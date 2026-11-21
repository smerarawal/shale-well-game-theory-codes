"""Regenerate CO2 transitions with variable 2-8 operator well locations.

The input must be the existing geology-split CO2 dataset. Its permeability and
porosity maps and train/validation/test geology assignments are preserved. New
solver trajectories are written to a separate output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .co2_xz_solver import CO2XZConfig, CO2XZSolver, initial_state
except ImportError:
    from co2_xz_solver import CO2XZConfig, CO2XZSolver, initial_state


COHORTS = ("friendly", "intermediate", "challenging", "tarbert", "upper_ness")
SPLITS = ("train", "validation", "test")
STATE_FIELDS = ("p_pa", "sg_mobile", "sg_trapped", "sg_max", "c_dissolved_kg_m3")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _geology_rows(data: dict[str, np.ndarray]):
    """One stage-0 row per geology, retaining the supplied split exactly."""
    selected, seen = [], set()
    for i in np.flatnonzero(data["stage_index"] == 0):
        key = (int(data["geology_seed"][i]), int(data["geology_index"][i]))
        if key not in seen:
            seen.add(key)
            selected.append(int(i))
    return selected


def _schedule(n_ops: int, stages: int, qmax: float, rng: np.random.Generator):
    levels = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]) * qmax
    rates = rng.choice(levels, size=(stages, n_ops), replace=True)
    for t in range(1, stages):
        keep = rng.random(n_ops) < 0.5
        rates[t, keep] = rates[t - 1, keep]
    # Add both all-active and all-shut-in examples to broaden total-rate coverage.
    if rng.random() < 0.5:
        rates[0] = qmax
    if rng.random() < 0.25:
        rates[-1] = 0.0
    return rates


def _make_archive(rows, source, *, episodes, stages, stage_days, qmax,
                  min_spacing_cells, seed, max_dt_days, max_steps,
                  solubility_kg_m3, dissolution_rate_s, land_coefficient):
    sample = rows[0]
    shape = source["phi"][sample].shape
    nz, nx = shape
    dx_ft, dz_ft = float(source["dx_ft"][sample]), float(source["dz_ft"][sample])
    if min_spacing_cells < 0:
        raise ValueError("min-spacing-cells must be nonnegative")
    # One common lower-row band keeps well depth comparable while x locations vary.
    cells = [(nz - 2, x) for x in range(1, nx - 1)]
    if not cells:
        raise ValueError("grid has no legal interior well cells")
    feasible = []
    for cell in cells:
        if not feasible or all(abs(cell[1] - other[1]) >= min_spacing_cells
                               for other in feasible):
            feasible.append(cell)
    if len(feasible) < 8:
        raise ValueError("grid cannot fit eight candidate wells at requested cell spacing")

    states = {f"{k}_{edge}": [] for k in STATE_FIELDS for edge in ("in", "out")}
    states.update({k: [] for k in ("kx_mD", "kz_mD", "phi")})
    row_fields = {k: [] for k in (
        "geology_seed", "geology_index", "episode_id", "schedule_id", "schedule_name",
        "stage_index", "n_operators", "rate_operator_0_m3_s", "rate_operator_1_m3_s",
        "rate_total_m3_s", "operator_rates_m3_s", "well_locations_zx",
        "well_rate_map", "well_count_map", "dx_ft", "dy_ft", "dz_ft",
        "geology_metadata_json")}
    metrics = []
    for geology_row in rows:
        geo_seed = int(source["geology_seed"][geology_row])
        geo_index = int(source["geology_index"][geology_row])
        rng = np.random.default_rng(seed + geo_seed * 1009 + geo_index * 9176)
        kx = np.asarray(source["kx_mD"][geology_row], dtype=np.float64)
        kz = np.asarray(source["kz_mD"][geology_row], dtype=np.float64)
        phi = np.asarray(source["phi"][geology_row], dtype=np.float64)
        solver = CO2XZSolver(CO2XZConfig(
            dx_ft=dx_ft, dy_ft=float(source["dy_ft"][geology_row]), dz_ft=dz_ft,
            max_dt_s=max_dt_days * 86400.0, max_steps=max_steps,
            solubility_kg_m3_brine=solubility_kg_m3,
            dissolution_rate_s_inv=dissolution_rate_s,
            land_coefficient=land_coefficient))
        prior_meta = source["geology_metadata_json"][geology_row]
        if isinstance(prior_meta, bytes):
            prior_meta = prior_meta.decode("utf-8")
        for episode in range(episodes):
            # Ensure systematic count coverage, then randomize counts in later episodes.
            n_ops = (2, 4, 6, 8)[episode] if episode < 4 else int(rng.integers(2, 9))
            locations = [feasible[i] for i in rng.choice(len(feasible), n_ops, replace=False)]
            rates_by_stage = _schedule(n_ops, stages, qmax, rng)
            state = initial_state(shape)
            for stage in range(stages):
                rates = rates_by_stage[stage]
                rate_map = np.zeros(shape, dtype=np.float32)
                count_map = np.zeros(shape, dtype=np.float32)
                wells = []
                for (z, x), q in zip(locations, rates):
                    if q > 0:
                        wells.append((z, x, float(q)))
                        rate_map[z, x] += float(q)
                        count_map[z, x] += 1.0
                prior = {k: state[k].copy() for k in STATE_FIELDS}
                state, diag, _ = solver.advance(kx, kz, phi, state, wells,
                                                stage_days * 86400.0)
                for key in STATE_FIELDS:
                    states[f"{key}_in"].append(prior[key].astype(np.float32))
                    states[f"{key}_out"].append(state[key].astype(np.float32))
                for key, value in (("kx_mD", kx), ("kz_mD", kz), ("phi", phi)):
                    states[key].append(value.astype(np.float32))
                row_fields["geology_seed"].append(geo_seed)
                row_fields["geology_index"].append(geo_index)
                row_fields["episode_id"].append(episode)
                row_fields["schedule_id"].append(episode)
                row_fields["schedule_name"].append(f"random_n{n_ops}_ep{episode:03d}")
                row_fields["stage_index"].append(stage)
                row_fields["n_operators"].append(n_ops)
                row_fields["rate_operator_0_m3_s"].append(float(rates[0]))
                row_fields["rate_operator_1_m3_s"].append(float(rates[1]) if n_ops > 1 else 0.0)
                row_fields["rate_total_m3_s"].append(float(rates.sum()))
                padded = np.zeros(8, dtype=np.float32); padded[:n_ops] = rates
                padded_locs = np.full((8, 2), -1, dtype=np.int16)
                padded_locs[:n_ops] = locations
                row_fields["operator_rates_m3_s"].append(padded)
                row_fields["well_locations_zx"].append(padded_locs)
                row_fields["well_rate_map"].append(rate_map)
                row_fields["well_count_map"].append(count_map)
                row_fields["dx_ft"].append(dx_ft)
                row_fields["dy_ft"].append(float(source["dy_ft"][geology_row]))
                row_fields["dz_ft"].append(dz_ft)
                row_fields["geology_metadata_json"].append(prior_meta)
                metrics.append({"geology_seed": geo_seed, "geology_index": geo_index,
                                "episode_id": episode, "n_operators": n_ops,
                                "stage": stage, "total_rate_m3_s": float(rates.sum()), **diag})
    archive = {k: np.stack(v) for k, v in states.items()}
    archive.update({k: np.asarray(v) for k, v in row_fields.items()})
    return archive, metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True, help="existing geology-split CO2 dataset")
    ap.add_argument("--out-dir", required=True, help="new output directory; must differ from input")
    ap.add_argument("--episodes-per-geology", type=int, default=12)
    ap.add_argument("--cohorts", nargs="+", choices=COHORTS, default=list(COHORTS))
    ap.add_argument("--stages", type=int, default=6)
    ap.add_argument("--stage-days", type=float, default=30.0)
    ap.add_argument("--rate-m3-s-per-operator", type=float, default=1e-7)
    ap.add_argument("--min-spacing-cells", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--max-dt-days", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--solubility-kg-m3", type=float, default=30.0)
    ap.add_argument("--dissolution-rate-s", type=float, default=2e-7)
    ap.add_argument("--land-coefficient", type=float, default=2.0)
    a = ap.parse_args()
    if a.episodes_per_geology < 8 or a.stages < 1 or a.rate_m3_s_per_operator <= 0:
        ap.error("use at least 8 episodes/geology, positive stages, and positive operator rate")
    source_dir, out_dir = Path(a.input_dir).resolve(), Path(a.out_dir).resolve()
    if source_dir == out_dir or source_dir in out_dir.parents:
        ap.error("output must be a new sibling directory, never inside the source dataset")
    config_path = source_dir / "dataset_config.json"
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, all_metrics = [], []
    for cohort in a.cohorts:
        for split in SPLITS:
            path = source_dir / split / f"{cohort}.npz"
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=False) as z:
                source = {k: z[k] for k in z.files}
            rows = _geology_rows(source)
            archive, metrics = _make_archive(
                rows, source, episodes=a.episodes_per_geology, stages=a.stages,
                stage_days=a.stage_days, qmax=a.rate_m3_s_per_operator,
                min_spacing_cells=a.min_spacing_cells, seed=a.seed,
                max_dt_days=a.max_dt_days, max_steps=a.max_steps,
                solubility_kg_m3=a.solubility_kg_m3,
                dissolution_rate_s=a.dissolution_rate_s,
                land_coefficient=a.land_coefficient)
            split_dir = out_dir / split; split_dir.mkdir(exist_ok=True)
            out_path = split_dir / f"{cohort}.npz"
            np.savez_compressed(out_path, **archive)
            all_metrics.extend({"cohort": cohort, "split": split, **m} for m in metrics)
            manifest.append({"cohort": cohort, "split": split,
                             "geology_count": len(rows), "transition_count": len(metrics),
                             "file": str(out_path.relative_to(out_dir))})
            print(f"{cohort}/{split}: {len(rows)} geologies, {len(metrics)} transitions -> {out_path}", flush=True)
    with (out_dir / "stage_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0]))
        writer.writeheader(); writer.writerows(all_metrics)
    (out_dir / "split_manifest.csv").write_text(
        "cohort,split,geology_count,transition_count,file\n" + "".join(
            f"{m['cohort']},{m['split']},{m['geology_count']},{m['transition_count']},{m['file']}\n"
            for m in manifest), encoding="utf-8")
    config = {"description": "CO2 multiwell trajectories; variable 2-8 operators and locations",
              "input_dataset_dir": str(source_dir),
              "input_dataset_config_sha256": _sha256(config_path),
              "input_dataset_description": base_config.get("description"),
              "physics_limits": base_config.get("physics_limits", []),
              "parameters": {"episodes_per_geology": a.episodes_per_geology,
                             "stages": a.stages, "stage_days": a.stage_days,
                             "rate_m3_s_per_operator": a.rate_m3_s_per_operator,
                             "min_spacing_cells": a.min_spacing_cells,
                             "max_dt_days": a.max_dt_days, "max_steps": a.max_steps,
                             "solubility_kg_m3": a.solubility_kg_m3,
                             "dissolution_rate_s": a.dissolution_rate_s,
                             "land_coefficient": a.land_coefficient,
                             "seed": a.seed},
              "n_operators_range": [2, 8], "location_input": "aggregate well_rate_map plus active well_count_map",
              "split_policy": "preserved exactly from source geology split; no geology moved between splits",
              "transitions": len(all_metrics), "manifest": manifest}
    (out_dir / "dataset_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote new CO2 multiwell dataset: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
