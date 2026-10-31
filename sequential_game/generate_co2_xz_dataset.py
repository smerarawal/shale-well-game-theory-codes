"""Generate split-by-geology CO2 x-z transition data for surrogate training.

Input is the geology archive written by ``co2_experiment.py --spe10-xz``.
The three synthetic scalar-k cohorts are converted to directional mD fields by
rank-mapping their spatial patterns to the pooled SPE10 x-z Kx/Kz marginals.
This is an explicit synthetic scaling convention, not a rock-physics model.

Writes one compressed transition archive per cohort and split. Train/validation/
test are separated by synthetic geology seed or SPE10 y-slice, never by stage.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from co2_xz_solver import CO2XZConfig, CO2XZSolver, initial_state


COHORTS = ("friendly", "intermediate", "challenging", "tarbert", "upper_ness")
SPLITS = ("train", "validation", "test")


def _scalar(v):
    return v.item() if hasattr(v, "item") else v


def _metadata(raw):
    value = _scalar(raw)
    return json.loads(value) if isinstance(value, str) else dict(value)


def _resample(a, target_z, target_x, *, geometric=False):
    """Geometric/log K block reduction; nearest replication only when enlarging."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("geology fields must be 2D [z,x]")
    if target_z >= a.shape[0] and target_x >= a.shape[1]:
        zi = np.minimum((np.arange(target_z) * a.shape[0] / target_z).astype(int), a.shape[0] - 1)
        xi = np.minimum((np.arange(target_x) * a.shape[1] / target_x).astype(int), a.shape[1] - 1)
        return a[np.ix_(zi, xi)].copy()
    out = np.empty((target_z, target_x), dtype=np.float64)
    for iz in range(target_z):
        z0, z1 = int(np.floor(iz * a.shape[0] / target_z)), int(np.ceil((iz + 1) * a.shape[0] / target_z))
        z1 = max(z1, z0 + 1)
        for ix in range(target_x):
            x0, x1 = int(np.floor(ix * a.shape[1] / target_x)), int(np.ceil((ix + 1) * a.shape[1] / target_x))
            x1 = max(x1, x0 + 1)
            block = np.maximum(a[z0:z1, x0:x1], 1e-30)
            out[iz, ix] = np.exp(np.log(block).mean()) if geometric else block.mean()
    return out


def _split_geologies(cases, *, spe10, seed):
    """Return index lists, keeping all matching y sections in one split."""
    if spe10:
        result = {name: [] for name in SPLITS}
        for i, case in enumerate(cases):
            y = int(case["metadata"]["spe10_y_index"])
            if y < 73 or y >= 147:
                result["train"].append(i)
            elif y < 110:
                result["validation"].append(i)
            else:
                result["test"].append(i)
        if any(not result[name] for name in SPLITS):
            raise ValueError("SPE10 archive must include sampled y-slices from all three spatial split blocks")
        return result
    groups = {}
    for i, case in enumerate(cases):
        group = int(case["seed"])
        groups.setdefault(group, []).append(i)
    group_ids = np.asarray(sorted(groups), dtype=np.int64)
    if len(group_ids) < 3:
        raise ValueError("need at least 3 distinct geology seeds/y slices per cohort for train/validation/test")
    rng = np.random.default_rng(seed)
    rng.shuffle(group_ids)
    n = len(group_ids)
    n_test = max(1, int(round(0.15 * n)))
    n_val = max(1, int(round(0.15 * n)))
    while n_test + n_val >= n:
        if n_val > 1:
            n_val -= 1
        else:
            n_test -= 1
    assignment = {g: "train" for g in group_ids}
    for g in group_ids[:n_test]:
        assignment[int(g)] = "test"
    for g in group_ids[n_test:n_test + n_val]:
        assignment[int(g)] = "validation"
    result = {name: [] for name in SPLITS}
    for group, indices in groups.items():
        result[assignment[group]].extend(indices)
    return result


def _action_schedules(episodes, stages, qmax, rng):
    templates = [
        ("symmetric_mid", np.full((stages, 2), 0.5 * qmax)),
        ("operator0_only", np.tile([qmax, 0.0], (stages, 1))),
        ("operator1_only", np.tile([0.0, qmax], (stages, 1))),
        ("alternating_asymmetric", np.asarray([
            [qmax, 0.25 * qmax] if t % 2 == 0 else [0.25 * qmax, qmax]
            for t in range(stages)]) ),
        ("symmetric_high", np.full((stages, 2), qmax)),
        ("alternating_off", np.asarray([
            [qmax, 0.0] if t % 2 == 0 else [0.0, qmax] for t in range(stages)]) ),
    ]
    schedules = templates[:min(episodes, len(templates))]
    levels = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]) * qmax
    while len(schedules) < episodes:
        rates = rng.choice(levels, size=(stages, 2), replace=True)
        # Introduce temporal persistence while retaining stage-to-stage action changes.
        for t in range(1, stages):
            if rng.random() < 0.5:
                rates[t] = rates[t - 1]
        schedules.append((f"random_{len(schedules):03d}", rates))
    return schedules


def _build_cases(npz, target_n):
    # Map synthetic spatial ranks onto the empirical directional permeability
    # distributions present in the SPE10 x-z archive.
    ref_x = np.concatenate([np.asarray(npz[f"spe10_xz_{r}_kx_mD"], float).ravel()
                            for r in ("tarbert", "upper_ness")])
    ref_z = np.concatenate([np.asarray(npz[f"spe10_xz_{r}_kz_mD"], float).ravel()
                            for r in ("tarbert", "upper_ness")])
    ref_x = ref_x[np.isfinite(ref_x) & (ref_x > 0)]
    ref_z = ref_z[np.isfinite(ref_z) & (ref_z > 0)]
    if min(len(ref_x), len(ref_z)) == 0:
        raise ValueError("SPE10 x-z Kx/Kz calibration fields are empty")
    cases_by_cohort = {name: [] for name in COHORTS}

    for cohort in ("friendly", "intermediate", "challenging"):
        karr = np.asarray(npz[f"{cohort}_k"], float)
        phiarr = np.asarray(npz[f"{cohort}_phi"], float)
        seeds = np.asarray(npz[f"{cohort}_seed"], int)
        metadata = np.asarray(npz[f"{cohort}_metadata_json"], dtype=object)
        for i, k in enumerate(karr):
            ranks = (np.argsort(np.argsort(k.ravel(), kind="stable"), kind="stable") /
                     max(k.size - 1, 1)).reshape(k.shape)
            kx = np.quantile(ref_x, ranks)
            kz = np.quantile(ref_z, ranks)
            meta = _metadata(metadata[i])
            # Common 1,200 ft by 100 ft synthetic x-z domain; these dimensions
            # are declared scenario assumptions, not field measurements.
            cases_by_cohort[cohort].append({
                "kx": _resample(kx, target_n, target_n, geometric=True),
                "kz": _resample(kz, target_n, target_n, geometric=True),
                "phi": _resample(phiarr[i], target_n, target_n),
                "seed": int(seeds[i]), "metadata": {
                    **meta, "source_type": "synthetic_rank_mapped_to_SPE10_marginals",
                    "dx_ft": 1200.0 / target_n, "dy_ft": 10.0,
                    "dz_ft": 100.0 / target_n,
                    "permeability_mapping": "synthetic k rank mapped to pooled SPE10 x-z Kx/Kz marginal distributions",
                    "scaling_status": "scenario assumption; requires expert calibration"}})

    for region in ("tarbert", "upper_ness"):
        prefix = f"spe10_xz_{region}"
        kxs, kzs, phis = (np.asarray(npz[f"{prefix}_{key}"], float)
                          for key in ("kx_mD", "kz_mD", "phi"))
        seeds = np.asarray(npz[f"{prefix}_seed"], int)
        metadata = np.asarray(npz[f"{prefix}_metadata_json"], dtype=object)
        for i in range(len(kxs)):
            meta = _metadata(metadata[i])
            z_extent_ft = float(meta["dz_ft"]) * kxs[i].shape[0]
            cases_by_cohort[region].append({
                "kx": _resample(kxs[i], target_n, target_n, geometric=True),
                "kz": _resample(kzs[i], target_n, target_n, geometric=True),
                "phi": _resample(phis[i], target_n, target_n),
                "seed": int(seeds[i]),
                "metadata": {**meta, "dx_ft": 1200.0 / target_n,
                             "dy_ft": float(meta["dy_ft"]),
                             "dz_ft": z_extent_ft / target_n,
                             "source_type": "SPE10_raw_directional_xz"}})
    return cases_by_cohort


def _generate_split(cases, indices, cohort, split, args, schedule_seed):
    # sg_max is required to make the Land-style hysteresis state Markov.
    state_fields = ("p_pa", "sg_mobile", "sg_trapped", "sg_max", "c_dissolved_kg_m3")
    states = {f"{key}_{edge}": [] for key in state_fields for edge in ("in", "out")}
    states.update({key: [] for key in ("kx_mD", "kz_mD", "phi")})
    labels = {key: [] for key in (
        "geology_seed", "geology_index", "episode_id", "schedule_id", "schedule_name",
        "stage_index", "rate_operator_0_m3_s", "rate_operator_1_m3_s", "dx_ft", "dy_ft", "dz_ft",
        "geology_metadata_json")}
    metric_rows = []
    for gi in indices:
        case = cases[gi]
        episode_key = (int(case["metadata"]["spe10_y_index"])
                       if cohort in ("tarbert", "upper_ness") else int(case["seed"]))
        rng = np.random.default_rng(schedule_seed + episode_key)
        schedules = _action_schedules(args.episodes_per_geology, args.stages,
                                      args.rate_m3_s, rng)
        kx, kz, phi = case["kx"], case["kz"], case["phi"]
        meta = case["metadata"]
        config = CO2XZConfig(
            dx_ft=float(meta["dx_ft"]), dy_ft=float(meta["dy_ft"]),
            dz_ft=float(meta["dz_ft"]), max_dt_s=args.max_dt_days * 86400.0,
            max_steps=args.max_steps, solubility_kg_m3_brine=args.solubility_kg_m3,
            dissolution_rate_s_inv=args.dissolution_rate_s,
            land_coefficient=args.land_coefficient)
        solver = CO2XZSolver(config)
        nz, nx = phi.shape
        well_z = nz - 2
        well_x = (max(1, nx // 3), min(nx - 2, (2 * nx) // 3))
        for episode_id, (schedule_name, rates_by_stage) in enumerate(schedules):
            state = initial_state(phi.shape)
            for stage, rates in enumerate(rates_by_stage):
                prior = {key: state[key].copy() for key in state_fields}
                wells = [(well_z, well_x[op], float(rates[op]))
                         for op in range(2) if rates[op] > 0]
                # Also advance shut-in stages: existing mobile CO2 can still
                # segregate and dissolve after an operator stops injecting.
                state, diag, _ = solver.advance(
                    kx, kz, phi, state, wells, args.stage_days * 86400.0)
                for key in state_fields:
                    states[f"{key}_in"].append(prior[key].astype(np.float32))
                    states[f"{key}_out"].append(state[key].astype(np.float32))
                for key, value in (("kx_mD", kx), ("kz_mD", kz), ("phi", phi)):
                    states[key].append(value.astype(np.float32))
                labels["geology_seed"].append(int(case["seed"]))
                labels["geology_index"].append(int(gi))
                labels["episode_id"].append(int(episode_id))
                labels["schedule_id"].append(int(episode_id))
                labels["schedule_name"].append(schedule_name)
                labels["stage_index"].append(int(stage))
                labels["rate_operator_0_m3_s"].append(float(rates[0]))
                labels["rate_operator_1_m3_s"].append(float(rates[1]))
                for key in ("dx_ft", "dy_ft", "dz_ft"):
                    labels[key].append(float(meta[key]))
                labels["geology_metadata_json"].append(json.dumps(meta, sort_keys=True))
                metric_rows.append({"cohort": cohort, "split": split,
                                    "geology_seed": int(case["seed"]),
                                    "episode_id": episode_id, "schedule": schedule_name,
                                    "stage": stage,
                                    "rate_operator_0_m3_s": float(rates[0]),
                                    "rate_operator_1_m3_s": float(rates[1]), **diag})
    archive = {key: np.stack(value) for key, value in states.items()}
    archive.update({key: np.asarray(value) for key, value in labels.items()})
    return archive, metric_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-n", type=int, default=24)
    parser.add_argument("--episodes-per-geology", type=int, default=4)
    parser.add_argument("--stages", type=int, default=6)
    parser.add_argument("--stage-days", type=float, default=30.0)
    parser.add_argument("--rate-m3-s", type=float, default=1e-7,
                        help="maximum operator injection rate; schedules use 0 to 100%% of this value")
    parser.add_argument("--max-dt-days", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--solubility-kg-m3", type=float, default=30.0)
    parser.add_argument("--dissolution-rate-s", type=float, default=2e-7)
    parser.add_argument("--land-coefficient", type=float, default=2.0)
    args = parser.parse_args()
    if min(args.target_n, args.episodes_per_geology, args.stages) < 1:
        parser.error("target grid, episode count, and stage count must be positive")

    input_dir, out_dir = Path(args.input_dir), Path(args.out_dir)
    archive_path = input_dir / "co2_geology_cohorts.npz"
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing geology archive: {archive_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics, split_manifest = [], []
    with np.load(archive_path, allow_pickle=True) as npz:
        cases_by_cohort = _build_cases(npz, args.target_n)
    for ci, cohort in enumerate(COHORTS):
        cases = cases_by_cohort[cohort]
        if not cases:
            raise ValueError(f"No geology cases found for cohort {cohort}")
        is_spe10 = cohort in ("tarbert", "upper_ness")
        # Use identical y-slice split assignments and paired action schedules
        # for Tarbert and Ness, so comparisons share both geology rows and actions.
        split_seed = args.seed + 50_000 if is_spe10 else args.seed + ci
        split_indices = _split_geologies(cases, spe10=is_spe10, seed=split_seed)
        for split in SPLITS:
            indices = sorted(split_indices[split])
            if not indices:
                raise ValueError(f"Empty {split} split for {cohort}")
            schedule_seed = args.seed + 91_001 if is_spe10 else args.seed + ci * 100_003
            data, metrics = _generate_split(cases, indices, cohort, split, args,
                                            schedule_seed=schedule_seed)
            split_dir = out_dir / split
            split_dir.mkdir(exist_ok=True)
            path = split_dir / f"{cohort}.npz"
            np.savez_compressed(path, **data)
            all_metrics.extend(metrics)
            seeds = [int(cases[i]["seed"]) for i in indices]
            group_ids = [int(cases[i]["metadata"].get("spe10_y_index", cases[i]["seed"]))
                         if is_spe10 else int(cases[i]["seed"]) for i in indices]
            split_manifest.append({"cohort": cohort, "split": split,
                                   "geology_count": len(indices),
                                   "transition_count": len(metrics),
                                   "geology_seeds": json.dumps(seeds),
                                   "geology_group_ids": json.dumps(group_ids),
                                   "file": str(path.relative_to(out_dir))})
            print(f"{cohort}/{split}: {len(indices)} geologies, {len(metrics)} transitions -> {path}",
                  flush=True)
    with (out_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(split_manifest[0]))
        writer.writeheader(); writer.writerows(split_manifest)
    with (out_dir / "stage_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0]))
        writer.writeheader(); writer.writerows(all_metrics)
    config = {"description": "CO2 x-z solver labels; exploratory physical/scaling assumptions",
              "parameters": vars(args),
              "cohorts": list(COHORTS),
              "dynamic_state_fields": ["p_pa", "sg_mobile", "sg_trapped", "sg_max", "c_dissolved_kg_m3"],
              "split_unit": "synthetic geology seed; SPE10 y slice",
              "transitions": len(all_metrics),
              "synthetic_scaling": "rank-map scalar synthetic k to pooled SPE10 directional Kx/Kz marginals; fixed 1200 ft x 100 ft domain",
              "grid_resampling": "24x24; geometric block means for permeability downsampling; nearest replication only when upsampling",
              "physics_limits": ["experimental 2D x-z IMPES model", "not field calibrated", "no capillary pressure", "closed outer boundaries", "constant phase properties", "simplified dissolution and residual trapping", "not a 3D model"]}
    (out_dir / "dataset_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Complete: {len(all_metrics)} stage transitions in {out_dir}")
    print("No surrogate has been trained by this script; fit preprocessing only on train splits.")


if __name__ == "__main__":
    main()
