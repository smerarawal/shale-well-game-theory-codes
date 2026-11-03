"""Label surrogate-planned game visits with the CO2 solver.

Each input JSON schedule is a two-operator, multi-stage rate path produced by
``co2_differentiable_game.py``. This script replays the path from a held-in
training geology with the actual CO2 solver, saves solver-labelled transitions
in the existing trainer NPZ schema, and appends them only to the training split.
It is the data-collection step of an equilibrium-aware / DAgger-style loop.

Example:
  python collect_co2_equilibrium_data.py --base-data-dir DATA --schedule-json br.json \\
      --out-dir DATA_ACTIVE
  # Repeat after planning additional training geologies, pointing --base-data-dir
  # at the last active dataset so prior labels remain included.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .co2_xz_solver import CO2XZConfig, CO2XZSolver
    from .train_co2_xz_surrogate import COHORTS, STATE_KEYS
except ImportError:
    from co2_xz_solver import CO2XZConfig, CO2XZSolver
    from train_co2_xz_surrogate import COHORTS, STATE_KEYS


def _solver_config(data_dir: Path) -> tuple[CO2XZConfig, float]:
    cfg_path = data_dir / "dataset_config.json"
    if not cfg_path.exists():
        return CO2XZConfig(), 30.0
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    p = payload.get("parameters", {})
    days = float(p.get("stage_days", 30.0))
    conf = CO2XZConfig(
        max_dt_s=float(p.get("max_dt_days", 1.0)) * 86400.0,
        max_steps=int(p.get("max_steps", 20000)),
        solubility_kg_m3_brine=float(p.get("solubility_kg_m3", 30.0)),
        dissolution_rate_s_inv=float(p.get("dissolution_rate_s", 2e-7)),
        land_coefficient=float(p.get("land_coefficient", 2.0)))
    return conf, days


def _load_json_schedules(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    schedules = value if isinstance(value, list) else [value]
    for item in schedules:
        if not isinstance(item, dict) or "cohort" not in item or "geology_seed" not in item:
            raise ValueError("each schedule needs cohort and geology_seed")
        if "rates_m3_s_by_stage_operator" not in item:
            raise ValueError("schedule is missing rates_m3_s_by_stage_operator")
        rates = np.asarray(item["rates_m3_s_by_stage_operator"], dtype=float)
        if rates.ndim != 2 or rates.shape[1] != 2 or np.any(~np.isfinite(rates)) or np.any(rates < 0):
            raise ValueError("rates must be a finite, nonnegative [stage, 2] array")
        item["rates_m3_s_by_stage_operator"] = rates
    return schedules


def _transition_rows(data: dict[str, np.ndarray], row: int,
                     rates: np.ndarray, cohort: str, episode_id: int,
                     solver: CO2XZSolver, stage_days: float) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    kx, kz, phi = data["kx_mD"][row], data["kz_mD"][row], data["phi"][row]
    state = {key: np.asarray(data[f"{key}_in"][row], dtype=np.float64).copy()
             for key in STATE_KEYS}
    state["time_s"] = float(data.get("time_s_in", np.zeros(len(data["geology_seed"])))[row]) if "time_s_in" in data else 0.0
    state["injected_co2_kg"] = 0.0
    h, w = phi.shape
    wz, wx = h - 2, (max(1, w // 3), min(w - 2, (2 * w) // 3))
    stage_seconds = stage_days * 86400.0
    records: dict[str, list[Any]] = {f"{key}_{edge}": [] for key in STATE_KEYS for edge in ("in", "out")}
    records.update({key: [] for key in ("kx_mD", "kz_mD", "phi")})
    labels: dict[str, list[Any]] = {key: [] for key in (
        "geology_seed", "geology_index", "episode_id", "schedule_id", "schedule_name",
        "stage_index", "rate_operator_0_m3_s", "rate_operator_1_m3_s",
        "dx_ft", "dy_ft", "dz_ft", "geology_metadata_json")}
    metrics: list[dict[str, Any]] = []
    metadata = str(data["geology_metadata_json"][row]) if "geology_metadata_json" in data else "{}"
    geology_seed = int(data["geology_seed"][row])
    geology_index = int(data["geology_index"][row])
    dx, dy, dz = (float(data[f"{key}_ft"][row]) for key in ("dx", "dy", "dz"))
    for stage, rates_by_operator in enumerate(rates):
        before = {key: state[key].copy() for key in STATE_KEYS}
        wells = [(wz, wx[op], float(rate)) for op, rate in enumerate(rates_by_operator) if rate > 0]
        state, diag, _ = solver.advance(kx, kz, phi, state, wells, stage_seconds)
        for key in STATE_KEYS:
            records[f"{key}_in"].append(before[key].astype(np.float32))
            records[f"{key}_out"].append(np.asarray(state[key]).astype(np.float32))
        for key, value in (("kx_mD", kx), ("kz_mD", kz), ("phi", phi)):
            records[key].append(np.asarray(value).astype(np.float32))
        for key, value in (("geology_seed", geology_seed), ("geology_index", geology_index),
                           ("episode_id", episode_id), ("schedule_id", episode_id),
                           ("schedule_name", "equilibrium_aware_best_response"),
                           ("stage_index", stage),
                           ("rate_operator_0_m3_s", rates_by_operator[0]),
                           ("rate_operator_1_m3_s", rates_by_operator[1]),
                           ("dx_ft", dx), ("dy_ft", dy), ("dz_ft", dz),
                           ("geology_metadata_json", metadata)):
            labels[key].append(value)
        metrics.append({"cohort": cohort, "geology_seed": geology_seed,
                        "episode_id": episode_id, "stage_index": stage,
                        "rate_operator_0_m3_s": float(rates_by_operator[0]),
                        "rate_operator_1_m3_s": float(rates_by_operator[1]),
                        **diag})
    result = {key: np.asarray(value) for key, value in records.items()}
    for key, value in labels.items():
        if key in ("geology_seed", "geology_index", "episode_id", "schedule_id", "stage_index"):
            result[key] = np.asarray(value, dtype=np.int64)
        elif key == "schedule_name" or key == "geology_metadata_json":
            result[key] = np.asarray(value, dtype=str)
        else:
            result[key] = np.asarray(value, dtype=np.float32)
    return result, metrics


def _append_npz(path: Path, additions: list[dict[str, np.ndarray]]) -> None:
    with np.load(path, allow_pickle=False) as z:
        merged = {key: z[key] for key in z.files}
    for addition in additions:
        if set(addition) != set(merged):
            missing = sorted(set(merged) - set(addition))
            extra = sorted(set(addition) - set(merged))
            raise ValueError(f"active label schema mismatch (missing={missing}, extra={extra})")
        for key in merged:
            merged[key] = np.concatenate((merged[key], addition[key]), axis=0)
    np.savez_compressed(path, **merged)


def collect(base_data_dir: Path, schedule_path: Path, out_dir: Path) -> Path:
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}; use a new output path")
    shutil.copytree(base_data_dir, out_dir)
    cfg, default_stage_days = _solver_config(base_data_dir)
    solver = CO2XZSolver(cfg)
    schedules = _load_json_schedules(schedule_path)
    by_cohort: dict[str, list[dict[str, np.ndarray]]] = {}
    manifest: list[dict[str, Any]] = []
    for item in schedules:
        cohort = str(item["cohort"])
        seed = int(item["geology_seed"])
        if cohort not in COHORTS:
            raise ValueError(f"unknown cohort {cohort!r}; expected one of {COHORTS}")
        rates = np.asarray(item["rates_m3_s_by_stage_operator"], dtype=float)
        source = base_data_dir / "train" / f"{cohort}.npz"
        with np.load(source, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        candidates = np.flatnonzero((data["geology_seed"] == seed) & (data["stage_index"] == 0))
        if not len(candidates):
            raise ValueError(f"training geology seed {seed} not found in {source}")
        row = int(candidates[0])
        episode_id = int(data["episode_id"][data["geology_seed"] == seed].max()) + 1
        stage_days = float(item.get("stage_days", default_stage_days))
        addition, metrics = _transition_rows(data, row, rates, cohort, episode_id,
                                             solver, stage_days)
        by_cohort.setdefault(cohort, []).append(addition)
        manifest.extend(metrics)
    for cohort, additions in by_cohort.items():
        _append_npz(out_dir / "train" / f"{cohort}.npz", additions)
    with (out_dir / "equilibrium_label_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader(); writer.writerows(manifest)
    meta = {"source_data": str(base_data_dir), "schedule_file": str(schedule_path),
            "added_transitions": len(manifest), "added_episodes": len(schedules),
            "label_source": "CO2XZSolver", "split_policy": "appended to train only",
            "ensemble_used": False}
    (out_dir / "equilibrium_label_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    dataset_config = out_dir / "dataset_config.json"
    if dataset_config.exists():
        config_payload = json.loads(dataset_config.read_text(encoding="utf-8"))
        config_payload["transitions"] = int(config_payload.get("transitions", 0)) + len(manifest)
        config_payload["equilibrium_aware_added_transitions"] = len(manifest)
        config_payload["equilibrium_aware_schedule_file"] = str(schedule_path)
        dataset_config.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-data-dir", type=Path, required=True)
    ap.add_argument("--schedule-json", type=Path, required=True,
                    help="one planner JSON or a JSON list of planner records")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    print(f"Wrote solver-labelled equilibrium data to {collect(args.base_data_dir, args.schedule_json, args.out_dir)}")


if __name__ == "__main__":
    main()
