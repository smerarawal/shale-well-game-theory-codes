"""Generate a small solver-labelled CO2 x-z trajectory pilot from SPE10 inputs.

Example (after ``co2_experiment.py --spe10-xz``):

    python sequential_game/generate_co2_xz_pilot.py \
      --input-dir /kaggle/working/co2_geologies \
      --out-dir /kaggle/working/co2_xz_pilot

This creates development data only. Do not train/evaluate a final surrogate from
this pilot; it has a small geology count and is not a held-out split.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from co2_xz_solver import CO2XZConfig, CO2XZSolver, initial_state


def _meta_value(value):
    return value.item() if hasattr(value, "item") else value


def _coarsen(a, fz, fx, *, geometric=False):
    a = np.asarray(a, dtype=np.float64)
    nz = a.shape[0] // fz * fz
    nx = a.shape[1] // fx * fx
    cropped = a[:nz, :nx].reshape(nz // fz, fz, nx // fx, fx)
    if geometric:
        return np.exp(np.log(np.maximum(cropped, 1e-12)).mean(axis=(1, 3)))
    return cropped.mean(axis=(1, 3))


def _schedules(stages, rate):
    base = np.full((stages, 2), rate, dtype=np.float64)
    asymmetric = np.empty_like(base)
    for t in range(stages):
        asymmetric[t] = (1.5 * rate, 0.5 * rate) if t % 2 == 0 else (0.5 * rate, 1.5 * rate)
    return {"symmetric": base, "alternating_asymmetric": asymmetric}


def _save_region(npz, region, out_dir, args, schedule_map):
    prefix = f"spe10_xz_{region}"
    keys = [f"{prefix}_{name}" for name in ("kx_mD", "kz_mD", "phi", "seed", "metadata_json")]
    missing = [key for key in keys if key not in npz]
    if missing:
        raise KeyError(f"Input archive is missing {missing}; regenerate geology inputs with --spe10-xz")
    kxs, kzs, phis, seeds, metadata = (np.asarray(npz[key]) for key in keys)
    count = min(args.geologies_per_region, len(kxs))
    transition_rows = []
    metadata_for_output = None
    states = {key: [] for key in (
        "p_in_pa", "sg_mobile_in", "sg_trapped_in", "c_dissolved_in_kg_m3",
        "p_out_pa", "sg_mobile_out", "sg_trapped_out", "c_dissolved_out_kg_m3",
        "kx_mD", "kz_mD", "phi")}
    labels = {key: [] for key in ("geology_seed", "geology_index", "schedule_id", "stage_index", "rate_operator_0_m3_s", "rate_operator_1_m3_s")}

    for gi in range(count):
        meta = _meta_value(metadata[gi])
        kx = _coarsen(kxs[gi], args.coarsen_z, args.coarsen_x, geometric=True)
        kz = _coarsen(kzs[gi], args.coarsen_z, args.coarsen_x, geometric=True)
        phi = _coarsen(phis[gi], args.coarsen_z, args.coarsen_x)
        local_meta = dict(meta)
        local_meta["pilot_coarsen_z"] = args.coarsen_z
        local_meta["pilot_coarsen_x"] = args.coarsen_x
        if metadata_for_output is None:
            metadata_for_output = local_meta
        config = CO2XZConfig(
            dx_ft=float(meta["dx_ft"]) * args.coarsen_x,
            dz_ft=float(meta["dz_ft"]) * args.coarsen_z,
            dy_ft=float(meta["dy_ft"]),
            max_dt_s=args.max_dt_days * 86400.0,
            max_steps=args.max_steps,
            solubility_kg_m3_brine=args.solubility_kg_m3,
            dissolution_rate_s_inv=args.dissolution_rate_s,
            land_coefficient=args.land_coefficient)
        solver = CO2XZSolver(config)
        nz, nx = phi.shape
        well_z = nz - 2
        well_x = (max(1, nx // 3), min(nx - 2, (2 * nx) // 3))
        for schedule_id, (schedule_name, rates_by_stage) in enumerate(schedule_map.items()):
            state = initial_state(phi.shape)
            for stage, rates in enumerate(rates_by_stage):
                prior = {key: state[key].copy() for key in (
                    "p_pa", "sg_mobile", "sg_trapped", "c_dissolved_kg_m3")}
                wells = [(well_z, well_x[op], float(rates[op])) for op in range(2) if rates[op] > 0]
                state, diag, _ = solver.advance(
                    kx, kz, phi, state, wells, args.stage_days * 86400.0)
                for key, value in (
                    ("p_in_pa", prior["p_pa"]), ("sg_mobile_in", prior["sg_mobile"]),
                    ("sg_trapped_in", prior["sg_trapped"]),
                    ("c_dissolved_in_kg_m3", prior["c_dissolved_kg_m3"]),
                    ("p_out_pa", state["p_pa"]), ("sg_mobile_out", state["sg_mobile"]),
                    ("sg_trapped_out", state["sg_trapped"]),
                    ("c_dissolved_out_kg_m3", state["c_dissolved_kg_m3"]),
                    ("kx_mD", kx), ("kz_mD", kz), ("phi", phi)):
                    states[key].append(value.astype(np.float32))
                for key, value in (
                    ("geology_seed", int(seeds[gi])), ("geology_index", gi),
                    ("schedule_id", schedule_id), ("stage_index", stage),
                    ("rate_operator_0_m3_s", float(rates[0])),
                    ("rate_operator_1_m3_s", float(rates[1]))):
                    labels[key].append(value)
                transition_rows.append({
                    "region": region, "geology_seed": int(seeds[gi]),
                    "schedule": schedule_name, "stage": stage,
                    "rate_operator_0_m3_s": float(rates[0]),
                    "rate_operator_1_m3_s": float(rates[1]), **diag})
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = {**{k: np.stack(v) for k, v in states.items()},
               **{k: np.asarray(v) for k, v in labels.items()},
               "grid_metadata_json": np.asarray([metadata_for_output], dtype=object)}
    np.savez_compressed(out_dir / f"{region}_pilot_transitions.npz", **archive)
    return transition_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True, help="folder with co2_geology_cohorts.npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--geologies-per-region", type=int, default=2)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--stage-days", type=float, default=30.0)
    ap.add_argument("--rate-m3-s", type=float, default=1e-7)
    ap.add_argument("--max-dt-days", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--land-coefficient", type=float, default=2.0)
    ap.add_argument("--solubility-kg-m3", type=float, default=30.0)
    ap.add_argument("--dissolution-rate-s", type=float, default=2e-7)
    ap.add_argument("--coarsen-z", type=int, default=1,
                    help="additional pilot coarsening; arithmetic porosity and geometric K means")
    ap.add_argument("--coarsen-x", type=int, default=1)
    args = ap.parse_args()
    if args.geologies_per_region < 1 or args.stages < 1 or args.stage_days <= 0 or args.rate_m3_s <= 0:
        ap.error("geology count, stages, stage duration, and injection rate must be positive")
    if args.coarsen_x < 1 or args.coarsen_z < 1:
        ap.error("coarsening factors must be positive integers")
    input_dir, out_dir = Path(args.input_dir), Path(args.out_dir)
    archive = input_dir / "co2_geology_cohorts.npz"
    if not archive.exists():
        raise FileNotFoundError(f"Missing geology archive: {archive}")
    schedule_map = _schedules(args.stages, args.rate_m3_s)
    all_rows = []
    with np.load(archive, allow_pickle=True) as npz:
        for region in ("tarbert", "upper_ness"):
            all_rows.extend(_save_region(npz, region, out_dir, args, schedule_map))
    with (out_dir / "pilot_stage_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    config_doc = {
        "model": "CO2XZSolver experimental 2D vertical x-z pilot",
        "physics_scope": "two-phase Darcy IMPES; constant densities/viscosities; anisotropic Kx/Kz; gravity; local finite-rate dissolution; Land-style hysteretic residual trapping; no capillary pressure; closed boundaries",
        "units": {"K": "mD", "length": "ft", "pressure": "Pa", "rate": "m3/s", "time": "s", "mass": "kg"},
        "parameters": vars(args),
        "schedule_names": list(schedule_map),
        "transitions": len(all_rows),
        "split_warning": "development pilot only; split future train/validation/test by geology seed; do not use this pilot as confirmatory evaluation or claim final CO2 surrogate training data",
        "limitations": ["not field calibrated", "not a fully coupled 3D model", "pressure equation uses constant total compressibility IMPES approximation", "trapping and dissolution laws are simplified and require expert review"]}
    (out_dir / "pilot_config.json").write_text(json.dumps(config_doc, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_rows)} solver-labelled stage transitions to {out_dir}")
    print("Pilot only: confirm mass-balance residuals and gravity checks before training; held-out geology evaluation still requires more independent seeds.")


if __name__ == "__main__":
    main()
