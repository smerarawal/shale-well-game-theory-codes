"""Compare fixed-substep CO2 pressure priors with validation labels.

The path mode here uses the labelled endpoint saturation and is explicitly an
oracle diagnostic. It estimates the benefit available if the saturation path
were known; it is not a deployable surrogate score. A learned endpoint predictor
must replace that oracle before reporting prior2 as a model result.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from .co2_pressure_prior import compare_prior_to_labels
    from .train_co2_xz_surrogate import COHORTS, load_cohort
except ImportError:
    from co2_pressure_prior import compare_prior_to_labels
    from train_co2_xz_surrogate import COHORTS, load_cohort


def _subset(data, count: int, seed: int):
    n = len(data["p_pa_in"])
    if n <= count:
        return data
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(n, size=count, replace=False))
    return {key: (value[ids] if isinstance(value, np.ndarray) and value.ndim > 0
                  and len(value) == n else value)
            for key, value in data.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cohorts", nargs="+", default=list(COHORTS), choices=COHORTS)
    parser.add_argument("--n-sub", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--samples-per-cohort", type=int, default=24)
    parser.add_argument("--stage-days", type=float, default=30.0)
    parser.add_argument("--gravity-sign", type=int, default=1, choices=[-1, 0, 1])
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    if args.samples_per_cohort < 1 or any(n < 1 for n in args.n_sub):
        parser.error("sample count and substep counts must be positive")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cohort_index, cohort in enumerate(args.cohorts):
        # Validation only: substep choice is not made from the test split.
        data = load_cohort(args.data_dir, "validation", cohort)
        data = _subset(data, args.samples_per_cohort, args.seed + cohort_index)
        endpoint = {key: data[f"{key}_out"] for key in
                    ("sg_mobile", "sg_trapped", "sg_max", "c_dissolved_kg_m3", "p_pa")}
        for n_sub in args.n_sub:
            for mode in ("frozen", "path"):
                scores = compare_prior_to_labels(
                    data, n_sub=n_sub, mode=mode,
                    endpoint=endpoint if mode == "path" else None,
                    stage_days=args.stage_days, gravity_sign=args.gravity_sign)
                rows.append({"cohort": cohort, "n_sub": n_sub,
                             "mode": mode,
                             "path_endpoint": "label_oracle" if mode == "path" else "start_state",
                             "transitions": len(data["p_pa_in"]), **scores})
                print(f"{cohort:14s} n_sub={n_sub:2d} {mode:6s} "
                      f"p_RMSE={scores['pressure_rmse_pa']:.4g} Pa "
                      f"pmax_MAE={scores['pmax_abs_mean_pa']:.4g} Pa",
                      flush=True)

    json_path = out_dir / "pressure_prior_diagnostic.json"
    csv_path = out_dir / "pressure_prior_diagnostic.csv"
    json_path.write_text(json.dumps({"purpose": "validation-only pressure-prior diagnostic",
                                    "path_mode_note": "label-endpoint oracle; not a deployable prior2 score",
                                    "substeps": args.n_sub, "rows": rows}, indent=2),
                         encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
