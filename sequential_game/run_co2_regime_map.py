"""Paired CO2 geology x institution regime-map experiment (solver reference).

Runs a coarse iterated-best-response approximation and a coordinated joint
schedule on each selected training geology, with identical wells, actions,
capacity, seeds, and geology across instruments. Outputs a row per
geology/instrument, including both Coop-Nash gap and noncooperative welfare.

Start with ``--max-geologies-per-cohort 1`` to estimate cost. The physical and
economic coefficients in --config must be reviewed/calibrated before treating
any rankings as substantive. This is a 2D x-z pilot, not a 3D study.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .co2_research_studies import geology_descriptors
    from .co2_solver_game_search import (
        coordinate_best_responses,
        coordinate_best_responses_multistart,
        coordinate_cooperative_plan,
    )
except ImportError:
    from co2_research_studies import geology_descriptors
    from co2_solver_game_search import (
        coordinate_best_responses,
        coordinate_best_responses_multistart,
        coordinate_cooperative_plan,
    )


def _instrument_arms(config: dict[str, Any], names: list[str], prices: list[float]) -> list[tuple[str, dict[str, Any]]]:
    arms = []
    for name in names:
        arm = copy.deepcopy(config)
        if name in {"proportional", "cea", "cel", "fixed_equal"}:
            arm["pipeline"]["dispatch_rule"] = name
            arms.append((name, arm))
        elif name == "price":
            for price in prices:
                priced = copy.deepcopy(arm)
                priced["pipeline"]["tariff_per_kg"] = float(price)
                arms.append((f"price_{price:g}", priced))
        else:
            raise ValueError(f"unknown instrument {name}")
    return arms


def _bootstrap_mean_ci(values: list[float], *, rng: np.random.Generator,
                       reps: int = 2000) -> list[float] | None:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        return None
    samples = rng.choice(array, size=(reps, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def summarize_paired_results(rows: list[dict[str, Any]], *, seed: int = 0) -> dict[str, Any]:
    """Geology-level means/CIs and paired policy contrasts within each cohort."""
    rng = np.random.default_rng(seed)
    by_key: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((str(row["cohort"]), str(row["instrument"])), {})[
            int(row["geology_seed"])
        ] = row
    cohorts = sorted({key[0] for key in by_key})
    instruments = sorted({key[1] for key in by_key})
    summaries = []
    paired_contrasts = []
    for cohort in cohorts:
        for instrument in instruments:
            cases = by_key.get((cohort, instrument), {})
            if not cases:
                continue
            gaps = [float(row["commons_gap_coop_minus_nash"]) for row in cases.values()]
            nash = [float(row["nash_like_welfare"]) for row in cases.values()]
            coop = [float(row["cooperative_welfare"]) for row in cases.values()]
            summaries.append({
                "cohort": cohort,
                "instrument": instrument,
                "paired_geologies_n": len(cases),
                "mean_commons_gap_coop_minus_nash": float(np.mean(gaps)),
                "median_commons_gap": float(np.median(gaps)),
                "gap_mean_95pct_bootstrap_ci": _bootstrap_mean_ci(gaps, rng=rng),
                "positive_gap_fraction": float(np.mean(np.asarray(gaps) > 0)),
                "mean_nash_like_welfare": float(np.mean(nash)),
                "mean_cooperative_welfare": float(np.mean(coop)),
                "interpretation": "exploratory if geology count is below the preregistered precision target",
            })
        baseline = by_key.get((cohort, "proportional"), {})
        for instrument in instruments:
            if instrument == "proportional":
                continue
            treatment = by_key.get((cohort, instrument), {})
            common = sorted(set(baseline).intersection(treatment))
            if not common:
                continue
            differences = [
                float(treatment[g]["nash_like_welfare"])
                - float(baseline[g]["nash_like_welfare"])
                for g in common
            ]
            paired_contrasts.append({
                "cohort": cohort,
                "contrast": f"{instrument} minus proportional in Nash-like welfare",
                "paired_geologies_n": len(common),
                "mean_paired_difference": float(np.mean(differences)),
                "mean_difference_95pct_bootstrap_ci": _bootstrap_mean_ci(differences, rng=rng),
                "positive_difference_fraction": float(np.mean(np.asarray(differences) > 0)),
            })
    return {
        "summary_by_cohort_and_instrument": summaries,
        "paired_instrument_contrasts": paired_contrasts,
        "bootstrap_unit": "independent geology seed",
        "bootstrap_replicates": 2000,
        "seed": int(seed),
    }


def run(data_dir: Path, config_path: Path, out_path: Path, *, split: str,
        max_geologies_per_cohort: int, rate_max_m3_s: float, rate_levels: int,
        response_rounds: int, coop_sweeps: int, instruments: list[str],
        prices: list[float], diagnose_final: bool = False,
        equilibrium_starts: int = 1, rng_seed: int = 0) -> list[dict[str, Any]]:
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    arms = _instrument_arms(base_config, instruments, prices)
    cohorts = ("friendly", "intermediate", "challenging", "tarbert", "upper_ness")
    rows = []
    counterfactual_rows = []
    for cohort in cohorts:
        path = data_dir / split / f"{cohort}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        first_rows = []
        seen = set()
        for i in np.flatnonzero(data["stage_index"] == 0):
            geology_seed = int(data["geology_seed"][i])
            if geology_seed not in seen:
                seen.add(geology_seed); first_rows.append((geology_seed, int(i)))
        first_rows = first_rows[:max_geologies_per_cohort]
        for geology_seed, row_id in first_rows:
            geo = {key: data[key][row_id] for key in ("kx_mD", "kz_mD", "phi")}
            desc = geology_descriptors(geo["kx_mD"], geo["phi"], geo["kz_mD"],
                                       dx_ft=float(data["dx_ft"][row_id]),
                                       dz_ft=float(data["dz_ft"][row_id]))
            for arm_name, arm_config in arms:
                # Preserve the grid spacing recorded with this particular
                # geology; SPE10 Tarbert and Ness have different vertical extents.
                arm_config.setdefault("solver", {})["dx_ft"] = float(data["dx_ft"][row_id])
                arm_config["solver"]["dy_ft"] = float(data["dy_ft"][row_id])
                arm_config["solver"]["dz_ft"] = float(data["dz_ft"][row_id])
                t0 = time.perf_counter()
                if equilibrium_starts > 1:
                    nash = coordinate_best_responses_multistart(
                        geo, arm_config, rate_max_m3_s=rate_max_m3_s,
                        rate_levels=rate_levels, response_rounds=response_rounds,
                        starts=equilibrium_starts, seed=rng_seed + geology_seed,
                        cohort=cohort, geology_seed=geology_seed, split=split)
                else:
                    nash = coordinate_best_responses(
                        geo, arm_config, rate_max_m3_s=rate_max_m3_s,
                        rate_levels=rate_levels, response_rounds=response_rounds,
                        diagnose_final=diagnose_final, cohort=cohort,
                        geology_seed=geology_seed, split=split)
                coop = coordinate_cooperative_plan(
                    geo, arm_config, rate_max_m3_s=rate_max_m3_s,
                    rate_levels=rate_levels, sweeps=coop_sweeps,
                    cohort=cohort, geology_seed=geology_seed, split=split)
                for policy_type, result in (("nash_like", nash), ("cooperative", coop)):
                    for counterfactual in result["counterfactuals"]:
                        counterfactual["state_id"] = f"{arm_name}_{counterfactual['state_id']}"
                        counterfactual["instrument"] = arm_name
                        counterfactual["policy_type"] = policy_type
                        counterfactual_rows.append(counterfactual)
                result_row = {
                    "cohort": cohort, "geology_seed": geology_seed, "instrument": arm_name,
                    "price_per_kg": (float(arm_name.removeprefix("price_"))
                                     if arm_name.startswith("price_") else None),
                    "nash_like_welfare": nash["project_resource_welfare"],
                    "project_resource_welfare": nash["project_resource_welfare"],
                    "cooperative_welfare": coop["project_resource_welfare"],
                    "commons_gap_coop_minus_nash": coop["project_resource_welfare"] - nash["project_resource_welfare"],
                    "best_instrument_by_nash_welfare": "",
                    "best_instrument_by_smallest_gap": "",
                    "nash_operator_payoffs_json": json.dumps(nash["operator_payoff"]),
                    "cooperative_operator_payoffs_json": json.dumps(coop["operator_payoff"]),
                    "nash_rate_schedule_json": json.dumps(nash["rates_m3_s_by_stage_operator"]),
                    "cooperative_rate_schedule_json": json.dumps(coop["rates_m3_s_by_stage_operator"]),
                    "nash_equilibrium_diagnostics_json": json.dumps(nash["equilibrium_diagnostics"]),
                    "nash_multi_start_diagnostics_json": json.dumps(nash.get("multi_start_diagnostics")),
                    "nash_solver_evaluations": nash["solver_evaluations"],
                    "cooperative_solver_evaluations": coop["solver_evaluations"],
                    "runtime_seconds": time.perf_counter() - t0,
                    **desc,
                }
                for operator, payoff in enumerate(nash["operator_payoff"]):
                    result_row[f"nash_operator_{operator}_payoff"] = payoff
                for operator, payoff in enumerate(coop["operator_payoff"]):
                    result_row[f"cooperative_operator_{operator}_payoff"] = payoff
                rows.append(result_row)
                print(f"{cohort} seed={geology_seed} arm={arm_name} gap={rows[-1]['commons_gap_coop_minus_nash']:.6g} "
                      f"runtime={rows[-1]['runtime_seconds']:.1f}s", flush=True)
    # Instrument choice is evaluated within a cohort using Nash welfare as the
    # principal deployment criterion; gap minimization is retained separately.
    by_cohort: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_cohort.setdefault(row["cohort"], []).append(row)
    for cohort_rows in by_cohort.values():
        score_nash: dict[str, list[float]] = {}
        score_gap: dict[str, list[float]] = {}
        for row in cohort_rows:
            score_nash.setdefault(row["instrument"], []).append(float(row["nash_like_welfare"]))
            score_gap.setdefault(row["instrument"], []).append(float(row["commons_gap_coop_minus_nash"]))
        best_welfare = max(score_nash, key=lambda name: float(np.mean(score_nash[name])))
        smallest_gap = min(score_gap, key=lambda name: float(np.mean(score_gap[name])))
        for row in cohort_rows:
            row["best_instrument_by_nash_welfare"] = best_welfare
            row["best_instrument_by_smallest_gap"] = smallest_gap
    if not rows:
        raise ValueError("no geologies found; check --data-dir and --split")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    if counterfactual_rows:
        cf_path = out_path.with_name(out_path.stem + "_counterfactuals.csv")
        with cf_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(counterfactual_rows[0]))
            writer.writeheader(); writer.writerows(counterfactual_rows)
    summary = summarize_paired_results(rows, seed=rng_seed)
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    ap.add_argument("--max-geologies-per-cohort", type=int, default=1)
    ap.add_argument("--rate-max-m3-s", type=float, required=True)
    ap.add_argument("--rate-levels", type=int, default=5)
    ap.add_argument("--response-rounds", type=int, default=2)
    ap.add_argument("--coop-sweeps", type=int, default=2)
    ap.add_argument("--diagnose-final", action="store_true",
                    help="measure final unilateral rate-grid deviation gains; adds solver rollouts")
    ap.add_argument("--equilibrium-starts", type=int, default=1,
                    help="number of initial schedules for convergence/stability diagnostics")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instruments", nargs="+", default=["proportional", "cea", "cel", "fixed_equal", "price"],
                    choices=["proportional", "cea", "cel", "fixed_equal", "price"])
    ap.add_argument("--prices", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0])
    args = ap.parse_args()
    if args.max_geologies_per_cohort < 1:
        ap.error("max-geologies-per-cohort must be >= 1")
    rows = run(args.data_dir, args.config, args.out, split=args.split,
               max_geologies_per_cohort=args.max_geologies_per_cohort,
               rate_max_m3_s=args.rate_max_m3_s, rate_levels=args.rate_levels,
               response_rounds=args.response_rounds, coop_sweeps=args.coop_sweeps,
               instruments=args.instruments, prices=args.prices,
               diagnose_final=args.diagnose_final,
               equilibrium_starts=args.equilibrium_starts, rng_seed=args.seed)
    print(f"Wrote {len(rows)} paired geology/instrument results to {args.out}")


if __name__ == "__main__":
    main()
