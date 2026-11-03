"""Solver-reference evaluation and small discrete best-response search.

This provides a solver environment for checking differentiable-surrogate plans.
It is intentionally a coarse, auditable search, not an exact equilibrium
solver. The JSON config must state all capacity/payoff assumptions. Every
candidate schedule is rolled out using the actual CO2 x-z solver and game
transition kernel.

Config example:
{
  "horizon": 6, "stage_duration_s": 2592000, "discount_factor": 0.95238,
  "pipeline": {"capacity_kg_per_stage": 1000, "dispatch_rule": "proportional"},
  "payoff": {"storage_value_per_kg": 1, "pressure_cap_pa": 2000000,
              "pressure_damage_per_pa2": 1e-10},
  "wells_by_operator": [[10, 4], [10, 8]],
  "solver": {"dx_ft": 20, "dz_ft": 2, "dy_ft": 10, "max_dt_s": 86400}
}

Examples:
  python co2_solver_game_search.py evaluate --geology geo.npz --config game.json \\
      --schedule plan.json --out solver_check.json
  python co2_solver_game_search.py search --geology geo.npz --config game.json \\
      --cohort challenging --geology-seed 41000 --rate-max-m3-s 1e-7 \\
      --rate-levels 5 --response-rounds 2 --out solver_nash_like.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from .co2_pipeline_game import (CO2GameConfig, CO2PayoffConfig, OperatorAction,
                                    PipelineConfig, advance_stage, new_game_state)
    from .co2_xz_solver import CO2XZConfig, CO2XZSolver
except ImportError:
    from co2_pipeline_game import (CO2GameConfig, CO2PayoffConfig, OperatorAction,
                                   PipelineConfig, advance_stage, new_game_state)
    from co2_xz_solver import CO2XZConfig, CO2XZSolver


def make_config(raw: Mapping[str, Any]) -> tuple[CO2GameConfig, CO2XZConfig, tuple[tuple[int, int], ...]]:
    pipe_raw = dict(raw["pipeline"])
    payoff_raw = dict(raw.get("payoff", {}))
    game = CO2GameConfig(
        n_operators=len(raw["wells_by_operator"]), horizon=int(raw.get("horizon", 6)),
        stage_duration_s=float(raw.get("stage_duration_s", 30 * 86400)),
        discount_factor=float(raw.get("discount_factor", 1 / 1.05)),
        storage_cap_kg=raw.get("storage_cap_kg"),
        storage_dispatch_rule=raw.get("storage_dispatch_rule", "proportional"),
        pipeline=PipelineConfig(**pipe_raw), payoff=CO2PayoffConfig(**payoff_raw))
    solver = CO2XZConfig(**dict(raw.get("solver", {})))
    wells = tuple(tuple(map(int, c)) for c in raw["wells_by_operator"])
    if len(wells) < 2:
        raise ValueError("config must provide at least two well cells, one per operator")
    return game, solver, wells


def evaluate_schedule(geology: Mapping[str, np.ndarray], raw_config: Mapping[str, Any],
                      rates: np.ndarray, *, gravity_sign: int = 1) -> dict[str, Any]:
    game_cfg, solver_cfg, wells = make_config(raw_config)
    n_ops = game_cfg.n_operators
    if rates.shape != (game_cfg.horizon, n_ops) or np.any(~np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError(f"schedule must be nonnegative [horizon={game_cfg.horizon},operators={n_ops}]")
    solver = CO2XZSolver(solver_cfg)
    shape = np.asarray(geology["phi"]).shape
    for z, x in wells:
        if not (0 <= z < shape[0] and 0 <= x < shape[1]):
            raise ValueError(f"well {(z, x)} is outside grid {shape}")
    state = new_game_state(solver, shape, game_cfg.n_operators)
    stage_records = []
    start = time.perf_counter()
    for stage, pair in enumerate(rates):
        pressure_before = float(np.max(state.reservoir["p_pa"]))
        plume_before = float(np.mean(state.reservoir["sg_mobile"] >= 0.01))
        permit_remaining = (None if game_cfg.storage_cap_kg is None else
                            max(0.0, float(game_cfg.storage_cap_kg - state.cumulative_injected_kg_by_operator.sum())))
        actions = []
        for op in range(n_ops):
            z, x = wells[op]
            well_action = ((z, x, float(pair[op])),) if pair[op] > 0 else ()
            new = (wells[op],) if stage == 0 else ()
            actions.append(OperatorAction(wells=well_action, new_wells=new))
        state, record = advance_stage(solver=solver, config=game_cfg, state=state,
                                      geology=geology, actions=actions,
                                      gravity_sign=gravity_sign)
        diag = record["solver_diagnostics"]
        stage_records.append({"stage": stage,
                              "pressure_before_pa": pressure_before,
                              "plume_fraction_before": plume_before,
                              "permit_remaining_kg": permit_remaining,
                              "pressure_max_pa": record["pressure_max_pa"],
                              "injected_kg_by_operator": record["storage_accepted_kg_by_operator"],
                              "operator_payoff": record["discounted_operator_payoff_by_operator"],
                              "operator_payoff_before_damage_liability": (
                                  np.asarray(record["discounted_operator_payoff_by_operator"], dtype=float)
                                  + np.asarray(record["damage_liability_by_operator"], dtype=float)
                                  * float(record["discount_factor_applied"])
                              ).tolist(),
                              "damage_liability_by_operator": record["damage_liability_by_operator"],
                              "shared_damage_cost": record["shared_damage_cost"],
                              "discounted_project_resource_welfare": record["discounted_project_resource_welfare"],
                              "mass_balance_residual_kg": diag["mass_balance_residual_kg"],
                              "mobile_co2_kg": diag["mobile_gas_kg"],
                              "trapped_co2_kg": diag["trapped_gas_kg"],
                              "dissolved_co2_kg": diag["dissolved_kg"],
                              "top_boundary_mobile_gas_fraction": diag["top_boundary_mobile_gas_fraction"]})
    return {"rates_m3_s_by_stage_operator": rates.tolist(),
            "operator_payoff": state.cumulative_payoff_by_operator.tolist(),
            "project_resource_welfare": float(state.cumulative_welfare),
            "final_injected_kg_by_operator": state.cumulative_injected_kg_by_operator.tolist(),
            "runtime_seconds": float(time.perf_counter() - start),
            "stage_records": stage_records,
            "solution_label": "solver_rollout; no equilibrium claim"}


def coordinate_best_responses(geology: Mapping[str, np.ndarray], raw_config: Mapping[str, Any],
                              *, rate_max_m3_s: float, rate_levels: int,
                              response_rounds: int, initial: np.ndarray | None = None,
                              diagnose_final: bool = False,
                              cohort: str = "", geology_seed: int | str = "",
                              split: str = "train") -> dict[str, Any]:
    game_cfg, _, _ = make_config(raw_config)
    if rate_max_m3_s <= 0 or rate_levels < 2 or response_rounds < 1:
        raise ValueError("rate maximum must be positive, levels >=2 and rounds >=1")
    levels = np.linspace(0.0, rate_max_m3_s, rate_levels)
    n_ops = game_cfg.n_operators
    schedule = (np.full((game_cfg.horizon, n_ops), rate_max_m3_s * 0.5, dtype=float)
                if initial is None else np.asarray(initial, dtype=float).copy())
    if schedule.shape != (game_cfg.horizon, n_ops):
        raise ValueError(f"initial schedule shape does not match horizon x {n_ops}")
    trace = []
    seen_schedules: dict[tuple[float, ...], int] = {}
    cycle_start_round = None
    counterfactuals = []
    evaluations = 0
    baseline = evaluate_schedule(geology, raw_config, schedule)
    evaluations += 1
    for round_id in range(response_rounds):
        signature = tuple(np.round(schedule.ravel(), 14).tolist())
        if signature in seen_schedules and cycle_start_round is None:
            cycle_start_round = seen_schedules[signature]
        seen_schedules.setdefault(signature, round_id + 1)
        improvements = []
        for player in range(n_ops):
            start_payoff = baseline["operator_payoff"][player]
            for stage in range(game_cfg.horizon):
                best_rate = schedule[stage, player]
                best_outcome = baseline
                best_value = baseline["operator_payoff"][player]
                context = baseline["stage_records"][stage]
                state_id = f"{cohort}_{geology_seed}_r{round_id}_p{player}_t{stage}"
                for action_index, candidate in enumerate(levels):
                    trial = schedule.copy()
                    trial[stage, player] = candidate
                    outcome = evaluate_schedule(geology, raw_config, trial)
                    evaluations += 1
                    value = outcome["operator_payoff"][player]
                    counterfactuals.append({
                        "state_id": state_id, "split": split, "cohort": cohort,
                        "geology_seed": geology_seed, "round": round_id + 1,
                        "player": player, "stage": stage,
                        "pressure_before_pa": context["pressure_before_pa"],
                        "plume_fraction_before": context["plume_fraction_before"],
                        "permit_remaining_kg": context["permit_remaining_kg"],
                        "action_id": f"rate_{action_index}",
                        "action_rate_m3_s": float(candidate),
                        "decision_payoff": float(value),
                    })
                    if value > best_value + 1e-10:
                        best_rate, best_value, best_outcome = float(candidate), float(value), outcome
                schedule[stage, player] = best_rate
                baseline = best_outcome
            improvements.append(float(baseline["operator_payoff"][player] - start_payoff))
        trace.append({"round": round_id + 1, "player_payoff_change": improvements,
                      "payoffs": baseline["operator_payoff"],
                      "schedule": schedule.tolist()})
    final = evaluate_schedule(geology, raw_config, schedule)
    evaluations += 1
    final_deviations = None
    if diagnose_final:
        final_deviations = []
        baseline_payoffs = np.asarray(final["operator_payoff"], dtype=float)
        for player in range(game_cfg.n_operators):
            for stage in range(game_cfg.horizon):
                best_gain = 0.0
                best_rate = float(schedule[stage, player])
                for candidate in levels:
                    trial = schedule.copy()
                    trial[stage, player] = candidate
                    outcome = evaluate_schedule(geology, raw_config, trial)
                    evaluations += 1
                    gain = float(outcome["operator_payoff"][player] - baseline_payoffs[player])
                    if gain > best_gain:
                        best_gain, best_rate = gain, float(candidate)
                final_deviations.append({
                    "operator": player, "stage": stage,
                    "best_found_gain": best_gain,
                    "best_deviation_rate_m3_s": best_rate,
                })
    return {**final, "coordinate_search_trace": trace,
            "counterfactuals": counterfactuals,
            "solver_evaluations": evaluations,
            "equilibrium_diagnostics": {
                "final_schedule_grid_deviations": final_deviations,
                "max_final_grid_deviation_gain": (
                    None if final_deviations is None else
                    max((row["best_found_gain"] for row in final_deviations), default=0.0)
                ),
                "cycle_start_round": cycle_start_round,
                "unique_round_start_profiles": len(seen_schedules),
                "status": "computed on declared rate grid" if diagnose_final else "not requested",
            },
            "search": {"rate_levels_m3_s": levels.tolist(), "response_rounds": response_rounds},
            "solution_label": "coarse coordinate iterated best response; not an exact Nash equilibrium"}


def coordinate_cooperative_plan(geology: Mapping[str, np.ndarray], raw_config: Mapping[str, Any],
                                *, rate_max_m3_s: float, rate_levels: int,
                                sweeps: int = 2, initial: np.ndarray | None = None,
                                cohort: str = "", geology_seed: int | str = "",
                                split: str = "train") -> dict[str, Any]:
    """Coordinate-search a joint schedule maximizing total resource welfare."""
    game_cfg, _, _ = make_config(raw_config)
    if rate_max_m3_s <= 0 or rate_levels < 2 or sweeps < 1:
        raise ValueError("rate maximum must be positive, levels >=2 and sweeps >=1")
    levels = np.linspace(0.0, rate_max_m3_s, rate_levels)
    n_ops = game_cfg.n_operators
    schedule = (np.full((game_cfg.horizon, n_ops), rate_max_m3_s * 0.5, dtype=float)
                if initial is None else np.asarray(initial, dtype=float).copy())
    if schedule.shape != (game_cfg.horizon, n_ops):
        raise ValueError(f"initial schedule shape does not match horizon x {n_ops}")
    evaluations = 0
    outcome = evaluate_schedule(geology, raw_config, schedule); evaluations += 1
    trace = []
    counterfactuals = []
    for sweep in range(sweeps):
        start = outcome["project_resource_welfare"]
        for stage in range(game_cfg.horizon):
            for player in range(n_ops):
                best_rate = schedule[stage, player]
                best_outcome = outcome
                best_welfare = outcome["project_resource_welfare"]
                context = outcome["stage_records"][stage]
                state_id = f"{cohort}_{geology_seed}_coop_s{sweep}_p{player}_t{stage}"
                for action_index, candidate in enumerate(levels):
                    trial = schedule.copy()
                    trial[stage, player] = candidate
                    candidate_outcome = evaluate_schedule(geology, raw_config, trial)
                    evaluations += 1
                    counterfactuals.append({
                        "state_id": state_id, "split": split, "cohort": cohort,
                        "geology_seed": geology_seed, "round": sweep + 1,
                        "player": player, "stage": stage,
                        "pressure_before_pa": context["pressure_before_pa"],
                        "plume_fraction_before": context["plume_fraction_before"],
                        "permit_remaining_kg": context["permit_remaining_kg"],
                        "action_id": f"rate_{action_index}",
                        "action_rate_m3_s": float(candidate),
                        "decision_payoff": float(candidate_outcome["project_resource_welfare"]),
                    })
                    if candidate_outcome["project_resource_welfare"] > best_welfare + 1e-10:
                        best_rate = float(candidate)
                        best_welfare = candidate_outcome["project_resource_welfare"]
                        best_outcome = candidate_outcome
                schedule[stage, player] = best_rate
                outcome = best_outcome
        trace.append({"sweep": sweep + 1,
                      "welfare_improvement": float(outcome["project_resource_welfare"] - start),
                      "project_resource_welfare": outcome["project_resource_welfare"]})
    return {**outcome, "coordinate_search_trace": trace,
            "counterfactuals": counterfactuals,
            "solver_evaluations": evaluations,
            "search": {"rate_levels_m3_s": levels.tolist(), "sweeps": sweeps},
            "solution_label": "cooperative joint schedule from coarse coordinate search; not a global optimum"}


def coordinate_best_responses_multistart(
    geology: Mapping[str, np.ndarray],
    raw_config: Mapping[str, Any],
    *,
    rate_max_m3_s: float,
    rate_levels: int,
    response_rounds: int,
    starts: int = 3,
    seed: int = 0,
    cohort: str = "",
    geology_seed: int | str = "",
    split: str = "test",
) -> dict[str, Any]:
    """Run best responses from low/high/mid/random schedules and report stability."""
    game_cfg, _, _ = make_config(raw_config)
    if starts < 1:
        raise ValueError("starts must be >= 1")
    levels = np.linspace(0.0, rate_max_m3_s, rate_levels)
    initial_profiles = [np.zeros((game_cfg.horizon, game_cfg.n_operators)),
                        np.full((game_cfg.horizon, game_cfg.n_operators), rate_max_m3_s),
                        np.full((game_cfg.horizon, game_cfg.n_operators), rate_max_m3_s * 0.5)]
    rng = np.random.default_rng(seed)
    while len(initial_profiles) < starts:
        indices = rng.integers(0, len(levels), size=(game_cfg.horizon, game_cfg.n_operators))
        initial_profiles.append(levels[indices])
    initial_profiles = initial_profiles[:starts]

    outcomes = []
    total_evaluations = 0
    for start_id, initial in enumerate(initial_profiles):
        result = coordinate_best_responses(
            geology, raw_config,
            rate_max_m3_s=rate_max_m3_s,
            rate_levels=rate_levels,
            response_rounds=response_rounds,
            initial=initial,
            diagnose_final=True,
            cohort=cohort,
            geology_seed=geology_seed,
            split=split,
        )
        total_evaluations += int(result["solver_evaluations"])
        outcomes.append({"start_id": start_id, **result})
    selected = min(
        outcomes,
        key=lambda item: (
            float(item["equilibrium_diagnostics"]["max_final_grid_deviation_gain"]),
            -float(item["project_resource_welfare"]),
        ),
    )
    schedules = [np.asarray(row["rates_m3_s_by_stage_operator"]) for row in outcomes]
    modal_match_fraction = max(
        sum(np.array_equal(candidate, other) for other in schedules) / len(schedules)
        for candidate in schedules
    )
    return {
        **selected,
        "solver_evaluations": total_evaluations,
        "multi_start_diagnostics": {
            "starts": starts,
            "seed": int(seed),
            "selected_start_id": selected["start_id"],
            "exact_schedule_match_fraction_to_modal_start": float(modal_match_fraction),
            "per_start": [
                {
                    "start_id": row["start_id"],
                    "project_resource_welfare": row["project_resource_welfare"],
                    "max_final_grid_deviation_gain": row["equilibrium_diagnostics"][
                        "max_final_grid_deviation_gain"
                    ],
                    "cycle_start_round": row["equilibrium_diagnostics"]["cycle_start_round"],
                    "rates_m3_s_by_stage_operator": row["rates_m3_s_by_stage_operator"],
                }
                for row in outcomes
            ],
        },
        "solution_label": (
            "best of finite multi-start coordinate best responses by lowest checked grid regret; "
            "not an exact Nash equilibrium"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "search", "cooperative"):
        sp = sub.add_parser(name)
        sp.add_argument("--geology", type=Path, required=True,
                        help="NPZ containing 2D kx_mD, kz_mD, phi fields")
        sp.add_argument("--row", type=int, default=0,
                        help="case index if fields are stored as [N,z,x]")
        sp.add_argument("--config", type=Path, required=True)
        sp.add_argument("--out", type=Path, required=True)
        if name == "evaluate":
            sp.add_argument("--schedule", type=Path, required=True)
        elif name in ("search", "cooperative"):
            sp.add_argument("--cohort", required=True)
            sp.add_argument("--geology-seed", type=int, required=True)
            sp.add_argument("--split", choices=("train", "validation", "test"), default="test")
            sp.add_argument("--rate-max-m3-s", type=float, required=True)
            sp.add_argument("--rate-levels", type=int, default=5)
            sp.add_argument("--initial-schedule", type=Path,
                            help="optional planner JSON to initialize solver response search")
            if name == "search":
                sp.add_argument("--response-rounds", type=int, default=2)
                sp.add_argument("--starts", type=int, default=1,
                                help="initial schedules; >1 enables low/high/mid/random diagnostics")
                sp.add_argument("--seed", type=int, default=0)
                sp.add_argument("--diagnose-final", action="store_true",
                                help="evaluate all unilateral stage-rate deviations from the final schedule")
            else:
                sp.add_argument("--sweeps", type=int, default=2)
    args = ap.parse_args()
    with np.load(args.geology, allow_pickle=False) as z:
        geology = {k: z[k] for k in ("kx_mD", "kz_mD", "phi")}
    for key, value in geology.items():
        if value.ndim == 3:
            if not 0 <= args.row < len(value):
                raise IndexError(f"--row {args.row} is outside {key} with {len(value)} cases")
            geology[key] = value[args.row]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.command == "evaluate":
        planned = json.loads(args.schedule.read_text(encoding="utf-8"))
        rates = np.asarray(planned["rates_m3_s_by_stage_operator"], dtype=float)
        result = evaluate_schedule(geology, config, rates)
        result.update({"planner_record": planned, "validation": "solver physical rollout of supplied schedule"})
    elif args.command == "search":
        initial = None
        if args.initial_schedule is not None:
            initial = np.asarray(json.loads(args.initial_schedule.read_text(encoding="utf-8"))[
                "rates_m3_s_by_stage_operator"], dtype=float)
        if args.starts > 1:
            if initial is not None:
                ap.error("--initial-schedule cannot be combined with --starts > 1")
            result = coordinate_best_responses_multistart(
                geology, config, rate_max_m3_s=args.rate_max_m3_s,
                rate_levels=args.rate_levels, response_rounds=args.response_rounds,
                starts=args.starts, seed=args.seed, cohort=args.cohort,
                geology_seed=args.geology_seed, split=args.split)
        else:
            result = coordinate_best_responses(geology, config,
                                               rate_max_m3_s=args.rate_max_m3_s,
                                               rate_levels=args.rate_levels,
                                               response_rounds=args.response_rounds,
                                               diagnose_final=args.diagnose_final,
                                               initial=initial,
                                               cohort=args.cohort,
                                               geology_seed=args.geology_seed,
                                               split=args.split)
        result.update({"cohort": args.cohort, "geology_seed": args.geology_seed})
    else:
        initial = None
        if args.initial_schedule is not None:
            initial = np.asarray(json.loads(args.initial_schedule.read_text(encoding="utf-8"))[
                "rates_m3_s_by_stage_operator"], dtype=float)
        result = coordinate_cooperative_plan(geology, config,
                                             rate_max_m3_s=args.rate_max_m3_s,
                                             rate_levels=args.rate_levels,
                                             sweeps=args.sweeps,
                                             initial=initial,
                                             cohort=args.cohort,
                                             geology_seed=args.geology_seed,
                                             split=args.split)
        result.update({"cohort": args.cohort, "geology_seed": args.geology_seed})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote solver-game result to {args.out}")


if __name__ == "__main__":
    main()
