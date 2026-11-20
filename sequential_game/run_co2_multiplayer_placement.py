"""Solver-based CO2 placement game for more than two operators.

Example:
  python sequential_game/run_co2_multiplayer_placement.py \
    --data-dir /kaggle/input/notebooks/smerarawal/transition-geology/co2_xz_dataset_v2 \
    --out-dir /kaggle/working/co2_multiplayer_placement --n-operators 8 \
    --cohorts challenging

This runner uses exhaustive one-stage action profiles. Keep the joint profile
count modest; it grows as (actions per operator) ** n_operators. Current CO2
surrogates are two-operator/fixed-cell models and are not used for this
multi-operator location experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

try:
    from .co2_equilibrium_lab import (
        analyze_n_player,
        evaluate_solver_stage_game,
        make_placement_action_set,
        n_player_best_response_dynamics,
        summarize_dynamics,
    )
    from .co2_pipeline_game import CO2GameConfig, CO2PayoffConfig, PipelineConfig
    from .co2_well_placement_game import CO2WellPlacementGame, PlacementRules
    from .co2_xz_solver import CO2XZConfig, CO2XZSolver
except ImportError:
    from co2_equilibrium_lab import (
        analyze_n_player, evaluate_solver_stage_game, make_placement_action_set,
        n_player_best_response_dynamics, summarize_dynamics,
    )
    from co2_pipeline_game import CO2GameConfig, CO2PayoffConfig, PipelineConfig
    from co2_well_placement_game import CO2WellPlacementGame, PlacementRules
    from co2_xz_solver import CO2XZConfig, CO2XZSolver


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_geologies(data_dir: Path, cohorts: list[str], per_cohort: int):
    items = []
    for cohort in cohorts:
        path = data_dir / "test" / f"{cohort}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as z:
            stages = np.flatnonzero(z["stage_index"] == 0)
            seen = set()
            for i in stages:
                seed = int(z["geology_seed"][i])
                if seed in seen:
                    continue
                seen.add(seed)
                items.append({
                    "cohort": cohort,
                    "geology_seed": seed,
                    "geology_index": int(z["geology_index"][i]),
                    "kx_mD": np.asarray(z["kx_mD"][i], dtype=np.float64),
                    "kz_mD": np.asarray(z["kz_mD"][i], dtype=np.float64),
                    "phi": np.asarray(z["phi"][i], dtype=np.float64),
                    "dx_ft": float(z["dx_ft"][i]),
                    "dy_ft": float(z["dy_ft"][i]),
                    "dz_ft": float(z["dz_ft"][i]),
                })
                if len([x for x in items if x["cohort"] == cohort]) >= per_cohort:
                    break
    if not items:
        raise RuntimeError("No held-out stage-0 geologies found")
    return items


def candidate_sites(n_ops: int, shape: tuple[int, int], dx_ft: float,
                    min_spacing_ft: float):
    nz, nx = shape
    if nx < 2 * n_ops + 2:
        raise ValueError(f"grid width {nx} is too small for {n_ops} operators")
    x_positions = np.rint(np.linspace(1, nx - 2, 2 * n_ops)).astype(int)
    if len(set(x_positions.tolist())) != len(x_positions):
        raise ValueError("candidate sites collide; reduce operators or spacing")
    if len(x_positions) > 1 and np.min(np.diff(x_positions)) * dx_ft < min_spacing_ft:
        raise ValueError(
            "candidate sites cannot meet min spacing on this grid; reduce operator count "
            "or set a physically justified smaller --min-spacing-ft"
        )
    z = nz - 2
    return tuple(((z, int(x_positions[2 * op])),
                  (z, int(x_positions[2 * op + 1]))) for op in range(n_ops))


def make_game(geo, n_ops, *, stage_days, full_rate, capacity_fraction,
              dispatch_rule, pressure_damage, pressure_cap_pa, solver_parameters,
              min_spacing_ft):
    seconds = stage_days * 86400.0
    rho = 650.0
    cap = n_ops * full_rate * seconds * rho * capacity_fraction
    sp = solver_parameters
    solver = CO2XZSolver(CO2XZConfig(
        dx_ft=geo["dx_ft"], dy_ft=geo["dy_ft"], dz_ft=geo["dz_ft"],
        solubility_kg_m3_brine=float(sp.get("solubility_kg_m3", 30.0)),
        dissolution_rate_s_inv=float(sp.get("dissolution_rate_s", 2e-7)),
        land_coefficient=float(sp.get("land_coefficient", 2.0)),
        max_dt_s=float(sp.get("max_dt_days", 1.0)) * 86400.0,
        max_steps=int(sp.get("max_steps", 20000)),
    ))
    config = CO2GameConfig(
        n_operators=n_ops, horizon=1, stage_duration_s=seconds,
        discount_factor=1.0,
        pipeline=PipelineConfig(capacity_kg_per_stage=cap,
                                source_supply_kg_per_stage=cap,
                                dispatch_rule=dispatch_rule),
        payoff=CO2PayoffConfig(
            storage_value_per_kg=1.0,
            pressure_cap_pa=pressure_cap_pa,
            pressure_damage_per_pa2=pressure_damage,
        ),
    )
    sites = candidate_sites(n_ops, geo["phi"].shape, geo["dx_ft"], min_spacing_ft)
    rules = PlacementRules(
        candidate_cells_by_operator=tuple(frozenset(x) for x in sites),
        min_spacing_ft=min_spacing_ft,
        max_wells_per_operator=1,
        max_new_wells_per_operator_per_stage=1,
        max_rate_m3_s_per_well=full_rate,
        drilling_lead_stages=0,
    )
    return CO2WellPlacementGame(solver=solver, config=config,
                                placement_rules=rules), sites


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if hasattr(x, "__dataclass_fields__"):
        return {k: jsonable(getattr(x, k)) for k in x.__dataclass_fields__}
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-operators", type=int, default=8)
    ap.add_argument("--cohorts", nargs="+", default=["challenging"])
    ap.add_argument("--geologies-per-cohort", type=int, default=1)
    ap.add_argument("--dispatch-rules", nargs="+", default=["proportional"])
    ap.add_argument("--capacity-fractions", nargs="+", type=float, default=[0.5])
    ap.add_argument("--pressure-damages", nargs="+", type=float, default=[1e-9])
    ap.add_argument("--pressure-cap-mpa", type=float, default=5.0)
    ap.add_argument("--min-spacing-ft", type=float, default=50.0)
    ap.add_argument("--max-joint-profiles", type=int, default=10000)
    ap.add_argument("--cvar-alpha", type=float, default=0.25,
                    help="lower-tail fraction for risk-sensitive payoff summaries")
    ap.add_argument("--risk-weights", nargs="+", type=float, default=[0.0, 0.5, 1.0],
                    help="blend mean payoff (0) with lower-tail CVaR payoff (1)")
    ap.add_argument("--starts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261121)
    a = ap.parse_args()
    if a.n_operators < 2:
        raise ValueError("use --n-operators 2 or more")
    data_dir = Path(a.data_dir).expanduser().resolve()
    out_dir = Path(a.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = data_dir / "dataset_config.json"
    dataset_config = json.loads(config_path.read_text(encoding="utf-8"))
    params = dataset_config.get("parameters", {})
    geologies = load_geologies(data_dir, a.cohorts, a.geologies_per_cohort)
    if not 0 < a.cvar_alpha <= 1:
        raise ValueError("--cvar-alpha must be in (0, 1]")
    stage_days = float(params.get("stage_days", 30.0))
    full_rate = float(params.get("rate_m3_s", 1e-7))
    arms = []
    for dispatch, capacity_fraction, damage in itertools.product(
            a.dispatch_rules, a.capacity_fractions, a.pressure_damages):
        per_geo = []
        for geo in geologies:
            game, sites = make_game(
                geo, a.n_operators, stage_days=stage_days, full_rate=full_rate,
                capacity_fraction=capacity_fraction, dispatch_rule=dispatch,
                pressure_damage=damage, pressure_cap_pa=a.pressure_cap_mpa * 1e6,
                solver_parameters=params, min_spacing_ft=a.min_spacing_ft)
            # Three actions per operator: do not drill, or drill at either
            # candidate site and inject at the configured full rate.
            action_sets = [
                make_placement_action_set(op, sites[op], [full_rate])
                for op in range(a.n_operators)
            ]
            profile_count = int(np.prod([len(x) for x in action_sets]))
            if profile_count > a.max_joint_profiles:
                raise ValueError(
                    f"{a.n_operators} operators × {len(action_sets[0])} actions gives "
                    f"{profile_count} joint profiles, above --max-joint-profiles="
                    f"{a.max_joint_profiles}. Reduce operators/actions or raise the cap."
                )
            result = evaluate_solver_stage_game(
                game, game.initial_state(geo["phi"].shape), geo, action_sets)
            per_geo.append({"cohort": geo["cohort"],
                            "geology_seed": geo["geology_seed"],
                            "geology_index": geo["geology_index"],
                            "sites_by_operator": sites,
                            "result": result})
            print(f"Done {dispatch=} {capacity_fraction=} {damage=} "
                  f"{geo['cohort']=} {geo['geology_seed']=}; profiles={profile_count}",
                  flush=True)
        payoff_stack = np.stack([x["result"]["payoffs"] for x in per_geo])
        welfare_stack = np.stack([x["result"]["resource_welfare"] for x in per_geo])
        expected_payoffs = payoff_stack.mean(axis=0)
        expected_welfare = welfare_stack.mean(axis=0)
        eq = analyze_n_player(expected_payoffs)
        paths = n_player_best_response_dynamics(
            expected_payoffs, starts=a.starts, update="alternating", seed=a.seed)
        risk_summaries = []
        tail_n = max(1, int(np.ceil(a.cvar_alpha * payoff_stack.shape[0])))
        lower_tail_payoffs = np.sort(payoff_stack, axis=0)[:tail_n].mean(axis=0)
        payoff_std = payoff_stack.std(axis=0, ddof=1) if payoff_stack.shape[0] > 1 \
            else np.zeros_like(expected_payoffs)
        for risk_weight in a.risk_weights:
            if not 0.0 <= risk_weight <= 1.0:
                raise ValueError("each --risk-weights value must lie in [0,1]")
            adjusted = ((1.0 - risk_weight) * expected_payoffs
                        + risk_weight * lower_tail_payoffs)
            risk_eq = analyze_n_player(adjusted)
            risk_summaries.append({
                "risk_weight_on_lower_tail_cvar": float(risk_weight),
                "pure_nash_profiles": risk_eq["pure_nash_profiles"],
                "welfare_maximizers_under_adjusted_payoffs": risk_eq["welfare_maximizers"],
                "payoffs_for_game_solution": adjusted,
            })
        arms.append({
            "dispatch_rule": dispatch,
            "capacity_fraction_of_n_full_rate_wells": capacity_fraction,
            "pressure_damage_per_pa2": damage,
            "pressure_cap_pa": a.pressure_cap_mpa * 1e6,
            "n_operators": a.n_operators,
            "actions_per_operator": len(action_sets[0]),
            "joint_profile_count": profile_count,
            "expected_payoff_pure_nash_profiles": eq["pure_nash_profiles"],
            "expected_welfare_maximizers": eq["welfare_maximizers"],
            "best_response_convergence": summarize_dynamics(paths),
            "expected_payoffs": expected_payoffs,
            "lower_tail_cvar_payoffs": lower_tail_payoffs,
            "payoff_std_across_geologies": payoff_std,
            "risk_attitude_equilibria": risk_summaries,
            "expected_resource_welfare": expected_welfare,
            "per_geology": per_geo,
        })
    result = {
        "provenance": {
            "dataset_dir": str(data_dir),
            "dataset_config_sha256": sha256(config_path),
            "dataset_description": dataset_config.get("description"),
            "n_operators": a.n_operators,
            "max_site_candidates_per_operator": 2,
            "candidate_menu": "out or one of two candidate sites at full rate",
            "solution_concept": "exact pure Nash for enumerated one-stage N-player action games; best-response paths are empirical convergence diagnostics",
            "surrogate": "not used: current merged checkpoint has two fixed operator rate channels",
            "brine_separation": "CO2 modules and metrics only",
        },
        "options": vars(a),
        "geology_manifest": [{k: g[k] for k in (
            "cohort", "geology_seed", "geology_index", "dx_ft", "dy_ft", "dz_ft")}
            for g in geologies],
        "arms": arms,
    }
    output = out_dir / "co2_multiplayer_placement.json"
    output.write_text(json.dumps(jsonable(result), indent=2), encoding="utf-8")
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
