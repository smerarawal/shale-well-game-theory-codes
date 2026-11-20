"""CO2-only equilibrium and well-placement pilot using held-out geologies.

Run from the repository root, for example:
  python sequential_game/run_co2_equilibrium_pilot.py \
    --data-dir /kaggle/input/notebooks/smerarawal/transition-geology/co2_xz_dataset_v2 \
    --out-dir /kaggle/working/co2_equilibrium_pilot \
    --pressure-tail-checkpoint /kaggle/input/.../ablation_pressure_tail/co2_ccs_gnn_s0.pt \
    --unroll3-checkpoint /kaggle/input/.../ablation_unroll3/co2_ccs_gnn_s0.pt

This is an explicit finite-action pilot. Its one-stage games are exact pure-NE
games for the enumerated actions. The surrogate planner is the fieldwise merge
of pressure-tail and unroll-3 CCS-GNN checkpoints. Best-response/fictitious/logit
trajectories are learning-dynamics diagnostics, not an exact multi-stage
Markov-perfect equilibrium. Placement is solver-scored because current CO2
checkpoints encode fixed injection cells only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .co2_equilibrium_lab import (
        aggregate_geology_games,
        analyze_bimatrix,
        best_response_dynamics,
        compare_planner_to_solver,
        evaluate_solver_stage_game,
        evaluate_surrogate_stage_game,
        fictitious_play,
        load_co2_checkpoint,
        logit_response_dynamics,
        make_placement_action_set,
        make_pressure_tail_unroll3_hybrid,
        summarize_dynamics,
    )
    from .co2_pipeline_game import (
        CO2GameConfig,
        CO2PayoffConfig,
        PipelineConfig,
    )
    from .co2_well_placement_game import (
        CO2WellPlacementGame,
        PlacementRules,
    )
    from .co2_xz_solver import CO2XZConfig, CO2XZSolver
except ImportError:
    from co2_equilibrium_lab import (
        aggregate_geology_games, analyze_bimatrix, best_response_dynamics,
        compare_planner_to_solver, evaluate_solver_stage_game,
        evaluate_surrogate_stage_game, fictitious_play,
        load_co2_checkpoint, logit_response_dynamics,
        make_placement_action_set, make_pressure_tail_unroll3_hybrid,
        summarize_dynamics,
    )
    from co2_pipeline_game import CO2GameConfig, CO2PayoffConfig, PipelineConfig
    from co2_well_placement_game import CO2WellPlacementGame, PlacementRules
    from co2_xz_solver import CO2XZConfig, CO2XZSolver


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_geologies(data_dir: Path, cohorts: list[str], limit: int) -> list[dict[str, Any]]:
    out = []
    for cohort in cohorts:
        path = data_dir / "test" / f"{cohort}.npz"
        if not path.is_file():
            print(f"Skip absent cohort: {path}", flush=True)
            continue
        with np.load(path, allow_pickle=False) as z:
            data = {key: z[key] for key in z.files}
        stage0 = np.flatnonzero(np.asarray(data["stage_index"]) == 0)
        chosen, seeds = [], set()
        for row in stage0:
            seed = int(data["geology_seed"][row])
            if seed in seeds:
                continue
            seeds.add(seed)
            chosen.append(int(row))
            if len(chosen) >= limit:
                break
        for row in chosen:
            out.append({
                "cohort": cohort,
                "geology_seed": int(data["geology_seed"][row]),
                "geology_index": int(data["geology_index"][row]),
                "kx_mD": np.asarray(data["kx_mD"][row], dtype=np.float64),
                "kz_mD": np.asarray(data["kz_mD"][row], dtype=np.float64),
                "phi": np.asarray(data["phi"][row], dtype=np.float64),
                "dx_ft": float(data["dx_ft"][row]),
                "dy_ft": float(data["dy_ft"][row]),
                "dz_ft": float(data["dz_ft"][row]),
                "schedule_name": str(data["schedule_name"][row]),
                "source_file": str(path),
            })
    if not out:
        raise RuntimeError("no stage-0 held-out geologies found in requested cohorts")
    return out


def _label_actions(actions, rates):
    labels = ["out"]
    for k, cell in enumerate(actions[1:]):
        labels.append(f"site{k}: {cell.placements[0].z_index},{cell.placements[0].x_index}; "
                      f"q={rates[k]:.3g} m3/s")
    return labels


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_jsonable(getattr(obj, k)) for k in obj.__dataclass_fields__}
    return obj


def _scenario_game(geo, *, stage_days, full_rate, capacity_fraction,
                   dispatch_rule, pressure_damage, pressure_cap_pa,
                   solver_parameters=None, max_wells_per_operator=1):
    seconds = stage_days * 86400.0
    rho = 650.0
    cap = 2.0 * full_rate * seconds * rho * capacity_fraction
    sp = solver_parameters or {}
    solver = CO2XZSolver(CO2XZConfig(
        dx_ft=geo["dx_ft"], dy_ft=geo["dy_ft"], dz_ft=geo["dz_ft"],
        solubility_kg_m3_brine=float(sp.get("solubility_kg_m3", 30.0)),
        dissolution_rate_s_inv=float(sp.get("dissolution_rate_s", 2.0e-7)),
        land_coefficient=float(sp.get("land_coefficient", 2.0)),
        max_dt_s=float(sp.get("max_dt_days", 1.0)) * 86400.0,
        max_steps=int(sp.get("max_steps", 20000)),
    ))
    config = CO2GameConfig(
        n_operators=2,
        horizon=1,
        stage_duration_s=seconds,
        discount_factor=1.0,
        pipeline=PipelineConfig(capacity_kg_per_stage=cap,
                                source_supply_kg_per_stage=cap,
                                dispatch_rule=dispatch_rule),
        payoff=CO2PayoffConfig(
            storage_value_per_kg=1.0,
            drilling_cost_per_well=0.0,
            pressure_cap_pa=pressure_cap_pa,
            pressure_damage_per_pa2=pressure_damage,
        ),
    )
    nz, nx = geo["phi"].shape
    # Candidate locations are separated by operator lease halves and lie away
    # from boundaries. The exact same cells are used on every paired geology.
    z = nz - 2
    candidates = (
        ((z, max(1, nx // 6)), (z, max(1, nx // 3))),
        ((z, min(nx - 2, 2 * nx // 3)), (z, min(nx - 2, 5 * nx // 6))),
    )
    rules = PlacementRules(
        candidate_cells_by_operator=tuple(frozenset(x) for x in candidates),
        max_wells_per_operator=max_wells_per_operator,
        max_new_wells_per_operator_per_stage=1,
        max_rate_m3_s_per_well=full_rate,
        drilling_lead_stages=0,
    )
    return CO2WellPlacementGame(solver=solver, config=config,
                                placement_rules=rules), candidates


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cohorts", nargs="+", default=["friendly", "challenging", "tarbert", "upper_ness"])
    ap.add_argument("--geologies-per-cohort", type=int, default=1)
    ap.add_argument("--dispatch-rules", nargs="+", default=["proportional", "fixed_equal"])
    ap.add_argument("--capacity-fractions", nargs="+", type=float, default=[0.5, 1.0])
    ap.add_argument("--pressure-damages", nargs="+", type=float, default=[0.0, 1e-9])
    ap.add_argument("--pressure-cap-mpa", type=float, default=5.0)
    ap.add_argument("--pressure-tail-checkpoint")
    ap.add_argument("--unroll3-checkpoint")
    ap.add_argument("--pressure-tail-checkpoints", nargs="+")
    ap.add_argument("--unroll3-checkpoints", nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--starts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--skip-placement", action="store_true")
    a = ap.parse_args()
    data_dir = Path(a.data_dir).expanduser().resolve()
    out_dir = Path(a.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = data_dir / "dataset_config.json"
    dataset_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    p = dataset_cfg.get("parameters", {})
    stage_days = float(p.get("stage_days", 30.0))
    full_rate = float(p.get("rate_m3_s", 1.0e-7))
    pressure_cap = a.pressure_cap_mpa * 1.0e6
    geologies = _load_geologies(data_dir, a.cohorts, a.geologies_per_cohort)
    print(f"Selected {len(geologies)} held-out stage-0 geologies", flush=True)

    models = {}
    tail_paths = list(a.pressure_tail_checkpoints or []) + (
        [a.pressure_tail_checkpoint] if a.pressure_tail_checkpoint else [])
    unroll_paths = list(a.unroll3_checkpoints or []) + (
        [a.unroll3_checkpoint] if a.unroll3_checkpoint else [])
    if bool(tail_paths) != bool(unroll_paths):
        raise ValueError("merge-only mode requires both pressure-tail and unroll-3 checkpoint(s)")
    if tail_paths:
        tail_by_seed = {}
        unroll_by_seed = {}
        for path in tail_paths:
            item = load_co2_checkpoint(path, device=a.device)
            seed = item["blob"].get("seed")
            if seed in tail_by_seed:
                raise ValueError(f"duplicate pressure-tail seed {seed}")
            tail_by_seed[seed] = path
        for path in unroll_paths:
            item = load_co2_checkpoint(path, device=a.device)
            seed = item["blob"].get("seed")
            if seed in unroll_by_seed:
                raise ValueError(f"duplicate unroll-3 seed {seed}")
            unroll_by_seed[seed] = path
        if set(tail_by_seed) != set(unroll_by_seed):
            raise ValueError("merge-only mode requires matched seed IDs in both checkpoint lists")
        for seed in sorted(tail_by_seed, key=str):
            name = f"merged_ccs_gnn_seed{seed}"
            models[name] = make_pressure_tail_unroll3_hybrid(
                tail_by_seed[seed], unroll_by_seed[seed], device=a.device)

    arms = []
    for dispatch in a.dispatch_rules:
        for cap_fraction in a.capacity_fractions:
            for damage in a.pressure_damages:
                arm = {"dispatch_rule": dispatch,
                       "capacity_fraction_of_two_full_wells": cap_fraction,
                       "pressure_damage_per_pa2": damage,
                       "pressure_cap_pa": pressure_cap}
                game_results = []
                model_results = {name: [] for name in models}
                for geo in geologies:
                    game, candidate_cells = _scenario_game(
                        geo, stage_days=stage_days, full_rate=full_rate,
                        capacity_fraction=cap_fraction, dispatch_rule=dispatch,
                        pressure_damage=damage, pressure_cap_pa=pressure_cap,
                        solver_parameters=p)
                    state = game.initial_state(geo["phi"].shape)
                    half_full = (0.5 * full_rate, full_rate)

                    # Fixed training-supported cells: same action menu across
                    # the solver and all surrogate planners.
                    h, w = geo["phi"].shape
                    supported = ((h - 2, max(1, w // 3)),
                                 (h - 2, min(w - 2, 2 * w // 3)))
                    fixed_actions = [
                        make_placement_action_set(op, (supported[op],), half_full)
                        for op in range(2)
                    ]
                    fixed_labels = [
                        ["out", f"half@{supported[op]}", f"full@{supported[op]}"]
                        for op in range(2)
                    ]
                    solver_fixed = evaluate_solver_stage_game(
                        game, state, geo, fixed_actions)
                    game_results.append({
                        "cohort": geo["cohort"], "geology_seed": geo["geology_seed"],
                        "fixed_solver": solver_fixed,
                    })
                    for name, model in models.items():
                        try:
                            fore = evaluate_surrogate_stage_game(
                                game, state, geo, fixed_actions, model,
                                dx_ft=geo["dx_ft"], dy_ft=geo["dy_ft"], dz_ft=geo["dz_ft"])
                        except ValueError as exc:
                            # Location restriction is intentional. If current
                            # candidate geometry is unsupported, fail visibly.
                            raise RuntimeError(f"surrogate {name} cannot score this action set: {exc}") from exc
                        model_results[name].append({
                            "cohort": geo["cohort"], "geology_seed": geo["geology_seed"],
                            "forecast": fore,
                        })

                    if not a.skip_placement:
                        # Placement menu changes the well cell. It is solver
                        # scored, never passed through the fixed-cell surrogate.
                        placement_actions = [
                            make_placement_action_set(op, candidate_cells[op], half_full)
                            for op in range(2)
                        ]
                        placement = evaluate_solver_stage_game(
                            game, state, geo, placement_actions)
                        game_results[-1]["placement_solver"] = placement

                solver_mean = aggregate_geology_games(
                    [x["fixed_solver"]["payoffs"] for x in game_results],
                    [x["fixed_solver"]["resource_welfare"] for x in game_results])
                solver_analysis = analyze_bimatrix(solver_mean["expected_payoffs"])
                analysis = {
                    "n_heldout_geologies": len(game_results),
                    "fixed_solver_expected_payoff_pure_nash": solver_analysis.pure_nash_profiles,
                    "fixed_solver_expected_welfare_maximizers": solver_analysis.welfare_maximizers,
                    "fixed_solver_learning_dynamics": {},
                    "surrogate_planners": {},
                }
                for update in ("alternating", "simultaneous"):
                    paths = best_response_dynamics(
                        solver_mean["expected_payoffs"], starts=a.starts,
                        update=update, seed=a.seed)
                    analysis["fixed_solver_learning_dynamics"][update] = summarize_dynamics(paths)
                analysis["fictitious_play_solver"] = fictitious_play(
                    solver_mean["expected_payoffs"], starts=10, seed=a.seed)
                analysis["logit_response_solver"] = logit_response_dynamics(
                    solver_mean["expected_payoffs"], seed=a.seed)
                for name, results in model_results.items():
                    mean = aggregate_geology_games(
                        [x["forecast"]["payoffs"] for x in results],
                        [x["forecast"]["resource_welfare"] for x in results])
                    model_analysis = analyze_bimatrix(mean["expected_payoffs"])
                    selected_profiles = model_analysis.pure_nash_profiles or model_analysis.welfare_maximizers
                    per_geology_replay = []
                    for profile in selected_profiles:
                        for case in game_results:
                            table = case["fixed_solver"]
                            case_analysis = table["analysis"]
                            i, j = profile
                            regret = case_analysis.nash_regret[i, j]
                            per_geology_replay.append({
                                "model_profile": [i, j],
                                "cohort": case["cohort"],
                                "geology_seed": case["geology_seed"],
                                "solver_regret_by_operator": regret.tolist(),
                                "solver_epsilon_sum": float(regret.sum()),
                                "solver_payoff_by_operator": table["payoffs"][i, j].tolist(),
                                "solver_welfare": float(table["resource_welfare"][i, j]),
                                "solver_pressure_max_pa": float(table["pressure_max_pa"][i, j]),
                            })
                    eps = np.asarray([x["solver_epsilon_sum"] for x in per_geology_replay])
                    solver_replays = compare_planner_to_solver(
                        solver_mean["expected_payoffs"], mean["expected_payoffs"],
                        action_labels=fixed_labels,
                        solver_pressure_pa=np.mean(
                            [x["fixed_solver"]["pressure_max_pa"] for x in game_results], axis=0))
                    analysis["surrogate_planners"][name] = {
                        "checkpoint": results[0]["forecast"]["checkpoint"],
                        "expected_payoff_pure_nash": model_analysis.pure_nash_profiles,
                        "expected_welfare_maximizers": model_analysis.welfare_maximizers,
                        "solver_replay_and_deviation_regret": solver_replays,
                        "solver_replay_by_geology": per_geology_replay,
                        "solver_epsilon_summary_across_selected_profiles_and_geologies": ({
                            "median": float(np.median(eps)),
                            "p95": float(np.quantile(eps, 0.95)),
                            "max": float(np.max(eps)),
                            "fraction_within_1e-6": float(np.mean(eps <= 1e-6)),
                        } if eps.size else None),
                        "learning_dynamics": summarize_dynamics(best_response_dynamics(
                            mean["expected_payoffs"], starts=a.starts,
                            update="alternating", seed=a.seed)),
                    }
                placement_summary = None
                placement_results = [x["placement_solver"] for x in game_results
                                     if "placement_solver" in x]
                if placement_results:
                    place_mean = aggregate_geology_games(
                        [x["payoffs"] for x in placement_results],
                        [x["resource_welfare"] for x in placement_results])
                    pa = analyze_bimatrix(place_mean["expected_payoffs"])
                    placement_summary = {
                        "expected_payoff_pure_nash": pa.pure_nash_profiles,
                        "expected_welfare_maximizers": pa.welfare_maximizers,
                        "action_labels": [
                            ["out", "site0-half", "site0-full", "site1-half", "site1-full"],
                            ["out", "site0-half", "site0-full", "site1-half", "site1-full"],
                        ],
                        "solver_only": True,
                        "learning_dynamics": summarize_dynamics(best_response_dynamics(
                            place_mean["expected_payoffs"], starts=a.starts,
                            update="alternating", seed=a.seed)),
                    }
                arm.update({"analysis": analysis,
                            "fixed_solver_mean": solver_mean,
                            "per_geology_solver": game_results,
                            "placement_expected_game": placement_summary})
                arms.append(arm)
                print(f"Done {dispatch=} {cap_fraction=} {damage=}; "
                      f"pure NE={solver_analysis.pure_nash_profiles}", flush=True)

    provenance = {
        "dataset_dir": str(data_dir),
        "dataset_config_sha256": _hash_file(config_path),
        "dataset_description": dataset_cfg.get("description"),
        "dataset_physics_limits": dataset_cfg.get("physics_limits", []),
        "code_module": "sequential_game.co2_equilibrium_lab",
        "solver": "CO2XZSolver; dataset dx/dy/dz; stage duration from dataset_config",
        "equilibrium_claim": "one-stage pure Nash in the enumerated action game; learning dynamics diagnostics",
        "surrogate_claim": "merge-only: pressure-tail pressure delta plus unroll-3 other state deltas, one-stage forecasts on fixed trained well cells; selected profiles solver replayed",
        "placement_claim": "one-stage placement equilibria scored by solver over supplied held-out geology sample; not full MPE",
        "model_checkpoints": {k: {
            "path": v["checkpoint"],
            "component_paths": v["checkpoint"].split("|"),
            "component_sha256": [_hash_file(Path(x)) for x in v["checkpoint"].split("|")],
            "backbone": v["blob"].get("backbone"),
            "seed": v["blob"].get("seed"),
            "training_args": v["blob"].get("args", {}),
        } for k, v in models.items()},
        "co2_and_brine_separation": "CO2-only modules; no brine game results imported",
    }
    result = {"provenance": provenance,
              "options": vars(a),
              "geology_manifest": [{k: g[k] for k in (
                  "cohort", "geology_seed", "geology_index", "dx_ft", "dy_ft",
                  "dz_ft", "schedule_name", "source_file")} for g in geologies],
              "arms": arms}
    path = out_dir / "co2_equilibrium_pilot.json"
    path.write_text(json.dumps(_to_jsonable(result), indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
