"""Eight-operator CO2 game using the aggregate-map CCS-GNN hybrid.

The merged surrogate scores the full one-stage action-profile tensor. Its pure
Nash profiles and welfare maximizers are then replayed with the CO2 solver;
solver unilateral deviations report replay regret. This is a one-stage game
experiment, not a sequential Markov-perfect equilibrium result.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

try:
    from .co2_equilibrium_lab import (
        analyze_n_player,
        evaluate_surrogate_n_player_stage_game,
        make_placement_action_set,
        make_pressure_tail_unroll3_hybrid,
    )
    from .run_co2_multiplayer_placement import load_geologies, make_game, jsonable
except ImportError:
    from co2_equilibrium_lab import (
        analyze_n_player, evaluate_surrogate_n_player_stage_game,
        make_placement_action_set, make_pressure_tail_unroll3_hybrid,
    )
    from run_co2_multiplayer_placement import load_geologies, make_game, jsonable


def _solver_profile_and_deviations(game, state, geo, action_sets, profile,
                                   predicted_states):
    def score(indices):
        actions = tuple(action_sets[p][indices[p]] for p in range(len(indices)))
        _, record = game.step(deepcopy(state), geo, actions)
        return record

    next_state, base = game.step(
        deepcopy(state), geo,
        tuple(action_sets[p][profile[p]] for p in range(len(profile))))
    base_payoffs = np.asarray(base["discounted_operator_payoff_by_operator"], dtype=float)
    deviations = []
    regrets = np.zeros(len(profile), dtype=float)
    for player, action_set in enumerate(action_sets):
        best = base_payoffs[player]
        best_action = profile[player]
        for action_i in range(len(action_set)):
            if action_i == profile[player]:
                continue
            changed = list(profile); changed[player] = action_i
            alt = score(tuple(changed))
            value = float(alt["discounted_operator_payoff_by_operator"][player])
            if value > best:
                best, best_action = value, action_i
        regrets[player] = max(0.0, best - base_payoffs[player])
        deviations.append({"operator": player, "best_response_action_index": best_action,
                           "solver_regret": float(regrets[player])})
    solver_state = next_state.core.reservoir
    field_errors = {}
    for key in ("p_pa", "sg_mobile", "sg_trapped", "sg_max", "c_dissolved_kg_m3"):
        err = np.asarray(predicted_states[key])[0] - np.asarray(solver_state[key])
        field_errors[key] = {"rmse": float(np.sqrt(np.mean(err ** 2))),
                             "mae": float(np.mean(np.abs(err))),
                             "max_abs": float(np.max(np.abs(err)))}
    return {"profile": list(profile), "solver_payoffs": base_payoffs,
            "solver_welfare": float(base["discounted_project_resource_welfare"]),
            "solver_peak_pressure_pa": float(base["pressure_max_pa"]),
            "solver_mass_balance_residual_kg": base["solver_diagnostics"].get("mass_balance_residual_kg"),
            "surrogate_vs_solver_field_errors": field_errors,
            "solver_injected_kg_by_operator": base["storage_accepted_kg_by_operator"],
            "max_solver_unilateral_regret": float(regrets.max()),
            "mean_solver_unilateral_regret": float(regrets.mean()),
            "deviations": deviations}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="new multiwell dataset directory")
    ap.add_argument("--pressure-checkpoint", required=True)
    ap.add_argument("--dynamics-checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-operators", type=int, default=8)
    ap.add_argument("--cohorts", nargs="+", default=["challenging"])
    ap.add_argument("--geologies-per-cohort", type=int, default=1)
    ap.add_argument("--dispatch-rule", default="proportional")
    ap.add_argument("--capacity-fraction", type=float, default=0.5)
    ap.add_argument("--pressure-damage", type=float, default=1e-9)
    ap.add_argument("--pressure-cap-mpa", type=float, default=5.0)
    ap.add_argument("--min-spacing-ft", type=float, default=50.0)
    ap.add_argument("--max-solver-replays", type=int, default=20,
                    help="maximum surrogate NE/welfare profiles to replay; each includes all unilateral deviations")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.n_operators != 8:
        raise ValueError("this first location-randomized training recipe is scoped to 8 operators")
    if a.max_solver_replays < 1:
        raise ValueError("max-solver-replays must be positive")
    data_dir = Path(a.data_dir).resolve()
    out_dir = Path(a.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = data_dir / "dataset_config.json"
    dataset_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    params = dataset_cfg.get("parameters", {})
    geologies = load_geologies(data_dir, a.cohorts, a.geologies_per_cohort)
    if not geologies:
        raise RuntimeError("No held-out geologies found in test split")
    surrogate = make_pressure_tail_unroll3_hybrid(
        a.pressure_checkpoint, a.dynamics_checkpoint, device=a.device)
    if surrogate["blob"].get("well_input_mode") != "aggregate_well_map_v1":
        raise ValueError("both checkpoints must be trained on this multiwell dataset")
    full_rate = float(params["rate_m3_s_per_operator"])
    stage_days = float(params["stage_days"])
    outputs = []
    for geo in geologies:
        game, sites = make_game(
            geo, a.n_operators, stage_days=stage_days, full_rate=full_rate,
            capacity_fraction=a.capacity_fraction, dispatch_rule=a.dispatch_rule,
            pressure_damage=a.pressure_damage,
            pressure_cap_pa=a.pressure_cap_mpa * 1e6,
            solver_parameters=params, min_spacing_ft=a.min_spacing_ft)
        action_sets = [make_placement_action_set(op, sites[op], [full_rate])
                       for op in range(a.n_operators)]
        state = game.initial_state(geo["phi"].shape)
        result = evaluate_surrogate_n_player_stage_game(
            game, state, geo, action_sets, surrogate,
            dx_ft=geo["dx_ft"], dy_ft=geo["dy_ft"], dz_ft=geo["dz_ft"])
        analysis = analyze_n_player(result["payoffs"])
        candidate_profiles = list(analysis["pure_nash_profiles"])
        selection = "surrogate_pure_nash"
        if not candidate_profiles:
            candidate_profiles = list(analysis["welfare_maximizers"])
            selection = "surrogate_welfare_fallback_no_pure_nash"
        # Also include surrogate welfare maximizers as a separate planning reference.
        welfare_profiles = list(analysis["welfare_maximizers"])
        profiles = []
        profile_labels = {}
        for label, values in ((selection, candidate_profiles),
                              ("surrogate_welfare_maximizer", welfare_profiles)):
            for profile in values:
                key = tuple(map(int, profile))
                if key not in profile_labels:
                    profiles.append(key); profile_labels[key] = []
                profile_labels[key].append(label)
        profiles = profiles[:a.max_solver_replays]
        solver_replays = []
        for profile in profiles:
            replay = _solver_profile_and_deviations(
                game, state, geo, action_sets, profile, result["predicted_states"][profile])
            replay["selection_labels"] = profile_labels[profile]
            replay["surrogate_payoffs"] = result["payoffs"][profile]
            replay["surrogate_welfare"] = float(result["resource_welfare"][profile])
            replay["surrogate_peak_pressure_pa"] = float(result["pressure_max_pa"][profile])
            solver_replays.append(replay)
        outputs.append({"cohort": geo["cohort"], "geology_seed": geo["geology_seed"],
                        "geology_index": geo["geology_index"], "sites_by_operator": sites,
                        "n_operators": a.n_operators, "action_counts": result["action_counts"],
                        "joint_profiles_scored_by_surrogate": int(np.prod(result["action_counts"])),
                        "surrogate_pure_nash_profiles": analysis["pure_nash_profiles"],
                        "surrogate_welfare_maximizers": analysis["welfare_maximizers"],
                        "surrogate_welfare_max": float(np.max(result["resource_welfare"])),
                        "solver_replays": solver_replays})
        print(f"Scored 8-player surrogate game for {geo['cohort']} seed {geo['geology_seed']}; "
              f"NE={len(analysis['pure_nash_profiles'])}, solver replays={len(solver_replays)}", flush=True)
    output = {"provenance": {"dataset_dir": str(data_dir),
                              "dataset_config_sha256": __import__("hashlib").sha256(cfg_path.read_bytes()).hexdigest(),
                              "dataset_description": dataset_cfg.get("description"),
                              "surrogate_components": surrogate["surrogate_components"],
                              "n_operators": a.n_operators,
                              "joint_game_scope": "one-stage terminal placement/rate game",
                              "equilibrium_scope": "surrogate pure Nash candidates with solver replay and unilateral deviation regret",
                              "limitations": ["one tested geology is a case study, not uncertainty evidence",
                                              "the multi-stage Markov-perfect equilibrium is not established"]},
              "options": vars(a), "per_geology": outputs}
    path = out_dir / "co2_multiwell_surrogate_game.json"
    path.write_text(json.dumps(jsonable(output), indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
