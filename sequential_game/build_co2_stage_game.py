"""Build a solver-backed one-stage normal-form game for mechanism experiments.

The output is a *static initial-state benchmark* for equilibrium and
institutional comparisons. It is not a Markov-perfect equilibrium result.
Each action is a per-operator injection-rate choice at one fixed well site.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .co2_solver_game_search import evaluate_schedule, make_config
except ImportError:
    from co2_solver_game_search import evaluate_schedule, make_config


def load_geology(path: Path, row: int = 0) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        geology = {key: archive[key] for key in ("kx_mD", "kz_mD", "phi")}
    for key, value in geology.items():
        if value.ndim == 3:
            if not 0 <= row < value.shape[0]:
                raise IndexError(f"--row {row} is outside {key} with {value.shape[0]} cases")
            geology[key] = value[row]
        if geology[key].ndim != 2:
            raise ValueError(f"{key} must be a single [z,x] geology or a [N,z,x] archive")
    if not (geology["kx_mD"].shape == geology["kz_mD"].shape == geology["phi"].shape):
        raise ValueError("kx_mD, kz_mD, and phi must have matching shapes")
    return geology


def build_stage_game(
    geology: dict[str, np.ndarray],
    raw_config: dict[str, Any],
    rate_levels_m3_s: list[float],
    *,
    max_profiles: int = 20_000,
) -> dict[str, Any]:
    """Enumerate simultaneous rate choices and record payoffs/outcomes."""
    game_cfg, _, wells = make_config(raw_config)
    n_ops = game_cfg.n_operators
    rates = np.asarray(rate_levels_m3_s, dtype=np.float64)
    if rates.ndim != 1 or len(rates) < 2 or np.any(~np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("rate_levels_m3_s must contain at least two finite nonnegative rates")
    if np.any(np.diff(rates) < 0):
        raise ValueError("rate_levels_m3_s must be in ascending order")
    n_profiles = len(rates) ** n_ops
    if n_profiles > max_profiles:
        raise ValueError(
            f"{n_profiles} profiles exceed max_profiles={max_profiles}; reduce operators or rate levels"
        )
    one_stage_config = dict(raw_config)
    one_stage_config["horizon"] = 1
    action_shape = tuple([len(rates)] * n_ops)
    payoffs = np.zeros(action_shape + (n_ops,), dtype=np.float64)
    gross_payoffs = np.zeros_like(payoffs)
    welfare = np.zeros(action_shape, dtype=np.float64)
    harm_liability = np.zeros_like(payoffs)
    delivered_mass = np.zeros_like(payoffs)
    solver_calls = 0

    for action_indices in itertools.product(range(len(rates)), repeat=n_ops):
        profile_rates = rates[np.asarray(action_indices)]
        schedule = profile_rates.reshape(1, n_ops)
        result = evaluate_schedule(geology, one_stage_config, schedule)
        solver_calls += result["solver_evaluations"] if "solver_evaluations" in result else 1
        record = result["stage_records"][0]
        payoffs[action_indices] = np.asarray(result["operator_payoff"], dtype=np.float64)
        gross_payoffs[action_indices] = np.asarray(
            record["operator_payoff_before_damage_liability"], dtype=np.float64
        )
        welfare[action_indices] = result["project_resource_welfare"]
        harm_liability[action_indices] = np.asarray(
            record["damage_liability_by_operator"], dtype=np.float64
        )
        delivered_mass[action_indices] = np.asarray(
            result["final_injected_kg_by_operator"], dtype=np.float64
        )

    return {
        "rate_levels_m3_s": rates,
        "payoffs": payoffs,
        "gross_payoffs_before_damage_liability": gross_payoffs,
        "project_resource_welfare": welfare,
        "damage_liability_by_operator": harm_liability,
        "injected_kg_by_operator": delivered_mass,
        "number_of_operators": n_ops,
        "number_of_profiles": n_profiles,
        "solver_evaluations": solver_calls,
        "stage_duration_s": game_cfg.stage_duration_s,
        "wells_by_operator": wells,
        "interpretation": (
            "one-stage initial-state solver game; not a dynamic Markov equilibrium; "
            "payoff units and physics inherit the supplied scenario"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geology", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rate-max-m3-s", type=float, required=True)
    parser.add_argument("--rate-levels", type=int, default=3)
    parser.add_argument("--max-profiles", type=int, default=20_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.rate_max_m3_s <= 0 or args.rate_levels < 2:
        parser.error("--rate-max-m3-s must be positive and --rate-levels >= 2")
    geology = load_geology(args.geology, args.row)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    levels = np.linspace(0.0, args.rate_max_m3_s, args.rate_levels).tolist()
    result = build_stage_game(geology, config, levels, max_profiles=args.max_profiles)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        rate_levels_m3_s=result["rate_levels_m3_s"],
        payoffs=result["payoffs"],
        gross_payoffs_before_damage_liability=result["gross_payoffs_before_damage_liability"],
        project_resource_welfare=result["project_resource_welfare"],
        damage_liability_by_operator=result["damage_liability_by_operator"],
        injected_kg_by_operator=result["injected_kg_by_operator"],
    )
    meta = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    meta["rate_levels_m3_s"] = levels
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {result['number_of_profiles']} profiles to {args.out}")


if __name__ == "__main__":
    main()
