"""Equilibrium and institutional experiments for the CO2 injection game.

This module operates on finite action/payoff tables. ``build_co2_stage_game.py``
can produce a solver-backed initial-state stage game. The mechanism results are
therefore stage-game benchmarks; they do not by themselves establish a dynamic
Markov-perfect equilibrium.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import linprog


def _validate_payoffs(payoffs: np.ndarray) -> tuple[int, tuple[int, ...]]:
    u = np.asarray(payoffs, dtype=np.float64)
    n_players = u.shape[-1] if u.ndim >= 3 else 0
    action_shape = u.shape[:-1]
    if n_players < 2 or len(action_shape) != n_players or any(n < 1 for n in action_shape):
        raise ValueError("payoffs must have shape (n_actions_p0, ..., n_actions_pN, n_players)")
    if np.any(~np.isfinite(u)):
        raise ValueError("payoff tensor must contain only finite values")
    return n_players, action_shape


def enumerate_equilibria(payoffs: np.ndarray, tolerance: float = 1e-9) -> dict[str, Any]:
    """Enumerate pure Nash profiles and compute unilateral grid regrets."""
    u = np.asarray(payoffs, dtype=np.float64)
    n_players, action_shape = _validate_payoffs(u)
    profiles = list(itertools.product(*(range(k) for k in action_shape)))
    diagnostics = []
    pure_nash = []
    for profile in profiles:
        current = u[profile]
        gains = []
        best_deviations = []
        for player in range(n_players):
            alternatives = []
            for action in range(action_shape[player]):
                trial = list(profile)
                trial[player] = action
                alternatives.append((float(u[tuple(trial)][player]), action))
            best_value, best_action = max(alternatives)
            gain = max(0.0, best_value - float(current[player]))
            gains.append(gain)
            best_deviations.append({"action": int(best_action), "gain": gain})
        total_welfare = float(np.sum(current))
        row = {"profile": list(map(int, profile)), "payoffs": current.tolist(),
               "total_payoff": total_welfare, "unilateral_regret_by_player": gains,
               "best_deviation_by_player": best_deviations,
               "max_unilateral_regret": max(gains, default=0.0)}
        diagnostics.append(row)
        if max(gains, default=0.0) <= tolerance:
            pure_nash.append(row)
    planner = max(diagnostics, key=lambda item: item["total_payoff"])
    return {
        "number_of_players": n_players,
        "action_counts": list(action_shape),
        "profiles_checked": len(profiles),
        "pure_nash_profiles": pure_nash,
        "pure_nash_count": len(pure_nash),
        "utilitarian_private_payoff_profile": planner,
        "best_grid_approximate_equilibrium": min(
            diagnostics, key=lambda item: item["max_unilateral_regret"]
        ),
        "profile_diagnostics": diagnostics,
        "regret_definition": (
            "largest payoff gain from any one-player action deviation in the declared finite action grid"
        ),
    }


def welfare_maximizing_correlated_equilibrium(
    payoffs: np.ndarray, *, social_welfare: np.ndarray | None = None,
    obedience_tolerance: float = 1e-8
) -> dict[str, Any]:
    """Solve the utilitarian correlated-equilibrium LP with scipy HiGHS."""
    u = np.asarray(payoffs, dtype=np.float64)
    n_players, action_shape = _validate_payoffs(u)
    profiles = list(itertools.product(*(range(k) for k in action_shape)))
    n_profiles = len(profiles)
    profile_index = {profile: i for i, profile in enumerate(profiles)}
    if social_welfare is None:
        welfare = np.sum(u, axis=-1)
        objective_basis = "sum of player utilities"
    else:
        welfare = np.asarray(social_welfare, dtype=np.float64)
        if welfare.shape != action_shape or np.any(~np.isfinite(welfare)):
            raise ValueError("social_welfare must be a finite tensor matching the action dimensions")
        objective_basis = "supplied project resource welfare"
    objective = -np.asarray([float(welfare[p]) for p in profiles])
    obedience_rows = []
    for player in range(n_players):
        for recommended in range(action_shape[player]):
            for deviation in range(action_shape[player]):
                if deviation == recommended:
                    continue
                coeff = np.zeros(n_profiles, dtype=np.float64)
                for idx, profile in enumerate(profiles):
                    if profile[player] != recommended:
                        continue
                    deviated = list(profile)
                    deviated[player] = deviation
                    coeff[idx] = -(
                        u[profile][player] - u[tuple(deviated)][player]
                    )  # linprog uses A_ub x <= b_ub
                obedience_rows.append(coeff)
    a_ub = np.asarray(obedience_rows) if obedience_rows else None
    b_ub = np.zeros(len(obedience_rows), dtype=np.float64) if obedience_rows else None
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=np.ones((1, n_profiles)),
        b_eq=np.ones(1),
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"correlated-equilibrium LP failed: {result.message}")
    probabilities = np.clip(result.x, 0.0, 1.0)
    probabilities /= probabilities.sum()
    expected = np.zeros(n_players, dtype=np.float64)
    expected_welfare = 0.0
    support = []
    max_violation = 0.0
    for profile, probability in zip(profiles, probabilities):
        if probability > 1e-8:
            expected += probability * u[profile]
            expected_welfare += probability * float(welfare[profile])
            support.append({"profile": list(profile), "probability": float(probability),
                            "payoffs": u[profile].tolist()})
    for player in range(n_players):
        for recommended in range(action_shape[player]):
            for deviation in range(action_shape[player]):
                if deviation == recommended:
                    continue
                gain = 0.0
                for profile, probability in zip(profiles, probabilities):
                    if profile[player] != recommended:
                        continue
                    deviated = list(profile)
                    deviated[player] = deviation
                    gain += probability * (u[tuple(deviated)][player] - u[profile][player])
                max_violation = max(max_violation, float(gain))
    return {
        "expected_payoff_by_player": expected.tolist(),
        "expected_total_payoff": float(expected.sum()),
        "expected_project_resource_welfare": float(expected_welfare),
        "support": support,
        "max_obedience_violation": max_violation,
        "obedience_pass": max_violation <= obedience_tolerance,
        "solver": "scipy.optimize.linprog(method='highs')",
        "objective_basis": objective_basis,
        "interpretation": (
            "the mediator draws a joint action profile and privately recommends each player's action; "
            "each recommendation is incentive-compatible in expectation"
        ),
    }


def allocate_capacity_auction(
    true_marginal_values: Sequence[Sequence[float]],
    bids: Sequence[Sequence[float]],
    capacity_blocks: int,
    *,
    mechanism: str = "vcg",
    reserve_bid: float = 0.0,
) -> dict[str, Any]:
    """Clear a multi-unit private-value auction with prefix allocations.

    Each row is an operator's ordered marginal value/bid schedule. The
    allocation search chooses a prefix length for each operator, so block 2
    cannot be won without block 1 even if the supplied marginal schedule is
    nonmonotone. Equal-score allocations first prefer fewer zero-value blocks,
    then the allocation with the most balanced block counts. VCG payments use
    Clarke-pivot externalities; pay-as-bid is a comparison.
    """
    values = np.asarray(true_marginal_values, dtype=np.float64)
    bid_array = np.asarray(bids, dtype=np.float64)
    if values.ndim != 2 or bid_array.shape != values.shape:
        raise ValueError("values and bids must be matching [operator, capacity_block] arrays")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(bid_array)):
        raise ValueError("values and bids must be finite")
    if not 0 <= capacity_blocks <= values.size:
        raise ValueError("capacity_blocks must be between zero and the total offered blocks")
    if mechanism not in {"vcg", "pay_as_bid"}:
        raise ValueError("mechanism must be 'vcg' or 'pay_as_bid'")

    n_ops, n_blocks = values.shape

    def best_allocation(exclude: int | None = None) -> tuple[tuple[int, ...], float]:
        players = [i for i in range(n_ops) if i != exclude]
        best_counts = tuple(0 for _ in range(n_ops))
        best_score = 0.0

        def balance_penalty(counts: Sequence[int]) -> float:
            average = sum(counts) / max(len(counts), 1)
            return float(sum((count - average) ** 2 for count in counts))

        for partial in itertools.product(*(range(n_blocks + 1) for _ in players)):
            if sum(partial) > capacity_blocks:
                continue
            counts = [0] * n_ops
            for player, count in zip(players, partial):
                counts[player] = count
            feasible = all(
                np.all(bid_array[player, :count] >= reserve_bid)
                for player, count in enumerate(counts)
            )
            if not feasible:
                continue
            score = sum(float(np.sum(bid_array[player, :count]))
                        for player, count in enumerate(counts))
            tied_score = abs(score - best_score) <= 1e-12
            fewer_blocks = sum(counts) < sum(best_counts)
            equally_few_blocks = sum(counts) == sum(best_counts)
            more_balanced = balance_penalty(counts) < balance_penalty(best_counts) - 1e-12
            if score > best_score + 1e-12 or (
                    tied_score and (fewer_blocks or (equally_few_blocks and more_balanced))):
                best_counts, best_score = tuple(counts), score
        return best_counts, best_score

    allocated, _ = best_allocation()
    true_value_by_operator = np.asarray([
        float(np.sum(values[player, :count]))
        for player, count in enumerate(allocated)
    ])
    winning_bid_sum = np.asarray([
        float(np.sum(bid_array[player, :count]))
        for player, count in enumerate(allocated)
    ])
    true_welfare = float(true_value_by_operator.sum())
    payments = np.zeros(n_ops, dtype=np.float64)
    if mechanism == "pay_as_bid":
        payments[:] = winning_bid_sum
    else:
        for operator in range(n_ops):
            welfare_others_with = sum(
                float(np.sum(bid_array[player, :allocated[player]]))
                for player in range(n_ops) if player != operator
            )
            _, welfare_others_without = best_allocation(exclude=operator)
            payments[operator] = max(0.0, welfare_others_without - welfare_others_with)
    winners = [
        {"operator": player, "blocks": int(allocated[player]),
         "marginal_bids": bid_array[player, :allocated[player]].tolist(),
         "marginal_values": values[player, :allocated[player]].tolist()}
        for player in range(n_ops) if allocated[player] > 0
    ]
    return {
        "mechanism": mechanism,
        "capacity_blocks": int(capacity_blocks),
        "allocated_blocks_by_operator": list(allocated),
        "true_value_by_operator": true_value_by_operator.tolist(),
        "payment_by_operator": payments.tolist(),
        "private_surplus_by_operator": (true_value_by_operator - payments).tolist(),
        "auction_revenue": float(payments.sum()),
        "total_allocated_true_private_value": true_welfare,
        "social_welfare_under_additive_private_value_assumption": true_welfare,
        "winners": winners,
        "reserve_bid": float(reserve_bid),
        "tie_break_rule": "maximize reported bid sum; on ties prefer fewer blocks, then balance counts across operators",
        "caveat": (
            "single-stage additive-value auction; VCG efficiency/truthfulness needs the stated "
            "private-value assumptions and does not model reservoir externalities by itself"
        ),
    }


def marginal_values_from_stage_game(game_path: Path) -> tuple[np.ndarray, float]:
    """Estimate ordered private marginal-value blocks from a no-cap solver table.

    For each operator, differences its private payoff between adjacent
    unilateral rate levels with rivals at zero. Use a no-pipeline-cap,
    no-storage-cap table with the chosen private-liability rule and tariff;
    otherwise dispatch censors the marginal willingness to pay. The result is
    a scenario estimate, not a reported/validated private margin.
    """
    with np.load(game_path, allow_pickle=False) as game:
        rates = np.asarray(game["rate_levels_m3_s"], dtype=np.float64)
        private = np.asarray(game["payoffs"], dtype=np.float64)
        injected = np.asarray(game["injected_kg_by_operator"], dtype=np.float64)
    n_ops, action_shape = _validate_payoffs(private)
    if injected.shape != private.shape:
        raise ValueError("injected-mass tensor shape does not match the payoff tensor")
    if len(set(action_shape)) != 1 or action_shape[0] != len(rates):
        raise ValueError("stage game must use the same rate grid for every operator")
    zero = [0] * n_ops
    values = np.zeros((n_ops, len(rates) - 1), dtype=np.float64)
    block_masses = []
    for operator in range(n_ops):
        for level in range(1, len(rates)):
            low_profile, high_profile = zero.copy(), zero.copy()
            low_profile[operator], high_profile[operator] = level - 1, level
            delta_mass = float(
                injected[tuple(high_profile)][operator]
                - injected[tuple(low_profile)][operator]
            )
            if delta_mass <= 0:
                raise ValueError(
                    "no positive marginal injected mass; build this valuation table with "
                    "nonbinding pipeline/source/storage capacity"
                )
            values[operator, level - 1] = (
                private[tuple(high_profile)][operator]
                - private[tuple(low_profile)][operator]
            )
            block_masses.append(delta_mass)
    block_mass = float(np.mean(block_masses))
    if not np.allclose(block_masses, block_mass, rtol=1e-5, atol=1e-8):
        raise ValueError("operator rate increments do not imply a common auction block mass")
    return values, block_mass


def financial_assurance_payoffs(
    gross_private_payoffs: np.ndarray,
    attributable_damage: np.ndarray,
    bond_by_operator: Sequence[float],
    *,
    capital_cost_rate: float = 0.0,
) -> dict[str, np.ndarray]:
    """Apply capped collectible liability and the cost of tied-up bond capital.

    ``gross_private_payoffs`` must exclude damage liability, while
    ``attributable_damage`` assigns the physical damage responsibility by
    operator for each action profile. Bond payments are transfers; physical
    project welfare must still subtract the full damage exactly once.
    """
    gross = np.asarray(gross_private_payoffs, dtype=np.float64)
    damage = np.asarray(attributable_damage, dtype=np.float64)
    n_players, _ = _validate_payoffs(gross)
    if damage.shape != gross.shape or np.any(damage < 0) or np.any(~np.isfinite(damage)):
        raise ValueError("attributable_damage must be a matching finite nonnegative payoff tensor")
    bonds = np.asarray(bond_by_operator, dtype=np.float64)
    if bonds.shape != (n_players,) or np.any(bonds < 0) or np.any(~np.isfinite(bonds)):
        raise ValueError("bond_by_operator must contain one finite nonnegative amount per operator")
    if not np.isfinite(capital_cost_rate) or capital_cost_rate < 0:
        raise ValueError("capital_cost_rate must be finite and nonnegative")
    collectible = np.minimum(damage, bonds.reshape((1,) * (gross.ndim - 1) + (n_players,)))
    capital_cost = bonds * capital_cost_rate
    net = gross - collectible - capital_cost.reshape((1,) * (gross.ndim - 1) + (n_players,))
    return {
        "private_payoffs": net,
        "secured_liability": collectible,
        "unsecured_damage": damage - collectible,
        "capital_cost_by_operator": capital_cost,
    }


def grim_trigger_threshold(
    payoffs: np.ndarray,
    cooperative_profile: Sequence[int],
    punishment_profile: Sequence[int],
    *,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Compute the infinite/indefinite-horizon grim-trigger discount threshold."""
    u = np.asarray(payoffs, dtype=np.float64)
    n_players, action_shape = _validate_payoffs(u)
    coop, punish = tuple(cooperative_profile), tuple(punishment_profile)
    if len(coop) != n_players or len(punish) != n_players:
        raise ValueError("profiles must give one action index per player")
    if any(not 0 <= coop[i] < action_shape[i] or not 0 <= punish[i] < action_shape[i]
           for i in range(n_players)):
        raise ValueError("profile action index is outside the action grid")
    punishment_check = enumerate_equilibria(u, tolerance=tolerance)
    punishment_nash = any(tuple(row["profile"]) == punish
                          for row in punishment_check["pure_nash_profiles"])
    result = []
    for player in range(n_players):
        coop_value = float(u[coop][player])
        punish_value = float(u[punish][player])
        deviation_profile_values = []
        for action in range(action_shape[player]):
            trial = list(coop)
            trial[player] = action
            deviation_profile_values.append((float(u[tuple(trial)][player]), action))
        deviation_value, deviation_action = max(deviation_profile_values)
        one_shot_gain = max(0.0, deviation_value - coop_value)
        continuation_gain = coop_value - punish_value
        if one_shot_gain <= tolerance:
            threshold = 0.0
        elif continuation_gain <= tolerance:
            threshold = None
        else:
            threshold = one_shot_gain / (one_shot_gain + continuation_gain)
        result.append({
            "player": player,
            "cooperation_payoff": coop_value,
            "best_one_shot_deviation_payoff": deviation_value,
            "best_deviation_action": int(deviation_action),
            "one_shot_deviation_gain": one_shot_gain,
            "punishment_payoff": punish_value,
            "minimum_discount_factor": threshold,
        })
    thresholds = [row["minimum_discount_factor"] for row in result]
    return {
        "player_incentives": result,
        "minimum_discount_factor_for_all_players": (
            None if any(value is None for value in thresholds)
            else float(max(thresholds, default=0.0))
        ),
        "punishment_profile_is_pure_nash": punishment_nash,
        "continuation_assumption": "infinite or indefinite repetition with public monitoring and credible trigger punishment",
        "finite_horizon_warning": (
            "this threshold does not establish cooperation in a known finite-horizon game; "
            "with a unique stage-game Nash equilibrium, backward induction selects it in every period"
        ),
    }


def discounted_stationary_comparison(
    payoffs: np.ndarray,
    cooperative_profile: Sequence[int],
    nash_profile: Sequence[int],
    horizons: Iterable[int],
    discount_factor: float,
    *,
    project_welfare: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Compare repeated stationary profiles as accounting paths over finite H."""
    u = np.asarray(payoffs, dtype=np.float64)
    n_players, action_shape = _validate_payoffs(u)
    if not 0 < discount_factor <= 1:
        raise ValueError("discount_factor must lie in (0,1]")
    coop, nash = tuple(cooperative_profile), tuple(nash_profile)
    welfare = np.sum(u, axis=-1) if project_welfare is None else np.asarray(project_welfare, dtype=np.float64)
    if welfare.shape != action_shape:
        raise ValueError("project_welfare must match the payoff action dimensions")
    rows = []
    for horizon in horizons:
        if horizon < 1:
            raise ValueError("horizons must be positive integers")
        weights = np.asarray([discount_factor ** stage for stage in range(horizon)])
        coop_total = u[coop] * weights.sum()
        nash_total = u[nash] * weights.sum()
        rows.append({
            "horizon": int(horizon),
            "discount_factor": float(discount_factor),
            "cooperative_payoff_by_player": coop_total.tolist(),
            "nash_payoff_by_player": nash_total.tolist(),
            "cooperative_project_welfare": float(welfare[coop] * weights.sum()),
            "nash_project_welfare": float(welfare[nash] * weights.sum()),
            "cooperative_minus_nash_total_welfare": float(
                (welfare[coop] - welfare[nash]) * weights.sum()
            ),
            "interpretation": "stationary repeated stage-payoff comparison; physical reservoir state is held fixed",
        })
    return rows


def weitzman_price_quantity_sensitivity(spec: dict[str, Any]) -> dict[str, Any]:
    """Monte Carlo price-vs-cap sensitivity for a transparent linear-quadratic toy model."""
    required = ("mb_intercept_mean", "md_intercept_mean", "mb_slope", "md_slope",
                "mb_intercept_sd", "md_intercept_sd")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"missing parameters: {missing}")
    b0, d0, beta, delta, sd_b, sd_d = (float(spec[k]) for k in required)
    corr = float(spec.get("intercept_correlation", 0.0))
    draws = int(spec.get("draws", 20_000))
    seed = int(spec.get("seed", 0))
    if beta <= 0 or delta < 0 or sd_b < 0 or sd_d < 0 or not -1 <= corr <= 1 or draws < 100:
        raise ValueError("slopes/uncertainties/correlation/draw count are outside valid ranges")
    rng = np.random.default_rng(seed)
    covariance = np.asarray([[sd_b ** 2, corr * sd_b * sd_d],
                             [corr * sd_b * sd_d, sd_d ** 2]])
    shocks = rng.multivariate_normal([0.0, 0.0], covariance, size=draws)
    social_slope = beta + delta
    quantity = max(0.0, (b0 - d0) / social_slope)
    price = max(0.0, d0 + delta * quantity)
    q_price = np.maximum((b0 + shocks[:, 0] - price) / beta, 0.0)

    def surplus(q: np.ndarray | float) -> np.ndarray:
        return ((b0 + shocks[:, 0] - d0 - shocks[:, 1]) * q
                - 0.5 * social_slope * np.square(q))

    welfare_quantity = surplus(quantity)
    welfare_price = surplus(q_price)
    return {
        "quantity_cap": quantity,
        "per_unit_price": price,
        "mean_welfare_quantity": float(np.mean(welfare_quantity)),
        "mean_welfare_price": float(np.mean(welfare_price)),
        "mean_price_minus_quantity_welfare": float(np.mean(welfare_price - welfare_quantity)),
        "mean_quantity_under_price": float(np.mean(q_price)),
        "sd_quantity_under_price": float(np.std(q_price, ddof=1)),
        "price_zero_quantity_fraction": float(np.mean(q_price == 0.0)),
        "parameters": spec,
        "caveat": (
            "illustrative linear-quadratic uncertainty model only; use the solver-backed cap/tariff sweep "
            "for CO2 conclusions, and calibrate marginal-value/damage uncertainty before interpretation"
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze-game")
    analyze.add_argument("--game", type=Path, required=True, help="NPZ from build_co2_stage_game.py")
    analyze.add_argument("--out", type=Path, required=True)
    analyze.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 6, 12])
    analyze.add_argument("--discount-factor", type=float, default=1.0 / 1.05)
    analyze.add_argument("--bond-capital-cost-rate", type=float, default=0.0,
                         help="per-stage opportunity-cost fraction on secured bond capital")
    auction = sub.add_parser("auction")
    auction.add_argument("--values", type=Path, help="JSON with values_by_operator, block_kg")
    auction.add_argument("--game", type=Path,
                         help="no-cap solver stage-game NPZ; derives marginal values from unilateral schedules")
    auction.add_argument("--capacity-blocks", type=int, required=True)
    auction.add_argument("--shading", nargs="+", type=float, default=[0.5, 0.75, 1.0])
    auction.add_argument("--out", type=Path, required=True)
    weitzman = sub.add_parser("weitzman")
    weitzman.add_argument("--spec", type=Path, required=True)
    weitzman.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "analyze-game":
        with np.load(args.game, allow_pickle=False) as data:
            payoffs = data["payoffs"]
            gross = data["gross_payoffs_before_damage_liability"]
            damage = data["damage_liability_by_operator"]
            welfare = data["project_resource_welfare"]
            rate_levels = data["rate_levels_m3_s"].tolist()
        analysis = enumerate_equilibria(payoffs)
        analysis["correlated_equilibrium"] = welfare_maximizing_correlated_equilibrium(
            payoffs, social_welfare=welfare
        )
        pure_nash = analysis["pure_nash_profiles"]
        welfare_opt_index = tuple(map(int, np.unravel_index(np.argmax(welfare), welfare.shape)))
        welfare_opt = {
            "profile": list(welfare_opt_index),
            "project_resource_welfare": float(welfare[welfare_opt_index]),
            "private_payoffs": payoffs[welfare_opt_index].tolist(),
        }
        analysis["project_welfare_optimum"] = welfare_opt
        cooperative = welfare_opt["profile"]
        punishment = (min(pure_nash, key=lambda row: row["total_payoff"])["profile"]
                      if pure_nash else analysis["best_grid_approximate_equilibrium"]["profile"])
        analysis["grim_trigger_incentives"] = grim_trigger_threshold(payoffs, cooperative, punishment)
        analysis["stationary_horizon_sensitivity"] = discounted_stationary_comparison(
            payoffs, cooperative, punishment, args.horizons, args.discount_factor,
            project_welfare=welfare,
        )
        analysis["action_rate_levels_m3_s"] = rate_levels
        analysis["source_game_project_welfare"] = welfare.tolist()
        analysis["bond_sensitivity"] = []
        max_damage = float(np.max(damage)) if damage.size else 0.0
        for fraction in (0.0, 0.25, 0.5, 1.0, 2.0):
            bond = np.full(payoffs.shape[-1], max_damage * fraction)
            adjusted = financial_assurance_payoffs(
                gross, damage, bond,
                capital_cost_rate=args.bond_capital_cost_rate,
            )
            result = enumerate_equilibria(adjusted["private_payoffs"])
            result["bond_fraction_of_max_profile_damage"] = fraction
            result["bond_per_operator"] = bond.tolist()
            result["max_unsecured_damage_any_profile"] = float(
                np.max(adjusted["unsecured_damage"])
            )
            analysis["bond_sensitivity"].append({
                "bond_fraction_of_max_profile_damage": fraction,
                "bond_per_operator": bond.tolist(),
                "pure_nash_profiles": result["pure_nash_profiles"],
                "pure_nash_count": result["pure_nash_count"],
                "best_grid_approximate_equilibrium": result["best_grid_approximate_equilibrium"],
                "capital_cost_rate_per_stage": args.bond_capital_cost_rate,
                "capital_cost_by_operator": adjusted["capital_cost_by_operator"].tolist(),
                "max_unsecured_damage_any_profile": result["max_unsecured_damage_any_profile"],
                "project_welfare_at_pure_nash_profiles": [
                    {"profile": row["profile"],
                     "project_resource_welfare_before_bond_capital_cost": float(
                         welfare[tuple(row["profile"])]
                     ),
                     "project_resource_welfare_net_of_bond_capital_cost": float(
                         welfare[tuple(row["profile"])]
                         - np.sum(adjusted["capital_cost_by_operator"])
                     )}
                    for row in result["pure_nash_profiles"]
                ],
            })
    elif args.command == "auction":
        if (args.values is None) == (args.game is None):
            parser.error("provide exactly one of --values or --game")
        if args.game is not None:
            values, block_kg = marginal_values_from_stage_game(args.game)
            values_spec = {"values_source": "solver-derived unilateral marginal payoff blocks",
                           "block_kg": block_kg,
                           "caveat": "requires a no-cap/no-storage-cap stage-game table"}
        else:
            values_spec = json.loads(args.values.read_text(encoding="utf-8"))
            values = np.asarray(values_spec["values_by_operator"], dtype=np.float64)
            block_kg = float(values_spec["block_kg"])
        outcomes = []
        for shade in args.shading:
            if shade < 0:
                parser.error("bid shading factors must be nonnegative")
            bids = values * shade
            for mechanism in ("vcg", "pay_as_bid"):
                result = allocate_capacity_auction(
                    values, bids, args.capacity_blocks, mechanism=mechanism,
                    reserve_bid=float(values_spec.get("reserve_bid", 0.0)),
                )
                result["bid_shading_factor"] = shade
                result["block_mass_kg"] = block_kg
                result["allocated_mass_kg_by_operator"] = (
                    np.asarray(result["allocated_blocks_by_operator"])
                    * block_kg
                ).tolist()
                outcomes.append(result)
        result_payload = {"assumptions": values_spec, "outcomes": outcomes}
    else:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        result_payload = weitzman_price_quantity_sensitivity(spec)

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_safe(analysis if args.command == "analyze-game"
                                               else result_payload), indent=2), encoding="utf-8")
    print(f"Wrote {args.command} results to {out_path}")


if __name__ == "__main__":
    main()
