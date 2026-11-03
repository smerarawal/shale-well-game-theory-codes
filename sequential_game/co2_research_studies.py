"""Analysis tools for the CO2 geology/game research studies.

This module deliberately separates measured geology descriptors from game
outcomes.  It also provides solver finite-difference pressure externalities,
an auditable Pigouvian-price calculation, and shallow-tree policy distillation.
No coefficients are silently calibrated by this module.

Examples (Kaggle or local Python):
  python co2_research_studies.py describe-geologies --archive geology.npz --out descriptors.csv
  python co2_research_studies.py regime-map --descriptors descriptors.csv --outcomes game_results.csv --out regime_map.csv
  python co2_research_studies.py interference --geology geology_case.npz --out interference.json \\
      --well-cells 10,4 10,8 --rates 1e-7 1e-7 --rate-step 1e-8 \\
      --pressure-cap-pa 2000000 --pressure-damage-per-pa2 1e-10
  python co2_research_studies.py distill-policy --payoffs game_action_payoffs.csv --out policy.json \\
      --objective operator_0_payoff --features stage pressure_pa plume_fraction permit_remaining

The two-dimensional x-z solver and the game payoff coefficients remain
experimental/scenario assumptions.  Price outputs are therefore model-unit
estimates until the physical and economic scales are validated.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _connected_component(mask: np.ndarray) -> tuple[int, int, bool]:
    """Return largest 4-neighbor component, count, and x-spanning flag."""
    seen = np.zeros(mask.shape, dtype=bool)
    largest = 0
    count = 0
    spans = False
    nz, nx = mask.shape
    for z0, x0 in zip(*np.where(mask & ~seen)):
        count += 1
        stack = [(int(z0), int(x0))]
        seen[z0, x0] = True
        left, right, size = x0 == 0, x0 == nx - 1, 0
        while stack:
            z, x = stack.pop()
            size += 1
            left |= x == 0
            right |= x == nx - 1
            for zz, xx in ((z - 1, x), (z + 1, x), (z, x - 1), (z, x + 1)):
                if 0 <= zz < nz and 0 <= xx < nx and mask[zz, xx] and not seen[zz, xx]:
                    seen[zz, xx] = True
                    stack.append((zz, xx))
        largest = max(largest, size)
        spans |= bool(left and right)
    return largest, count, spans


def _correlation_length(field: np.ndarray, axis: int, spacing: float) -> float:
    """Estimate the e-folding autocorrelation length along one grid axis."""
    a = np.asarray(field, dtype=np.float64)
    a = a - a.mean()
    if axis == 0:
        max_lag = a.shape[0] - 1
        corr = [float(np.mean(a[:max_lag + 1 - lag] * a[lag:])) for lag in range(max_lag + 1)]
    else:
        max_lag = a.shape[1] - 1
        corr = [float(np.mean(a[:, :max_lag + 1 - lag] * a[:, lag:])) for lag in range(max_lag + 1)]
    if not corr or corr[0] <= 1e-20:
        return 0.0
    corr = np.asarray(corr) / corr[0]
    target = math.exp(-1.0)
    for lag in range(1, len(corr)):
        if corr[lag] <= target:
            lo, hi = corr[lag - 1], corr[lag]
            frac = 0.0 if abs(hi - lo) < 1e-12 else float(np.clip((target - lo) / (hi - lo), 0, 1))
            return float((lag - 1 + frac) * spacing)
    return float((len(corr) - 1) * spacing)


def geology_descriptors(kx: np.ndarray, phi: np.ndarray, kz: np.ndarray | None = None,
                        *, dx_ft: float = 1.0, dz_ft: float = 1.0,
                        high_k_quantile: float = 0.8) -> dict[str, float | int]:
    """Descriptive-only geology features for regime-map analyses."""
    kx = np.asarray(kx, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    if kx.ndim != 2 or phi.shape != kx.shape or np.any(kx <= 0) or np.any(phi <= 0):
        raise ValueError("kx and phi must be matching, positive [z,x] fields")
    kz_arr = kx if kz is None else np.asarray(kz, dtype=np.float64)
    if kz_arr.shape != kx.shape or np.any(kz_arr <= 0):
        raise ValueError("kz must be positive and match kx")
    logkx = np.log(kx)
    threshold = float(np.quantile(kx, high_k_quantile))
    largest, component_count, spans = _connected_component(kx >= threshold)
    return {
        "logkx_std": float(logkx.std()),
        "logkz_std": float(np.log(kz_arr).std()),
        "phi_mean": float(phi.mean()),
        "phi_std": float(phi.std()),
        "kx_kz_geomean_ratio": float(np.exp(np.mean(logkx - np.log(kz_arr)))),
        "high_k_threshold": threshold,
        "high_k_fraction": float(np.mean(kx >= threshold)),
        "largest_high_k_component_fraction": float(largest / kx.size),
        "high_k_component_count": int(component_count),
        "high_k_spans_x": int(spans),
        "corr_length_x_ft": _correlation_length(logkx, 1, dx_ft),
        "corr_length_z_ft": _correlation_length(logkx, 0, dz_ft),
    }


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def describe_archive(archive_path: Path) -> list[dict[str, Any]]:
    """Read synthetic/SPE10 geology cohort NPZ and return one row per field."""
    rows: list[dict[str, Any]] = []
    with np.load(archive_path, allow_pickle=True) as z:
        for key in z.files:
            if not key.endswith("_phi"):
                continue
            source_cohort = key[:-4]
            cohort = source_cohort
            if source_cohort.startswith("spe10_xz_"):
                cohort = source_cohort.removeprefix("spe10_xz_")
            elif source_cohort.startswith("spe10_"):
                cohort = source_cohort.removeprefix("spe10_")
            phi_arr = np.asarray(z[key])
            if phi_arr.ndim == 2:
                phi_arr = phi_arr[None]
            # Preserve the archive prefix for field lookup (for example,
            # ``spe10_xz_tarbert_kx_mD``), while keeping the public cohort
            # label concise (``tarbert``).
            kx_key = (f"{source_cohort}_kx_mD" if f"{source_cohort}_kx_mD" in z
                      else f"{source_cohort}_k")
            kz_key = f"{source_cohort}_kz_mD" if f"{source_cohort}_kz_mD" in z else kx_key
            if kx_key not in z:
                continue
            kx_arr = np.asarray(z[kx_key])
            kz_arr = np.asarray(z[kz_key]) if kz_key in z else kx_arr
            if kx_arr.ndim == 2:
                kx_arr, kz_arr = kx_arr[None], kz_arr[None]
            seeds = (np.asarray(z[f"{source_cohort}_seed"]) if f"{source_cohort}_seed" in z
                     else np.arange(len(kx_arr)))
            metas = (np.asarray(z[f"{source_cohort}_metadata_json"], dtype=object)
                     if f"{source_cohort}_metadata_json" in z
                     else np.array([{}] * len(kx_arr), dtype=object))
            mults = (np.asarray(z[f"{source_cohort}_mult"]) if f"{source_cohort}_mult" in z
                     else np.ones(len(kx_arr)))
            for i, (kx, kz, phi) in enumerate(zip(kx_arr, kz_arr, phi_arr)):
                meta = _metadata(metas[i]) if i < len(metas) else {}
                desc = geology_descriptors(kx, phi, kz,
                                           dx_ft=float(meta.get("dx_ft", 1.0)),
                                           dz_ft=float(meta.get("dz_ft", 1.0)))
                rows.append({"cohort": cohort, "source_cohort": source_cohort,
                             "geology_seed": int(seeds[i]),
                             "mult": float(mults[i]), **meta, **desc})
    return rows


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def join_regime_outcomes(descriptors: Sequence[Mapping[str, Any]],
                         outcomes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Join outcome records to descriptor rows using cohort and geology seed."""
    by_key = {(str(r.get("cohort", "")), str(r.get("geology_seed", r.get("seed", "")))): r
              for r in descriptors}
    joined = []
    for outcome in outcomes:
        cohort = str(outcome.get("cohort", ""))
        seed = str(outcome.get("geology_seed", outcome.get("seed", "")))
        desc = by_key.get((cohort, seed))
        if desc is None:
            continue
        joined.append({**desc, **outcome})
    return joined


def _rank(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(order), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return ranks


def rank_correlations(rows: Sequence[Mapping[str, Any]], outcome: str,
                      descriptors: Sequence[str]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for feature in descriptors:
        pairs = []
        for row in rows:
            try:
                pairs.append((float(row[feature]), float(row[outcome])))
            except (KeyError, TypeError, ValueError):
                continue
        if len(pairs) < 3:
            result[feature] = None
            continue
        x, y = np.asarray(pairs).T
        rx, ry = _rank(x), _rank(y)
        if np.std(rx) <= 0 or np.std(ry) <= 0:
            result[feature] = None
        else:
            result[feature] = float(np.corrcoef(rx, ry)[0, 1])
    return result


def _load_solver():
    try:
        from .co2_xz_solver import CO2XZConfig, CO2XZSolver, initial_state
    except ImportError:
        from co2_xz_solver import CO2XZConfig, CO2XZSolver, initial_state
    return CO2XZConfig, CO2XZSolver, initial_state


def estimate_interference(
    geology: Mapping[str, np.ndarray], well_cells: Sequence[tuple[int, int]],
    rates_m3_s: Sequence[float], *, rate_step_m3_s: float,
    stage_days: float, pressure_cap_pa: float, pressure_damage_per_pa2: float,
    solver_config: Any | None = None, pipeline_capacity_kg_per_stage: float | None = None,
    dispatch_rule: str = "proportional",
) -> dict[str, Any]:
    """Finite-difference cross-pressure and damage matrix using the CO2 solver.

    Matrix convention: ``matrix[sensor_operator][perturbed_injector]``.
    Cross-pressure entries use pressure at each operator's well cell. The
    Pigouvian externality uses the current game kernel's pooled pmax pressure
    penalty and computes the finite-difference change in the portion allocated
    to rival operators. This keeps the price derivative aligned with the
    implemented payoff rule while retaining a spatial interference matrix.
    """
    CO2XZConfig, CO2XZSolver, initial_state = _load_solver()
    kx, kz, phi = (np.asarray(geology[k], dtype=np.float64)
                   for k in ("kx_mD", "kz_mD", "phi"))
    n = len(well_cells)
    if n < 2 or len(rates_m3_s) != n:
        raise ValueError("provide at least two well cells and one rate per operator")
    if rate_step_m3_s <= 0 or stage_days <= 0 or pressure_cap_pa < 0 or pressure_damage_per_pa2 < 0:
        raise ValueError("finite-difference step/duration must be positive and penalty parameters nonnegative")
    if solver_config is None:
        solver_config = CO2XZConfig()
    solver = CO2XZSolver(solver_config)
    dt = float(stage_days) * 86400.0
    rho = float(solver.cfg.rho_co2_kg_m3)

    try:
        from .co2_pipeline_game import dispatch_mass
    except ImportError:
        from co2_pipeline_game import dispatch_mass

    def delivered(rates: np.ndarray) -> np.ndarray:
        requested_mass = rates * rho * dt
        if pipeline_capacity_kg_per_stage is None:
            return requested_mass
        return dispatch_mass(requested_mass, pipeline_capacity_kg_per_stage, dispatch_rule)

    def solve(rates: np.ndarray) -> tuple[np.ndarray, float]:
        actual_rates = delivered(rates) / (rho * dt)
        state, diag, _ = solver.advance(
            kx, kz, phi, initial_state(phi.shape),
            [(int(z), int(x), float(q)) for (z, x), q in zip(well_cells, actual_rates) if q > 0], dt)
        p = state["p_pa"]
        at_wells = np.asarray([p[z, x] for z, x in well_cells], dtype=np.float64)
        return at_wells, float(diag["pmax_pa"])

    base_rates = np.asarray(rates_m3_s, dtype=np.float64)
    if np.any(base_rates < rate_step_m3_s):
        raise ValueError("base rates must be at least rate_step_m3_s for centered differences")
    base_p, base_pmax = solve(base_rates)
    pressure = np.zeros((n, n), dtype=np.float64)
    pmax_sensitivity = np.zeros(n, dtype=np.float64)
    damage_price = np.zeros(n, dtype=np.float64)
    for injector in range(n):
        plus, minus = base_rates.copy(), base_rates.copy()
        plus[injector] += rate_step_m3_s
        minus[injector] -= rate_step_m3_s
        actual_plus = delivered(plus)
        actual_minus = delivered(minus)
        marginal_injected_mass = (actual_plus[injector] - actual_minus[injector]) / 2.0
        if marginal_injected_mass <= 1e-12:
            raise ValueError("pipeline dispatch leaves no positive marginal injection for this operator")
        p_plus, pmax_plus = solve(plus)
        p_minus, pmax_minus = solve(minus)
        dp_dkg = (p_plus - p_minus) / (2.0 * marginal_injected_mass)
        dmax_dkg = (pmax_plus - pmax_minus) / (2.0 * marginal_injected_mass)
        pressure[:, injector] = dp_dkg / 1000.0  # kPa per kg over the declared stage
        pmax_sensitivity[injector] = dmax_dkg / 1000.0
        masses_plus = actual_plus
        masses_minus = actual_minus
        rival_share_plus = (masses_plus.sum() - masses_plus[injector]) / max(masses_plus.sum(), 1e-12)
        rival_share_minus = (masses_minus.sum() - masses_minus[injector]) / max(masses_minus.sum(), 1e-12)
        damage_plus = (pressure_damage_per_pa2
                       * max(0.0, pmax_plus - pressure_cap_pa) ** 2 * rival_share_plus)
        damage_minus = (pressure_damage_per_pa2
                        * max(0.0, pmax_minus - pressure_cap_pa) ** 2 * rival_share_minus)
        external = (damage_plus - damage_minus) / (2.0 * marginal_injected_mass)
        damage_price[injector] = external
    return {
        "units": {"rates": "m3/s", "stage_duration": "days", "pressure": "Pa; matrix in kPa/kg",
                  "damage_price": "payoff units/kg injected during the declared stage"},
        "stage_days": float(stage_days), "base_rates_m3_s": base_rates.tolist(),
        "base_pressure_at_wells_pa": base_p.tolist(), "base_pmax_pa": base_pmax,
        "pressure_sensitivity_kpa_per_kg": pressure.tolist(),
        "pmax_sensitivity_kpa_per_kg": pmax_sensitivity.tolist(),
        "pigouvian_external_damage_price_per_kg": damage_price.tolist(),
        "pressure_cap_pa": float(pressure_cap_pa),
        "pressure_damage_per_pa2": float(pressure_damage_per_pa2),
        "pipeline_capacity_kg_per_stage": pipeline_capacity_kg_per_stage,
        "dispatch_rule": dispatch_rule if pipeline_capacity_kg_per_stage is not None else "uncapped",
        "damage_definition": "finite difference of pooled pmax pressure penalty times the rival injection-share allocation used by co2_pipeline_game",
        "caveat": "local centered finite difference from the experimental 2D x-z solver; not calibrated currency or a full dynamic equilibrium price",
    }


def distill_policy(payoff_rows: Sequence[Mapping[str, str]], *, objective: str,
                   feature_names: Sequence[str], max_depth: int = 3,
                   min_samples_leaf: int = 5, seed: int = 0) -> dict[str, Any]:
    """Fit a shallow decision tree to action-level solver payoff tables.

    Expected CSV: one row per ``state_id, action_id`` with feature columns,
    ``split`` (train/validation/test), and the objective payoff. Each state's
    selected teacher action is the payoff-maximizing feasible action. Regret is
    measured on held-out states by looking up the tree-selected action's actual
    solver payoff in the same counterfactual table.
    """
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("scikit-learn is required for policy distillation") from exc
    grouped: dict[str, list[Mapping[str, str]]] = {}
    for row in payoff_rows:
        grouped.setdefault(str(row["state_id"]), []).append(row)
    states = []
    for state_id, rows in grouped.items():
        if any(feature not in rows[0] for feature in feature_names):
            raise ValueError("feature column missing from payoff table")
        feasible = [r for r in rows if r.get(objective, "") not in ("", "nan", "None")]
        if not feasible:
            continue
        teacher = max(feasible, key=lambda r: float(r[objective]))
        states.append({"state_id": state_id,
                       "split": str(teacher.get("split", "train")),
                       "x": [float(teacher[f]) for f in feature_names],
                       "teacher_action": str(teacher["action_id"]),
                       "actions": {str(r["action_id"]): float(r[objective]) for r in feasible}})
    train = [s for s in states if s["split"] == "train"]
    test = [s for s in states if s["split"] in ("test", "validation")]
    if len(train) < 2 or not test:
        raise ValueError("need at least two train states and one validation/test state")
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                   random_state=seed)
    model.fit(np.asarray([s["x"] for s in train]), np.asarray([s["teacher_action"] for s in train]))
    regrets, matches = [], []
    for state in test:
        action = str(model.predict(np.asarray([state["x"]]))[0])
        if action not in state["actions"]:
            # A tree can select an action absent from a particular state's
            # feasible set; mark the decision infeasible rather than hiding it.
            regrets.append(None)
            matches.append(False)
            continue
        best = max(state["actions"].values())
        regrets.append(float(best - state["actions"][action]))
        matches.append(action == state["teacher_action"])
    valid_regrets = [r for r in regrets if r is not None]
    return {
        "objective": objective, "features": list(feature_names),
        "max_depth": int(max_depth), "min_samples_leaf": int(min_samples_leaf),
        "train_states": len(train), "heldout_states": len(test),
        "heldout_action_match_fraction": float(np.mean(matches)),
        "heldout_infeasible_fraction": float(1.0 - len(valid_regrets) / len(test)),
        "heldout_mean_payoff_regret": float(np.mean(valid_regrets)) if valid_regrets else None,
        "heldout_p95_payoff_regret": float(np.percentile(valid_regrets, 95)) if valid_regrets else None,
        "tree_rules": export_text(model, feature_names=list(feature_names)),
        "interpretation": "regret is evaluated on the provided solver payoff table; a policy tree is not an equilibrium proof",
    }


def compare_price_sweep(theoretical: Mapping[str, Any], rows: Sequence[Mapping[str, str]],
                        *, price_column: str = "price_per_kg",
                        welfare_column: str = "project_resource_welfare") -> dict[str, Any]:
    """Compare a local solver-derived externality price with empirical sweep.

    The empirical table must contain one record per paired case and price.
    The function averages welfare at each tested price; callers should retain
    the paired records and bootstrap over geology seeds for inference.
    """
    groups: dict[float, list[float]] = {}
    for row in rows:
        try:
            price, welfare = float(row[price_column]), float(row[welfare_column])
        except (KeyError, ValueError):
            continue
        groups.setdefault(price, []).append(welfare)
    if not groups:
        raise ValueError("empirical price table has no valid price/welfare rows")
    means = {str(price): float(np.mean(values)) for price, values in sorted(groups.items())}
    best_price = max(groups, key=lambda price: float(np.mean(groups[price])))
    local = np.asarray(theoretical.get("pigouvian_external_damage_price_per_kg", []), dtype=float)
    local_price = float(np.mean(local)) if local.size else None
    return {
        "theoretical_local_price_per_kg_by_operator": local.tolist(),
        "theoretical_mean_local_price_per_kg": local_price,
        "empirical_mean_project_welfare_by_price": means,
        "best_tested_empirical_price_per_kg": float(best_price),
        "absolute_gap_price_per_kg": None if local_price is None else float(abs(best_price - local_price)),
        "caveat": "local derivative and empirical optimum are comparable only when they use the same payoff units, capacity regime, and geology distribution",
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    desc = sub.add_parser("describe-geologies")
    desc.add_argument("--archive", required=True, type=Path)
    desc.add_argument("--out", required=True, type=Path)
    regime = sub.add_parser("regime-map")
    regime.add_argument("--descriptors", required=True, type=Path)
    regime.add_argument("--outcomes", required=True, type=Path)
    regime.add_argument("--out", required=True, type=Path)
    regime.add_argument("--summary-out", type=Path)
    regime.add_argument("--outcome", default="commons_gap_coop_minus_nash")
    regime.add_argument("--direction", choices=("maximize", "minimize"), default="minimize",
                        help="whether lower or higher outcome defines the preferred arm")
    inter = sub.add_parser("interference")
    inter.add_argument("--geology", required=True, type=Path,
                       help="NPZ with 2D kx_mD, kz_mD, phi arrays")
    inter.add_argument("--out", required=True, type=Path)
    inter.add_argument("--well-cells", required=True, nargs="+", help="one z,x pair per operator")
    inter.add_argument("--rates", required=True, nargs="+", type=float)
    inter.add_argument("--rate-step", type=float, default=1e-8)
    inter.add_argument("--stage-days", type=float, default=30.0)
    inter.add_argument("--pressure-cap-pa", required=True, type=float)
    inter.add_argument("--pressure-damage-per-pa2", required=True, type=float)
    inter.add_argument("--pipeline-capacity-kg-per-stage", type=float)
    inter.add_argument("--dispatch-rule", choices=("proportional", "cea", "cel", "fixed_equal"),
                       default="proportional")
    inter.add_argument("--dx-ft", type=float, default=20.0)
    inter.add_argument("--dz-ft", type=float, default=2.0)
    inter.add_argument("--dy-ft", type=float, default=10.0)
    inter.add_argument("--max-dt-days", type=float, default=1.0)
    inter.add_argument("--row", type=int, default=0,
                       help="case index if the geology NPZ stores [N,z,x] arrays")
    dist = sub.add_parser("distill-policy")
    dist.add_argument("--payoffs", required=True, type=Path)
    dist.add_argument("--out", required=True, type=Path)
    dist.add_argument("--objective", required=True)
    dist.add_argument("--features", required=True, nargs="+")
    dist.add_argument("--max-depth", type=int, default=3)
    dist.add_argument("--min-samples-leaf", type=int, default=5)
    price = sub.add_parser("compare-price")
    price.add_argument("--theoretical", required=True, type=Path)
    price.add_argument("--empirical", required=True, type=Path)
    price.add_argument("--out", required=True, type=Path)
    price.add_argument("--price-column", default="price_per_kg")
    price.add_argument("--welfare-column", default="project_resource_welfare")
    args = parser.parse_args()

    if args.command == "describe-geologies":
        rows = describe_archive(args.archive)
        write_csv(rows, args.out)
        print(f"Wrote {len(rows)} field descriptors to {args.out}")
    elif args.command == "regime-map":
        joined = join_regime_outcomes(_read_csv(args.descriptors), _read_csv(args.outcomes))
        if not joined:
            raise ValueError("no joined rows; outcome cohort and geology_seed/seed must match descriptors")
        write_csv(joined, args.out)
        corr = rank_correlations(joined, args.outcome,
                                 ("largest_high_k_component_fraction", "high_k_spans_x",
                                  "corr_length_x_ft", "corr_length_z_ft", "logkx_std", "phi_std"))
        summary = {"outcome": args.outcome, "selection_direction": args.direction,
                   "joined_rows": len(joined),
                   "spearman_by_descriptor": corr,
                   "best_instrument_by_cohort": {}}
        by_cohort: dict[str, list[dict[str, str]]] = {}
        for row in joined:
            by_cohort.setdefault(str(row.get("cohort", "unknown")), []).append(row)
        for cohort, rows in by_cohort.items():
            instrument_key = "instrument" if "instrument" in rows[0] else "best_instrument"
            totals: dict[str, list[float]] = {}
            for row in rows:
                if row.get(instrument_key):
                    try:
                        totals.setdefault(str(row[instrument_key]), []).append(float(row[args.outcome]))
                    except (ValueError, KeyError):
                        pass
            means = {key: float(np.mean(vals)) for key, vals in totals.items()}
            if means:
                summary["best_instrument_by_cohort"][cohort] = {
                    "mean_outcome_by_instrument": means,
                    "selected_instrument_by_mean_outcome": (
                        min(means, key=means.get) if args.direction == "minimize"
                        else max(means, key=means.get))}
        out_summary = args.summary_out or args.out.with_suffix(".summary.json")
        out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote joined regime map to {args.out}; summary to {out_summary}")
    elif args.command == "interference":
        with np.load(args.geology, allow_pickle=False) as z:
            geo = {k: z[k] for k in ("kx_mD", "kz_mD", "phi")}
        for key, value in geo.items():
            if value.ndim == 3:
                if not 0 <= args.row < len(value):
                    raise IndexError(f"--row {args.row} is outside {key} with {len(value)} cases")
                geo[key] = value[args.row]
        cells = [tuple(map(int, pair.split(","))) for pair in args.well_cells]
        if len(cells) != len(args.rates):
            raise ValueError("--well-cells and --rates must have the same length")
        CO2XZConfig, _, _ = _load_solver()
        cfg = CO2XZConfig(dx_ft=args.dx_ft, dz_ft=args.dz_ft, dy_ft=args.dy_ft,
                          max_dt_s=args.max_dt_days * 86400.0)
        result = estimate_interference(
            geo, cells, args.rates, rate_step_m3_s=args.rate_step,
            stage_days=args.stage_days, pressure_cap_pa=args.pressure_cap_pa,
            pressure_damage_per_pa2=args.pressure_damage_per_pa2, solver_config=cfg,
            pipeline_capacity_kg_per_stage=args.pipeline_capacity_kg_per_stage,
            dispatch_rule=args.dispatch_rule)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote pressure interference and local price estimate to {args.out}")
    elif args.command == "distill-policy":
        result = distill_policy(_read_csv(args.payoffs), objective=args.objective,
                                feature_names=args.features, max_depth=args.max_depth,
                                min_samples_leaf=args.min_samples_leaf)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote policy tree and solver-table regret to {args.out}")
    else:
        theoretical = json.loads(args.theoretical.read_text(encoding="utf-8"))
        result = compare_price_sweep(theoretical, _read_csv(args.empirical),
                                    price_column=args.price_column,
                                    welfare_column=args.welfare_column)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote theoretical/empirical price comparison to {args.out}")


if __name__ == "__main__":
    _main()
