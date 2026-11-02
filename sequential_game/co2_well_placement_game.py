"""Well-placement extension for the shared-supply CO2 sequential game.

This adapter gives wells stable identities and makes placement an explicit
operator action while reusing ``co2_pipeline_game.advance_stage`` for source
allocation, the shared pipeline bottleneck, reservoir injection, and payoff
accounting. The reservoir solver currently represents wells as point sources
on an x-z grid; trajectories and network routing belong in later adapters.

Monitoring wells can be placed and charged as wells, but they only provide
information when a caller supplies ``monitoring_update``. No observations or
belief updates are fabricated by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from .co2_pipeline_game import (
        CO2GameConfig,
        CO2GameState,
        CO2XZSolver,
        OperatorAction as GridOperatorAction,
        advance_stage as advance_grid_stage,
        new_game_state,
    )
except ImportError:  # direct execution from sequential_game/
    from co2_pipeline_game import (
        CO2GameConfig,
        CO2GameState,
        CO2XZSolver,
        OperatorAction as GridOperatorAction,
        advance_stage as advance_grid_stage,
        new_game_state,
    )


WELL_TYPES = {"injector", "monitor"}


@dataclass(frozen=True)
class PlacementRules:
    """Placement constraints; omitted candidate lists mean any grid cell.

    ``candidate_cells_by_operator`` is an optional sequence with one set of
    legal ``(z, x)`` cells per operator. ``min_spacing_ft`` applies between
    all wells in physical grid coordinates, accounting for dx and dz. Limits count both injector and monitor
    wells. ``drilling_lead_stages=0`` means a new injector can operate in its
    placement stage; positive values model drilling/commissioning delay.
    """

    candidate_cells_by_operator: tuple[frozenset[tuple[int, int]], ...] | None = None
    min_spacing_ft: float = 0.0
    max_wells_per_operator: int | tuple[int, ...] | None = None
    max_new_wells_per_operator_per_stage: int | None = None
    max_rate_m3_s_per_well: float | None = None
    drilling_lead_stages: int = 0

    def __post_init__(self):
        if not np.isfinite(self.min_spacing_ft) or self.min_spacing_ft < 0:
            raise ValueError("min_spacing_ft must be finite and nonnegative")
        if self.drilling_lead_stages < 0:
            raise ValueError("drilling_lead_stages must be nonnegative")
        if (self.max_new_wells_per_operator_per_stage is not None
                and self.max_new_wells_per_operator_per_stage < 0):
            raise ValueError("max_new_wells_per_operator_per_stage must be nonnegative")
        if (self.max_rate_m3_s_per_well is not None
                and (not np.isfinite(self.max_rate_m3_s_per_well)
                     or self.max_rate_m3_s_per_well < 0)):
            raise ValueError("max_rate_m3_s_per_well must be finite and nonnegative")
        caps = self.max_wells_per_operator
        if caps is not None:
            values = (caps,) if isinstance(caps, int) else tuple(caps)
            if not values or any(v < 0 for v in values):
                raise ValueError("max_wells_per_operator must contain nonnegative limits")


@dataclass(frozen=True)
class WellPlacement:
    """A requested well installation with a stable ID."""

    well_id: str
    z_index: int
    x_index: int
    well_type: str = "injector"


@dataclass(frozen=True)
class WellRate:
    """Requested CO2 rate for one previously or newly placed injector."""

    well_id: str
    rate_m3_s: float


@dataclass(frozen=True)
class OperatorDecision:
    """One operator's joint placement and per-well control action."""

    placements: tuple[WellPlacement, ...] = ()
    rates: tuple[WellRate, ...] = ()


@dataclass(frozen=True)
class WellAsset:
    """A drilled, persistent well in the Markov state."""

    well_id: str
    owner: int
    z_index: int
    x_index: int
    well_type: str
    drilled_stage: int
    ready_stage: int
    last_realized_rate_m3_s: float = 0.0
    cumulative_injected_kg: float = 0.0


@dataclass
class CO2MarkovState:
    """Complete game state: physical/economic state plus persistent wells."""

    core: CO2GameState
    wells: tuple[WellAsset, ...] = ()

    @property
    def stage(self) -> int:
        return self.core.stage


MonitoringUpdate = Callable[
    [int, Mapping[str, Any], Mapping[str, np.ndarray], tuple[WellAsset, ...],
     Sequence[OperatorDecision], tuple[Any, ...]],
    tuple[Sequence[Any], Sequence[Any]],
]


def new_markov_state(
    solver: CO2XZSolver,
    shape: tuple[int, int],
    config: CO2GameConfig,
    *,
    initial_beliefs: Sequence[Any] | None = None,
) -> CO2MarkovState:
    """Create an empty-well state for any configured number of operators."""
    n_ops = config.n_operators
    if initial_beliefs is None:
        beliefs = tuple({} for _ in range(n_ops))
    else:
        beliefs = tuple(initial_beliefs)
        if len(beliefs) != n_ops:
            raise ValueError("initial_beliefs must contain one belief per operator")
    core = new_game_state(solver, shape, n_ops, belief_state=beliefs)
    return CO2MarkovState(core=core)


def _operator_cap(rules: PlacementRules, operator: int, n_ops: int) -> int | None:
    caps = rules.max_wells_per_operator
    if caps is None:
        return None
    if isinstance(caps, int):
        return caps
    if len(caps) != n_ops:
        raise ValueError("max_wells_per_operator tuple must have one entry per operator")
    return int(caps[operator])


def _validate_and_translate(
    *,
    state: CO2MarkovState,
    decisions: Sequence[OperatorDecision],
    rules: PlacementRules,
    shape: tuple[int, int],
    n_ops: int,
    dx_ft: float,
    dz_ft: float,
) -> tuple[tuple[WellAsset, ...], list[GridOperatorAction], dict[str, Any]]:
    if len(decisions) != n_ops:
        raise ValueError("one OperatorDecision is required for each operator")
    if rules.candidate_cells_by_operator is not None and len(rules.candidate_cells_by_operator) != n_ops:
        raise ValueError("candidate_cells_by_operator must have one candidate set per operator")

    registry = {well.well_id: well for well in state.wells}
    if len(registry) != len(state.wells):
        raise ValueError("well IDs in the state must be unique")
    new_assets: list[WellAsset] = []
    placed_cells: dict[tuple[int, int], str] = {}
    for well in state.wells:
        if not 0 <= well.owner < n_ops:
            raise ValueError(f"well {well.well_id!r} has an invalid operator owner")
        if well.well_type not in WELL_TYPES or well.ready_stage < well.drilled_stage:
            raise ValueError(f"well {well.well_id!r} has invalid type or commissioning stages")
        cell = (well.z_index, well.x_index)
        if cell in placed_cells:
            raise ValueError(f"multiple wells occupy site {cell}")
        placed_cells[cell] = well.well_id
    new_wells_per_operator = [0] * n_ops
    rate_controls_by_operator: list[list[tuple[str, float, WellAsset]]] = [
        [] for _ in range(n_ops)
    ]
    stage_events: list[dict[str, Any]] = []

    for operator, decision in enumerate(decisions):
        candidate_cells = (
            None if rules.candidate_cells_by_operator is None
            else rules.candidate_cells_by_operator[operator]
        )
        for placement in decision.placements:
            if not isinstance(placement.well_id, str) or not placement.well_id.strip():
                raise ValueError("well_id must be a nonempty stable identifier")
            well_id = placement.well_id.strip()
            if (int(placement.z_index) != placement.z_index
                    or int(placement.x_index) != placement.x_index):
                raise ValueError("well placement indices must be integers")
            cell = (int(placement.z_index), int(placement.x_index))
            if well_id in registry or any(w.well_id == well_id for w in new_assets):
                raise ValueError(f"well_id {well_id!r} has already been used")
            if placement.well_type not in WELL_TYPES:
                raise ValueError(f"well_type must be one of {sorted(WELL_TYPES)}")
            if not (0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]):
                raise ValueError(f"well {well_id!r} is outside the reservoir grid: {cell}")
            if candidate_cells is not None and cell not in candidate_cells:
                raise ValueError(f"operator {operator} cannot place a well at {cell}")
            if cell in placed_cells:
                raise ValueError(f"site {cell} is already occupied by well {placed_cells[cell]!r}")
            for old_cell in placed_cells:
                distance_ft = hypot(
                    (cell[0] - old_cell[0]) * dz_ft,
                    (cell[1] - old_cell[1]) * dx_ft,
                )
                if distance_ft < rules.min_spacing_ft:
                    raise ValueError(
                        f"well {well_id!r} violates minimum well spacing at {old_cell}"
                    )
            new_wells_per_operator[operator] += 1
            if (rules.max_new_wells_per_operator_per_stage is not None
                    and new_wells_per_operator[operator]
                    > rules.max_new_wells_per_operator_per_stage):
                raise ValueError(f"operator {operator} exceeds its per-stage drilling limit")
            cap = _operator_cap(rules, operator, n_ops)
            old_count = sum(w.owner == operator for w in state.wells)
            if cap is not None and old_count + new_wells_per_operator[operator] > cap:
                raise ValueError(f"operator {operator} exceeds its total well limit")

            asset = WellAsset(
                well_id=well_id,
                owner=operator,
                z_index=cell[0],
                x_index=cell[1],
                well_type=placement.well_type,
                drilled_stage=state.stage,
                ready_stage=state.stage + rules.drilling_lead_stages,
            )
            new_assets.append(asset)
            placed_cells[cell] = well_id
            stage_events.append({
                "well_id": well_id,
                "operator": operator,
                "well_type": asset.well_type,
                "cell_zx": list(cell),
                "drilled_stage": asset.drilled_stage,
                "ready_stage": asset.ready_stage,
            })

        seen_rate_ids: set[str] = set()
        for rate_action in decision.rates:
            if rate_action.well_id in seen_rate_ids:
                raise ValueError(f"duplicate rate control for well {rate_action.well_id!r}")
            seen_rate_ids.add(rate_action.well_id)
            asset = registry.get(rate_action.well_id)
            if asset is None:
                asset = next((w for w in new_assets if w.well_id == rate_action.well_id), None)
            if asset is None:
                raise ValueError(f"unknown well ID {rate_action.well_id!r}")
            if asset.owner != operator:
                raise ValueError(f"operator {operator} does not own well {asset.well_id!r}")
            if asset.well_type != "injector":
                raise ValueError(f"monitor well {asset.well_id!r} cannot inject CO2")
            if asset.ready_stage > state.stage:
                raise ValueError(
                    f"well {asset.well_id!r} is not commissioned until stage {asset.ready_stage}"
                )
            if not np.isfinite(rate_action.rate_m3_s) or rate_action.rate_m3_s < 0:
                raise ValueError("well rates must be finite and nonnegative; zero means shut in")
            if (rules.max_rate_m3_s_per_well is not None
                    and rate_action.rate_m3_s > rules.max_rate_m3_s_per_well):
                raise ValueError(f"well {asset.well_id!r} exceeds its maximum injection rate")
            rate_controls_by_operator[operator].append(
                (asset.well_id, float(rate_action.rate_m3_s), asset)
            )

    all_assets = tuple(state.wells) + tuple(new_assets)
    translated = []
    for operator in range(n_ops):
        solver_rates = tuple(
            (asset.z_index, asset.x_index, rate)
            for _, rate, asset in rate_controls_by_operator[operator]
            if rate > 0
        )
        # The core transition keeps ownership of every drilled site, including
        # wells that are shut in and monitoring wells that have no injection.
        all_new_cells = tuple(
            (asset.z_index, asset.x_index)
            for asset in new_assets if asset.owner == operator
        )
        translated.append(GridOperatorAction(wells=solver_rates, new_wells=all_new_cells))

    rate_events = []
    for operator, controls in enumerate(rate_controls_by_operator):
        for well_id, rate, asset in controls:
            rate_events.append({
                "well_id": well_id,
                "operator": operator,
                "well_type": asset.well_type,
                "requested_rate_m3_s": rate,
            })
    return all_assets, translated, {"placements": stage_events, "rate_controls": rate_events}


def advance_markov_stage(
    *,
    solver: CO2XZSolver,
    config: CO2GameConfig,
    state: CO2MarkovState,
    geology: Mapping[str, np.ndarray],
    decisions: Sequence[OperatorDecision],
    placement_rules: PlacementRules = PlacementRules(),
    gravity_sign: int = 1,
    risk_metrics: Mapping[str, float] | None = None,
    monitoring_update: MonitoringUpdate | None = None,
) -> tuple[CO2MarkovState, dict[str, Any]]:
    """Apply placement, shared dispatch, reservoir, and payoff updates for a stage."""
    if state.stage >= config.horizon:
        raise ValueError("game is already at or beyond its terminal stage")
    kx = np.asarray(geology["kx_mD"], dtype=np.float64)
    kz = np.asarray(geology["kz_mD"], dtype=np.float64)
    phi = np.asarray(geology["phi"], dtype=np.float64)
    if kx.ndim != 2 or kz.shape != kx.shape or phi.shape != kx.shape:
        raise ValueError("geology arrays kx_mD, kz_mD, phi must share a [z,x] shape")
    n_ops = config.n_operators
    if state.core.cumulative_injected_kg_by_operator.shape != (n_ops,):
        raise ValueError("state operator ledgers do not match game configuration")

    all_assets, translated, placement_and_rate_events = _validate_and_translate(
        state=state,
        decisions=decisions,
        rules=placement_rules,
        shape=kx.shape,
        n_ops=n_ops,
        dx_ft=solver.cfg.dx_ft,
        dz_ft=solver.cfg.dz_ft,
    )
    def belief_adapter(stage, reservoir, observed_geology, _grid_actions, beliefs):
        if monitoring_update is None:
            return tuple(() for _ in range(n_ops)), beliefs
        available_monitors = tuple(
            well for well in all_assets
            if well.well_type == "monitor" and well.ready_stage <= stage + 1
        )
        return monitoring_update(
            stage, reservoir, observed_geology, available_monitors, decisions, beliefs
        )

    next_core, record = advance_grid_stage(
        solver=solver,
        config=config,
        state=state.core,
        geology=geology,
        actions=translated,
        gravity_sign=gravity_sign,
        risk_metrics=risk_metrics,
        belief_update=belief_adapter,
    )

    requested_mass = np.asarray(record["requested_mass_kg_by_operator"], dtype=np.float64)
    delivered_mass = np.asarray(record["pipeline_delivered_kg_by_operator"], dtype=np.float64)
    scale = np.divide(delivered_mass, requested_mass, out=np.zeros(n_ops), where=requested_mass > 0)
    for event in placement_and_rate_events["rate_controls"]:
        op = event["operator"]
        event["actual_rate_m3_s"] = event["requested_rate_m3_s"] * scale[op]
        event["pipeline_dispatch_factor"] = float(scale[op])

    realized_rate_by_well = {
        event["well_id"]: event["actual_rate_m3_s"]
        for event in placement_and_rate_events["rate_controls"]
    }
    old_assets_by_id = {well.well_id: well for well in state.wells}
    stage_injected_kg_per_rate = config.stage_duration_s * solver.cfg.rho_co2_kg_m3
    updated_assets = tuple(
        WellAsset(
            well_id=well.well_id,
            owner=well.owner,
            z_index=well.z_index,
            x_index=well.x_index,
            well_type=well.well_type,
            drilled_stage=well.drilled_stage,
            ready_stage=well.ready_stage,
            last_realized_rate_m3_s=realized_rate_by_well.get(well.well_id, 0.0),
            cumulative_injected_kg=(
                old_assets_by_id.get(well.well_id, well).cumulative_injected_kg
                + realized_rate_by_well.get(well.well_id, 0.0) * stage_injected_kg_per_rate
            ),
        )
        for well in all_assets
    )

    record["well_placements"] = placement_and_rate_events["placements"]
    record["well_rate_controls"] = placement_and_rate_events["rate_controls"]
    record["well_registry"] = [
        {
            "well_id": well.well_id,
            "owner": well.owner,
            "well_type": well.well_type,
            "cell_zx": [well.z_index, well.x_index],
            "drilled_stage": well.drilled_stage,
            "ready_stage": well.ready_stage,
            "commissioned_for_next_stage": well.ready_stage <= state.stage + 1,
            "last_realized_rate_m3_s": well.last_realized_rate_m3_s,
            "cumulative_injected_kg": well.cumulative_injected_kg,
        }
        for well in updated_assets
    ]
    record["well_placement_model"] = "candidate_grid_sites_v1"
    record["monitoring_model"] = "callback" if monitoring_update is not None else "not_configured"
    return CO2MarkovState(core=next_core, wells=updated_assets), record


class CO2WellPlacementGame:
    """Reusable environment wrapper; planning/equilibrium methods plug in above it."""

    def __init__(
        self,
        *,
        solver: CO2XZSolver,
        config: CO2GameConfig,
        placement_rules: PlacementRules = PlacementRules(),
        monitoring_update: MonitoringUpdate | None = None,
    ):
        self.solver = solver
        self.config = config
        self.placement_rules = placement_rules
        self.monitoring_update = monitoring_update

    def initial_state(
        self,
        shape: tuple[int, int],
        *,
        initial_beliefs: Sequence[Any] | None = None,
    ) -> CO2MarkovState:
        return new_markov_state(
            self.solver, shape, self.config, initial_beliefs=initial_beliefs
        )

    def step(
        self,
        state: CO2MarkovState,
        geology: Mapping[str, np.ndarray],
        decisions: Sequence[OperatorDecision],
        *,
        gravity_sign: int = 1,
        risk_metrics: Mapping[str, float] | None = None,
    ) -> tuple[CO2MarkovState, dict[str, Any]]:
        return advance_markov_stage(
            solver=self.solver,
            config=self.config,
            state=state,
            geology=geology,
            decisions=decisions,
            placement_rules=self.placement_rules,
            gravity_sign=gravity_sign,
            risk_metrics=risk_metrics,
            monitoring_update=self.monitoring_update,
        )

