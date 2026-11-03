"""Stage transition kernel for a CO2 pipeline + reservoir Markov game.

This is the first integration layer, not a Nash/cooperative planner and not a
pipeline-hydraulics simulator. Version 1 represents one shared pipeline by a
per-stage mass capacity and explicit transport cost. It dispatches CO2, passes
the resulting well rates to ``CO2XZSolver``, and records state/payoff components
needed by a later partially observed Markov-game solver.

Units: pipeline capacities and ledgers are kg per stage; well rates are m^3/s;
duration is seconds; pressure is Pa; permeability is mD; porosity and saturation
are fractions. Costs/values are caller-defined model units. The defaults avoid
inventing a calibrated storage value, pressure threshold, or pipeline tariff.

The current CO2 solver has closed boundaries and no production/leakage, so CO2
injected during a game remains in the modeled formation. The terminal storage
credit therefore uses cumulative injected mass. If production, leakage, or
open boundaries are added, implement an operator tracer/component ledger before
continuing to use that attribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from .co2_xz_solver import CO2XZSolver, initial_state as co2_initial_state
except ImportError:  # direct execution from sequential_game/
    from co2_xz_solver import CO2XZSolver, initial_state as co2_initial_state


DISPATCH_RULES = {"proportional", "cea", "cel", "fixed_equal"}


@dataclass(frozen=True)
class PipelineConfig:
    """Single shared-pipe abstraction; capacity is a mass per game stage."""

    capacity_kg_per_stage: float
    source_supply_kg_per_stage: float | None = None
    dispatch_rule: str = "proportional"
    resource_cost_per_kg: float = 0.0
    tariff_per_kg: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.capacity_kg_per_stage) or self.capacity_kg_per_stage < 0:
            raise ValueError("pipeline capacity must be a finite nonnegative kg/stage value")
        if self.source_supply_kg_per_stage is not None and (
                not np.isfinite(self.source_supply_kg_per_stage)
                or self.source_supply_kg_per_stage < 0):
            raise ValueError("source supply must be None or a finite nonnegative kg/stage value")
        if self.dispatch_rule not in DISPATCH_RULES:
            raise ValueError(f"dispatch_rule must be one of {sorted(DISPATCH_RULES)}")
        if (not np.isfinite(self.resource_cost_per_kg) or not np.isfinite(self.tariff_per_kg)
                or self.resource_cost_per_kg < 0 or self.tariff_per_kg < 0):
            raise ValueError("pipeline resource cost and tariff must be nonnegative")

    @property
    def available_mass_kg_per_stage(self) -> float:
        if self.source_supply_kg_per_stage is None:
            return float(self.capacity_kg_per_stage)
        return float(min(self.capacity_kg_per_stage, self.source_supply_kg_per_stage))


@dataclass(frozen=True)
class CO2PayoffConfig:
    """Explicit scenario coefficients; not calibrated economic estimates."""

    storage_value_per_kg: float = 0.0
    drilling_cost_per_well: float = 0.0
    pressure_cap_pa: float | None = None
    pressure_damage_per_pa2: float = 0.0
    plume_overlap_damage: float = 0.0
    boundary_contact_damage: float = 0.0

    def __post_init__(self):
        values = (self.storage_value_per_kg, self.drilling_cost_per_well,
                  self.pressure_damage_per_pa2, self.plume_overlap_damage,
                  self.boundary_contact_damage)
        if any(not np.isfinite(v) or v < 0 for v in values):
            raise ValueError("payoff coefficients must be finite and nonnegative")
        if self.pressure_damage_per_pa2 > 0 and (
                self.pressure_cap_pa is None or not np.isfinite(self.pressure_cap_pa)):
            raise ValueError("a finite pressure_cap_pa is required when pressure damage is active")
        if self.pressure_cap_pa is not None and self.pressure_cap_pa < 0:
            raise ValueError("pressure_cap_pa must be nonnegative")


@dataclass(frozen=True)
class CO2GameConfig:
    n_operators: int
    horizon: int = 6
    stage_duration_s: float = 30.0 * 86400.0
    discount_factor: float = 1.0 / 1.05
    storage_cap_kg: float | None = None
    storage_dispatch_rule: str = "proportional"
    pipeline: PipelineConfig | None = None
    payoff: CO2PayoffConfig = field(default_factory=CO2PayoffConfig)

    def __post_init__(self):
        if self.n_operators < 2:
            raise ValueError("the shared-resource game requires at least two operators")
        if self.horizon < 1 or self.stage_duration_s <= 0:
            raise ValueError("horizon and stage duration must be positive")
        if not np.isfinite(self.discount_factor) or not 0 < self.discount_factor <= 1:
            raise ValueError("discount_factor must lie in (0,1]")
        if self.storage_cap_kg is not None and (
                not np.isfinite(self.storage_cap_kg) or self.storage_cap_kg < 0):
            raise ValueError("storage_cap_kg must be None or finite and nonnegative")
        if self.storage_dispatch_rule not in DISPATCH_RULES:
            raise ValueError(f"storage_dispatch_rule must be one of {sorted(DISPATCH_RULES)}")
        if self.pipeline is None:
            raise ValueError("a PipelineConfig is required; use a large explicit capacity for an uncapped scenario")


@dataclass(frozen=True)
class OperatorAction:
    """One operator's requested well controls for one stage.

    ``wells`` contains ``(z_index, x_index, requested_rate_m3_s)`` for every
    active injection well. A single dispatch scaling factor is applied to that
    operator's rates. ``new_wells`` lists cells drilled during this stage. A
    newly drilled well may remain shut in during its commissioning stage; it
    does not need a positive rate in ``wells``. A higher-level placement
    adapter can use stable well IDs and translate them into grid coordinates.
    """

    wells: tuple[tuple[int, int, float], ...] = ()
    new_wells: tuple[tuple[int, int], ...] = ()


@dataclass
class CO2GameState:
    """Augmented state at the start of a stage."""

    stage: int
    reservoir: dict[str, Any]
    cumulative_injected_kg_by_operator: np.ndarray
    cumulative_pipeline_delivered_kg_by_operator: np.ndarray
    active_wells_by_operator: tuple[frozenset[tuple[int, int]], ...] = ()
    cumulative_welfare: float = 0.0
    cumulative_payoff_by_operator: np.ndarray | None = None
    belief_state: tuple[Any, ...] = ()


def new_game_state(solver: CO2XZSolver, shape: tuple[int, int], n_operators: int,
                   *, belief_state: Sequence[Any] = ()) -> CO2GameState:
    """Create a zero-CO2 initial state compatible with the x-z solver."""
    if n_operators < 2:
        raise ValueError("n_operators must be at least two")
    reservoir = co2_initial_state(shape)
    return CO2GameState(
        stage=0,
        reservoir=reservoir,
        cumulative_injected_kg_by_operator=np.zeros(n_operators, dtype=np.float64),
        cumulative_pipeline_delivered_kg_by_operator=np.zeros(n_operators, dtype=np.float64),
        active_wells_by_operator=tuple(frozenset() for _ in range(n_operators)),
        cumulative_payoff_by_operator=np.zeros(n_operators, dtype=np.float64),
        belief_state=tuple(belief_state),
    )


def dispatch_mass(requested_kg: Sequence[float], capacity_kg: float,
                  rule: str = "proportional") -> np.ndarray:
    """Allocate a shared stage capacity under an explicit rationing rule.

    ``fixed_equal`` gives each operator an equal fixed share; unused shares are
    not redistributed. CEA equalizes awards up to each request. CEL equalizes
    request reductions, floored at zero.
    """
    req = np.asarray(requested_kg, dtype=np.float64)
    cap = float(capacity_kg)
    if req.ndim != 1 or req.size < 1 or np.any(~np.isfinite(req)) or np.any(req < 0):
        raise ValueError("requested_kg must be a finite nonnegative vector")
    if cap < 0 or np.isnan(cap):
        raise ValueError("capacity_kg must be nonnegative")
    if rule not in DISPATCH_RULES:
        raise ValueError(f"rule must be one of {sorted(DISPATCH_RULES)}")
    if req.sum() <= cap + 1e-12:
        return req.copy()
    if cap == 0:
        return np.zeros_like(req)
    if rule == "proportional":
        return req * (cap / req.sum())
    if rule == "fixed_equal":
        return np.minimum(req, cap / req.size)
    if rule == "cea":
        lo, hi = 0.0, float(req.max())
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if np.minimum(req, mid).sum() < cap:
                lo = mid
            else:
                hi = mid
        return np.minimum(req, 0.5 * (lo + hi))
    # CEL: choose a common loss that leaves exactly the available capacity.
    lo, hi = 0.0, float(req.max())
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.maximum(req - mid, 0.0).sum() > cap:
            lo = mid
        else:
            hi = mid
    return np.maximum(req - 0.5 * (lo + hi), 0.0)


def _validate_actions(actions: Sequence[OperatorAction], n_operators: int,
                      shape: tuple[int, int]) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    if len(actions) != n_operators:
        raise ValueError("one OperatorAction is required for each operator")
    requested_rate = np.zeros(n_operators, dtype=np.float64)
    combined_wells: list[tuple[int, int, float]] = []
    used_cells: set[tuple[int, int]] = set()
    for op, action in enumerate(actions):
        own_cells = set()
        for z, x, rate in action.wells:
            cell = (int(z), int(x))
            if not (0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]):
                raise ValueError(f"operator {op} requested a well outside the reservoir grid: {cell}")
            if cell in used_cells:
                raise ValueError(f"two operators cannot inject from the same cell: {cell}")
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError("requested well rates must be finite and positive")
            used_cells.add(cell)
            own_cells.add(cell)
            requested_rate[op] += float(rate)
            combined_wells.append((cell[0], cell[1], float(rate)))
        new_cells = [tuple(map(int, cell)) for cell in action.new_wells]
        if len(set(new_cells)) != len(new_cells):
            raise ValueError("new_wells must contain unique cells")
        for cell in new_cells:
            if not (0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]):
                raise ValueError(f"operator {op} placed a well outside the reservoir grid: {cell}")
    return requested_rate, combined_wells


def advance_stage(
    *,
    solver: CO2XZSolver,
    config: CO2GameConfig,
    state: CO2GameState,
    geology: Mapping[str, np.ndarray],
    actions: Sequence[OperatorAction],
    gravity_sign: int = 1,
    risk_metrics: Mapping[str, float] | None = None,
    belief_update: Callable[[int, Mapping[str, Any], Mapping[str, np.ndarray],
                             Sequence[OperatorAction], tuple[Any, ...]],
                            tuple[Sequence[Any], Sequence[Any]]] | None = None,
) -> tuple[CO2GameState, dict[str, Any]]:
    """Apply one joint action and return ``(next_state, stage_record)``.

    The transition is deterministic conditional on geology, action, and the
    solver. ``belief_update`` is an optional observation hook returning
    ``(observations_by_operator, next_beliefs_by_operator)``. The hook is where
    the later Bayesian Markov-game layer will add noisy/public/private
    monitoring. No belief update is fabricated if that model is absent.

    ``geology`` must contain matching arrays ``kx_mD``, ``kz_mD``, and ``phi``.
    ``risk_metrics`` may provide validated stage values ``plume_overlap`` and
    ``boundary_contact_fraction`` in [0,1]. Nonzero payoff weights require
    these diagnostics; the current solver does not compute operator-attributed
    plume overlap or containment risk by itself.
    """
    n_ops = config.n_operators
    if state.stage >= config.horizon:
        raise ValueError("game is already at or beyond its terminal stage")
    if state.cumulative_injected_kg_by_operator.shape != (n_ops,):
        raise ValueError("state operator ledger does not match game configuration")
    if state.cumulative_pipeline_delivered_kg_by_operator.shape != (n_ops,):
        raise ValueError("pipeline ledger does not match game configuration")
    if state.cumulative_payoff_by_operator is None:
        prior_payoffs = np.zeros(n_ops, dtype=np.float64)
    else:
        prior_payoffs = np.asarray(state.cumulative_payoff_by_operator, dtype=np.float64)
        if prior_payoffs.shape != (n_ops,):
            raise ValueError("payoff ledger does not match game configuration")

    kx = np.asarray(geology["kx_mD"], dtype=np.float64)
    kz = np.asarray(geology["kz_mD"], dtype=np.float64)
    phi = np.asarray(geology["phi"], dtype=np.float64)
    if kx.ndim != 2 or kz.shape != kx.shape or phi.shape != kx.shape:
        raise ValueError("geology arrays kx_mD, kz_mD, phi must share a [z,x] shape")
    requested_rate, _ = _validate_actions(actions, n_ops, kx.shape)
    prior_wells = state.active_wells_by_operator or tuple(frozenset() for _ in range(n_ops))
    if len(prior_wells) != n_ops:
        raise ValueError("active well ledger does not match game configuration")
    next_wells_by_operator = []
    globally_owned = set()
    for op, (action, old_cells) in enumerate(zip(actions, prior_wells)):
        old_cells = frozenset(tuple(map(int, c)) for c in old_cells)
        new_cells = frozenset(tuple(map(int, c)) for c in action.new_wells)
        if new_cells.intersection(old_cells):
            raise ValueError(f"operator {op} attempted to drill an already active well")
        controlled = old_cells | new_cells
        action_cells = {(int(z), int(x)) for z, x, _ in action.wells}
        if not action_cells.issubset(controlled):
            raise ValueError(f"operator {op} requested injection from an unowned, undrilled well")
        if globally_owned.intersection(controlled):
            raise ValueError("a well cell cannot be controlled by two operators")
        globally_owned.update(controlled)
        next_wells_by_operator.append(controlled)
    requested_mass = (requested_rate * config.stage_duration_s
                      * solver.cfg.rho_co2_kg_m3)

    pipe = config.pipeline
    pipeline_capacity_allocation = dispatch_mass(
        requested_mass, pipe.available_mass_kg_per_stage, pipe.dispatch_rule)
    if config.storage_cap_kg is None:
        delivered_mass = pipeline_capacity_allocation.copy()
    else:
        remaining_storage = max(
            0.0, config.storage_cap_kg - float(state.cumulative_injected_kg_by_operator.sum()))
        if config.storage_dispatch_rule == "fixed_equal":
            # A fixed storage quota is cumulative across stages, unlike the
            # pipeline's per-stage fixed share.
            quota_left = np.maximum(
                config.storage_cap_kg / n_ops - state.cumulative_injected_kg_by_operator,
                0.0)
            delivered_mass = np.minimum(pipeline_capacity_allocation, quota_left)
        else:
            delivered_mass = dispatch_mass(
                pipeline_capacity_allocation, remaining_storage, config.storage_dispatch_rule)

    # The stage-capacity v1 model has no linepack or surface buffer. Therefore
    # only mass accepted by both constraints is physically sent to the wells.
    scale = np.divide(delivered_mass, requested_mass,
                      out=np.zeros_like(delivered_mass), where=requested_mass > 0)
    solver_wells: list[tuple[int, int, float]] = []
    for op, action in enumerate(actions):
        for z, x, rate in action.wells:
            actual_rate = float(rate) * scale[op]
            if actual_rate > 0:
                solver_wells.append((int(z), int(x), actual_rate))

    reservoir_next, diagnostics, _ = solver.advance(
        kx, kz, phi, state.reservoir, solver_wells,
        config.stage_duration_s, gravity_sign=gravity_sign)
    actual_injected = float(diagnostics["injected_co2_kg_stage"])
    planned_injected = float(delivered_mass.sum())
    if not np.isclose(actual_injected, planned_injected, rtol=2e-7, atol=1e-7):
        raise RuntimeError(
            "pipeline/storage dispatch and reservoir injection disagree: "
            f"dispatched={planned_injected:.9g} kg, solver={actual_injected:.9g} kg")

    injected_by_op = delivered_mass.copy()
    pipeline_cost_by_op = delivered_mass * pipe.resource_cost_per_kg
    pipeline_tariff_by_op = delivered_mass * pipe.tariff_per_kg
    new_well_counts = np.asarray([len(a.new_wells) for a in actions], dtype=np.float64)
    drilling_cost_by_op = new_well_counts * config.payoff.drilling_cost_per_well

    pmax = float(diagnostics["pmax_pa"])
    pressure_cost = 0.0
    if config.payoff.pressure_damage_per_pa2 > 0:
        exceed = max(0.0, pmax - float(config.payoff.pressure_cap_pa))
        pressure_cost = config.payoff.pressure_damage_per_pa2 * exceed ** 2

    risks = dict(risk_metrics or {})
    for key, coefficient in (("plume_overlap", config.payoff.plume_overlap_damage),
                             ("boundary_contact_fraction", config.payoff.boundary_contact_damage)):
        if coefficient > 0 and key not in risks:
            raise ValueError(f"risk_metrics[{key!r}] is required when its payoff weight is nonzero")
    overlap = float(risks.get("plume_overlap", 0.0))
    boundary = float(risks.get("boundary_contact_fraction", 0.0))
    if not (0.0 <= overlap <= 1.0 and 0.0 <= boundary <= 1.0):
        raise ValueError("plume and boundary risk metrics must lie in [0,1]")
    plume_cost = config.payoff.plume_overlap_damage * overlap
    boundary_cost = config.payoff.boundary_contact_damage * boundary
    shared_damage = pressure_cost + plume_cost + boundary_cost
    liability_share = (injected_by_op / injected_by_op.sum()
                       if injected_by_op.sum() > 0 else np.zeros(n_ops))
    allocated_damage = shared_damage * liability_share

    terminal = state.stage + 1 == config.horizon
    storage_credit_by_op = np.zeros(n_ops, dtype=np.float64)
    if terminal and config.payoff.storage_value_per_kg > 0:
        storage_credit_by_op = (
            state.cumulative_injected_kg_by_operator + injected_by_op
        ) * config.payoff.storage_value_per_kg

    operator_payoff = (storage_credit_by_op - drilling_cost_by_op - pipeline_cost_by_op
                       - allocated_damage - pipeline_tariff_by_op)
    resource_welfare = float(
        storage_credit_by_op.sum() - drilling_cost_by_op.sum()
        - pipeline_cost_by_op.sum() - shared_damage)
    discount_applied = config.discount_factor ** state.stage
    discounted_operator_payoff = operator_payoff * discount_applied
    discounted_resource_welfare = resource_welfare * discount_applied

    next_injected = state.cumulative_injected_kg_by_operator + injected_by_op
    next_delivered = state.cumulative_pipeline_delivered_kg_by_operator + delivered_mass
    observations: Sequence[Any] = ()
    next_beliefs = state.belief_state
    if belief_update is not None:
        observations, next_beliefs = belief_update(
            state.stage, reservoir_next, geology, actions, state.belief_state)
        if len(next_beliefs) != n_ops:
            raise ValueError("belief_update must return one next belief per operator")
        if len(observations) != n_ops:
            raise ValueError("belief_update must return one observation record per operator")

    next_state = CO2GameState(
        stage=state.stage + 1,
        reservoir=reservoir_next,
        cumulative_injected_kg_by_operator=next_injected,
        cumulative_pipeline_delivered_kg_by_operator=next_delivered,
        active_wells_by_operator=tuple(next_wells_by_operator),
        cumulative_welfare=state.cumulative_welfare + discounted_resource_welfare,
        cumulative_payoff_by_operator=prior_payoffs + discounted_operator_payoff,
        belief_state=tuple(next_beliefs),
    )
    record = {
        "stage": state.stage,
        "terminal": terminal,
        "requested_mass_kg_by_operator": requested_mass.tolist(),
        "pipeline_capacity_allocation_kg_by_operator": pipeline_capacity_allocation.tolist(),
        "pipeline_delivered_kg_by_operator": delivered_mass.tolist(),
        "storage_accepted_kg_by_operator": injected_by_op.tolist(),
        "pipeline_curtailed_kg_by_operator": (requested_mass - pipeline_capacity_allocation).tolist(),
        "reservoir_curtailed_kg_by_operator": (pipeline_capacity_allocation - delivered_mass).tolist(),
        "pipeline_resource_cost_by_operator": pipeline_cost_by_op.tolist(),
        "pipeline_tariff_transfer_by_operator": pipeline_tariff_by_op.tolist(),
        "new_wells_by_operator": new_well_counts.astype(int).tolist(),
        "shared_damage_cost": float(shared_damage),
        "damage_liability_by_operator": allocated_damage.tolist(),
        "drilling_cost_by_operator": drilling_cost_by_op.tolist(),
        "pressure_max_pa": pmax,
        "pressure_damage_cost": pressure_cost,
        "plume_overlap": risks.get("plume_overlap"),
        "boundary_contact_fraction": risks.get("boundary_contact_fraction"),
        "plume_damage_cost": plume_cost,
        "boundary_damage_cost": boundary_cost,
        "operator_payoff_by_operator": operator_payoff.tolist(),
        "discount_factor_applied": discount_applied,
        "discounted_operator_payoff_by_operator": discounted_operator_payoff.tolist(),
        "storage_credit_by_operator": storage_credit_by_op.tolist(),
        "project_resource_welfare": resource_welfare,
        "discounted_project_resource_welfare": discounted_resource_welfare,
        "solver_diagnostics": diagnostics,
        "observations_by_operator": list(observations),
        "pipeline_model": "stage_capacity_v1",
        "reservoir_model": "co2_xz_solver",
    }
    return next_state, record
