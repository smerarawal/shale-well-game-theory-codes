# CO₂ shared-supply sequential game: quick start

The game environment is split into a pipeline/reservoir transition and a well-placement adapter:

- `co2_pipeline_game.py` applies the common source limit, shared stage-capacity dispatch, reservoir solver, storage cap, and payoff ledger.
- `co2_well_placement_game.py` adds persistent well IDs, operator ownership, placement constraints, drilling delays, and a monitoring-observation callback.
- `co2_xz_solver.py` remains the reservoir reference model.

The pipeline layer is deliberately a single shared bottleneck. Total mass available in a stage is `min(source_supply_kg_per_stage, capacity_kg_per_stage)`. This does not represent pipe routes, pressure, linepack, or compressor dynamics.

## Minimal stage interaction

```python
import numpy as np

from sequential_game.co2_xz_solver import CO2XZSolver, CO2XZConfig
from sequential_game.co2_pipeline_game import CO2GameConfig, PipelineConfig
from sequential_game.co2_well_placement_game import (
    CO2WellPlacementGame,
    OperatorDecision,
    PlacementRules,
    WellPlacement,
    WellRate,
)

solver = CO2XZSolver(CO2XZConfig())
config = CO2GameConfig(
    n_operators=2,
    horizon=6,
    pipeline=PipelineConfig(
        source_supply_kg_per_stage=250_000.0,
        capacity_kg_per_stage=300_000.0,
        dispatch_rule="proportional",
    ),
)
rules = PlacementRules(
    candidate_cells_by_operator=(
        frozenset((z, x) for z in range(2, 10) for x in range(1, 5)),
        frozenset((z, x) for z in range(2, 10) for x in range(7, 11)),
    ),
    min_spacing_ft=80.0,
    max_wells_per_operator=3,
    max_new_wells_per_operator_per_stage=1,
    max_rate_m3_s_per_well=1.0e-4,
    drilling_lead_stages=0,
)
game = CO2WellPlacementGame(solver=solver, config=config, placement_rules=rules)
state = game.initial_state((12, 12))

geology = {
    "kx_mD": np.full((12, 12), 100.0),
    "kz_mD": np.full((12, 12), 50.0),
    "phi": np.full((12, 12), 0.20),
}
joint_decision = (
    OperatorDecision(
        placements=(WellPlacement("op0-inj0", 5, 2),),
        rates=(WellRate("op0-inj0", 1.0e-4),),
    ),
    OperatorDecision(
        placements=(WellPlacement("op1-inj0", 5, 9),),
        rates=(WellRate("op1-inj0", 1.0e-4),),
    ),
)
state, stage_record = game.step(state, geology, joint_decision)
```

Operators act simultaneously: pass one `OperatorDecision` per operator, in the same order as `n_operators`. Each stage record contains nominations, allocations, curtailment, per-well requested/realized rate, placement and commissioning information, reservoir diagnostics, and payoff components. A zero rate shuts in a commissioned injector; omitting a well from the rate list also means it does not inject that stage.

## Monitoring and beliefs

Monitor wells can be placed with `WellPlacement(..., well_type="monitor")`. They incur the configured per-well drilling cost. To make monitoring change later decisions, pass a `monitoring_update` callback to `CO2WellPlacementGame`. The callback receives the stage, post-transition reservoir state, geology, commissioned monitors, joint action, and current per-operator beliefs; it must return one observation and next belief per operator. Without the callback, monitor wells are recorded but produce no information.

## Current scope and next layer

This is a Markov-game **environment**, not yet a Nash solver or a Caers-style belief-state planner. `CO2GameConfig` payoff defaults are zero-valued placeholders. Before a planner can make meaningful welfare-maximizing decisions, choose and justify scenario payoff coefficients or a calibrated economic ledger. Then add policy modules that consume each operator's information state and call `game.step`; keep equilibrium search separate from physical transition code. The current CO₂ solver also represents point injectors on a 2D x-z grid and does not model well trajectories, brine production, pipeline network flow, or leakage. Monitoring behavior must be supplied as a caller-defined observation callback.
