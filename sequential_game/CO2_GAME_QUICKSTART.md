# CO₂ shared-supply sequential game: quick start

The game environment is split into a pipeline/reservoir transition and a well-placement adapter:

- `co2_pipeline_game.py` applies the common source limit, shared stage-capacity dispatch, reservoir solver, storage cap, and payoff ledger.
- `co2_well_placement_game.py` adds persistent well IDs, operator ownership, placement constraints, drilling delays, and a monitoring-observation callback.
- `co2_xz_solver.py` remains the reservoir reference model.

The pipeline layer is deliberately a single shared bottleneck. Total mass available in a stage is `min(source_supply_kg_per_stage, capacity_kg_per_stage)`. This does not represent pipe routes, pressure, linepack, or compressor dynamics.

## Kaggle setup

For this transition-environment pilot, use **CPU**; the reference reservoir solver uses NumPy/SciPy and does not use CUDA. No CO₂ dataset or checkpoint is needed for the synthetic plumbing example below. Turn Kaggle Internet on if you clone from GitHub.

Run this as the first notebook cell:

```python
from pathlib import Path
import subprocess
import sys

REPO = Path("/kaggle/working/shale-well-game-theory-codes")
if not REPO.exists():
    subprocess.run([
        "git", "clone", "--depth", "1",
        "--branch", "codex/ccs-lstm-fno-variants",
        "https://github.com/smerarawal/shale-well-game-theory-codes.git",
        str(REPO),
    ], check=True)
sys.path.insert(0, str(REPO))
```

Then run the “Minimal stage interaction” code in a second cell. It advances **one 30-day stage** and returns `state` plus `stage_record`. To continue the game, call `game.step(...)` again with the returned state and the next simultaneous decisions. For an offline Kaggle session, add a ZIP of the feature branch as a notebook input and extract it under `/kaggle/working` instead of cloning.

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
        source_supply_kg_per_stage=1_200.0,
        capacity_kg_per_stage=1_000.0,
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
    max_rate_m3_s_per_well=1.0e-6,
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
        rates=(WellRate("op0-inj0", 1.0e-6),),
    ),
    OperatorDecision(
        placements=(WellPlacement("op1-inj0", 5, 9),),
        rates=(WellRate("op1-inj0", 1.0e-6),),
    ),
)
state, stage_record = game.step(state, geology, joint_decision)
```

Operators act simultaneously: pass one `OperatorDecision` per operator, in the same order as `n_operators`. Each stage record contains nominations, allocations, curtailment, per-well requested/realized rate, placement and commissioning information, reservoir diagnostics, and payoff components. A zero rate shuts in a commissioned injector; omitting a well from the rate list also means it does not inject that stage.

In this tiny plumbing example, the two requests total about 3,370 kg for the 30-day stage, but the shared pipe can carry only 1,000 kg. Proportional dispatch should therefore deliver about 500 kg per operator. These numbers are chosen to demonstrate the code path safely; they are not calibrated engineering or economic values.

## Monitoring and beliefs

Monitor wells can be placed with `WellPlacement(..., well_type="monitor")`. They incur the configured per-well drilling cost. To make monitoring change later decisions, pass a `monitoring_update` callback to `CO2WellPlacementGame`. The callback receives the stage, post-transition reservoir state, geology, commissioned monitors, joint action, and current per-operator beliefs; it must return one observation and next belief per operator. Without the callback, monitor wells are recorded but produce no information.

## Current scope and next layer

This is a Markov-game **environment**, not yet a Nash solver or a Caers-style belief-state planner. `CO2GameConfig` payoff defaults are zero-valued placeholders. Before a planner can make meaningful welfare-maximizing decisions, choose and justify scenario payoff coefficients or a calibrated economic ledger. Then add policy modules that consume each operator's information state and call `game.step`; keep equilibrium search separate from physical transition code. The current CO₂ solver also represents point injectors on a 2D x-z grid and does not model well trajectories, brine production, pipeline network flow, or leakage. Monitoring behavior must be supplied as a caller-defined observation callback.

## Experiment suite

For the new CO₂ work, use [`CO2_RESEARCH_EXECUTION.md`](CO2_RESEARCH_EXECUTION.md)
for the paired geology/instrument regime map, solver-derived interference and
price analysis, surrogate/solver planner comparison, and exact Kaggle cells
for the five institution ideas: correlated equilibrium, capacity auction and
price-vs-quantity, financial assurance, horizon/Folk incentives, and
equilibrium/welfare-gap diagnostics. Read [`CO2_STUDY_PLAN.md`](CO2_STUDY_PLAN.md)
for the physics gates, study design, and 3D roadmap. The initial institution
experiments use one-stage fixed-well payoff tables as a diagnostic; they are
not dynamic Markov-perfect equilibrium results.
