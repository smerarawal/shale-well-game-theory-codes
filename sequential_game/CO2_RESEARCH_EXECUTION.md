# CO₂ research workflow (ensemble excluded)

This execution layer covers the CO2 institution experiments, geology regime
study, solver-derived pressure interference/Pigouvian price, a differentiable
surrogate best-response planner, equilibrium-aware solver relabeling, and
policy-tree distillation. It does **not** implement an ensemble. All flow and game rollouts here use the
current experimental 2D x-z solver. These are research tools, not field
forecasts or a calibrated pipeline simulator.

## What was added

- `co2_research_studies.py`: directional geology descriptors (high-K component
  connectivity/spanning, x/z correlation lengths, K anisotropy); descriptor to
  game-outcome joins; centered finite-difference interference matrix and local
  pressure-damage price; comparison against an empirical price sweep; and a
  shallow decision-tree policy with held-out solver-table regret.
- `co2_solver_game_search.py`: solver-backed schedule evaluation, a coarse
  iterated best-response benchmark, and a coordinated welfare search. These
  are approximate finite searches and do not certify equilibrium/global
  optimality.
- `run_co2_regime_map.py`: paired geology/instrument sweep. For each selected
  training geology it runs both Nash-like and coordinated searches with matched
  configuration, then records the Coop−Nash gap, operator payoffs, descriptors,
  and solver-call counts. It tests proportional, CEA, CEL, fixed-equal, and a
  tariff/price sweep.
- `build_co2_stage_game.py` and `co2_institution_experiments.py`: solver-derived
  finite action tables; pure-Nash and unilateral-regret diagnostics; a
  welfare-maximizing correlated-equilibrium LP; capacity-auction allocation
  and payment comparisons; financial-assurance sensitivity; horizon and
  grim-trigger incentive calculations; and a labeled Weitzman price-vs-cap
  toy model. Exact scope and Kaggle cells appear below.
- `co2_differentiable_game.py`: alternating open-loop best responses through a
  trained single surrogate; it emits a six-stage rate schedule. This is a
  differentiable planner proposal, not a solver result.
- `collect_co2_equilibrium_data.py`: replays planned schedules with the real
  solver and appends the resulting transitions to a fresh copy of the training
  split only. Retrain using that copied dataset, then repeat planning and
  labeling for the next active-learning round.
- The CO2 trainer now supports pressure-tail and pressure-cap-sensitive losses,
  pressure-aware checkpoint selection, and validation-rollout checkpoint
  selection. Metrics include pressure-cap violation classification and, when
  an expert-reviewed cap and damage coefficient are supplied, pressure-penalty
  error in the same payoff units as the game.

## Game-readiness refinement in the same run

The existing three-seed CCS/CCS-GNN numbers are a starting benchmark, not
evidence that either model is ready to drive a game. The CCS-GNN has not yet
shown a consistent win: its current pressure and plume metrics trade off, and
long rollouts drift. Run a matched refinement with CCS retained as the control.

1. **Reference arm:** train `ccs` and `ccs_gnn` on the existing training split,
   seeds 0, 1, 2, with the prior one-step objective. Keep outputs in a new
   directory so `--resume` cannot silently reuse unrelated checkpoints.
2. **Refined arm:** train the same two backbones and seeds with six-stage
   pushforward windows, modest validity and mass-balance losses, pressure-tail
   loss, and validation-rollout checkpoint selection. Add cap-penalty loss only
   after the pressure cap and damage coefficient have been reviewed in Pa and
   payoff units. Do not copy the old brine cap into this CO2 case.
3. **Equilibrium-aware round:** run the differentiable planner on a few
   *training* geologies from each regime, then replay those schedules through
   the solver. Append labels to train only and retrain both backbones from
   scratch on the augmented set. Keep validation and test untouched.
4. **Decision validation:** for held-out validation/test geologies, send the
   same surrogate-planned schedule to the solver and initialize the solver's
   coarse best-response and cooperative searches from that exact schedule
   (`co2_solver_game_search.py ... --initial-schedule PLAN.json`). Report the
   initial-to-response payoff changes as approximate unilateral deviation
   gains, and the planner-vs-cooperative welfare gap as welfare regret, along with stagewise
   pressure-cap classification/penalty error, plume IoU, mass residual, and
   elapsed solver versus surrogate-planner time. These discrete searches are
   approximate references, not equilibrium certificates.
5. **Keep CCS-GNN only if it improves the game-level outcomes across seeds and
   held-out geology regimes without unacceptable pressure violations or mass
   residuals.** Compare against CCS and the existing model artifacts; do not
   choose from field L2 or plume IoU alone. If gains vary across seeds/geologies,
   report that uncertainty rather than naming a single universal winner.

The refined trainer options are:

```bash
python train_co2_xz_surrogate.py --data-dir DATA --out-dir OUT/refined \
  --backbones ccs ccs_gnn --seeds 0 1 2 --epochs 80 --patience 12 \
  --rollout-unroll 6 --rollout-validation-weight 0.25 \
  --rollout-validation-every 5 --valid-loss-weight 0.1 \
  --mass-loss-weight 0.1 --pmax-loss-weight 0.5 \
  --pressure-tail-loss-weight 0.25 --pressure-tail-fraction 0.1 \
  --device cuda
```

Those weights are an initial experimental setting, not calibrated constants.
Use a separate output folder for the matched reference arm. If the expert has
approved a pressure cap and damage coefficient, add
`--pressure-cap-pa <cap>` and `--pressure-damage-per-pa2 <coefficient>` to get
penalty-scale error metrics, and optionally add
`--pressure-penalty-loss-weight 0.25` to train against that game quantity.
Report both one-step and autoregressive six-stage test metrics for each cohort
and seed. The training script itself does not prove a good equilibrium policy;
the solver game comparison is the decision-level test.

## Scenario file

Create a JSON file for the case with values reviewed by your reservoir/game
expert. At minimum it must contain:

```json
{
  "horizon": 6,
  "stage_duration_s": 2592000,
  "discount_factor": 0.9523809524,
  "pipeline": {
    "capacity_kg_per_stage": 1000.0,
    "dispatch_rule": "proportional",
    "resource_cost_per_kg": 0.0,
    "tariff_per_kg": 0.0
  },
  "payoff": {
    "storage_value_per_kg": 1.0,
    "drilling_cost_per_well": 0.0,
    "pressure_cap_pa": 2000000.0,
    "pressure_damage_per_pa2": 1e-10,
    "plume_overlap_damage": 0.0,
    "boundary_contact_damage": 0.0
  },
  "wells_by_operator": [[22, 8], [22, 16]],
  "solver": {"max_dt_s": 86400.0}
}
```

The numbers above are **syntactic examples only**, not recommended physical or
economic values. Replace capacity, pressure limit/penalty, storage credit,
injection rates, well cells, and solver grid dimensions with reviewed values.
Use a separate scenario file for each calibrated case. A pressure cap is in Pa;
the old brine-game cap of 6 must not be carried over as 6 Pa or 6 MPa by
assumption. Pipeline capacity is kg/stage; the current code still has one
shared scalar bottleneck, not a real network topology.

`co2_stage_game_template.json` is a small runnable-shape template. Its values
are placeholders. It sets pressure damage and plume damage to zero, so the
financial-assurance sweep will be mechanically present but will not change
incentives until you set a reviewed, nonzero damage model. Likewise, its well
cells assume a 24×24 section. Review all grid coordinates and coefficients
before running scientific comparisons.

## The five requested institution and equilibrium experiments: Kaggle cells

These cells use the new CO2 repository branch after the repo, CO2 dataset, and
a reviewed scenario JSON are attached/mounted. They run on CPU because each
payoff profile calls the SciPy reservoir solver. Keep the GPU for surrogate
training. Start with 3 actions per operator (9 solver profiles for two
operators); 5 actions means 25 profiles. These are one-stage fixed-well
institution/equilibrium pilots, not six-stage Markov-game results.

First set paths and make a **scarcity** scenario for CE/Nash/cooperative/bond
comparisons. Use a geology NPZ with `kx_mD`, `kz_mD`, and `phi`, each a single
2D field or a batch with its first dimension as case. The template's well
coordinates fit 24×24 only.

```python
# Institution experiment cell 1: paths and scenarios.
from pathlib import Path
import json, subprocess, sys

REPO = Path("/kaggle/working/shale-well-game-theory-codes")
DATA = Path("/kaggle/input/<owner>/<co2-dataset>/co2_xz_dataset_v2")
GEOLOGY = DATA / "validation" / "friendly.npz"
SCENARIO_TEMPLATE = REPO / "sequential_game" / "co2_stage_game_template.json"
OUT = Path("/kaggle/working/co2_institution_pilot")
OUT.mkdir(parents=True, exist_ok=True)
assert (REPO / "sequential_game" / "build_co2_stage_game.py").exists(), REPO
assert GEOLOGY.exists(), GEOLOGY

scenario = json.loads(SCENARIO_TEMPLATE.read_text())
# REPLACE these with expert-reviewed CO2 case values before interpreting.
scenario["pipeline"]["capacity_kg_per_stage"] = 150.0
SCARCITY = OUT / "scenario_scarcity.json"
SCARCITY.write_text(json.dumps(scenario, indent=2))

# Separate uncapped case for private marginal values in the capacity auction.
uncapped = json.loads(json.dumps(scenario))
uncapped["pipeline"]["capacity_kg_per_stage"] = 1.0e12
uncapped["pipeline"]["source_supply_kg_per_stage"] = None
uncapped["storage_cap_kg"] = None
UNCAPPED = OUT / "scenario_uncapped.json"
UNCAPPED.write_text(json.dumps(uncapped, indent=2))
```

Build two payoff tables using the same geology row, rate choices, well
locations, and solver settings. For a pilot, select one row from the
validation split; a confirmatory run should pair multiple held-out geology
seeds.

```python
# Institution experiment cell 2: build scarcity and uncapped solver tables.
BUILD = REPO / "sequential_game" / "build_co2_stage_game.py"
RATE_MAX_M3_S = 1e-7  # illustrative smoke-test rate; replace with reviewed value
for tag, scenario_path in (("scarcity", SCARCITY), ("uncapped", UNCAPPED)):
    subprocess.run([
        sys.executable, str(BUILD), "--geology", str(GEOLOGY), "--row", "0",
        "--config", str(scenario_path), "--rate-max-m3-s", str(RATE_MAX_M3_S),
        "--rate-levels", "3", "--out", str(OUT / f"{tag}_game.npz"),
    ], check=True)
```

Run the equilibrium audit, correlated equilibrium, welfare-optimal profile,
bond sensitivity, and horizon/trigger calculations. The horizon comparison
holds the reservoir state and stage payoff fixed; it is not a six-stage
dynamic simulation.

```python
# Institution experiment cell 3: Experiments 1, 3, 4, and 5.
ANALYZE = REPO / "sequential_game" / "co2_institution_experiments.py"
subprocess.run([
    sys.executable, str(ANALYZE), "analyze-game",
    "--game", str(OUT / "scarcity_game.npz"),
    "--out", str(OUT / "equilibrium_ce_bond_horizon.json"),
    "--horizons", "1", "3", "6", "12", "30",
    "--discount-factor", "0.9523809524",
], check=True)
```

Run the capacity auction and separate Weitzman price-vs-quantity toy. The
auction derives private marginal values from the uncapped solver table, then
reports VCG and pay-as-bid allocations at several bid-shading factors. The
Weitzman assumptions below must be reviewed; they are not estimated reservoir
values. These calculations do not yet include strategic dynamic reservoir
externalities inside the auction mechanism.

```python
# Institution experiment cell 4: Experiment 2.
subprocess.run([
    sys.executable, str(ANALYZE), "auction",
    "--game", str(OUT / "uncapped_game.npz"),
    "--capacity-blocks", "2", "--shading", "0.5", "0.75", "1.0",
    "--out", str(OUT / "capacity_auction.json"),
], check=True)

WEITZMAN_SPEC = {
    "mb_intercept_mean": 10.0, "md_intercept_mean": 2.0,
    "mb_slope": 1.0, "md_slope": 0.5,
    "mb_intercept_sd": 2.0, "md_intercept_sd": 1.0,
    "intercept_correlation": 0.0, "draws": 20000, "seed": 0
}
SPEC = OUT / "weitzman_assumptions.json"
SPEC.write_text(json.dumps(WEITZMAN_SPEC, indent=2))
subprocess.run([
    sys.executable, str(ANALYZE), "weitzman", "--spec", str(SPEC),
    "--out", str(OUT / "weitzman_price_quantity.json"),
], check=True)
```

Interpret these together: Nash/deviation gains check unilateral stability on
the declared rate grid; CE asks whether private recommendations can improve
coordination; the auction compares capacity allocation/payment rules under
private-value assumptions; bonds change incentives only when modeled damages
and capital costs are nonzero; horizon accounting is distinct from an
indefinite-horizon trigger condition. Separately, the regime map varies
geology and policy, while surrogate studies compare planner proposals with
solver rollouts. None alone validates a full Markov game.

## Kaggle workflow

Use the CO₂ dataset notebook output as an input dataset. The GPU is useful for
the differentiable planner and model retraining. The reservoir solver, geology
descriptors, finite-difference matrix, discrete game searches, and solver
labeling run on CPU. Keep output directories unique between active-learning
rounds.

```python
# Cell 1: set paths and ensure the repository is the current branch.
from pathlib import Path
import subprocess

REPO = Path("/kaggle/working/surrogate_repo")
if not REPO.exists():
    subprocess.run([
        "git", "clone", "--branch", "codex/ccs-lstm-fno-variants",
        "--single-branch",
        "https://github.com/smerarawal/shale-well-game-theory-codes.git",
        str(REPO),
    ], check=True)
else:
    subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], check=True)

DATA = Path("/kaggle/input/notebooks/<owner>/<co2-dataset-notebook>/co2_xz_dataset_v2")
OUT = Path("/kaggle/working/co2_research")
assert (DATA / "train" / "friendly.npz").exists(), DATA
```

Replace the `<owner>/<...>` path with Kaggle's actual mounted input path. Upload
the scenario JSON as a Kaggle dataset input, then set `SCENARIO` to its path.

```python
# Cell 2: compute the field-only regime descriptors from the raw geology archive.
import subprocess, sys
GEOLOGY_ARCHIVE = Path("/kaggle/input/<owner>/<geology-dataset>/co2_geology_cohorts.npz")
DESCRIPTORS = OUT / "geology_descriptors.csv"
subprocess.run([
    sys.executable, str(REPO / "sequential_game" / "co2_research_studies.py"),
    "describe-geologies", "--archive", str(GEOLOGY_ARCHIVE), "--out", str(DESCRIPTORS)
], check=True)
```

```python
# Cell 3: small validation-split pilot. Start with one geology per cohort,
# three rate levels, one best-response round, and one cooperative sweep.
SCENARIO = Path("/kaggle/input/<owner>/<scenario-dataset>/scenario.json")
REGIME_CSV = OUT / "regime_map_pilot.csv"
subprocess.run([
    sys.executable, str(REPO / "sequential_game" / "run_co2_regime_map.py"),
    "--data-dir", str(DATA), "--config", str(SCENARIO), "--out", str(REGIME_CSV),
    "--split", "validation", "--max-geologies-per-cohort", "1",
    "--rate-max-m3-s", "<reviewed-rate>", "--rate-levels", "3",
    "--response-rounds", "1", "--coop-sweeps", "1",
], check=True)
```

The pilot is a cost/implementation check, not a final result. The CLI reports
solver rollout evaluations and elapsed time. A full regime study should use
paired geology seeds across institutions, independent held-out seeds for
confirmation, and uncertainty intervals bootstrapped over geology—not merely
over transition rows.

The regime runner also writes `<output-stem>_counterfactuals.csv`. Run it
separately on train, validation, and test splits if you want policy-tree data;
fit on train rows and assess only on held-out validation/test rows.

For the separate finite-difference study, pass a training-cohort NPZ and its
case row. The wells must match the scenario; set the pressure cap and damage
coefficient to expert-reviewed values.

```python
INTERFERENCE = OUT / "interference.json"
subprocess.run([
    sys.executable, str(REPO / "sequential_game" / "co2_research_studies.py"),
    "interference", "--geology", str(DATA / "train" / "friendly.npz"),
    "--row", "0", "--well-cells", "22,8", "22,16", "--rates",
    "<rate0>", "<rate1>", "--rate-step", "<finite-difference-rate>",
    "--stage-days", "30", "--pressure-cap-pa", "<reviewed-cap-pa>",
    "--pressure-damage-per-pa2", "<reviewed-coefficient>", "--out", str(INTERFERENCE)
], check=True)
```

Run a matched empirical tariff sweep with `run_co2_regime_map.py --instruments
price --prices ...`; the prices and welfare must use consistent units. Compare
the local derivative and empirical price after the same paired cases finish:

```python
subprocess.run([
    sys.executable, str(REPO / "sequential_game" / "co2_research_studies.py"),
    "compare-price", "--theoretical", str(INTERFERENCE),
    "--empirical", str(OUT / "price_sweep.csv"),
    "--price-column", "price_per_kg", "--welfare-column", "project_resource_welfare",
    "--out", str(OUT / "price_comparison.json")
], check=True)
```

For equilibrium-aware data, first run the differentiable planner on a held-in
training geology with a trained CCS or CCS-GNN checkpoint, then have the solver
label that schedule. Retrain with the augmented dataset; never append active
labels to validation or test.

```python
PLAN = OUT / "best_response.json"
subprocess.run([
    sys.executable, str(REPO / "sequential_game" / "co2_differentiable_game.py"),
    "--data-dir", str(DATA), "--checkpoint", "/kaggle/input/<model>/co2_ccs_s0.pt",
    "--cohort", "challenging", "--geology-seed", "<training-seed>",
    "--rate-max-m3-s", "<reviewed-rate>", "--pressure-cap-pa", "<reviewed-cap-pa>",
    "--pressure-damage-per-pa2", "<reviewed-coefficient>", "--out", str(PLAN)
], check=True)

ACTIVE = Path("/kaggle/working/co2_xz_dataset_active_1")
subprocess.run([
    sys.executable, str(REPO / "sequential_game" / "collect_co2_equilibrium_data.py"),
    "--base-data-dir", str(DATA), "--schedule-json", str(PLAN), "--out-dir", str(ACTIVE)
], check=True)
```

```python
# Fit a fresh output directory so the previous run is not resumed accidentally.
TRAINED = OUT / "active_round_1_models"
subprocess.run([
    sys.executable, str(REPO / "sequential_game" / "train_co2_xz_surrogate.py"),
    "--data-dir", str(ACTIVE), "--out-dir", str(TRAINED),
    "--backbones", "ccs", "ccs_gnn", "--seeds", "0", "1", "2",
    "--epochs", "60", "--rollout-unroll", "6", "--device", "cuda"
], check=True)
```

Make another planner/label round only if solver regret or rollout metrics
improve. To verify a planned schedule, run `co2_solver_game_search.py evaluate`
with the same geology and scenario; for a discrete solver-reference response,
run `search`.

## Policy distillation input

The decision-tree tool needs a counterfactual solver table: one row per
`state_id, action_id`, with a `split` column, state feature columns (e.g. stage,
remaining permit, pressure, plume fraction), and the payoff column to optimize.
The tree is trained on train states and measured on validation/test states.
Its held-out regret is computed by looking up the tree-selected action in the
same solver counterfactual table. It does not infer missing action payoffs.

```bash
python co2_research_studies.py distill-policy --payoffs counterfactuals.csv \
  --out policy_tree.json --objective decision_payoff \
  --features stage pressure_before_pa plume_fraction_before --max-depth 3
```

For the generated counterfactual CSV, use `decision_payoff` and these exact
feature names: `stage`, `pressure_before_pa`, `plume_fraction_before`. Add
`permit_remaining_kg` only when a finite storage cap is configured. Filter the
CSV to `policy_type=nash_like` or `policy_type=cooperative` before fitting the
two trees separately.

## Computation and interpretation

- Ensemble is deliberately omitted. No disagreement-triggered solver calls are
  included in this implementation.
- The differentiable planner uses one model. The solver-reference coordinate
  searches can be much more expensive: each candidate rate requires a fresh
  full-horizon solver rollout. Start with one validation geology and three
  action levels; scale the paired design only after measuring wall time.
- A price derivative is local to the chosen geology, well cells, action point,
  stage duration, pressure-damage definition, and solver. It is not a universal
  market price.
- Regime correlations are descriptive. A geology-to-instrument rule requires
  held-out geologies and uncertainty intervals.
- The current data and code remain 2D x-z. A real 3D solver, physical pipeline
  topology, calibrated capacity/costs, expert-reviewed constitutive physics,
  and a pressure-cap conversion are separate prerequisites for 3D claims.
