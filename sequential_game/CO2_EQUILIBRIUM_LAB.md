# CO₂ equilibrium and well-placement pilot

This is a CO₂-only addition. It imports the CO₂ stage solver and well-placement adapter; it does not import the brine game or combine brine payoffs.

## What this pilot tests

`run_co2_equilibrium_pilot.py` creates paired finite games for held-out stage-0 geologies.

1. **Fixed-location rate game:** use one fieldwise merged surrogate. Pressure-tail CCS-GNN supplies the pressure change; three-step-unroll CCS-GNN supplies mobile gas, trapped gas, historical maximum gas, and dissolved-CO₂ changes. The merged planner's selected profiles are checked by solver replay and unilateral deviation regret. This run does not compare the component architectures.
2. **Placement game:** each operator chooses no well, or one of two candidate sites and a half/full rate. This is evaluated with the CO₂ solver across the held-out geology sample. The current surrogate training features hard-code two injection cells, so variable-site forecasts are out of distribution and deliberately prohibited.
3. **Equilibrium dynamics:** report all pure Nash profiles and all welfare-maximizing profiles for the expected-payoff game. Then compare random-start alternating and simultaneous best response, fictitious play, and damped logit response over several initializations/temperatures. Report exact unilateral regret at the selected terminal profiles, cycles, and non-convergence.
4. **Sensitivity arms:** vary pipeline dispatch, capacity, and the exploratory pressure-damage weight. These payoff coefficients are scenario assumptions, not calibrated monetary estimates.

The pilot solves a one-stage game exactly for the enumerated actions. Repeating a stage-local equilibrium at future states would be a myopic state-conditioned feedback policy; it is not a finite-horizon Markov-perfect equilibrium. A proper MPE result still requires continuation values or policy-level deviation checks across full trajectories.

## More than two operators

The physical CO₂ game and placement adapter support a configurable number of operators. The original equilibrium pilot was restricted to two players because its payoff analysis used a bimatrix. `co2_equilibrium_lab.py` now also has an n-player payoff-tensor solver and multi-start best-response dynamics, and `run_co2_multiplayer_placement.py` runs an exact one-stage placement game for configurable operator counts.

The multi-operator experiment offers each operator three actions: do not drill, or drill at either of two candidate cells and inject at the configured full rate. For four operators this is 81 joint profiles per geology and scenario arm; each profile is evaluated by the CO₂ solver. It reports exact pure Nash profiles, welfare maxima, and the frequency with which random-start best response reaches a pure Nash profile.

This runner is solver-only. Existing merged checkpoints have exactly two fixed operator-rate channels and do not support many operators or variable sites. For a valid multi-operator surrogate, regenerate CO₂ trajectories with randomized legal well locations and multiple operators, train on a total well-rate map (and owner/tracer channels if needed for attribution), preserve geology-level splits, then validate action regret against solver replays.

Example Kaggle command for four operators, one geology per cohort, and a single pilot arm:

```python
cmd = [sys.executable,
       str(REPO / 'sequential_game' / 'run_co2_multiplayer_placement.py'),
       '--data-dir', str(DATA_DIR),
       '--out-dir', '/kaggle/working/co2_multiplayer_placement',
       '--n-operators', '4',
       '--cohorts', 'friendly', 'challenging', 'tarbert', 'upper_ness',
       '--geologies-per-cohort', '1',
       '--dispatch-rules', 'proportional',
       '--capacity-fractions', '0.5',
       '--pressure-damages', '1e-9',
       '--min-spacing-ft', '80']
subprocess.run(cmd, check=True)
```

The current 24×24 domain and 80 ft spacing support only a modest number of distinct candidate cells. The action-profile count grows exponentially as `3 ** n_operators`; keep the exact game small, and move to sampled deviations or double-oracle methods for larger operator populations. The 80 ft spacing and pressure/payoff values are pilot assumptions that need scenario justification.

## Kaggle inputs and runtime

Attach:

- the CO₂ transition dataset as a Kaggle notebook/dataset input, at `/kaggle/input/notebooks/smerarawal/transition-geology/co2_xz_dataset_v2` (or set `DATA_DIR` to its actual mount);
- a code copy containing the repository's `sequential_game/` directory;
- optional `co2_ccs_s*.pt`, pressure-tail CCS-GNN, and unroll-3 CCS-GNN checkpoints from the **same dataset version and trainer run**. Checkpoint paths must be copied from that run's `/kaggle/input` mount. Do not use brine checkpoints.

Use **CPU** for the exact solver and placement search: the CO₂ IMPES reference is NumPy/SciPy CPU code. GPU is optional for surrogate forward passes; it will not speed up solver replays. Start with one held-out geology per cohort and the default small menu. Save `/kaggle/working/co2_equilibrium_pilot/` as a Kaggle output dataset so JSON results persist after the session.

If GitHub internet access is available, clone the branch that contains this file. If not, attach a ZIP of that source tree and extract it before importing.

```python
from pathlib import Path
import subprocess, sys

REPO = Path('/kaggle/working/shale-well-game-theory-codes')
if not REPO.exists():
    subprocess.run([
        'git', 'clone', '--depth', '1',
        '--branch', 'codex/co2-architecture-benchmark',
        'https://github.com/smerarawal/shale-well-game-theory-codes.git', str(REPO)
    ], check=True)
sys.path.insert(0, str(REPO))
```

## Run cells

First confirm the dataset and outputs:

```python
from pathlib import Path
DATA_DIR = Path('/kaggle/input/notebooks/smerarawal/transition-geology/co2_xz_dataset_v2')
assert (DATA_DIR / 'dataset_config.json').is_file()
assert all((DATA_DIR / split / 'friendly.npz').is_file()
           for split in ('train', 'validation', 'test'))
print((DATA_DIR / 'dataset_config.json').read_text()[:2500])
```

Then run the pilot. Supply both matched checkpoint families to enable the merged surrogate. The script can also run the solver-only placement experiment if both lists are empty. Use matched seed IDs; it builds one merged model for each seed.

```python
from pathlib import Path
import subprocess, sys

OUT = Path('/kaggle/working/co2_equilibrium_pilot')
PRESSURE_TAIL = []  # e.g. ['/kaggle/input/.../ablation_pressure_tail/co2_ccs_gnn_s0.pt', ...]
UNROLL3 = []        # e.g. ['/kaggle/input/.../ablation_unroll3/co2_ccs_gnn_s0.pt', ...]
cmd = [sys.executable, str(REPO / 'sequential_game' / 'run_co2_equilibrium_pilot.py'),
       '--data-dir', str(DATA_DIR), '--out-dir', str(OUT),
       '--cohorts', 'friendly', 'challenging', 'tarbert', 'upper_ness',
       '--geologies-per-cohort', '1', '--device', 'cpu']
for flag, values in [('--pressure-tail-checkpoints', PRESSURE_TAIL),
                     ('--unroll3-checkpoints', UNROLL3)]:
    if values:
        cmd += [flag, *values]
print(' '.join(cmd))
subprocess.run(cmd, check=True)
```

For a larger follow-up, increase `--geologies-per-cohort` to 2–5, add `intermediate`, add dispatch/capacity/pressure-damage values, and use matched seeds. This multiplies solver profiles; use a Kaggle CPU session with enough remaining time and persist outputs incrementally if the run is long.

The output `co2_equilibrium_pilot.json` includes hashes, geology IDs, game arms, payoff tensors, exact pure-Nash sets, solver replays of surrogate-selected equilibria, unilateral solver regrets, pressure matrices, and learning-dynamics traces.

## Interpretation and limits

- A model can forecast fields well yet induce poor actions. The main surrogate endpoint here is solver-replayed unilateral regret for the model-selected profile, alongside pressure and welfare—not its predicted welfare alone.
- The fieldwise hybrid is accepted only if it improves solver-replayed action quality over each component on matched geologies. It must be assessed across seeds and geologies before selection.
- Nash convergence depends on the update rule, tie-breaking, starting profile, and action grid. A high multi-start convergence fraction describes that finite game and algorithm, not convergence for every continuous policy.
- Expected-game equilibria are computed from the sample mean of selected test geologies. The lower-tail payoffs are available in the analysis utility as a risk sensitivity, but a small geology sample is not a calibrated posterior.
- Placement effects are evaluated with two fixed candidate sites per operator and a point-well 2D x–z solver. Do not interpret these as optimized trajectories or a 3D field plan.
- Existing zero trapped-CO₂ and pressure range audits remain prerequisites before interpreting trapped-gas or cap-risk conclusions. A closed-boundary simulation and an exploratory 5 MPa cap are not field validation.

## Well placement connection to Jef Caers

The useful connection is decision-making across geological realizations, not a claim that this pilot implements Caers's full workflow. Caers and coauthors' metric-space work frames uncertainty in terms of decision-relevant responses and ensembles of earth models ([Caers, Park & Scheidt, “Modeling uncertainty in metric space”](https://iamg09.stanford.edu/IAMG09%20Presentations/JCaersMetricFull%20Paper.pdf)). For this game, that means evaluate each candidate pair of locations across matched geologies, preserve per-geology response distributions, and compare expected and conservative payoffs before choosing a site.

The sequential extension should update the geology ensemble/beliefs after declared monitoring observations and re-optimize later locations, following the closed-loop field-development idea described in this Stanford report ([Gharib Shirangi, “Closed-loop Field Development Optimization”](https://pangea.stanford.edu/departments/ere/dropbox/scrf/documents/reports/26/SCRF2013_Report26/19.SCRF2013mehr2.pdf)). Neither belief updates nor history matching are fabricated in the pilot. A location-aware surrogate dataset must randomize candidate cells/locations and preserve geology-level train/validation/test splits before surrogate placement planning is enabled.
