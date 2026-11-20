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

The multi-operator experiment offers each operator three actions: do not drill, or drill at either of two candidate cells and inject at the configured full rate. For eight operators this is 6,561 joint profiles per geology and scenario arm; each profile is evaluated by the CO₂ solver. It reports exact pure Nash profiles, welfare maxima, and the frequency with which random-start best response reaches a pure Nash profile. The default run is deliberately one cohort, one geology, one dispatch rule, one capacity setting, and one damage weight; expand it after measuring runtime.

This runner is solver-only. Existing merged checkpoints have exactly two fixed operator-rate channels and do not support many operators or variable sites. For a valid multi-operator surrogate, regenerate CO₂ trajectories with randomized legal well locations and multiple operators, train on a total well-rate map (and owner/tracer channels if needed for attribution), preserve geology-level splits, then validate action regret against solver replays.

Example Kaggle command for eight operators, one challenging geology, and a single pilot arm:

```python
cmd = [sys.executable,
       str(REPO / 'sequential_game' / 'run_co2_multiplayer_placement.py'),
       '--data-dir', str(DATA_DIR),
       '--out-dir', '/kaggle/working/co2_multiplayer_placement',
       '--n-operators', '8',
       '--cohorts', 'challenging',
       '--geologies-per-cohort', '1',
       '--dispatch-rules', 'proportional',
       '--capacity-fractions', '0.5',
       '--pressure-damages', '1e-9',
       '--min-spacing-ft', '50']
subprocess.run(cmd, check=True)
```

The current 24×24 domain and default 50 ft spacing provide 16 distinct candidate cells for eight operators, with two alternatives per operator. The 50 ft value is one horizontal grid cell for the current 1200 ft / 24-cell grid; it is an exploratory spacing assumption, not an engineering recommendation. The action-profile count grows exponentially as `3 ** n_operators`; eight operators means 6,561 solver profiles per geology/arm. The default exact-enumeration limit is 10,000 profiles. Larger populations require sampled deviations, double-oracle methods, or a larger/structured site model. The pressure and payoff values remain pilot assumptions.

## Kaggle inputs and runtime

Attach:

- the CO₂ transition dataset as a Kaggle notebook/dataset input, at `/kaggle/input/notebooks/smerarawal/transition-geology/co2_xz_dataset_v2` (or set `DATA_DIR` to its actual mount);
- a code copy containing the repository's `sequential_game/` directory;
- the pressure-tail and unroll-3 CCS-GNN checkpoint families from the **same dataset version and trainer run** if you also want the two-operator merged-surrogate pilot. The multi-operator placement runner needs no checkpoint. Do not use brine checkpoints.

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
- The fieldwise hybrid is used as one planner. Solver replay measures its action regret, pressure, and payoff; it is not a component-architecture comparison.
- Nash convergence depends on the update rule, tie-breaking, starting profile, and action grid. A high multi-start convergence fraction describes that finite game and algorithm, not convergence for every continuous policy.
- Expected-game equilibria use mean payoffs across selected test geologies. The multi-operator runner also computes lower-tail CVaR payoffs and blends them with the mean to show risk-attitude sensitivity. A small geology sample is not a calibrated posterior; use multiple cohorts/geologies before interpreting the risk sweep.
- Placement effects are evaluated with two fixed candidate sites per operator and a point-well 2D x–z solver. Do not interpret these as optimized trajectories or a 3D field plan.
- Existing zero trapped-CO₂ and pressure range audits remain prerequisites before interpreting trapped-gas or cap-risk conclusions. A closed-boundary simulation and an exploratory 5 MPa cap are not field validation.

## Caers-related research strands and how they map to this project

“All of Caers’s work” spans many Earth-science problems, so the implementation uses the strands relevant to geological uncertainty and reservoir decisions rather than claiming to reproduce every method or paper.

1. **Geological realizations and distance-based uncertainty.** Caers and coauthors develop ways to compare and work with ensembles of geologic models in metric/distance spaces. Here, each candidate placement/profile is evaluated over the same held-out geology set; the output retains per-geology outcomes and a lower-tail payoff summary. A next step is response-aware model screening so a smaller set of solver geologies preserves diversity in pressure/plume responses, not only permeability maps. [Caers, Park & Scheidt, “Modeling uncertainty in metric space”](https://iamg09.stanford.edu/IAMG09%20Presentations/JCaersMetricFull%20Paper.pdf)
2. **Decision-centered uncertainty (Bayesian Evidential Learning).** The chain should connect observations, geological models, predicted responses, and decisions. Our decision endpoints are action regret, pressure risk, storage/payoff, and equilibrium sensitivity. This is why the merged surrogate is judged through the solver-checked action it selects, not just field RMSE. [SCERF’s BEL overview](https://scerf.stanford.edu/bayesian-evidential-learning-0)
3. **Risk preferences and value of information.** Report payoff distributions across geologies and vary how much the decision weights the mean versus the lower-tail outcomes. That is the role of the pilot’s lower-tail-CVaR payoff sensitivity; the payoff weights are still exploratory. Monitoring-well actions only become valuable-information experiments after an observation-noise model updates the geology belief and its value is computed against a no-monitoring policy. [Caers et al. on value of information for spatial Earth decisions](https://pangea.stanford.edu/departments/ere/dropbox/scrf/documents/reports/23/SCRF2010_Report23/SCRF2010_20.WTrainor_VOI_SCRF2.pdf), [Athens & Caers on geothermal development risk and information value](https://www.sciencedirect.com/science/article/pii/S0306261919316198)
4. **Sequential decisions and closed-loop model updating.** A later well decision should use the observations generated after earlier placement/injection and update the geology ensemble before re-optimizing. Our current script has one-stage game equilibria; the monitoring callback is only an interface and no observation model is invented. This fits the general ensemble/update/re-optimize loop in [the reservoir-management workflow paper](https://www.sciencedirect.com/science/article/pii/S092041052200715X).
5. **Geologic model checking.** Keep the geologic prior and solver assumptions explicit, preserve source hashes, and separate numerical closure from field validity. Caers’s work on history matching under geological control warns against modifying/upscaling models in ways that break their geological relationships. [Caers, “History matching under geological control”](https://pangea.stanford.edu/departments/ere/dropbox/scrf/documents/reports/16/SCRF2003_Report16/SCRF2003_Jef_Caers.pdf)

The current multi-operator code implements the first decision experiment: candidate-site portfolios across a small set of geology realizations, an expected-payoff Nash game, a mean/lower-tail risk sensitivity, and best-response convergence diagnostics. It does not yet implement distance-space posterior inference, history matching, monitoring-driven belief updates, or sequential re-optimization.

Before enabling merged-surrogate forecasts for many operators or new sites, regenerate CO₂ trajectories with randomized well positions and a larger operator count, then split by geology. The current checkpoints only encode two fixed operator rate channels.
