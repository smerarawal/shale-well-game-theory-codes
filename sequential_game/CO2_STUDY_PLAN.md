# CO₂ injection study plan

## Aim

Extend the sequential injection game from brine/water injection to CO₂ storage, then test whether the commons gap and the best policy instrument depend on geology. Keep all existing water results as a separate baseline. Do not combine water and CO₂ welfare or surrogate scores.

The central research question is:

> How do rock structure and operator position change pressure and plume externalities, the Nash–cooperative welfare gap, and the instrument that best manages shared storage?

The study should combine the six proposed ideas, with equilibrium-aware data generation as the main surrogate contribution.

## Dimensionality target: build toward 3D

The eventual target is a 3D CO2 game. The current reservoir solver, stage adapter, game state, and SPE10 patch sampler are 2D and square-grid-oriented; they are not a 3D implementation. The existing SPE10 patch path extracts a horizontal x-y layer, so gravity along one of those axes is not the SPE10 vertical direction. Do not describe those current patches as a 3D buoyant-plume experiment.

Use a staged path while keeping the new APIs dimension-aware:

1. **Physics-development case: vertical 2D x-z section.** Fix a y index in SPE10 and retain the x and z coordinates, vertical permeability, porosity, layer order, and physical cell dimensions. This allows buoyant rise and vertical layering at much lower cost than 3D. The two operators then occupy different x positions in the same section; state this limitation because it is not a full separate-lease geometry.
2. **Coarse 3D validation case.** After the vertical section passes the physical checks, build a small x-y-z model and rerun a few paired arms: cap-24 Nash versus cooperative, a capacity allocation, and a price treatment. Use identical geology, starting state, and action choices wherever possible. The purpose is to see which conclusions survive 3D, not to claim a fully powered 3D study.
3. **Full 3D game and surrogate.** Expand only after measuring runtime and numerical stability on the coarse 3D case. Make well completions (which layers a well intersects) part of the model specification and, if appropriate, the action space. Train a new 3D surrogate on solver trajectories; the existing 2D GNN+prior2 and CCSNet checkpoints cannot be reused as validated 3D CO2 surrogates.

Do not assume the estimated 15–30x slowdown or study runtime in meeting notes is measured. Benchmark pressure-solve time, transport substeps, and complete planner schedules on the intended hardware before setting sample size. The current SciPy sparse solver is CPU code; two Kaggle T4 GPUs do not automatically accelerate it. A 3D pressure backend and a 3D surrogate are separate engineering tasks.

## Three geology cohorts

Use three synthetic regimes to test whether geology makes equilibrium computation easier or harder, plus a distinct SPE10 benchmark cohort. Here “suited” means the game’s best-response process is more likely to settle to a stable equilibrium; it does not mean the rock favors the human researcher or one operator.

1. **Equilibrium-friendly synthetic cohort:** weaker cross-operator pressure/plume coupling and smoother payoff responses. Best-response iteration is expected to converge consistently from multiple starting action profiles.
2. **Intermediate synthetic cohort:** moderate heterogeneity and cross-operator coupling between the friendly and challenging regimes. Randomize geology seeds, permeability multipliers, and well layouts within fixed bounds, as in the earlier experiments. This is the middle regime, not a post-hoc selection of geologies whose measured outcomes happen to be halfway between the endpoints.
3. **Equilibrium-challenging synthetic cohort:** stronger cross-operator coupling, frequently binding pressure/plume constraints, or sharper action-payoff changes. Best-response iteration is expected to be slower, more sensitive to its starting point, or prone to cycling.
4. **SPE10 benchmark cohort:** Tarbert and Upper Ness property fields as heterogeneity stress tests. SPE10 Model 2 is a synthetic waterflood benchmark, not field-measured CO₂ data. Tarbert is comparatively smooth; Upper Ness is channelized. For vertical buoyancy work, use a vertical x-z section first and the full 3D volume for the later validation. Keep the existing horizontal-patch CO2 cohort as a separate lateral-heterogeneity debug dataset only. Describe all such runs as CO2 simulations on SPE10 rock properties, not as an official or field-calibrated SPE10 CO2 benchmark. Consider SPE11B as a CO2-specific benchmark extension; it is a benchmark case, not a real field dataset.

### Geology-input visualization and descriptive analysis

After generating the NPZ and manifest, create figures and cohort summaries with:

```bash
python sequential_game/visualize_co2_geologies.py --input-dir /path/to/co2_geologies
```

The script writes six-sample synthetic-regime panels, separate SPE10 horizontal-patch and vertical-section figures, descriptor-distribution figures split by normalized inputs versus raw SPE10 x-z sections, and a cohort-level descriptor CSV under `figures/`. Synthetic maps share color scales across the three regimes. Horizontal SPE10 patches are explicitly labeled as per-patch normalized inputs; they are not raw-contrast comparisons. Vertical SPE10 plots retain separate Kx/Kz and porosity, use log10(mD) for permeability, and use physical x/z coordinates with the SPE10 cell spacing. The descriptor plots summarize input properties before flow and keep the normalized and raw-K scales apart. They cannot validate that friendly/intermediate/challenging cases actually produce the intended interference or equilibrium difficulty; that requires solver-derived measurements.

To connect the controlled middle regime to SPE10 without obscuring the benchmark, add an optional **SPE10-derived randomized bridge set**: vary vertical slice location or 3D subvolume, well placement/completions, and bounded permeability multipliers while retaining SPE10 spatial structure. Report original SPE10 fields and randomized variants separately. Do not blend SPE10 arrays numerically with unrelated synthetic fields unless they are first mapped to a common grid and that transformation is documented.

Define the synthetic regimes by generation settings and pre-game physical diagnostics, then freeze them before evaluating policy outcomes. The intermediate regime should be specified by parameter ranges between the friendly and challenging generator settings, not by selecting fields after observing convergence. Measure equilibrium behavior separately: iterations to a best-response residual tolerance, convergence rate, cycling/non-convergence, and whether different starting profiles reach different equilibria. This avoids selecting geologies using the very equilibrium results being reported. Also record whether each geology is physically easy or difficult for storage; that is a separate axis from whether the game is easy to solve.

### Geology descriptors to save

For every field, calculate and store before policy outcomes are examined:

- permeability and porosity distributions, anisotropy, and vertical layering;
- horizontal and vertical correlation lengths;
- connected high-permeability channel fraction and connectivity between injection sites;
- standardized single-well injectivity / pressure response;
- cross-well pressure interference under a fixed rate perturbation;
- plume footprint and boundary-contact risk under a standard injection schedule;
- best-response convergence diagnostics on a small, fixed pilot action set, used only to verify the designed synthetic regime and then reported separately from confirmatory policy comparisons.

These descriptors support the rule “for this geology class, use this instrument.” Keep the descriptors continuous as well as using any later regime bins.

## Gate 1: define and validate CO₂ physics

The current game adapter injects the wetting water phase. The current transport treatment also uses total-flow direction for saturation upwinding, which cannot reliably represent counter-current buoyant segregation. Changing the phase label or reversing a gravity sign is not sufficient.

**Implementation status:** the pushed `co2_solver` path provides phase-specific Darcy-flux plumbing and a brine-filled initial condition. The working tree contains `co2_experiment.py` for randomized synthetic regimes, the existing scaled horizontal SPE10 patch cohort, and a new data-only vertical x-z SPE10 cohort. `spe10_geologies.py` now exposes raw Kx/Ky/Kz in mD with porosity in `[z,y,x]` order; x-z cases retain Kx and Kz separately plus cell spacing and slice metadata. These directional sections are **not consumable by the existing scalar-k game or solver yet**. Optional geometric-mean coarsening is data reduction, not transmissibility-preserving upscaling. The flow prototype still tracks pressure and brine saturation only; it has **not** implemented CO₂ dissolution or residual trapping, the payoff ledger is not yet connected to `game3.py`, and none of this path is physically validated. Do not interpret generated game results until those pieces are integrated and validated.

The requested physical scope is a CO₂–brine model with mobile free-phase CO₂, dissolved CO₂ in brine, residual/trapped CO₂, pressure, phase-specific relative permeability, gravity, and phase-aware wells. A consistent state should carry pressure, brine saturation, mobile CO₂ saturation, trapped CO₂ inventory, and dissolved CO₂ concentration (or mass fraction). Define the component inventory in mass units and make dissolution and trapping transfers conservative: dissolution transfers CO₂ from mobile gas to aqueous concentration; residual trapping transfers it from mobile to immobile gas. Trapped CO₂ should not keep contributing gas mobility. A history-dependent trapping relation (for example, a documented Land-type residual-trapping law) and a documented solubility/mass-transfer model are needed.

The first implementation can use a clearly labelled simplified model with constant reference densities/viscosities, a documented solubility relation and finite mass-transfer rate, and a hysteretic residual-trapping rule. It must still conserve the CO₂ component. Thermal effects, geochemistry, geomechanics, and field-calibrated PVT can remain outside scope if stated. Record every constitutive parameter and unit; until calibrated, call outputs normalized/nondimensional research simulations rather than field forecasts.

Before a game run, require these physical checks:

1. With no wells, buoyancy moves the lighter CO₂ upward and brine counterflows; reversing gravity reverses segregation.
2. A no-gravity control does not develop vertical drift from the gravity term.
3. Phase-by-phase mass balance closes over many steps, and saturation remains bounded without silent mass loss from clipping.
4. CO₂ injection and brine production sources have the intended phase rates and a stable pressure response.
5. The plume direction and mass balance persist under timestep and grid refinement.

Save diagnostics for every check. If the current solver cannot pass them, use a CO₂-capable benchmark simulator or implement a suitable phase-flux formulation before interpreting any game or surrogate result.

## Gate 2: define the CO₂ game and outcomes

Retain the existing finite-horizon sequential-game structure, but use a separate CO₂ configuration and result namespace. Define decisions, shared constraints, and rewards before sweeping instruments.

The selected game objective includes **both** pressure costs and storage/plume value. Track physical outcomes separately from welfare:

- injected and retained CO₂ mass;
- mobile, residually trapped, and dissolved mass where represented by the model;
- plume footprint, overlap between operators, and contact with domain or caprock boundaries;
- pressure maximum, pressure at rival wells, and time above a chosen pressure threshold;
- permit use, drilling, each operator’s payoff, total welfare, and the Coop − Nash gap.

State the unit and calibration of stage duration, injection rate, pressure, storage capacity, and monetary payoffs. The current project’s stage-time units are not yet calibrated to years or tonnes, so do not present them as field forecasts until calibrated.

Compare the existing policy instruments where their definitions transfer cleanly (for example, shared cap, fixed allocation, price, and cooperative coordination). Treat a price such as 5–10 as a candidate carried over from the water study, not as a CO₂ result or assumed optimum.

Define an explicit payoff ledger before policy sweeps. The code scaffold in `co2_experiment.py` expresses stage welfare as injection revenue plus a credit for CO₂ retained at the evaluation horizon, less drilling cost, pressure-damage cost, and plume-overlap/containment-risk cost. Count stored CO₂ once: attribute retained mass to its injecting operator with an operator tracer or component ledger, and define how shared plume/pressure damages are allocated to private payoffs. Keep transfers separate from total resource welfare. Normalize or calibrate all terms to compatible units before interpreting the “best” price or welfare gap.

## Study design

Use paired comparisons: every instrument and planner treatment sees the same geology, initial state, action choices, and random seeds. Split train, validation, and test sets by geology seed, never by adjacent states from one geology. For synthetic regimes, balance operator labels and well layouts so that equilibrium difficulty comes from geology/game coupling rather than a built-in player advantage. Keep original SPE10 fields held out as benchmark evaluations when SPE10-derived variants appear in training or development data.

Start with a small paired pilot in each cohort to verify outputs and measure runtime. Then choose the full sample size using the pilot’s paired variance and a stated precision target. A reasonable planning range is 24–36 independent geologies per cohort, but the final number should follow runtime and interval-width calculations. Report achieved sample size and paired confidence intervals; do not compare treatments on different geology sets.

Use a consistent primary endpoint: paired total-welfare gap, Coop − Nash. Secondary outcomes include each operator’s payoff, injection and permit utilization, pressure and plume metrics, and whether the policy instrument changes the sign or size of the gap. Report geology-level values, not only pooled averages.

## Research sequence

### A. Regime map and policy instruments

Once the CO₂ solver passes Gate 1, run paired Nash and cooperative policies across the three cohorts and the instrument set. For SPE10, compare matched Tarbert and Upper Ness layer bands and quantify channel connectivity and correlation length. Estimate how those descriptors relate to the commons gap, plume interaction, and best-performing instrument. Hold out some geology seeds for confirmation of the final geology-to-instrument rule.

### B. Physics-based price estimate

For each operator’s standardized rate perturbation, use the solver to estimate the marginal effects on the other operator’s pressure, plume overlap, and containment risk. Convert those cross-effects into marginal damage in the same payoff units, then sum harm imposed on rivals to obtain a candidate Pigouvian price per injected unit. Compare it against an empirical price sweep on the same paired geologies. Report both the derivative estimates and the price units; a pressure sensitivity alone is not a price until multiplied by the payoff derivative of the relevant harm.

### C. Equilibrium-aware surrogate training — primary ML experiment

Compare two label-allocation strategies at equal solver-label budgets:

- **Random baseline:** train on broad, randomly sampled states as in the earlier surrogate work.
- **Equilibrium-aware loop:** train an initial model, run game policies, collect visited states near Nash/cooperative trajectories and pressure/plume constraints, label those states with the validated solver, retrain, and repeat.

Evaluate on held-out geologies by one-step field errors and, more importantly, decision regret: the difference in realized solver welfare between the solver-chosen action and the surrogate-planned action. Plot regret, pressure/plume violations, and runtime against solver labels. Keep the solver as the environment and welfare reference during the policy comparison.

### D. Disagreement-triggered solver calls

Train an ensemble. During planning, request an exact solver label when model disagreement is high and the predicted state is near pressure, plume, or containment constraints. Compare solver-call cost against decision regret and violation rates. Treat this as an extension of the equilibrium-aware loop so both methods share the same label budget and held-out geologies.

### E. Differentiable game and sensitivities

After the simulator is validated, first differentiate through a pressure/surrogate model and compare gradients with finite-difference solver sensitivities. Then test gradient best responses against the current discrete action search on the same cases. Do not treat gradients as reliable near phase-front changes, clipping, or other nonsmooth events without explicit checks.

### F. Policy distillation

After stable policies are available, distill Nash and cooperative choices into small decision trees or rules based on stage, remaining permit, pressure, and plume indicators. Test rules such as “inject at the high rate until a pressure/plume threshold, then taper” on held-out geology seeds using the solver. Report the welfare retained and safety violations relative to the full policy.

## Surrogate architecture plan for CO₂

### Architecture candidates to recompute for CO₂

The previously shared architecture comparison is an **initial screen on one randomized synthetic source**. Its L2-style scores were used to identify candidates, not as results on the new friendly/intermediate/challenging cohorts, SPE10, or CO₂ physics. Do not present or copy those values into the CO₂ results. Recompute every reported architecture metric on the CO₂ solver's held-out geology sets before drawing conclusions.

Use these as candidates for the new experiment, not as pre-ranked CO₂ winners:

- **Core field-model candidates:** CCSNet, U-Net baseline, U-FNO, trajectory-trained ResUNet, and plain FNO.
- **Game-specific carry-over candidate:** GNN + prior2, because it performed well in the separate six-stage brine game. This is a different task and is included to test transfer of the game-oriented design, not as a CO₂ performance claim. The earlier CPU speedup was measured on the old 2D setup and must be remeasured.
- **One exploratory architecture hybrid:** CCSNet + ConvLSTM trajectory memory, to test whether recurrent state helps a multi-stage game. Do not create every pairwise backbone combination.
- **Defer from the first CO₂ screen:** DeepONet, E2C, DeepONet-KAN, FINN, GNS, nested FNO, and physics-informed ConvLSTM/FNO variants. Their initial single-source screen does not justify spending the first CO₂ budget on them. `GNS` is also not the same implementation as the earlier game-specific `GNN + prior2` candidate.

The existing local repository defines 13 different **2D stage-surrogate** backbones. The previous architecture list has 16 models, while the separate whole-trajectory screen supplied here has 15 rows. These inventories are not interchangeable; this CO₂ plan uses the candidate names above and requires their appropriate 3D or x-z implementations before training.

### Staged comparison and physics hybrids

Do not start this comparison until the CO₂ solver, trajectory dataset, 3D/x-z model inputs, and physical priors refer to the same state variables and pass the Gate 1 checks. The existing architecture code and `prior`/`prior2` currently operate on 2D pressure and brine saturation; they are not drop-in CO₂/3D models.

1. **Pilot architecture screen:** on identical train/validation/test geology splits, train the five field-model candidates plus GNN+prior2 and the optional CCSNet-ConvLSTM candidate. Use one seed to screen, not to declare a winner. Start with a common mixed training set spanning friendly, intermediate, challenging synthetic cases and a defined SPE10 training subset. Hold out entire geology seeds, vertical slices, and original SPE10 regions as appropriate.
2. **Prior ablation:** compare `none`, `prior`, and `prior2` on CCSNet, U-Net, and GNN. Compare `none` and `prior2` on the remaining screen candidates. This tests the value of a CO₂-specific differentiable pressure solve without multiplying every backbone by every hybrid. `prior` must use current phase mobilities; `prior2` must recompute pressure using the predicted mobile saturation path. Both must include anisotropic K, vertical gravity, well sources, and the same pressure equation as the validated simulator.
3. **Confirmatory run:** take the best two or three candidates by held-out geology-level decision regret and physical safety metrics, then train at least three seeds each. Preserve seed-level results and report mean, standard deviation, and geology-bootstrap confidence intervals. Include a solver-only reference. Benchmark inference and planning time against the same 3D solver on the same hardware; the old 4.3x result is not a 3D speed claim.
4. **Trajectory/game validation:** score both teacher-forced one-step transitions and autoregressive multi-stage rollouts. Evaluate the actual planner actions using solver rollouts, so a small field error cannot conceal a poor action ranking or a pressure/plume violation.

The requested earlier comparison refers to 16 individual architectures, while the supplied whole-trajectory table contains 15 ranked rows and the current local `stage_surrogate.py` defines 13 different 2D backbones. Keep these experiment inventories separate. Use the five evidence-backed trajectory models above plus the prior GNN game candidate for the first CO₂ screen; do not claim this reproduces either previous 16-model experiment.

### Required CO₂ error analysis

Report each metric per geology and per stage, including the final autoregressive stage. Preserve the same test cases across architectures and seeds.

| Error family | Required measures |
|---|---|
| Pressure field | RMSE, MAE, max-absolute error, R²; p_max absolute error, signed bias and high percentile; top-decile-pressure RMSE; well-cell RMSE |
| Phase fields | Brine and mobile-CO₂ saturation RMSE/MAE/max error; plume footprint IoU/F1 at declared saturation cutoffs; plume area/volume and front displacement |
| CO₂ accounting | Dissolved, mobile, and residual-trapped mass absolute/relative error; total component mass-balance residual; retained-mass error |
| Safety/payoff | Pressure-cap violation classification and rate; error in the pressure penalty `30 × max(0, p_max − 6)^2` if that cap and calibration are retained; plume overlap and boundary-contact error; payoff error and solver-measured action regret |
| Robustness and cost | Metrics by friendly/intermediate/challenging, rough/smooth, Tarbert/Ness, more-well and unseen-slice sets; stage-wise error growth; parameters, peak memory, train time, batch-1 inference time and full planner runtime |

The pressure penalty above is inherited from the current game experiment, not yet calibrated for CO₂. The CO₂ payoff also includes retained storage and plume/containment terms, so report errors in each payoff component and total welfare separately. The supplied five-model OOD p_max range and negative SPE10 bias make pressure-tail, well-cell, and p_max-bias analysis mandatory, not optional.

## Result and artifact requirements

Keep three sources of uncertainty distinct: physical-model error, game-solution/search error, and surrogate prediction/planning error. Preserve complete trajectories and metadata: fluid model/version, geology source/seed, cohort assignment, operator mapping, gravity, grid and timestep, initial state, wells/actions, phase rates, solver diagnostics, model/planner versions, and all payoff components.

Create separate outputs for water and CO₂. The CO₂ summary should include cohort sample counts, paired intervals, price sweep, regime-map plots, regret-versus-label curves, solver-call cost, and policy-distillation performance. Mark pilot results and incomplete analyses as exploratory.

## Scope and external benchmark sources

- SPE10 is described by SINTEF as a Model 2 benchmark with a comparatively smooth shallow-marine Tarbert section and a fluvial Upper Ness section: https://www.sintef.no/projectweb/geoscale/results/msmfem/spe10/.
- SPE11 is an SPE comparative benchmark specifically for geological CO₂ storage, with cases motivated by laboratory, 2D field-transect, and 3D field-model settings. It is a benchmark rather than an observed field dataset. Background: https://www.pnnl.gov/publications/benchmarking-co2-storage-simulations-results-11th-society-petroleum-engineers. Full comparison data are described in the paper at https://doi.org/10.1016/j.ijggc.2025.104519 and dataset DOI https://doi.org/10.18419/DARUS-4750.
