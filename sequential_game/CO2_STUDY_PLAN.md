# CO₂ injection study plan

## Aim

Extend the sequential injection game from brine/water injection to CO₂ storage, then test whether the commons gap and the best policy instrument depend on geology. Keep all existing water results as a separate baseline. Do not combine water and CO₂ welfare or surrogate scores.

The central research question is:

> How do rock structure and operator position change pressure and plume externalities, the Nash–cooperative welfare gap, and the instrument that best manages shared storage?

The study should combine the six proposed ideas, with equilibrium-aware data generation as the main surrogate contribution.

## Three geology cohorts

Use three synthetic regimes to test whether geology makes equilibrium computation easier or harder, plus a distinct SPE10 benchmark cohort. Here “suited” means the game’s best-response process is more likely to settle to a stable equilibrium; it does not mean the rock favors the human researcher or one operator.

1. **Equilibrium-friendly synthetic cohort:** weaker cross-operator pressure/plume coupling and smoother payoff responses. Best-response iteration is expected to converge consistently from multiple starting action profiles.
2. **Intermediate synthetic cohort:** moderate heterogeneity and cross-operator coupling between the friendly and challenging regimes. Randomize geology seeds, permeability multipliers, and well layouts within fixed bounds, as in the earlier experiments. This is the middle regime, not a post-hoc selection of geologies whose measured outcomes happen to be halfway between the endpoints.
3. **Equilibrium-challenging synthetic cohort:** stronger cross-operator coupling, frequently binding pressure/plume constraints, or sharper action-payoff changes. Best-response iteration is expected to be slower, more sensitive to its starting point, or prone to cycling.
4. **SPE10 benchmark cohort:** Tarbert and Upper Ness property fields as heterogeneity stress tests. SPE10 Model 2 is a synthetic waterflood benchmark, not field-measured CO₂ data. Tarbert is comparatively smooth; Upper Ness is channelized. Run the original property fields as a distinct benchmark after CO₂ physics validation. Describe these as CO₂ simulations on SPE10 rock fields, not as a validated SPE10 CO₂ benchmark. Consider SPE11B as a CO₂-specific benchmark extension; it is a benchmark case, not a real field dataset.

To connect the controlled middle regime to SPE10 without obscuring the benchmark, add an optional **SPE10-derived randomized bridge set**: vary field seed/layer band where available, well placement, and bounded permeability multipliers while retaining the SPE10 spatial structure. Report original SPE10 fields and randomized variants separately. Do not blend SPE10 arrays numerically with unrelated synthetic fields unless they are first mapped to a common grid and that transformation is documented.

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

**Implementation status:** a separate `co2_solver` stage-model path and phase-specific Darcy-flux option have now been added as a prototype. The existing `solver` path retains its historical behavior. The CO₂ path is not yet validated and must not be used to claim game, equilibrium, or surrogate results until the checks below are completed.

First specify the model’s scope. A feasible first version is a two-phase, immiscible CO₂–brine model with explicit density contrast, phase-specific relative permeability, gravity, wells, pressure, and saturation. Record which processes are omitted, such as dissolution, capillary trapping, thermal effects, geochemistry, and geomechanics. This is a simplified research model unless those processes are added and calibrated.

Before a game run, require these physical checks:

1. With no wells, buoyancy moves the lighter CO₂ upward and brine counterflows; reversing gravity reverses segregation.
2. A no-gravity control does not develop vertical drift from the gravity term.
3. Phase-by-phase mass balance closes over many steps, and saturation remains bounded without silent mass loss from clipping.
4. CO₂ injection and brine production sources have the intended phase rates and a stable pressure response.
5. The plume direction and mass balance persist under timestep and grid refinement.

Save diagnostics for every check. If the current solver cannot pass them, use a CO₂-capable benchmark simulator or implement a suitable phase-flux formulation before interpreting any game or surrogate result.

## Gate 2: define the CO₂ game and outcomes

Retain the existing finite-horizon sequential-game structure, but use a separate CO₂ configuration and result namespace. Define decisions, shared constraints, and rewards before sweeping instruments.

Track physical outcomes separately from welfare:

- injected and retained CO₂ mass;
- mobile, residually trapped, and dissolved mass where represented by the model;
- plume footprint, overlap between operators, and contact with domain or caprock boundaries;
- pressure maximum, pressure at rival wells, and time above a chosen pressure threshold;
- permit use, drilling, each operator’s payoff, total welfare, and the Coop − Nash gap.

State the unit and calibration of stage duration, injection rate, pressure, storage capacity, and monetary payoffs. The current project’s stage-time units are not yet calibrated to years or tonnes, so do not present them as field forecasts until calibrated.

Compare the existing policy instruments where their definitions transfer cleanly (for example, shared cap, fixed allocation, price, and cooperative coordination). Treat a price such as 5–10 as a candidate carried over from the water study, not as a CO₂ result or assumed optimum.

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

## Result and artifact requirements

Keep three sources of uncertainty distinct: physical-model error, game-solution/search error, and surrogate prediction/planning error. Preserve complete trajectories and metadata: fluid model/version, geology source/seed, cohort assignment, operator mapping, gravity, grid and timestep, initial state, wells/actions, phase rates, solver diagnostics, model/planner versions, and all payoff components.

Create separate outputs for water and CO₂. The CO₂ summary should include cohort sample counts, paired intervals, price sweep, regime-map plots, regret-versus-label curves, solver-call cost, and policy-distillation performance. Mark pilot results and incomplete analyses as exploratory.

## Scope and external benchmark sources

- SPE10 is described by SINTEF as a Model 2 benchmark with a comparatively smooth shallow-marine Tarbert section and a fluvial Upper Ness section: https://www.sintef.no/projectweb/geoscale/results/msmfem/spe10/.
- SPE11 is an SPE comparative benchmark specifically for geological CO₂ storage, with cases motivated by laboratory, 2D field-transect, and 3D field-model settings. It is a benchmark rather than an observed field dataset. Background: https://www.pnnl.gov/publications/benchmarking-co2-storage-simulations-results-11th-society-petroleum-engineers. Full comparison data are described in the paper at https://doi.org/10.1016/j.ijggc.2025.104519 and dataset DOI https://doi.org/10.18419/DARUS-4750.
