# CO₂ Pipeline–Reservoir Sequential Markov Game: Design Blueprint

## Purpose and scope

Build a second game alongside the existing water/brine-injection game. It represents operators sharing both a CO₂ delivery system and a geological storage formation. The initial research question is:

> How do pipeline capacity, reservoir geology, and operator differences shape injection decisions, storage outcomes, and the welfare gap between noncooperative and coordinated policies?

This design specification accompanies the CO₂ game environment in `co2_pipeline_game.py` and `co2_well_placement_game.py`. The checkout also contains a water game, an experimental vertical x–z CO₂/brine reservoir solver, and a CO₂ payoff scaffold. It does not yet contain a pipeline network model or a game-theoretic policy/equilibrium runner. Keep existing water results in their own namespace; do not pool water and CO₂ welfare or surrogate errors.

## Modeling boundary

The model has two coupled physical subsystems:

1. **Pipeline delivery:** one common CO₂ supply feeds one shared pipeline-capacity bottleneck serving all operators. The tighter of source availability and pipeline capacity limits total stage delivery. The current game allocates that shared quantity between operators; it does not route flow to individual wellheads.
2. **Subsurface storage:** delivered CO₂ is injected into a brine-filled formation. Pressure, buoyant migration, plume interaction, dissolution, residual trapping, and possible boundary contact evolve over time. Brine is displaced as CO₂ occupies pore volume.

The first game version should use a transparent, stage-level pipeline capacity and cost abstraction. It should not claim to simulate pipeline hydraulics. Add pressure, temperature, linepack, compressor dynamics, and routing only after the user’s pipeline implementation and units are available. A CO₂ transport model and a geological storage model are separate components of the broader CCS chain; NETL’s cost methodology explicitly treats pipeline rate, distance, elevation, pumps, and diameter as transport factors. [NETL transport and storage cost guidelines](https://netl.doe.gov/projects/files/QualityGuidelinesforEnergySystemStudiesCarbonDioxideTransportandStorageCostsinNETLStudies_073124.pdf)

## Game class and time

- **Players:** start with two storage operators for comparison, but keep the operator count configurable. State/action structures must not assume exactly two operators or one well per operator.
- **Horizon:** six stages for direct comparability with the existing game, but configurable. A stage is a declared operational interval (for example, a 30-day pilot step); this duration is a scenario parameter until calibrated.
- **Information:** geology and some reservoir parameters are fixed but initially unknown to operators. Operators observe their own actions, delivered quantities, pipeline signals available to them, and noisy monitoring data. The observation model must state which pressure/plume measurements are public, private, or unavailable.
- **Solution concept:** finite-horizon partially observable stochastic game. The maintained information state is the joint belief over unknown geology/parameters plus the physical state. Nash and cooperative policies are computed approximations; finite-horizon best-response results must not be called a proven exact Markov-perfect Bayesian equilibrium.
- **Randomness:** geology seed, observation noise, and any explicitly modeled outages or disturbances. A paired comparison fixes these random inputs across policies.

## Markov state

At the beginning of stage `t`, define the full state as

`S_t = (t, X_t^res, X_t^pipe, W_t, C_t, L_t, B_t, Z_t)`.

The belief state is needed because the physical geology is hidden. The augmented state is Markov if the stored variables are sufficient to compute the next physical state, observation distribution, and payoff.

### Reservoir state `X_res`

Use fields on the CO₂ solver grid:

- pressure `p_pa[z,x]`;
- mobile gas saturation `sg_mobile[z,x]`;
- residual trapped gas saturation/inventory `sg_trapped[z,x]`;
- dissolved CO₂ concentration or component mass `c_dissolved_kg_m3[z,x]`;
- historical maximum gas saturation `sg_max[z,x]` for hysteretic trapping;
- cumulative injected component mass and simulation time;
- per-operator CO₂ tracer/component inventories, so retained storage can be attributed without counting shared storage twice.

Brine saturation is derived as `Sw = 1 - sg_mobile - sg_trapped` in the current prototype. If brine production is added, carry produced brine mass/volume and its operator/source attribution as ledger state.

The current x–z pilot has mobile/trapped/dissolved CO₂ and `sg_max`, but no producing wells. Its closed boundaries and simplified IMPES physics define the first-game scope. Do not let the game silently assume brine production, leakage, or field-calibrated behavior that the solver does not simulate.

### Pipeline state `X_pipe`

**Version 1, stage-level abstraction:**

- one shared throughput limit and one common source-availability limit (mass per stage);
- exogenous availability/outage flag, if outages are part of the scenario;
- remaining contracted/source quantity, if supply is limited;
- cumulative delivered mass and resource cost by operator.

No within-stage linepack or hydraulic pressure is represented in v1. A capacity number is a scenario constraint, not a claim about a real pipe.

**Version 2, dynamic network:** add linepack/mass inventory, node and segment pressures, compressor operating state, temperature/composition as required by the chosen pipeline model, and topology. These variables must come from the actual pipeline code/physics rather than guessed game constants.

### Wells, permits, and accounts

- persistent well IDs, site, type, operator ownership, drilling/commissioning stage, last realized rate, cumulative injection by well, and remaining drill budget;
- shared pipeline capacity already used during the stage if dispatch is sequential;
- cumulative storage/injection permit used and any operator-specific quota balances;
- cumulative operator payoffs split into revenue/storage credit, pipeline cost, compression energy, drilling cost, pressure damage, plume/containment damage, and transfers.

### Future well-placement and monitoring decisions

The game must grow beyond a small fixed set of wells. The current transition kernel already permits a variable-size set of grid-cell wells per operator and a `new_wells` action. That is a useful integration hook, but it is only simple cell selection; it is not an optimized, geometry-constrained well-placement model.

For a future Caers-style sequential placement workflow, extend the game while preserving the current shared-supply and shared-pipeline boundary:

- Give every well a stable ID, owner, type (injector or monitor), candidate site or trajectory, drilling/activation stage, completion, last realized rate, and cumulative injected mass. Do not identify a well by its position in an array.
- Put candidate locations and geometric constraints in a separate placement module: eligible zones, minimum spacing, fault/boundary setbacks, trajectory constraints, site exclusivity, maximum well counts, and drilling budgets. The game validates proposed placements through this module; the reservoir solver receives the resulting well controls.
- Make placement, drilling/activation, shut-in, and rate control explicit actions. If drilling has a lead time, store planned, under-construction, and active wells separately. Do not silently let a newly selected location inject immediately unless that scenario says so.
- Let each operator choose only from its private/legal candidate set and control only its own wells. The dispatcher continues to aggregate nominations across operators and allocate the common source/pipeline throughput before reservoir advancement.
- Add monitoring-well placement and observations as distinct actions only when an observation model is implemented. Store uncertain geology in the belief state; monitoring changes beliefs and can therefore change later placement and injection actions.
- Keep pipeline dispatch operator-level in the current model. A future network can add site-level routing through a separate adapter without changing the game’s well identities or reservoir state.

This is compatible with the general approach in Wang et al., *Optimizing Carbon Storage Operations for Long-Term Safety*, which formulates sequential injector/monitor placement under geological uncertainty as belief-state planning and uses a flow surrogate to make repeated planning simulations tractable. This prototype does not yet implement that paper's POMDP, monitoring, 3D physics, or surrogate.

### Hidden parameters and belief `B_t`

The fixed hidden parameter `theta` may include geology, permeability multiplier, and pre-registered uncertain constitutive parameters. Each operator has a belief `b^i_t(theta)` consistent with its observations. If monitoring is public, beliefs may be common; if not, keep operator beliefs distinct. Do not put the true geology in the action-selection information set for learning arms.

### Exogenous state `Z_t`

Represent only declared external processes, such as source availability, a sampled pipeline outage, or scheduled maintenance. If these are absent from the experiment, omit them rather than adding unmotivated stochasticity.

## Actions

At each stage, each operator chooses an action simultaneously from its information set. The number of operators and each operator's active/candidate well sets are variable:

`a^i_t = (q^i_t, d^i_t, w^i_t, r^i_t, m^i_t)`

where:

- `q_i`: CO₂ delivery nomination or requested mass for the stage;
- `d_i`: pipeline/capacity bid or priority request if an allocation mechanism is under study;
- `w_i`: drill/activate/hold action and well site/completion choice; v1 supports adding grid-cell injection wells, while monitoring wells and trajectories are future extensions;
- `r_i`: requested injection rate schedule across owned active wells;
- `m_i`: monitoring choice, only if monitoring has a modeled cost and changes observations.

Version 1 can discretize `q_i` and `r_i` using a small action grid aligned to the current game’s discrete rate choices. Separate pipeline nomination from reservoir injection rate: delivery can be capped upstream, while the reservoir may accept less because of the stage injection decision or a reservoir restriction.

If brine extraction is a research objective, add it as a distinct action (well site and rate) with a brine-handling cost and a corresponding solver source/sink. It cannot be approximated as negative CO₂ injection.

## One-stage transition

Given `S_t` and the joint action `a_t`:

1. **Validate nominations.** Enforce nonnegative requests, source availability, and each operator’s local technical bounds.
2. **Dispatch pipeline delivery.** Apply the specified network/capacity rule to obtain delivered CO₂ `D_t^i`. Log requests, allocations, unmet requests, active capacity constraints, and transport costs. In v1, use an explicit rule such as proportional curtailment, fixed shares, or a price-clearing dispatch. Never hide dispatch inside the reservoir solver.
3. **Map delivery to well controls.** Allocate delivered CO₂ across the operator’s injection wells using the declared action/rule. Enforce per-well rate and cumulative storage limits.
4. **Advance the CO₂–brine reservoir.** Call a CO₂-specific transition adapter with geology `(Kx,Kz,phi)`, current state, wells/rates, duration, and physics config. It returns next pressure and phase/component states plus numerical diagnostics.
5. **Update physical ledgers.** Record injected mass, mobile/trapped/dissolved mass by operator, brine displacement, plume overlap, boundary contact, pressure maxima, permit use, and any modeled leakage/production. Check mass balance; do not silently clip away component mass.
6. **Generate observations.** Sample measurements from the explicit observation model and update each operator’s belief. Public information must be identical across players; private measurements update only the receiving operator.
7. **Accrue payoffs.** Apply the payoff ledger below. Apply price payments as transfers between players or to a regulator account; exclude pure transfers from total resource welfare.
8. **Advance stage.** Increment `t`; apply declared exogenous availability/outage transitions.

## Constraints and policy mechanisms

Keep the following limits distinct because they represent different bottlenecks:

1. **Source availability:** captured/supplied CO₂ available for delivery.
2. **Pipeline throughput:** rate or stage mass deliverable through each network edge.
3. **Reservoir injection limit:** well/formation pressure or operational rate limit.
4. **Shared storage permit:** cumulative mass or pore-space allowance assigned to the project.
5. **Containment/plume constraints:** pressure, caprock/boundary contact, or plume interaction threshold.

Candidate rules should include a baseline unallocated shared pool, proportional curtailment, CEA, CEL, fixed quota, and a pipeline price/auction. Specify where each rule acts: a pipeline allocation acts on deliveries; a storage permit acts on cumulative reservoir injection. A quota at one layer does not automatically satisfy the other layer’s constraint.

## Payoff and welfare ledger

The existing `score_co2_stage` is a scaffold, not a calibrated objective. Define stage `t` operator payoff as:

`u_i,t = V_i,storage + R_i,service - C_i,pipeline - C_i,compression - C_i,drill - D_i,pressure - D_i,plume - D_i,containment + T_i`.

Each term must be computed and logged separately:

- `V_storage`: value/credit for CO₂ retained at a declared evaluation horizon; count each CO₂ component mass once and attribute it via a tracer/ledger;
- `R_service`: injection or storage-service revenue, with a clearly stated payer and recipient;
- `C_pipeline`: transport resource cost by edge, distance, throughput, or supplied pipeline model;
- `C_compression`: compression/booster energy and cost if represented;
- `C_drill`: well construction/operation cost;
- `D_pressure`: shared pressure harm, with the private allocation rule declared;
- `D_plume`: overlap or interference cost, only if supported by physical diagnostics and a declared coefficient;
- `D_containment`: boundary/contact/leakage-risk cost, only if the model computes the corresponding diagnostic;
- `T_i`: transfers such as permit prices, pipeline tariffs, or taxes. They affect private incentives but cancel from project-wide resource welfare when both payer and recipient are included.

Project welfare is the discounted sum of resource benefits and costs before internal transfers. Keep private payoffs and project welfare side by side. Report all value coefficients as **scenario weights in model units** until calibrated to currency, mass, time, and a storage-service contract. The current defaults in `CO2PayoffConfig` are placeholders; do not treat `injection_margin=20`, `stored_mass_credit=1`, or damage weights as empirically estimated values.

Do not reward injection volume alone as “storage”: injected CO₂ may remain mobile, dissolve, become residually trapped, or leave the modeled domain. State the accounting horizon and what retained means. The payoff scaffold’s current retained-mass argument must be supplied from a validated CO₂ component/tracer ledger.

## Policy benchmarks and solution methods

- **Nash-like benchmark:** iterated best response using each operator’s private payoff, belief, pipeline dispatch mechanism, and finite look-ahead. Save initial action profiles, rounds, residuals, cycles, and best-response deviations.
- **Coordinated benchmark:** joint policy maximizing total resource welfare with the same state, action bounds, forecast model, and pipeline/reservoir transition. Define how its gains are distributed among operators; joint welfare alone does not imply every operator benefits.
- **Solver-reference planning:** all candidate actions are valued with the research simulator.
- **Surrogate planning, solver environment:** surrogate forecasts candidate trajectories, but the research simulator advances the realized state and supplies scored outcomes. This is the primary surrogate decision-quality test.
- **All-surrogate rollout:** optional diagnostic only; clearly label model-dependent realized outcomes.

Use a stronger deviation search after the core policy comparison. A finite search returns a best-found deviation lower bound, not an equilibrium proof. Measure both realized solver welfare regret of surrogate-selected actions and physical-constraint violations.

## Experimental sequence

### Phase 0 — collect the pipeline interface

Before coding the pipeline transition, record from the user’s pipeline model:

- topology and nodes/edges; source and injection-site mapping;
- whether inputs/outputs are mass rate, volumetric rate, or mass per interval;
- pressure and temperature units, inlet/outlet bounds, and whether CO₂ phase/composition is constrained;
- capacity by edge and time, linepack/inventory behavior, and how outages/maintenance are represented;
- compression/booster energy and transport cost equations;
- how deliveries are assigned to operators and injection sites;
- which decisions are controlled by operators versus a pipeline owner/regulator.

Until this interface is known, use a documented synthetic stage-capacity block for game design only; do not label it a simulated pipeline.

### Phase 1 — deterministic plumbing pilot

Use two operators, one common source, one shared-capacity bottleneck, a fixed stage delivery cap, no outage, and the current vertical x–z reservoir solver. Start with a fixed placement set for a controlled baseline, then vary placement and well count as a separate treatment. Fix geology, action grid, and stage duration within each paired comparison. Check the conservation identity:

`source mass = delivered mass + pipeline loss (if modeled) + undelivered/curtailed mass`;

and separately:

`initial reservoir CO₂ + injected CO₂ = final mobile + trapped + dissolved CO₂ + produced/leaked CO₂ + numerical residual`.

With no pipeline loss and no production/leakage, delivered mass must equal injected mass or remain explicitly in a modeled pipeline inventory. This pilot establishes interface consistency, not a game-theoretic result.

### Phase 2 — core paired game

On paired geologies and seeds, compare Nash-like and coordinated policies under:

1. unconstrained pipeline / unconstrained storage reference;
2. pipeline throughput cap only;
3. reservoir storage cap only;
4. both constraints together.

Report the paired welfare gap, each operator payoff, deliveries, injection, bottleneck usage, pressure, retained-component masses, plume overlap/contact, wells, and equilibrium diagnostics. This factorial contrast identifies whether the commons is upstream, underground, or coupled.

### Phase 3 — institutions and heterogeneity

Compare proportional/CEA/CEL/fixed allocation separately at the pipeline and storage layers, then selected combinations. Vary operator margins and pipeline access/cost position. Keep private distributional outcomes alongside total welfare; a mechanism can raise welfare while disadvantaging one operator.

### Phase 4 — CO₂ surrogate

First validate one-step and six-stage rollouts by held-out geology. Then use a surrogate planner with solver environment on the same held-out geologies as solver planning. Train/test splits must be by geology seed and, for SPE10, by spatial blocks/slices. Do not transfer the 2D brine GNN+prior2 checkpoint as a validated CO₂ model; the current CO₂ trainer is separate.

### Phase 5 — dynamic pipeline and 3D extension

Only after the stage-capacity game works should the model incorporate hydraulic/linepack state, compressors, outages, or branching networks. Then extend the reservoir from the current 2D vertical x–z pilot to coarse 3D and check whether policy findings survive. These are separate increases in model fidelity and compute cost.

## Required output record

Every stage record should include:

- study/experiment/arm, code version, solver version, fluid and grid, stage duration, units and calibration status;
- geology source, seed, cohort, permeability multiplier, spatial split, and operator mapping;
- initial and final pipeline state; nominations, routed flow, delivered/curtailed mass, active edge constraints, outage, pressure/linepack if modeled, and pipeline cost;
- well actions, completions, requested and realized rates, permit balances, and injection by operator;
- reservoir state/diagnostics: pressure, mobile/trapped/dissolved CO₂, brine saturation, mass-balance residual, plume metrics, boundary contact, solver steps/runtime;
- observation values, noise seed, each operator’s belief summary, chosen actions, best-response convergence;
- payoff components by operator, transfers, project welfare, and the exact policy-model/environment-model labels;
- if a surrogate is used: checkpoint hash, architecture, training data/split, seed, device, prediction errors, action regret, runtime, and violation metrics.

Write one atomic result per geology/arm and preserve complete trajectories for training and audit. Never encode missing jobs as zeros. Compare treatments only on the same completed geology set.

## Measures and interpretation

Primary endpoint: paired total-resource-welfare difference `W_coordinated - W_Nash-like` at the same geology and exogenous seeds. Report mean, median, geology-bootstrap 95% interval, positive share, paired `n`, and the per-geology distribution.

Secondary endpoints:

- operator-specific payoffs and payoff inequality;
- pipeline throughput utilization, curtailment, bottleneck frequency, and resource/transfer cost;
- injected, mobile, trapped, dissolved, and retained CO₂ masses;
- brine displacement/production if implemented;
- pressure maximum, pressure at rival wells, and time/volume above declared constraints;
- plume footprint, overlap, and boundary/caprock contact;
- drilling, rate schedules, and storage-cap utilization;
- Nash residual/deviation lower bound and convergence/cycling;
- solver and surrogate runtime, surrogate action regret, and physical-violation rate.

Keep three uncertainty sources separate: (1) physical model and discretization error, (2) game solution/search error, and (3) surrogate prediction/planning error. Keep transport/pipeline uncertainty as an additional subsystem uncertainty when the pipeline is modeled.

## Implementation map in this repository

- Water-game baseline: `game3.py`, `run_study7.py`, and `README_STUDY7.md`.
- CO₂ reservoir pilot: `co2_xz_solver.py`, `test_co2_xz_solver.py`, `generate_co2_xz_pilot.py`, `generate_co2_xz_dataset.py`.
- CO₂ payoff/geology scaffold: `co2_experiment.py`; it is not wired into `game3.py`.
- Initial stage transition kernel: `co2_pipeline_game.py`. It implements a single shared stage-capacity abstraction, explicit dispatch (proportional/CEA/CEL/fixed equal), storage-cap curtailment, CO₂ solver calls, operator well ledgers, and separated pipeline resource costs/tariff transfers.
- Well-placement game adapter: `co2_well_placement_game.py`. It adds stable well IDs, variable well counts, candidate-site, spacing, per-operator well-count, per-stage drilling, per-well rate, and commissioning-delay constraints. It supports injector and monitor well records; monitor observations require an explicit callback. It is still a transition environment, not a policy planner, equilibrium solver, well-trajectory model, or pipeline-hydraulics model.
- CO₂ surrogate workflow: `train_co2_xz_surrogate.py` and `CO2_STAGED_RUN.md`; it is separate from the brine trainer.
- Keep CO₂ policy/planner integration in CO₂-specific modules rather than changing the water game’s state keys or reward calculation.

## Explicit unresolved decisions

These cannot be responsibly fixed from the current repository alone:

1. Whether operators own pipeline capacity, purchase it from a common carrier, or face a regulator-allocated shared limit.
3. Whether brine is only displaced in place or can be produced, treated, and sold/disposed.
4. The calibrated value of stored CO₂, injection service, transport, compression, pressure damage, plume overlap, and containment risk.
5. Whether source supply, outages, monitoring, leakage, or post-injection retention are in the first game scope.
6. The target stage duration and whether a six-stage game represents operations, planning periods, or another timescale.

These should become explicit scenario configurations, not hidden constants. The game can be coded modularly once the pipeline interface and payoff units are supplied.
