# Study v7: confirm the commons, then extend to plume physics

## Decision

Do **not** run a broad v6 rerun plus CO2 plume cases in the same study. First establish whether the water-injection results survive a larger, paired geology sample and the stronger deviation search. Add the plume as a separate phase after the solver passes buoyancy and mass-balance checks. Keep the old v5/v6 outputs as historical/pilot results; label any pooled analysis separately.

The current game transition calls a NumPy solver, so a Kaggle GPU is not expected to accelerate the reservoir step unless the solver is moved to a GPU backend. Benchmark CPU and GPU setup only if a CUDA implementation exists; otherwise spend the four-hour window on parallel CPU workers and measured preflight.

## Main design choices

- Primary sampling unit: **geology**, not a rerun of a solver seed. Use 36 new geologies, IDs 24–59, with unique field seeds and a balanced permeability-multiplier cycle spanning the existing 0.5–2.0 range. Keep the original geologies 0–23 as the historical sample.
- Pair every treatment on the same geology. Keep model version, initial state, action grid, stage time, and observation/noise seeds fixed across paired arms. Store a manifest of these values.
- Target n=36 new paired geologies. If the runtime preflight cannot fit the full design, stop at a complete paired prefix of at least n=24; report achieved n and do not compare arms with different geology sets.
- Use H=3 for the primary comparison to match v6. Retain H=6 as a planner-horizon audit on a smaller paired subset; do not mix H=3 and H=6 welfare levels.
- Pre-register the primary endpoint as the paired welfare gap, Coop − Nash. Report the mean, median, paired 95% bootstrap interval, fraction positive, and number of geologies. Also report each operator's payoff and physical outcomes.

## Run order

### Tier 0 — correctness and runtime gate (4–6 geologies)

Run a short paired smoke sample before filling the time budget. Check that every arm uses identical geology IDs, the shared permit is enforced, fixed allocations cannot exceed each operator's share, welfare recomputes from payoff components, and output files resume without duplicates. Measure seconds per complete paired geology. Estimate the number of geologies that fit, then set the run prefix. Do not count this small preflight as confirmatory evidence.

### Experiment 18 — confirm the core permit gap

**Question:** Does the H=3 water-injection gap persist on new geologies, and is it caused by the shared permit?

**Arms:** close-well layout, cap 24 vs no shared cap; Nash vs cooperative planner. Run all four arms on each new geology (IDs 24–59), H=3, same model settings.

**Primary readout:** paired Coop − Nash gap for each cap, plus the paired difference-in-gaps: (gap at cap 24) − (gap without cap). Also report injected volume, wells drilled, peak pressure, total drilling cost, and the operators' separate payoffs.

**Planner check:** repeat Nash and Coop at H=6 on a preselected subset of 12 geologies. This checks the v5 horizon effect without making every run expensive.

### Experiment 19 — resolve the non-monotone scarcity curve

**Question:** Is the v6 cap pattern real or a small-sample/discrete-action artifact?

**Arms:** caps 12, 16, 20, 24, 32, 48; Nash and Coop, close layout, H=3; paired on the same new geologies. Use the same n=36 if Tier 0 timing allows, otherwise n=24.

**Readouts:** paired welfare gap and interval at each cap, fraction of geologies with positive gap, whether the cap actually binds, injection utilization, well count, and chosen rate schedule. Do not infer “more scarcity causes a larger gap” from a single cap; look for a repeatable shape across geologies.

### Experiment 20 — identify why rationing rules matter

**Question:** Does fixed allocation help by preventing early capture of the pool, by discouraging extra drilling, or both?

**Arms:** proportional, CEA, CEL, fixed equal allocation at caps 16 and 24; Nash only for the rule comparison, plus base Coop as a welfare reference. Keep the paired geology set fixed.

**Required mechanism logs:** per-stage requested and served injection by operator, cumulative permit use, new wells, rates, remaining permit, and operator payoff. This distinguishes a timing race from claims inflation more directly than end-of-game welfare alone.

### Experiment 21 — price versus quantity with unequal operators

**Question:** Does a price allocate capacity better than an equal quota when margins differ, and who receives the gain?

**Arms:** margins (20,20), (24,16), (28,12); price 0/5/10 and fixed equal allocation at cap 24. Use paired geologies. Include price revenue as a transfer in operator payoffs, but keep it out of total resource welfare to match the earlier convention; report both gross welfare and transfers explicitly.

**Readouts:** total welfare, each operator's payoff, injection by operator, permit utilization, and policy gain relative to the same-geometry Nash baseline. Treat price 10 as a candidate calibration, not a universally optimal price.

### Experiment 22 — stronger deviation search on the expanded sample

**Question:** Is the Nash gap still several times larger than profitable deviations when more geologies are tested?

Run the strong two-stage deviation search on the H=3 cap-24 Nash path for all geologies that fit. Keep the plain search as a paired diagnostic. For the cooperative path, report the joint-schedule improvement separately from unilateral Nash epsilon.

Epsilon is a **lower bound** on the best possible deviation because the search does not enumerate every schedule. Report its search space, number of simulations, and stopping rule with the estimate. Do not call it a proof of equilibrium.

### Experiment 23 — targeted uncertainty check (only after core tiers)

**Question:** Does monitoring still help, and does the welfare gap remain when players learn the geology?

Repeat only the prior vs repaired-belief Nash, learned Nash, and learned Coop arms on a paired subset of at least 24 geologies if the measured runtime permits. Report the sign by geology as well as the mean and interval. Keep the H=3 perfect-information comparison on exactly the same geologies.

Exp14's one-at-a-time sensitivity can follow using the new baseline; Exp16 coalitions are lower priority until the two-player results are confirmed. Do not spend the first run on coalition variants.

## Four-hour priority if runtime is tight

1. Tier 0 preflight.
2. Exp 18 at cap 24 vs no cap, H=3, paired Nash/Coop.
3. Exp 19 cap sweep at H=3.
4. Exp 22 strong epsilon at cap 24.
5. Exp 20 rationing-mechanism logs at caps 16 and 24.
6. Exp 21 heterogeneity/prices, then Exp 23 learning if time remains.

Save each completed geology-arm as an atomic JSON file and checkpoint the exact manifest. Resume only missing jobs. Do not start a tier whose paired sample cannot be completed within the remaining time.

## CO2/plume experiment — separate, gated study

The current sequential-game stage adapter injects the wetting water phase. The solver's gravity notes flag that saturation transport is upwinded using total-flow direction, which cannot represent counter-current buoyant flow reliably. A lighter injected phase is not made physically valid by simply reversing a sign.

Before a CO2 game run, implement and pass these solver tests:

1. A no-well segregation test: a light plume rises and brine moves in the opposing direction under gravity; reversing gravity reverses segregation.
2. A no-gravity control: the plume does not drift vertically from a gravity term.
3. Phase mass balance and saturation bounds over many steps; no silent clipping-based mass loss.
4. Injection/production source tests for each phase and stable pressure response.
5. Grid-refinement and timestep checks showing the plume direction and mass balance persist.

After those pass, define a new, clearly labeled CO2 experiment: a plume-based shared-capacity externality with plume footprint/overlap and pressure as separate outcomes. Keep water results as the baseline and report CO2 separately; do not pool their welfare levels. Exp IDs 24+ should be assigned only after the physics tests are accepted.

## Preserve data for the surrogate stage

The current stage adapter isolates the state transition behind the advance(geo, state, wells) interface. Keep the game solver authoritative for the next experiment, but save transition data so a surrogate can later replace that call and be checked on the same trajectories.

For every stage, save:

- study, experiment, arm, geology ID, field seed, permeability multiplier, model/code version, fluid, gravity settings, configuration, and random seeds;
- input permeability and porosity, pressure and saturation state;
- well locations, operator ownership, rates, and action choices;
- next pressure and saturation state, elapsed solver time, and mass-balance diagnostics;
- stage rewards split into revenue, drilling cost, liability, price transfer, and total welfare.

Keep complete trajectories, not only final summaries. Split surrogate training and validation by **geology seed** so adjacent states from the same geology cannot leak into both sets. First compare one-step transition errors on held-out geologies; then compare surrogate-driven game policy outcomes with the real solver on the exact same held-out geologies.

## Stop/report rules

- Report paired sample size and confidence interval; show old pilot geologies separately from new confirmatory geologies.
- Mark incomplete tiers as incomplete. Do not fill missing outcomes with zero.
- Report intervals that include zero as uncertain, and do not overread n=2–3 intervals.
- Keep physical validation, game-theoretic approximation error, and surrogate prediction error as three separate sources of uncertainty.

## Running the implementation

`run_study7.py` uses the existing water solver. A GPU is not expected to speed up this NumPy implementation. From the repository root in Kaggle:

```bash
cd sequential_game
python run_study7.py --hours 3.7 --out study7_results --tiers A,B,C,D --procs 4 --trajectories
python analyze7.py --dir study7_results
```

If Kaggle's CPU allocation differs, first measure runtime with `--hours 0.15 --tiers A --procs 2`. The runner resumes by skipping completed JSON jobs and runs tiers in priority order. An interrupted tier can be incomplete; `analyze7.py` reports paired sample counts and does not fill missing outcomes with zero. `--trajectories` writes compressed `.npz` true-state transitions for episode jobs. Keep those files with the JSON results for later surrogate training.

Tier A is the core H=3 cap-24 versus uncapped comparison plus the H=6 subset. Add B for the capacity sweep, C for rationing, and D for deviation and learning checks. Already-started jobs finish after the deadline; no new job starts once the time budget expires.
