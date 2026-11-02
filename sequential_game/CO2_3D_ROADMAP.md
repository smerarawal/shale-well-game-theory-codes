# CO2 surrogate path from 2D x-z to 3D

## Current evidence

The present dataset and solver are 2D x-z slices coarsened to 24 x 24 cells.
In the uploaded three-seed comparison, CCS-GNN has the strongest balanced
neural-only result at the sixth rollout stage: mean pressure RMSE about 244 kPa,
mean absolute pmax error about 649 kPa, plume IoU about 0.89 at mobile gas
saturation 0.01, and p95 predicted stage mass-balance error about 137 kg. CCS
has lower mass-balance error (about 112 kg), while plain GNN is competitive on
pressure. These averages hide substantially harder cases in the challenging
synthetic cohort. The results establish a useful 2D baseline; they do not
validate a 3D model or establish game-payoff accuracy.

## Recommended 3D phases

### Phase 1: specify the physical model and geometry

- Define the target grid, cell dimensions, well locations/completions, pressure
  reference, boundary conditions, and whether the game uses leases in separate
  horizontal regions.
- Keep x and y horizontal and z vertical. Preserve anisotropic Kx, Ky, and Kz;
  do not infer 3D behavior by extruding a 2D x-z slice.
- Decide a coarse SPE10 representation before data generation. The native
  60 x 220 x 85 grid is roughly 1.12 million cells, so begin with a cropped and
  coarsened 3D test block and document the upscaling for each permeability
  direction.

### Phase 2: extend and validate the solver

- Implement a 3D finite-volume pressure solve with x, y, and z face
  transmissibilities, anisotropic permeability, well source terms, and the
  chosen boundary conditions.
- Include vertical gravity consistently in phase fluxes. Carry the CO2 state
  variables needed for Markov rollouts: pressure, mobile gas, trapped gas,
  historical maximum gas saturation, and dissolved CO2.
- Verify closed-domain mass conservation, hydrostatic equilibrium, buoyant rise
  in a homogeneous column, a one-dimensional limiting case, well/source
  accounting, and grid/time-step refinement. Compare a 3D model with a
  one-cell-wide y slice against the existing 2D solver where assumptions match.
- Have a reservoir-physics reviewer validate the flux, trapping, dissolution,
  and boundary formulations before generating the large training set.

### Phase 3: build a small 3D trajectory dataset

- Start with homogeneous and layered synthetic blocks, then connected and
  disconnected channel cases, then cropped Tarbert and Upper Ness blocks.
- Split by independent geology realization, not by transition rows, to prevent
  the same rock field leaking across train and test.
- Track solver mass residuals, phase saturation bounds, buoyant segregation,
  plume extent, and pressure peaks for each sample before training.

### Phase 4: train and compare 3D surrogates

- Begin with 3D CCS/U-Net and a 3D permeability-aware GNN; compare CCS-GNN only
  after both components have reliable 3D baselines.
- Train one-step and full six-stage rollout variants, preserve the CO2 pressure
  prior as a separate ablation, and evaluate pressure, pmax, mobile/trapped gas,
  dissolved CO2, plume overlap at multiple cutoffs, mass balance, and inference
  time against the 3D solver.
- Re-select the architecture in 3D. The 2D winner may not remain the winner
  because lateral connectivity and 3D well interference add new structure.

## Immediate 2D experiment before the 3D port

Drop GNN-LSTM. Compare CCS and CCS-GNN with three seeds using six-stage
pushforward training, first without a plume loss and then with a modest soft-Dice
loss. Report threshold sensitivity at mobile-gas cutoffs 0.005, 0.01, and 0.02,
plus precision/recall, plume-area error, pressure/pmax, mass balance, and
per-geology results. Tune the plume-loss weight on validation data and reserve
the test split for final reporting.
