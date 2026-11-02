# CO2 staged surrogate run

Use `notebooks/co2_staged_surrogate_run.ipynb` in Kaggle with a GPU accelerator.
Attach the saved `co2_xz_dataset_v2` output and the prior multiseed surrogate
output containing `co2_gnn_s{0,1,2}.pt` and `co2_ccs_s{0,1,2}.pt`.

The workflow has four phases:

1. Select a fixed pressure-prior substep count from validation-only cases. The
   `path` diagnostic uses labelled endpoint saturation as an oracle and is
   reported only as a diagnostic upper bound.
2. Train pressure residual networks for prior-lite mobility features, frozen
   stage-start mobility, and a saturation path supplied by the frozen base
   surrogate. The last mode is the CO2 prior2 analogue; it is not the brine
   pressure prior.
3. Train matched one-step and three-stage pushforward models with optional soft
   mass-balance and raw-state validity losses.
4. Train CCSNet/GNN ConvLSTM variants and a dual CCSNet/GNN spatial branch using
   the same three-stage training and physical penalties.

The pressure prior calls the CO2 solver's permeability, mobility, transmissibility,
gravity, and pressure assembly helpers. It holds pressure substeps fixed and does
not model the stage's adaptive transport trajectory. The endpoint-path prior
linearly interpolates predicted mobile and trapped saturation. Both are
approximations to be compared with solver labels. The surrogate result is only
accuracy relative to the exploratory 2D solver; it is not field validation.

`train_co2_xz_surrogate.py` keeps the original defaults when both new loss weights
are zero and `--rollout-unroll 1` is selected. Use `--rollout-unroll 3` to train
consecutive stage windows, detaching predicted states between stages to limit
GPU memory. Use `--mass-loss-weight` and `--valid-loss-weight` for soft penalties;
all prediction metrics continue to be computed after the inference projection.
