# CO₂ Solver Replay Audit and Architecture Expert Brief

**Prepared:** 2026-11-10  
**Scope:** CO₂ work only. Keep all brine data, notebooks, code, and conclusions separate and unchanged.

## What I need from the architecture expert

Please review the current CO₂ x-z surrogate/planner/solver setup and recommend a model and training plan that improves held-out pressure and plume predictions, especially under heterogeneous geology. Before proposing a training run, complete the no-training checks in “First diagnostics” below. In particular:

1. Reconcile the code's actual model names and implementations with these candidates: graph-only GNN, CCSNet/CNN, CCS-GNN hybrid, and compact U-Net. The current checkpoint labels are only `ccs` and `ccs_gnn`; do not assume these map one-to-one to the expert's names without inspecting `make_model` and checkpoint metadata.
2. Assess Fourier-MIONet as the CO₂-specific operator candidate suggested by the user’s research screenshot; say whether a compact stagewise adaptation is worthwhile at this grid/data size. Treat U-FNO as a further reference/ablation, not a presumed winner. Consider recurrent R-U-Net only if it fits the six-stage transition task better.
3. Inspect the solver’s trapped-CO₂ and top-boundary accounting and determine whether the all-zero outputs below are expected or a reporting/implementation issue.
4. Explain why both learned planners produce almost identical schedules/outcomes and whether the current payoff and capacity setup can distinguish surrogate quality.
5. Propose a fair training/evaluation matrix, including architecture-compatible planner interface changes, if needed.

Please separate literature-backed expectations from hypotheses about this particular simplified solver. Do not claim an architecture winner from papers run on different simulators or geometries.

## Current assets and run identity

The attached Kaggle output reports a seed-0 paired planner/solver replay on the 50 test geologies (10 per cohort), once for each of `ccs` and `ccs_gnn`, for 100 rows total.

- Dataset: `/kaggle/input/notebooks/smerarawal/transition-geology/co2_xz_dataset_v2`
- Planner/trainer/search code: `/kaggle/input/datasets/smerarawal/co2-research-bundle/co2_research_run_bundle/sequential_game`
- Solver dependencies: `/kaggle/input/notebooks/smerarawal/scenario-json-creation/shale-well-game-theory-codes/sequential_game`
- Checkpoints used: `/kaggle/input/notebooks/smerarawal/scenario-json-creation/co2_surrogate_refined/co2_ccs_s0.pt` and `co2_ccs_gnn_s0.pt`
- Scenario: `co2_surrogate_scenario_exploratory.json`; six 30-day stages, 24×24 grid, 5 MPa pressure cap, exploratory and not field calibrated.
- Attached result files: `summary_by_architecture.txt` and `solver_replayed_game_results.txt`.

The CSVs identify input paths but do not contain a dataset hash, code hash, or training-data hash for the checkpoints. Verify whether the checkpoints were trained from this exact dataset version and split before treating the model comparison as controlled.

## Findings verified from the attached result rows

| Finding | CCS | CCS-GNN |
|---|---:|---:|
| Solver replay rows | 50 | 50 |
| Mean project welfare | 1510.07485 | 1510.07568 |
| Median project welfare | 1516.32 | 1516.32 |
| Mean cap exceedance over cases | 0.99275 MPa | 0.99273 MPa |
| Cases with any stage above 5 MPa | 6/50 (12%) | 6/50 (12%) |
| Mean of each replay's max absolute mass residual | 1.2102×10⁻¹² kg | 1.1994×10⁻¹² kg |
| Maximum of replay max-absolute mass residual | 2.3874×10⁻¹² kg | 2.3874×10⁻¹² kg |
| Median solver replay time | 0.545 s | 0.542 s |

The six violating geologies are identical for both backbones:

| Cohort | Geology seed | Max pressure | Exceedance | Stages over cap |
|---|---:|---:|---:|---:|
| friendly | 41040 | 5.4034 MPa | 0.4034 MPa | 2/6 |
| challenging | 241026 | 11.2263 MPa | 6.2263 MPa | 5/6 |
| challenging | 241041 | 32.9069 MPa | 27.9069 MPa | 6/6 |
| challenging | 241042 | 13.4844 MPa | 8.4844 MPa | 6/6 |
| challenging | 241048 | 5.0276 MPa | 0.0276 MPa | 1/6 |
| challenging | 241049 | 11.5890 MPa | 6.5890 MPa | 6/6 |

Five of ten challenging cases breach the cap; one of ten friendly cases does; none of the other three cohorts do. The worst case is geology 241041, where all six stages exceed the cap.

The mean cap exceedance is dominated by the worst case. For CCS, six positive exceedances sum to 49.6375 MPa; geology 241041 contributes 27.9069 MPa (56.2%). Median exceedance is 0 MPa; linear-interpolated 90th, 95th, and 99th percentiles are approximately 0.065, 6.426, and 18.390 MPa. CCS-GNN figures are nearly identical. Therefore show breach count, median, quantiles, and maximum; do not use the mean alone. The challenging cohort's 5/10 breach fraction has a broad approximate Wilson 95% interval of 23.7%–76.3%, assuming independent binomial sampling; treat it as imprecise, not as a stable population rate.

Matched outcome differences are tiny: maximum absolute paired peak-pressure difference is 5.34×10⁻⁵ MPa (about 53 Pa); mean absolute paired peak-pressure difference is 2.08×10⁻⁵ MPa. Mean absolute welfare difference is 0.00083, maximum 0.03762. This shows near-identical **replayed outcomes**, not necessarily identical predicted fields. The attached CSV does not contain the stage/operator rate schedules or direct field-prediction deltas; inspect the per-case JSON outputs to compare those.

The replayed outcomes cannot discriminate the two surrogates. Both models breach on the same six cases, pointing to a shared planner/payoff/data cause or to the planner choosing effectively identical actions. The model's predictions may differ even when its resulting schedule does not. Check `rates_m3_s_by_stage_operator` and the stagewise predicted/solver pressure before attributing the breaches to architecture.

The configured pipeline capacity is 252.72 kg/stage for six stages, or 1516.32 kg total. Pipeline resource cost and tariff are zero, and storage value is 1 normalized unit/kg. Most cases achieve welfare 1516.32 exactly. This creates a broad welfare plateau, so welfare alone is a weak architecture discriminator; pressure safety and field prediction metrics must be reported separately.

Across the 100 rows, final mobile + trapped + dissolved CO₂ sums to 1516.32 kg (within floating-point tolerance). Reported solver mass residuals are around 10⁻¹² kg. These residuals are **solver replay** diagnostics, not surrogate mass-prediction errors; the same is true of the solver runtime column. They cannot be used to rank model mass accuracy or inference speed. Measure those directly from the predicted states and model forward pass.

`final_trapped_co2_kg` is zero in all 100 final rows; `max_top_boundary_mobile_gas_fraction` is zero in all rows. The outputs establish those facts but do not establish their cause. If `sg_trapped` is zero across the complete training dataset, remove trapped-field error from model ranking and verify whether `sg_max` adds any dynamic information. Because the scenario has closed outer boundaries and the modeled vertical rise may be limited over 180 days, zero top-boundary contact is plausible, but check the metric definition and vertical orientation. Neither zero should be labeled a code bug without a source audit.

Most cases reach welfare 1516.32, exactly 6 × 252.72 kg/stage × 1 normalized unit/kg. The objective is therefore at a capacity plateau for most cases. The configured pressure cap/damage is not a calibrated safety constraint; a soft penalty can rationally permit violations. Keep this welfare and the 5 MPa cap out of claims about field performance and architecture ranking.

## Issues to verify in source and case JSONs

These are audit questions, not confirmed bugs. Complete them before training:

- Trace `sg_trapped` updates through the x-z solver. Does `evaluate_schedule` aggregate trapped mass using cell volume, porosity, CO₂ density, and the same saturation convention as the dataset? Is trapped mass included in total mass accounting exactly once?
- Is residual trapping expected during six consecutive injection stages with this model's simplified trapping law? Check all six stage records, not only the final CSV column.
- How is `max_top_boundary_mobile_gas_fraction` computed? Confirm row/column orientation, active/inactive cells, threshold, and whether closed boundaries imply this value should remain zero.
- Compare `rates_m3_s_by_stage_operator` in matched case JSONs. Are CCS and CCS-GNN actually selecting the same schedule? If yes, find whether this follows from capacity saturation, flat payoffs, planner projection/clipping, deterministic rate search, or model outputs.
- Confirm the pressure cap is a soft damage term rather than a hard feasibility constraint. The observed breaches show that the current planning pipeline does not guarantee the 5 MPa limit.
- Inspect pressure evolution by stage for all six breaching cases and compare predicted pressure maxima against solver maxima. The CSV has only solver replay maxima; raw prediction metrics are required to identify whether the error is in prediction or decision optimization.
- Check whether zero plume-boundary contact follows from the closed boundary setup; do not interpret it as evidence of field containment.
- Confirm code/dataset/checkpoint provenance and exact checkpoint architecture metadata before comparing these rows with training metrics.

### First diagnostics (no training)

1. Load each of the six breach case JSONs. Compare stagewise `planner` predicted peak pressure with the solver's stagewise peak pressure, then compare the two backbones' `rates_m3_s_by_stage_operator`. First determine whether the model predicted the exceedance and whether the plans actually differ.
2. Load `sg_trapped` across every split and cohort in the 7,488-transition dataset. Report min/max/nonzero count. Also report whether `sg_max` differs from the running maximum of `sg_mobile`; if not, it adds no new dynamic information for this dataset.
3. Compute the training-split and validation-split pressure-maximum ranges from the true labels, using the same pressure definition and units as `evaluate_schedule`. Compare those ranges with the six breach cases, especially 241041 at 32.91 MPa. Do not tune on the test split.
4. Verify checkpoint training-data/split hashes and model metadata. If provenance is absent, mark the current checkpoints historical and retrain the baselines in the controlled benchmark.
5. Verify whether the 5 MPa cap is enforced as a hard constraint or represented only by the soft `pressure_damage_per_pa2` payoff. If it is soft, the benchmark should separately report welfare-optimal and cap-feasible policies.

## Architecture and experiment recommendation

### First benchmark: controlled architecture comparison

Primary set, three seeds each: graph-only GNN, CCSNet/CNN, a CCS-GNN hybrid, and a compact U-Net. First reconcile these labels with the implementations and checkpoint metadata. The most directly relevant additional architecture is a compact, stagewise Fourier-MIONet adaptation: the paper is specifically about multiphase CO₂ geologic sequestration and combines multiple input branches with Fourier/U-FNO decoding ([paper](https://arxiv.org/abs/2303.04778), [journal version](https://doi.org/10.1016/j.ress.2024.110392)). Use one reference run first and expand to three seeds only if it is competitive. Its reported efficiency/generalization comes from a different geometry, grid, and solver; its codebase also uses a modified DeepXDE, so it is not a direct plug-in for the current PyTorch trainer ([official code and Apache-2.0 license](https://github.com/lu-group/fourier-mionet-gcs)). U-FNO can be a later ablation if useful; skip plain FNO unless needed to isolate the Fourier path. Recurrent R-U-Net is a later option if rollout error proves dominant; the current state is already formulated as a Markov transition ([study](https://sccs.stanford.edu/sites/g/files/sbiybj17761/files/media/file/surrogate-model-for-geological-co2-storage-and-its-use-in-hierarchical-mcmc-history-matching.pdf)).

User-reported results from separate brine/water work suggest GNN may help pressure/pmax, CCSNet plume/mass, and U-Net may be a strong low-cost control. These are hypotheses for the CO₂ benchmark only; do not mix their data, metrics, checkpoints, or conclusions into the CO₂ report.

### Fair protocol

1. Freeze the exact `co2_xz_dataset_v2` config, split manifest, scenario, solver files, and code version; hash them in a manifest. Keep all outputs in a CO₂-specific folder.
2. Verify existing checkpoints were trained on those exact data/split hashes. If that cannot be established, retrain the four primary candidates (graph-only GNN, CCSNet/CNN, CCS-GNN hybrid, compact U-Net) under one harness. If it can be established, reuse only the verified matching checkpoints and train the new candidates under matching data, losses, and selection rules. Add one Fourier-MIONet reference run if its stagewise adaptation fits the same input/output contract.
3. Use the existing geology-level train/validation/test split without moving samples between splits. Keep the test set out of model selection and hyperparameter tuning. Use seeds 0, 1, and 2 for each architecture.
4. Use a matched training budget and the same target fields, normalizations, loss components, and checkpoint-selection rule. Match parameter counts as closely as practical, especially for the graph-only GNN versus CCS-GNN hybrid, and report the actual counts. Allow a small, equal-budget validation-only hyperparameter search for each architecture if needed. Save training and validation curves and best checkpoints.
5. Report metrics by architecture × seed × cohort × stage, not just one overall scalar: pressure RMSE/MAE, well-cell and upper-decile pressure errors, peak-pressure bias/absolute error, mobile-plume MAE and IoU at the existing 0.01 threshold, dissolved field errors, predicted mass error, saturation bounds, rollout drift, model forward latency, parameter count, and memory. Include trapped-field metrics only if the dataset audit finds nonconstant `sg_trapped` labels. Solver mass residual and solver runtime belong in a separate solver-validation table.
6. For decision quality, use solver replay only after the prediction benchmark. Reuse the current CCS/CCS-GNN replay rows only if their checkpoint/data provenance matches. Replay the same held-out geologies for new architectures through the same planner and solver. If provenance does not match, rerun all architectures on matched cases. Keep normalized welfare and the exploratory 5 MPa cap in a separate scenario-diagnostic table; do not use them to rank architectures or make field-safety claims until payoffs and limits are calibrated.

### Likely higher-value changes after the baseline benchmark

The expert should consider the CO₂ pressure prior on the top two models, a high-pressure-weighted loss, training-only oversampling of high-pressure/challenging training cases, and unroll-3 training. Test these as separate ablations after the architecture baseline so their effects are identifiable. A solver-verified action shield or conservative pressure bound can then enforce a policy limit in the exploratory game. The current 5 MPa value is uncalibrated, so this is a logic/sensitivity experiment, not a field-safety claim. A lower prediction error alone does not guarantee policy feasibility.

## Suggested expert deliverables

1. A concise source audit explaining trapped-mass and boundary-metric behavior, with file/function references.
2. A recommended architecture set (including whether Fourier-MIONet is viable and whether U-FNO/plain FNO adds a useful ablation) and why it suits 24×24 x-z fields and the available sample count.
3. A fair Kaggle run matrix with compute estimate, training/evaluation outputs, and provenance fields.
4. A pressure-safety plan that keeps the architecture comparison separate from the policy shield/loss ablations.
5. Any code changes required to expose the chosen architectures through the existing trainer and differentiable planner interfaces.

## Important limits

This is an exploratory 2D x-z solver with simplified physics and normalized payoffs, not a field-calibrated CO₂ storage assessment. The architecture benchmark can establish comparative performance for this dataset and solver only. It must not be combined with the separate brine work.
