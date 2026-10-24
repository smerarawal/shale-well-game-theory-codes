# Consolidated surrogate experiment

`run_consolidated_sweep.sh` evaluates the original six U-Net cells (`none`, `prior`, `prior2` × `game`, `mix`) alongside the same hybrid/data matrix for the architecture baselines and new coupled models. The default is one seed, 10 architectures × 3 hybrid modes × 2 training-data conditions (60 runs), dispatched two at a time to Kaggle's T4x2 GPUs. Set `BACKBONES`, `HYBRIDS`, `DATASETS`, `SEEDS`, or `EPOCHS` to reduce or extend the sweep. The `mix` condition concatenates the game's training fields with SPE10 train-band patches. Tarbert and Upper Ness test-band patches remain held out for evaluation.

## Architecture choices

| Name | Design | Motivation |
|---|---|---|
| `unet` | Shared U-Net trunk | Reproduce the original six-cell baseline. |
| `ccs` | Separate pressure and saturation U-Net decoders | Test task-specific output heads. |
| `gnn` | Per-cell grid message passing with permeability edge features | Test local transport modeling. |
| `split` | U-Net pressure, GNN saturation | HGNS-style division of labor. |
| `runet` | U-Net with an intra-stage ConvLSTM bottleneck | Recurrent convolution baseline. |
| `gnnlstm` | GNN latent state with LSTM memory carried between stages | Recurrent-graph candidate; trained with six-stage unrolling. |
| `ccslstm` | CCSNet-style dual-decoder encoder with ConvLSTM hidden/cell state carried between stages | Tests whether spatial recurrence improves six-stage pressure and saturation rollouts. Unlike `runet`, memory persists across game stages. |
| `ccsfno` | CCSNet local encoder-decoder fused with an FNO feature branch at the bottleneck | Tests local plume detail plus long-range pressure context with separate pressure/saturation decoders. |
| `fno`, `ufno` | Fourier operator and U-Net/Fourier operator | Global operator baselines. |
| `ugnn` | Pressure U-Net → saturation GNN → pressure correction U-Net | New bidirectionally coupled CNN/graph candidate. |
| `fgnn` | Pressure FNO → saturation GNN → pressure correction FNO | New bidirectionally coupled operator/graph candidate. |

HGNS motivates separating global pressure prediction from graph-based flow prediction ([Wu et al., KDD 2022](https://arxiv.org/abs/2206.07680)). The U-FNO work motivates combining Fourier global context and CNN spatial detail ([Wen et al., 2022](https://arxiv.org/abs/2109.03697)). MGN-LSTM motivates recurrent graph state for long rollouts and reduced temporal error accumulation ([Ju et al., Computers & Geosciences, 2024](https://doi.org/10.1016/j.cageo.2024.105711)); the `gnnlstm` implementation adapts this idea to the current fixed 2D grid and six-stage game.

The comparison should focus on stage-6 p-max and penalty error, calibration bias, well-cell pressure, saturation front/plume metrics, and CPU time per full stage. Training logs, validation reports, the consolidated Markdown table, and CPU timing report are written under `OUT`.

## Focused CCS follow-up

After the main sweep, run `run_ccs_extensions.sh`. It compares a retrained CCSNet control against `ccslstm` and `ccsfno`, using mixed game + SPE10 training data, `prior2`, six-stage unrolling, and seeds 0, 1, and 2 by default. The control is retrained in a separate output folder so its training unroll matches the new variants. Existing generated datasets are copied from `BASE_OUT`; model checkpoints and metrics remain in `OUT`. To reduce runtime, override `SEEDS`, `EPOCHS`, `GPU_IDS`, or `MAX_PARALLEL`.

Kaggle example:

```bash
REPO=/kaggle/working/surrogate_repo/sequential_game \
SPE=/kaggle/input/datasets/<owner>/spe10model2 \
BASE_OUT=/kaggle/working/consolidated_final \
OUT=/kaggle/working/ccs_extensions \
bash /kaggle/working/surrogate_repo/sequential_game/run_ccs_extensions.sh
```
