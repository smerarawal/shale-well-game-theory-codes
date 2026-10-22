# Final surrogate run on Kaggle

This run is a seeded confirmation of the strongest single-seed results, not another indiscriminate sweep.

## What the final run compares

`run_final_multiseed.sh` schedules 66 model fits across seeds 0, 1, and 2:

- `unet`, `ccs`, and `ccsgnn`: every hybrid (`none`, `prior`, `prior2`) on both `game` and `mix` training data.
- `gnn`, `split`, `ugnn`, and `fgnn`: `prior2` with `mix` training data, the strongest-supported setting from the preceding screen.

Every final fit uses the same 100-epoch budget, width 48, two-stage training unroll, and fixed loss weights. Validation reports cover game geologies and the held-out Tarbert and Upper Ness SPE10 test bands. The final markdown report shows means and standard deviations across seeds. The CPU benchmark uses seed 0 once per configuration to avoid repeating identical timing work three times.

The new `ccsgnn` candidate is a CCSNet-inspired shared encoder with separate pressure and saturation decoders, a permeability-aware graph residual for saturation conditioned on the pressure estimate, and a pressure correction conditioned on the predicted saturation change. Its exchange heads start at zero, so the network begins as the CCS-style dual-decoder baseline and learns the extra coupling as a residual. It is a project-specific architecture hypothesis, not a claim that a new published architecture has been established.

Training adds a small smooth L1 term on `log1p(game_penalty/2)` to the existing high-pressure, p-max, and saturation losses. The direct p-max loss remains, so gradients still guide pressure predictions near/below the penalty cap. Training on a two-stage unroll reduces the gap between one-step teacher-forced training and six-stage autoregressive evaluation.

## Move the existing data to the new Kaggle account

The prior output is about 320 MB and contains the generated solver datasets as well as old one-seed checkpoints. The final run reuses only the generated `.npz` datasets from that prior output; it must not count the old one-seed checkpoints as final seed results because the training objective and unroll have changed.

Preferred transfer:

1. On the old account, create a **private Kaggle Dataset** from the notebook version's `consolidated` Output Data.
2. Add the new Kaggle account as a dataset collaborator with view access. Do the same for the private `spe10model2` dataset, or upload the two SPE10 `.INC` files as a private dataset on the new account.
3. On the new account, make a notebook with **Internet on** and **GPU T4 x2**. Add the old results dataset and SPE10 dataset as inputs.

If collaborator access is inconvenient, download the old notebook output ZIP and upload it as a private dataset on the new account. Keep the `consolidated/` directory inside the ZIP. Notebook access and dataset access are separate Kaggle permissions; sharing a notebook alone does not grant access to its private data inputs.

## Kaggle cells

### Cell 1 — confirm attached inputs and GPUs

```python
from pathlib import Path
import torch

print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
print("Inputs:")
for p in Path("/kaggle/input").iterdir():
    print(" ", p)
```

### Cell 2 — get the project code

After the final-run branch is pushed, clone it here (or use its pinned commit SHA):

```python
!git clone --branch codex/final-surrogate-run --single-branch https://github.com/smerarawal/shale-well-game-theory-codes.git /kaggle/working/repo
```

### Cell 3 — restore output and locate SPE10

On the first run, the input is the old single-seed `consolidated` dataset, so only `.npz` solver data are copied. On later notebook versions, attach the latest final notebook output; its `consolidated_final` folder is restored in full so completed checkpoints/metrics/validations are reused.

```python
from pathlib import Path
from collections import Counter
import os
import shutil

root = Path("/kaggle/input")
perm = list(root.rglob("SPE10MODEL2_PERM.INC"))
phi = list(root.rglob("SPE10MODEL2_PHI.INC"))
if not perm or not phi:
    raise FileNotFoundError("Attach the SPE10 dataset containing both SPE10MODEL2 .INC files.")
spe_dir = perm[0].parent
if not (spe_dir / "SPE10MODEL2_PHI.INC").exists():
    spe_dir = phi[0].parent

# Prefer a previous final-run output directory, if present.
final_dirs = {p.parent for p in root.rglob("*_metrics.json")
              if p.parent.name == "consolidated_final"}
out_dir = Path("/kaggle/working/consolidated_final")
if final_dirs:
    source = max(final_dirs, key=lambda d: len(list(d.glob("*_metrics.json"))))
    shutil.copytree(source, out_dir, dirs_exist_ok=True)
    restore_mode = "resume prior final run"
else:
    old_data_dirs = {p.parent for p in root.rglob("stage_train.npz")}
    if not old_data_dirs:
        raise FileNotFoundError(
            "No prior stage_train.npz found. Attach the old consolidated output dataset."
        )
    source = max(old_data_dirs, key=lambda d: len(list(d.glob("*.npz"))))
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in source.glob("*.npz"):
        shutil.copy2(f, out_dir / f.name)
    restore_mode = "reuse prior generated data; train fresh final seeds"

os.environ["SPE_DIR"] = str(spe_dir)
os.environ["OUT_DIR"] = str(out_dir)
print("SPE10:", spe_dir)
print("Restored from:", source)
print("Mode:", restore_mode)
print("NPZ files:", len(list(out_dir.glob("*.npz"))))
print("Existing final model metrics:", len(list(out_dir.glob("*_metrics.json"))))
```

### Cell 4 — check code and the SPE10 parser

```python
%cd /kaggle/working/repo/sequential_game
!python spe10_geologies.py --data_dir "$SPE_DIR" --inspect
```

### Cell 5 — run or resume

```bash
%%bash
set -e
cd /kaggle/working/repo/sequential_game
export REPO=/kaggle/working/repo/sequential_game
export SPE="$SPE_DIR"
export OUT="$OUT_DIR"
export SEEDS="0 1 2"
export EPOCHS=100
export UNROLL=2
export GPU_IDS="0 1"
export MAX_PARALLEL=2
bash run_final_multiseed.sh
```

Save and run all. A Kaggle run may need more than one version. When it stops, wait for the version's output to finish saving, then add the latest version's **Notebook Output Files** to the next version and rerun from Cell 1. The restore cell will reuse completed final jobs and their validation JSONs. Models without both a checkpoint and metrics JSON are trained again.

Expected final files:

- `consolidated_final/final_multiseed_report.md`
- `consolidated_final/cpu_speed_seed0.md`
- per-seed checkpoints, metrics, validation reports, and training logs

The full schedule is similar in job count to the old 60-fit run, but uses three seeds on a selected set and a two-stage unroll. Budget roughly 20–35 hours of T4x2 wall time, split across saved Kaggle versions if necessary; actual time depends on epoch durations and Kaggle availability.
