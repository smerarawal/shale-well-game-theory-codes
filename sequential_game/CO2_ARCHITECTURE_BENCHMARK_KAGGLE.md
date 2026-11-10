# CO₂ architecture benchmark on Kaggle

This run uses only the corrected-physics CO₂ transition dataset. It does not attach, read, or write the brine/SPE10 water-work outputs. The input dataset is read-only; all checkpoints and reports go under `/kaggle/working/co2_architecture_run`.

## Kaggle settings and inputs

- **Accelerator:** one GPU, preferably Tesla T4. Training uses CUDA when available. Do not select T4 x2; this script trains one model at a time and does not distribute models across GPUs.
- **Internet:** turn on for the code-clone cell.
- **Attach as input:** `smerarawal/transition-geology/co2_xz_dataset_v2`, mounted at `/kaggle/input/notebooks/smerarawal/transition-geology/co2_xz_dataset_v2`.
- **Do not attach:** `spe10model2`, brine datasets, or old surrogate checkpoints. The corrected CO₂ labels already contain the SPE10 Tarbert and Upper Ness cohorts. Old checkpoints were trained on the old physics/data and are not loaded.
- The previous preflight imported code from `/kaggle/input/datasets/smerarawal/co2-research-bundle/co2_research_run_bundle/sequential_game`. That is the older trainer snapshot: it confirms the 14-channel/five-target data contract, but it does not include `compact_unet`, `fourier_mionet`, or the pressure-tail option. Cell 3 fetches the updated trainer from this branch. If Kaggle Internet is unavailable, first publish this branch's `sequential_game/` code as a Kaggle Dataset and attach that updated dataset instead.
- Run all cells in one Kaggle version. If Kaggle stops the run, save the notebook output, attach that output to a new version, and set `RESUME_FROM` in Cell 1 to the saved output folder. The run directories are separate by experiment so configurations cannot overwrite one another.

The principal comparison is CCSNet (`ccs`), CCS-GNN (`ccs_gnn`), graph-only GNN (`gnn`), and compact U-Net (`compact_unet`), three seeds each. Fourier-MIONet is a single-seed reference first; it is an independent 2D PyTorch adaptation, not a reproduction of the published 3D model. The last cell selects two primary models using validation-only stage-6 metrics and runs separate pressure-tail and three-step-unroll ablations. The fixed test split is reported but never used to choose architectures.

## Kaggle cells

### Cell 1 — runtime and paths

```python
from pathlib import Path
import json, os, torch

DATA = Path("/kaggle/input/notebooks/smerarawal/transition-geology/co2_xz_dataset_v2")
ROOT = Path("/kaggle/working/co2_architecture_run")
CODE = Path("/kaggle/working/shale-well-game-theory-codes")
RESUME_FROM = None  # Set to a prior notebook-output folder only when resuming.

assert (DATA / "dataset_config.json").is_file(), f"Missing corrected CO2 dataset: {DATA}"
assert (DATA / "split_manifest.csv").is_file()
assert (DATA / "stage_metrics.csv").is_file()
if RESUME_FROM:
    import shutil
    shutil.copytree(RESUME_FROM, ROOT, dirs_exist_ok=True)
ROOT.mkdir(parents=True, exist_ok=True)

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
print("Dataset config:", json.dumps(json.loads((DATA / "dataset_config.json").read_text()), indent=2))
```

### Cell 2 — audit the attached labels before training (CPU)

This reports the observed pressure range, whether trapped gas varies in each split, and the largest stored solver mass residual. It does not alter the dataset.

```python
import csv, hashlib
import numpy as np

cohorts = ("friendly", "intermediate", "challenging", "tarbert", "upper_ness")
mass_residual_by_split = {s: 0.0 for s in ("train", "validation", "test")}
with (DATA / "stage_metrics.csv").open(newline="") as f:
    for row in csv.DictReader(f):
        split = row.get("split")
        value = row.get("mass_balance_residual_kg")
        if split in mass_residual_by_split and value not in (None, ""):
            mass_residual_by_split[split] = max(mass_residual_by_split[split], abs(float(value)))
for split in ("train", "validation", "test"):
    pmin, pmax, trapped_nonzero, trapped_total = np.inf, -np.inf, 0, 0
    residual_max = 0.0
    for cohort in cohorts:
        f = DATA / split / f"{cohort}.npz"
        with np.load(f, allow_pickle=False) as z:
            for key in ("p_pa_in", "p_pa_out"):
                pmin = min(pmin, float(z[key].min()))
                pmax = max(pmax, float(z[key].max()))
            for key in ("sg_trapped_in", "sg_trapped_out"):
                trapped_nonzero += int(np.count_nonzero(np.abs(z[key]) > 1e-12))
                trapped_total += int(z[key].size)
        residual_max = mass_residual_by_split[split]
    print(f"{split}: pressure=[{pmin:.6g}, {pmax:.6g}] Pa; "
          f"nonzero trapped cells={trapped_nonzero}/{trapped_total}; "
          f"max stored mass residual={residual_max:.6g} kg")

print("Input hashes:")
for rel in ("dataset_config.json", "split_manifest.csv", "stage_metrics.csv"):
    h = hashlib.sha256((DATA / rel).read_bytes()).hexdigest()
    print(rel, h)
```

If trapped gas is constant zero in all splits, keep `sg_trapped` metrics out of architecture claims. The model still predicts that channel because the trainer's common state format includes it. Pressure values are in the dataset's solver units, and this scenario is exploratory rather than field-calibrated.

### Cell 3 — fetch the CO₂-only code branch

```python
import subprocess

if not (CODE / ".git").exists():
    subprocess.run([
        "git", "clone", "--depth", "1", "--branch", "codex/co2-architecture-benchmark",
        "https://github.com/smerarawal/shale-well-game-theory-codes.git", str(CODE)
    ], check=True)
else:
    subprocess.run(["git", "-C", str(CODE), "pull", "--ff-only"], check=True)

SEQ = CODE / "sequential_game"
TRAINER = SEQ / "train_co2_xz_surrogate.py"
assert TRAINER.is_file()
print("Code branch:", subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "--short", "HEAD"], text=True).strip())
print("Trainer:", TRAINER)
```

### Cell 4 — train the four primary architectures (GPU)

Four architectures × three seeds. This is the main confirmatory architecture screen. Keep its output folder unchanged for a clean resume.

```python
import subprocess, sys

BASE = ROOT / "baseline_multiseed"
cmd = [sys.executable, str(TRAINER),
       "--data-dir", str(DATA), "--out-dir", str(BASE),
       "--backbones", "ccs", "ccs_gnn", "gnn", "compact_unet",
       "--seeds", "0", "1", "2", "--epochs", "60", "--patience", "10",
       "--batch-size", "64", "--width", "32", "--rollout-unroll", "1",
       "--device", "cuda" if torch.cuda.is_available() else "cpu"]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True, cwd=SEQ)
```

### Cell 5 — Fourier-MIONet reference (GPU)

This is one initial reference run, kept separate because it has one seed and is not part of the three-seed primary ranking.

```python
MIONET = ROOT / "fourier_mionet_reference"
cmd = [sys.executable, str(TRAINER),
       "--data-dir", str(DATA), "--out-dir", str(MIONET),
       "--backbones", "fourier_mionet", "--seeds", "0",
       "--epochs", "60", "--patience", "10", "--batch-size", "64",
       "--width", "24", "--rollout-unroll", "1",
       "--device", "cuda" if torch.cuda.is_available() else "cpu"]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True, cwd=SEQ)
```

### Cell 6 — validation-only report and top-two selection

The selection score averages ranks across stage-6 pressure-tail RMSE, p-max 95th-percentile error, mobile-gas RMSE, and plume IoU. Lower rank is better; plume IoU is ranked higher-is-better. Test metrics are intentionally excluded from selection.

```python
import json
from collections import defaultdict
import numpy as np

summaries = json.loads((BASE / "run_summary.json").read_text())
primary = ("ccs", "ccs_gnn", "gnn", "compact_unet")
metrics = ("pressure_top_decile_rmse_pa", "pmax_abs_p95_pa",
           "sg_mobile_rmse", "plume_iou_sg_ge_0p01")
values = defaultdict(lambda: defaultdict(list))
for run in summaries:
    name = run["backbone"]
    if name not in primary:
        continue
    for cohort_result in run["autoregressive_validation"].values():
        stage6 = cohort_result.get("by_stage", {}).get("5", {})
        for metric in metrics:
            if metric in stage6:
                values[name][metric].append(float(stage6[metric]))

means = {name: {m: float(np.mean(values[name][m])) for m in metrics} for name in primary}
ranks = {name: [] for name in primary}
for metric in metrics:
    descending = metric == "plume_iou_sg_ge_0p01"
    order = sorted(primary, key=lambda n: means[n][metric], reverse=descending)
    for rank, name in enumerate(order, 1):
        ranks[name].append(rank)
score = {name: float(np.mean(ranks[name])) for name in primary}
ranked = sorted(primary, key=lambda n: score[n])
print("Validation stage-6 means across held-out geologies:")
for name in primary:
    print(name, "rank_score=", round(score[name], 3), means[name])
SELECTED = ranked[:2]
print("Top two by predeclared validation rank composite:", SELECTED)
```

### Cell 7 — separate pressure-tail and three-stage-unroll ablations (GPU)

The pressure-tail run changes only the loss emphasis. The unroll-3 run starts from the ordinary loss and changes only the training rollout length. Each uses three seeds and its own output directory.

```python
import subprocess, sys

def train_ablation(name, extra_args):
    out = ROOT / name
    cmd = [sys.executable, str(TRAINER),
           "--data-dir", str(DATA), "--out-dir", str(out),
           "--backbones", *SELECTED, "--seeds", "0", "1", "2",
           "--epochs", "60", "--patience", "10", "--batch-size", "64",
           "--width", "32", "--device", "cuda" if torch.cuda.is_available() else "cpu",
           *extra_args]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=SEQ)

train_ablation("ablation_pressure_tail", [
    "--rollout-unroll", "1", "--pressure-tail-weight", "0.5",
    "--pressure-tail-quantile", "0.90"])
train_ablation("ablation_unroll3", ["--rollout-unroll", "3"])
```

### Cell 8 — write a compact run manifest and confirm outputs

```python
import hashlib, json, subprocess
import csv

manifest = {
    "code_commit": subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True).strip(),
    "dataset_config_sha256": hashlib.sha256((DATA / "dataset_config.json").read_bytes()).hexdigest(),
    "dataset_manifest_sha256": hashlib.sha256((DATA / "split_manifest.csv").read_bytes()).hexdigest(),
    "dataset_stage_metrics_sha256": hashlib.sha256((DATA / "stage_metrics.csv").read_bytes()).hexdigest(),
    "selected_validation_top2": SELECTED,
    "primary": {"architectures": ["ccs", "ccs_gnn", "gnn", "compact_unet"], "seeds": [0, 1, 2]},
    "fourier_mionet": {"seed": 0, "reference_only": True},
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
}
(ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
metric_rows = []
for group_name in ("baseline_multiseed", "fourier_mionet_reference",
                   "ablation_pressure_tail", "ablation_unroll3"):
    summary_path = ROOT / group_name / "run_summary.json"
    if not summary_path.is_file():
        continue
    for run in json.loads(summary_path.read_text()):
        for split in ("autoregressive_validation", "autoregressive_test"):
            cohort_results = run.get(split, {})
            for cohort, cohort_result in cohort_results.items():
                stage6 = cohort_result.get("by_stage", {}).get("5", {})
                metric_rows.append({
                    "group": group_name, "split": split, "backbone": run["backbone"],
                    "seed": run["seed"], "cohort": cohort, "params": run["params"],
                    "best_epoch": run["best_epoch"],
                    "cpu_inference_ms_batch1": run.get("cpu_inference_ms_sample_batch1"),
                    **{m: stage6.get(m) for m in (
                        "pressure_top_decile_rmse_pa", "pmax_abs_mean_pa", "pmax_bias_pa",
                        "pmax_abs_p95_pa", "sg_mobile_rmse", "sg_trapped_rmse",
                        "c_dissolved_kg_m3_rmse", "plume_iou_sg_ge_0p01",
                        "predicted_mass_balance_abs_p95_kg")}
                })
if metric_rows:
    with (ROOT / "stage6_metrics_by_run.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0]))
        writer.writeheader(); writer.writerows(metric_rows)
    print("Wrote:", ROOT / "stage6_metrics_by_run.csv", "rows:", len(metric_rows))
for p in sorted(ROOT.rglob("*.pt")):
    print(p.relative_to(ROOT), p.stat().st_size)
print("Manifest:", ROOT / "run_manifest.json")
```

Save a Kaggle notebook version after completion so the output directory is retained. Keep `baseline_multiseed`, `fourier_mionet_reference`, `ablation_pressure_tail`, and `ablation_unroll3` as distinct result groups. Solver replay mass residuals and runtime are solver metrics; they are not surrogate prediction errors. The trainer reports predicted mass residual separately and records CPU/GPU network-only latency separately from solver runtime.
