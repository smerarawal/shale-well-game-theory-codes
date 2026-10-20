"""Summarize validation reports from run_consolidated_sweep.sh."""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--out", default=None)
a = ap.parse_args()

R = defaultdict(lambda: defaultdict(list))
M = defaultdict(list)
pat = re.compile(r"val_([a-z0-9]+)_(none|prior|prior2)_(game|mix)_s(\d+)_(game|tarbert|ness)\.json")
for f in sorted(glob.glob(os.path.join(a.dir, "val_*.json"))):
    m = pat.fullmatch(os.path.basename(f))
    if not m:
        continue
    bb, hybrid, data, seed, ev = m.groups()
    blob = json.load(open(f, encoding="utf-8"))
    R[(bb, hybrid, data)][ev].append(next(v for k, v in blob.items() if not k.endswith("_onestep")))
    mf = os.path.join(a.dir, f"{bb}_{hybrid}_{data}_s{seed}_metrics.json")
    if ev == "game" and os.path.exists(mf):
        M[(bb, hybrid, data)].append(json.load(open(mf, encoding="utf-8")))


def mean_std(xs):
    v = np.asarray(xs, float)
    if not v.size:
        return "-"
    return f"{v.mean():.3g}" + (f" ±{v.std():.2g}" if len(v) > 1 else "")


backbones = ("unet", "ccs", "gnn", "split", "runet", "gnnlstm", "fno", "ufno", "ugnn", "fgnn")
lines = [
    "# Consolidated surrogate architecture sweep\n",
    "Rollout over six stages against solver rollouts. Penalty = 30*max(0,p_max-6)^2. The U-Net rows reproduce the six baseline hybrid/data cells; other rows compare backbone families under the same conditions.\n",
]
for ev in ("game", "tarbert", "ness"):
    lines += [f"\n## Eval set: {ev}\n",
              "| backbone | hybrid | train data | seeds | params | train s | pmax abs k=3 | pmax abs k=6 | pmax p95 k=6 | pmax bias k=6 | penalty abs k=3 | penalty abs k=6 | penalty bias k=6 | frac pen<=2 k=6 | well RMSE k=6 | well MAE k=6 | field RMSE k=6 | field MAE k=6 | field R² k=6 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    rows = []
    for key, evals in R.items():
        bb, hybrid, data = key
        runs = evals.get(ev, [])
        if not runs:
            continue
        g = lambda name, idx: mean_std([r[name][idx] for r in runs])
        meta = M[key]
        params = mean_std([m["params"] / 1e6 for m in meta if "params" in m]) + "M" if meta else "-"
        seconds = mean_std([m["train_seconds"] for m in meta if "train_seconds" in m]) if meta else "-"
        rows.append((float(np.mean([r["penalty_abs"][5] for r in runs])),
                     f"| {bb} | {hybrid} | {data} | {len(runs)} | {params} | {seconds} | {g('pmax_abs', 2)} | {g('pmax_abs', 5)} | {g('pmax_abs_p95', 5)} | {g('pmax_bias', 5)} | {g('penalty_abs', 2)} | {g('penalty_abs', 5)} | {g('penalty_bias', 5)} | {g('frac_pen_err_le_tol', 5)} | {g('well_rmse', 5)} | {g('well_mae', 5)} | {g('field_rmse', 5)} | {g('field_mae', 5)} | {g('field_r2', 5)} |"))
    lines.extend(row for _, row in sorted(rows))
    lines += [f"\n### Saturation rollout metrics: {ev} (stage 6)\n",
              "| backbone | hybrid | train data | Sw RMSE | Sw MAE | Sw R² | Sw inventory bias | Sw inventory abs error | IoU Sw>0.5 | IoU Sw>0.3 | plume area bias | plume area abs | edge miss | edge false alarm | edge max-Sw error | front RMSE | well Sw RMSE |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    srows = []
    for (bb, hybrid, data), evals in R.items():
        runs = evals.get(ev, [])
        if not runs:
            continue
        g = lambda name: mean_std([r[name][5] for r in runs if name in r])
        srows.append((float(np.nanmean([r["sw_iou_50"][5] for r in runs if "sw_iou_50" in r])),
                      f"| {bb} | {hybrid} | {data} | {g('sw_rmse')} | {g('sw_mae')} | {g('sw_r2')} | {g('sw_inventory_bias')} | {g('sw_inventory_abs')} | {g('sw_iou_50')} | {g('sw_iou_30')} | {g('sw_area_bias_30')} | {g('sw_area_abs_30')} | {g('sw_edge_miss_30')} | {g('sw_edge_false_30')} | {g('sw_edge_max_err')} | {g('sw_front_rmse')} | {g('sw_well_rmse')} |"))
    lines.extend(row for _, row in sorted(srows, key=lambda x: -x[0]))

    lines += [f"\n### Pressure/penalty error strata: {ev} (stage 6)\n",
              "| backbone | hybrid | train data | stratum | n | pmax abs | pmax bias | penalty abs | penalty bias |",
              "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for (bb, hybrid, data), evals in R.items():
        for run in evals.get(ev, []):
            for name, vals in run.get("strata", {}).items():
                lines.append(f"| {bb} | {hybrid} | {data} | {name} | {vals['n']} | {vals['pmax_abs_k6']:.3f} | "
                             f"{vals['pmax_bias_k6']:.3f} | {vals['penalty_abs_k6']:.2f} | {vals['penalty_bias_k6']:.2f} |")

out = "\n".join(lines) + "\n"
print(out)
with open(a.out or os.path.join(a.dir, "consolidated_sweep.md"), "w", encoding="utf-8") as f:
    f.write(out)
