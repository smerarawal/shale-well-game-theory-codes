"""
summarize_arch_sweep.py -- one table per eval set from run_arch_sweep.sh output.

Files: <dir>/val_<hybrid>-<backbone>[-uN]_mix_s<seed>_<evalset>.json and <dir>/<same tag>_metrics.json
(parameter count, training seconds). Rows: backbone x hybrid (x unroll), mean over seeds at stage k=6 / k=3.
Eval sets: game = the game's 24 truths; tarbert / ness = held-out SPE10 patches (same style as training data).
"""
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

R = defaultdict(lambda: defaultdict(list))        # (hybrid, backbone) -> evalset -> [val json]
META = defaultdict(list)                          # (hybrid, backbone) -> [metrics json]
for f in sorted(glob.glob(os.path.join(a.dir, "val_*_mix_s*_*.json"))):
    m = re.match(r"val_(none|prior2|prior)-([a-z0-9-]+)_mix_s(\d+)_(\w+)\.json", os.path.basename(f))
    if not m:
        continue
    h, bb, s, ev = m.groups()
    j = json.load(open(f))
    R[(h, bb)][ev].append(next(v for k, v in j.items() if not k.endswith("_onestep")))
    mf = os.path.join(a.dir, f"{h}-{bb}_mix_s{s}_metrics.json")
    if ev == "game" and os.path.exists(mf):
        META[(h, bb)].append(json.load(open(mf)))


def ms(vals):
    v = np.array(vals, float)
    return f"{v.mean():.3g}" + (f" ±{v.std():.2g}" if len(v) > 1 else "")


lines = ["# Backbone sweep inside the physics hybrid (train data: game + SPE10 train band)\n",
         "Rollout over 6 stages vs solver rollouts; penalty = 30*max(0,p_max-6)^2. Backbone suffix -uN = trained on N chained stages.\n"]
for ev in ("game", "tarbert", "ness"):
    lines += [f"\n## eval set: {ev}\n",
              "| hybrid | backbone | seeds | params | train s | pmax err k=3 | pmax err k=6 | pmax bias k=6 | penalty err k=6 | frac pen<=2 k=6 | well RMSE k=6 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for (h, bb), d in R.items():
        runs = d.get(ev)
        if not runs:
            continue
        g = lambda key, i: ms([r[key][i] for r in runs])
        meta = META.get((h, bb), [])
        params = f"{meta[0]['params'] / 1e6:.2f}M" if meta and "params" in meta[0] else "-"
        secs = ms([x["train_seconds"] for x in meta if "train_seconds" in x]) if meta and "train_seconds" in meta[0] else "-"
        key = float(np.mean([r["penalty_abs"][5] for r in runs]))
        rows.append((key, f"| {h} | {bb} | {len(runs)} | {params} | {secs} | {g('pmax_abs', 2)} | {g('pmax_abs', 5)} | {g('pmax_bias', 5)} | "
                          f"{g('penalty_abs', 5)} | {g('frac_pen_err_le_tol', 5)} | {g('well_rmse', 5)} |"))
    lines += [r for _, r in sorted(rows)]
txt = "\n".join(lines)
print(txt)
open(a.out or os.path.join(a.dir, "arch_sweep.md"), "w").write(txt + "\n")
