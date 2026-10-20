"""
summarize_hybrid_2x2.py -- collect the rollout-validation JSONs written by run_hybrid_2x2.sh into one table.

Files: <dir>/val_<hybrid>_<data>_s<seed>_<evalset>.json   (hybrid in none|prior|prior2, data in game|mix)
For every (hybrid, data) and eval set: mean (and seed std when >1 seed) at stage k=6 and k=3 of
p_max |err|, p_max bias, penalty |err| (welfare units), fraction of rollouts with penalty error <= tol,
plus well-cell RMSE at k=6. 'game' = the game's own 24 truths, 'tarbert'/'ness' = held-out SPE10 patches
(OOD for data=game, held-out-but-same-style for data=mix).
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

R = defaultdict(lambda: defaultdict(list))        # (hybrid,data) -> evalset -> [json per seed]
for f in sorted(glob.glob(os.path.join(a.dir, "val_*_s*_*.json"))):
    m = re.match(r"val_(none|prior2|prior)_(game|mix)_s(\d+)_(\w+)\.json", os.path.basename(f))
    if not m:
        continue
    h, dname, s, ev = m.groups()
    j = json.load(open(f))
    R[(h, dname)][ev].append(next(v for k, v in j.items() if not k.endswith("_onestep")))

def ms(vals):
    v = np.array(vals, float)
    return f"{v.mean():.3g}" + (f" ±{v.std():.2g}" if len(v) > 1 else "")

lines = ["# Hybrid 2x2: {none, prior, prior2} x {game fields only, game + SPE10 train band}\n",
         "Rollout over 6 stages against solver rollouts. Penalty = 30*max(0,p_max-6)^2.\n"]
evs = sorted({e for d in R.values() for e in d}, key=lambda e: ["game", "tarbert", "ness"].index(e) if e in ("game", "tarbert", "ness") else 9)
order = [(h, d) for d in ("game", "mix") for h in ("none", "prior", "prior2") if (h, d) in R]
for ev in evs:
    lines += [f"\n## eval set: {ev}\n", "| hybrid | train data | seeds | pmax err k=3 | pmax err k=6 | pmax bias k=6 | penalty err k=3 | penalty err k=6 | frac pen<=tol k=6 | well RMSE k=6 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for h, d in order:
        runs = R[(h, d)].get(ev)
        if not runs:
            continue
        g = lambda key, i: ms([r[key][i] for r in runs])
        lines.append(f"| {h} | {d} | {len(runs)} | {g('pmax_abs', 2)} | {g('pmax_abs', 5)} | {g('pmax_bias', 5)} | "
                     f"{g('penalty_abs', 2)} | {g('penalty_abs', 5)} | {g('frac_pen_err_le_tol', 5)} | {g('well_rmse', 5)} |")
txt = "\n".join(lines)
print(txt)
open(a.out or os.path.join(a.dir, "hybrid_2x2.md"), "w").write(txt + "\n")
