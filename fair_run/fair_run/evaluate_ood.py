"""
evaluate_ood.py -- evaluates every trained model (all seeds, best-validation
weights) on the OOD sets. Uses the same input/pressure normalisation statistics
as training (from run_config.json) and the same metric code as the main run.

    python evaluate_ood.py --out final_run
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from model_zoo_v4 import make_all_architectures_v4, ARCH_COST
from run_fair_comparison import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="final_run")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.out, "run_config.json")))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zoo = make_all_architectures_v4(cfg["H"], cfg["W"], cfg["T"], cfg["stats"])

    sets = {}
    for p in sorted(glob.glob(os.path.join(a.out, "ood", "ood_*.npz"))):
        if p.endswith(".tmp.npz") or "chunks" in p:
            continue
        d = np.load(p)
        x = np.stack([d["permeability"], d["porosity"], d["well_mask"]], axis=1).astype(np.float32)
        y = np.stack([d["final_pressure"], d["final_saturation"]], axis=1).astype(np.float32)
        sets[os.path.basename(p)[:-4]] = (torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev))
    if not sets:
        print("no OOD sets found"); return

    res = {}
    for arch in ARCH_COST:
        wfiles = sorted(glob.glob(os.path.join(a.out, "weights", f"{arch}__seed*.pt")))
        if not wfiles:
            continue
        per_set = {n: [] for n in sets}
        for wf in wfiles:
            m = zoo[arch]().to(dev)
            m.load_state_dict(torch.load(wf, map_location=dev))
            for n, (x, y) in sets.items():
                per_set[n].append(evaluate(m, x, y))
        res[arch] = {n: {k: float(np.mean([r[k] for r in rs])) for k in ("balanced", "rel_pressure", "rel_saturation")}
                     | {"balanced_std": float(np.std([r["balanced"] for r in rs])), "n_seeds": len(rs)}
                     for n, rs in per_set.items()}
        print(arch, {n: round(v["balanced"], 4) for n, v in res[arch].items()}, flush=True)
    json.dump(res, open(os.path.join(a.out, "ood", "ood_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
