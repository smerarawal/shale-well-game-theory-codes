"""
benchmark_inference.py -- clean, SEQUENTIAL inference timing of the trained
(best-validation, seed-0) weights of every architecture. Run it after
training finishes and with nothing else on the GPU, so the speed column is not
contaminated by the two training workers sharing the machine.

    python benchmark_inference.py --out final_run
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from model_zoo_v4 import make_all_architectures_v4, ARCH_COST


@torch.no_grad()
def time_model(model, x, warmup=10, reps=50):
    dev = x.device
    for _ in range(warmup):
        model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="final_run")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.out, "run_config.json")))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zoo = make_all_architectures_v4(cfg["H"], cfg["W"], cfg["T"], cfg["stats"])
    H, W = cfg["H"], cfg["W"]
    res = {}
    for arch in ARCH_COST:
        wpath = os.path.join(a.out, "weights", f"{arch}__seed{a.seed}.pt")
        if not os.path.exists(wpath):
            print(f"skip {arch}: no weights")
            continue
        m = zoo[arch]().to(dev)
        m.load_state_dict(torch.load(wpath, map_location=dev))
        m.eval()
        r = {}
        for bs in (1, 64):
            x = torch.randn(bs, 3, H, W, device=dev) * 0.1
            sec = time_model(m, x)
            r[f"ms_per_sample_bs{bs}"] = 1000 * sec / bs
        r["device"] = str(dev)
        res[arch] = r
        print(f"{arch:22s} bs1={r['ms_per_sample_bs1']:.3f} ms/sample   bs64={r['ms_per_sample_bs64']:.4f} ms/sample", flush=True)
    os.makedirs(os.path.join(a.out, "bench"), exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "bench", "inference.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
