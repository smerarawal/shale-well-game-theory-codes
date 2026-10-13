"""
run_demo.py -- small end-to-end demonstration of the sequential Bayesian game on the real solver.
12x12 grid, 2 operators, 4 stages, 8-particle belief, 6 hidden geologies. NOT a paper result:
the grid, horizon, and payoff units are toy-scale and uncalibrated. Its job is to show the
pipeline works and that the effects (externality, value of information, cost of non-cooperation)
are measurable. Swap SolverStageModel for a stage-wise surrogate to scale it up.

    python run_demo.py --out demo_results.json
"""
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

from game import Cfg, make_geology, run_episode
from stage_model import SolverStageModel

PRIOR_SEEDS = list(range(11, 19))
TRUE_SEEDS = [101, 102, 103, 104, 105, 106]


def one(args):
    true_seed, mode, info = args
    cfg, model = Cfg(), SolverStageModel(stage_time=1.0)
    prior = [make_geology(s, cfg.n) for s in PRIOR_SEEDS]
    t0 = time.time()
    tot, log = run_episode(model, cfg, make_geology(true_seed, cfg.n), prior, mode, info, seed=true_seed)
    return {"true_seed": true_seed, "mode": mode, "info": info, "payoff": tot.tolist(),
            "welfare": float(tot.sum()), "secs": round(time.time() - t0, 1), "log": log}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_results.json")
    a = ap.parse_args()
    jobs = [(s, m, i) for s in TRUE_SEEDS for m in ("nash", "coop") for i in ("prior", "learn", "oracle")]
    with Pool(2) as pool:
        rows = pool.map(one, jobs, chunksize=1)
    json.dump({"cfg": Cfg().__dict__, "rows": rows}, open(a.out, "w"), indent=1, default=list)

    def stat(mode, info, key="welfare"):
        v = np.array([r[key] for r in rows if r["mode"] == mode and r["info"] == info])
        return v
    print(f"{'mode':5} {'info':7} {'welfare mean':>13} {'std':>7}   per-seed")
    for m in ("nash", "coop"):
        for i in ("prior", "learn", "oracle"):
            v = stat(m, i)
            print(f"{m:5} {i:7} {v.mean():13.1f} {v.std():7.1f}   {np.round(v, 0).tolist()}")
    print()
    for i in ("prior", "learn", "oracle"):
        n, c = stat("nash", i), stat("coop", i)
        print(f"cost of non-cooperation (coop - nash welfare), info={i}: mean {np.mean(c - n):.1f}  per-seed {np.round(c - n, 1).tolist()}")
    for m in ("nash", "coop"):
        print(f"value of learning ({m}): learn - prior = {np.mean(stat(m,'learn')-stat(m,'prior')):.1f}  "
              f"| value of perfect info: oracle - learn = {np.mean(stat(m,'oracle')-stat(m,'learn')):.1f}")
