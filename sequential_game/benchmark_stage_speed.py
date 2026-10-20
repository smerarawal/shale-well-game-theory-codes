"""Time the full CPU solver and each trained StageSurrogate on identical saved transitions.

Usage: python benchmark_stage_speed.py --data stage_eval_game.npz --ckpt_dir hyb --out hyb/cpu_speed
The solver and surrogate timings include Python/model overhead and cover one complete stage.
"""
import argparse
import glob
import json
import os
import statistics
import time

import numpy as np
import torch

from stage_model import SolverStageModel
from stage_surrogate import StageNet, StageSurrogate
from train_stage_surrogate import load


def timed(fn, repeats):
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="saved evaluation episodes; stage 6 is benchmarked")
    ap.add_argument("--ckpt_dir", required=True, help="directory with consolidated *.pt checkpoints")
    ap.add_argument("--out", required=True, help="output prefix, without extension")
    ap.add_argument("--cases", type=int, default=3, help="number of evenly spaced episodes to benchmark")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    d = load(a.data)
    E, T, n, _ = d["well_rate"].shape
    eps = np.unique(np.linspace(0, E - 1, min(a.cases, E), dtype=int))
    stage = T - 1
    samples = []
    for e in eps:
        geo = {"k": d["permeability"][e].astype(float), "phi": d["porosity"][e].astype(float)}
        seq = []
        for s in range(T):
            wells = [(int(i), int(j), float(d["well_rate"][e, s, i, j]))
                     for i, j in zip(*np.nonzero(d["well_rate"][e, s]))]
            seq.append(({"p": d["p_in"][e, s].astype(float), "Sw": d["Sw_in"][e, s].astype(float)}, wells))
        samples.append((geo, seq))

    sim = SolverStageModel(stage_time=float(np.mean(d["elapsed"])))
    solver_ms = []
    for geo, seq in samples:
        state, wells = seq[stage]
        sim.advance(geo, state, wells)  # warm-up
        solver_ms.extend(timed(lambda: sim.advance(geo, state, wells), a.repeats))

    results = []
    for path in sorted(glob.glob(os.path.join(a.ckpt_dir, "*.pt"))):
        blob = torch.load(path, map_location="cpu")
        net = StageNet.from_blob(blob)
        net.load_state_dict(blob["state_dict"])
        model = StageSurrogate(net=net, stage_time=blob.get("stage_time", 1.0), device="cpu")
        sur_ms = []
        for geo, seq in samples:
            memory_state = seq[0][0]
            # Warm recurrent models through the earlier stages; reset hidden state by
            # beginning from the saved stage-1 state, then pass its memory forward.
            model.initial_state(n)
            for prior_state, prior_wells in seq[:stage]:
                prior_state = dict(prior_state)
                if "_memory" in memory_state:
                    prior_state["_memory"] = memory_state["_memory"]
                memory_state, _ = model.advance(geo, prior_state, prior_wells)
            state, wells = seq[stage]
            state = dict(state)
            if "_memory" in memory_state:
                state["_memory"] = memory_state["_memory"]
            for _ in range(a.warmup):
                model.advance(geo, state, wells)
            sur_ms.extend(timed(lambda: model.advance(geo, state, wells), a.repeats))
        solver_med = statistics.median(solver_ms)
        sur_med = statistics.median(sur_ms)
        results.append({"checkpoint": os.path.basename(path), "median_ms": sur_med,
                        "p90_ms": float(np.percentile(sur_ms, 90)), "solver_median_ms": solver_med,
                        "speedup_median": solver_med / sur_med if sur_med else None,
                        "samples": len(samples), "repeats_per_sample": a.repeats})
        print(f"{os.path.basename(path)}: {sur_med:.3f} ms/stage; solver {solver_med:.3f} ms; "
              f"{solver_med / sur_med:.2f}x")

    payload = {"device": "cpu", "threads": a.threads, "eval_data": a.data, "stage_index": stage + 1,
               "solver_median_ms": statistics.median(solver_ms), "solver_p90_ms": float(np.percentile(solver_ms, 90)),
               "cases": len(samples), "repeats": a.repeats, "models": results}
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lines = ["# CPU stage runtime benchmark\n",
             f"Eval file: `{a.data}`; stage {stage + 1}; {len(samples)} episode(s), {a.repeats} repeats each; {a.threads} CPU thread(s).",
             "Times include full stage call overhead. Solver and surrogates use the same geology, state, and well schedule.\n",
             "| checkpoint | surrogate median ms | surrogate p90 ms | solver median ms | median speedup |",
             "|---|---:|---:|---:|---:|"]
    for r in sorted(results, key=lambda v: v["median_ms"]):
        lines.append(f"| {r['checkpoint']} | {r['median_ms']:.3f} | {r['p90_ms']:.3f} | {r['solver_median_ms']:.3f} | {r['speedup_median']:.2f}x |")
    with open(a.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
