"""Run the paired, water-injection follow-up defined in README_STUDY7.md.

Example (Kaggle notebook, CPU recommended for the current NumPy solver):
    python run_study7.py --hours 3.7 --out study7_results --tiers A,B,C,D --procs 4 --trajectories

Outputs are one atomic JSON per completed geology/arm. Optional compressed NPZ files
contain the true geology and stage input/output states for surrogate training.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import glob
import json
import time
import traceback
from multiprocessing import Pool

import numpy as np

from eps_check import open_loop_eps, two_stage_block_eps
from game3 import Cfg, particle_set_v3, run_episode, schedule_from_log
from run_study3 import INF, true_geo
from stage_model import SolverStageModel

DEADLINE = [None]
GEOS = range(24, 60)
BASE = dict(layout="close", horizon=3, cap_total=24.0)


def job(exp, tag, geo, cfg, mode="nash", info="oracle", kind="episode", **extra):
    return {"exp": exp, "tag": tag, "geo": geo, "cfg": cfg, "mode": mode,
            "info": info, "kind": kind, "name": f"{exp}__{tag}__g{geo:02d}", **extra}


def plan(tiers="ABCD"):
    out = {k: [] for k in "ABCD"}
    for g in GEOS:
        # A: core paired gap, including uncapped control. H=6 is a fixed audit subset.
        for cap_tag, cap in (("cap24", 24.0), ("uncapped", INF)):
            for mode in ("nash", "coop"):
                out["A"].append(job("e18", f"{cap_tag}_{mode}_H3", g,
                                    {**BASE, "cap_total": cap}, mode))
        if g < 36:
            for mode in ("nash", "coop"):
                out["A"].append(job("e18", f"cap24_{mode}_H6", g,
                                    {**BASE, "horizon": 6}, mode))
        # B: paired cap-response curve.
        for cap in (12, 16, 20, 24, 32, 48):
            for mode in ("nash", "coop"):
                out["B"].append(job("e19", f"cap{cap}_{mode}", g,
                                    {**BASE, "cap_total": float(cap)}, mode))
        # C: cap-allocation rules. Cooperative reference uses the original shared-pool rule.
        for cap in (16, 24):
            for rule in ("prorata", "cea", "cel", "fixed"):
                out["C"].append(job("e20", f"cap{cap}_{rule}_nash", g,
                                    {**BASE, "cap_total": float(cap), "rationing_rule": rule}, "nash"))
            out["C"].append(job("e20", f"cap{cap}_coop", g,
                                {**BASE, "cap_total": float(cap)}, "coop"))
        # D: margins and price instruments; price is charged in Nash private payoffs only.
        for margins_tag, margins in (("equal", (20.0, 20.0)),
                                     ("op0_high_24_16", (24.0, 16.0)),
                                     ("op0_high_28_12", (28.0, 12.0))):
            for price in (0.0, 5.0, 10.0):
                out["D"].append(job("e21", f"{margins_tag}_price{price:g}", g,
                                    {**BASE, "margins": margins, "inj_price": price}, "nash"))
            out["D"].append(job("e21", f"{margins_tag}_fixed", g,
                                {**BASE, "margins": margins, "alloc": True}, "nash"))
        # Stronger unilateral deviation diagnostic, paired with the same cap-24 Nash path.
        out["D"].append(job("e22", "cap24_nash_two_stage_eps", g, BASE,
                            "nash", kind="two_stage_eps"))
        out["D"].append(job("e22", "cap24_nash_one_stage_eps", g, BASE,
                            "nash", kind="one_stage_eps"))
        # E: prioritize uncertainty only after the physical/policy tiers.
        learn = dict(BASE, **{"belief_mode": "v4", "prune_thr": 0.01,
                              "temper": 0.5, "model_err": 0.1, "jitter": 0.15})
        for tag, mode, info in (("nash_prior", "nash", "prior"),
                                ("nash_learn", "nash", "learn"),
                                ("coop_learn", "coop", "learn")):
            if g < 48:
                out["D"].append(job("e23", tag, g, learn, mode, info))
    return [x for tier in "ABCD" if tier in tiers for x in out[tier]]


def _write_npz(path, geo, log):
    np.savez_compressed(
        path, permeability=geo["k"], porosity=geo["phi"],
        pressure_in=np.asarray([x["state_in"]["p"] for x in log], np.float32),
        saturation_in=np.asarray([x["state_in"]["Sw"] for x in log], np.float32),
        pressure_out=np.asarray([x["state_out"]["p"] for x in log], np.float32),
        saturation_out=np.asarray([x["state_out"]["Sw"] for x in log], np.float32),
        wells=np.asarray([x["new_wells"] for x in log], dtype=object),
        rates=np.asarray([x["rates"] for x in log], np.float32),
        served_rates=np.asarray([x["served_rates"] for x in log], np.float32),
        requested_vol=np.asarray([x["requested_vol"] for x in log], np.float32),
        served_vol=np.asarray([x["served_vol"] for x in log], np.float32))


def _work(args):
    j, outdir, trajectories = args
    path = os.path.join(outdir, j["name"] + ".json")
    if os.path.exists(path) or (DEADLINE[0] is not None and time.time() >= DEADLINE[0]):
        return None
    try:
        cfg = Cfg(**j["cfg"])
        geo = true_geo(j["geo"], cfg.n)
        model = SolverStageModel(cfg.stage_time)
        prior = particle_set_v3(cfg.n, field_seeds=(11, 12, 13, 14))
        seed = 8000 + j["geo"]
        rec = {k: v for k, v in j.items() if k != "cfg"}
        rec.update(cfg={k: ("inf" if isinstance(v, float) and np.isinf(v) else v)
                        for k, v in j["cfg"].items()}, field_seed=int(101 + j["geo"]),
                   mult=float(geo["mult"]), seed=seed, fluid="water", solver="solver_v3")
        started = time.time()
        cap_states = trajectories and j["kind"] == "episode"
        summary, log = run_episode(model, cfg, geo, prior, j["mode"], j["info"],
                                   seed=seed, pool_seeds=tuple(range(11, 19)),
                                   capture_states=cap_states)
        rec.update(summary)
        if j["kind"] == "two_stage_eps":
            rec["eps"] = two_stage_block_eps(model, cfg, geo, schedule_from_log(log, cfg))
        elif j["kind"] == "one_stage_eps":
            rec["eps"] = open_loop_eps(model, cfg, geo, schedule_from_log(log, cfg), objective="own")
        elif j["kind"] == "episode":
            rec["log"] = log
        if cap_states:
            npz_path = os.path.join(outdir, j["name"] + "_trajectory.npz")
            _write_npz(npz_path, geo, log)
            rec["trajectory_file"] = os.path.basename(npz_path)
            for stage in log:
                stage.pop("state_in", None); stage.pop("state_out", None)
        rec["secs"] = round(time.time() - started, 2)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        os.replace(tmp, path)
        return j["name"], rec["secs"]
    except Exception:
        with open(os.path.join(outdir, j["name"] + ".err"), "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        return j["name"], -1


def _init(deadline):
    DEADLINE[0] = deadline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=3.7)
    ap.add_argument("--out", default="study7_results")
    ap.add_argument("--tiers", default="A,B,C,D", help="A=core; B=cap sweep; C=rationing; D=deviations and learning")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--count_only", action="store_true")
    ap.add_argument("--trajectories", action="store_true", help="save true-state NPZ files for surrogate training")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tiers = a.tiers.replace(",", "").upper()
    jobs = plan(tiers)
    todo = [j for j in jobs if not os.path.exists(os.path.join(a.out, j["name"] + ".json"))]
    print(f"Study 7: {len(todo)} jobs remaining in tiers {tiers}; geologies 24-59; processes={a.procs}", flush=True)
    if a.count_only:
        return
    start = time.time()
    deadline = start + max(0.0, a.hours) * 3600 if a.hours > 0 else None
    done = 0
    with Pool(a.procs, initializer=_init, initargs=(deadline,)) as pool:
        for result in pool.imap(_work, [(j, a.out, a.trajectories) for j in todo], chunksize=1):
            if result is not None:
                done += 1
                if done % 5 == 0 or result[1] < 0:
                    print(f"[{(time.time()-start)/60:.1f} min] {done}/{len(todo)} completed; {result[0]} {result[1]}s", flush=True)
    print(f"Completed {done} jobs in {(time.time()-start)/3600:.2f} h. Analyze with: python analyze7.py --dir {a.out}", flush=True)


if __name__ == "__main__":
    main()
