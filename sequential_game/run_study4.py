"""
run_study4.py -- study v4 (continues study v3; same engine, same result folder, fully resumable).

    python run_study4.py --out seq3_results --resume_from <old results dir(s)>      # no time limit by default
    python run_study4.py --count_only                                               # how many jobs are left
    python run_study4.py --tiers A,B                                                # run only some tiers

What is new (the rest of the v3 plan is appended as tier D, so ONE command finishes everything):
  Exp 5  belief repair    : ablation of four fixes to the overconfident posterior (tempered likelihood,
                            model-error term, more fields in the prior/pool, bigger jitter) with new
                            PREDICTIVE calibration diagnostics (one-step-ahead pressure z-scores, peak-pressure
                            predictive interval). b0 = v3 belief re-run with the diagnostics.
  Exp 6  planner check    : the weak-setting results (spacing without permit, m2/m3/causer) re-done at H=3,
                            where the planner error is smaller (v3 showed the H=2 gap shrinking at H>=3).
  Exp 7  learning arms    : prior / learn / ramp-up / coop / m2 / causer / spacing, all with the repaired
                            belief (b4), on geologies 0-11 first, then 12-23.
  Exp 8  dose-response    : liability multiplier m in {1.25,1.5,2.5,4} (weak and strong settings) to see why
                            m=2 closes 75% of the gap and m=3 only 4%.

Tiers (every prefix is a complete paired sample over geologies):
  A  geologies 0-11 : Exp 5, Exp 6, Exp 8          (belief repair first, then the cheap oracle checks)
  B  geologies 0-11 : Exp 7
  C  geologies 12-23: Exp 6, Exp 8, Exp 5, Exp 7
  D  the unfinished v3 plan (oracle experiments 1/3/4 on more geologies, Exp 2 on 16-23)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, glob, json, shutil, time
from multiprocessing import Pool

from game3 import Cfg, make_geology, particle_set_v3, run_episode
import run_study3 as r3
from run_study3 import INF, J, POLICIES, STRONG, WEAK, lay_cfg, true_geo

DEADLINE = [None]
LEARN = dict(belief_mode="v4", prune_thr=0.01)
FULL = dict(**LEARN, temper=0.5, model_err=0.1, jitter=0.15)   # b4: all fixes together

BELIEFS = {                                  # tag -> (cfg overrides, number of prior fields)
    "b0": (LEARN, 2),
    "b1_temper": ({**LEARN, "temper": 0.5}, 2),
    "b2_modelerr": ({**LEARN, "model_err": 0.1}, 2),
    "b3_fields4": (LEARN, 4),
    "b4_full": (FULL, 4),
}


def Jf(exp, tag, g, cfgkw, mode="nash", info="oracle", fields=2, **extra):
    return J(exp, tag, g, "episode", cfgkw, mode, info, fields=fields, **extra)


def exp5(g, t):
    for tag, (kw, nf) in BELIEFS.items():
        t.append(Jf("e5", f"{tag}", g, kw, "nash", "learn", nf))


def exp6(g, t):
    for lay in ("far", "mid", "close"):
        for mo in ("nash", "coop"):
            t.append(Jf("e6", f"{lay}_inf_H3_{mo}", g, {**lay_cfg(lay, INF), "horizon": 3}, mo))
    for pol in ("m2", "m3", "causer"):
        t.append(Jf("e6", f"weak_{pol}_H3", g, {**WEAK, "horizon": 3, **POLICIES[pol]}, "nash"))
    t.append(Jf("e6", "weak_base_H3_noext", g, {**WEAK, "horizon": 3, "extrap": False}, "nash"))


def exp8(g, t):
    for m in (1.25, 1.5, 2.5, 4.0):
        t.append(Jf("e8", f"weak_m{m:g}", g, {**WEAK, "liab_mult": m}))
        t.append(Jf("e8", f"strong_m{m:g}", g, {**STRONG, "liab_mult": m}))


def exp7(g, t):
    kw, nf = FULL, 4
    t.append(Jf("e7", "nash_prior", g, FULL, "nash", "prior", nf))
    t.append(Jf("e7", "nash_learn", g, FULL, "nash", "learn", nf))
    t.append(Jf("e7", "nash_learn_pilot", g, {**FULL, "pilot_rate": 0.5}, "nash", "learn", nf))
    t.append(Jf("e7", "coop_learn", g, FULL, "coop", "learn", nf))
    t.append(Jf("e7", "weak_m2_learn", g, {**FULL, "liab_mult": 2.0}, "nash", "learn", nf))
    t.append(Jf("e7", "weak_causer_learn", g, {**FULL, "liab_rule": "causer"}, "nash", "learn", nf))
    for lay in ("far", "close"):
        for mo in ("nash", "coop"):
            t.append(Jf("e7", f"{lay}_learn_{mo}", g, {**FULL, "layout": lay}, mo, "learn", nf))


def plan(tiers="ABCD"):
    T = {k: [] for k in "ABCD"}
    for g in range(12):
        exp5(g, T["A"])
    for g in range(12):
        exp6(g, T["A"]); exp8(g, T["A"])
    for g in range(12):
        exp7(g, T["B"])
    for g in range(12, 24):
        exp6(g, T["C"]); exp8(g, T["C"])
    for g in range(12, 24):
        exp5(g, T["C"])
    for g in range(12, 24):
        exp7(g, T["C"])
    T["D"] = [dict(j, fields=2) for j in r3.plan()]
    return [j for k in "ABCD" if k in tiers for j in T[k]]


def work(args):
    job, outdir = args
    path = os.path.join(outdir, job["name"] + ".json")
    if os.path.exists(path) or (DEADLINE[0] and time.time() > DEADLINE[0]):
        return None
    try:
        return _work(job, outdir, path)
    except Exception:
        import traceback
        open(os.path.join(outdir, job["name"] + ".err"), "w").write(traceback.format_exc())
        return job["name"], -1


def _work(job, outdir, path):
    from eps_check import open_loop_eps, osd_eps
    from game3 import schedule_from_log
    from stage_model import SolverStageModel
    cfg = Cfg(**job["cfg"])
    model = SolverStageModel(1.0)
    geo = true_geo(job["geo"])
    nf = job.get("fields", 2)
    seeds = (11, 12) if nf == 2 else (11, 12, 13, 14)
    pool_seeds = (11, 12, 13, 14) if nf == 2 else tuple(range(11, 19))
    pool = particle_set_v3(cfg.n, field_seeds=seeds)
    seed = 1000 + job["geo"]
    t0 = time.time()
    rec = {k: v for k, v in job.items() if k != "cfg"}
    rec["cfg"] = {k: (str(v) if v == INF else v) for k, v in job["cfg"].items()}
    rec["mult"] = geo["mult"]
    if job["kind"] == "episode":
        summ, log = run_episode(model, cfg, geo, pool, job["mode"], job["info"], seed=seed, pool_seeds=pool_seeds)
        rec.update(summ)
        rec["log"] = log
    elif job["kind"] == "eps":
        summ, log = run_episode(model, cfg, geo, pool, job["mode"], "oracle", seed=seed)
        rec.update(summ)
        rec["eps"] = open_loop_eps(model, cfg, geo, schedule_from_log(log, cfg), objective=job["objective"])
    elif job["kind"] == "osd":
        base = run_episode(model, cfg, geo, pool, "nash", "oracle", seed=seed)
        rec["welfare"] = base[0]["welfare"]
        rec["eps"] = osd_eps(model, cfg, geo, job["t0"], "nash", seed, pool, base=base)
    rec["secs"] = round(time.time() - t0, 1)
    tmp = path + ".tmp"
    json.dump(rec, open(tmp, "w"))
    os.replace(tmp, path)
    return job["name"], rec["secs"]


def init(deadline):
    DEADLINE[0] = deadline


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=0.0, help="0 = no limit; otherwise no new job starts after this")
    ap.add_argument("--out", default="seq3_results")
    ap.add_argument("--resume_from", nargs="*", default=[], help="folder(s) with earlier results to copy in")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--tiers", default="ABCD")
    ap.add_argument("--count_only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for src in a.resume_from:
        for f in glob.glob(os.path.join(src, "**", "e[1-9]__*.json"), recursive=True):
            d = os.path.join(a.out, os.path.basename(f))
            if not os.path.exists(d):
                shutil.copy(f, d)
    jobs = plan(a.tiers.replace(",", ""))
    seen, todo = set(), []
    for j in jobs:
        if j["name"] in seen or os.path.exists(os.path.join(a.out, j["name"] + ".json")):
            continue
        seen.add(j["name"])
        todo.append(j)
    by = {}
    for j in todo:
        by[j["exp"]] = by.get(j["exp"], 0) + 1
    print(f"procs={a.procs} jobs_left={len(todo)}  by experiment: {dict(sorted(by.items()))}", flush=True)
    if a.count_only:
        raise SystemExit
    start = time.time()
    deadline = start + a.hours * 3600 if a.hours > 0 else None
    done = 0
    with Pool(a.procs, initializer=init, initargs=(deadline,)) as pool:
        for r in pool.imap(work, [(j, a.out) for j in todo], chunksize=1):
            if r is None:
                continue
            done += 1
            if done % 5 == 0:
                print(f"[{(time.time()-start)/60:6.1f} min] finished {done}/{len(todo)}  last={r[0]} {r[1]}s", flush=True)
    print(f"finished {done} jobs in {(time.time()-start)/3600:.2f} h. Now run: python analyze3.py --dir {a.out}; python analyze4.py --dir {a.out}", flush=True)
