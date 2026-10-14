"""
run_study3.py -- Experiments 1-4 in ONE time-budgeted, resumable CPU run (real solver in the loop).

    python run_study3.py --hours 3.4 --out seq3_results
Re-run the same command after any interruption; finished jobs (one json each) are skipped.
Use --resume_from <dir> to seed --out with results kept from an earlier session.

  Exp 1  equilibrium quality  : planner horizon convergence, open-loop deviation epsilon,
                                one-shot-deviation epsilon (reactive continuation)
  Exp 2  belief + ramp-up     : rejuvenated particle filter vs old belief, value of monitoring,
                                mandatory low-rate first stage ("ramp-up")
  Exp 3  externality strength : well spacing {far, mid, close} x shared injection permit
  Exp 4  policy decomposition : liability multiplier, owner-of-peak-pays, per-operator quota,
                                separated into externality correction (oracle) and precaution (learning)

Jobs are ordered in TIERS so every prefix is a complete, perm-balanced, paired sample:
  tier 1  geologies 0-11 : all cheap oracle experiments (1, 3, 4)
  tier 2  geologies 0-7  : learning experiments (2), one-shot-deviation check, learning runs of 3 and 4
  tier 3  Exp 2 on geologies 8-15, coop-learn on 0-7, then the tier-1 experiments on geologies 12-23
  tier 4  geologies 24-35: tier-1 experiments, Exp 2 on 16-23
No job STARTS after the deadline; the last jobs end within ~10 minutes of it.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, glob, json, shutil, time
from multiprocessing import Pool

from game3 import Cfg, make_geology, particle_set_v3, run_episode, schedule_from_log
from eps_check import open_loop_eps, osd_eps
from stage_model import SolverStageModel

TRUE_MULTS = [1.0, 0.6, 1.7, 0.7, 1.4, 0.5, 2.0, 0.85, 1.2]   # same cycle as study v1/v2 (off-grid values included)
INF = float("inf")
DEADLINE = [None]
LEARN = dict(belief_mode="v3", prune_thr=0.01)


def true_geo(idx, n=12):
    return make_geology(101 + idx, TRUE_MULTS[idx % len(TRUE_MULTS)], n)


def H_cfg(H, extrap=True):
    return dict(horizon=H, extrap=extrap)


def lay_cfg(layout, cap):
    return dict(layout=layout, cap_total=cap, horizon=6, extrap=False) if cap < INF else \
        dict(layout=layout, cap_total=INF, horizon=2, extrap=True)


POLICIES = {"base": {}, "m2": {"liab_mult": 2.0}, "m3": {"liab_mult": 3.0},
            "causer": {"liab_rule": "causer"}, "quota": {"rates": (0.0, 0.5, 0.75)}}
STRONG = lay_cfg("close", 24.0)      # strong-externality setting for Exp 4
WEAK = lay_cfg("v2", INF)            # the study-v2 setting


def J(exp, tag, g, kind, cfgkw, mode="nash", info="oracle", **extra):
    return dict(exp=exp, tag=tag, geo=g, kind=kind, cfg=cfgkw, mode=mode, info=info, name=f"{exp}__{tag}__g{g:02d}", **extra)


def plan():
    tiers = {1: [], 2: [], 3: [], 4: []}

    def exp1(g, t):
        for H in (1, 2, 3, 4, 6):
            for mo in ("nash", "coop"):
                t.append(J("e1", f"{mo}_H{H}", g, "episode", H_cfg(H), mo))
        t.append(J("e1", "nash_H2_noextrap", g, "episode", H_cfg(2, False), "nash"))
        t.append(J("e1", "eps_nash_H2", g, "eps", H_cfg(2), "nash", objective="own"))
        t.append(J("e1", "eps_coop_H2", g, "eps", H_cfg(2), "coop", objective="sum"))

    def exp3_oracle(g, t):
        cfgs = [(l, c) for l in ("far", "mid", "close") for c in (INF, 24.0)] + [("close", 16.0), ("v2", 24.0)]
        for lay, cap in cfgs:
            for mo in ("nash", "coop"):
                t.append(J("e3", f"{lay}_cap{cap:g}_{mo}", g, "episode", lay_cfg(lay, cap), mo))

    def exp4_oracle(g, t):
        for pol, kw in POLICIES.items():
            t.append(J("e4", f"weak_{pol}_oracle", g, "episode", {**WEAK, **kw}, "nash"))
            t.append(J("e4", f"strong_{pol}_oracle", g, "episode", {**STRONG, **kw}, "nash"))

    def exp2(g, t, coop=False):
        t.append(J("e2", "nash_prior", g, "episode", LEARN, "nash", "prior"))
        t.append(J("e2", "nash_learn", g, "episode", LEARN, "nash", "learn"))
        t.append(J("e2", "nash_learn_pilot", g, "episode", {**LEARN, "pilot_rate": 0.5}, "nash", "learn"))
        if coop:
            t.append(J("e2", "coop_learn", g, "episode", LEARN, "coop", "learn"))

    def exp4_learn(g, t):
        for pol in ("m2", "causer", "quota"):
            t.append(J("e4", f"weak_{pol}_learn", g, "episode", {**LEARN, **POLICIES[pol]}, "nash", "learn"))

    def exp3_learn(g, t):
        for lay in ("far", "close"):
            for mo in ("nash", "coop"):
                t.append(J("e3", f"{lay}_learn_{mo}", g, "episode", {**LEARN, "layout": lay}, mo, "learn"))

    def osd(g, t):
        for t0 in (0, 1, 2):
            t.append(J("e1", f"osd_t{t0}", g, "osd", H_cfg(2), "nash", t0=t0))

    for g in range(12):
        exp1(g, tiers[1]); exp3_oracle(g, tiers[1]); exp4_oracle(g, tiers[1])
    for g in range(8):
        exp2(g, tiers[2]); osd(g, tiers[2]); exp4_learn(g, tiers[2]); exp3_learn(g, tiers[2])
    for g in range(8, 16):
        exp2(g, tiers[3])
    for g in range(8):
        tiers[3].append(J("e2", "coop_learn", g, "episode", LEARN, "coop", "learn"))
    for g in range(12, 24):
        exp1(g, tiers[3]); exp3_oracle(g, tiers[3]); exp4_oracle(g, tiers[3])
    for g in range(24, 36):
        exp1(g, tiers[4]); exp3_oracle(g, tiers[4]); exp4_oracle(g, tiers[4])
    for g in range(16, 24):
        exp2(g, tiers[4])
    return [j for t in (1, 2, 3, 4) for j in tiers[t]]


def work(args):
    job, outdir = args
    path = os.path.join(outdir, job["name"] + ".json")
    if os.path.exists(path) or time.time() > DEADLINE[0]:
        return None
    try:
        return _work(job, outdir, path)
    except Exception as e:  # never let one bad job kill the run
        import traceback
        open(os.path.join(outdir, job["name"] + ".err"), "w").write(traceback.format_exc())
        return job["name"], -1


def _work(job, outdir, path):
    cfg = Cfg(**job["cfg"])
    model = SolverStageModel(1.0)
    geo = true_geo(job["geo"])
    pool = particle_set_v3(cfg.n)
    seed = 1000 + job["geo"]
    t0 = time.time()
    rec = {k: v for k, v in job.items() if k != "cfg"}
    rec["cfg"] = {k: (str(v) if v == INF else v) for k, v in job["cfg"].items()}
    rec["mult"] = geo["mult"]
    if job["kind"] == "episode":
        summ, log = run_episode(model, cfg, geo, pool, job["mode"], job["info"], seed=seed)
        rec.update(summ)
        rec["log"] = log
    elif job["kind"] == "eps":
        summ, log = run_episode(model, cfg, geo, pool, job["mode"], "oracle", seed=seed)
        e = open_loop_eps(model, cfg, geo, schedule_from_log(log, cfg), objective=job["objective"])
        rec.update(summ)
        rec["eps"] = e
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
    ap.add_argument("--hours", type=float, default=3.4)
    ap.add_argument("--out", default="seq3_results")
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--count_only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.resume_from:
        for f in glob.glob(os.path.join(a.resume_from, "**", "e[1-4]__*.json"), recursive=True):
            d = os.path.join(a.out, os.path.basename(f))
            if not os.path.exists(d):
                shutil.copy(f, d)
    todo = [j for j in plan() if not os.path.exists(os.path.join(a.out, j["name"] + ".json"))]
    print(f"procs={a.procs} jobs_left={len(todo)} budget={a.hours}h (no new job starts after the deadline)", flush=True)
    if a.count_only:
        raise SystemExit
    start = time.time()
    deadline = start + a.hours * 3600
    done = 0
    with Pool(a.procs, initializer=init, initargs=(deadline,)) as pool:
        for r in pool.imap(work, [(j, a.out) for j in todo], chunksize=1):
            if r is None:
                continue
            done += 1
            if done % 10 == 0:
                print(f"[{(time.time()-start)/60:6.1f} min] finished {done}/{len(todo)}  last={r[0]} {r[1]}s", flush=True)
    print(f"finished {done} jobs in {(time.time()-start)/3600:.2f} h. Now run: python analyze3.py --dir {a.out}", flush=True)
