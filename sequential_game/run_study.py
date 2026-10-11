"""
run_study.py -- time-budgeted, resumable Kaggle/CPU study for the sequential Bayesian game (v2).

    python run_study.py --hours 3.3 --out seq_results
Re-run the same command after a crash: finished jobs (one json file each) are skipped.
Use --resume_from <dir> to seed `--out` with results saved from a previous session.

Jobs are ordered so every prefix is a complete, perm-balanced, paired sample:
  tier 1: 2-operator geologies 0..11, all conditions
  tier 2: 3-operator geologies 0..5 (nash/coop with learning + scripted baselines)
  tier 3: 2-operator geologies 12..35
No job STARTS after the deadline; the last jobs finish within ~10 minutes of it.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, glob, json, shutil, time
from multiprocessing import Pool

from game2 import Cfg, make_geology, particle_set, run_episode
from stage_model import SolverStageModel

TRUE_MULTS = [1.0, 0.6, 1.7, 0.7, 1.4, 0.5, 2.0, 0.85, 1.2]   # perm-balanced cycle, off-grid values included
DEADLINE = [None]


def true_geo(idx, n=12):
    return make_geology(101 + idx, TRUE_MULTS[idx % len(TRUE_MULTS)], n)


def cfg_for(nops, m):
    return Cfg(n_ops=nops, liab_mult=m)


def conditions(nops):
    if nops == 2:
        c = [("static", "prior", 1.0), ("bau", "prior", 1.0)]
        c += [(mo, inf, 1.0) for mo in ("nash", "coop") for inf in ("prior", "learn", "oracle")]
        c += [("nash", "learn", 2.0)]
        return c
    return [("static", "prior", 1.0), ("bau", "prior", 1.0), ("nash", "learn", 1.0), ("coop", "learn", 1.0)]


def jobs(outdir):
    plan = []
    def add(nops, g):
        for mo, inf, m in conditions(nops):
            plan.append((nops, g, mo, inf, m))
    for g in range(12): add(2, g)
    for g in range(6): add(3, g)
    for g in range(12, 36): add(2, g)
    return [j for j in plan if not os.path.exists(path(outdir, j))]


def path(outdir, j):
    nops, g, mo, inf, m = j
    return os.path.join(outdir, f"n{nops}_g{g:02d}_{mo}_{inf}_m{m:g}.json")


def work(args):
    j, outdir = args
    if time.time() > DEADLINE[0]:
        return None
    nops, g, mo, inf, m = j
    cfg, model = cfg_for(nops, m), SolverStageModel(1.0)
    t0 = time.time()
    tg = true_geo(g)
    summ, log = run_episode(model, cfg, tg, particle_set(cfg.n), mo, inf, seed=1000 + g)
    rec = dict(nops=nops, geo=g, mult=tg["mult"], mode=mo, info=inf, liab_mult=m, secs=round(time.time() - t0, 1),
               **summ, log=log)
    json.dump(rec, open(path(outdir, j), "w"))
    return j, rec["secs"], rec["welfare"]


def init(deadline):
    DEADLINE[0] = deadline


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=3.3)
    ap.add_argument("--out", default="seq_results")
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.resume_from:
        for f in glob.glob(os.path.join(a.resume_from, "**", "n*_g*.json"), recursive=True):
            d = os.path.join(a.out, os.path.basename(f))
            if not os.path.exists(d):
                shutil.copy(f, d)
    todo = jobs(a.out)
    start = time.time()
    deadline = start + a.hours * 3600
    print(f"procs={a.procs} jobs_left={len(todo)} budget={a.hours}h  (no new job starts after the deadline)", flush=True)
    done = 0
    with Pool(a.procs, initializer=init, initargs=(deadline,)) as pool:
        for r in pool.imap(work, [(j, a.out) for j in todo], chunksize=1):
            if r is None:
                continue
            done += 1
            if done % 5 == 0:
                print(f"[{(time.time()-start)/60:6.1f} min] finished {done}/{len(todo)}  last={r[0]} {r[1]}s", flush=True)
    print(f"finished {done} jobs in {(time.time()-start)/3600:.2f} h. Now run: python analyze.py --dir {a.out}", flush=True)
