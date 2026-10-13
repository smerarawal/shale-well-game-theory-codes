"""
run_study5.py -- study v5: is the permit-commons result real, does it survive uncertainty, and what policy fits it?
Same engine / same results folder as studies v3 and v4; resumable; one json per job.

    python run_study5.py --out seq3_results --resume_from <old results dir(s)> --hours 11
    python run_study5.py --count_only
    python run_study5.py --tiers A,B

Setting for everything here = the "strong" setting of study v3/v4: close wells + shared injection permit of 24
(pro-rata curtailment). Perfect-information baselines for it already exist in the folder (e3 close_cap24_nash / _coop).

  Exp 9   equilibrium quality of the permit result: planner horizon sweep H=2,3,4 (H=6 already exists), more
          best-response rounds, open-loop deviation epsilon on the Nash and the cooperative path.
  Exp 10  permit setting under uncertainty: prior / learn / coop-learn / m2 / causer with the repaired belief (b4).
  Exp 11  policies that fit a capacity commons: a per-unit injection price (a transfer, seen by Nash players only)
          and a fixed equal allocation of the permit, with perfect information and under learning.

Tiers (each prefix is a complete paired sample over geologies):
  A  geologies 0-11 : Exp 9 + Exp 11 (perfect information)       -- cheap, answers "is the gap real?"
  B  geologies 0-11 : Exp 10 + Exp 11 under learning
  C  geologies 12-23: Exp 9 + Exp 11 (perfect information)
  D  geologies 12-23: Exp 10 + Exp 11 under learning
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, glob, shutil, time
from multiprocessing import Pool

import run_study4 as r4
from run_study4 import FULL, Jf
from run_study3 import STRONG, J

LEARN_H = 3                       # planner horizon for the learning arms (see README_STUDY5: cost vs. Exp 9 evidence)
SL = {**STRONG, "horizon": LEARN_H}    # strong setting at the learning horizon
PRICES = (2.0, 5.0, 10.0, 20.0)   # per unit injected volume; revenue is 20 per unit volume
LPRICES = (5.0, 10.0)


def exp9(g, t):
    for H in (2, 3, 4):
        for mo in ("nash", "coop"):
            t.append(Jf("e9", f"{mo}_H{H}", g, {**STRONG, "horizon": H}, mo))
    t.append(Jf("e9", "nash_H6_br6", g, {**STRONG, "br_rounds": 6}, "nash"))
    t.append(J("e9", "eps_nash_H6", g, "eps", STRONG, "nash", "oracle", objective="own", fields=2))
    t.append(J("e9", "eps_coop_H6", g, "eps", STRONG, "coop", "oracle", objective="sum", fields=2))


def exp11_oracle(g, t):
    for p in PRICES:
        t.append(Jf("e11", f"price{p:g}_oracle", g, {**STRONG, "inj_price": p}, "nash"))
    t.append(Jf("e11", "alloc_oracle", g, {**STRONG, "alloc": True}, "nash"))
    t.append(Jf("e11", "alloc_coop_oracle", g, {**STRONG, "alloc": True}, "coop"))


def exp10(g, t):
    nf = 4
    t.append(Jf("e10", "nash_prior", g, {**SL, **FULL}, "nash", "prior", nf))
    t.append(Jf("e10", "nash_learn", g, {**SL, **FULL}, "nash", "learn", nf))
    t.append(Jf("e10", "coop_learn", g, {**SL, **FULL}, "coop", "learn", nf))
    t.append(Jf("e10", "m2_learn", g, {**SL, **FULL, "liab_mult": 2.0}, "nash", "learn", nf))
    t.append(Jf("e10", "causer_learn", g, {**SL, **FULL, "liab_rule": "causer"}, "nash", "learn", nf))


def exp11_learn(g, t):
    for p in LPRICES:
        t.append(Jf("e11", f"price{p:g}_learn", g, {**SL, **FULL, "inj_price": p}, "nash", "learn", 4))
    t.append(Jf("e11", "alloc_learn", g, {**SL, **FULL, "alloc": True}, "nash", "learn", 4))


def plan(tiers="ABCD"):
    T = {k: [] for k in "ABCD"}
    for g in range(12):
        exp9(g, T["A"]); exp11_oracle(g, T["A"])
    for g in range(12):
        exp10(g, T["B"]); exp11_learn(g, T["B"])
    for g in range(12, 24):
        exp9(g, T["C"]); exp11_oracle(g, T["C"])
    for g in range(12, 24):
        exp10(g, T["D"]); exp11_learn(g, T["D"])
    return [j for k in "ABCD" if k in tiers for j in T[k]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=0.0, help="0 = no limit; otherwise no new job starts after this")
    ap.add_argument("--out", default="seq3_results")
    ap.add_argument("--resume_from", nargs="*", default=[])
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--tiers", default="ABCD")
    ap.add_argument("--count_only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for src in a.resume_from:
        for f in glob.glob(os.path.join(src, "**", "e*__*.json"), recursive=True):
            d = os.path.join(a.out, os.path.basename(f))
            if not os.path.exists(d):
                shutil.copy(f, d)
    todo, seen = [], set()
    for j in plan(a.tiers.replace(",", "")):
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
    with Pool(a.procs, initializer=r4.init, initargs=(deadline,)) as pool:
        for r in pool.imap(r4.work, [(j, a.out) for j in todo], chunksize=1):
            if r is None:
                continue
            done += 1
            if done % 5 == 0:
                print(f"[{(time.time()-start)/60:6.1f} min] finished {done}/{len(todo)}  last={r[0]} {r[1]}s", flush=True)
    print(f"finished {done} jobs in {(time.time()-start)/3600:.2f} h. Now run: python analyze5.py --dir {a.out}", flush=True)
