"""
run_study6.py -- study v6: why is there a permit gap, how does it scale, which instrument fits, who gains?
Same engine / results folder as studies v3-v5; resumable; one json per job. All runs use perfect information
(oracle) and the study-v5 H=3 permit setting (close wells, permit 24, planner horizon 3, no extrapolation) unless
stated, so they are cheap (about 10-40 s each) and directly comparable with Exp 9 of study v5.

    python run_study6.py --out seq3_results --resume_from <old results dir(s)> --tiers A
    python run_study6.py --count_only

  Exp 12  scarcity curve       : permit cap in {12,16,20,24,32,48}, Nash vs coop. Is the commons gap a function of
                                 scarcity (zero when the cap never binds, zero when nothing is left to fight over)?
  Exp 13  rationing rules      : the permit as a CLAIMS PROBLEM. proportional (current), constrained equal awards (cea),
                                 constrained equal losses (cel), fixed equal allocation. Tests whether the gap is a
                                 claims-inflation race (drilling / rate) or a timing race.
  Exp 14  sensitivity          : penalty weight, pressure cap and well cost each halved/doubled (one at a time):
                                 does the gap and the ranking of instruments survive other payoff scales?
  Exp 15  heterogeneity        : asymmetric margins (24,16) and (28,12): price vs quantity instruments (Weitzman).
                                 Rigid shares should lose more when operators differ; a price should not.
  Exp 16  three operators      : coalition-structure runs -> gamma-characteristic function, Shapley value, core.
  Exp 17  certification        : stronger deviation search (pair moves) on the H=3 permit Nash and coop paths:
                                 how large is epsilon really, relative to the gap?

Tiers (each prefix is a complete paired sample over geologies):
  A  geologies 0-11  : Exp 12, 13, 15            (cheap; answers scarcity / rules / heterogeneity)
  B  geologies 0-7   : Exp 14;  geologies 0-11 : Exp 16
  C  geologies 0-11  : Exp 17                    (slow: pair-move deviation search, ~10 min per job)
  D  geologies 12-23 : Exp 12, 13, 15            (doubles the sample for tier A)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, glob, shutil, time
from multiprocessing import Pool

import run_study4 as r4
from run_study4 import Jf
from run_study3 import STRONG, J

BASE = {**STRONG, "horizon": 3}                     # close layout, permit 24, planner H=3 (extrap off) = study v5 Exp 9 at H=3
CAPS = (12.0, 16.0, 20.0, 24.0, 32.0, 48.0)         # 48 ~ never binds (max possible volume with 8 wells x 6 stages)
RULE_CAPS = (16.0, 24.0)
VARIANTS = {"lam10": {"lam": 10.0}, "lam90": {"lam": 90.0}, "pcap4": {"p_cap": 4.0}, "pcap8": {"p_cap": 8.0},
            "cw9": {"c_well": 9.0}, "cw36": {"c_well": 36.0}}
HET = {"het24_16": (24.0, 16.0), "het28_12": (28.0, 12.0)}   # mean margin 20, as in the symmetric game
C3 = dict(n_ops=3, cap_total=36.0, horizon=3)       # 3 operators, v2 layout, permit 36 (same per-operator scarcity as 24 for two)
PARTS = {"123": ((0, 1, 2),), "12_3": ((0, 1), (2,)), "13_2": ((0, 2), (1,)), "23_1": ((1, 2), (0,)),
         "1_2_3": ((0,), (1,), (2,))}


def exp12(g, t):
    for cap in CAPS:
        for mo in ("nash", "coop"):
            t.append(Jf("e12", f"cap{cap:g}_{mo}", g, {**BASE, "cap_total": cap}, mo))


def exp13(g, t):
    for cap in RULE_CAPS:
        t.append(Jf("e13", f"cap{cap:g}_alloc", g, {**BASE, "cap_total": cap, "alloc": True}, "nash"))
        t.append(Jf("e13", f"cap{cap:g}_cea", g, {**BASE, "cap_total": cap, "rule": "cea"}, "nash"))
        t.append(Jf("e13", f"cap{cap:g}_cel", g, {**BASE, "cap_total": cap, "rule": "cel"}, "nash"))


def exp14(g, t):
    for var, kw in VARIANTS.items():
        b = {**BASE, **kw}
        t.append(Jf("e14", f"{var}_nash", g, b, "nash"))
        t.append(Jf("e14", f"{var}_coop", g, b, "coop"))
        t.append(Jf("e14", f"{var}_alloc", g, {**b, "alloc": True}, "nash"))
        t.append(Jf("e14", f"{var}_cea", g, {**b, "rule": "cea"}, "nash"))


def exp15(g, t):
    for het, mg in HET.items():
        b = {**BASE, "margins": mg}
        t.append(Jf("e15", f"{het}_nash", g, b, "nash"))
        t.append(Jf("e15", f"{het}_coop", g, b, "coop"))
        t.append(Jf("e15", f"{het}_alloc", g, {**b, "alloc": True}, "nash"))
        t.append(Jf("e15", f"{het}_cea", g, {**b, "rule": "cea"}, "nash"))
        t.append(Jf("e15", f"{het}_price5", g, {**b, "inj_price": 5.0}, "nash"))
        t.append(Jf("e15", f"{het}_price10", g, {**b, "inj_price": 10.0}, "nash"))


def exp16(g, t):
    for part, groups in PARTS.items():
        t.append(Jf("e16", f"part_{part}", g, {**C3, "groups": groups}, "groups"))


def exp17(g, t):
    t.append(J("e17", "eps_nash_H3", g, "eps_strong", BASE, "nash", "oracle", objective="own", fields=2, sweeps=3, max_sims=1200))
    t.append(J("e17", "eps_coop_H3", g, "eps_strong", BASE, "coop", "oracle", objective="sum", fields=2, sweeps=3, max_sims=2500))


def plan(tiers="ABCD"):
    T = {k: [] for k in "ABCD"}
    for g in range(12):
        exp12(g, T["A"]); exp13(g, T["A"]); exp15(g, T["A"])
    for g in range(8):
        exp14(g, T["B"])
    for g in range(12):
        exp16(g, T["B"])
    for g in range(12):
        exp17(g, T["C"])
    for g in range(12, 24):
        exp12(g, T["D"]); exp13(g, T["D"]); exp15(g, T["D"])
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
            if done % 10 == 0:
                print(f"[{(time.time()-start)/60:6.1f} min] finished {done}/{len(todo)}  last={r[0]} {r[1]}s", flush=True)
    print(f"finished {done} jobs in {(time.time()-start)/3600:.2f} h. Now run: python analyze6.py --dir {a.out}", flush=True)
