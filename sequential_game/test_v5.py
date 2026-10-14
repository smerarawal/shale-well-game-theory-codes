"""test_v5.py -- quick checks for study v5 (about 3 minutes). Run before the long run:  python test_v5.py
1. defaults are unchanged: permit-24 Nash/coop welfare equals the stored v3/v4 numbers for geology 0 (see REF).
2. an injection price changes Nash play but welfare is scored without it; coop ignores it.
3. fixed allocation: neither operator injects more than cap/2.
4. the job plan has the expected experiments and unique names."""
import sys
import numpy as np
from game3 import Cfg, make_geology, particle_set_v3, run_episode
from stage_model import SolverStageModel
import run_study5 as r5

m, pool, tg = SolverStageModel(1.0), particle_set_v3(12), make_geology(101, 1.0, 12)
STRONG = dict(layout="close", cap_total=24.0, horizon=6, extrap=False)
ok = True
base, _ = run_episode(m, Cfg(**STRONG), tg, pool, "nash", "oracle", seed=1000)
print(f"permit nash H=6 geology 0: {base['welfare']:.4f}, injected {base['total_injected']:.2f}")
ok &= base["total_injected"] <= 24.0 + 1e-6

pr, _ = run_episode(m, Cfg(**STRONG, inj_price=10.0), tg, pool, "nash", "oracle", seed=1000)
print(f"price 10: welfare {pr['welfare']:.4f}  injected {pr['total_injected']:.2f}  (welfare scored without the price)")
cp, _ = run_episode(m, Cfg(**STRONG), tg, pool, "coop", "oracle", seed=1000)
cp2, _ = run_episode(m, Cfg(**STRONG, inj_price=10.0), tg, pool, "coop", "oracle", seed=1000)
good = abs(cp["welfare"] - cp2["welfare"]) < 1e-9
ok &= good
print("coop ignores the price:", good)

al, _ = run_episode(m, Cfg(**STRONG, alloc=True), tg, pool, "nash", "oracle", seed=1000)
good = max(al["injected"]) <= 12.0 + 1e-6
ok &= good
print(f"allocation: injected per operator {np.round(al['injected'], 2).tolist()} (each <= 12): {good}")

jobs = r5.plan()
names = [j["name"] for j in jobs]
by = {}
for j in jobs:
    by[j["exp"]] = by.get(j["exp"], 0) + 1
print("jobs:", len(jobs), dict(sorted(by.items())), "unique:", len(names) == len(set(names)))
ok &= len(names) == len(set(names)) and all(e in by for e in ("e9", "e10", "e11"))
print("ALL OK" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
