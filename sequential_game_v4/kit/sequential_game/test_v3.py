"""test_v3.py -- sanity checks (about 1 minute). Run before the long study:  python test_v3.py
1. game3 with default Cfg reproduces the v2 numbers exactly (geology 0).
2. schedule replay reproduces the played welfare exactly.
3. a permit really curtails injection; 'causer' liability puts the whole penalty on one operator."""
import sys, time
import numpy as np
from game3 import Cfg, make_geology, particle_set_v3, run_episode, schedule_from_log, simulate_schedule
from stage_model import SolverStageModel

m, pool = SolverStageModel(1.0), particle_set_v3(12)
tg, cfg = make_geology(101, 1.0, 12), Cfg()
ref = {("static", "prior"): -722.3757367435082, ("bau", "prior"): -181.74342917414543,
       ("nash", "oracle"): 466.4620638635338, ("coop", "oracle"): 473.74179145292385}
ok = True
for (mode, info), r in ref.items():
    s, log = run_episode(m, cfg, tg, pool, mode, info, seed=1000)
    good = abs(s["welfare"] - r) < 1e-9
    ok &= good
    print(f"v2 reproduction {mode:6s} {info:6s} {s['welfare']:.6f} vs {r:.6f}  {'OK' if good else 'MISMATCH'}")
    if mode == "nash":
        tot = simulate_schedule(m, cfg, tg, schedule_from_log(log, cfg)).sum()
        good = abs(tot - s["welfare"]) < 1e-9
        ok &= good
        print(f"schedule replay {tot:.6f}  {'OK' if good else 'MISMATCH'}")
s, _ = run_episode(m, Cfg(cap_total=24.0, horizon=6, extrap=False), tg, pool, "nash", "oracle", seed=1000)
good = s["total_injected"] <= 24.0 + 1e-6
ok &= good
print(f"permit 24: total injected {s['total_injected']:.2f}  {'OK' if good else 'VIOLATED'}")
s, log = run_episode(m, Cfg(liab_rule="causer"), tg, pool, "bau", "prior", seed=1000)
print(f"causer rule runs, welfare {s['welfare']:.1f}")
print("ALL OK" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
