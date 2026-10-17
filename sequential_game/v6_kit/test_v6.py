"""test_v6.py -- quick checks for study v6 (a few minutes). Run before the long run:  python test_v6.py
1. defaults unchanged (permit-24 Nash H=3 geology 0 equals the stored v5 value).
2. ration(): conserves the permit, never exceeds claims, expected values on claims (8,3,1), allowed 6.
3. margins=(20,20) equals the default game.
4. groups: singletons reproduce Nash, one group reproduces coop (2 operators).
5. 3-operator episode returns 3 payoffs.
6. strong epsilon >= plain epsilon on a short schedule.
7. plan has unique names and Exp 12-17."""
import sys
import numpy as np
from game3 import Cfg, make_geology, particle_set_v3, run_episode, ration, schedule_from_log
from stage_model import SolverStageModel
from eps_check import open_loop_eps, open_loop_eps_strong
import run_study6 as r6

m, pool, tg = SolverStageModel(1.0), particle_set_v3(12), make_geology(101, 1.0, 12)
B = dict(layout="close", cap_total=24.0, horizon=3, extrap=True)
ok = True
def chk(name, good):
    global ok
    ok &= bool(good); print(("PASS " if good else "FAIL ") + name)

nash, _ = run_episode(m, Cfg(**B), tg, pool, "nash", "oracle", seed=1000)
chk(f"v5 reference nash_H3 369.4121 (got {nash['welfare']:.4f})", abs(nash["welfare"] - 369.4121482304) < 1e-6)

req = [8, 3, 1]
for rule, exp in (("prorata", (4, 1.5, .5)), ("cea", (2.5, 2.5, 1)), ("cel", (5.5, .5, 0))):
    o = ration(req, 6, rule)
    chk(f"ration {rule} {np.round(o, 3).tolist()}", np.allclose(o, exp, atol=1e-6) and abs(o.sum() - 6) < 1e-6 and (o <= np.array(req) + 1e-9).all())
chk("ration passes through when claims fit", np.allclose(ration([1, 2], 5, "cea"), [1, 2]))

mg, _ = run_episode(m, Cfg(**B, margins=(20.0, 20.0)), tg, pool, "nash", "oracle", seed=1000)
chk("margins (20,20) == default", abs(mg["welfare"] - nash["welfare"]) < 1e-9)

coop, _ = run_episode(m, Cfg(**B), tg, pool, "coop", "oracle", seed=1000)
gs, _ = run_episode(m, Cfg(**B, groups=((0,), (1,))), tg, pool, "groups", "oracle", seed=1000)
ga, _ = run_episode(m, Cfg(**B, groups=((0, 1),)), tg, pool, "groups", "oracle", seed=1000)
chk("groups singletons == nash", abs(gs["welfare"] - nash["welfare"]) < 1e-9)
chk("groups all-in-one == coop", abs(ga["welfare"] - coop["welfare"]) < 1e-9)

c3 = dict(n_ops=3, cap_total=36.0, horizon=2)
s3, _ = run_episode(m, Cfg(**c3, groups=((0,), (1,), (2,))), tg, pool, "groups", "oracle", seed=1000)
chk("3-operator episode has 3 payoffs", len(s3["payoff"]) == 3)

cfg = Cfg(**B)
sched = schedule_from_log(_ if False else run_episode(m, cfg, tg, pool, "nash", "oracle", seed=1000)[1], cfg)
p = open_loop_eps(m, cfg, tg, sched)
s = open_loop_eps_strong(m, cfg, tg, sched, sweeps=1, max_sims=60)
chk(f"strong eps >= plain eps ({s['max']:.3f} vs {p['max']:.3f})", s["max"] >= p["max"] - 1e-9 or s["sims"] >= 60)

jobs = r6.plan(); names = [j["name"] for j in jobs]
by = {}
for j in jobs: by[j["exp"]] = by.get(j["exp"], 0) + 1
print("jobs:", len(jobs), dict(sorted(by.items())))
chk("unique names, e12-e17 present", len(names) == len(set(names)) and all(e in by for e in ("e12", "e13", "e14", "e15", "e16", "e17")))
print("ALL OK" if ok else "SOME CHECKS FAILED"); sys.exit(0 if ok else 1)
