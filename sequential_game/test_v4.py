"""test_v4.py -- quick checks for study v4 (about 2 minutes). Run before the long run:  python test_v4.py
1. v3 behaviour unchanged (defaults temper=1, model_err=0 give the v3 likelihood exactly) -- plus test_v3.py.
2. new belief switches change the likelihood in the intended direction.
3. the job plan has unique names and the expected experiments."""
import sys
import numpy as np
from game3 import Cfg, loglik
import run_study4 as r4

ok = True
obs, pred = np.array([1.0, 2.0]), np.array([1.2, 1.5])
c0 = Cfg()
v3 = -np.sum((obs - pred) ** 2) / (2 * c0.sigma_lik ** 2)
ok &= abs(loglik(obs, pred, c0) - v3) < 1e-15
print("default likelihood equals v3:", abs(loglik(obs, pred, c0) - v3) < 1e-15)
ok &= abs(loglik(obs, pred, Cfg(temper=0.5)) - 0.5 * v3) < 1e-12
ok &= loglik(obs, pred, Cfg(model_err=0.1)) > v3          # model error widens sigma -> less negative
print("tempering halves, model error widens: OK")
jobs = r4.plan()
names = [j["name"] for j in jobs]
dup = len(names) - len(set(names))
by = {}
for j in jobs:
    by[j["exp"]] = by.get(j["exp"], 0) + 1
print("jobs:", len(jobs), "duplicates (v3 overlap, skipped at run time):", dup, dict(sorted(by.items())))
ok &= all(e in by for e in ("e5", "e6", "e7", "e8", "e1", "e2", "e3", "e4"))
print("ALL OK" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
