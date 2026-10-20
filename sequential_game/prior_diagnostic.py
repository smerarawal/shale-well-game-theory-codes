"""
prior_diagnostic.py -- how good is the physics prior ALONE (no learned part)?

Compares, on stored solver transitions (one stage each, teacher-forced), the pressure from
physics_prior.pressure_prior against the solver's p_out for:
  * lambda frozen at Sw_in          (what the hybrid sees at inference, pass 1)
  * lambda frozen at the TRUE midpoint (Sw_in + Sw_out)/2   (oracle: upper bound for a
    'learned saturation + physics pressure' hybrid)
  * lambda at the true Sw_out       (oracle, other end)
and sweeps the number of substeps. Reports field RMSE, p_max error/bias and the penalty error.

    python prior_diagnostic.py --data stage_eval_game.npz
"""
import argparse

import numpy as np
import torch

from physics_prior import pressure_prior

pen = lambda pm: 30.0 * np.maximum(0.0, pm - 6.0) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_sub", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--max_rows", type=int, default=1500)
    a = ap.parse_args()
    d = np.load(a.data)
    E, T = d["well_rate"].shape[:2]
    rep = lambda x: np.repeat(x[:, None], T, 1).reshape(E * T, *x.shape[1:])
    fl = lambda x: x.reshape(E * T, *x.shape[2:])
    sel = np.random.default_rng(0).permutation(E * T)[:a.max_rows]
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x[sel])).double()
    k, phi = t(rep(d["permeability"])), t(rep(d["porosity"]))
    wr, p_in, s_in, s_out, p_out = t(fl(d["well_rate"])), t(fl(d["p_in"])), t(fl(d["Sw_in"])), t(fl(d["Sw_out"])), t(fl(d["p_out"]))
    pm_t = p_out.flatten(1).amax(1).numpy()
    print(f"{len(sel)} transitions; true p_max mean {pm_t.mean():.2f}; stage_time {d['elapsed'].mean():.3f}")
    print("| prior | n_sub | field RMSE | p_max bias | p_max abs | penalty abs |\n|---|---|---|---|---|---|")
    base = ((p_in - p_out) ** 2).flatten(1).mean(1).sqrt().mean()
    pm_in = p_in.flatten(1).amax(1).numpy()
    print(f"| p_in carried (no physics) | - | {base:.4f} | {(pm_in - pm_t).mean():+.3f} | {np.abs(pm_in - pm_t).mean():.3f} | {np.abs(pen(pm_in) - pen(pm_t)).mean():.2f} |")
    for nm, sw, se in (("lambda frozen Sw_in", s_in, None), ("lambda frozen true mid", 0.5 * (s_in + s_out), None),
                       ("lambda frozen true Sw_out", s_out, None),
                       ("lambda linear Sw_in->true Sw_out (oracle)", s_in, s_out)):
        for ns in a.n_sub:
            p = pressure_prior(k, phi, sw, wr, p_in, float(d["elapsed"].mean()), n_sub=ns, sw_end=se)
            pm = p.flatten(1).amax(1).numpy()
            r = ((p - p_out) ** 2).flatten(1).mean(1).sqrt().mean()
            print(f"| {nm} | {ns} | {r:.4f} | {(pm - pm_t).mean():+.3f} | {np.abs(pm - pm_t).mean():.3f} | {np.abs(pen(pm) - pen(pm_t)).mean():.2f} |")


if __name__ == "__main__":
    main()
