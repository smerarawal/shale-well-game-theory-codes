"""
validate_stage_surrogate.py -- payoff-relevant validation of a stage surrogate against the
real solver (handoff plan step 5).

For each episode the surrogate is rolled out from the game's initial state (p=0, Sw=0.2)
using the SAME well/rate sequence the solver saw, chaining its own predictions. The
true states are the solver rollout stored by make_stage_dataset.py. Reported per stage k=1..6:
  * p_max error (signed, absolute) and the game-penalty error 30*max(0,p_max-6)^2,
  * pressure error at well cells (the noisy observation that drives the belief),
  * field pressure RMSE and saturation RMSE,
stratified by permeability multiplier (low <= 0.7, mid, high >= 1.4) and well count.
Also one-step (teacher-forced) errors from stored states, for comparison.

    python validate_stage_surrogate.py --ckpt stage_unet.pt --train_data stage_train.npz \
        --eval_data stage_eval_game.npz --tol 2.0
"""
import argparse
import json

import numpy as np
import torch

from stage_surrogate import StageNet, StageSurrogate
from train_stage_surrogate import episode_split, load

CAP, LAM = 6.0, 30.0
pen = lambda pm: LAM * np.maximum(0.0, pm - CAP) ** 2


@torch.no_grad()
def rollout(net, d, eps):
    """-> predicted p_out, Sw_out as numpy [E,T,n,n] from chained predictions."""
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
    k, phi, wr = t(d["permeability"][eps]), t(d["porosity"][eps]), t(d["well_rate"][eps])
    E, T, n, _ = wr.shape
    p, sw = torch.zeros(E, n, n), torch.full((E, n, n), 0.2)
    P, S = [], []
    for s in range(T):
        p, sw = net.step(k, phi, wr[:, s], p, sw)
        P.append(p.numpy()); S.append(sw.numpy())
    return np.stack(P, 1), np.stack(S, 1)


@torch.no_grad()
def one_step(net, d, eps):
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
    E, T = d["well_rate"][eps].shape[:2]
    k = t(np.repeat(d["permeability"][eps][:, None], T, 1).reshape(E * T, *d["permeability"].shape[1:]))
    phi = t(np.repeat(d["porosity"][eps][:, None], T, 1).reshape(E * T, *d["porosity"].shape[1:]))
    f = lambda key: t(d[key][eps].reshape(E * T, *d[key].shape[2:]))
    p, sw = net.step(k, phi, f("well_rate"), f("p_in"), f("Sw_in"))
    return p.numpy().reshape(E, T, *p.shape[1:]), sw.numpy().reshape(E, T, *sw.shape[1:])


def stage_metrics(pp, ps, d, eps):
    """pp, ps: predicted [E,T,n,n]. Returns dict of per-stage arrays [T]."""
    pt, st, wr = d["p_out"][eps], d["Sw_out"][eps], d["well_rate"][eps]
    E, T = pt.shape[:2]
    pm_t, pm_p = pt.reshape(E, T, -1).max(-1), pp.reshape(E, T, -1).max(-1)
    dpm = pm_p - pm_t
    dpen = pen(pm_p) - pen(pm_t)
    wmask = wr > 0
    well_rmse = np.array([np.sqrt(((pp[:, s] - pt[:, s])[wmask[:, s]] ** 2).mean()) for s in range(T)])
    return {"pmax_true_mean": pm_t.mean(0), "pmax_bias": dpm.mean(0), "pmax_abs": np.abs(dpm).mean(0),
            "pmax_abs_p95": np.percentile(np.abs(dpm), 95, axis=0),
            "penalty_abs": np.abs(dpen).mean(0), "penalty_bias": dpen.mean(0), "penalty_true_mean": pen(pm_t).mean(0),
            "well_rmse": well_rmse,
            "field_rmse": np.sqrt(((pp - pt) ** 2).reshape(E, T, -1).mean(-1)).mean(0),
            "sw_rmse": np.sqrt(((ps - st) ** 2).reshape(E, T, -1).mean(-1)).mean(0),
            "frac_pen_err_le_tol": None}


def run(net, d, eps, name, tol, lines, out):
    pp, ps = rollout(net, d, eps)
    m = stage_metrics(pp, ps, d, eps)
    dpen = np.abs(pen(pp.reshape(len(eps), pp.shape[1], -1).max(-1)) - pen(d["p_out"][eps].reshape(len(eps), pp.shape[1], -1).max(-1)))
    m["frac_pen_err_le_tol"] = (dpen <= tol).mean(0)
    op, _ = one_step(net, d, eps)
    ms = stage_metrics(op, op * 0 + 0.2, d, eps)
    out[name] = {k: (v.tolist() if v is not None else None) for k, v in m.items()}
    out[name + "_onestep"] = {k: v.tolist() for k, v in ms.items() if v is not None and k != "sw_rmse"}
    fmt = lambda a: " | ".join(f"{x:.3f}" for x in a)
    lines += [f"\n### {name} ({len(eps)} episodes)\n", "| metric | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |", "|---|---|---|---|---|---|---|"]
    for key, lab in (("pmax_true_mean", "true p_max (mean)"), ("pmax_bias", "p_max bias (rollout)"),
                     ("pmax_abs", "p_max |err| (rollout)"), ("pmax_abs_p95", "p_max |err| p95"),
                     ("penalty_abs", f"penalty |err| (welfare units)"), ("frac_pen_err_le_tol", f"frac penalty err <= {tol}"),
                     ("well_rmse", "well-cell RMSE"), ("field_rmse", "field p RMSE"), ("sw_rmse", "Sw RMSE")):
        lines.append(f"| {lab} | {fmt(m[key])} |")
    lines.append(f"| p_max |err| (one-step, teacher-forced) | {fmt(ms['pmax_abs'])} |")
    # strata
    mult, nw = d["mult"][eps], d["n_wells"][eps]
    lines += ["\nStrata (p_max |err| / penalty |err| at k=6 and mean over k=1..6):\n", "| stratum | n | pmax err k=6 | pen err k=6 | pmax err mean k | pen err mean k |", "|---|---|---|---|---|---|"]
    strata = {"mult <= 0.7": mult <= 0.7, "0.7 < mult < 1.4": (mult > 0.7) & (mult < 1.4), "mult >= 1.4": mult >= 1.4,
              "wells at k=6 <= 4": nw[:, -1] <= 4, "wells at k=6 in 5..6": (nw[:, -1] >= 5) & (nw[:, -1] <= 6), "wells at k=6 >= 7": nw[:, -1] >= 7}
    pm_t = d["p_out"][eps].reshape(len(eps), pp.shape[1], -1).max(-1)
    pm_p = pp.reshape(len(eps), pp.shape[1], -1).max(-1)
    for nm, msk in strata.items():
        if msk.sum() < 3:
            continue
        e, pe = np.abs(pm_p[msk] - pm_t[msk]), np.abs(pen(pm_p[msk]) - pen(pm_t[msk]))
        lines.append(f"| {nm} | {int(msk.sum())} | {e[:, -1].mean():.3f} | {pe[:, -1].mean():.2f} | {e.mean():.3f} | {pe.mean():.2f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_data", default=None, help="held-out test episodes of this file are used")
    ap.add_argument("--eval_data", default=None, help="evaluation episodes (all used), e.g. game geologies or SPE10")
    ap.add_argument("--eval_name", default="game geologies true_geo(0..23)", help="label for the eval data in the report")
    ap.add_argument("--tol", type=float, default=2.0, help="penalty-error tolerance in welfare units (proposal)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    blob = torch.load(a.ckpt, map_location="cpu")
    net = StageNet.from_blob(blob)
    net.load_state_dict(blob["state_dict"])
    net.eval()

    # interface consistency: advance() must equal the batched step used above
    d0 = load(a.eval_data or a.train_data)
    sur = StageSurrogate(net=net, stage_time=blob["stage_time"])
    geo = {"k": d0["permeability"][0].astype(float), "phi": d0["porosity"][0].astype(float)}
    wells = [(int(i), int(j), float(d0["well_rate"][0, 0][i, j])) for i, j in zip(*np.nonzero(d0["well_rate"][0, 0]))]
    nxt, el = sur.advance(geo, sur.initial_state(geo["k"].shape[0]), wells)
    ref, _ = rollout(net, d0, np.array([0]))
    assert np.allclose(nxt["p"], ref[0, 0], atol=1e-4), "advance() disagrees with batched rollout"

    lines, out = [f"# Stage surrogate validation: {a.ckpt}\n", f"Penalty = 30*max(0, p_max-6)^2; tolerance proposal: {a.tol} welfare units.\n"], {}
    if a.train_data:
        d = load(a.train_data)
        _, _, te = episode_split(d["permeability"].shape[0])
        run(net, d, te, "held-out test episodes (random policies)", a.tol, lines, out)
    if a.eval_data:
        d = load(a.eval_data)
        run(net, d, np.arange(d["permeability"].shape[0]), a.eval_name, a.tol, lines, out)
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        open(a.out + ".md", "w").write(txt + "\n")
        json.dump(out, open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
