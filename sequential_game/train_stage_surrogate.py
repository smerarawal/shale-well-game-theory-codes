"""
train_stage_surrogate.py -- train the stage-wise surrogate on make_stage_dataset.py output.

Split is BY EPISODE (80/10/10, fixed split seed): the 6 transitions of an episode share
one geology and are strongly correlated, so a row-level split would leak.

Loss (physical units, see handoff section 6): pressure MSE with extra weight where the
true pressure is high (the game penalty is quadratic above 6) + a direct per-sample
p_max term + a saturation term. Saturation is state, not scored, so it gets a lower weight.

    python train_stage_surrogate.py --data stage_train.npz --out stage_unet.pt --backbone unet --epochs 40
    python train_stage_surrogate.py --data stage_train.npz stage_spe10_train.npz --hybrid prior2 --out h.pt

Several --data files are concatenated before the episode split (e.g. random fields + channelised SPE10 patches).
"""
import argparse
import json
import time

import numpy as np
import torch

from stage_surrogate import BACKBONES, StageNet, P_SCALE, SWC, SOR

SPLIT_SEED = 1234


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def episode_split(E, seed=SPLIT_SEED):
    perm = np.random.default_rng(seed).permutation(E)
    n_tr, n_va = int(0.8 * E), int(0.1 * E)
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


def flatten(d, eps):
    """-> tensors of per-transition rows for the given episodes."""
    k = np.repeat(d["permeability"][eps][:, None], d["p_in"].shape[1], 1)
    phi = np.repeat(d["porosity"][eps][:, None], d["p_in"].shape[1], 1)
    f = lambda a: torch.from_numpy(np.ascontiguousarray(a.reshape(-1, *a.shape[2:]))).float()
    return {"k": f(k), "phi": f(phi), "wr": f(d["well_rate"][eps]), "p_in": f(d["p_in"][eps]),
            "sw_in": f(d["Sw_in"][eps]), "p_out": f(d["p_out"][eps]), "sw_out": f(d["Sw_out"][eps])}


def loss_unroll(net, D, idx, L, hi_w, pmax_w, sat_w):
    """Multi-stage loss: start from the TRUE state at row idx, chain L stage predictions (each fed with the
    previous PREDICTED state, gradients flow through the physics solves) and average the per-stage loss.
    Rows are episode-major with T stages each, so row idx+j is stage j of the same episode."""
    p, sw = D["p_in"][idx], D["sw_in"][idx]
    tot = 0.0
    for j in range(L):
        r = idx + j
        p, sw = net.step(D["k"][idx], D["phi"][idx], D["wr"][r], p, sw)
        b = {"p_out": D["p_out"][r], "sw_out": D["sw_out"][r]}
        tot = tot + loss_terms(p, sw, b, hi_w, pmax_w, sat_w)[0]
    return tot / L


def loss_fn(net, b, hi_w, pmax_w, sat_w):
    p, sw = net.step(b["k"], b["phi"], b["wr"], b["p_in"], b["sw_in"])
    return loss_terms(p, sw, b, hi_w, pmax_w, sat_w)


def loss_terms(p, sw, b, hi_w, pmax_w, sat_w):
    ramp = ((b["p_out"] - 3.0) / 3.0).clamp(0, 1)                       # 0 below p=3, 1 above p=6
    lp = ((1 + hi_w * ramp) * (p - b["p_out"]) ** 2).mean() / P_SCALE ** 2
    pm_p, pm_t = p.flatten(1).amax(1), b["p_out"].flatten(1).amax(1)
    lm = ((pm_p - pm_t) ** 2).mean() / P_SCALE ** 2
    ls = ((sw - b["sw_out"]) ** 2).mean() / (1 - SOR - SWC) ** 2
    return lp + pmax_w * lm + sat_w * ls, lp, lm, ls


@torch.no_grad()
def evaluate(net, data, bs=512):
    net.eval()
    se, pe, ae = [], [], []
    for i in range(0, data["k"].shape[0], bs):
        b = {k: v[i:i + bs] for k, v in data.items()}
        p, sw = net.step(b["k"], b["phi"], b["wr"], b["p_in"], b["sw_in"])
        se.append(((p - b["p_out"]) ** 2).flatten(1).mean(1))
        pe.append(p.flatten(1).amax(1) - b["p_out"].flatten(1).amax(1))
        ae.append(((sw - b["sw_out"]) ** 2).flatten(1).mean(1))
    se, pe, ae = torch.cat(se), torch.cat(pe), torch.cat(ae)
    return {"p_rmse": float(se.sqrt().mean()), "pmax_abs": float(pe.abs().mean()),
            "pmax_bias": float(pe.mean()), "sw_rmse": float(ae.sqrt().mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--hybrid", default="none", choices=["none", "prior", "prior2"])
    ap.add_argument("--n_sub", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="unet", choices=list(BACKBONES))
    ap.add_argument("--unroll", type=int, default=1,
                    help="train on L chained stage predictions (gradients through the solves); 1 = one-step teacher forcing")
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hi_w", type=float, default=3.0, help="extra weight on high-pressure cells")
    ap.add_argument("--pmax_w", type=float, default=5.0, help="weight of the direct p_max loss term")
    ap.add_argument("--sat_w", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    parts = [load(p) for p in a.data]
    d = {k: np.concatenate([q[k] for q in parts]) for k in parts[0] if k != "policy"}
    tr, va, te = episode_split(d["permeability"].shape[0])
    D = {n: {k: v.to(a.device) for k, v in flatten(d, e).items()} for n, e in (("train", tr), ("val", va), ("test", te))}
    stage_time = float(d["elapsed"].mean())
    print(f"episodes train/val/test = {len(tr)}/{len(va)}/{len(te)}; rows = "
          f"{D['train']['k'].shape[0]}/{D['val']['k'].shape[0]}/{D['test']['k'].shape[0]}; stage_time {stage_time}")

    net = StageNet(a.backbone, a.width, a.hybrid, a.n_sub, stage_time).to(a.device)
    n_params = sum(p.numel() for p in net.parameters())
    print("params:", n_params)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    N = D["train"]["k"].shape[0]
    T = d["p_in"].shape[1]
    # start rows for unrolled training: stages 0..T-L of each episode (rows are episode-major, T per episode)
    starts = torch.nonzero((torch.arange(N, device=a.device) % T) <= T - a.unroll).squeeze(1)
    NS = starts.numel()
    steps_per_epoch = (NS + a.bs - 1) // a.bs
    total, warm = a.epochs * steps_per_epoch, 3 * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / total))))

    best, best_ep, t0 = float("inf"), -1, time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        perm = starts[torch.randperm(NS, device=a.device)]
        run = 0.0
        for i in range(0, NS, a.bs):
            idx = perm[i:i + a.bs]
            if a.unroll == 1:
                b = {k: v[idx] for k, v in D["train"].items()}
                loss, *_ = loss_fn(net, b, a.hi_w, a.pmax_w, a.sat_w)
            else:
                loss = loss_unroll(net, D["train"], idx, a.unroll, a.hi_w, a.pmax_w, a.sat_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            run += float(loss) * len(idx)
        v = evaluate(net, D["val"])
        score = v["p_rmse"] + v["pmax_abs"]           # pressure-relevant selection, not saturation
        if score < best:
            best, best_ep = score, ep
            torch.save({"state_dict": net.state_dict(), "backbone": a.backbone, "width": a.width, "hybrid": a.hybrid, "n_sub": a.n_sub,
                        "stage_time": stage_time, "args": vars(a)}, a.out)
        if ep % 5 == 0 or ep == 1:
            print(f"ep {ep:3d} train {run / NS:.5f} | val p_rmse {v['p_rmse']:.4f} pmax_abs {v['pmax_abs']:.4f} "
                  f"sw_rmse {v['sw_rmse']:.4f} | {time.time() - t0:.0f}s", flush=True)

    blob = torch.load(a.out, map_location=a.device)
    net.load_state_dict(blob["state_dict"])
    res = {"best_epoch": best_ep, "params": n_params, "train_seconds": round(time.time() - t0), "unroll": a.unroll,
           "backbone": a.backbone, "hybrid": a.hybrid, "val": evaluate(net, D["val"]), "test": evaluate(net, D["test"])}
    json.dump(res, open(a.out.replace(".pt", "_metrics.json"), "w"), indent=1)
    print("best epoch", best_ep, "| val", res["val"], "\n test", res["test"])


if __name__ == "__main__":
    main()
