"""
train_e2c_v2.py -- trajectory training for E2CTrajectoryV2 with the two
E2C loss terms:
  (a) autoencoding:  decode(encode(x_t)) ~ x_t at every real frame
  (b) prediction:    z_0 -> transition x (T-1) -> decode each z_t ~ x_t

Variants (--variant):
  pooled       exactly the handoff spec: encoder sees (p, s) only; rock/well
               information reaches the transition only via the 16-d
               globally-average-pooled context vector.
  spatial_enc  DEVIATION from the spec, for diagnosis: encoder also sees
               the static channels (perm, poro, well_mask), so z_0 can
               carry well position. Everything else identical.

Loss is inverse-variance weighted per channel (pressure normalised to unit
std; saturation divided by its std inside the loss).
"""
import argparse
import time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

from model_zoo_v3_e2c_v2 import E2CTrajectoryV2, E2CEncoderV2


def load(path):
    d = np.load(path)
    P, S = d["pressure_traj"], d["saturation_traj"]
    p_scale = float(P.std())
    s_std = float(S.std())
    static = np.stack([
        (d["permeability"] - d["permeability"].mean()) / (d["permeability"].std() + 1e-8),
        (d["porosity"] - d["porosity"].mean()) / (d["porosity"].std() + 1e-8),
        d["well_mask"] / (np.abs(d["well_mask"]).max() + 1e-8)], axis=1)
    X = np.stack([P / p_scale, S], axis=2)  # [N,T,2,H,W]
    return (torch.tensor(X, dtype=torch.float32), torch.tensor(static, dtype=torch.float32),
            p_scale, s_std)


class Runner:
    def __init__(self, model, variant):
        self.m, self.variant = model, variant

    def enc(self, x, S):
        if self.variant == "spatial_enc":
            x = torch.cat([x, S], dim=1)
        return self.m.encoder(x)

    def rollout(self, x0, S, n):
        m = self.m
        ctx = m.context_encoder(S)
        dt = torch.ones(x0.shape[0], 1, device=x0.device)
        z = self.enc(x0, S)
        zs = [z]
        for _ in range(n):
            z = m.transition(z, ctx, dt)
            zs.append(z)
        return zs


def swap_norms(module):
    """DIAGNOSTIC DEVIATION: BatchNorm2d -> GroupNorm, BatchNorm1d -> LayerNorm
    (batch-independent, so train/eval behave identically)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            c = child.num_features
            setattr(module, name, nn.GroupNorm(8 if c % 8 == 0 else 1, c))
        elif isinstance(child, nn.BatchNorm1d):
            setattr(module, name, nn.LayerNorm(child.num_features))
        else:
            swap_norms(child)


def chan_loss(pred, tgt, s_std):
    lp = ((pred[:, 0] - tgt[:, 0]) ** 2).mean()
    ls = (((pred[:, 1] - tgt[:, 1]) / s_std) ** 2).mean()
    return lp + ls, lp.item(), ls.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="small_traj.npz")
    ap.add_argument("--variant", choices=["pooled", "spatial_enc"], default="pooled")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--lam_pred", type=float, default=1.0)
    ap.add_argument("--norm", choices=["bn", "gn"], default="bn")
    ap.add_argument("--batch", type=int, default=0, help="0 = full batch")
    ap.add_argument("--n_test", type=int, default=0, help="hold out the last n samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    X, S, p_scale, s_std = load(a.data)
    Xte = Ste = None
    if a.n_test:
        Xte, Ste = X[-a.n_test:], S[-a.n_test:]
        X, S = X[:-a.n_test], S[:-a.n_test]
    N, T, _, H, W = X.shape
    print(f"data N={N} T={T} {H}x{W} | p_scale={p_scale:.3f} s_std={s_std:.4f} | variant={a.variant}", flush=True)

    m = E2CTrajectoryV2(H=H, W=W, latent_dim=a.latent)
    if a.variant == "spatial_enc":
        m.encoder = E2CEncoderV2(5, H, W, a.latent)
    if a.norm == "gn":
        swap_norms(m)
    run = Runner(m, a.variant)
    print("params:", sum(p.numel() for p in m.parameters()), flush=True)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    t0 = time.time()
    bs = a.batch or N
    for ep in range(1, a.epochs + 1):
        perm = torch.randperm(N)
        tot = {"loss": 0.0, "ae": 0.0, "pr": 0.0, "p": 0.0, "s": 0.0}
        nb = 0
        for b0 in range(0, N, bs):
            idx = perm[b0:b0 + bs]
            if len(idx) < 2:
                continue
            Xb, Sb = X[idx], S[idx]
            m.train()
            opt.zero_grad()
            # (a) autoencoding at every real frame
            ae = 0.0
            for t in range(T):
                rec = m.decode(run.enc(Xb[:, t], Sb))
                l, _, _ = chan_loss(rec, Xb[:, t], s_std)
                ae = ae + l / T
            # (b) prediction by latent rollout from frame 0
            zs = run.rollout(Xb[:, 0], Sb, T - 1)
            pr, lp_, ls_ = 0.0, 0.0, 0.0
            for t in range(1, T):
                l, lp, ls = chan_loss(m.decode(zs[t]), Xb[:, t], s_std)
                pr = pr + l / (T - 1)
                lp_ += lp / (T - 1); ls_ += ls / (T - 1)
            loss = ae + a.lam_pred * pr
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            tot["loss"] += loss.item(); tot["ae"] += float(ae.detach()); tot["pr"] += float(pr.detach())
            tot["p"] += lp_; tot["s"] += ls_; nb += 1
        sched.step()
        if ep % a.log_every == 0 or ep == 1:
            print(f"ep {ep:4d} loss={tot['loss']/nb:.4f} ae={tot['ae']/nb:.4f} pred={tot['pr']/nb:.4f} "
                  f"(p={tot['p']/nb:.4f}, s={tot['s']/nb:.4f}) {time.time()-t0:.0f}s", flush=True)

    def report(tag, train_mode, X, S):
        m.train(train_mode)
        with torch.no_grad():
            zs = run.rollout(X[:, 0], S, T - 1)
            pred = m.decode(zs[-1])
            ae_pred = m.decode(run.enc(X[:, -1], S))
            tgt = X[:, -1]

            def rel(pr, tg):
                return ((pr - tg).flatten(1).norm(dim=1) / (tg.flatten(1).norm(dim=1) + 1e-12)).mean().item()

            pp, tp = pred[:, 0] * p_scale, tgt[:, 0] * p_scale
            print(f"\n=== {tag} | variant={a.variant} norm={a.norm} epochs={a.epochs} ===")
            print(f"final frame rel L2  pressure={rel(pred[:,0], tgt[:,0]):.4f}  saturation={rel(pred[:,1], tgt[:,1]):.4f}")
            print(f"AE-only (no transition)  pressure={rel(ae_pred[:,0], tgt[:,0]):.4f}  saturation={rel(ae_pred[:,1], tgt[:,1]):.4f}")
            print(f"true pressure range [{tp.min():.2f}, {tp.max():.2f}] | pred [{pp.min():.2f}, {pp.max():.2f}]")
            print("per-frame rel L2 p:", [round(rel(m.decode(zs[t])[:, 0], X[:, t, 0]), 3) for t in range(T)],
                  "s:", [round(rel(m.decode(zs[t])[:, 1], X[:, t, 1]), 3) for t in range(T)])
            print("latent norms by step:", [round(z.norm(dim=1).mean().item(), 2) for z in zs])

    report("TRAIN SET, EVAL MODE", False, X, S)
    if Xte is not None:
        report("HELD-OUT TEST SET, EVAL MODE", False, Xte, Ste)
    if a.norm == "bn":
        report("TRAIN SET, TRAIN MODE (batch stats; diagnostic only)", True, X, S)


if __name__ == "__main__":
    main()
