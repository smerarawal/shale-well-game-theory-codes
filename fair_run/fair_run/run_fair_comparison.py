"""
run_fair_comparison.py -- the FINAL fair, resumable comparison of all 13
surrogates.

PROTOCOL (identical for every architecture)
  data      one dataset, fixed 80/10/10 train/val/test split (split seed fixed).
  sweep     pilot runs (seed 0, --pilot_epochs) over an LR grid; the two
            physics-informed models also sweep the physics weight. Winner =
            best VALIDATION balanced error. The test set is never touched here.
  final     --seeds full runs (--epochs) at the winning setting. Best-VALIDATION
            checkpoint is restored before the single test evaluation.
  metrics   all in raw physical units: relative L2 for pressure, saturation,
            combined (stacked channels, as in the earlier table) and
            "balanced" = mean(pressure, saturation), which is the selection
            metric (combined is dominated by the larger-magnitude pressure).
  speed     NOT measured here (workers share the machine); run
            benchmark_inference.py afterwards, sequentially.

RESUMABILITY
  * every sweep/final run writes a JSON when finished; finished runs are skipped.
  * unfinished runs checkpoint every --ckpt_every epochs and resume exactly
    (model, optimiser, scheduler, best-so-far, history, per-epoch data order).
  * --time_budget_hours: workers stop gracefully (checkpoint saved, exit code 3)
    before a Kaggle session limit, so the notebook ends normally and its output
    is kept. Continue in a new session with --import_dir <previous output>.

TWO GPUs: start two workers (--worker 0/1 --n_workers 2), each pinned with
CUDA_VISIBLE_DEVICES. Architectures are assigned to workers by a deterministic
greedy cost balance; all of one architecture's jobs run on one worker.

    python run_fair_comparison.py --out final_run --epochs 300 --seeds 3
"""
import argparse
import glob
import json
import math
import os
import random
import shutil
import sys
import time

import numpy as np
import torch

from model_zoo_v4 import make_all_architectures_v4, ARCH_COST, PHYSICS_ARCHS
from physics_residual_v3 import pressure_residual
from train_utils_v3 import compute_channel_weights, weighted_channel_loss, relative_l2_per_sample, count_parameters

LR_GRID = [3e-4, 1e-3, 3e-3]
SLOW_ARCHS = {"finn", "gns", "gns_traj"}  # slow and not contenders: capped at --slow_seeds seeds (documented)
PW_GRID = [3.0, 10.0, 30.0, 100.0]  # residual is ~0.01 vs main loss ~0.1: weights <1 are inert (checked)


class Paused(Exception):
    pass


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def atomic_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def atomic_torch(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def log(worker, msg):
    print(f"[w{worker} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def import_previous(src, dst):
    n = 0
    for sub in ("data", "sweep", "results", "weights", "ckpt"):
        for root, _, files in os.walk(os.path.join(src, sub)):
            for fn in files:
                if fn.endswith(".tmp"):
                    continue
                s = os.path.join(root, fn)
                d = os.path.join(dst, os.path.relpath(s, src))
                if not os.path.exists(d):
                    os.makedirs(os.path.dirname(d), exist_ok=True)
                    shutil.copy2(s, d)
                    n += 1
    return n


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_data(path, split_seed=1234):
    d = np.load(path)
    x = np.stack([d["permeability"], d["porosity"], d["well_mask"]], axis=1).astype(np.float32)
    y = np.stack([d["final_pressure"], d["final_saturation"]], axis=1).astype(np.float32)
    yt = np.stack([d["pressure_traj"], d["saturation_traj"]], axis=2).astype(np.float32)
    assert np.allclose(yt[:, -1], y, atol=1e-5), "final frame != last trajectory snapshot"
    N = x.shape[0]
    perm = np.random.default_rng(split_seed).permutation(N)
    n_tr, n_va = int(0.8 * N), int(0.1 * N)
    idx = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
    tr = idx["train"]
    stats = {
        "in_mean": [float(x[tr, 0].mean()), float(x[tr, 1].mean()), 0.0],
        "in_std": [float(x[tr, 0].std()) + 1e-8, float(x[tr, 1].std()) + 1e-8, float(np.abs(x[tr, 2]).max())],
        "p_mean": float(y[tr, 0].mean()),
        "p_std": float(y[tr, 0].std()),
    }
    wp, ws = compute_channel_weights(y[tr, 0], y[tr, 1])
    return x, y, yt, idx, stats, wp, ws


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, X, Y, bs=128):
    model.eval()
    rc, rp, rs = [], [], []
    for i in range(0, X.shape[0], bs):
        pred, y = model(X[i:i + bs]), Y[i:i + bs]
        rc.append(relative_l2_per_sample(pred, y))
        rp.append(relative_l2_per_sample(pred[:, 0:1], y[:, 0:1]))
        rs.append(relative_l2_per_sample(pred[:, 1:2], y[:, 1:2]))
    rc, rp, rs = (torch.cat(v).cpu().numpy() for v in (rc, rp, rs))
    bal = 0.5 * (rp + rs)
    f = lambda a: float(a) if np.isfinite(a) else float("inf")
    return {
        "rel_combined": f(rc.mean()), "rel_pressure": f(rp.mean()), "rel_saturation": f(rs.mean()),
        "balanced": f(bal.mean()), "balanced_median": f(np.median(bal)),
        "balanced_p95": f(np.percentile(bal, 95)), "balanced_max": f(bal.max()),
    }


# ---------------------------------------------------------------------------
# one training run (sweep pilot or final)
# ---------------------------------------------------------------------------
def train_run(ctx, arch, lr, pw, epochs, seed, ckpt_path):
    dev = ctx["device"]
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = ctx["zoo"][arch]().to(dev)
    n_params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warm = max(1, min(3, epochs // 10))
    lam = lambda e: (e + 1) / warm if e < warm else max(0.01, 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, epochs - warm))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)

    state = {"epoch": 0, "best_val": float("inf"), "best_epoch": -1, "best_state": None, "history": [], "elapsed": 0.0}
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        state = ck["state"]
        log(ctx["worker"], f"  resumed {os.path.basename(ckpt_path)} at epoch {state['epoch']}/{epochs}")

    X, Y, YT = ctx["X"]["train"], ctx["Y"]["train"], ctx["YT"]["train"]
    Xv, Yv = ctx["X"]["val"], ctx["Y"]["val"]
    wp, ws, bs = ctx["wp"], ctx["ws"], ctx["batch_size"]
    has_traj = model.has_traj
    use_phys = pw > 0
    N = X.shape[0]
    t_start = time.time()

    def save_ckpt():
        state["elapsed"] += time.time() - t_start_ref[0]
        t_start_ref[0] = time.time()
        atomic_torch({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                      "state": state}, ckpt_path)

    t_start_ref = [time.time()]
    for epoch in range(state["epoch"], epochs):
        model.train()
        g = torch.Generator(); g.manual_seed(seed * 100003 + epoch)
        perm = torch.randperm(N, generator=g).to(dev)
        tot, nb = 0.0, 0
        for b0 in range(0, N, bs):
            idx = perm[b0:b0 + bs]
            if idx.numel() < 2:
                continue
            x, y = X[idx], Y[idx]
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss, _, _ = weighted_channel_loss(pred, y, wp, ws)
            if has_traj:
                loss = loss + model.traj_loss(x, YT[idx], wp, ws)
            if use_phys:
                loss = loss + pw * pressure_residual(pred, x)
            if not torch.isfinite(loss):
                return {"status": "diverged", "n_params": n_params, "best_val": float("inf"),
                        "epoch_reached": epoch}, None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        state["epoch"] = epoch + 1

        if (epoch + 1) % ctx["eval_every"] == 0 or epoch + 1 == epochs:
            v = evaluate(model, Xv, Yv)
            state["history"].append({"epoch": epoch + 1, "train_loss": tot / max(nb, 1), **v})
            if v["balanced"] < state["best_val"]:
                state["best_val"], state["best_epoch"] = v["balanced"], epoch + 1
                state["best_state"] = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            if (epoch + 1) % (ctx["eval_every"] * 4) == 0 or epoch + 1 == epochs:
                log(ctx["worker"], f"  {arch} ep {epoch + 1}/{epochs} loss={tot / max(nb, 1):.4f} "
                    f"val(p={v['rel_pressure']:.4f}, s={v['rel_saturation']:.4f}, bal={v['balanced']:.4f}) "
                    f"best={state['best_val']:.4f}@{state['best_epoch']}")

        if (epoch + 1) % ctx["ckpt_every"] == 0 and epoch + 1 < epochs:
            save_ckpt()
        if ctx["deadline"] and time.time() > ctx["deadline"] and epoch + 1 < epochs:
            save_ckpt()
            raise Paused(f"{arch} paused at epoch {epoch + 1}/{epochs}")

    state["elapsed"] += time.time() - t_start_ref[0]
    if state["best_state"] is None:
        return {"status": "diverged", "n_params": n_params, "best_val": float("inf")}, None
    return {"status": "complete", "n_params": n_params, "best_val": state["best_val"],
            "best_epoch": state["best_epoch"], "history": state["history"],
            "train_seconds": state["elapsed"]}, (model, state["best_state"])


# ---------------------------------------------------------------------------
# per-architecture pipeline: sweep -> pick -> finals
# ---------------------------------------------------------------------------
def sweep_configs(arch):
    pws = PW_GRID if arch in PHYSICS_ARCHS else [0.0]
    return [(lr, pw) for lr in LR_GRID for pw in pws]


def run_arch(ctx, arch):
    out, a = ctx["out"], ctx["args"]
    w = ctx["worker"]
    chosen = None

    if "sweep" in a.stages or "final" in a.stages:
        for lr, pw in sweep_configs(arch):
            path = os.path.join(out, "sweep", f"{arch}__lr{lr:g}__pw{pw:g}.json")
            if os.path.exists(path):
                continue
            log(w, f"SWEEP {arch} lr={lr:g} pw={pw:g} ({a.pilot_epochs} epochs)")
            res, _ = train_run(ctx, arch, lr, pw, a.pilot_epochs, 0,
                               os.path.join(out, "ckpt", f"{arch}__sweep_lr{lr:g}_pw{pw:g}.pt"))
            res.update({"arch": arch, "lr": lr, "pw": pw, "epochs": a.pilot_epochs})
            res.pop("history", None)
            atomic_json(res, path)
            ck = os.path.join(out, "ckpt", f"{arch}__sweep_lr{lr:g}_pw{pw:g}.pt")
            if os.path.exists(ck):
                os.remove(ck)
            log(w, f"  -> {res['status']} best_val={res['best_val']:.4f}")

    # pick the winner from validation results
    sweeps = [json.load(open(p)) for p in glob.glob(os.path.join(out, "sweep", f"{arch}__*.json"))]
    ok = [s for s in sweeps if s["status"] == "complete" and math.isfinite(s["best_val"])]
    if ok:
        chosen = min(ok, key=lambda s: s["best_val"])
        log(w, f"{arch}: chosen lr={chosen['lr']:g} pw={chosen['pw']:g} "
               f"(pilot val={chosen['best_val']:.4f}; {len(ok)}/{len(sweep_configs(arch))} configs usable)")
    else:
        log(w, f"{arch}: ALL sweep configs diverged -- skipping finals")
        return

    if "final" not in a.stages:
        return
    n_final = min(a.seeds, a.slow_seeds) if arch in SLOW_ARCHS else a.seeds
    for seed in range(n_final):
        path = os.path.join(out, "results", f"{arch}__seed{seed}.json")
        if os.path.exists(path):
            continue
        log(w, f"FINAL {arch} seed={seed} lr={chosen['lr']:g} pw={chosen['pw']:g} ({a.epochs} epochs)")
        ck = os.path.join(out, "ckpt", f"{arch}__final_seed{seed}.pt")
        res, mb = train_run(ctx, arch, chosen["lr"], chosen["pw"], a.epochs, seed, ck)
        res.update({"arch": arch, "seed": seed, "lr": chosen["lr"], "pw": chosen["pw"], "epochs": a.epochs})
        if mb is not None:
            model, best_state = mb
            model.load_state_dict(best_state)
            res["val"] = evaluate(model, ctx["X"]["val"], ctx["Y"]["val"])
            res["test"] = evaluate(model, ctx["X"]["test"], ctx["Y"]["test"])
            atomic_torch(best_state, os.path.join(out, "weights", f"{arch}__seed{seed}.pt"))
            log(w, f"  -> TEST p={res['test']['rel_pressure']:.4f} s={res['test']['rel_saturation']:.4f} "
                   f"bal={res['test']['balanced']:.4f} (best epoch {res['best_epoch']}, {res['train_seconds']:.0f}s)")
        else:
            log(w, f"  -> {res['status']}")
        atomic_json(res, path)
        if os.path.exists(ck):
            os.remove(ck)


# ---------------------------------------------------------------------------
def assign_archs(archs, n_workers, n_seeds, epochs, pilot_epochs, slow_seeds=2):
    def cost(a):
        n_sw = len(sweep_configs(a))
        ns = min(n_seeds, slow_seeds) if a in SLOW_ARCHS else n_seeds
        return ARCH_COST[a] * (n_sw * pilot_epochs / epochs + ns)
    loads = [0.0] * n_workers
    buckets = [[] for _ in range(n_workers)]
    for a in sorted(archs, key=cost, reverse=True):
        k = loads.index(min(loads))
        buckets[k].append(a); loads[k] += cost(a)
    return buckets, loads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="final_run")
    ap.add_argument("--dataset", default=None, help="default: <out>/data/dataset.npz")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--pilot_epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--slow_seeds", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--stages", default="sweep,final")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--time_budget_hours", type=float, default=0.0, help="0 = unlimited")
    ap.add_argument("--import_dir", default=None)
    args = ap.parse_args()
    args.stages = args.stages.split(",")

    out = args.out
    os.makedirs(out, exist_ok=True)
    if args.import_dir:
        n = import_previous(args.import_dir, out)
        print(f"imported {n} files from {args.import_dir}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_path = args.dataset or os.path.join(out, "data", "dataset.npz")
    x, y, yt, idx, stats, wp, ws = load_data(ds_path)
    N, _, H, W = x.shape
    T = yt.shape[1]
    if args.worker == 0:
        atomic_json({"stats": stats, "wp": wp, "ws": ws, "H": H, "W": W, "T": T, "N": int(N),
                     "n_train": len(idx["train"]), "n_val": len(idx["val"]), "n_test": len(idx["test"]),
                     "args": vars(args)}, os.path.join(out, "run_config.json"))

    ctx = {
        "device": dev, "worker": args.worker, "out": out, "args": args, "wp": wp, "ws": ws,
        "batch_size": args.batch_size, "eval_every": args.eval_every, "ckpt_every": args.ckpt_every,
        "deadline": time.time() + args.time_budget_hours * 3600 if args.time_budget_hours else 0,
        "zoo": make_all_architectures_v4(H, W, T, stats),
        "X": {}, "Y": {}, "YT": {},
    }
    for s in ("train", "val", "test"):
        ctx["X"][s] = torch.from_numpy(x[idx[s]]).to(dev)
        ctx["Y"][s] = torch.from_numpy(y[idx[s]]).to(dev)
    ctx["YT"]["train"] = torch.from_numpy(yt[idx["train"]]).to(dev)

    archs = [a for a in ARCH_COST if (not args.only or a in args.only)]
    buckets, loads = assign_archs(archs, args.n_workers, args.seeds, args.epochs, args.pilot_epochs, args.slow_seeds)
    mine = buckets[args.worker]
    log(args.worker, f"device={dev} data N={N} {H}x{W} T={T} split={len(idx['train'])}/{len(idx['val'])}/{len(idx['test'])}")
    log(args.worker, f"channel weights wp={wp:.4f} ws={ws:.4f}; assigned (cost {loads[args.worker]:.0f}): {mine}")

    try:
        for arch in mine:
            run_arch(ctx, arch)
    except Paused as e:
        log(args.worker, f"PAUSED (time budget): {e}. Re-run the same command (new session: add --import_dir).")
        sys.exit(3)
    log(args.worker, "ALL ASSIGNED JOBS COMPLETE")
    open(os.path.join(out, f"done_worker{args.worker}.flag"), "w").write("ok")


if __name__ == "__main__":
    main()
