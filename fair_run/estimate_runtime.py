"""
estimate_runtime.py -- GPU PREFLIGHT. For every architecture: builds the model,
runs real training steps (same loss path as the run: weighted MSE, plus the
trajectory loss / physics residual where they apply) on random data of the real
shapes, and reports seconds/step, peak GPU memory and a MEASURED extrapolation
to the full protocol. Fails loudly (exit 1) if any architecture errors, produces
a non-finite loss or non-finite gradients, so you find out in ~5 minutes rather
than 5 hours.

    python estimate_runtime.py --out final_run --n_train 3200 --epochs 300
"""
import argparse
import json
import math
import os
import sys
import time

import torch

from model_zoo_v4 import make_all_architectures_v4, ARCH_COST, PHYSICS_ARCHS
from physics_residual_v3 import pressure_residual
from train_utils_v3 import weighted_channel_loss
import run_fair_comparison as rfc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="final_run")
    ap.add_argument("--n_train", type=int, default=3200)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--T", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--pilot_epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--slow_seeds", type=int, default=2)
    ap.add_argument("--n_workers", type=int, default=0, help="0 = number of visible GPUs (min 1)")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nw = a.n_workers or max(1, torch.cuda.device_count())
    cfg_path = os.path.join(a.out, "run_config.json")
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        stats, H, T, n_train = cfg["stats"], cfg["H"], cfg["T"], cfg["n_train"]
    else:
        stats = dict(in_mean=[0.1, 0.15, 0.0], in_std=[0.05, 0.05, 1.4], p_mean=-1.0, p_std=3.0)
        H, T, n_train = a.H, a.T, a.n_train
        ds_path = os.path.join(a.out, "data", "dataset.npz")
        if os.path.exists(ds_path):  # real shapes from the generated dataset (80% train split)
            import numpy as np
            shp = np.load(ds_path)["pressure_traj"].shape
            n_train, T, H = int(0.8 * shp[0]), shp[1], shp[2]
    print(f"device={dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU: estimates are NOT GPU numbers'}), "
          f"{nw} worker(s), grid {H}x{H}, T={T}, n_train={n_train}, batch={a.batch_size}", flush=True)

    zoo = make_all_architectures_v4(H, H, T, stats)
    B = a.batch_size
    iters_per_epoch = math.ceil(n_train / B)
    x = torch.randn(B, 3, H, H, device=dev) * 0.1
    x[:, 2] = 0
    x[:, 2, H // 6, H // 6] = 1.0
    x[:, 2, (5 * H) // 8, H // 3] = -1.0
    y = torch.randn(B, 2, H, H, device=dev)
    yt = torch.randn(B, T, 2, H, H, device=dev)
    wp, ws = 0.05, 1.95

    rows, failures = {}, []
    for arch in ARCH_COST:
        if a.only and arch not in a.only:
            continue
        try:
            if dev.type == "cuda":
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            m = zoo[arch]().to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            times = []
            for i in range(a.steps):
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                pred = m(x)
                loss, _, _ = weighted_channel_loss(pred, y, wp, ws)
                if m.has_traj:
                    loss = loss + m.traj_loss(x, yt, wp, ws)
                if arch in PHYSICS_ARCHS:
                    loss = loss + 10.0 * pressure_residual(pred, x)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
                if not (torch.isfinite(loss) and torch.isfinite(gn)):
                    raise RuntimeError(f"non-finite loss/grad at step {i}")
            sec = float(sorted(times[3:])[len(times[3:]) // 2])  # median after warmup
            mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else float("nan")
            rows[arch] = {"sec_per_step": sec, "epoch_sec": sec * iters_per_epoch * 1.05, "peak_gb": mem,
                          "params": sum(p.numel() for p in m.parameters())}
            print(f"  ok  {arch:22s} {sec * 1000:8.1f} ms/step  ~{rows[arch]['epoch_sec']:7.1f} s/epoch  "
                  f"peak {mem:5.2f} GB  {rows[arch]['params']:>10,} params", flush=True)
            del m, opt
        except Exception as e:  # noqa
            failures.append((arch, repr(e)))
            print(f"  FAIL {arch}: {e!r}", flush=True)

    if failures:
        print("\nPREFLIGHT FAILED for:", [f[0] for f in failures])
        sys.exit(1)

    def total_sec(arch):
        n_sw = len(rfc.sweep_configs(arch))
        ns = min(a.seeds, a.slow_seeds) if arch in rfc.SLOW_ARCHS else a.seeds
        return rows[arch]["epoch_sec"] * (n_sw * a.pilot_epochs + ns * a.epochs)

    archs = list(rows)
    rfc.ARCH_COST.update({k: rows[k]["epoch_sec"] for k in archs})  # balance with MEASURED cost
    buckets, loads = rfc.assign_archs(archs, nw, a.seeds, a.epochs, a.pilot_epochs, a.slow_seeds)
    per_worker = [sum(total_sec(k) for k in b) / 3600 for b in buckets]
    print("\n arch                   sweep+final GPU-hours")
    for k in sorted(archs, key=total_sec, reverse=True):
        print(f"  {k:22s} {total_sec(k) / 3600:6.2f}")
    print(f"\nTOTAL GPU-hours: {sum(per_worker):.1f}   |  wall clock with {nw} worker(s): ~{max(per_worker):.1f} h "
          f"(+ data generation, + ~10% slack when both GPUs run at once)")
    print("worker assignment:", {i: b for i, b in enumerate(buckets)})
    print("NOTE: estimates use random data at the real shapes; real runs add periodic validation (~5%, included).")
    os.makedirs(a.out, exist_ok=True)
    json.dump({"rows": rows, "per_worker_hours": per_worker, "n_workers": nw, "device": str(dev)},
              open(os.path.join(a.out, "estimate.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
