"""
make_dataset.py -- sharded, parallel, RESUMABLE trajectory dataset generator.

Run this on a CPU-only Kaggle notebook (data generation is CPU-bound and
would otherwise burn GPU quota). Each shard is written atomically, so if the
session ends you just re-run the same command (optionally with --import_dir
pointing at the previous session's output) and only the missing shards are
generated. When all shards exist they are merged into <out>/dataset.npz.

    python make_dataset.py --out final_run/data --n_samples 4000 --nx 32 --ny 32 \
        --gravity --n_pressure_steps 150 --n_snapshots 5 --n_proc 4

Stores per-snapshot trajectories AND the final frame (final_* == last
snapshot), so one file serves final-field models and E2C.
"""
import argparse
import glob
import os
import shutil
import sys
import time
from multiprocessing import Pool

import numpy as np

from data_gen_v3_trajectory import generate_dataset

KEYS = ["permeability", "porosity", "well_mask", "pressure_traj", "saturation_traj"]


def _gen(job):
    (i, out, n, nx, ny, base_seed, gravity, steps, snaps, deadline) = job
    path = os.path.join(out, "shards", f"shard_{i:04d}.npz")
    if os.path.exists(path):
        return i, "exists"
    if deadline and time.time() > deadline:
        return i, "skipped_budget"
    t0 = time.time()
    ds = generate_dataset(n, nx, ny, base_seed + i * n, gravity, steps, snaps)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **{k: ds[k].astype(np.float32) for k in KEYS})
    os.replace(tmp, path)
    return i, f"done in {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="final_run/data")
    ap.add_argument("--n_samples", type=int, default=4000)
    ap.add_argument("--shard_size", type=int, default=100)
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20_000)
    ap.add_argument("--gravity", action="store_true")
    ap.add_argument("--n_pressure_steps", type=int, default=150)
    ap.add_argument("--n_snapshots", type=int, default=5)
    ap.add_argument("--n_proc", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--time_budget_hours", type=float, default=0.0, help="0 = unlimited")
    ap.add_argument("--import_dir", default=None, help="previous session's data dir to copy shards from")
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "shards"), exist_ok=True)
    if a.import_dir:
        for f in glob.glob(os.path.join(a.import_dir, "shards", "shard_*.npz")):
            dst = os.path.join(a.out, "shards", os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copy2(f, dst)
        print(f"imported shards from {a.import_dir}")

    n_shards = a.n_samples // a.shard_size
    deadline = time.time() + a.time_budget_hours * 3600 if a.time_budget_hours else 0
    jobs = [(i, a.out, a.shard_size, a.nx, a.ny, a.seed, a.gravity, a.n_pressure_steps, a.n_snapshots, deadline)
            for i in range(n_shards)]
    missing = [j for j in jobs if not os.path.exists(os.path.join(a.out, "shards", f"shard_{j[0]:04d}.npz"))]
    print(f"{n_shards} shards of {a.shard_size}: {n_shards - len(missing)} present, {len(missing)} to generate "
          f"on {a.n_proc} processes", flush=True)

    if missing:
        t0 = time.time()
        with Pool(a.n_proc) as pool:
            for k, (i, msg) in enumerate(pool.imap_unordered(_gen, missing), 1):
                print(f"[{k}/{len(missing)}] shard {i}: {msg}  (elapsed {time.time() - t0:.0f}s)", flush=True)

    present = sorted(glob.glob(os.path.join(a.out, "shards", "shard_*.npz")))
    present = [p for p in present if not p.endswith(".tmp.npz")]
    if len(present) < n_shards:
        print(f"PAUSED: {len(present)}/{n_shards} shards. Re-run to continue.")
        sys.exit(3)

    merged = {k: np.concatenate([np.load(p)[k] for p in present], axis=0) for k in KEYS}
    merged["final_pressure"] = merged["pressure_traj"][:, -1]
    merged["final_saturation"] = merged["saturation_traj"][:, -1]
    tmp = os.path.join(a.out, "dataset.tmp.npz")
    np.savez(tmp, **merged)
    os.replace(tmp, os.path.join(a.out, "dataset.npz"))
    print("merged dataset:", {k: v.shape for k, v in merged.items()})
    p = merged["final_pressure"]
    print(f"final pressure range [{p.min():.2f}, {p.max():.2f}] std {p.std():.3f}; "
          f"nan={bool(np.isnan(p).any())}")


if __name__ == "__main__":
    main()
