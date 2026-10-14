"""
make_ood_dataset.py -- out-of-distribution test sets (final frames only),
generated with the SAME solver settings as training (gravity ON, rho_w=1.0,
rho_n=0.6, g=1.5, same pressure steps / snapshot spacing) but a shifted
input distribution:

  ood_more_wells   5-7 wells per sample (training: 2-4)
  ood_rough_perm   permeability correlation length 2  (training: 6) -> rougher
  ood_smooth_perm  permeability correlation length 12 (training: 6) -> smoother

Seeds are disjoint from the training pool. Resumable (per-chunk atomic files);
re-running only generates missing chunks.

    python make_ood_dataset.py --out final_run/ood --n_per_set 200 --n_proc 4
"""
import argparse
import glob
import os
import time
from multiprocessing import Pool

import numpy as np

from data_gen_v3 import random_permeability, random_porosity, random_wells, make_well_mask
from solver_v3 import solve_two_phase

OOD_SETS = {
    "ood_more_wells": dict(n_wells_range=(5, 7), corr_len=6),
    "ood_rough_perm": dict(n_wells_range=(2, 4), corr_len=2),
    "ood_smooth_perm": dict(n_wells_range=(2, 4), corr_len=12),
}
CHUNK = 25


def gen_sample(nx, ny, seed, cfg, steps):
    k = random_permeability(nx, ny, seed, corr_len=cfg["corr_len"])
    phi = random_porosity(nx, ny, seed)
    locs, rates, inj = random_wells(nx, ny, seed, n_wells_range=cfg["n_wells_range"])
    wm = make_well_mask(nx, ny, locs, rates, inj)
    p, s, _ = solve_two_phase(k, phi, well_locations=locs, well_rates=rates, well_is_water_injector=inj,
                              n_pressure_steps=steps, save_every=max(1, steps // 5),
                              add_gravity=True, rho_w=1.0, rho_n=0.6, g=1.5)
    return k, phi, wm, p[-1], s[-1]


def _job(job):
    name, ci, out, nx, ny, steps, base = job
    path = os.path.join(out, "chunks", f"{name}__{ci:03d}.npz")
    if os.path.exists(path):
        return name, ci, "exists"
    cfg = OOD_SETS[name]
    set_idx = list(OOD_SETS).index(name)
    t0 = time.time()
    rows = [gen_sample(nx, ny, base + set_idx * 100_000 + ci * CHUNK + j, cfg, steps) for j in range(CHUNK)]
    tmp = path + ".tmp.npz"
    np.savez(tmp, **{k: np.stack([r[i] for r in rows]).astype(np.float32)
                     for i, k in enumerate(["permeability", "porosity", "well_mask", "final_pressure", "final_saturation"])})
    os.replace(tmp, path)
    return name, ci, f"done in {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="final_run/ood")
    ap.add_argument("--n_per_set", type=int, default=200)
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--n_pressure_steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=900_000)
    ap.add_argument("--n_proc", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--import_dir", default=None)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "chunks"), exist_ok=True)
    if a.import_dir:
        import shutil
        for f in glob.glob(os.path.join(a.import_dir, "chunks", "*.npz")):
            d = os.path.join(a.out, "chunks", os.path.basename(f))
            if not os.path.exists(d) and not f.endswith(".tmp.npz"):
                shutil.copy2(f, d)
    n_chunks = max(1, a.n_per_set // CHUNK)
    jobs = [(n, ci, a.out, a.nx, a.ny, a.n_pressure_steps, a.seed) for n in OOD_SETS for ci in range(n_chunks)]
    todo = [j for j in jobs if not os.path.exists(os.path.join(a.out, "chunks", f"{j[0]}__{j[1]:03d}.npz"))]
    print(f"{len(jobs)} chunks, {len(todo)} to generate on {a.n_proc} procs", flush=True)
    if todo:
        t0 = time.time()
        with Pool(a.n_proc) as pool:
            for k, (n, ci, msg) in enumerate(pool.imap_unordered(_job, todo), 1):
                print(f"[{k}/{len(todo)}] {n} chunk {ci}: {msg} ({time.time() - t0:.0f}s)", flush=True)
    for name in OOD_SETS:
        files = sorted(glob.glob(os.path.join(a.out, "chunks", f"{name}__*.npz")))
        files = [f for f in files if not f.endswith(".tmp.npz")]
        if len(files) < n_chunks:
            print(f"PAUSED: {name} has {len(files)}/{n_chunks} chunks"); raise SystemExit(3)
        keys = ["permeability", "porosity", "well_mask", "final_pressure", "final_saturation"]
        merged = {k: np.concatenate([np.load(f)[k] for f in files[:n_chunks]], axis=0) for k in keys}
        tmp = os.path.join(a.out, f"{name}.tmp.npz")
        np.savez(tmp, **merged)
        os.replace(tmp, os.path.join(a.out, f"{name}.npz"))
        p = merged["final_pressure"]
        print(f"{name}: {p.shape[0]} samples, pressure range [{p.min():.1f}, {p.max():.1f}], "
              f"mean #wells={np.mean((merged['well_mask'] != 0).sum(axis=(1, 2))):.1f}")


if __name__ == "__main__":
    main()
