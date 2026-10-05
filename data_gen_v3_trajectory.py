"""
data_gen_v3_trajectory.py -- like data_gen_v3.py but keeps every saved
snapshot (pressure + saturation) so trajectory models (E2C, R-U-Net,
GNS) can be supervised at intermediate times, not just the final frame.

Output npz keys: permeability, porosity, well_mask  [N,H,W]
                 pressure_traj, saturation_traj      [N,T,H,W]
                 final_pressure, final_saturation    [N,H,W]  (= last frame)

    python data_gen_v3_trajectory.py --out small_traj.npz --n_samples 16 \
        --nx 16 --ny 16 --gravity --n_pressure_steps 100 --n_snapshots 5
"""
import argparse
import numpy as np

from solver_v3 import solve_two_phase
from data_gen_v3 import random_permeability, random_porosity, random_wells, make_well_mask


def generate_sample(nx, ny, seed, add_gravity, n_pressure_steps, n_snapshots):
    k = random_permeability(nx, ny, seed)
    phi = random_porosity(nx, ny, seed)
    locs, rates, is_inj = random_wells(nx, ny, seed)
    wm = make_well_mask(nx, ny, locs, rates, is_inj)
    kwargs = dict(add_gravity=True, rho_w=1.0, rho_n=0.6, g=1.5) if add_gravity else {}
    save_every = max(1, n_pressure_steps // n_snapshots)
    p_hist, s_hist, _ = solve_two_phase(
        k, phi, well_locations=locs, well_rates=rates, well_is_water_injector=is_inj,
        n_pressure_steps=n_pressure_steps, save_every=save_every, **kwargs)
    return k, phi, wm, np.stack(p_hist), np.stack(s_hist)


def generate_dataset(n_samples, nx, ny, seed, add_gravity, n_pressure_steps, n_snapshots):
    out = {k: [] for k in ("permeability", "porosity", "well_mask", "pressure_traj", "saturation_traj")}
    for s in range(n_samples):
        k, phi, wm, P, S = generate_sample(nx, ny, seed + s, add_gravity, n_pressure_steps, n_snapshots)
        for key, v in zip(out, (k, phi, wm, P, S)):
            out[key].append(v)
        if (s + 1) % max(1, n_samples // 10) == 0:
            print(f"  generated {s + 1}/{n_samples}", flush=True)
    T = min(len(a) for a in out["pressure_traj"])  # guard against ragged snapshot counts
    out = {k: np.stack([a[:T] if a.ndim == 3 else a for a in v]) for k, v in out.items()}
    out["final_pressure"] = out["pressure_traj"][:, -1]
    out["final_saturation"] = out["saturation_traj"][:, -1]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_v3_gravity_trajectory.npz")
    ap.add_argument("--n_samples", type=int, default=1000)
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gravity", action="store_true")
    ap.add_argument("--n_pressure_steps", type=int, default=150)
    ap.add_argument("--n_snapshots", type=int, default=5)
    a = ap.parse_args()
    ds = generate_dataset(a.n_samples, a.nx, a.ny, a.seed, a.gravity, a.n_pressure_steps, a.n_snapshots)
    np.savez_compressed(a.out, **ds)
    print("saved", a.out, {k: v.shape for k, v in ds.items()})
