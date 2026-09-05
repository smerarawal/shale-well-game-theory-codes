"""
data_gen_trajectory.py  (v2 -- matches your real solver_v2.py signatures)

solver_v2.solve_pressure_diffusion_v2() ALREADY returns the full pressure
trajectory (history, shape (nt, nx, ny)) on every call -- it was never
final-state-only. The v1/v2 game-theory scripts presumably just discard
everything except history[-1]. This script keeps the whole thing.

No well_mask generator exists in solver_v2.py -- wells there are
(well_locations, well_rates) lists. This script generates random wells
matching v1's convention (1-8 wells, production-only i.e. negative rates)
and ALSO builds a dense well_mask array (well count/magnitude at each cell)
for the FNO's input channels, since a list of (i,j) tuples isn't directly
usable as a fixed-size network input.
"""

import numpy as np
import time

from solver_v2 import random_anisotropic_permeability, random_porosity_field, solve_pressure_diffusion_v2

GRID = 32
N_SAMPLES = 2000
NT = 50                  # physics timesteps per sample (solver_v2 default)
N_SAVED_STEPS = 20       # subsample history down to this many trajectory frames for the FNO target
MIN_WELLS, MAX_WELLS = 1, 8
RATE_MIN, RATE_MAX = -3.0, -1.0   # production-only, matches v1's "production-only" convention
MU, CT = 1.0, 1.0
NON_DARCY_BETA = 0.0     # set > 0 if you want v2's Forchheimer correction included


def random_wells(nx, ny, rng):
    n_wells = rng.integers(MIN_WELLS, MAX_WELLS + 1)
    ys = rng.integers(0, nx, n_wells)
    xs = rng.integers(0, ny, n_wells)
    well_locations = list(zip(ys.tolist(), xs.tolist()))
    well_rates = rng.uniform(RATE_MIN, RATE_MAX, n_wells).tolist()

    well_mask = np.zeros((nx, ny), dtype=np.float32)
    for (i, j), r in zip(well_locations, well_rates):
        well_mask[i, j] = -r  # positive magnitude in the mask channel
    return well_locations, well_rates, well_mask


def subsample_trajectory(history, n_saved_steps):
    """history: (nt, nx, ny) -> (n_saved_steps, nx, ny), evenly spaced,
    always including the final timestep."""
    nt = history.shape[0]
    idx = np.linspace(0, nt - 1, n_saved_steps).round().astype(int)
    return history[idx]


def generate_dataset(n_samples=N_SAMPLES, grid_size=GRID, nt=NT,
                      n_saved_steps=N_SAVED_STEPS, out_path="dataset_v2_trajectory.npz",
                      seed=0):
    rng = np.random.default_rng(seed)

    kx_all = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)
    ky_all = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)
    poro_all = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)
    wells_all = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)
    pressure_traj_all = np.zeros((n_samples, n_saved_steps, grid_size, grid_size), dtype=np.float32)

    t0 = time.time()
    for i in range(n_samples):
        sample_seed = int(rng.integers(0, 2**31 - 1))
        kx, ky = random_anisotropic_permeability(grid_size, grid_size, seed=sample_seed)
        phi = random_porosity_field(grid_size, grid_size, seed=sample_seed)
        well_locations, well_rates, well_mask = random_wells(grid_size, grid_size, rng)

        history, _dt = solve_pressure_diffusion_v2(
            kx, ky, phi, well_locations, well_rates,
            mu=MU, ct=CT, dx=1.0, dt=None, nt=nt,
            non_darcy_beta=NON_DARCY_BETA,
        )

        kx_all[i] = kx
        ky_all[i] = ky
        poro_all[i] = phi
        wells_all[i] = well_mask
        pressure_traj_all[i] = subsample_trajectory(history, n_saved_steps)

        if (i + 1) % 200 == 0:
            print(f"generated {i + 1}/{n_samples}   {time.time() - t0:.1f}s")

    np.savez_compressed(
        out_path,
        kx=kx_all, ky=ky_all, porosity=poro_all, well_mask=wells_all,
        pressure_trajectory=pressure_traj_all,
    )
    elapsed = time.time() - t0
    print(f"generated {n_samples} trajectory samples in {elapsed:.1f}s "
          f"({1000 * elapsed / n_samples:.1f} ms/sample)")
    size_mb = (pressure_traj_all.nbytes + kx_all.nbytes + ky_all.nbytes
               + poro_all.nbytes + wells_all.nbytes) / 1e6
    print(f"saved {out_path}, size MB: {size_mb:.2f}")


if __name__ == "__main__":
    generate_dataset()
