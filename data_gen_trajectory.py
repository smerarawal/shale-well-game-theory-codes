"""
data_gen_trajectory.py

Generates (kx, ky, porosity, well_mask) -> pressure TRAJECTORY dataset,
instead of just the final-timestep pressure field.

This unlocks the real discounted-EUR payoff: EUR = sum_t discount(t) * rate(t),
where rate(t) can be derived from the local pressure gradient at each well over
time, instead of the single-snapshot proxy used everywhere in v1/v2 so far.

ASSUMED INTERFACE (swap these two lines for your real solver_v2 functions):
    from solver_v2 import solve_pressure_field_v2
    perm_field, poro_field, well_mask = make_random_geology_v2(grid_size)

If your actual solver_v2.solve_pressure_field_v2() only returns the final
pressure state, the minimal change needed there is: instead of discarding
intermediate timesteps in the time-integration loop, append a copy of `p`
to a list every `save_every` steps and return the stacked array.
"""

import numpy as np
import time

GRID = 32
N_SAMPLES = 2000
N_SAVED_STEPS = 20       # how many trajectory snapshots to keep per sample
SAVE_EVERY_N_PHYSICS_STEPS = 5  # subsample the solver's internal timestep

# --- swap this block for your real imports -------------------------------
try:
    from solver_v2 import solve_pressure_field_v2, make_random_geology_v2
    HAVE_REAL_SOLVER = True
except ImportError:
    HAVE_REAL_SOLVER = False
    print("WARNING: solver_v2 not found on path — using a stub solver so "
          "this script is runnable/testable standalone. Replace with your "
          "real solver_v2 import before using the output for real results.")
# ---------------------------------------------------------------------------


def stub_make_random_geology(grid_size):
    """Fallback geology generator (Gaussian-smoothed noise), only used if
    solver_v2 isn't importable. Matches v2's (kx, ky, porosity, well_mask)
    channel convention."""
    from scipy.ndimage import gaussian_filter
    kx = gaussian_filter(np.random.rand(grid_size, grid_size), sigma=2) * 0.18 + 0.02
    ky = gaussian_filter(np.random.rand(grid_size, grid_size), sigma=2) * 0.18 + 0.02
    porosity = gaussian_filter(np.random.rand(grid_size, grid_size), sigma=3) * 0.15 + 0.05
    well_mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    n_wells = np.random.randint(1, 9)
    ys = np.random.randint(0, grid_size, n_wells)
    xs = np.random.randint(0, grid_size, n_wells)
    well_mask[ys, xs] = 1.0
    return kx.astype(np.float32), ky.astype(np.float32), porosity.astype(np.float32), well_mask


def stub_solve_trajectory(kx, ky, porosity, well_mask, n_saved_steps):
    """Fallback trajectory solver: crude explicit diffusion stepping so the
    script produces a plausible (T, H, W) trajectory shape. Replace with a
    call into your real, validated solver_v2 time-integration loop."""
    grid = kx.shape[0]
    p = np.zeros((grid, grid), dtype=np.float32)
    dt = 0.01
    trajectory = np.zeros((n_saved_steps, grid, grid), dtype=np.float32)
    step = 0
    for t in range(n_saved_steps * SAVE_EVERY_N_PHYSICS_STEPS):
        lap = (
            np.roll(p, 1, 0) + np.roll(p, -1, 0)
            + np.roll(p, 1, 1) + np.roll(p, -1, 1) - 4 * p
        )
        k_eff = 0.5 * (kx + ky)
        p = p + dt * (k_eff * lap / (porosity + 1e-3)) - well_mask * dt * 5.0
        if (t + 1) % SAVE_EVERY_N_PHYSICS_STEPS == 0:
            trajectory[step] = p
            step += 1
    return trajectory


def generate_dataset(n_samples=N_SAMPLES, grid_size=GRID, n_saved_steps=N_SAVED_STEPS, out_path="dataset_v2_trajectory.npz"):
    kx_all = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)
    ky_all = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)
    poro_all = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)
    wells_all = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)
    pressure_traj_all = np.zeros((n_samples, n_saved_steps, grid_size, grid_size), dtype=np.float32)

    t0 = time.time()
    for i in range(n_samples):
        if HAVE_REAL_SOLVER:
            kx, ky, porosity, well_mask = make_random_geology_v2(grid_size)
            traj = solve_pressure_field_v2(
                kx, ky, porosity, well_mask,
                save_trajectory=True, n_saved_steps=n_saved_steps,
            )
        else:
            kx, ky, porosity, well_mask = stub_make_random_geology(grid_size)
            traj = stub_solve_trajectory(kx, ky, porosity, well_mask, n_saved_steps)

        kx_all[i] = kx
        ky_all[i] = ky
        poro_all[i] = porosity
        wells_all[i] = well_mask
        pressure_traj_all[i] = traj

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
    print(f"saved {out_path}, size MB: {np.round(np.prod(pressure_traj_all.shape) * 4 / 1e6 + 4 * n_samples * grid_size * grid_size * 4 / 1e6, 2)}")


if __name__ == "__main__":
    generate_dataset()
