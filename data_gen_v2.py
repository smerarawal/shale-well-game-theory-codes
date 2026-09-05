"""
data_gen_v2.py -- dataset generation using solver_v2.py's richer physics.

Input channels change from 2 (permeability, well_mask) to 4:
    kx, ky (anisotropic permeability), porosity, well_mask

Run:
    python data_gen_v2.py
"""
import numpy as np

from solver_v2 import (
    random_anisotropic_permeability, random_porosity_field,
    solve_pressure_diffusion_v2,
)

NX, NY = 32, 32


def sample_well_configuration(nx, ny, max_wells=8, margin=2):
    n_wells = np.random.randint(1, max_wells + 1)
    locs = [
        (np.random.randint(margin, nx - margin), np.random.randint(margin, ny - margin))
        for _ in range(n_wells)
    ]
    rates = -np.random.uniform(0.5, 1.0, size=n_wells)
    return locs, rates


def compute_well_payoffs_v2(history, well_locations, dt):
    """Same simplified proxy as data_gen.py's compute_well_payoffs, adapted for v2 history shape."""
    n_steps = history.shape[0]
    payoffs = np.zeros(len(well_locations))
    for step in range(n_steps):
        for w_idx, (wi, wj) in enumerate(well_locations):
            payoffs[w_idx] += abs(history[step, wi, wj]) * dt
    return payoffs


def generate_dataset_v2(n_samples=2000, nx=NX, ny=NY, max_wells=8, nt=150, seed=None):
    if seed is not None:
        np.random.seed(seed)

    kxs = np.zeros((n_samples, nx, ny))
    kys = np.zeros((n_samples, nx, ny))
    phis = np.zeros((n_samples, nx, ny))
    well_masks = np.zeros((n_samples, nx, ny))
    final_pressures = np.zeros((n_samples, nx, ny))
    all_payoffs, all_locs, all_rates = [], [], []

    for s in range(n_samples):
        kx, ky = random_anisotropic_permeability(nx, ny)
        phi = random_porosity_field(nx, ny)
        locs, rates = sample_well_configuration(nx, ny, max_wells)

        history, dt = solve_pressure_diffusion_v2(kx, ky, phi, locs, rates, nt=nt)
        payoffs = compute_well_payoffs_v2(history, locs, dt)

        mask = np.zeros((nx, ny))
        for (wi, wj), r in zip(locs, rates):
            mask[wi, wj] = r

        kxs[s] = kx; kys[s] = ky; phis[s] = phi
        well_masks[s] = mask
        final_pressures[s] = history[-1]
        all_payoffs.append(payoffs); all_locs.append(locs); all_rates.append(rates)

        if (s + 1) % 200 == 0:
            print(f"  generated {s+1}/{n_samples}")

    return {
        "kx": kxs, "ky": kys, "porosity": phis, "well_mask": well_masks,
        "final_pressure": final_pressures,
        "payoffs": all_payoffs, "well_locations": all_locs, "well_rates": all_rates,
    }


if __name__ == "__main__":
    import time

    t0 = time.time()
    ds = generate_dataset_v2(n_samples=2000, nx=NX, ny=NY, nt=150, seed=42)
    elapsed = time.time() - t0
    print(f"generated {len(ds['kx'])} samples in {elapsed:.1f}s "
          f"({elapsed/len(ds['kx'])*1000:.1f} ms/sample)")

    np.savez_compressed(
        "dataset_v2_2000.npz",
        kx=ds["kx"], ky=ds["ky"], porosity=ds["porosity"],
        well_mask=ds["well_mask"], final_pressure=ds["final_pressure"],
    )
    import os
    print("saved dataset_v2_2000.npz, size MB:", os.path.getsize("dataset_v2_2000.npz") / 1e6)
