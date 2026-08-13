"""
Synthetic training data generation for the FNO surrogate.

Generates random permeability fields + well configurations, runs the FD
solver, and derives a per-well payoff (EUR-style: cumulative production,
penalized by interference from neighboring wells).
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from solver import solve_pressure_diffusion


def random_permeability_field(nx=32, ny=32, correlation_length=5, k_min=0.02, k_max=0.2):
    """
    Smooth random field standing in for a heterogeneous reservoir.
    correlation_length controls geological 'patchiness' (higher = smoother).
    Rescaled to [k_min, k_max] -- permeability must stay strictly positive.
    """
    raw = np.random.randn(nx, ny)
    field = gaussian_filter(raw, sigma=correlation_length)
    field = (field - field.min()) / (field.max() - field.min() + 1e-10)
    return k_min + field * (k_max - k_min)


def compute_well_payoffs(history, well_locations, well_rates, dt, discount_rate=0.0):
    """
    EUR-style per-well payoff: discounted cumulative production.

    For a producer (negative rate), instantaneous production rate scales
    with local pressure drawdown relative to a virgin-pressure baseline
    (here 0, since p starts at 0 and wells deplete it). We approximate
    per-step production as proportional to |local pressure| * |well_rate
    sign|, which increases when interference deepens local drawdown --
    NOTE: this is a simplified proxy, not a full material-balance model.
    Discounting (if discount_rate > 0) applies standard exponential decay
    per timestep, matching typical EUR/NPV treatment in reservoir economics.

    Returns: payoffs, shape (n_wells,)
    """
    n_steps = history.shape[0]
    payoffs = np.zeros(len(well_locations))
    for step in range(n_steps):
        discount = 1.0 / ((1.0 + discount_rate) ** step) if discount_rate > 0 else 1.0
        for w_idx, (wi, wj) in enumerate(well_locations):
            local_p = history[step, wi, wj]
            # production contribution: magnitude of local pressure depletion
            payoffs[w_idx] += discount * abs(local_p) * dt
    return payoffs


def sample_well_configuration(nx, ny, max_wells=8, margin=2):
    """Random well count/locations/rates, kept off the domain boundary."""
    n_wells = np.random.randint(1, max_wells + 1)
    locs = [
        (np.random.randint(margin, nx - margin), np.random.randint(margin, ny - margin))
        for _ in range(n_wells)
    ]
    rates = -np.random.uniform(0.5, 1.0, size=n_wells)  # producers only, for now
    return locs, rates


def generate_dataset(n_samples=2000, nx=32, ny=32, max_wells=8, nt=150, seed=None):
    """
    Returns a dict of stacked arrays ready for tensor conversion:
        permeability: (N, nx, ny)
        well_mask:    (N, nx, ny)  -- 1 at well cells, 0 elsewhere (rate-weighted)
        final_pressure: (N, nx, ny)
        payoffs: list of length-N arrays (variable length per sample -- wells vary)
        well_locations, well_rates: lists of length N
    """
    if seed is not None:
        np.random.seed(seed)

    perms = np.zeros((n_samples, nx, ny))
    well_masks = np.zeros((n_samples, nx, ny))
    final_pressures = np.zeros((n_samples, nx, ny))
    all_payoffs = []
    all_locs = []
    all_rates = []

    for s in range(n_samples):
        perm = random_permeability_field(nx, ny)
        locs, rates = sample_well_configuration(nx, ny, max_wells)

        history, dt = solve_pressure_diffusion(perm, locs, rates, nt=nt)
        payoffs = compute_well_payoffs(history, locs, rates, dt)

        mask = np.zeros((nx, ny))
        for (wi, wj), r in zip(locs, rates):
            mask[wi, wj] = r

        perms[s] = perm
        well_masks[s] = mask
        final_pressures[s] = history[-1]
        all_payoffs.append(payoffs)
        all_locs.append(locs)
        all_rates.append(rates)

    return {
        "permeability": perms,
        "well_mask": well_masks,
        "final_pressure": final_pressures,
        "payoffs": all_payoffs,
        "well_locations": all_locs,
        "well_rates": all_rates,
    }


if __name__ == "__main__":
    import time

    t0 = time.time()
    ds = generate_dataset(n_samples=200, nx=32, ny=32, nt=150, seed=0)
    elapsed = time.time() - t0
    print(f"generated {len(ds['permeability'])} samples in {elapsed:.1f}s "
          f"({elapsed/len(ds['permeability'])*1000:.1f} ms/sample)")
    print("permeability shape:", ds["permeability"].shape)
    print("final_pressure shape:", ds["final_pressure"].shape)
    print("example payoffs (sample 0):", ds["payoffs"][0])

    np.savez_compressed(
        "sample_dataset_200.npz",
        permeability=ds["permeability"],
        well_mask=ds["well_mask"],
        final_pressure=ds["final_pressure"],
    )
    print("saved sample_dataset_200.npz")
