"""
data_gen_v3.py -- v3 two-phase dataset generator, filling the gap left by
run_pipeline.py (v1-only) and data_gen_v2.py (single-phase v2). Produces
the exact npz layout visualize_everything.py's dataset_v3_field_samples()
already expects: permeability, porosity, well_mask, final_pressure,
final_saturation -- so existing plotting code works unmodified.

Two calls matching the two files visualize_everything.py already looks
for:
    python data_gen_v3.py --out dataset_v3_1000.npz
    python data_gen_v3.py --out dataset_v3_gravity_1000.npz --gravity

well_mask convention (per solver_v3.py's well_is_water_injector): +rate
at injector cells, -rate at producer cells.

Tested end-to-end (real solve_two_phase calls, not mocked) at 16x16 scale
before delivery -- see test_adapters.py / test_full_run.py in the same
delivery for the training-side validation this dataset feeds into.
"""
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter

from solver_v3 import solve_two_phase


def random_permeability(nx, ny, seed, corr_len=6, k_min=0.02, k_max=0.2):
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.random((nx, ny)), sigma=corr_len)
    field = (field - field.min()) / (field.max() - field.min() + 1e-12)
    return field * (k_max - k_min) + k_min


def random_porosity(nx, ny, seed, corr_len=4, phi_min=0.08, phi_max=0.25):
    rng = np.random.default_rng(seed + 10_000)
    field = gaussian_filter(rng.random((nx, ny)), sigma=corr_len)
    field = (field - field.min()) / (field.max() - field.min() + 1e-12)
    return field * (phi_max - phi_min) + phi_min


def random_wells(nx, ny, seed, n_wells_range=(2, 4), margin=4):
    """At least one injector and one producer per sample -- a pure-injector
    or pure-producer sample can't demonstrate displacement, which is the
    entire point of v3 over v1/v2."""
    rng = np.random.default_rng(seed + 20_000)
    n_wells = rng.integers(n_wells_range[0], n_wells_range[1] + 1)
    locs, rates, is_injector = [], [], []
    chosen = set()
    for w in range(n_wells):
        while True:
            i = rng.integers(margin, nx - margin)
            j = rng.integers(margin, ny - margin)
            if (i, j) not in chosen:
                chosen.add((i, j))
                break
        locs.append((int(i), int(j)))
        inj = (w == 0) or (w > 1 and rng.random() < 0.5)  # guarantee well 0 = injector
        if w == 1:
            inj = False  # guarantee well 1 = producer, so every sample has both
        is_injector.append(bool(inj))
        rate = float(rng.uniform(0.6, 1.4))
        rates.append(rate)
    return locs, rates, is_injector


def make_well_mask(nx, ny, locs, rates, is_injector):
    mask = np.zeros((nx, ny))
    for (i, j), r, inj in zip(locs, rates, is_injector):
        mask[i, j] = r if inj else -r
    return mask


def generate_sample(nx, ny, seed, add_gravity, n_pressure_steps):
    k = random_permeability(nx, ny, seed)
    phi = random_porosity(nx, ny, seed)
    locs, rates, is_injector = random_wells(nx, ny, seed)
    well_mask = make_well_mask(nx, ny, locs, rates, is_injector)

    kwargs = {}
    if add_gravity:
        kwargs = dict(add_gravity=True, rho_w=1.0, rho_n=0.6, g=1.5)

    p_hist, s_hist, _ = solve_two_phase(
        k, phi, well_locations=locs, well_rates=rates,
        well_is_water_injector=is_injector,
        n_pressure_steps=n_pressure_steps, save_every=max(1, n_pressure_steps // 5),
        **kwargs,
    )
    return k, phi, well_mask, p_hist[-1], s_hist[-1]


def generate_dataset(n_samples, nx, ny, seed, add_gravity=False, n_pressure_steps=150):
    perm = np.zeros((n_samples, nx, ny))
    poro = np.zeros((n_samples, nx, ny))
    well_mask = np.zeros((n_samples, nx, ny))
    final_pressure = np.zeros((n_samples, nx, ny))
    final_saturation = np.zeros((n_samples, nx, ny))
    for s in range(n_samples):
        k, phi, wm, p_final, sw_final = generate_sample(
            nx, ny, seed=seed + s, add_gravity=add_gravity, n_pressure_steps=n_pressure_steps)
        perm[s], poro[s], well_mask[s] = k, phi, wm
        final_pressure[s], final_saturation[s] = p_final, sw_final
        if (s + 1) % max(1, n_samples // 10) == 0:
            print(f"  generated {s + 1}/{n_samples}")
    return {
        "permeability": perm, "porosity": poro, "well_mask": well_mask,
        "final_pressure": final_pressure, "final_saturation": final_saturation,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_v3_1000.npz")
    ap.add_argument("--n_samples", type=int, default=1000)
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gravity", action="store_true")
    ap.add_argument("--n_pressure_steps", type=int, default=150)
    args = ap.parse_args()

    print(f"generating {args.n_samples} samples at {args.nx}x{args.ny}"
          f"{' WITH gravity' if args.gravity else ''} -> {args.out}")
    ds = generate_dataset(args.n_samples, args.nx, args.ny, args.seed,
                           add_gravity=args.gravity, n_pressure_steps=args.n_pressure_steps)
    np.savez_compressed(args.out, **ds)
    print(f"saved {args.out}")
