"""
data_gen_v3_multiphase.py -- Part C of richer_physics_v3_spec.md.

This is the file validate_solver_v3.py's own "GATE CLEARED" message points
to next. It did not exist in the repo -- this is the dataset generator for
solver_v3.py's validated two-phase (IMPES) physics, mirroring data_gen.py /
data_gen_v2.py's structure so it drops straight into the existing
comparison_protocol.py / run_surrogate_comparison.py / eval_utils.py
pipeline without changes to those files.

Input channels (3, matching solver_v3.py's isotropic-k signature --
solve_two_phase takes a single k field, not kx/ky, so this is NOT simply
v2's 4-channel setup plus a channel):
    permeability, porosity, well_mask (signed: +rate at injectors,
    -rate at producers, encoding BOTH well location and type in one
    channel the same way v1/v2 already do)

Output channels (2, replacing v1/v2's single final_pressure):
    final_pressure, final_saturation

Well configurations deliberately mix injectors and producers in the same
sample (not injection-only) -- this is the capability the physics spec
flagged as a genuine advantage over the benchmark paper (Chen & Hosseini,
arXiv:2508.11618), whose model is injection-only. Exercise it in the
training data or the surrogate never sees combined-scenario examples.

Per-well EUR-style game-theory payoffs (the two-channel pressure+plume
payoff described in the physics spec) are NOT computed here -- that is a
separate downstream step once the surrogates themselves are validated,
not part of "implement the surrogates." Skipped deliberately, not missed.

Gravity (added after this file's first version): pass --gravity to generate
a dataset using solver_v3.py's add_gravity=True path (buoyant non-wetting
phase segregates toward shallower j; see solver_v3.py's solve_two_phase
docstring for the depth-axis convention and validate_solver_v3.py's B.3
test for the validation this rests on). Produces a DIFFERENT dataset file
(dataset_v3_gravity_N.npz, not dataset_v3_N.npz) -- these are physically
different scenarios, not a config toggle on the same data, and should not
be mixed into one training set without the surrogate being told which
physics regime each sample came from.

Run:
    python data_gen_v3_multiphase.py [n_samples] [--gravity]
    (defaults to 1000 -- see the timing note below before raising this)

Timing note: solve_two_phase's pressure solve is a direct sparse linear
solve assembled with a per-cell Python loop (see solver_v3.py's
solve_pressure_implicit), which is the actual cost driver, not well count.
Measured ~1.7s/sample at 32x32 with n_pressure_steps=100; at this script's
default n_pressure_steps=150 (matching v1/v2's nt=150) that's
~2.5s/sample -> ~40min for 1000 samples, ~85min for 2000. Start with 1000;
regenerate at 2000 only once the pipeline below is confirmed working end
to end on the smaller set.
"""
import numpy as np

from data_gen import random_permeability_field
from solver_v2 import random_porosity_field
from solver_v3 import solve_two_phase

NX, NY = 32, 32


def sample_well_configuration_v3(nx, ny, max_wells=6, margin=3):
    """
    Random well count/locations/rates/types. Unlike data_gen.py and
    data_gen_v2.py (producers only), this mixes injectors and producers --
    each well independently 50/50 -- so the dataset actually contains the
    combined production+injection scenarios v3's solver can represent but
    v1/v2 never could.
    """
    n_wells = np.random.randint(2, max_wells + 1)
    locs = [
        (np.random.randint(margin, nx - margin), np.random.randint(margin, ny - margin))
        for _ in range(n_wells)
    ]
    rates = np.random.uniform(0.5, 1.5, size=n_wells)  # magnitudes only --
    # solve_two_phase takes the sign via well_is_water_injector, not via
    # the sign of well_rates itself (see solver_v3.py's source_total loop)
    is_injector = np.random.rand(n_wells) < 0.5
    if not is_injector.any():
        # force at least one injector -- an all-producer sample has no
        # water source at all, so Sw stays at Swc everywhere and the
        # saturation channel degenerates to a constant field for that
        # sample, which is a wasted training example, not a meaningful
        # "pure depletion" case (solver_v3's compressible pressure eq.
        # already covers pure depletion fine without needing Sw variation)
        is_injector[np.random.randint(n_wells)] = True
    return locs, rates, is_injector


def generate_dataset_v3(n_samples=1000, nx=NX, ny=NY, max_wells=6,
                         n_pressure_steps=150, seed=None,
                         add_gravity=False, rho_w=1.0, rho_n=0.7, g=1.0):
    if seed is not None:
        np.random.seed(seed)

    perms = np.zeros((n_samples, nx, ny))
    phis = np.zeros((n_samples, nx, ny))
    well_masks = np.zeros((n_samples, nx, ny))
    final_pressures = np.zeros((n_samples, nx, ny))
    final_saturations = np.zeros((n_samples, nx, ny))
    all_locs, all_rates, all_is_injector = [], [], []

    for s in range(n_samples):
        perm = random_permeability_field(nx, ny)
        phi = random_porosity_field(nx, ny)
        locs, rates, is_injector = sample_well_configuration_v3(nx, ny, max_wells)

        p_hist, s_hist, dt = solve_two_phase(
            perm, phi, well_locations=locs, well_rates=rates,
            well_is_water_injector=is_injector,
            nx=nx, ny=ny, n_pressure_steps=n_pressure_steps, save_every=30,
            add_gravity=add_gravity, rho_w=rho_w, rho_n=rho_n, g=g,
        )

        mask = np.zeros((nx, ny))
        for (wi, wj), r, inj in zip(locs, rates, is_injector):
            mask[wi, wj] = r if inj else -r

        perms[s] = perm
        phis[s] = phi
        well_masks[s] = mask
        final_pressures[s] = p_hist[-1]
        final_saturations[s] = s_hist[-1]
        all_locs.append(locs); all_rates.append(rates); all_is_injector.append(is_injector)

        if (s + 1) % 50 == 0:
            print(f"  generated {s+1}/{n_samples}")

    return {
        "permeability": perms, "porosity": phis, "well_mask": well_masks,
        "final_pressure": final_pressures, "final_saturation": final_saturations,
        "well_locations": all_locs, "well_rates": all_rates, "well_is_injector": all_is_injector,
    }


if __name__ == "__main__":
    import sys
    import time

    args = sys.argv[1:]
    use_gravity = "--gravity" in args
    args = [a for a in args if a != "--gravity"]
    n_samples = int(args[0]) if args else 1000

    t0 = time.time()
    ds = generate_dataset_v3(n_samples=n_samples, nx=NX, ny=NY, n_pressure_steps=150,
                              seed=42, add_gravity=use_gravity, rho_w=1.0, rho_n=0.6, g=2.0)
    elapsed = time.time() - t0
    print(f"generated {len(ds['permeability'])} samples in {elapsed:.1f}s "
          f"({elapsed/len(ds['permeability'])*1000:.1f} ms/sample)"
          f"{' [gravity ON]' if use_gravity else ''}")

    print("final_saturation range across dataset:",
          ds["final_saturation"].min(), "to", ds["final_saturation"].max())
    print("sanity: fraction of cells at connate Sw=0.2 in sample 0:",
          float(np.mean(np.isclose(ds["final_saturation"][0], 0.2))))

    out_path = f"dataset_v3_gravity_{n_samples}.npz" if use_gravity else f"dataset_v3_{n_samples}.npz"
    np.savez_compressed(
        out_path,
        permeability=ds["permeability"], porosity=ds["porosity"],
        well_mask=ds["well_mask"], final_pressure=ds["final_pressure"],
        final_saturation=ds["final_saturation"],
    )
    import os
    print(f"saved {out_path}, size MB:", os.path.getsize(out_path) / 1e6)
