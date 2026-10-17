"""
make_spe10_ood32.py -- 32x32 OOD sets for the BENCHMARK checkpoints, built from SPE10 patches
(see spe10_geologies.py for the patch recipe, the rescaling to the training range, and the
ASSUMED, unverified file ordering).

Same task as the benchmark data: random 2-4 wells (random_wells, injectors + producers),
gravity on, 150 pressure steps, final pressure/saturation. Output files are named like the other
OOD sets (ood_spe10_tarbert.npz, ood_spe10_ness.npz) and have the same keys, so they can be
dropped into final_run/ood/ for evaluate_ood.py, or passed to evaluate_extended_metrics.py with
--extra_ood_dir.

The solver run uses root solver_v3.py; its extra arguments (p_init, t_end, return_time) default to
the old behaviour, so the fields are produced exactly like the benchmark data.

    python make_spe10_ood32.py --data_dir /kaggle/input/spe10 --out_dir /kaggle/working/spe10_ood \
        --n_cases 200 --workers 4
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from data_gen_v3 import make_well_mask, random_wells  # noqa: E402
from solver_v3 import solve_two_phase  # noqa: E402
from spe10_geologies import TARBERT, UPPER_NESS, load_spe10, sample_patches  # noqa: E402

REGIONS = {"tarbert": TARBERT, "ness": UPPER_NESS}


def solve_case(args):
    patch, seed, n, steps = args
    k, phi = patch["k"], patch["phi"]
    locs, rates, inj = random_wells(n, n, seed)
    wm = make_well_mask(n, n, locs, rates, inj)
    p_hist, s_hist, _ = solve_two_phase(
        k, phi, well_locations=locs, well_rates=rates, well_is_water_injector=inj,
        n_pressure_steps=steps, save_every=max(1, steps // 5), add_gravity=True, rho_w=1.0, rho_n=0.6, g=1.5)
    return k, phi, wm, np.stack(p_hist), np.stack(s_hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder with spe_perm.dat and spe_phi.dat")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_cases", type=int, default=200)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--coarsen", type=int, default=1, help="1 = native cells (32-cell crops need width >= 32; SPE10 is 60 wide)")
    ap.add_argument("--regions", nargs="+", default=["tarbert", "ness"], choices=list(REGIONS))
    ap.add_argument("--seed", type=int, default=1_900_000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    logk, phi = load_spe10(a.data_dir)
    for ri, region in enumerate(a.regions):
        patches = sample_patches(logk, phi, REGIONS[region], a.n_cases, a.n, a.coarsen, a.seed + ri)
        jobs = [(p, a.seed + 1000 * ri + c, a.n, a.steps) for c, p in enumerate(patches)]
        t0 = time.time()
        with Pool(a.workers) as pool:
            res = pool.map(solve_case, jobs, chunksize=2)
        out = {"permeability": np.stack([r[0] for r in res]), "porosity": np.stack([r[1] for r in res]),
               "well_mask": np.stack([r[2] for r in res]),
               "pressure_traj": np.stack([r[3] for r in res]), "saturation_traj": np.stack([r[4] for r in res]),
               "spe10_layer": np.array([p["layer"] for p in patches]),
               "spe10_logk_range": np.array([p["logk_range"] for p in patches])}
        out["final_pressure"], out["final_saturation"] = out["pressure_traj"][:, -1], out["saturation_traj"][:, -1]
        path = os.path.join(a.out_dir, f"ood_spe10_{region}.npz")
        np.savez_compressed(path, **out)
        print(f"{region}: {len(res)} cases -> {path} ({time.time() - t0:.0f}s); "
              f"final p range {out['final_pressure'].min():.2f}..{out['final_pressure'].max():.2f}", flush=True)


if __name__ == "__main__":
    main()
