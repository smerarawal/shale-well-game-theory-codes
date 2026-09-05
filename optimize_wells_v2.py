"""
optimize_wells_v2.py -- same role as optimize_wells.py (load model, predict
pressure, compute payoff, greedy placement) but wired to the v2 FNO
(kx, ky, porosity, well_mask -> pressure) instead of the original
(permeability, well_mask -> pressure).

IMPORTANT -- this is NOT a drop-in replacement despite matching function
names. v1's predict_pressure(model, perm, ...) took a single permeability
array. v2's predict_pressure(model, scenario, ...) takes a `scenario` dict
with kx/ky/porosity instead. Any Stage 2/3 script you repoint at this
module (shapley_values.py, blotto_solver.py, blotto_constrained.py,
core_and_nucleolus.py) needs every call site that builds/passes `perm`
updated to build/pass a `scenario` dict via generate_scenario() instead.
The import line changes from:
    from optimize_wells import load_fno, predict_pressure, total_payoff, load_norm_stats, WELL_RATE
to:
    from optimize_wells_v2 import load_fno, predict_pressure, total_payoff, load_norm_stats, WELL_RATE
but the `perm = random_permeability_field(...)` lines in those scripts
must also become `scenario = generate_scenario(...)`, and every
`predict_pressure(model, perm, ...)` call becomes
`predict_pressure(model, scenario, ...)`.

Run standalone for a smoke test:
    python optimize_wells_v2.py
"""
import numpy as np
import torch
from neuralop.models import FNO

from solver_v2 import random_anisotropic_permeability, random_porosity_field

NPZ_PATH = "dataset_v2_2000.npz"
CKPT_PATH = "fno_surrogate_v2.pt"
NX, NY = 32, 32
WELL_RATE = -0.8
MARGIN = 2
K_WELLS = 5
N_CANDIDATES = 200


def load_norm_stats(npz_path=NPZ_PATH):
    d = np.load(npz_path)
    kx = d["kx"].astype(np.float32); ky = d["ky"].astype(np.float32)
    phi = d["porosity"].astype(np.float32); pressure = d["final_pressure"].astype(np.float32)
    return (kx.mean(), kx.std(), ky.mean(), ky.std(),
            phi.mean(), phi.std(), pressure.mean(), pressure.std())


def load_fno(ckpt_path=CKPT_PATH, device="cpu"):
    model = FNO(n_modes=(16, 16), hidden_channels=32, in_channels=4, out_channels=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    return model


def generate_scenario(seed=None):
    """Convenience: build one fresh (kx, ky, porosity) triple, replacing
    the single `perm` field every v1 script used to generate/pass around."""
    if seed is not None:
        np.random.seed(seed)
    kx, ky = random_anisotropic_permeability(NX, NY)
    phi = random_porosity_field(NX, NY)
    return {"kx": kx, "ky": ky, "porosity": phi}


def predict_pressure(model, scenario, well_locs, well_rates, norm_stats, device):
    """
    Takes `scenario` (a dict with kx/ky/porosity) instead of a single
    `perm` array -- see module docstring for what this means for callers.
    """
    kx_mean, kx_std, ky_mean, ky_std, phi_mean, phi_std, p_mean, p_std = norm_stats
    mask = np.zeros((NX, NY), dtype=np.float32)
    for (wi, wj), r in zip(well_locs, well_rates):
        mask[wi, wj] = r

    x = np.stack([
        (scenario["kx"] - kx_mean) / kx_std,
        (scenario["ky"] - ky_mean) / ky_std,
        (scenario["porosity"] - phi_mean) / phi_std,
        mask,
    ], axis=0)[None, ...].astype(np.float32)
    x_t = torch.from_numpy(x).to(device)
    with torch.no_grad():
        pred_norm = model(x_t)
    pred = pred_norm.cpu().numpy()[0, 0] * p_std + p_mean
    return pred


def total_payoff(pressure, well_locs):
    """Unchanged from v1 -- same proxy payoff definition."""
    return sum(abs(pressure[wi, wj]) for (wi, wj) in well_locs)


def greedy_well_placement(model, scenario, k, candidates, norm_stats, device):
    chosen = []
    remaining = list(candidates)
    payoff_trajectory = []

    for step in range(k):
        best_loc = None
        best_payoff = -np.inf
        for loc in remaining:
            trial_locs = chosen + [loc]
            trial_rates = [WELL_RATE] * len(trial_locs)
            pressure = predict_pressure(model, scenario, trial_locs, trial_rates, norm_stats, device)
            payoff = total_payoff(pressure, trial_locs)
            if payoff > best_payoff:
                best_payoff = payoff
                best_loc = loc
        chosen.append(best_loc)
        remaining.remove(best_loc)
        payoff_trajectory.append(best_payoff)
        print(f"  step {step+1}/{k}: added well at {best_loc}, total payoff = {best_payoff:.3f}")

    return chosen, payoff_trajectory


def random_search_baseline(model, scenario, k, candidates, norm_stats, device, n_trials=200):
    best_locs, best_payoff = None, -np.inf
    for _ in range(n_trials):
        trial_locs = list(map(tuple, np.array(candidates)[
            np.random.choice(len(candidates), size=k, replace=False)
        ]))
        trial_rates = [WELL_RATE] * k
        pressure = predict_pressure(model, scenario, trial_locs, trial_rates, norm_stats, device)
        payoff = total_payoff(pressure, trial_locs)
        if payoff > best_payoff:
            best_payoff = payoff
            best_locs = trial_locs
    return best_locs, best_payoff


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm_stats = load_norm_stats()
    model = load_fno(device=device)

    scenario = generate_scenario(seed=7)
    all_interior = [(i, j) for i in range(MARGIN, NX - MARGIN) for j in range(MARGIN, NY - MARGIN)]
    candidates = [all_interior[i] for i in np.random.choice(len(all_interior), N_CANDIDATES, replace=False)]

    print(f"Optimizing placement of {K_WELLS} wells (v2 physics: anisotropic k + porosity)")
    greedy_locs, greedy_trajectory = greedy_well_placement(model, scenario, K_WELLS, candidates, norm_stats, device)
    random_locs, random_payoff = random_search_baseline(model, scenario, K_WELLS, candidates, norm_stats, device)

    print(f"\nGreedy payoff:  {greedy_trajectory[-1]:.3f}")
    print(f"Random payoff:  {random_payoff:.3f}")
    print(f"Greedy improvement over random: {(greedy_trajectory[-1] / random_payoff - 1) * 100:.1f}%")


if __name__ == "__main__":
    main()
