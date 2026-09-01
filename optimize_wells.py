"""
Stage 1: single-agent well placement optimization.

Given a fixed permeability field, choose k well locations (out of a large
candidate grid) to maximize total payoff, using the trained FNO as a fast
payoff function instead of the FD solver (that's the whole point of Stage 0).

Two methods, so you can sanity-check one against the other:
  1. Greedy submodular selection -- add wells one at a time, each time
     picking the location with the highest marginal payoff gain. Standard
     approach for "choose k of n" problems with diminishing returns
     (interference between wells makes this submodular-ish, not
     provably submodular without further analysis -- flag this to your
     supervisor as an assumption, not a proven property).
  2. Random search baseline -- for comparison, so you know if greedy is
     actually doing anything useful.

Run:
    pip install torch neuraloperator numpy scipy matplotlib
    python optimize_wells.py

Expects dataset_2000.npz (for normalization stats) and fno_surrogate.pt
in the same directory.
"""
import numpy as np
import torch
from neuralop.models import FNO

from data_gen import random_permeability_field

NPZ_PATH = "dataset_2000.npz"
CKPT_PATH = "fno_surrogate.pt"
NX, NY = 32, 32
WELL_RATE = -0.8          # fixed production rate per well (matches training distribution)
MARGIN = 2                # keep wells off the boundary, matches training data_gen
K_WELLS = 5                # how many wells this agent gets to place
N_CANDIDATES = 200         # candidate location pool size (subsample the grid)


def load_norm_stats(npz_path):
    d = np.load(npz_path)
    perm = d["permeability"].astype(np.float32)
    pressure = d["final_pressure"].astype(np.float32)
    return perm.mean(), perm.std(), pressure.mean(), pressure.std()


def load_fno(ckpt_path, device):
    model = FNO(n_modes=(16, 16), hidden_channels=32, in_channels=2, out_channels=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    return model


def predict_pressure(model, perm, well_locs, well_rates, norm_stats, device):
    """Single fast forward pass through the FNO. Returns (nx, ny) pressure field."""
    perm_mean, perm_std, p_mean, p_std = norm_stats
    mask = np.zeros((NX, NY), dtype=np.float32)
    for (wi, wj), r in zip(well_locs, well_rates):
        mask[wi, wj] = r

    x = np.stack([(perm - perm_mean) / perm_std, mask], axis=0)[None, ...].astype(np.float32)
    x_t = torch.from_numpy(x).to(device)
    with torch.no_grad():
        pred_norm = model(x_t)
    pred = pred_norm.cpu().numpy()[0, 0] * p_std + p_mean
    return pred


def total_payoff(pressure, well_locs):
    """
    Proxy payoff: sum of |pressure drawdown| at well locations.
    NOTE: this is a simplified single-timestep proxy (FNO only predicts
    final pressure, not the full time history the FD solver gives you).
    If you need the discounted-cumulative-production version from Stage 0's
    compute_well_payoffs, you'd need an FNO trained to output the full
    trajectory, not just the final field -- flag this simplification
    when presenting results.
    """
    return sum(abs(pressure[wi, wj]) for (wi, wj) in well_locs)


def greedy_well_placement(model, perm, k, candidates, norm_stats, device):
    """
    Greedy submodular-style selection: repeatedly add the candidate
    location that gives the largest marginal payoff increase.
    Returns chosen locations and the payoff trajectory (for plotting).
    """
    chosen = []
    remaining = list(candidates)
    payoff_trajectory = []

    for step in range(k):
        best_loc = None
        best_payoff = -np.inf
        for loc in remaining:
            trial_locs = chosen + [loc]
            trial_rates = [WELL_RATE] * len(trial_locs)
            pressure = predict_pressure(model, perm, trial_locs, trial_rates, norm_stats, device)
            payoff = total_payoff(pressure, trial_locs)
            if payoff > best_payoff:
                best_payoff = payoff
                best_loc = loc

        chosen.append(best_loc)
        remaining.remove(best_loc)
        payoff_trajectory.append(best_payoff)
        print(f"  step {step+1}/{k}: added well at {best_loc}, "
              f"total payoff = {best_payoff:.3f}")

    return chosen, payoff_trajectory


def random_search_baseline(model, perm, k, candidates, norm_stats, device, n_trials=200):
    """Best of n_trials random k-subsets, for comparison against greedy."""
    best_locs = None
    best_payoff = -np.inf
    for _ in range(n_trials):
        trial_locs = list(map(tuple, np.array(candidates)[
            np.random.choice(len(candidates), size=k, replace=False)
        ]))
        trial_rates = [WELL_RATE] * k
        pressure = predict_pressure(model, perm, trial_locs, trial_rates, norm_stats, device)
        payoff = total_payoff(pressure, trial_locs)
        if payoff > best_payoff:
            best_payoff = payoff
            best_locs = trial_locs
    return best_locs, best_payoff


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm_stats = load_norm_stats(NPZ_PATH)
    model = load_fno(CKPT_PATH, device)

    np.random.seed(7)
    perm = random_permeability_field(NX, NY)

    # candidate pool: random subsample of interior grid points
    all_interior = [(i, j) for i in range(MARGIN, NX - MARGIN) for j in range(MARGIN, NY - MARGIN)]
    candidates = [all_interior[i] for i in np.random.choice(len(all_interior), N_CANDIDATES, replace=False)]

    print(f"Optimizing placement of {K_WELLS} wells from {N_CANDIDATES} candidates")
    print("Greedy submodular selection:")
    greedy_locs, greedy_trajectory = greedy_well_placement(model, perm, K_WELLS, candidates, norm_stats, device)
    greedy_final_payoff = greedy_trajectory[-1]

    print("\nRandom search baseline (200 trials):")
    random_locs, random_payoff = random_search_baseline(model, perm, K_WELLS, candidates, norm_stats, device)
    print(f"  best random payoff = {random_payoff:.3f}")

    print(f"\nGreedy payoff:  {greedy_final_payoff:.3f}")
    print(f"Random payoff:  {random_payoff:.3f}")
    print(f"Greedy improvement over random: {(greedy_final_payoff / random_payoff - 1) * 100:.1f}%")

    # save results for inspection / plotting
    np.savez(
        "stage1_optimization_result.npz",
        permeability=perm,
        greedy_locs=np.array(greedy_locs),
        greedy_trajectory=np.array(greedy_trajectory),
        random_locs=np.array(random_locs),
        random_payoff=random_payoff,
    )
    print("\nsaved stage1_optimization_result.npz")


if __name__ == "__main__":
    main()
