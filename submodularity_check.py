"""
Stage 3c: submodularity check. Stage 1's greedy well placement is only
guaranteed near-optimal (the classic 1 - 1/e bound) if the payoff function
is submodular -- meaning each additional well gives diminishing marginal
returns as more wells are already placed. This was FLAGGED AS AN
ASSUMPTION, not proven, in optimize_wells.py. This script tests it
empirically: add wells one at a time in random order, many times, and
check whether marginal gains are (on average) decreasing.

This is an empirical check, not a formal proof -- report it as evidence
for/against the assumption, not as a guarantee.

Run:
    python submodularity_check.py
"""
import numpy as np
import torch

from optimize_wells import load_fno, predict_pressure, total_payoff, load_norm_stats, WELL_RATE, MARGIN
from data_gen import random_permeability_field

NPZ_PATH = "dataset_2000.npz"
CKPT_PATH = "fno_surrogate.pt"
NX, NY = 32, 32
N_TRIALS = 30       # random orderings to test
MAX_WELLS = 10      # add up to this many wells per trial
N_CANDIDATES = 300


def marginal_gains_for_random_order(model, perm, candidates, norm_stats, device, max_wells, seed):
    rng = np.random.default_rng(seed)
    order = [candidates[i] for i in rng.choice(len(candidates), max_wells, replace=False)]

    chosen = []
    prev_payoff = 0.0
    gains = []
    for loc in order:
        chosen.append(loc)
        rates = [WELL_RATE] * len(chosen)
        pressure = predict_pressure(model, perm, chosen, rates, norm_stats, device)
        payoff = total_payoff(pressure, chosen)
        gains.append(payoff - prev_payoff)
        prev_payoff = payoff

    return gains


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm_stats = load_norm_stats(NPZ_PATH)
    model = load_fno(CKPT_PATH, device)

    np.random.seed(41)
    perm = random_permeability_field(NX, NY)
    all_interior = [(i, j) for i in range(MARGIN, NX - MARGIN) for j in range(MARGIN, NY - MARGIN)]
    candidates = [all_interior[i] for i in np.random.choice(len(all_interior), N_CANDIDATES, replace=False)]

    all_gains = []
    for trial in range(N_TRIALS):
        gains = marginal_gains_for_random_order(model, perm, candidates, norm_stats, device, MAX_WELLS, seed=trial)
        all_gains.append(gains)
        print(f"trial {trial}: marginal gains = " + "  ".join(f"{g:.2f}" for g in gains))

    all_gains = np.array(all_gains)  # (N_TRIALS, MAX_WELLS)
    mean_gain_by_position = all_gains.mean(axis=0)

    print("\n--- Mean marginal gain by well count (averaged over trials) ---")
    for i, g in enumerate(mean_gain_by_position):
        print(f"  well #{i+1}: mean marginal gain = {g:.3f}")

    is_decreasing = all(
        mean_gain_by_position[i] >= mean_gain_by_position[i + 1] - 1e-6
        for i in range(len(mean_gain_by_position) - 1)
    )
    n_violations = sum(
        1 for i in range(len(mean_gain_by_position) - 1)
        if mean_gain_by_position[i + 1] > mean_gain_by_position[i]
    )
    print(f"\nMonotonically non-increasing on average: {is_decreasing}")
    print(f"Violations (gain went UP with more wells already placed): {n_violations} out of "
          f"{len(mean_gain_by_position)-1} transitions")
    print("(some violations are expected due to interference geometry / FNO approximation noise; "
          "a large or consistent violation count would mean the submodularity assumption doesn't "
          "hold well here, and greedy's near-optimality guarantee doesn't strictly apply)")

    np.savez("stage3_submodularity_result.npz", all_gains=all_gains, mean_gain_by_position=mean_gain_by_position)
    print("\nsaved stage3_submodularity_result.npz")


if __name__ == "__main__":
    main()
