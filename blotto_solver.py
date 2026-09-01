"""
Stage 2b: Colonel Blotto framing -- two rival agents independently choose
where to place a fixed budget of wells on the same/overlapping grid, each
maximizing their OWN payoff given the other's placement. No coordination.

Classic Blotto is only analytically solvable for simplified symmetric
cases -- here we approximate a Nash equilibrium via iterated best response:
alternately let each agent re-optimize its placement (via the Stage 1
greedy method) against the other's current fixed placement, repeat until
neither wants to move (or a max iteration cap).

Run:
    pip install torch neuraloperator numpy scipy
    python blotto_solver.py

Expects dataset_2000.npz and fno_surrogate.pt in the same directory.
"""
import numpy as np
import torch

from optimize_wells import (
    load_fno, predict_pressure, total_payoff, load_norm_stats,
    WELL_RATE, MARGIN,
)
from data_gen import random_permeability_field

NPZ_PATH = "dataset_2000.npz"
CKPT_PATH = "fno_surrogate.pt"
NX, NY = 32, 32
K_WELLS_PER_AGENT = 4
N_CANDIDATES = 150
MAX_ROUNDS = 10


def best_response(model, perm, own_k, opponent_locs, candidates, norm_stats, device):
    """
    Greedy best response: place own_k wells to maximize OWN payoff, given
    the opponent's wells are already fixed on the grid (their pressure
    interference is included in every evaluation).
    Same greedy logic as Stage 1's optimize_wells.py, extended to hold
    a fixed opponent well set as background interference.
    """
    chosen = []
    remaining = list(candidates)

    for _ in range(own_k):
        best_loc = None
        best_marginal = -np.inf
        for loc in remaining:
            trial_own = chosen + [loc]
            all_locs = trial_own + opponent_locs
            all_rates = [WELL_RATE] * len(all_locs)
            pressure = predict_pressure(model, perm, all_locs, all_rates, norm_stats, device)
            own_payoff = total_payoff(pressure, trial_own)  # only OWN wells count for own payoff
            if own_payoff > best_marginal:
                best_marginal = own_payoff
                best_loc = loc
        chosen.append(best_loc)
        remaining.remove(best_loc)

    return chosen, best_marginal


def iterated_best_response(model, perm, k_a, k_b, candidates, norm_stats, device, max_rounds=MAX_ROUNDS, seed=0):
    """
    Alternate best-response between agent A and agent B until placements
    stop changing (approximate pure-strategy Nash equilibrium) or the
    round cap is hit. Returns final placements and payoff history.
    """
    rng = np.random.default_rng(seed)
    # random initial placement for B, so A's first move isn't against an empty grid
    locs_b = [candidates[i] for i in rng.choice(len(candidates), k_b, replace=False)]
    locs_a = []

    history = []
    for round_num in range(max_rounds):
        avail_for_a = [c for c in candidates if c not in locs_b]
        new_locs_a, payoff_a = best_response(model, perm, k_a, locs_b, avail_for_a, norm_stats, device)

        avail_for_b = [c for c in candidates if c not in new_locs_a]
        new_locs_b, payoff_b = best_response(model, perm, k_b, new_locs_a, avail_for_b, norm_stats, device)

        history.append({"round": round_num, "payoff_a": payoff_a, "payoff_b": payoff_b})
        print(f"round {round_num}: agent A payoff={payoff_a:.3f}, agent B payoff={payoff_b:.3f}")

        converged = (new_locs_a == locs_a) and (new_locs_b == locs_b)
        locs_a, locs_b = new_locs_a, new_locs_b
        if converged:
            print(f"converged after {round_num} rounds (neither agent wants to move)")
            break
    else:
        print(f"hit max_rounds={max_rounds} without full convergence -- "
              "check the payoff history above for oscillation vs near-stability")

    return locs_a, locs_b, history


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm_stats = load_norm_stats(NPZ_PATH)
    model = load_fno(CKPT_PATH, device)

    np.random.seed(23)
    perm = random_permeability_field(NX, NY)

    all_interior = [(i, j) for i in range(MARGIN, NX - MARGIN) for j in range(MARGIN, NY - MARGIN)]
    candidates = [all_interior[i] for i in np.random.choice(len(all_interior), N_CANDIDATES, replace=False)]

    print(f"Solving Blotto: agent A and B each place {K_WELLS_PER_AGENT} wells "
          f"from {N_CANDIDATES} shared candidates\n")

    locs_a, locs_b, history = iterated_best_response(
        model, perm, K_WELLS_PER_AGENT, K_WELLS_PER_AGENT, candidates, norm_stats, device
    )

    print(f"\nFinal placement A: {locs_a}")
    print(f"Final placement B: {locs_b}")

    np.savez(
        "stage2_blotto_result.npz",
        permeability=perm,
        locs_a=np.array(locs_a),
        locs_b=np.array(locs_b),
        payoff_history=history,
        allow_pickle=True,
    )
    print("saved stage2_blotto_result.npz")


if __name__ == "__main__":
    main()
