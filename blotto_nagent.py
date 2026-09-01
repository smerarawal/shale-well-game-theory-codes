"""
Stage 3b: N-agent Blotto. So far only 2 rival agents. Real shale acreage
often has 3+ competing operators. This generalizes iterated best response
to N agents, cycling through each in turn (round-robin), each optimizing
against everyone else's current fixed placement.

Run:
    python blotto_nagent.py
"""
import numpy as np
import torch

from optimize_wells import load_fno, predict_pressure, total_payoff, load_norm_stats, WELL_RATE, MARGIN
from data_gen import random_permeability_field

NPZ_PATH = "dataset_2000.npz"
CKPT_PATH = "fno_surrogate.pt"
NX, NY = 32, 32
N_AGENTS = 4
K_WELLS_PER_AGENT = 3
N_CANDIDATES = 200
MAX_ROUNDS = 15


def best_response_vs_others(model, perm, own_k, other_agents_locs, candidates, norm_stats, device):
    """
    Same greedy logic as the 2-agent version, but "opponent" is now the
    pooled wells of ALL other agents combined (their interference stacks).
    """
    other_locs = [loc for locs in other_agents_locs for loc in locs]

    chosen = []
    remaining = list(candidates)
    for _ in range(own_k):
        best_loc = None
        best_payoff = -np.inf
        for loc in remaining:
            trial_own = chosen + [loc]
            all_locs = trial_own + other_locs
            all_rates = [WELL_RATE] * len(all_locs)
            pressure = predict_pressure(model, perm, all_locs, all_rates, norm_stats, device)
            own_payoff = total_payoff(pressure, trial_own)
            if own_payoff > best_payoff:
                best_payoff = own_payoff
                best_loc = loc
        chosen.append(best_loc)
        remaining.remove(best_loc)

    return chosen, best_payoff


def iterated_best_response_nagent(model, perm, n_agents, k_per_agent, candidates, norm_stats, device,
                                    max_rounds=MAX_ROUNDS, seed=0):
    """
    Round-robin best response: agent 0 responds to {1,2,...,n-1}'s current
    placements, then agent 1 responds to the updated set, etc, one full
    pass = one round. Repeat until no agent's placement changes.
    """
    rng = np.random.default_rng(seed)
    # random initial placements for all agents
    placements = []
    used = set()
    for a in range(n_agents):
        avail = [c for c in candidates if c not in used]
        picks = [avail[i] for i in rng.choice(len(avail), k_per_agent, replace=False)]
        placements.append(picks)
        used.update(picks)

    history = []
    for round_num in range(max_rounds):
        old_placements = [list(p) for p in placements]
        round_payoffs = []

        for a in range(n_agents):
            others = [placements[o] for o in range(n_agents) if o != a]
            used_by_others = set(loc for locs in others for loc in locs)
            avail = [c for c in candidates if c not in used_by_others]
            new_locs, payoff = best_response_vs_others(model, perm, k_per_agent, others, avail, norm_stats, device)
            placements[a] = new_locs
            round_payoffs.append(payoff)

        history.append({"round": round_num, "payoffs": round_payoffs})
        print(f"round {round_num}: payoffs = " + "  ".join(f"agent{i}={p:.3f}" for i, p in enumerate(round_payoffs)))

        converged = all(placements[a] == old_placements[a] for a in range(n_agents))
        if converged:
            print(f"converged after {round_num} rounds")
            break
    else:
        print(f"hit max_rounds={max_rounds} without converging -- check history for oscillation")

    return placements, history


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm_stats = load_norm_stats(NPZ_PATH)
    model = load_fno(CKPT_PATH, device)

    np.random.seed(31)
    perm = random_permeability_field(NX, NY)

    all_interior = [(i, j) for i in range(MARGIN, NX - MARGIN) for j in range(MARGIN, NY - MARGIN)]
    candidates = [all_interior[i] for i in np.random.choice(len(all_interior), N_CANDIDATES, replace=False)]

    print(f"{N_AGENTS}-agent Blotto: each places {K_WELLS_PER_AGENT} wells "
          f"from {N_CANDIDATES} shared candidates\n")

    placements, history = iterated_best_response_nagent(
        model, perm, N_AGENTS, K_WELLS_PER_AGENT, candidates, norm_stats, device
    )

    print("\nFinal placements:")
    for i, locs in enumerate(placements):
        print(f"  agent {i}: {locs}")

    np.savez(
        "stage3_nagent_blotto_result.npz",
        permeability=perm,
        placements=placements,
        history=history,
        allow_pickle=True,
    )
    print("saved stage3_nagent_blotto_result.npz")


if __name__ == "__main__":
    main()
