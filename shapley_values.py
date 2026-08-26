"""
Stage 2a: Shapley values for cooperative multi-agent well placement.

Setup: N agents, each already assigned a fixed subset of wells on the
shared grid (who places where is decided elsewhere -- this script answers
"given these placements, how should the total payoff be fairly split?").

Exact Shapley values: factorial in N, only run for N <= ~6.
Monte Carlo approximation: scales to larger N via random permutation sampling.

Run:
    pip install torch neuraloperator numpy scipy
    python shapley_values.py

Expects dataset_2000.npz and fno_surrogate.pt in the same directory
(reuses load_fno / predict_pressure / total_payoff from optimize_wells.py).
"""
import itertools
import math
import numpy as np
import torch

from optimize_wells import load_fno, predict_pressure, total_payoff, load_norm_stats, WELL_RATE
from data_gen import random_permeability_field

NPZ_PATH = "dataset_2000.npz"
CKPT_PATH = "fno_surrogate.pt"
NX, NY = 32, 32


def coalition_payoff(model, perm, agent_wells, coalition, norm_stats, device):
    """
    Total payoff of a coalition (subset of agent indices): pool all wells
    belonging to agents in the coalition, run them through the FNO
    TOGETHER (so interference between different agents' wells is captured),
    and return the total payoff.

    agent_wells: dict {agent_id: [(i,j), ...]} -- each agent's well locations
    coalition: tuple/list of agent_ids in this coalition
    """
    if len(coalition) == 0:
        return 0.0
    locs = []
    for a in coalition:
        locs.extend(agent_wells[a])
    rates = [WELL_RATE] * len(locs)
    pressure = predict_pressure(model, perm, locs, rates, norm_stats, device)
    return total_payoff(pressure, locs)


def exact_shapley_values(model, perm, agent_wells, norm_stats, device):
    """
    Exact Shapley value via the permutation formula:
        phi_i = (1/N!) * sum over all permutations of
                [v(coalition up to and including i) - v(coalition before i)]

    Only tractable for small N (<=6 or so -- N! grows fast).
    Returns dict {agent_id: shapley_value}.
    """
    agents = list(agent_wells.keys())
    n = len(agents)
    shapley = {a: 0.0 for a in agents}

    # cache coalition payoffs so we don't recompute the same subset twice
    cache = {}

    def get_payoff(coalition):
        key = tuple(sorted(coalition))
        if key not in cache:
            cache[key] = coalition_payoff(model, perm, agent_wells, key, norm_stats, device)
        return cache[key]

    n_perms = 0
    for perm_order in itertools.permutations(agents):
        n_perms += 1
        coalition = []
        prev_payoff = 0.0
        for agent in perm_order:
            coalition.append(agent)
            new_payoff = get_payoff(coalition)
            marginal = new_payoff - prev_payoff
            shapley[agent] += marginal
            prev_payoff = new_payoff

    for a in agents:
        shapley[a] /= n_perms

    return shapley, cache


def monte_carlo_shapley(model, perm, agent_wells, norm_stats, device, n_samples=500, seed=0):
    """
    Monte Carlo approximation: sample random permutations instead of
    enumerating all N!. Scales to larger N. Returns dict {agent_id: value}.
    """
    rng = np.random.default_rng(seed)
    agents = list(agent_wells.keys())
    n = len(agents)
    shapley = {a: 0.0 for a in agents}
    cache = {}

    def get_payoff(coalition):
        key = tuple(sorted(coalition))
        if key not in cache:
            cache[key] = coalition_payoff(model, perm, agent_wells, key, norm_stats, device)
        return cache[key]

    for _ in range(n_samples):
        order = list(agents)
        rng.shuffle(order)
        coalition = []
        prev_payoff = 0.0
        for agent in order:
            coalition.append(agent)
            new_payoff = get_payoff(coalition)
            shapley[agent] += new_payoff - prev_payoff
            prev_payoff = new_payoff

    for a in agents:
        shapley[a] /= n_samples

    return shapley, cache


def verify_properties(shapley, grand_coalition_payoff, agent_wells, tol=1e-6):
    """
    Sanity checks from cooperative game theory:
      - Efficiency: sum of Shapley values == grand coalition payoff
      - Null player: an agent with zero wells gets zero Shapley value
    """
    total = sum(shapley.values())
    eff_ok = abs(total - grand_coalition_payoff) < max(tol, 0.01 * abs(grand_coalition_payoff))
    print(f"Efficiency check: sum(shapley)={total:.4f} vs grand coalition={grand_coalition_payoff:.4f} "
          f"-> {'PASS' if eff_ok else 'FAIL'}")

    for a, wells in agent_wells.items():
        if len(wells) == 0:
            null_ok = abs(shapley[a]) < tol
            print(f"Null player check (agent {a}, 0 wells): shapley={shapley[a]:.6f} "
                  f"-> {'PASS' if null_ok else 'FAIL'}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm_stats = load_norm_stats(NPZ_PATH)
    model = load_fno(CKPT_PATH, device)

    np.random.seed(11)
    perm = random_permeability_field(NX, NY)

    # example: 4 agents, each with a few wells, placed close enough to interfere
    agent_wells = {
        "A": [(8, 8), (10, 10)],
        "B": [(9, 9)],            # deliberately near A -> interference
        "C": [(20, 20), (22, 22), (20, 24)],
        "D": [],                   # null player -- should get 0 shapley value
    }

    print("Agents and well counts:", {a: len(w) for a, w in agent_wells.items()})

    print("\n--- Exact Shapley values ---")
    exact_vals, cache = exact_shapley_values(model, perm, agent_wells, norm_stats, device)
    for a, v in exact_vals.items():
        print(f"  agent {a}: {v:.4f}")

    grand_coalition = tuple(sorted(agent_wells.keys()))
    grand_payoff = cache[grand_coalition]
    verify_properties(exact_vals, grand_payoff, agent_wells)

    print("\n--- Monte Carlo Shapley approximation (500 samples) ---")
    mc_vals, _ = monte_carlo_shapley(model, perm, agent_wells, norm_stats, device, n_samples=500)
    for a, v in mc_vals.items():
        exact_v = exact_vals[a]
        err = abs(v - exact_v) / (abs(exact_v) + 1e-8) * 100
        print(f"  agent {a}: mc={v:.4f}  exact={exact_v:.4f}  error={err:.1f}%")

    np.savez(
        "stage2_shapley_result.npz",
        permeability=perm,
        exact_shapley=exact_vals,
        mc_shapley=mc_vals,
        grand_coalition_payoff=grand_payoff,
        allow_pickle=True,
    )
    print("\nsaved stage2_shapley_result.npz")


if __name__ == "__main__":
    main()
