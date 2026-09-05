"""
shapley_values_v2.py -- shapley_values.py fully repointed to the v2
physics (anisotropic permeability + porosity) FNO. This is the worked
example of the repointing pattern described in optimize_wells_v2.py's
docstring -- use this as the template for rewiring blotto_solver.py,
blotto_constrained.py, and core_and_nucleolus.py the same way if you
need those on v2 physics too.

Changes from shapley_values.py:
  - imports from optimize_wells_v2 instead of optimize_wells
  - `perm = random_permeability_field(...)` -> `scenario = generate_scenario(...)`
  - every `coalition_payoff(model, perm, ...)` -> `coalition_payoff(model, scenario, ...)`

Run:
    python shapley_values_v2.py

Expects dataset_v2_2000.npz and fno_surrogate_v2.pt in the same directory.
"""
import itertools
import numpy as np
import torch

from optimize_wells_v2 import load_fno, predict_pressure, total_payoff, load_norm_stats, WELL_RATE, generate_scenario

NX, NY = 32, 32


def coalition_payoff(model, scenario, agent_wells, coalition, norm_stats, device):
    if len(coalition) == 0:
        return 0.0
    locs = []
    for a in coalition:
        locs.extend(agent_wells[a])
    rates = [WELL_RATE] * len(locs)
    pressure = predict_pressure(model, scenario, locs, rates, norm_stats, device)
    return total_payoff(pressure, locs)


def exact_shapley_values(model, scenario, agent_wells, norm_stats, device):
    agents = list(agent_wells.keys())
    n = len(agents)
    shapley = {a: 0.0 for a in agents}
    cache = {}

    def get_payoff(coalition):
        key = tuple(sorted(coalition))
        if key not in cache:
            cache[key] = coalition_payoff(model, scenario, agent_wells, key, norm_stats, device)
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


def verify_properties(shapley, grand_coalition_payoff, agent_wells, tol=1e-6):
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
    norm_stats = load_norm_stats()
    model = load_fno(device=device)

    scenario = generate_scenario(seed=11)

    agent_wells = {
        "A": [(8, 8), (10, 10)],
        "B": [(9, 9)],
        "C": [(20, 20), (22, 22), (20, 24)],
        "D": [],
    }

    print("Agents and well counts (v2 physics: anisotropic k + porosity):",
          {a: len(w) for a, w in agent_wells.items()})

    exact_vals, cache = exact_shapley_values(model, scenario, agent_wells, norm_stats, device)
    print("\n--- Exact Shapley values (v2) ---")
    for a, v in exact_vals.items():
        print(f"  agent {a}: {v:.4f}")

    grand_coalition = tuple(sorted(agent_wells.keys()))
    grand_payoff = cache[grand_coalition]
    verify_properties(exact_vals, grand_payoff, agent_wells)

    print("\nCompare these values against your v1 shapley_values.py run "
          "(A=8.6913, B=5.0132, C=9.7689, D=0.0000) to see how much the "
          "richer physics (anisotropy + porosity) changes the fairness result.")

    np.savez("stage2_shapley_v2_result.npz", exact_shapley=exact_vals, grand_coalition_payoff=grand_payoff, allow_pickle=True)
    print("\nsaved stage2_shapley_v2_result.npz")


if __name__ == "__main__":
    main()
