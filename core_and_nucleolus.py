"""
core_and_nucleolus.py -- two more cooperative game theory solution
concepts, beyond Shapley values, computed on the SAME coalition payoffs
your shapley_values.py already produces.

THE CORE: the set of payoff splits where no sub-coalition would want to
break away and go it alone. A split x is "in the core" if, for every
possible sub-coalition S, the sum of x's shares for members of S is at
least as much as S could get by defecting (v(S)). If the core is empty,
NO split can satisfy every possible sub-coalition simultaneously -- a
real, reportable finding about whether stable cooperation is even
possible for this group of agents.

THE NUCLEOLUS: a single specific point INSIDE the core (when the core is
non-empty) that minimizes the maximum "complaint" (dissatisfaction) of
any coalition -- the fairest single point according to a different
criterion than Shapley's. Computed here via the standard iterative LP
approach (successively fixing which constraints are "tight" and
resolving).

Run:
    pip install scipy
    python core_and_nucleolus.py

Expects dataset_2000.npz and fno_surrogate.pt in the same directory
(reuses coalition_payoff logic from shapley_values.py).
"""
import itertools
import numpy as np
import torch
from scipy.optimize import linprog

from optimize_wells import load_fno, load_norm_stats
from shapley_values import coalition_payoff, exact_shapley_values
from data_gen import random_permeability_field

NPZ_PATH = "dataset_2000.npz"
CKPT_PATH = "fno_surrogate.pt"
NX, NY = 32, 32


def all_coalition_payoffs(model, perm, agent_wells, norm_stats, device):
    """Precompute v(S) for every possible sub-coalition S (2^N of them)."""
    agents = list(agent_wells.keys())
    payoffs = {}
    for r in range(len(agents) + 1):
        for combo in itertools.combinations(agents, r):
            payoffs[combo] = coalition_payoff(model, perm, agent_wells, combo, norm_stats, device)
    return payoffs


def check_core_nonempty(agents, coalition_payoffs):
    """
    Check whether the core is non-empty by solving a feasibility LP:
    find x (payoff to each agent) such that:
      - sum(x) == v(grand coalition)          [efficiency]
      - for every proper sub-coalition S: sum(x_i for i in S) >= v(S)  [stability]
    If this LP is feasible, the core is non-empty and `x` returned is
    ONE point in the core (not necessarily the "best" one -- that's what
    the nucleolus is for).
    """
    n = len(agents)
    grand = tuple(sorted(agents))
    grand_payoff = coalition_payoffs[grand]

    # minimize 0 (pure feasibility problem) subject to core constraints
    c = np.zeros(n)

    A_ub = []
    b_ub = []
    for r in range(1, n):  # proper non-empty sub-coalitions only
        for combo in itertools.combinations(agents, r):
            # constraint: -sum(x_i for i in combo) <= -v(combo)
            row = [-1.0 if a in combo else 0.0 for a in agents]
            A_ub.append(row)
            b_ub.append(-coalition_payoffs[tuple(sorted(combo))])

    A_eq = [[1.0] * n]
    b_eq = [grand_payoff]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=[(None, None)] * n, method="highs")

    return result.success, (dict(zip(agents, result.x)) if result.success else None)


def compute_nucleolus(agents, coalition_payoffs, max_iterations=20):
    """
    Standard iterative nucleolus algorithm: repeatedly minimize the
    largest "excess" (how much a coalition is unhappy: v(S) - x(S)),
    then fix whichever constraints become tight (equalities) and
    re-minimize the remaining slack, until the allocation is fully
    pinned down. This is the textbook approach (Maschler et al.) --
    implemented directly rather than via a library, since no
    lightweight one is guaranteed available in your environment.
    """
    n = len(agents)
    grand = tuple(sorted(agents))
    grand_payoff = coalition_payoffs[grand]

    all_coalitions = [tuple(sorted(c)) for r in range(1, n)
                       for c in itertools.combinations(agents, r)]

    fixed_equalities = []  # list of (coalition, value) pairs pinned so far
    remaining = list(all_coalitions)

    x = None
    for iteration in range(max_iterations):
        if not remaining:
            break

        # variables: x_1..x_n, plus epsilon (the max excess we're minimizing)
        c = np.zeros(n + 1)
        c[-1] = 1.0  # minimize epsilon

        A_ub = []
        b_ub = []
        for coalition in remaining:
            # v(S) - sum(x_i for i in S) <= epsilon
            row = [-1.0 if a in coalition else 0.0 for a in agents] + [-1.0]
            A_ub.append(row)
            b_ub.append(-coalition_payoffs[coalition])

        A_eq = [[1.0] * n + [0.0]]
        b_eq = [grand_payoff]
        for coalition, value in fixed_equalities:
            row = [1.0 if a in coalition else 0.0 for a in agents] + [0.0]
            A_eq.append(row)
            b_eq.append(value)

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=[(None, None)] * n + [(None, None)], method="highs")

        if not result.success:
            print(f"iteration {iteration}: LP infeasible -- stopping (core may be empty)")
            break

        x = result.x[:n]
        epsilon = result.x[-1]

        # find which coalitions are now "tight" (excess == epsilon exactly)
        newly_tight = []
        still_slack = []
        for coalition in remaining:
            coalition_sum = sum(x[agents.index(a)] for a in coalition)
            excess = coalition_payoffs[coalition] - coalition_sum
            if abs(excess - epsilon) < 1e-6:
                newly_tight.append(coalition)
            else:
                still_slack.append(coalition)

        for coalition in newly_tight:
            coalition_sum = sum(x[agents.index(a)] for a in coalition)
            fixed_equalities.append((coalition, coalition_sum))

        remaining = still_slack
        print(f"iteration {iteration}: epsilon={epsilon:.4f}, "
              f"{len(newly_tight)} coalitions newly fixed, {len(remaining)} remaining")

        if not newly_tight:
            print("no progress -- stopping to avoid infinite loop")
            break

    return dict(zip(agents, x)) if x is not None else None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm_stats = load_norm_stats(NPZ_PATH)
    model = load_fno(CKPT_PATH, device)

    np.random.seed(11)
    perm = random_permeability_field(NX, NY)

    agent_wells = {
        "A": [(8, 8), (10, 10)],
        "B": [(9, 9)],
        "C": [(20, 20), (22, 22), (20, 24)],
    }
    agents = list(agent_wells.keys())

    print("Computing all coalition payoffs...")
    coalition_payoffs = all_coalition_payoffs(model, perm, agent_wells, norm_stats, device)

    print("\n--- Shapley values (for comparison) ---")
    shapley, _ = exact_shapley_values(model, perm, agent_wells, norm_stats, device)
    for a, v in shapley.items():
        print(f"  {a}: {v:.4f}")

    print("\n--- Core check ---")
    core_nonempty, example_point = check_core_nonempty(agents, coalition_payoffs)
    print(f"Core non-empty: {core_nonempty}")
    if core_nonempty:
        print("Example point in the core:")
        for a, v in example_point.items():
            print(f"  {a}: {v:.4f}")
        # check whether Shapley itself happens to lie in the core
        shapley_in_core = all(
            sum(shapley[a] for a in coalition) >= coalition_payoffs[coalition] - 1e-4
            for coalition in coalition_payoffs if len(coalition) < len(agents)
        )
        print(f"\nIs the Shapley allocation itself in the core? {shapley_in_core}")
    else:
        print("Core is EMPTY -- no allocation can satisfy every sub-coalition's "
              "stability constraint simultaneously. This means: under this exact "
              "geology and these exact well positions, there is no way to split "
              "profits that prevents SOME sub-group of agents from being better "
              "off breaking away and negotiating separately. Worth reporting as-is.")

    print("\n--- Nucleolus ---")
    nucleolus = compute_nucleolus(agents, coalition_payoffs)
    if nucleolus:
        print("Nucleolus allocation:")
        for a, v in nucleolus.items():
            print(f"  {a}: {v:.4f}")

        print("\n--- Comparison: Shapley vs Nucleolus ---")
        for a in agents:
            diff = shapley[a] - nucleolus[a]
            print(f"  {a}: shapley={shapley[a]:.4f}  nucleolus={nucleolus[a]:.4f}  diff={diff:+.4f}")

    np.savez(
        "core_nucleolus_result.npz",
        shapley=shapley,
        core_nonempty=core_nonempty,
        nucleolus=nucleolus,
        allow_pickle=True,
    )
    print("\nsaved core_nucleolus_result.npz")


if __name__ == "__main__":
    main()
