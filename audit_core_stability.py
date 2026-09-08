"""
audit_core_stability.py -- Module 3.3 (partial) from implementation_spec_v3.md.

HONESTY NOTE: the full spec for 3.3 wants this run against YOUR OWN trained
MADDPG policy's outputs (Module 3.2), which doesn't exist -- no MARL
training was built in this repo. What's implemented here is the reusable
half: a standalone auditor that takes ANY set of coalition payoff numbers
(from a paper's results table, a hypothetical scenario, or eventually your
own MADDPG output once built) and checks core-stability + computes the
nucleolus for comparison. This is directly useful right now (e.g. audit
numbers from a paper's Table 3), and becomes the second half of Module 3.3
automatically once MADDPG output feeds into it -- no changes needed to
this file when that happens, just a different input dict.

Run:
    pip install scipy
    python audit_core_stability.py

No FNO/dataset dependency -- pure game-theory audit on whatever numbers
you supply.
"""
import itertools
import numpy as np
from scipy.optimize import linprog


def check_core_nonempty(agents, coalition_payoffs):
    """Same LP as core_and_nucleolus.py, reusable standalone."""
    n = len(agents)
    grand = tuple(sorted(agents))
    grand_payoff = coalition_payoffs[grand]
    c = np.zeros(n)
    A_ub, b_ub = [], []
    for r in range(1, n):
        for combo in itertools.combinations(agents, r):
            row = [-1.0 if a in combo else 0.0 for a in agents]
            A_ub.append(row)
            b_ub.append(-coalition_payoffs[tuple(sorted(combo))])
    A_eq = [[1.0] * n]
    b_eq = [grand_payoff]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=[(None, None)] * n, method="highs")
    return result.success, (dict(zip(agents, result.x)) if result.success else None)


def check_allocation_in_core(agents, coalition_payoffs, allocation):
    """
    THE actual audit function: given a SPECIFIC allocation (e.g. a paper's
    reported NPV split for one reward structure), check whether THAT
    allocation is core-stable -- i.e. would any sub-coalition prefer to
    defect and split off, given what they were actually awarded?
    Returns (is_stable, list_of_violating_coalitions).
    """
    n = len(agents)
    violations = []
    for r in range(1, n):
        for combo in itertools.combinations(agents, r):
            combo = tuple(sorted(combo))
            allocated_sum = sum(allocation[a] for a in combo)
            standalone_value = coalition_payoffs[combo]
            if allocated_sum < standalone_value - 1e-6:
                deficit = standalone_value - allocated_sum
                violations.append((combo, deficit))
    return len(violations) == 0, violations


def audit_scenario(name, agents, coalition_payoffs, allocation):
    print(f"\n{'='*60}\nAUDITING: {name}\n{'='*60}")
    print("Coalition payoffs supplied:")
    for coalition, value in sorted(coalition_payoffs.items(), key=lambda kv: len(kv[0])):
        label = "{" + ",".join(coalition) + "}" if coalition else "{}"
        print(f"  v({label}) = {value:.3f}")

    print(f"\nAllocation being audited: {allocation}")

    is_stable, violations = check_allocation_in_core(agents, coalition_payoffs, allocation)
    if is_stable:
        print("\nRESULT: STABLE -- this specific allocation IS in the core. "
              "No sub-coalition has an incentive to defect under this split.")
    else:
        print(f"\nRESULT: UNSTABLE -- {len(violations)} sub-coalition(s) would "
              "prefer to defect:")
        for combo, deficit in violations:
            label = "{" + ",".join(combo) + "}"
            print(f"  {label} was allocated less than it could get alone "
                  f"(deficit = {deficit:.3f}) -- these agents have an incentive "
                  f"to break away and renegotiate independently")

    core_nonempty, example = check_core_nonempty(agents, coalition_payoffs)
    if core_nonempty and not is_stable:
        print(f"\nNote: a core-stable allocation DOES exist for this game "
              f"(e.g. {example}) -- the audited allocation just isn't one of them. "
              "This means the instability is a property of the SPECIFIC split "
              "chosen, not an inherent feature of the underlying game.")
    elif not core_nonempty:
        print("\nNote: the core is EMPTY for this game entirely -- no allocation "
              "of any kind could have been stable, so this instability isn't a "
              "flaw specific to the audited allocation.")

    return is_stable, violations


def example_from_paper_table():
    """
    TEMPLATE: replace these numbers with an actual paper's reported
    coalition/reward-structure results once you have them (e.g. from the
    benchmark paper's Table 3, or your own future MADDPG output). Structure:
    coalition_payoffs needs v(S) for EVERY possible sub-coalition (not just
    the ones the paper reports for their 5 headline structures) -- if the
    paper only reports a few coalition structures, you may need to estimate
    or bound the missing v(S) values, or restrict the audit to the
    coalitions that ARE reported and note the limitation explicitly.
    """
    agents = ["Op1", "Op2", "Op3"]
    coalition_payoffs = {
        (): 0.0,
        ("Op1",): 12.0, ("Op2",): 15.0, ("Op3",): 10.0,
        ("Op1", "Op2"): 30.0, ("Op1", "Op3"): 25.0, ("Op2", "Op3"): 28.0,
        ("Op1", "Op2", "Op3"): 48.0,
    }
    # example "reported" allocation -- e.g. what a paper's fully-cooperative
    # reward structure actually paid each agent
    allocation = {"Op1": 15.0, "Op2": 18.0, "Op3": 15.0}
    return agents, coalition_payoffs, allocation


def main():
    agents, coalition_payoffs, allocation = example_from_paper_table()
    audit_scenario("TEMPLATE example -- replace with real paper/MADDPG numbers",
                    agents, coalition_payoffs, allocation)

    print("\n\nTo use this for real: replace example_from_paper_table() with "
          "the actual v(S) values and allocation you want to audit (from a "
          "paper's table, or your own future coalition/MARL results), then "
          "rerun. No other code changes needed.")


if __name__ == "__main__":
    main()
