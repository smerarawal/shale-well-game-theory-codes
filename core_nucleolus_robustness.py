"""
core_nucleolus_robustness.py -- checks whether the empty-core finding
from core_and_nucleolus.py is a one-off feature of one geology/agent-set,
or a consistent pattern. Sweeps across BOTH multiple random geologies AND
increasing numbers of agents.

NOTE: this still uses the ORIGINAL trained FNO (permeability + well_mask
only) -- solver_v2.py's porosity/anisotropy upgrades are validated as
standalone physics but have NOT been trained into the FNO surrogate yet.
Integrating them requires: (1) regenerating the dataset with solver_v2.py
producing (kx, ky, phi, well_mask) as 4 input channels instead of 2,
(2) retraining fno_train.py with in_channels=4, (3) rerunning Stage 1-3
against the new model. That's a substantial follow-up stage, not done
here -- flagged so you don't assume this run reflects the richer physics.

Run:
    python core_nucleolus_robustness.py

Expects dataset_2000.npz and fno_surrogate.pt in the same directory.
"""
import itertools
import numpy as np
import torch

from optimize_wells import load_fno, load_norm_stats
from shapley_values import exact_shapley_values
from core_and_nucleolus import all_coalition_payoffs, check_core_nonempty, compute_nucleolus
from data_gen import random_permeability_field

NPZ_PATH = "dataset_2000.npz"
CKPT_PATH = "fno_surrogate.pt"
NX, NY = 32, 32
N_GEOLOGIES = 8


def make_agent_wells(n_agents, seed):
    """
    Generate a random agent/well configuration for a given agent count.
    Wells kept spread out but with enough overlap potential to interfere.
    """
    rng = np.random.default_rng(seed)
    agent_wells = {}
    used = set()
    for i in range(n_agents):
        agent_id = chr(ord("A") + i)
        n_wells = rng.integers(1, 3)  # 1-2 wells per agent, keeps N! tractable for exact Shapley
        locs = []
        attempts = 0
        while len(locs) < n_wells and attempts < 50:
            loc = (int(rng.integers(4, NX - 4)), int(rng.integers(4, NY - 4)))
            if loc not in used:
                locs.append(loc)
                used.add(loc)
            attempts += 1
        agent_wells[agent_id] = locs
    return agent_wells


def run_sweep(model, norm_stats, device, agent_counts=(3, 4, 5), n_geologies=N_GEOLOGIES):
    results = []
    for n_agents in agent_counts:
        print(f"\n{'='*60}\nAGENT COUNT = {n_agents}\n{'='*60}")
        for g in range(n_geologies):
            np.random.seed(5000 + g)
            perm = random_permeability_field(NX, NY)
            agent_wells = make_agent_wells(n_agents, seed=100 * n_agents + g)

            coalition_payoffs = all_coalition_payoffs(model, perm, agent_wells, norm_stats, device)
            agents = list(agent_wells.keys())

            core_ok, _ = check_core_nonempty(agents, coalition_payoffs)

            shapley, _ = exact_shapley_values(model, perm, agent_wells, norm_stats, device)
            nucleolus = compute_nucleolus(agents, coalition_payoffs) if core_ok else None

            max_diff = None
            if nucleolus:
                max_diff = max(abs(shapley[a] - nucleolus[a]) for a in agents)

            print(f"  geology {g}: core_nonempty={core_ok}  "
                  f"max|shapley-nucleolus| diff={max_diff}")

            results.append({
                "n_agents": n_agents, "geology": g,
                "core_nonempty": core_ok, "max_shapley_nucleolus_diff": max_diff,
            })

    return results


def summarize(results):
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    by_agent_count = {}
    for r in results:
        by_agent_count.setdefault(r["n_agents"], []).append(r)

    for n_agents, rs in by_agent_count.items():
        n_empty = sum(1 for r in rs if not r["core_nonempty"])
        n_total = len(rs)
        diffs = [r["max_shapley_nucleolus_diff"] for r in rs if r["max_shapley_nucleolus_diff"] is not None]
        print(f"\n{n_agents} agents ({n_total} geologies tested):")
        print(f"  empty core in {n_empty}/{n_total} geologies ({n_empty/n_total*100:.0f}%)")
        if diffs:
            print(f"  when core non-empty: mean max shapley-vs-nucleolus disagreement = {np.mean(diffs):.3f}")

    print("\nInterpretation guide:")
    print("- If empty-core rate stays high across ALL agent counts: this reservoir/")
    print("  interference structure fundamentally resists stable cooperative splits,")
    print("  regardless of how many operators are involved -- a strong, general finding.")
    print("- If empty-core rate changes with agent count (e.g. more agents -> more/less")
    print("  empty cores): the instability is scale-dependent -- also worth reporting,")
    print("  with a hypothesis for why (e.g. more agents = more possible sub-coalitions")
    print("  to violate stability, mechanically making the core harder to satisfy).")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm_stats = load_norm_stats(NPZ_PATH)
    model = load_fno(CKPT_PATH, device)

    results = run_sweep(model, norm_stats, device, agent_counts=(3, 4, 5), n_geologies=N_GEOLOGIES)
    summarize(results)

    np.savez("core_nucleolus_robustness_result.npz", results=results, allow_pickle=True)
    print("\nsaved core_nucleolus_robustness_result.npz")


if __name__ == "__main__":
    main()
