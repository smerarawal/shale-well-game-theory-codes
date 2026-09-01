"""
Stage 2c: constrained-region Colonel Blotto.

The Stage 2b run converged trivially in 1 round because agents A and B
simply moved to opposite corners of a big candidate pool -- avoidance,
not real competition. This version confines BOTH agents to the same
restricted sub-region of the grid, so they cannot simply avoid each other
and must genuinely compete for overlapping territory. This is the more
interesting/harder equilibrium.

Run:
    pip install torch neuraloperator numpy scipy
    python blotto_constrained.py

Expects dataset_2000.npz and fno_surrogate.pt in the same directory.
"""
import numpy as np
import torch

from optimize_wells import load_fno, predict_pressure, total_payoff, load_norm_stats, WELL_RATE
from blotto_solver import best_response, iterated_best_response
from data_gen import random_permeability_field

NPZ_PATH = "dataset_2000.npz"
CKPT_PATH = "fno_surrogate.pt"
NX, NY = 32, 32
K_WELLS_PER_AGENT = 4

# the contested sub-region: both agents' candidates come ONLY from here,
# so they can't just retreat to opposite corners like Stage 2b did
REGION_X = (10, 22)   # inclusive-exclusive range along x
REGION_Y = (10, 22)   # same size region -> 12x12 = 144 candidate cells


def build_constrained_candidates(region_x, region_y):
    return [(i, j) for i in range(*region_x) for j in range(*region_y)]


def run_multiple_seeds(model, perm, candidates, n_seeds=5):
    """
    Run iterated best response from several different random starting
    points for agent B. Nash equilibria aren't always unique -- running
    multiple seeds tells you whether you consistently land on the same
    outcome (robust equilibrium) or get different outcomes depending on
    who moves first / where they start (multiple equilibria -- also a
    valid, reportable finding, not a bug).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm_stats = load_norm_stats(NPZ_PATH)

    results = []
    for seed in range(n_seeds):
        print(f"\n=== seed {seed} ===")
        locs_a, locs_b, history = iterated_best_response(
            model, perm, K_WELLS_PER_AGENT, K_WELLS_PER_AGENT,
            candidates, norm_stats, device, seed=seed
        )
        final_payoff_a = history[-1]["payoff_a"]
        final_payoff_b = history[-1]["payoff_b"]
        n_rounds = len(history)
        results.append({
            "seed": seed,
            "locs_a": locs_a,
            "locs_b": locs_b,
            "payoff_a": final_payoff_a,
            "payoff_b": final_payoff_b,
            "n_rounds": n_rounds,
        })
    return results


def summarize(results):
    print("\n=== Summary across seeds ===")
    payoffs_a = [r["payoff_a"] for r in results]
    payoffs_b = [r["payoff_b"] for r in results]
    rounds = [r["n_rounds"] for r in results]

    print(f"agent A payoff: mean={np.mean(payoffs_a):.3f}  std={np.std(payoffs_a):.3f}")
    print(f"agent B payoff: mean={np.mean(payoffs_b):.3f}  std={np.std(payoffs_b):.3f}")
    print(f"rounds to converge: mean={np.mean(rounds):.1f}  max={np.max(rounds)}")

    # check whether wells actually overlap/cluster together now (real contestation)
    for r in results:
        locs_a = set(r["locs_a"])
        locs_b = set(r["locs_b"])
        min_dist = min(
            abs(ax - bx) + abs(ay - by)
            for (ax, ay) in locs_a for (bx, by) in locs_b
        )
        print(f"  seed {r['seed']}: closest A-B well pair distance = {min_dist} grid cells")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm_stats = load_norm_stats(NPZ_PATH)
    model = load_fno(CKPT_PATH, device)

    np.random.seed(23)
    perm = random_permeability_field(NX, NY)

    candidates = build_constrained_candidates(REGION_X, REGION_Y)
    print(f"Constrained region: {REGION_X[1]-REGION_X[0]}x{REGION_Y[1]-REGION_Y[0]} "
          f"= {len(candidates)} candidate cells (both agents share this pool)")

    results = run_multiple_seeds(model, perm, candidates, n_seeds=5)
    summarize(results)

    np.savez(
        "stage2c_constrained_blotto_result.npz",
        permeability=perm,
        results=results,
        region_x=REGION_X,
        region_y=REGION_Y,
        allow_pickle=True,
    )
    print("\nsaved stage2c_constrained_blotto_result.npz")


if __name__ == "__main__":
    main()
