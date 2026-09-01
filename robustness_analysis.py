"""
Stage 3a: robustness across geology. Everything so far (Shapley values,
Blotto equilibrium) was computed on ONE random permeability field. A
reviewer's first question will be "does this hold in general, or did you
get lucky with one map?" This runs the same analyses across multiple
independently generated permeability fields and reports how much the
results vary.

Run:
    python robustness_analysis.py
"""
import numpy as np
import torch

from optimize_wells import load_fno, load_norm_stats, WELL_RATE, MARGIN
from shapley_values import exact_shapley_values
from blotto_solver import iterated_best_response
from data_gen import random_permeability_field

NPZ_PATH = "dataset_2000.npz"
CKPT_PATH = "fno_surrogate.pt"
NX, NY = 32, 32
N_GEOLOGIES = 10          # number of independent permeability fields to test
K_WELLS_PER_AGENT = 4
N_CANDIDATES = 150


def run_shapley_across_geologies(model, norm_stats, device, n_geologies=N_GEOLOGIES):
    agent_wells = {
        "A": [(8, 8), (10, 10)],
        "B": [(9, 9)],
        "C": [(20, 20), (22, 22), (20, 24)],
    }
    results = []
    for g in range(n_geologies):
        np.random.seed(1000 + g)  # different geology each time, offset from other seed ranges
        perm = random_permeability_field(NX, NY)
        shapley, cache = exact_shapley_values(model, perm, agent_wells, norm_stats, device)
        results.append(shapley)
        print(f"geology {g}: " + "  ".join(f"{a}={v:.3f}" for a, v in shapley.items()))
    return results


def summarize_shapley_robustness(results):
    agents = list(results[0].keys())
    print("\n--- Shapley robustness across geologies ---")
    for a in agents:
        vals = [r[a] for r in results]
        cv = np.std(vals) / (np.mean(vals) + 1e-8) * 100  # coefficient of variation
        print(f"agent {a}: mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  "
              f"coefficient of variation={cv:.1f}%")
    print("(low CV = agent's relative share is stable across geologies; "
          "high CV = this agent's fair share is very geology-dependent)")


def run_blotto_across_geologies(model, norm_stats, device, n_geologies=N_GEOLOGIES):
    results = []
    for g in range(n_geologies):
        np.random.seed(2000 + g)
        perm = random_permeability_field(NX, NY)
        all_interior = [(i, j) for i in range(MARGIN, NX - MARGIN) for j in range(MARGIN, NY - MARGIN)]
        candidates = [all_interior[i] for i in np.random.choice(len(all_interior), N_CANDIDATES, replace=False)]

        locs_a, locs_b, history = iterated_best_response(
            model, perm, K_WELLS_PER_AGENT, K_WELLS_PER_AGENT, candidates, norm_stats, device, seed=g
        )
        final = history[-1]
        results.append({
            "geology": g,
            "payoff_a": final["payoff_a"],
            "payoff_b": final["payoff_b"],
            "n_rounds": len(history),
        })
        print(f"geology {g}: payoff_a={final['payoff_a']:.3f}  "
              f"payoff_b={final['payoff_b']:.3f}  rounds={len(history)}")
    return results


def summarize_blotto_robustness(results):
    payoffs_a = [r["payoff_a"] for r in results]
    payoffs_b = [r["payoff_b"] for r in results]
    rounds = [r["n_rounds"] for r in results]
    print("\n--- Blotto robustness across geologies ---")
    print(f"agent A payoff: mean={np.mean(payoffs_a):.3f}  std={np.std(payoffs_a):.3f}")
    print(f"agent B payoff: mean={np.mean(payoffs_b):.3f}  std={np.std(payoffs_b):.3f}")
    print(f"rounds to converge: mean={np.mean(rounds):.1f}  max={np.max(rounds)}  "
          f"(all hitting max_rounds without converging would be a red flag -- check this)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm_stats = load_norm_stats(NPZ_PATH)
    model = load_fno(CKPT_PATH, device)

    print(f"=== Shapley robustness across {N_GEOLOGIES} geologies ===")
    shapley_results = run_shapley_across_geologies(model, norm_stats, device)
    summarize_shapley_robustness(shapley_results)

    print(f"\n=== Blotto robustness across {N_GEOLOGIES} geologies ===")
    blotto_results = run_blotto_across_geologies(model, norm_stats, device)
    summarize_blotto_robustness(blotto_results)

    np.savez(
        "stage3_robustness_result.npz",
        shapley_results=shapley_results,
        blotto_results=blotto_results,
        allow_pickle=True,
    )
    print("\nsaved stage3_robustness_result.npz")


if __name__ == "__main__":
    main()
