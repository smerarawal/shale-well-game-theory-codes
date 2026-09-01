"""
Visualize Stage 2 results for presentation. Reads the .npz files saved by
shapley_values.py, blotto_solver.py, and blotto_constrained.py.

Run locally (or in Kaggle) after those scripts have produced their .npz files:
    pip install numpy matplotlib
    python visualize_stage2.py
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_shapley(npz_path="stage2_shapley_result.npz", out_path="shapley_plot.png"):
    d = np.load(npz_path, allow_pickle=True)
    exact = d["exact_shapley"].item()
    mc = d["mc_shapley"].item()

    agents = list(exact.keys())
    exact_vals = [exact[a] for a in agents]
    mc_vals = [mc[a] for a in agents]

    x = np.arange(len(agents))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width/2, exact_vals, width, label="Exact Shapley")
    ax.bar(x + width/2, mc_vals, width, label="Monte Carlo Shapley")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("Payoff share")
    ax.set_title("Shapley value allocation by agent")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


def plot_blotto(npz_path="stage2_blotto_result.npz", out_path="blotto_plot.png"):
    d = np.load(npz_path, allow_pickle=True)
    perm = d["permeability"]
    locs_a = d["locs_a"]
    locs_b = d["locs_b"]

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(perm.T, origin="lower", cmap="viridis", alpha=0.6)
    ax.scatter(locs_a[:, 0], locs_a[:, 1], c="red", marker="^", s=120, label="Agent A wells", edgecolors="black")
    ax.scatter(locs_b[:, 0], locs_b[:, 1], c="blue", marker="^", s=120, label="Agent B wells", edgecolors="black")
    ax.set_title("Blotto equilibrium well placement")
    ax.legend()
    plt.colorbar(im, ax=ax, fraction=0.046, label="permeability")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


def plot_constrained_blotto_summary(npz_path="stage2c_constrained_blotto_result.npz", out_path="constrained_blotto_summary.png"):
    d = np.load(npz_path, allow_pickle=True)
    results = d["results"]

    seeds = [r["seed"] for r in results]
    payoffs_a = [r["payoff_a"] for r in results]
    payoffs_b = [r["payoff_b"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(seeds, payoffs_a, "o-", label="Agent A", color="red")
    ax.plot(seeds, payoffs_b, "o-", label="Agent B", color="blue")
    ax.set_xlabel("Random seed (different starting conditions)")
    ax.set_ylabel("Equilibrium payoff")
    ax.set_title("Constrained-region Blotto: payoff stability across seeds")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


if __name__ == "__main__":
    import os

    if os.path.exists("stage2_shapley_result.npz"):
        plot_shapley()
    if os.path.exists("stage2_blotto_result.npz"):
        plot_blotto()
    if os.path.exists("stage2c_constrained_blotto_result.npz"):
        plot_constrained_blotto_summary()
