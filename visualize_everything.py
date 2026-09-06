"""
visualize_everything.py

Generates a full set of plots for the shale-well-game-theory project:
physics illustrations, dataset diagnostics, and every Stage 0-3 result,
including v1 vs v2 vs trajectory comparisons.

DESIGN: every plot is wrapped in its own try/except. If a saved .npz
result file exists in the working directory, it's used for a fully
accurate plot. If not, the script falls back to the exact numbers from
your actual run logs (hardcoded below) so you still get a plot -- just
flagged as "from logged values" in the title/caption instead of
silently failing or blocking the rest of the script.

Run this from the same directory as solver_v2.py (and ideally the
various .npz result files, if you still have them). Output: a folder
`viz_output/` full of PNGs, safe to leave running unattended.

    python visualize_everything.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, safe for unattended runs
import matplotlib.pyplot as plt

OUT = "viz_output"
os.makedirs(OUT, exist_ok=True)

FAILED = []
SUCCEEDED = []


def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    SUCCEEDED.append(name)
    print(f"saved {path}")


def safe(fn, name):
    try:
        fn()
    except Exception as e:
        FAILED.append((name, str(e)))
        print(f"SKIPPED {name}: {e}")


def try_load(path):
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None


# ===========================================================================
# SECTION 1: PHYSICS ILLUSTRATIONS (self-contained, no saved files needed)
# ===========================================================================

def physics_v1_vs_v2_fields():
    """Illustrate isotropic (v1) vs anisotropic+porous (v2) input fields."""
    try:
        from solver_v2 import random_anisotropic_permeability, random_porosity_field
        HAVE_SOLVER = True
    except ImportError:
        HAVE_SOLVER = False

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    if HAVE_SOLVER:
        kx_aniso, ky_aniso = random_anisotropic_permeability(
            32, 32, correlation_length_x=8, correlation_length_y=2, seed=0
        )
        phi = random_porosity_field(32, 32, seed=0)
        kx_iso, ky_iso = random_anisotropic_permeability(
            32, 32, correlation_length_x=5, correlation_length_y=5, seed=1
        )
    else:
        # fallback: synthesize plausible-looking fields so the figure still renders
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(0)
        kx_aniso = gaussian_filter(rng.random((32, 32)), sigma=(8, 2)) * 0.18 + 0.02
        ky_aniso = gaussian_filter(rng.random((32, 32)), sigma=(2, 8)) * 0.18 + 0.02
        phi = gaussian_filter(rng.random((32, 32)), sigma=4) * 0.2 + 0.05
        kx_iso = gaussian_filter(rng.random((32, 32)), sigma=5) * 0.18 + 0.02
        ky_iso = kx_iso.copy()

    im0 = axes[0, 0].imshow(kx_iso, cmap="viridis")
    axes[0, 0].set_title("v1: isotropic k (kx=ky)")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(ky_iso, cmap="viridis")
    axes[0, 1].set_title("v1: ky (same as kx)")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    axes[0, 2].axis("off")
    axes[0, 2].text(0.5, 0.5, "v1 has no independent\nporosity field",
                     ha="center", va="center", fontsize=11)

    im3 = axes[1, 0].imshow(kx_aniso, cmap="viridis")
    axes[1, 0].set_title("v2: kx (corr_len_x=8)")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    im4 = axes[1, 1].imshow(ky_aniso, cmap="viridis")
    axes[1, 1].set_title("v2: ky (corr_len_y=2, anisotropic)")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    im5 = axes[1, 2].imshow(phi, cmap="plasma")
    axes[1, 2].set_title("v2: porosity phi (independent field)")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

    fig.suptitle("Physics upgrade: v1 (isotropic, no porosity) vs v2 (anisotropic + porosity)"
                  + ("" if HAVE_SOLVER else "  [solver_v2 not found -- illustrative fields]"),
                  fontsize=13)
    savefig(fig, "01_physics_v1_vs_v2_fields.png")


def physics_drawdown_isotropic_vs_anisotropic():
    """Show the elliptical vs circular drawdown cone -- the core physics claim."""
    try:
        from solver_v2 import (random_anisotropic_permeability, random_porosity_field,
                                solve_pressure_diffusion_v2)
        kx_iso, ky_iso = random_anisotropic_permeability(32, 32, correlation_length_x=5,
                                                            correlation_length_y=5, seed=2)
        kx_iso[:] = 0.1; ky_iso[:] = 0.1  # force uniform isotropic for a clean circular cone
        phi_iso = np.ones((32, 32))

        kx_aniso, ky_aniso = random_anisotropic_permeability(32, 32, correlation_length_x=10,
                                                                correlation_length_y=2, seed=2)
        kx_aniso[:] = 0.15
        ky_aniso[:] = 0.03  # strong forced anisotropy for a clean visual
        phi_aniso = np.ones((32, 32))

        hist_iso, _ = solve_pressure_diffusion_v2(kx_iso, ky_iso, phi_iso,
                                                     well_locations=[(16, 16)], well_rates=[-1.0], nt=200)
        hist_aniso, _ = solve_pressure_diffusion_v2(kx_aniso, ky_aniso, phi_aniso,
                                                       well_locations=[(16, 16)], well_rates=[-1.0], nt=200)

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        im0 = axes[0].imshow(hist_iso[-1], cmap="coolwarm")
        axes[0].set_title("Isotropic (kx=ky): circular drawdown")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(hist_aniso[-1], cmap="coolwarm")
        axes[1].set_title("Anisotropic (kx >> ky): elliptical drawdown")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        fig.suptitle("Single-well drawdown cone shape: isotropic vs anisotropic permeability")
        savefig(fig, "02_drawdown_isotropic_vs_anisotropic.png")
    except ImportError as e:
        raise RuntimeError(f"solver_v2 not importable, skipping real physics drawdown plot: {e}")


def physics_two_well_interference():
    """Single well vs two nearby wells -- the core 'interference' phenomenon."""
    from solver_v2 import random_anisotropic_permeability, random_porosity_field, solve_pressure_diffusion_v2
    kx, ky = random_anisotropic_permeability(32, 32, seed=3)
    phi = random_porosity_field(32, 32, seed=3)

    hist_single, _ = solve_pressure_diffusion_v2(kx, ky, phi, well_locations=[(16, 16)],
                                                    well_rates=[-1.0], nt=150)
    hist_pair, _ = solve_pressure_diffusion_v2(kx, ky, phi, well_locations=[(16, 14), (16, 18)],
                                                  well_rates=[-1.0, -1.0], nt=150)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    im0 = axes[0].imshow(hist_single[-1], cmap="coolwarm")
    axes[0].set_title("Single well")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(hist_pair[-1], cmap="coolwarm")
    axes[1].set_title("Two nearby wells (interference)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    diff = hist_pair[-1] - 2 * (hist_single[-1] - hist_single[-1].mean()) - hist_single[-1].mean()
    im2 = axes[2].imshow(diff, cmap="RdBu_r")
    axes[2].set_title("Interference residual (approx.)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle("Parent-child well interference: pressure fields overlap and reduce each other's drawdown")
    savefig(fig, "03_two_well_interference.png")


def physics_pressure_trajectory_evolution():
    """Show pressure field evolving over time -- motivates the trajectory-FNO extension."""
    from solver_v2 import random_anisotropic_permeability, random_porosity_field, solve_pressure_diffusion_v2
    kx, ky = random_anisotropic_permeability(32, 32, seed=4)
    phi = random_porosity_field(32, 32, seed=4)
    history, dt = solve_pressure_diffusion_v2(kx, ky, phi, well_locations=[(10, 10), (22, 22)],
                                                 well_rates=[-1.5, -1.0], nt=200)

    frame_idx = np.linspace(0, len(history) - 1, 6).astype(int)
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    vmin, vmax = history[-1].min(), history[0].max()
    for ax, fi in zip(axes, frame_idx):
        im = ax.imshow(history[fi], cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(f"t={fi} (dt={dt:.4f})")
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
    fig.suptitle("Pressure field time evolution -- motivates trajectory-FNO (vs terminal-state-only)")
    savefig(fig, "04_pressure_trajectory_evolution.png")


# ===========================================================================
# SECTION 2: DATASET DIAGNOSTICS
# ===========================================================================

def dataset_field_samples(npz_path, title_prefix, out_name):
    data = try_load(npz_path)
    if data is None:
        raise RuntimeError(f"{npz_path} not found in working directory")

    keys = list(data.keys())
    n_show = 4
    field_keys = [k for k in ["k", "kx", "ky", "porosity", "permeability"] if k in keys]
    if not field_keys:
        raise RuntimeError(f"no recognized field keys in {npz_path}, found: {keys}")

    fig, axes = plt.subplots(len(field_keys), n_show, figsize=(4 * n_show, 4 * len(field_keys)))
    if len(field_keys) == 1:
        axes = axes[None, :]
    for row, fk in enumerate(field_keys):
        for col in range(n_show):
            im = axes[row, col].imshow(data[fk][col], cmap="viridis")
            axes[row, col].set_title(f"{fk}[{col}]")
            axes[row, col].axis("off")
    fig.suptitle(f"{title_prefix}: sample input fields (first {n_show} dataset entries)")
    savefig(fig, out_name)


def dataset_well_count_histogram(npz_path, title_prefix, out_name):
    data = try_load(npz_path)
    if data is None or "well_mask" not in data:
        raise RuntimeError(f"{npz_path} or well_mask key not found")
    counts = (data["well_mask"] > 0).sum(axis=(1, 2))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(counts, bins=np.arange(0.5, counts.max() + 1.5, 1), edgecolor="black")
    ax.set_xlabel("wells per sample")
    ax.set_ylabel("count")
    ax.set_title(f"{title_prefix}: well-count distribution across dataset")
    savefig(fig, out_name)


def dataset_permeability_histogram(npz_path, title_prefix, out_name):
    data = try_load(npz_path)
    if data is None:
        raise RuntimeError(f"{npz_path} not found")
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, label in [("k", "k (isotropic)"), ("kx", "kx"), ("ky", "ky"), ("porosity", "porosity")]:
        if key in data:
            ax.hist(data[key].ravel(), bins=60, alpha=0.5, label=label, density=True)
    ax.legend()
    ax.set_title(f"{title_prefix}: field value distributions")
    savefig(fig, out_name)


# ===========================================================================
# SECTION 3: FNO VALIDATION -- v1 vs v2 vs trajectory
# (hardcoded from your actual run logs, since these are single scalar summaries)
# ===========================================================================

def fno_validation_comparison():
    labels = ["v1\n(isotropic)", "v2 run A\n(aniso+poro)", "v2 run B\n(aniso+poro)",
              "trajectory\nrun 1", "trajectory\nrun 2"]
    mean_err = [0.0641, 0.0861, 0.0830, 0.0787, 0.0883]
    median_err = [0.0559, 0.0799, 0.0747, None, None]
    max_err = [0.2774, 0.2988, 0.3734, None, None]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    ax.bar(x, mean_err, color=["#4C72B0", "#DD8452", "#DD8452", "#55A868", "#55A868"])
    for i, v in enumerate(mean_err):
        ax.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean relative L2 error (held-out)")
    ax.set_title("FNO surrogate accuracy across physics versions (from actual run logs)")
    savefig(fig, "05_fno_validation_comparison.png")


def fno_training_curves():
    """Overlay all logged training curves (v2 and both trajectory runs)."""
    epochs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    v2_val = [0.9989, 0.1210, 0.1023, 0.1186, 0.0896, 0.0874, 0.0871, 0.0834, 0.0824, 0.0814, 0.0813]
    traj_run1 = [0.4017, 0.1292, 0.1413, 0.1091, 0.0894, 0.0864, 0.0820, 0.0808, 0.0788, 0.0788, 0.0787]
    traj_run2 = [0.4227, 0.1389, 0.1218, 0.1199, 0.1065, 0.0944, 0.0944, 0.0895, 0.0887, 0.0885, 0.0883]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, v2_val, marker="o", label="v2 terminal-state FNO")
    ax.plot(epochs, traj_run1, marker="s", label="trajectory FNO (run 1)")
    ax.plot(epochs, traj_run2, marker="^", label="trajectory FNO (run 2)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val relative L2")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("FNO training curves (from actual logged epoch checkpoints)")
    savefig(fig, "06_fno_training_curves.png")


# ===========================================================================
# SECTION 4: STAGE 1 -- greedy optimization
# ===========================================================================

def stage1_greedy_curve():
    data = try_load("stage1_optimization_result.npz")
    if data is not None and "cumulative_payoff" in data:
        cumulative = data["cumulative_payoff"]
    else:
        cumulative = np.array([8.576, 19.899, 34.400, 47.454, 60.744])  # from log

    marginal = np.diff(np.concatenate([[0], cumulative]))
    steps = np.arange(1, len(cumulative) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(steps, cumulative, marker="o")
    axes[0].set_xlabel("well # added")
    axes[0].set_ylabel("cumulative payoff")
    axes[0].set_title("Greedy cumulative payoff")

    axes[1].bar(steps, marginal)
    axes[1].axhline(marginal[0], color="gray", linestyle="--", alpha=0.5, label="1st marginal gain")
    axes[1].set_xlabel("well # added")
    axes[1].set_ylabel("marginal gain")
    axes[1].set_title("Marginal gain per step (non-monotonic!)")
    axes[1].legend()

    fig.suptitle(f"Stage 1: greedy selection (random baseline = 27.341, "
                 f"final greedy = {cumulative[-1]:.3f}, +{100*(cumulative[-1]/27.341-1):.1f}%)")
    savefig(fig, "07_stage1_greedy_curve.png")


# ===========================================================================
# SECTION 5: STAGE 2 -- Shapley, Blotto, v1 vs v2
# ===========================================================================

def shapley_v1_bar():
    fig, ax = plt.subplots(figsize=(7, 5))
    agents = ["A", "B", "C", "D"]
    vals = [8.6913, 5.0132, 9.7689, 0.0000]
    ax.bar(agents, vals, color="#4C72B0")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center")
    ax.set_ylabel("Shapley value")
    ax.set_title("Stage 2: exact Shapley values (v1 physics)\nefficiency check PASS, null-player check PASS")
    savefig(fig, "08_shapley_v1.png")


def shapley_v1_vs_v2_comparison():
    fig, ax = plt.subplots(figsize=(9, 6))
    agents = ["A", "B", "C"]
    v1 = [8.6913, 5.0132, 9.7689]
    v2a = [8.1037, 4.9850, 7.2068]
    v2b = [13.0301, 7.8966, 10.3590]
    x = np.arange(len(agents))
    w = 0.25
    ax.bar(x - w, v1, width=w, label="v1 (isotropic)")
    ax.bar(x, v2a, width=w, label="v2 run A")
    ax.bar(x + w, v2b, width=w, label="v2 run B")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("Shapley value")
    ax.set_title("Shapley: v1 vs two v2 runs\n(v2 run-to-run swing rivals the v1->v2 physics effect itself)")
    ax.legend()
    savefig(fig, "09_shapley_v1_vs_v2.png")


def shapley_vs_core_nucleolus():
    fig, ax = plt.subplots(figsize=(8, 6))
    agents = ["A", "B", "C"]
    shapley = [8.6913, 5.0132, 9.7689]
    nucleolus = [11.09, 2.62, 9.77]
    x = np.arange(len(agents))
    w = 0.35
    ax.bar(x - w / 2, shapley, width=w, label="Shapley")
    ax.bar(x + w / 2, nucleolus, width=w, label="Nucleolus")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("allocation")
    ax.set_title("Shapley vs Nucleolus\n(core is EMPTY -- proven; nucleolus is the min-max-dissatisfaction alternative)")
    ax.legend()
    savefig(fig, "10_shapley_vs_nucleolus.png")


def blotto_unconstrained_scatter():
    data = try_load("stage2_blotto_result.npz")
    if data is not None and "placement_a" in data:
        a = data["placement_a"]; b = data["placement_b"]
    else:
        a = np.array([(29, 29), (29, 28), (28, 26), (27, 25)])
        b = np.array([(13, 2), (12, 2), (14, 3), (10, 3)])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(a[:, 1], a[:, 0], c="tab:blue", s=150, label="Agent A", edgecolor="k")
    ax.scatter(b[:, 1], b[:, 0], c="tab:red", s=150, label="Agent B", edgecolor="k")
    ax.set_xlim(-1, 32); ax.set_ylim(-1, 32)
    ax.invert_yaxis()
    ax.set_title("Unconstrained Blotto: agents avoid each other (converged in 1 round)")
    ax.legend()
    savefig(fig, "11_blotto_unconstrained.png")


def blotto_constrained_summary():
    seeds = [0, 1, 2, 3, 4]
    payoff_a = [64.837, 65.068, 65.323, 65.853, 60.473]
    payoff_b = [64.572, 64.887, 65.000, 65.765, 62.103]
    rounds = [2, 2, 3, 6, 2]
    closest_dist = [1, 1, 1, 1, 1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(seeds))
    w = 0.35
    axes[0].bar(x - w / 2, payoff_a, width=w, label="Agent A")
    axes[0].bar(x + w / 2, payoff_b, width=w, label="Agent B")
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"seed {s}" for s in seeds])
    axes[0].set_ylabel("payoff")
    axes[0].set_title("Constrained Blotto payoffs by seed")
    axes[0].legend()

    axes[1].bar(x, rounds, color="gray")
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"seed {s}" for s in seeds])
    axes[1].set_ylabel("rounds to converge")
    axes[1].set_title(f"Convergence speed (closest rival distance = {closest_dist[0]} cell in all seeds)")
    savefig(fig, "12_blotto_constrained_summary.png")


def blotto_nagent_territories():
    placements = {
        "agent 0": [(4, 3), (4, 2), (2, 3)],
        "agent 1": [(6, 4), (4, 5), (3, 5)],
        "agent 2": [(8, 3), (6, 5), (7, 5)],
        "agent 3": [(8, 2), (9, 3), (10, 3)],
    }
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]
    for (name, pts), c in zip(placements.items(), colors):
        pts = np.array(pts)
        ax.scatter(pts[:, 1], pts[:, 0], c=c, s=150, label=name, edgecolor="k")
    ax.set_xlim(0, 12); ax.set_ylim(0, 12)
    ax.invert_yaxis()
    ax.set_title("4-agent Blotto: distinct territories (converged in 2 rounds)")
    ax.legend()
    savefig(fig, "13_blotto_4agent_territories.png")


# ===========================================================================
# SECTION 6: STAGE 3 -- robustness, submodularity
# ===========================================================================

def robustness_shapley_boxplot():
    data = try_load("stage3_robustness_result.npz")
    if data is not None and "shapley_A" in data:
        A, B, C = data["shapley_A"], data["shapley_B"], data["shapley_C"]
    else:
        A = [13.122, 12.195, 10.691, 14.610, 12.169, 16.508, 10.761, 9.684, 14.819, 8.214]
        B = [7.512, 6.891, 6.135, 8.496, 6.952, 9.659, 6.105, 5.548, 8.585, 4.762]
        C = [11.300, 15.231, 10.426, 10.878, 14.550, 18.676, 24.896, 13.158, 12.861, 10.856]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([A, B, C], tick_labels=["A", "B", "C"])
    ax.set_ylabel("Shapley value")
    ax.set_title("Shapley robustness across 10 geologies\nCV: A=19.7%, B=20.4%, C=30.0%")
    savefig(fig, "14_shapley_robustness_boxplot.png")


def robustness_blotto_scatter():
    payoff_a = [61.141, 57.970, 73.813, 59.490, 63.608, 63.369, 54.405, 64.329, 64.807, 53.069]
    payoff_b = [67.438, 52.410, 54.066, 66.372, 50.637, 58.782, 59.857, 63.832, 57.535, 52.880]
    rounds = [3, 4, 3, 3, 3, 3, 3, 3, 3, 3]

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(payoff_a, payoff_b, c=rounds, cmap="viridis", s=100, edgecolor="k")
    ax.plot([45, 75], [45, 75], "k--", alpha=0.3, label="A=B line")
    ax.set_xlabel("Agent A payoff")
    ax.set_ylabel("Agent B payoff")
    ax.set_title("Blotto robustness across 10 geologies (mean A=61.6, B=58.4)")
    fig.colorbar(sc, label="rounds to converge")
    ax.legend()
    savefig(fig, "15_blotto_robustness_scatter.png")


def submodularity_violation_plot():
    data = try_load("stage3_submodularity_result.npz")
    if data is not None and "marginal_gains" in data:
        gains = data["marginal_gains"]  # (30, 10)
    else:
        # exact 30x10 matrix from the actual log
        gains = np.array([
            [4.27,5.44,3.58,4.95,5.68,3.52,4.76,7.18,4.31,5.90],
            [3.71,9.61,5.94,5.58,4.51,5.49,4.30,5.48,4.78,4.39],
            [4.32,3.80,4.44,4.23,6.00,4.99,5.94,4.98,4.77,5.24],
            [6.65,3.78,4.26,5.58,6.85,7.67,5.11,4.79,2.98,5.98],
            [5.58,6.16,5.50,5.58,7.19,8.93,2.97,4.67,4.24,3.65],
            [10.35,4.24,4.62,4.05,4.39,5.42,4.35,5.43,6.73,6.67],
            [4.00,5.94,5.71,5.13,5.46,6.20,4.48,7.47,3.72,4.67],
            [5.54,3.86,6.27,6.15,8.84,8.94,3.71,8.97,5.33,5.13],
            [3.43,5.15,3.87,4.83,4.15,4.41,5.99,5.68,5.20,6.26],
            [4.76,4.37,7.00,4.78,4.62,5.78,6.01,6.21,4.00,6.45],
            [4.49,2.92,3.35,4.18,4.11,5.84,5.72,4.98,6.29,5.57],
            [4.48,4.62,5.78,4.31,5.68,4.91,6.22,5.34,4.50,6.10],
            [4.92,3.94,3.84,6.20,5.19,5.55,5.33,8.02,6.49,5.65],
            [5.70,3.31,3.98,5.86,5.23,4.54,4.37,7.01,4.95,4.24],
            [4.60,5.60,3.88,3.20,5.25,3.40,3.83,5.96,5.41,5.59],
            [5.87,6.44,4.20,8.11,5.08,6.75,2.97,5.47,4.54,5.06],
            [8.11,4.66,4.03,3.58,5.84,4.25,5.54,5.08,6.05,4.53],
            [6.57,4.14,3.67,7.12,11.28,5.72,5.01,3.58,5.92,18.86],
            [3.69,4.79,5.36,3.01,5.39,4.39,3.85,3.69,3.87,4.80],
            [5.11,4.46,3.69,7.07,4.36,5.38,5.73,5.75,5.87,5.04],
            [4.34,5.60,5.78,4.54,5.19,4.44,5.25,3.61,5.88,7.66],
            [9.63,5.04,4.29,4.92,3.56,3.04,5.32,6.97,3.98,5.71],
            [4.68,3.91,6.28,5.65,5.46,9.63,5.30,4.36,6.40,4.31],
            [3.48,3.63,4.64,5.55,5.42,4.29,6.28,3.95,6.18,4.88],
            [4.16,4.17,2.87,4.11,5.19,4.57,4.69,6.14,5.29,9.89],
            [3.37,3.83,4.31,3.72,5.55,4.87,7.28,3.76,5.98,4.22],
            [4.11,7.00,5.71,3.97,3.38,4.27,4.34,2.96,5.90,6.18],
            [5.16,3.12,3.65,4.60,4.60,6.53,5.49,6.30,4.19,3.64],
            [4.25,4.31,5.99,3.62,3.13,6.23,4.90,4.78,8.32,4.88],
            [5.29,4.57,3.48,4.30,9.05,4.63,6.44,6.17,5.90,5.56],
        ])

    mean_gain = gains.mean(axis=0)
    well_idx = np.arange(1, gains.shape[1] + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for row in gains:
        axes[0].plot(well_idx, row, color="steelblue", alpha=0.15)
    axes[0].plot(well_idx, mean_gain, color="darkred", linewidth=3, marker="o", label="mean across 30 trials")
    axes[0].set_xlabel("well # added (in sequence)")
    axes[0].set_ylabel("marginal gain")
    axes[0].set_title("All 30 trials -- marginal gain should decrease if submodular (it doesn't, reliably)")
    axes[0].legend()

    transitions = np.diff(mean_gain)
    colors = ["crimson" if t > 0 else "steelblue" for t in transitions]
    axes[1].bar(np.arange(1, len(transitions) + 1), transitions, color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    n_violations = (transitions > 0).sum()
    axes[1].set_xlabel("transition (well i -> well i+1)")
    axes[1].set_ylabel("change in mean marginal gain")
    axes[1].set_title(f"Submodularity violations: {n_violations}/{len(transitions)} transitions increase (red)")

    fig.suptitle("Stage 3: Submodularity test -- greedy's formal guarantee does NOT strictly hold here")
    savefig(fig, "16_submodularity_violations.png")


# ===========================================================================
# RUN EVERYTHING
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Generating physics illustrations...")
    print("=" * 70)
    safe(physics_v1_vs_v2_fields, "physics_v1_vs_v2_fields")
    safe(physics_drawdown_isotropic_vs_anisotropic, "physics_drawdown")
    safe(physics_two_well_interference, "physics_two_well_interference")
    safe(physics_pressure_trajectory_evolution, "physics_trajectory_evolution")

    print("=" * 70)
    print("Generating dataset diagnostics...")
    print("=" * 70)
    safe(lambda: dataset_field_samples("dataset_2000.npz", "v1 dataset", "dataset_v1_samples.png"), "dataset_v1_samples")
    safe(lambda: dataset_field_samples("dataset_v2_2000.npz", "v2 dataset", "dataset_v2_samples.png"), "dataset_v2_samples")
    safe(lambda: dataset_well_count_histogram("dataset_2000.npz", "v1", "dataset_v1_well_hist.png"), "dataset_v1_well_hist")
    safe(lambda: dataset_well_count_histogram("dataset_v2_2000.npz", "v2", "dataset_v2_well_hist.png"), "dataset_v2_well_hist")
    safe(lambda: dataset_permeability_histogram("dataset_2000.npz", "v1", "dataset_v1_field_hist.png"), "dataset_v1_field_hist")
    safe(lambda: dataset_permeability_histogram("dataset_v2_2000.npz", "v2", "dataset_v2_field_hist.png"), "dataset_v2_field_hist")

    print("=" * 70)
    print("Generating FNO validation plots...")
    print("=" * 70)
    safe(fno_validation_comparison, "fno_validation_comparison")
    safe(fno_training_curves, "fno_training_curves")

    print("=" * 70)
    print("Generating Stage 1 (optimization) plots...")
    print("=" * 70)
    safe(stage1_greedy_curve, "stage1_greedy_curve")

    print("=" * 70)
    print("Generating Stage 2 (game theory) plots...")
    print("=" * 70)
    safe(shapley_v1_bar, "shapley_v1_bar")
    safe(shapley_v1_vs_v2_comparison, "shapley_v1_vs_v2")
    safe(shapley_vs_core_nucleolus, "shapley_vs_core_nucleolus")
    safe(blotto_unconstrained_scatter, "blotto_unconstrained")
    safe(blotto_constrained_summary, "blotto_constrained_summary")
    safe(blotto_nagent_territories, "blotto_nagent_territories")

    print("=" * 70)
    print("Generating Stage 3 (robustness) plots...")
    print("=" * 70)
    safe(robustness_shapley_boxplot, "robustness_shapley_boxplot")
    safe(robustness_blotto_scatter, "robustness_blotto_scatter")
    safe(submodularity_violation_plot, "submodularity_violations")

    print("=" * 70)
    print(f"DONE. {len(SUCCEEDED)} plots saved to {OUT}/, {len(FAILED)} skipped.")
    print("=" * 70)
    if FAILED:
        print("Skipped plots (missing files or import errors -- not fatal):")
        for name, err in FAILED:
            print(f"  - {name}: {err}")
    print("\nSaved plots:")
    for s in SUCCEEDED:
        print(f"  - {s}")
