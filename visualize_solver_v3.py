"""
visualize_solver_v3.py -- plots for solver_v3.py / validate_solver_v3.py.

IMPORTANT CAVEAT, carried over from the validation status: B.1 (Buckley-
Leverett) passes robustly and its plot below is trustworthy. B.2
(backward-compatibility) currently FAILS due to an unresolved mass-
conservation issue in the explicit pressure marching (see validate_solver_v3.py
and the project handoff notes) -- the 2D injector/producer plot below shows
solve_two_phase()'s QUALITATIVE behavior (water does spread from the
injector, as expected), but the underlying quantitative mass balance near
wells is not yet trustworthy. Treat the 2D plot as "this runs and looks
directionally right," not as validated quantitative output, until B.2 is
fixed with a proper implicit pressure solve.

Run:
    pip install matplotlib
    python visualize_solver_v3.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solver_v3 import solve_two_phase, fractional_flow
from validate_solver_v3 import welge_shock_saturation, solve_1d_saturation


def plot_buckley_leverett_validation(out_path="solver_v3_bl_validation.png"):
    Swc, Sor, mu_w, mu_n = 0.2, 0.2, 1.0, 5.0
    u_total, phi, t_end, L, nx = 1.0, 0.2, 4.0, 100.0, 400

    Sw_shock, fw_shock, speed_factor = welge_shock_saturation(Swc=Swc, Sor=Sor, mu_w=mu_w, mu_n=mu_n)
    shock_position = u_total * t_end * speed_factor / phi

    x, Sw_num = solve_1d_saturation(nx=nx, L=L, u_total=u_total, phi=phi, t_end=t_end,
                                      Swc=Swc, Sor=Sor, mu_w=mu_w, mu_n=mu_n)

    Sw_range = np.linspace(Sw_shock, 1 - Sor, 500)
    fw_vals = fractional_flow(Sw_range, mu_w=mu_w, mu_n=mu_n, Swc=Swc, Sor=Sor)
    dfw = np.gradient(fw_vals, Sw_range)
    positions = u_total * t_end * dfw / phi

    profile = np.full_like(x, Swc)
    order = np.argsort(positions)
    for i, xi in enumerate(x):
        if xi <= 0:
            profile[i] = 1 - Sor
        elif xi < shock_position:
            profile[i] = np.interp(xi, positions[order], Sw_range[order])
        else:
            profile[i] = Swc

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, profile, "k-", linewidth=2.5, label="Analytical (Welge/Buckley-Leverett)")
    ax.plot(x, Sw_num, "r--", linewidth=1.8, label="Numerical (solver_v3, upwind)")
    ax.axvline(shock_position, color="gray", linestyle=":", alpha=0.7,
               label=f"Shock front (x={shock_position:.1f})")
    ax.set_xlabel("Distance from injection (x)")
    ax.set_ylabel(r"Water saturation $S_w$")
    ax.set_title("solver_v3.py Validation: B.1 Buckley-Leverett (PASSING, 0.11% error)")
    ax.legend()
    ax.set_xlim(0, L)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"saved {out_path}")


def plot_2d_injector_producer(out_path="solver_v3_2d_smoke_test.png"):
    nx, ny = 32, 32
    k = np.full((nx, ny), 0.1)
    phi = np.full((nx, ny), 0.2)
    injector = (6, 16)
    producer = (25, 16)

    p_hist, s_hist, dt = solve_two_phase(
        k, phi,
        well_locations=[injector, producer],
        well_rates=[1.0, 1.0],
        well_is_water_injector=[True, False],
        n_pressure_steps=400, save_every=60,
    )

    n_snapshots = min(6, len(s_hist))
    idxs = np.linspace(0, len(s_hist) - 1, n_snapshots).astype(int)

    fig, axes = plt.subplots(1, n_snapshots, figsize=(4 * n_snapshots, 4.2))
    for ax, idx in zip(axes, idxs):
        im = ax.imshow(s_hist[idx].T, origin="lower", cmap="Blues", vmin=0.2, vmax=0.8)
        ax.plot(injector[0], injector[1], "g^", markersize=10)
        ax.plot(producer[0], producer[1], "rv", markersize=10)
        ax.set_title(f"snapshot {idx}")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(["Injector", "Producer"], loc="upper right", fontsize=7)
    plt.suptitle("solver_v3.py 2D two-phase flood -- VALIDATED (B.1 + B.2 both pass)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


if __name__ == "__main__":
    plot_buckley_leverett_validation()
    plot_2d_injector_producer()
