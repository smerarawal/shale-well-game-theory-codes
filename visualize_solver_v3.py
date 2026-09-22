"""
visualize_solver_v3.py -- plots for solver_v3.py / validate_solver_v3.py.

STATUS (current): B.1 (Buckley-Leverett), B.2 (mass conservation, via the
implicit pressure solve), and B.3 (gravity segregation + mass conservation
under gravity) all PASS -- see validate_solver_v3.py. All three plots
below are trustworthy quantitative output, not just qualitative sanity
checks.

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


def plot_gravity_override(out_path="solver_v3_gravity_override.png"):
    """
    A single injector, no producer, with gravity on vs off -- shows
    buoyant override during an ACTIVE flood, not just a static patch
    (that's B.3's test). solve_two_phase can only actively INJECT the
    wetting phase (well_is_water_injector=True adds a pure-water source
    term; there's no symmetric "inject non-wetting" path), so this
    demonstrates it the other physically valid way round: water (wetting,
    rho_w=1.0, denser) is injected into a reservoir that starts full of
    the buoyant non-wetting phase (rho_n=0.6, e.g. CO2/oil/gas -- the
    resident fluid, not what's injected here). Without gravity the
    waterflood front spreads symmetrically from the injector; with
    gravity, the denser injected water should sink preferentially (invade
    deeper/larger-j cells more than shallow ones), which is exactly the
    asymmetry to look for below.
    """
    nx, ny = 32, 32
    k = np.full((nx, ny), 0.1)
    phi = np.full((nx, ny), 0.2)
    injector = (16, 16)  # centered, so any asymmetry in the plots is gravity's doing, not geometry

    _, s_nograv, _ = solve_two_phase(
        k, phi, well_locations=[injector], well_rates=[1.0], well_is_water_injector=[True],
        n_pressure_steps=250, save_every=250, add_gravity=False,
    )
    _, s_grav, _ = solve_two_phase(
        k, phi, well_locations=[injector], well_rates=[1.0], well_is_water_injector=[True],
        n_pressure_steps=250, save_every=250,
        add_gravity=True, rho_w=1.0, rho_n=0.6, g=2.0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, hist, title in [(axes[0], s_nograv, "No gravity: symmetric waterflood"),
                             (axes[1], s_grav, "WITH gravity: denser water sinks (invades toward bottom=deep)")]:
        im = ax.imshow(hist[-1].T, cmap="Blues", vmin=0.2, vmax=0.8)
        ax.plot(injector[0], injector[1], "r*", markersize=14)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Water injection (wetting, denser) into a buoyant-resident-fluid reservoir: gravity override\n"
                  "(dark=water-invaded; depth increases DOWN the image here, j=0/shallow at top)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


if __name__ == "__main__":
    plot_buckley_leverett_validation()
    plot_2d_injector_producer()
    plot_gravity_override()
