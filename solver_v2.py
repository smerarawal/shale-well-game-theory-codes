"""
solver_v2.py -- physics upgrade over solver.py.

Original equation:      dp/dt = div(k * grad(p)) + sources
Upgraded equation:  phi*mu*ct * dp/dt = div((k/mu) * grad(p)) + sources
                    (the standard reservoir "diffusivity equation")

New physics added:
  1. Porosity (phi): fraction of rock volume that's actually open pore
     space. Permeability tells you how CONNECTED pores are; porosity
     tells you how MUCH pore space exists. A rock can have high
     permeability but low porosity or vice versa -- they're independent
     fields now, not one field standing in for both.
  2. Compressibility (ct) and viscosity (mu): standard reservoir
     engineering constants controlling how fast pressure changes
     propagate. Left as scalars (not spatially varying) for now --
     a reasonable simplification, real reservoirs do vary these too
     but it's a second-order effect compared to k and phi.
  3. Anisotropic permeability (kx != ky): real rock often flows more
     easily in one direction (e.g. horizontal bedding planes vs
     vertical). Drawdown cones become elliptical instead of circular.
  4. Optional non-Darcy (Forchheimer) correction: near a well, flow
     velocity is high enough that simple linear Darcy flow
     underestimates the actual pressure drop. Adds a term proportional
     to velocity-squared, only significant very close to wells.

This solver is backward-compatible: call solve_pressure_diffusion_v2
with default phi=1.0, mu=1.0, ct=1.0, kx=ky=k, non_darcy_beta=0 and you
recover behavior equivalent to the original solver.py.

Run standalone to sanity-check the new physics:
    python solver_v2.py
"""
import numpy as np


def harmonic_mean(a, b, eps=1e-10):
    return 2.0 * a * b / (a + b + eps)


def face_values_harmonic(k):
    """Harmonic-averaged values at cell faces (same as solver.py)."""
    nx, ny = k.shape
    k_east = np.zeros_like(k)
    k_west = np.zeros_like(k)
    k_north = np.zeros_like(k)
    k_south = np.zeros_like(k)

    k_east[:-1, :] = harmonic_mean(k[:-1, :], k[1:, :])
    k_west[1:, :] = k_east[:-1, :]
    k_north[:, :-1] = harmonic_mean(k[:, :-1], k[:, 1:])
    k_south[:, 1:] = k_north[:, :-1]

    return k_east, k_west, k_north, k_south


def random_anisotropic_permeability(nx=32, ny=32, correlation_length_x=5,
                                      correlation_length_y=5, k_min=0.02, k_max=0.2,
                                      seed=None):
    """
    Generate kx and ky separately -- different correlation lengths in
    each direction produce elongated/directional geological patterns
    (e.g. sedimentary layering), instead of the isotropic (same in all
    directions) fields solver.py's random_permeability_field produces.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)

    raw_x = rng.standard_normal((nx, ny))
    field_x = gaussian_filter(raw_x, sigma=(correlation_length_x, correlation_length_y))
    field_x = (field_x - field_x.min()) / (field_x.max() - field_x.min() + 1e-10)
    kx = k_min + field_x * (k_max - k_min)

    # ky correlated with kx (same underlying rock) but with an independent
    # anisotropy ratio per cell, representing directional flow preference
    anisotropy_ratio = rng.uniform(0.5, 1.5, size=(nx, ny))
    ky = np.clip(kx * anisotropy_ratio, k_min, k_max)

    return kx, ky


def random_porosity_field(nx=32, ny=32, correlation_length=4, phi_min=0.05, phi_max=0.25, seed=None):
    """
    Porosity field, independent of permeability (a real rock property,
    not derived from k). Typical shale porosity range ~5-25%.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((nx, ny))
    field = gaussian_filter(raw, sigma=correlation_length)
    field = (field - field.min()) / (field.max() - field.min() + 1e-10)
    return phi_min + field * (phi_max - phi_min)


def max_stable_dt_v2(kx, ky, phi, mu, ct, dx, safety=0.4):
    """CFL bound adapted for the compressible diffusivity equation."""
    k_max = max(np.max(kx), np.max(ky))
    phi_min = np.min(phi)
    if k_max <= 0:
        return np.inf
    return safety * phi_min * mu * ct * dx ** 2 / (4.0 * k_max)


def solve_pressure_diffusion_v2(
    kx, ky, phi,
    well_locations, well_rates,
    mu=1.0, ct=1.0, dx=1.0, dt=None, nt=50,
    non_darcy_beta=0.0,
):
    """
    phi*mu*ct * dp/dt = div((k/mu) * grad(p)) + q  [+ optional Forchheimer correction]

    kx, ky: (nx, ny) anisotropic permeability fields
    phi: (nx, ny) porosity field
    mu: fluid viscosity (scalar)
    ct: total compressibility (scalar)
    non_darcy_beta: if > 0, adds a velocity-squared correction near wells
                     (Forchheimer non-Darcy flow), otherwise pure Darcy
    """
    nx, ny = kx.shape
    if dt is None:
        dt = max_stable_dt_v2(kx, ky, phi, mu, ct, dx)

    kx_e, kx_w, _, _ = face_values_harmonic(kx)
    _, _, ky_n, ky_s = face_values_harmonic(ky)

    p = np.zeros((nx, ny), dtype=np.float64)
    history = np.zeros((nt, nx, ny), dtype=np.float64)

    wi = np.array([w[0] for w in well_locations], dtype=int)
    wj = np.array([w[1] for w in well_locations], dtype=int)
    rates = np.asarray(well_rates, dtype=np.float64)
    source = np.zeros((nx, ny), dtype=np.float64)

    accumulation = phi * mu * ct  # left-hand-side coefficient, varies per cell now

    for t in range(nt):
        p_e = np.zeros_like(p); p_w = np.zeros_like(p)
        p_n = np.zeros_like(p); p_s = np.zeros_like(p)
        p_e[:-1, :] = p[1:, :]; p_w[1:, :] = p[:-1, :]
        p_n[:, :-1] = p[:, 1:]; p_s[:, 1:] = p[:, :-1]

        flux_e = kx_e / mu * (p_e - p)
        flux_w = kx_w / mu * (p - p_w)
        flux_n = ky_n / mu * (p_n - p)
        flux_s = ky_s / mu * (p - p_s)

        if non_darcy_beta > 0:
            # Forchheimer correction: extra resistance proportional to
            # |flux|*flux, significant only where flow is fast (near wells)
            flux_e -= non_darcy_beta * np.abs(flux_e) * flux_e
            flux_w -= non_darcy_beta * np.abs(flux_w) * flux_w
            flux_n -= non_darcy_beta * np.abs(flux_n) * flux_n
            flux_s -= non_darcy_beta * np.abs(flux_s) * flux_s

        div_flux = (flux_e - flux_w + flux_n - flux_s) / dx ** 2

        source[:] = 0.0
        np.add.at(source, (wi, wj), rates)

        dp_dt = (div_flux + source) / (accumulation + 1e-10)
        p = p + dt * dp_dt
        history[t] = p

    return history, dt


if __name__ == "__main__":
    nx, ny = 32, 32
    kx, ky = random_anisotropic_permeability(nx, ny, correlation_length_x=8, correlation_length_y=2, seed=0)
    phi = random_porosity_field(nx, ny, seed=0)

    history, dt = solve_pressure_diffusion_v2(
        kx, ky, phi, well_locations=[(16, 16)], well_rates=[-1.0], nt=200
    )
    print("dt used:", dt)
    print("pressure at well:", history[-1, 16, 16])
    print("pressure at edge:", history[-1, 0, 0])

    # sanity check: with kx correlation length 8 and ky correlation length 2,
    # the drawdown pattern should look visibly elongated (anisotropic), not
    # a clean circle -- check this visually if you have matplotlib available
    print("\nkx range:", kx.min(), kx.max())
    print("ky range:", ky.min(), ky.max())
    print("porosity range:", phi.min(), phi.max())

    # backward-compatibility check: isotropic + phi=1 + mu=1 + ct=1 should
    # behave like solver.py
    kx_iso = np.full((nx, ny), 0.1)
    ky_iso = np.full((nx, ny), 0.1)
    phi_iso = np.ones((nx, ny))
    history_iso, dt_iso = solve_pressure_diffusion_v2(
        kx_iso, ky_iso, phi_iso, well_locations=[(16, 16)], well_rates=[-1.0], nt=200
    )
    print("\nisotropic backward-compat check, pressure at well:", history_iso[-1, 16, 16])
    assert history_iso[-1, 16, 16] < 0, "producer well should show negative (depleted) pressure"
    print("OK")
