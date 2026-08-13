"""
Finite-difference solver for 2D pressure diffusion in a heterogeneous reservoir.

    dp/dt = div( k(x,y) * grad(p) ) + sources

Vectorized with NumPy (no Python loops over grid cells). Uses harmonic
averaging of permeability between neighboring cells, which is the standard
approach in reservoir simulation for flux continuity across cell faces.
"""
import numpy as np


def harmonic_mean(a, b, eps=1e-10):
    """Harmonic mean of two permeability arrays (elementwise)."""
    return 2.0 * a * b / (a + b + eps)


def face_permeabilities(k):
    """
    Compute harmonic-averaged permeability at cell faces.
    k: (nx, ny)
    Returns k_east, k_west, k_north, k_south, each (nx, ny), zero-padded
    at domain boundaries (no-flow boundary condition).
    """
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


def max_stable_dt(k, dx, safety=0.4):
    """
    CFL-style stability bound for the explicit scheme:
        dt <= safety * dx^2 / (4 * max(k))
    """
    k_max = np.max(k)
    if k_max <= 0:
        return np.inf
    return safety * dx ** 2 / (4.0 * k_max)


def solve_pressure_diffusion(
    permeability,
    well_locations,
    well_rates,
    dx=1.0,
    dt=None,
    nt=50,
    record_every=1,
):
    """
    Explicit, vectorized finite-difference solve of the pressure diffusion
    equation with no-flow (Neumann) boundaries.

    permeability: (nx, ny) array, k > 0
    well_locations: list of (i, j) grid indices
    well_rates: list of source (+) / sink (-) strengths, same length as
                well_locations
    dx: grid spacing
    dt: timestep. If None, computed automatically from the CFL bound.
    nt: number of timesteps
    record_every: store history every N steps (reduces memory for long runs)

    Returns:
        history: (n_recorded, nx, ny) pressure fields
        dt_used: float
    """
    k = np.asarray(permeability, dtype=np.float64)
    nx, ny = k.shape

    if dt is None:
        dt = max_stable_dt(k, dx)

    k_e, k_w, k_n, k_s = face_permeabilities(k)

    p = np.zeros((nx, ny), dtype=np.float64)
    n_records = nt // record_every
    history = np.zeros((n_records, nx, ny), dtype=np.float64)

    # source term as a dense grid (added each step)
    source = np.zeros((nx, ny), dtype=np.float64)
    wi = np.array([w[0] for w in well_locations], dtype=int)
    wj = np.array([w[1] for w in well_locations], dtype=int)
    rates = np.asarray(well_rates, dtype=np.float64)

    rec_idx = 0
    for t in range(nt):
        p_e = np.zeros_like(p)
        p_w = np.zeros_like(p)
        p_n = np.zeros_like(p)
        p_s = np.zeros_like(p)
        p_e[:-1, :] = p[1:, :]
        p_w[1:, :] = p[:-1, :]
        p_n[:, :-1] = p[:, 1:]
        p_s[:, 1:] = p[:, :-1]

        # flux-conservative divergence: div(k grad p)
        div_kgradp = (
            k_e * (p_e - p) - k_w * (p - p_w)
            + k_n * (p_n - p) - k_s * (p - p_s)
        ) / dx ** 2

        source[:] = 0.0
        np.add.at(source, (wi, wj), rates)

        p = p + dt * (div_kgradp + source)

        if (t + 1) % record_every == 0:
            history[rec_idx] = p
            rec_idx += 1

    return history[:rec_idx], dt


if __name__ == "__main__":
    # quick sanity check: single injector in homogeneous field should
    # produce a radially symmetric pressure bump
    nx, ny = 32, 32
    k = np.ones((nx, ny)) * 0.1
    history, dt_used = solve_pressure_diffusion(
        k, well_locations=[(16, 16)], well_rates=[1.0], dx=1.0, nt=200
    )
    print("dt used:", dt_used)
    print("final pressure at well:", history[-1, 16, 16])
    print("final pressure at edge:", history[-1, 0, 0])
    assert history[-1, 16, 16] > history[-1, 0, 0], "pressure should decay from well"
    print("OK: pressure decays radially from injector")
