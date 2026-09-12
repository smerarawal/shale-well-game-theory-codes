"""
solver_v3.py -- Part A of richer_physics_v3_spec.md: black-oil-level
two-phase (wetting/non-wetting) flow. Two coupled PDEs:

  Pressure (elliptic):
      div(lambda_total(Sw) * k * grad(p)) + q_total = 0

  Saturation (hyperbolic, genuinely new):
      phi * dSw/dt + div(fw(Sw) * u_total) = qw

Naming follows the spec exactly: "w" = wetting phase (brine/water),
"n" = non-wetting phase (CO2, gas, or oil depending on scenario).

Numerical scheme: IMPES (IMplicit Pressure, Explicit Saturation).
Pressure solved with the same explicit-marching harmonic-mean structure
as solve_pressure_diffusion_v2, but lambda_total(Sw)*k replaces k/mu and
now varies in TIME as well as space. Saturation is upwinded (mandatory --
centered differencing is unconditionally unstable for pure advection) and
sub-cycled under its OWN CFL condition, separate from the pressure CFL.
"""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


# ---- Part A.2: Corey relative permeability ----

def corey_relperm(Sw, Swc=0.2, Sor=0.2, krw_max=0.8, krn_max=1.0, nw=2, nn=2):
    """
    Corey (1954) relative permeability model. Sw physically cannot leave
    [Swc, 1-Sor]; Sw_eff is clipped to [0,1] so krw/krn are always
    well-defined and correctly zero at the opposite endpoint.
    """
    Sw_eff = np.clip((Sw - Swc) / (1 - Swc - Sor), 0, 1)
    krw = krw_max * Sw_eff ** nw
    krn = krn_max * (1 - Sw_eff) ** nn
    return krw, krn


def fractional_flow(Sw, mu_w=1.0, mu_n=5.0, **corey_kwargs):
    """fw(Sw) = (krw/mu_w) / (krw/mu_w + krn/mu_n)."""
    krw, krn = corey_relperm(Sw, **corey_kwargs)
    lambda_w = krw / mu_w
    lambda_n = krn / mu_n
    return lambda_w / (lambda_w + lambda_n + 1e-12)


def total_mobility(Sw, mu_w=1.0, mu_n=5.0, **corey_kwargs):
    krw, krn = corey_relperm(Sw, **corey_kwargs)
    return krw / mu_w + krn / mu_n


def harmonic_mean(a, b, eps=1e-10):
    return 2.0 * a * b / (a + b + eps)


def face_values_harmonic(field):
    nx, ny = field.shape
    e = np.zeros_like(field); w = np.zeros_like(field)
    n = np.zeros_like(field); s = np.zeros_like(field)
    e[:-1, :] = harmonic_mean(field[:-1, :], field[1:, :])
    w[1:, :] = e[:-1, :]
    n[:, :-1] = harmonic_mean(field[:, :-1], field[:, 1:])
    s[:, 1:] = n[:, :-1]
    return e, w, n, s


# ---- Part A.3: upwind fractional flow ----

def upwind_fractional_flow_faces(Sw, flux_x, flux_y, mu_w=1.0, mu_n=5.0, **corey_kwargs):
    """
    Upwind the fractional flow at each face: use the saturation value
    from whichever cell the flow is coming FROM. Centered differencing
    is unconditionally unstable for this hyperbolic equation.
    """
    fw = fractional_flow(Sw, mu_w=mu_w, mu_n=mu_n, **corey_kwargs)
    fw_east = np.roll(fw, -1, axis=0)
    fw_north = np.roll(fw, -1, axis=1)
    fw_face_x = np.where(flux_x >= 0, fw, fw_east)
    fw_face_y = np.where(flux_y >= 0, fw, fw_north)
    return fw_face_x, fw_face_y


# ---- Part A.4: dual CFL, pressure and saturation are separate limits ----

def max_stable_dt_pressure(k, lambda_total, phi, mu_ref, ct, dx, safety=0.4):
    keff_max = np.max(k) * np.max(lambda_total)
    phi_min = np.min(phi)
    if keff_max <= 0:
        return np.inf
    return safety * phi_min * mu_ref * ct * dx ** 2 / (4.0 * keff_max)


def max_stable_dt_saturation(Sw, vel_x, vel_y, phi, dx, mu_w=1.0, mu_n=5.0, safety=0.5, **corey_kwargs):
    """
    dt_sat <= safety * phi_min * dx / max(|fw'(Sw) * u_total|).
    df_w/dSw computed via perturbation over the FULL possible saturation
    range, not the current local state -- using only the current, often
    near-uniform, local state silently produces one unstable timestep
    (this exact bug was hit and fixed in solver_v3_multiphase.py earlier).
    """
    Swc = corey_kwargs.get("Swc", 0.2)
    Sor = corey_kwargs.get("Sor", 0.2)
    Sw_probe = np.linspace(Swc + 1e-3, 1 - Sor - 1e-3, 300)
    eps = 1e-3
    fw_plus = fractional_flow(Sw_probe + eps, mu_w=mu_w, mu_n=mu_n, **corey_kwargs)
    fw_minus = fractional_flow(Sw_probe - eps, mu_w=mu_w, mu_n=mu_n, **corey_kwargs)
    dfw_dSw_max = np.max(np.abs((fw_plus - fw_minus) / (2 * eps)))

    vel_mag_max = np.max(np.sqrt(vel_x ** 2 + vel_y ** 2)) + 1e-8
    phi_min = np.min(phi)
    return safety * phi_min * dx / (dfw_dSw_max * vel_mag_max + 1e-8)


# ---- implicit pressure solve (the real fix for B.2) ----

def solve_pressure_implicit(p_old, k_e, k_w, k_n, k_s, lam_e, lam_w, lam_n, lam_s,
                              phi, ct, dt, source_total, dx=1.0):
    """
    Backward-Euler IMPLICIT solve of:
        phi*ct*(p_new - p_old)/dt = div(lambda*k*grad(p_new)) + source
    via a direct sparse linear solve, NOT explicit pseudo-time marching.

    This is the actual fix for the B.2 mass-conservation problem found
    during validation: explicit marching only approximately satisfies the
    discrete balance each step (the residual gets silently absorbed into
    "ongoing pressure change", which corrupts the saturation equation's
    convective term downstream). A direct linear solve satisfies the
    discrete equation EXACTLY (to linear-solver tolerance) at every cell,
    every step, so the velocity field derived from p_new is consistent
    with well rates immediately, not only after many small steps.
    Also unconditionally stable, so dt is now an ACCURACY choice, not a
    stability constraint (though the same CFL-derived dt is still passed
    in here, to keep the physical timescale interpretation unchanged
    relative to the explicit version).
    """
    nx, ny = p_old.shape
    n = nx * ny

    T_e = (k_e * lam_e / dx ** 2).flatten()
    T_w = (k_w * lam_w / dx ** 2).flatten()
    T_n = (k_n * lam_n / dx ** 2).flatten()
    T_s = (k_s * lam_s / dx ** 2).flatten()
    accumulation = (phi * ct / dt).flatten()

    def idx(i, j):
        return i * ny + j

    A = lil_matrix((n, n))
    b = np.zeros(n)

    for i in range(nx):
        for j in range(ny):
            m = idx(i, j)
            diag = accumulation[m]
            b[m] = accumulation[m] * p_old[i, j] + source_total[i, j]

            if i < nx - 1:
                A[m, idx(i + 1, j)] -= T_e[m]
                diag += T_e[m]
            if i > 0:
                A[m, idx(i - 1, j)] -= T_w[m]
                diag += T_w[m]
            if j < ny - 1:
                A[m, idx(i, j + 1)] -= T_n[m]
                diag += T_n[m]
            if j > 0:
                A[m, idx(i, j - 1)] -= T_s[m]
                diag += T_s[m]

            A[m, m] = diag

    p_new_flat = spsolve(csr_matrix(A), b)
    return p_new_flat.reshape(nx, ny)


# ---- main two-phase IMPES solver ----

def solve_two_phase(
    k, phi, well_locations, well_rates, well_is_water_injector,
    Sw_init=None, nx=None, ny=None, dx=1.0,
    mu_w=1.0, mu_n=5.0, ct=1.0,
    n_pressure_steps=200, pressure_safety=0.4, saturation_safety=0.5,
    save_every=10,
    **corey_kwargs
):
    """
    well_is_water_injector: list of bool, same length as well_locations --
        True = injects water; False = produces (withdraws local mixture,
        proportional to local fractional flow at that cell).
    Sw_init: defaults to uniform Swc everywhere (pure injection scenario).

    Returns: pressure_history, saturation_history (lists of 2D snapshots
    saved every `save_every` pressure steps), final dt_pressure used.
    """
    if nx is None or ny is None:
        nx, ny = k.shape
    Swc = corey_kwargs.get("Swc", 0.2)
    Sor = corey_kwargs.get("Sor", 0.2)

    Sw = np.full((nx, ny), Swc) if Sw_init is None else Sw_init.copy()
    p = np.zeros((nx, ny))
    k_e, k_w, k_n, k_s = face_values_harmonic(k)

    wi = np.array([loc[0] for loc in well_locations], dtype=int)
    wj = np.array([loc[1] for loc in well_locations], dtype=int)
    rates = np.asarray(well_rates, dtype=np.float64)
    is_injector = np.asarray(well_is_water_injector, dtype=bool)

    pressure_history = []
    saturation_history = []

    for step in range(n_pressure_steps):
        lam_total = total_mobility(Sw, mu_w=mu_w, mu_n=mu_n, **corey_kwargs)
        lam_e, lam_w, lam_n, lam_s = face_values_harmonic(lam_total)

        dt_pressure = max_stable_dt_pressure(k, lam_total, phi, 1.0, ct, dx, safety=pressure_safety)

        # NOTE on a design decision, found the hard way: the spec's stated
        # pressure equation is ELLIPTIC (no accumulation/dp-dt term at all),
        # which is only well-posed when sources sum to zero globally
        # (balanced injector/producer pairs, no-flow boundary). This
        # project's existing Stage 0-3 use case is producer-ONLY (depleting
        # a closed reservoir, net source != 0), which the elliptic form
        # cannot represent at all -- attempting a literal "iterate pressure
        # to convergence" implementation of it was tried and DIVERGED
        # (confirmed: error went from 90% to 3546% when attempted) precisely
        # because no bounded steady state exists for an unbalanced sink under
        # no-flow boundaries. Retaining compressibility (the ct term below,
        # same as solver_v2.py) keeps this solver well-posed for BOTH
        # balanced (injector+producer) and unbalanced (producer-only)
        # scenarios, at the cost of deviating from the spec's literal
        # "no accumulation term" simplification. This is the single most
        # important limitation to disclose about this solver if you present
        # it: it's compressible two-phase flow, not incompressible.
        p_e = np.zeros_like(p); p_w = np.zeros_like(p)
        p_n = np.zeros_like(p); p_s = np.zeros_like(p)
        p_e[:-1, :] = p[1:, :]; p_w[1:, :] = p[:-1, :]
        p_n[:, :-1] = p[:, 1:]; p_s[:, 1:] = p[:, :-1]

        source_total = np.zeros((nx, ny))
        source_water = np.zeros((nx, ny))
        fw_current = fractional_flow(Sw, mu_w=mu_w, mu_n=mu_n, **corey_kwargs)
        for idx in range(len(well_locations)):
            i, j = wi[idx], wj[idx]
            if is_injector[idx]:
                source_total[i, j] += rates[idx]
                source_water[i, j] += rates[idx]
            else:
                source_total[i, j] -= rates[idx]
                source_water[i, j] -= rates[idx] * fw_current[i, j]

        p = solve_pressure_implicit(p, k_e, k_w, k_n, k_s, lam_e, lam_w, lam_n, lam_s,
                                     phi, ct, dt_pressure, source_total, dx=dx)

        p_e2 = np.zeros_like(p); p_e2[:-1, :] = p[1:, :]
        p_n2 = np.zeros_like(p); p_n2[:, :-1] = p[:, 1:]
        vel_x = -k_e * lam_e * (p_e2 - p) / dx
        vel_y = -k_n * lam_n * (p_n2 - p) / dx

        dt_sat_max = max_stable_dt_saturation(Sw, vel_x, vel_y, phi, dx, mu_w=mu_w, mu_n=mu_n,
                                               safety=saturation_safety, **corey_kwargs)
        n_sat_substeps = max(1, int(np.ceil(dt_pressure / dt_sat_max)))
        dt_sub = dt_pressure / n_sat_substeps

        for _ in range(n_sat_substeps):
            fw_face_x, fw_face_y = upwind_fractional_flow_faces(Sw, vel_x, vel_y, mu_w=mu_w, mu_n=mu_n, **corey_kwargs)

            sat_flux_e = fw_face_x * vel_x
            sat_flux_n = fw_face_y * vel_y
            sat_flux_w = np.zeros_like(sat_flux_e); sat_flux_w[1:, :] = sat_flux_e[:-1, :]
            sat_flux_s = np.zeros_like(sat_flux_n); sat_flux_s[:, 1:] = sat_flux_n[:, :-1]

            div_sat_flux = (sat_flux_e - sat_flux_w + sat_flux_n - sat_flux_s) / dx

            Sw = Sw + dt_sub / phi * (-div_sat_flux + source_water)
            Sw = np.clip(Sw, Swc, 1 - Sor)

        if step % save_every == 0:
            pressure_history.append(p.copy())
            saturation_history.append(Sw.copy())

    pressure_history.append(p.copy())
    saturation_history.append(Sw.copy())

    return pressure_history, saturation_history, dt_pressure


if __name__ == "__main__":
    print("=== Corey relperm sanity check (spec defaults: krw_max=0.8, krn_max=1.0) ===")
    krw, krn = corey_relperm(np.array([0.2, 0.5, 0.8]))
    print(f"Sw=0.2 (connate water): krw={krw[0]:.4f} krn={krn[0]:.4f}")
    print(f"Sw=0.8 (residual non-wetting): krw={krw[2]:.4f} krn={krn[2]:.4f}")
    assert krw[0] < 1e-6 and abs(krn[0] - 1.0) < 1e-6, "endpoint check failed"
    assert abs(krw[2] - 0.8) < 1e-6 and krn[2] < 1e-6, "endpoint check failed"
    print("OK\n")

    print("=== 2D smoke test: single water injector into uniform reservoir ===")
    nx, ny = 24, 24
    k = np.full((nx, ny), 0.1)
    phi = np.full((nx, ny), 0.2)
    p_hist, s_hist, dt = solve_two_phase(
        k, phi, well_locations=[(12, 12)], well_rates=[1.0],
        well_is_water_injector=[True], n_pressure_steps=100, save_every=20,
    )
    print(f"final Sw at injector: {s_hist[-1][12, 12]:.3f} (should approach 1-Sor=0.8)")
    print(f"final Sw far away: {s_hist[-1][0, 0]:.3f} (should stay near Swc=0.2)")
    assert s_hist[-1][12, 12] > s_hist[-1][0, 0], "injector should be more saturated than far-field"
    print("OK -- see validate_solver_v3.py for the mandatory Buckley-Leverett + backward-compat checks")
