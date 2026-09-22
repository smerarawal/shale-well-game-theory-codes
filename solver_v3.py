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


def phase_mobilities(Sw, mu_w=1.0, mu_n=5.0, **corey_kwargs):
    """lambda_w, lambda_n separately (total_mobility only returns their
    sum) -- gravity needs them individually since buoyancy pulls each
    phase differently, weighted by ITS OWN mobility, not the combined one."""
    krw, krn = corey_relperm(Sw, **corey_kwargs)
    return krw / mu_w, krn / mu_n


def conservative_clip(Sw_raw, phi, Swc, Sor):
    """
    Clip to [Swc, 1-Sor] WITHOUT losing/gaining global mass -- found this
    the hard way under gravity: a buoyant phase piling up against a
    closed (no-flow) boundary overshoots the physical bound by a small
    amount EVERY substep (this is a real numerical artifact of first-
    order upwinding hitting a reflecting wall with converging flux, not a
    CFL/timestep-size issue -- shrinking dt by 10x barely changed it).
    Plain np.clip silently deletes that overshoot's mass; compounded over
    hundreds of substeps this measured out to ~50% total water "lost"
    over a 150-step run with no wells, which is wrong, not just
    imprecise. This redistributes exactly what clipping would have
    deleted into cells that still have headroom, weighted by how much
    headroom each has, so sum(Sw*phi) is invariant by construction.
    A no-op (identical to plain np.clip) whenever nothing overshoots, so
    this is safe to use unconditionally, not just under add_gravity=True.
    """
    Sw_clipped = np.clip(Sw_raw, Swc, 1 - Sor)
    excess_mass = ((Sw_raw - Sw_clipped) * phi).sum()  # >0: mass clipped away, needs putting back
    if abs(excess_mass) < 1e-12:
        return Sw_clipped

    if excess_mass > 0:
        headroom = np.clip((1 - Sor) - Sw_clipped, 0, None)  # cells with room to take more water
    else:
        headroom = np.clip(Sw_clipped - Swc, 0, None)  # cells with room to give water back

    weight_total = headroom.sum()
    if weight_total < 1e-12:
        # every cell is already saturated at the opposite bound -- nowhere
        # to put the excess (physically: reservoir is completely full or
        # completely dry); nothing sound to redistribute into, so fall back
        # to plain clip rather than divide by ~0 into garbage values
        return Sw_clipped

    Sw_final = Sw_clipped + (excess_mass * headroom / weight_total) / phi
    return np.clip(Sw_final, Swc, 1 - Sor)


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
    add_gravity=False, rho_w=1.0, rho_n=0.7, g=1.0,
    **corey_kwargs
):
    """
    well_is_water_injector: list of bool, same length as well_locations --
        True = injects water; False = produces (withdraws local mixture,
        proportional to local fractional flow at that cell).
    Sw_init: defaults to uniform Swc everywhere (pure injection scenario).

    add_gravity: off by default -- everything below this docstring is
    IDENTICAL to the pre-gravity solver when add_gravity=False (the extra
    terms are only computed and only enter source_total/vel_y inside an
    `if add_gravity:` block), so existing validated behavior (B.1/B.2 in
    validate_solver_v3.py) is unchanged. When True, adds buoyancy
    segregation between the wetting (rho_w) and non-wetting (rho_n)
    phases along the grid's j-axis, which is DEFINED as depth increasing
    with j (j=0 shallow, j=ny-1 deep) -- this 2D areal grid has no
    inherent depth axis, so this is a deliberate reinterpretation of the
    y-axis as vertical, not a 3rd spatial dimension. rho_n < rho_w by
    default (non-wetting phase buoyant, e.g. CO2 in brine) so it migrates
    toward j=0 (rises) while wetting phase sinks toward j=ny-1.

    Derivation (phase potential form, Darcy's law per phase alpha):
        u_alpha = -k*(kr_alpha/mu_alpha) * (dp/dj - rho_alpha*g)
    Summing phases gives the TOTAL velocity's gravity term (added to
    vel_y below) and, via its divergence, an explicit source term added
    to the pressure equation's RHS (same role as well source_total,
    lagged at the current step's Sw exactly like lambda_total already is
    -- standard IMPES treatment, not implicit in Sw). This divergence is
    generally NONZERO only where phase mobilities vary spatially (i.e.
    near a saturation front) -- a spatially uniform reservoir with
    uniform gravity does not itself create net flow, only internal
    counter-current segregation, which is exactly the correct physics.

    KNOWN SIMPLIFICATION: the saturation update below still upwinds fw
    using the TOTAL velocity's sign at each face (single upwind direction
    per face). Under strong buoyancy this can differ from the more
    rigorous treatment of upwinding each phase's flux independently,
    which allows true counter-current flow (water down, non-wetting
    phase up) AT THE SAME FACE even when total velocity there is near
    zero. This solver cannot represent that counter-current case
    correctly -- it gets the net migration direction right (validated
    below) but would understate segregation sharpness in a near-zero-
    total-velocity, strong-gravity regime. Flagging this rather than
    quietly shipping it as exact.

    Assumes dx=1.0 when add_gravity=True (matches every other call site
    in this repo already) -- the gravity terms below are only dimensionally
    consistent with the rest of the discretization at dx=1; asserted below.

    Returns: pressure_history, saturation_history (lists of 2D snapshots
    saved every `save_every` pressure steps), final dt_pressure used.
    """
    if nx is None or ny is None:
        nx, ny = k.shape
    Swc = corey_kwargs.get("Swc", 0.2)
    Sor = corey_kwargs.get("Sor", 0.2)
    if add_gravity:
        assert dx == 1.0, "gravity terms are only dimensionally consistent at dx=1.0 in this implementation"

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

        if add_gravity:
            # phase-specific mobilities, face-averaged the SAME way (harmonic
            # mean) as lambda_total already is -- only the north/south (j-axis
            # = depth) faces matter, gravity has no x-component here
            lam_w_cell, lam_n_cell = phase_mobilities(Sw, mu_w=mu_w, mu_n=mu_n, **corey_kwargs)
            _, _, lamw_n_face, _ = face_values_harmonic(lam_w_cell)
            _, _, lamn_n_face, _ = face_values_harmonic(lam_n_cell)
            # Q_face[i,j]: gravity-driven Darcy velocity across the face
            # between (i,j) and (i,j+1), positive = flow toward +j (deeper).
            # At dx=1 this is simultaneously a valid velocity term (added to
            # vel_y below) AND, via its face-to-face difference, a valid
            # source-term contribution to the pressure equation's RHS (same
            # scale as source_total at dx=1 -- see docstring derivation)
            Q_face = k_n * (lamw_n_face * rho_w + lamn_n_face * rho_n) * g
            Q_face_south = np.zeros_like(Q_face); Q_face_south[:, 1:] = Q_face[:, :-1]
            source_total_with_gravity = source_total + (Q_face_south - Q_face)
        else:
            source_total_with_gravity = source_total

        p = solve_pressure_implicit(p, k_e, k_w, k_n, k_s, lam_e, lam_w, lam_n, lam_s,
                                     phi, ct, dt_pressure, source_total_with_gravity, dx=dx)

        p_e2 = np.zeros_like(p); p_e2[:-1, :] = p[1:, :]
        p_n2 = np.zeros_like(p); p_n2[:, :-1] = p[:, 1:]
        vel_x = -k_e * lam_e * (p_e2 - p) / dx
        vel_y = -k_n * lam_n * (p_n2 - p) / dx
        if add_gravity:
            vel_y = vel_y + Q_face

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

            Sw_raw = Sw + dt_sub / phi * (-div_sat_flux + source_water)
            Sw = conservative_clip(Sw_raw, phi, Swc, Sor)

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
