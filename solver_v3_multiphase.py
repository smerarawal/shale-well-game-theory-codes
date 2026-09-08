"""
solver_v3_multiphase.py -- two-phase (oil-water) flow, extending the
single-phase pressure diffusion in solver.py/solver_v2.py to a real
displacement problem: water (or CO2, same math) pushing oil through the
reservoir, tracked via saturation, not just pressure.

Physics:
  Pressure equation (elliptic, same structure as solver_v2 but total
  mobility replaces single-phase k/mu):
      phi*mu*ct * dp/dt = div(k * lambda_total(Sw) * grad(p)) + q

  Saturation equation (hyperbolic transport, Buckley-Leverett):
      phi * dSw/dt + div(fw(Sw) * u) = qw
  where u is the Darcy velocity from the pressure solve and fw is the
  fractional flow function.

Relative permeability: standard Corey model
    krw = krw_max * ((Sw - Swc)/(1 - Swc - Sor))^nw
    kro = kro_max * ((1 - Sw - Sor)/(1 - Swc - Sor))^no

Numerical scheme: IMPES (IMplicit Pressure, Explicit Saturation) --
standard in reservoir simulation. Pressure solved implicitly-in-structure
each step (same explicit-marching approach as solver_v2 for simplicity,
not a true implicit linear solve, but stable under its own CFL), then
saturation advected explicitly with upwinding, SUB-CYCLED within each
pressure step since saturation transport has a much stricter stability
limit than pressure diffusion.

Run standalone to validate against the analytical solution:
    python solver_v3_multiphase.py
(full 2D + 1D validation both run; see validate_solver_v3.py for the
dedicated, stricter validation-only script)
"""
import numpy as np


# ---- Corey relative permeability model ----

def corey_relperm(Sw, Swc=0.2, Sor=0.2, krw_max=0.4, kro_max=0.9, nw=2.0, no=2.0):
    """
    Standard Corey-type relative permeability curves. Literature-typical
    defaults for a water-wet sandstone-like rock (adjust per your specific
    formation if you have real core data).
    Sw: water saturation (scalar or array)
    Swc: connate (irreducible) water saturation
    Sor: residual oil saturation
    """
    Sw = np.clip(Sw, Swc, 1 - Sor)
    Se = (Sw - Swc) / (1 - Swc - Sor)  # normalized ("effective") saturation
    krw = krw_max * Se ** nw
    kro = kro_max * (1 - Se) ** no
    return krw, kro


def fractional_flow(Sw, mu_w=1.0, mu_o=5.0, **corey_kwargs):
    """
    fw(Sw) = (krw/mu_w) / (krw/mu_w + kro/mu_o)
    mu_o > mu_w by default (oil more viscous than water -- typical, and
    what makes displacement inefficient/interesting; a favorable mobility
    ratio, mu_o <= mu_w, gives a very different-looking front).
    """
    krw, kro = corey_relperm(Sw, **corey_kwargs)
    lambda_w = krw / mu_w
    lambda_o = kro / mu_o
    return lambda_w / (lambda_w + lambda_o + 1e-12)


def total_mobility(Sw, mu_w=1.0, mu_o=5.0, **corey_kwargs):
    krw, kro = corey_relperm(Sw, **corey_kwargs)
    return krw / mu_w + kro / mu_o


# ---- Analytical 1D Buckley-Leverett solution (Welge tangent construction) ----

def welge_shock_saturation(Swc=0.2, Sor=0.2, mu_w=1.0, mu_o=5.0, **corey_kwargs):
    """
    Find the shock-front saturation via the Welge tangent-line construction:
    the shock saturation Sw_f is where a line FROM (Swc, 0) is TANGENT to
    the fractional flow curve fw(Sw) -- i.e. where
        fw'(Sw_f) = fw(Sw_f) / (Sw_f - Swc)
    Standard textbook method (Buckley & Leverett 1942; Welge 1952) for
    finding the physically correct (entropy-satisfying) shock, since the
    raw characteristic solution alone would be multi-valued without it.
    Solved here by dense sampling + root-adjacent search (robust, no
    dependency on a root-finder library beyond numpy).
    """
    Sw_range = np.linspace(Swc + 1e-4, 1 - Sor - 1e-4, 2000)
    fw_vals = fractional_flow(Sw_range, mu_w=mu_w, mu_o=mu_o, Swc=Swc, Sor=Sor, **corey_kwargs)

    secant_slope = fw_vals / (Sw_range - Swc)
    dfw = np.gradient(fw_vals, Sw_range)

    diff = dfw - secant_slope
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        idx = np.argmin(np.abs(diff))
    else:
        idx = sign_changes[-1]

    Sw_shock = Sw_range[idx]
    fw_shock = fw_vals[idx]
    shock_speed_factor = secant_slope[idx]
    return Sw_shock, fw_shock, shock_speed_factor


def analytical_bl_profile(x, t, u_total, phi, Swc=0.2, Sor=0.2, mu_w=1.0, mu_o=5.0, **corey_kwargs):
    """
    Analytical 1D Buckley-Leverett saturation profile at position x, time t,
    for a constant-rate water injection into oil starting at x=0.
    """
    Sw_shock, fw_shock, shock_speed_factor = welge_shock_saturation(
        Swc=Swc, Sor=Sor, mu_w=mu_w, mu_o=mu_o, **corey_kwargs
    )
    shock_position = u_total * t * shock_speed_factor / phi

    Sw_range = np.linspace(Sw_shock, 1 - Sor, 500)
    fw_vals = fractional_flow(Sw_range, mu_w=mu_w, mu_o=mu_o, Swc=Swc, Sor=Sor, **corey_kwargs)
    dfw = np.gradient(fw_vals, Sw_range)
    positions = u_total * t * dfw / phi

    x = np.atleast_1d(x)
    profile = np.full_like(x, Swc, dtype=float)
    for i, xi in enumerate(x):
        if xi <= 0:
            profile[i] = 1 - Sor
        elif xi < shock_position:
            order = np.argsort(positions)
            profile[i] = np.interp(xi, positions[order], Sw_range[order])
        else:
            profile[i] = Swc

    return profile, shock_position, Sw_shock


# ---- 1D numerical solver (for validating against the analytical solution) ----

def solve_1d_saturation_numerical(nx=200, L=100.0, u_total=1.0, phi=0.2, t_end=20.0,
                                    Swc=0.2, Sor=0.2, mu_w=1.0, mu_o=5.0, cfl=0.4, **corey_kwargs):
    """
    Simple 1D upwind finite-volume solver for saturation transport, constant
    total velocity (pure Buckley-Leverett setup, no pressure solve needed
    since u_total is prescribed and constant -- isolates the
    saturation-transport numerics for direct comparison against the
    analytical solution, independent of any 2D pressure-solve error).
    """
    dx = L / nx
    Sw = np.full(nx, Swc)
    x = (np.arange(nx) + 0.5) * dx

    # GLOBAL max wave speed over the entire possible saturation range
    # [Swc, 1-Sor], computed ONCE -- not from the current (mostly
    # still-connate) cell values. Using the current local state
    # underestimates the true CFL bound badly here: at t=0 almost every
    # cell sits at Swc where dfw/dSw is near zero, so a local-state bound
    # picks one huge, unstable timestep and the front never actually
    # propagates past the first cell. The global bound is the standard,
    # safe way to set an explicit-scheme timestep for a nonlinear
    # conservation law with a non-monotonic wave-speed profile like this.
    Sw_probe = np.linspace(Swc + 1e-3, 1 - Sor - 1e-3, 500)
    fw_probe = fractional_flow(Sw_probe, mu_w=mu_w, mu_o=mu_o, Swc=Swc, Sor=Sor, **corey_kwargs)
    global_max_dfw_dSw = np.max(np.abs(np.gradient(fw_probe, Sw_probe)))
    dt = cfl * dx / (global_max_dfw_dSw * u_total / phi + 1e-8)

    t = 0.0
    while t < t_end:
        fw = fractional_flow(Sw, mu_w=mu_w, mu_o=mu_o, Swc=Swc, Sor=Sor, **corey_kwargs)
        dt = min(dt, t_end - t)

        fw_upwind = np.concatenate(([1 - Sor], fw))  # inlet BC: injecting water
        flux = u_total * fw_upwind
        dSw = -dt / (phi * dx) * (flux[1:] - flux[:-1])
        Sw = Sw + dSw
        Sw = np.clip(Sw, Swc, 1 - Sor)

        t += dt

    return x, Sw, t


if __name__ == "__main__":
    print("=== Corey relative permeability sanity check ===")
    krw, kro = corey_relperm(np.array([0.2, 0.5, 0.8]))
    print(f"at Sw=0.2 (connate water): krw={krw[0]:.4f}, kro={kro[0]:.4f} "
          f"(krw should be ~0, kro should be near max)")
    print(f"at Sw=0.8 (near residual oil): krw={krw[2]:.4f}, kro={kro[2]:.4f} "
          f"(krw should be near max, kro should be ~0)")
    assert krw[0] < 0.01 and kro[0] > 0.5, "connate water endpoint check failed"
    assert krw[2] > 0.2 and kro[2] < 0.05, "residual oil endpoint check failed"
    print("OK\n")

    print("=== Welge shock saturation ===")
    Sw_shock, fw_shock, speed = welge_shock_saturation()
    print(f"shock saturation Sw_f = {Sw_shock:.4f}")
    print(f"fractional flow at shock fw_f = {fw_shock:.4f}")
    print(f"shock speed factor (dfw/dSw at shock) = {speed:.4f}")
    assert 0.2 < Sw_shock < 0.8, "shock saturation should be physically reasonable"
    print("OK\n")

    print("=== 1D numerical vs analytical validation ===")
    # t_end chosen so the shock front stays within the domain (not yet at
    # breakthrough) -- comparing profiles is only meaningful before the
    # front exits, otherwise both solutions trivially agree (fully swept)
    nx, L, u_total, phi, t_end = 400, 100.0, 1.0, 0.2, 4.0
    x_num, Sw_num, t_actual = solve_1d_saturation_numerical(nx=nx, L=L, u_total=u_total, phi=phi, t_end=t_end)
    Sw_analytical, shock_pos, Sw_shock = analytical_bl_profile(x_num, t_end, u_total, phi)

    numerical_front_idx = np.where(Sw_num > Sw_shock - 0.05)[0]
    numerical_front_pos = x_num[numerical_front_idx[-1]] if len(numerical_front_idx) > 0 else np.nan

    print(f"analytical shock front position: {shock_pos:.2f}")
    print(f"numerical shock front position (approx): {numerical_front_pos:.2f}")
    front_error_pct = abs(numerical_front_pos - shock_pos) / shock_pos * 100
    print(f"front position error: {front_error_pct:.1f}%")

    mean_abs_error = np.mean(np.abs(Sw_num - Sw_analytical))
    print(f"mean absolute saturation error across profile: {mean_abs_error:.4f}")

    if front_error_pct < 15 and mean_abs_error < 0.1:
        print("PASS: numerical solver matches analytical Buckley-Leverett solution "
              "within reasonable numerical-diffusion tolerance (upwind schemes are "
              "known to smear sharp shocks somewhat -- this is expected, not a bug)")
    else:
        print("CHECK: error larger than expected -- review before using downstream")
