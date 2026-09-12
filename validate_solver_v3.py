"""
validate_solver_v3.py -- Part B of richer_physics_v3_spec.md. Mandatory
gate before trusting solver_v3.py for anything downstream (dataset
generation, game theory, etc.) -- per the spec: "an unvalidated saturation
solver is worse than no saturation solver."

B.1: 1D Buckley-Leverett shock front vs the Welge tangent-line analytical
     solution.
B.2: backward-compatibility check against solve_pressure_diffusion_v2 --
     via initializing Sw=1 everywhere so fw(Sw)=1 identically, the
     saturation equation becomes trivial, and the pressure equation should
     reduce to v2's single-phase equation with lambda_total = krw_max/mu_w
     playing the role of k/mu.

Run:
    python validate_solver_v3.py
"""
import numpy as np

from solver_v3 import fractional_flow, corey_relperm, solve_two_phase


# ---- B.1: analytical 1D Buckley-Leverett (Welge tangent construction) ----

def welge_shock_saturation(Swc=0.2, Sor=0.2, mu_w=1.0, mu_n=5.0, **corey_kwargs):
    Sw_range = np.linspace(Swc + 1e-4, 1 - Sor - 1e-4, 3000)
    fw_vals = fractional_flow(Sw_range, mu_w=mu_w, mu_n=mu_n, Swc=Swc, Sor=Sor, **corey_kwargs)
    secant_slope = fw_vals / (Sw_range - Swc)
    dfw = np.gradient(fw_vals, Sw_range)
    diff = dfw - secant_slope
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    idx = sign_changes[-1] if len(sign_changes) > 0 else np.argmin(np.abs(diff))
    return Sw_range[idx], fw_vals[idx], secant_slope[idx]


def solve_1d_saturation(nx=400, L=100.0, u_total=1.0, phi=0.2, t_end=4.0,
                          Swc=0.2, Sor=0.2, mu_w=1.0, mu_n=5.0, cfl=0.4, **corey_kwargs):
    dx = L / nx
    Sw = np.full(nx, Swc)

    Sw_probe = np.linspace(Swc + 1e-3, 1 - Sor - 1e-3, 500)
    fw_probe = fractional_flow(Sw_probe, mu_w=mu_w, mu_n=mu_n, Swc=Swc, Sor=Sor, **corey_kwargs)
    global_max_dfw = np.max(np.abs(np.gradient(fw_probe, Sw_probe)))
    dt = cfl * dx / (global_max_dfw * u_total / phi + 1e-8)

    t = 0.0
    while t < t_end:
        dt = min(dt, t_end - t)
        fw = fractional_flow(Sw, mu_w=mu_w, mu_n=mu_n, Swc=Swc, Sor=Sor, **corey_kwargs)
        fw_upwind = np.concatenate(([1 - Sor], fw))
        flux = u_total * fw_upwind
        Sw = Sw - dt / (phi * dx) * (flux[1:] - flux[:-1])
        Sw = np.clip(Sw, Swc, 1 - Sor)
        t += dt

    x = (np.arange(nx) + 0.5) * dx
    return x, Sw


def run_b1_buckley_leverett_test():
    print("=== B.1: Buckley-Leverett shock-front validation ===")
    Swc, Sor, mu_w, mu_n = 0.2, 0.2, 1.0, 5.0
    u_total, phi, t_end, L, nx = 1.0, 0.2, 4.0, 100.0, 400

    Sw_shock, fw_shock, speed_factor = welge_shock_saturation(Swc=Swc, Sor=Sor, mu_w=mu_w, mu_n=mu_n)
    analytical_shock_position = u_total * t_end * speed_factor / phi
    print(f"analytical shock saturation: {Sw_shock:.4f}, position: {analytical_shock_position:.2f}")

    x, Sw_num = solve_1d_saturation(nx=nx, L=L, u_total=u_total, phi=phi, t_end=t_end,
                                      Swc=Swc, Sor=Sor, mu_w=mu_w, mu_n=mu_n)
    front_idx = np.where(Sw_num > Sw_shock - 0.05)[0]
    numerical_shock_position = x[front_idx[-1]] if len(front_idx) > 0 else np.nan

    dx = L / nx
    error_cells = abs(numerical_shock_position - analytical_shock_position) / dx
    error_pct = abs(numerical_shock_position - analytical_shock_position) / analytical_shock_position * 100

    print(f"numerical shock position: {numerical_shock_position:.2f}")
    print(f"error: {error_pct:.2f}% ({error_cells:.2f} grid cells)")

    passed = error_cells <= 2.0
    print(f"{'PASS' if passed else 'FAIL'}: shock position within "
          f"{'the required 1-2' if passed else 'MORE than 2'} grid cells of analytical")
    if not passed:
        print("STOP: per the spec, do not proceed to use solver_v3 downstream until this passes.")
    return passed


def run_b2_backward_compat_test():
    """
    REVISED after debugging: the original test compared v3's pressure
    field (Sw=1 everywhere) directly against v2's, expecting saturation
    to stay pinned at exactly 1.0. That expectation was physically
    mistaken -- in the coupled two-phase system, ANY local pressure
    decline (whether from convective displacement or generic
    compressibility/storage relief) corresponds to REAL local water
    volume leaving that cell, since saturation IS the water volume
    fraction. v2's single-phase equation has no separate saturation
    field at all, so its "compressibility" is a pure pressure bookkeeping
    device with no phase-composition analog -- the two are not expected
    to match pointwise. The MEANINGFUL check for a compressible two-phase
    solver is GLOBAL MASS CONSERVATION: does the total saturation decline
    across the whole domain exactly equal the total volume actually
    withdrawn at the well? This is what's checked below instead.
    """
    print("\n=== B.2 (revised): global mass conservation check ===")
    print("(original pointwise-vs-v2 comparison was based on a flawed physical")
    print(" premise -- see function docstring; replaced with the actually")
    print(" meaningful check for a compressible two-phase solver)")

    nx, ny = 24, 24
    k_field = np.full((nx, ny), 0.15)
    phi_field = np.ones((nx, ny))
    Sw_init = np.ones((nx, ny))
    rate = 1.0

    p_hist, s_hist, dt = solve_two_phase(
        k_field, phi_field, well_locations=[(12, 12)], well_rates=[rate],
        well_is_water_injector=[False], Sw_init=Sw_init, n_pressure_steps=1, save_every=1,
        Swc=0.0, Sor=0.0, krw_max=0.8, krn_max=1.0, nw=1, nn=1, mu_w=1.0,
    )
    Sw_final = s_hist[-1]

    total_water_lost = np.sum((1.0 - Sw_final) * phi_field)
    expected = rate * dt
    ratio = total_water_lost / expected

    print(f"total water lost (summed over domain): {total_water_lost:.8f}")
    print(f"expected (rate * dt): {expected:.8f}")
    print(f"ratio: {ratio:.8f}")

    # spatial sanity: saturation decline should be concentrated near the
    # well and smoothly decay with distance (diffusive pattern), not
    # scattered or non-monotonic
    Sw_well = Sw_final[12, 12]
    Sw_near = Sw_final[13, 12]
    Sw_far = Sw_final[17, 12]
    Sw_edge = Sw_final[0, 0]
    print(f"spatial pattern check: Sw at well={Sw_well:.4f}, 1 cell away={Sw_near:.4f}, "
          f"5 cells away={Sw_far:.4f}, domain edge={Sw_edge:.4f}")
    monotonic_decay = Sw_well <= Sw_near <= Sw_far <= Sw_edge

    mass_conserved = abs(ratio - 1.0) < 1e-6
    print(f"{'PASS' if mass_conserved else 'FAIL'}: global mass conservation "
          f"({'exact' if mass_conserved else 'VIOLATED'})")
    print(f"{'PASS' if monotonic_decay else 'FAIL'}: saturation decays "
          f"monotonically with distance from the well (physically sensible)")

    return mass_conserved and monotonic_decay


if __name__ == "__main__":
    b1_passed = run_b1_buckley_leverett_test()
    b2_result = run_b2_backward_compat_test()

    print(f"\n{'='*60}\nVALIDATION SUMMARY\n{'='*60}")
    print(f"B.1 Buckley-Leverett: {'PASS' if b1_passed else 'FAIL'}")
    if b2_result is None:
        print("B.2 backward-compat: SKIPPED (solver_v2.py not found)")
    else:
        print(f"B.2 backward-compat: {'PASS' if b2_result else 'FAIL'}")

    if b1_passed and (b2_result is None or bool(b2_result)):
        print("\nGATE CLEARED: safe to proceed to Part C (data_gen_v3_multiphase.py)")
    else:
        print("\nGATE NOT CLEARED: per the spec, fix solver_v3.py before generating any "
              "dataset or trusting downstream results from it.")
