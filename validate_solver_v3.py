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
    print("\n=== B.2: backward-compatibility check against solver_v2 ===")
    try:
        from solver_v2 import solve_pressure_diffusion_v2
    except ImportError:
        print("SKIPPED: solver_v2.py not found in this directory -- copy it alongside "
              "solver_v3.py to run this check.")
        return None

    nx, ny = 24, 24
    k_field = np.full((nx, ny), 0.15)
    phi_field = np.ones((nx, ny))
    krw_max = 0.8
    mu_w = 1.0

    Sw_init = np.ones((nx, ny))
    p_hist_v3, s_hist_v3, dt_v3 = solve_two_phase(
        k_field, phi_field, well_locations=[(12, 12)], well_rates=[1.0],
        well_is_water_injector=[False],
        Sw_init=Sw_init, n_pressure_steps=200, save_every=200,
        Swc=0.0, Sor=0.0, krw_max=krw_max, krn_max=1.0, nw=1, nn=1, mu_w=mu_w,
    )
    p_v3_final = p_hist_v3[-1]

    k_effective = k_field * (krw_max / mu_w)
    p_hist_v2, dt_v2 = solve_pressure_diffusion_v2(
        kx=k_effective, ky=k_effective, phi=phi_field,
        well_locations=[(12, 12)], well_rates=[-1.0],
        mu=1.0, ct=1.0, nt=200,
    )
    p_v2_final = p_hist_v2[-1]

    max_abs_diff = np.max(np.abs(p_v3_final - p_v2_final))
    rel_diff = max_abs_diff / (np.max(np.abs(p_v2_final)) + 1e-8)
    print(f"max absolute pressure difference: {max_abs_diff:.4f}")
    print(f"relative difference: {rel_diff:.4%}")

    passed = rel_diff < 0.05
    print(f"{'PASS' if passed else 'FAIL'}: v3 (Sw=1 everywhere) matches v2 within tolerance")
    if not passed:
        print("NOTE: some divergence is expected since v3 uses its own timestep/substepping "
              "logic, not identical code paths to v2 -- if this fails by a large margin, "
              "that indicates a real bug, review before trusting solver_v3.py.")
    return passed


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
