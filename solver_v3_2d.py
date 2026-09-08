"""
solver_v3_2d.py -- 2D extension of solver_v3_multiphase.py's validated
Buckley-Leverett physics. The 1D module validated the saturation-transport
numerics against the analytical solution (0.4% front-position error); this
applies that same validated physics to a real 2D reservoir scenario: one
injector well pushing water into oil, one producer pulling fluid out,
tracking the saturation front sweeping across the grid between them.

Scheme: IMPES (IMplicit Pressure, Explicit Saturation), standard in
reservoir simulation --
  1. Solve pressure using TOTAL mobility (water+oil combined), same
     explicit-marching structure as solver_v2.py.
  2. Compute Darcy velocity from the pressure field.
  3. Advect saturation using the velocity field + fractional flow,
     upwinded, SUB-CYCLED with its own (smaller) CFL-stable timestep
     since saturation transport is stiffer than pressure diffusion.

Run:
    python solver_v3_2d.py
"""
import numpy as np

from solver_v3_multiphase import corey_relperm, fractional_flow, total_mobility

NX, NY = 32, 32


def harmonic_mean(a, b, eps=1e-10):
    return 2.0 * a * b / (a + b + eps)


def face_values_harmonic(k):
    nx, ny = k.shape
    k_east = np.zeros_like(k); k_west = np.zeros_like(k)
    k_north = np.zeros_like(k); k_south = np.zeros_like(k)
    k_east[:-1, :] = harmonic_mean(k[:-1, :], k[1:, :])
    k_west[1:, :] = k_east[:-1, :]
    k_north[:, :-1] = harmonic_mean(k[:, :-1], k[:, 1:])
    k_south[:, 1:] = k_north[:, :-1]
    return k_east, k_west, k_north, k_south


def solve_2d_two_phase(
    k, phi, injector_loc, producer_loc, injection_rate=1.0,
    nx=NX, ny=NY, dx=1.0, mu_w=1.0, mu_o=5.0,
    n_pressure_steps=300, saturation_cfl=0.4,
    Swc=0.2, Sor=0.2, **corey_kwargs
):
    """
    k: (nx, ny) permeability field (isotropic for this pass -- combining
       with solver_v2's anisotropic kx/ky is a straightforward extension,
       not done here to keep this scenario's physics easy to sanity-check)
    phi: (nx, ny) porosity field
    injector_loc, producer_loc: (i, j) grid cells
    injection_rate: water injection rate at injector (producer rate is
                    set equal and opposite, for a simple balanced pattern)

    Returns: Sw (final saturation field), pressure (final pressure field),
             saturation_history (list of Sw snapshots for animation/plotting)
    """
    k_e, k_w, k_n, k_s = face_values_harmonic(k)

    p = np.zeros((nx, ny))
    Sw = np.full((nx, ny), Swc)
    saturation_history = [Sw.copy()]

    for step in range(n_pressure_steps):
        lam_total = total_mobility(Sw, mu_w=mu_w, mu_o=mu_o, Swc=Swc, Sor=Sor, **corey_kwargs)
        lam_e, lam_w, lam_n, lam_s = face_values_harmonic(lam_total)

        dt_pressure = 0.4 * dx ** 2 / (4.0 * np.max(k) * np.max(lam_total) + 1e-8)

        p_e = np.zeros_like(p); p_w = np.zeros_like(p)
        p_n = np.zeros_like(p); p_s = np.zeros_like(p)
        p_e[:-1, :] = p[1:, :]; p_w[1:, :] = p[:-1, :]
        p_n[:, :-1] = p[:, 1:]; p_s[:, 1:] = p[:, :-1]

        flux_e = k_e * lam_e * (p_e - p)
        flux_w = k_w * lam_w * (p - p_w)
        flux_n = k_n * lam_n * (p_n - p)
        flux_s = k_s * lam_s * (p - p_s)
        div_flux = (flux_e - flux_w + flux_n - flux_s) / dx ** 2

        source = np.zeros((nx, ny))
        source[injector_loc] = injection_rate
        source[producer_loc] = -injection_rate

        p = p + dt_pressure * (div_flux + source)

        p_e2 = np.zeros_like(p); p_e2[:-1, :] = p[1:, :]
        p_n2 = np.zeros_like(p); p_n2[:, :-1] = p[:, 1:]
        vel_x = -k_e * lam_e * (p_e2 - p) / dx
        vel_y = -k_n * lam_n * (p_n2 - p) / dx
        vel_mag = np.sqrt(vel_x ** 2 + vel_y ** 2) + 1e-8

        Sw_probe = np.linspace(Swc + 1e-3, 1 - Sor - 1e-3, 200)
        fw_probe = fractional_flow(Sw_probe, mu_w=mu_w, mu_o=mu_o, Swc=Swc, Sor=Sor, **corey_kwargs)
        global_max_dfw = np.max(np.abs(np.gradient(fw_probe, Sw_probe)))
        dt_sat = saturation_cfl * dx / (global_max_dfw * np.max(vel_mag) / np.min(phi) + 1e-8)
        n_subcycles = max(1, int(np.ceil(dt_pressure / dt_sat)))
        dt_sub = dt_pressure / n_subcycles

        for _ in range(n_subcycles):
            fw = fractional_flow(Sw, mu_w=mu_w, mu_o=mu_o, Swc=Swc, Sor=Sor, **corey_kwargs)

            fw_e = np.zeros_like(fw); fw_w = np.zeros_like(fw)
            fw_n = np.zeros_like(fw); fw_s = np.zeros_like(fw)
            fw_e[:-1, :] = fw[1:, :]; fw_w[1:, :] = fw[:-1, :]
            fw_n[:, :-1] = fw[:, 1:]; fw_s[:, 1:] = fw[:, :-1]

            sat_flux_e = np.where(vel_x >= 0, fw, fw_e) * vel_x
            sat_flux_n = np.where(vel_y >= 0, fw, fw_n) * vel_y

            sat_flux_w = np.zeros_like(fw); sat_flux_w[1:, :] = sat_flux_e[:-1, :]
            sat_flux_s = np.zeros_like(fw); sat_flux_s[:, 1:] = sat_flux_n[:, :-1]

            div_sat_flux = (sat_flux_e - sat_flux_w + sat_flux_n - sat_flux_s) / dx

            sat_source = np.zeros((nx, ny))
            sat_source[injector_loc] = injection_rate

            Sw = Sw + dt_sub / phi * (-div_sat_flux + sat_source)
            Sw = np.clip(Sw, Swc, 1 - Sor)

        if step % 30 == 0:
            saturation_history.append(Sw.copy())

    saturation_history.append(Sw.copy())
    return Sw, p, saturation_history


if __name__ == "__main__":
    np.random.seed(5)
    k = np.full((NX, NY), 0.1)
    phi = np.full((NX, NY), 0.2)
    injector = (5, 16)
    producer = (26, 16)

    print(f"Running 2D two-phase flow: injector at {injector}, producer at {producer}")
    Sw_final, p_final, history = solve_2d_two_phase(k, phi, injector, producer, n_pressure_steps=200)

    print(f"final saturation range: {Sw_final.min():.3f} to {Sw_final.max():.3f}")
    print(f"saturation at injector: {Sw_final[injector]:.3f} (should be near max, well-flooded)")
    print(f"saturation at producer: {Sw_final[producer]:.3f} (should show breakthrough if front reached it)")
    print(f"saturation far from both wells: {Sw_final[16, 5]:.3f} (should be near connate, unswept)")

    assert Sw_final[injector] > Sw_final[16, 5], "injector should be more water-flooded than far-field"
    print("\nOK: saturation correctly higher near injector than in unswept far-field region")

    np.save("saturation_history_2d.npy", np.array(history))
    print("saved saturation_history_2d.npy for visualization")
