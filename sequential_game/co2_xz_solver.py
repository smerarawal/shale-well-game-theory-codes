"""Experimental 2D vertical x-z CO2/brine IMPES solver for pilot studies.

This is a dimensionally consistent, anisotropic, finite-volume research model,
not a field-calibrated simulator. It uses constant phase densities/viscosities,
no capillary pressure, local kinetic dissolution, and a simplified Land-style
residual-trapping rule. Use only after the included conservation and gravity
checks pass for the chosen grid and timestep.

Axes: arrays are [z, x], z index 0 is shallow, and positive gravity points
toward increasing z index (downward). Horizontal SPE10 patches are not inputs.
Kx/Kz are supplied in mD, lengths in ft, pressure in Pa, rates in m^3/s, and
time in seconds. Face boundaries are closed/no-flow.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
except ImportError:  # Tiny local invariant tests can run without SciPy installed.
    coo_matrix = None
    spsolve = None


MD_TO_M2 = 9.869233e-16
FT_TO_M = 0.3048


@dataclass(frozen=True)
class CO2XZConfig:
    dx_ft: float = 20.0
    dz_ft: float = 2.0
    dy_ft: float = 10.0
    mu_brine_pa_s: float = 1.0e-3
    mu_co2_pa_s: float = 6.0e-5
    rho_brine_kg_m3: float = 1030.0
    rho_co2_kg_m3: float = 650.0
    gravity_m_s2: float = 9.81
    total_compressibility_pa_inv: float = 1.0e-9
    swc: float = 0.15
    krw_endpoint: float = 1.0
    krg_endpoint: float = 1.0
    corey_brine_exp: float = 2.0
    corey_co2_exp: float = 2.0
    land_coefficient: float = 2.0
    solubility_kg_m3_brine: float = 30.0
    dissolution_rate_s_inv: float = 2.0e-7
    transport_cfl: float = 0.35
    max_dt_s: float = 86400.0
    max_steps: int = 20000

    def __post_init__(self):
        positive = (self.dx_ft, self.dz_ft, self.dy_ft, self.gravity_m_s2, self.mu_brine_pa_s,
                    self.mu_co2_pa_s, self.rho_brine_kg_m3,
                    self.rho_co2_kg_m3, self.total_compressibility_pa_inv,
                    self.max_dt_s)
        if any(v <= 0 for v in positive):
            raise ValueError("lengths, viscosities, densities, compressibility, and max_dt must be positive")
        if not 0 <= self.swc < 1:
            raise ValueError("swc must be in [0, 1)")
        if self.land_coefficient < 0 or self.solubility_kg_m3_brine < 0 or self.dissolution_rate_s_inv < 0:
            raise ValueError("trapping and dissolution parameters must be nonnegative")
        if not 0 < self.transport_cfl <= 0.5:
            raise ValueError("transport_cfl must be in (0, 0.5]")


def initial_state(shape, gas_seed=None, dissolved_kg_m3=0.0):
    """Create brine-filled state; optional gas_seed is an initial mobile Sg map."""
    nz, nx = map(int, shape)
    if nz < 2 or nx < 2:
        raise ValueError("x-z grid must be at least 2 by 2")
    sg = np.zeros((nz, nx), dtype=np.float64) if gas_seed is None else np.asarray(gas_seed, dtype=np.float64).copy()
    if sg.shape != (nz, nx) or np.any(sg < 0) or np.any(sg >= 1):
        raise ValueError("gas_seed must match grid and lie in [0,1)")
    c = np.broadcast_to(np.asarray(dissolved_kg_m3, dtype=np.float64), (nz, nx)).copy()
    if np.any(c < 0):
        raise ValueError("initial dissolved concentration must be nonnegative")
    return {"p_pa": np.zeros((nz, nx), dtype=np.float64),
            "sg_mobile": sg,
            "sg_trapped": np.zeros((nz, nx), dtype=np.float64),
            "sg_max": sg.copy(),
            "c_dissolved_kg_m3": c,
            "time_s": 0.0,
            "injected_co2_kg": 0.0}


def _harmonic(a, b):
    return 2.0 * a * b / np.maximum(a + b, 1e-300)


def _divergence(qx, qz):
    """Finite-volume divergence of positive-east and positive-down face rates."""
    nz, nxm1 = qx.shape
    nx = nxm1 + 1
    div = np.zeros((nz, nx), dtype=np.float64)
    div[:, :-1] += qx
    div[:, 1:] -= qx
    div[:-1, :] += qz
    div[1:, :] -= qz
    return div


def _outflow(qx, qz):
    """Cellwise sum of outgoing positive volumetric face rates."""
    nz, nxm1 = qx.shape
    nx = nxm1 + 1
    out = np.zeros((nz, nx), dtype=np.float64)
    out[:, :-1] += np.maximum(qx, 0.0)
    out[:, 1:] += np.maximum(-qx, 0.0)
    out[:-1, :] += np.maximum(qz, 0.0)
    out[1:, :] += np.maximum(-qz, 0.0)
    return out


class CO2XZSolver:
    """Two-phase x-z IMPES flow with dissolved and residually trapped CO2.

    Wells are ``(z_index, x_index, rate_m3_s)`` tuples for pure CO2 injection.
    The stage is closed at external faces; production wells are not implemented.
    """

    def __init__(self, config=CO2XZConfig()):
        self.cfg = config
        self.dx = config.dx_ft * FT_TO_M
        self.dz = config.dz_ft * FT_TO_M
        self.dy = config.dy_ft * FT_TO_M
        self.cell_volume = self.dx * self.dy * self.dz

    def _validate_fields(self, kx_mD, kz_mD, phi):
        kx, kz, poro = (np.asarray(a, dtype=np.float64) for a in (kx_mD, kz_mD, phi))
        if kx.ndim != 2 or kz.shape != kx.shape or poro.shape != kx.shape:
            raise ValueError("Kx, Kz, and porosity must be matching [z,x] arrays")
        if np.any(~np.isfinite(kx)) or np.any(~np.isfinite(kz)) or np.any(~np.isfinite(poro)):
            raise ValueError("rock fields must be finite")
        if np.any(kx <= 0) or np.any(kz <= 0) or np.any((poro <= 0) | (poro >= 1)):
            raise ValueError("permeability must be positive and porosity must lie in (0,1)")
        return kx * MD_TO_M2, kz * MD_TO_M2, poro

    def _mobilities(self, sg_mobile, sg_trapped):
        cfg = self.cfg
        sw = 1.0 - sg_mobile - sg_trapped
        if np.any(sw < cfg.swc - 1e-9):
            raise FloatingPointError("brine saturation fell below connate-water saturation")
        se_w = np.clip((sw - cfg.swc) / (1.0 - cfg.swc), 0.0, 1.0)
        se_g = np.clip(sg_mobile / (1.0 - cfg.swc), 0.0, 1.0)
        lam_w = cfg.krw_endpoint * se_w ** cfg.corey_brine_exp / cfg.mu_brine_pa_s
        lam_g = cfg.krg_endpoint * se_g ** cfg.corey_co2_exp / cfg.mu_co2_pa_s
        return sw, lam_w, lam_g

    def _face_coefficients(self, kx, kz, lam_w, lam_g):
        # Permeability is harmonic across cell interfaces. Lagged phase
        # mobilities use arithmetic averaging so a phase can enter a cell where
        # its local saturation is zero; harmonic mobility would lock the phase
        # front because its endpoint mobility is exactly zero.
        kx_f = _harmonic(kx[:, :-1], kx[:, 1:])
        kz_f = _harmonic(kz[:-1, :], kz[1:, :])
        lwx, lgx = 0.5 * (lam_w[:, :-1] + lam_w[:, 1:]), 0.5 * (lam_g[:, :-1] + lam_g[:, 1:])
        lwz, lgz = 0.5 * (lam_w[:-1, :] + lam_w[1:, :]), 0.5 * (lam_g[:-1, :] + lam_g[1:, :])
        txw = kx_f * lwx * (self.dy * self.dz) / self.dx
        txg = kx_f * lgx * (self.dy * self.dz) / self.dx
        tzw = kz_f * lwz * (self.dy * self.dx) / self.dz
        tzg = kz_f * lgz * (self.dy * self.dx) / self.dz
        return txw, txg, tzw, tzg

    def _pressure(self, p_old, phi, txw, txg, tzw, tzg, qwell, dt, gravity_sign):
        nz, nx = p_old.shape
        n = nz * nx
        cfg = self.cfg
        accum = phi * cfg.total_compressibility_pa_inv * self.cell_volume / dt
        diag = accum.copy()
        rhs = accum * p_old + qwell.copy()
        rows, cols, vals = [], [], []

        def add_edges(t, vertical=False):
            nonlocal diag, rhs
            if vertical:
                ii, jj = np.indices(t.shape)
                a = (ii * nx + jj).ravel()
                b = ((ii + 1) * nx + jj).ravel()
                grav_rate = ((cfg.rho_brine_kg_m3 * tx_dummy + cfg.rho_co2_kg_m3 * tg_dummy)
                             * cfg.gravity_m_s2 * gravity_sign * self.dz).ravel()
            else:
                ii, jj = np.indices(t.shape)
                a = (ii * nx + jj).ravel()
                b = (ii * nx + jj + 1).ravel()
                grav_rate = np.zeros(t.size)
            tv = t.ravel()
            np.add.at(diag.ravel(), a, tv)
            np.add.at(diag.ravel(), b, tv)
            # Diagonal edge terms are already accumulated in ``diag`` above;
            # append only the two off-diagonal entries here.
            rows.extend((a, b))
            cols.extend((b, a))
            vals.extend((-tv, -tv))
            # Total gravity-face rate enters the pressure RHS as -div(G).
            if vertical:
                rhs[:-1, :] -= grav_rate.reshape(t.shape)
                rhs[1:, :] += grav_rate.reshape(t.shape)

        # For pressure gravity, mix phase transmissibilities and densities.
        tx_dummy, tg_dummy = txw, txg
        add_edges(txw + txg, vertical=False)
        tx_dummy, tg_dummy = tzw, tzg
        add_edges(tzw + tzg, vertical=True)
        rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag.ravel())
        row_idx, col_idx, values = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        if coo_matrix is None:
            # Dense fallback is only intended for small development grids; the
            # Kaggle/production path uses SciPy's sparse direct solver.
            matrix = np.zeros((n, n), dtype=np.float64)
            np.add.at(matrix, (row_idx, col_idx), values)
            pressure = np.linalg.solve(matrix, rhs.ravel())
        else:
            matrix = coo_matrix((values, (row_idx, col_idx)), shape=(n, n)).tocsr()
            pressure = spsolve(matrix, rhs.ravel())
        return pressure.reshape(nz, nx)

    def _phase_fluxes(self, p, kx, kz, lamw, lamg, gravity_sign):
        cfg = self.cfg
        kx_f = _harmonic(kx[:, :-1], kx[:, 1:])
        kz_f = _harmonic(kz[:-1, :], kz[1:, :])
        drive_x = p[:, :-1] - p[:, 1:]
        gz = cfg.gravity_m_s2 * gravity_sign * self.dz
        drive_zw = p[:-1, :] - p[1:, :] + cfg.rho_brine_kg_m3 * gz
        drive_zg = p[:-1, :] - p[1:, :] + cfg.rho_co2_kg_m3 * gz
        # Use phase-potential upwinding for transport fluxes. This prevents a
        # phase with zero upstream saturation from spuriously leaving a cell.
        lwx = np.where(drive_x >= 0, lamw[:, :-1], lamw[:, 1:])
        lgx = np.where(drive_x >= 0, lamg[:, :-1], lamg[:, 1:])
        lwz = np.where(drive_zw >= 0, lamw[:-1, :], lamw[1:, :])
        lgz = np.where(drive_zg >= 0, lamg[:-1, :], lamg[1:, :])
        qxw = kx_f * lwx * (self.dy * self.dz) / self.dx * drive_x
        qxg = kx_f * lgx * (self.dy * self.dz) / self.dx * drive_x
        qzw = kz_f * lwz * (self.dy * self.dx) / self.dz * drive_zw
        qzg = kz_f * lgz * (self.dy * self.dx) / self.dz * drive_zg
        return qxw, qzw, qxg, qzg

    def _stable_dt(self, phi, sg_mobile, sg_trapped, qxw, qzw, qxg, qzg, wells):
        sw = 1.0 - sg_mobile - sg_trapped
        pv = phi * self.cell_volume
        dt = self.cfg.max_dt_s
        for saturation, out in ((sw, _outflow(qxw, qzw)), (sg_mobile, _outflow(qxg, qzg))):
            active = out > 1e-30
            if np.any(active):
                capacity = pv * np.maximum(saturation, 1e-10)
                dt = min(dt, float(np.min(self.cfg.transport_cfl * capacity[active] / out[active])))
        for z, x, rate in wells:
            available = pv[z, x] * max(1.0 - self.cfg.swc - sg_mobile[z, x] - sg_trapped[z, x], 1e-8)
            dt = min(dt, self.cfg.transport_cfl * available / rate)
        return max(dt, 1e-6)

    def _dissolved_transport(self, c, sw, phi, qxw, qzw, dt):
        cx = np.where(qxw >= 0, c[:, :-1], c[:, 1:])
        cz = np.where(qzw >= 0, c[:-1, :], c[1:, :])
        mass = phi * sw * self.cell_volume * c
        mass -= dt * _divergence(qxw * cx, qzw * cz)
        if np.any(mass < -1e-9):
            raise FloatingPointError("dissolved-CO2 transport produced negative mass; reduce timestep")
        return np.maximum(mass, 0.0)

    def component_masses(self, state, phi):
        pv = np.asarray(phi, dtype=np.float64) * self.cell_volume
        gas_pore_volume = pv * (state["sg_mobile"] + state["sg_trapped"])
        mobile_kg = self.cfg.rho_co2_kg_m3 * float(np.sum(pv * state["sg_mobile"]))
        trapped_kg = self.cfg.rho_co2_kg_m3 * float(np.sum(pv * state["sg_trapped"]))
        water_volume = pv * (1.0 - state["sg_mobile"] - state["sg_trapped"])
        dissolved_kg = float(np.sum(water_volume * state["c_dissolved_kg_m3"]))
        return {"mobile_gas_kg": mobile_kg, "trapped_gas_kg": trapped_kg,
                "dissolved_kg": dissolved_kg,
                "total_co2_kg": mobile_kg + trapped_kg + dissolved_kg}

    def advance(self, kx_mD, kz_mD, phi, state, wells, duration_s, *, gravity_sign=1,
                save_snapshots=False):
        """Advance one stage; return (new state, diagnostics, optional snapshots).

        ``gravity_sign`` is +1 for depth-positive-down gravity, -1 to reverse
        gravity for a symmetry validation, and 0 for a no-gravity control.
        """
        kx, kz, poro = self._validate_fields(kx_mD, kz_mD, phi)
        if gravity_sign not in (-1, 0, 1):
            raise ValueError("gravity_sign must be -1, 0, or 1")
        wells = [(int(z), int(x), float(q)) for z, x, q in wells]
        if any(not (0 <= z < poro.shape[0] and 0 <= x < poro.shape[1]) or q <= 0 for z, x, q in wells):
            raise ValueError("CO2 injection wells require valid cells and positive rates")
        shape = poro.shape
        st = {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in state.items()}
        for key in ("p_pa", "sg_mobile", "sg_trapped", "sg_max", "c_dissolved_kg_m3"):
            if st[key].shape != shape:
                raise ValueError(f"state {key} shape does not match geology")
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if any(np.any(~np.isfinite(st[key])) for key in
               ("p_pa", "sg_mobile", "sg_trapped", "sg_max", "c_dissolved_kg_m3")):
            raise ValueError("state fields must be finite")
        sw0 = 1.0 - st["sg_mobile"] - st["sg_trapped"]
        if (np.any(st["sg_mobile"] < 0) or np.any(st["sg_trapped"] < 0)
                or np.any(sw0 < self.cfg.swc - 1e-9)
                or np.any(st["c_dissolved_kg_m3"] < 0)):
            raise ValueError("state contains invalid saturation or dissolved concentration")

        start = self.component_masses(st, poro)
        initial_pressure = st["p_pa"].copy()
        total_injected = 0.0
        time_left = float(duration_s)
        steps = 0
        mass_residual_max = 0.0
        min_sw = float(np.min(1.0 - st["sg_mobile"] - st["sg_trapped"]))
        snapshots = []
        qwell = np.zeros(shape, dtype=np.float64)
        qgas = np.zeros(shape, dtype=np.float64)
        for z, x, rate in wells:
            qwell[z, x] += rate
            qgas[z, x] += rate

        while time_left > max(1e-9, duration_s * 1e-14):
            steps += 1
            if steps > self.cfg.max_steps:
                raise RuntimeError(f"CO2 stage exceeded max_steps={self.cfg.max_steps}; reduce stage duration/rate or increase grid coarsening")
            sg_mob, sg_trap = st["sg_mobile"], st["sg_trapped"]
            sw, lamw, lamg = self._mobilities(sg_mob, sg_trap)
            txw, txg, tzw, tzg = self._face_coefficients(kx, kz, lamw, lamg)
            dt = min(time_left, self.cfg.max_dt_s)

            # Pressure is implicit; redo with a smaller dt if explicit phase
            # transport would violate its cellwise CFL bound.
            stable = dt
            cfl_converged = False
            # Pressure depends on dt through accumulation, so the transport
            # CFL bound is a fixed-point condition rather than a one-shot
            # estimate. Heterogeneous SPE10 fields can need more than a few
            # backtracks before that condition settles.
            for _ in range(64):
                p_new = self._pressure(st["p_pa"], poro, txw, txg, tzw, tzg,
                                       qwell, dt, gravity_sign)
                qxw, qzw, qxg, qzg = self._phase_fluxes(p_new, kx, kz, lamw, lamg, gravity_sign)
                stable = self._stable_dt(poro, sg_mob, sg_trap, qxw, qzw, qxg, qzg, wells)
                if stable >= dt * (1 - 1e-10):
                    cfl_converged = True
                    break
                next_dt = min(dt, stable)
                if next_dt >= dt * (1 - 1e-14):
                    break
                dt = next_dt
            if not cfl_converged:
                raise RuntimeError(
                    "could not satisfy CO2 transport CFL bound after 64 pressure/transport iterations "
                    f"(trial dt={dt:.6g} s, latest stable dt={stable:.6g} s, "
                    f"minimum configured timestep=1e-6 s); check injection rate, grid units, "
                    "and whether a cell is at its saturation limit")

            old_mass = self.component_masses(st, poro)["total_co2_kg"]
            pv = poro * self.cell_volume
            sg_candidate = sg_mob - dt * _divergence(qxg, qzg) / pv + dt * qgas / pv
            if np.any(sg_candidate < -1e-8) or np.any(sg_candidate + sg_trap > 1.0 - self.cfg.swc + 1e-8):
                raise FloatingPointError("mobile gas update exceeded saturation bounds; reduce timestep or injection rate")
            sg_candidate = np.maximum(sg_candidate, 0.0)

            # Hysteretic residual trapping is evaluated on the transport update
            # before dissolution. Therefore a later local dissolution transfer
            # cannot itself trigger trapping; this models transport retreat as
            # an imbibition proxy, not dissolution as imbibition.
            gas_total_candidate = sg_candidate + sg_trap
            previous_total = sg_mob + sg_trap
            sg_max = np.maximum(st["sg_max"], gas_total_candidate)
            retreat = gas_total_candidate < previous_total - 1e-12
            residual_target = sg_max / (1.0 + self.cfg.land_coefficient * sg_max)
            target_trapped = np.where(retreat, np.minimum(gas_total_candidate, residual_target), sg_trap)
            sg_trap_new = np.maximum(sg_trap, target_trapped)
            sg_mob_new = gas_total_candidate - sg_trap_new

            # Advect aqueous CO2 conservatively with the phase-specific brine
            # flux before local finite-rate dissolution.
            sw_new = 1.0 - sg_mob_new - sg_trap_new
            dissolved_mass = self._dissolved_transport(st["c_dissolved_kg_m3"], sw, poro, qxw, qzw, dt)
            water_volume = pv * sw_new
            dissolved_capacity = np.maximum(
                self.cfg.solubility_kg_m3_brine * water_volume - dissolved_mass, 0.0)
            mobile_mass = self.cfg.rho_co2_kg_m3 * pv * sg_mob_new
            transfer = np.minimum(mobile_mass, dissolved_capacity *
                                  (1.0 - np.exp(-self.cfg.dissolution_rate_s_inv * dt)))
            mobile_mass -= transfer
            dissolved_mass += transfer
            sg_mob_new = mobile_mass / (self.cfg.rho_co2_kg_m3 * pv)
            sw_after_dissolution = 1.0 - sg_mob_new - sg_trap_new
            water_volume = pv * sw_after_dissolution
            c_new = np.divide(dissolved_mass, water_volume, out=np.zeros_like(dissolved_mass),
                              where=water_volume > 1e-20)

            injected = self.cfg.rho_co2_kg_m3 * sum(rate for _, _, rate in wells) * dt
            next_state = {"p_pa": p_new, "sg_mobile": sg_mob_new,
                          "sg_trapped": sg_trap_new, "sg_max": sg_max,
                          "c_dissolved_kg_m3": c_new,
                          "time_s": float(st["time_s"] + dt),
                          "injected_co2_kg": float(st["injected_co2_kg"] + injected)}
            end_mass = self.component_masses(next_state, poro)["total_co2_kg"]
            residual = end_mass - old_mass - injected
            mass_residual_max = max(mass_residual_max, abs(residual))
            total_injected += injected
            st = next_state
            time_left -= dt
            min_sw = min(min_sw, float(np.min(sw_new)))
            if save_snapshots:
                snapshots.append({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()})

        end = self.component_masses(st, poro)
        plume = st["sg_mobile"] >= 0.01
        diagnostics = {"steps": steps, "duration_s": float(duration_s),
                      "pmax_pa": float(np.max(st["p_pa"])),
                      "pmax_increase_pa": float(np.max(st["p_pa"] - initial_pressure)),
                      "min_sw": min_sw,
                      "max_sg_mobile": float(np.max(st["sg_mobile"])),
                      "max_sg_trapped": float(np.max(st["sg_trapped"])),
                      "mobile_gas_kg": end["mobile_gas_kg"],
                      "trapped_gas_kg": end["trapped_gas_kg"],
                      "dissolved_kg": end["dissolved_kg"],
                      "total_co2_kg": end["total_co2_kg"],
                      "injected_co2_kg_stage": total_injected,
                      "mass_balance_residual_kg": end["total_co2_kg"] - start["total_co2_kg"] - total_injected,
                      "max_step_mass_residual_kg": mass_residual_max,
                      "plume_cell_fraction_sg_ge_0p01": float(plume.mean()),
                      "top_boundary_mobile_gas_fraction": float(plume[0, :].mean())}
        return st, diagnostics, snapshots


def gas_vertical_centroid_m(state, phi, dz_ft):
    """Depth-coordinate center of total gas inventory; lower values are shallower."""
    gas = np.asarray(phi) * (state["sg_mobile"] + state["sg_trapped"])
    total = gas.sum()
    if total <= 1e-20:
        return float("nan")
    z = (np.arange(gas.shape[0]) + 0.5) * dz_ft * FT_TO_M
    return float((gas * z[:, None]).sum() / total)
