"""CO2-in-brine stage transition using phase-specific Darcy fluxes.

This is an experimental, immiscible two-phase model. It represents pressure,
buoyancy, phase-specific relative permeability, and brine saturation. It does
not model dissolution, capillary trapping, thermal effects, geochemistry, or
geomechanics, and its solver units are not calibrated to field years or tonnes.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver_v3 import solve_two_phase  # noqa: E402


class CO2BrineStageModel:
    """Ground-truth CO2 injection transition for the sequential game."""

    def __init__(self, stage_time=1.0, max_steps=400, gravity=True,
                 rho_brine=1.0, rho_co2=0.6, g=1.5,
                 Swc=0.2, Sgr_max=0.02, mu_brine=1.0, mu_co2=5.0,
                 ct=1.0):
        self.stage_time = float(stage_time)
        self.max_steps = int(max_steps)
        self.gravity = bool(gravity)
        self.rho_brine = float(rho_brine)
        self.rho_co2 = float(rho_co2)
        self.g = float(g)
        self.Swc = float(Swc)
        self.Sgr_max = float(Sgr_max)
        self.mu_brine = float(mu_brine)
        self.mu_co2 = float(mu_co2)
        self.ct = float(ct)
        if not (0.0 <= self.Swc < 1.0) or not (0.0 <= self.Sgr_max < 1.0 - self.Swc):
            raise ValueError("require 0 <= Swc < 1 and 0 <= Sgr_max < 1-Swc")
        if self.rho_co2 >= self.rho_brine and self.gravity:
            raise ValueError("buoyant-rise configuration requires rho_co2 < rho_brine")

    def initial_state(self, n, Swc=None):
        # Start with brine-filled pore space and zero mobile CO2 saturation.
        # Swc is accepted for compatibility with the common stage-model API;
        # this model's configured Swc remains authoritative.
        sw_init = 1.0
        return {"p": np.zeros((n, n), dtype=np.float64),
                "Sw": np.full((n, n), sw_init, dtype=np.float64)}

    def advance(self, geo, state, wells):
        """Advance one game stage; all supplied wells inject the CO2 phase."""
        locs = [(int(i), int(j)) for i, j, _ in wells]
        rates = [float(rate) for _, _, rate in wells]
        # The current flow prototype does not yet represent hysteretic residual
        # gas. Keep its Corey endpoint at pure brine; Sgr_max is reserved for
        # the forthcoming mass-conservative trapping extension.
        corey = {"Swc": self.Swc, "Sor": 0.0}
        ph, sh, _, elapsed = solve_two_phase(
            geo["k"], geo["phi"], locs, rates, [False] * len(locs),
            well_modes=["co2_injector"] * len(locs),
            Sw_init=np.asarray(state["Sw"], dtype=np.float64),
            p_init=np.asarray(state["p"], dtype=np.float64),
            mu_w=self.mu_brine, mu_n=self.mu_co2, ct=self.ct,
            n_pressure_steps=self.max_steps, t_end=self.stage_time,
            save_every=self.max_steps + 1, return_time=True,
            add_gravity=self.gravity, rho_w=self.rho_brine,
            rho_n=self.rho_co2, g=self.g, phase_fluxes=True,
            **corey)
        return {"p": ph[-1], "Sw": sh[-1]}, elapsed
