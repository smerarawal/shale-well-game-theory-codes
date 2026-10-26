"""Small physical-invariant checks for the experimental CO2 x-z pilot solver."""
import unittest

import numpy as np

from co2_xz_solver import CO2XZConfig, CO2XZSolver, gas_vertical_centroid_m, initial_state


class CO2XZSolverChecks(unittest.TestCase):
    def setUp(self):
        self.shape = (8, 8)
        self.kx = np.full(self.shape, 500.0)
        self.kz = np.full(self.shape, 500.0)
        self.phi = np.full(self.shape, 0.2)
        self.cfg = CO2XZConfig(dx_ft=20, dz_ft=2, dy_ft=10,
                               max_dt_s=3600, max_steps=5000,
                               solubility_kg_m3_brine=0,
                               dissolution_rate_s_inv=0,
                               land_coefficient=0)
        self.solver = CO2XZSolver(self.cfg)

    def _bubble_state(self):
        sg = np.zeros(self.shape)
        sg[5, 4] = 0.3
        return initial_state(self.shape, gas_seed=sg)

    def test_no_gravity_has_no_vertical_drift(self):
        state = self._bubble_state()
        before = gas_vertical_centroid_m(state, self.phi, self.cfg.dz_ft)
        result, diag, _ = self.solver.advance(self.kx, self.kz, self.phi, state, [], 86400,
                                              gravity_sign=0)
        after = gas_vertical_centroid_m(result, self.phi, self.cfg.dz_ft)
        self.assertAlmostEqual(after, before, places=10)
        self.assertAlmostEqual(diag["pmax_increase_pa"], 0.0, places=10)

    def test_buoyancy_reverses_when_gravity_reverses(self):
        state = self._bubble_state()
        before = gas_vertical_centroid_m(state, self.phi, self.cfg.dz_ft)
        up, _, _ = self.solver.advance(self.kx, self.kz, self.phi, state, [], 2 * 86400,
                                       gravity_sign=1)
        down, _, _ = self.solver.advance(self.kx, self.kz, self.phi, state, [], 2 * 86400,
                                         gravity_sign=-1)
        z_up = gas_vertical_centroid_m(up, self.phi, self.cfg.dz_ft)
        z_down = gas_vertical_centroid_m(down, self.phi, self.cfg.dz_ft)
        self.assertLess(z_up, before)
        self.assertGreater(z_down, before)

    def test_buoyancy_trend_is_stable_under_timestep_refinement(self):
        state = self._bubble_state()
        centroids = []
        for max_dt in (3600.0, 1800.0):
            cfg = CO2XZConfig(dx_ft=20, dz_ft=2, dy_ft=10,
                              max_dt_s=max_dt, max_steps=5000,
                              solubility_kg_m3_brine=0,
                              dissolution_rate_s_inv=0,
                              land_coefficient=0)
            solver = CO2XZSolver(cfg)
            result, _, _ = solver.advance(self.kx, self.kz, self.phi, state, [], 2 * 86400,
                                          gravity_sign=1)
            centroids.append(gas_vertical_centroid_m(result, self.phi, cfg.dz_ft))
        before = gas_vertical_centroid_m(state, self.phi, self.cfg.dz_ft)
        self.assertLess(centroids[0], before)
        self.assertLess(centroids[1], before)
        self.assertLess(abs(centroids[1] - centroids[0]), 0.1)

    def test_retreating_mobile_gas_becomes_immobile_without_mass_loss(self):
        cfg = CO2XZConfig(dx_ft=20, dz_ft=2, dy_ft=10,
                          max_dt_s=1800, max_steps=5000,
                          solubility_kg_m3_brine=0,
                          dissolution_rate_s_inv=0,
                          land_coefficient=3.0)
        solver = CO2XZSolver(cfg)
        state = self._bubble_state()
        result, diag, _ = solver.advance(self.kx, self.kz, self.phi, state, [], 86400)
        self.assertGreater(diag["trapped_gas_kg"], 0)
        self.assertLess(abs(diag["mass_balance_residual_kg"]), 1e-8)
        self.assertLess(float(np.max(result["sg_mobile"])), float(np.max(state["sg_mobile"])))

    def test_injection_dissolution_and_trapping_conserve_component_mass(self):
        cfg = CO2XZConfig(dx_ft=20, dz_ft=2, dy_ft=10,
                          max_dt_s=1800, max_steps=5000,
                          solubility_kg_m3_brine=30,
                          dissolution_rate_s_inv=1e-4,
                          land_coefficient=3.0)
        solver = CO2XZSolver(cfg)
        state = initial_state(self.shape)
        state, first, _ = solver.advance(self.kx, self.kz, self.phi, state,
                                         [(7, 2, 1e-7), (7, 5, 1e-7)], 8 * 3600)
        state, second, _ = solver.advance(self.kx, self.kz, self.phi, state, [], 2 * 86400)
        self.assertGreater(first["injected_co2_kg_stage"], 0)
        self.assertGreater(first["pmax_increase_pa"], 0)
        self.assertGreater(first["dissolved_kg"], 0)
        self.assertLess(abs(first["mass_balance_residual_kg"]), 1e-8)
        self.assertLess(abs(second["mass_balance_residual_kg"]), 1e-8)
        self.assertGreaterEqual(float(np.min(state["sg_mobile"])), -1e-12)
        self.assertGreaterEqual(float(np.min(state["sg_trapped"])), -1e-12)
        self.assertGreaterEqual(float(np.min(1 - state["sg_mobile"] - state["sg_trapped"])), cfg.swc - 1e-9)


if __name__ == "__main__":
    unittest.main()
