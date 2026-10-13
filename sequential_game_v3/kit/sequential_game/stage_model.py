"""
stage_model.py -- one-year transition models for the sequential game.

A stage model advances the reservoir state (pressure, saturation) by one fixed
span of simulated time under a given set of injection wells. The game code only
calls `advance`, so the real solver can be swapped for a stage-wise surrogate
(e.g. CCSNet with state inputs) without touching the game logic.

Units are the solver's nondimensional units. One stage = `stage_time` solver time
units; it is NOT yet calibrated to years or tons (see the study document).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver_v3 import solve_two_phase  # noqa: E402


class SolverStageModel:
    """Ground-truth transition: solver_v3 run for a fixed simulated time."""

    def __init__(self, stage_time=1.0, gravity=True, max_steps=400):
        self.stage_time = stage_time
        self.gravity = gravity
        self.max_steps = max_steps

    def initial_state(self, n, Swc=0.2):
        return {"p": np.zeros((n, n)), "Sw": np.full((n, n), Swc)}

    def advance(self, geo, state, wells):
        """wells: list of (i, j, rate), all injectors. Returns (next_state, elapsed_time)."""
        locs = [(i, j) for i, j, _ in wells]
        rates = [r for _, _, r in wells]
        kw = dict(add_gravity=True, rho_w=1.0, rho_n=0.6, g=1.5) if self.gravity else {}
        ph, sh, _, t = solve_two_phase(
            geo["k"], geo["phi"], locs, rates, [True] * len(locs),
            Sw_init=state["Sw"], p_init=state["p"], n_pressure_steps=self.max_steps,
            t_end=self.stage_time, save_every=10 ** 9, return_time=True, **kw)
        return {"p": ph[-1], "Sw": sh[-1]}, t
