"""Pressure-only CO2 physics prior using the conventions in co2_xz_solver.

The prior intentionally reuses the solver's pressure assembly and transmissibility
helpers. It does not reproduce transport, trapping, dissolution, or the solver's
adaptive stage trajectory. ``frozen`` holds the starting mobilities fixed;
``path`` interpolates from starting saturation to a supplied predicted endpoint.
Both are approximate stage priors and must be assessed against held-out labels.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import numpy as np

try:  # Direct execution from sequential_game or package import from repo root.
    from .co2_xz_solver import CO2XZConfig, CO2XZSolver
except ImportError:
    from co2_xz_solver import CO2XZConfig, CO2XZSolver


def _as_fields(state: Mapping[str, np.ndarray], i: int) -> dict[str, np.ndarray]:
    return {name: np.asarray(value[i], dtype=np.float64)
            for name, value in state.items()}


def _config_for_row(data: Mapping[str, np.ndarray], i: int,
                    base: CO2XZConfig | None = None) -> CO2XZConfig:
    base = base or CO2XZConfig()
    return replace(base,
                   dx_ft=float(data["dx_ft"][i]),
                   dy_ft=float(data["dy_ft"][i]),
                   dz_ft=float(data["dz_ft"][i]))


def pressure_prior_one(data: Mapping[str, np.ndarray], i: int, *,
                       n_sub: int = 8, mode: str = "frozen",
                       endpoint: Mapping[str, np.ndarray] | None = None,
                       stage_days: float = 30.0,
                       gravity_sign: int = 1,
                       config: CO2XZConfig | None = None) -> np.ndarray:
    """Return a pressure *increment* [Pa] for one labelled transition.

    ``mode='frozen'`` uses the stage-start mobile/trapped saturation throughout.
    ``mode='path'`` requires endpoint states and linearly interpolates mobile and
    trapped saturation over the fixed prior substeps. The interpolation is an
    approximation to the solver's adaptive transport path, not an exact replay.
    """
    if n_sub < 1:
        raise ValueError("n_sub must be positive")
    if mode not in {"frozen", "path"}:
        raise ValueError("mode must be 'frozen' or 'path'")
    if mode == "path" and endpoint is None:
        raise ValueError("path prior requires endpoint saturation fields")

    cfg = _config_for_row(data, i, config)
    solver = CO2XZSolver(cfg)
    kx, kz, phi = solver._validate_fields(data["kx_mD"][i], data["kz_mD"][i],
                                          data["phi"][i])
    p = np.asarray(data["p_pa_in"][i], dtype=np.float64).copy()
    start_mobile = np.asarray(data["sg_mobile_in"][i], dtype=np.float64)
    start_trapped = np.asarray(data["sg_trapped_in"][i], dtype=np.float64)
    if mode == "path":
        end_mobile = np.asarray(endpoint["sg_mobile"][i], dtype=np.float64)
        end_trapped = np.asarray(endpoint["sg_trapped"][i], dtype=np.float64)
    else:
        end_mobile, end_trapped = start_mobile, start_trapped

    nz, nx = p.shape
    qwell = np.zeros_like(p)
    wz = nz - 2
    x0, x1 = max(1, nx // 3), min(nx - 2, (2 * nx) // 3)
    qwell[wz, x0] += float(data["rate_operator_0_m3_s"][i])
    qwell[wz, x1] += float(data["rate_operator_1_m3_s"][i])
    dt = float(stage_days) * 86400.0 / n_sub

    for step in range(n_sub):
        # The solver evaluates lagged mobilities at the beginning of each
        # pressure step. This fixed interpolation is only the prior's path.
        alpha = step / n_sub if mode == "path" else 0.0
        sg_m = (1.0 - alpha) * start_mobile + alpha * end_mobile
        sg_t = (1.0 - alpha) * start_trapped + alpha * end_trapped
        _, lam_w, lam_g = solver._mobilities(sg_m, sg_t)
        txw, txg, tzw, tzg = solver._face_coefficients(kx, kz, lam_w, lam_g)
        p = solver._pressure(p, phi, txw, txg, tzw, tzg, qwell, dt,
                             gravity_sign)
    return p - np.asarray(data["p_pa_in"][i], dtype=np.float64)


def pressure_prior_batch(data: Mapping[str, np.ndarray], *, n_sub: int = 8,
                         mode: str = "frozen",
                         endpoint: Mapping[str, np.ndarray] | None = None,
                         stage_days: float = 30.0,
                         gravity_sign: int = 1,
                         config: CO2XZConfig | None = None) -> np.ndarray:
    """Compute pressure increments [N,Z,X]. Intended for cached preprocessing."""
    n = len(data["p_pa_in"])
    return np.stack([pressure_prior_one(data, i, n_sub=n_sub, mode=mode,
                                        endpoint=endpoint, stage_days=stage_days,
                                        gravity_sign=gravity_sign, config=config)
                     for i in range(n)]).astype(np.float32)


def compare_prior_to_labels(data: Mapping[str, np.ndarray], *, n_sub: int,
                            mode: str = "frozen",
                            endpoint: Mapping[str, np.ndarray] | None = None,
                            stage_days: float = 30.0,
                            gravity_sign: int = 1,
                            config: CO2XZConfig | None = None) -> dict[str, float]:
    """Summarize stage-pressure error of the prior alone, with peak diagnostics."""
    pred = pressure_prior_batch(data, n_sub=n_sub, mode=mode, endpoint=endpoint,
                                stage_days=stage_days, gravity_sign=gravity_sign,
                                config=config) + data["p_pa_in"]
    truth = np.asarray(data["p_pa_out"])
    error = pred - truth
    peak_error = pred.reshape(len(pred), -1).max(1) - truth.reshape(len(truth), -1).max(1)
    return {"pressure_rmse_pa": float(np.sqrt(np.mean(error ** 2))),
            "pressure_mae_pa": float(np.mean(np.abs(error))),
            "pmax_abs_mean_pa": float(np.mean(np.abs(peak_error))),
            "pmax_bias_pa": float(np.mean(peak_error)),
            "pmax_abs_p95_pa": float(np.percentile(np.abs(peak_error), 95))}
