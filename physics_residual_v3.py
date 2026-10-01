"""
physics_residual_v3.py -- pressure-equation residual for the
physics_informed_fno / pi_convlstm loss terms, reusing solver_v3.py's own
harmonic-mean face convention so the residual is computed the same way
the actual solver computes its fluxes (not an approximation of it).

Residual form: div(k * lambda_total(Sw) * grad(p)) + source ~= 0 (the
PRESSURE equation only -- a full saturation-transport residual needs the
Darcy velocity field, which isn't recoverable from a final-field-only
prediction, same limitation flagged in the earlier delivery's
pi_convlstm adapter). Still gives genuine physics signal on the pressure
channel, which is the channel most architectures were shown to weight
too heavily anyway -- a residual that further sharpens pressure accuracy
is a reasonable complement to the loss reweighting, not a workaround for
skipping the harder saturation residual.
"""
import torch
from solver_v3 import corey_relperm


def face_values_harmonic_torch(field):
    eps = 1e-10
    e = torch.zeros_like(field); w = torch.zeros_like(field)
    n = torch.zeros_like(field); s = torch.zeros_like(field)
    a, b = field[:, :-1, :], field[:, 1:, :]
    e[:, :-1, :] = 2.0 * a * b / (a + b + eps)
    w[:, 1:, :] = e[:, :-1, :]
    a, b = field[:, :, :-1], field[:, :, 1:]
    n[:, :, :-1] = 2.0 * a * b / (a + b + eps)
    s[:, :, 1:] = n[:, :, :-1]
    return e, w, n, s


def total_mobility_torch(Sw, Swc=0.2, Sor=0.2, krw_max=0.8, krn_max=1.0, nw=2, nn=2, mu_w=1.0, mu_n=5.0):
    Sw_eff = torch.clamp((Sw - Swc) / (1 - Swc - Sor), 0, 1)
    krw = krw_max * Sw_eff ** nw
    krn = krn_max * (1 - Sw_eff) ** nn
    return krw / mu_w + krn / mu_n


def pressure_residual(pred, x, dx=1.0):
    """pred: (B, 2, H, W) = [pressure, saturation]. x: (B, 3, H, W) =
    [permeability, porosity, well_mask]."""
    p = pred[:, 0]
    Sw = pred[:, 1]
    k = x[:, 0]
    well_mask = x[:, 2]
    lam_total = total_mobility_torch(Sw)
    k_e, k_w, k_n, k_s = face_values_harmonic_torch(k * lam_total)
    p_e = torch.zeros_like(p); p_e[:, :-1, :] = p[:, 1:, :]
    p_w = torch.zeros_like(p); p_w[:, 1:, :] = p[:, :-1, :]
    p_n = torch.zeros_like(p); p_n[:, :, :-1] = p[:, :, 1:]
    p_s = torch.zeros_like(p); p_s[:, :, 1:] = p[:, :, :-1]
    flux_e = k_e * (p_e - p); flux_w = k_w * (p - p_w)
    flux_n = k_n * (p_n - p); flux_s = k_s * (p - p_s)
    div_flux = (flux_e - flux_w + flux_n - flux_s) / dx ** 2
    return ((div_flux + well_mask) ** 2).mean()
