"""
pi_convlstm.py -- Physics-informed ConvLSTM (Chen, Gildin, Killough,
arXiv:2305.09056). Same author (Jungang Chen) as the benchmark CCS paper
this whole project is compared against. Trained via the flow-equation
residual (same idea as fno_physics_informed.py), applied to the recurrent
ConvLSTM architecture from runet_recurrent.py instead of the one-shot FNO
-- so the residual has to be evaluated at EVERY predicted step of the
rollout, not just once on a single output field.

Reuses runet_recurrent.py's ConvLSTMCell/UNetDecode/RUNet building blocks
directly rather than redefining them, and reuses the harmonic-mean flux
convention from fno_physics_informed.py's face_values_harmonic_torch so
the residual is computed the same way both places.

Run: python pi_convlstm.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from runet_recurrent import RUNet


def face_values_harmonic_torch(field):
    """Same convention as fno_physics_informed.py's version -- duplicated
    here (not imported) so this file stays standalone-runnable without
    also requiring fno_physics_informed.py's torch/solver_v2 coupling."""
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


def pressure_residual(pressure, kx, ky, well_mask, dx=1.0):
    """Same steady-form div(k grad p) + source residual as the plain
    PI-FNO, applied per-step here."""
    p = pressure.squeeze(1)
    kx_e, kx_w, _, _ = face_values_harmonic_torch(kx)
    _, _, ky_n, ky_s = face_values_harmonic_torch(ky)
    p_e = torch.zeros_like(p); p_e[:, :-1, :] = p[:, 1:, :]
    p_w = torch.zeros_like(p); p_w[:, 1:, :] = p[:, :-1, :]
    p_n = torch.zeros_like(p); p_n[:, :, :-1] = p[:, :, 1:]
    p_s = torch.zeros_like(p); p_s[:, :, 1:] = p[:, :, :-1]
    flux_e = kx_e * (p_e - p); flux_w = kx_w * (p - p_w)
    flux_n = ky_n * (p_n - p); flux_s = ky_s * (p - p_s)
    div_flux = (flux_e - flux_w + flux_n - flux_s) / dx ** 2
    return ((div_flux + well_mask) ** 2).mean()


def saturation_residual(sat_curr, sat_prev, u_total_x, u_total_y, phi, dt, dx=1.0):
    """Transport-equation residual: phi * dS/dt + div(f_w * u_total) = 0
    (source term omitted at well cells is left to well_mask upstream,
    matching how solver_v3.py separates pressure and saturation sources).
    f_w here uses the SAME saturation-dependent Corey fractional-flow
    convention as solver_v3.py (S_w-only proxy: f_w = S_w, i.e. straight
    upwind advection of a scalar -- swap in the real Corey f_w(S_w) if
    matching solver_v3.py's rel-perm curves exactly is needed for the
    validation-against-solver step)."""
    dSdt = (sat_curr - sat_prev) / dt
    f_w = sat_curr.squeeze(1)  # proxy fractional flow; see docstring
    flux_x = f_w * u_total_x
    flux_y = f_w * u_total_y
    div_x = torch.zeros_like(f_w); div_x[:, 1:-1, :] = (flux_x[:, 2:, :] - flux_x[:, :-2, :]) / (2 * dx)
    div_y = torch.zeros_like(f_w); div_y[:, :, 1:-1] = (flux_y[:, :, 2:] - flux_y[:, :, :-2]) / (2 * dx)
    residual = phi.squeeze(1) * dSdt.squeeze(1) + div_x + div_y
    return (residual ** 2).mean()


def combined_step_loss(pred, target, kx, ky, phi, well_mask, dynamic_prev,
                        u_total_x, u_total_y, dt, physics_weight):
    data_loss = F.mse_loss(pred, target)
    if physics_weight == 0.0:
        return data_loss, data_loss, torch.tensor(0.0)  # same explicit short-circuit as PI-FNO
    pressure_pred, sat_pred = pred[:, 0:1], pred[:, 1:2]
    sat_prev = dynamic_prev[:, 1:2]
    phys = pressure_residual(pressure_pred, kx, ky, well_mask) \
        + saturation_residual(sat_pred, sat_prev, u_total_x, u_total_y, phi, dt)
    total = data_loss + physics_weight * phys
    return total, data_loss, phys


def rollout_with_physics_loss(model, static, dynamic_init, target_traj, kx, ky, phi,
                               well_mask, u_total_x, u_total_y, dt, physics_weight):
    """Physics residual evaluated at EVERY step of the rollout, not once --
    the actual extra cost of applying physics-informed training to a
    recurrent model vs. a one-shot one."""
    B, _, H, W = static.shape
    state = model.cell.init_state(B, H, W, static.device)
    dynamic = dynamic_init
    total_loss = torch.tensor(0.0)
    total_data = torch.tensor(0.0)
    total_phys = torch.tensor(0.0)
    n_steps = target_traj.shape[1]
    for t in range(n_steps):
        dynamic, state = model.forward_step(static, dynamic, state)
        step_total, step_data, step_phys = combined_step_loss(
            dynamic, target_traj[:, t], kx, ky, phi, well_mask, dynamic,
            u_total_x, u_total_y, dt, physics_weight)
        total_loss = total_loss + step_total
        total_data = total_data + step_data
        total_phys = total_phys + step_phys
    return total_loss / n_steps, total_data / n_steps, total_phys / n_steps


if __name__ == "__main__":
    torch.manual_seed(0)
    B, nx, ny, n_steps = 3, 48, 48, 4
    model = RUNet(static_ch=3, dynamic_ch=2, hidden_ch=24, out_ch=2)
    static = torch.rand(B, 3, nx, ny)
    dynamic_init = torch.rand(B, 2, nx, ny)
    target_traj = torch.rand(B, n_steps, 2, nx, ny)
    kx = torch.rand(B, nx, ny) * 0.18 + 0.02
    ky = torch.rand(B, nx, ny) * 0.18 + 0.02
    phi = torch.rand(B, 1, nx, ny) * 0.1 + 0.15
    well_mask = torch.zeros(B, nx, ny); well_mask[:, 24, 24] = -0.8
    u_total_x = torch.randn(B, nx, ny) * 0.01
    u_total_y = torch.randn(B, nx, ny) * 0.01

    print("=== physics_weight=0 reduction check (REQUIRED, same as PI-FNO) ===")
    loss_w0, data_w0, phys_w0 = rollout_with_physics_loss(
        model, static, dynamic_init, target_traj, kx, ky, phi, well_mask,
        u_total_x, u_total_y, dt=0.1, physics_weight=0.0)
    assert phys_w0.item() == 0.0
    print(f"loss(weight=0)={loss_w0.item():.4f} == data-only loss: {data_w0.item():.4f}")
    print("PASS reduction check")

    print("=== nonzero weight + gradient flow check ===")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss_w1, data_w1, phys_w1 = rollout_with_physics_loss(
        model, static, dynamic_init, target_traj, kx, ky, phi, well_mask,
        u_total_x, u_total_y, dt=0.1, physics_weight=0.05)
    loss_w1.backward()
    opt.step()
    print(f"loss(weight=0.05)={loss_w1.item():.4f} data={data_w1.item():.4f} phys={phys_w1.item():.4f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"param count: {n_params:,}")
    print("PASS")
