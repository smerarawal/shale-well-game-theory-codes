"""
runet_recurrent.py -- Recurrent R-U-Net (Tang, Liu, Durlofsky, J. Comput.
Phys. 413, 2020; extended to 3D CO2 flow-geomechanics in Tang, Ju,
Durlofsky 2022). ConvLSTM + U-Net: predicts ONE step ahead at a time,
state (hidden + cell) carried forward across steps, instead of the
trajectory FNO's one-shot fixed-horizon rollout.

Why this matters for the MARL extension specifically: trajectory FNO
takes a FIXED control sequence and emits the FULL trajectory in one
forward pass -- there is no point at which a new decision (a different
agent's well-rate choice at step t) can be inserted mid-rollout without
recomputing the whole thing from scratch. R-U-Net's state is carried
step-by-step, so a control change at step t only requires forward passes
from t onward, reusing everything before it. That's the actual
requirement for a MARL environment where each agent's action arrives
one control period at a time.

Run: python runet_recurrent.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        # single conv produces all 4 gates at once (input, forget, output, cell)
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=pad)
        self.hidden_ch = hidden_ch

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch, H, W, device):
        z = torch.zeros(batch, self.hidden_ch, H, W, device=device)
        return z, z.clone()


class UNetDecode(nn.Module):
    """Thin U-Net head mapping ConvLSTM hidden state -> the next-step field
    (pressure and saturation). Local convs, since this is spatial
    refinement, not global mixing -- that's the ConvLSTM's job across time."""

    def __init__(self, hidden_ch, out_ch, width=32):
        super().__init__()
        self.down = nn.Conv2d(hidden_ch, width, 3, stride=2, padding=1)
        self.up = nn.ConvTranspose2d(width, width, 4, stride=2, padding=1)
        self.out = nn.Conv2d(width, out_ch, 1)

    def forward(self, h):
        d = F.gelu(self.down(h))
        u = F.gelu(self.up(d))
        if u.shape[-2:] != h.shape[-2:]:
            u = F.interpolate(u, size=h.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(u)


class RUNet(nn.Module):
    def __init__(self, static_ch=3, dynamic_ch=2, hidden_ch=32, out_ch=2):
        """static_ch: kx, ky, well_mask (constant across the rollout).
        dynamic_ch: pressure, saturation from the previous step (fed back
        in at every step). out_ch=2: predicted (pressure, saturation)."""
        super().__init__()
        self.cell = ConvLSTMCell(static_ch + dynamic_ch, hidden_ch)
        self.decode = UNetDecode(hidden_ch, out_ch)

    def forward_step(self, static, dynamic_prev, state):
        x = torch.cat([static, dynamic_prev], dim=1)
        h, c = self.cell(x, state)
        out = self.decode(h)
        pressure, saturation = out[:, 0:1], torch.sigmoid(out[:, 1:2])
        return torch.cat([pressure, saturation], dim=1), (h, c)

    def rollout(self, static, dynamic_init, n_steps):
        """Runs n_steps forward, state carried the whole way -- this is the
        capability the trajectory FNO structurally lacks: a caller can stop
        here, hand back control to a MARL loop, get a new action-dependent
        `static` (well_mask changed), and resume from (h, c) without
        redoing steps 0..t."""
        B, _, H, W = static.shape
        state = self.cell.init_state(B, H, W, static.device)
        dynamic = dynamic_init
        outputs = []
        for _ in range(n_steps):
            dynamic, state = self.forward_step(static, dynamic, state)
            outputs.append(dynamic)
        return torch.stack(outputs, dim=1), state  # (B, n_steps, 2, H, W)


def train_step(model, opt, static, dynamic_init, target_traj):
    """target_traj: (B, n_steps, 2, H, W), teacher-forced against the
    model's own free rollout (no ground-truth feedback each step -- tests
    whether errors compound over the horizon, the known failure mode this
    family needs validating against)."""
    model.train()
    opt.zero_grad()
    n_steps = target_traj.shape[1]
    pred_traj, _ = model.rollout(static, dynamic_init, n_steps)
    loss = F.mse_loss(pred_traj, target_traj)
    loss.backward()
    opt.step()
    return loss.item()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, nx, ny, n_steps = 3, 48, 48, 6
    model = RUNet(static_ch=3, dynamic_ch=2, hidden_ch=24, out_ch=2)
    static = torch.rand(B, 3, nx, ny)
    dynamic_init = torch.rand(B, 2, nx, ny)
    target_traj = torch.rand(B, n_steps, 2, nx, ny)

    print("=== rollout shape check ===")
    pred_traj, state = model.rollout(static, dynamic_init, n_steps)
    assert pred_traj.shape == (B, n_steps, 2, nx, ny)
    print(f"OK: rollout shape {tuple(pred_traj.shape)}")

    print("=== mid-rollout resume check (the actual capability claim) ===")
    half_traj, state_half = model.rollout(static, dynamic_init, n_steps // 2)
    # resume from state_half with a CHANGED static field (simulates a new
    # mid-episode well-rate decision) without recomputing steps 0..n/2
    new_static = static.clone()
    new_static[:, 2] *= 1.5  # well_mask changed
    dyn_at_half = half_traj[:, -1]
    resumed_out, _ = model.forward_step(new_static, dyn_at_half, state_half)
    assert resumed_out.shape == (B, 2, nx, ny)
    print("OK: resumed from mid-rollout state with a modified control input")

    print("=== gradient flow check ===")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    l0 = train_step(model, opt, static, dynamic_init, target_traj)
    l1 = train_step(model, opt, static, dynamic_init, target_traj)
    print(f"loss step0={l0:.4f} step1={l1:.4f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"param count: {n_params:,}")
    print("PASS")
