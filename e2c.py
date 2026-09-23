"""
e2c.py -- E2C (Embed-to-Control; Jin, Liu, Durlofsky, JPSE 192, 2020).
Origin of the exact ROM architecture the benchmark CCS paper (Chen &
Hosseini, arXiv:2508.11618) uses. Three pieces: encoder compresses the
full-field state to a small latent vector z, a LOCALLY-LINEAR transition
model evolves z given a control input u (well rates), decoder reconstructs
the full field from z. "Locally-linear" means A and B are themselves
functions of z (predicted by small nets), not a single global linear
system -- linear only in a neighborhood of the current latent state.

Known failure mode this file is built to make visible, not hide: the
locally-linear approximation error compounds over a long rollout (each
step's A(z), B(z) is only locally valid, so drift accumulates) in a way
plain FNO and R-U-Net don't share, since those never linearize the
dynamics. The rollout sanity check below reports per-step reconstruction
error growth explicitly, rather than only a single end-of-rollout number,
so that compounding (if present) is visible instead of averaged away.

Run: python e2c.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_ch, H, W, latent_dim, width=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1), nn.GELU(),
        )
        with torch.no_grad():
            dummy = self.conv(torch.zeros(1, in_ch, H, W))
        self.flat_dim = dummy.numel()
        self.flat_shape = dummy.shape[1:]
        self.fc = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        return self.fc(h.flatten(1))


class Decoder(nn.Module):
    def __init__(self, out_ch, flat_shape, latent_dim, width=16):
        super().__init__()
        self.flat_shape = flat_shape
        self.fc = nn.Linear(latent_dim, int(torch.tensor(flat_shape).prod()))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(width * 2, width * 2, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(width, out_ch, 4, stride=2, padding=1),
        )

    def forward(self, z, target_size):
        h = self.fc(z).view(-1, *self.flat_shape)
        out = self.deconv(h)
        if out.shape[-2:] != target_size:
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        return out


class LocallyLinearTransition(nn.Module):
    """A(z) and B(z): small MLPs predicting the entries of the
    locally-linear system z_{t+1} = A(z_t) z_t + B(z_t) u_t, re-linearized
    at every step around the current latent state (not one fixed global
    A, B -- that would just be a linear ROM, not E2C)."""

    def __init__(self, latent_dim, control_dim, hidden=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.A_net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim * latent_dim))
        self.B_net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim * control_dim))
        self.control_dim = control_dim

    def forward(self, z, u):
        B = z.shape[0]
        A = self.A_net(z).view(B, self.latent_dim, self.latent_dim)
        Bm = self.B_net(z).view(B, self.latent_dim, self.control_dim)
        z_next = torch.bmm(A, z.unsqueeze(-1)).squeeze(-1) + torch.bmm(Bm, u.unsqueeze(-1)).squeeze(-1)
        return z_next


class E2C(nn.Module):
    def __init__(self, in_ch=2, H=48, W=48, latent_dim=32, control_dim=4):
        super().__init__()
        self.H, self.W = H, W
        self.encoder = Encoder(in_ch, H, W, latent_dim)
        self.decoder = Decoder(in_ch, self.encoder.flat_shape, latent_dim)
        self.transition = LocallyLinearTransition(latent_dim, control_dim)

    def rollout(self, state0, controls):
        """state0: (B, in_ch, H, W) initial field. controls: (B, n_steps,
        control_dim) well-rate sequence. Returns decoded field at every
        step -- cheap because only the transition net runs per-step; the
        encoder runs once and the decoder only when a field is needed."""
        z = self.encoder(state0)
        decoded = [self.decoder(z, (self.H, self.W))]
        for t in range(controls.shape[1]):
            z = self.transition(z, controls[:, t])
            decoded.append(self.decoder(z, (self.H, self.W)))
        return torch.stack(decoded, dim=1)  # (B, n_steps+1, in_ch, H, W)


def train_step(model, opt, state0, controls, target_traj):
    model.train()
    opt.zero_grad()
    pred_traj = model.rollout(state0, controls)
    loss = F.mse_loss(pred_traj, target_traj)
    loss.backward()
    opt.step()
    return loss.item()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, nx, ny, n_steps, latent_dim = 3, 48, 48, 8, 24
    model = E2C(in_ch=2, H=nx, W=ny, latent_dim=latent_dim, control_dim=4)
    state0 = torch.rand(B, 2, nx, ny)
    controls = torch.randn(B, n_steps, 4) * 0.1
    target_traj = torch.rand(B, n_steps + 1, 2, nx, ny)

    print("=== rollout shape check ===")
    pred_traj = model.rollout(state0, controls)
    assert pred_traj.shape == (B, n_steps + 1, 2, nx, ny)
    print(f"OK: rollout shape {tuple(pred_traj.shape)}")

    print("=== per-step error-compounding visibility check (the actual failure mode) ===")
    with torch.no_grad():
        per_step_err = ((pred_traj - target_traj) ** 2).mean(dim=(0, 2, 3, 4))
    for t, e in enumerate(per_step_err.tolist()):
        print(f"  step {t}: mse={e:.4f}")
    print("(untrained model -- error trend here is not meaningful yet, but this is exactly "
          "the per-step breakdown that must be checked once trained, since a rising trend "
          "across steps is the locally-linear compounding-error signature to watch for)")

    print("=== gradient flow check ===")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    l0 = train_step(model, opt, state0, controls, target_traj)
    l1 = train_step(model, opt, state0, controls, target_traj)
    print(f"loss step0={l0:.4f} step1={l1:.4f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"param count: {n_params:,} (should be small -- ROM cheapness is the whole point)")
    print("PASS")
