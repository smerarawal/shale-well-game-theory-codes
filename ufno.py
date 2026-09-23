"""
ufno.py -- U-FNO (Wen, Li, Azizzadenesheli, Anandkumar, Benson, Adv. Water
Resources 163, 2022). Interleaves standard FNO spectral-conv layers with
U-Net conv blocks. Motivation per the paper: plain FNO over-smooths sharp
saturation fronts (Fourier truncation kills high-frequency shock content);
the U-Net blocks operating in physical space restore local sharpness that
the spectral path throws away.

Architecture: lifting -> [FNO layer] x n_fno_plain -> [FNO layer + U-Net
block, summed] x n_fno_unet -> projection. Matches the paper's design of
plain FNO layers early (global structure) and FNO+U-Net layers late
(local sharpening near the output).

Built for v3's saturation field, the first architecture in the spec that
genuinely requires v3 to exist (front-sharpening only matters once there
IS a front to sharpen -- v1/v2 pressure fields are smooth/diffusive).

Run: python ufno.py   (shape + gradient-flow sanity check, no data needed)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Standard FNO spectral conv: truncated 2D FFT, learned complex weights
    on the low modes only, inverse FFT. Same building block as fno_train.py."""

    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1, m2 = min(self.modes1, H), min(self.modes2, W // 2 + 1)
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(H, W))


class UNetBlock(nn.Module):
    """Small local conv block (the 'U' part is really just a down/up conv
    pair in the original U-FNO paper's per-layer block, not a full U-Net) --
    operates in physical space, so it can represent sharp local gradients
    that the truncated-mode spectral path cannot."""

    def __init__(self, ch):
        super().__init__()
        self.down = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        self.up = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
        self.norm = nn.InstanceNorm2d(ch)

    def forward(self, x):
        h = F.gelu(self.norm(self.down(x)))
        h = self.up(h)
        if h.shape[-2:] != x.shape[-2:]:
            h = F.interpolate(h, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return h


class FNOLayer(nn.Module):
    def __init__(self, ch, modes1, modes2, with_unet):
        super().__init__()
        self.spectral = SpectralConv2d(ch, ch, modes1, modes2)
        self.pointwise = nn.Conv2d(ch, ch, 1)
        self.unet = UNetBlock(ch) if with_unet else None

    def forward(self, x):
        out = self.spectral(x) + self.pointwise(x)
        if self.unet is not None:
            out = out + self.unet(x)
        return F.gelu(out)


class UFNO(nn.Module):
    def __init__(self, in_ch=4, out_ch=1, width=32, modes1=16, modes2=16,
                 n_fno_plain=2, n_fno_unet=2):
        """in_ch: kx, ky, well_mask, S_w_prev (previous saturation, for a
        one-step-ahead front-advection target). out_ch=1: S_w at next step."""
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        layers = [FNOLayer(width, modes1, modes2, with_unet=False) for _ in range(n_fno_plain)]
        layers += [FNOLayer(width, modes1, modes2, with_unet=True) for _ in range(n_fno_unet)]
        self.layers = nn.ModuleList(layers)
        self.proj1 = nn.Conv2d(width, width * 2, 1)
        self.proj2 = nn.Conv2d(width * 2, out_ch, 1)

    def forward(self, x):
        h = self.lift(x)
        for layer in self.layers:
            h = layer(h)
        h = F.gelu(self.proj1(h))
        return torch.sigmoid(self.proj2(h))  # S_w in [0, 1] by construction


def train_step(model, opt, x, target):
    model.train()
    opt.zero_grad()
    pred = model(x)
    loss = F.mse_loss(pred, target)
    loss.backward()
    opt.step()
    return loss.item()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, nx, ny = 4, 64, 64
    model = UFNO(in_ch=4, out_ch=1, width=24, modes1=12, modes2=12)
    x = torch.rand(B, 4, nx, ny)
    target = torch.rand(B, 1, nx, ny).clamp(0.2, 0.8)  # physical S_w range

    print("=== forward shape check ===")
    pred = model(x)
    assert pred.shape == (B, 1, nx, ny), pred.shape
    assert (pred >= 0).all() and (pred <= 1).all(), "S_w must stay in [0,1]"
    print(f"OK: output shape {tuple(pred.shape)}, range [{pred.min():.3f}, {pred.max():.3f}]")

    print("=== gradient flow check ===")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss0 = train_step(model, opt, x, target)
    loss1 = train_step(model, opt, x, target)
    print(f"loss step0={loss0:.4f} step1={loss1:.4f} (should move)")
    assert loss0 != loss1
    n_params = sum(p.numel() for p in model.parameters())
    print(f"param count: {n_params:,}")
    print("PASS")
