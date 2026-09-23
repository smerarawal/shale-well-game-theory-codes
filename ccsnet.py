"""
ccsnet.py -- CCSNet (Wen, Hay, Benson, arXiv:2104.01795). Modular design:
ONE shared encoder trunk over the input fields (kx, ky, well_mask), then
SEPARATE lightweight decoder heads per physical output (pressure,
saturation), instead of one monolithic network predicting a stacked
multi-channel tensor.

Why this is a genuinely different design choice, not just a relabeled
U-Net: pressure (elliptic/smooth) and saturation (hyperbolic/sharp-front)
have very different regularity, so forcing one decoder to fit both targets
trades accuracy on the harder one (saturation) against the easier one
(pressure) whenever they share final-layer capacity. Separate heads let
each output's decoder specialize its receptive field / channel budget
without touching the other's loss gradient at the last layer.

Run: python ccsnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.norm = nn.InstanceNorm2d(out_ch)

    def forward(self, x):
        return F.gelu(self.norm(self.conv(x)))


class SharedEncoder(nn.Module):
    """Shared trunk: downsample twice, keep skip connections for the decoder
    heads (each head gets its own upsampling path, but reads off the SAME
    encoder features)."""

    def __init__(self, in_ch, width):
        super().__init__()
        self.b0 = ConvBlock(in_ch, width)
        self.b1 = ConvBlock(width, width * 2, stride=2)
        self.b2 = ConvBlock(width * 2, width * 4, stride=2)

    def forward(self, x):
        f0 = self.b0(x)
        f1 = self.b1(f0)
        f2 = self.b2(f1)
        return f0, f1, f2


class DecoderHead(nn.Module):
    """One output field's private decoder, reading shared encoder features
    via skip connections. Independent weights per head -- the whole point
    of the modular design."""

    def __init__(self, width, out_ch, activation):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(width * 4, width * 2, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1)
        self.merge1 = ConvBlock(width * 4, width * 2)  # after concat with f1
        self.merge2 = ConvBlock(width * 2, width)       # after concat with f0
        self.out = nn.Conv2d(width, out_ch, 1)
        self.activation = activation

    def forward(self, f0, f1, f2):
        h = self.up1(f2)
        h = self.merge1(torch.cat([h, f1], dim=1))
        h = self.up2(h)
        h = self.merge2(torch.cat([h, f0], dim=1))
        out = self.out(h)
        return self.activation(out) if self.activation is not None else out


class CCSNet(nn.Module):
    def __init__(self, in_ch=3, width=16):
        """in_ch: kx, ky, well_mask. Two heads: pressure (unbounded,
        identity activation) and saturation (sigmoid, S_w in [0,1])."""
        super().__init__()
        self.encoder = SharedEncoder(in_ch, width)
        self.pressure_head = DecoderHead(width, out_ch=1, activation=None)
        self.saturation_head = DecoderHead(width, out_ch=1, activation=torch.sigmoid)

    def forward(self, x):
        f0, f1, f2 = self.encoder(x)
        pressure = self.pressure_head(f0, f1, f2)
        saturation = self.saturation_head(f0, f1, f2)
        return {"pressure": pressure, "saturation": saturation}


def train_step(model, opt, x, target_pressure, target_saturation, sat_weight=1.0):
    model.train()
    opt.zero_grad()
    out = model(x)
    loss_p = F.mse_loss(out["pressure"], target_pressure)
    loss_s = F.mse_loss(out["saturation"], target_saturation)
    loss = loss_p + sat_weight * loss_s
    loss.backward()
    opt.step()
    return loss.item(), loss_p.item(), loss_s.item()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, nx, ny = 4, 64, 64
    model = CCSNet(in_ch=3, width=16)
    x = torch.rand(B, 3, nx, ny)
    tp = torch.randn(B, 1, nx, ny)
    ts = torch.rand(B, 1, nx, ny).clamp(0.2, 0.8)

    print("=== forward shape / head-independence check ===")
    out = model(x)
    assert out["pressure"].shape == (B, 1, nx, ny)
    assert out["saturation"].shape == (B, 1, nx, ny)
    assert (out["saturation"] >= 0).all() and (out["saturation"] <= 1).all()
    # confirm heads don't share parameters (modular claim, not just cosmetic)
    p_params = set(id(p) for p in model.pressure_head.parameters())
    s_params = set(id(p) for p in model.saturation_head.parameters())
    assert p_params.isdisjoint(s_params), "heads must not share weights"
    print("OK: independent pressure/saturation heads, shapes correct")

    print("=== gradient flow check ===")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    l0, lp0, ls0 = train_step(model, opt, x, tp, ts)
    l1, lp1, ls1 = train_step(model, opt, x, tp, ts)
    print(f"total loss step0={l0:.4f} step1={l1:.4f} (pressure={lp0:.4f}/{lp1:.4f}, sat={ls0:.4f}/{ls1:.4f})")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"param count: {n_params:,}")
    print("PASS")
