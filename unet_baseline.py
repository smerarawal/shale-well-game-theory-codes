"""
unet_baseline.py -- Step 2.2 of surrogate_comparison_spec.md.

Standard U-Net (Ronneberger et al. 2015 style): encoder downsamples via
strided pooling, bottleneck, decoder upsamples via transposed conv with
skip connections from the matching encoder level. This is the direct test
of "does FNO's Fourier/global-structure inductive bias actually buy
anything over a standard local-convolution architecture for this physics."

Sized for 32x32 inputs (2 downsampling levels -> 8x8 bottleneck; going
deeper on a 32x32 grid leaves too little spatial resolution to be
meaningful).

Run:
    pip install torch
    python unet_baseline.py
(standalone run does a forward-pass shape check only, no training data
needed -- see run_surrogate_comparison.py for actual training)
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 convs with ReLU, the standard U-Net building block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSurrogate(nn.Module):
    """
    in_channels: 2 for v1 (permeability, well_mask), 4 for v2
                 (kx, ky, porosity, well_mask)
    out_channels: 1 for pressure-only, 2 for pressure+saturation (v3)
    """
    def __init__(self, in_channels=2, out_channels=1, base_channels=32):
        super().__init__()
        c = base_channels

        self.enc1 = ConvBlock(in_channels, c)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(c, c * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(c * 2, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)   # c*4 in because of skip-connection concat
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)

        self.final = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.final(d1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = UNetSurrogate(in_channels=2, out_channels=1, base_channels=32)
    x = torch.randn(4, 2, 32, 32)
    y = model(x)
    print(f"input shape: {x.shape}")
    print(f"output shape: {y.shape}")
    assert y.shape == (4, 1, 32, 32), f"unexpected output shape {y.shape}"
    print(f"parameter count: {count_parameters(model):,}")
    print("OK -- forward pass shape check passed")
