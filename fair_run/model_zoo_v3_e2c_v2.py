"""
model_zoo_v3_e2c_v2.py -- E2C rebuilt to match the real reference
implementation (jungangc/CCS_E2CO-RL, MSE2C.py / MSE2C_layers.py).

1. Orthogonal weight init on every layer (encoder, decoder, transition).
2. Encoder/decoder keep a SPATIAL bottleneck (conv downsample -> residual
   conv blocks AT that resolution -> THEN flatten).
3. Transition is dt-aware: dt concatenated into the transition's input,
   control scaled by dt before entering B.

KNOWN ISSUE: orthogonal init on the hypernetwork layers that PRODUCE the
A/B matrices does not make the produced A matrix itself orthogonal --
untrained latent norm still decays fast (2.3 -> 0.36 -> 0.05 over 5
steps). This may not matter -- the real test is whether TRAINING
(prediction + autoencoding loss on real trajectory data) fixes the
dynamics, not whether an untrained net preserves norm. Don't gate further
work on the untrained-norm number; run the real training test instead.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def orthogonal_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        identity = x
        a = F.relu(self.bn1(self.conv1(x)))
        a = self.bn2(self.conv2(a))
        return identity + a


class E2CEncoderV2(nn.Module):
    def __init__(self, in_ch, H, W, latent_dim, base=16):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch, base, stride=2),
            ConvBNReLU(base, base * 2, stride=1),
            ConvBNReLU(base * 2, base * 4, stride=2),
            ConvBNReLU(base * 4, base * 8, stride=1))
        self.res = nn.Sequential(
            ResidualConvBlock(base * 8), ResidualConvBlock(base * 8), ResidualConvBlock(base * 8))
        with torch.no_grad():
            dummy = self.res(self.conv(torch.zeros(1, in_ch, H, W)))
        self.flat_shape = tuple(dummy.shape[1:])
        self.fc = nn.Linear(dummy.numel(), latent_dim)
        self.apply(orthogonal_init)

    def forward(self, x):
        h = self.res(self.conv(x))
        return self.fc(h.flatten(1))


class E2CDecoderV2(nn.Module):
    def __init__(self, out_ch, flat_shape, latent_dim, base=16):
        super().__init__()
        self.flat_shape = flat_shape
        n_flat = 1
        for d in flat_shape:
            n_flat *= d
        self.fc = nn.Linear(latent_dim, n_flat)
        self.res = nn.Sequential(
            ResidualConvBlock(flat_shape[0]), ResidualConvBlock(flat_shape[0]), ResidualConvBlock(flat_shape[0]))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(flat_shape[0], base * 4, 3, stride=1, padding=1), nn.BatchNorm2d(base * 4), nn.ReLU(),
            nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2), nn.BatchNorm2d(base * 2), nn.ReLU(),
            nn.ConvTranspose2d(base * 2, base, 3, stride=1, padding=1), nn.BatchNorm2d(base), nn.ReLU(),
            nn.ConvTranspose2d(base, base, 2, stride=2), nn.BatchNorm2d(base), nn.ReLU(),
            nn.Conv2d(base, out_ch, 3, padding=1))
        self.apply(orthogonal_init)

    def forward(self, z, target_size):
        h = F.relu(self.fc(z)).view(-1, *self.flat_shape)
        h = self.res(h)
        out = self.deconv(h)
        if out.shape[-2:] != tuple(target_size):
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        return out


class DtAwareTransition(nn.Module):
    def __init__(self, latent_dim, control_dim, hidden=128):
        super().__init__()
        self.latent_dim, self.control_dim = latent_dim, control_dim
        self.trans_encoder = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU())
        self.A_layer = nn.Linear(hidden, latent_dim * latent_dim)
        self.B_layer = nn.Linear(hidden, latent_dim * control_dim)
        self.apply(orthogonal_init)

    def forward(self, z, u, dt):
        B = z.shape[0]
        hz = self.trans_encoder(torch.cat([z, dt], dim=-1))
        A = self.A_layer(hz).view(B, self.latent_dim, self.latent_dim)
        Bm = self.B_layer(hz).view(B, self.latent_dim, self.control_dim)
        u_dt = u * dt
        return torch.bmm(A, z.unsqueeze(-1)).squeeze(-1) + torch.bmm(Bm, u_dt.unsqueeze(-1)).squeeze(-1)


class StaticContextEncoderV2(nn.Module):
    def __init__(self, in_ch=3, context_dim=16, width=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1))
        self.proj = nn.Linear(width * 2, context_dim)

    def forward(self, x):
        return self.proj(self.net(x).flatten(1))


class E2CTrajectoryV2(nn.Module):
    def __init__(self, H=32, W=32, latent_dim=64, context_dim=16, static_ch=3, base=16):
        super().__init__()
        self.H, self.W = H, W
        self.context_encoder = StaticContextEncoderV2(static_ch, context_dim)
        self.encoder = E2CEncoderV2(2, H, W, latent_dim, base=base)
        self.decoder = E2CDecoderV2(2, self.encoder.flat_shape, latent_dim, base=base)
        self.transition = DtAwareTransition(latent_dim, context_dim)
        self.n_steps = 1

    def encode(self, state):
        return self.encoder(state)

    def decode(self, z):
        from model_zoo_v3 import split_output
        return split_output(self.decoder(z, (self.H, self.W)))

    def set_steps(self, n_steps):
        self.n_steps = n_steps
        return self

    def forward_trajectory(self, state0, static, dt_value=1.0):
        context = self.context_encoder(static)
        B = state0.shape[0]
        dt = torch.full((B, 1), dt_value, device=state0.device, dtype=state0.dtype)
        z = self.encode(state0)
        zs = [z]
        for _ in range(self.n_steps):
            z = self.transition(z, context, dt)
            zs.append(z)
        return zs
