"""
stage_surrogate.py -- learned one-stage transition model with the SAME interface as
stage_model.SolverStageModel, so game3 can swap it in without any change:

    model = StageSurrogate("stage_ckpt.pt")          # or StageSurrogate(net=..., stage_time=1.0)
    state = model.initial_state(n)                   # {"p": zeros, "Sw": Swc}
    state, elapsed = model.advance(geo, state, wells) # wells = [(i, j, rate), ...], all injectors

Design (see SURROGATE_HANDOFF.md sections 6-8):
  * input channels: log-permeability, porosity, rate-weighted well map, p_in, Sw_in
    (fixed, data-independent scalings, so a checkpoint needs no dataset statistics);
  * the network predicts the CHANGE of the state (dp, dSw): p_out = p_in + dp,
    Sw_out = clip(Sw_in + dSw, Swc, 1 - Sor), the solver's own saturation bounds;
  * units stay solver units; elapsed time returned is exactly stage_time.

The backbone is a small U-Net ("unet"), a CCSNet-style two-decoder U-Net ("ccs"), a grid-graph message-passing
network with permeability edge features and separate pressure/saturation heads ("gnn"), an HGNS-style split
(U-Net for pressure, GNN for saturation: "split"), a U-Net with a ConvLSTM bottleneck ("runet") or a small FNO ("fno").

Hybrid modes (physics_prior.py = the solver's own implicit pressure step, batched in torch):
  hybrid="none"    pure learned model (above).
  hybrid="prior"   p_a = physics pressure advance with lambda frozen at Sw_in is fed to the net as
                   an extra channel; the net predicts the RESIDUAL:  p_out = p_a + dp.
  hybrid="prior2"  as "prior", then the net's own Sw_out is used to re-solve pressure with lambda
                   following Sw_in -> Sw_out:  p_out = p_b(Sw_out_pred) + dp.  Differentiable, so it
                   trains end to end; pressure then depends on the true k/phi geometry through
                   physics and on the learned saturation only through the mobility.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from physics_prior import pressure_prior

SWC, SOR = 0.2, 0.2
# fixed input scalings (benchmark k in [0.02, 0.2]; game k = that x mult in [0.5, 2] -> [0.01, 0.4])
LOGK_MEAN, LOGK_STD = float(np.log(0.07)), 0.9
PHI_MEAN, PHI_STD = 0.165, 0.05
P_SCALE = 6.0            # the game's pressure cap; p is fed and predicted in units of this
DP_SCALE = 3.0           # typical per-stage pressure change, in units of P_SCALE * 1
DS_SCALE = 0.3
IN_CH = 5
RES_SCALE = 0.5          # typical size of the residual on top of the physics prior (hybrid modes)


def encode(k, phi, well_rate, p_in, sw_in):
    """All args are tensors [B,H,W]; returns [B,5,H,W]."""
    return torch.stack([(torch.log(k.clamp_min(1e-6)) - LOGK_MEAN) / LOGK_STD,
                        (phi - PHI_MEAN) / PHI_STD,
                        well_rate,
                        p_in / P_SCALE,
                        (sw_in - SWC) / (1 - SOR - SWC)], dim=1)


def _block(ic, oc, stride=1, norm=True):
    layers = [nn.Conv2d(ic, oc, 3, stride=stride, padding=1)]
    if norm:
        layers.append(nn.GroupNorm(min(8, oc), oc))
    layers.append(nn.GELU())
    return nn.Sequential(*layers)


class _Decoder(nn.Module):
    def __init__(self, c, out_ch):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1)
        self.m1 = _block(c * 4, c * 2)
        self.up2 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)
        self.m2 = _block(c * 2, c)
        self.out = nn.Conv2d(c, out_ch, 1)

    def forward(self, f0, f1, f2):
        h = self.up1(f2)
        h = F.interpolate(h, size=f1.shape[-2:], mode="bilinear", align_corners=False) if h.shape[-2:] != f1.shape[-2:] else h
        h = self.m1(torch.cat([h, f1], 1))
        h = self.up2(h)
        h = F.interpolate(h, size=f0.shape[-2:], mode="bilinear", align_corners=False) if h.shape[-2:] != f0.shape[-2:] else h
        return self.out(self.m2(torch.cat([h, f0], 1)))


BACKBONES = ("unet", "ccs", "gnn", "split", "runet", "fno", "ufno")

_DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))        # neighbour offsets (dy, dx) of the 4-neighbour grid graph


def _shift(t, dy, dx):
    """out[..., i, j] = t[..., i+dy, j+dx], zero outside the grid."""
    H, W = t.shape[-2:]
    tp = F.pad(t, (1, 1, 1, 1))
    return tp[..., 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]


class _UNet(nn.Module):
    """Single-decoder U-Net, n_out channels (same blocks as the 'unet' backbone)."""

    def __init__(self, in_ch, c, n_out):
        super().__init__()
        self.e0 = nn.Sequential(_block(in_ch, c), _block(c, c))
        self.e1 = nn.Sequential(_block(c, c * 2, stride=2), _block(c * 2, c * 2))
        self.e2 = nn.Sequential(_block(c * 2, c * 4, stride=2), _block(c * 4, c * 4))
        self.dec = _Decoder(c, n_out)

    def forward(self, x):
        f0 = self.e0(x)
        f1 = self.e1(f0)
        return self.dec(f0, f1, self.e2(f1))


class _GridGNN(nn.Module):
    """Message passing on the 4-neighbour grid graph (nodes = cells, MeshGraphNet-style).
    Edge features come from the permeability field -- harmonic-mean k across the face and the log-k jump,
    i.e. what the physics' transmissibility depends on; one message MLP per direction (gravity acts along j);
    domain boundaries have no edges. A global-mean context is added in every layer so the receptive field is
    the whole grid (pressure responds globally). One head per output = separate pressure / saturation
    decoders (the CCS idea). Returns a tuple of [B,H,W] tensors."""

    def __init__(self, in_ch, c, n_layers=6, n_out=2):
        super().__init__()
        self.n_layers = n_layers
        self.enc = nn.Sequential(nn.Conv2d(in_ch, c, 1), nn.GELU(), nn.Conv2d(c, c, 1))
        self.msg = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Conv2d(2 * c + 2, c, 1), nn.GELU(), nn.Conv2d(c, c, 1))
                                                 for _ in _DIRS]) for _ in range(n_layers)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Conv2d(2 * c, c, 1), nn.GELU(), nn.Conv2d(c, c, 1)) for _ in range(n_layers)])
        self.glob = nn.ModuleList([nn.Conv2d(c, c, 1) for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.GroupNorm(min(8, c), c) for _ in range(n_layers)])
        self.heads = nn.ModuleList([nn.Sequential(nn.Conv2d(c, c, 1), nn.GELU(), nn.Conv2d(c, 1, 1)) for _ in range(n_out)])

    @staticmethod
    def edge_features(x):
        lk = x[:, 0:1]                                           # normalised log k
        k = torch.exp(lk * LOGK_STD + LOGK_MEAN)
        ones = torch.ones(1, 1, *x.shape[-2:], device=x.device, dtype=x.dtype)
        out = []
        for dy, dx in _DIRS:
            m = _shift(ones, dy, dx)                             # 0 where the neighbour is outside the grid
            kn = _shift(k, dy, dx)
            kh = 2 * k * kn / (k + kn + 1e-9)
            f1 = (torch.log(kh + 1e-9) - LOGK_MEAN) / LOGK_STD
            f2 = _shift(lk, dy, dx) - lk
            out.append((torch.cat([f1, f2], 1) * m, m))
        return out

    def forward(self, x):
        edges = self.edge_features(x)
        h = self.enc(x)
        for l in range(self.n_layers):
            agg = 0
            for d, (dy, dx) in enumerate(_DIRS):
                e, m = edges[d]
                agg = agg + self.msg[l][d](torch.cat([h, _shift(h, dy, dx), e], 1)) * m
            h = self.norm[l](h + self.upd[l](torch.cat([h, agg], 1)) + self.glob[l](h.mean((2, 3), keepdim=True)))
        return tuple(hd(h)[:, 0] for hd in self.heads)


class _Split(nn.Module):
    """HGNS-style division of labour (Wu et al., hybrid graph network simulator): a U-Net for the global
    pressure change, a graph network for the local saturation change."""

    def __init__(self, in_ch, c):
        super().__init__()
        self.pnet = _UNet(in_ch, c, 1)
        self.snet = _GridGNN(in_ch, c, n_out=1)

    def forward(self, x):
        return self.pnet(x)[:, 0], self.snet(x)[0]


class _ConvLSTMCell(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(2 * ch, 4 * ch, 3, padding=1)

    def forward(self, x, h, c):
        i, f, g, o = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class _RUNet(nn.Module):
    """R-U-Net flavour: U-Net whose bottleneck is refined by a ConvLSTM unrolled n_rec times within the stage
    (the hidden state is reset every stage: the game state (p, Sw) is already Markov)."""

    def __init__(self, in_ch, c, n_rec=4):
        super().__init__()
        self.n_rec = n_rec
        self.e0 = nn.Sequential(_block(in_ch, c), _block(c, c))
        self.e1 = nn.Sequential(_block(c, c * 2, stride=2), _block(c * 2, c * 2))
        self.e2 = nn.Sequential(_block(c * 2, c * 4, stride=2), _block(c * 4, c * 4))
        self.proj, self.cell, self.back = nn.Conv2d(c * 4, c * 2, 1), _ConvLSTMCell(c * 2), nn.Conv2d(c * 2, c * 4, 1)
        self.dec = _Decoder(c, 2)

    def forward(self, x):
        f0 = self.e0(x)
        f1 = self.e1(f0)
        f2 = self.e2(f1)
        z = self.proj(f2)
        h, cs = torch.zeros_like(z), torch.zeros_like(z)
        for _ in range(self.n_rec):
            h, cs = self.cell(z, h, cs)
        o = self.dec(f0, f1, f2 + self.back(h))
        return o[:, 0], o[:, 1]


class _SpectralConv2d(nn.Module):
    def __init__(self, ci, co, modes):
        super().__init__()
        s = 1.0 / (ci * co)
        self.m = modes
        self.w1 = nn.Parameter(s * torch.randn(ci, co, modes, modes, 2))     # real storage of complex weights
        self.w2 = nn.Parameter(s * torch.randn(ci, co, modes, modes, 2))

    def forward(self, x):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x)
        m1, m2 = min(self.m, H // 2), min(self.m, W // 2 + 1)
        out = torch.zeros(B, self.w1.shape[1], H, W // 2 + 1, dtype=xf.dtype, device=x.device)
        w1, w2 = torch.view_as_complex(self.w1), torch.view_as_complex(self.w2)
        out[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m1, :m2], w1[:, :, :m1, :m2])
        out[:, :, H - m1:, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, H - m1:, :m2], w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out, s=(H, W))


class _UNetPath(nn.Module):
    """Small conv encoder-decoder used as the U-Net path of a U-FNO layer (one down/up level)."""

    def __init__(self, c):
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(c, c, 3, stride=2, padding=1), nn.GELU(), nn.Conv2d(c, c, 3, padding=1), nn.GELU())
        self.up = nn.ConvTranspose2d(c, c, 4, stride=2, padding=1)
        self.out = nn.Conv2d(2 * c, c, 3, padding=1)

    def forward(self, x):
        h = self.up(self.down(x))
        if h.shape[-2:] != x.shape[-2:]:
            h = F.interpolate(h, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(torch.cat([h, x], 1))


class _FNO(nn.Module):
    """Small Fourier neural operator, two output heads. u_layers > 0 makes the LAST u_layers layers U-FNO layers
    (Fourier path + pointwise path + a U-Net path, as in U-FNO / CCSNet's building block)."""

    def __init__(self, in_ch, c, modes=6, n_layers=4, u_layers=0):
        super().__init__()
        self.lift = nn.Conv2d(in_ch + 2, c, 1)
        self.n_layers, self.u_layers = n_layers, u_layers
        self.spec = nn.ModuleList([_SpectralConv2d(c, c, modes) for _ in range(n_layers)])
        self.pw = nn.ModuleList([nn.Conv2d(c, c, 1) for _ in range(n_layers)])
        self.unet = nn.ModuleList([_UNetPath(c) for _ in range(u_layers)])
        self.heads = nn.ModuleList([nn.Sequential(nn.Conv2d(c, c, 1), nn.GELU(), nn.Conv2d(c, 1, 1)) for _ in range(2)])

    def forward(self, x):
        B, _, H, W = x.shape
        gy = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        gx = torch.linspace(0, 1, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        h = self.lift(torch.cat([x, gy, gx], 1))
        for l, (s, p) in enumerate(zip(self.spec, self.pw)):
            z = s(h) + p(h)
            if l >= self.n_layers - self.u_layers:
                z = z + self.unet[l - (self.n_layers - self.u_layers)](h)
            h = F.gelu(z) if l < self.n_layers - 1 else z
        return tuple(hd(h)[:, 0] for hd in self.heads)


def _make_core(backbone, in_ch, c):
    return {"gnn": lambda: _GridGNN(in_ch, c, n_out=2), "split": lambda: _Split(in_ch, c),
            "runet": lambda: _RUNet(in_ch, c), "fno": lambda: _FNO(in_ch, c),
            "ufno": lambda: _FNO(in_ch, c, u_layers=2)}[backbone]()


class StageNet(nn.Module):
    """x:[B,5,H,W] -> (dp, dSw) each [B,H,W], in scaled units."""

    def __init__(self, backbone="unet", width=48, hybrid="none", n_sub=8, stage_time=1.0):
        super().__init__()
        assert hybrid in ("none", "prior", "prior2")
        assert backbone in BACKBONES, f"backbone must be one of {BACKBONES}"
        self.backbone, self.hybrid, self.n_sub, self.stage_time = backbone, hybrid, n_sub, stage_time
        c = width
        in_ch = IN_CH + (hybrid != "none")
        if backbone in ("unet", "ccs"):
            self.e0 = nn.Sequential(_block(in_ch, c), _block(c, c))
            self.e1 = nn.Sequential(_block(c, c * 2, stride=2), _block(c * 2, c * 2))
            self.e2 = nn.Sequential(_block(c * 2, c * 4, stride=2), _block(c * 4, c * 4))
            if backbone == "ccs":
                self.dp = _Decoder(c, 1)
                self.ds = _Decoder(c, 1)
            else:
                self.dec = _Decoder(c, 2)
        else:
            self.core = _make_core(backbone, in_ch, c)

    def forward(self, x):
        if hasattr(self, "core"):
            return self.core(x)
        f0 = self.e0(x)
        f1 = self.e1(f0)
        f2 = self.e2(f1)
        if self.backbone == "ccs":
            dp, ds = self.dp(f0, f1, f2)[:, 0], self.ds(f0, f1, f2)[:, 0]
        else:
            o = self.dec(f0, f1, f2)
            dp, ds = o[:, 0], o[:, 1]
        return dp, ds

    @classmethod
    def from_blob(cls, blob):
        return cls(blob.get("backbone", "unet"), blob.get("width", 48), blob.get("hybrid", "none"),
                   blob.get("n_sub", 8), blob.get("stage_time", 1.0))

    def step(self, k, phi, well_rate, p_in, sw_in):
        """Physical-units one-step prediction: returns (p_out, sw_out) tensors [B,H,W]."""
        if self.hybrid != "none" and k.shape[0] > 256:      # keep the dense [B,144,144] systems small
            outs = [self.step(*(a[i:i + 256] for a in (k, phi, well_rate, p_in, sw_in))) for i in range(0, k.shape[0], 256)]
            return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])
        x = encode(k, phi, well_rate, p_in, sw_in)
        if self.hybrid == "none":
            dp, ds = self(x)
            return p_in + dp * DP_SCALE, (sw_in + ds * DS_SCALE).clamp(SWC, 1 - SOR)
        with torch.no_grad():                 # the frozen-lambda prior is a function of the inputs only
            p_a = pressure_prior(k, phi, sw_in, well_rate, p_in, self.stage_time, self.n_sub, dtype=torch.float32)
        dp, ds = self(torch.cat([x, ((p_a - p_in) / DP_SCALE).unsqueeze(1)], 1))
        sw_out = (sw_in + ds * DS_SCALE).clamp(SWC, 1 - SOR)
        base = p_a if self.hybrid == "prior" else pressure_prior(k, phi, sw_in, well_rate, p_in, self.stage_time,
                                                                 self.n_sub, sw_end=sw_out, dtype=torch.float32)
        return base + dp * RES_SCALE, sw_out


class StageSurrogate:
    """Drop-in replacement for stage_model.SolverStageModel."""

    def __init__(self, ckpt=None, net=None, stage_time=1.0, device="cpu"):
        if net is None:
            blob = torch.load(ckpt, map_location=device)
            net = StageNet.from_blob(blob)
            net.load_state_dict(blob["state_dict"])
            stage_time = blob.get("stage_time", stage_time)
        self.net = net.to(device).eval()
        self.stage_time = stage_time
        self.device = device

    def initial_state(self, n, Swc=SWC):
        return {"p": np.zeros((n, n)), "Sw": np.full((n, n), Swc)}

    @torch.no_grad()
    def advance(self, geo, state, wells):
        """wells: [(i, j, rate), ...] (all injectors). Returns ({'p','Sw'}, elapsed)."""
        n = geo["k"].shape[0]
        wr = np.zeros((n, n), np.float32)
        for i, j, r in wells:
            wr[i, j] += r
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)[None]
        p, sw = self.net.step(t(geo["k"]), t(geo["phi"]), t(wr), t(state["p"]), t(state["Sw"]))
        return {"p": p[0].double().cpu().numpy(), "Sw": sw[0].double().cpu().numpy()}, self.stage_time
