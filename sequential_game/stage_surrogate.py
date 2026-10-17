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

The backbone is a small U-Net ("unet") or a CCSNet-style two-decoder U-Net ("ccs").

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


class StageNet(nn.Module):
    """x:[B,5,H,W] -> (dp, dSw) each [B,H,W], in scaled units."""

    def __init__(self, backbone="unet", width=48, hybrid="none", n_sub=8, stage_time=1.0):
        super().__init__()
        assert hybrid in ("none", "prior", "prior2")
        self.backbone, self.hybrid, self.n_sub, self.stage_time = backbone, hybrid, n_sub, stage_time
        c = width
        self.e0 = nn.Sequential(_block(IN_CH + (hybrid != "none"), c), _block(c, c))
        self.e1 = nn.Sequential(_block(c, c * 2, stride=2), _block(c * 2, c * 2))
        self.e2 = nn.Sequential(_block(c * 2, c * 4, stride=2), _block(c * 4, c * 4))
        if backbone == "ccs":
            self.dp = _Decoder(c, 1)
            self.ds = _Decoder(c, 1)
        else:
            self.dec = _Decoder(c, 2)

    def forward(self, x):
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
