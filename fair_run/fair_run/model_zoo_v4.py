"""
model_zoo_v4.py -- the architectures for the FINAL fair comparison.

Everything is built on model_zoo_v3.py (same building blocks), with these
changes. Every change is listed because each one is a protocol decision that
should be defensible in the paper.

UNIFORM PROTOCOL (applies to ALL 13 models via FieldNormWrapper):
  * inputs standardised with TRAIN-set statistics (permeability, porosity:
    z-score; well_mask: divided by max |value|).
  * two coordinate channels (x, y in [0,1]) appended -> every model sees the
    same 5 input channels. FNOs assume periodic boundaries; the reservoir has
    no-flow boundaries, so position information is standard practice.
  * pressure output is de-normalised (out*p_std + p_mean) inside the wrapper,
    so the loss and ALL metrics are computed in raw physical units, exactly
    the definition used in the earlier Kaggle table.

ARCHITECTURE FIXES (each addresses a diagnosed cause, not a hunch):
  * deeponet: the old branch saw 150 RANDOM cells out of 1024 -> a well cell
    was visible with probability ~15%. Now the branch sees EVERY grid cell
    (all-grid sensors), and the trunk uses Fourier features (plain-coordinate
    trunks are known to be too smooth for sharp fields).
  * hybrid_deeponet_kan: the FNO branch global-average-pooled (destroying well
    position). Now pools to 8x8 and flattens before the latent projection.
  * e2c: replaced by the reference-matched E2C v2 (orthogonal init, spatial
    bottleneck, dt-aware transition) trained on TRAJECTORIES with an
    autoencoding term + latent-rollout prediction term. GroupNorm/LayerNorm
    instead of BatchNorm (BatchNorm eval-mode statistics broke the earlier
    runs). Its initial state is the solver's true initial condition
    (p=0, Sw=Swc), so the final-field interface is unchanged. E2C therefore
    uses intermediate-time supervision the other models do not; this is a
    documented asymmetry, required for E2C to function at all.
  * nested_fno: located wells via x[:, -1]; with coordinate channels appended
    that would silently be the y-coordinate. Now uses channel 2.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_zoo_v3 import (
    PlainFNO, UNetSurrogate, PhysicsInformedFNO, CCSNetV3, UFNOV3, NestedFNOV3,
    RUNetV3, PIConvLSTMV3, GNSV3, FINNV3, SpectralConv2d, KANTrunkV3, split_output,
    GNSLayerV3, build_grid_edges,
)
from model_zoo_v3_e2c_v2 import (
    E2CEncoderV2, E2CDecoderV2, DtAwareTransition, StaticContextEncoderV2,
)

N_STATIC = 5  # 3 standardised fields + 2 coordinate channels


# ---------------------------------------------------------------------------
# uniform wrapper
# ---------------------------------------------------------------------------
class FieldNormWrapper(nn.Module):
    def __init__(self, model, stats, H, W):
        super().__init__()
        self.model = model
        t = lambda v: torch.tensor(v, dtype=torch.float32)
        self.register_buffer("in_mean", t(stats["in_mean"]).view(1, 3, 1, 1))
        self.register_buffer("in_std", t(stats["in_std"]).view(1, 3, 1, 1))
        self.register_buffer("p_mean", t(stats["p_mean"]))
        self.register_buffer("p_std", t(stats["p_std"]))
        ii, jj = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
        self.register_buffer("coords", torch.stack([ii, jj], dim=0).unsqueeze(0))
        self.has_traj = hasattr(model, "traj_outputs")
        if hasattr(model, "configure"):
            model.configure(float(stats["p_mean"]), float(stats["p_std"]))

    def prep(self, x):
        xn = (x - self.in_mean) / self.in_std
        return torch.cat([xn, self.coords.expand(x.shape[0], -1, -1, -1)], dim=1)

    def _denorm(self, out):
        return torch.cat([out[:, 0:1] * self.p_std + self.p_mean, out[:, 1:2]], dim=1)

    def forward(self, x):
        return self._denorm(self.model(self.prep(x)))

    def traj_loss(self, x, ytraj, wp, ws):
        """ytraj: raw (B,T,2,H,W). Autoencoding + latent-rollout prediction,
        both measured in raw units with the same channel weights."""
        xn = self.prep(x)
        frames_n = torch.cat([(ytraj[:, :, 0:1] - self.p_mean) / self.p_std, ytraj[:, :, 1:2]], dim=2)
        ae, pr = self.model.traj_outputs(xn, frames_n)
        total = 0.0
        for out in (ae, pr):
            if out is None:  # models without an autoencoding term (recurrent rollouts)
                continue
            out = torch.cat([out[:, :, 0:1] * self.p_std + self.p_mean, out[:, :, 1:2]], dim=2)
            total = total + wp * F.mse_loss(out[:, :, 0], ytraj[:, :, 0]) + ws * F.mse_loss(out[:, :, 1], ytraj[:, :, 1])
        return total


# ---------------------------------------------------------------------------
# nested FNO (well-mask channel index fix)
# ---------------------------------------------------------------------------
class NestedFNOV4(NestedFNOV3):
    def _well_centers(self, x):
        well_mask = x[:, 2]
        B, H, W = well_mask.shape
        idx = well_mask.abs().reshape(B, -1).argmax(dim=1)
        return torch.stack([idx // W, idx % W], dim=1)


# ---------------------------------------------------------------------------
# DeepONet with all-grid sensors + Fourier-feature trunk
# ---------------------------------------------------------------------------
class FourierFeatures(nn.Module):
    def __init__(self, in_dim=2, n_freq=6):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freq)) * math.pi)
        self.out_dim = in_dim + 2 * in_dim * n_freq

    def forward(self, c):
        a = (c.unsqueeze(-1) * self.freqs).flatten(1)
        return torch.cat([c, torch.sin(a), torch.cos(a)], dim=-1)


class _DeepONetHalfV4(nn.Module):
    def __init__(self, n_in, latent=96, hidden=256, trunk_hidden=192):
        super().__init__()
        self.branch = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(),
                                    nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, latent))
        self.ff = FourierFeatures(2, 6)
        self.trunk = nn.Sequential(nn.Linear(self.ff.out_dim, trunk_hidden), nn.GELU(),
                                   nn.Linear(trunk_hidden, trunk_hidden), nn.GELU(), nn.Linear(trunk_hidden, latent))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, sensors, coords):
        return torch.einsum("bl,ql->bq", self.branch(sensors), self.trunk(self.ff(coords))) + self.bias


class DeepONetV4(nn.Module):
    def __init__(self, in_ch=N_STATIC, H=32, W=32, latent_dim=96):
        super().__init__()
        self.H, self.W = H, W
        self.pressure_net = _DeepONetHalfV4(in_ch * H * W, latent=latent_dim)
        self.saturation_net = _DeepONetHalfV4(in_ch * H * W, latent=latent_dim)
        ys, xs = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
        self.register_buffer("coords", torch.stack([xs.flatten(), ys.flatten()], dim=-1))

    def forward(self, x):
        s = x.flatten(1)  # ALL grid cells are sensors
        p = self.pressure_net(s, self.coords).view(-1, 1, self.H, self.W)
        sat = self.saturation_net(s, self.coords).view(-1, 1, self.H, self.W)
        return split_output(torch.cat([p, sat], dim=1))


# ---------------------------------------------------------------------------
# Hybrid DeepONet + KAN with a spatial (non-global-pooled) FNO branch
# ---------------------------------------------------------------------------
class FNOBranchV4(nn.Module):
    def __init__(self, in_ch, latent_dim, width=40, modes=14, n_layers=3, pool=8):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.pool = nn.AdaptiveAvgPool2d(pool)
        self.proj = nn.Sequential(nn.Linear(width * pool * pool, 256), nn.GELU(), nn.Linear(256, latent_dim))

    def forward(self, x):
        h = self.lift(x)
        for s, p in zip(self.spectral, self.pointwise):
            h = F.gelu(s(h) + p(h))
        return self.proj(self.pool(h).flatten(1))


class _HybridKANHalfV4(nn.Module):
    def __init__(self, in_ch, coord_dim, latent_dim):
        super().__init__()
        self.branch = FNOBranchV4(in_ch, latent_dim)
        self.trunk = KANTrunkV3(coord_dim, latent_dim)
        self.mixer = nn.Sequential(nn.Linear(2 * latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, 1))

    def forward(self, field_input, query_coords):
        b = self.branch(field_input)
        t = self.trunk(query_coords)
        Bn, Q = b.shape[0], t.shape[0]
        mixed = torch.cat([b.unsqueeze(1).expand(Bn, Q, -1), t.unsqueeze(0).expand(Bn, Q, -1)], dim=-1)
        return self.mixer(mixed).squeeze(-1)


class HybridDeepONetKANV4(nn.Module):
    def __init__(self, in_ch=N_STATIC, H=32, W=32, latent_dim=56):
        super().__init__()
        self.H, self.W = H, W
        self.pressure_net = _HybridKANHalfV4(in_ch, 3, latent_dim)
        self.saturation_net = _HybridKANHalfV4(in_ch, 3, latent_dim)
        ys, xs = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
        self.register_buffer("coords", torch.stack([xs.flatten(), ys.flatten(), torch.ones(H * W)], dim=-1))

    def forward(self, x):
        p = self.pressure_net(x, self.coords).view(-1, 1, self.H, self.W)
        s = self.saturation_net(x, self.coords).view(-1, 1, self.H, self.W)
        return split_output(torch.cat([p, s], dim=1))


# ---------------------------------------------------------------------------
# E2C v4: reference-matched E2C v2, trajectory-trained, norm-layer swapped
# ---------------------------------------------------------------------------
def _swap_norms(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            c = child.num_features
            setattr(module, name, nn.GroupNorm(8 if c % 8 == 0 else 1, c))
        elif isinstance(child, nn.BatchNorm1d):
            setattr(module, name, nn.LayerNorm(child.num_features))
        else:
            _swap_norms(child)


class E2CV4(nn.Module):
    SW_INIT = 0.2  # solver's initial Sw = Swc (solve_two_phase default)

    def __init__(self, static_ch=N_STATIC, H=32, W=32, latent_dim=128, context_dim=16, n_steps=6):
        super().__init__()
        self.H, self.W, self.n_steps = H, W, n_steps
        self.encoder = E2CEncoderV2(2 + static_ch, H, W, latent_dim)
        self.decoder = E2CDecoderV2(2, self.encoder.flat_shape, latent_dim)
        self.transition = DtAwareTransition(latent_dim, context_dim)
        self.context_encoder = StaticContextEncoderV2(static_ch, context_dim)
        _swap_norms(self)
        self.register_buffer("init_state", torch.zeros(1, 2, 1, 1))
        self.init_state[:, 1] = self.SW_INIT

    def configure(self, p_mean, p_std):
        # raw initial pressure is 0 -> normalised value
        self.init_state[:, 0] = (0.0 - p_mean) / p_std

    def _decode(self, z):
        return split_output(self.decoder(z, (self.H, self.W)))

    def _rollout(self, static, n):
        B = static.shape[0]
        ctx = self.context_encoder(static)
        dt = torch.ones(B, 1, device=static.device)
        frame0 = self.init_state.expand(B, 2, self.H, self.W)
        z = self.encoder(torch.cat([frame0, static], dim=1))
        zs = [z]
        for _ in range(n):
            z = self.transition(z, ctx, dt)
            zs.append(z)
        return zs

    def forward(self, static):
        return self._decode(self._rollout(static, self.n_steps)[-1])

    def traj_outputs(self, static, frames_n):
        T = frames_n.shape[1]
        ae = torch.stack([self._decode(self.encoder(torch.cat([frames_n[:, k], static], dim=1)))
                          for k in range(T)], dim=1)
        zs = self._rollout(static, T)
        pr = torch.stack([self._decode(zs[k + 1]) for k in range(T)], dim=1)
        return ae, pr


# ---------------------------------------------------------------------------
# Trajectory-trained recurrent variants (R-U-Net, PI-ConvLSTM, GNS).
# Same backbones as the final-frame versions, but rolled out for T steps from
# the solver's true initial state (p=0, Sw=Swc), one step per snapshot, and
# supervised at EVERY snapshot (prediction loss only; no autoencoding term).
# The plain "runet"/"pi_convlstm"/"gns" entries stay final-frame-only, so the
# comparison shows what intermediate-time supervision buys each backbone.
# ---------------------------------------------------------------------------
class RUNetTrajV4(RUNetV3):
    def __init__(self, static_ch=N_STATIC, hidden_ch=64, width=48, n_steps=6):
        super().__init__(static_ch, hidden_ch, width, n_rollout_steps=n_steps)
        self.n_steps = n_steps
        self.register_buffer("init_state", torch.zeros(1, 2, 1, 1))
        self.init_state[:, 1] = 0.2

    def configure(self, p_mean, p_std):
        self.init_state[:, 0] = (0.0 - p_mean) / p_std

    def _rollout_all(self, static):
        B, _, H, W = static.shape
        state = self.cell.init_state(B, H, W, static.device)
        dyn = self.init_state.expand(B, 2, H, W)
        outs = []
        for _ in range(self.n_steps):
            h, c = self.cell(torch.cat([static, dyn], dim=1), state)
            state = (h, c)
            dyn = self._decode(h)
            outs.append(dyn)
        return torch.stack(outs, dim=1)

    def forward(self, x):
        return self._rollout_all(x)[:, -1]

    def traj_outputs(self, static, frames_n):
        return None, self._rollout_all(static)


class PIConvLSTMTrajV4(RUNetTrajV4):
    """Physics-informed-ness lives in the loss (pressure residual on the final field)."""
    pass


class GNSTrajV4(nn.Module):
    def __init__(self, H=32, W=32, static_dim=N_STATIC, node_hidden=64, edge_hidden=64,
                 layers_per_step=3, n_steps=6):
        super().__init__()
        self.H, self.W, self.n_steps = H, W, n_steps
        self.encode = nn.Linear(static_dim + 2, node_hidden)
        self.layers = nn.ModuleList([GNSLayerV3(node_hidden, edge_hidden) for _ in range(layers_per_step)])
        self.decode = nn.Linear(node_hidden, 2)
        self.register_buffer("edge_index", build_grid_edges(H, W, "cpu"))
        self.register_buffer("init_state", torch.zeros(1, 2, 1, 1))
        self.init_state[:, 1] = 0.2

    def configure(self, p_mean, p_std):
        self.init_state[:, 0] = (0.0 - p_mean) / p_std

    def _step(self, static_nodes, dyn):
        B, n, _ = static_nodes.shape
        h = self.encode(torch.cat([static_nodes, dyn], dim=-1))
        for layer in self.layers:
            h = layer(h, self.edge_index, n)
        return split_output(self.decode(h).permute(0, 2, 1).reshape(B, 2, self.H, self.W))

    def _rollout_all(self, x):
        from torch.utils.checkpoint import checkpoint
        B, C, H, W = x.shape
        n = H * W
        static_nodes = x.permute(0, 2, 3, 1).reshape(B, n, C)
        dyn = self.init_state.expand(B, 2, H, W).permute(0, 2, 3, 1).reshape(B, n, 2)
        outs = []
        for _ in range(self.n_steps):
            # per-step activation checkpointing: message passing over ~4k edges x 64 ch x 32 samples x 3 layers
            # x 6 steps does not fit comfortably in memory otherwise (recompute costs ~30% more time)
            if self.training and torch.is_grad_enabled():
                field = checkpoint(self._step, static_nodes, dyn, use_reentrant=False)
            else:
                field = self._step(static_nodes, dyn)
            outs.append(field)
            dyn = field.permute(0, 2, 3, 1).reshape(B, n, 2)
        return torch.stack(outs, dim=1)

    def forward(self, x):
        return self._rollout_all(x)[:, -1]

    def traj_outputs(self, static, frames_n):
        return None, self._rollout_all(static)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
# relative training cost (seconds per 100 epochs / 800 samples in the earlier
# Kaggle run; e2c/deeponet are estimates). Used ONLY to balance workers.
ARCH_COST = {
    "finn": 225, "gns": 149, "pi_convlstm": 78, "runet": 77, "ccsnet": 66, "ufno": 58,
    "hybrid_deeponet_kan": 70, "nested_fno": 57, "physics_informed_fno": 49, "unet_baseline": 47,
    "plain_fno": 46, "e2c": 120, "deeponet": 25,
    "runet_traj": 150, "pi_convlstm_traj": 150, "gns_traj": 500,
}
PHYSICS_ARCHS = {"physics_informed_fno", "pi_convlstm", "pi_convlstm_traj"}


def make_all_architectures_v4(H, W, T_snap, stats):
    C = N_STATIC
    raw = {
        "plain_fno": lambda: PlainFNO(in_ch=C, width=48, modes=16),
        "unet_baseline": lambda: UNetSurrogate(in_ch=C, base_ch=48),
        "physics_informed_fno": lambda: PhysicsInformedFNO(in_ch=C, width=48, modes=16),
        "ccsnet": lambda: CCSNetV3(in_ch=C, width=48),
        "ufno": lambda: UFNOV3(in_ch=C, width=48, modes1=16, modes2=16),
        "nested_fno": lambda: NestedFNOV4(in_ch=C, H=H, W=W, coarse_width=40, fine_width=48),
        "runet": lambda: RUNetV3(static_ch=C, hidden_ch=64, width=48),
        "pi_convlstm": lambda: PIConvLSTMV3(static_ch=C, hidden_ch=64, width=48),
        "e2c": lambda: E2CV4(static_ch=C, H=H, W=W, n_steps=T_snap),
        "deeponet": lambda: DeepONetV4(in_ch=C, H=H, W=W),
        "gns": lambda: GNSV3(H=H, W=W, static_dim=C, node_hidden=64, edge_hidden=64, n_layers=5),
        "hybrid_deeponet_kan": lambda: HybridDeepONetKANV4(in_ch=C, H=H, W=W),
        "finn": lambda: FINNV3(static_dim=C, n_steps=10, dt=0.05),
        "runet_traj": lambda: RUNetTrajV4(static_ch=C, n_steps=T_snap),
        "pi_convlstm_traj": lambda: PIConvLSTMTrajV4(static_ch=C, n_steps=T_snap),
        "gns_traj": lambda: GNSTrajV4(H=H, W=W, static_dim=C, n_steps=T_snap),
    }
    return {k: (lambda f=f: FieldNormWrapper(f(), stats, H, W)) for k, f in raw.items()}
