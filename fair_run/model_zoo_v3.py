"""
model_zoo_v3.py -- all 13 field-predicting architectures (everything from
surrogate_comparison_spec.md except near_well_hybrid, which predicts a
scalar well-index correction, not a field, and isn't comparable here),
with CAPACITY REBALANCED across architectures and self-contained (no
dependency on neuraloperator -- a plain_fno built from the same
SpectralConv2d block as u_fno is included, so this file runs standalone).

WHY THIS FILE EXISTS (vs. reusing adapters_v3.py as-is): the first pass
left wildly uneven parameter counts -- e.g. ccsnet at width=16 (~150K
params) next to plain_fno/hybrid_deeponet_kan at 600K-1.3M. A model with
an order of magnitude less capacity will look like it "underperforms" the
architecture family when it's really just underpowered -- that's a
capacity confound, not an architectural finding. Every constructor below
targets roughly 500K-1.1M parameters (actual counts printed at
construction time in run_final_v3_comparison.py, not assumed). Two
documented exceptions, NOT bumped to match:
  - FINN: small parameter count is a STRUCTURAL property (only the
    constitutive flux relation is learned; the accounting is fixed), not
    a capacity shortfall -- inflating it would mean adding width to a
    part of the architecture the paper's whole point is to keep small.
  - GNS: message-passing width scales differently (cost is O(n_edges) per
    layer, not O(width^2) like a dense conv) -- bumped as far as is
    reasonable (node_hidden=64, edge_hidden=64, n_layers=5) without
    making every training step edge-count-dominated and slow.

All architectures take x = (B, 3, H, W) = [permeability, porosity,
well_mask] and return (B, 2, H, W) = [pressure, saturation] (saturation
sigmoid-bounded to [0,1], pressure left unbounded), matching the dataset
layout data_gen_v3.py produces.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# shared building blocks
# ===========================================================================

class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.out_ch = out_ch

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1, m2 = min(self.modes1, H), min(self.modes2, W // 2 + 1)
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(H, W))


class UNetBlock(nn.Module):
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


def split_output(raw):
    """Every architecture's final layer emits 2 RAW channels; this applies
    the one activation rule used consistently everywhere: pressure
    unbounded, saturation sigmoid-bounded to [0,1] (matches solver_v3.py's
    Sw in [Swc, 1-Sor] physically)."""
    return torch.cat([raw[:, 0:1], torch.sigmoid(raw[:, 1:2])], dim=1)


# ===========================================================================
# 1. Plain FNO (self-contained -- no neuraloperator dependency)
# ===========================================================================
class PlainFNO(nn.Module):
    def __init__(self, in_ch=3, width=48, modes=16, n_layers=4):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.proj1 = nn.Conv2d(width, width * 2, 1)
        self.proj2 = nn.Conv2d(width * 2, 2, 1)

    def forward(self, x):
        h = self.lift(x)
        for s, p in zip(self.spectral, self.pointwise):
            h = F.gelu(s(h) + p(h))
        h = F.gelu(self.proj1(h))
        return split_output(self.proj2(h))


# ===========================================================================
# 2. U-Net baseline (proper encoder/decoder with skip connections --
#    the earlier stand-in's skip logic was sloppy; this is the clean version)
# ===========================================================================
class UNetSurrogate(nn.Module):
    def __init__(self, in_ch=3, base_ch=48):
        super().__init__()
        c = base_ch
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, c, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(c, c, 3, padding=1), nn.GELU())
        self.pool1 = nn.Conv2d(c, c * 2, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(nn.Conv2d(c * 2, c * 2, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(c * 2, c * 2, 3, padding=1), nn.GELU())
        self.pool2 = nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(nn.Conv2d(c * 4, c * 4, 3, padding=1), nn.GELU())
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(nn.Conv2d(c * 4, c * 2, 3, padding=1), nn.GELU())
        self.up1 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(nn.Conv2d(c * 2, c, 3, padding=1), nn.GELU())
        self.out = nn.Conv2d(c, 2, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.gelu(self.pool1(e1)))
        b = self.bottleneck(F.gelu(self.pool2(e2)))
        d2 = self.up2(b)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return split_output(self.out(d1))


# ===========================================================================
# 3. Physics-informed FNO -- same PlainFNO backbone, physics-ness is a
#    training-loss property (see train_utils_v3.py), not architectural
# ===========================================================================
class PhysicsInformedFNO(PlainFNO):
    pass


# ===========================================================================
# 4. CCSNet -- width bumped 16 -> 48 (the capacity fix the user flagged)
# ===========================================================================
class CCSNetV3(nn.Module):
    def __init__(self, in_ch=3, width=48):
        super().__init__()

        def conv_block(ic, oc, stride=1):
            return nn.Sequential(nn.Conv2d(ic, oc, 3, stride=stride, padding=1), nn.InstanceNorm2d(oc), nn.GELU())

        self.b0 = conv_block(in_ch, width)
        self.b1 = conv_block(width, width * 2, stride=2)
        self.b2 = conv_block(width * 2, width * 4, stride=2)

        def decoder_head(out_ch):
            return nn.ModuleDict({
                "up1": nn.ConvTranspose2d(width * 4, width * 2, 4, stride=2, padding=1),
                "merge1": conv_block(width * 4, width * 2),
                "up2": nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1),
                "merge2": conv_block(width * 2, width),
                "out": nn.Conv2d(width, out_ch, 1),
            })

        self.pressure_head = decoder_head(1)
        self.saturation_head = decoder_head(1)

    def _decode(self, head, f0, f1, f2):
        h = head["up1"](f2)
        if h.shape[-2:] != f1.shape[-2:]:
            h = F.interpolate(h, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        h = head["merge1"](torch.cat([h, f1], dim=1))
        h = head["up2"](h)
        if h.shape[-2:] != f0.shape[-2:]:
            h = F.interpolate(h, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        h = head["merge2"](torch.cat([h, f0], dim=1))
        return head["out"](h)

    def forward(self, x):
        f0 = self.b0(x)
        f1 = self.b1(f0)
        f2 = self.b2(f1)
        pressure = self._decode(self.pressure_head, f0, f1, f2)
        saturation = self._decode(self.saturation_head, f0, f1, f2)
        return split_output(torch.cat([pressure, saturation], dim=1))


# ===========================================================================
# 5. U-FNO -- width bumped 32 -> 48, matches PlainFNO for a fair spectral
#    vs spectral+local comparison at EQUAL capacity
# ===========================================================================
class UFNOV3(nn.Module):
    def __init__(self, in_ch=3, width=48, modes1=16, modes2=16, n_plain=2, n_unet=2):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.plain_layers = nn.ModuleList([
            nn.ModuleDict({"spectral": SpectralConv2d(width, width, modes1, modes2),
                            "pointwise": nn.Conv2d(width, width, 1)})
            for _ in range(n_plain)])
        self.unet_layers = nn.ModuleList([
            nn.ModuleDict({"spectral": SpectralConv2d(width, width, modes1, modes2),
                            "pointwise": nn.Conv2d(width, width, 1),
                            "unet": UNetBlock(width)})
            for _ in range(n_unet)])
        self.proj1 = nn.Conv2d(width, width * 2, 1)
        self.proj2 = nn.Conv2d(width * 2, 2, 1)

    def forward(self, x):
        h = self.lift(x)
        for layer in self.plain_layers:
            h = F.gelu(layer["spectral"](h) + layer["pointwise"](h))
        for layer in self.unet_layers:
            h = F.gelu(layer["spectral"](h) + layer["pointwise"](h) + layer["unet"](h))
        h = F.gelu(self.proj1(h))
        return split_output(self.proj2(h))


# ===========================================================================
# 6. Nested FNO -- coarse+fine widths bumped to match
# ===========================================================================
class _SmallFNO(nn.Module):
    def __init__(self, in_ch, out_ch, width, modes, n_layers=3):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.proj = nn.Conv2d(width, out_ch, 1)

    def forward(self, x):
        h = self.lift(x)
        for s, p in zip(self.spectral, self.pointwise):
            h = F.gelu(s(h) + p(h))
        return self.proj(h)


class NestedFNOV3(nn.Module):
    def __init__(self, in_ch=3, H=32, W=32, coarse_width=40, fine_width=48, coarse_modes=12, fine_modes=14):
        super().__init__()
        self.coarse_size = max(8, H // 2)
        self.patch_size = max(8, min(H, self.coarse_size + self.coarse_size // 2))
        self.coarse_fno = _SmallFNO(in_ch, 2, coarse_width, coarse_modes)
        self.fine_fno = _SmallFNO(in_ch + 2, 2, fine_width, fine_modes)

    def _well_centers(self, x):
        well_mask = x[:, -1]
        B, H, W = well_mask.shape
        flat = well_mask.abs().reshape(B, -1)
        idx = flat.argmax(dim=1)
        return torch.stack([idx // W, idx % W], dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        centers = self._well_centers(x)
        x_coarse = F.interpolate(x, size=(self.coarse_size, self.coarse_size), mode="bilinear", align_corners=False)
        coarse_out = self.coarse_fno(x_coarse)
        coarse_up = F.interpolate(coarse_out, size=(H, W), mode="bilinear", align_corners=False)

        p = self.patch_size
        half = p // 2
        patches_in, patches_coarse, boxes = [], [], []
        for b in range(B):
            r, c = centers[b].tolist()
            r0 = max(0, min(H - p, r - half)); c0 = max(0, min(W - p, c - half))
            patches_in.append(x[b:b + 1, :, r0:r0 + p, c0:c0 + p])
            patches_coarse.append(coarse_up[b:b + 1, :, r0:r0 + p, c0:c0 + p])
            boxes.append((r0, c0))
        patch_in = torch.cat(patches_in, dim=0)
        patch_coarse = torch.cat(patches_coarse, dim=0)
        fine_out = self.fine_fno(torch.cat([patch_in, patch_coarse], dim=1))

        full_out = coarse_up.clone()
        for b in range(B):
            r0, c0 = boxes[b]
            full_out[b:b + 1, :, r0:r0 + p, c0:c0 + p] = fine_out[b:b + 1]
        return split_output(full_out)


# ===========================================================================
# 7/8. R-U-Net and PI-ConvLSTM (same backbone) -- hidden_ch bumped 32 -> 64
# ===========================================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=pad)
        self.hidden_ch = hidden_ch

    def forward(self, x, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        return o * torch.tanh(c_next), c_next

    def init_state(self, batch, H, W, device):
        z = torch.zeros(batch, self.hidden_ch, H, W, device=device)
        return z, z.clone()


class RUNetV3(nn.Module):
    def __init__(self, static_ch=3, hidden_ch=64, width=48, n_rollout_steps=3):
        super().__init__()
        self.cell = ConvLSTMCell(static_ch + 2, hidden_ch)
        self.down = nn.Conv2d(hidden_ch, width, 3, stride=2, padding=1)
        self.up = nn.ConvTranspose2d(width, width, 4, stride=2, padding=1)
        self.out = nn.Conv2d(width, 2, 1)
        self.n_rollout_steps = n_rollout_steps

    def _decode(self, h):
        d = F.gelu(self.down(h))
        u = F.gelu(self.up(d))
        if u.shape[-2:] != h.shape[-2:]:
            u = F.interpolate(u, size=h.shape[-2:], mode="bilinear", align_corners=False)
        return split_output(self.out(u))

    def rollout(self, static, dynamic_init, n_steps):
        B, _, H, W = static.shape
        state = self.cell.init_state(B, H, W, static.device)
        dynamic = dynamic_init
        for _ in range(n_steps):
            x = torch.cat([static, dynamic], dim=1)
            h, c = self.cell(x, state)
            state = (h, c)
            dynamic = self._decode(h)
        return dynamic

    def forward(self, x):
        B, _, H, W = x.shape
        dynamic_init = torch.zeros(B, 2, H, W, device=x.device)
        return self.rollout(x, dynamic_init, self.n_rollout_steps)


class PIConvLSTMV3(RUNetV3):
    """Same backbone -- physics-informed-ness lives in the loss function at
    training time (train_utils_v3.py), exactly matching how
    physics_informed_fno relates to plain_fno."""
    pass


# ===========================================================================
# 9. E2C -- latent_dim bumped 32 -> 64
# ===========================================================================
class E2CEncoder(nn.Module):
    def __init__(self, in_ch, H, W, latent_dim, width=24):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1), nn.GELU())
        with torch.no_grad():
            dummy = self.conv(torch.zeros(1, in_ch, H, W))
        self.flat_shape = dummy.shape[1:]
        self.fc = nn.Linear(dummy.numel(), latent_dim)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


class E2CDecoder(nn.Module):
    def __init__(self, out_ch, flat_shape, latent_dim, width=24):
        super().__init__()
        self.flat_shape = flat_shape
        self.fc = nn.Linear(latent_dim, int(torch.tensor(flat_shape).prod()))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(width * 2, width * 2, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(width, out_ch, 4, stride=2, padding=1))

    def forward(self, z, target_size):
        h = self.fc(z).view(-1, *self.flat_shape)
        out = self.deconv(h)
        if out.shape[-2:] != target_size:
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        return out


class E2CTransition(nn.Module):
    def __init__(self, latent_dim, control_dim, hidden=96):
        super().__init__()
        self.latent_dim = latent_dim
        self.A_net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, latent_dim ** 2))
        self.B_net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, latent_dim * control_dim))
        self.control_dim = control_dim

    def forward(self, z, u):
        B = z.shape[0]
        A = self.A_net(z).view(B, self.latent_dim, self.latent_dim)
        Bm = self.B_net(z).view(B, self.latent_dim, self.control_dim)
        return torch.bmm(A, z.unsqueeze(-1)).squeeze(-1) + torch.bmm(Bm, u.unsqueeze(-1)).squeeze(-1)


class E2CV3(nn.Module):
    def __init__(self, in_ch=3, H=32, W=32, latent_dim=64, control_dim=4, n_rollout_steps=4):
        super().__init__()
        self.H, self.W = H, W
        self.n_rollout_steps = n_rollout_steps
        self.encoder = E2CEncoder(in_ch, H, W, latent_dim)
        self.decoder = E2CDecoder(2, self.encoder.flat_shape, latent_dim)
        self.transition = E2CTransition(latent_dim, control_dim)

    @staticmethod
    def _controls_from_well_mask(well_mask, n_steps):
        pos = well_mask.clamp(min=0); neg = (-well_mask).clamp(min=0)
        total_inj = pos.sum(dim=(1, 2)); total_prod = neg.sum(dim=(1, 2))
        H, W = well_mask.shape[1], well_mask.shape[2]
        ys, xs = torch.meshgrid(torch.linspace(0, 1, H, device=well_mask.device),
                                 torch.linspace(0, 1, W, device=well_mask.device), indexing="ij")
        weight = pos + 1e-6
        cx = (weight * xs).sum(dim=(1, 2)) / weight.sum(dim=(1, 2))
        cy = (weight * ys).sum(dim=(1, 2)) / weight.sum(dim=(1, 2))
        control = torch.stack([total_inj, total_prod, cx, cy], dim=1)
        return control.unsqueeze(1).repeat(1, n_steps, 1)

    def forward(self, x):
        well_mask = x[:, -1]
        z = self.encoder(x)
        controls = self._controls_from_well_mask(well_mask, self.n_rollout_steps)
        for t in range(self.n_rollout_steps):
            z = self.transition(z, controls[:, t])
        out = self.decoder(z, (self.H, self.W))
        return split_output(out)


# ===========================================================================
# 10. DeepONet -- two branch/trunk pairs, latent_dim bumped 48 -> 80
# ===========================================================================
class _DeepONetHalf(nn.Module):
    def __init__(self, n_sensors, coord_dim=2, latent_dim=80, hidden=160):
        super().__init__()
        self.branch = nn.Sequential(nn.Linear(n_sensors, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, latent_dim))
        self.trunk = nn.Sequential(nn.Linear(coord_dim, hidden), nn.GELU(),
                                    nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, latent_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, sensor_values, query_coords):
        b = self.branch(sensor_values)
        t = self.trunk(query_coords)
        return torch.einsum("bl,ql->bq", b, t) + self.bias


class DeepONetV3(nn.Module):
    def __init__(self, in_ch=3, H=32, W=32, n_sensors=150, latent_dim=80):
        super().__init__()
        self.H, self.W = H, W
        self.in_ch = in_ch
        rows = torch.randint(0, H, (n_sensors,)); cols = torch.randint(0, W, (n_sensors,))
        self.register_buffer("sensor_idx", torch.stack([rows, cols], dim=1))
        self.pressure_net = _DeepONetHalf(n_sensors * in_ch, latent_dim=latent_dim)
        self.saturation_net = _DeepONetHalf(n_sensors * in_ch, latent_dim=latent_dim)

    def _sample_sensors(self, x):
        rows, cols = self.sensor_idx[:, 0], self.sensor_idx[:, 1]
        return x[:, :, rows, cols].reshape(x.shape[0], -1)

    def forward(self, x):
        device = x.device
        sensor_values = self._sample_sensors(x)
        ys, xs = torch.meshgrid(torch.linspace(0, 1, self.H, device=device),
                                 torch.linspace(0, 1, self.W, device=device), indexing="ij")
        coords = torch.stack([xs.flatten(), ys.flatten()], dim=-1)
        pressure = self.pressure_net(sensor_values, coords).view(-1, 1, self.H, self.W)
        saturation = self.saturation_net(sensor_values, coords).view(-1, 1, self.H, self.W)
        return split_output(torch.cat([pressure, saturation], dim=1))


# ===========================================================================
# 11. GNS -- node_hidden/edge_hidden bumped 32 -> 64, n_layers 4 -> 5
#     (see module docstring for why this isn't pushed further to match
#     the ~800K target the dense-conv architectures hit)
# ===========================================================================
def build_grid_edges(H, W, device):
    idx = torch.arange(H * W, device=device).view(H, W)
    src, dst = [], []
    src.append(idx[:, :-1].flatten()); dst.append(idx[:, 1:].flatten())
    src.append(idx[:, 1:].flatten()); dst.append(idx[:, :-1].flatten())
    src.append(idx[:-1, :].flatten()); dst.append(idx[1:, :].flatten())
    src.append(idx[1:, :].flatten()); dst.append(idx[:-1, :].flatten())
    return torch.stack([torch.cat(src), torch.cat(dst)], dim=0)


class GNSLayerV3(nn.Module):
    def __init__(self, node_dim, edge_hidden):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(2 * node_dim, edge_hidden), nn.GELU(),
                                       nn.Linear(edge_hidden, edge_hidden))
        self.node_mlp = nn.Sequential(nn.Linear(node_dim + edge_hidden, 96), nn.GELU(),
                                       nn.Linear(96, node_dim))

    def forward(self, node_feat, edge_index, n_nodes):
        B = node_feat.shape[0]
        src, dst = edge_index
        messages = self.edge_mlp(torch.cat([node_feat[:, src], node_feat[:, dst]], dim=-1))
        agg = torch.zeros(B, n_nodes, messages.shape[-1], device=node_feat.device)
        agg.index_add_(1, dst, messages)
        return node_feat + self.node_mlp(torch.cat([node_feat, agg], dim=-1))


class GNSV3(nn.Module):
    def __init__(self, H=32, W=32, static_dim=3, node_hidden=64, edge_hidden=64, n_layers=5):
        super().__init__()
        self.H, self.W = H, W
        self.encode = nn.Linear(static_dim + 2, node_hidden)
        self.layers = nn.ModuleList([GNSLayerV3(node_hidden, edge_hidden) for _ in range(n_layers)])
        self.decode = nn.Linear(node_hidden, 2)
        self.register_buffer("edge_index", build_grid_edges(H, W, "cpu"))

    def forward(self, x):
        B, C, H, W = x.shape
        static_nodes = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        dynamic = torch.zeros(B, H * W, 2, device=x.device)
        n_nodes = H * W
        h = self.encode(torch.cat([static_nodes, dynamic], dim=-1))
        for layer in self.layers:
            h = layer(h, self.edge_index, n_nodes)
        out = dynamic + self.decode(h)
        out_field = out.permute(0, 2, 1).reshape(B, 2, H, W)
        return split_output(out_field)


# ===========================================================================
# 12. Hybrid DeepONet+KAN -- latent_dim bumped 32 -> 56
# ===========================================================================
class FNOBranchV3(nn.Module):
    def __init__(self, in_ch, latent_dim, width=40, modes=14, n_layers=3):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(width, latent_dim)

    def forward(self, x):
        h = self.lift(x)
        for s, p in zip(self.spectral, self.pointwise):
            h = F.gelu(s(h) + p(h))
        return self.proj(self.pool(h).flatten(1))


class KANLayerV3(nn.Module):
    def __init__(self, in_dim, out_dim, n_basis=10, grid_range=(-2.0, 2.0)):
        super().__init__()
        self.register_buffer("centers", torch.linspace(*grid_range, n_basis))
        self.width = (grid_range[1] - grid_range[0]) / n_basis
        self.spline_weight = nn.Parameter(torch.randn(in_dim, out_dim, n_basis) * 0.1)
        self.base_weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)

    def forward(self, x):
        diff = x.unsqueeze(-1) - self.centers.view(1, 1, -1)
        basis = torch.exp(-0.5 * (diff / self.width) ** 2)
        spline_out = torch.einsum("bik,iok->bo", basis, self.spline_weight)
        base_out = torch.einsum("bi,io->bo", torch.tanh(x), self.base_weight)
        return spline_out + base_out


class KANTrunkV3(nn.Module):
    def __init__(self, coord_dim, latent_dim, hidden=48, n_layers=2):
        super().__init__()
        layers = [KANLayerV3(coord_dim, hidden)]
        for _ in range(n_layers - 1):
            layers.append(KANLayerV3(hidden, hidden))
        layers.append(KANLayerV3(hidden, latent_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, coords):
        h = coords
        for layer in self.layers:
            h = layer(h)
        return h


class _HybridKANHalf(nn.Module):
    def __init__(self, in_ch, coord_dim, latent_dim):
        super().__init__()
        self.branch = FNOBranchV3(in_ch, latent_dim)
        self.trunk = KANTrunkV3(coord_dim, latent_dim)
        self.mixer = nn.Sequential(nn.Linear(2 * latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, 1))

    def forward(self, field_input, query_coords):
        b = self.branch(field_input)
        t = self.trunk(query_coords)
        Bn, Q = b.shape[0], t.shape[0]
        mixed = torch.cat([b.unsqueeze(1).expand(Bn, Q, -1), t.unsqueeze(0).expand(Bn, Q, -1)], dim=-1)
        return self.mixer(mixed).squeeze(-1)


class HybridDeepONetKANV3(nn.Module):
    def __init__(self, in_ch=3, H=32, W=32, latent_dim=56):
        super().__init__()
        self.H, self.W = H, W
        self.pressure_net = _HybridKANHalf(in_ch, 3, latent_dim)
        self.saturation_net = _HybridKANHalf(in_ch, 3, latent_dim)

    def forward(self, x):
        device = x.device
        ys, xs = torch.meshgrid(torch.linspace(0, 1, self.H, device=device),
                                 torch.linspace(0, 1, self.W, device=device), indexing="ij")
        ts = torch.ones_like(xs)
        coords = torch.stack([xs.flatten(), ys.flatten(), ts.flatten()], dim=-1)
        pressure = self.pressure_net(x, coords).view(-1, 1, self.H, self.W)
        saturation = self.saturation_net(x, coords).view(-1, 1, self.H, self.W)
        return split_output(torch.cat([pressure, saturation], dim=1))


# ===========================================================================
# 13. FINN -- intentionally NOT capacity-bumped, see module docstring.
#     Two independent flux channels (pressure-like, saturation-like),
#     slightly deeper flux MLP than the original scaffold (64 -> 96
#     hidden) since the original's 2.7K params was arguably UNDER even
#     FINN's own modest target, not just small-by-design.
# ===========================================================================
class LearnedFluxV3(nn.Module):
    def __init__(self, state_dim, hidden=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * state_dim, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, state_left, state_right):
        return self.net(torch.cat([state_left, state_right], dim=-1)).squeeze(-1)


class FINNV3(nn.Module):
    def __init__(self, static_dim=3, n_steps=10, dt=0.05):
        super().__init__()
        self.pressure_flux = LearnedFluxV3(1 + static_dim)
        self.saturation_flux = LearnedFluxV3(1 + static_dim)
        self.n_steps = n_steps
        self.dt = dt

    def _update_channel(self, flux_net, channel, static, clamp_range=None):
        combined = torch.cat([channel, static], dim=1)
        left_x = combined[:, :, :-1, :].permute(0, 2, 3, 1); right_x = combined[:, :, 1:, :].permute(0, 2, 3, 1)
        flux_x = flux_net(left_x, right_x)
        left_y = combined[:, :, :, :-1].permute(0, 2, 3, 1); right_y = combined[:, :, :, 1:].permute(0, 2, 3, 1)
        flux_y = flux_net(left_y, right_y)
        B, _, H, W = channel.shape
        div = torch.zeros(B, H, W, device=channel.device)
        div[:, :-1, :] += flux_x; div[:, 1:, :] -= flux_x
        div[:, :, :-1] += flux_y; div[:, :, 1:] -= flux_y
        next_channel = channel[:, 0] - div * self.dt
        if clamp_range is not None:
            next_channel = next_channel.clamp(*clamp_range)
        return next_channel.unsqueeze(1)

    def forward(self, x):
        B, _, H, W = x.shape
        pressure = torch.zeros(B, 1, H, W, device=x.device)
        saturation = torch.full((B, 1, H, W), 0.2, device=x.device)
        for _ in range(self.n_steps):
            pressure = self._update_channel(self.pressure_flux, pressure, x)
            saturation = self._update_channel(self.saturation_flux, saturation, x, clamp_range=(0.0, 1.0))
        return torch.cat([pressure, saturation], dim=1)


# ===========================================================================
# factory -- every constructor needs H, W (several bake grid size in)
# ===========================================================================
def make_all_architectures(H, W):
    return {
        "plain_fno": lambda: PlainFNO(in_ch=3, width=48, modes=16),
        "unet_baseline": lambda: UNetSurrogate(in_ch=3, base_ch=48),
        "physics_informed_fno": lambda: PhysicsInformedFNO(in_ch=3, width=48, modes=16),
        "ccsnet": lambda: CCSNetV3(in_ch=3, width=48),
        "ufno": lambda: UFNOV3(in_ch=3, width=48, modes1=16, modes2=16),
        "nested_fno": lambda: NestedFNOV3(in_ch=3, H=H, W=W, coarse_width=40, fine_width=48),
        "runet": lambda: RUNetV3(static_ch=3, hidden_ch=64, width=48),
        "pi_convlstm": lambda: PIConvLSTMV3(static_ch=3, hidden_ch=64, width=48),
        "e2c": lambda: E2CV3(in_ch=3, H=H, W=W, latent_dim=64),
        "deeponet": lambda: DeepONetV3(in_ch=3, H=H, W=W, n_sensors=min(150, H * W // 3), latent_dim=80),
        "gns": lambda: GNSV3(H=H, W=W, static_dim=3, node_hidden=64, edge_hidden=64, n_layers=5),
        "hybrid_deeponet_kan": lambda: HybridDeepONetKANV3(in_ch=3, H=H, W=W, latent_dim=56),
        "finn": lambda: FINNV3(static_dim=3, n_steps=10, dt=0.05),
    }
