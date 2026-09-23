"""
hybrid_deeponet_kan.py -- Hybrid DeepONet combining FNO + MLP + KAN
(Kolmogorov-Arnold Network) inside one branch/trunk structure (arXiv:
2511.02962, 2025 / ScienceDirect 2026). Validated in the source paper on
SPE10, the standard reservoir-engineering heterogeneity benchmark --
flagged in the spec as the closest published match to this project's
exact multiphase target, so this is the one file in the set worth taking
most seriously as a real accuracy contender, not just a scaffold-for-
completeness entry.

Structure: BRANCH is an FNO tower over the input fields (spatial
structure, reuses the SpectralConv2d block from ufno.py/nested_fno.py) --
this is the "spatial learning" half. TRUNK is a KAN over the query
coordinates (temporal/positional structure) -- the "temporal learning"
half, decoupled from the branch per the paper's design. An MLP mixes the
two latent vectors before the final dot product, instead of a bare dot
product like plain DeepONet, since branch and trunk here come from very
different architectures (FNO's pooled features vs. KAN's spline-basis
features) and a plain dot product would be a coordinate-scale mismatch.

KAN layer implementation: learnable 1D spline (here, a fixed small
B-spline-like basis of Gaussian bumps per edge, per the standard
"replace weights with learnable univariate functions" KAN idea) instead
of a fixed nonlinearity + learned linear weight -- the actual structural
difference from an MLP, kept explicit rather than approximated away with
a bigger MLP.

Run: python hybrid_deeponet_kan.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FNOBranch(nn.Module):
    """The 'spatial learning' half: pools an FNO tower's output to a fixed
    latent vector per sample."""

    def __init__(self, in_ch, latent_dim, width=24, modes=10, n_layers=3):
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
        h = self.pool(h).flatten(1)
        return self.proj(h)


class KANLayer(nn.Module):
    """One KAN layer: for each (input, output) pair, a learnable univariate
    function implemented as a weighted sum of fixed Gaussian radial basis
    functions along that input's range, replacing the usual
    weight-times-input + fixed-nonlinearity of an MLP layer. n_basis
    centers are fixed (spec-simple), only the per-edge basis WEIGHTS and a
    residual linear term are learned (standard KAN practice: spline part +
    a base linear term)."""

    def __init__(self, in_dim, out_dim, n_basis=8, grid_range=(-2.0, 2.0)):
        super().__init__()
        centers = torch.linspace(grid_range[0], grid_range[1], n_basis)
        self.register_buffer("centers", centers)
        self.width = (grid_range[1] - grid_range[0]) / n_basis
        self.spline_weight = nn.Parameter(torch.randn(in_dim, out_dim, n_basis) * 0.1)
        self.base_weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)

    def forward(self, x):
        # x: (B, in_dim) -> basis: (B, in_dim, n_basis)
        diff = x.unsqueeze(-1) - self.centers.view(1, 1, -1)
        basis = torch.exp(-0.5 * (diff / self.width) ** 2)
        spline_out = torch.einsum("bik,iok->bo", basis, self.spline_weight)
        base_out = torch.einsum("bi,io->bo", torch.tanh(x), self.base_weight)  # base activation = tanh, per KAN convention
        return spline_out + base_out


class KANTrunk(nn.Module):
    """The 'temporal learning' half: stacked KAN layers over query
    coordinates (x, y, t)."""

    def __init__(self, coord_dim, latent_dim, hidden=32, n_layers=2):
        super().__init__()
        layers = [KANLayer(coord_dim, hidden)]
        for _ in range(n_layers - 1):
            layers.append(KANLayer(hidden, hidden))
        layers.append(KANLayer(hidden, latent_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, coords):
        h = coords
        for layer in self.layers:
            h = layer(h)
        return h


class HybridDeepONetKAN(nn.Module):
    def __init__(self, in_ch=3, coord_dim=3, latent_dim=48):
        super().__init__()
        self.branch = FNOBranch(in_ch, latent_dim)
        self.trunk = KANTrunk(coord_dim, latent_dim)
        # MLP mixer instead of a bare dot product -- branch (FNO-pooled) and
        # trunk (KAN-spline) latents come from structurally different
        # feature spaces, so mix before combining rather than assume they're
        # already on comparable scales.
        self.mixer = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim), nn.GELU(),
            nn.Linear(latent_dim, 1))

    def forward(self, field_input, query_coords):
        """field_input: (B, in_ch, H, W). query_coords: (Q, coord_dim)
        shared query set (x, y, t) -- t included, unlike plain DeepONet's
        (x, y) only, since the paper's trunk covers space AND time."""
        b = self.branch(field_input)                       # (B, latent_dim)
        t = self.trunk(query_coords)                        # (Q, latent_dim)
        Bn, Q = b.shape[0], t.shape[0]
        b_exp = b.unsqueeze(1).expand(Bn, Q, -1)
        t_exp = t.unsqueeze(0).expand(Bn, Q, -1)
        mixed = torch.cat([b_exp, t_exp], dim=-1)
        return self.mixer(mixed).squeeze(-1)                # (B, Q)


def train_step(model, opt, field_input, query_coords, target_values):
    model.train()
    opt.zero_grad()
    pred = model(field_input, query_coords)
    loss = F.mse_loss(pred, target_values)
    loss.backward()
    opt.step()
    return loss.item()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, nx, ny = 3, 48, 48
    model = HybridDeepONetKAN(in_ch=3, coord_dim=3, latent_dim=32)
    field_input = torch.rand(B, 3, nx, ny)
    query_coords = torch.rand(20, 3)  # (x, y, t), off-grid + multi-time in one query set
    target_values = torch.randn(B, 20)

    print("=== spatiotemporal off-grid query check ===")
    pred = model(field_input, query_coords)
    assert pred.shape == (B, 20)
    print(f"OK: {tuple(pred.shape)} -- queries include arbitrary (x, y, t), not just a fixed spatial grid")

    print("=== KAN layer basis-function sanity check ===")
    kan = KANLayer(4, 6, n_basis=8)
    x = torch.randn(5, 4)
    out = kan(x)
    assert out.shape == (5, 6)
    print(f"OK: KAN layer output shape {tuple(out.shape)}")

    print("=== gradient flow check ===")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    l0 = train_step(model, opt, field_input, query_coords, target_values)
    l1 = train_step(model, opt, field_input, query_coords, target_values)
    print(f"loss step0={l0:.4f} step1={l1:.4f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"param count: {n_params:,}")
    print("PASS")
