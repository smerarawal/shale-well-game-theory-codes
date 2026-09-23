"""
deeponet.py -- DeepONet (Lu, Jin, Karniadakis et al., 2021). Branch/trunk
architecture: the BRANCH net encodes the input function (kx, ky,
well_mask fields, sampled at a fixed set of sensor points), the TRUNK net
encodes the query coordinates (x, y) where the output is evaluated, and
the two are combined by a dot product -- this decouples "what the input
looks like" from "where you're asking for the output," which is DeepONet's
whole structural difference from a convolutional field-to-field map like
FNO/U-Net (those don't separate the two).

Per the direct comparison paper (arXiv:2509.06154, 2025): DeepONet needs
substantially more training data than FNO for comparable accuracy --
flagged here explicitly as a known disadvantage on this project's current
2000-sample budget, not glossed over. Included anyway because it's a
structurally distinct baseline worth having in the comparison table, and
because the trunk/query decoupling means DeepONet can natively evaluate
at OFF-GRID points (a capability none of the field-to-field models here
have, even if not needed by the current use case).

Run: python deeponet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BranchNet(nn.Module):
    """Encodes the input function, sampled at n_sensors fixed locations,
    into a latent_dim vector."""

    def __init__(self, n_sensors, latent_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_sensors, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, sensor_values):
        return self.net(sensor_values)


class TrunkNet(nn.Module):
    """Encodes query coordinates into the SAME latent_dim, so branch and
    trunk outputs can be dot-producted together."""

    def __init__(self, coord_dim, latent_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coord_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, coords):
        return self.net(coords)


class DeepONet(nn.Module):
    def __init__(self, n_sensors, coord_dim=2, latent_dim=64):
        super().__init__()
        self.branch = BranchNet(n_sensors, latent_dim)
        self.trunk = TrunkNet(coord_dim, latent_dim)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, sensor_values, query_coords):
        """sensor_values: (B, n_sensors). query_coords: (Q, coord_dim) --
        SHARED across the batch, or (B, Q, coord_dim) for per-sample query
        sets; both handled. Returns (B, Q): output value at each query
        point for each batch item."""
        b = self.branch(sensor_values)              # (B, latent_dim)
        if query_coords.dim() == 2:
            t = self.trunk(query_coords)             # (Q, latent_dim)
            out = torch.einsum("bl,ql->bq", b, t)
        else:
            t = self.trunk(query_coords)             # (B, Q, latent_dim)
            out = torch.einsum("bl,bql->bq", b, t)
        return out + self.bias

    def predict_grid(self, sensor_values, H, W, device):
        """Convenience: evaluate on a full regular H x W grid -- comparable
        output shape to the field-to-field models, even though DeepONet
        doesn't natively need a grid (this is for apples-to-apples
        comparison in comparison_protocol.py, not a structural requirement)."""
        ys, xs = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device), indexing="ij")
        coords = torch.stack([xs.flatten(), ys.flatten()], dim=-1)  # (H*W, 2)
        out = self.forward(sensor_values, coords)                  # (B, H*W)
        return out.view(-1, 1, H, W)


def sample_sensors(field_stack, sensor_idx):
    """field_stack: (B, C, H, W). sensor_idx: (n_sensors, 2) fixed (row,
    col) sampling locations, shared across the dataset (DeepONet's branch
    input width is fixed at train time, unlike a conv net's spatial input)."""
    B, C, H, W = field_stack.shape
    rows, cols = sensor_idx[:, 0], sensor_idx[:, 1]
    sampled = field_stack[:, :, rows, cols]  # (B, C, n_sensors)
    return sampled.reshape(B, -1)            # (B, C * n_sensors)


def train_step(model, opt, sensor_values, query_coords, target_values):
    model.train()
    opt.zero_grad()
    pred = model(sensor_values, query_coords)
    loss = F.mse_loss(pred, target_values)
    loss.backward()
    opt.step()
    return loss.item()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, nx, ny = 4, 64, 64
    n_sensors, in_ch = 100, 3
    model = DeepONet(n_sensors=n_sensors * in_ch, coord_dim=2, latent_dim=48)

    field_stack = torch.rand(B, in_ch, nx, ny)
    sensor_idx = torch.randint(0, nx, (n_sensors, 2))
    sensor_values = sample_sensors(field_stack, sensor_idx)
    assert sensor_values.shape == (B, n_sensors * in_ch)

    print("=== off-grid query evaluation check (the structural capability) ===")
    off_grid_coords = torch.rand(37, 2)  # arbitrary, non-grid-aligned points
    pred_off_grid = model(sensor_values, off_grid_coords)
    assert pred_off_grid.shape == (B, 37)
    print(f"OK: evaluated at {37} arbitrary off-grid points, shape {tuple(pred_off_grid.shape)}")

    print("=== full-grid comparison-shape check ===")
    pred_grid = model.predict_grid(sensor_values, nx, ny, sensor_values.device)
    assert pred_grid.shape == (B, 1, nx, ny)
    print(f"OK: grid-shape output {tuple(pred_grid.shape)} for comparison_protocol.py")

    print("=== gradient flow check ===")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    target = torch.randn(B, 37)
    l0 = train_step(model, opt, sensor_values, off_grid_coords, target)
    l1 = train_step(model, opt, sensor_values, off_grid_coords, target)
    print(f"loss step0={l0:.4f} step1={l1:.4f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"param count: {n_params:,}")
    print("PASS (caveat carried from spec: expect this to need more training data than FNO)")
