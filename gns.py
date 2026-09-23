"""
gns.py -- Graph Neural Simulator, following the architecture family used
in the same comparison paper as deeponet.py (arXiv:2509.06154, 2025) that
introduced the FR (Full Rollout) vs AR (Autoregressive) distinction: the
paper found FNO-FR (what trajectory_fno does) has appreciable error
specifically in EARLY timesteps but generally outperforms FNO-AR after
that, while GNS had the LOWEST error growth over long horizons among
everything they tested -- error plateaus rather than compounds. That's
the property this file exists to make available for a long-horizon MARL
episode where late-horizon accuracy matters as much as early.

Grid cells are graph nodes (one node per cell, holding kx, ky, pressure,
saturation as features); graph edges are the SAME cell-face adjacency
solver_v3.py's harmonic-mean flux already uses (4-connected: E/W/N/S) --
so the message-passing structure mirrors the discretization's own
connectivity, not an arbitrary graph. Message passing computes a learned
flux-like quantity per edge, aggregates into each node, updates node
state -- autoregressive (AR), one physical step at a time, like R-U-Net,
but with graph structure instead of conv kernels.

Run: python gns.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_grid_edges(H, W, device):
    """4-connected grid adjacency (E/W/N/S), same connectivity as
    solver_v3.py's harmonic-mean face structure. Returns (2, n_edges)
    edge_index in both directions (message passing is symmetric here)."""
    idx = torch.arange(H * W, device=device).view(H, W)
    src, dst = [], []
    # horizontal edges (E-W)
    src.append(idx[:, :-1].flatten()); dst.append(idx[:, 1:].flatten())
    src.append(idx[:, 1:].flatten()); dst.append(idx[:, :-1].flatten())
    # vertical edges (N-S)
    src.append(idx[:-1, :].flatten()); dst.append(idx[1:, :].flatten())
    src.append(idx[1:, :].flatten()); dst.append(idx[:-1, :].flatten())
    return torch.stack([torch.cat(src), torch.cat(dst)], dim=0)


class EdgeMLP(nn.Module):
    """Computes a learned per-edge message from the two endpoint node
    states -- plays the role solver_v3.py's harmonic-mean transmissibility
    plays in the real discretization, except learned instead of closed-form."""

    def __init__(self, node_dim, edge_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * node_dim, edge_hidden), nn.GELU(),
            nn.Linear(edge_hidden, edge_hidden))

    def forward(self, node_feat, edge_index):
        src, dst = edge_index
        h = torch.cat([node_feat[:, src], node_feat[:, dst]], dim=-1)
        return self.net(h)  # (B, n_edges, edge_hidden)


class NodeUpdateMLP(nn.Module):
    """Aggregates incoming edge messages (sum, since flux exchange is
    additive/conservative in spirit -- same reasoning FINN uses explicitly)
    and updates each node's state."""

    def __init__(self, node_dim, edge_hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim + edge_hidden, 64), nn.GELU(),
            nn.Linear(64, out_dim))

    def forward(self, node_feat, aggregated_messages):
        h = torch.cat([node_feat, aggregated_messages], dim=-1)
        return self.net(h)


class GNSLayer(nn.Module):
    def __init__(self, node_dim, edge_hidden):
        super().__init__()
        self.edge_mlp = EdgeMLP(node_dim, edge_hidden)
        self.node_mlp = NodeUpdateMLP(node_dim, edge_hidden, node_dim)

    def forward(self, node_feat, edge_index, n_nodes):
        B = node_feat.shape[0]
        messages = self.edge_mlp(node_feat, edge_index)          # (B, n_edges, edge_hidden)
        dst = edge_index[1]
        agg = torch.zeros(B, n_nodes, messages.shape[-1], device=node_feat.device)
        agg.index_add_(1, dst, messages)                          # sum incoming messages per node
        delta = self.node_mlp(node_feat, agg)
        return node_feat + delta                                  # residual update


class GNS(nn.Module):
    def __init__(self, static_dim=3, dynamic_dim=2, node_hidden=32, edge_hidden=32, n_layers=4):
        """static_dim: kx, ky, well_mask. dynamic_dim: pressure, saturation.
        Node feature = concat(static, dynamic), fixed size node_hidden
        after an initial encoder."""
        super().__init__()
        self.encode = nn.Linear(static_dim + dynamic_dim, node_hidden)
        self.layers = nn.ModuleList([GNSLayer(node_hidden, edge_hidden) for _ in range(n_layers)])
        self.decode = nn.Linear(node_hidden, dynamic_dim)

    def step(self, static_nodes, dynamic_nodes, edge_index):
        """One physical timestep -- AR, like R-U-Net, so it shares the
        mid-rollout-resume capability, on top of the paper-reported
        long-horizon error-plateau advantage."""
        n_nodes = static_nodes.shape[1]
        h = self.encode(torch.cat([static_nodes, dynamic_nodes], dim=-1))
        for layer in self.layers:
            h = layer(h, edge_index, n_nodes)
        delta = self.decode(h)
        return dynamic_nodes + delta  # predict the CHANGE, not the raw field (standard GNS convention)

    def rollout(self, static_nodes, dynamic_init, edge_index, n_steps):
        dynamic = dynamic_init
        outputs = []
        for _ in range(n_steps):
            dynamic = self.step(static_nodes, dynamic, edge_index)
            outputs.append(dynamic)
        return torch.stack(outputs, dim=1)  # (B, n_steps, n_nodes, dynamic_dim)


def field_to_nodes(field_stack):
    """(B, C, H, W) -> (B, H*W, C) node feature layout."""
    B, C, H, W = field_stack.shape
    return field_stack.permute(0, 2, 3, 1).reshape(B, H * W, C)


def train_step(model, opt, static_nodes, dynamic_init, edge_index, target_traj):
    model.train()
    opt.zero_grad()
    n_steps = target_traj.shape[1]
    pred_traj = model.rollout(static_nodes, dynamic_init, edge_index, n_steps)
    loss = F.mse_loss(pred_traj, target_traj)
    loss.backward()
    opt.step()
    return loss.item()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W, n_steps = 3, 24, 24, 6
    device = "cpu"
    edge_index = build_grid_edges(H, W, device)
    n_nodes = H * W
    n_edges = edge_index.shape[1]
    print(f"grid {H}x{W} -> {n_nodes} nodes, {n_edges} edges (4-connected, matches solver face connectivity)")

    model = GNS(static_dim=3, dynamic_dim=2, node_hidden=24, edge_hidden=24, n_layers=3)
    static_field = torch.rand(B, 3, H, W)
    dynamic_field = torch.rand(B, 2, H, W)
    static_nodes = field_to_nodes(static_field)
    dynamic_init = field_to_nodes(dynamic_field)
    target_traj = torch.rand(B, n_steps, n_nodes, 2)

    print("=== rollout shape check ===")
    pred_traj = model.rollout(static_nodes, dynamic_init, edge_index, n_steps)
    assert pred_traj.shape == (B, n_steps, n_nodes, 2)
    print(f"OK: rollout shape {tuple(pred_traj.shape)}")

    print("=== single-step AR + mid-rollout resume check ===")
    half = model.rollout(static_nodes, dynamic_init, edge_index, n_steps // 2)
    resumed = model.step(static_nodes, half[:, -1], edge_index)
    assert resumed.shape == (B, n_nodes, 2)
    print("OK: resumed mid-rollout, same capability as R-U-Net")

    print("=== gradient flow check ===")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    l0 = train_step(model, opt, static_nodes, dynamic_init, edge_index, target_traj)
    l1 = train_step(model, opt, static_nodes, dynamic_init, edge_index, target_traj)
    print(f"loss step0={l0:.4f} step1={l1:.4f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"param count: {n_params:,}")
    print("PASS")
