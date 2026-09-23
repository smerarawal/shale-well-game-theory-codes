"""
finn.py -- FINN (Finite Volume Neural Network; Karlbauer et al., arXiv:
2111.11798, 2022). Network structure directly MIRRORS the finite-volume
method: explicit control volumes (grid cells), explicit flux exchange
between neighbors that is enforced to be antisymmetric (what leaves cell
i through a face equals what enters cell j through that same face) --
conservation is hard-coded into the architecture, not hoped-for from
training data. Only the constitutive relationship (what the flux actually
IS, as a function of the state) is learned; the accounting (how fluxes
combine into a cell update) is fixed, closed-form finite-volume
bookkeeping identical in structure to solver_v3.py's own update step.

Per the spec: this hard-coded conservation structure is why FINN has the
strongest published out-of-distribution generalization of anything on
the surrogate list -- a model that CAN'T violate conservation even on
inputs unlike its training data is a different reliability class from one
that only tends not to, learned end-to-end. The explicit
flux-antisymmetry check below is the one property every other file in
this set does NOT enforce structurally (FNO/U-Net/GNS/etc. can all learn
to violate conservation; only FINN's forward pass makes it impossible).

Run: python finn.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedFlux(nn.Module):
    """Constitutive relation: given the two neighboring cells' states,
    predicts the flux crossing their shared face. This is the ONLY learned
    part of FINN -- everything downstream (how fluxes update cell state)
    is fixed accounting, not learned."""

    def __init__(self, state_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * state_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1))

    def forward(self, state_left, state_right):
        """state_left, state_right: (..., state_dim) states of the two
        cells sharing a face, always called in a FIXED (e.g. west-to-east)
        orientation so the SAME flux value can be reused with a sign flip
        for the neighbor's own conservation equation -- see
        conservative_update below."""
        h = torch.cat([state_left, state_right], dim=-1)
        return self.net(h).squeeze(-1)


class FINN(nn.Module):
    def __init__(self, state_dim=2, static_dim=3):
        """state_dim: (pressure, saturation) -- the conserved-ish quantities
        being updated. static_dim: kx, ky, well_mask, fed alongside state
        into the flux net so the learned flux can depend on permeability."""
        super().__init__()
        self.flux_net = LearnedFlux(state_dim + static_dim)

    def compute_face_fluxes(self, state, static, axis):
        """axis=2: E-W faces (grid rows fixed, columns adjacent).
        axis=3: N-S faces. Returns flux at every INTERNAL face, computed
        ONCE per face (not once per cell) -- this is what makes the
        antisymmetry structural rather than approximate: cell i's outflow
        through a face and cell j's inflow through that SAME face are
        literally the same tensor value, just added with opposite sign,
        never two independently-learned numbers that happen to be trained
        toward agreeing."""
        combined = torch.cat([state, static], dim=1)  # (B, state_dim+static_dim, H, W)
        if axis == 2:
            left = combined[:, :, :-1, :]
            right = combined[:, :, 1:, :]
        else:
            left = combined[:, :, :, :-1]
            right = combined[:, :, :, 1:]
        left = left.permute(0, 2, 3, 1)   # (B, ., ., C) for the MLP
        right = right.permute(0, 2, 3, 1)
        flux = self.flux_net(left, right)  # (B, ., .) -- one value per internal face
        return flux

    def conservative_update(self, state, static, dt, source):
        """Fixed finite-volume accounting: cell state changes by
        -(flux_out - flux_in) * dt / volume + source, using the SAME face
        flux tensors for both neighboring cells (antisymmetric by
        construction, not by loss penalty). Structurally identical to how
        solver_v3.py accumulates E/W/N/S face fluxes into a cell update."""
        B, C, H, W = state.shape
        flux_x = self.compute_face_fluxes(state, static, axis=2)  # (B, H-1, W)
        flux_y = self.compute_face_fluxes(state, static, axis=3)  # (B, H, W-1)

        div = torch.zeros(B, H, W, device=state.device)
        # cell i loses flux_x[i] to its south neighbor, that neighbor gains
        # the SAME value -- enforced by construction, not two learned copies
        div[:, :-1, :] += flux_x
        div[:, 1:, :] -= flux_x
        div[:, :, :-1] += flux_y
        div[:, :, 1:] -= flux_y

        state_flat = state[:, 0]  # apply to the first state channel (pressure-like) for this scaffold
        state_next0 = state_flat - div * dt + source * dt
        state_next = torch.cat([state_next0.unsqueeze(1), state[:, 1:]], dim=1)
        return state_next

    def rollout(self, state0, static, source, dt, n_steps):
        state = state0
        outputs = []
        for _ in range(n_steps):
            state = self.conservative_update(state, static, dt, source)
            outputs.append(state)
        return torch.stack(outputs, dim=1)


def check_global_conservation(finn, state0, static, dt):
    """The property FINN is built to guarantee: with no source/sink and
    no-flow (zero) boundary, total mass in the domain should be EXACTLY
    conserved to floating-point precision after one step -- not
    approximately, not 'small error', exactly, because flux antisymmetry
    is structural. This is the test the other 10 architectures in this
    set cannot pass by construction (they could only pass it by luck or
    by an explicit penalty term that never reaches exactly zero)."""
    zero_source = torch.zeros_like(state0[:, 0])
    total_before = state0[:, 0].sum(dim=(1, 2))
    state_next = finn.conservative_update(state0, static, dt, zero_source)
    total_after = state_next[:, 0].sum(dim=(1, 2))
    return total_before, total_after


def train_step(model, opt, state0, static, source, dt, target_traj):
    model.train()
    opt.zero_grad()
    n_steps = target_traj.shape[1]
    pred_traj = model.rollout(state0, static, source, dt, n_steps)
    loss = F.mse_loss(pred_traj, target_traj)
    loss.backward()
    opt.step()
    return loss.item()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, nx, ny, n_steps = 3, 32, 32, 5
    model = FINN(state_dim=2, static_dim=3)
    state0 = torch.rand(B, 2, nx, ny)
    static = torch.rand(B, 3, nx, ny)
    source = torch.zeros(B, nx, ny)

    print("=== rollout shape check ===")
    target_traj = torch.rand(B, n_steps, 2, nx, ny)
    pred_traj = model.rollout(state0, static, source, dt=0.01, n_steps=n_steps)
    assert pred_traj.shape == (B, n_steps, 2, nx, ny)
    print(f"OK: rollout shape {tuple(pred_traj.shape)}")

    print("=== STRUCTURAL conservation check (the property unique to this architecture) ===")
    total_before, total_after = check_global_conservation(model, state0, static, dt=0.01)
    diff = (total_after - total_before).abs()
    print(f"total (pressure-channel) mass before: {total_before.tolist()}")
    print(f"total (pressure-channel) mass after:  {total_after.tolist()}")
    print(f"max abs diff: {diff.max().item():.2e} (should be ~1e-6 or tighter, float32 roundoff only)")
    assert diff.max().item() < 1e-4, "flux antisymmetry should make this exact to floating-point precision"
    print("PASS: conservation holds structurally, before any training")

    print("=== gradient flow check ===")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    l0 = train_step(model, opt, state0, static, source, 0.01, target_traj)
    l1 = train_step(model, opt, state0, static, source, 0.01, target_traj)
    print(f"loss step0={l0:.4f} step1={l1:.4f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"param count: {n_params:,} (small -- only the constitutive flux relation is learned)")
    print("PASS")
