"""
near_well_hybrid.py -- Intrusive hybrid / near-well ML model, per the
taxonomy in "A Machine-Learned Near-Well Model in OPM Flow" (arXiv
2601.11193): full-replacement (every other file in this directory) /
non-intrusive hybrid (ML corrects simulator output after the fact, e.g.
PoroTwin) / intrusive hybrid (ML embedded INSIDE the simulator, replacing
only one sub-piece).

This file is the third kind: a small local net replaces only the
near-well upscaling correction (the standard reservoir-sim trick of
using an analytic Peaceman-type correction near a well because the grid
is too coarse to resolve the true near-well pressure/saturation
gradient), while the rest of the timestep -- global pressure solve,
saturation transport -- still runs through solver_v3.py exactly as-is.
The net only ever sees a small patch and only ever touches one term.

Per the spec: NOT currently needed, since the existing full-replacement
FNO is already fast enough for current call volume (Shapley/Blotto don't
need per-cell near-well accuracy, they need a fast scalar payoff). Built
anyway per this session's scaffold-everything decision, but the honest
framing stays deliberately-not-needed rather than a gap.

Run: python near_well_hybrid.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NearWellCorrectionNet(nn.Module):
    """Small local net: patch of coarse-grid fields in, a scalar (or small
    field) correction to the well-cell pressure/rate relationship out.
    Deliberately tiny -- it replaces ONE closed-form correction term, not
    a whole physics module."""

    def __init__(self, in_ch=4, patch_size=9, hidden=32):
        super().__init__()
        self.patch_size = patch_size
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(hidden, 1)  # single scalar correction factor

    def forward(self, patch):
        """patch: (B, in_ch, patch_size, patch_size) -- coarse-grid kx, ky,
        well rate, and the solver's own coarse-cell pressure, cropped
        around one well. Returns a multiplicative correction to that
        well's Peaceman-equivalent index (dimensionless, centered near 1
        via a softplus so it starts as a near-identity correction)."""
        h = self.net(patch).flatten(1)
        return F.softplus(self.head(h)) + 0.5  # centered so init correction ~= 1.0-ish


def extract_well_patch(field_stack, well_rc, patch_size):
    """field_stack: (B, C, H, W). well_rc: (B, 2) integer row/col. Crops
    the patch_size x patch_size window centered on each well -- the ONLY
    data this hybrid model is allowed to see, by design (intrusive-hybrid
    means small footprint, not basin-wide context)."""
    B, C, H, W = field_stack.shape
    half = patch_size // 2
    patches = []
    for b in range(B):
        r, c = well_rc[b].tolist()
        r0 = max(0, min(H - patch_size, r - half))
        c0 = max(0, min(W - patch_size, c - half))
        patches.append(field_stack[b:b + 1, :, r0:r0 + patch_size, c0:c0 + patch_size])
    return torch.cat(patches, dim=0)


def apply_correction_in_solver_step(coarse_well_pressure, correction_factor, target_rate):
    """Stand-in for the actual intrusive hook: solver_v3.py's well-source
    term would call this instead of its closed-form Peaceman index at
    exactly the point it currently computes the well-cell source. Kept as
    a free function (not a method) so it's obvious this plugs into
    EXISTING solver code, not a replacement solver."""
    return target_rate * correction_factor


def train_step(model, opt, patches, target_correction):
    """Trained against a reference: correction factor that makes the
    coarse-grid well-cell pressure match a locally-refined (or analytic
    Peaceman) fine-grid solution -- i.e. supervised on the SAME quantity
    the closed-form correction currently approximates, so this is a
    drop-in comparison, not a new target."""
    model.train()
    opt.zero_grad()
    pred = model(patches)
    loss = F.mse_loss(pred, target_correction)
    loss.backward()
    opt.step()
    return loss.item()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, nx, ny, patch = 5, 64, 64, 9
    model = NearWellCorrectionNet(in_ch=4, patch_size=patch)
    field_stack = torch.rand(B, 4, nx, ny)  # kx, ky, well_mask, coarse pressure
    well_rc = torch.tensor([[32, 32], [10, 10], [50, 50], [5, 58], [60, 4]])

    print("=== patch extraction + forward shape check ===")
    patches = extract_well_patch(field_stack, well_rc, patch)
    assert patches.shape == (B, 4, patch, patch)
    correction = model(patches)
    assert correction.shape == (B, 1)
    assert (correction > 0).all(), "correction factor must stay positive (it multiplies a rate)"
    print(f"OK: patches {tuple(patches.shape)}, correction range [{correction.min():.3f}, {correction.max():.3f}]")

    print("=== intrusive-hook stand-in check ===")
    target_rate = torch.tensor([-0.8, -0.5, -1.2, -0.3, -0.6]).unsqueeze(-1)
    corrected = apply_correction_in_solver_step(None, correction, target_rate)
    assert corrected.shape == target_rate.shape
    print(f"OK: correction applied to well rates -> {corrected.flatten().tolist()}")

    print("=== gradient flow check ===")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    target_correction = torch.rand(B, 1) * 0.5 + 0.8
    l0 = train_step(model, opt, patches, target_correction)
    l1 = train_step(model, opt, patches, target_correction)
    print(f"loss step0={l0:.4f} step1={l1:.4f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"param count: {n_params:,} (tiny by design -- one sub-piece, not a solver)")
    print("PASS")
