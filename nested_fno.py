"""
nested_fno.py -- Nested FNO (Wen et al., Energy & Environmental Science
16(4), 2023). Two FNOs at different spatial scales feeding into each
other: a COARSE FNO sees the whole basin at low resolution (captures
long-range pressure interference between wells cheaply), a FINE FNO runs
only on a small patch around each well at full resolution (captures the
sharp near-well saturation gradient the coarse model can't resolve). The
fine model's input is conditioned on the coarse model's output, upsampled
and cropped to the patch -- coarse solve is reused, not redone.

Directly relevant to the flagged trajectory-FNO issue: the current
trajectory FNO spends the same per-cell network capacity everywhere, even
though accuracy only matters near wells (where EUR-relevant saturation
gradients live) and basin-wide grid cells far from any well are nearly
uniform. Nested FNO is a structural fix, not just a bigger network.

Run: python nested_fno.py
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


class SmallFNO(nn.Module):
    """One FNO tower, reused for both the coarse and fine branches (same
    block, different resolution/modes/channel width)."""

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


class NestedFNO(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, coarse_size=32, patch_size=16,
                 coarse_width=24, fine_width=32, coarse_modes=10, fine_modes=12):
        """coarse_size: side length the full basin is downsampled to before
        the coarse FNO. patch_size: side length of the near-well window the
        fine FNO refines, cropped from the full-resolution grid."""
        super().__init__()
        self.coarse_size = coarse_size
        self.patch_size = patch_size
        self.coarse_fno = SmallFNO(in_ch, out_ch, coarse_width, coarse_modes)
        # fine FNO sees: original fields on the patch + upsampled coarse output on the patch
        self.fine_fno = SmallFNO(in_ch + out_ch, out_ch, fine_width, fine_modes)

    def forward(self, x, well_centers):
        """x: (B, in_ch, H, W) full-resolution basin fields.
        well_centers: (B, 2) integer (row, col) center of the patch to
        refine per batch item (e.g. the injector/producer location driving
        the payoff-relevant cell)."""
        B, C, H, W = x.shape
        x_coarse = F.interpolate(x, size=(self.coarse_size, self.coarse_size), mode="bilinear", align_corners=False)
        coarse_out = self.coarse_fno(x_coarse)                      # (B, out_ch, coarse, coarse)
        coarse_up = F.interpolate(coarse_out, size=(H, W), mode="bilinear", align_corners=False)

        p = self.patch_size
        half = p // 2
        patches_in, patches_coarse, boxes = [], [], []
        for b in range(B):
            r, c = well_centers[b].tolist()
            r0 = max(0, min(H - p, r - half))
            c0 = max(0, min(W - p, c - half))
            patches_in.append(x[b:b + 1, :, r0:r0 + p, c0:c0 + p])
            patches_coarse.append(coarse_up[b:b + 1, :, r0:r0 + p, c0:c0 + p])
            boxes.append((r0, c0))
        patch_in = torch.cat(patches_in, dim=0)
        patch_coarse = torch.cat(patches_coarse, dim=0)
        fine_input = torch.cat([patch_in, patch_coarse], dim=1)
        fine_out = self.fine_fno(fine_input)                        # (B, out_ch, p, p)

        # stitch fine patch back over the coarse-upsampled full field
        full_out = coarse_up.clone()
        for b in range(B):
            r0, c0 = boxes[b]
            full_out[b:b + 1, :, r0:r0 + p, c0:c0 + p] = fine_out[b:b + 1]
        return full_out, coarse_out, fine_out


def train_step(model, opt, x, well_centers, target_full, coarse_target, fine_weight=2.0):
    """Two loss terms: coarse FNO supervised directly on a downsampled
    target (so it doesn't just become a free-floating conditioning input),
    fine patch supervised at full resolution with higher weight since
    that's where EUR-relevant accuracy lives."""
    model.train()
    opt.zero_grad()
    full_out, coarse_out, fine_out = model(x, well_centers)
    loss_coarse = F.mse_loss(coarse_out, coarse_target)
    p = model.patch_size
    half = p // 2
    fine_targets = []
    for b in range(x.shape[0]):
        r, c = well_centers[b].tolist()
        H, W = target_full.shape[-2:]
        r0 = max(0, min(H - p, r - half))
        c0 = max(0, min(W - p, c - half))
        fine_targets.append(target_full[b:b + 1, :, r0:r0 + p, c0:c0 + p])
    fine_target = torch.cat(fine_targets, dim=0)
    loss_fine = F.mse_loss(fine_out, fine_target)
    loss = loss_coarse + fine_weight * loss_fine
    loss.backward()
    opt.step()
    return loss.item(), loss_coarse.item(), loss_fine.item()


if __name__ == "__main__":
    torch.manual_seed(0)
    B, nx, ny = 3, 64, 64
    model = NestedFNO(in_ch=3, out_ch=1, coarse_size=32, patch_size=16)
    x = torch.rand(B, 3, nx, ny)
    well_centers = torch.tensor([[32, 32], [20, 45], [50, 10]])
    target_full = torch.randn(B, 1, nx, ny)
    coarse_target = F.interpolate(target_full, size=(32, 32), mode="bilinear", align_corners=False)

    print("=== forward shape check ===")
    full_out, coarse_out, fine_out = model(x, well_centers)
    assert full_out.shape == (B, 1, nx, ny)
    assert coarse_out.shape == (B, 1, 32, 32)
    assert fine_out.shape == (B, 1, 16, 16)
    print(f"OK: full={tuple(full_out.shape)} coarse={tuple(coarse_out.shape)} fine={tuple(fine_out.shape)}")

    print("=== gradient flow check ===")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    l0 = train_step(model, opt, x, well_centers, target_full, coarse_target)
    l1 = train_step(model, opt, x, well_centers, target_full, coarse_target)
    print(f"loss step0={l0} step1={l1}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"param count: {n_params:,}")
    print("PASS")
