"""CO2-only PyTorch architecture and training utilities.

These are standalone extensions for the 24x24 CO2 x-z transition task. They do
not modify the existing CCS/CCS-GNN implementation or any brine code. The
FourierMIONet2D class is a compact, stagewise adaptation inspired by the
Fourier-MIONet paper; it is not a reproduction of the paper's 3D DeepXDE model.

Tensor convention: all field tensors use [batch, channel, z, x]. Controls are
per-case vectors [batch, n_controls]; include well-location maps in geology
channels if well positions can vary. Pressure-aware loss thresholds must be
computed from the TRAIN split only and passed in the same normalized scale as
the model's pressure target channel.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler


def _group_count(channels: int, max_groups: int = 8) -> int:
    """Choose a GroupNorm divisor that works for arbitrary channel widths."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SpectralConv2d(nn.Module):
    """Low-mode Fourier convolution for [B,C,Z,X] fields."""

    def __init__(self, in_channels: int, out_channels: int,
                 modes_z: int = 8, modes_x: int = 8) -> None:
        super().__init__()
        if modes_z < 1 or modes_x < 1:
            raise ValueError("Fourier mode counts must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_z = modes_z
        self.modes_x = modes_x
        scale = 1.0 / max(1, in_channels * out_channels) ** 0.5
        shape = (in_channels, out_channels, modes_z, modes_x)
        self.weight_pos = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_neg = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def _mix(x_ft: Tensor, weight: Tensor) -> Tensor:
        return torch.einsum("bihw,iohw->bohw", x_ft, weight)

    def forward(self, x: Tensor) -> Tensor:
        b, _, nz, nx = x.shape
        # Use float32 for FFT support and numerical consistency under AMP.
        x32 = x.float()
        x_ft = torch.fft.rfft2(x32, norm="ortho")
        out_ft = torch.zeros(
            b, self.out_channels, nz, nx // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        mz = min(self.modes_z, max(1, (nz - 1) // 2))
        mx = min(self.modes_x, nx // 2 + 1)
        out_ft[:, :, :mz, :mx] = self._mix(
            x_ft[:, :, :mz, :mx], self.weight_pos[:, :, :mz, :mx]
        )
        out_ft[:, :, -mz:, :mx] = self._mix(
            x_ft[:, :, -mz:, :mx], self.weight_neg[:, :, :mz, :mx]
        )
        y = torch.fft.irfft2(out_ft, s=(nz, nx), norm="ortho")
        return y.to(dtype=x.dtype)


class _TwoScaleCorrection(nn.Module):
    """Small encoder/decoder path that restores local spatial detail."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.full = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
        )
        self.coarse = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
        )
        self.mix = nn.Conv2d(2 * width, width, 1)

    def forward(self, x: Tensor) -> Tensor:
        local = self.full(x)
        coarse = F.avg_pool2d(x, kernel_size=2, ceil_mode=True)
        coarse = self.coarse(coarse)
        coarse = F.interpolate(coarse, size=x.shape[-2:], mode="bilinear",
                               align_corners=False)
        return self.mix(torch.cat((local, coarse), dim=1))


class _FourierUNetBlock(nn.Module):
    def __init__(self, width: int, modes_z: int, modes_x: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_z, modes_x)
        self.local = nn.Conv2d(width, width, kernel_size=1)
        self.multiscale = _TwoScaleCorrection(width)
        self.norm = nn.GroupNorm(num_groups=_group_count(width), num_channels=width)

    def forward(self, x: Tensor) -> Tensor:
        y = self.spectral(x) + self.local(x) + self.multiscale(x)
        return x + F.gelu(self.norm(y))


def _coordinates(batch: int, nz: int, nx: int, stage: Optional[Tensor],
                 device: torch.device, dtype: torch.dtype) -> Tensor:
    z = torch.linspace(-1.0, 1.0, nz, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, nx, device=device, dtype=dtype)
    zz, xx = torch.meshgrid(z, x, indexing="ij")
    coords = torch.stack((zz, xx), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
    if stage is None:
        stage_map = torch.zeros(batch, 1, nz, nx, device=device, dtype=dtype)
    else:
        stage = torch.as_tensor(stage, device=device, dtype=dtype).reshape(batch, -1)
        stage_map = stage[:, :1, None, None].expand(-1, 1, nz, nx)
    return torch.cat((coords, stage_map), dim=1)


class FourierMIONet2D(nn.Module):
    """Multi-input Fourier operator for one CO2 stage transition.

    Inputs are separate geology and dynamic-state fields plus per-case controls.
    A scalar/stage trunk is combined with the field branches, and a Fourier plus
    multiscale decoder predicts the next state. With ``residual=True`` the head
    predicts a state increment and returns ``state + increment``. Use matching
    normalized input/output conventions from the existing CO2 trainer.
    """

    def __init__(self, n_geology: int, n_state: int, n_controls: int,
                 width: int = 32, modes_z: int = 8, modes_x: int = 8,
                 n_operator_blocks: int = 2, residual: bool = True) -> None:
        super().__init__()
        if min(n_geology, n_state, n_controls, width, n_operator_blocks) < 1:
            raise ValueError("Input sizes, width, and block count must be positive")
        self.n_state = n_state
        self.residual = residual
        self.geology_branch = nn.Sequential(
            nn.Conv2d(n_geology, width, 1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
        )
        self.state_branch = nn.Sequential(
            nn.Conv2d(n_state, width, 1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
        )
        self.control_branch = nn.Sequential(
            nn.Linear(n_controls, width), nn.GELU(), nn.Linear(width, width)
        )
        self.trunk = nn.Sequential(
            nn.Conv2d(3, width, 1), nn.GELU(), nn.Conv2d(width, width, 1)
        )
        self.branch_project = nn.Conv2d(width, width, 1)
        self.operator_blocks = nn.Sequential(*[
            _FourierUNetBlock(width, modes_z, modes_x)
            for _ in range(n_operator_blocks)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, n_state, 1),
        )

    def forward(self, geology: Tensor, state: Tensor, controls: Tensor,
                stage: Optional[Tensor] = None) -> Tensor:
        if geology.ndim != 4 or state.ndim != 4 or controls.ndim != 2:
            raise ValueError("Expected geology/state [B,C,Z,X], controls [B,C]")
        if geology.shape[0] != state.shape[0] or state.shape[1] != self.n_state:
            raise ValueError("Incompatible batch or state channel dimensions")
        if geology.shape[-2:] != state.shape[-2:]:
            raise ValueError("Geology and state grids must match")
        b, _, nz, nx = state.shape
        branch = (self.geology_branch(geology) + self.state_branch(state)
                  + self.control_branch(controls).unsqueeze(-1).unsqueeze(-1))
        coords = _coordinates(b, nz, nx, stage, state.device, state.dtype)
        trunk = self.trunk(coords)
        latent = self.branch_project(branch * trunk)
        delta = self.head(self.operator_blocks(latent))
        return state + delta if self.residual else delta


class _DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels), nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class CompactUNet2D(nn.Module):
    """Compact U-Net baseline with the same stage-transition call signature."""

    def __init__(self, n_geology: int, n_state: int, n_controls: int,
                 width: int = 24, residual: bool = True) -> None:
        super().__init__()
        self.n_state = n_state
        self.residual = residual
        self.control_channels = n_controls
        self.enc1 = _DoubleConv(n_geology + n_state + n_controls + 3, width)
        self.enc2 = _DoubleConv(width, 2 * width)
        self.mid = _DoubleConv(2 * width, 4 * width)
        self.dec2 = _DoubleConv(6 * width, 2 * width)
        self.dec1 = _DoubleConv(3 * width, width)
        self.head = nn.Conv2d(width, n_state, 1)

    def forward(self, geology: Tensor, state: Tensor, controls: Tensor,
                stage: Optional[Tensor] = None) -> Tensor:
        if geology.ndim != 4 or state.ndim != 4 or controls.ndim != 2:
            raise ValueError("Expected geology/state [B,C,Z,X], controls [B,C]")
        b, _, nz, nx = state.shape
        if controls.shape != (b, self.control_channels):
            raise ValueError("Unexpected control vector shape")
        coords = _coordinates(b, nz, nx, stage, state.device, state.dtype)
        control_map = controls[:, :, None, None].expand(-1, -1, nz, nx)
        x0 = torch.cat((geology, state, control_map, coords), dim=1)
        e1 = self.enc1(x0)
        e2 = self.enc2(F.avg_pool2d(e1, 2, ceil_mode=True))
        mid = self.mid(F.avg_pool2d(e2, 2, ceil_mode=True))
        d2 = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat((d1, e1), dim=1))
        out = self.head(d1)
        return state + out if self.residual else out


class PressureAwareStateLoss(nn.Module):
    """State loss with a training-derived high-pressure tail and peak term.

    ``pressure_tail_threshold`` and ``channel_scales`` use the same normalized
    space as the model tensors. Determine them from train data only. This loss
    does not encode the exploratory 5 MPa game cap or claim physical safety.
    """

    def __init__(self, pressure_channel: int,
                 pressure_tail_threshold: float,
                 channel_scales: Optional[Sequence[float]] = None,
                 tail_weight: float = 2.0, peak_weight: float = 0.5,
                 smooth_l1_beta: float = 1.0) -> None:
        super().__init__()
        if tail_weight < 0 or peak_weight < 0 or smooth_l1_beta <= 0:
            raise ValueError("Loss weights must be nonnegative and beta positive")
        self.pressure_channel = pressure_channel
        self.pressure_tail_threshold = float(pressure_tail_threshold)
        self.tail_weight = float(tail_weight)
        self.peak_weight = float(peak_weight)
        self.smooth_l1_beta = float(smooth_l1_beta)
        if channel_scales is None:
            self.register_buffer("channel_scales", torch.empty(0))
        else:
            scales = torch.as_tensor(channel_scales, dtype=torch.float32)
            if torch.any(scales <= 0):
                raise ValueError("All channel scales must be positive")
            self.register_buffer("channel_scales", scales)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError("Prediction and target must match [B,C,Z,X]")
        if not 0 <= self.pressure_channel < prediction.shape[1]:
            raise ValueError("pressure_channel is outside the output channels")
        if self.channel_scales.numel():
            if self.channel_scales.numel() != prediction.shape[1]:
                raise ValueError("channel_scales must have one value per channel")
            scales = self.channel_scales.to(prediction).view(1, -1, 1, 1)
        else:
            scales = torch.ones(1, prediction.shape[1], 1, 1,
                                device=prediction.device, dtype=prediction.dtype)

        err = (prediction - target) / scales
        base = F.smooth_l1_loss(err, torch.zeros_like(err), beta=self.smooth_l1_beta)
        p_scale = scales[:, self.pressure_channel:self.pressure_channel + 1]
        p_pred = prediction[:, self.pressure_channel:self.pressure_channel + 1]
        p_true = target[:, self.pressure_channel:self.pressure_channel + 1]
        high = (p_true >= self.pressure_tail_threshold).to(prediction.dtype)
        tail = (((p_pred - p_true) / p_scale).square() * (1.0 + self.tail_weight * high)).mean()
        pred_peak = p_pred.amax(dim=(-2, -1))
        true_peak = p_true.amax(dim=(-2, -1))
        peak = (((pred_peak - true_peak) / p_scale.flatten()[0]).square()).mean()
        return base + self.tail_weight * tail + self.peak_weight * peak


def make_pressure_weighted_sampler(
    pressure_max_pa: Tensor,
    geology_ids: Optional[Tensor] = None,
    quantile: float = 0.90,
    max_oversample_boost: float = 4.0,
    num_samples: Optional[int] = None,
) -> tuple[WeightedRandomSampler, float]:
    """Build a training-only sampler that emphasizes high-pressure samples.

    If ``geology_ids`` is supplied, each geology first receives equal base
    weight, then high-pressure samples within it receive the oversampling
    boost. Never pass validation or test labels to this helper.
    Returns the sampler and the train-derived threshold in the input units.
    """
    pmax = torch.as_tensor(pressure_max_pa, dtype=torch.float64).flatten().cpu()
    if pmax.numel() == 0 or not torch.isfinite(pmax).all():
        raise ValueError("pressure_max_pa must be nonempty and finite")
    if not 0.0 < quantile < 1.0 or max_oversample_boost < 0:
        raise ValueError("quantile must be in (0,1); boost must be nonnegative")
    threshold = float(torch.quantile(pmax, quantile))
    span = max(float(pmax.max() - threshold), 1e-12)
    pressure_factor = 1.0 + max_oversample_boost * (
        (pmax - threshold).clamp_min(0.0) / span
    ).clamp_max(1.0)

    weights = pressure_factor
    if geology_ids is not None:
        gids = torch.as_tensor(geology_ids).flatten().cpu()
        if gids.numel() != pmax.numel():
            raise ValueError("geology_ids and pressure_max_pa lengths must match")
        _, inverse, counts = torch.unique(gids, return_inverse=True,
                                          return_counts=True)
        weights = pressure_factor / counts[inverse].to(pressure_factor.dtype)
    if float(weights.sum()) <= 0:
        raise ValueError("Sampler weights must have positive sum")
    sampler = WeightedRandomSampler(
        weights=weights.double(),
        num_samples=int(num_samples or weights.numel()),
        replacement=True,
    )
    return sampler, threshold


def unroll3_loss(model: nn.Module, geology: Tensor, initial_state: Tensor,
                 controls: Tensor, targets: Tensor, loss_fn: nn.Module,
                 stages: Optional[Tensor] = None,
                 steps: int = 3) -> tuple[Tensor, Tensor]:
    """Autoregressive multi-step loss for stagewise transition models.

    Args:
        geology: [B,Cg,Z,X], held fixed through the rollout.
        initial_state: [B,Cs,Z,X], state before the first transition.
        controls: [B,T,Ccontrol], one control vector per transition.
        targets: [B,T,Cs,Z,X], solver states after each transition.
        loss_fn: per-step state loss, e.g. PressureAwareStateLoss.
        stages: optional normalized or embedded stage values [B,T].
        steps: number of autoregressive transitions; use 3 for unroll-3.

    Gradients flow through all rollout steps. This increases memory use; train
    with appropriately small batches and do not substitute teacher forcing here.
    """
    if controls.ndim != 3 or targets.ndim != 5:
        raise ValueError("controls must be [B,T,C], targets [B,T,C,Z,X]")
    if controls.shape[:2] != targets.shape[:2] or steps < 1 or steps > targets.shape[1]:
        raise ValueError("Invalid rollout length or mismatched sequence dimensions")
    if stages is not None and stages.shape[:2] != targets.shape[:2]:
        raise ValueError("stages must have shape [B,T]")
    state = initial_state
    losses = []
    predictions = []
    for t in range(steps):
        stage_t = None if stages is None else stages[:, t]
        pred = model(geology, state, controls[:, t], stage_t)
        truth = targets[:, t]
        losses.append(loss_fn(pred, truth))
        predictions.append(pred)
        state = pred
    return torch.stack(losses).mean(), torch.stack(predictions, dim=1)


__all__ = [
    "CompactUNet2D",
    "FourierMIONet2D",
    "PressureAwareStateLoss",
    "SpectralConv2d",
    "count_parameters",
    "make_pressure_weighted_sampler",
    "unroll3_loss",
]
