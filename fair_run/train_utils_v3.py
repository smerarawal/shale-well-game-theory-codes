"""
train_utils_v3.py -- the REFINED loss function. This is the actual fix
for ccsnet (and, less visibly, every other architecture) underperforming:
plain unweighted MSE on stacked [pressure, saturation] lets whichever
channel has larger raw magnitude dominate the gradient -- pressure in
this solver spans roughly [-15, +10] (compressible two-phase flow, see
solver_v3.py's own docstring on why it's not elliptic/bounded), while
saturation is bounded to [Swc, 1-Sor] ~ [0.2, 0.8]. Unweighted MSE
effectively trains almost entirely on pressure and lets saturation ride
along on whatever capacity is left -- a LOSS-FUNCTION problem, not purely
a capacity one, though undersized models (ccsnet at width=16) make it
worse by leaving even less spare capacity for the underweighted channel.

Fix: per-channel INVERSE-VARIANCE weighting, computed once from the
TRAIN split (never test -- would leak test statistics into the loss),
exactly the same principle as feature standardization, applied to the
loss instead of the data so every architecture's existing output
activation (raw pressure, sigmoid saturation) stays untouched.
"""
import time
import torch
import torch.nn.functional as F


def compute_channel_weights(train_pressure, train_saturation):
    """train_pressure, train_saturation: (N, H, W) numpy or tensor, TRAIN
    split only. Returns (w_pressure, w_saturation) with w_i = 1/var_i,
    normalized so they sum to 2 (keeps the combined loss on a similar
    overall scale to plain MSE, so existing LR choices stay roughly
    sensible -- avoids silently requiring a new LR sweep on top of the
    loss-function change)."""
    var_p = float(train_pressure.var()) + 1e-8
    var_s = float(train_saturation.var()) + 1e-8
    w_p_raw, w_s_raw = 1.0 / var_p, 1.0 / var_s
    total = w_p_raw + w_s_raw
    return 2.0 * w_p_raw / total, 2.0 * w_s_raw / total


def weighted_channel_loss(pred, target, w_pressure, w_saturation):
    """pred, target: (B, 2, H, W). Returns (total, mse_pressure, mse_saturation)
    -- the per-channel values are reported separately in results, not just
    folded into one number, so a reader can see exactly where an
    architecture's error actually lives (the 'well-rounded' part of the
    comparison)."""
    mse_p = F.mse_loss(pred[:, 0], target[:, 0])
    mse_s = F.mse_loss(pred[:, 1], target[:, 1])
    total = w_pressure * mse_p + w_saturation * mse_s
    return total, mse_p, mse_s


def relative_l2_per_sample(pred, target, eps=1e-8):
    B = pred.shape[0]
    num = torch.norm((pred - target).reshape(B, -1), dim=1)
    den = torch.norm(target.reshape(B, -1), dim=1) + eps
    return num / den


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_loader, device, epochs, lr, w_pressure, w_saturation,
                 weight_decay=1e-4, physics_loss_fn=None, physics_weight=0.0):
    """Shared training loop for every architecture. physics_loss_fn, if
    given, is called as physics_loss_fn(pred, x) -> scalar residual and
    added at physics_weight -- used only for physics_informed_fno and
    pi_convlstm; every other architecture trains with physics_weight=0,
    which exactly reduces this to the plain weighted-MSE loop (same
    required consistency property fno_physics_informed.py already enforces)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss, mse_p, mse_s = weighted_channel_loss(pred, y, w_pressure, w_saturation)
            if physics_loss_fn is not None and physics_weight > 0:
                loss = loss + physics_weight * physics_loss_fn(pred, x)
            loss.backward()
            opt.step()
        scheduler.step()
    return model, time.time() - t0


def evaluate_model(model, test_loader, device, n_timing_batches=5):
    """Reports combined AND per-channel relative L2 -- the 'well-rounded'
    metric set the user asked for, not just one averaged number that can
    hide a model doing well on pressure and badly on saturation (or vice
    versa) behind an OK-looking mean."""
    model.eval()
    combined, pressure_only, saturation_only = [], [], []
    inference_times = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            t0 = time.time()
            pred = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            if batch_idx < n_timing_batches:
                inference_times.append(elapsed / x.shape[0])
            combined.extend(relative_l2_per_sample(pred, y).cpu().tolist())
            pressure_only.extend(relative_l2_per_sample(pred[:, 0:1], y[:, 0:1]).cpu().tolist())
            saturation_only.extend(relative_l2_per_sample(pred[:, 1:2], y[:, 1:2]).cpu().tolist())
    import numpy as np
    combined, pressure_only, saturation_only = map(np.array, (combined, pressure_only, saturation_only))
    return {
        "mean_rel_l2_combined": float(combined.mean()),
        "median_rel_l2_combined": float(np.median(combined)),
        "max_rel_l2_combined": float(combined.max()),
        "mean_rel_l2_pressure": float(pressure_only.mean()),
        "mean_rel_l2_saturation": float(saturation_only.mean()),
        "n_parameters": count_parameters(model),
        "inference_ms_per_sample": float(np.mean(inference_times) * 1000) if inference_times else None,
    }
