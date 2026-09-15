"""
eval_utils.py -- Step 3 of surrogate_comparison_spec.md.

Every architecture gets evaluated with EXACTLY this function on the SAME
held-out test set -- no per-architecture custom metrics, no cherry-picking
which numbers to report. Reports mean/median/max relative L2 (not just
mean, which hides worst-case failures), per-frame breakdown for trajectory
outputs, inference speed, and parameter count.

Import and call full_evaluation(model, test_loader, ...) from
run_surrogate_comparison.py -- not meant to be run standalone.
"""
import time
import torch
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def relative_l2_per_sample(pred, target, eps=1e-8):
    """
    pred, target: (B, C, H, W) or (B, T, C, H, W) for trajectory outputs.
    Returns per-sample relative L2 error, flattening all dims except batch.
    """
    B = pred.shape[0]
    pred_flat = pred.reshape(B, -1)
    target_flat = target.reshape(B, -1)
    num = torch.norm(pred_flat - target_flat, dim=1)
    den = torch.norm(target_flat, dim=1) + eps
    return (num / den)


def relative_l2_by_frame(pred, target, eps=1e-8):
    """
    For trajectory-output models only. pred, target: (B, T, C, H, W).
    Returns a (T,) array of mean relative L2 error at each timestep,
    across the whole batch -- this is what answers "are early frames
    artificially easy, inflating the averaged number" with evidence.
    """
    T = pred.shape[1]
    errors = []
    for t in range(T):
        pred_t = pred[:, t].reshape(pred.shape[0], -1)
        target_t = target[:, t].reshape(target.shape[0], -1)
        num = torch.norm(pred_t - target_t, dim=1)
        den = torch.norm(target_t, dim=1) + eps
        errors.append((num / den).mean().item())
    return np.array(errors)


def full_evaluation(model, test_loader, device, is_trajectory=False, n_timing_batches=5):
    """
    model: trained model in eval() mode.
    test_loader: yields (x, y) batches. y is (B, C, H, W) normally, or
                 (B, T, C, H, W) if is_trajectory=True.
    Returns a dict matching the spec's required metric set exactly.
    """
    model.eval()
    all_rel_l2 = []
    frame_errors_accum = []
    inference_times = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            t0 = time.time()
            pred = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            if batch_idx < n_timing_batches:
                inference_times.append(elapsed / x.shape[0])

            rel_l2 = relative_l2_per_sample(pred, y)
            all_rel_l2.extend(rel_l2.cpu().numpy().tolist())

            if is_trajectory:
                frame_errors_accum.append(relative_l2_by_frame(pred, y))

    all_rel_l2 = np.array(all_rel_l2)

    results = {
        "mean_rel_l2": float(all_rel_l2.mean()),
        "median_rel_l2": float(np.median(all_rel_l2)),
        "max_rel_l2": float(all_rel_l2.max()),
        "n_parameters": count_parameters(model),
        "inference_ms_per_sample": float(np.mean(inference_times) * 1000) if inference_times else None,
    }

    if is_trajectory and frame_errors_accum:
        results["rel_l2_by_frame"] = np.mean(np.stack(frame_errors_accum), axis=0).tolist()
        # per the spec's Step 5 priority #2: for trajectory models, the
        # FINAL frame's error is what actually drives EUR-payoff accuracy
        # downstream -- surface it explicitly, don't bury it in the average
        results["final_frame_rel_l2"] = results["rel_l2_by_frame"][-1]

    return results
