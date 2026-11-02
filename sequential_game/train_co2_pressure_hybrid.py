"""Train pressure-residual hybrids around frozen CO2 surrogate checkpoints.

For each base GNN/CCS checkpoint, this trains a correction network for a
solver-derived stage pressure prior. ``frozen`` fixes mobility at stage start;
``path`` uses the frozen base model's predicted endpoint saturation. The latter
is a CO2 prior2 analogue. Other state fields remain predictions from the base
model, so this experiment isolates the pressure-prior effect.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

try:
    from . import train_co2_xz_surrogate as base
    from .co2_pressure_prior import pressure_prior_batch, pressure_prior_one
except ImportError:
    import train_co2_xz_surrogate as base
    from co2_pressure_prior import pressure_prior_batch, pressure_prior_one


def _cat(cohorts):
    return {key: np.concatenate([d[key] for d in cohorts.values()])
            for key in cohorts[next(iter(cohorts))]}


def _load_base(checkpoint, device):
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    model = base.make_model(blob["backbone"], int(blob["width"])).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    stats = {k: np.asarray(v, dtype=np.float32) for k, v in blob["stats"].items()}
    return model, stats, float(blob["rate_scale"]), blob


def _predict_base(model, data, stats, rate_scale, device, dynamic=None):
    delta = base.predict_delta(model, data, stats, rate_scale, device, dynamic=dynamic)
    return delta, base._to_states(data, delta, dynamic)


def _lite_channels(data, dynamic=None):
    """Cellwise total mobility and density-weighted mobility proxy features."""
    sg_m = (dynamic or {}).get("sg_mobile", data["sg_mobile_in"])
    sg_t = (dynamic or {}).get("sg_trapped", data["sg_trapped_in"])
    sw = 1.0 - sg_m - sg_t
    sew = np.clip((sw - 0.15) / 0.85, 0.0, 1.0)
    seg = np.clip(sg_m / 0.85, 0.0, 1.0)
    lam_w = sew ** 2 / 1.0e-3
    lam_g = seg ** 2 / 6.0e-5
    total = lam_w + lam_g
    grav = (1030.0 * lam_w + 650.0 * lam_g)
    return np.stack((np.log(np.maximum(total, 1e-12)),
                     np.log(np.maximum(grav, 1e-12))), axis=1).astype(np.float32)


def _prior(data, mode, n_sub, stage_days, device, base_model, base_stats, rate_scale):
    if mode == "lite":
        return np.zeros_like(data["p_pa_in"], dtype=np.float32)
    endpoint = None
    if mode == "path":
        _, endpoint = _predict_base(base_model, data, base_stats, rate_scale, device)
    return pressure_prior_batch(data, n_sub=n_sub, mode=mode, endpoint=endpoint,
                                stage_days=stage_days)


def _augmented_x(data, mode, n_sub, stage_days, device, base_model,
                 base_stats, rate_scale):
    prior = _prior(data, mode, n_sub, stage_days, device, base_model,
                   base_stats, rate_scale)
    x = base.build_x(data, rate_scale)
    return np.concatenate((x, prior[:, None]), axis=1), prior


def _eval_one_step(model, xstats, ystats, prior, data, base_model,
                   base_stats, rate_scale, device, args):
    base_delta, states = _predict_base(base_model, data, base_stats, rate_scale, device)
    extra = (_lite_channels(data) if args.mode == "lite" else prior[:, None])
    x = np.concatenate((base.build_x(data, rate_scale), extra,
                        base_delta[:, :1]), axis=1)
    xn = torch.from_numpy(((x - xstats["x_mean"]) / xstats["x_std"]).astype(np.float32)).to(device)
    with torch.no_grad():
        corr = model(xn).cpu().numpy() * ystats["y_std"] + ystats["y_mean"]
    states["p_pa"] = data["p_pa_in"] + prior + corr[:, 0]
    truth = {k: data[f"{k}_out"] for k in base.STATE_KEYS}
    return base.score_states(states, truth, data, args)


def _eval_rollout(model, xstats, ystats, data, mode, n_sub, stage_days,
                  base_model, base_stats, rate_scale, device, args):
    keys = list(zip(data["geology_seed"].tolist(), data["episode_id"].tolist()))
    order = np.lexsort((data["stage_index"], data["episode_id"], data["geology_seed"]))
    pred_rows, true_rows, in_rows, selected, stage_rows = (
        {k: [] for k in base.STATE_KEYS}, {k: [] for k in base.STATE_KEYS},
        {k: [] for k in base.STATE_KEYS}, [], [])
    start = 0
    while start < len(order):
        group, key = [], keys[order[start]]
        while start < len(order) and keys[order[start]] == key:
            group.append(order[start]); start += 1
        group.sort(key=lambda j: int(data["stage_index"][j]))
        if len(group) != 6:
            continue
        state = {k: data[f"{k}_in"][group[0]:group[0] + 1].copy() for k in base.STATE_KEYS}
        for j in group:
            one = {k: v[j:j + 1] for k, v in data.items()
                   if isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == len(data["geology_seed"])}
            for k in base.STATE_KEYS:
                one[f"{k}_in"] = state[k]
                in_rows[k].append(state[k][0].copy())
            base_delta, base_state = _predict_base(base_model, one, base_stats,
                                                    rate_scale, device, state)
            endpoint = base_state if mode == "path" else None
            if mode == "lite":
                prior = np.zeros_like(state["p_pa"], dtype=np.float32)
                lite = _lite_channels(one, state)
                x = np.concatenate((base.build_x(one, rate_scale, state), lite,
                                    base_delta[:, :1]), axis=1)
            else:
                prior = pressure_prior_one(one, 0, n_sub=n_sub, mode=mode,
                                           endpoint=endpoint, stage_days=stage_days)[None]
                x = np.concatenate((base.build_x(one, rate_scale, state), prior[:, None],
                                    base_delta[:, :1]), axis=1)
            xn = torch.from_numpy(((x - xstats["x_mean"]) / xstats["x_std"]).astype(np.float32)).to(device)
            with torch.no_grad():
                corr = model(xn).cpu().numpy() * ystats["y_std"] + ystats["y_mean"]
            base_state["p_pa"] = state["p_pa"] + prior + corr[:, 0]
            state = base_state
            for k in base.STATE_KEYS:
                pred_rows[k].append(state[k][0].copy())
                true_rows[k].append(data[f"{k}_out"][j])
            selected.append(j)
            stage_rows.append(int(data["stage_index"][j]))
    pred = {k: np.stack(v) for k, v in pred_rows.items()}
    truth = {k: np.stack(v) for k, v in true_rows.items()}
    subset = {k: v[selected] for k, v in data.items()
              if isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == len(data["geology_seed"])}
    for k in base.STATE_KEYS:
        subset[f"{k}_in"] = np.stack(in_rows[k])
    overall = base.score_states(pred, truth, subset, args)
    by_stage = {}
    stage_rows = np.asarray(stage_rows)
    for stage in sorted(np.unique(stage_rows).tolist()):
        mask = stage_rows == stage
        stage_data = {k: v[mask] for k, v in subset.items()}
        stage_pred = {k: v[mask] for k, v in pred.items()}
        stage_truth = {k: v[mask] for k, v in truth.items()}
        by_stage[str(stage)] = base.score_states(stage_pred, stage_truth,
                                                 stage_data, args)
    overall["by_stage"] = by_stage
    return overall


def _prepare_split(data, mode, n_sub, stage_days, device,
                   base_model, base_stats, rate_scale, base_delta=None, base_states=None):
    x = base.build_x(data, rate_scale)
    if base_delta is None:
        base_delta, base_states = _predict_base(base_model, data, base_stats, rate_scale, device)
    endpoint = base_states if mode == "path" else None
    if mode == "lite":
        prior = np.zeros_like(data["p_pa_in"], dtype=np.float32)
        extra = _lite_channels(data)
    else:
        prior = pressure_prior_batch(data, n_sub=n_sub, mode=mode,
                                     endpoint=endpoint, stage_days=stage_days)
        extra = prior[:, None]
    residual = base.build_y(data)[:, :1].copy()
    residual[:, 0] -= prior
    if mode == "lite":
        extra = np.concatenate((extra, base_delta[:, :1]), axis=1)
    else:
        extra = np.concatenate((extra, base_delta[:, :1]), axis=1)
    return np.concatenate((x, extra), axis=1), residual


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--base-dir", required=True, help="folder with co2_{backbone}_s{seed}.pt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--backbones", nargs="+", choices=["gnn", "ccs"], default=["gnn", "ccs"])
    ap.add_argument("--modes", nargs="+", choices=["lite", "frozen", "path"], default=["lite", "frozen", "path"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-sub", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--learning-rate", type=float, default=8e-4)
    ap.add_argument("--stage-days", type=float, default=30.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device, out_dir = torch.device(args.device), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {split: {c: base.load_cohort(args.data_dir, split, c) for c in base.COHORTS}
              for split in ("train", "validation", "test")}
    summary_path = out_dir / "pressure_hybrid_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
    done = {(r["backbone"], r["mode"], r["seed"]) for r in summary}
    for backbone in args.backbones:
        for seed in args.seeds:
            base_path = Path(args.base_dir) / f"co2_{backbone}_s{seed}.pt"
            if not base_path.exists():
                raise FileNotFoundError(f"Missing base checkpoint: {base_path}")
            base_model, base_stats, rate_scale, base_blob = _load_base(base_path, device)
            precomputed = {}
            for split, cohorts in splits.items():
                precomputed[split] = {}
                for cohort, data in cohorts.items():
                    bd, bs = _predict_base(base_model, data, base_stats, rate_scale, device)
                    precomputed[split][cohort] = (bd, bs)
            train_data = _cat(splits["train"])
            val_data = _cat(splits["validation"])
            test_data = _cat(splits["test"])
            train_delta = np.concatenate([precomputed["train"][c][0] for c in base.COHORTS])
            train_states = {k: np.concatenate([precomputed["train"][c][1][k] for c in base.COHORTS])
                            for k in base.STATE_KEYS}
            for mode in args.modes:
                if (backbone, mode, seed) in done:
                    continue
                t0 = time.time()
                torch.manual_seed(seed + 7000); np.random.seed(seed + 7000)
                xtr, ytr = _prepare_split(train_data, mode, args.n_sub, args.stage_days,
                                          device, base_model, base_stats, rate_scale,
                                          train_delta, train_states)
                xva, yva = _prepare_split(val_data, mode, args.n_sub, args.stage_days,
                                          device, base_model, base_stats, rate_scale)
                xstats = {"x_mean": xtr.mean((0, 2, 3), keepdims=True).astype(np.float32),
                          "x_std": xtr.std((0, 2, 3), keepdims=True).clip(1e-6).astype(np.float32)}
                ystats = {"y_mean": ytr.mean((0, 2, 3), keepdims=True).astype(np.float32),
                          "y_std": ytr.std((0, 2, 3), keepdims=True).clip(1e-6).astype(np.float32)}
                xn = ((xtr - xstats["x_mean"]) / xstats["x_std"]).astype(np.float32)
                yn = ((ytr - ystats["y_mean"]) / ystats["y_std"]).astype(np.float32)
                model = base.make_model(backbone, args.width, cin=xtr.shape[1], cout=1).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
                best, best_epoch, stale = float("inf"), 0, 0
                checkpoint = out_dir / f"co2_{backbone}_{mode}_s{seed}.pt"
                for epoch in range(1, args.epochs + 1):
                    model.train(); order = np.random.permutation(len(xn)); loss_sum = 0.0
                    for start in range(0, len(order), args.batch_size):
                        ids = order[start:start + args.batch_size]
                        xb = torch.from_numpy(xn[ids]).to(device)
                        yb = torch.from_numpy(yn[ids]).to(device)
                        pred = model(xb)
                        loss = F.mse_loss(pred, yb)
                        opt.zero_grad(set_to_none=True); loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                        loss_sum += float(loss.detach()) * len(ids)
                    model.eval(); val_loss = 0.0
                    with torch.no_grad():
                        for start in range(0, len(xva), args.batch_size):
                            xb = torch.from_numpy(((xva[start:start + args.batch_size]
                                                   - xstats["x_mean"]) / xstats["x_std"]).astype(np.float32)).to(device)
                            yb = (yva[start:start + args.batch_size] - ystats["y_mean"]) / ystats["y_std"]
                            val_loss += float(F.mse_loss(model(xb), torch.from_numpy(yb).to(device), reduction="sum"))
                    score = val_loss / max(yva.size, 1)
                    if score < best:
                        best, best_epoch, stale = score, epoch, 0
                        torch.save({"state_dict": model.state_dict(), "backbone": backbone,
                                    "mode": mode, "seed": seed, "n_sub": args.n_sub,
                                    "width": args.width, "xstats": {k: v.tolist() for k, v in xstats.items()},
                                    "ystats": {k: v.tolist() for k, v in ystats.items()},
                                    "base_checkpoint": str(base_path)}, checkpoint)
                    else:
                        stale += 1
                    if epoch == 1 or epoch % 5 == 0:
                        print(f"{backbone}/{mode}/s{seed} ep={epoch} train={loss_sum/max(len(xn),1):.5g} "
                              f"val={score:.5g} best={best:.5g}", flush=True)
                    if stale >= args.patience:
                        break
                blob = torch.load(checkpoint, map_location=device, weights_only=False)
                model.load_state_dict(blob["state_dict"])
                xstats = {k: np.asarray(v, np.float32) for k, v in blob["xstats"].items()}
                ystats = {k: np.asarray(v, np.float32) for k, v in blob["ystats"].items()}
                item = {"backbone": backbone, "mode": mode, "seed": seed,
                        "n_sub": args.n_sub, "best_epoch": best_epoch,
                        "train_seconds": round(time.time() - t0), "best_val_mse": best,
                        "one_step_test": {}, "autoregressive_test": {}}
                for cohort, data in splits["test"].items():
                    prior = _prior(data, mode, args.n_sub, args.stage_days, device,
                                   base_model, base_stats, rate_scale)
                    item["one_step_test"][cohort] = _eval_one_step(
                        model, xstats, ystats, prior, data, base_model, base_stats,
                        rate_scale, device, SimpleNamespace(stage_days=args.stage_days,
                                                           mode=mode))
                    item["autoregressive_test"][cohort] = _eval_rollout(
                        model, xstats, ystats, data, mode, args.n_sub, args.stage_days,
                        base_model, base_stats, rate_scale, device,
                        SimpleNamespace(stage_days=args.stage_days))
                summary.append(item)
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                (out_dir / f"co2_{backbone}_{mode}_s{seed}_metrics.json").write_text(
                    json.dumps(item, indent=2), encoding="utf-8")
                print(f"Saved {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
