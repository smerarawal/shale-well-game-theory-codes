"""Train CO2-specific one-stage surrogate backbones on split-by-geology data.

This is deliberately separate from train_stage_surrogate.py: it predicts pressure,
mobile/trapped gas, historical maximum gas saturation, and dissolved CO2 for the
experimental CO2 x-z solver. The first comparison is teacher-forced one-step
training with six-stage autoregressive test rollouts. It does not use the brine
pressure prior or claim field accuracy.

Example:
  python train_co2_xz_surrogate.py --data-dir /kaggle/working/co2_xz_dataset \
      --out-dir /kaggle/working/co2_surrogate_runs --backbones unet ccs gnn fno ufno \
      --seeds 0 1 2 --epochs 60 --device cuda
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


COHORTS = ("friendly", "intermediate", "challenging", "tarbert", "upper_ness")
STATE_KEYS = ("p_pa", "sg_mobile", "sg_trapped", "sg_max", "c_dissolved_kg_m3")
SWC = 0.15
RHO_CO2 = 650.0


def _block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                         nn.GroupNorm(min(8, cout), cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1),
                         nn.GroupNorm(min(8, cout), cout), nn.GELU())


class _Decoder(nn.Module):
    def __init__(self, width, cout):
        super().__init__()
        self.b1 = _block(width * 6, width * 2)
        self.b0 = _block(width * 3, width)
        self.out = nn.Conv2d(width, cout, 1)

    def forward(self, f0, f1, f2):
        x = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.b1(torch.cat((x, f1), 1))
        x = F.interpolate(x, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.b0(torch.cat((x, f0), 1)))


class _Encoder(nn.Module):
    def __init__(self, cin, width):
        super().__init__()
        self.e0 = _block(cin, width)
        self.e1 = _block(width, width * 2, 2)
        self.e2 = _block(width * 2, width * 4, 2)

    def forward(self, x):
        f0 = self.e0(x)
        f1 = self.e1(f0)
        f2 = self.e2(f1)
        return f0, f1, f2


class UNet(nn.Module):
    def __init__(self, cin, width, cout):
        super().__init__()
        self.enc = _Encoder(cin, width)
        self.dec = _Decoder(width, cout)

    def forward(self, x):
        return self.dec(*self.enc(x))


class CCSNet(nn.Module):
    """Shared encoder with distinct pressure and CO2-state decoders."""
    def __init__(self, cin, width, cout):
        super().__init__()
        self.enc = _Encoder(cin, width)
        self.pressure = _Decoder(width, 1)
        self.co2 = _Decoder(width, cout - 1)

    def forward(self, x):
        f = self.enc(x)
        return torch.cat((self.pressure(*f), self.co2(*f)), 1)


def _shift(x, dz, dx):
    h, w = x.shape[-2:]
    p = F.pad(x, (1, 1, 1, 1))
    return p[..., 1 + dz:1 + dz + h, 1 + dx:1 + dx + w]


class GridGNN(nn.Module):
    """Four-neighbour message passing with permeability-jump edge features."""
    def __init__(self, cin, width, cout, layers=5):
        super().__init__()
        self.enc = nn.Conv2d(cin, width, 1)
        self.msg = nn.ModuleList([nn.ModuleList([
            nn.Sequential(nn.Conv2d(width * 2 + 2, width, 1), nn.GELU(),
                          nn.Conv2d(width, width, 1)) for _ in range(4)])
            for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Conv2d(width * 5, width, 1),
                                                nn.GELU(), nn.Conv2d(width, width, 1),
                                                nn.GELU()) for _ in range(layers)])
        self.head = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(),
                                  nn.Conv2d(width, cout, 1))
        self.dirs = ((0, 1), (0, -1), (1, 0), (-1, 0))

    def forward(self, x):
        h = self.enc(x)
        kx, kz = x[:, 0:1], x[:, 1:2]
        for layers in zip(self.msg, self.upd):
            msg_layers, update = layers
            messages = []
            for (dz, dx), net in zip(self.dirs, msg_layers):
                neigh = _shift(h, dz, dx)
                mask = _shift(torch.ones_like(kx), dz, dx)
                edge = torch.cat((((_shift(kx, dz, dx) - kx).abs() * mask),
                                  ((_shift(kz, dz, dx) - kz).abs() * mask)), 1)
                messages.append(net(torch.cat((h, neigh, edge), 1)))
            h = h + update(torch.cat((h, *messages), 1))
        return self.head(h)


class SpectralConv2d(nn.Module):
    def __init__(self, cin, cout, modes=6):
        super().__init__()
        self.cin, self.cout, self.modes = cin, cout, modes
        scale = 1.0 / max(1, cin * cout)
        self.w_top = nn.Parameter(scale * torch.randn(cin, cout, modes, modes, 2))
        self.w_bottom = nn.Parameter(scale * torch.randn(cin, cout, modes, modes, 2))

    def forward(self, x):
        b, _, h, w = x.shape
        xf = torch.fft.rfft2(x, norm="ortho")
        yf = torch.zeros((b, self.cout, h, w // 2 + 1), device=x.device, dtype=xf.dtype)
        mz, mx = min(self.modes, h // 2), min(self.modes, w // 2 + 1)
        wt = torch.view_as_complex(self.w_top[:, :, :mz, :mx].contiguous())
        wb = torch.view_as_complex(self.w_bottom[:, :, :mz, :mx].contiguous())
        yf[:, :, :mz, :mx] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :mz, :mx], wt)
        yf[:, :, -mz:, :mx] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -mz:, :mx], wb)
        return torch.fft.irfft2(yf, s=(h, w), norm="ortho")


class FNO(nn.Module):
    def __init__(self, cin, width, cout):
        super().__init__()
        self.lift = nn.Conv2d(cin, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width) for _ in range(4)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(4)])
        self.head = nn.Sequential(nn.Conv2d(width, width * 2, 1), nn.GELU(),
                                  nn.Conv2d(width * 2, cout, 1))

    def forward(self, x):
        x = self.lift(x)
        for spec, local in zip(self.spec, self.local):
            x = F.gelu(spec(x) + local(x))
        return self.head(x)


class UFNO(nn.Module):
    """Global FNO path fused with a local U-Net path."""
    def __init__(self, cin, width, cout):
        super().__init__()
        self.global_path = FNO(cin, width, width)
        self.local_path = UNet(cin, width, width)
        self.fuse = nn.Sequential(_block(width * 2, width), nn.Conv2d(width, cout, 1))

    def forward(self, x):
        return self.fuse(torch.cat((self.global_path(x), self.local_path(x)), 1))


def make_model(name, width):
    makers = {"unet": UNet, "ccs": CCSNet, "gnn": GridGNN,
              "fno": FNO, "ufno": UFNO}
    if name not in makers:
        raise ValueError(f"unknown backbone {name}")
    return makers[name](14, width, len(STATE_KEYS))


def _npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_cohort(data_dir, split, cohort):
    return _npz(Path(data_dir) / split / f"{cohort}.npz")


def build_x(data, rate_scale, dynamic=None):
    n, h, w = data["kx_mD"].shape
    dyn = dynamic or {key: data[f"{key}_in"] for key in STATE_KEYS}
    xcoord = np.broadcast_to(np.linspace(-1, 1, w, dtype=np.float32)[None, None, :], (n, h, w))
    zcoord = np.broadcast_to(np.linspace(-1, 1, h, dtype=np.float32)[None, :, None], (n, h, w))
    r0 = np.zeros((n, h, w), dtype=np.float32)
    r1 = np.zeros_like(r0)
    wz = h - 2
    wx0, wx1 = max(1, w // 3), min(w - 2, (2 * w) // 3)
    r0[:, wz, wx0] = data["rate_operator_0_m3_s"] / rate_scale
    r1[:, wz, wx1] = data["rate_operator_1_m3_s"] / rate_scale
    dx = np.log(np.maximum(data["dx_ft"], 1e-6)).astype(np.float32)[:, None, None]
    dz = np.log(np.maximum(data["dz_ft"], 1e-6)).astype(np.float32)[:, None, None]
    dx = np.broadcast_to(dx, (n, h, w)); dz = np.broadcast_to(dz, (n, h, w))
    return np.stack((np.log(np.maximum(data["kx_mD"], 1e-12)),
                     np.log(np.maximum(data["kz_mD"], 1e-12)), data["phi"],
                     dyn["p_pa"], dyn["sg_mobile"], dyn["sg_trapped"], dyn["sg_max"],
                     dyn["c_dissolved_kg_m3"] / 30.0, r0, r1,
                     xcoord, zcoord, dx, dz), axis=1).astype(np.float32)


def build_y(data):
    return np.stack([data[f"{key}_out"] - data[f"{key}_in"] for key in STATE_KEYS], axis=1).astype(np.float32)


def fit_stats(x, y):
    xm = x.mean(axis=(0, 2, 3), keepdims=True)
    xs = x.std(axis=(0, 2, 3), keepdims=True).clip(1e-6)
    ym = y.mean(axis=(0, 2, 3), keepdims=True)
    ys = y.std(axis=(0, 2, 3), keepdims=True).clip(1e-6)
    return {"x_mean": xm.astype(np.float32), "x_std": xs.astype(np.float32),
            "y_mean": ym.astype(np.float32), "y_std": ys.astype(np.float32)}


def _to_states(data, pred_delta, dynamic=None):
    base = dynamic or {key: data[f"{key}_in"] for key in STATE_KEYS}
    result = {}
    for i, key in enumerate(STATE_KEYS):
        result[key] = base[key] + pred_delta[:, i]
    result["sg_trapped"] = np.maximum(result["sg_trapped"], base["sg_trapped"])
    result["sg_trapped"] = np.clip(result["sg_trapped"], 0, 1 - SWC)
    result["sg_mobile"] = np.clip(result["sg_mobile"], 0, 1 - SWC - result["sg_trapped"])
    result["sg_max"] = np.clip(np.maximum.reduce((result["sg_max"], base["sg_max"],
                                                  result["sg_mobile"] + result["sg_trapped"])),
                                0, 1 - SWC)
    result["c_dissolved_kg_m3"] = np.clip(result["c_dissolved_kg_m3"], 0, 30.0)
    return result


@torch.no_grad()
def predict_delta(model, data, stats, rate_scale, device, batch=128, dynamic=None):
    x = build_x(data, rate_scale, dynamic)
    pred = []
    model.eval()
    for i in range(0, len(x), batch):
        xt = torch.from_numpy((x[i:i + batch] - stats["x_mean"]) / stats["x_std"]).to(device)
        pred.append(model(xt).cpu().numpy())
    dy = np.concatenate(pred) * stats["y_std"] + stats["y_mean"]
    return dy


def score_states(pred, truth, data, args):
    out = {}
    for i, key in enumerate(STATE_KEYS):
        e = pred[key] - truth[key]
        mse = float(np.mean(e ** 2))
        denom = float(np.sum((truth[key] - truth[key].mean()) ** 2)) + 1e-20
        out[f"{key}_rmse"] = float(np.sqrt(mse))
        out[f"{key}_mae"] = float(np.mean(np.abs(e)))
        out[f"{key}_max_abs"] = float(np.max(np.abs(e)))
        out[f"{key}_r2"] = float(1.0 - np.sum(e ** 2) / denom)
    ptrue, ppred = truth["p_pa"], pred["p_pa"]
    pmax_e = ppred.reshape(len(ppred), -1).max(1) - ptrue.reshape(len(ptrue), -1).max(1)
    out["pmax_abs_mean_pa"] = float(np.mean(np.abs(pmax_e)))
    out["pmax_bias_pa"] = float(np.mean(pmax_e))
    out["pmax_abs_p95_pa"] = float(np.percentile(np.abs(pmax_e), 95))
    top_errors, well_errors, plume_iou, plume_fraction = [], [], [], []
    h, w = ptrue.shape[-2:]
    well_z, well_x = h - 2, (max(1, w // 3), min(w - 2, (2 * w) // 3))
    for i in range(len(ptrue)):
        thresh = np.quantile(ptrue[i], 0.9)
        mask = ptrue[i] >= thresh
        top_errors.extend((ppred[i][mask] - ptrue[i][mask]).ravel().tolist())
        well_errors.extend((ppred[i, well_z, list(well_x)] - ptrue[i, well_z, list(well_x)]).tolist())
        a, b = pred["sg_mobile"][i] >= 0.01, truth["sg_mobile"][i] >= 0.01
        union = np.logical_or(a, b).sum()
        plume_iou.append(1.0 if union == 0 else np.logical_and(a, b).sum() / union)
        plume_fraction.append(abs(float(a.mean()) - float(b.mean())))
    out["pressure_top_decile_rmse_pa"] = float(np.sqrt(np.mean(np.square(top_errors))))
    out["pressure_well_cell_rmse_pa"] = float(np.sqrt(np.mean(np.square(well_errors))))
    out["plume_iou_sg_ge_0p01"] = float(np.mean(plume_iou))
    out["plume_fraction_abs_error"] = float(np.mean(plume_fraction))
    phi = data["phi"]
    cell_m3 = (data["dx_ft"] * 0.3048) * (data["dy_ft"] * 0.3048) * (data["dz_ft"] * 0.3048)
    def mass(s):
        pv = phi * cell_m3[:, None, None]
        gas = RHO_CO2 * pv * (s["sg_mobile"] + s["sg_trapped"])
        water = pv * (1.0 - s["sg_mobile"] - s["sg_trapped"]) * s["c_dissolved_kg_m3"]
        return (gas + water).sum(axis=(1, 2))
    injected = RHO_CO2 * (data["rate_operator_0_m3_s"] + data["rate_operator_1_m3_s"]) * args.stage_days * 86400.0
    residual = mass(pred) - mass({key: data[f"{key}_in"] for key in STATE_KEYS}) - injected
    out["predicted_mass_balance_abs_mean_kg"] = float(np.mean(np.abs(residual)))
    out["predicted_mass_balance_abs_p95_kg"] = float(np.percentile(np.abs(residual), 95))
    return out


def _pred_batch(model, data, stats, rate_scale, device):
    delta = predict_delta(model, data, stats, rate_scale, device)
    return _to_states(data, delta)


def evaluate_one_step(model, datasets, stats, rate_scale, device, args):
    results = {}
    for cohort, data in datasets.items():
        truth = {key: data[f"{key}_out"] for key in STATE_KEYS}
        pred = _pred_batch(model, data, stats, rate_scale, device)
        results[cohort] = score_states(pred, truth, data, args)
    return results


def evaluate_rollout(model, datasets, stats, rate_scale, device, args):
    results = {}
    for cohort, data in datasets.items():
        keys = list(zip(data["geology_seed"].tolist(), data["episode_id"].tolist()))
        order = np.lexsort((data["stage_index"], data["episode_id"], data["geology_seed"]))
        pred_rows, true_rows, input_states, input_rows = ({k: [] for k in STATE_KEYS},
                                                          {k: [] for k in STATE_KEYS},
                                                          {k: [] for k in STATE_KEYS}, [])
        start = 0
        while start < len(order):
            first = order[start]
            group = []
            key = keys[first]
            while start < len(order) and keys[order[start]] == key:
                group.append(order[start]); start += 1
            group = sorted(group, key=lambda j: int(data["stage_index"][j]))
            if len(group) != 6:
                continue
            state = {k: data[f"{k}_in"][group[0]:group[0] + 1].copy() for k in STATE_KEYS}
            for j in group:
                one = {k: v[j:j + 1] for k, v in data.items()
                       if isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == len(data["geology_seed"])}
                for k in STATE_KEYS:
                    input_states[k].append(state[k][0].copy())
                dy = predict_delta(model, one, stats, rate_scale, device, dynamic=state)
                pstate = _to_states(one, dy, dynamic=state)
                for k in STATE_KEYS:
                    pred_rows[k].append(pstate[k][0])
                    true_rows[k].append(data[f"{k}_out"][j])
                input_rows.append(j)
                state = {k: pstate[k] for k in STATE_KEYS}
        if input_rows:
            ids = np.asarray(input_rows, dtype=int)
            pred = {k: np.stack(v) for k, v in pred_rows.items()}
            truth = {k: np.stack(v) for k, v in true_rows.items()}
            subset = {k: v[ids] for k, v in data.items()
                      if isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == len(data["geology_seed"])}
            for k in STATE_KEYS:
                subset[f"{k}_in"] = np.stack(input_states[k])
            results[cohort] = score_states(pred, truth, subset, args)
        else:
            results[cohort] = {"error": "no six-stage episodes found"}
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--backbones", nargs="+", default=["unet", "ccs", "gnn", "fno", "ufno"],
                    choices=["unet", "ccs", "gnn", "fno", "ufno"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=8e-4)
    ap.add_argument("--stage-days", type=float, default=30.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = {c: load_cohort(data_dir, "train", c) for c in COHORTS}
    val = {c: load_cohort(data_dir, "validation", c) for c in COHORTS}
    test = {c: load_cohort(data_dir, "test", c) for c in COHORTS}
    rate_scale = max(float(max(np.max(d["rate_operator_0_m3_s"]),
                               np.max(d["rate_operator_1_m3_s"])) for d in train.values()), 1e-12)
    x_parts = [build_x(d, rate_scale) for d in train.values()]
    y_parts = [build_y(d) for d in train.values()]
    x_train, y_train = np.concatenate(x_parts), np.concatenate(y_parts)
    stats = fit_stats(x_train, y_train)
    X = ((x_train - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
    Y = ((y_train - stats["y_mean"]) / stats["y_std"]).astype(np.float32)
    p_in = np.concatenate([d["p_pa_in"] for d in train.values()])
    best_metrics = []
    device = torch.device(args.device)
    for backbone in args.backbones:
        for seed in args.seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            model = make_model(backbone, args.width).to(device)
            params = sum(p.numel() for p in model.parameters())
            opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
            best, best_epoch, stale, t0 = float("inf"), 0, 0, time.time()
            ckpt = out_dir / f"co2_{backbone}_s{seed}.pt"
            val_X = {c: ((build_x(d, rate_scale) - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
                     for c, d in val.items()}
            val_Y = {c: ((build_y(d) - stats["y_mean"]) / stats["y_std"]).astype(np.float32)
                     for c, d in val.items()}
            for epoch in range(1, args.epochs + 1):
                model.train(); perm = np.random.permutation(len(X)); run = 0.0
                for i in range(0, len(perm), args.batch_size):
                    ids = perm[i:i + args.batch_size]
                    xb = torch.from_numpy(X[ids]).to(device)
                    yb = torch.from_numpy(Y[ids]).to(device)
                    pb = torch.from_numpy(p_in[ids]).to(device)
                    target_p = pb + torch.from_numpy(y_train[ids, 0]).to(device)
                    pred = model(xb)
                    loss_field = F.mse_loss(pred, yb)
                    p_pred = pb + (pred[:, 0] * stats["y_std"][0, 0, 0, 0] + stats["y_mean"][0, 0, 0, 0])
                    pmax_loss = F.mse_loss(p_pred.flatten(1).amax(1), target_p.flatten(1).amax(1)) / (stats["y_std"][0, 0, 0, 0] ** 2 + 1e-12)
                    loss = loss_field + 0.5 * pmax_loss
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                    run += float(loss.detach()) * len(ids)
                model.eval(); val_loss = 0.0; val_n = 0
                with torch.no_grad():
                    for c in COHORTS:
                        vx = torch.from_numpy(val_X[c]).to(device)
                        vy = torch.from_numpy(val_Y[c]).to(device)
                        for i in range(0, len(vx), args.batch_size):
                            pv = model(vx[i:i + args.batch_size])
                            val_loss += float(F.mse_loss(pv, vy[i:i + args.batch_size], reduction="sum"))
                            val_n += vy[i:i + args.batch_size].numel()
                score = val_loss / max(val_n, 1)
                if score < best:
                    best, best_epoch, stale = score, epoch, 0
                    torch.save({"state_dict": model.state_dict(), "backbone": backbone,
                                "width": args.width, "seed": seed, "params": params,
                                "rate_scale": rate_scale,
                                "stats": {k: v.tolist() for k, v in stats.items()},
                                "args": vars(args)}, ckpt)
                else:
                    stale += 1
                if epoch == 1 or epoch % 5 == 0:
                    print(f"{backbone} seed {seed} epoch {epoch}: train={run/len(X):.5f} "
                          f"val={score:.5f} best={best:.5f} ({best_epoch})", flush=True)
                if stale >= args.patience:
                    break
            blob = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(blob["state_dict"])
            item = {"backbone": backbone, "seed": seed, "params": params,
                    "best_epoch": best_epoch, "train_seconds": round(time.time() - t0),
                    "best_val_mse": best,
                    "one_step_test": evaluate_one_step(model, test, stats, rate_scale, device, args),
                    "autoregressive_test": evaluate_rollout(model, test, stats, rate_scale, device, args)}
            metric_path = out_dir / f"co2_{backbone}_s{seed}_metrics.json"
            metric_path.write_text(json.dumps(item, indent=2), encoding="utf-8")
            best_metrics.append(item)
            print(f"Saved {ckpt} and {metric_path}", flush=True)
    (out_dir / "run_summary.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    print(f"Completed {len(best_metrics)} model/seed runs in {out_dir}")


if __name__ == "__main__":
    main()
