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
import math
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
        self.co2 = _Decoder(width, cout - 1) if cout > 1 else None

    def forward(self, x):
        f = self.enc(x)
        p = self.pressure(*f)
        return p if self.co2 is None else torch.cat((p, self.co2(*f)), 1)


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

    def features(self, x):
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
        return h

    def forward(self, x):
        return self.head(self.features(x))


class CCSGNN(nn.Module):
    """Dual spatial branch: CCS encoder/decoder plus permeability-aware GNN."""
    def __init__(self, cin, width, cout):
        super().__init__()
        self.ccs_encoder = _Encoder(cin, width)
        self.gnn = GridGNN(cin, width, cout, layers=5)
        self.gnn.head = nn.Identity()  # fusion consumes node features directly
        self.fuse = _block(width * 2, width)
        self.pressure = _Decoder(width, 1)
        self.co2 = _Decoder(width, cout - 1) if cout > 1 else None

    def forward(self, x):
        f0, f1, f2 = self.ccs_encoder(x)
        g = self.gnn.features(x)
        f0 = self.fuse(torch.cat((f0, g), dim=1))
        p = self.pressure(f0, f1, f2)
        return p if self.co2 is None else torch.cat((p, self.co2(f0, f1, f2)), dim=1)


class ConvLSTMCell(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.gates = nn.Conv2d(channels * 2, channels * 4, kernel_size,
                               padding=kernel_size // 2)

    def forward(self, x, hidden=None):
        if hidden is None:
            h = torch.zeros_like(x); c = torch.zeros_like(x)
        else:
            h, c = hidden
        i, f, o, g = self.gates(torch.cat((x, h), dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, (h, c)


class CCSConvLSTM(nn.Module):
    """CCS encoder-decoder with a ConvLSTM state at the bottleneck."""
    def __init__(self, cin, width, cout):
        super().__init__()
        self.enc = _Encoder(cin, width)
        # _Encoder's bottleneck (f2) has 4 * width channels.
        self.temporal = ConvLSTMCell(width * 4)
        self.dec = _Decoder(width, cout)

    def step(self, x, hidden=None):
        f0, f1, f2 = self.enc(x)
        h, hidden = self.temporal(f2, hidden)
        return self.dec(f0, f1, h), hidden

    def forward(self, x):
        return self.step(x, None)[0]


class GNNConvLSTM(nn.Module):
    """Permeability-aware GNN features with recurrent stage memory."""
    def __init__(self, cin, width, cout):
        super().__init__()
        self.gnn = GridGNN(cin, width, cout)
        self.gnn.head = nn.Identity()  # recurrent head replaces the GNN output head
        self.temporal = ConvLSTMCell(width)
        self.head = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(),
                                  nn.Conv2d(width, cout, 1))

    def step(self, x, hidden=None):
        h, hidden = self.temporal(self.gnn.features(x), hidden)
        return self.head(h), hidden

    def forward(self, x):
        return self.step(x, None)[0]


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


def make_model(name, width, cin=14, cout=len(STATE_KEYS)):
    makers = {"unet": UNet, "ccs": CCSNet, "gnn": GridGNN,
              "fno": FNO, "ufno": UFNO, "ccs_gnn": CCSGNN,
              "ccs_lstm": CCSConvLSTM, "gnn_lstm": GNNConvLSTM}
    if name not in makers:
        raise ValueError(f"unknown backbone {name}")
    return makers[name](cin, width, cout)


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


def _sequence_windows(datasets, unroll):
    """Return row-index windows, keeping geology/schedule episodes intact."""
    windows, offset = [], 0
    for data in datasets.values():
        groups = {}
        for i, (geology, episode) in enumerate(zip(data["geology_seed"], data["episode_id"])):
            groups.setdefault((int(geology), int(episode)), []).append(i)
        for rows in groups.values():
            rows.sort(key=lambda j: int(data["stage_index"][j]))
            for start in range(max(0, len(rows) - unroll + 1)):
                window = rows[start:start + unroll]
                stages = [int(data["stage_index"][j]) for j in window]
                if stages == list(range(stages[0], stages[0] + unroll)):
                    windows.append([offset + j for j in window])
        offset += len(data["geology_seed"])
    return np.asarray(windows, dtype=np.int64)


def _torch_project(state, previous):
    """Inference projection used between pushforward training steps."""
    result = {k: state[:, i] for i, k in enumerate(STATE_KEYS)}
    result["sg_trapped"] = torch.maximum(result["sg_trapped"], previous[:, 2]).clamp(0.0, 1.0 - SWC)
    result["sg_mobile"] = result["sg_mobile"].clamp_min(0.0)
    result["sg_mobile"] = torch.minimum(result["sg_mobile"], 1.0 - SWC - result["sg_trapped"])
    result["sg_max"] = torch.maximum(result["sg_max"], previous[:, 3])
    result["sg_max"] = torch.maximum(result["sg_max"], result["sg_mobile"] + result["sg_trapped"])
    result["sg_max"] = result["sg_max"].clamp(0.0, 1.0 - SWC)
    land_cap = result["sg_max"] / (1.0 + 2.0 * result["sg_max"])
    result["sg_trapped"] = torch.minimum(result["sg_trapped"], land_cap)
    dissolved_cap = 30.0 * (1.0 - result["sg_mobile"] - result["sg_trapped"])
    result["c_dissolved_kg_m3"] = torch.minimum(
        result["c_dissolved_kg_m3"].clamp_min(0.0), dissolved_cap)
    return torch.stack([result[k] for k in STATE_KEYS], dim=1)


def _batch_loss(pred, y_target, state_in, pv, injected, y_std, y_mean,
                valid_weight, mass_weight, plume_weight=0.0,
                plume_threshold=0.01, plume_temperature=0.002,
                pmax_weight=0.5, pressure_tail_weight=0.0,
                pressure_tail_fraction=0.1, pressure_penalty_weight=0.0,
                pressure_cap_pa=None):
    """Field loss plus configurable game-pressure and physical penalties."""
    target_norm = (y_target - y_mean) / y_std
    loss_field = F.mse_loss(pred, target_norm)
    delta = pred * y_std + y_mean
    raw = state_in + delta
    target_out = state_in + y_target
    pmax_loss = (F.mse_loss(raw[:, 0].flatten(1).amax(1),
                            target_out[:, 0].flatten(1).amax(1))
                 / (y_std[0, 0, 0, 0].square() + 1e-12))
    pressure_tail_loss = pred.new_zeros(())
    pressure_penalty_loss = pred.new_zeros(())
    if pressure_tail_weight > 0:
        flat_p = raw[:, 0].flatten(1)
        flat_t = target_out[:, 0].flatten(1)
        count = max(1, int(math.ceil(flat_t.shape[1] * pressure_tail_fraction)))
        threshold = flat_t.topk(count, dim=1).values[:, -1:]
        mask = flat_t >= threshold
        scale2 = y_std[0, 0, 0, 0].square() + 1e-12
        pressure_tail_loss = (((flat_p - flat_t).square() * mask).sum(1)
                              / mask.sum(1).clamp_min(1) / scale2).mean()
    if pressure_penalty_weight > 0:
        if pressure_cap_pa is None or pressure_cap_pa <= 0:
            raise ValueError("pressure penalty loss requires a positive --pressure-cap-pa")
        pmax_pred = raw[:, 0].flatten(1).amax(1)
        pmax_true = target_out[:, 0].flatten(1).amax(1)
        scale = torch.clamp_min(y_std[0, 0, 0, 0], max(float(pressure_cap_pa), 1.0))
        exceed_pred = F.relu(pmax_pred - pressure_cap_pa) / scale
        exceed_true = F.relu(pmax_true - pressure_cap_pa) / scale
        pressure_penalty_loss = F.mse_loss(exceed_pred.square(), exceed_true.square())
    sm, st, smx, cd = raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4]
    valid = (F.relu(-sm).square().mean()
             + F.relu(state_in[:, 2] - st).square().mean()
             + F.relu(-st).square().mean()
             + F.relu(sm + st - (1.0 - SWC)).square().mean()
             + F.relu(state_in[:, 3] - smx).square().mean()
             + F.relu(sm + st - smx).square().mean()
             + F.relu(st - smx / (1.0 + 2.0 * smx.clamp_min(0.0))).square().mean()
             + F.relu(-cd / 30.0).square().mean()
             + F.relu((cd - 30.0 * (1.0 - sm - st)) / 30.0).square().mean())
    mass_out = (RHO_CO2 * pv * (sm + st) + pv * (1.0 - sm - st) * cd).sum((1, 2))
    mass_in = (RHO_CO2 * pv * (state_in[:, 1] + state_in[:, 2])
               + pv * (1.0 - state_in[:, 1] - state_in[:, 2]) * state_in[:, 4]).sum((1, 2))
    mass = (((mass_out - mass_in - injected) / injected.abs().clamp_min(1.0)) ** 2).mean()
    # Soft Dice aligns the mobile-CO2 support with the evaluation plume mask
    # while keeping the continuous state-field and physics losses active.
    plume_target = (target_out[:, 1] >= plume_threshold).to(raw.dtype)
    plume_prob = torch.sigmoid((raw[:, 1] - plume_threshold) /
                               max(plume_temperature, 1e-6))
    intersection = (plume_prob * plume_target).sum((1, 2))
    plume_dice = 1.0 - ((2.0 * intersection + 1e-6) /
                        (plume_prob.sum((1, 2)) + plume_target.sum((1, 2)) + 1e-6))
    plume_loss = plume_dice.mean()
    return (loss_field + pmax_weight * pmax_loss
            + pressure_tail_weight * pressure_tail_loss
            + pressure_penalty_weight * pressure_penalty_loss
            + valid_weight * valid + mass_weight * mass
            + plume_weight * plume_loss)


def _train_pushforward_epoch(model, optimizer, windows, sequence_data,
                             state_train, y_train, pv_train, inject_train,
                             rate_scale, stats, device, args):
    """Train consecutive stages, detaching each predicted state."""
    order = np.random.permutation(len(windows))
    y_std = torch.as_tensor(stats["y_std"], device=device)
    y_mean = torch.as_tensor(stats["y_mean"], device=device)
    run = 0.0
    for start in range(0, len(order), args.batch_size):
        seq = windows[order[start:start + args.batch_size]]
        dynamic = state_train[seq[:, 0]].copy()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step in range(args.rollout_unroll):
            ids = seq[:, step]
            sample = {key: value[ids] for key, value in sequence_data.items()}
            dyn = {key: dynamic[:, i] for i, key in enumerate(STATE_KEYS)}
            x = build_x(sample, rate_scale, dyn)
            x = ((x - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
            state_in = torch.from_numpy(dynamic).to(device)
            true_out = torch.from_numpy(state_train[ids] + y_train[ids]).to(device)
            target_delta = true_out - state_in
            pred = model(torch.from_numpy(x).to(device))
            losses.append(_batch_loss(
                pred, target_delta, state_in,
                torch.from_numpy(pv_train[ids]).to(device),
                torch.from_numpy(inject_train[ids]).to(device),
                y_std, y_mean, args.valid_loss_weight, args.mass_loss_weight,
                getattr(args, "plume_loss_weight", 0.0),
                getattr(args, "plume_threshold", 0.01),
                getattr(args, "plume_temperature", 0.002),
                getattr(args, "pmax_loss_weight", 0.5),
                getattr(args, "pressure_tail_loss_weight", 0.0),
                getattr(args, "pressure_tail_fraction", 0.1),
                getattr(args, "pressure_penalty_loss_weight", 0.0),
                getattr(args, "pressure_cap_pa", None)))
            raw_out = state_in + pred * y_std + y_mean
            dynamic = _torch_project(raw_out, state_in).detach().cpu().numpy()
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        run += float(loss.detach()) * len(seq)
    return run


def _to_states(data, pred_delta, dynamic=None):
    base = dynamic or {key: data[f"{key}_in"] for key in STATE_KEYS}
    result = {}
    for i, key in enumerate(STATE_KEYS):
        result[key] = base[key] + pred_delta[:, i]
    result["sg_trapped"] = np.maximum(result["sg_trapped"], base["sg_trapped"])
    result["sg_trapped"] = np.clip(result["sg_trapped"], 0, 1 - SWC)
    result["sg_max"] = np.clip(np.maximum.reduce((result["sg_max"], base["sg_max"],
                                                   result["sg_mobile"] + result["sg_trapped"])),
                                0, 1 - SWC)
    land_cap = result["sg_max"] / (1.0 + 2.0 * result["sg_max"])
    result["sg_trapped"] = np.minimum(result["sg_trapped"], land_cap)
    result["sg_mobile"] = np.clip(result["sg_mobile"], 0, 1 - SWC - result["sg_trapped"])
    result["sg_max"] = np.clip(np.maximum(result["sg_max"],
                                           result["sg_mobile"] + result["sg_trapped"]),
                                0, 1 - SWC)
    dissolved_cap = 30.0 * (1.0 - result["sg_mobile"] - result["sg_trapped"])
    result["c_dissolved_kg_m3"] = np.clip(result["c_dissolved_kg_m3"], 0, dissolved_cap)
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
    pressure_cap = getattr(args, "pressure_cap_pa", None)
    pressure_damage = getattr(args, "pressure_damage_per_pa2", None)
    if pressure_cap is not None:
        true_max = ptrue.reshape(len(ptrue), -1).max(1)
        pred_max = ppred.reshape(len(ppred), -1).max(1)
        true_violation = true_max > pressure_cap
        pred_violation = pred_max > pressure_cap
        out["pressure_cap_pa"] = float(pressure_cap)
        out["pressure_cap_violation_true_fraction"] = float(np.mean(true_violation))
        out["pressure_cap_violation_pred_fraction"] = float(np.mean(pred_violation))
        out["pressure_cap_violation_classification_error"] = float(
            np.mean(true_violation != pred_violation))
        out["pressure_cap_exceedance_abs_error_mean_pa"] = float(np.mean(np.abs(
            np.maximum(pred_max - pressure_cap, 0.0)
            - np.maximum(true_max - pressure_cap, 0.0))))
        if pressure_damage is not None:
            true_penalty = pressure_damage * np.maximum(true_max - pressure_cap, 0.0) ** 2
            pred_penalty = pressure_damage * np.maximum(pred_max - pressure_cap, 0.0) ** 2
            out["pressure_penalty_abs_error_mean"] = float(np.mean(np.abs(pred_penalty - true_penalty)))
            out["pressure_penalty_bias"] = float(np.mean(pred_penalty - true_penalty))
    top_errors, well_errors, plume_iou, plume_fraction = [], [], [], []
    plume_thresholds = (0.005, 0.01, 0.02)
    threshold_ious = {threshold: [] for threshold in plume_thresholds}
    plume_precision, plume_recall = [], []
    h, w = ptrue.shape[-2:]
    well_z, well_x = h - 2, (max(1, w // 3), min(w - 2, (2 * w) // 3))
    for i in range(len(ptrue)):
        thresh = np.quantile(ptrue[i], 0.9)
        mask = ptrue[i] >= thresh
        top_errors.extend((ppred[i][mask] - ptrue[i][mask]).ravel().tolist())
        well_errors.extend((ppred[i, well_z, list(well_x)] - ptrue[i, well_z, list(well_x)]).tolist())
        a, b = (pred["sg_mobile"][i] >= 0.01,
                truth["sg_mobile"][i] >= 0.01)
        union = np.logical_or(a, b).sum()
        intersection = np.logical_and(a, b).sum()
        plume_iou.append(1.0 if union == 0 else intersection / union)
        plume_fraction.append(abs(float(a.mean()) - float(b.mean())))
        plume_precision.append(1.0 if a.sum() == 0 and b.sum() == 0
                               else intersection / max(int(a.sum()), 1))
        plume_recall.append(1.0 if a.sum() == 0 and b.sum() == 0
                            else intersection / max(int(b.sum()), 1))
        for threshold in plume_thresholds:
            pred_mask = pred["sg_mobile"][i] >= threshold
            true_mask = truth["sg_mobile"][i] >= threshold
            threshold_union = np.logical_or(pred_mask, true_mask).sum()
            threshold_intersection = np.logical_and(pred_mask, true_mask).sum()
            threshold_ious[threshold].append(
                1.0 if threshold_union == 0 else threshold_intersection / threshold_union)
    out["pressure_top_decile_rmse_pa"] = float(np.sqrt(np.mean(np.square(top_errors))))
    out["pressure_well_cell_rmse_pa"] = float(np.sqrt(np.mean(np.square(well_errors))))
    out["plume_iou_sg_ge_0p01"] = float(np.mean(plume_iou))
    out["plume_fraction_abs_error"] = float(np.mean(plume_fraction))
    out["plume_precision_sg_ge_0p01"] = float(np.mean(plume_precision))
    out["plume_recall_sg_ge_0p01"] = float(np.mean(plume_recall))
    for threshold, values in threshold_ious.items():
        key = str(threshold).replace(".", "p")
        out[f"plume_iou_sg_ge_{key}"] = float(np.mean(values))
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


def _raw_validity(data, delta):
    """Violation fractions before the inference projection is applied."""
    raw = {key: data[f"{key}_in"] + delta[:, i]
           for i, key in enumerate(STATE_KEYS)}
    sm, st = raw["sg_mobile"], raw["sg_trapped"]
    smx, cd = raw["sg_max"], raw["c_dissolved_kg_m3"]
    land = smx / np.maximum(1.0 + 2.0 * smx, 1e-12)
    dissolved_cap = 30.0 * (1.0 - sm - st)
    return {"negative_mobile_fraction": float(np.mean(sm < 0)),
            "trapped_decrease_fraction": float(np.mean(st < data["sg_trapped_in"])),
            "saturation_sum_over_0p85_fraction": float(np.mean(sm + st > 1.0 - SWC)),
            "sgmax_decrease_fraction": float(np.mean(smx < data["sg_max_in"])),
            "land_trapping_bound_violation_fraction": float(np.mean(st > land)),
            "dissolution_capacity_violation_fraction": float(np.mean(cd > dissolved_cap)),
            "negative_dissolved_fraction": float(np.mean(cd < 0))}


def _inference_ms_batch1(model, data, stats, rate_scale, device, repeats=30):
    x = build_x({key: value[:1] for key, value in data.items()
                 if isinstance(value, np.ndarray) and value.ndim > 0
                 and len(value) == len(data["geology_seed"])}, rate_scale)
    xn = torch.from_numpy(((x - stats["x_mean"]) / stats["x_std"]).astype(np.float32)).to(device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(xn)
        sync(); t0 = time.perf_counter()
        for _ in range(repeats):
            model(xn)
        sync()
    return 1000.0 * (time.perf_counter() - t0) / repeats


def evaluate_one_step(model, datasets, stats, rate_scale, device, args):
    results = {}
    for cohort, data in datasets.items():
        truth = {key: data[f"{key}_out"] for key in STATE_KEYS}
        delta = predict_delta(model, data, stats, rate_scale, device)
        pred = _to_states(data, delta)
        results[cohort] = score_states(pred, truth, data, args)
        results[cohort]["raw_validity_before_projection"] = _raw_validity(data, delta)
    return results


def evaluate_rollout(model, datasets, stats, rate_scale, device, args):
    results = {}
    for cohort, data in datasets.items():
        keys = list(zip(data["geology_seed"].tolist(), data["episode_id"].tolist()))
        order = np.lexsort((data["stage_index"], data["episode_id"], data["geology_seed"]))
        pred_rows, true_rows, input_states, input_rows, stage_rows = (
            {k: [] for k in STATE_KEYS}, {k: [] for k in STATE_KEYS},
            {k: [] for k in STATE_KEYS}, [], [])
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
            hidden = None
            for j in group:
                one = {k: v[j:j + 1] for k, v in data.items()
                       if isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == len(data["geology_seed"])}
                for k in STATE_KEYS:
                    input_states[k].append(state[k][0].copy())
                if hasattr(model, "step"):
                    xx = build_x(one, rate_scale, state)
                    xn = torch.from_numpy(((xx - stats["x_mean"]) / stats["x_std"]).astype(np.float32)).to(device)
                    with torch.no_grad():
                        out, hidden = model.step(xn, hidden)
                    dy = out.cpu().numpy() * stats["y_std"] + stats["y_mean"]
                else:
                    dy = predict_delta(model, one, stats, rate_scale, device, dynamic=state)
                pstate = _to_states(one, dy, dynamic=state)
                for k in STATE_KEYS:
                    pred_rows[k].append(pstate[k][0])
                    true_rows[k].append(data[f"{k}_out"][j])
                input_rows.append(j)
                stage_rows.append(int(data["stage_index"][j]))
                state = {k: pstate[k] for k in STATE_KEYS}
        if input_rows:
            ids = np.asarray(input_rows, dtype=int)
            pred = {k: np.stack(v) for k, v in pred_rows.items()}
            truth = {k: np.stack(v) for k, v in true_rows.items()}
            subset = {k: v[ids] for k, v in data.items()
                      if isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == len(data["geology_seed"])}
            for k in STATE_KEYS:
                subset[f"{k}_in"] = np.stack(input_states[k])
            overall = score_states(pred, truth, subset, args)
            by_stage = {}
            stage_rows = np.asarray(stage_rows)
            for stage in sorted(np.unique(stage_rows).tolist()):
                mask = stage_rows == stage
                stage_data = {k: v[mask] for k, v in subset.items()}
                stage_pred = {k: v[mask] for k, v in pred.items()}
                stage_truth = {k: v[mask] for k, v in truth.items()}
                by_stage[str(stage)] = score_states(stage_pred, stage_truth, stage_data, args)
            overall["by_stage"] = by_stage
            results[cohort] = overall
        else:
            results[cohort] = {"error": "no six-stage episodes found"}
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--backbones", nargs="+", default=["unet", "ccs", "gnn", "fno", "ufno"],
                    choices=["unet", "ccs", "gnn", "fno", "ufno", "ccs_gnn", "ccs_lstm", "gnn_lstm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=8e-4)
    ap.add_argument("--pmax-loss-weight", type=float, default=0.5,
                    help="weight for normalized max-pressure regression")
    ap.add_argument("--pressure-tail-loss-weight", type=float, default=0.0,
                    help="weight for pressure error on target top-tail cells")
    ap.add_argument("--pressure-tail-fraction", type=float, default=0.1,
                    help="fraction of highest target-pressure cells used for tail loss")
    ap.add_argument("--pressure-penalty-loss-weight", type=float, default=0.0,
                    help="weight for matching pressure-cap exceedance-squared penalty")
    ap.add_argument("--pressure-cap-pa", type=float, default=None,
                    help="expert-reviewed game pressure cap in Pa")
    ap.add_argument("--pressure-damage-per-pa2", type=float, default=None,
                    help="damage coefficient used to report payoff-scale pressure penalty error")
    ap.add_argument("--mass-loss-weight", type=float, default=0.0,
                    help="soft stage mass-balance loss weight; zero preserves baseline training")
    ap.add_argument("--valid-loss-weight", type=float, default=0.0,
                    help="penalize raw invalid saturation/trapping/dissolution outputs before clipping")
    ap.add_argument("--rollout-unroll", type=int, choices=[1, 3, 6], default=1,
                    help="independent transitions or 3-/6-stage pushforward windows")
    ap.add_argument("--rollout-validation-weight", type=float, default=0.0,
                    help="weight for normalized six-stage validation rollout score in checkpoint selection")
    ap.add_argument("--rollout-validation-every", type=int, default=5,
                    help="evaluate validation rollouts every N epochs when rollout weight is nonzero")
    ap.add_argument("--plume-loss-weight", type=float, default=0.0,
                    help="soft-Dice weight for the mobile-CO2 plume mask")
    ap.add_argument("--plume-threshold", type=float, default=0.01,
                    help="mobile gas saturation cutoff defining plume cells")
    ap.add_argument("--plume-temperature", type=float, default=0.002,
                    help="soft threshold width for the differentiable plume mask")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                    help="reuse completed seed checkpoints and metrics already in the output folder")
    ap.add_argument("--stage-days", type=float, default=30.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not 0.0 < args.pressure_tail_fraction <= 1.0:
        raise ValueError("--pressure-tail-fraction must be in (0, 1]")
    if args.pressure_penalty_loss_weight > 0 and not args.pressure_cap_pa:
        raise ValueError("--pressure-penalty-loss-weight requires --pressure-cap-pa")
    if args.pressure_damage_per_pa2 is not None and not args.pressure_cap_pa:
        raise ValueError("--pressure-damage-per-pa2 requires --pressure-cap-pa")
    if min(args.pmax_loss_weight, args.pressure_tail_loss_weight,
           args.pressure_penalty_loss_weight, args.rollout_validation_weight) < 0:
        raise ValueError("pressure loss weights must be nonnegative")
    if args.rollout_validation_every < 1:
        raise ValueError("--rollout-validation-every must be >= 1")
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = {c: load_cohort(data_dir, "train", c) for c in COHORTS}
    val = {c: load_cohort(data_dir, "validation", c) for c in COHORTS}
    test = {c: load_cohort(data_dir, "test", c) for c in COHORTS}
    rate_scale = max(
        max(
            float(max(np.max(d["rate_operator_0_m3_s"]),
                      np.max(d["rate_operator_1_m3_s"])))
            for d in train.values()
        ),
        1e-12,
    )
    x_parts = [build_x(d, rate_scale) for d in train.values()]
    y_parts = [build_y(d) for d in train.values()]
    x_train, y_train = np.concatenate(x_parts), np.concatenate(y_parts)
    pv_parts, inject_parts, state_parts = [], [], []
    for d in train.values():
        cell_volume = ((d["dx_ft"] * 0.3048) * (d["dy_ft"] * 0.3048)
                       * (d["dz_ft"] * 0.3048))
        pv_parts.append(d["phi"] * cell_volume[:, None, None])
        inject_parts.append(RHO_CO2 * (d["rate_operator_0_m3_s"]
                                      + d["rate_operator_1_m3_s"])
                            * args.stage_days * 86400.0)
        state_parts.append(np.stack([d[f"{key}_in"] for key in STATE_KEYS], axis=1))
    pv_train = np.concatenate(pv_parts).astype(np.float32)
    inject_train = np.concatenate(inject_parts).astype(np.float32)
    state_train = np.concatenate(state_parts).astype(np.float32)
    sequence_keys = ("kx_mD", "kz_mD", "phi", "rate_operator_0_m3_s",
                     "rate_operator_1_m3_s", "dx_ft", "dy_ft", "dz_ft")
    sequence_data = {key: np.concatenate([d[key] for d in train.values()])
                     for key in sequence_keys}
    sequence_windows = (_sequence_windows(train, args.rollout_unroll)
                        if args.rollout_unroll > 1 else None)
    if args.rollout_unroll > 1 and not len(sequence_windows):
        raise ValueError("No consecutive episode windows found for rollout-aware training")
    stats = fit_stats(x_train, y_train)
    X = ((x_train - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
    Y = ((y_train - stats["y_mean"]) / stats["y_std"]).astype(np.float32)
    p_in = np.concatenate([d["p_pa_in"] for d in train.values()])
    best_metrics = []
    device = torch.device(args.device)
    for backbone in args.backbones:
        for seed in args.seeds:
            ckpt = out_dir / f"co2_{backbone}_s{seed}.pt"
            metric_path = out_dir / f"co2_{backbone}_s{seed}_metrics.json"
            if args.resume and ckpt.exists() and metric_path.exists():
                best_metrics.append(json.loads(metric_path.read_text(encoding="utf-8")))
                print(f"[resume] reuse {backbone} seed {seed}", flush=True)
                continue
            torch.manual_seed(seed); np.random.seed(seed)
            model = make_model(backbone, args.width).to(device)
            params = sum(p.numel() for p in model.parameters())
            opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
            best, best_val_mse, best_epoch, stale, t0 = float("inf"), float("inf"), 0, 0, time.time()
            val_X = {c: ((build_x(d, rate_scale) - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
                     for c, d in val.items()}
            val_Y = {c: ((build_y(d) - stats["y_mean"]) / stats["y_std"]).astype(np.float32)
                     for c, d in val.items()}
            for epoch in range(1, args.epochs + 1):
                model.train(); run = 0.0
                y_std_t = torch.as_tensor(stats["y_std"], device=device)
                y_mean_t = torch.as_tensor(stats["y_mean"], device=device)
                if args.rollout_unroll == 1:
                    order = np.random.permutation(len(X))
                    batches = [order[i:i + args.batch_size]
                               for i in range(0, len(order), args.batch_size)]
                else:
                    order = np.random.permutation(len(sequence_windows))
                    batches = [sequence_windows[order[i:i + args.batch_size]]
                               for i in range(0, len(order), args.batch_size)]
                for batch_ids in batches:
                    opt.zero_grad(set_to_none=True)
                    if args.rollout_unroll == 1:
                        ids = batch_ids
                        xb = torch.from_numpy(X[ids]).to(device)
                        state_in = torch.from_numpy(state_train[ids]).to(device)
                        target_delta = torch.from_numpy(y_train[ids]).to(device)
                        pv = torch.from_numpy(pv_train[ids]).to(device)
                        injected = torch.from_numpy(inject_train[ids]).to(device)
                        pred = model(xb)
                        loss = _batch_loss(pred, target_delta, state_in, pv,
                                           injected, y_std_t, y_mean_t,
                                           args.valid_loss_weight, args.mass_loss_weight,
                                           args.plume_loss_weight, args.plume_threshold,
                                           args.plume_temperature,
                                           args.pmax_loss_weight,
                                           args.pressure_tail_loss_weight,
                                           args.pressure_tail_fraction,
                                           args.pressure_penalty_loss_weight,
                                           args.pressure_cap_pa)
                        n_items = len(ids)
                    else:
                        seq = batch_ids
                        dynamic = state_train[seq[:, 0]].copy()
                        hidden = None
                        step_losses = []
                        for step in range(args.rollout_unroll):
                            ids = seq[:, step]
                            sample = {key: value[ids] for key, value in sequence_data.items()}
                            dynamic_map = {key: dynamic[:, si]
                                           for si, key in enumerate(STATE_KEYS)}
                            x_np = build_x(sample, rate_scale, dynamic_map)
                            x_norm = ((x_np - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
                            xb = torch.from_numpy(x_norm).to(device)
                            state_in = torch.from_numpy(dynamic).to(device)
                            true_out = torch.from_numpy(state_train[ids] + y_train[ids]).to(device)
                            target_delta = true_out - state_in
                            pv = torch.from_numpy(pv_train[ids]).to(device)
                            injected = torch.from_numpy(inject_train[ids]).to(device)
                            if hasattr(model, "step"):
                                pred, hidden = model.step(xb, hidden)
                            else:
                                pred = model(xb)
                            step_losses.append(_batch_loss(
                                pred, target_delta, state_in, pv, injected,
                                y_std_t, y_mean_t, args.valid_loss_weight,
                                args.mass_loss_weight, args.plume_loss_weight,
                                args.plume_threshold, args.plume_temperature,
                                args.pmax_loss_weight,
                                args.pressure_tail_loss_weight,
                                args.pressure_tail_fraction,
                                args.pressure_penalty_loss_weight,
                                args.pressure_cap_pa))
                            # Push forward but detach state at each stage to
                            # bound memory use during the rollout unroll.
                            raw_out = state_in + pred * y_std_t + y_mean_t
                            dynamic = _torch_project(raw_out, state_in).detach().cpu().numpy()
                        loss = torch.stack(step_losses).mean()
                        n_items = len(seq)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                    run += float(loss.detach()) * n_items
                model.eval(); val_loss = 0.0; val_n = 0
                val_plume_sum = 0.0; val_plume_n = 0
                val_pmax_sum = 0.0; val_tail_sum = 0.0; val_penalty_sum = 0.0
                val_case_n = 0
                y_std_t = torch.as_tensor(stats["y_std"], device=device)
                y_mean_t = torch.as_tensor(stats["y_mean"], device=device)
                with torch.no_grad():
                    for c in COHORTS:
                        vx = torch.from_numpy(val_X[c]).to(device)
                        vy = torch.from_numpy(val_Y[c]).to(device)
                        state_in_np = np.stack(
                            [val[c][f"{key}_in"] for key in STATE_KEYS], axis=1
                        ).astype(np.float32)
                        state_in_all = torch.from_numpy(state_in_np).to(device)
                        for i in range(0, len(vx), args.batch_size):
                            sl = slice(i, i + args.batch_size)
                            pred_norm = model(vx[sl])
                            target_norm = vy[sl]
                            val_loss += float(F.mse_loss(pred_norm, target_norm,
                                                         reduction="sum"))
                            val_n += target_norm.numel()
                            state_in = state_in_all[sl]
                            pred_out = state_in + pred_norm * y_std_t + y_mean_t
                            target_out = state_in + target_norm * y_std_t + y_mean_t
                            pred_p = pred_out[:, 0].flatten(1)
                            true_p = target_out[:, 0].flatten(1)
                            pscale2 = y_std_t[0, 0, 0, 0].square() + 1e-12
                            val_pmax_sum += float(F.mse_loss(
                                pred_p.amax(1), true_p.amax(1), reduction="sum") / pscale2)
                            tail_count = max(1, int(math.ceil(
                                true_p.shape[1] * args.pressure_tail_fraction)))
                            tail_threshold = true_p.topk(tail_count, dim=1).values[:, -1:]
                            tail_mask = true_p >= tail_threshold
                            tail_per_case = (((pred_p - true_p).square() * tail_mask).sum(1)
                                             / tail_mask.sum(1).clamp_min(1) / pscale2)
                            val_tail_sum += float(tail_per_case.sum())
                            if args.pressure_cap_pa is not None:
                                scale = torch.clamp_min(
                                    y_std_t[0, 0, 0, 0], max(float(args.pressure_cap_pa), 1.0))
                                pred_excess = F.relu(pred_p.amax(1) - args.pressure_cap_pa) / scale
                                true_excess = F.relu(true_p.amax(1) - args.pressure_cap_pa) / scale
                                val_penalty_sum += float(F.mse_loss(
                                    pred_excess.square(), true_excess.square(), reduction="sum"))
                            val_case_n += len(pred_p)
                            if args.plume_loss_weight > 0:
                                pred_prob = torch.sigmoid(
                                    (pred_out[:, 1] - args.plume_threshold)
                                    / max(args.plume_temperature, 1e-6))
                                target_mask = (target_out[:, 1] >= args.plume_threshold).to(pred_prob.dtype)
                                inter = (pred_prob * target_mask).sum((1, 2))
                                dice = 1.0 - (2.0 * inter + 1e-6) / (
                                    pred_prob.sum((1, 2)) + target_mask.sum((1, 2)) + 1e-6
                                )
                                val_plume_sum += float(dice.sum())
                                val_plume_n += len(dice)
                val_mse = val_loss / max(val_n, 1)
                val_plume_dice_loss = val_plume_sum / max(val_plume_n, 1)
                val_pmax_loss = val_pmax_sum / max(val_case_n, 1)
                val_pressure_tail_loss = val_tail_sum / max(val_case_n, 1)
                val_pressure_penalty_loss = val_penalty_sum / max(val_case_n, 1)
                score = (val_mse + args.pmax_loss_weight * val_pmax_loss
                         + args.pressure_tail_loss_weight * val_pressure_tail_loss
                         + args.pressure_penalty_loss_weight * val_pressure_penalty_loss
                         + args.plume_loss_weight * val_plume_dice_loss)
                rollout_score = None
                rollout_check = (args.rollout_validation_weight > 0 and
                                 (epoch == 1 or epoch % args.rollout_validation_every == 0
                                  or epoch == args.epochs))
                if rollout_check:
                    rollout_results = evaluate_rollout(
                        model, val, stats, rate_scale, device, args)
                    channel_scales = np.asarray(stats["y_std"]).reshape(-1)
                    cohort_scores = []
                    for result in rollout_results.values():
                        if "p_pa_rmse" not in result:
                            continue
                        field_score = np.mean([
                            result[f"{key}_rmse"] / max(float(channel_scales[i]), 1e-12)
                            for i, key in enumerate(STATE_KEYS)])
                        pmax_score = result["pmax_abs_mean_pa"] / max(float(channel_scales[0]), 1e-12)
                        cohort_scores.append(float(field_score + pmax_score))
                    if cohort_scores:
                        rollout_score = float(np.mean(cohort_scores))
                        score += args.rollout_validation_weight * rollout_score
                select_this_epoch = args.rollout_validation_weight == 0 or rollout_check
                if select_this_epoch and score < best:
                    best, best_val_mse, best_epoch, stale = score, val_mse, epoch, 0
                    torch.save({"state_dict": model.state_dict(), "backbone": backbone,
                                "width": args.width, "seed": seed, "params": params,
                                "rate_scale": rate_scale,
                                "stats": {k: v.tolist() for k, v in stats.items()},
                                "args": vars(args)}, ckpt)
                elif select_this_epoch:
                    stale += 1
                if epoch == 1 or epoch % 5 == 0:
                    print(f"{backbone} seed {seed} epoch {epoch}: train={run/len(X):.5f} "
                          f"val_mse={val_mse:.5f} val_pmax={val_pmax_loss:.5f} "
                          f"val_tail={val_pressure_tail_loss:.5f} "
                          f"val_penalty={val_pressure_penalty_loss:.5f} "
                          f"val_rollout={rollout_score if rollout_score is not None else float('nan'):.5f} "
                          f"val_objective={score:.5f} "
                          f"best={best:.5f} ({best_epoch})", flush=True)
                if stale >= args.patience:
                    break
            blob = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(blob["state_dict"])
            item = {"backbone": backbone, "seed": seed, "params": params,
                    "best_epoch": best_epoch, "train_seconds": round(time.time() - t0),
                    "inference_ms_sample_batch1": _inference_ms_batch1(
                        model, next(iter(test.values())), stats, rate_scale, device),
                    "inference_scope": "network forward only; excludes preprocessing and solver",
                    "best_val_mse": best_val_mse,
                    "best_val_objective": best,
                    "one_step_test": evaluate_one_step(model, test, stats, rate_scale, device, args),
                    "autoregressive_test": evaluate_rollout(model, test, stats, rate_scale, device, args)}
            metric_path.write_text(json.dumps(item, indent=2), encoding="utf-8")
            best_metrics.append(item)
            print(f"Saved {ckpt} and {metric_path}", flush=True)
    (out_dir / "run_summary.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    print(f"Completed {len(best_metrics)} model/seed runs in {out_dir}")


if __name__ == "__main__":
    main()
