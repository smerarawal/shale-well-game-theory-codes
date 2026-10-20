"""
evaluate_extended_metrics.py -- extra error measures for the best architectures
(requested by the advisor), plus the game-relevant pressure quantities.

For every architecture / seed / data set it reports, per channel (pressure,
saturation): RMSE, MAE, max absolute error, relative L2, normalised RMSE
(RMSE / per-sample range of the true field) and R^2. For pressure it also
reports game-relevant quantities:
  * p_max error (signed bias, absolute, relative) -- the game penalty reads this,
  * pressure error at well cells (RMSE / MAE) -- the belief update reads this,
  * error in the high-pressure region (top decile of each true field, and
    cells with true p > --p_thresh if any exist),
  * penalty proxy 30*max(0, p_max - cap)^2 evaluated on prediction vs truth,
  * signed error / RMSE binned by true pressure level (calibration).

CAUTION: pressure in this benchmark (32x32, 150 steps, producers present) is not on
the game's scale (12x12, injectors only). --p_thresh and --cap are only meaningful
here as relative diagnostics until the stage-wise 12x12 data exist.

    python evaluate_extended_metrics.py --out final_run
    python evaluate_extended_metrics.py --out final_run --archs ccsnet unet_baseline
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from model_zoo_v4 import make_all_architectures_v4
from run_fair_comparison import load_data

DEFAULT_ARCHS = ["ccsnet", "unet_baseline", "ufno", "runet_traj", "plain_fno"]


@torch.no_grad()
def predict(model, X, bs=128):
    model.eval()
    return torch.cat([model(X[i:i + bs]) for i in range(0, X.shape[0], bs)]).cpu().numpy()


def channel_metrics(pred, true):
    """pred, true: [N,H,W]. Returns dict of sample-averaged metrics."""
    N = pred.shape[0]
    err = (pred - true).reshape(N, -1)
    t = true.reshape(N, -1)
    mse = (err ** 2).mean(1)
    rmse = np.sqrt(mse)
    sst = ((t - t.mean(1, keepdims=True)) ** 2).sum(1) + 1e-12
    rng = t.max(1) - t.min(1) + 1e-12
    return {
        "rmse": float(rmse.mean()),
        "rmse_median": float(np.median(rmse)),
        "rmse_p95": float(np.percentile(rmse, 95)),
        "mae": float(np.abs(err).mean(1).mean()),
        "max_abs_err": float(np.abs(err).max(1).mean()),
        "max_abs_err_worst": float(np.abs(err).max()),
        "rel_l2": float((np.linalg.norm(err, axis=1) / (np.linalg.norm(t, axis=1) + 1e-8)).mean()),
        "nrmse_range": float((rmse / rng).mean()),
        "r2": float((1.0 - (err ** 2).sum(1) / sst).mean()),
        "bias": float(err.mean()),
    }


def pressure_game_metrics(pred, true, wmask, p_thresh, cap):
    N = pred.shape[0]
    pm_t, pm_p = true.reshape(N, -1).max(1), pred.reshape(N, -1).max(1)
    d = pm_p - pm_t
    pen = lambda pm: 30.0 * np.maximum(0.0, pm - cap) ** 2
    out = {
        "pmax_bias": float(d.mean()),
        "pmax_abs_err": float(np.abs(d).mean()),
        "pmax_abs_err_p95": float(np.percentile(np.abs(d), 95)),
        "pmax_rel_err": float((np.abs(d) / (np.abs(pm_t) + 1e-8)).mean()),
        "penalty_proxy_abs_err": float(np.abs(pen(pm_p) - pen(pm_t)).mean()),
        "penalty_proxy_frac_samples_over_cap": float((pm_t > cap).mean()),
    }
    wc = wmask != 0
    if wc.any():
        e = (pred - true)[wc]
        out["well_rmse"] = float(np.sqrt((e ** 2).mean()))
        out["well_mae"] = float(np.abs(e).mean())
        out["well_bias"] = float(e.mean())
    q90 = np.percentile(true.reshape(N, -1), 90, axis=1)[:, None, None]
    top = true >= q90
    e = (pred - true)[top]
    out["top_decile_rmse"] = float(np.sqrt((e ** 2).mean()))
    out["top_decile_bias"] = float(e.mean())
    hi = true > p_thresh
    out["p_thresh"] = p_thresh
    out["frac_cells_above_thresh"] = float(hi.mean())
    if hi.sum() > 0:
        e = (pred - true)[hi]
        out["above_thresh_rmse"] = float(np.sqrt((e ** 2).mean()))
        out["above_thresh_bias"] = float(e.mean())
    # calibration vs true pressure level (pooled quantile bins)
    edges = np.quantile(true.ravel(), np.linspace(0, 1, 6))
    bins = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (true >= a) & (true <= b)
        e = (pred - true)[m]
        bins.append({"p_lo": float(a), "p_hi": float(b), "bias": float(e.mean()),
                     "rmse": float(np.sqrt((e ** 2).mean()))})
    out["by_true_pressure_quintile"] = bins
    return out


def evaluate_set(model, x, y, p_thresh, cap, dev):
    X = torch.from_numpy(x).to(dev)
    pred = predict(model, X)
    wmask = x[:, 2]
    res = {"pressure": channel_metrics(pred[:, 0], y[:, 0]),
           "saturation": channel_metrics(pred[:, 1], y[:, 1]),
           "pressure_game": pressure_game_metrics(pred[:, 0], y[:, 0], wmask, p_thresh, cap)}
    return res


def flat_mean(list_of_dicts):
    """Average nested dicts of floats over seeds; bins are averaged elementwise."""
    out = {}
    for k, v in list_of_dicts[0].items():
        if isinstance(v, dict):
            out[k] = flat_mean([d[k] for d in list_of_dicts])
        elif isinstance(v, list):
            out[k] = [flat_mean([d[k][i] for d in list_of_dicts]) for i in range(len(v))]
        else:
            vals = [d[k] for d in list_of_dicts if k in d]
            out[k] = float(np.mean(vals))
            if len(vals) > 1:
                out[k + "_seedstd"] = float(np.std(vals))
    return out


def write_markdown(res, path):
    lines = ["# Extended error metrics (mean over seeds)\n",
             "Pressure in benchmark units; see the CAUTION in evaluate_extended_metrics.py.\n"]
    sets = sorted({s for a in res.values() for s in a})
    for s in sets:
        lines += [f"\n## {s}\n",
                  "| arch | P RMSE | P MAE | P max-abs | P R2 | S RMSE | S MAE | S max-abs | S R2 | pmax bias | pmax |err| | well RMSE | top-decile RMSE |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for a, d in res.items():
            if s not in d:
                continue
            P, S, G = d[s]["pressure"], d[s]["saturation"], d[s]["pressure_game"]
            lines.append(
                f"| {a} | {P['rmse']:.4f} | {P['mae']:.4f} | {P['max_abs_err']:.4f} | {P['r2']:.4f} | "
                f"{S['rmse']:.4f} | {S['mae']:.4f} | {S['max_abs_err']:.4f} | {S['r2']:.4f} | "
                f"{G['pmax_bias']:+.4f} | {G['pmax_abs_err']:.4f} | {G.get('well_rmse', float('nan')):.4f} | "
                f"{G['top_decile_rmse']:.4f} |")
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="final_run")
    ap.add_argument("--archs", nargs="+", default=DEFAULT_ARCHS)
    ap.add_argument("--p_thresh", type=float, default=5.0)
    ap.add_argument("--cap", type=float, default=6.0)
    ap.add_argument("--extra_ood_dir", default=None, help="extra ood_*.npz sets (e.g. SPE10), read in addition to <out>/ood")
    ap.add_argument("--report_dir", default=None, help="default: <out>/report (use a writable dir on Kaggle)")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.out, "run_config.json")))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zoo = make_all_architectures_v4(cfg["H"], cfg["W"], cfg["T"], cfg["stats"])

    # in-distribution test split (same split seed as training)
    x, y, _, idx, _, _, _ = load_data(os.path.join(a.out, "data", "dataset.npz"))
    te = idx["test"]
    sets = {"test": (x[te], y[te])}
    ood_files = sorted(glob.glob(os.path.join(a.out, "ood", "ood_*.npz")))
    if a.extra_ood_dir:
        ood_files += sorted(glob.glob(os.path.join(a.extra_ood_dir, "ood_*.npz")))
    for p in ood_files:
        if p.endswith(".tmp.npz") or "chunks" in p:
            continue
        d = np.load(p)
        sets[os.path.basename(p)[:-4]] = (
            np.stack([d["permeability"], d["porosity"], d["well_mask"]], axis=1).astype(np.float32),
            np.stack([d["final_pressure"], d["final_saturation"]], axis=1).astype(np.float32))

    res = {}
    for arch in a.archs:
        wfiles = sorted(glob.glob(os.path.join(a.out, "weights", f"{arch}__seed*.pt")))
        if not wfiles:
            print(f"[skip] no weights for {arch}")
            continue
        per_set = {n: [] for n in sets}
        for wf in wfiles:
            m = zoo[arch]().to(dev)
            m.load_state_dict(torch.load(wf, map_location=dev))
            for n, (xs, ys) in sets.items():
                per_set[n].append(evaluate_set(m, xs, ys, a.p_thresh, a.cap, dev))
        res[arch] = {n: flat_mean(v) | {"n_seeds": len(v)} for n, v in per_set.items()}
        t = res[arch]["test"]
        print(f"{arch}: P rmse {t['pressure']['rmse']:.4f} mae {t['pressure']['mae']:.4f} "
              f"| S rmse {t['saturation']['rmse']:.4f} | pmax |err| {t['pressure_game']['pmax_abs_err']:.4f}", flush=True)

    rd = a.report_dir or os.path.join(a.out, "report")
    os.makedirs(rd, exist_ok=True)
    json.dump(res, open(os.path.join(rd, "extended_metrics.json"), "w"), indent=1)
    write_markdown(res, os.path.join(rd, "extended_metrics.md"))
    print("wrote report/extended_metrics.json and .md")


if __name__ == "__main__":
    main()
