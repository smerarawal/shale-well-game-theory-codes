"""Make scale-aware figures and pre-flow summaries for generated CO2 geologies.

Run after ``co2_experiment.py`` has written the NPZ and manifest, e.g.::

    python visualize_co2_geologies.py --input-dir /kaggle/working/co2_geologies

The figures describe rock-property inputs only. They do not establish that a
cohort produces its intended flow-interference or equilibrium regime.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


SYNTHETIC = ("friendly", "intermediate", "challenging")
SPE10_XY = ("spe10_tarbert", "spe10_upper_ness")
SPE10_XZ = ("spe10_xz_tarbert", "spe10_xz_upper_ness")


def _get(npz, key):
    if key not in npz:
        return None
    return np.asarray(npz[key])


def _limits(arrays, q=(0.0, 1.0), fallback=(0.0, 1.0)):
    values = [np.asarray(a, dtype=float).ravel() for a in arrays if a is not None]
    values = [v[np.isfinite(v)] for v in values if v.size]
    if not values:
        return fallback
    values = np.concatenate(values)
    lo, hi = np.quantile(values, q)
    if not np.isfinite(lo + hi) or hi <= lo:
        return fallback
    return float(lo), float(hi)


def _save_synthetic(npz, out_dir, count):
    import matplotlib.pyplot as plt

    selected = {}
    for cohort in SYNTHETIC:
        k, phi = _get(npz, f"{cohort}_k"), _get(npz, f"{cohort}_phi")
        if k is None or phi is None:
            continue
        n = min(count, len(k), len(phi))
        selected[cohort] = (k[:n], phi[:n], _get(npz, f"{cohort}_seed")[:n])
    if not selected:
        return
    kmin, kmax = _limits([v[0] for v in selected.values()], fallback=(0.005, 0.4))
    pmin, pmax = _limits([v[1] for v in selected.values()], fallback=(0.08, 0.25))
    cohorts = list(selected)
    n = min(len(v[0]) for v in selected.values())
    fig, axes = plt.subplots(2 * len(cohorts), n, figsize=(2.0 * n, 1.75 * 2 * len(cohorts)),
                             squeeze=False, constrained_layout=True)
    for ci, cohort in enumerate(cohorts):
        for part, label, vmin, vmax, unit in (
                (0, f"{cohort} | scaled K", kmin, kmax, "permeability input"),
                (1, f"{cohort} | porosity", pmin, pmax, "porosity fraction")):
            r = 2 * ci + part
            source, seeds = selected[cohort][part], selected[cohort][2]
            for i in range(n):
                ax = axes[r, i]
                im = ax.imshow(source[i], cmap="viridis", vmin=vmin, vmax=vmax,
                               origin="upper", interpolation="nearest")
                ax.set_title(f"seed {int(seeds[i])}", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            axes[r, 0].set_ylabel(label)
            fig.colorbar(im, ax=axes[r, :], shrink=.82,
                         label=f"{unit} (shared scale across synthetic regimes)")
    fig.suptitle("Randomized synthetic geology samples — inputs only", fontsize=12)
    fig.savefig(out_dir / "synthetic_regimes_samples.png", dpi=180)
    plt.close(fig)


def _save_spe10_xy(npz, out_dir, count):
    import matplotlib.pyplot as plt

    available = []
    for cohort in SPE10_XY:
        k, phi = _get(npz, f"{cohort}_k"), _get(npz, f"{cohort}_phi")
        if k is not None and phi is not None:
            seeds = _get(npz, f"{cohort}_seed")
            available.append((cohort, k[:count], phi[:count], seeds[:count]))
    if not available:
        return
    n = min(len(item[1]) for item in available)
    kmin, kmax = _limits([v[1] for v in available], fallback=(0.02, 0.2))
    pmin, pmax = _limits([v[2] for v in available], fallback=(0.08, 0.25))
    fig, axes = plt.subplots(2 * len(available), n, figsize=(2.0 * n, 1.8 * 2 * len(available)),
                             squeeze=False, constrained_layout=True)
    for ci, (cohort, k, phi, seeds) in enumerate(available):
        region = cohort.replace("spe10_", "").replace("_", " ").title()
        for part, label, source, vmin, vmax, bar in (
                (0, f"{region} | scaled Kx", k, kmin, kmax, "scaled Kx proxy"),
                (1, f"{region} | porosity", phi, pmin, pmax, "per-patch normalized porosity fraction")):
            r = 2 * ci + part
            for i in range(n):
                ax = axes[r, i]
                im = ax.imshow(source[i], cmap="viridis", vmin=vmin, vmax=vmax,
                               origin="upper", interpolation="nearest")
                ax.set_title(f"seed {int(seeds[i])}", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            axes[r, 0].set_ylabel(label)
            fig.colorbar(im, ax=axes[r, :], shrink=.82,
                         label=f"{bar} (shared across Tarbert and Ness panels)")
    fig.suptitle("SPE10 horizontal x-y patches — normalized patch inputs, not raw contrast", fontsize=12)
    fig.savefig(out_dir / "spe10_horizontal_patches.png", dpi=180)
    plt.close(fig)


def _save_spe10_xz(npz, out_dir, count):
    import matplotlib.pyplot as plt

    cohorts = []
    for cohort in SPE10_XZ:
        kx, kz, phi = (_get(npz, f"{cohort}_{key}") for key in ("kx_mD", "kz_mD", "phi"))
        if kx is None or kz is None or phi is None:
            continue
        meta = _get(npz, f"{cohort}_metadata_json")
        cohorts.append((cohort, kx[:count], kz[:count], phi[:count], meta[:count]))
    if not cohorts:
        return
    # Use log10 mD and a common, robust scale for both directions and regions.
    logk = [np.log10(np.maximum(a, 1e-12)) for item in cohorts for a in (item[1], item[2])]
    kmin, kmax = _limits(logk, q=(0.01, 0.99), fallback=(-3.0, 4.0))
    pmin, pmax = _limits([item[3] for item in cohorts], q=(0.0, 0.99), fallback=(0.0, 0.4))
    n = min(len(item[1]) for item in cohorts)
    for direction, var_idx in (("Kx", 1), ("Kz", 2), ("Porosity", 3)):
        fig, axes = plt.subplots(len(cohorts), n, figsize=(6.8 * n, 1.0 * len(cohorts)),
                                 squeeze=False, constrained_layout=True)
        im = None
        for r, (cohort, kx, kz, phi, metas) in enumerate(cohorts):
            for c in range(n):
                meta = metas[c].item() if hasattr(metas[c], "item") else metas[c]
                dx = float(meta.get("dx_ft", 20.0))
                dz = float(meta.get("dz_ft", 2.0))
                if var_idx == 1:
                    data, vmin, vmax, cmap, bar = np.log10(np.maximum(kx[c], 1e-12)), kmin, kmax, "viridis", "log10 Kx (mD)"
                elif var_idx == 2:
                    data, vmin, vmax, cmap, bar = np.log10(np.maximum(kz[c], 1e-12)), kmin, kmax, "viridis", "log10 Kz (mD)"
                else:
                    data, vmin, vmax, cmap, bar = phi[c], pmin, pmax, "viridis", "porosity (fraction)"
                nz, nx = data.shape
                extent = (0, nx * dx, nz * dz, 0)
                ax = axes[r, c]
                im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper",
                               extent=extent, aspect="equal", interpolation="nearest")
                ax.set_title(f"{cohort.replace('spe10_xz_', '').replace('_', ' ').title()} | y={meta.get('spe10_y_index', '?')}", fontsize=8)
                ax.set_ylabel("z (ft)" if c == 0 else "")
                ax.set_xlabel("x (ft)")
        fig.colorbar(im, ax=axes, shrink=.82, label=bar)
        fig.suptitle(f"SPE10 vertical x-z sections: {direction} — physical aspect and coordinates", fontsize=12)
        fig.savefig(out_dir / f"spe10_vertical_xz_{direction.lower()}.png", dpi=180)
        plt.close(fig)


def _write_descriptor_summary(manifest_path, out_dir):
    if not manifest_path.exists():
        return
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    metrics = ("logk_std", "phi_mean", "phi_std", "high_k_fraction",
               "largest_high_k_component_fraction", "high_k_component_count")
    groups = {}
    for row in rows:
        groups.setdefault(row.get("cohort", "unknown"), []).append(row)
    summary = []
    for cohort, group in sorted(groups.items()):
        item = {"cohort": cohort, "n_fields": len(group)}
        for metric in metrics:
            vals = np.asarray([float(r[metric]) for r in group if r.get(metric)], dtype=float)
            if vals.size:
                item[f"{metric}_mean"] = float(vals.mean())
                item[f"{metric}_sd"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
                item[f"{metric}_min"] = float(vals.min())
                item[f"{metric}_max"] = float(vals.max())
        summary.append(item)
    out_file = out_dir / "geology_descriptor_summary.csv"
    columns = sorted({key for row in summary for key in row})
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader(); writer.writerows(summary)

    import matplotlib.pyplot as plt
    # Keep normalized synthetic/xy patches separate from raw directional SPE10
    # sections: their K contrast is intentionally on different scales.
    plot_groups = (
        ("normalized_inputs", [r for r in rows if not r.get("cohort", "").startswith("spe10_xz_")],
         "Normalized synthetic and horizontal-patch input descriptors"),
        ("raw_spe10_xz", [r for r in rows if r.get("cohort", "").startswith("spe10_xz_")],
         "Raw directional SPE10 x-z input descriptors"),
    )
    for suffix, subset, heading in plot_groups:
        if not subset:
            continue
        names = list(dict.fromkeys(r["cohort"] for r in subset))
        fig, axes = plt.subplots(1, 3, figsize=(max(11, 2.2 * len(names)), 4), constrained_layout=True)
        for ax, metric, title in zip(axes,
                ("logk_std", "high_k_fraction", "largest_high_k_component_fraction"),
                ("standard deviation of log K", "fraction above 80th-percentile K", "largest high-K component / cells")):
            data = [[float(r[metric]) for r in subset if r.get("cohort") == name and r.get(metric)]
                    for name in names]
            ax.boxplot(data, labels=names, showmeans=True)
            ax.set_title(title); ax.tick_params(axis="x", rotation=55); ax.grid(axis="y", alpha=.25)
        fig.suptitle(heading + " — descriptive only, before flow", fontsize=12)
        fig.savefig(out_dir / f"geology_descriptor_distributions_{suffix}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="folder containing co2_geology_cohorts.npz and optional manifest CSV")
    parser.add_argument("--out-dir", default=None, help="figure directory; defaults to input-dir/figures")
    parser.add_argument("--samples", type=int, default=6, help="number of fields per cohort to visualize")
    args = parser.parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = in_dir / "co2_geology_cohorts.npz"
    if not archive.exists():
        raise FileNotFoundError(f"Missing geology archive: {archive}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401
    with np.load(archive, allow_pickle=True) as npz:
        _save_synthetic(npz, out_dir, args.samples)
        _save_spe10_xy(npz, out_dir, args.samples)
        _save_spe10_xz(npz, out_dir, min(args.samples, 2))
    _write_descriptor_summary(in_dir / "co2_geology_manifest.csv", out_dir)
    print(f"Wrote visualization and summary files to {out_dir}")
    print("These are input-property diagnostics only; flow and equilibrium regimes require solver-derived measurements.")


if __name__ == "__main__":
    main()
