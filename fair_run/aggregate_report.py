"""
aggregate_report.py -- turns the per-run JSONs into the final table + verdict.

    python aggregate_report.py --out final_run

Writes <out>/report/final_table.csv and <out>/report/final_verdict.md.
Ranking metric = test "balanced" error = mean(pressure rel-L2, saturation
rel-L2), averaged over seeds. A difference smaller than the seed-to-seed std
of the leader is flagged as a statistical tie rather than a win.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="final_run")
    a = ap.parse_args()

    runs = {}
    for p in sorted(glob.glob(os.path.join(a.out, "results", "*.json"))):
        r = json.load(open(p))
        runs.setdefault(r["arch"], []).append(r)
    bench_path = os.path.join(a.out, "bench", "inference.json")
    bench = json.load(open(bench_path)) if os.path.exists(bench_path) else {}

    rows = []
    for arch, rs in runs.items():
        good = [r for r in rs if r.get("status") == "complete" and "test" in r]
        if not good:
            rows.append({"arch": arch, "n_seeds": 0, "diverged": len(rs)})
            continue
        g = lambda k, sub="test": np.array([r[sub][k] for r in good])
        row = {
            "arch": arch, "n_seeds": len(good), "diverged": len(rs) - len(good),
            "lr": good[0]["lr"], "pw": good[0]["pw"], "params": good[0]["n_params"],
            "bal_mean": g("balanced").mean(), "bal_std": g("balanced").std(),
            "p_mean": g("rel_pressure").mean(), "p_std": g("rel_pressure").std(),
            "s_mean": g("rel_saturation").mean(), "s_std": g("rel_saturation").std(),
            "comb_mean": g("rel_combined").mean(), "p95_mean": g("balanced_p95").mean(),
            "best_epoch_mean": float(np.mean([r["best_epoch"] for r in good])),
            "epochs": good[0]["epochs"], "train_min": float(np.mean([r["train_seconds"] for r in good])) / 60,
            "ms_bs1": bench.get(arch, {}).get("ms_per_sample_bs1"),
            "ms_bs64": bench.get(arch, {}).get("ms_per_sample_bs64"),
        }
        rows.append(row)

    ok = sorted([r for r in rows if r.get("n_seeds")], key=lambda r: r["bal_mean"])
    bad = [r for r in rows if not r.get("n_seeds")]
    os.makedirs(os.path.join(a.out, "report"), exist_ok=True)

    with open(os.path.join(a.out, "report", "final_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)

    L = ["# Final surrogate comparison (fair protocol)\n"]
    L.append("Ranking metric: test **balanced** relative-L2 = mean(pressure, saturation), mean ± std over seeds. "
             "All numbers are on the held-out test split, at the best-validation checkpoint.\n")
    L.append("| rank | architecture | balanced | pressure | saturation | combined* | p95 balanced | params | lr (pw) | best ep | ms/sample bs1 | ms/sample bs64 | seeds |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    fmt = lambda v, d=3: "n/a" if v is None else f"{v:.{d}f}"
    for i, r in enumerate(ok, 1):
        pw = f" (pw={r['pw']:g})" if r["pw"] else ""
        L.append(f"| {i} | {r['arch']} | {r['bal_mean']:.4f} ± {r['bal_std']:.4f} | {r['p_mean']:.4f} ± {r['p_std']:.4f} | "
                 f"{r['s_mean']:.4f} ± {r['s_std']:.4f} | {r['comb_mean']:.4f} | {r['p95_mean']:.4f} | {r['params']:,} | "
                 f"{r['lr']:g}{pw} | {r['best_epoch_mean']:.0f}/{r['epochs']} | {fmt(r['ms_bs1'])} | {fmt(r['ms_bs64'], 4)} | {r['n_seeds']} |")
    L.append("\n*combined = relative L2 of the stacked [pressure, saturation] fields (the earlier table's definition); "
             "it is dominated by pressure because pressure has the larger magnitude.\n")
    if bad:
        L.append("**No usable result:** " + ", ".join(f"{r['arch']} (diverged runs: {r['diverged']})" for r in bad) + "\n")

    if ok:
        best = ok[0]
        L.append("## Verdict\n")
        L.append(f"**Most accurate overall: {best['arch']}** (balanced {best['bal_mean']:.4f} ± {best['bal_std']:.4f}).")
        if len(ok) > 1:
            second = ok[1]
            gap = second["bal_mean"] - best["bal_mean"]
            if gap < best["bal_std"]:
                L.append(f"- {second['arch']} is within one seed-std of the leader ({gap:.4f} < {best['bal_std']:.4f}): "
                         "treat as a **statistical tie**; decide on speed/params.")
            else:
                L.append(f"- Leads {second['arch']} by {gap:.4f} ({gap / max(second['bal_mean'], 1e-9):.0%} relative), "
                         f"larger than its seed-std ({best['bal_std']:.4f}).")
        bp = min(ok, key=lambda r: r["p_mean"]); bs = min(ok, key=lambda r: r["s_mean"])
        L.append(f"- Best pressure: {bp['arch']} ({bp['p_mean']:.4f}); best saturation: {bs['arch']} ({bs['s_mean']:.4f})"
                 f"{' -- same model' if bp['arch'] == bs['arch'] else ' -- different models; weigh by which channel your payoff uses'}.")
        timed = [r for r in ok if r["ms_bs64"] is not None]
        if timed:
            fast = min(timed, key=lambda r: r["ms_bs64"])
            L.append(f"- Fastest (bs64): {fast['arch']} at {fast['ms_bs64']:.4f} ms/sample.")
            # Pareto front on (balanced error, ms_bs64)
            front = [r for r in timed if not any(o["bal_mean"] <= r["bal_mean"] and o["ms_bs64"] <= r["ms_bs64"]
                                                 and (o["bal_mean"] < r["bal_mean"] or o["ms_bs64"] < r["ms_bs64"]) for o in timed)]
            L.append("- Pareto-optimal (error vs speed): " + ", ".join(r["arch"] for r in sorted(front, key=lambda r: r["bal_mean"])))
        else:
            L.append("- Speed not benchmarked yet: run `benchmark_inference.py`.")
        trained_traj = [r["arch"] for r in ok if r["arch"] in ("e2c", "runet_traj", "pi_convlstm_traj", "gns_traj")]
        if trained_traj:
            L.append("- NOTE: " + ", ".join(trained_traj) + " use intermediate-time (trajectory) supervision that the other models do not; "
                     "compare each to its final-frame-only twin in the section below.")
        slow = [r["arch"] for r in ok if r["arch"] in ("finn", "gns", "gns_traj") and r["n_seeds"] < 3]
        if slow:
            L.append("- NOTE: " + ", ".join(slow) + " were run with fewer seeds (slow, not contenders): their std is less reliable.")
        L.append("- NOTE: physics-informed variants use the pressure-equation residual only (no velocity in the data), "
                 "with the physics weight chosen on validation from {3, 10, 30, 100}.")
    # ---- trajectory-supervision pairs ---------------------------------------
    by = {r["arch"]: r for r in ok}
    pairs = [("runet", "runet_traj"), ("pi_convlstm", "pi_convlstm_traj"), ("gns", "gns_traj")]
    pairs = [(a_, b_) for a_, b_ in pairs if a_ in by and b_ in by]
    if pairs:
        L.append("\n## Effect of intermediate-time (trajectory) supervision\n")
        L.append("| backbone | final-frame only | trajectory-trained | change |")
        L.append("|---|---|---|---|")
        for a_, b_ in pairs:
            x, y = by[a_]["bal_mean"], by[b_]["bal_mean"]
            L.append(f"| {a_} | {x:.4f} | {y:.4f} | {(y - x) / x:+.0%} |")

    # ---- OOD ------------------------------------------------------------------
    ood_path = os.path.join(a.out, "ood", "ood_results.json")
    if os.path.exists(ood_path):
        ood = json.load(open(ood_path))
        names = sorted({n for v in ood.values() for n in v})
        L.append("\n## Out-of-distribution robustness (balanced rel-L2, mean over seeds)\n")
        L.append("ood_more_wells = 5-7 wells (train 2-4); ood_rough_perm = corr. length 2 (train 6); "
                 "ood_smooth_perm = corr. length 12. Ratio = OOD error / in-distribution test error.\n")
        L.append("| architecture | in-dist | " + " | ".join(f"{n} (ratio)" for n in names) + " | mean OOD |")
        L.append("|---|---|" + "---|" * (len(names) + 1))
        rows_o = []
        for r in ok:
            if r["arch"] not in ood:
                continue
            vals = [ood[r["arch"]][n]["balanced"] for n in names if n in ood[r["arch"]]]
            rows_o.append((float(np.mean(vals)), r, ood[r["arch"]]))
        for m_, r, o in sorted(rows_o, key=lambda t: t[0]):
            cells = " | ".join(f"{o[n]['balanced']:.4f} ({o[n]['balanced'] / r['bal_mean']:.2f}x)" if n in o else "n/a" for n in names)
            L.append(f"| {r['arch']} | {r['bal_mean']:.4f} | {cells} | {m_:.4f} |")
        if rows_o:
            best_ood = min(rows_o, key=lambda t: t[0])
            L.append(f"\n- Most robust (lowest mean OOD error): **{best_ood[1]['arch']}** ({best_ood[0]:.4f}).")
            if ok and best_ood[1]["arch"] != ok[0]["arch"]:
                L.append(f"- NOTE: the in-distribution winner ({ok[0]['arch']}) is not the most robust model; decide which matters for your use.")

    open(os.path.join(a.out, "report", "final_verdict.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
