"""
analyze4.py -- answers for study v4 (Experiments 5-8). Run AFTER (or together with) analyze3.py on the same folder.
All comparisons are PAIRED over geologies, 95% bootstrap CIs. Welfare in uncalibrated solver units.

    python analyze4.py --dir seq3_results
"""
import argparse, glob, json, os
import numpy as np

rng = np.random.default_rng(0)
INFS = "inf"


def load(d):
    R = {}
    for f in sorted(glob.glob(os.path.join(d, "e[1-9]__*.json"))):
        r = json.load(open(f))
        R.setdefault((r["exp"], r["tag"]), {})[r["geo"]] = r
    return R


def boot(x, B=4000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    if len(x) < 2:
        return (float(x.mean()), np.nan, np.nan)
    m = rng.choice(x, (B, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt(t, d=1):
    m, lo, hi = t
    if np.isnan(m):
        return "n/a"
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]" if not np.isnan(lo) else f"{m:.{d}f}"


def stratum(m):
    return "low-perm (<=0.7)" if m <= 0.7 else ("high-perm (>=1.4)" if m >= 1.4 else "mid-perm")


def W(R, key, geos=None):
    d = R.get(key, {})
    return {g: r["welfare"] for g, r in d.items() if geos is None or g in geos}


def diff(R, a, b, key="welfare"):
    """paired a - b over common geologies -> (list, geos)"""
    da, db = R.get(a, {}), R.get(b, {})
    g = sorted(set(da) & set(db))
    return [da[i][key] - db[i][key] for i in g], g


def line(label, R, a, b, d=1):
    x, g = diff(R, a, b)
    return f"- **{label}**: {fmt(boot(x), d)}  (n={len(g)}; {np.mean(np.array(x) > 0) * 100:.0f}% up, {np.mean(np.array(x) < 0) * 100:.0f}% down)" if len(g) else f"- {label}: no paired data yet"


def gap_line(label, R, coop, nash):
    return line(label, R, coop, nash)


def share(x_gain, x_gap):
    return 100 * np.mean(x_gain) / np.mean(x_gap) if len(x_gap) and abs(np.mean(x_gap)) > 1e-9 else float("nan")


def main(d):
    R = load(d)
    out = []
    P = out.append
    n_files = sum(len(v) for v in R.values())
    P(f"# Study v4: Experiments 5-8\n\nEpisodes/jobs loaded (all experiments): {n_files}. Welfare units are uncalibrated solver units.\n")

    # ------------------------------------------------------------------ Exp 5
    P("## Experiment 5: repairing the belief\n")
    conds = [t for (e, t) in R if e == "e5"]
    order = ["b0", "b1_temper", "b2_modelerr", "b3_fields4", "b4_full"]
    conds = [c for c in order if ("e5", c) in R]
    if conds:
        T = max(len(R[("e5", c)]) for c in conds)
        P("Welfare under learning (Nash, study-v2 geometry, H=2) and distance to the perfect-information oracle on the same geologies.\n")
        P("| belief | n | welfare | oracle - welfare | vs b0 (paired) |")
        P("|---|---|---|---|---|")
        for c in conds:
            g = sorted(R[("e5", c)])
            w = np.array([R[("e5", c)][i]["welfare"] for i in g])
            orc = [R.get(("e1", "nash_H2"), {}).get(i, {}).get("welfare", np.nan) for i in g]
            gapv = np.array(orc) - w
            gapv = gapv[~np.isnan(gapv)]
            dv, dg = diff(R, ("e5", c), ("e5", "b0"))
            P(f"| {c} | {len(g)} | {fmt(boot(w))} | {fmt(boot(gapv))} | {fmt(boot(dv)) if c != 'b0' else '-'} |")
        P("\nCalibration by stage (targets: z_rms ~ 1, coverage ~ 90%). 'mult cov' = truth inside the posterior 90% interval of the "
          "permeability multiplier; 'z cov' = observed well pressures inside the one-step-ahead 90% predictive interval; "
          "'peak cov' = true field-peak pressure inside the predictive 90% interval.\n")
        for c in conds:
            runs = list(R[("e5", c)].values())
            Tn = max(len(r["log"]) for r in runs)
            rows = []
            for t in range(Tn):
                L = [(r["log"][t], r["mult"]) for r in runs if len(r["log"]) > t]
                err = np.mean([abs(e["mult_hat"] - m) for e, m in L])
                mc = np.mean([e["q05"] <= m <= e["q95"] for e, m in L]) * 100
                zr = np.mean([e["z_rms"] for e, _ in L if "z_rms" in e])
                zc = np.mean([e["z_cov90"] for e, _ in L if "z_cov90" in e]) * 100
                pc = np.mean([e["pm_q05"] - 0.05 <= e["pmax"] <= e["pm_q95"] + 0.05 for e, _ in L if "pm_q05" in e]) * 100
                es = np.mean([e["ess"] for e, _ in L])
                rows.append((err, mc, zr, zc, pc, es))
            P(f"**{c}** (n={len(runs)}, rejuvenations/episode {np.mean([r.get('n_rejuv', 0) for r in runs]):.1f})\n")
            P("| stage | |mult err| | mult cov % | z_rms | z cov % | peak cov % | ESS |")
            P("|---|---|---|---|---|---|---|")
            for t, (err, mc, zr, zc, pc, es) in enumerate(rows):
                P(f"| t{t} | {err:.3f} | {mc:.0f} | {zr:.2f} | {zc:.0f} | {pc:.0f} | {es:.1f} |")
            P("")
        P("Reading guide: a fix helps only if coverage moves toward 90% AND welfare (or the oracle gap) improves. "
          "If |mult err| stays flat while coverage is low, the posterior is narrowing without learning (overconfidence).\n")
    else:
        P("No Exp 5 results yet.\n")

    # ------------------------------------------------------------------ Exp 6
    P("## Experiment 6: do the weak-setting results survive a better planner (H=3)?\n")
    P("| layout (no permit) | n | nash H=3 | coop H=3 | coop - nash H=3 | coop - nash H=2 (same geologies) |")
    P("|---|---|---|---|---|---|")
    rows = [("v2", ("e1", "nash_H3"), ("e1", "coop_H3"), ("e1", "nash_H2"), ("e1", "coop_H2"))]
    for lay in ("far", "mid", "close"):
        rows.append((lay, ("e6", f"{lay}_inf_H3_nash"), ("e6", f"{lay}_inf_H3_coop"),
                     ("e3", f"{lay}_capinf_nash"), ("e3", f"{lay}_capinf_coop")))
    for lab, nk, ck, n2, c2 in rows:
        g = sorted(set(R.get(nk, {})) & set(R.get(ck, {})) & set(R.get(n2, {})) & set(R.get(c2, {})))
        if not g:
            P(f"| {lab} | 0 | | | | |")
            continue
        n3 = np.array([R[nk][i]["welfare"] for i in g]); c3 = np.array([R[ck][i]["welfare"] for i in g])
        n2v = np.array([R[n2][i]["welfare"] for i in g]); c2v = np.array([R[c2][i]["welfare"] for i in g])
        P(f"| {lab} | {len(g)} | {fmt(boot(n3))} | {fmt(boot(c3))} | {fmt(boot(c3 - n3))} | {fmt(boot(c2v - n2v))} |")
    P("\nPolicies at H=3 (weak setting, perfect information):\n")
    P("| policy | n | gain over base Nash H=3 | share of (coop - nash) H=3 gap closed |")
    P("|---|---|---|---|")
    for pol in ("m2", "m3", "causer"):
        key = ("e6", f"weak_{pol}_H3")
        x, g = diff(R, key, ("e1", "nash_H3"))
        gap, gg = diff(R, ("e1", "coop_H3"), ("e1", "nash_H3"))
        gapd = {i: v for i, v in zip(gg, gap)}
        gapc = [gapd[i] for i in g if i in gapd]
        P(f"| {pol} | {len(g)} | {fmt(boot(x))} | {share(x, gapc):.0f}% |" if len(g) else f"| {pol} | 0 | | |")
    P("")
    P(line("extrapolated continuation vs none at H=3 (none - with)", R, ("e6", "weak_base_H3_noext"), ("e1", "nash_H3")))
    P("")

    # ------------------------------------------------------------------ Exp 7
    P("## Experiment 7: learning arms with the repaired belief (b4)\n")
    c7 = ["nash_prior", "nash_learn", "nash_learn_pilot", "coop_learn"]
    P("| condition | n | welfare |")
    P("|---|---|---|")
    for c in c7 + ["weak_m2_learn", "weak_causer_learn"]:
        w = list(W(R, ("e7", c)).values())
        P(f"| {c} | {len(w)} | {fmt(boot(w))} |")
    orc = list(W(R, ("e1", "nash_H2")).values())
    P(f"| oracle (perfect info, Nash H=2) | {len(orc)} | {fmt(boot(orc))} |\n")
    P(line("value of monitoring (learn - prior)", R, ("e7", "nash_learn"), ("e7", "nash_prior")))
    P(line("value of perfect information (oracle - learn)", R, ("e1", "nash_H2"), ("e7", "nash_learn")))
    P(line("ramp-up effect (pilot - learn)", R, ("e7", "nash_learn_pilot"), ("e7", "nash_learn")))
    P(line("coop - nash under learning", R, ("e7", "coop_learn"), ("e7", "nash_learn")))
    P(line("repaired belief vs v3 belief (e7 learn - e2 learn), same geologies", R, ("e7", "nash_learn"), ("e2", "nash_learn")))
    P("")
    x, g = diff(R, ("e7", "nash_learn_pilot"), ("e7", "nash_learn"))
    if g:
        P("Ramp-up by permeability stratum (pilot - learn):\n")
        P("| stratum | n | effect |")
        P("|---|---|---|")
        for s in ("low-perm (<=0.7)", "mid-perm", "high-perm (>=1.4)"):
            sel = [v for v, i in zip(x, g) if stratum(R[("e7", "nash_learn")][i]["mult"]) == s]
            P(f"| {s} | {len(sel)} | {fmt(boot(sel))} |")
        P("")
    P("Policies under learning (gain over Nash-learn) next to the same policy with perfect information:\n")
    P("| policy | n | gain, learning | gain, perfect info | difference (precaution effect) |")
    P("|---|---|---|---|---|")
    for pol, lk, ok in (("m2", ("e7", "weak_m2_learn"), ("e4", "weak_m2_oracle")),
                        ("causer", ("e7", "weak_causer_learn"), ("e4", "weak_causer_oracle"))):
        xl, gl = diff(R, lk, ("e7", "nash_learn"))
        xo, go = diff(R, ok, ("e4", "weak_base_oracle"))
        od = {i: v for i, v in zip(go, xo)}
        gg = [i for i in gl if i in od]
        a = [xl[gl.index(i)] for i in gg]; b = [od[i] for i in gg]
        P(f"| {pol} | {len(gg)} | {fmt(boot(a))} | {fmt(boot(b))} | {fmt(boot(np.array(a) - np.array(b)))} |" if gg else f"| {pol} | 0 | | | |")
    P("\nSpacing under uncertainty (coop - nash), learning vs perfect information on the same geologies:\n")
    P("| layout | n | learning | perfect info |")
    P("|---|---|---|---|")
    for lay in ("far", "close"):
        xl, gl = diff(R, ("e7", f"{lay}_learn_coop"), ("e7", f"{lay}_learn_nash"))
        xo, go = diff(R, ("e3", f"{lay}_capinf_coop"), ("e3", f"{lay}_capinf_nash"))
        od = {i: v for i, v in zip(go, xo)}
        gg = [i for i in gl if i in od]
        a = [xl[gl.index(i)] for i in gg]; b = [od[i] for i in gg]
        P(f"| {lay} | {len(gg)} | {fmt(boot(a))} | {fmt(boot(b))} |" if gg else f"| {lay} | 0 | | |")
    P("")

    # ------------------------------------------------------------------ Exp 8
    P("## Experiment 8: liability-multiplier dose-response (perfect information, Nash)\n")
    for setting, base, gapk in (("weak", ("e4", "weak_base_oracle"), (("e1", "coop_H2"), ("e1", "nash_H2"))),
                                ("strong", ("e4", "strong_base_oracle"), (("e3", "close_cap24_coop"), ("e3", "close_cap24_nash")))):
        gap, gg = diff(R, gapk[0], gapk[1])
        P(f"**{setting} setting** (gap to close: {fmt(boot(gap))}, n={len(gg)})\n")
        P("| m | n | gain over m=1 | share of gap closed |")
        P("|---|---|---|---|")
        keys = [(1.25, ("e8", f"{setting}_m1.25")), (1.5, ("e8", f"{setting}_m1.5")), (2.0, ("e4", f"{setting}_m2_oracle")),
                (2.5, ("e8", f"{setting}_m2.5")), (3.0, ("e4", f"{setting}_m3_oracle")), (4.0, ("e8", f"{setting}_m4"))]
        gd = {i: v for i, v in zip(gg, gap)}
        for m, k in keys:
            x, g = diff(R, k, base)
            gp = [gd[i] for i in g if i in gd]
            P(f"| {m:g} | {len(g)} | {fmt(boot(x))} | {share(x, gp):.0f}% |" if len(g) else f"| {m:g} | 0 | | |")
        P("")
    P("A non-monotonic curve (gain peaks near one multiplier and falls past it) means over-correction: the "
      "liability price stops being a transfer and starts distorting production.\n")

    P("## Caveats\n- Receding-horizon planner: see Experiment 6 and study v3 Experiment 1 for how far from equilibrium.\n"
      "- The true geology's random field is never in the belief's field pool, so the multiplier posterior also absorbs "
      "field mismatch; use the predictive checks (z, peak coverage), not only multiplier coverage.\n"
      "- Welfare units are uncalibrated; signs and ranks are the findings. 12x12 grid, closed domain, all wells injectors.\n")
    txt = "\n".join(out)
    open(os.path.join(d, "analysis4.md"), "w").write(txt)
    print(txt)
    try:
        plots(R, d)
    except Exception as e:  # plotting must never hide the tables
        print(f"(plot skipped: {e})")


def plots(R, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for c in [c for c in ("b0", "b1_temper", "b2_modelerr", "b3_fields4", "b4_full") if ("e5", c) in R]:
        runs = list(R[("e5", c)].values())
        Tn = min(len(r["log"]) for r in runs)
        ax[0].plot([np.mean([r["log"][t]["z_cov90"] for r in runs]) * 100 for t in range(Tn)], marker="o", label=c)
        ax[1].plot([np.mean([r["log"][t]["q05"] <= r["mult"] <= r["log"][t]["q95"] for r in runs]) * 100 for t in range(Tn)], marker="o", label=c)
    ax[0].set_title("predictive 90% coverage of well pressures"); ax[1].set_title("posterior 90% coverage of multiplier")
    for a in ax[:2]:
        a.axhline(90, ls="--", c="k", lw=0.8); a.set_xlabel("stage"); a.set_ylabel("%"); a.legend(fontsize=7)
    for setting, base in (("weak", ("e4", "weak_base_oracle")), ("strong", ("e4", "strong_base_oracle"))):
        ms, gs = [], []
        for m, k in ((1.25, ("e8", f"{setting}_m1.25")), (1.5, ("e8", f"{setting}_m1.5")), (2.0, ("e4", f"{setting}_m2_oracle")),
                     (2.5, ("e8", f"{setting}_m2.5")), (3.0, ("e4", f"{setting}_m3_oracle")), (4.0, ("e8", f"{setting}_m4"))):
            x, g = diff(R, k, base)
            if len(g):
                ms.append(m); gs.append(np.mean(x))
        ax[2].plot([1.0] + ms, [0.0] + gs, marker="o", label=setting)
    ax[2].set_title("welfare gain vs liability multiplier m"); ax[2].set_xlabel("m"); ax[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(d, "study4.png"), dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="seq3_results")
    main(ap.parse_args().dir)
