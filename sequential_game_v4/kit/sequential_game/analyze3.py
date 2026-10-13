"""
analyze3.py -- turns seq3_results/*.json (run_study3.py) into the answers for Experiments 1-4.
All comparisons are PAIRED over geologies, with 95% bootstrap CIs. Welfare is in uncalibrated
solver units (relative scale only).

    python analyze3.py --dir seq3_results [--baseline_dir <study-v2 results>]
"""
import argparse, glob, json, os
import numpy as np

rng = np.random.default_rng(0)


def load(d):
    R = {}
    for f in sorted(glob.glob(os.path.join(d, "e[1-4]__*.json"))):
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


def main(d, out, baseline_dir=None):
    R = load(d)
    L = []
    P = L.append
    W = lambda r: r["welfare"]

    def geos(*keys):
        s = None
        for k in keys:
            g = set(R.get(k, {}))
            s = g if s is None else s & g
        return sorted(s or [])

    def diff(a, b, f=W):
        g = geos(a, b)
        return g, np.array([f(R[a][i]) - f(R[b][i]) for i in g])

    def line(title, a, b, f=W, note=""):
        g, x = diff(a, b, f)
        if len(g) == 0:
            P(f"- {title}: no paired data yet")
            return g, x
        P(f"- **{title}**: {fmt(boot(x))}  (n={len(g)}; {np.mean(x>1e-6)*100:.0f}% up, {np.mean(x<-1e-6)*100:.0f}% down) {note}")
        return g, x

    n_files = sum(len(v) for v in R.values())
    P("# Study v3: Experiments 1-4\n")
    P(f"Episodes/jobs loaded: {n_files}. Welfare units are uncalibrated solver units.\n")

    # ------------------------------------------------------------------ Exp 1
    P("## Experiment 1: equilibrium quality\n")
    P("### 1a. Planner horizon convergence (perfect information, study-v2 geometry)\n")
    P("| horizon H | n | nash welfare | coop welfare | coop - nash |\n|---|---|---|---|---|")
    for H in (1, 2, 3, 4, 6):
        a, b = ("e1", f"nash_H{H}"), ("e1", f"coop_H{H}")
        gn, gc = geos(a), geos(b)
        gg, x = diff(b, a)
        P(f"| {H} | {len(gg)} | {fmt(boot([W(R[a][i]) for i in gn]))} | {fmt(boot([W(R[b][i]) for i in gc]))} | {fmt(boot(x))} |")
    P("")
    for mo in ("nash", "coop"):
        for H in (1, 2, 3, 4):
            line(f"{mo}: H={H} minus H=6 (how much a longer lookahead changes welfare)", ("e1", f"{mo}_H{H}"), ("e1", f"{mo}_H6"))
    line("nash H=2: no continuation extrapolation minus with extrapolation", ("e1", "nash_H2_noextrap"), ("e1", "nash_H2"))

    P("\n### 1b. Open-loop deviation epsilon (greedy search, so a LOWER bound on the true epsilon)\n")
    for tag, lab in (("eps_nash_H2", "Nash path: best unilateral schedule change (max over operators)"),
                     ("eps_coop_H2", "Coop path: welfare gained by changing any single action (planner sub-optimality)")):
        rr = R.get(("e1", tag), {})
        if rr:
            key = "max" if "nash" in tag else "sum"
            e = np.array([r["eps"][key] for r in rr.values()])
            wel = np.array([r["eps"]["base_welfare"] for r in rr.values()])
            P(f"- **{lab}**: {fmt(boot(e))} welfare units = {fmt(boot(100*e/np.maximum(np.abs(wel),1)))}% of welfare; "
              f">1 unit in {np.mean(e>1)*100:.0f}% of geologies (n={len(e)})")
    gg, gap = diff(("e1", "coop_H2"), ("e1", "nash_H2"))
    if len(gg):
        P(f"- For scale, the measured cost of non-cooperation at H=2 is {fmt(boot(gap))} (n={len(gg)}). "
          "If it is not clearly larger than the epsilons above, it is not distinguishable from planner error.")

    P("\n### 1c. One-shot-deviation epsilon with reactive continuation (Nash planner, H=2)\n")
    P("| stage t0 | n | eps (max over operators) | geologies with eps > 1 |\n|---|---|---|---|")
    for t0 in (0, 1, 2):
        rr = R.get(("e1", f"osd_t{t0}"), {})
        if rr:
            e = np.array([r["eps"]["max"] for r in rr.values()])
            P(f"| {t0} | {len(e)} | {fmt(boot(e))} | {np.mean(e>1)*100:.0f}% |")

    # ------------------------------------------------------------------ Exp 2
    P("\n## Experiment 2: belief fix and ramp-up\n")
    prior, learn, pilot, oracle = ("e2", "nash_prior"), ("e2", "nash_learn"), ("e2", "nash_learn_pilot"), ("e1", "nash_H2")
    P("| condition | n | welfare | peak pressure |\n|---|---|---|---|")
    for lab, k in (("prior only", prior), ("learn (v3 belief)", learn), ("learn + ramp-up (stage-0 rate <= 0.5)", pilot), ("oracle", oracle)):
        v = list(R.get(k, {}).values())
        if v:
            P(f"| {lab} | {len(v)} | {fmt(boot([W(r) for r in v]))} | {np.mean([r['pmax_peak'] for r in v]):.2f} |")
    P("")
    line("value of monitoring (learn - prior)", learn, prior)
    line("value of perfect information (oracle - learn)", oracle, learn)
    g1, x1 = diff(learn, prior)
    g2, x2 = diff(oracle, prior)
    c = sorted(set(g1) & set(g2))
    if c:
        a_ = np.mean([x1[g1.index(i)] for i in c]); b_ = np.mean([x2[g2.index(i)] for i in c])
        if abs(b_) > 1e-6:
            P(f"- monitoring captures {100*a_/b_:.0f}% of the oracle-minus-prior value (n={len(c)})")
    line("ramp-up effect (learn+ramp-up - learn)", pilot, learn)
    line("coop under learning minus nash under learning", ("e2", "coop_learn"), learn)
    P("\nBy permeability stratum:\n")
    P("| stratum | n | learn - prior | oracle - learn | ramp-up - learn |\n|---|---|---|---|---|")
    for s in ("low-perm (<=0.7)", "mid-perm", "high-perm (>=1.4)"):
        gs = [g for g, r in R.get(learn, {}).items() if stratum(r["mult"]) == s]
        def sub(a, b):
            g, x = diff(a, b)
            return fmt(boot([xx for gg_, xx in zip(g, x) if gg_ in gs]))
        P(f"| {s} | {len(gs)} | {sub(learn, prior)} | {sub(oracle, learn)} | {sub(pilot, learn)} |")

    P("\n### Belief diagnostics (learn mode)\n")
    v = list(R.get(learn, {}).values())
    if v:
        T = len(v[0]["log"])
        P("- effective sample size by stage: " + ", ".join(f"t{t}: {np.mean([r['log'][t]['ess'] for r in v]):.1f}" for t in range(T)) + " (of 16 particles)")
        P("- |posterior-mean multiplier - truth| by stage: " + ", ".join(f"t{t}: {np.mean([abs(r['log'][t]['mult_hat']-r['mult']) for r in v]):.3f}" for t in range(T)))
        cov = [np.mean([r['log'][t]['q05'] <= r['mult'] <= r['log'][t]['q95'] for r in v]) for t in range(T)]
        P("- truth inside the posterior 90% interval, by stage: " + ", ".join(f"t{t}: {100*cv:.0f}%" for t, cv in enumerate(cov)) + " (target ~90%)")
        P(f"- rejuvenation events per episode: {np.mean([r['n_rejuv'] for r in v]):.1f}")
    pv = list(R.get(prior, {}).values())
    if pv:
        P(f"- no-learning baseline error: {np.mean([abs(r['log'][-1]['mult_hat']-r['mult']) for r in pv]):.3f}")
    if baseline_dir:
        old = {}
        for f in glob.glob(os.path.join(baseline_dir, "**", "n2_g*_nash_learn_m1.json"), recursive=True):
            r = json.load(open(f)); old[r["geo"]] = r
        gcm = sorted(set(old) & set(R.get(learn, {})))
        if gcm:
            dx = np.array([R[learn][i]["welfare"] - old[i]["welfare"] for i in gcm])
            P(f"\n- vs study-v2 belief (10 particles, collapsing): new learn minus old learn = {fmt(boot(dx))} (n={len(gcm)})")
            eo = np.mean([abs(old[i]['log'][-1]['mult_hat'] - old[i]['mult']) for i in gcm])
            en = np.mean([abs(R[learn][i]['log'][-1]['mult_hat'] - R[learn][i]['mult']) for i in gcm])
            P(f"- final posterior error: old {eo:.3f} vs new {en:.3f}; old final ESS {np.mean([old[i]['log'][-1]['ess'] for i in gcm]):.1f}")

    # ------------------------------------------------------------------ Exp 3
    P("\n## Experiment 3: externality strength (perfect information)\n")
    P("| layout | permit | n | nash | coop | coop - nash | coop/nash | peak p (nash / coop) |\n|---|---|---|---|---|---|---|---|")
    cfgs = [("far", "inf"), ("mid", "inf"), ("close", "inf"), ("v2", "24"), ("far", "24"), ("mid", "24"), ("close", "24"), ("close", "16")]
    for lay, cap in cfgs:
        a, b = ("e3", f"{lay}_cap{cap}_nash"), ("e3", f"{lay}_cap{cap}_coop")
        if lay == "v2" and cap == "inf":
            a, b = ("e1", "nash_H2"), ("e1", "coop_H2")
        g = geos(a, b)
        if not g:
            continue
        gap = np.array([W(R[b][i]) - W(R[a][i]) for i in g])
        ratio = np.array([W(R[b][i]) / W(R[a][i]) for i in g if W(R[a][i]) > 1])
        P(f"| {lay} | {cap} | {len(g)} | {fmt(boot([W(R[a][i]) for i in g]))} | {fmt(boot([W(R[b][i]) for i in g]))} | "
          f"{fmt(boot(gap))} | {np.mean(ratio):.3f} | {np.mean([R[a][i]['pmax_peak'] for i in g]):.1f} / {np.mean([R[b][i]['pmax_peak'] for i in g]):.1f} |")
    P("\nCooperation gap vs spacing (permit off):")
    for lay in ("far", "mid", "close"):
        g, x = diff(("e3", f"{lay}_capinf_coop"), ("e3", f"{lay}_capinf_nash"))
        if len(g):
            P(f"- {lay}: {fmt(boot(x))} (n={len(g)})")
    P("\nCooperation gap vs permit (close layout):")
    for cap in ("inf", "24", "16"):
        g, x = diff(("e3", f"close_cap{cap}_coop"), ("e3", f"close_cap{cap}_nash"))
        if len(g):
            P(f"- permit {cap}: {fmt(boot(x))} (n={len(g)})")
    P("\nUnder uncertainty (learning, v3 belief), spacing only:")
    for lay in ("far", "close"):
        g, x = diff(("e3", f"{lay}_learn_coop"), ("e3", f"{lay}_learn_nash"))
        g0, x0 = diff(("e3", f"{lay}_capinf_coop"), ("e3", f"{lay}_capinf_nash"))
        cm = sorted(set(g) & set(g0))
        if len(g):
            P(f"- {lay}: coop - nash with learning {fmt(boot(x))} (n={len(g)})"
              + (f"; same geologies with perfect info {fmt(boot([x0[g0.index(i)] for i in cm]))}" if cm else ""))

    # ------------------------------------------------------------------ Exp 4
    P("\n## Experiment 4: policy decomposition\n")
    P("Welfare is always scored with the true penalty; policies change only what operators optimise (or their feasible set for the quota).")
    for setting, ref_nash, ref_coop in (("weak", ("e1", "nash_H2"), ("e1", "coop_H2")),
                                        ("strong", ("e3", "close_cap24_nash"), ("e3", "close_cap24_coop"))):
        P(f"\n### Setting: {setting} ({'study-v2 geometry, no permit' if setting=='weak' else 'close wells + permit 24'}), perfect information\n")
        P("| policy | n | gain over base policy | share of coop-minus-nash gap closed |\n|---|---|---|---|")
        base = ("e4", f"{setting}_base_oracle")
        for pol in ("m2", "m3", "causer", "quota"):
            k = ("e4", f"{setting}_{pol}_oracle")
            g, x = diff(k, base)
            gc, gap = diff(ref_coop, base)
            cm = sorted(set(g) & set(gc))
            share = (np.mean([x[g.index(i)] for i in cm]) / np.mean([gap[gc.index(i)] for i in cm])) if cm and abs(np.mean([gap[gc.index(i)] for i in cm])) > 1e-6 else float("nan")
            P(f"| {pol} | {len(g)} | {fmt(boot(x))} | {share*100:.0f}% |" if not np.isnan(share) else f"| {pol} | {len(g)} | {fmt(boot(x))} | n/a |")
        gcb, gapb = diff(ref_coop, base)
        if len(gcb):
            P(f"\nCoop minus base Nash (the gap to close): {fmt(boot(gapb))} (n={len(gcb)})")
    P("\n### Learning versus perfect information: externality correction vs precaution (weak setting)\n")
    P("| policy | gain with perfect information | gain with learning | difference = precaution effect |\n|---|---|---|---|")
    base_l = ("e2", "nash_learn")
    for pol in ("m2", "causer", "quota"):
        kl, ko = ("e4", f"weak_{pol}_learn"), ("e4", f"weak_{pol}_oracle")
        bo = ("e4", "weak_base_oracle")
        g = geos(kl, ko, bo, base_l)
        if not g:
            continue
        gl = np.array([W(R[kl][i]) - W(R[base_l][i]) for i in g])
        go = np.array([W(R[ko][i]) - W(R[bo][i]) for i in g])
        P(f"| {pol} (n={len(g)}) | {fmt(boot(go))} | {fmt(boot(gl))} | {fmt(boot(gl-go))} |")

    P("\n## Caveats\n- Receding-horizon planner, not a full equilibrium: see 1a-1c for how far it is from one.\n"
      "- Greedy deviation search gives lower bounds on epsilon.\n- Welfare units are uncalibrated; the sign and rank of effects are the findings, not the levels.\n"
      "- 12x12 grid, all wells injectors in a closed domain.")
    txt = "\n".join(L)
    open(os.path.join(out, "analysis3.md"), "w").write(txt)
    print(txt)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        labs, vals, errs = [], [], []
        for lay, cap in cfgs:
            a, b = ("e3", f"{lay}_cap{cap}_nash"), ("e3", f"{lay}_cap{cap}_coop")
            if lay == "v2" and cap == "inf":
                continue
            g, x = diff(b, a)
            if len(g):
                m, lo, hi = boot(x); labs.append(f"{lay}\n{cap}"); vals.append(m); errs.append([[m - lo if not np.isnan(lo) else 0], [hi - m if not np.isnan(hi) else 0]])
        if labs:
            ax[0].bar(range(len(labs)), vals, yerr=np.array(errs).reshape(len(labs), 2).T, color="#4C78A8")
            ax[0].set_xticks(range(len(labs))); ax[0].set_xticklabels(labs, fontsize=7); ax[0].set_title("Cooperation gap by layout and permit"); ax[0].set_ylabel("coop - nash (welfare)")
        labs, vals, errs = [], [], []
        for pol in ("m2", "m3", "causer", "quota"):
            g, x = diff(("e4", f"strong_{pol}_oracle"), ("e4", "strong_base_oracle"))
            if len(g):
                m, lo, hi = boot(x); labs.append(pol); vals.append(m); errs.append([m - lo if not np.isnan(lo) else 0, hi - m if not np.isnan(hi) else 0])
        if labs:
            ax[1].bar(range(len(labs)), vals, yerr=np.array(errs).T, color="#E45756")
            ax[1].set_xticks(range(len(labs))); ax[1].set_xticklabels(labs); ax[1].set_title("Policy gain, strong-externality setting")
        plt.tight_layout(); plt.savefig(os.path.join(out, "study3.png"), dpi=140)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="seq3_results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline_dir", default=None)
    a = ap.parse_args()
    main(a.dir, a.out or a.dir, a.baseline_dir)
