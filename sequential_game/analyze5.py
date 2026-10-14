"""
analyze5.py -- answers for study v5 (Experiments 9-11): the permit-commons result.
Paired over geologies, 95% bootstrap CIs, welfare in uncalibrated solver units.

    python analyze5.py --dir seq3_results
"""
import argparse, glob, json, os
import numpy as np

from analyze4 import boot, fmt, diff, share, stratum

STRONG_NASH, STRONG_COOP = ("e3", "close_cap24_nash"), ("e3", "close_cap24_coop")   # perfect-information H=6 baselines


def load(d):
    R = {}
    for f in sorted(glob.glob(os.path.join(d, "e*__*.json"))):
        r = json.load(open(f))
        R.setdefault((r["exp"], r["tag"]), {})[r["geo"]] = r
    return R


def line(label, R, a, b, d=1):
    x, g = diff(R, a, b)
    if not len(g):
        return f"- {label}: no paired data yet"
    return f"- **{label}**: {fmt(boot(x), d)}  (n={len(g)}; {np.mean(np.array(x) > 0) * 100:.0f}% up, {np.mean(np.array(x) < 0) * 100:.0f}% down)"


def wel(R, key):
    return [r["welfare"] for r in R.get(key, {}).values()]


def early_rate(r, stages=3):
    """average chosen rate over the first stages (race / front-loading indicator)"""
    return float(np.mean([np.mean(e["rates"]) for e in r["log"][:stages]]))


def main(d):
    R = load(d)
    out, P = [], None
    P = out.append
    P("# Study v5: the permit commons (Experiments 9-11)\n")
    P("Setting: close wells + shared injection permit of 24 (pro-rata curtailment). Welfare units are uncalibrated.\n")

    # ------------------------------------------------------------- Exp 9
    P("## Experiment 9: is the permit result a planner artifact?\n")
    P("| planner | n | nash | coop | coop - nash |")
    P("|---|---|---|---|---|")
    rows = [(f"H={H}", ("e9", f"nash_H{H}"), ("e9", f"coop_H{H}")) for H in (2, 3, 4)] + [("H=6 (study v3)", STRONG_NASH, STRONG_COOP)]
    for lab, nk, ck in rows:
        g = sorted(set(R.get(nk, {})) & set(R.get(ck, {})))
        if not g:
            P(f"| {lab} | 0 | | | |")
            continue
        n_ = np.array([R[nk][i]["welfare"] for i in g]); c_ = np.array([R[ck][i]["welfare"] for i in g])
        P(f"| {lab} | {len(g)} | {fmt(boot(n_))} | {fmt(boot(c_))} | {fmt(boot(c_ - n_))} |")
    P("")
    for H in (2, 3, 4):
        P(line(f"nash H={H} minus nash H=6", R, ("e9", f"nash_H{H}"), STRONG_NASH))
    P(line("nash with 6 best-response rounds minus 3 rounds (H=6)", R, ("e9", "nash_H6_br6"), STRONG_NASH))
    P("")
    P("**Open-loop deviation epsilon** (greedy search, a LOWER bound on the true epsilon):\n")
    en = [r["eps"]["max"] for r in R.get(("e9", "eps_nash_H6"), {}).values()]
    ec = [r["eps"]["sum"] for r in R.get(("e9", "eps_coop_H6"), {}).values()]
    gap, gg = diff(R, STRONG_COOP, STRONG_NASH)
    if en:
        P(f"- Nash path, best unilateral schedule change: {fmt(boot(en))} (n={len(en)}; >1 unit in {np.mean(np.array(en) > 1) * 100:.0f}% of geologies)")
    if ec:
        P(f"- Coop path, welfare gained by changing a single action: {fmt(boot(ec))} (n={len(ec)})")
    if gg:
        P(f"- Measured permit gap (coop - nash, H=6): {fmt(boot(gap))} (n={len(gg)})")
        if en:
            P(f"- Ratio: gap / Nash epsilon = {np.mean(gap) / max(np.mean(en), 1e-9):.0f}x. "
              f"Reading: the gap is real if it is many times epsilon and the horizon sweep leaves it unchanged.")
    P("")

    # ------------------------------------------------------------- Exp 10
    P("## Experiment 10: the permit setting under uncertainty (repaired belief b4, planner H=3)\n")
    P("| condition | n | welfare | total injected | early-stage mean rate |")
    P("|---|---|---|---|---|")
    for c in ("nash_prior", "nash_learn", "coop_learn", "m2_learn", "causer_learn"):
        k = ("e10", c)
        rs = list(R.get(k, {}).values())
        if not rs:
            P(f"| {c} | 0 | | | |")
            continue
        P(f"| {c} | {len(rs)} | {fmt(boot([r['welfare'] for r in rs]))} | {np.mean([r['total_injected'] for r in rs]):.1f} | "
          f"{np.mean([early_rate(r) for r in rs]):.2f} |")
    for lab, k in (("oracle nash H=3", ("e9", "nash_H3")), ("oracle coop H=3", ("e9", "coop_H3"))):
        rs = list(R.get(k, {}).values())
        if rs:
            P(f"| {lab} | {len(rs)} | {fmt(boot([r['welfare'] for r in rs]))} | {np.mean([r['total_injected'] for r in rs]):.1f} | "
              f"{np.mean([early_rate(r) for r in rs]):.2f} |")
    P("")
    P(line("value of monitoring (learn - prior)", R, ("e10", "nash_learn"), ("e10", "nash_prior")))
    P(line("value of perfect information (oracle nash H=3 - learn)", R, ("e9", "nash_H3"), ("e10", "nash_learn")))
    P(line("coop - nash UNDER LEARNING (does the commons gap survive uncertainty?)", R, ("e10", "coop_learn"), ("e10", "nash_learn")))
    P(line("coop - nash with perfect information, H=3 (same geologies)", R, ("e9", "coop_H3"), ("e9", "nash_H3")))
    P(line("m2 gain under learning", R, ("e10", "m2_learn"), ("e10", "nash_learn")))
    P(line("causer-pays gain under learning", R, ("e10", "causer_learn"), ("e10", "nash_learn")))
    x, g = diff(R, ("e10", "coop_learn"), ("e10", "nash_learn"))
    if g:
        P("\nCoop - nash under learning by permeability stratum:\n")
        P("| stratum | n | effect |")
        P("|---|---|---|")
        for s in ("low-perm (<=0.7)", "mid-perm", "high-perm (>=1.4)"):
            sel = [v for v, i in zip(x, g) if stratum(R[("e10", "nash_learn")][i]["mult"]) == s]
            P(f"| {s} | {len(sel)} | {fmt(boot(sel))} |")
    P("")

    # ------------------------------------------------------------- Exp 11
    P("## Experiment 11: policies that fit a capacity commons\n")
    P("Perfect information, H=6, over the base Nash path of the same setting. 'share closed' = gain / (coop - nash).\n")
    P("| policy | n | gain over base nash | share of gap closed |")
    P("|---|---|---|---|")
    gd = {i: v for i, v in zip(gg, gap)}
    for lab, k in [(f"injection price {p:g}", ("e11", f"price{p:g}_oracle")) for p in (2.0, 5.0, 10.0, 20.0)] + \
                  [("equal fixed allocation", ("e11", "alloc_oracle"))]:
        x, g = diff(R, k, STRONG_NASH)
        gp = [gd[i] for i in g if i in gd]
        P(f"| {lab} | {len(g)} | {fmt(boot(x))} | {share(x, gp):.0f}% |" if len(g) else f"| {lab} | 0 | | |")
    P("\nReference: liability multiplier (study v3/v4, same setting): m=1.5 +5.9, m=2 +6.2, m=3 +7.5 (12-20% of the gap).\n")
    P(line("equal allocation, then cooperative planner, minus free-pool cooperative optimum (cost of rigid shares)", R, ("e11", "alloc_coop_oracle"), STRONG_COOP))
    P("")
    P("Under learning (planner H=3), gain over Nash-learn:\n")
    P("| policy | n | gain |")
    P("|---|---|---|")
    for lab, k in [(f"injection price {p:g}", ("e11", f"price{p:g}_learn")) for p in (5.0, 10.0)] + [("equal fixed allocation", ("e11", "alloc_learn"))]:
        x, g = diff(R, k, ("e10", "nash_learn"))
        P(f"| {lab} | {len(g)} | {fmt(boot(x))} |")
    P("")
    P("## Caveats\n- Planner is receding-horizon; Exp 9 is what says how much that matters here.\n"
      "- Learning arms use H=3 (cost); perfect-information comparators for them are the H=3 runs of Exp 9.\n"
      "- An injection price is treated as a pure transfer: operators pay it, welfare ignores it.\n"
      "- Welfare units are uncalibrated; signs and ranks are the findings. 12x12 grid, closed domain.\n")
    txt = "\n".join(out)
    open(os.path.join(d, "analysis5.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="seq3_results")
    main(ap.parse_args().dir)
