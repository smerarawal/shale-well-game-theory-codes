"""
analyze.py -- turns seq_results/*.json into the study's answers (analysis.md + CSV + PNG).

Research questions (each answered with paired differences over geologies and 95% bootstrap CIs):
  Q1 Externality  : how much welfare does non-cooperation lose?   coop - nash, same information
  Q2 Information  : what is monitoring worth?                    learn - prior ; oracle - learn ;
                    share of the perfect-information value that monitoring captures
  Q3 Regulation   : can a liability multiplier close the gap?     nash(m=2) - nash(m=1) vs coop - nash
  Q4 Planning     : what is strategic play worth vs naive rules?  nash/learn - bau, - static
  Q5 Scale        : does the cost of non-cooperation grow with 3 operators vs 2?
  Q6 Physical     : stored volume, peak pressure, cap violation, by solution concept
  Q7 Belief       : does the posterior find the true permeability level?
"""
import argparse, glob, json, os
import numpy as np

rng = np.random.default_rng(0)


def load(d):
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "n*_g*.json")))]
    return rows


def key(r):
    return (r["nops"], r["mode"], r["info"], r["liab_mult"])


def boot(x, B=4000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else np.nan, np.nan, np.nan
    m = rng.choice(x, (B, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt(t, d=1):
    m, lo, hi = t
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]" if not np.isnan(lo) else f"{m:.{d}f}"


def stratum(mult):
    return "low-perm (<=0.7)" if mult <= 0.7 else ("high-perm (>=1.4)" if mult >= 1.4 else "mid-perm")


def main(d, out):
    rows = load(d)
    W = {}   # (key) -> {geo: row}
    for r in rows:
        W.setdefault(key(r), {})[r["geo"]] = r
    L = []
    P = L.append

    def paired(k1, k2, f=lambda r: r["welfare"]):
        a, b = W.get(k1, {}), W.get(k2, {})
        g = sorted(set(a) & set(b))
        return g, np.array([f(a[i]) - f(b[i]) for i in g])

    def report(title, k1, k2, note=""):
        g, x = paired(k1, k2)
        if len(g) == 0:
            P(f"- {title}: no paired data yet")
            return None
        P(f"- **{title}**: {fmt(boot(x))}  (n={len(g)} geologies; {np.mean(x > 1e-6)*100:.0f}% positive, {np.mean(x < -1e-6)*100:.0f}% negative) {note}")
        return g, x

    n = {k: len(v) for k, v in W.items()}
    P("# Sequential Bayesian shared-reservoir game: results\n")
    P(f"Episodes loaded: {len(rows)}. Welfare units are the solver's uncalibrated units (a relative scale, not dollars).\n")
    P("## Completed cells (geologies per condition)\n")
    P("| operators | solution | information | liab. mult | n |\n|---|---|---|---|---|")
    for k in sorted(n):
        P(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]:g} | {n[k]} |")

    P("\n## Mean welfare per condition (95% bootstrap CI over geologies)\n")
    P("| operators | solution | information | m | welfare | peak pressure | injected | cap violation |\n|---|---|---|---|---|---|---|---|")
    for k in sorted(W):
        v = list(W[k].values())
        P(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]:g} | {fmt(boot([r['welfare'] for r in v]))} | "
          f"{np.mean([r['pmax_peak'] for r in v]):.2f} | {np.mean([r['total_injected'] for r in v]):.1f} | "
          f"{np.mean([r['cap_violation'] for r in v]):.2f} |")

    P("\n## Q1 Externality: cost of non-cooperation (coop - nash), 2 operators\n")
    for inf in ("prior", "learn", "oracle"):
        report(f"info={inf}", (2, "coop", inf, 1.0), (2, "nash", inf, 1.0))
    P("\nA coop-minus-nash value that is negative on some geologies is an approximation artefact (both are "
      "receding-horizon heuristics), not a violation of theory; the negative share is reported as a diagnostic.\n")

    P("## Q2 Value of information, 2 operators\n")
    for mo in ("nash", "coop"):
        report(f"{mo}: value of monitoring (learn - prior)", (2, mo, "learn", 1.0), (2, mo, "prior", 1.0))
        report(f"{mo}: value of perfect information (oracle - learn)", (2, mo, "oracle", 1.0), (2, mo, "learn", 1.0))
        g1, x1 = paired((2, mo, "learn", 1.0), (2, mo, "prior", 1.0)) if (2, mo, "learn", 1.0) in W and (2, mo, "prior", 1.0) in W else ([], np.array([]))
        g2, x2 = paired((2, mo, "oracle", 1.0), (2, mo, "prior", 1.0)) if (2, mo, "oracle", 1.0) in W and (2, mo, "prior", 1.0) in W else ([], np.array([]))
        c = sorted(set(g1) & set(g2))
        if c:
            a = np.mean([x1[g1.index(i)] for i in c]); b = np.mean([x2[g2.index(i)] for i in c])
            P(f"  - {mo}: monitoring captures {100*a/b:.0f}% of the oracle-minus-prior value" if abs(b) > 1e-6 else "")
    P("\nBy permeability stratum (nash, learn - prior and oracle - learn):\n")
    P("| stratum | n | learn - prior | oracle - learn |\n|---|---|---|---|")
    for s in ("low-perm (<=0.7)", "mid-perm", "high-perm (>=1.4)"):
        gs = [g for g, r in W.get((2, "nash", "prior", 1.0), {}).items() if stratum(r["mult"]) == s]
        def sub(k1, k2):
            g, x = paired(k1, k2)
            return fmt(boot([xx for gg, xx in zip(g, x) if gg in gs]))
        P(f"| {s} | {len(gs)} | {sub((2,'nash','learn',1.0),(2,'nash','prior',1.0))} | {sub((2,'nash','oracle',1.0),(2,'nash','learn',1.0))} |")

    P("\n## Q3 Regulation: liability multiplier m=2 for Nash players (2 operators, learning)\n")
    r1 = report("nash m=2 minus nash m=1", (2, "nash", "learn", 2.0), (2, "nash", "learn", 1.0))
    r2 = report("coop minus nash m=1 (the gap to close)", (2, "coop", "learn", 1.0), (2, "nash", "learn", 1.0))
    r3 = report("coop minus nash m=2 (remaining gap)", (2, "coop", "learn", 1.0), (2, "nash", "learn", 2.0))

    P("\n## Q4 Value of strategic planning vs naive rules (2 operators)\n")
    for base in ("bau", "static"):
        report(f"nash/learn minus {base}", (2, "nash", "learn", 1.0), (2, base, "prior", 1.0))
        report(f"coop/learn minus {base}", (2, "coop", "learn", 1.0), (2, base, "prior", 1.0))
    for base in ("bau", "static"):
        v = [r["welfare"] for r in W.get((2, base, "prior", 1.0), {}).values()]
        if v:
            P(f"- {base}: negative welfare in {100*np.mean(np.array(v) < 0):.0f}% of geologies (mean {np.mean(v):.0f})")

    P("\n## Q5 Scale: 2 vs 3 operators (common geologies)\n")
    for nops in (2, 3):
        report(f"{nops} ops: coop - nash (learn)", (nops, "coop", "learn", 1.0), (nops, "nash", "learn", 1.0))
    ga, xa = paired((2, "coop", "learn", 1.0), (2, "nash", "learn", 1.0)) if (2, "coop", "learn", 1.0) in W else ([], [])
    gb, xb = paired((3, "coop", "learn", 1.0), (3, "nash", "learn", 1.0)) if (3, "coop", "learn", 1.0) in W else ([], [])
    c = sorted(set(ga) & set(gb))
    if c:
        d3 = np.array([xb[gb.index(i)] - xa[ga.index(i)] for i in c])
        P(f"- gap(3 ops) - gap(2 ops) on the same {len(c)} geologies: {fmt(boot(d3))}")

    P("\n## Q7 Belief learning (learn mode)\n")
    for mo in ("nash", "coop"):
        v = list(W.get((2, mo, "learn", 1.0), {}).values())
        if v:
            T = len(v[0]["log"])
            P(f"- {mo}: |posterior-mean multiplier - true| by stage: " +
              ", ".join(f"t{t}: {np.mean([abs(r['log'][t]['mult_hat']-r['mult']) for r in v]):.2f}" for t in range(T)) +
              f"; ESS final {np.mean([r['log'][-1]['ess'] for r in v]):.1f} of 10")
    v = list(W.get((2, "nash", "prior", 1.0), {}).values())
    if v:
        P(f"- no-learning baseline error: {np.mean([abs(r['log'][-1]['mult_hat']-r['mult']) for r in v]):.2f}")

    P("\n## Fairness of the split (2 ops, nash/learn): |payoff gap| between operators\n")
    v = list(W.get((2, "nash", "learn", 1.0), {}).values())
    c = list(W.get((2, "coop", "learn", 1.0), {}).values())
    if v and c:
        P(f"- nash {np.mean([r['gini_gap'] for r in v]):.1f}  vs coop {np.mean([r['gini_gap'] for r in c]):.1f} "
          "(coop maximises the sum, not equity; a core/Shapley split would reallocate)")
    open(os.path.join(out, "analysis.md"), "w").write("\n".join(L))
    print("\n".join(L))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ks = [k for k in sorted(W) if k[0] == 2]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ms = [boot([r["welfare"] for r in W[k].values()]) for k in ks]
        ax.bar(range(len(ks)), [m[0] for m in ms], yerr=[[m[0]-m[1] for m in ms], [m[2]-m[0] for m in ms]], color="#4C78A8")
        ax.set_xticks(range(len(ks))); ax.set_xticklabels([f"{k[1]}\n{k[2]}\nm{k[3]:g}" for k in ks], fontsize=7)
        ax.set_ylabel("welfare (solver units)"); ax.set_title("Mean welfare, 2 operators (95% CI over geologies)")
        plt.tight_layout(); plt.savefig(os.path.join(out, "welfare.png"), dpi=140)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="seq_results"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); main(a.dir, a.out or a.dir)
