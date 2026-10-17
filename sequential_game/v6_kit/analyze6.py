"""
analyze6.py -- answers for study v6 (Experiments 12-17). Paired over geologies, 95% bootstrap CIs,
welfare in uncalibrated solver units. Everything is perfect-information, planner horizon 3 (see run_study6.py).

    python analyze6.py --dir seq3_results
"""
import argparse, glob, itertools, json, os
from math import factorial
import numpy as np

from analyze4 import boot, fmt, share

CAPS = (12.0, 16.0, 20.0, 24.0, 32.0, 48.0)
RULE_CAPS = (16.0, 24.0)
VARIANTS = ("lam10", "lam90", "pcap4", "pcap8", "cw9", "cw36")
HET = ("het24_16", "het28_12")
PARTS = ("123", "12_3", "13_2", "23_1", "1_2_3")


def load(d):
    R = {}
    for f in sorted(glob.glob(os.path.join(d, "e*__*.json"))):
        r = json.load(open(f))
        R.setdefault((r["exp"], r["tag"]), {})[r["geo"]] = r
    return R


def common(R, *keys):
    g = set.intersection(*[set(R.get(k, {})) for k in keys]) if keys else set()
    return sorted(g)


def vec(R, key, field, geos):
    return np.array([R[key][g][field] if field != "wel" else R[key][g]["welfare"] for g in geos], float)


def pair_diff(R, a, b, field="welfare"):
    g = common(R, a, b)
    return np.array([R[a][i][field] - R[b][i][field] for i in g], float), g


def mean_of(R, key, geos, f):
    return float(np.mean([f(R[key][g]) for g in geos])) if geos else float("nan")


def n_drilled(r):
    return r["n_wells"]


def early_rate(r, stages=3):
    return float(np.mean([np.mean(e["rates"]) for e in r["log"][:stages]]))


def gain_row(R, label, key, base, coop, d=1):
    g = common(R, key, base, coop)
    if not g:
        return f"| {label} | 0 | | | |"
    gain = np.array([R[key][i]["welfare"] - R[base][i]["welfare"] for i in g])
    gap = np.array([R[coop][i]["welfare"] - R[base][i]["welfare"] for i in g])
    return (f"| {label} | {len(g)} | {fmt(boot(gain), d)} | {share(gain, gap):.0f}% | "
            f"{mean_of(R, key, g, n_drilled):.1f} / {mean_of(R, base, g, n_drilled):.1f} |")


def shapley(v, n=3):
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for k in range(n):
            for S in itertools.combinations(others, k):
                w = factorial(k) * factorial(n - k - 1) / factorial(n)
                phi[i] += w * (v[tuple(sorted(S + (i,)))] - (v[tuple(sorted(S))] if S else 0.0))
    return phi


def least_core(v, n=3):
    """min eps s.t. sum x = v(N), x(S) + eps >= v(S) for every proper coalition S. eps <= 0: core non-empty."""
    from scipy.optimize import linprog
    N = tuple(range(n))
    A, b = [], []
    for k in range(1, n):
        for S in itertools.combinations(range(n), k):
            A.append([-1.0 if i in S else 0.0 for i in range(n)] + [-1.0]); b.append(-v[S])
    res = linprog([0.0] * n + [1.0], A_ub=A, b_ub=b, A_eq=[[1.0] * n + [0.0]], b_eq=[v[N]],
                  bounds=[(None, None)] * (n + 1))
    return float(res.x[-1]), res.x[:n]


def main(d):
    R = load(d)
    out = []
    P = out.append
    P("# Study v6: scarcity, rationing rules, sensitivity, heterogeneity, coalitions, certification (Experiments 12-17)\n")
    P("Setting unless stated: close wells, shared injection permit, planner horizon 3 (no extrapolation), perfect information. "
      "Welfare units are uncalibrated. Gaps at H=3 are LOWER than at the full horizon (study v5: 14.8 / 24.5 / 29.8 / 37.1 at H=2/3/4/6), "
      "so read ranks and shares, not absolute sizes.\n")

    # ---------------------------------------------------------------- Exp 12
    P("## Experiment 12: how does the commons gap depend on scarcity?\n")
    P("| permit cap | n | nash | coop | coop - nash | gap / nash | wells nash / coop | injected nash | peak p nash / coop |")
    P("|---|---|---|---|---|---|---|---|---|")
    gaps = {}
    for cap in CAPS:
        kn, kc = ("e12", f"cap{cap:g}_nash"), ("e12", f"cap{cap:g}_coop")
        g = common(R, kn, kc)
        if not g:
            P(f"| {cap:g} | 0 | | | | | | | |")
            continue
        n_ = vec(R, kn, "welfare", g); c_ = vec(R, kc, "welfare", g)
        gaps[cap] = float(np.mean(c_ - n_))
        P(f"| {cap:g} | {len(g)} | {fmt(boot(n_))} | {fmt(boot(c_))} | {fmt(boot(c_ - n_))} | {100 * np.mean(c_ - n_) / np.mean(n_):.1f}% | "
          f"{mean_of(R, kn, g, n_drilled):.1f} / {mean_of(R, kc, g, n_drilled):.1f} | {mean_of(R, kn, g, lambda r: r['total_injected']):.1f} | "
          f"{mean_of(R, kn, g, lambda r: r['pmax_peak']):.2f} / {mean_of(R, kc, g, lambda r: r['pmax_peak']):.2f} |")
    kn, kc = ("e6", "close_inf_H3_nash"), ("e6", "close_inf_H3_coop")
    g = common(R, kn, kc)
    if g:
        n_ = vec(R, kn, "welfare", g); c_ = vec(R, kc, "welfare", g)
        P(f"| no permit (study v4, same geometry) | {len(g)} | {fmt(boot(n_))} | {fmt(boot(c_))} | {fmt(boot(c_ - n_))} | | | | |")
    P("")
    if len(gaps) >= 3:
        cap_star = max(gaps, key=gaps.get)
        P(f"Reading: the gap is largest at cap {cap_star:g} ({gaps[cap_star]:.1f}). A gap that rises then falls with the cap means the "
          f"commons problem needs a binding-but-not-hopeless permit; a gap that keeps rising as the cap loosens means the "
          f"pressure externality, not the permit, is doing the work.\n")

    # ---------------------------------------------------------------- Exp 13
    P("## Experiment 13: the permit as a claims problem (rationing rules)\n")
    P("Rules when claims (requested volumes = rate x wells) exceed the remaining permit: proportional (current: inflating a claim, e.g. by drilling, "
      "raises the share), constrained equal awards cea (a claim above the equal share earns nothing extra), constrained equal losses cel, "
      "fixed equal allocation alloc (no pooling at all). 'share closed' = gain over proportional Nash / (coop - proportional Nash).\n")
    for cap in RULE_CAPS:
        base, coop = ("e12", f"cap{cap:g}_nash"), ("e12", f"cap{cap:g}_coop")
        P(f"**Permit {cap:g}**\n")
        P("| rule (Nash play) | n | gain over proportional | share of gap closed | wells drilled: rule / proportional |")
        P("|---|---|---|---|---|")
        for lab, key in (("cea (equal awards)", ("e13", f"cap{cap:g}_cea")), ("cel (equal losses)", ("e13", f"cap{cap:g}_cel")),
                         ("fixed equal allocation", ("e13", f"cap{cap:g}_alloc"))):
            P(gain_row(R, lab, key, base, coop))
        g = common(R, base, coop)
        if g:
            P(f"| (coop, for scale) | {len(g)} | {fmt(boot(vec(R, coop, 'welfare', g) - vec(R, base, 'welfare', g)))} | 100% | "
              f"{mean_of(R, coop, g, n_drilled):.1f} / {mean_of(R, base, g, n_drilled):.1f} |")
        P("")
    P("Reading: if cea closes most of the gap, the gap is a CLAIMS-INFLATION race (extra wells and rates bought to enlarge a share) that a "
      "better pooled rule removes at no rigidity cost. If only the fixed allocation closes it, the race is about TIMING of use of a shared pool. "
      "Compare 'wells drilled' between Nash and coop: fewer wells under coop is the signature of rent-seeking drilling.\n")

    # ---------------------------------------------------------------- Exp 14
    P("## Experiment 14: do the conclusions survive other payoff scales? (one parameter changed at a time)\n")
    P("| variant | n | nash | coop - nash | alloc gain (share) | cea gain (share) |")
    P("|---|---|---|---|---|---|")
    rows = [("base (lam 30, cap 6, well cost 18)", ("e12", "cap24_nash"), ("e12", "cap24_coop"), ("e13", "cap24_alloc"), ("e13", "cap24_cea"))]
    for v in VARIANTS:
        rows.append((v, ("e14", f"{v}_nash"), ("e14", f"{v}_coop"), ("e14", f"{v}_alloc"), ("e14", f"{v}_cea")))
    for lab, kn, kc, ka, ke in rows:
        g = common(R, kn, kc, ka, ke)
        if not g:
            P(f"| {lab} | 0 | | | | |")
            continue
        n_, c_ = vec(R, kn, "welfare", g), vec(R, kc, "welfare", g)
        ga, ge = vec(R, ka, "welfare", g) - n_, vec(R, ke, "welfare", g) - n_
        P(f"| {lab} | {len(g)} | {fmt(boot(n_))} | {fmt(boot(c_ - n_))} | {fmt(boot(ga))} ({share(ga, c_ - n_):.0f}%) | {fmt(boot(ge))} ({share(ge, c_ - n_):.0f}%) |")
    P("\nReading: a conclusion is robust if the sign and the ordering of the instruments are the same across rows. "
      "A row where the gap vanishes shows the regime in which the commons problem is absent.\n")

    # ---------------------------------------------------------------- Exp 15
    P("## Experiment 15: price vs quantity when operators differ\n")
    P("Margins (a, b) with mean 20; same permit 24. alloc = equal fixed shares (a quantity instrument), price = per-unit injection charge (a price "
      "instrument; a pure transfer, welfare ignores it), cea = pooled equal awards. 'share closed' relative to coop - nash of the same row.\n")
    P("| operators | instrument | n | gain over nash | share of gap closed | payoff op0 / op1 |")
    P("|---|---|---|---|---|---|")
    arms = [("symmetric (20,20)", ("e12", "cap24_nash"), ("e12", "cap24_coop"),
             [("alloc", ("e13", "cap24_alloc")), ("cea", ("e13", "cap24_cea"))])]
    for h in HET:
        mg = h.replace("het", "").replace("_", ",")
        arms.append((f"({mg})", ("e15", f"{h}_nash"), ("e15", f"{h}_coop"),
                     [("alloc", ("e15", f"{h}_alloc")), ("cea", ("e15", f"{h}_cea")),
                      ("price 5", ("e15", f"{h}_price5")), ("price 10", ("e15", f"{h}_price10"))]))
    for lab, kn, kc, pol in arms:
        g = common(R, kn, kc)
        if g:
            pn = np.mean([R[kn][i]["payoff"][:2] for i in g], axis=0)
            P(f"| {lab} | nash (no instrument) | {len(g)} | 0 | 0% | {pn[0]:.0f} / {pn[1]:.0f} |")
            pc = np.mean([R[kc][i]["payoff"][:2] for i in g], axis=0)
            P(f"| {lab} | coop | {len(g)} | {fmt(boot(vec(R, kc, 'welfare', g) - vec(R, kn, 'welfare', g)))} | 100% | {pc[0]:.0f} / {pc[1]:.0f} |")
        for plab, pk in pol:
            gg = common(R, pk, kn, kc)
            if not gg:
                P(f"| {lab} | {plab} | 0 | | | |")
                continue
            gain = vec(R, pk, "welfare", gg) - vec(R, kn, "welfare", gg)
            gap = vec(R, kc, "welfare", gg) - vec(R, kn, "welfare", gg)
            pp = np.mean([R[pk][i]["payoff"][:2] for i in gg], axis=0)
            P(f"| {lab} | {plab} | {len(gg)} | {fmt(boot(gain))} | {share(gain, gap):.0f}% | {pp[0]:.0f} / {pp[1]:.0f} |")
    P("\nReading (Weitzman): a quantity instrument fixes the allocation without knowing who is more productive, so it should lose more as operators "
      "differ; a price lets the productive operator keep injecting. Compare the alloc row across the three operator pairs.\n")

    # ---------------------------------------------------------------- Exp 16
    P("## Experiment 16: three operators -- coalition values, Shapley value, core\n")
    P("Partition-function game with the gamma-characteristic function: v(S) = total payoff of S when S cooperates as one block and the others play "
      "Nash individually; v({i}) = payoff of i when everybody plays Nash. v(N) = best welfare found over the grand-coalition planner and all "
      "partitions (a coalition can replicate any partition outcome, and the receding-horizon planner is not exactly optimal).\n")
    geos = common(R, *[("e16", f"part_{p}") for p in PARTS])
    if geos:
        V, PH, ext, planner_err, vN_l, sing, vpair = [], [], [], 0, [], [], []
        eps_l, nonempty = [], 0
        for g in geos:
            pay = {p: np.array(R[("e16", f"part_{p}")][g]["payoff"]) for p in PARTS}
            wel = {p: pay[p].sum() for p in PARTS}
            v = {(0,): pay["1_2_3"][0], (1,): pay["1_2_3"][1], (2,): pay["1_2_3"][2],
                 (0, 1): pay["12_3"][0] + pay["12_3"][1], (0, 2): pay["13_2"][0] + pay["13_2"][1], (1, 2): pay["23_1"][0] + pay["23_1"][1]}
            vN = max(wel.values())
            planner_err += int(wel["123"] < vN - 1e-9)
            v[(0, 1, 2)] = vN
            phi = shapley(v)
            e, x = least_core(v)
            eps_l.append(e); nonempty += int(e <= 1e-9)
            PH.append(phi); sing.append(pay["1_2_3"]); vN_l.append(vN); vpair.append([v[(0, 1)], v[(0, 2)], v[(1, 2)]])
            ext.append([pay["12_3"][2] - pay["1_2_3"][2], pay["13_2"][1] - pay["1_2_3"][1], pay["23_1"][0] - pay["1_2_3"][0]])
        PH, sing, ext = np.array(PH), np.array(sing), np.array(ext)
        P(f"n = {len(geos)} geologies.\n")
        P("| quantity | value |")
        P("|---|---|")
        P(f"| v({{i}}) = Nash payoff, op0 / op1 / op2 | {sing.mean(0)[0]:.1f} / {sing.mean(0)[1]:.1f} / {sing.mean(0)[2]:.1f} |")
        P(f"| v({{i,j}}) for 01 / 02 / 12 | {np.mean(vpair, 0)[0]:.1f} / {np.mean(vpair, 0)[1]:.1f} / {np.mean(vpair, 0)[2]:.1f} |")
        P(f"| v(N) (best found) | {fmt(boot(vN_l))} |")
        P(f"| gain from full cooperation v(N) - sum of Nash payoffs | {fmt(boot(np.array(vN_l) - sing.sum(1)))} |")
        P(f"| Shapley value op0 / op1 / op2 | {PH.mean(0)[0]:.1f} / {PH.mean(0)[1]:.1f} / {PH.mean(0)[2]:.1f} |")
        P(f"| Shapley minus Nash payoff (who gains from cooperating) | {(PH - sing).mean(0)[0]:.1f} / {(PH - sing).mean(0)[1]:.1f} / {(PH - sing).mean(0)[2]:.1f} |")
        P(f"| least-core epsilon (<= 0: core non-empty) | {fmt(boot(eps_l), 2)}; core non-empty in {nonempty}/{len(geos)} geologies |")
        P(f"| externality: outsider's payoff change when the OTHER two cooperate (op2 / op1 / op0 outside) | {fmt(boot(ext.ravel()))} (mean over the three cases; > 0 = others' cooperation helps the outsider) |")
        P(f"| grand-coalition planner below the best partition outcome | {planner_err}/{len(geos)} geologies (planner is a receding-horizon heuristic) |\n")
        P("Reading: a non-empty core means a stable split of the cooperative surplus exists. If outsiders GAIN when the others cooperate (positive externality, as the "
          "mean above shows when > 0), staying out is tempting (free riding) and small coalitions are fragile; if they lose, joining is defensive.\n")
    else:
        P("no complete geologies yet (needs all five partition runs).\n")

    # ---------------------------------------------------------------- Exp 17
    P("## Experiment 17: how big is epsilon really? (stronger deviation search)\n")
    P("Both searches are lower bounds on the true epsilon. 'plain' = single-stage changes (studies v3-v5). 'strong' = also pair moves "
      "(two stages changed at once) and more sweeps, a strictly larger search space on the same path.\n")
    kn, kc = ("e17", "eps_nash_H3"), ("e17", "eps_coop_H3")
    gn, gc = sorted(R.get(kn, {})), sorted(R.get(kc, {}))
    if gn:
        pl = np.array([R[kn][g]["eps_plain"]["max"] for g in gn]); st = np.array([R[kn][g]["eps"]["max"] for g in gn])
        P(f"- Nash path: plain {fmt(boot(pl))}, strong {fmt(boot(st))} (n={len(gn)}; strong > 1 in {np.mean(st > 1) * 100:.0f}% of geologies; "
          f"mean simulations {np.mean([R[kn][g]['eps']['sims'] for g in gn]):.0f})")
    if gc:
        pl = np.array([R[kc][g]["eps_plain"]["sum"] for g in gc]); st = np.array([R[kc][g]["eps"]["sum"] for g in gc])
        P(f"- Coop path (welfare gain from a better joint schedule): plain {fmt(boot(pl))}, strong {fmt(boot(st))} (n={len(gc)})")
    g = common(R, ("e12", "cap24_nash"), ("e12", "cap24_coop"), kn) if gn else []
    if g:
        gap = vec(R, ("e12", "cap24_coop"), "welfare", g) - vec(R, ("e12", "cap24_nash"), "welfare", g)
        st = np.array([R[kn][i]["eps"]["max"] for i in g])
        P(f"- Measured gap on the same geologies: {fmt(boot(gap))}; gap / strong Nash epsilon = {('%.1fx' % (np.mean(gap) / np.mean(st))) if np.mean(st) > 1e-6 else 'epsilon found = 0 (no profitable deviation found)'}")
    P("\nReading: the gap is credible if it stays several times the strong epsilon. If the strong search finds much more than the plain one, "
      "the earlier epsilons were too optimistic and the reported gaps need a wider margin.\n")

    P("## Caveats\n- Perfect information and planner horizon 3: gaps are lower than at the full horizon (study v5), ranks and shares are the finding.\n"
      "- Exp 14 changes one parameter at a time; interactions are not tested.\n"
      "- Exp 16 uses the gamma-characteristic function and the best welfare found for v(N); other characteristic functions give other cores.\n"
      "- Exp 17 searches are lower bounds on epsilon, so a small epsilon is evidence, not proof, of an equilibrium.\n"
      "- n is 12 geologies per arm (8 for Exp 14); welfare units are uncalibrated; 12x12 grid, closed domain.\n")
    txt = "\n".join(out)
    open(os.path.join(d, "analysis6.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="seq3_results")
    main(ap.parse_args().dir)
