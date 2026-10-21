"""Paired summaries for the Study 7 follow-up; writes study7_analysis.md."""
import argparse
import glob
import json
import os

import numpy as np


def load(folder):
    rows = {}
    for path in glob.glob(os.path.join(folder, "e*__*.json")):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        rows.setdefault((r["exp"], r["tag"]), {})[int(r["geo"])] = r
    return rows


def ci(x, seed=37, draws=10000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return "n=0"
    rng = np.random.default_rng(seed)
    b = np.mean(x[rng.integers(0, len(x), (draws, len(x)))], axis=1)
    lo, hi = np.quantile(b, [0.025, 0.975])
    return f"{np.mean(x):.3f} [{lo:.3f}, {hi:.3f}]"


def paired(rows, a, b, field="welfare"):
    aa, bb = rows.get(a, {}), rows.get(b, {})
    ids = sorted(set(aa) & set(bb))
    return ids, np.asarray([aa[g][field] - bb[g][field] for g in ids], float)


def table_gap(rows, exp, pairs):
    out = ["| contrast (A − B) | paired n | mean [95% bootstrap CI] | positive |", "|---|---:|---:|---:|"]
    for label, a, b in pairs:
        ids, x = paired(rows, (exp, a), (exp, b))
        out.append(f"| {label} | {len(ids)} | {ci(x)} | {100*np.mean(x > 0):.0f}% |" if len(ids) else f"| {label} | 0 | — | — |")
    return out


def main(folder):
    rows = load(folder)
    lines = ["# Study 7 results", "", "Welfare is the discounted resource payoff before injection-price transfers. Intervals are geology-level paired bootstrap CIs. Incomplete arms are shown with their achieved paired n.", "",
             "## Experiment 18 — shared permit and planner gap", "",
             *table_gap(rows, "e18", [
                 ("Coop − Nash, cap 24, H=3", "cap24_coop_H3", "cap24_nash_H3"),
                 ("Coop − Nash, uncapped, H=3", "uncapped_coop_H3", "uncapped_nash_H3"),
                 ]), "",
             "| difference-in-gaps: (cap24 Coop−Nash) − (uncapped Coop−Nash) | paired n | mean [95% CI] | positive |", "|---|---:|---:|---:|"]
    arms = [rows.get(("e18", tag), {}) for tag in ("cap24_coop_H3", "cap24_nash_H3", "uncapped_coop_H3", "uncapped_nash_H3")]
    ids = sorted(set.intersection(*(set(x) for x in arms))) if arms else []
    did = [arms[0][g]["welfare"] - arms[1][g]["welfare"] - arms[2][g]["welfare"] + arms[3][g]["welfare"] for g in ids]
    lines.append(f"| gap difference | {len(ids)} | {ci(did)} | {100*np.mean(np.asarray(did)>0):.0f}% |" if ids else "| gap difference | 0 | — | — |")
    lines += ["",
             "## Experiment 19 — capacity response", "",
             "| cap | paired n | Coop − Nash mean [95% CI] | positive geologies | Nash binds permit (%) |", "|---:|---:|---:|---:|---:|"]
    for cap in (12, 16, 20, 24, 32, 48):
        a, b = rows.get(("e19", f"cap{cap}_coop"), {}), rows.get(("e19", f"cap{cap}_nash"), {})
        ids = sorted(set(a) & set(b))
        gap = np.asarray([a[g]["welfare"] - b[g]["welfare"] for g in ids])
        binds = [b[g]["total_injected"] >= cap - 1e-5 for g in ids]
        lines.append(f"| {cap} | {len(ids)} | {ci(gap) if ids else '—'} | {100*np.mean(gap>0):.0f}% | {100*np.mean(binds):.0f}% |" if ids else f"| {cap} | 0 | — | — | — |")
    lines += ["", "## Experiment 20 — rationing rules", "", "| cap/rule | paired n | welfare change vs pro-rata Nash | injected volume change |", "|---|---:|---:|---:|"]
    for cap in (16, 24):
        base = rows.get(("e20", f"cap{cap}_prorata_nash"), {})
        for rule in ("cea", "cel", "fixed"):
            other = rows.get(("e20", f"cap{cap}_{rule}_nash"), {})
            ids = sorted(set(base) & set(other))
            dw = [other[g]["welfare"] - base[g]["welfare"] for g in ids]
            dv = [other[g]["total_injected"] - base[g]["total_injected"] for g in ids]
            lines.append(f"| {cap}/{rule} | {len(ids)} | {ci(dw) if ids else '—'} | {ci(dv) if ids else '—'} |")
    lines += ["", "Read stage-level requested_vol and served_vol in JSON to see whether a rule changes requests, the distribution of curtailment, or only final totals.", "",
              "## Experiment 21 — heterogeneous margins and price", "", "| margins | arm | n | welfare mean [95% CI] | operator 0 net payoff | operator 1 net payoff |", "|---|---|---:|---:|---:|---:|"]
    for margin_tag in ("equal", "op0_high", "op1_high"):
        for arm in ("price0", "price5", "price10", "fixed"):
            rs = list(rows.get(("e21", f"{margin_tag}_{arm}"), {}).values())
            if not rs:
                lines.append(f"| {margin_tag} | {arm} | 0 | — | — | — |")
                continue
            lines.append(f"| {margin_tag} | {arm} | {len(rs)} | {ci([r['welfare'] for r in rs])} | {np.mean([r['payoff_net_price'][0] for r in rs]):.3f} | {np.mean([r['payoff_net_price'][1] for r in rs]):.3f} |")
    lines += ["", "Paired price effects relative to the same-margin price0 arm:", "", "| margins | price | paired n | resource-welfare change [95% CI] |", "|---|---:|---:|---:|"]
    for margin_tag in ("equal", "op0_high", "op1_high"):
        for price in (5, 10):
            ids, change = paired(rows, ("e21", f"{margin_tag}_price{price}"), ("e21", f"{margin_tag}_price0"))
            lines.append(f"| {margin_tag} | {price} | {len(ids)} | {ci(change)} |" if ids else f"| {margin_tag} | {price} | 0 | — |")
    lines += ["", "Price is a transfer: it changes Nash choices and net operator payoffs, but is excluded from resource welfare. Compare a price arm only to the same-margin price0 arm.", "",
              "## Experiment 22 — deviation diagnostic", "", "| search | n | maximum unilateral gain mean [95% CI] |", "|---|---:|---:|"]
    for tag in ("cap24_nash_one_stage_eps", "cap24_nash_two_stage_eps"):
        rs = list(rows.get(("e22", tag), {}).values())
        lines.append(f"| {tag} | {len(rs)} | {ci([r['eps']['max'] for r in rs]) if rs else '—'} |")
    lines += ["", "Both are lower bounds on full equilibrium epsilon. The two-stage move expands the search to coordinated adjacent action pairs by one operator, with other players' schedules held fixed; it is not a proof of equilibrium.", "",
              "## Experiment 23 — learning", "", *table_gap(rows, "e23", [("learned Coop − learned Nash", "coop_learn", "nash_learn")]), "",
              "A positive mean alone is not decisive when the paired interval spans zero. Treat Exp18 at H=3 as the primary result; H=6 is a horizon sensitivity check, not pooled with H=3.", ""]
    text = "\n".join(lines)
    path = os.path.join(folder, "study7_analysis.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="study7_results")
    main(parser.parse_args().dir)
