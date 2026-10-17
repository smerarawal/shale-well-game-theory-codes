"""
eps_check.py -- how far is the planner's play from an equilibrium?  (Experiment 1)

Two measures, both under perfect information (deterministic dynamics, real solver):

1. open_loop_eps  -- take the played path. Hold the OTHER operators' scheduled actions fixed and let
   one operator search over its own whole 6-stage schedule (coordinate ascent: change one stage's
   action at a time, keep any improvement, up to 2 sweeps). eps_o = best payoff found - played payoff.
   For the cooperative path the same search maximises the SUM over any operator's actions, so the
   gain is the planner's sub-optimality in welfare. The search is greedy, so every number is a
   LOWER BOUND on the true epsilon (a better deviation may exist).

2. osd_eps -- one-shot-deviation check at stage t0 with REACTIVE continuation. At stage t0 operator o
   plays any of its options while the others keep the planner's t0 action; from t0+1 everybody
   (including o) re-plans with the same planner. eps_o(t0) = max_a Q_o(a) - Q_o(planner action).
   In a finite game, eps = 0 for all states on the path means the planner is subgame perfect
   against one-shot deviations at those states.

Compare eps with the effect sizes you report (e.g. cost of non-cooperation): an effect smaller than
eps is not distinguishable from planner error.
"""
import copy
import itertools

import numpy as np

from game3 import (options_for, run_episode, schedule_from_log, simulate_schedule)


def _alts(cfg, o):
    rates = cfg.rates if cfg.n_ops <= 2 else (0.0, 1.0)
    return [(r, d) for r in rates for d in (None, "dyn")]


def _cur_key(entry):
    r, s = entry
    return (r, s is not None)


def open_loop_eps(model, cfg, geo, sched, objective="own", sweeps=2):
    """objective 'own': per-operator deviation gain (others fixed). 'sum': welfare improvement
    over joint single-action changes. Returns dict with per-op gains (own) or the sum gain."""
    base = simulate_schedule(model, cfg, geo, sched)
    out = {}
    ops_sets = [[o] for o in range(cfg.n_ops)] if objective == "own" else [list(range(cfg.n_ops))]
    for ops in ops_sets:
        cur = copy.deepcopy(sched)
        val = lambda v: v[ops[0]] if objective == "own" else v.sum()
        curval = val(base)
        for _ in range(sweeps):
            improved = False
            for t in range(cfg.T):
                for o in ops:
                    bestv, besta = curval, None
                    for a in _alts(cfg, o):
                        if (a[0], a[1] is not None) == _cur_key(cur[t][o]):
                            continue
                        trial = copy.deepcopy(cur)
                        trial[t][o] = a
                        v = val(simulate_schedule(model, cfg, geo, trial))
                        if v > bestv + 1e-9:
                            bestv, besta = v, a
                    if besta is not None:
                        cur[t][o] = besta
                        curval, improved = bestv, True
            if not improved:
                break
        key = f"op{ops[0]}" if objective == "own" else "sum"
        out[key] = float(curval - val(base))
    if objective == "own":
        out["max"] = max(out.values())
    out["base_welfare"] = float(base.sum())
    return out


def osd_eps(model, cfg, geo, t0, mode="nash", seed=0, pool=None, base=None):
    """One-shot-deviation epsilon at stage t0 on the planner's own path (oracle information)."""
    if base is None:
        base = run_episode(model, cfg, geo, pool, mode, "oracle", seed=seed)
    summ, log = base
    prefix = [e["joint"] for e in log[:t0]]
    res = {}
    for o in range(cfg.n_ops):
        base_q = summ["payoff"][o]
        best_q, tried, seen = base_q, 0, set()
        planned = tuple(log[t0]["joint"][o][:1]) + (log[t0]["joint"][o][1] is not None,)
        seen.add(planned)
        for a in _alts(cfg, o):
            key = (a[0], a[1] is not None)
            if key in seen:
                continue
            seen.add(key)
            s2, l2 = run_episode(model, cfg, geo, pool, mode, "oracle", seed=seed,
                                 force={t0: {o: (a[0], a[1] is not None)}}, prefix=prefix)
            # a drill request with no free site resolves to "no drill": identical to an already seen option
            tried += 1
            best_q = max(best_q, s2["payoff"][o])
        res[f"op{o}"] = float(best_q - base_q)
    res["max"] = max(res.values())
    return res


def open_loop_eps_strong(model, cfg, geo, sched, objective="own", sweeps=3, max_sims=3000):
    """Tighter lower bound than open_loop_eps (study v6). Per round: coordinate ascent over single-stage
    changes, then PAIR moves (two schedule cells changed jointly, any stages / any operators allowed by the
    objective), repeated until a round brings no improvement, `sweeps` rounds or `max_sims` simulations
    (per operator set). Still a search, so still a LOWER bound on the true epsilon, but a strictly larger
    search space than open_loop_eps (every move open_loop_eps can make is also tried here).
    Returns the same keys as open_loop_eps plus 'sims' (simulations used)."""
    base = simulate_schedule(model, cfg, geo, sched)
    out, used = {}, 0
    ops_sets = [[o] for o in range(cfg.n_ops)] if objective == "own" else [list(range(cfg.n_ops))]
    for ops in ops_sets:
        val = (lambda v, o=ops[0]: v[o]) if objective == "own" else (lambda v: v.sum())
        cells = [(t, o) for t in range(cfg.T) for o in ops]
        cur, curval, sims = copy.deepcopy(sched), val(base), 0
        for _ in range(sweeps):
            improved = False
            for (t, o) in cells:                                   # single moves
                bestv, besta = curval, None
                for a in _alts(cfg, o):
                    if (a[0], a[1] is not None) == _cur_key(cur[t][o]):
                        continue
                    trial = copy.deepcopy(cur)
                    trial[t][o] = a
                    v = val(simulate_schedule(model, cfg, geo, trial))
                    sims += 1
                    if v > bestv + 1e-9:
                        bestv, besta = v, a
                if besta is not None:
                    cur[t][o], curval, improved = besta, bestv, True
            for (t1, o1), (t2, o2) in itertools.combinations(cells, 2):   # pair moves
                if sims >= max_sims:
                    break
                bestv, best = curval, None
                for a1 in _alts(cfg, o1):
                    if (a1[0], a1[1] is not None) == _cur_key(cur[t1][o1]):
                        continue
                    for a2 in _alts(cfg, o2):
                        if (a2[0], a2[1] is not None) == _cur_key(cur[t2][o2]):
                            continue
                        trial = copy.deepcopy(cur)
                        trial[t1][o1], trial[t2][o2] = a1, a2
                        v = val(simulate_schedule(model, cfg, geo, trial))
                        sims += 1
                        if v > bestv + 1e-9:
                            bestv, best = v, (a1, a2)
                if best is not None:
                    cur[t1][o1], cur[t2][o2] = best
                    curval, improved = bestv, True
            if not improved or sims >= max_sims:
                break
        key = f"op{ops[0]}" if objective == "own" else "sum"
        out[key] = float(curval - val(base))
        used += sims
    if objective == "own":
        out["max"] = max(v for k, v in out.items() if k.startswith("op"))
    out["base_welfare"] = float(base.sum())
    out["sims"] = used
    return out
