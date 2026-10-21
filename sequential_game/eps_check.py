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


def two_stage_block_eps(model, cfg, geo, sched, sweeps=1):
    """Greedy unilateral deviation search that changes two adjacent stages jointly.

    Other operators' complete schedules stay fixed. Each candidate block enumerates the
    Cartesian product of the configured rate levels and (no drill, next preferred site).
    This remains a lower bound on equilibrium epsilon; it is stronger than one-stage moves,
    but it does not enumerate arbitrary full-horizon schedules.
    """
    base = simulate_schedule(model, cfg, geo, sched)
    out = {"base_payoff": base.tolist(), "base_welfare": float(base.sum()),
           "move_set": "unilateral adjacent two-stage blocks", "sweeps": int(sweeps),
           "candidate_simulations": 0}
    rates = cfg.rates if cfg.n_ops <= 2 else (0.0, 1.0)
    actions = [(r, d) for r in rates for d in (None, "dyn")]
    for op in range(cfg.n_ops):
        cur = copy.deepcopy(sched)
        curv = float(base[op])
        tried = 0
        for _ in range(sweeps):
            improved = False
            for t in range(max(1, cfg.T - 1)):
                end = min(t + 2, cfg.T)
                bestv, best = curv, None
                for a0 in actions:
                    for a1 in (actions if end - t == 2 else (None,)):
                        trial = copy.deepcopy(cur)
                        trial[t][op] = a0
                        if a1 is not None:
                            trial[t + 1][op] = a1
                        vals = simulate_schedule(model, cfg, geo, trial)
                        tried += 1
                        if vals[op] > bestv + 1e-9:
                            bestv, best = float(vals[op]), trial
                if best is not None:
                    cur, curv, improved = best, bestv, True
            if not improved:
                break
        out[f"op{op}"] = max(0.0, curv - float(base[op]))
        out["candidate_simulations"] += tried
    out["max"] = max(out[f"op{o}"] for o in range(cfg.n_ops))
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
