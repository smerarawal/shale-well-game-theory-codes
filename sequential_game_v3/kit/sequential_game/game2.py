"""
game2.py -- comprehensive sequential Bayesian game (v2) on the real solver.

Model: a finite-horizon, partially observable Markov (stochastic) game.
  state    : reservoir (pressure, saturation) + each operator's wells and rate setting
  hidden   : geology theta = (random field, permeability multiplier); shared by all operators
  actions  : per operator per stage: rate level in cfg.rates (applied to all its wells)
             x {no drill, drill the next free preferred site}
  belief   : K-particle posterior over theta (public pressure monitoring at every well,
             Gaussian noise sigma_obs, likelihood tempered with sigma_lik > sigma_obs because
             the particle set is a coarse, misspecified cover of the truth)
  payoff_o : margin*rate*time per well - drilling cost - liability share of a FIELD-WIDE
             pressure penalty lam*max(0, max_cell_p - p_cap)^2, split pro rata by injected
             volume (unitisation / proportional liability). Pore space + caprock limit are
             the common pool; each operator internalises only its share of the penalty.
  solution : "nash"  = iterated best response over the stage game (receding horizon,
                       `horizon` stages of simulation + extrapolated continuation value)
             "coop"  = joint action maximising total welfare, same valuation
             "bau"   = business as usual: everybody drills ASAP at full rate, no planning
             "static"= initial wells only, full rate, no planning
  liab_mult m: regulator scales the penalty seen by Nash players (decision payoff uses
             lam*m). Welfare is ALWAYS scored with m = 1.
This is an approximate equilibrium (receding horizon, not a full Markov perfect Bayesian
equilibrium); see ANALYSIS notes in run_study.py output.
"""
import itertools
import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_gen_v3 import random_permeability, random_porosity  # noqa: E402

MULTS = (0.5, 0.7, 1.0, 1.4, 2.0)


@dataclass
class Cfg:
    n: int = 12
    T: int = 6
    n_ops: int = 2
    init_sites: tuple = ((2, 2), (9, 9), (2, 9))
    prefs: tuple = (((5, 5), (2, 5), (9, 6)), ((6, 6), (9, 2), (6, 9)), ((5, 9), (9, 5), (3, 7)))
    rates: tuple = (0.0, 0.5, 1.0)
    max_new: int = 3
    margin: float = 20.0
    c_well: float = 18.0
    p_cap: float = 6.0
    lam: float = 30.0
    gamma: float = 1 / 1.05
    sigma_obs: float = 0.3
    sigma_lik: float = 0.8
    horizon: int = 2
    extrap: bool = True
    liab_mult: float = 1.0


def make_geology(field_seed, mult, n):
    return {"k": random_permeability(n, n, field_seed) * mult, "phi": random_porosity(n, n, field_seed),
            "mult": mult, "seed": field_seed}


def particle_set(n, field_seeds=(11, 12), mults=MULTS):
    return [make_geology(s, m, n) for s in field_seeds for m in mults]


def play_stage(model, geo, state, wells, rates, new_wells, cfg):
    """wells: tuple (op,i,j); rates: per-operator rate. Returns next state and components:
    gross[o], drill[o], pen (scalar field penalty, m=1), share[o], pmax."""
    active = [(i, j, max(rates[o], 1e-6)) for o, i, j in wells]
    nxt, t = model.advance(geo, state, active)
    gross = np.zeros(cfg.n_ops)
    vol = np.zeros(cfg.n_ops)
    for o, _, _ in wells:
        gross[o] += cfg.margin * rates[o] * t
        vol[o] += rates[o] * t
    drill = np.zeros(cfg.n_ops)
    for o, _, _ in new_wells:
        drill[o] += cfg.c_well
    pmax = float(nxt["p"].max())
    pen = cfg.lam * max(0.0, pmax - cfg.p_cap) ** 2
    share = vol / vol.sum() if vol.sum() > 1e-9 else np.zeros(cfg.n_ops)
    return nxt, {"gross": gross, "drill": drill, "pen": pen, "share": share, "pmax": pmax, "vol": vol}


def payoff(c, cfg, mult=1.0):
    return c["gross"] - c["drill"] - mult * c["pen"] * c["share"]


class Belief:
    def __init__(self, geos, model, cfg):
        self.geos = geos
        self.states = [model.initial_state(cfg.n) for _ in geos]
        self.w = np.full(len(geos), 1.0 / len(geos))

    def update(self, obs, wells, cfg):
        pred = [np.array([s["p"][i, j] for _, i, j in wells]) for s in self.states]
        logl = np.array([-np.sum((obs - p) ** 2) / (2 * cfg.sigma_lik ** 2) for p in pred])
        w = self.w * np.exp(logl - logl.max())
        w = np.maximum(w / w.sum(), 1e-3)
        self.w = w / w.sum()

    def ess(self):
        return float(1.0 / np.sum(self.w ** 2))

    def mult_mean(self):
        return float(sum(w * g["mult"] for w, g in zip(self.w, self.geos)))


def joint_value(model, cfg, belief, wells, rates, new_wells, t, mult):
    """Expected discounted payoff vector of (rates, new_wells) now, rates held afterwards,
    plus extrapolated continuation. Simulated on every belief particle."""
    H = min(cfg.horizon, cfg.T - t)
    total = np.zeros(cfg.n_ops)
    allw = wells + new_wells
    for geo, st, w in zip(belief.geos, belief.states, belief.w):
        acc, r = np.zeros(cfg.n_ops), None
        for h in range(H):
            st, c = play_stage(model, geo, st, allw, rates, new_wells if h == 0 else (), cfg)
            r = payoff(c, cfg, mult)
            acc += cfg.gamma ** h * r
        rest = cfg.T - t - H
        if cfg.extrap and rest > 0:
            last = r + (c["drill"] if H >= 1 else 0)  # recurring part: no drilling cost
            acc += sum(cfg.gamma ** (H + k) for k in range(rest)) * last
        total += w * acc
    return total


def options_for(op, wells, rates_allowed, cfg, n_new, others_sites=()):
    taken = {(i, j) for _, i, j in wells} | {s for s in others_sites if s is not None}
    free = [s for s in cfg.prefs[op] if s not in taken]
    drill_opts = [None] + (free[:1] if n_new[op] < cfg.max_new and free else [])
    return [(r, s) for r in rates_allowed for s in drill_opts]


def choose(model, cfg, belief, wells, prev_rates, n_new, t, mode):
    ops = range(cfg.n_ops)
    rates_allowed = cfg.rates if cfg.n_ops <= 2 else (0.0, 1.0)
    m_dec = cfg.liab_mult if mode == "nash" else 1.0
    cache = {}

    def val(j):
        key = tuple(j)
        if key not in cache:
            rates = tuple(a[0] for a in j)
            new = tuple((o, a[1][0], a[1][1]) for o, a in enumerate(j) if a[1] is not None)
            cache[key] = joint_value(model, cfg, belief, wells, rates, new, t, m_dec)
        return cache[key]

    opts = [options_for(o, wells, rates_allowed, cfg, n_new) for o in ops]
    if mode == "coop":
        best, bestv = None, -np.inf
        for j in itertools.product(*opts):
            s = [a[1] for a in j if a[1] is not None]
            if len(s) != len(set(s)):
                continue
            v = val(j).sum()
            if v > bestv + 1e-9:
                best, bestv = list(j), v
        return best
    # Nash by iterated best response, start from "continue previous rate, no drill"
    joint = [(min(rates_allowed, key=lambda r: abs(r - prev_rates[o])), None) for o in ops]
    for _ in range(3):
        changed = False
        for o in ops:
            others = [joint[q][1] for q in ops if q != o]
            cur, curv = joint[o], val(joint)[o]
            for a in options_for(o, wells, rates_allowed, cfg, n_new, others):
                j2 = list(joint)
                j2[o] = a
                v = val(j2)[o]
                if v > curv + 1e-9:
                    cur, curv = a, v
            if cur != joint[o]:
                joint[o], changed = cur, True
        if not changed:
            break
    return joint


def scripted(cfg, wells, n_new, mode):
    """bau: drill next preferred site every stage at full rate. static: never drill."""
    joint = []
    for o in range(cfg.n_ops):
        taken = {(i, j) for _, i, j in wells}
        free = [s for s in cfg.prefs[o] if s not in taken]
        drill = free[0] if (mode == "bau" and free and n_new[o] < cfg.max_new) else None
        joint.append((1.0, drill))
    return joint


def run_episode(model, cfg, true_geo, prior_geos, mode, info, seed=0):
    rng = np.random.default_rng(seed)
    geos = [true_geo] if info == "oracle" else prior_geos
    belief = Belief(geos, model, cfg)
    true_state = model.initial_state(cfg.n)
    wells = tuple((o,) + tuple(cfg.init_sites[o]) for o in range(cfg.n_ops))
    n_new = [0] * cfg.n_ops
    prev_rates = [1.0] * cfg.n_ops
    total, log = np.zeros(cfg.n_ops), []
    inj = np.zeros(cfg.n_ops)
    pmax_hist, viol = [], 0.0
    for t in range(cfg.T):
        if mode in ("bau", "static"):
            joint = scripted(cfg, wells, n_new, mode)
        else:
            joint = choose(model, cfg, belief, wells, prev_rates, n_new, t, mode)
        rates = tuple(a[0] for a in joint)
        new = tuple((o, a[1][0], a[1][1]) for o, a in enumerate(joint) if a[1] is not None)
        for o, _, _ in new:
            n_new[o] += 1
        wells = wells + new
        prev_rates = list(rates)
        true_state, c = play_stage(model, true_geo, true_state, wells, rates, new, cfg)
        r = payoff(c, cfg, 1.0)
        total += cfg.gamma ** t * r
        inj += c["vol"]
        pmax_hist.append(c["pmax"])
        viol += max(0.0, c["pmax"] - cfg.p_cap)
        pred_err = None
        if info != "oracle" and mode not in ("bau", "static"):
            pass
        for k, (g, s) in enumerate(zip(belief.geos, belief.states)):
            belief.states[k], _ = play_stage(model, g, s, wells, rates, new, cfg)
        if info == "learn":
            obs = np.array([true_state["p"][i, j] for _, i, j in wells]) + rng.normal(0, cfg.sigma_obs, len(wells))
            belief.update(obs, wells, cfg)
        log.append({"stage": t, "rates": list(rates), "new_wells": [list(w) for w in new],
                    "reward": r.round(2).tolist(), "pmax": round(c["pmax"], 2),
                    "ess": round(belief.ess(), 2), "mult_hat": round(belief.mult_mean(), 3)})
    summary = {"payoff": total.tolist(), "welfare": float(total.sum()), "injected": inj.tolist(),
               "total_injected": float(inj.sum()), "n_wells": len(wells), "pmax_peak": float(max(pmax_hist)),
               "cap_violation": float(viol), "gini_gap": float(abs(total[0] - total[-1]))}
    return summary, log
