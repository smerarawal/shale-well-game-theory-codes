"""
game.py -- sequential Bayesian game of CO2-style injection by several operators.

Setting (a deliberately small, fully inspectable instance):
  * N operators share one closed reservoir. Each starts with one injection well.
  * Each stage (a "year") every operator chooses: hold, or drill one new well at one
    of its two preferred free sites. Wells inject at a fixed rate.
  * The geology (permeability, porosity) is hidden. All operators share a belief:
    K candidate geologies with weights, updated by Bayes' rule from noisy pressure
    readings at every well (public monitoring).
  * Payoff per stage = margin * injected mass - drilling cost - penalty on pressure
    above a cap at the operator's own wells. Discounted by gamma per stage. The
    shared pore space is the externality: others' injection raises pressure everywhere.

Decision rules (approximations, stated honestly):
  * "nash": each stage, iterated best response over the stage game, where each joint
    action is valued by simulating `horizon` stages ahead on every belief particle
    with no further drilling after this stage. This is a one-stage-lookahead Bayesian
    Nash, not a full Markov perfect Bayesian equilibrium.
  * "coop": the joint action maximizing the sum of payoffs under the same valuation.

Information modes: "prior" (never learn), "learn" (Bayes update), "oracle" (knows geology).
"""
import itertools
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_gen_v3 import random_permeability, random_porosity  # noqa: E402


@dataclass
class Cfg:
    n: int = 12
    T: int = 4
    n_ops: int = 2
    init_sites: tuple = ((2, 2), (9, 9))
    prefs: tuple = (((5, 5), (2, 9), (9, 2), (2, 5), (6, 6), (9, 6)),
                    ((6, 6), (9, 2), (2, 9), (9, 6), (5, 5), (2, 5)))
    rate: float = 1.0
    margin: float = 20.0
    c_well: float = 18.0
    p_cap: float = 5.0
    lam: float = 6.0
    gamma: float = 1 / 1.05
    sigma: float = 0.3
    horizon: int = 2


def make_geology(seed, n):
    return {"k": random_permeability(n, n, seed), "phi": random_porosity(n, n, seed)}


def play_stage(model, geo, state, wells, new_wells, cfg):
    """wells: tuple of (op, i, j) active this stage (new ones included).
    Returns next state and the per-operator reward vector for the stage."""
    nxt, t = model.advance(geo, state, [(i, j, cfg.rate) for _, i, j in wells])
    r = np.zeros(cfg.n_ops)
    for o, i, j in wells:
        r[o] += cfg.margin * cfg.rate * t - cfg.lam * max(0.0, nxt["p"][i, j] - cfg.p_cap) ** 2
    for o, _, _ in new_wells:
        r[o] -= cfg.c_well
    return nxt, r


class Belief:
    def __init__(self, geos, model, cfg):
        self.geos = geos
        self.states = [model.initial_state(cfg.n) for _ in geos]
        self.w = np.full(len(geos), 1.0 / len(geos))

    def update(self, obs, wells, sigma):
        logl = np.array([-np.sum((obs - np.array([s["p"][i, j] for _, i, j in wells])) ** 2) / (2 * sigma ** 2)
                         for s in self.states])
        w = self.w * np.exp(logl - logl.max())
        w = np.maximum(w / w.sum(), 1e-3)
        self.w = w / w.sum()


def joint_value(model, cfg, belief, wells, new_wells, t):
    """Expected discounted payoff vector of applying `new_wells` now, then holding."""
    H = min(cfg.horizon, cfg.T - t)
    total = np.zeros(cfg.n_ops)
    allw = wells + new_wells
    for geo, st, w in zip(belief.geos, belief.states, belief.w):
        acc = np.zeros(cfg.n_ops)
        for h in range(H):
            st, r = play_stage(model, geo, st, allw, new_wells if h == 0 else (), cfg)
            acc += cfg.gamma ** h * r
        total += w * acc
    return total


def options_for(op, wells, cfg, others_sites=()):
    taken = {(i, j) for _, i, j in wells} | set(s for s in others_sites if s is not None)
    free = [s for s in cfg.prefs[op] if s not in taken]
    return [None] + free[:2]


def to_new_wells(joint):
    return tuple((o, s[0], s[1]) for o, s in enumerate(joint) if s is not None)


def choose(model, cfg, belief, wells, t, mode):
    cache = {}

    def val(j):
        key = tuple(j)
        if key not in cache:
            cache[key] = joint_value(model, cfg, belief, wells, to_new_wells(j), t)
        return cache[key]

    if mode == "coop":
        best, bestv = None, -np.inf
        for j in itertools.product(*[options_for(o, wells, cfg) for o in range(cfg.n_ops)]):
            sites = [s for s in j if s is not None]
            if len(sites) != len(set(sites)):
                continue
            v = val(list(j)).sum()
            if v > bestv + 1e-9:
                best, bestv = list(j), v
        return best
    joint = [None] * cfg.n_ops
    for _ in range(4):
        changed = False
        for op in range(cfg.n_ops):
            cur, curv = joint[op], val(joint)[op]
            others = [joint[o] for o in range(cfg.n_ops) if o != op]
            for opt in options_for(op, wells, cfg, others):
                j2 = list(joint)
                j2[op] = opt
                v = val(j2)[op]
                if v > curv + 1e-9:
                    cur, curv = opt, v
            if cur != joint[op]:
                joint[op], changed = cur, True
        if not changed:
            break
    return joint


def run_episode(model, cfg, true_geo, prior_geos, mode, info, seed=0):
    rng = np.random.default_rng(seed)
    geos = [true_geo] if info == "oracle" else prior_geos
    belief = Belief(geos, model, cfg)
    true_state = model.initial_state(cfg.n)
    wells = tuple((o,) + tuple(cfg.init_sites[o]) for o in range(cfg.n_ops))
    total, log = np.zeros(cfg.n_ops), []
    for t in range(cfg.T):
        joint = choose(model, cfg, belief, wells, t, mode)
        new = to_new_wells(joint)
        wells = wells + new
        true_state, r = play_stage(model, true_geo, true_state, wells, new, cfg)
        total += cfg.gamma ** t * r
        for k, (g, s) in enumerate(zip(belief.geos, belief.states)):
            belief.states[k], _ = play_stage(model, g, s, wells, new, cfg)
        if info == "learn":
            obs = np.array([true_state["p"][i, j] for _, i, j in wells]) + rng.normal(0, cfg.sigma, len(wells))
            belief.update(obs, wells, cfg.sigma)
        log.append({"stage": t, "new_wells": [list(w) for w in new], "reward": r.round(2).tolist(),
                    "p_at_wells": [round(float(true_state["p"][i, j]), 2) for _, i, j in wells],
                    "belief_max_w": round(float(belief.w.max()), 3)})
    return total, log
