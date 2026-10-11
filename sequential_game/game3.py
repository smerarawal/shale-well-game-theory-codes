"""
game3.py -- v3 of the sequential Bayesian shared-reservoir game. SUPERSET of game2.py:
with default Cfg() every number is identical to v2 (verified in test_v3.py). New switches:

  belief_mode="v3"  16 particles, resample + jitter + exact re-weighting (rejuvenation) when
                    ESS < rejuv_frac*K, optional pruning of negligible particles in planning
  layout            "far"/"mid"/"close": operator well geometry (spacing sweep)
  cap_total         shared injection-volume permit; requested rates are curtailed pro rata
                    when cumulative volume would exceed it (capacity commons / race)
  liab_rule         "prorata" (v2) or "causer": operator owning the well nearest the pressure
                    peak pays the whole penalty
  pilot_rate        ramp-up rule: stage-0 rates capped at this value
  rates             set e.g. (0, .5, .75) for a per-operator injection quota
  force / prefix    deviation experiments (one-shot-deviation check)
Original v2 description follows.

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
    # ---- v3 switches (defaults reproduce v2 exactly)
    layout: str = "v2"
    cap_total: float = float("inf")
    liab_rule: str = "prorata"
    pilot_rate: float = None
    belief_mode: str = "v2"
    prune_thr: float = 0.0
    rejuv_frac: float = 0.5
    jitter: float = 0.1
    stage_time: float = 1.0

    def __post_init__(self):
        if self.layout != "v2":
            lay = LAYOUTS[self.layout]
            self.init_sites = lay["init"] + tuple(self.init_sites[len(lay["init"]):])
            self.prefs = lay["prefs"] + tuple(self.prefs[len(lay["prefs"]):])


# Operator well geometry (12x12 grid). Cross-operator nearest distance: far 5, mid 4, close 2 cells.
LAYOUTS = {
    "far": dict(init=((3, 2), (3, 9)), prefs=(((7, 2), (5, 3), (9, 2)), ((7, 9), (5, 8), (9, 9)))),
    "mid": dict(init=((3, 3), (3, 7)), prefs=(((7, 3), (5, 4), (9, 3)), ((7, 7), (5, 8), (9, 7)))),
    "close": dict(init=((3, 4), (3, 6)), prefs=(((7, 4), (5, 5), (9, 4)), ((7, 6), (5, 7), (9, 6)))),
}


def make_geology(field_seed, mult, n):
    return {"k": random_permeability(n, n, field_seed) * mult, "phi": random_porosity(n, n, field_seed),
            "mult": mult, "seed": field_seed}


def particle_set(n, field_seeds=(11, 12), mults=MULTS):
    return [make_geology(s, m, n) for s in field_seeds for m in mults]


def particle_set_v3(n, field_seeds=(11, 12), n_mults=8):
    """Log-spaced permeability multipliers 0.5..2.0; truths in the study are deliberately off-grid."""
    return [make_geology(sd, float(m), n) for sd in field_seeds for m in np.geomspace(0.5, 2.0, n_mults)]


def play_stage(model, geo, state, wells, rates, new_wells, cfg):
    """wells: tuple (op,i,j); rates: per-operator rate. Returns next state and components:
    gross[o], drill[o], pen (scalar field penalty, m=1), share[o], pmax."""
    cum = state.get("cum", 0.0)
    rates = tuple(rates)
    if np.isfinite(cfg.cap_total):  # shared permit: curtail every operator pro rata if it would be exceeded
        nw = np.zeros(cfg.n_ops)
        for o, _, _ in wells:
            nw[o] += 1
        req = float(sum(rates[o] * cfg.stage_time * nw[o] for o in range(cfg.n_ops)))
        allowed = max(0.0, cfg.cap_total - cum)
        f = 1.0 if (req <= allowed + 1e-12 or req <= 0) else allowed / req
        rates = tuple(r * f for r in rates)
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
    if cfg.liab_rule == "causer" and pen > 0:
        pi, pj = np.unravel_index(int(np.argmax(nxt["p"])), nxt["p"].shape)
        owner = min(wells, key=lambda w: ((w[1] - pi) ** 2 + (w[2] - pj) ** 2, w[0]))[0]
        share = np.zeros(cfg.n_ops)
        share[owner] = 1.0
    else:
        share = vol / vol.sum() if vol.sum() > 1e-9 else np.zeros(cfg.n_ops)
    nxt = dict(nxt)
    nxt["cum"] = cum + float(vol.sum())
    return nxt, {"gross": gross, "drill": drill, "pen": pen, "share": share, "pmax": pmax, "vol": vol}


def payoff(c, cfg, mult=1.0):
    return c["gross"] - c["drill"] - mult * c["pen"] * c["share"]


def wquantile(vals, w, q):
    o = np.argsort(vals)
    v, ww = np.asarray(vals)[o], np.asarray(w)[o]
    c = np.cumsum(ww) / ww.sum()
    return float(v[min(np.searchsorted(c, q), len(v) - 1)])


class Belief:
    """v2 belief: fixed particles, Bayes weights with a 1e-3 floor (collapses to ~1 particle)."""

    def __init__(self, geos, model, cfg, **_):
        self.geos = geos
        self.model, self.cfg = model, cfg
        self.states = [model.initial_state(cfg.n) for _ in geos]
        self.w = np.full(len(geos), 1.0 / len(geos))

    def advance(self, wells, rates, new, cfg):
        for k, (g, s) in enumerate(zip(self.geos, self.states)):
            self.states[k], _ = play_stage(self.model, g, s, wells, rates, new, cfg)

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

    def mult_q(self, q):
        return wquantile([g["mult"] for g in self.geos], self.w, q)

    n_rejuv = 0


class BeliefV3(Belief):
    """Particle filter with exact re-weighting and rejuvenation. When ESS < rejuv_frac*K the
    particles are resampled (systematic), each is jittered (log-normal on the permeability
    multiplier, occasionally a new random field from the pool), its whole history is replayed
    on the solver, and weights are set from the likelihood of ALL past observations."""

    def __init__(self, geos, model, cfg, pool_seeds=(11, 12, 13, 14), seed=0, **_):
        super().__init__(geos, model, cfg)
        self.pool_seeds = tuple(pool_seeds)
        self.rng = np.random.default_rng(seed + 777)
        self.hist = []  # (wells, rates, new, obs or None)
        self.n_rejuv = 0

    def advance(self, wells, rates, new, cfg):
        super().advance(wells, rates, new, cfg)
        self.hist.append([wells, tuple(rates), tuple(new), None])

    def update(self, obs, wells, cfg):
        pred = [np.array([s["p"][i, j] for _, i, j in wells]) for s in self.states]
        logl = np.array([-np.sum((obs - p) ** 2) / (2 * cfg.sigma_lik ** 2) for p in pred])
        w = self.w * np.exp(logl - logl.max())
        w = np.maximum(w / w.sum(), 1e-12)
        self.w = w / w.sum()
        self.hist[-1][3] = np.asarray(obs)
        if self.ess() < cfg.rejuv_frac * len(self.geos):
            self.rejuvenate(cfg)

    def rejuvenate(self, cfg):
        K = len(self.geos)
        u = (self.rng.random() + np.arange(K)) / K  # systematic resampling
        idx = np.minimum(np.searchsorted(np.cumsum(self.w), u), K - 1)
        geos, states, logls = [], [], []
        for i in idx:
            g = self.geos[i]
            mult = float(np.clip(g["mult"] * np.exp(self.rng.normal(0, cfg.jitter)), 0.35, 3.0))
            seed = g["seed"] if self.rng.random() > 0.3 else int(self.rng.choice(self.pool_seeds))
            ng = make_geology(seed, mult, cfg.n)
            st, ll = self.model.initial_state(cfg.n), 0.0
            for wells, rates, new, obs in self.hist:
                st, _ = play_stage(self.model, ng, st, wells, rates, new, cfg)
                if obs is not None:
                    pred = np.array([st["p"][a, b] for _, a, b in wells])
                    ll += -np.sum((obs - pred) ** 2) / (2 * cfg.sigma_lik ** 2)
            geos.append(ng), states.append(st), logls.append(ll)
        logls = np.array(logls)
        w = np.exp(logls - logls.max())
        self.geos, self.states, self.w = geos, states, w / w.sum()
        self.n_rejuv += 1


def joint_value(model, cfg, belief, wells, rates, new_wells, t, mult):
    """Expected discounted payoff vector of (rates, new_wells) now, rates held afterwards,
    plus extrapolated continuation. Simulated on every belief particle."""
    H = min(cfg.horizon, cfg.T - t)
    total = np.zeros(cfg.n_ops)
    allw = wells + new_wells
    wts = belief.w
    if cfg.prune_thr > 0:  # skip negligible particles in planning (v3 only)
        keep = wts >= cfg.prune_thr
        if not keep.any():
            keep = wts >= wts.max()
        wts = wts * keep
        wts = wts / wts.sum()
    for geo, st, w in zip(belief.geos, belief.states, wts):
        if w == 0.0:
            continue
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
    if cfg.pilot_rate is not None and t == 0:  # ramp-up rule
        rates_allowed = tuple(r for r in rates_allowed if r <= cfg.pilot_rate + 1e-9)
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


def resolve_action(op, rate, drill, wells, cfg, n_new, joint):
    """(rate, drill flag) -> (rate, site|None): next free preferred site, avoiding other operators' sites."""
    if not drill:
        return (rate, None)
    others = [joint[q][1] for q in range(cfg.n_ops) if q != op and joint[q] is not None]
    sites = [a[1] for a in options_for(op, wells, (rate,), cfg, n_new, others) if a[1] is not None]
    return (rate, sites[0] if sites else None)


def run_episode(model, cfg, true_geo, prior_geos, mode, info, seed=0, force=None, prefix=None,
                pool_seeds=(11, 12, 13, 14)):
    """force={t: {op: (rate, drill_flag)}} overrides one operator's action at stage t AFTER planning
    (others keep the planner's action). prefix=[joint,...] replays given joints for the first stages
    instead of planning (used by the one-shot-deviation check)."""
    rng = np.random.default_rng(seed)
    geos = [true_geo] if info == "oracle" else prior_geos
    BeliefCls = BeliefV3 if (cfg.belief_mode == "v3" and info != "oracle") else Belief
    belief = BeliefCls(geos, model, cfg, pool_seeds=pool_seeds, seed=seed)
    true_state = model.initial_state(cfg.n)
    wells = tuple((o,) + tuple(cfg.init_sites[o]) for o in range(cfg.n_ops))
    n_new = [0] * cfg.n_ops
    prev_rates = [1.0] * cfg.n_ops
    total, log = np.zeros(cfg.n_ops), []
    inj = np.zeros(cfg.n_ops)
    pmax_hist, viol = [], 0.0
    for t in range(cfg.T):
        if prefix is not None and t < len(prefix):
            joint = [(a[0], tuple(a[1]) if a[1] is not None else None) for a in prefix[t]]
        elif mode in ("bau", "static"):
            joint = scripted(cfg, wells, n_new, mode)
        else:
            joint = choose(model, cfg, belief, wells, prev_rates, n_new, t, mode)
        if force and t in force:
            joint = list(joint)
            for o, (rate, drill) in force[t].items():
                joint[o] = resolve_action(o, rate, drill, wells, cfg, n_new, joint)
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
        belief.advance(wells, rates, new, cfg)
        if info == "learn":
            obs = np.array([true_state["p"][i, j] for _, i, j in wells]) + rng.normal(0, cfg.sigma_obs, len(wells))
            belief.update(obs, wells, cfg)
        log.append({"stage": t, "rates": list(rates), "new_wells": [list(w) for w in new],
                    "joint": [[a[0], list(a[1]) if a[1] is not None else None] for a in joint],
                    "reward": r.round(2).tolist(), "pmax": round(c["pmax"], 2),
                    "ess": round(belief.ess(), 2), "mult_hat": round(belief.mult_mean(), 3),
                    "q05": round(belief.mult_q(0.05), 3), "q95": round(belief.mult_q(0.95), 3)})
    summary = {"payoff": total.tolist(), "welfare": float(total.sum()), "injected": inj.tolist(),
               "total_injected": float(inj.sum()), "n_wells": len(wells), "pmax_peak": float(max(pmax_hist)),
               "cap_violation": float(viol), "gini_gap": float(abs(total[0] - total[-1])),
               "n_rejuv": int(getattr(belief, "n_rejuv", 0))}
    return summary, log


# ---------------------------------------------------------------------------------------------
# schedule replay (equilibrium-quality checks): sched[t][op] = (rate, site) with site in
# {None, (i, j), "dyn"}; "dyn" means "drill the next free preferred site at that time".
# ---------------------------------------------------------------------------------------------
def schedule_from_log(log, cfg):
    sched = []
    for e in log:
        row = []
        for o in range(cfg.n_ops):
            site = next((tuple(w[1:3]) for w in e["new_wells"] if w[0] == o), None)
            row.append((e["rates"][o], site))
        sched.append(row)
    return sched


def simulate_schedule(model, cfg, true_geo, sched):
    state = model.initial_state(cfg.n)
    wells = tuple((o,) + tuple(cfg.init_sites[o]) for o in range(cfg.n_ops))
    n_new = [0] * cfg.n_ops
    total = np.zeros(cfg.n_ops)
    for t in range(cfg.T):
        taken = {(i, j) for _, i, j in wells}
        sites = []
        for o in range(cfg.n_ops):
            s = sched[t][o][1]
            if s is not None and s != "dyn" and (tuple(s) in taken or n_new[o] >= cfg.max_new):
                s = None
            sites.append(s)
        for o in range(cfg.n_ops):
            if sites[o] == "dyn":
                tk = taken | {tuple(x) for q, x in enumerate(sites) if q != o and x not in (None, "dyn")}
                free = [x for x in cfg.prefs[o] if x not in tk]
                sites[o] = free[0] if (free and n_new[o] < cfg.max_new) else None
        new = tuple((o,) + tuple(sites[o]) for o in range(cfg.n_ops) if sites[o] is not None)
        for o, _, _ in new:
            n_new[o] += 1
        wells = wells + new
        rates = tuple(sched[t][o][0] for o in range(cfg.n_ops))
        state, c = play_stage(model, true_geo, state, wells, rates, new, cfg)
        total += cfg.gamma ** t * payoff(c, cfg, 1.0)
    return total
