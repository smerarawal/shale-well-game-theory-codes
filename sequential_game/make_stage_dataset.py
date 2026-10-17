"""
make_stage_dataset.py -- stage-wise training data for a game-ready surrogate.

Each EPISODE is one geology (random field x permeability multiplier) driven through
T=6 stages by a random game-like policy, on the real solver (SolverStageModel,
fixed t_end = stage_time). Every stage is one transition

    (k, phi, well_rate map, p_in, Sw_in)  ->  (p_out, Sw_out)

so the data carry (a) realistic carried states (they come from solver rollouts, not
random noise), (b) 1..8 injector wells, rates in {0, 0.5, 1} (applied as max(rate,1e-6)
exactly like game3.play_stage), (c) the game's 12x12 grid and fixed stage time, and
(d) a permeability multiplier in [0.5, 2] with extra weight on the low-permeability
tail (mult <= 0.7), where pressure errors are costliest.

Transitions of one episode are strongly correlated: SPLIT BY EPISODE, never by row.
Whole 6-stage rollouts for evaluation are the rows of one episode in stage order
(well_rate[ep, 0..5] are the actions, p_out[ep, s] the true states).

Field seeds: training/validation use seeds >= 1_000_000. The game's own geologies
(true_geo(i) = seed 101+i; particle fields 11, 12) are written by --eval_geos to a
separate file and must never enter training.

    python make_stage_dataset.py --out stage_train.npz --n_episodes 2000 --workers 4
    python make_stage_dataset.py --out stage_eval_game.npz --eval_geos --workers 4
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from game3 import LAYOUTS, make_geology  # noqa: E402
from stage_model import SolverStageModel  # noqa: E402

TRUE_MULTS = [1.0, 0.6, 1.7, 0.7, 1.4, 0.5, 2.0, 0.85, 1.2]  # run_study3.TRUE_MULTS (game truths)
RATES = (0.0, 0.5, 1.0)
T_STAGES = 6
MAX_NEW = 3


def draw_mult(rng):
    """log-uniform on [0.5, 2]; half of the episodes forced into the low-perm tail [0.5, 0.7]."""
    lo, hi = (0.5, 0.7) if rng.random() < 0.5 else (0.5, 2.0)
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def draw_policy(rng):
    """Policy families: random rates; aggressive (rate 1, drill early); mixed with idle stages."""
    u = rng.random()
    return "aggressive" if u < 0.3 else ("mixed" if u < 0.6 else "random")


def free_random_site(rng, n, taken, margin=2):
    for _ in range(100):
        s = (int(rng.integers(margin, n - margin)), int(rng.integers(margin, n - margin)))
        if s not in taken:
            return s
    return None


def run_episode(args):
    ep, field_seed, mult, n, policy_seed, stage_time, random_site_frac, patch = args
    rng = np.random.default_rng(policy_seed)
    if patch is None:
        geo = make_geology(field_seed, mult, n)
    else:  # SPE10 patch: same k*mult convention as game3.make_geology
        geo = {"k": patch["k"] * mult, "phi": patch["phi"], "mult": mult, "seed": field_seed}
    model = SolverStageModel(stage_time=stage_time, gravity=True)
    layout = LAYOUTS[["far", "mid", "close"][int(rng.integers(3))]]
    wells = [(o,) + tuple(layout["init"][o]) for o in range(2)]       # (op, i, j)
    prefs = [list(layout["prefs"][o]) for o in range(2)]
    n_new = [0, 0]
    policy = draw_policy(rng)
    state = model.initial_state(n)
    rows = []
    for t in range(T_STAGES):
        # rates per operator
        if policy == "aggressive":
            rates = [1.0, 1.0]
        elif policy == "mixed":
            rates = [float(rng.choice(RATES, p=[0.3, 0.3, 0.4])) for _ in range(2)]
        else:
            rates = [float(rng.choice(RATES)) for _ in range(2)]
        # drilling: at most one new well per operator per stage, max MAX_NEW each
        taken = {(i, j) for _, i, j in wells}
        for o in range(2):
            p_drill = 0.8 if policy == "aggressive" else 0.35
            if n_new[o] < MAX_NEW and rng.random() < p_drill:
                site = None
                if rng.random() < random_site_frac:
                    site = free_random_site(rng, n, taken)
                else:
                    for s in prefs[o]:
                        if tuple(s) not in taken:
                            site = tuple(s)
                            break
                if site is not None:
                    wells.append((o,) + site)
                    taken.add(site)
                    n_new[o] += 1
        active = [(i, j, max(rates[o], 1e-6)) for o, i, j in wells]
        nxt, elapsed = model.advance(geo, state, active)
        wr = np.zeros((n, n), np.float32)
        for i, j, r in active:
            wr[i, j] = r
        rows.append((wr, state["p"].astype(np.float32), state["Sw"].astype(np.float32),
                     nxt["p"].astype(np.float32), nxt["Sw"].astype(np.float32), len(active), elapsed))
        state = nxt
    return ep, geo["k"].astype(np.float32), geo["phi"].astype(np.float32), mult, field_seed, policy, rows


def build_jobs(a):
    rng = np.random.default_rng(a.seed)
    jobs = []
    if a.spe10_dir:  # OOD eval set: channelised / smooth real-rock patches, never used for training
        from spe10_geologies import load_spe10, sample_patches, TARBERT, UPPER_NESS
        logk, phi = load_spe10(a.spe10_dir)
        layers = {"tarbert": TARBERT, "ness": UPPER_NESS, "both": list(TARBERT) + list(UPPER_NESS)}[a.spe10_region]
        base = {"all": 2_000_000, "test": 2_000_000, "train": 3_000_000}[a.spe10_band]
        for e, pt in enumerate(sample_patches(logk, phi, layers, a.n_episodes, a.n, a.spe10_coarsen, a.seed, band=a.spe10_band)):
            jobs.append((e, base + e, draw_mult(rng), a.n, int(rng.integers(1 << 31)), a.stage_time, a.random_site_frac if a.spe10_band == "train" else 0.0, pt))
    elif a.eval_geos:  # the game's own 24 truths (+ the two particle fields at a few multipliers)
        for i in range(a.n_eval_geos):
            for rep in range(a.reps):
                jobs.append((len(jobs), 101 + i, TRUE_MULTS[i % len(TRUE_MULTS)], a.n,
                             a.seed + 7919 * (i * a.reps + rep + 1), a.stage_time, 0.0, None))
    else:
        for e in range(a.n_episodes):
            jobs.append((e, 1_000_000 + a.seed * 100_000 + e, draw_mult(rng), a.n,
                         int(rng.integers(1 << 31)), a.stage_time, a.random_site_frac, None))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_episodes", type=int, default=2000)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--stage_time", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--random_site_frac", type=float, default=0.15)
    ap.add_argument("--eval_geos", action="store_true", help="episodes on the game's true_geo(0..23)")
    ap.add_argument("--n_eval_geos", type=int, default=24)
    ap.add_argument("--spe10_dir", default=None, help="folder with spe_perm.dat/spe_phi.dat -> OOD eval set from SPE10 patches")
    ap.add_argument("--spe10_region", choices=["tarbert", "ness", "both"], default="ness")
    ap.add_argument("--spe10_band", choices=["all", "train", "test"], default="all",
                    help="train/test: disjoint halves of the long axis (training-time channelised data vs held-out patches)")
    ap.add_argument("--spe10_coarsen", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3, help="policy replicates per game geology (eval_geos)")
    a = ap.parse_args()

    jobs = build_jobs(a)
    t0 = time.time()
    res = []
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap(run_episode, jobs, chunksize=2), 1):
            res.append(r)
            if k % max(1, len(jobs) // 20) == 0:
                el = time.time() - t0
                print(f"  {k}/{len(jobs)} episodes, {el:.0f}s elapsed, ~{el / k * (len(jobs) - k):.0f}s left", flush=True)
    res.sort(key=lambda r: r[0])

    E = len(res)
    n = a.n
    out = {
        "permeability": np.stack([r[1] for r in res]), "porosity": np.stack([r[2] for r in res]),
        "mult": np.array([r[3] for r in res], np.float32), "field_seed": np.array([r[4] for r in res]),
        "policy": np.array([r[5] for r in res]),
    }
    names = ("well_rate", "p_in", "Sw_in", "p_out", "Sw_out")
    for q, nm in enumerate(names):
        out[nm] = np.stack([np.stack([row[q] for row in r[6]]) for r in res])          # [E,T,n,n]
    out["n_wells"] = np.array([[row[5] for row in r[6]] for r in res])                 # [E,T]
    out["elapsed"] = np.array([[row[6] for row in r[6]] for r in res], np.float32)     # [E,T]
    np.savez_compressed(a.out, **out)
    pm = out["p_out"].reshape(E, T_STAGES, -1).max(-1)
    print(f"saved {a.out}: {E} episodes x {T_STAGES} stages = {E * T_STAGES} transitions, "
          f"{time.time() - t0:.0f}s")
    print(f"elapsed per stage: min {out['elapsed'].min():.6f} max {out['elapsed'].max():.6f}")
    print(f"p_max over transitions: min {pm.min():.2f} median {np.median(pm):.2f} max {pm.max():.2f}; "
          f"fraction > cap(6): {(pm > 6).mean():.3f}")
    print(f"wells per stage: {np.bincount(out['n_wells'].ravel())}")


if __name__ == "__main__":
    main()
