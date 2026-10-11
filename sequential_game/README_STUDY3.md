# Study v3: Experiments 1-4 (real solver in the loop, CPU only)

Files: game3.py (engine, superset of game2: defaults reproduce v2 exactly), eps_check.py (equilibrium checks),
run_study3.py (tiered, resumable, time-budgeted driver), analyze3.py (paired CIs), test_v3.py (1-minute sanity check).

Run:
    python test_v3.py
    python run_study3.py --hours 3.4 --out seq3_results
    python analyze3.py --dir seq3_results [--baseline_dir <study-v2 results folder>]

Experiments
1 Equilibrium quality: horizon convergence (H=1..6), open-loop deviation epsilon, one-shot-deviation epsilon.
2 Belief + ramp-up: rejuvenated 16-particle filter (ESS, posterior error, 90% coverage), value of monitoring,
  mandatory low-rate first stage.
3 Externality strength: well spacing {far, mid, close} x shared injection permit {none, 24, 16}.
4 Policies: liability multiplier m=2,3, owner-of-peak-pays, per-operator quota 0.75; gains split into
  externality correction (perfect information) and precaution (learning).

Tiers (every prefix is a complete paired sample): 1 = oracle experiments on geologies 0-11; 2 = learning experiments
and deviation checks on 0-7; 3 = more learning runs, coop-learn, oracle experiments on 12-23; 4 = 24-35.
