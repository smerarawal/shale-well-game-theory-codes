# Sequential Bayesian shared-reservoir game (v2)

Finite-horizon (6 stages) partially observable Markov game on the real two-phase solver (12x12).
Hidden geology (random field x permeability multiplier), public pressure monitoring, shared K=10-particle belief.
Actions per operator per stage: rate in {0, 0.5, 1} (all its wells) x {no drill, drill next preferred site}.
Payoff: margin*injection - drilling cost - pro-rata share of a FIELD-WIDE pressure penalty
lam*max(0, p_max - cap)^2 (proportional liability: pore space and caprock limit are a common pool).

Solution concepts: nash (iterated best response, receding horizon + extrapolated continuation),
coop (joint welfare max), bau/static (scripted baselines). Information: prior / learn (Bayes) / oracle.

Questions (analyze.py): Q1 cost of non-cooperation | Q2 value of monitoring and perfect information |
Q3 liability multiplier | Q4 strategic planning vs naive | Q5 2 vs 3 operators | Q6 physical outcomes |
Q7 belief accuracy

Run: python run_study.py --hours 3.2 --out seq_results && python analyze.py --dir seq_results

Limits: receding-horizon approximation (not a full Markov perfect Bayesian equilibrium); solver units
uncalibrated; all wells are injectors in a closed reservoir; 12x12 grid; coarse particle cover.
