# Study v4 (continues study v3)

    python test_v3.py && python test_v4.py                       # ~3 min sanity checks
    python run_study4.py --out seq3_results --resume_from <old results dir(s)> --hours 11
    python analyze3.py --dir seq3_results ; python analyze4.py --dir seq3_results

Re-run the same command to continue; finished jobs (one json each) are skipped. `--hours 0` = no limit.
`--tiers A,B` runs only those tiers (A: belief ablation + H=3 planner check + m dose-response on geologies 0-11;
B: learning arms with the repaired belief; C: same on geologies 12-23; D: the unfinished study-v3 plan).

New engine switches in game3.py (defaults reproduce v3 exactly): `temper` (likelihood exponent),
`model_err` (relative model-error term in the likelihood), predictive diagnostics in every learn-mode log entry
(`z_rms`, `z_cov90`, `pm_q05/pm_q95/pm_mean`).
