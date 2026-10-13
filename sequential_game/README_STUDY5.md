# Study v5: the permit commons

    python test_v3.py && python test_v4.py && python test_v5.py
    python run_study5.py --out seq3_results --resume_from <old results dir(s)> --hours 11
    python analyze5.py --dir seq3_results

Resumable (one json per job). `--tiers A,B` runs only those tiers (A: Exp 9 + Exp 11 oracle on geologies 0-11,
~20 min on 4 cores; B: Exp 10 + Exp 11 under learning on 0-11, ~3 h; C/D: the same on geologies 12-23).

Engine switches added in game3.py (defaults reproduce v4 exactly): `inj_price` (per-unit injected volume charge seen by
Nash players only; a transfer, welfare ignores it), `alloc` (permit split into equal fixed per-operator allowances),
`br_rounds` (iterated best-response rounds, default 3). Learning arms use planner H=3 (`LEARN_H` in run_study5.py)
because H=6 with 32 particles is several times slower; Exp 9 tests whether H=3 and H=6 agree in this setting.
