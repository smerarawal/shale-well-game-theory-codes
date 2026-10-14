# Shale-well game theory + CO2-storage surrogates

Three layers, one repo:

| layer | where | status |
|---|---|---|
| Physics solver (IMPES two-phase flow with gravity) | `solver_v3.py`, `data_gen_v3.py`, `validate_solver_v3.py` | live |
| Sequential Bayesian Markov game on the solver | `sequential_game/` | live (engine `game3.py`, studies v3-v5) |
| 16-architecture surrogate benchmark + OOD tests | `fair_run/` | live, not yet connected to the game |

## Run
    cd sequential_game && python test_v3.py && python test_v4.py && python test_v5.py
    python run_study5.py --count_only          # then see README_STUDY5.md
Surrogate benchmark: see `fair_run/KAGGLE_RUNBOOK.md`.

## Units and caveats
Solver units are non-dimensional (uncalibrated); welfare numbers are in those units, so signs and ranks
are the findings, not magnitudes. The game uses all-injector wells in a closed 12x12 domain.

## Legacy code (not on main)
Old code lives on `archive/*` branches (each self-contained). See `ARCHIVE.md`.
