# Shale-well game theory + CO2-storage surrogates

This repository keeps the current solver and sequential game, plus the 16-model surrogate benchmark.

| Area | Location | Status |
|---|---|---|
| Two-phase physics solver | `solver_v3.py`, `data_gen_v3.py`, `validate_solver_v3.py` | Current |
| Sequential Bayesian Markov game | `sequential_game/` | Current engine and studies v3-v5 |
| 16-architecture surrogate benchmark and OOD evaluation | `fair_run/` | Current benchmark; not yet connected to the game |

## Run

```text
cd sequential_game && python test_v3.py && python test_v4.py && python test_v5.py
python run_study5.py --count_only
```

For the surrogate benchmark, see `fair_run/KAGGLE_RUNBOOK.md`.

## Units and caveats

Solver units are non-dimensional and uncalibrated. Welfare signs and ranks are the findings, not the magnitudes. The game uses all-injector wells in a closed 12x12 domain.

## Archived code

Legacy code and exported sequential-game kits are preserved on `archive/*` branches. See `ARCHIVE.md` for the branch map and the full pre-cleanup snapshot.
