# Final fair run: one command

## What it does (all in `python run_all.py`)
1. **Data**: 4000 samples, 32x32, solver_v3 WITH gravity (rho_w=1.0, rho_n=0.6, g=1.5), 150 pressure steps, 6 snapshots each (trajectory + final frame). Split 80/10/10, fixed.
2. **OOD sets** (200 each): 5-7 wells (train 2-4), rough permeability (corr. length 2 vs 6), smooth permeability (12 vs 6).
3. **GPU preflight**: measured time and memory for every architecture; stops before training if anything errors or goes non-finite; prints the MEASURED total-hours estimate.
4. **Training**, 16 models, one worker per GPU: per-model LR sweep {3e-4, 1e-3, 3e-3} (60 epochs, seed 0, chosen on validation); physics-informed models also sweep physics weight {3, 10, 30, 100}; then 300 epochs x 3 seeds (FINN, GNS, GNS-traj: 2 seeds, slow and not contenders). Best-validation checkpoint, test set touched once.
5. **Inference benchmark** (sequential, bs1 and bs64), 6. **OOD evaluation**, 7. **Report**: `final_run/report/final_verdict.md`.

The 16 models = the original 13 + trajectory-trained twins of R-U-Net, PI-ConvLSTM and GNS (so you can see what intermediate-time supervision buys).

## Steps

### One-time: get the code into your repo
On your computer, unzip `fair_run.zip` into the root of your clone of `shale-well-game-theory-codes`, then:
```
git add fair_run && git commit -m "final fair comparison run" && git push
```
(No GitHub? Upload `fair_run.zip` as a Kaggle Dataset instead and use the "Option B" cell below.)

### Kaggle notebook setup
New notebook -> Settings: Accelerator **GPU T4 x2**, Internet **On**, Persistence: Files only.

### Optional 3-minute sanity test (do this first)
Tests the two-GPU launch, the one thing I could not test without a GPU:
```
!git clone https://github.com/smerarawal/shale-well-game-theory-codes /kaggle/working/repo
%cd /kaggle/working/repo/fair_run
!python run_all.py --quick --out /kaggle/working/quick_test
```
It should end with "DONE" and a report. (Rankings from it are meaningless: 3 epochs, 60 samples.)

### The real run (cell 1; use Save Version -> Save & Run All so it keeps running if you close the tab)
```
!git clone https://github.com/smerarawal/shale-well-game-theory-codes /kaggle/working/repo
%cd /kaggle/working/repo/fair_run
!python run_all.py
```
Option B (zip as Kaggle Dataset): `!cp -r /kaggle/input/<dataset-name>/fair_run /kaggle/working/ && cd /kaggle/working/fair_run && python run_all.py`

Output goes to `/kaggle/working/final_run`. Early on it prints the preflight table with the measured GPU-hour estimate. A heartbeat line prints every 5 minutes.

### If it prints "PAUSED" (time budget reached; checkpoints saved, nothing lost)
The default budget is 10.5 h so the notebook ends normally before Kaggle's 12 h limit and its output is kept. Then:
1. Let the version finish saving.
2. New notebook (same settings) -> Add Input -> **Notebook Output** -> pick the previous version.
3. Run the same cell, adding `--prev`:
```
!git clone https://github.com/smerarawal/shale-well-game-theory-codes /kaggle/working/repo
%cd /kaggle/working/repo/fair_run
!python run_all.py --prev /kaggle/input/<previous-notebook-name>/final_run
```
Finished runs are skipped; the interrupted one resumes from its last checkpoint (saved every 10 epochs). Repeat until it prints "DONE".

### Save GPU quota (optional)
Data generation (~1 h, CPU only) can run in a CPU-only notebook first: `!python run_all.py --stages data,ood_data`. Then attach that output and run the real run with `--prev`.

## Reading the result
`final_run/report/final_verdict.md`: ranked table (balanced = mean of pressure and saturation rel-L2, mean +- std over seeds, plus both channels separately), speed, Pareto set, trajectory-supervision effect, OOD robustness. A gap smaller than the leader's seed-std is reported as a tie.

## Known limits (stated, not hidden)
- Physics-informed variants use the pressure-equation residual only: the data has no velocity, so a saturation-transport residual is not possible without changing the solver to save fluxes.
- E2C and the *_traj models use intermediate-time supervision the others do not; each has a final-frame-only twin except E2C.
- Estimates before the preflight are extrapolated; the preflight replaces them with measurements.
- Not included: the downstream Shapley/Blotto check against the real solver.
