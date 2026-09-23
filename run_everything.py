"""
run_everything.py -- single entry point for a full session: v1 pipeline
(dataset + game theory), v2 dataset, v3 datasets (both physics variants)
+ validation gate, every CURRENTLY-WIRED surrogate comparison, and the
full visualization suite.

WHAT THIS DOES NOT DO: train nested_fno.py, runet_recurrent.py,
pi_convlstm.py, e2c.py, gns.py, deeponet.py, hybrid_deeponet_kan.py,
finn.py, or near_well_hybrid.py on real data. Those 9 files are
implemented and pass their own standalone shape/gradient-flow checks
(`python <file>.py` on synthetic random tensors) but are NOT wired into
any real dataset or comparison harness yet -- see run_surrogate_
comparison_v3.py's module docstring for exactly why each one needs
different data (trajectory snapshots, sensor/query points, or a
conservation-law-structured split) that doesn't exist yet. Running this
script does not change that; it's flagged here so "run everything" means
what's actually wired up, not a false claim to cover all 14 architectures
from the original spec.

Expected wall-clock: dataset generation for v3 is the slow part (~40min
EACH for the no-gravity and gravity variants, single-threaded numpy, no
GPU involved -- see data_gen_v3_multiphase.py's own timing note). Every
surrogate training stage needs a GPU to finish in reasonable time (100
epochs x up to 8 architectures per dataset); on CPU this will take hours
per stage. Recommend Colab/Kaggle with a GPU runtime for everything from
Stage 2 onward.

Run:
    pip install torch neuraloperator matplotlib
    python run_everything.py
    python run_everything.py --skip-v3-gravity   # skip the gravity dataset+comparison
    python run_everything.py --only v3            # v3 dataset+comparison+validation only
"""
import argparse
import os
import subprocess
import sys


def run(cmd, allow_fail=False):
    print(f"\n{'='*70}\n>>> {cmd}\n{'='*70}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"FAILED: {cmd}")
        if not allow_fail:
            sys.exit(1)
    return result.returncode == 0


def stage_v1(args):
    print("\n########## STAGE V1: dataset + FNO + game theory ##########")
    # run_pipeline.py already handles dataset_2000.npz -> fno_train.py ->
    # validate_fno.py -> Stage 1/2/3 game theory -> visualize_stage2.py.
    # Not duplicating that logic here, just invoking it.
    run("python run_pipeline.py")

    print("\n---- v1 surrogate architecture comparison (plain FNO vs U-Net vs PI-FNO) ----")
    run("python comparison_protocol.py dataset_2000.npz")  # (re)build the split FOR v1's dataset --
    # NOTE: fixed_split.npz is a single file, always describing whichever
    # dataset comparison_protocol.py last ran against. Every stage below
    # that trains against a different dataset rebuilds it immediately
    # before training, specifically to avoid a stale split silently
    # mismatching a different dataset's sample count.
    run("python run_surrogate_comparison.py")


def stage_v2(args):
    print("\n########## STAGE V2: anisotropic + porosity dataset ##########")
    if os.path.exists("dataset_v2_2000.npz"):
        print("dataset_v2_2000.npz already exists -- skipping generation")
    else:
        run("python data_gen_v2.py")
    run("python validate_fno_v2.py", allow_fail=True)
    run("python fno_train_v2.py", allow_fail=True)
    print("NOTE: there is no run_surrogate_comparison_v2.py -- v2 was single-phase "
          "and scientifically superseded by v3 for the interesting (saturation-front) "
          "comparison. Not building a v2-specific multi-architecture harness; v1's "
          "covers the pressure-only case and v3's covers pressure+saturation.")


def stage_v3(args):
    print("\n########## STAGE V3: two-phase (gravity-off) ##########")
    run("python validate_solver_v3.py")  # hard gate -- must PASS (exits nonzero on FAIL)

    if os.path.exists("dataset_v3_1000.npz"):
        print("dataset_v3_1000.npz already exists -- skipping generation (~40min if regenerated)")
    else:
        run("python data_gen_v3_multiphase.py 1000")
    run("python comparison_protocol.py dataset_v3_1000.npz")
    run("python run_surrogate_comparison_v3.py dataset_v3_1000.npz")
    # results land in surrogate_comparison_results_v3.json -- rename before
    # the gravity stage overwrites it, if you're running both in one session
    if not args.skip_v3_gravity:
        run("cp surrogate_comparison_results_v3.json surrogate_comparison_results_v3_nogravity.json",
            allow_fail=True)

    if not args.skip_v3_gravity:
        print("\n########## STAGE V3-GRAVITY: two-phase (buoyancy on) ##########")
        if os.path.exists("dataset_v3_gravity_1000.npz"):
            print("dataset_v3_gravity_1000.npz already exists -- skipping generation (~40min if regenerated)")
        else:
            run("python data_gen_v3_multiphase.py 1000 --gravity")
        run("python comparison_protocol.py dataset_v3_gravity_1000.npz")
        run("python run_surrogate_comparison_v3.py dataset_v3_gravity_1000.npz")
        run("cp surrogate_comparison_results_v3.json surrogate_comparison_results_v3_gravity.json",
            allow_fail=True)
        # visualize_everything.py's surrogate_comparison_bars() reads the
        # PLAIN "surrogate_comparison_results_v3.json" name -- leave the
        # gravity run's output under that name (the last one written) so
        # the plot picks it up; the *_nogravity/_gravity copies above are
        # for you to diff by hand, not consumed by any script.


def stage_visualize(args):
    print("\n########## VISUALIZATION ##########")
    run("python visualize_solver_v3.py", allow_fail=True)
    run("python visualize_everything.py", allow_fail=True)
    print("\nAll plots in viz_output/ (+3 solver_v3_*.png at repo root from visualize_solver_v3.py)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["v1", "v2", "v3"], default=None,
                         help="Run only one stage's pipeline (skips the others AND skips final visualization)")
    parser.add_argument("--skip-v3-gravity", action="store_true",
                         help="Skip the gravity dataset + comparison (still ~40min saved)")
    args = parser.parse_args()

    if args.only == "v1":
        stage_v1(args)
        return
    if args.only == "v2":
        stage_v2(args)
        return
    if args.only == "v3":
        stage_v3(args)
        return

    stage_v1(args)
    stage_v2(args)
    stage_v3(args)
    stage_visualize(args)

    print("\n" + "=" * 70)
    print("DONE. Push the generated datasets + result JSONs so next session")
    print("skips straight past the slow parts:")
    print("  git add dataset_2000.npz dataset_v2_2000.npz dataset_v3_1000.npz "
          "dataset_v3_gravity_1000.npz fno_surrogate.pt surrogate_comparison_results*.json")
    print("  git commit -m 'cache datasets + surrogate comparison results'")
    print("  git push")
    print("=" * 70)


if __name__ == "__main__":
    main()
