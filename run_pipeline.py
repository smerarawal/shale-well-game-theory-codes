"""
Single entry point for a fresh Kaggle/Colab session. Checks what already
exists on disk and ONLY regenerates/retrains what's missing -- if you've
pushed dataset_2000.npz and fno_surrogate.pt to GitHub and pulled them,
this skips straight to Stage 2 in seconds.

Run this ONE script every session instead of the individual stage scripts:
    python run_pipeline.py
"""
import os
import subprocess
import sys


def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"FAILED: {cmd}")
        sys.exit(1)


def main():
    # Stage 0a: dataset
    if os.path.exists("dataset_2000.npz"):
        print("dataset_2000.npz already exists -- skipping generation")
    else:
        print("dataset_2000.npz missing -- generating (takes ~10s)")
        run(
            'python -c "'
            "from data_gen import generate_dataset; import numpy as np; "
            "ds = generate_dataset(n_samples=2000, nx=32, ny=32, nt=150, seed=42); "
            "np.savez_compressed('dataset_2000.npz', permeability=ds['permeability'], "
            "well_mask=ds['well_mask'], final_pressure=ds['final_pressure'])"
            '"'
        )

    # Stage 0b: FNO checkpoint
    if os.path.exists("fno_surrogate.pt"):
        print("fno_surrogate.pt already exists -- skipping training")
    else:
        print("fno_surrogate.pt missing -- training FNO (few minutes on GPU)")
        run("python fno_train.py")

    # Stage 0c: validation (cheap, always safe to (re)run for a fresh report)
    print("\nrunning validation...")
    run("python validate_fno.py")

    # Stage 1: single-agent optimization
    print("\nrunning Stage 1 optimization...")
    run("python optimize_wells.py")

    # Stage 2: game theory
    print("\nrunning Stage 2 (Shapley + Blotto)...")
    run("python shapley_values.py")
    run("python blotto_solver.py")
    run("python blotto_constrained.py")

    # Stage 3: robustness, N-agent generalization, submodularity check
    print("\nrunning Stage 3 (robustness + extensions)...")
    run("python robustness_analysis.py")
    run("python blotto_nagent.py")
    run("python submodularity_check.py")

    # visualization
    print("\ngenerating plots...")
    run("python visualize_stage2.py")

    print("\nDONE. If dataset_2000.npz / fno_surrogate.pt were freshly generated "
          "this run, push them to GitHub now so next session skips straight past "
          "Stage 0:\n"
          "  git add dataset_2000.npz fno_surrogate.pt\n"
          "  git commit -m 'cache dataset + trained fno'\n"
          "  git push")


if __name__ == "__main__":
    main()
