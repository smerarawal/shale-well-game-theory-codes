"""
run_all.py -- ONE command for the whole final comparison. Resumable.

    python run_all.py --out /kaggle/working/final_run

Stages (each skipped automatically if already complete):
  1 data        sharded trajectory dataset (CPU, parallel)
  2 ood_data    out-of-distribution test sets (CPU, parallel)
  3 preflight   measured GPU time / memory per architecture; aborts if anything breaks
  4 train       LR sweeps + finals, one worker per GPU, per-architecture, checkpointed
  5 benchmark   sequential inference timing (bs1, bs64)
  6 ood_eval    every model on the OOD sets
  7 report      final table + verdict (final_run/report/final_verdict.md)

If the session limit is near, stages stop cleanly (exit code 3, nothing lost).
Start a new session, attach the previous version's output, and run the SAME
command with  --prev /kaggle/input/<previous-output>/final_run
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

T_START = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))


def py(script, *args, env=None):
    cmd = [sys.executable, os.path.join(HERE, script), *map(str, args)]
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=HERE, env=env).returncode


def import_prev(prev, out):
    if not prev or not os.path.isdir(prev):
        return
    n = 0
    have_ds = os.path.exists(os.path.join(prev, "data", "dataset.npz"))
    for sub in ("data", "ood", "sweep", "results", "weights", "ckpt"):
        for root, _, files in os.walk(os.path.join(prev, sub)):
            if have_ds and os.sep + "shards" in root + os.sep and sub == "data":
                continue  # merged dataset exists: no need to copy the shards too
            for fn in files:
                if fn.endswith(".tmp") or ".tmp." in fn:
                    continue
                s = os.path.join(root, fn)
                d = os.path.join(out, os.path.relpath(s, prev))
                if not os.path.exists(d):
                    os.makedirs(os.path.dirname(d), exist_ok=True)
                    shutil.copy2(s, d)
                    n += 1
    print(f"imported {n} files from {prev}", flush=True)


def hours_left(budget):
    return budget - (time.time() - T_START) / 3600


def pause(msg, out, prev_hint=True):
    print(f"\n=== PAUSED: {msg}")
    print("Nothing is lost. Start a NEW session (the previous version's output saved), attach it as input, and run:")
    print(f"  python run_all.py --out {out} --prev /kaggle/input/<previous-output-name>/{os.path.basename(out)}")
    sys.exit(3)


def main():
    ap = argparse.ArgumentParser()
    default_out = "/kaggle/working/final_run" if os.path.isdir("/kaggle/working") else "final_run"
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--prev", default=None, help="previous session's final_run dir to resume from")
    ap.add_argument("--budget_hours", type=float, default=10.5, help="stop cleanly after this many hours")
    ap.add_argument("--stages", default="data,ood_data,preflight,train,benchmark,ood_eval,report")
    ap.add_argument("--quick", action="store_true", help="tiny smoke configuration (minutes, CPU ok)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict architectures (debug)")
    ap.add_argument("--n_samples", type=int, default=4000)
    ap.add_argument("--n_ood", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--pilot_epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--slow_seeds", type=int, default=2)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--n_workers", type=int, default=0, help="0 = number of GPUs")
    ap.add_argument("--n_proc", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()

    if a.quick:
        a.n_samples, a.n_ood, a.epochs, a.pilot_epochs, a.seeds, a.slow_seeds, a.grid = 60, 25, 3, 1, 1, 1, 16
    steps = 100 if a.quick else 150
    shard = 10 if a.quick else 100
    stages = a.stages.split(",")
    out = a.out
    os.makedirs(out, exist_ok=True)
    import_prev(a.prev, out)
    only = ["--only", *a.only] if a.only else []

    import torch
    n_gpu = torch.cuda.device_count()
    nw = a.n_workers or max(1, n_gpu)
    print(f"out={out} | GPUs={n_gpu} -> {nw} worker(s) | CPUs={a.n_proc} | budget={a.budget_hours}h | quick={a.quick}", flush=True)

    ds = os.path.join(out, "data", "dataset.npz")
    if "data" in stages and not os.path.exists(ds):
        rc = py("make_dataset.py", "--out", os.path.join(out, "data"), "--n_samples", a.n_samples, "--shard_size", shard,
                "--nx", a.grid, "--ny", a.grid, "--gravity", "--n_pressure_steps", steps, "--n_snapshots", 5,
                "--n_proc", a.n_proc, "--time_budget_hours", max(0.05, hours_left(a.budget_hours) - 0.2))
        if rc == 3:
            pause("data generation reached the time budget", out)
        if rc != 0:
            sys.exit(f"data generation failed (exit {rc})")

    ood_ok = all(os.path.exists(os.path.join(out, "ood", f"{n}.npz")) for n in ("ood_more_wells", "ood_rough_perm", "ood_smooth_perm"))
    if "ood_data" in stages and not ood_ok:
        rc = py("make_ood_dataset.py", "--out", os.path.join(out, "ood"), "--n_per_set", a.n_ood, "--nx", a.grid,
                "--ny", a.grid, "--n_pressure_steps", steps, "--n_proc", a.n_proc)
        if rc == 3:
            pause("OOD data generation interrupted", out)
        if rc != 0:
            sys.exit(f"OOD data generation failed (exit {rc})")

    common = ["--epochs", a.epochs, "--pilot_epochs", a.pilot_epochs, "--seeds", a.seeds, "--slow_seeds", a.slow_seeds]
    if "preflight" in stages and not os.path.exists(os.path.join(out, "estimate.json")):
        rc = py("estimate_runtime.py", "--out", out, *common, "--n_workers", nw, *only,
                *(["--steps", 6] if a.quick else []))
        if rc != 0:
            sys.exit("PREFLIGHT FAILED: an architecture errored or produced non-finite values. Nothing was trained. "
                     "Read the FAIL lines above (and send them to me).")

    if "train" in stages:
        train_budget = hours_left(a.budget_hours) - 0.4  # reserve for benchmark + OOD eval + report
        if train_budget < 0.1:
            pause("no time budget left for training in this session", out)
        procs = []
        os.makedirs(os.path.join(out, "logs"), exist_ok=True)
        for k in range(nw):
            env = dict(os.environ)
            if n_gpu:
                env["CUDA_VISIBLE_DEVICES"] = str(k % n_gpu)
            cmd = [sys.executable, os.path.join(HERE, "run_fair_comparison.py"), "--out", out, *map(str, common),
                   "--worker", str(k), "--n_workers", str(nw), "--time_budget_hours", f"{train_budget:.3f}", *map(str, only),
                   *(["--batch_size", "16", "--eval_every", "1", "--ckpt_every", "1"] if a.quick else [])]
            logf = open(os.path.join(out, "logs", f"worker{k}.log"), "a")
            print("$ " + " ".join(cmd) + f"   (log: {logf.name})", flush=True)
            procs.append((k, subprocess.Popen(cmd, cwd=HERE, env=env, stdout=logf, stderr=subprocess.STDOUT)))
        # progress heartbeat every 5 minutes
        while any(p.poll() is None for _, p in procs):
            time.sleep(300 if not a.quick else 5)
            done = len(glob.glob(os.path.join(out, "results", "*.json")))
            print(f"[{(time.time() - T_START) / 3600:5.2f}h] finals complete: {done} | "
                  f"sweeps complete: {len(glob.glob(os.path.join(out, 'sweep', '*.json')))} | "
                  f"{hours_left(a.budget_hours):.1f}h of budget left", flush=True)
        rcs = [p.returncode for _, p in procs]
        print("worker exit codes:", rcs)
        if any(r == 3 for r in rcs):
            pause("training reached the time budget (checkpoints saved)", out)
        if any(r != 0 for r in rcs):
            sys.exit(f"a training worker failed (codes {rcs}); see {out}/logs/worker*.log")

    if "benchmark" in stages and not os.path.exists(os.path.join(out, "bench", "inference.json")):
        py("benchmark_inference.py", "--out", out)
    if "ood_eval" in stages and not os.path.exists(os.path.join(out, "ood", "ood_results.json")):
        py("evaluate_ood.py", "--out", out)
    if "report" in stages:
        py("aggregate_report.py", "--out", out)
        print(f"\nDONE in {(time.time() - T_START) / 3600:.2f} h this session. Report: {out}/report/final_verdict.md")


if __name__ == "__main__":
    main()
