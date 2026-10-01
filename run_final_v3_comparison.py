"""
run_final_v3_comparison.py -- THE final, conclusive comparison: all 13
field-predicting architectures, capacity-rebalanced (model_zoo_v3.py),
trained with the per-channel-weighted loss that fixes the dominant-channel
problem (train_utils_v3.py), on REAL v3-with-gravity data, with a written
verdict at the end following the comparison spec's own priority order
(inference speed > final-field accuracy > params > mean L2) -- not just a
table.

Run on your Kaggle T4x2:
    python data_gen_v3.py --out dataset_v3_gravity_1000.npz --gravity \
        --n_samples 1000 --nx 32 --ny 32 --n_pressure_steps 150
    python run_final_v3_comparison.py --dataset dataset_v3_gravity_1000.npz

Single GPU is used by default (torch.device('cuda') picks GPU 0) -- these
are all well under a T4's 16GB, so DataParallel across your second T4
isn't necessary for any one architecture to fit or train at reasonable
speed; it would mainly help if you wanted to train several architectures
CONCURRENTLY in separate processes, which this script doesn't do (runs
them sequentially, so results are directly comparable under identical
otherwise-idle-GPU conditions -- a confound this script deliberately
avoids introducing).
"""
import argparse
import json
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from model_zoo_v3 import make_all_architectures
from train_utils_v3 import compute_channel_weights, train_model, evaluate_model
from physics_residual_v3 import pressure_residual

PHYSICS_ARCHITECTURES = {"physics_informed_fno", "pi_convlstm"}
PHYSICS_WEIGHT = 0.05  # single weight, not a sweep -- see README_FINAL.md for why


class SplitDatasetV3(Dataset):
    def __init__(self, dataset_path, idx, p_mean=None, p_std=None):
        data = np.load(dataset_path)
        perm = data["permeability"].astype(np.float32)[idx]
        poro = data["porosity"].astype(np.float32)[idx]
        mask = data["well_mask"].astype(np.float32)[idx]
        pressure = data["final_pressure"].astype(np.float32)[idx]
        saturation = data["final_saturation"].astype(np.float32)[idx]
        self.x = np.stack([perm, poro, mask], axis=1)
        self.y = np.stack([pressure, saturation], axis=1)
        self.pressure_raw = pressure
        self.saturation_raw = saturation

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i])


def make_fixed_split(n_samples, seed=42, train_frac=0.8):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    n_train = int(n_samples * train_frac)
    return idx[:n_train], idx[n_train:]


def write_verdict(results, out_path):
    """Writes the actual comparison spec's Step 5 verdict FROM the measured
    numbers -- not a template filled with placeholders. Priority order:
    (1) inference speed, (2) final-field accuracy (here: combined
    relative L2, since this dataset is final-field-only, not trajectory),
    (3) parameter count, (4) mean relative L2 as tie-breaker only."""
    by_speed = sorted(results.items(), key=lambda kv: kv[1]["inference_ms_per_sample"])
    by_accuracy = sorted(results.items(), key=lambda kv: kv[1]["mean_rel_l2_combined"])
    by_params = sorted(results.items(), key=lambda kv: kv[1]["n_parameters"])
    by_saturation = sorted(results.items(), key=lambda kv: kv[1]["mean_rel_l2_saturation"])
    by_pressure = sorted(results.items(), key=lambda kv: kv[1]["mean_rel_l2_pressure"])

    lines = ["# Surrogate comparison verdict (v3, two-phase, with gravity)\n"]
    lines.append("## Full results\n")
    lines.append("| architecture | mean_rel_l2 | pressure_rel_l2 | saturation_rel_l2 | params | infer(ms) | train(s) |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["mean_rel_l2_combined"]):
        lines.append(f"| {name} | {r['mean_rel_l2_combined']:.4f} | {r['mean_rel_l2_pressure']:.4f} | "
                      f"{r['mean_rel_l2_saturation']:.4f} | {r['n_parameters']:,} | "
                      f"{r['inference_ms_per_sample']:.3f} | {r['training_wall_clock_seconds']:.0f} |")

    lines.append("\n## Priority-ordered verdict\n")
    lines.append(f"**1. Inference speed** (Shapley/Blotto call volume is the real bottleneck): "
                 f"fastest is **{by_speed[0][0]}** at {by_speed[0][1]['inference_ms_per_sample']:.3f} ms/sample, "
                 f"slowest is {by_speed[-1][0]} at {by_speed[-1][1]['inference_ms_per_sample']:.3f} ms/sample "
                 f"({by_speed[-1][1]['inference_ms_per_sample'] / max(by_speed[0][1]['inference_ms_per_sample'], 1e-6):.1f}x spread).")
    lines.append(f"\n**2. Final-field accuracy**: most accurate overall is **{by_accuracy[0][0]}** "
                 f"(combined rel L2 = {by_accuracy[0][1]['mean_rel_l2_combined']:.4f}). "
                 f"Split by channel: best on pressure is {by_pressure[0][0]} "
                 f"({by_pressure[0][1]['mean_rel_l2_pressure']:.4f}), best on saturation is "
                 f"{by_saturation[0][0]} ({by_saturation[0][1]['mean_rel_l2_saturation']:.4f}) -- "
                 f"{'the same architecture leads both channels' if by_pressure[0][0] == by_saturation[0][0] else 'NOT the same architecture, worth noting if your downstream payoff weights one channel more than the other'}.")
    lines.append(f"\n**3. Parameter count**: smallest is {by_params[0][0]} ({by_params[0][1]['n_parameters']:,}), "
                 f"largest is {by_params[-1][0]} ({by_params[-1][1]['n_parameters']:,}).")
    lines.append(f"\n**4. Mean relative L2 (tie-breaker only)**: {by_accuracy[0][0]} leads at "
                 f"{by_accuracy[0][1]['mean_rel_l2_combined']:.4f}, "
                 f"{by_accuracy[1][0]} second at {by_accuracy[1][1]['mean_rel_l2_combined']:.4f}.")

    lines.append("\n## Reading this table correctly")
    lines.append("- These numbers come from whatever --n_samples/--epochs this run used "
                 "(see the JSON's `run_config` block) -- small-scale numbers establish "
                 "RELATIVE ranking and confirm nothing is broken; re-run at full scale "
                 "(1000+ samples, 100+ epochs) before treating any single number as final.")
    lines.append("- `mean_rel_l2_saturation` is the one that matters most for the MARL/game-theory "
                 "payoff work specifically (plume-channel interference depends on the saturation "
                 "field, not just pressure) -- don't let a good combined score hide a bad "
                 "saturation score.")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote verdict to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_json", default="surrogate_comparison_results_v3_final.json")
    ap.add_argument("--out_verdict", default="surrogate_comparison_verdict_v3.md")
    ap.add_argument("--only", nargs="*", default=None, help="subset of architecture names to run")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data = np.load(args.dataset)
    n_samples = data["permeability"].shape[0]
    H, W = data["permeability"].shape[1], data["permeability"].shape[2]
    train_idx, test_idx = make_fixed_split(n_samples)
    print(f"dataset: {n_samples} samples at {H}x{W}, {len(train_idx)} train / {len(test_idx)} test")

    train_ds = SplitDatasetV3(args.dataset, train_idx)
    test_ds = SplitDatasetV3(args.dataset, test_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    w_pressure, w_saturation = compute_channel_weights(train_ds.pressure_raw, train_ds.saturation_raw)
    print(f"channel loss weights (train-set inverse-variance, sum=2): "
          f"w_pressure={w_pressure:.4f}, w_saturation={w_saturation:.4f}")

    architectures = make_all_architectures(H, W)
    if args.only:
        architectures = {k: v for k, v in architectures.items() if k in args.only}

    results = {}
    for name, ctor in architectures.items():
        print(f"\n=== {name} ===")
        model = ctor().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {n_params:,} parameters")
        physics_fn = pressure_residual if name in PHYSICS_ARCHITECTURES else None
        phys_w = PHYSICS_WEIGHT if name in PHYSICS_ARCHITECTURES else 0.0
        model, train_time = train_model(model, train_loader, device, args.epochs, args.lr,
                                         w_pressure, w_saturation,
                                         physics_loss_fn=physics_fn, physics_weight=phys_w)
        eval_results = evaluate_model(model, test_loader, device)
        eval_results["training_wall_clock_seconds"] = train_time
        results[name] = eval_results
        print(f"  mean_rel_l2={eval_results['mean_rel_l2_combined']:.4f} "
              f"(pressure={eval_results['mean_rel_l2_pressure']:.4f}, "
              f"saturation={eval_results['mean_rel_l2_saturation']:.4f}) "
              f"infer={eval_results['inference_ms_per_sample']:.3f}ms train={train_time:.0f}s")

    output = {
        "run_config": {"dataset": args.dataset, "n_samples": int(n_samples), "H": int(H), "W": int(W),
                        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                        "w_pressure": w_pressure, "w_saturation": w_saturation,
                        "physics_weight": PHYSICS_WEIGHT, "device": str(device)},
        "results": results,
    }
    with open(args.out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nsaved {args.out_json}")
    write_verdict(results, args.out_verdict)


if __name__ == "__main__":
    main()
