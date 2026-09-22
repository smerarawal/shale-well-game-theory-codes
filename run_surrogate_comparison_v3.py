"""
run_surrogate_comparison_v3.py -- v3 (two-phase) version of
run_surrogate_comparison.py.

Trains and evaluates plain FNO, U-Net baseline, and physics-informed FNO
(sweeping physics_weight over {0, 0.01, 0.1, 1.0} -- the full sweep the
spec calls for; run_surrogate_comparison.py only ever ran weight=0.1) on
dataset_v3_*.npz, predicting BOTH final_pressure and final_saturation as a
2-channel output instead of v1's single final_pressure channel.

3 input channels here (permeability, porosity, well_mask), not v2's 4 --
solver_v3.py takes a single isotropic k field, not kx/ky, so there is no
second permeability channel to stack.

U-FNO (the architecture actually motivated by this exact saturation-front
problem, per the physics spec's Priority 2.4) is deliberately NOT included
here -- it doesn't exist in this repo yet. This script is what makes
building it worthwhile in the first place (a real Sw output to test it
against); building U-FNO itself is the next file, not this one.

Physics loss here penalizes the residual on the PRESSURE channel only
(pred[:, 0:1]) -- physics_residual_loss encodes the elliptic pressure
equation, not the saturation transport equation, so applying it to the
saturation channel would be penalizing the wrong PDE. The saturation
channel still gets trained, just through the plain MSE data term.

Run:
    pip install torch neuraloperator
    python data_gen_v3_multiphase.py 1000      # once, if dataset_v3_1000.npz doesn't exist
    python comparison_protocol.py dataset_v3_1000.npz   # once
    python run_surrogate_comparison_v3.py [dataset_v3_1000.npz]

Expects the named dataset (default dataset_v3_1000.npz) and fixed_split.npz
(built FROM that same dataset -- rerun comparison_protocol.py if you
regenerate the dataset at a different sample count) in the same directory.
"""
import json
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from neuralop.models import FNO

from unet_baseline import UNetSurrogate
from fno_physics_informed import physics_residual_loss
from eval_utils import full_evaluation, count_parameters

DATASET_PATH = sys.argv[1] if len(sys.argv) > 1 else "dataset_v3_1000.npz"
SPLIT_PATH = "fixed_split.npz"
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
PHYSICS_WEIGHTS = [0.0, 0.01, 0.1, 1.0]


class SplitDatasetV3(Dataset):
    """3 input channels (permeability, porosity, well_mask), 2 output
    channels (final_pressure, final_saturation). Norm stats from TRAIN
    indices only, same leakage-avoidance rule as v1's SplitDataset."""

    def __init__(self, dataset_path, split_path, subset):
        data = np.load(dataset_path)
        split = np.load(split_path)
        idx = split[f"{subset}_idx"]
        train_idx = split["train_idx"]

        perm = data["permeability"].astype(np.float32)
        poro = data["porosity"].astype(np.float32)
        mask = data["well_mask"].astype(np.float32)
        pressure = data["final_pressure"].astype(np.float32)
        saturation = data["final_saturation"].astype(np.float32)

        self.perm_mean, self.perm_std = perm[train_idx].mean(), perm[train_idx].std()
        self.poro_mean, self.poro_std = poro[train_idx].mean(), poro[train_idx].std()
        self.mask_mean, self.mask_std = mask[train_idx].mean(), mask[train_idx].std() + 1e-8
        self.p_mean, self.p_std = pressure[train_idx].mean(), pressure[train_idx].std()
        self.s_mean, self.s_std = saturation[train_idx].mean(), saturation[train_idx].std()

        perm_n = (perm[idx] - self.perm_mean) / self.perm_std
        poro_n = (poro[idx] - self.poro_mean) / self.poro_std
        mask_n = (mask[idx] - self.mask_mean) / self.mask_std
        pressure_n = (pressure[idx] - self.p_mean) / self.p_std
        saturation_n = (saturation[idx] - self.s_mean) / self.s_std

        self.x = np.stack([perm_n, poro_n, mask_n], axis=1)
        self.y = np.stack([pressure_n, saturation_n], axis=1)
        self.perm_raw = perm[idx]
        self.mask_raw = mask[idx]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return (torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i]),
                torch.from_numpy(self.perm_raw[i]), torch.from_numpy(self.mask_raw[i]))


def make_loader(subset, shuffle):
    ds = SplitDatasetV3(DATASET_PATH, SPLIT_PATH, subset)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle), ds


def train_plain_fno(device):
    train_loader, _ = make_loader("train", shuffle=True)
    model = FNO(n_modes=(16, 16), hidden_channels=32, in_channels=3, out_channels=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for x, y, perm_raw, mask_raw in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
        scheduler.step()
    return model, time.time() - t0


def train_unet(device):
    train_loader, _ = make_loader("train", shuffle=True)
    model = UNetSurrogate(in_channels=3, out_channels=2, base_channels=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for x, y, perm_raw, mask_raw in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
        scheduler.step()
    return model, time.time() - t0


def train_physics_informed_fno_v3(device, physics_weight):
    train_loader, _ = make_loader("train", shuffle=True)
    model = FNO(n_modes=(16, 16), hidden_channels=32, in_channels=3, out_channels=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for x, y, perm_raw, mask_raw in train_loader:
            x, y = x.to(device), y.to(device)
            perm_raw, mask_raw = perm_raw.to(device), mask_raw.to(device)
            opt.zero_grad()
            pred = model(x)
            data_loss = F.mse_loss(pred, y)
            if physics_weight == 0.0:
                # same explicit short-circuit as fno_physics_informed.py's
                # combined_loss, for the same reason: guarantee byte-for-
                # byte identical behavior to plain_fno at weight=0, not
                # rely on 0 * phys_loss cooperating numerically
                total_loss = data_loss
            else:
                # physics residual is defined on the PRESSURE channel
                # (index 0) only -- pred is normalized, but the residual
                # equation is linear in p, so penalizing the residual of
                # the normalized field is an equally valid (rescaled)
                # penalty; kx=ky=perm_raw since solver_v3 is isotropic
                phys_loss = physics_residual_loss(pred[:, 0:1], perm_raw, perm_raw, mask_raw)
                total_loss = data_loss + physics_weight * phys_loss
            total_loss.backward()
            opt.step()
        scheduler.step()
    return model, time.time() - t0


ARCHITECTURES = {"plain_fno": train_plain_fno, "unet_baseline": train_unet}
for w in PHYSICS_WEIGHTS:
    ARCHITECTURES[f"physics_informed_fno_w{w}"] = (
        lambda device, w=w: train_physics_informed_fno_v3(device, physics_weight=w)
    )


def run_full_comparison():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader, _ = make_loader("test", shuffle=False)

    class EvalWrapper:
        def __init__(self, loader):
            self.loader = loader

        def __iter__(self):
            for x, y, perm_raw, mask_raw in self.loader:
                yield x, y

    results = {}
    for name, train_fn in ARCHITECTURES.items():
        print(f"\n=== Training {name} ===")
        model, train_time = train_fn(device)
        eval_results = full_evaluation(model, EvalWrapper(test_loader), device)
        eval_results["training_wall_clock_seconds"] = train_time
        results[name] = eval_results
        print(f"{name}: mean_rel_l2={eval_results['mean_rel_l2']:.4f}, "
              f"params={eval_results['n_parameters']:,}, "
              f"inference={eval_results['inference_ms_per_sample']:.2f}ms, "
              f"train_time={train_time:.0f}s")

    with open("surrogate_comparison_results_v3.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved surrogate_comparison_results_v3.json")

    print(f"\n{'='*76}\nSUMMARY TABLE (v3, two-phase: mean_rel_l2 is over BOTH channels stacked)\n{'='*76}")
    print(f"{'architecture':<32}{'mean_rel_l2':<14}{'params':<12}{'infer(ms)':<12}{'train(s)':<10}")
    for name, r in results.items():
        print(f"{name:<32}{r['mean_rel_l2']:<14.4f}{r['n_parameters']:<12,}"
              f"{r['inference_ms_per_sample']:<12.2f}{r['training_wall_clock_seconds']:<10.0f}")

    print("\nCheck first: physics_informed_fno_w0.0's mean_rel_l2 should match "
          "plain_fno's almost exactly (same required reduction property as v1's "
          "check in fno_physics_informed.py, now verified in the actual training "
          "loop, not just on random tensors). If it doesn't, something upstream "
          "of this script broke that invariant -- stop and check before trusting "
          "the rest of the table.")
    print("\nPer the spec's Step 5: write the verdict yourself, in this order:")
    print("  1. inference speed (Shapley/Blotto call volume is the real bottleneck)")
    print("  2. accuracy on the frame/output the payoff is computed from")
    print("  3. parameter count / retraining cost")
    print("  4. mean relative L2 -- only as a tie-breaker after 1-3")
    print("Note eval_utils' mean_rel_l2 here is computed over the flattened "
          "(pressure, saturation) pair together -- if you need the two channels' "
          "error broken apart, slice pred[:,0]/pred[:,1] vs y[:,0]/y[:,1] and "
          "call eval_utils.relative_l2_per_sample on each separately.")


if __name__ == "__main__":
    run_full_comparison()
