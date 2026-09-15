"""
run_surrogate_comparison.py -- Step 4 of surrogate_comparison_spec.md.

Trains and evaluates every Priority 1-3 architecture (plain FNO, U-Net
baseline, physics-informed FNO at weight=0.1) on the IDENTICAL
fixed_split.npz, with the SAME batch size/optimizer/schedule/epoch count,
and reports every metric from eval_utils.py in one table.

Per the spec's own priority list, U-FNO (2.4) is deliberately NOT included
here -- only worth building once v3 multi-phase data with a saturation
channel exists.

Run:
    pip install torch neuraloperator
    python comparison_protocol.py dataset_2000.npz   # once, if not already done
    python run_surrogate_comparison.py

Expects dataset_2000.npz and fixed_split.npz in the same directory.
"""
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from neuralop.models import FNO

from unet_baseline import UNetSurrogate
from fno_physics_informed import combined_loss
from eval_utils import full_evaluation, count_parameters

DATASET_PATH = "dataset_2000.npz"
SPLIT_PATH = "fixed_split.npz"
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3


class SplitDataset(Dataset):
    """Loads dataset_2000.npz, applies the FIXED split's indices -- never
    regenerates its own split. Norm stats computed from TRAIN indices only
    (standard practice, not explicit in the spec's text -- normalizing
    using test-set statistics would leak information; flagged since it's
    an addition beyond the spec's literal wording)."""

    def __init__(self, dataset_path, split_path, subset):
        data = np.load(dataset_path)
        split = np.load(split_path)
        idx = split[f"{subset}_idx"]
        train_idx = split["train_idx"]

        perm = data["permeability"].astype(np.float32)
        mask = data["well_mask"].astype(np.float32)
        pressure = data["final_pressure"].astype(np.float32)

        self.perm_mean, self.perm_std = perm[train_idx].mean(), perm[train_idx].std()
        self.p_mean, self.p_std = pressure[train_idx].mean(), pressure[train_idx].std()

        perm_n = (perm[idx] - self.perm_mean) / self.perm_std
        pressure_n = (pressure[idx] - self.p_mean) / self.p_std
        mask_sub = mask[idx]

        self.x = np.stack([perm_n, mask_sub], axis=1)
        self.y = pressure_n[:, None, :, :]
        self.kx_raw = perm[idx]
        self.mask_raw = mask_sub

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return (torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i]),
                torch.from_numpy(self.kx_raw[i]), torch.from_numpy(self.mask_raw[i]))


def make_loader(subset, shuffle):
    ds = SplitDataset(DATASET_PATH, SPLIT_PATH, subset)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle), ds


def train_plain_fno(device):
    train_loader, train_ds = make_loader("train", shuffle=True)

    model = FNO(n_modes=(16, 16), hidden_channels=32, in_channels=2, out_channels=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for x, y, kx_raw, mask_raw in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            opt.step()
        scheduler.step()
    train_time = time.time() - t0
    return model, train_time


def train_unet(device):
    train_loader, train_ds = make_loader("train", shuffle=True)

    model = UNetSurrogate(in_channels=2, out_channels=1, base_channels=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for x, y, kx_raw, mask_raw in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            opt.step()
        scheduler.step()
    train_time = time.time() - t0
    return model, train_time


def train_physics_informed_fno(device, physics_weight=0.1):
    train_loader, train_ds = make_loader("train", shuffle=True)

    model = FNO(n_modes=(16, 16), hidden_channels=32, in_channels=2, out_channels=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for x, y, kx_raw, mask_raw in train_loader:
            x, y = x.to(device), y.to(device)
            kx_raw, mask_raw = kx_raw.to(device), mask_raw.to(device)
            opt.zero_grad()
            pred = model(x)
            # physics residual needs kx, ky separately -- this dataset
            # (v1, isotropic) only has one permeability field, so kx=ky
            total_loss, data_loss, phys_loss = combined_loss(
                pred, y, kx_raw, kx_raw, mask_raw, physics_weight=physics_weight
            )
            total_loss.backward()
            opt.step()
        scheduler.step()
    train_time = time.time() - t0
    return model, train_time


ARCHITECTURES = {
    "plain_fno": train_plain_fno,
    "unet_baseline": train_unet,
    "physics_informed_fno_w0.1": lambda device: train_physics_informed_fno(device, physics_weight=0.1),
}


def run_full_comparison():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader, _ = make_loader("test", shuffle=False)

    class EvalWrapper:
        """Wraps the 4-tuple training loader down to (x, y) pairs, so
        eval_utils.full_evaluation stays architecture-agnostic -- it
        doesn't need to know which architectures used kx/mask during
        training."""
        def __init__(self, loader):
            self.loader = loader

        def __iter__(self):
            for x, y, kx_raw, mask_raw in self.loader:
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

    with open("surrogate_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved surrogate_comparison_results.json")

    print(f"\n{'='*70}\nSUMMARY TABLE\n{'='*70}")
    print(f"{'architecture':<28}{'mean_rel_l2':<14}{'params':<12}{'infer(ms)':<12}{'train(s)':<10}")
    for name, r in results.items():
        print(f"{name:<28}{r['mean_rel_l2']:<14.4f}{r['n_parameters']:<12,}"
              f"{r['inference_ms_per_sample']:<12.2f}{r['training_wall_clock_seconds']:<10.0f}")

    print("\nPER THE SPEC'S STEP 5: do not stop at this table. Write the verdict")
    print("yourself, weighing in this order for YOUR project's actual usage:")
    print("  1. inference speed (Shapley/Blotto call volume is the real bottleneck)")
    print("  2. accuracy on the frame/output that the payoff is computed from")
    print("  3. parameter count / retraining cost")
    print("  4. mean relative L2 -- only as a tie-breaker after 1-3")


if __name__ == "__main__":
    run_full_comparison()
