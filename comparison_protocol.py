"""
comparison_protocol.py -- Step 1 of surrogate_comparison_spec.md.

Run ONCE. Every architecture trained afterward MUST load this exact split
from disk, never regenerate it. This is the single most important step in
the whole comparison -- skipping it or regenerating the split differently
per architecture is the most common way this kind of comparison silently
becomes unfair.

Run:
    python comparison_protocol.py [dataset_path]
    (defaults to dataset_2000.npz if no path given)
"""
import numpy as np


def create_fixed_split(dataset_path, seed=42, train_frac=0.85, val_frac=0.10):
    data = np.load(dataset_path)
    if "kx" in data:
        n = len(data["kx"])
    elif "k" in data:
        n = len(data["k"])
    elif "permeability" in data:
        n = len(data["permeability"])
    else:
        raise KeyError(f"Could not find a permeability-like array in {dataset_path} "
                        f"to infer sample count. Available keys: {list(data.keys())}")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    split = {
        "train_idx": idx[:n_train],
        "val_idx": idx[n_train:n_train + n_val],
        "test_idx": idx[n_train + n_val:],  # held out -- NEVER touched until final comparison
    }
    np.savez("fixed_split.npz", **split)
    print(f"train={n_train}, val={n_val}, test={n - n_train - n_val}")
    print(f"saved fixed_split.npz (source dataset: {dataset_path}, seed={seed})")
    return split


if __name__ == "__main__":
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "dataset_2000.npz"
    create_fixed_split(dataset_path)
