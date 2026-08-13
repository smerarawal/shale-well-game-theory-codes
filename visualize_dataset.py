"""
Visualize samples from dataset_2000.npz locally.

    pip install numpy matplotlib
    python visualize_dataset.py

Produces dataset_preview.png in the same folder: rows = permeability,
well_mask, final_pressure; columns = different samples. Also prints
basic dataset stats so you have numbers to quote, not just a picture.

Edit SAMPLE_INDICES below to look at specific samples (e.g. ones with
many wells clustered together, to show interference).
"""
import numpy as np
import matplotlib.pyplot as plt

NPZ_PATH = "dataset_2000.npz"
SAMPLE_INDICES = [0, 1, 2]   # change these to look at other samples
N_TO_SHOW = len(SAMPLE_INDICES)


def main():
    d = np.load(NPZ_PATH)
    perm = d["permeability"]
    mask = d["well_mask"]
    pressure = d["final_pressure"]

    print(f"dataset: {perm.shape[0]} samples, grid {perm.shape[1]}x{perm.shape[2]}")
    print(f"permeability   min={perm.min():.4f} max={perm.max():.4f} mean={perm.mean():.4f}")
    print(f"final_pressure min={pressure.min():.4f} max={pressure.max():.4f} mean={pressure.mean():.4f}")
    well_counts = (mask != 0).sum(axis=(1, 2))
    print(f"wells per sample: min={well_counts.min()} max={well_counts.max()} mean={well_counts.mean():.1f}")

    fig, axes = plt.subplots(3, N_TO_SHOW, figsize=(4 * N_TO_SHOW, 12))
    row_specs = [
        ("permeability", perm, "viridis"),
        ("well_mask", mask, "coolwarm"),
        ("final_pressure", pressure, "RdBu_r"),
    ]
    for row, (name, arr, cmap) in enumerate(row_specs):
        for col, idx in enumerate(SAMPLE_INDICES):
            ax = axes[row, col]
            im = ax.imshow(arr[idx].T, origin="lower", cmap=cmap)
            ax.set_title(f"{name} — sample {idx}")
            plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig("dataset_preview.png", dpi=130)
    print("saved dataset_preview.png")


if __name__ == "__main__":
    main()
