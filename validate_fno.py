"""
Stage 0 validation: compare the trained FNO surrogate against the FD
solver (ground truth) on FRESH cases it never saw during training, and
check inference speed for repeated batch queries.

Run in the same Colab session (or reload the checkpoint later):
    python validate_fno.py
"""
import time
import numpy as np
import torch
from neuralop.models import FNO

from solver import solve_pressure_diffusion
from data_gen import random_permeability_field, sample_well_configuration
from baseline_cnn import relative_l2_error

NPZ_PATH = "dataset_2000.npz"   # only used to recover normalization stats
CKPT_PATH = "fno_surrogate.pt"
N_TEST = 100
NX, NY = 32, 32


def load_norm_stats(npz_path):
    d = np.load(npz_path)
    perm = d["permeability"].astype(np.float32)
    pressure = d["final_pressure"].astype(np.float32)
    return perm.mean(), perm.std(), pressure.mean(), pressure.std()


def main():
    perm_mean, perm_std, p_mean, p_std = load_norm_stats(NPZ_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FNO(n_modes=(16, 16), hidden_channels=32, in_channels=2, out_channels=1).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    # generate N_TEST fresh cases (different seed region than training)
    np.random.seed(999)
    rel_l2_errors = []
    infer_times = []

    for i in range(N_TEST):
        perm = random_permeability_field(NX, NY)
        locs, rates = sample_well_configuration(NX, NY, max_wells=8)

        # ground truth via FD solver
        history, dt = solve_pressure_diffusion(perm, locs, rates, nt=150)
        true_pressure = history[-1]

        mask = np.zeros((NX, NY), dtype=np.float32)
        for (wi, wj), r in zip(locs, rates):
            mask[wi, wj] = r

        x = np.stack([(perm - perm_mean) / perm_std, mask], axis=0)[None, ...]
        x_t = torch.from_numpy(x.astype(np.float32)).to(device)

        t0 = time.time()
        with torch.no_grad():
            pred_norm = model(x_t)
        infer_times.append(time.time() - t0)

        pred = pred_norm.cpu().numpy()[0, 0] * p_std + p_mean

        num = np.linalg.norm(pred - true_pressure)
        den = np.linalg.norm(true_pressure) + 1e-8
        rel_l2_errors.append(num / den)

    rel_l2_errors = np.array(rel_l2_errors)
    infer_times = np.array(infer_times)

    print(f"held-out cases: {N_TEST}")
    print(f"relative L2 error: mean={rel_l2_errors.mean():.4f}  "
          f"median={np.median(rel_l2_errors):.4f}  "
          f"max={rel_l2_errors.max():.4f}")
    print(f"single-sample inference: mean={infer_times.mean()*1000:.2f}ms  "
          f"(device={device})")

    # batch inference speed (what an optimization loop will actually do)
    batch = 64
    perms_batch = np.stack([random_permeability_field(NX, NY) for _ in range(batch)])
    masks_batch = np.zeros((batch, NX, NY), dtype=np.float32)
    for b in range(batch):
        locs, rates = sample_well_configuration(NX, NY, max_wells=8)
        for (wi, wj), r in zip(locs, rates):
            masks_batch[b, wi, wj] = r
    x_batch = np.stack([
        (perms_batch - perm_mean) / perm_std,
        masks_batch
    ], axis=1).astype(np.float32)
    x_batch_t = torch.from_numpy(x_batch).to(device)

    t0 = time.time()
    with torch.no_grad():
        _ = model(x_batch_t)
    batch_time = time.time() - t0
    print(f"batch inference: {batch} samples in {batch_time*1000:.1f}ms "
          f"({batch_time/batch*1000:.2f}ms/sample amortized)")

    if rel_l2_errors.mean() < 0.15:
        print("PASS: FNO is accurate enough to use as the surrogate going forward")
    elif rel_l2_errors.mean() < 0.3:
        print("MARGINAL: usable but consider more training data/epochs before trusting it in the game-theory loop")
    else:
        print("FAIL: error too high, do not use for the optimization stage yet")


if __name__ == "__main__":
    main()
