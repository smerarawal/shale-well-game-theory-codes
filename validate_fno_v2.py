"""
validate_fno_v2.py -- validates the 4-channel FNO against solver_v2's
actual physics on fresh held-out cases, same structure as validate_fno.py.

Run:
    python validate_fno_v2.py

Expects dataset_v2_2000.npz and fno_surrogate_v2.pt in the same directory.
"""
import time
import numpy as np
import torch
from neuralop.models import FNO

from solver_v2 import random_anisotropic_permeability, random_porosity_field, solve_pressure_diffusion_v2
from data_gen_v2 import sample_well_configuration

NPZ_PATH = "dataset_v2_2000.npz"
CKPT_PATH = "fno_surrogate_v2.pt"
N_TEST = 100
NX, NY = 32, 32


def load_norm_stats_v2(npz_path):
    d = np.load(npz_path)
    kx = d["kx"].astype(np.float32); ky = d["ky"].astype(np.float32)
    phi = d["porosity"].astype(np.float32); pressure = d["final_pressure"].astype(np.float32)
    return (kx.mean(), kx.std(), ky.mean(), ky.std(),
            phi.mean(), phi.std(), pressure.mean(), pressure.std())


def main():
    stats = load_norm_stats_v2(NPZ_PATH)
    kx_mean, kx_std, ky_mean, ky_std, phi_mean, phi_std, p_mean, p_std = stats

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FNO(n_modes=(16, 16), hidden_channels=32, in_channels=4, out_channels=1).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=False))
    model.eval()

    np.random.seed(999)
    rel_l2_errors = []
    infer_times = []

    for i in range(N_TEST):
        kx, ky = random_anisotropic_permeability(NX, NY)
        phi = random_porosity_field(NX, NY)
        locs, rates = sample_well_configuration(NX, NY, max_wells=8)

        history, dt = solve_pressure_diffusion_v2(kx, ky, phi, locs, rates, nt=150)
        true_pressure = history[-1]

        mask = np.zeros((NX, NY), dtype=np.float32)
        for (wi, wj), r in zip(locs, rates):
            mask[wi, wj] = r

        x = np.stack([
            (kx - kx_mean) / kx_std, (ky - ky_mean) / ky_std,
            (phi - phi_mean) / phi_std, mask
        ], axis=0)[None, ...].astype(np.float32)
        x_t = torch.from_numpy(x).to(device)

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
          f"median={np.median(rel_l2_errors):.4f}  max={rel_l2_errors.max():.4f}")
    print(f"single-sample inference: mean={infer_times.mean()*1000:.2f}ms (device={device})")

    if rel_l2_errors.mean() < 0.15:
        print("PASS: v2 FNO (with porosity + anisotropy) is accurate enough to use going forward")
    elif rel_l2_errors.mean() < 0.3:
        print("MARGINAL: usable but consider more training data/epochs")
    else:
        print("FAIL: error too high -- the richer physics may need more training data "
              "than the original 2-channel version did, since the input space is bigger")


if __name__ == "__main__":
    main()
