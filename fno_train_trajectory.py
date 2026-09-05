"""
fno_train_trajectory.py

Trains an FNO to predict the FULL pressure trajectory (T, H, W) from static
inputs (kx, ky, porosity, well_mask), instead of just the final state.

Architecture choice: uses neuraloperator's FNO3d, treating time as the third
spatial-like dimension. Static inputs (kx, ky, porosity, well_mask) are
broadcast/repeated across the time axis and concatenated as extra input
channels at every timestep — a standard way to feed time-invariant fields
into a 3D FNO. Output has 1 channel (pressure) x T timesteps.

Once trained, `compute_discounted_eur()` turns a predicted trajectory into
the real discounted-EUR payoff instead of the single-snapshot proxy used in
optimize_wells.py / shapley_values.py so far.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from neuralop.models import FNO
    HAVE_NEURALOP = True
except ImportError:
    HAVE_NEURALOP = False
    print("WARNING: neuraloperator not importable in this environment — "
          "training loop below assumes it's available (as it is in your "
          "repo's environment, per your v2 training logs).")


class TrajectoryDataset(Dataset):
    """Wraps dataset_v2_trajectory.npz for spatio-temporal FNO training."""

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.kx = data["kx"]                     # (N, H, W)
        self.ky = data["ky"]                      # (N, H, W)
        self.porosity = data["porosity"]          # (N, H, W)
        self.well_mask = data["well_mask"]        # (N, H, W)
        self.pressure_traj = data["pressure_trajectory"]  # (N, T, H, W)
        self.n_samples, self.T, self.H, self.W = self.pressure_traj.shape

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Broadcast the 4 static fields across T timesteps, stack as channels.
        static = np.stack(
            [self.kx[idx], self.ky[idx], self.porosity[idx], self.well_mask[idx]],
            axis=0,
        )  # (4, H, W)
        static_t = np.repeat(static[:, None, :, :], self.T, axis=1)  # (4, T, H, W)

        x = torch.from_numpy(static_t).float()          # (4, T, H, W) input
        y = torch.from_numpy(
            self.pressure_traj[idx][None, :, :, :]
        ).float()                                        # (1, T, H, W) target
        return x, y


def relative_l2_error(pred, target):
    num = torch.norm(pred - target, dim=(-3, -2, -1))
    den = torch.norm(target, dim=(-3, -2, -1)) + 1e-8
    return (num / den).mean()


def train(npz_path="dataset_v2_trajectory.npz", n_epochs=100, batch_size=16,
          lr=1e-3, out_path="fno_surrogate_trajectory.pt"):
    dataset = TrajectoryDataset(npz_path)
    n_val = max(1, int(0.05 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FNO(
        n_modes=(8, 12, 12),        # (time, H, W) Fourier modes — start modest on time
        in_channels=4,
        out_channels=1,
        hidden_channels=32,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    loss_fn = nn.MSELoss()

    for epoch in range(n_epochs):
        model.train()
        train_mse = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_mse += loss.item() * x.size(0)
        train_mse /= n_train
        scheduler.step()

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            model.eval()
            val_rel_l2 = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    val_rel_l2 += relative_l2_error(pred, y).item() * x.size(0)
            val_rel_l2 /= n_val
            print(f"epoch {epoch} train_mse={train_mse:.4f} val_rel_l2={val_rel_l2:.4f}")

    torch.save(model.state_dict(), out_path)
    print(f"saved {out_path}")
    return model


def compute_discounted_eur(pressure_trajectory, well_mask, discount_rate=0.08,
                            dt_years=1.0 / 12, rate_scale=1.0):
    """
    Converts a predicted pressure TRAJECTORY into the real discounted-EUR
    payoff, replacing the single-snapshot proxy used everywhere in v1/v2.

    pressure_trajectory: (T, H, W) array/tensor of predicted pressure fields
    well_mask: (H, W) array marking well locations
    discount_rate: annual discount rate (e.g. 0.08 = 8%)
    dt_years: time between saved trajectory steps, in years
    rate_scale: converts local pressure-drawdown-per-step into a production
        rate proxy; calibrate against your solver's known well_rate units.

    Returns: scalar discounted EUR (estimated ultimate recovery), summed
    across all wells marked in well_mask.
    """
    if isinstance(pressure_trajectory, torch.Tensor):
        pressure_trajectory = pressure_trajectory.detach().cpu().numpy()
    T = pressure_trajectory.shape[0]
    well_ys, well_xs = np.nonzero(well_mask)

    eur = 0.0
    for t in range(1, T):
        # local production rate proxy: -dp/dt at each well cell (pressure
        # depletion rate). Replace with your solver's actual rate-from-
        # pressure relationship if you have one (e.g. via well index / PI).
        dp = pressure_trajectory[t - 1, well_ys, well_xs] - pressure_trajectory[t, well_ys, well_xs]
        rate_t = np.clip(dp, 0, None) * rate_scale
        discount_factor = 1.0 / (1.0 + discount_rate) ** (t * dt_years)
        eur += rate_t.sum() * discount_factor * dt_years

    return float(eur)


if __name__ == "__main__":
    model = train()
