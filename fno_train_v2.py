"""
fno_train_v2.py -- trains the FNO on the richer 4-channel physics
(kx, ky, porosity, well_mask -> final_pressure), instead of the original
2-channel (permeability, well_mask) version.

Run:
    pip install torch neuraloperator
    python fno_train_v2.py

Expects dataset_v2_2000.npz (from data_gen_v2.py) in the same directory.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from neuralop.models import FNO


class ReservoirDatasetV2(Dataset):
    """4 input channels instead of the original 2."""
    def __init__(self, npz_path):
        data = np.load(npz_path)
        kx = data["kx"].astype(np.float32)
        ky = data["ky"].astype(np.float32)
        porosity = data["porosity"].astype(np.float32)
        mask = data["well_mask"].astype(np.float32)
        pressure = data["final_pressure"].astype(np.float32)

        self.kx_mean, self.kx_std = kx.mean(), kx.std()
        self.ky_mean, self.ky_std = ky.mean(), ky.std()
        self.phi_mean, self.phi_std = porosity.mean(), porosity.std()
        self.p_mean, self.p_std = pressure.mean(), pressure.std()

        kx_n = (kx - self.kx_mean) / self.kx_std
        ky_n = (ky - self.ky_mean) / self.ky_std
        phi_n = (porosity - self.phi_mean) / self.phi_std
        pressure_n = (pressure - self.p_mean) / self.p_std

        self.x = np.stack([kx_n, ky_n, phi_n, mask], axis=1)  # (N, 4, nx, ny)
        self.y = pressure_n[:, None, :, :]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])


def relative_l2_error(pred, target, eps=1e-8):
    num = torch.norm(pred - target, dim=(-2, -1))
    den = torch.norm(target, dim=(-2, -1)) + eps
    return (num / den).mean().item()


def train_v2(npz_path="dataset_v2_2000.npz", epochs=100, batch_size=32, lr=1e-3, val_frac=0.15):
    full_ds = ReservoirDatasetV2(npz_path)
    n_val = int(len(full_ds) * val_frac)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # in_channels=4 now (kx, ky, porosity, well_mask), out_channels=1 (pressure)
    model = FNO(n_modes=(16, 16), hidden_channels=32, in_channels=4, out_channels=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= n_train
        scheduler.step()

        model.eval()
        val_l2 = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_l2 += relative_l2_error(pred, y) * x.size(0)
        val_l2 /= n_val

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}  train_mse={train_loss:.4f}  val_rel_l2={val_l2:.4f}")

    torch.save(model.state_dict(), "fno_surrogate_v2.pt")
    print("saved fno_surrogate_v2.pt")
    return model


if __name__ == "__main__":
    train_v2()
