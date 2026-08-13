"""
Stage 5a: baseline CNN surrogate. Validates the pipeline (data -> model ->
prediction) before you invest in FNO. Run this locally where torch is
installed:

    pip install torch
    python baseline_cnn.py

Expects dataset_2000.npz (from data_gen.py) in the same directory.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ReservoirDataset(Dataset):
    """
    Input: 2-channel image (permeability, well_mask)
    Target: 1-channel final pressure field
    """
    def __init__(self, npz_path):
        data = np.load(npz_path)
        perm = data["permeability"].astype(np.float32)
        mask = data["well_mask"].astype(np.float32)
        pressure = data["final_pressure"].astype(np.float32)

        # normalize inputs/targets (store stats if you need to invert later)
        self.perm_mean, self.perm_std = perm.mean(), perm.std()
        self.p_mean, self.p_std = pressure.mean(), pressure.std()

        perm = (perm - self.perm_mean) / self.perm_std
        pressure = (pressure - self.p_mean) / self.p_std

        self.x = np.stack([perm, mask], axis=1)  # (N, 2, nx, ny)
        self.y = pressure[:, None, :, :]          # (N, 1, nx, ny)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])


class BaselineCNN(nn.Module):
    """Plain conv net, no downsampling (keeps spatial resolution fixed)."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def relative_l2_error(pred, target, eps=1e-8):
    """Standard metric in the FNO/operator-learning literature."""
    num = torch.norm(pred - target, dim=(-2, -1))
    den = torch.norm(target, dim=(-2, -1)) + eps
    return (num / den).mean().item()


def train(npz_path="dataset_2000.npz", epochs=50, batch_size=32, lr=1e-3, val_frac=0.15):
    full_ds = ReservoirDataset(npz_path)
    n_val = int(len(full_ds) * val_frac)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BaselineCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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

        model.eval()
        val_l2 = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_l2 += relative_l2_error(pred, y) * x.size(0)
        val_l2 /= n_val

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}  train_mse={train_loss:.4f}  val_rel_l2={val_l2:.4f}")

    torch.save(model.state_dict(), "baseline_cnn.pt")
    print("saved baseline_cnn.pt")
    return model


if __name__ == "__main__":
    train()
