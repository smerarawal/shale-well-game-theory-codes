"""
Stage 5b: FNO surrogate, using the `neuraloperator` library (Zongyi Li
group) instead of reimplementing FNO layers.

Setup:
    pip install torch neuraloperator

Run only after baseline_cnn.py gives a sane val_rel_l2 (roughly < 0.2-0.3
is a reasonable bar for "the pipeline works" -- FNO should beat it).
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from neuralop.models import FNO

from baseline_cnn import ReservoirDataset, relative_l2_error


def train_fno(npz_path="dataset_2000.npz", epochs=100, batch_size=32, lr=1e-3, val_frac=0.15):
    full_ds = ReservoirDataset(npz_path)
    n_val = int(len(full_ds) * val_frac)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # in_channels=2 (permeability, well_mask), out_channels=1 (pressure)
    model = FNO(
        n_modes=(16, 16),      # frequency modes kept per spatial dim; tune vs grid size
        hidden_channels=32,
        in_channels=2,
        out_channels=1,
    ).to(device)

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

    torch.save(model.state_dict(), "fno_surrogate.pt")
    print("saved fno_surrogate.pt")
    return model


if __name__ == "__main__":
    train_fno()
