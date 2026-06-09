# Cell 1: imports & basic configuration

import os
import gc
import random
import math

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Use GPU if available (you already selected GPU T4 x2 in session options)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)



# Cell 2: load M5 datasets

DATA_DIR = "/kaggle/input/m5-forecasting-accuracy"

sales = pd.read_csv(os.path.join(DATA_DIR, "sales_train_evaluation.csv"))
calendar = pd.read_csv(os.path.join(DATA_DIR, "calendar.csv"))
sell_prices = pd.read_csv(os.path.join(DATA_DIR, "sell_prices.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

print("sales shape:", sales.shape)
print("calendar shape:", calendar.shape)
print("sell_prices shape:", sell_prices.shape)
print("sample_submission shape:", sample_sub.shape)

sales.head()



# Cell 3: build numpy matrix of demand time series + per-series scaling

# Keep only day columns: d_1 ... d_1941
day_cols = [c for c in sales.columns if c.startswith("d_")]
day_cols = sorted(day_cols, key=lambda x: int(x.split("_")[1]))  # sort by day index

values = sales[day_cols].values.astype("float32")  # shape: [n_series, n_days]
n_series, n_days = values.shape
print("n_series:", n_series, "n_days:", n_days)

# Official M5 validation setting
LAST_TRAIN_DAY = 1913   # we use d_1 ... d_1913
HORIZON = 28            # forecast horizon (28 days)
HISTORY = 90            # input history length for the model

print("HISTORY:", HISTORY, "HORIZON:", HORIZON, "LAST_TRAIN_DAY:", LAST_TRAIN_DAY)

# ---------- NEW: per-series scaling ----------
# Compute mean demand for each series over the training region
train_region = values[:, :LAST_TRAIN_DAY]          # columns 0 ... 1912
series_means = train_region.mean(axis=1)           # shape: [n_series]

# Define scale as mean, but avoid very small values
series_scales = series_means.copy()
series_scales[series_scales < 1.0] = 1.0           # floor at 1 to avoid exploding values

# Scaled time series matrix
values_scaled = values / series_scales[:, None]    # broadcasting over time dimension

print("values_scaled shape:", values_scaled.shape)



# Cell 4: Dataset that samples random sliding windows from all series (scaled data)

class M5NBeatsDataset(Dataset):
    """
    Dataset for global N-BEATS training on M5.
    It samples random (history, horizon) windows from all time series.
    """
    def __init__(
        self,
        all_series: np.ndarray,
        history: int,
        horizon: int,
        last_train_day: int,
        samples_per_series: int = 20,
    ):
        """
        all_series: array of shape [n_series, n_days] (already scaled)
        history: input length
        horizon: forecast length
        last_train_day: number of days we allow for training (e.g. 1913)
        samples_per_series: how many random windows per series
        """
        self.all_series = all_series
        self.history = history
        self.horizon = horizon
        self.last_train_day = last_train_day
        self.samples = []

        n_series, n_days = all_series.shape

        # NOTE: this matches your previous version (slight +1, but we keep it
        # to stay comparable with your old 60-epoch runs).
        max_start = last_train_day - history - horizon + 1
        max_start = max(max_start, 0)

        for s in range(n_series):
            for _ in range(samples_per_series):
                if max_start <= 0:
                    start = 0
                else:
                    start = np.random.randint(0, max_start + 1)
                self.samples.append((s, start))

        print("Total training windows:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s, start = self.samples[idx]
        x = self.all_series[s, start : start + self.history]
        y = self.all_series[s, start + self.history : start + self.history + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


train_dataset = M5NBeatsDataset(
    all_series=values_scaled,      # <-- use scaled matrix
    history=HISTORY,
    horizon=HORIZON,
    last_train_day=LAST_TRAIN_DAY,
    samples_per_series=10,         # moderate for speed
)

BATCH_SIZE = 1024
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)

batch_x, batch_y = next(iter(train_loader))
print("Batch x shape:", batch_x.shape, "Batch y shape:", batch_y.shape)



# Cell 5: define a regularized N-BEATS model (with dropout)

class NBeatsBlock(nn.Module):
    """
    Single generic N-BEATS block:
    - Fully connected stack
    - Backcast and forecast projections
    """
    def __init__(self, input_size, theta_size, hidden_size=256, nb_hid_layers=4, dropout_p=0.1):
        super().__init__()
        layers = []
        in_features = input_size
        for _ in range(nb_hid_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            # NEW: dropout for regularization
            layers.append(nn.Dropout(p=dropout_p))
            in_features = hidden_size
        self.fc = nn.Sequential(*layers)
        self.backcast_linear = nn.Linear(hidden_size, theta_size)
        self.forecast_linear = nn.Linear(hidden_size, theta_size)

    def forward(self, x):
        # x: [B, input_size]
        x = self.fc(x)
        backcast = self.backcast_linear(x)
        forecast = self.forecast_linear(x)
        return backcast, forecast


class NBeats(nn.Module):
    """
    Very simplified N-BEATS:
    - Multiple generic stacks
    - Residual connection on backcast
    - Final linear head to produce the forecast horizon
    """
    def __init__(
        self,
        history,
        horizon,
        n_stacks=3,
        hidden_size=256,
        nb_hid_layers=4,
        dropout_p=0.1,
    ):
        super().__init__()
        self.history = history
        self.horizon = horizon

        self.stacks = nn.ModuleList()
        for _ in range(n_stacks):
            block = NBeatsBlock(
                input_size=history,
                theta_size=history,      # backcast length
                hidden_size=hidden_size,
                nb_hid_layers=nb_hid_layers,
                dropout_p=dropout_p,
            )
            self.stacks.append(block)

        # Final head that maps the residual to forecast horizon
        self.forecast_head = nn.Linear(history, horizon)

    def forward(self, x):
        """
        x: [B, history]
        returns forecast: [B, horizon]
        """
        residual = x
        for block in self.stacks:
            backcast, _ = block(residual)
            residual = residual - backcast

        forecast = self.forecast_head(residual)
        return forecast


model = NBeats(
    history=HISTORY,
    horizon=HORIZON,
    n_stacks=3,
    hidden_size=256,
    nb_hid_layers=4,
    dropout_p=0.1,   # small dropout
).to(device)

criterion = nn.MSELoss()

# NEW: add small weight_decay for extra regularization
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

print(model)



# Cell 6: training loop (60 epochs to match your best run)

EPOCHS = 60   # you can later tune this within 40–80 if you want


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        # model input shape: [B, history]
        output = model(batch_x)
        loss = criterion(output, batch_y)

        loss.backward()
        # Gradient clipping keeps training stable
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(loader.dataset)


for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    print(f"Epoch {epoch}/{EPOCHS} - train_loss: {train_loss:.6f}")



# Cell 7: generate forecasts for all series (validation horizon)
# NOTE: we feed scaled history into the model, then rescale predictions back.

model.eval()
all_forecasts = np.zeros((n_series, HORIZON), dtype="float32")

with torch.no_grad():
    for i in range(n_series):
        # Scale for this series
        scale_i = series_scales[i]

        # Use the last HISTORY days before LAST_TRAIN_DAY as input (scaled)
        end = LAST_TRAIN_DAY               # this is a "count of days", 1-based
        start = end - HISTORY              # we will slice [start:end] (0-based)
        if start < 0:
            # If not enough history at the beginning, left-pad with zeros
            series_hist = values_scaled[i, :end]
            pad_len = HISTORY - series_hist.shape[0]
            x = np.concatenate([np.zeros(pad_len, dtype="float32"), series_hist])
        else:
            x = values_scaled[i, start:end]

        x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)  # shape [1, HISTORY]
        forecast_scaled = model(x_tensor).cpu().numpy().reshape(-1)

        # Rescale back to original demand units
        forecast = forecast_scaled * scale_i

        # Negative demand does not make sense here, so clip at 0
        all_forecasts[i, :] = np.clip(forecast, 0.0, None)

all_forecasts.shape



# Cell 8: fill sample_submission and save csv

sub = sample_sub.copy()

# Masks for validation and evaluation parts
val_mask = sub["id"].str.endswith("_validation")
eval_mask = sub["id"].str.endswith("_evaluation")

f_cols = [f"F{i}" for i in range(1, HORIZON + 1)]

# For validation part, row order matches sales_train_evaluation
sub.loc[val_mask, f_cols] = all_forecasts

# For evaluation part, as a simple baseline we reuse the same forecasts
# (better practice: re-train including validation period and then forecast evaluation)
sub.loc[eval_mask, f_cols] = all_forecasts

sub.head()

sub.to_csv("submission.csv", index=False)
print("Saved")





