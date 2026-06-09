# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ğŸ“¦ Setup
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pyarrow.parquet as pq

# ğŸ§  Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Load ADC info
adc_info = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/adc_info.csv").iloc[0]
adc_gain = adc_info['FGS1_adc_gain']
adc_offset = adc_info['FGS1_adc_offset']

# Custom Dataset Class
class FGS1Dataset(Dataset):
    def __init__(self, root_dir, planet_ids, targets_df):
        self.root_dir = root_dir
        self.planet_ids = planet_ids
        self.targets_df = targets_df

    def __len__(self):
        return len(self.planet_ids)

    def __getitem__(self, idx):
        planet_id = self.planet_ids[idx]
        signal_path = f"{self.root_dir}/{planet_id}/FGS1_signal_0.parquet"
        signal = pq.read_table(signal_path).to_pandas().values.astype('float64')
        signal = signal * adc_gain + adc_offset
        signal = signal.reshape(-1, 32, 32)  # (135000, 32, 32)
        signal = signal[:1000]  # Optional: truncate for speed
        signal = torch.tensor(signal).float()
        signal = signal.unsqueeze(1)  # Add channel dim -> (T, 1, 32, 32)

        # Target
        target_row = self.targets_df[self.targets_df.planet_id == planet_id]
        y = torch.tensor(target_row.drop(columns="planet_id").values[0]).float()  # 283 values


        return signal, y


class CNNRegressor(nn.Module):
    def __init__(self, out_dim=283):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 512), nn.ReLU(),
            nn.Linear(512, out_dim * 2)  # mean + log_std
        )

    def forward(self, x):  # x: (batch, T, 1, 32, 32)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)
        feats = feats.view(B, T, -1).mean(dim=1)  # Mean over time
        out = self.fc(feats)
        mean, log_std = out.chunk(2, dim=1)
        return mean, log_std


# Load targets
targets = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/train.csv")
planet_ids = targets['planet_id'].tolist()
train_ids, val_ids = train_test_split(planet_ids, test_size=0.1, random_state=42)

train_ds = FGS1Dataset("/kaggle/input/ariel-data-challenge-2025/train", train_ids, targets)
val_ds = FGS1Dataset("/kaggle/input/ariel-data-challenge-2025/train", val_ids, targets)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=2)



# assert torch.isfinite(X).all(), "X contains non-finite values!"
# assert torch.isfinite(y).all(), "y contains non-finite values!"


from tqdm.notebook import tqdm
import torch

model = CNNRegressor(out_dim=283).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def gll_loss(y_true, y_pred_mean, y_pred_log_std):
    log_std = torch.clamp(y_pred_log_std, min=-5, max=5)
    std = torch.exp(log_std)
    loss = log_std + ((y_true - y_pred_mean) ** 2) / (2 * std ** 2)
    return loss.mean()

EPOCHS = 20
best_val_loss = float('inf')
patience, wait = 5, 0

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        if not torch.isfinite(X).all() or not torch.isfinite(y).all():
            continue

        pred_mean, pred_log_std = model(X)
        loss = gll_loss(y, pred_mean, pred_log_std)
        if not torch.isfinite(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"âœ… Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.6f}")

    # ğŸ”� Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred_mean, pred_log_std = model(X)
            loss = gll_loss(y, pred_mean, pred_log_std)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"ğŸ”� Epoch {epoch+1} - Avg Val Loss: {avg_val_loss:.6f}")

    # â�¹ï¸� Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"â�¹ï¸� Early stopping at epoch {epoch+1}")
            break


import matplotlib.pyplot as plt

# ğŸ“Š Visualize Predictions
model.eval()
for i, (X, y_true) in enumerate(val_loader):
    X, y_true = X.to(device), y_true.to(device)
    with torch.no_grad():
        pred_mean, pred_log_std = model(X)

    # Detach tensors before converting to numpy
    mean = pred_mean[0].detach().cpu().numpy()
    std = torch.exp(pred_log_std[0]).detach().cpu().numpy()
    true = y_true[0].detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(true, label='True', linewidth=2)
    plt.plot(mean, label='Predicted', linestyle='--')
    plt.fill_between(
        range(283),
        mean - std,
        mean + std,
        alpha=0.3, label='Uncertainty'
    )
    plt.legend()
    plt.title(f"Validation Spectrum #{i}")
    plt.xlabel("Wavelength Bin")
    plt.ylabel("Flux")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if i == 2:
        break  # Show 3 examples



torch.save(model.state_dict(), 'best_model.pth')
print("âœ… Model saved!")


from sklearn.metrics import mean_absolute_error
from tqdm import tqdm  # âœ… make sure tqdm is imported

model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for X, y in tqdm(val_loader, desc="ğŸ”� Evaluating"):
        X, y = X.to(device), y.to(device)
        pred_mean, _ = model(X)
        all_preds.append(pred_mean.cpu())
        all_targets.append(y.cpu())

all_preds = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()

mae = mean_absolute_error(all_targets, all_preds)
print(f"ğŸ“� Validation MAE: {mae:.4f}")


import os
import torch
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm

# Constants
TEST_DIR = "/kaggle/input/ariel-data-challenge-2025/test"
adc_info = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/adc_info.csv").iloc[0]
adc_gain = adc_info['FGS1_adc_gain']
adc_offset = adc_info['FGS1_adc_offset']

model.eval()
submission_rows = []

with torch.no_grad():
    for planet_id in tqdm(os.listdir(TEST_DIR), desc="ğŸš€ Predicting"):
        obj_path = os.path.join(TEST_DIR, planet_id)
        if not os.path.isdir(obj_path):
            continue

        try:
            # Load signal file
            signal_path = os.path.join(obj_path, "FGS1_signal_0.parquet")
            signal = pq.read_table(signal_path).to_pandas().values.astype('float64')
            signal = signal * adc_gain + adc_offset
            signal = signal.reshape(-1, 32, 32)
            signal = signal[:1000]  # truncate for consistency
            signal = torch.tensor(signal).float().unsqueeze(1).unsqueeze(0).to(device)  # (1, T, 1, 32, 32)

            # Model prediction
            pred_mean, pred_log_std = model(signal)
            pred_mean = pred_mean.squeeze().cpu().numpy()
            pred_std = torch.exp(pred_log_std).squeeze().cpu().numpy()

            # Row: [planet_id, mu_0...mu_282, sigma_0...sigma_282]
            row = [planet_id] + pred_mean.tolist() + pred_std.tolist()
            submission_rows.append(row)

        except Exception as e:
            print(f"â�Œ Failed for {planet_id}: {e}")
            continue

# Prepare header
mu_cols = [f"mu_{i}" for i in range(283)]
sigma_cols = [f"sigma_{i}" for i in range(283)]
columns = ["planet_id"] + mu_cols + sigma_cols

# Save submission
submission_df = pd.DataFrame(submission_rows, columns=columns)
submission_df.to_csv("submission.csv", index=False)
print("âœ… submission.csv created successfully in required format!")



submission_df.head()

