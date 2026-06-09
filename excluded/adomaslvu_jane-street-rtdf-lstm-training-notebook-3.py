!pip install polars[gpu] torch --extra-index-url=https://pypi.nvidia.com

import glob
import gc
import polars as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 1) Loading the full training dataset (including what was previously val)
class LoadData:
    def __init__(self, file_paths, partition_ids=None):
        self.file_paths = file_paths
        self.partition_ids = partition_ids

    def load_and_concat(self):
        if self.partition_ids is not None:
            selected_files = [
                fp for fp in self.file_paths
                if any(f'partition_id={pid}' in fp for pid in self.partition_ids)
            ]
        else:
            selected_files = self.file_paths

        partitioned_data = [pl.scan_parquet(file_path) for file_path in selected_files]
        df = pl.concat(partitioned_data, rechunk=False)
        
        del partitioned_data
        gc.collect()
        
        return df

partition_ids = [5, 6, 7, 8, 9]
file_paths_all = sorted(glob.glob('/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/*/*.parquet'))
loader = LoadData(file_paths=file_paths_all, partition_ids=partition_ids)
df_selected = loader.load_and_concat()

# 2) Sort the entire dataset by date/time (no separate validation split)
df_selected = df_selected.sort(["date_id", "time_id"])
df_train = df_selected

# 3) Create lag columns for responders
all_cols = df_train.collect_schema().names()
target_cols = [col for col in all_cols if 'responder' in col]

train_lags = (
    df_train.select(
        ['date_id', 'time_id', 'symbol_id'] + 
        [pl.col(col).shift().over(['symbol_id', 'time_id'])
         .alias(f'lag_1_{col}') for col in target_cols]
    )
)

df_train = df_train.join(train_lags, on=['date_id', 'time_id', 'symbol_id'])

# 4) Fill NA's (simple fill with 0). Exclude 'responder_*' and 'weight' from training
excluded_features = [col for col in df_train.collect_schema().names() if col.startswith('responder_')] + ['weight']
cols_to_fill = [col for col in df_train.collect_schema().names() if col not in excluded_features]

def filling_na(cols_to_fill, df: pl.DataFrame):
    df = df.fill_null(0)
    return df

df_train = filling_na(cols_to_fill, df_train)
df_train = df_train.collect(engine='gpu')

# 5) Scale the data in chunks
batch_size = 1_000_000
scaler = StandardScaler()
n_train = df_train.height

y_train = df_train.select("responder_6").to_numpy()
w_train = df_train.select("weight").to_numpy()

# Partial fit the scaler
start_idx = 0
while start_idx < n_train:
    end_idx = min(start_idx + batch_size, n_train)
    
    chunk_df = df_train.slice(start_idx, end_idx - start_idx)
    chunk_array = chunk_df.select(cols_to_fill).to_numpy()
    scaler.partial_fit(chunk_array)
    
    del chunk_df, chunk_array
    gc.collect()
    start_idx = end_idx

# Transform the data fully
n_features = len(cols_to_fill)
X_train_scaled = np.empty((n_train, n_features), dtype=np.float32)

start_idx = 0
while start_idx < n_train:
    end_idx = min(start_idx + batch_size, n_train)
    chunk_df = df_train.slice(start_idx, end_idx - start_idx)
    
    chunk_array = chunk_df.select(cols_to_fill).to_numpy()
    scaled_array = scaler.transform(chunk_array)
    
    X_train_scaled[start_idx:end_idx] = scaled_array.astype(np.float32, copy=False)
    
    del chunk_df, chunk_array, scaled_array
    gc.collect()
    start_idx = end_idx

del df_train
gc.collect()

# Save the scaler
joblib.dump(scaler, "scaler.joblib")

# 6) Create a PyTorch Dataset with seq_length=1
class NumPySequenceDataset(Dataset):
    def __init__(self, X, y, w, seq_length):
        self.X = X
        self.y = y.reshape(-1)
        self.w = w.reshape(-1)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length + 1

    def __getitem__(self, idx):
        x_slice = self.X[idx : idx + self.seq_length]
        y_label = self.y[idx + self.seq_length - 1]
        w_label = self.w[idx + self.seq_length - 1]

        x_slice = torch.tensor(x_slice, dtype=torch.float32)
        y_label = torch.tensor(y_label, dtype=torch.float32)
        w_label = torch.tensor(w_label, dtype=torch.float32)

        return x_slice, y_label, w_label

seq_length = 1
train_dataset = NumPySequenceDataset(X_train_scaled, y_train, w_train, seq_length)

def weighted_mse_loss(y_pred, y_true, w):
    y_pred = y_pred.view(-1)
    sq_err = (y_true - y_pred)**2
    weighted_sq_err = w * sq_err
    return torch.mean(weighted_sq_err)

def compute_weighted_r2(model, loader, device):
    model.eval()
    numerator = 0.0
    denominator = 0.0
    with torch.no_grad():
        for batch_x, batch_y, batch_w in loader:
            batch_x, batch_y, batch_w = batch_x.to(device), batch_y.to(device), batch_w.to(device)
            y_pred = model(batch_x).view(-1)
            numerator += torch.sum(batch_w * (batch_y - y_pred)**2).item()
            denominator += torch.sum(batch_w * (batch_y**2)).item()
    if denominator == 0:
        return 0.0
    return 1.0 - (numerator / denominator)

# 7) Define the LSTM model (2-layer, seq_length=1)
class TwoLayerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout=0.2):
        super(TwoLayerLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, seq_length=1, input_dim]
        out, (hn, cn) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# 8) DataLoader + Training Loop
batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device("cuda")
print("Using device:", device)

input_dim = len(cols_to_fill)
hidden_dim = 32
dropout = 0.4
weight_decay = 1e-3
max_epochs = 50
lr = 5e-5

model = TwoLayerLSTM(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=1,
    dropout=dropout
).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y, batch_w in tqdm(loader, desc="Train", leave=False):
        batch_x, batch_y, batch_w = batch_x.to(device), batch_y.to(device), batch_w.to(device)
        optimizer.zero_grad()
        y_pred = model(batch_x).view(-1)
        loss = weighted_mse_loss(y_pred, batch_y, batch_w)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(batch_x)
    return running_loss / len(loader.dataset)

##################################################################
# Early stopping based on *training* Weighted R²
##################################################################
best_train_r2 = float('-inf')
no_improvement_count = 0
patience = 3  # early stop if no improvement for 3 consecutive epochs

for epoch in range(1, max_epochs + 1):
    print(f"\nEpoch {epoch}/{max_epochs}")
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    train_r2 = compute_weighted_r2(model, train_loader, device)
    print(f"  Train Loss: {train_loss:.4f}, Weighted R²: {train_r2:.4f}")

    # Check if train R² improved
    if train_r2 > best_train_r2:
        best_train_r2 = train_r2
        no_improvement_count = 0
        torch.save(model.state_dict(), "best_model_so_far.pth")
        print(f"  (New best training R²: {best_train_r2:.4f} at epoch {epoch})")
    else:
        no_improvement_count += 1
        print(f"  (No improvement, {no_improvement_count} epochs in a row)")

        if no_improvement_count >= patience:
            print(f"\nEarly stopping at epoch {epoch} due to no improvement "
                  f"in training R² for {patience} consecutive epochs.")
            break

print(f"Loading best model from final best training R²={best_train_r2:.4f}")
model.load_state_dict(torch.load("best_model_so_far.pth"))
print("Final model ready!")

