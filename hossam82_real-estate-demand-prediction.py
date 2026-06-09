"""
Deep Learning Real Estate Demand Prediction using Temporal Fusion Transformer
Minimal code for improved accuracy over baseline
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
pth = "/kaggle/input/china-real-estate-demand-prediction"

# Load key datasets
train_nht = pd.read_csv(f"{pth}/train/new_house_transactions.csv")
train_nhtns = pd.read_csv(f"{pth}/train/new_house_transactions_nearby_sectors.csv")
train_pht = pd.read_csv(f"{pth}/train/pre_owned_house_transactions.csv")
test = pd.read_csv(f"{pth}/test.csv")

# Parse test IDs
test[['month', 'sector']] = test['id'].str.split('_', expand=True)

month_map = {m: i for i, m in enumerate(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], 1)}

def process_data(df):
    df = df.copy()
    df['year'] = df['month'].str[:4].astype(int)
    df['month_num'] = df['month'].str[5:].map(month_map)
    df['sector_id'] = df['sector'].str.extract('(\d+)').astype(int)
    df['time'] = (df['year'] - 2019) * 12 + df['month_num'] - 1
    return df

train_nht = process_data(train_nht)
train_nhtns = process_data(train_nhtns)
train_pht = process_data(train_pht)
test = process_data(test)

# Merge datasets
data = train_nht.merge(train_nhtns, on=['sector', 'month', 'time', 'sector_id', 'year', 'month_num'], 
                       how='left', suffixes=('', '_nearby'))
data = data.merge(train_pht[['sector', 'month', 'amount_pre_owned_house_transactions']], 
                  on=['sector', 'month'], how='left')
data = data.fillna(0)

# Feature engineering
print("Engineering features...")
data = data.sort_values(['sector_id', 'time']).reset_index(drop=True)

# Add lags and rolling features
for lag in [1, 2, 3, 12]:
    data[f'amount_lag_{lag}'] = data.groupby('sector_id')['amount_new_house_transactions'].shift(lag)

for window in [3, 6, 12]:
    data[f'amount_roll_mean_{window}'] = data.groupby('sector_id')['amount_new_house_transactions'].transform(
        lambda x: x.rolling(window, min_periods=1).mean())
    data[f'amount_roll_std_{window}'] = data.groupby('sector_id')['amount_new_house_transactions'].transform(
        lambda x: x.rolling(window, min_periods=1).std())

# Seasonality
data['sin_month'] = np.sin(2 * np.pi * data['month_num'] / 12)
data['cos_month'] = np.cos(2 * np.pi * data['month_num'] / 12)

data = data.fillna(0)

# Select features
feature_cols = [
    'sector_id', 'time', 'month_num', 'sin_month', 'cos_month',
    'num_new_house_transactions', 'area_per_unit_new_house_transactions',
    'num_new_house_available_for_sale',
    'amount_lag_1', 'amount_lag_2', 'amount_lag_3', 'amount_lag_12',
    'amount_roll_mean_3', 'amount_roll_mean_6', 'amount_roll_mean_12',
    'amount_roll_std_3', 'amount_roll_std_6', 'amount_roll_std_12',
    'amount_pre_owned_house_transactions'
]

target_col = 'amount_new_house_transactions'

# Prepare sequences
def create_sequences(data, seq_len=12):
    sequences = []
    targets = []
    sectors = []
    
    for sector in data['sector_id'].unique():
        sector_data = data[data['sector_id'] == sector].sort_values('time')
        
        if len(sector_data) < seq_len + 1:
            continue
            
        for i in range(len(sector_data) - seq_len):
            seq = sector_data.iloc[i:i+seq_len][feature_cols].values
            target = sector_data.iloc[i+seq_len][target_col]
            sequences.append(seq)
            targets.append(target)
            sectors.append(sector)
    
    return np.array(sequences), np.array(targets), np.array(sectors)

print("Creating sequences...")
X, y, sector_ids = create_sequences(data, seq_len=12)

# Normalize
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_flat = X.reshape(-1, X.shape[-1])
X_scaled = scaler_X.fit_transform(X_flat).reshape(X.shape)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Split train/val
split_idx = int(len(X) * 0.85)
X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]

# PyTorch Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

# Model: Transformer + LSTM hybrid
class TemporalModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, 
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(d_model, d_model//2, num_layers=2, batch_first=True, dropout=dropout)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(d_model//2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last timestep
        x = self.fc(x)
        return x.squeeze()

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = TemporalModel(input_dim=X_train.shape[-1]).to(device)
criterion = nn.HuberLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

print(f"\nTraining on {len(X_train)} samples...")
best_val_loss = float('inf')
patience = 15
patience_counter = 0

for epoch in range(100):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            val_loss += criterion(y_pred, y_batch).item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), '/kaggle/working/best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load('/kaggle/working/best_model.pt'))
model.eval()

# Prepare test data - align with actual test months
print("\nGenerating predictions...")

# Get last time in training data
max_train_time = data['time'].max()
print(f"Last training time: {max_train_time}")

# Create predictions for each test row
predictions = []

for idx, row in test.iterrows():
    sector = row['sector_id']
    target_time = row['time']
    
    # Get historical data for this sector
    sector_data = data[data['sector_id'] == sector].sort_values('time')
    
    if len(sector_data) < 12:
        predictions.append(0)
        continue
    
    # Get the 12 most recent months before target
    history = sector_data[sector_data['time'] < target_time].tail(12)
    
    if len(history) < 12:
        # Pad with zeros if not enough history
        predictions.append(0)
        continue
    
    # Prepare sequence
    seq = history[feature_cols].values
    seq_scaled = scaler_X.transform(seq.reshape(-1, seq.shape[-1])).reshape(1, 12, -1)
    
    # Predict
    with torch.no_grad():
        pred_scaled = model(torch.FloatTensor(seq_scaled).to(device)).cpu().numpy()
    
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
    pred = max(0, pred)
    predictions.append(pred)

# Create submission
submission = test[['id']].copy()
submission['new_house_transaction_amount'] = predictions

submission.to_csv('/kaggle/working/submission.csv', index=False)

print("\n" + "="*60)
print("Submission saved!")
print(f"Total predictions: {len(submission)}")
print(f"Non-zero predictions: {(submission['new_house_transaction_amount'] > 0).sum()}")
print(f"Mean prediction: {submission['new_house_transaction_amount'].mean():.2f}")
print("="*60)
print("\nSample predictions:")
print(submission.head(10))

