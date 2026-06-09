#!/usr/bin/env python3
"""
Noise-Robust Feature-Gated MLP for Accident Risk
=================================================
- Learns per-feature attention weights with trainable bias
- Adds dropout for regularization
- Injects Gaussian noise during training for robustness
"""

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# =======================
# CONFIG
# =======================
DATA_FOLDER = "/kaggle/input/playground-series-s5e10"
EPOCHS = 100
BATCH_SIZE = 2048
LR = 1e-3
PATIENCE = 8
NOISE_STD = 0.05  # standard deviation of Gaussian noise
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# DATA LOADING
# =======================
df = pd.read_csv(f"{DATA_FOLDER}/train.csv")

road_type_map = {'urban': 2, 'rural': 1, 'highway': 3}
lighting_map = {'daylight': 3, 'dim': 2, 'night': 1}
weather_map = {'rainy': 3, 'clear': 1, 'foggy': 2}
time_of_day_map = {'afternoon': 2, 'evening': 3, 'morning': 1}

feature_columns = [
    'road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting',
    'weather', 'road_signs_present', 'public_road', 'time_of_day',
    'holiday', 'school_season', 'num_reported_accidents'
]

df_encoded = df.copy()
for col in feature_columns:
    if col == 'road_type':
        df_encoded[col] = df_encoded[col].map(road_type_map)
    elif col == 'lighting':
        df_encoded[col] = df_encoded[col].map(lighting_map)
    elif col == 'weather':
        df_encoded[col] = df_encoded[col].map(weather_map)
    elif col == 'time_of_day':
        df_encoded[col] = df_encoded[col].map(time_of_day_map)
    elif df_encoded[col].dtype == object or str(df_encoded[col].dtype).startswith('bool'):
        unique_vals = df_encoded[col].unique()
        val2id = {val: idx for idx, val in enumerate(unique_vals)}
        df_encoded[col] = df_encoded[col].map(val2id)

X = df_encoded[feature_columns].astype(float).values
y = df["accident_risk"].astype(float).values

# =======================
# DATASET
# =======================
class AccidentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
train_ds = AccidentDataset(X_train, y_train)
dev_ds = AccidentDataset(X_dev, y_dev)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    pin_memory=True, num_workers=4, prefetch_factor=4
)
dev_loader = DataLoader(
    dev_ds, batch_size=BATCH_SIZE, shuffle=False,
    pin_memory=True, num_workers=4, prefetch_factor=4
)
# =======================
# MODEL: Feature Gate + Dropout + Noise Injection
# =======================
class FeatureAttentionGate(nn.Module):
    """Learns an attention weight and bias per feature."""
    def __init__(self, num_features):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
            nn.Sigmoid()
        )
        self.bias = nn.Parameter(torch.zeros(num_features))  # learned bias per feature

    def forward(self, x):
        weights = self.fc(x)
        return x * (weights + self.bias)  # gated & biased


class AccidentRiskModel(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super().__init__()
        self.gate = FeatureAttentionGate(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, noise_std=0.0):
        # Gaussian noise injection (only during training)
        if self.training and noise_std > 0:
            noise = torch.randn_like(x) * noise_std
            x = x + noise
        x = self.gate(x)
        return self.fc(x).squeeze(-1)


# =======================
# TRAINING
# =======================
def compute_r2(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for Xb, yb in dataloader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            y_true.append(yb.cpu())
            y_pred.append(pred.cpu())
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0


input_dim = X.shape[1]
model = AccidentRiskModel(input_dim).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_r2, best_state, epochs_no_improve = -1e9, None, 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(xb, noise_std=NOISE_STD)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    dev_r2 = compute_r2(model, dev_loader, DEVICE)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss={avg_loss:.4f} | Dev R2={dev_r2:.4f}")

    if dev_r2 > best_r2:
        best_r2 = dev_r2
        best_state = model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

if best_state:
    model.load_state_dict(best_state)

# =======================
# TEST PREDICTION
# =======================
test_df = pd.read_csv(f"{DATA_FOLDER}/test.csv")
test_ids = test_df["id"]

test_encoded = test_df.copy()
for col in feature_columns:
    if col == "road_type":
        test_encoded[col] = test_encoded[col].map(road_type_map)
    elif col == "lighting":
        test_encoded[col] = test_encoded[col].map(lighting_map)
    elif col == "weather":
        test_encoded[col] = test_encoded[col].map(weather_map)
    elif col == "time_of_day":
        test_encoded[col] = test_encoded[col].map(time_of_day_map)
    elif test_encoded[col].dtype == object or str(test_encoded[col].dtype).startswith("bool"):
        val2id = {val: idx for idx, val in enumerate(df_encoded[col].unique())}
        test_encoded[col] = test_encoded[col].map(lambda v: val2id.get(v, 0))

X_test = test_encoded[feature_columns].astype(float).values
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

model.eval()
with torch.no_grad():
    preds = model(X_test_tensor).cpu().numpy()

pd.DataFrame({"id": test_ids, "accident_risk": preds}).to_csv("submission.csv", index=False)
print("Saved submission.csv ✅")



pd.DataFrame({"id": test_ids, "accident_risk": preds}).head()

