#!/usr/bin/env python3
"""
Noise-Aware MLP Regression for Accident Risk
- Handles Gaussian noise explicitly (predicts mean + variance)
- Converts categorical features to one-hot vectors
"""

import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# =======================
# CONFIG
# =======================
DATA_FOLDER = "/kaggle/input/playground-series-s5e10"
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# LOAD DATA
# =======================
df = pd.read_csv(f"{DATA_FOLDER}/train.csv")

# --- Identify feature types ---
target_col = "accident_risk"
id_col = "id"

numeric_cols = []
categorical_cols = []
for col in df.columns:
    if col in [target_col, id_col]:
        continue
    if np.issubdtype(df[col].dtype, np.number):
        numeric_cols.append(col)
    else:
        categorical_cols.append(col)

print("Numeric cols:", numeric_cols)
print("Categorical cols:", categorical_cols)

# --- One-hot encode categorical columns ---
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(categorical_cols),
    index=df.index,
)

df_encoded = pd.concat([df[numeric_cols], encoded_df], axis=1)
X = df_encoded.values.astype(np.float32)
y = df[target_col].astype(np.float32).values

# =======================
# DATASETS
# =======================
class AccidentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


X_train, X_dev, y_train, y_dev = train_test_split(
    X, y, test_size=0.2, random_state=42
)
train_ds = AccidentDataset(X_train, y_train)
dev_ds = AccidentDataset(X_dev, y_dev)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False)

# =======================
# MODEL: Mean + LogVar
# =======================
class AccidentRiskModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(64, 1)
        self.log_var_head = nn.Linear(64, 1)

    def forward(self, x):
        h = self.shared(x)
        mu = self.mean_head(h).squeeze(-1)
        log_var = self.log_var_head(h).squeeze(-1)
        return mu, log_var


# =======================
# LOSS + METRICS
# =======================
def gaussian_nll_loss(y_true, mu, log_var):
    # NLL under Gaussian noise assumption
    return torch.mean(0.5 * (log_var + (y_true - mu) ** 2 / torch.exp(log_var)))


@torch.no_grad()
def compute_rmse(model, dataloader):
    model.eval()
    y_true, y_pred = [], []
    for Xb, yb in dataloader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        mu, _ = model(Xb)
        y_true.append(yb.cpu())
        y_pred.append(mu.cpu())
    if not y_true:
        return 0.0
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    mse = torch.mean((y_true - y_pred) ** 2)
    return torch.sqrt(mse).item()


# =======================
# TRAINING LOOP
# =======================
input_dim = X.shape[1]
model = AccidentRiskModel(input_dim).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

best_r2, best_state, epochs_no_improve = -np.inf, None, 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        mu, log_var = model(xb)
        loss = gaussian_nll_loss(yb, mu, log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    dev_rmse = compute_rmse(model, dev_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss={avg_loss:.4f} | Dev RMSE={dev_rmse:.4f}")

    # For early stopping, lower RMSE is better
    if epoch == 0 or dev_rmse < best_r2:
        best_r2 = dev_rmse
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
# PREDICTION ON TEST SET
# =======================
test_df = pd.read_csv(f"{DATA_FOLDER}/test.csv")
test_encoded = encoder.transform(test_df[categorical_cols])
test_encoded_df = pd.DataFrame(
    test_encoded, columns=encoder.get_feature_names_out(categorical_cols)
)
test_full = pd.concat([test_df[numeric_cols].reset_index(drop=True),
                       test_encoded_df.reset_index(drop=True)], axis=1)

X_test = test_full.values.astype(np.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

model.eval()
with torch.no_grad():
    mu, log_var = model(X_test_tensor)
    preds = mu.cpu().numpy()
    pred_std = np.sqrt(np.exp(log_var.cpu().numpy()))  # uncertainty estimate

# Save submission
submission = pd.DataFrame({
    "id": test_df["id"],
    "accident_risk": preds,
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")

