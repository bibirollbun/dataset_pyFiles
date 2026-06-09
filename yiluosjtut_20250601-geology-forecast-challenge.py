import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Section 1: Load Data
TRAIN_PATH = '/kaggle/input/geology-forecast-challenge-open/data/train.csv'
TEST_PATH  = '/kaggle/input/geology-forecast-challenge-open/data/test.csv'
SUB_PATH   = '/kaggle/input/geology-forecast-challenge-open/data/sample_submission.csv'

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
sub   = pd.read_csv(SUB_PATH)

FEATURES = [c for c in test.columns if c != 'geology_id']
TARGETS  = [c for c in sub.columns if c != 'geology_id']


# Section 2: Preprocessing (Impute + Log + Scale)
train_feats = train[FEATURES].copy()
test_feats  = test[FEATURES].copy()

train_feats = train_feats.fillna(train_feats.mean())
test_feats  = test_feats.fillna(test_feats.mean())

X_raw      = np.log(31.0 + train_feats.values)
X_test_raw = np.log(31.0 + test_feats.values)

feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_raw)
X_test  = feature_scaler.transform(X_test_raw)

y_raw = train[TARGETS].copy().values.astype(np.float32)
target_scaler = StandardScaler()
y_scaled = target_scaler.fit_transform(y_raw)


# Section 3: Dataset / DataLoader Helper
def create_dataloader(X_data, y_data=None, batch_size=128, shuffle=False):
    if y_data is not None:
        ds = TensorDataset(
            torch.tensor(X_data, dtype=torch.float32),
            torch.tensor(y_data, dtype=torch.float32)
        )
    else:
        ds = TensorDataset(torch.tensor(X_data, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)



# Section 4: DNN Model Definition
class DNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.GELU(),

            nn.Linear(256, 128),
            nn.GELU(),

            nn.Linear(128, 64),
            nn.GELU(),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)



# Section 5: Training Loop with K-Fold CV, Early Stopping, LR Scheduler
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOLDS        = 5
EPOCHS       = 50
BATCH_SIZE   = 128
LR           = 1e-3
PATIENCE     = 5

n_train = X_train.shape[0]
n_test  = X_test.shape[0]
n_tgt   = len(TARGETS)

test_preds = np.zeros((n_test, n_tgt), dtype=np.float32)
oof_preds  = np.zeros((n_train, n_tgt), dtype=np.float32)

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    print(f"\nğŸ•� Fold {fold}/{FOLDS}")

    X_tr, y_tr = X_train[train_idx], y_scaled[train_idx]
    X_val, y_val = X_train[val_idx], y_scaled[val_idx]

    train_loader = create_dataloader(X_tr, y_tr, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = create_dataloader(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = create_dataloader(X_test, batch_size=BATCH_SIZE, shuffle=False)

    model     = DNN(input_dim=X_train.shape[1], output_dim=n_tgt).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )
    loss_fn   = nn.MSELoss()

    best_val_loss = np.inf
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, EPOCHS+1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        model.eval()
        val_preds_fold = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                pred = model(xb).cpu().numpy()
                val_preds_fold.append(pred)
        val_preds_fold = np.vstack(val_preds_fold)
        val_loss = mean_squared_error(y_val, val_preds_fold)

        print(f"Epoch {epoch:02d} | Train Loss {avg_train_loss:.6f} | Val Loss {val_loss:.6f}")

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("â�¹ Early stopping triggered.")
                break

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        fold_val_preds = []
        for xb, _ in val_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            fold_val_preds.append(pred)
        fold_val_preds = np.vstack(fold_val_preds)
        oof_preds[val_idx, :] = fold_val_preds

    model.eval()
    fold_test_preds = []
    with torch.no_grad():
        for xb, in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            fold_test_preds.append(pred)
        fold_test_preds = np.vstack(fold_test_preds)
        test_preds += fold_test_preds / FOLDS



# Section 6: Inverse Transform & Prepare Submission DataFrame
oof_preds_inv  = target_scaler.inverse_transform(oof_preds)
test_preds_inv = target_scaler.inverse_transform(test_preds)

oof_score = mean_squared_error(train[TARGETS].values, oof_preds_inv)
print(f"OOF RMSE (raw): {np.sqrt(oof_score):.6f}")

sub_df = pd.DataFrame(test_preds_inv, columns=TARGETS)
sub_df.insert(0, 'geology_id', test['geology_id'])


# Section 7: â€œAverage Trickâ€� Postâ€�Processing
df_sub = sub_df.copy()
numeric_values = df_sub.iloc[:, 1:].values
n_samples = numeric_values.shape[0]
data_reshaped = numeric_values.reshape(n_samples, 10, 300)
mean_across_realizations = data_reshaped.mean(axis=1)
mean_repeated = np.tile(mean_across_realizations[:, None, :], (1, 10, 1))
mean_repeated = mean_repeated.reshape(n_samples, 3000)
df_sub.iloc[:, 1:] = mean_repeated

df_sub.to_csv("submission_refined.csv", index=False)
df_sub.head()

