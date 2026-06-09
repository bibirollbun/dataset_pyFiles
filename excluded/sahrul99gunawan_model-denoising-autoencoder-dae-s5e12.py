# =========================================================================================
# THE "CHAMPION" PIPELINE: DAE + TRINITY ENSEMBLE (CatBoost, XGBoost, LightGBM)
# =========================================================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
import gc
import re

# Setup
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"ðŸš€ Running on: {device}")

# =========================================================================================
# 1. LOAD & PREPROCESS DATA
# =========================================================================================
print("\n[1/5] Loading & Processing Data...")

# Load Data (Sesuaikan path jika di Kaggle: '/kaggle/input/playground-series-s5e12/train.csv')
try:
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    submission = pd.read_csv('sample_submission.csv')
except FileNotFoundError:
    # Fallback untuk path Kaggle standar
    train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
    submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

target_col = 'diagnosed_diabetes'
train_len = len(train)

# Gabung untuk Preprocessing konsisten
X_all = pd.concat([train.drop(columns=[target_col, 'id']), test.drop(columns=['id'])], axis=0).reset_index(drop=True)

# --- Feature Engineering Medis ---
X_all['BMI_Age'] = X_all['bmi'] * X_all['age']
X_all['BP_Ratio'] = X_all['systolic_bp'] / (X_all['diastolic_bp'] + 1) # Hindari div/0
X_all['Chol_Ratio'] = X_all['cholesterol_total'] / (X_all['hdl_cholesterol'] + 1)
X_all['Activity_Intensity'] = X_all['physical_activity_minutes_per_week'] * X_all['heart_rate']

# --- Encoding & Scaling ---
cat_cols = X_all.select_dtypes(include=['object']).columns.tolist()
num_cols = X_all.select_dtypes(include=['int64', 'float64']).columns.tolist()

# One-Hot Encoding
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = ohe.fit_transform(X_all[cat_cols])
df_encoded = pd.DataFrame(encoded_cats, columns=ohe.get_feature_names_out(cat_cols))

# Scaling Numeric
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(X_all[num_cols])
df_nums = pd.DataFrame(scaled_nums, columns=num_cols)

# Gabung lagi
X_processed = pd.concat([df_nums, df_encoded], axis=1)

# Sanitasi nama kolom (Penting buat LightGBM/XGBoost biar ga error baca spasi)
X_processed.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_processed.columns]

print(f"Data siap! Dimensi: {X_processed.shape}")

# =========================================================================================
# 2. DENOISING AUTOENCODER (DAE) - DEEP LEARNING FEATURES
# =========================================================================================
print("\n[2/5] Training Denoising Autoencoder (Neural Network)...")

# Arsitektur Neural Network
class DAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(DAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64), # Bottleneck: Ini fitur barunya (64 kolom)
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.noise = nn.Dropout(p=0.2) # Rusak 20% data biar AI mikir keras

    def forward(self, x):
        x_noisy = self.noise(x) if self.training else x
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded)
        return decoded, encoded

# Prepare Data Loader
X_tensor = torch.FloatTensor(X_processed.values).to(device)
dataset = TensorDataset(X_tensor, X_tensor)
dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)

# Train Loop
dae = DAE(input_dim=X_processed.shape[1]).to(device)
optimizer = optim.Adam(dae.parameters(), lr=0.005)
criterion = nn.MSELoss()

epochs = 30 
for epoch in range(epochs):
    dae.train()
    loss_sum = 0
    for batch in dataloader:
        batch_x = batch[0]
        optimizer.zero_grad()
        output, _ = dae(batch_x)
        loss = criterion(output, batch_x)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    
    if (epoch+1) % 10 == 0:
        print(f"   Epoch {epoch+1}/{epochs} | Loss: {loss_sum/len(dataloader):.5f}")

# Extract Latent Features (Fitur Rahasia)
dae.eval()
with torch.no_grad():
    _, latent_features = dae(X_tensor)
    latent_features = latent_features.cpu().numpy()

# Gabung Fitur Neural Network ke Data Asli
df_latent = pd.DataFrame(latent_features, columns=[f'dae_{i}' for i in range(latent_features.shape[1])])
X_final = pd.concat([X_processed, df_latent], axis=1)

# Split Train & Test kembali
X_train = X_final.iloc[:train_len]
X_test = X_final.iloc[train_len:]
y_train = train[target_col]

print(f"Fitur Deep Learning ditambahkan. Total Fitur: {X_train.shape[1]}")
del dae, X_tensor, dataset, dataloader, X_processed
gc.collect()

# =========================================================================================
# 3. TRINITY ENSEMBLE TRAINING
# =========================================================================================
print("\n[3/5] Starting Ensemble Training...")

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
lgbm_preds = np.zeros(len(X_test))

# --- MODEL 1: CATBOOST ---
print("   >> Training CatBoost...")
cat_params = {
    'iterations': 2500, 'learning_rate': 0.03, 'depth': 6,
    'l2_leaf_reg': 5, 'loss_function': 'Logloss', 'eval_metric': 'AUC',
    'verbose': 0, 'early_stopping_rounds': 100,
    'allow_writing_files': False,
    'task_type': 'GPU' if torch.cuda.is_available() else 'CPU'
}

for fold, (train_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    model = CatBoostClassifier(**cat_params)
    model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx],
              eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]),
              use_best_model=True)
    cat_preds += model.predict_proba(X_test)[:, 1] / folds.n_splits
print("      CatBoost Done.")

# --- MODEL 2: XGBOOST ---
print("   >> Training XGBoost...")
xgb_params = {
    'n_estimators': 2000, 'learning_rate': 0.02, 'max_depth': 8,
    'subsample': 0.7, 'colsample_bytree': 0.7, 'objective': 'binary:logistic',
    'eval_metric': 'auc', 
    'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
    'random_state': 42, 'n_jobs': -1
}

for fold, (train_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    model = XGBClassifier(**xgb_params)
    model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx],
              eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
              verbose=False)
    xgb_preds += model.predict_proba(X_test)[:, 1] / folds.n_splits
print("      XGBoost Done.")

# --- MODEL 3: LIGHTGBM ---
print("   >> Training LightGBM...")
lgbm_params = {
    'n_estimators': 2500, 'learning_rate': 0.02, 'num_leaves': 64,
    'max_depth': -1, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'objective': 'binary', 'metric': 'auc', 'random_state': 42,
    'n_jobs': -1, 'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    model = LGBMClassifier(**lgbm_params)
    model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx],
              eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
              callbacks=[])
    lgbm_preds += model.predict_proba(X_test)[:, 1] / folds.n_splits
print("      LightGBM Done.")

# =========================================================================================
# 4. BLENDING & SUBMISSION
# =========================================================================================
print("\n[4/5] Blending Predictions...")

# Bobot Racikan: CatBoost (40%), XGBoost (35%), LightGBM (25%)
final_preds = (0.40 * cat_preds) + (0.35 * xgb_preds) + (0.25 * lgbm_preds)

print("\n[5/5] Saving Submission...")
submission['diagnosed_diabetes'] = final_preds
submission.to_csv('submission_champion_ensemble.csv', index=False)

print("\nâœ… SELESAI! File 'submission_champion_ensemble.csv' siap disubmit.")
print("Semoga tembus Top 10! ðŸ”¥")


