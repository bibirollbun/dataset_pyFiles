pip install -q pytorch-tabnet lightgbm catboost xgboost



!ls /input


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
from torch.utils.data import Dataset, DataLoader, TensorDataset

from pytorch_tabnet.tab_model import TabNetRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# RMSLE Metric
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.maximum(0, y_pred)))

# Dataset
df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")  # Replace with correct path
TARGET = "Calories"
NUM_FEATURES = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
CAT_FEATURES = ["Sex"]

# Preprocessing pipeline



preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), NUM_FEATURES),
    ('cat', OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)
])

X = df[NUM_FEATURES + CAT_FEATURES]
y = df[TARGET].values
X_proc = preprocessor.fit_transform(X)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = []
models = []

# Blend 5 models: LGBM, CatBoost, XGBoost, TabNet, PyTorch MLP
for fold, (train_idx, val_idx) in enumerate(kf.split(X_proc)):
    print(f"\nFold {fold+1}")
    X_train, X_val = X_proc[train_idx], X_proc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    preds_fold = []

    # LightGBM
    model_lgb = lgb.LGBMRegressor(n_estimators=1000)
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    preds_fold.append(model_lgb.predict(X_val))
    
    # CatBoost
    model_cat = CatBoostRegressor(verbose=0, iterations=1000, early_stopping_rounds=50)
    model_cat.fit(X_train, y_train, eval_set=(X_val, y_val))
    preds_fold.append(model_cat.predict(X_val))
    
    # XGBoost
    model_xgb = xgb.XGBRegressor(n_estimators=1000)
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    preds_fold.append(model_xgb.predict(X_val))

    # TabNet
    tabnet = TabNetRegressor()
    tabnet.fit(
        X_train=X_train, y_train=y_train.reshape(-1, 1),
        eval_set=[(X_val, y_val.reshape(-1, 1))],
        eval_metric=['rmse'], max_epochs=200,
        patience=20, verbose=0
    )
    preds_fold.append(tabnet.predict(X_val).ravel())

    # MLP on TPU
    class MLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.model(x).squeeze(1)

    device = xm.xla_device()
    model_mlp = MLP(X_train.shape[1]).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model_mlp.parameters(), lr=1e-3)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    for epoch in range(20):  # Short training for demonstration
        model_mlp.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model_mlp(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            xm.optimizer_step(optimizer)

    model_mlp.eval()
    with torch.no_grad():
        val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        preds_mlp = model_mlp(val_tensor).cpu().numpy()
    preds_fold.append(preds_mlp)

    # Blend predictions by mean
    blended = np.mean(preds_fold, axis=0)
    score = rmsle(y_val, blended)
    print(f"Fold RMSLE: {score:.4f}")
    oof_preds.extend(blended)
    models.append({
        'lgb': model_lgb,
        'cat': model_cat,
        'xgb': model_xgb,
        'tabnet': tabnet,
        'mlp': model_mlp
    })


final_score = rmsle(y, oof_preds)
print(f"\nOverall OOF RMSLE: {final_score:.4f}")



# Read test set
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
X_test = test_df[NUM_FEATURES + CAT_FEATURES]
X_test_proc = preprocessor.transform(X_test)

# Initialize list to hold test predictions from each fold
test_preds = []

for fold in range(5):
    fold_preds = []

    # Predict with each model saved per fold
    fold_preds.append(models[fold]['lgb'].predict(X_test_proc))
    fold_preds.append(models[fold]['cat'].predict(X_test_proc))
    fold_preds.append(models[fold]['xgb'].predict(X_test_proc))
    fold_preds.append(models[fold]['tabnet'].predict(X_test_proc).ravel())

    # MLP
    model_mlp = models[fold]['mlp']
    model_mlp.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_proc, dtype=torch.float32).to(device)
        preds_mlp = model_mlp(X_test_tensor).cpu().numpy()
    fold_preds.append(preds_mlp)

    # Average predictions from all 5 models for this fold
    test_preds.append(np.mean(fold_preds, axis=0))

# Average across folds
final_preds = np.mean(test_preds, axis=0)

# Prepare submission file
submission = pd.DataFrame({
    "id": test_df["id"],
    "Calories": final_preds
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv saved!")


