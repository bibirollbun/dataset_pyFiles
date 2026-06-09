!pip install pytorch-tabnet


%%time

import warnings
warnings.filterwarnings("ignore")

import gc
import torch
import numpy as np
import pandas as pd
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_log_error
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Check if CUDA is available and set the device string for TabNet
DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE_NAME}")

# Load the datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

# Target variable
TARGET = 'Calories'
FEATURES = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
CATEGORICAL_FEATURES = ['Sex']

# Preprocessing
def preprocess(df, is_train=True, encoders=None, scalers=None):
    df = df.copy()
    if encoders is None:
        encoders = {}
    if scalers is None:
        scalers = {}

    for col in CATEGORICAL_FEATURES:
        if col not in encoders:
            le = LabelEncoder() 
            df[col] = le.fit_transform(df[col].astype(str)) if is_train else le.transform(df[col].astype(str))
            encoders[col] = le
        else:
            df[col] = encoders[col].transform(df[col].astype(str))

    numerical_features = [col for col in FEATURES if col not in CATEGORICAL_FEATURES]
    for col in numerical_features:
        if col not in scalers:
            scaler = StandardScaler() 
            df[col] = scaler.fit_transform(df[[col]]) if is_train else scaler.transform(df[[col]])
            scalers[col] = scaler
        else:
            df[col] = scalers[col].transform(df[[col]])

    if is_train:
        return df[FEATURES].values, np.log1p(df[TARGET]).values, encoders, scalers
    else:
        return df[FEATURES].values, encoders, scalers

X, y, encoders, scalers = preprocess(train)
X_test, _, _ = preprocess(test, is_train=False, encoders=encoders, scalers=scalers)

# Prepare categorical feature information for TabNet
cat_idxs = [FEATURES.index(col) for col in CATEGORICAL_FEATURES]
cat_dims = [len(encoders[col].classes_) for col in CATEGORICAL_FEATURES]

# RMSLE Loss Function
def rmsle_loss(y_pred, y_true):
    # Ensure predictions are non-negative before log1p
    # Adding a small epsilon to avoid log(0) if relu makes it exactly 0
    y_pred_non_neg = torch.relu(y_pred) + 1e-6
    return torch.sqrt(F.mse_loss(torch.log1p(y_pred_non_neg), torch.log1p(y_true)))

# Define TabNet parameters
TABNET_PARAMS = {
    'n_d': 16,
    'n_a': 16,
    'n_steps': 3,
    'gamma': 1.5,
    'n_independent': 2,
    'n_shared': 2,
    'optimizer_fn': optim.AdamW,
    'optimizer_params': dict(lr=0.005, weight_decay=1e-5),
    'scheduler_fn': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'scheduler_params': dict(mode='min', factor=0.5, patience=5, verbose=False),
    'mask_type': 'sparsemax',
    'seed': 42,
    'verbose': 1,
    'device_name': DEVICE_NAME
}

# K-Fold Cross-Validation
NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(train), dtype=np.float32)
test_predictions = np.zeros(len(test), dtype=np.float32)
rmsle_scores = []

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    
    print(f"Fold {fold+1}")
    
    X_train, y_train = X[train_idx].astype(np.float32), y[train_idx].astype(np.float32).reshape(-1, 1)
    X_val, y_val = X[val_idx].astype(np.float32), y[val_idx].astype(np.float32).reshape(-1, 1)

    model = TabNetRegressor(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=1,
        **TABNET_PARAMS
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['rmse', 'rmsle'],
        patience=15,
        batch_size=512,
        virtual_batch_size=128,
        max_epochs=200,
        weights=0,
        drop_last=False,
        num_workers=0,
        from_unsupervised=None
    )

    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    oof_predictions[val_idx] = val_preds.flatten()
    test_predictions += test_preds.flatten() / NFOLDS

    # Calculate RMSLE on original scale for evaluation
    rmse_val = np.sqrt(mean_squared_log_error(np.expm1(y_val.flatten()), np.maximum(0, np.expm1(val_preds.flatten()))))
    rmsle_scores.append(rmse_val)
    print(f"Fold {fold+1} RMSLE: {rmse_val:.5f}")

    del model, X_train, y_train, X_val, y_val, val_preds, test_preds
    gc.collect()
    torch.cuda.empty_cache() 

print(f"\nMean OOF RMSLE: {np.mean(rmsle_scores):.5f}")

# Evaluate OOF predictions
oof_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y), np.maximum(0, np.expm1(oof_predictions))))
print(f"OOF RMSLE: {oof_rmsle:.5f}")

# Save OOF predictions to CSV
oof = pd.DataFrame({'id': train['id'], 'Calories': np.expm1(oof_predictions)})
oof.to_csv('oof_tabnet.csv', index=False)

# Create submission file
predictions = np.maximum(0, np.expm1(test_predictions))
submission = pd.DataFrame({'id': test['id'], 'Calories': predictions})
submission.to_csv('submission_tabnet.csv', index=False)

