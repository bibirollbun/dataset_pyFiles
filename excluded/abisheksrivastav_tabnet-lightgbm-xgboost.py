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


!pip install pytorch-tabnet lightgbm xgboost cudf-cu12 cuml-cu12 --quiet

import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from pytorch_tabnet.tab_model import TabNetRegressor
import lightgbm as lgb
import xgboost as xgb
from cuml.preprocessing import StandardScaler
import torch

# =======================
# Load datasets
# =======================
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

# Original dataset (adjust path if needed)
orig = pd.read_csv("/kaggle/input/bpm-prediction-challenge/Train.csv")

# =======================
# Preprocessing
# =======================
target = "BeatsPerMinute"
id_col = "id"

# Combine train + original dataset
train_full = pd.concat([train, orig], axis=0, ignore_index=True)

# Features
features = [col for col in train.columns if col not in [id_col, target]]

# Feature engineering
for df in [train_full, test]:
    df["LogDuration"] = np.log1p(df["TrackDurationMs"])
    df["EnergyRhythm"] = df["Energy"] * df["RhythmScore"]
    df["MoodAcoustic"] = df["MoodScore"] * df["AcousticQuality"]
    df["LoudnessNorm"] = df["AudioLoudness"] / (1 + df["Energy"])

features += ["LogDuration", "EnergyRhythm", "MoodAcoustic", "LoudnessNorm"]

X = train_full[features].values
y = train_full[target].values
X_test = test[features].values

# reshape target for TabNet
y = y.reshape(-1, 1)

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# =======================
# Cross-validation setup
# =======================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_tabnet = np.zeros(len(X))
oof_preds_lgb = np.zeros(len(X))
oof_preds_xgb = np.zeros(len(X))
test_preds_tabnet = np.zeros(len(X_test))
test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))

# =======================
# Training Loop
# =======================
for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
    print(f"FOLD {fold+1}")

    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    # -------- TabNet --------
    tabnet = TabNetRegressor(
        seed=42,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-3),
        mask_type="entmax"
    )
    tabnet.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=["rmse"],
        patience=20,
        batch_size=1024, virtual_batch_size=128,
        max_epochs=200,
        num_workers=2,
        drop_last=False
    )

    oof_preds_tabnet[valid_idx] = tabnet.predict(X_valid).squeeze()
    test_preds_tabnet += tabnet.predict(X_test).squeeze() / kf.n_splits

    # -------- LightGBM --------
    lgb_train = lgb.Dataset(X_train, y_train.squeeze())
    lgb_valid = lgb.Dataset(X_valid, y_valid.squeeze(), reference=lgb_train)

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "device": "gpu",
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5
    }

    model_lgb = lgb.train(
    lgb_params,
    lgb_train,
    valid_sets=[lgb_valid],
    num_boost_round=5000,
    callbacks=[
        lgb.early_stopping(100),
        lgb.log_evaluation(500)
    ]
)


    oof_preds_lgb[valid_idx] = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
    test_preds_lgb += model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / kf.n_splits

    # -------- XGBoost --------
    model_xgb = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=5000,
        random_state=42
    )

    model_xgb.fit(
        X_train, y_train.squeeze(),
        eval_set=[(X_valid, y_valid.squeeze())],
        eval_metric="rmse",
        early_stopping_rounds=100,
        verbose=500
    )

    oof_preds_xgb[valid_idx] = model_xgb.predict(X_valid)
    test_preds_xgb += model_xgb.predict(X_test) / kf.n_splits

    gc.collect()

# =======================
# Ensemble
# =======================
oof_final = (oof_preds_tabnet + oof_preds_lgb + oof_preds_xgb) / 3
test_final = (test_preds_tabnet + test_preds_lgb + test_preds_xgb) / 3

rmse = mean_squared_error(y, oof_final.reshape(-1, 1), squared=False)
print(f"OOF RMSE: {rmse:.5f}")

# =======================
# Submission
# =======================
sub[target] = test_final
sub.to_csv("submission.csv", index=False)
print("Submission file saved!")


