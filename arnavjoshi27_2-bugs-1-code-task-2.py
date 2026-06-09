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


# ============================================================
# ğŸš€ Mission: Repair - Final Ensemble (LGBM + XGB + CatBoost)
# Includes: ROC Curve, Confusion Matrix, RMSE Trend
# ============================================================

import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    mean_squared_error
)
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# âœ… Auto-detect dataset
# ============================================================
root_candidates = [p for p in Path("/kaggle/input").glob("*") if p.is_dir()]
for cand in root_candidates:
    if all((cand / f).exists() for f in ["train.csv", "test.csv", "sample_submission.csv"]):
        INPUT_ROOT = cand
        break
else:
    raise FileNotFoundError("â�Œ train/test/sample_submission.csv not found under /kaggle/input/*")

print(f"âœ… Dataset found at: {INPUT_ROOT}")

train = pd.read_csv(INPUT_ROOT / "train.csv")
test = pd.read_csv(INPUT_ROOT / "test.csv")
sample = pd.read_csv(INPUT_ROOT / "sample_submission.csv")

TARGETS = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
SEED = 42
N_FOLDS = 5
np.random.seed(SEED)

print("Train shape:", train.shape, "Test shape:", test.shape)

# ============================================================
# ğŸ§¹ Basic Preprocessing
# ============================================================
const_cols = [c for c in train.columns if train[c].nunique() <= 1]
train.drop(columns=const_cols, inplace=True, errors="ignore")
test.drop(columns=const_cols, inplace=True, errors="ignore")

features = [c for c in train.columns if c not in ["id"] + TARGETS]

# Encode categorical columns
for col in features:
    if train[col].dtype == "object":
        le = LabelEncoder()
        full = pd.concat([train[col].astype(str), test[col].astype(str)])
        le.fit(full)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

# Fill NaNs
train[features] = train[features].fillna(train[features].median())
test[features] = test[features].fillna(train[features].median())

# ============================================================
# âš™ Training (Ensemble + Visuals)
# ============================================================
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = pd.DataFrame(index=train.index, columns=TARGETS)
test_preds = pd.DataFrame(index=test.index, columns=TARGETS)

for target in TARGETS:
    print(f"\n================= TARGET: {target} =================")

    y = train[target]
    X = train[features]
    X_test = test[features]

    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    fold_rmse = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f" Fold {fold+1}/{N_FOLDS}")

        X_train, y_train = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # --- LightGBM ---
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1
        )
        # âœ… use callbacks for early stopping instead of argument
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=0)
            ]
        )

        # --- XGBoost ---
        xgb_model = xgb.XGBClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=SEED,
            verbosity=0
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=False
        )

        # --- CatBoost ---
        cat_model = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            eval_metric="AUC",
            random_seed=SEED,
            verbose=False,
            early_stopping_rounds=100
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

        # --- Ensemble (average of all 3) ---
        val_pred = (
            lgb_model.predict_proba(X_val)[:, 1] +
            xgb_model.predict_proba(X_val)[:, 1] +
            cat_model.predict_proba(X_val)[:, 1]
        ) / 3

        test_pred = (
            lgb_model.predict_proba(X_test)[:, 1] +
            xgb_model.predict_proba(X_test)[:, 1] +
            cat_model.predict_proba(X_test)[:, 1]
        ) / 3

        oof[val_idx] = val_pred
        preds += test_pred / N_FOLDS

        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        fold_rmse.append(rmse)

        del lgb_model, xgb_model, cat_model
        gc.collect()

    oof_preds[target] = oof
    test_preds[target] = preds

    auc = roc_auc_score(y, oof)
    print(f"âœ… AUC = {auc:.4f}")
    print(f"âœ… Avg RMSE = {np.mean(fold_rmse):.5f}")

    # =======================================================
    # ğŸ�¯ Visualization (ROC, Confusion Matrix, RMSE Trend)
    # =======================================================
    fpr, tpr, _ = roc_curve(y, oof)
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {target}")
    plt.legend()

    plt.subplot(1, 3, 2)
    y_pred_class = (oof > 0.5).astype(int)
    cm = confusion_matrix(y, y_pred_class)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=plt.gca(), colorbar=False)
    plt.title(f"Confusion Matrix - {target}")

    plt.subplot(1, 3, 3)
    plt.plot(range(1, N_FOLDS + 1), fold_rmse, marker='o')
    plt.title(f"RMSE Trend - {target}")
    plt.xlabel("Fold")
    plt.ylabel("RMSE")

    plt.tight_layout()
    plt.show()

# ============================================================
# ğŸ“Š Final Results
# ============================================================
mean_auc = np.mean([roc_auc_score(train[t], oof_preds[t]) for t in TARGETS])
print("\n================ FINAL RESULTS ================")
for t in TARGETS:
    print(f"{t}: AUC = {roc_auc_score(train[t], oof_preds[t]):.5f}")
print(f"\nğŸ”¥ Mean AUC: {mean_auc:.5f}")

# ============================================================
# ğŸ’¾ Submission
# ============================================================
submission = pd.DataFrame({"id": test["id"]})
for t in TARGETS:
    submission[t] = test_preds[t]
submission.to_csv("submission_final.csv", index=False)
print("\nâœ… submission_final.csv saved successfully!")




