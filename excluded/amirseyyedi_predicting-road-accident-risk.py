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


# =========================================
# Kaggle Playground 2025 - Accident Risk
# Single-cell final code (no CLI args needed)
# =========================================
import os, glob, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Try LightGBM; fallback to HistGradientBoostingRegressor if unavailable
USE_LGB = True
try:
    import lightgbm as lgb
except Exception:
    USE_LGB = False
    from sklearn.ensemble import HistGradientBoostingRegressor

TARGET = "accident_risk"
ID_COL = "id"

# -------- Helpers --------
def find_file(name: str) -> str:
    if os.path.exists(name):
        return name
    for path in glob.glob("/kaggle/input/**", recursive=True):
        if path.lower().endswith(f"/{name.lower()}") and os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Could not find {name} in working dir or /kaggle/input/")

def preprocess_tabular(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # Align columns between train (without target) and test
    X_tr_raw = train_df.drop(columns=[TARGET]).copy()
    X_te_raw = test_df.copy()
    common_cols = [c for c in X_tr_raw.columns if c in X_te_raw.columns]
    X_tr_raw = X_tr_raw[common_cols].reset_index(drop=True)
    X_te_raw = X_te_raw[common_cols].reset_index(drop=True)

    # Combine for consistent transforms
    X_all = pd.concat([X_tr_raw, X_te_raw], axis=0, ignore_index=True)

    cat_cols, num_cols = [], []
    for c in X_all.columns:
        if X_all[c].dtype == "object" or str(X_all[c].dtype).startswith("category"):
            cat_cols.append(c)
        else:
            num_cols.append(c)

    # Fill numeric with median
    for c in num_cols:
        X_all[c] = X_all[c].fillna(X_all[c].median())

    # Label-encode categoricals
    for c in cat_cols:
        X_all[c] = X_all[c].astype(str).fillna("__NA__")
        le = LabelEncoder()
        X_all[c] = le.fit_transform(X_all[c])

    X = X_all.iloc[:len(X_tr_raw)].reset_index(drop=True)
    X_test = X_all.iloc[len(X_tr_raw):].reset_index(drop=True)
    return X, X_test, common_cols

# -------- Load data --------
train_path = find_file("train.csv")
test_path  = find_file("test.csv")

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

assert TARGET in train.columns, f"Missing target '{TARGET}' in train.csv"
assert ID_COL in test.columns, f"Missing ID '{ID_COL}' in test.csv"

y = train[TARGET].astype(float).reset_index(drop=True)
X, X_test, feature_names = preprocess_tabular(train, test)

# -------- Train & Predict --------
folds = 5
seed = 42

oof = np.zeros(len(X), dtype=float)
test_pred = np.zeros(len(X_test), dtype=float)
fold_rmses = []
all_importances = []

if USE_LGB:
    print("Backend: LightGBM")
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=3000,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_samples=40,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=seed + fold
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
        te_pred  = model.predict(X_test, num_iteration=model.best_iteration_)
        rmse = mean_squared_error(y_val, val_pred, squared=False)
        fold_rmses.append(rmse)
        print(f"[Fold {fold}] RMSE: {rmse:.6f} (iters={model.best_iteration_})")

        oof[val_idx] = val_pred
        test_pred += te_pred / folds

        # Gain importance (fallback to split importance if needed)
        try:
            imp = model.booster_.feature_importance(importance_type="gain")
        except Exception:
            imp = model.feature_importances_
        all_importances.append(imp)

    oof_rmse = mean_squared_error(y, oof, squared=False)
    print(f"[OOF] RMSE: {oof_rmse:.6f} | mean={np.mean(fold_rmses):.6f}, std={np.std(fold_rmses):.6f}")

    importances = np.mean(np.vstack(all_importances), axis=0)
    fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)

else:
    print("Backend: HistGradientBoosting (LightGBM not available)")
    # Fast single split + refit on full, then pseudo-CV by bootstrapping (quick)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = HistGradientBoostingRegressor(
        learning_rate=0.08, max_iter=600, min_samples_leaf=50,
        early_stopping=True, validation_fraction=0.1, random_state=seed
    )
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, val_pred, squared=False)
    print(f"[Split] RMSE: {rmse:.6f}")

    final_model = HistGradientBoostingRegressor(
        learning_rate=0.08, max_iter=model.n_iter_, min_samples_leaf=50,
        early_stopping=False, random_state=seed
    )
    final_model.fit(X, y)
    test_pred = final_model.predict(X_test)
    oof_rmse = rmse
    fi = pd.DataFrame({"feature": feature_names, "importance": 0.0})

# -------- Save artifacts --------
sub = pd.DataFrame({
    ID_COL: test[ID_COL].values,
    TARGET: np.clip(test_pred, 0.0, 1.0)
})
sub.to_csv("submission.csv", index=False)

metrics = {
    "fold_rmse": [float(r) for r in fold_rmses],
    "oof_rmse": float(mean_squared_error(y, oof, squared=False)) if USE_LGB else float(oof_rmse)
}
with open("cv_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

fi.to_csv("feature_importance.csv", index=False)

print("\nSaved files:")
print(" - submission.csv")
print(" - cv_metrics.json")
print(" - feature_importance.csv")

# -------- Quick displays --------
# Top-20 feature importances (if any)
top = fi.head(min(20, len(fi)))
plt.figure(figsize=(8, max(5, int(0.35*len(top)))))
plt.barh(top["feature"][::-1], top["importance"][::-1])
plt.title("Top Feature Importances")
plt.tight_layout()
plt.show()

# Fold RMSE bar
if len(fold_rmses) > 0:
    plt.figure(figsize=(6,4))
    plt.bar([f"Fold {i}" for i in range(1, len(fold_rmses)+1)], fold_rmses)
    plt.title(f"CV RMSEs (mean={np.mean(fold_rmses):.5f}, std={np.std(fold_rmses):.5f})")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# Preview submission
display(sub.head())


