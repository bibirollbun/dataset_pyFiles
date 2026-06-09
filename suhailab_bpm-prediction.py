import os, sys, gc, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

# Try LightGBM; fall back to RF if unavailable
LGBM_AVAILABLE = True
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
except Exception as e:
    print("LightGBM not available, will use RandomForest. Reason:", e)
    LGBM_AVAILABLE = False


DATA_DIR = "/kaggle/input/playground-series-s5e9"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path  = os.path.join(DATA_DIR, "test.csv")

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

print("Train shape:", train.shape, " Test shape:", test.shape)
display(train.head())


# Missing values
na_pct = (train.isna().sum() / len(train) * 100).round(2).sort_values(ascending=False)
display(na_pct.to_frame("percent NULL (%)"))

# Numeric summary
display(train.describe().T)

# Target distribution
train['BeatsPerMinute'].hist(bins=40)
plt.title("BeatsPerMinute distribution")
plt.xlabel("BPM"); plt.ylabel("Count"); plt.show()

# Quick correlations (top/bottom 8)
corr = train.corr(numeric_only=True)['BeatsPerMinute'].drop('BeatsPerMinute').sort_values()
display(pd.concat([corr.head(8), corr.tail(8)]))


TARGET = "BeatsPerMinute"
ID = "id"

num_cols = train.drop(columns=[ID, TARGET]).select_dtypes(include=[np.number]).columns.tolist()

X = train[num_cols].copy()
y = train[TARGET].copy()
X_test = test[num_cols].copy()

X.shape, X_test.shape, len(num_cols)


kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

feature_importance = pd.DataFrame({"feature": num_cols, "importance": 0.0})

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    if LGBM_AVAILABLE:
        model = LGBMRegressor(
            n_estimators=4000,
            learning_rate=0.03,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
        )
        # feature importance
        feature_importance["importance"] += model.feature_importances_ / kf.n_splits
    else:
        model = RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        model.fit(X_tr, y_tr)

    va_pred = model.predict(X_va)
    oof[va_idx] = va_pred
    fold_rmse = rmse(y_va, va_pred)
    print(f"Fold {fold} RMSE: {fold_rmse:.5f}")

    test_preds += model.predict(X_test) / kf.n_splits
    del X_tr, X_va, y_tr, y_va, model
    gc.collect()

oof_rmse = rmse(y, oof)
print(f"\nOOF RMSE: {oof_rmse:.5f}")


if LGBM_AVAILABLE:
    fi = feature_importance.sort_values("importance", ascending=False).reset_index(drop=True)
    display(fi.head(20))
    fi.head(20).plot(kind="bar", x="feature", y="importance", figsize=(10,4))
    plt.title("Top LightGBM feature importances")
    plt.ylabel("Gain"); plt.tight_layout(); plt.show()
else:
    print("Feature importances shown only for LightGBM run.")


if LGBM_AVAILABLE:
    final_model = LGBMRegressor(
        n_estimators=int(1.2 * 4000),  # a bit more trees on full data
        learning_rate=0.03,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    final_model.fit(X, y)
    final_test_preds = final_model.predict(X_test)
else:
    final_model = RandomForestRegressor(
        n_estimators=800, n_jobs=-1, random_state=RANDOM_STATE
    )
    final_model.fit(X, y)
    final_test_preds = final_model.predict(X_test)

print("Full-data model ready.")


submission = pd.DataFrame({
    "id": test[ID],
    "BeatsPerMinute": final_test_preds
})

submission.head()


submission.to_csv("submission.csv", index=False)
print("Saved: submission.csv")

