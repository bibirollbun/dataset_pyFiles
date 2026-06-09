# strong_kaggle_pipeline_playground_s5e10.py
# Run in Kaggle notebook (GPU not needed). Assumes train.csv, test.csv, sample_submission.csv in working dir.

import os, gc, time, warnings
warnings.filterwarnings('ignore')
# ======================================================
# Predicting Road Accident Risk - Kaggle Playground S5E10
# Stacking Ensemble: LightGBM + XGBoost + CatBoost + Ridge
# ======================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


# ======================================================
# Predicting Road Accident Risk - Kaggle Playground S5E10
# Stacking Ensemble: LightGBM + XGBoost + CatBoost + Ridge
# ======================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# ------------------ Load Data ------------------
train_path = "/kaggle/input/playground-series-s5e10/train.csv"
test_path  = "/kaggle/input/playground-series-s5e10/test.csv"
sample_sub_path = "/kaggle/input/playground-series-s5e10/sample_submission.csv"

df = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

print("Train shape:", df.shape)
print("Test shape:", df_test.shape)

# ------------------ Features & Target ------------------
target = "accident_risk"
features = [col for col in df.columns if col not in ["id", target]]

X = df[features]
y = df[target]
X_test = df_test[features]


# ======================================================
# Helper Functions for OOF Training
# ======================================================

def train_lgb_oof(X, y, X_test, features, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        train_set = lgb.Dataset(X_train[features], label=y_train)
        valid_set = lgb.Dataset(X_valid[features], label=y_valid)

        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "seed": 42,
            "verbose": -1,
        }

        model = lgb.train(
            params,
            train_set,
            valid_sets=[valid_set],
            num_boost_round=10000,
            callbacks=[
                lgb.early_stopping(200),   # ✅ replaces early_stopping_rounds
                lgb.log_evaluation(200)    # log every 200 rounds
            ]
        )

        oof[valid_idx] = model.predict(X_valid[features], num_iteration=model.best_iteration)
        preds += model.predict(X_test[features], num_iteration=model.best_iteration) / n_splits

    return oof, preds


def train_xgb_oof(X, y, X_test, features, n_splits=5):
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(
            n_estimators=10000,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            tree_method="hist"
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=200,
            verbose=False
        )

        oof[val_idx] = model.predict(X_val)
        preds += model.predict(X_test) / n_splits

    score = mean_squared_error(y, oof, squared=False)
    print(f"XGBoost CV RMSE: {score:.5f}")
    return oof, preds


def train_cat_oof(X, y, X_test, features, n_splits=5):
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

        model = CatBoostRegressor(
            iterations=10000,
            learning_rate=0.03,
            depth=8,
            eval_metric="RMSE",
            random_seed=42,
            verbose=False,
            od_type="Iter",
            od_wait=200
        )

        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

        oof[val_idx] = model.predict(X_val)
        preds += model.predict(X_test) / n_splits

    score = mean_squared_error(y, oof, squared=False)
    print(f"CatBoost CV RMSE: {score:.5f}")
    return oof, preds


from sklearn.preprocessing import LabelEncoder

# Encode categorical columns
cat_cols = ["road_type", "lighting", "weather", "time_of_day"]

for col in cat_cols:
    le = LabelEncoder()
    # Fit on train+test to avoid unseen labels
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))



print("Training LightGBM...")
lgb_oof, lgb_pred = train_lgb_oof(X, y, X_test, features)

print("Training XGBoost...")
xgb_oof, xgb_pred = train_xgb_oof(X, y, X_test, features)

print("Training CatBoost...")
cat_oof, cat_pred = train_cat_oof(X, y, X_test, features)



# ======================================================
# Stacking (Meta Model)
# ======================================================
meta_train = np.vstack([lgb_oof, xgb_oof, cat_oof]).T
meta_test = np.vstack([lgb_pred, xgb_pred, cat_pred]).T

print("Meta train shape:", meta_train.shape)
print("Meta test shape:", meta_test.shape)

meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_train, y)
final_pred = meta_model.predict(meta_test)


# ======================================================
# Submission
# ======================================================
submission = pd.DataFrame({
    "id": df_test["id"],
    "accident_risk": np.clip(final_pred, 0, 1)
})
submission.to_csv("submission.csv", index=False)

print("Submission file saved as submission.csv ✅")

