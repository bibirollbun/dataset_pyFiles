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


# ============================================
# LightGBM + Optuna (5-Fold CV, categorical-aware)
# ============================================
import numpy as np, pandas as pd, optuna, lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

SEED, FOLDS, N_TRIALS = 42, 5, 40   # try 80–120 if you have time
np.random.seed(SEED)

def rmse(y_true, y_pred): 
    return mean_squared_error(y_true, y_pred, squared=False)

# --- Load & minimal FE (same as your main notebook) ---
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

y = train["accident_risk"].values
X = train.drop(["id","accident_risk"], axis=1).copy()
X_test = test.drop(["id"], axis=1).copy()

# cast categoricals to string, create a couple of proven FE features (keep in sync with CB)
cat_cols = X.select_dtypes(include=["object","bool","category"]).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype(str)
    X_test[c] = X_test[c].astype(str)

if {"road_type","lighting"}.issubset(X.columns):
    X["road_lighting"] = X["road_type"] + "_" + X["lighting"]
    X_test["road_lighting"] = X_test["road_type"] + "_" + X_test["lighting"]

bw = {"rain","snow","fog","storm"}
if "weather" in X.columns:
    X["bad_weather_flag"] = X["weather"].str.lower().isin(bw).astype(int)
    X_test["bad_weather_flag"] = X_test["weather"].str.lower().isin(bw).astype(int)

if {"holiday","school_season"}.issubset(X.columns):
    X["holiday_school_flag"] = ((X["holiday"]=="True") | (X["school_season"]=="True")).astype(int)
    X_test["holiday_school_flag"] = ((X_test["holiday"]=="True") | (X_test["school_season"]=="True")).astype(int)

if {"road_type","weather","lighting"}.issubset(X.columns):
    X["danger_combo"] = X["road_type"] + "_" + X["weather"] + "_" + X["lighting"]
    X_test["danger_combo"] = X_test["road_type"] + "_" + X_test["weather"] + "_" + X_test["lighting"]

# Update cat cols; cast to 'category' for LGB
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
X_lgb, X_test_lgb = X.copy(), X_test.copy()
for c in cat_cols:
    X_lgb[c] = X_lgb[c].astype("category")
    X_test_lgb[c] = X_test_lgb[c].astype("category")

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

def objective(trial):
    params = dict(
        objective="regression",
        metric="rmse",
        # learning
        learning_rate=trial.suggest_float("learning_rate", 0.012, 0.04, step=0.002),
        n_estimators=trial.suggest_int("n_estimators", 6000, 14000, step=1000),
        # tree shape / complexity
        num_leaves=trial.suggest_int("num_leaves", 48, 128, step=8),
        max_depth=trial.suggest_int("max_depth", -1, 10),  # -1 means no limit
        min_child_samples=trial.suggest_int("min_child_samples", 10, 80, step=5),
        # sampling
        subsample=trial.suggest_float("subsample", 0.7, 1.0, step=0.05),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0, step=0.05),
        # regularization
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 0.8, step=0.1),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 1.2, step=0.1),
        random_state=SEED,
        verbose=-1
    )

    oof = np.zeros(len(X_lgb))
    for tr_idx, va_idx in kf.split(X_lgb, y):
        X_tr, y_tr = X_lgb.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X_lgb.iloc[va_idx], y[va_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=600, verbose=False)]
        )
        oof[va_idx] = model.predict(X_va, num_iteration=model.best_iteration_)
    return rmse(y, np.clip(oof, 0, 1))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("Best params:", study.best_trial.params)
print("Best OOF:", study.best_value)

# Refit with best params and produce submission
best_params = {**study.best_trial.params, "objective": "regression", "metric":"rmse", "random_state": SEED, "verbose": -1}
oof = np.zeros(len(X_lgb))
test_preds = []

for tr_idx, va_idx in kf.split(X_lgb, y):
    X_tr, y_tr = X_lgb.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X_lgb.iloc[va_idx], y[va_idx]

    model = lgb.LGBMRegressor(**best_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=600, verbose=False)]
    )
    oof[va_idx] = model.predict(X_va, num_iteration=model.best_iteration_)
    test_preds.append(model.predict(X_test_lgb, num_iteration=model.best_iteration_))

final_oof = rmse(y, np.clip(oof,0,1))
print(f"Refit OOF RMSE: {final_oof:.5f}")

pred = np.clip(np.mean(np.column_stack(test_preds), axis=1), 0, 1)
pd.DataFrame({"id": test["id"], "accident_risk": pred}).to_csv("submission_lgb_optuna.csv", index=False)
print("Saved: submission_lgb_optuna.csv")


