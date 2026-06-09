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


# Final optimized stacked ensemble pipeline for Calories Burned prediction
# Target RMSLE < 0.054 on Kaggle Leaderboard

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.pipeline import make_pipeline

# === Load Data ===
df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

# === Preprocessing ===
def preprocess(df):
    df = df.copy()
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    df["Height_Weight"] = df["Height"] * df["Weight"]
    df["Age_HeartRate"] = df["Age"] * df["Heart_Rate"]
    df["Duration_BodyTemp"] = df["Duration"] * df["Body_Temp"]
    df["Age^2"] = df["Age"] ** 2
    df["Duration^2"] = df["Duration"] ** 2
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    return df

X = preprocess(df.drop(columns=["id", "Calories"]))
y = np.log1p(df["Calories"])
X_test = preprocess(df_test.drop(columns=["id"]))

# === Stratified CV ===
bins = pd.qcut(y, q=5, labels=False)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# === OOF and Test Placeholders ===
oof_preds_lgb, oof_preds_xgb, oof_preds_cat = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
test_preds_lgb, test_preds_xgb, test_preds_cat = [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, bins)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    lgb = LGBMRegressor(n_estimators=1000, learning_rate=0.01, num_leaves=128, subsample=0.8,
                        colsample_bytree=0.8, random_state=42)
    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
    oof_preds_lgb[val_idx] = lgb.predict(X_val)
    test_preds_lgb.append(lgb.predict(X_test))

    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, subsample=0.8,
                       colsample_bytree=0.8, random_state=42)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    oof_preds_xgb[val_idx] = xgb.predict(X_val)
    test_preds_xgb.append(xgb.predict(X_test))

    cat = CatBoostRegressor(n_estimators=1000, learning_rate=0.01, depth=6, random_seed=42,
                             verbose=0, task_type="GPU")
    cat.fit(X_train, y_train, eval_set=(X_val, y_val))
    oof_preds_cat[val_idx] = cat.predict(X_val)
    test_preds_cat.append(cat.predict(X_test))

# === Meta Features ===
oof_meta = np.vstack([oof_preds_lgb, oof_preds_xgb, oof_preds_cat]).T
test_meta = np.vstack([
    np.mean(test_preds_lgb, axis=0),
    np.mean(test_preds_xgb, axis=0),
    np.mean(test_preds_cat, axis=0)
]).T

# === Meta Model ===
meta_model = make_pipeline(StandardScaler(), RidgeCV())
meta_model.fit(oof_meta, y)
final_preds_log = meta_model.predict(test_meta)
final_preds_log = np.clip(final_preds_log, 0, None)
final_preds = np.expm1(final_preds_log)

# === Submission ===
sub = pd.DataFrame({"id": df_test["id"], "Calories": final_preds})
sub.to_csv("stacked_ensemble_submission.csv", index=False)

# === OOF RMSLE ===
oof_score = np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(meta_model.predict(oof_meta))))
print("OOF RMSLE (Stacked Ensemble):", round(oof_score, 5))


