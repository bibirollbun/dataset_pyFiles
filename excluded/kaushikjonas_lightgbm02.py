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


import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

target = "diagnosed_diabetes"
id_col = "id"

print(train.shape, test.shape)


cat_cols = [
    "gender", "ethnicity", "education_level", "income_level",
    "smoking_status", "employment_status",
    "family_history_diabetes", "hypertension_history", "cardiovascular_history"
]

for col in cat_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")


def add_features(df):
    # Lifestyle interactions
    df["activity_sleep_ratio"] = df["physical_activity_minutes_per_week"] / (df["sleep_hours_per_day"] + 0.1)
    df["screen_sleep_ratio"] = df["screen_time_hours_per_day"] / (df["sleep_hours_per_day"] + 0.1)

    # Metabolic interactions
    df["bmi_age"] = df["bmi"] * df["age"]
    df["waist_bmi_ratio"] = df["waist_to_hip_ratio"] / (df["bmi"] + 1)

    # Blood pressure
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["bp_mean"] = (df["systolic_bp"] + df["diastolic_bp"]) / 2
    df["bp_risk"] = 0.6 * df["systolic_bp"] + 0.4 * df["diastolic_bp"]

    # Cholesterol
    df["chol_hdl_ratio"] = df["cholesterol_total"] / (df["hdl_cholesterol"] + 1)
    df["ldl_hdl_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1)
    df["tg_hdl_ratio"] = df["triglycerides"] / (df["hdl_cholesterol"] + 1)

    # Combined metabolic
    df["metabolic_risk"] = (
        df["bmi"] +
        df["waist_to_hip_ratio"] +
        df["cholesterol_total"] +
        df["triglycerides"]
    )

    # Lifestyle + metabolic
    df["bmi_screen"] = df["bmi"] * df["screen_time_hours_per_day"]
    df["bmi_activity"] = df["bmi"] * df["physical_activity_minutes_per_week"]

    # Socioeconomic interaction (using codes)
    df["income_education"] = df["income_level"].cat.codes * df["education_level"].cat.codes

    # Lipid risk index
    df["lipid_risk"] = df["ldl_cholesterol"] + 0.5 * df["triglycerides"]

    # Lifestyle load
    df["lifestyle_load"] = df["screen_time_hours_per_day"] - df["physical_activity_minutes_per_week"] / 60.0

    return df

train = add_features(train)
test = add_features(test)


X = train.drop(columns=[target, id_col])
y = train[target]

X_test = test.drop(columns=[id_col])
X_test = X_test[X.columns]  # align columns


X_xgb = X.copy()
X_test_xgb = X_test.copy()

for col in cat_cols:
    X_xgb[col] = X_xgb[col].cat.codes
    X_test_xgb[col] = X_test_xgb[col].cat.codes


lgb_params = dict(
    n_estimators=3000,
    learning_rate=0.01,
    num_leaves=128,
    max_depth=-1,
    min_data_in_leaf=40,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l1=2.0,
    lambda_l2=2.0,
    objective="binary",
    random_state=42,
    n_jobs=-1
)

cb_params = dict(
    iterations=2000,
    learning_rate=0.02,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False
)

xgb_params = dict(
    n_estimators=1200,
    learning_rate=0.02,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    tree_method="hist",
    random_state=42
)


seeds = [42, 2024, 7]


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train))
oof_cb = np.zeros(len(train))
oof_xgb = np.zeros(len(train))

test_lgb = np.zeros(len(test))
test_cb = np.zeros(len(test))
test_xgb = np.zeros(len(test))

cat_idx = [X.columns.get_loc(c) for c in cat_cols]


for seed in seeds:
    print(f"\n=== Training with seed {seed} ===")

    lgb_params["random_state"] = seed
    cb_params["random_seed"] = seed
    xgb_params["random_state"] = seed

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"Seed {seed} - Fold {fold}")

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # LightGBM
        lgb = LGBMClassifier(**lgb_params)
        lgb.fit(X_tr, y_tr)
        oof_lgb[val_idx] += lgb.predict_proba(X_val)[:, 1] / len(seeds)
        test_lgb += lgb.predict_proba(X_test)[:, 1] / (len(seeds) * skf.n_splits)

        # CatBoost
        train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
        val_pool = Pool(X_val, y_val, cat_features=cat_idx)

        cb = CatBoostClassifier(**cb_params)
        cb.fit(train_pool, eval_set=val_pool)
        oof_cb[val_idx] += cb.predict_proba(X_val)[:, 1] / len(seeds)
        test_cb += cb.predict_proba(X_test)[:, 1] / (len(seeds) * skf.n_splits)

        # XGBoost
        X_tr_xgb = X_xgb.iloc[tr_idx]
        X_val_xgb = X_xgb.iloc[val_idx]

        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_tr_xgb, y_tr)
        oof_xgb[val_idx] += xgb.predict_proba(X_val_xgb)[:, 1] / len(seeds)
        test_xgb += xgb.predict_proba(X_test_xgb)[:, 1] / (len(seeds) * skf.n_splits)


auc_lgb = roc_auc_score(y, oof_lgb)
auc_cb = roc_auc_score(y, oof_cb)
auc_xgb = roc_auc_score(y, oof_xgb)

print("LightGBM AUC:", auc_lgb)
print("CatBoost AUC:", auc_cb)
print("XGBoost AUC:", auc_xgb)


meta_train = np.vstack([oof_lgb, oof_cb, oof_xgb]).T
meta_test = np.vstack([test_lgb, test_cb, test_xgb]).T

meta_model = LogisticRegression(max_iter=500)
meta_model.fit(meta_train, y)

meta_oof = meta_model.predict_proba(meta_train)[:, 1]
meta_pred = meta_model.predict_proba(meta_test)[:, 1]

print("Meta‑Learner AUC:", roc_auc_score(y, meta_oof))


blender = LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.01,
    num_leaves=64,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    random_state=42,
    objective="binary",
    n_jobs=-1
)

blender.fit(meta_train, y)
final_pred = blender.predict_proba(meta_test)[:, 1]

print("Blender AUC:", roc_auc_score(y, blender.predict_proba(meta_train)[:, 1]))




submission = pd.DataFrame({
    id_col: test[id_col],
    target: final_pred
})

submission.to_csv("/kaggle/working/submission01.csv", index=False)
print("Saved submission.csv")

