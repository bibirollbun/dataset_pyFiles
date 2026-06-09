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


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

SEED = 42
N_SPLITS = 5
TARGET = "loan_paid_back"
ID = "id"
VERBOSE = 200

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")

# Feature Engineering
def add_features(df):
    df = df.copy()
    df["grade"] = df["grade_subgrade"].str[0]
    df["loan_to_income"] = df["loan_amount"] / (df["annual_income"] + 1e-9)
    df["int_x_loan"] = df["interest_rate"] * df["loan_amount"]
    df["dti_bin"] = pd.cut(df["debt_to_income_ratio"], bins=8, labels=False).astype("int")
    df["log_annual_income"] = np.log1p(df["annual_income"])
    df["log_loan_amount"] = np.log1p(df["loan_amount"])
    return df

train = add_features(train)
test  = add_features(test)

# Categorical identification

cat_cols = train.select_dtypes(include="object").columns.tolist()
cat_cols += ["grade"]
cat_cols += ["dti_bin"]
cat_cols = list(set(cat_cols))


# Target Encoding (Leakage-safe)

def target_encode(train_df, test_df, cols, target, n_splits=5, smoothing=50):
    train_new = train_df.copy()
    test_new = test_df.copy()

    global_mean = train_df[target].mean()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for col in cols:
        oof = np.zeros(len(train_df))
        test_encoded = np.zeros((len(test_df), n_splits))

        for i, (tr_idx, val_idx) in enumerate(skf.split(train_df, train_df[target])):
            tr, val = train_df.iloc[tr_idx], train_df.iloc[val_idx]

            stats = tr.groupby(col)[target].agg(["mean", "count"])
            stats.columns = ["mean_target", "count"]

            val2 = val.merge(stats, left_on=col, right_index=True, how="left")
            m = val2["mean_target"].fillna(global_mean)
            c = val2["count"].fillna(0)

            smooth = (c * m + smoothing * global_mean) / (c + smoothing)
            oof[val_idx] = smooth.values

            test2 = test_df.merge(stats, left_on=col, right_index=True, how="left")
            m2 = test2["mean_target"].fillna(global_mean)
            c2 = test2["count"].fillna(0)
            smooth2 = (c2 * m2 + smoothing * global_mean) / (c2 + smoothing)
            test_encoded[:, i] = smooth2.values

        train_new[col + "_te"] = oof
        test_new[col + "_te"] = test_encoded.mean(axis=1)

    return train_new, test_new


te_cols = [c for c in cat_cols if train[c].nunique() < 500]

train_te, test_te = target_encode(train, test, te_cols, TARGET, n_splits=N_SPLITS)

# Prepare inputs for tree models

all_features = [c for c in train_te.columns if c not in [ID, TARGET]]

X_num = train_te[all_features].copy()
X_test_num = test_te[all_features].copy()

# Label encode categorical variables for LGB/XGB
for c in X_num.columns:
    if X_num[c].dtype == "object":
        le = LabelEncoder()
        full = pd.concat([X_num[c].astype(str), X_test_num[c].astype(str)])
        le.fit(full)
        X_num[c] = le.transform(X_num[c].astype(str))
        X_test_num[c] = le.transform(X_test_num[c].astype(str))

y = train[TARGET].values

# Stacking containers

oof_cat = np.zeros(len(train))
oof_lgb = np.zeros(len(train))
oof_xgb = np.zeros(len(train))

pred_cat = np.zeros(len(test))
pred_lgb = np.zeros(len(test))
pred_xgb = np.zeros(len(test))


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# Training Loop

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_num, y), 1):

    print(f"\n============ FOLD {fold} ============")

    X_tr_num, X_val_num = X_num.iloc[tr_idx], X_num.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    X_tr_cb = train_te.iloc[tr_idx][all_features]
    X_val_cb = train_te.iloc[val_idx][all_features]
    X_test_cb = test_te[all_features]

    # 1. CatBoost
    
    cb_model = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3,
        eval_metric="AUC",
        random_seed=SEED,
        early_stopping_rounds=200,
        use_best_model=True,
        verbose=VERBOSE
    )

    cb_model.fit(
        X_tr_cb, y_tr,
        eval_set=(X_val_cb, y_val),
        cat_features=[c for c in all_features if c in cat_cols],
        verbose=VERBOSE
    )

    oof_cat[val_idx] = cb_model.predict_proba(X_val_cb)[:, 1]
    pred_cat += cb_model.predict_proba(X_test_cb)[:, 1] / N_SPLITS

    
    # 2. LightGBM 
    params_lgb = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.5,
        "lambda_l2": 1.0,
        "seed": SEED,
        "verbosity": -1
    }

    dtrain = lgb.Dataset(X_tr_num, label=y_tr)
    dvalid = lgb.Dataset(X_val_num, label=y_val)

    callbacks = [
        lgb.early_stopping(stopping_rounds=150, verbose=True),
        lgb.log_evaluation(200)
    ]

    lgb_model = lgb.train(
        params_lgb,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dtrain, dvalid],
        callbacks=callbacks
    )

    oof_lgb[val_idx] = lgb_model.predict(
        X_val_num, num_iteration=lgb_model.best_iteration
    )
    pred_lgb += lgb_model.predict(
        X_test_num, num_iteration=lgb_model.best_iteration
    ) / N_SPLITS

   
    # 3. XGBoost
 
    xgb_model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        eval_metric="auc",
        random_state=SEED,
        tree_method="hist"
    )

    xgb_model.fit(
        X_tr_num, y_tr,
        eval_set=[(X_val_num, y_val)],
        early_stopping_rounds=150,
        verbose=False
    )

    oof_xgb[val_idx] = xgb_model.predict_proba(X_val_num)[:, 1]
    pred_xgb += xgb_model.predict_proba(X_test_num)[:, 1] / N_SPLITS

    # Scores
    blend_val = (oof_cat[val_idx] + oof_lgb[val_idx] + oof_xgb[val_idx]) / 3
    print(f"Fold {fold}: Cat={roc_auc_score(y_val, oof_cat[val_idx]):.5f}, "
          f"LGB={roc_auc_score(y_val, oof_lgb[val_idx]):.5f}, "
          f"XGB={roc_auc_score(y_val, oof_xgb[val_idx]):.5f}, "
          f"Blend={roc_auc_score(y_val, blend_val):.5f}")

# Stacking (meta-model)

X_meta_train = np.vstack([oof_cat, oof_lgb, oof_xgb]).T
X_meta_test  = np.vstack([pred_cat, pred_lgb, pred_xgb]).T

meta = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
meta.fit(X_meta_train, y)

meta_oof = meta.predict_proba(X_meta_train)[:, 1]
print("\nMeta OOF AUC:", roc_auc_score(y, meta_oof))

# Submission

final_preds = meta.predict_proba(X_meta_test)[:, 1]

submission = sample.copy()
submission[TARGET] = final_preds
submission.to_csv("/kaggle/working/submission.csv", index=False)

print("Saved → /kaggle/working/submission.csv")


