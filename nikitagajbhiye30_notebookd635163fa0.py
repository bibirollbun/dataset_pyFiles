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


# =========================================================
#  PLAYGROUND SERIES S5E11 - STRONG LGBM BASELINE
# =========================================================

import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# --------------- CONFIG ----------------
INPUT_DIR = "/kaggle/input/playground-series-s5e11"
OUTPUT_PATH = "/kaggle/working/submission.csv"
RANDOM_STATE = 42
N_SPLITS = 5
SEED_LIST = [42, 2025, 7]
# ---------------------------------------

# ========== LOAD DATA ==========
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

print("train:", train.shape, " test:", test.shape)
print(train.head())

# ========== BASIC PREPROCESSING ==========
def basic_preprocessing(train, test, target_col="loan_paid_back"):
    y = train[target_col].astype(int).copy()
    train = train.drop(columns=[target_col])
    train["__is_train"] = 1
    test["__is_train"] = 0
    df = pd.concat([train, test], ignore_index=True).reset_index(drop=True)
    return df, y

# ========== FEATURE ENGINEERING ==========
def feature_engineering(df):
    # log transforms
    for col in ["annual_income", "loan_amount", "interest_rate", "debt_to_income_ratio"]:
        if col in df.columns:
            df[col + "_log1p"] = np.log1p(df[col].clip(lower=0).astype(float))

    # credit banding
    if "credit_score" in df.columns:
        df["credit_score_bin"] = pd.cut(
            df["credit_score"].fillna(-1),
            bins=[-1, 549, 649, 699, 749, 799, 849, 1000],
            labels=False,
            include_lowest=True,
        ).astype(float)

    # ratio
    if "loan_amount" in df.columns and "annual_income" in df.columns:
        df["loan_income_ratio"] = df["loan_amount"] / (df["annual_income"].replace(0, np.nan) + 1e-9)

    # interaction feature
    if "grade_subgrade" in df.columns and "loan_purpose" in df.columns:
        df["grade_purpose"] = df["grade_subgrade"].astype(str) + "_" + df["loan_purpose"].astype(str)

    # frequency encoding
    cat_cols = [
        "gender", "marital_status", "education_level",
        "employment_status", "loan_purpose", "grade_subgrade", "grade_purpose"
    ]
    for c in cat_cols:
        if c in df.columns:
            vc = df[c].fillna("##na##").astype(str).value_counts(dropna=False)
            df[c + "_freq"] = df[c].fillna("##na##").astype(str).map(vc).astype(float)

    # nulls & zeros
    df["num_nulls"] = df.isna().sum(axis=1).astype(float)
    df["num_zero"] = (df == 0).sum(axis=1).astype(float)
    return df

# ========== LABEL ENCODING ==========
def label_encode_columns(df, cols):
    encoders = {}
    for c in cols:
        if c in df.columns:
            le = LabelEncoder()
            df[c] = df[c].fillna("##na##").astype(str)
            df[c] = le.fit_transform(df[c])
            encoders[c] = le
    return df, encoders

# ========== K-FOLD TARGET ENCODING ==========
def kfold_target_encode(df, y, cols, n_splits=5, seed=42, smooth=20):
    df = df.copy()
    train_idx = df["__is_train"] == 1
    train_positions = np.where(train_idx)[0]
    y_series = pd.Series(y, index=train_positions)

    for col in cols:
        if col not in df.columns:
            continue
        te_col = col + "_te"
        df[te_col] = np.nan
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        labels = y_series.values

        for tr, val in skf.split(train_positions, labels):
            tr_idx = train_positions[tr]
            val_idx = train_positions[val]
            stats = (pd.Series(y[tr], index=df.loc[tr_idx, col].values)
                     .groupby(level=0).agg(['mean','count']))
            means = stats['mean']
            counts = stats['count']
            global_mean = y.mean()
            smooth_vals = (means * counts + global_mean * smooth) / (counts + smooth)
            df.loc[val_idx, te_col] = df.loc[val_idx, col].map(smooth_vals)

        full_stats = (pd.Series(y, index=df.loc[train_positions, col].values)
                      .groupby(level=0).agg(['mean','count']))
        means = full_stats['mean']
        counts = full_stats['count']
        global_mean = y.mean()
        smooth_vals = (means * counts + global_mean * smooth) / (counts + smooth)
        df.loc[df["__is_train"] == 0, te_col] = df.loc[df["__is_train"] == 0, col].map(smooth_vals)
        df[te_col] = df[te_col].fillna(global_mean)
    return df

# ========== PIPELINE ==========
df, y = basic_preprocessing(train, test)
df = feature_engineering(df)

le_cols = ["gender", "marital_status", "education_level", "employment_status", "loan_purpose", "grade_subgrade"]
df, encoders = label_encode_columns(df, le_cols + ["grade_purpose"])

te_cols = ["gender", "marital_status", "education_level", "employment_status", "loan_purpose", "grade_subgrade", "grade_purpose"]
df = kfold_target_encode(df, y.values, te_cols, n_splits=N_SPLITS, seed=RANDOM_STATE, smooth=100)

features = [c for c in df.columns if c not in {"id", "__is_train"}]
cat_features = [c for c in le_cols if c in features]
if "grade_purpose" in features:
    cat_features.append("grade_purpose")

print("Number of features:", len(features))
print("Categorical features passed to LGBM:", cat_features)

# ========== MODEL TRAINING ==========
def train_lgbm_ensemble(df, y, features, cat_features, seeds=SEED_LIST, n_splits=N_SPLITS):
    train_mask = df["__is_train"] == 1
    X_all = df[features].copy()
    X_train = X_all.loc[train_mask].reset_index(drop=True)
    X_test = X_all.loc[~train_mask].reset_index(drop=True)

    oof_preds = np.zeros(X_train.shape[0])
    test_preds = np.zeros(X_test.shape[0])

    for seed in seeds:
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_data_in_leaf": 20,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "seed": seed,
            "verbosity": -1,
            "n_jobs": -1,
        }

        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr_idx, val_idx) in enumerate(folds.split(X_train, y)):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            lgb_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_features, free_raw_data=False)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_tr, categorical_feature=cat_features, free_raw_data=False)

            clf = lgb.train(
                params,
                lgb_tr,
                num_boost_round=3000,
                valid_sets=[lgb_tr, lgb_val],
                callbacks=[lgb.early_stopping(stopping_rounds=150), lgb.log_evaluation(period=500)]
            )

            oof_preds[val_idx] += clf.predict(X_val, num_iteration=clf.best_iteration) / len(seeds)
            test_preds += clf.predict(X_test, num_iteration=clf.best_iteration) / (n_splits * len(seeds))

            del clf, lgb_tr, lgb_val
            gc.collect()

    auc = roc_auc_score(y, oof_preds)
    print(f"OOF AUC: {auc:.6f}")
    return oof_preds, test_preds

oof, preds = train_lgbm_ensemble(df, y.values, features, cat_features)

# ========== SUBMISSION ==========
submission = sample.copy()
submission["loan_paid_back"] = preds.clip(0, 1)
submission.to_csv(OUTPUT_PATH, index=False)
print("Submission saved to:", OUTPUT_PATH)
submission.head()


