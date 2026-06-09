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


# Step 0: Setup & Imports

import os, gc, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

try:
    import lightgbm as lgb
except ImportError as e:
    raise SystemExit(
        "LightGBM not found. On Kaggle it should be available by default.\n"
        "If running locally, install with: pip install lightgbm"
    )

pd.set_option("display.max_columns", 200)
np.random.seed(42)

print("Versions -> pandas", pd.__version__, "| numpy", np.__version__)



# Step 1: Load Data (auto-detect Kaggle path)

INPUT_DIR = "/kaggle/input/home-credit-default-risk"
TRAIN_PATH = os.path.join(INPUT_DIR, "application_train.csv")
TEST_PATH  = os.path.join(INPUT_DIR, "application_test.csv")

# Fallback for local use
if not os.path.exists(TRAIN_PATH):
    INPUT_DIR = "../input/home-credit-default-risk"
    TRAIN_PATH = os.path.join(INPUT_DIR, "application_train.csv")
    TEST_PATH  = os.path.join(INPUT_DIR, "application_test.csv")

print("Reading CSVs...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print("train:", train.shape, "test:", test.shape)

print("\nTARGET distribution (proportion):")
print(train["TARGET"].value_counts(normalize=True).rename("proportion"))



# Step 2: Quick EDA (light checks)

print("\nSample rows:")
display(train.head(3))

print("\nMissingness (top 10):")
miss = train.isna().mean().sort_values(ascending=False).head(10)
display(miss.to_frame("missing_ratio"))

print("\nData types (counts):")
display(train.dtypes.value_counts())



# Step 3: Minimal, Robust Feature Engineering

def safe_mean(df, cols):
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series([np.nan]*len(df), index=df.index)
    return df[existing].mean(axis=1)

def add_basic_features(df):
    # Known anomaly in DAYS_EMPLOYED -> treat as missing + flag
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
        df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    
    # Ratios
    if {"AMT_CREDIT","AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1e-9)
    if {"AMT_ANNUITY","AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1e-9)
    if {"AMT_ANNUITY","AMT_CREDIT"}.issubset(df.columns):
        df["CREDIT_TERM"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1e-9)

    # Average of ext sources (robust to missing)
    df["EXT_SOURCES_MEAN"] = safe_mean(df, ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"])
    return df

train = add_basic_features(train)
test  = add_basic_features(test)

# Split features/target
y = train["TARGET"].astype(int)
X = train.drop(columns=["TARGET"])
X_test = test.copy()

print("Feature count (train):", X.shape[1])



# Step 4: Treat categoricals as pandas Categorical and align categories across train/test

cat_cols = [c for c in X.columns if X[c].dtype == "object"]

if len(cat_cols) > 0:
    combo = pd.concat([X[cat_cols], X_test[cat_cols]], axis=0)
    for c in cat_cols:
        cats = combo[c].astype("category").cat.categories
        X[c] = X[c].astype("category").cat.set_categories(cats)
        X_test[c] = X_test[c].astype("category").cat.set_categories(cats)

print(f"Categorical columns detected: {len(cat_cols)}")
print(cat_cols[:20])  # show a sample



# Step 5: LightGBM with 5-fold Stratified CV + AUC (callback-style early stopping)

# Optional: class imbalance helper (can be tuned)
neg, pos = (y == 0).sum(), (y == 1).sum()
scale_pos_weight = neg / max(pos, 1)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 50,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "verbose": -1,
    # Uncomment to try imbalance handling:
    # "scale_pos_weight": scale_pos_weight,
}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))
best_iterations = []
feature_importances = pd.DataFrame({"feature": X.columns})

for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), 1):
    X_tr, X_va = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[trn_idx], y.iloc[val_idx]

    # Build datasets; only pass categorical_feature if we actually have any
    ds_tr_kwargs = {"label": y_tr, "free_raw_data": False}
    ds_va_kwargs = {"label": y_va, "free_raw_data": False}
    if len(cat_cols) > 0:
        ds_tr_kwargs["categorical_feature"] = cat_cols
        ds_va_kwargs["categorical_feature"] = cat_cols

    lgb_tr = lgb.Dataset(X_tr, **ds_tr_kwargs)
    lgb_va = lgb.Dataset(X_va, **ds_va_kwargs)

    model = lgb.train(
        params,
        lgb_tr,
        num_boost_round=10000,
        valid_sets=[lgb_tr, lgb_va],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),  # replaces early_stopping_rounds=
            lgb.log_evaluation(period=200),           # replaces verbose_eval=
        ],
    )

    best_iterations.append(model.best_iteration)
    oof[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_pred += model.predict(X_test, num_iteration=model.best_iteration) / folds.n_splits

    feature_importances[f"fold_{fold}"] = model.feature_importance(importance_type="gain")

    del X_tr, X_va, y_tr, y_va, lgb_tr, lgb_va, model
    gc.collect()

cv_auc = roc_auc_score(y, oof)
print(f"\nCV AUC: {cv_auc:.5f}")
print("Best iterations per fold:", best_iterations, "| mean:", int(np.mean(best_iterations)))



# Step 6: Plot top feature importances (average gain)

fi_cols = [c for c in feature_importances.columns if c.startswith("fold_")]
feature_importances["avg_gain"] = feature_importances[fi_cols].mean(axis=1)
feature_importances = feature_importances.sort_values("avg_gain", ascending=False)

topn = 40
plt.figure(figsize=(8, max(6, int(topn*0.25))))
plt.barh(feature_importances["feature"].head(topn)[::-1], feature_importances["avg_gain"].head(topn)[::-1])
plt.title("Top Feature Importances (average gain)")
plt.xlabel("Average Gain")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

display(feature_importances.head(20))



# Step 7: Create submission file

submission = pd.DataFrame({
    "SK_ID_CURR": X_test["SK_ID_CURR"],
    "TARGET": test_pred
})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Saved:", submission_path)
display(submission.head())



# (Optional) Step 8: Add features from bureau.csv and re-train (quick recipe)

ADD_SECONDARY_TABLES = False  # <- set to True to enable

if ADD_SECONDARY_TABLES:
    bureau_path = os.path.join(INPUT_DIR, "bureau.csv")
    if not os.path.exists(bureau_path):
        raise FileNotFoundError("bureau.csv not found in input directory")

    print("Loading bureau tables...")
    bureau = pd.read_csv(bureau_path)
    
    # Numeric aggregations
    num_cols = bureau.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "SK_ID_CURR"]
    agg_spec = {c: ["mean","max","min","sum"] for c in num_cols}
    bureau_agg = bureau.groupby("SK_ID_CURR").agg(agg_spec)
    bureau_agg.columns = ["BUREAU_" + "_".join(col).upper() for col in bureau_agg.columns.ravel()]
    bureau_agg.reset_index(inplace=True)
    
    # Merge
    key = "SK_ID_CURR"
    X_ext = X.merge(bureau_agg, on=key, how="left")
    X_test_ext = X_test.merge(bureau_agg, on=key, how="left")
    
    # Re-train
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof2 = np.zeros(len(X_ext))
    test_pred2 = np.zeros(len(X_test_ext))
    feature_importances2 = pd.DataFrame({"feature": X_ext.columns})
    best_iterations2 = []
    
    lgb_cat_cols = [c for c in X_ext.columns if X_ext[c].dtype.name == "category"]
    
    for fold, (trn_idx, val_idx) in enumerate(folds.split(X_ext, y), 1):
        X_tr, X_va = X_ext.iloc[trn_idx], X_ext.iloc[val_idx]
        y_tr, y_va = y.iloc[trn_idx], y.iloc[val_idx]
        
        lgb_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=lgb_cat_cols, free_raw_data=False)
        lgb_va = lgb.Dataset(X_va, label=y_va, categorical_feature=lgb_cat_cols, free_raw_data=False)
        
        model = lgb.train(
            params,
            lgb_tr,
            num_boost_round=10000,
            valid_sets=[lgb_tr, lgb_va],
            valid_names=["train", "valid"],
            early_stopping_rounds=200,
            verbose_eval=200
        )
        
        best_iterations2.append(model.best_iteration)
        oof2[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
        test_pred2 += model.predict(X_test_ext, num_iteration=model.best_iteration) / folds.n_splits
        
        feature_importances2[f"fold_{fold}"] = model.feature_importance(importance_type="gain")
        
        del X_tr, X_va, y_tr, y_va, lgb_tr, lgb_va, model
        gc.collect()
    
    cv_auc2 = roc_auc_score(y, oof2)
    print(f"\nCV AUC with bureau features: {cv_auc2:.5f}")
    
    submission2 = pd.DataFrame({"SK_ID_CURR": X_test_ext["SK_ID_CURR"], "TARGET": test_pred2})
    submission2.to_csv("submission_with_bureau.csv", index=False)
    print("Saved:", "submission_with_bureau.csv")


