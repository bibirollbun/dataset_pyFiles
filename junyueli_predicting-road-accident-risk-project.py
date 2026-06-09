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


from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
def find_dataset_files():
    train_path = None
    test_path = None
    for p in INPUT_ROOT.rglob("train.csv"):
        train_path = p
        break
    for p in INPUT_ROOT.rglob("test.csv"):
        test_path = p
        break
    if train_path is None or test_path is None:
        raise FileNotFoundError("Could not find train.csv/test.csv under /kaggle/input. "
                                "Add the competition dataset to your notebook.")
    return train_path, test_path

train_csv, test_csv = find_dataset_files()
train_csv, test_csv



train = pd.read_csv(train_csv)
test  = pd.read_csv(test_csv)

print(train.shape, test.shape)
display(train.head())
display(train.describe())

TARGET = "accident_risk"
IDCOL  = "id"

assert TARGET in train.columns, "Target column not found."
assert IDCOL in train.columns and IDCOL in test.columns, "ID column not found in train/test."
print("Target range:", train[TARGET].min(), "->", train[TARGET].max())



# Categorical / boolean features based on the description
cat_cols = ["road_type", "lighting", "weather", "time_of_day"]
bool_cols = ["road_signs_present", "public_road", "holiday", "school_season"]

# Numerical are everything else (excluding target and id)
num_cols = [c for c in train.columns if c not in cat_cols + bool_cols + [TARGET, IDCOL]]

# Convert booleans to int
for c in bool_cols:
    if c in train.columns:
        train[c] = train[c].astype(int)
    if c in test.columns:
        test[c] = test[c].astype(int)

def add_simple_interactions(df):
    out = df.copy()
    if set(["curvature","speed_limit"]).issubset(out.columns):
        out["curve_x_speed"] = out["curvature"] * out["speed_limit"]
    if set(["num_reported_accidents","num_lanes"]).issubset(out.columns):
        denom = out["num_lanes"].replace(0, np.nan)
        out["acc_per_lane"] = (out["num_reported_accidents"] / denom).fillna(0)
    return out

train_fe = add_simple_interactions(train)
test_fe  = add_simple_interactions(test)

features = [c for c in train_fe.columns if c not in [TARGET, IDCOL]]
print("Feature count:", len(features))



# --- Fixed CatBoost CV cell: regression target + integer stratification labels ---

# (Re)imports
try:
    from catboost import CatBoostRegressor, Pool
except Exception:
    !pip -q install catboost
    from catboost import CatBoostRegressor, Pool

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
import numpy as np, pandas as pd, os, gc

# 1) Clean target as float, 1-D
y = pd.to_numeric(train_fe["accident_risk"], errors="coerce").astype(np.float32).values
X = train_fe[features].copy()
X_test = test_fe[features].copy()

# Safety checks
if np.isnan(y).any():
    raise ValueError("Found NaNs in target after coercion. Please inspect train_fe['accident_risk'].")

# 2) CatBoost categorical indices
cat_cols_present = [c for c in ["road_type","lighting","weather","time_of_day"] if c in X.columns]
cat_idx = [X.columns.get_loc(c) for c in cat_cols_present]

# 3) Build integer stratification labels to avoid 'unknown' target type errors
# Use rank to avoid identical bin edges, then qcut -> integer codes
ranks = pd.Series(y).rank(method="first")  # strictly increasing
y_bins = pd.qcut(ranks, q=50, duplicates="drop", labels=False).astype(int)

# If stratified split still fails for any reason, we’ll fall back to plain KFold
use_stratified = True
try:
    _ = np.bincount(y_bins)  # quick sanity check
except Exception:
    use_stratified = False

# 4) GPU if available
task_type = "GPU" if os.path.exists('/proc/driver/nvidia/version') else "CPU"
print("Using task_type:", task_type)

params = dict(
    loss_function="RMSE",      # regression
    eval_metric="RMSE",
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    random_seed=42,
    logging_level="Verbose",
    early_stopping_rounds=200,
    iterations=20000,          # rely on early stopping
    task_type=task_type
)

n_splits = 5
if use_stratified:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    split_iter = splitter.split(X, y_bins)
else:
    print("StratifiedKFold fallback → KFold (could not build valid strat labels).")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    split_iter = splitter.split(X)

oof = np.zeros(len(X), dtype=np.float32)
preds = np.zeros(len(X_test), dtype=np.float32)

for fold, idxs in enumerate(split_iter, 1):
    if use_stratified:
        trn_idx, val_idx = idxs
    else:
        trn_idx, val_idx = idxs

    print(f"\n===== Fold {fold} =====")
    X_tr, X_va = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_va = y[trn_idx], y[val_idx]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    valid_pool = Pool(X_va, y_va, cat_features=cat_idx)

    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=200)

    va_pred = model.predict(valid_pool).astype(np.float32)
    oof[val_idx] = va_pred
    rmse = mean_squared_error(y_va, va_pred, squared=False)
    print(f"[Fold {fold}] RMSE: {rmse:.6f}")

    test_pool = Pool(X_test, cat_features=cat_idx)
    preds += model.predict(test_pool).astype(np.float32) / n_splits

    del model, train_pool, valid_pool, test_pool
    gc.collect()

print("\nOOF RMSE:", mean_squared_error(y, oof, squared=False))



# Quick feature importance view on a smaller iteration count
full_pool = Pool(X, y, cat_features=cat_idx)
model_full = CatBoostRegressor(**{**params, "iterations": min(2000, params["iterations"])})
model_full.fit(full_pool, verbose=200)

imp = pd.DataFrame({
    "feature": X.columns,
    "importance": model_full.get_feature_importance(full_pool)
}).sort_values("importance", ascending=False)
imp.head(20)



sub = test[[IDCOL]].copy()
sub["accident_risk"] = np.clip(preds, 0.0, 1.0)
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Saved:", "/kaggle/working/submission.csv")
sub.head()



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

cat_cols = [c for c in ["road_type","lighting","weather","time_of_day"] if c in train.columns]
bool_cols = [c for c in ["road_signs_present","public_road","holiday","school_season"] if c in train.columns]
num_cols  = [c for c in train.columns if c not in cat_cols + bool_cols + [IDCOL, TARGET]]

# Use engineered columns too if present
extra_num = [c for c in ["curve_x_speed","acc_per_lane"] if c in train_fe.columns]

pre = ColumnTransformer(
    transformers=[
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", "passthrough", num_cols + bool_cols + extra_num),
    ]
)

hgb = HistGradientBoostingRegressor(
    learning_rate=0.07,
    max_leaf_nodes=63,
    max_depth=6,
    min_samples_leaf=100,
    max_bins=255,
    early_stopping=True,
    random_state=42
)

pipe = Pipeline([("pre", pre), ("model", hgb)])

X_train, X_valid, y_train, y_valid = train_test_split(train_fe[features], train_fe[TARGET], test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)
val_pred = pipe.predict(X_valid)
print("HGB Val RMSE:", mean_squared_error(y_valid, val_pred, squared=False))

test_pred = pipe.predict(test_fe[features])
sub_hgb = test[[IDCOL]].copy()
sub_hgb["accident_risk"] = np.clip(test_pred, 0, 1)
sub_hgb.to_csv("/kaggle/working/submission_hgbr.csv", index=False)
print("Saved:", "/kaggle/working/submission_hgbr.csv")


