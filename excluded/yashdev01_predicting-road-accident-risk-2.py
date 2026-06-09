import os
import gc
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import scipy.stats as st
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


pd.set_option("display.max_columns", None)
plt.rcParams["figure.dpi"] = 140

SEED = 42
np.random.seed(SEED)


def reduce_memory_usage(df: pd.DataFrame, verbose=True):
    start_memory = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type)[:3] != 'dat':
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_memory = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory reduced from {start_memory:.2f} MB to {end_memory:.2f} MB ({100*(start_memory - end_memory)/start_memory:.1f}% saved)")
    return df


TRAIN_PATH = "/kaggle/input/playground-series-s5e10/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e10/test.csv"
SYNTH_BASE = "/kaggle/input/simulated-roads-accident-data" 

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
print("train", train.shape, "test", test.shape)

orig_list = []
for k in [2, 10, 100]:
    path = f"{SYNTH_BASE}/synthetic_road_accidents_{k}k.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        orig_list.append(df)
    else:
        print("Missing synthetic file:", path)
if len(orig_list) == 0:
    raise FileNotFoundError("No synthetic/orig files found in provided path. Adjust SYNTH_BASE or filenames.")
orig = pd.concat(orig_list, axis=0, ignore_index=True)

orig['id'] = np.arange(len(orig)) + int(test['id'].max()) + 1

orig = orig[train.columns.intersection(orig.columns).tolist()]
print("orig shape:", orig.shape)

# reduce mem
train = reduce_memory_usage(train)
test = reduce_memory_usage(test)
orig = reduce_memory_usage(orig)


combine = pd.concat([train, test, orig], axis=0, ignore_index=True)
print("Combined shape:", combine.shape)

def f_baseline(X):
    return (
        0.3 * X["curvature"].fillna(0).astype(float)
        + 0.2 * (X["lighting"] == "night").astype(int)
        + 0.1 * (X["weather"] != "clear").astype(int)
        + 0.2 * (X["speed_limit"].fillna(0).astype(float) >= 60).astype(int)
        + 0.1 * (X["num_reported_accidents"].fillna(0).astype(float) > 2).astype(int)
    )

def clip_func(mu, sigma=0.05):
    a = (-mu) / sigma
    b = (1 - mu) / sigma
    Phi_a, Phi_b = st.norm.cdf(a), st.norm.cdf(b)
    phi_a, phi_b = st.norm.pdf(a), st.norm.pdf(b)
    return mu * (Phi_b - Phi_a) + sigma * (phi_a - phi_b) + 1 - Phi_b

mu = f_baseline(combine)
combine["y"] = clip_func(mu)

n_train = len(train)
n_test = len(test)
train["y"] = combine.loc[: n_train - 1, "y"].values
test["y"]  = combine.loc[n_train : n_train + n_test - 1, "y"].values

print("Baseline y computed and attached to train/test.")


TARGET = "accident_risk"

EXCLUDE = ["id", TARGET]
FEATURES = [c for c in train.columns if c not in EXCLUDE]
print("Number of FEATURES:", len(FEATURES))

# categorize features
CATS = []
NUMS = []
for c in FEATURES:
    if train[c].dtype == "object" or str(train[c].dtype).startswith("category"):
        CATS.append(c)
    else:
        NUMS.append(c)
print("CATS:", CATS)
print("NUMS (sample):", NUMS[:10])


TE = []
for c in FEATURES:
    if c not in orig.columns:
        continue
    tmp = orig.groupby(c)[TARGET].mean()
    n = f"TE_{c}"
    print(f"Creating TE: {n}")
    tmp.name = n
    train = train.merge(tmp, on=c, how='left')
    test  = test.merge(tmp, on=c, how='left')
    TE.append(n)

global_mean = orig[TARGET].mean() if TARGET in orig.columns else train[TARGET].mean()
for tcol in TE:
    train[tcol] = train[tcol].fillna(global_mean)
    test[tcol]  = test[tcol].fillna(global_mean)

print("Target encoding finished. TE columns count:", len(TE))



FEATURES_PLUS_TE = [c for c in FEATURES if c in train.columns] + TE
if "y" in FEATURES_PLUS_TE:
    FEATURES_PLUS_TE.remove("y")
print("Total features used for XGBoost:", len(FEATURES_PLUS_TE))


FOLDS = 7
SEED = SEED

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.001,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.6,
    "seed": SEED,
    "tree_method": "gpu_hist",        
    "predictor": "gpu_predictor",     
    "device": "cuda",                 
}

print("✅ Using GPU acceleration for XGBoost")
print("Parameters:")
for k, v in params.items():
    print(f"  {k}: {v}")

# Arrays for predictions
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print("\n" + "#" * 40)
    print(f"### Fold {fold + 1} / {FOLDS} ###")
    print("#" * 40)

    # Split features and labels
    X_train = train.iloc[train_idx][FEATURES_PLUS_TE].copy()
    X_valid = train.iloc[val_idx][FEATURES_PLUS_TE].copy()

    # Residual target = actual - baseline
    y_train = train.iloc[train_idx][TARGET].values - train.iloc[train_idx]["y"].values
    y_valid = train.iloc[val_idx][TARGET].values - train.iloc[val_idx]["y"].values

    # Baseline portions to add back
    y_valid_base = train.iloc[val_idx]["y"].values
    y_test_base  = test["y"].values

    # Convert object/category columns properly
    for df in [X_train, X_valid]:
        for c in df.select_dtypes(include=["object"]).columns:
            df[c] = df[c].astype("category")

    X_test = test[FEATURES_PLUS_TE].copy()
    for c in X_test.select_dtypes(include=["object"]).columns:
        X_test[c] = X_test[c].astype("category")

    # DMatrix for GPU
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval   = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
    dtest  = xgb.DMatrix(X_test, enable_categorical=True)

    # Train model with GPU
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100_000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=200,
        verbose_eval=200,
    )

    # Predict residuals and add baseline
    pred_val_resid = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    pred_test_resid = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

    oof_preds[val_idx] = pred_val_resid + y_valid_base
    test_preds += (pred_test_resid + y_test_base) / FOLDS

    # Fold RMSE
    fold_rmse = mean_squared_error(train.iloc[val_idx][TARGET].values, oof_preds[val_idx], squared=False)
    print(f"✅ Fold {fold + 1} RMSE: {fold_rmse:.6f}")

    del X_train, X_valid, X_test, dtrain, dval, dtest, model
    gc.collect()



oof_rmse = mean_squared_error(train[TARGET].values, oof_preds, squared=False)
print(f"\nFinal OOF RMSE: {oof_rmse:.6f}")

submission = pd.DataFrame({"id": test["id"].values, "accident_risk": test_preds})
submission.to_csv("submission_xgb_residuals.csv", index=False)
print("Saved: submission_xgb_residuals.csv")

plt.hist(submission["accident_risk"], bins=100)
plt.title("Histogram of Test Predictions")
plt.xlabel("accident_risk")
plt.show()


submission.head()




