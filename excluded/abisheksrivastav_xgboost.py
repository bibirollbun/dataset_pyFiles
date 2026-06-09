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


# Kaggle-ready: XGBoost on residuals using original data solution
import os, gc
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt

# ---------- Paths (update if needed) ----------
TRAIN_PATH = "/kaggle/input/playground-series-s5e10/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e10/test.csv"
SAMPLE_PATH = "/kaggle/input/playground-series-s5e10/sample_submission.csv"
# synthetic/original data files (the 2k/10k/100k example)
ORIG_GLOB = [
    "/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_2k.csv",
    "/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_10k.csv",
    "/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv",
]

# ---------- Config ----------
FOLDS = 7
SEED = 42
DEVICE = "cuda"   # set to "cpu" if no GPU
NUM_BOOST_ROUND = 100_000
EARLY_STOPPING = 200
VERBOSE_EVAL = 200

# ---------- Load ----------
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

# read and concat original/simulated files
orig_list = []
for path in ORIG_GLOB:
    if os.path.exists(path):
        orig_list.append(pd.read_csv(path))
    else:
        print("WARNING: missing original file:", path)
if len(orig_list) == 0:
    raise FileNotFoundError("No original data files found; update ORIG_GLOB path(s).")
orig = pd.concat(orig_list, axis=0, ignore_index=True)

# ensure orig has same columns as train: set ids beyond test id range, align columns
max_test_id = test['id'].max()
orig['id'] = np.arange(len(orig)) + int(max_test_id) + 1

# If orig has extra columns, select same columns as train (conservative)
orig = orig.loc[:, train.columns.intersection(orig.columns)]

# If `accident_risk` missing in orig, try to find target column name; otherwise assume last column is target
if 'accident_risk' not in orig.columns:
    # fall back to last column
    orig.columns = list(orig.columns[:-1]) + ['accident_risk']

# Reindex columns like example: keep order same as train
orig = orig.reindex(columns=train.columns)

print("Train shape:", train.shape)
print("Test shape: ", test.shape)
print("Orig shape: ", orig.shape)

# ---------- Combine as in example ----------
combine = pd.concat([train, test.assign(accident_risk=0.5), orig], axis=0, ignore_index=True)
print("Combine shape:", combine.shape)

# ---------- Feature engineering: add y (proxy) using given function ----------
# Your example function (we implement same logic)
import scipy.stats as stats

def f(X):
    return (
        0.3 * X["curvature"].astype(float).fillna(0.0)
        + 0.2 * (X["lighting"] == "night").astype(int)
        + 0.1 * (X["weather"] != "clear").astype(int)
        + 0.2 * (X["speed_limit"].fillna(0).astype(float) >= 60).astype(int)
        + 0.1 * (X["num_reported_accidents"].fillna(0).astype(float) > 2).astype(int)
    )

def clip_func(f):
    def clip_f(X):
        sigma = 0.05
        mu = f(X)
        a, b = -mu/sigma, (1-mu)/sigma
        Phi_a, Phi_b = stats.norm.cdf(a), stats.norm.cdf(b)
        phi_a, phi_b = stats.norm.pdf(a), stats.norm.pdf(b)
        return mu*(Phi_b-Phi_a) + sigma*(phi_a-phi_b) + 1 - Phi_b
    return clip_f

combine['y'] = clip_func(f)(combine)
# add 'y' to features
FEATURES = list(orig.columns[1:-1])  # like your example, all columns except id and target in orig
FEATURES.append("y")
TARGET = "accident_risk"

# ---------- Identify categorical vs numeric ----------
CATS = []
NUMS = []
for c in FEATURES:
    if combine[c].dtype == 'object' or combine[c].dtype.name == 'category':
        CATS.append(c)
    else:
        NUMS.append(c)
print("CATS:", CATS)
print("NUMS:", NUMS)

# ---------- Label factorize categorical columns ----------
SIZES = {}
for c in CATS:
    combine[c], _ = combine[c].factorize()
    SIZES[c] = int(combine[c].max()+1)
    combine[c] = combine[c].astype('int32')
print("Cardinality:", SIZES)

# ---------- Split back ----------
train = combine.iloc[:len(train)].copy().reset_index(drop=True)
test  = combine.iloc[len(train):len(train)+len(test)].copy().reset_index(drop=True)
orig  = combine.iloc[-len(orig):].copy().reset_index(drop=True)
print("Split shapes:", train.shape, test.shape, orig.shape)

# ---------- Target encoding based on original data ----------
TE_cols = []
for c in FEATURES:
    te_name = f"TE_{c}"
    tmp = orig.groupby(c)[TARGET].mean().rename(te_name)
    # merge to train/test (left join; if unseen in train/test becomes NaN)
    train = train.merge(tmp, on=c, how='left')
    test = test.merge(tmp, on=c, how='left')
    # fill NaNs in TE with global mean from orig to avoid NaNs
    train[te_name].fillna(orig[TARGET].mean(), inplace=True)
    test[te_name].fillna(orig[TARGET].mean(), inplace=True)
    TE_cols.append(te_name)
print("TE cols:", TE_cols)

# ---------- Reduce memory (downcast) ----------
def downcast_df(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

train = downcast_df(train)
test  = downcast_df(test)

# ---------- Prepare features for modeling ----------
ALL_FEATURES = FEATURES + TE_cols
print("Feature count:", len(ALL_FEATURES))

# ---------- XGBoost on residuals (KFold) ----------
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.6,
    "seed": SEED,
    "verbosity": 0,
}
# GPU device if available
if DEVICE == "cuda":
    params["tree_method"] = "gpu_hist"
else:
    params["tree_method"] = "hist"

print("XGBoost version:", xgb.__version__)
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(train), dtype=np.float32)
test_preds = np.zeros(len(test), dtype=np.float32)
best_models = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(train)):
    print("#"*30)
    print(f"FOLD {fold+1}/{FOLDS}")
    X_train = train.iloc[tr_idx][ALL_FEATURES].copy()
    y_train = train.iloc[tr_idx][TARGET].values - train.iloc[tr_idx]['y'].values  # residual target
    X_valid = train.iloc[val_idx][ALL_FEATURES].copy()
    y_valid = train.iloc[val_idx][TARGET].values - train.iloc[val_idx]['y'].values
    y_valid_y = train.iloc[val_idx]['y'].values  # to add back
    
    X_test = test[ALL_FEATURES].copy()
    y_test_y = test['y'].values
    
    # DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_valid, label=y_valid)
    dtest  = xgb.DMatrix(X_test)
    
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=VERBOSE_EVAL
    )
    best_iter = model.best_iteration if model.best_iteration is not None else model.num_boosted_rounds()
    print("Best iteration:", best_iter)
    
    # predictions: add back y (the base)
    oof_preds[val_idx] = model.predict(dval, iteration_range=(0, best_iter+1)) + y_valid_y
    test_preds += (model.predict(dtest, iteration_range=(0, best_iter+1)) + y_test_y) / FOLDS
    
    best_models.append(model)
    # cleanup
    del dtrain, dval, dtest, X_train, X_valid, X_test
    gc.collect()

# ---------- CV score ----------
cv_rmse = np.sqrt(np.mean((oof_preds - train[TARGET].values)**2))
baseline_rmse = np.sqrt(np.mean((train['y'].values - train[TARGET].values)**2))
print("Overall CV RMSE (model):", cv_rmse)
print("Baseline CV RMSE (orig y):", baseline_rmse)

# ---------- Optional: plot last model feature importance ----------
try:
    fig, ax = plt.subplots(figsize=(10,8))
    xgb.plot_importance(best_models[-1], max_num_features=40, importance_type='gain', ax=ax, show_values=False)
    ax.set_title("XGB Feature Importance (gain)")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Could not plot importance:", e)

# ---------- Save OOF and Submission ----------
np.save("oof_preds.npy", oof_preds)

sub = pd.read_csv(SAMPLE_PATH)
sub['accident_risk'] = np.clip(test_preds, 0.0, 1.0)
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv (first rows):")
print(sub.head())


