import numpy as np # linear algebra
import pandas as pd # data processing, (CSV file I/O)
import os, gc
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import tensorflow.keras.backend as K
print('Using TensorFlow version',tf.__version__)
import cupy as cp
import cudf
from cuml.ensemble import RandomForestClassifier as cuRF
from sklearn.ensemble import RandomForestClassifier as skRF
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import math, sys
import random
import matplotlib.pyplot as plt
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



csv_path = '/kaggle/input/amex-default-prediction/train_labels.csv'
df = pd.read_csv(csv_path)
print(df.head())


os.environ["CUDA_VISIBLE_DEVICES"]="0"

# TENSORFLOW : 8GB RAM
# RAPIDS: 7GB RAM
LIMIT = 8
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*LIMIT)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  except RuntimeError as e:
    print(e)
print('Restrict TensorFlow to max %iGB GPU RAM'%LIMIT)
print('RAPIDS to use %iGB GPU RAM'%(15-LIMIT))


PATH_TO_CUSTOMER_HASHES = None
PROCESS_DATA = True
PATH_TO_DATA = '/kaggle/working/data/'
TRAIN_MODEL = True
PATH_TO_MODEL = '/kaggle/working/model/'
INFER_TEST = True


# read only the header to get column names
cols = pd.read_csv('/kaggle/input/amex-default-prediction/train_data.csv', nrows=0).columns
print(f"{len(cols)} columns detected")

# just peek at dtypes from first few rows (cheap)
sample = pd.read_csv('/kaggle/input/amex-default-prediction/train_data.csv', nrows=1000)
print(sample.dtypes)

# count total rows (fast line count)
import subprocess, shlex
n = int(subprocess.check_output(shlex.split("wc -l /kaggle/input/amex-default-prediction/train_data.csv")).split()[0]) - 1
print(f"Total rows: {n:,}")
summary = sample.dtypes.value_counts()
print(summary)


np.random.seed(42); random.seed(42)
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 160)

DATA_DIR = "/kaggle/input/amex-default-prediction"
WORK_DIR = "/kaggle/working"

# knobs for chunked scans (tune based on RAM)
EDA_CHUNKSIZE = 2_000_000
EDA_SAMPLE_ROWS = 1_000_000  # for quick sample-based EDA (set None to scan full)


for fname in ["train_data.csv", "test_data.csv", "train_labels.csv", "sample_submission.csv"]:
    fpath = os.path.join(DATA_DIR, fname)
    size_gb = os.path.getsize(fpath) / (1024**3)
    print(f"{fname:22s}  {size_gb:6.2f} GB")


labels = pd.read_csv(f"{DATA_DIR}/train_labels.csv")
pos_rate = labels["target"].mean()
print(f"Train customers: {len(labels):,}")
print(f"Positive rate (target=1): {pos_rate:.4f}")
display(labels["target"].value_counts().rename_axis("target").to_frame("count"))


# EDA-3 â€” Per-customer statement counts + date range (chunked)
cust_counts = {}
min_date, max_date = None, None

reader = pd.read_csv(f"{DATA_DIR}/train_data.csv", usecols=["customer_ID", "S_2"], chunksize=EDA_CHUNKSIZE)
for i, chunk in enumerate(reader, 1):
    # track date range
    chunk["S_2"] = pd.to_datetime(chunk["S_2"])
    cmin, cmax = chunk["S_2"].min(), chunk["S_2"].max()
    min_date = cmin if min_date is None or cmin < min_date else min_date
    max_date = cmax if max_date is None or cmax > max_date else max_date

    # counts within chunk
    cnt = chunk["customer_ID"].value_counts()
    for cid, n in cnt.items():
        cust_counts[cid] = cust_counts.get(cid, 0) + int(n)

    if i % 2 == 0:
        print(f"[EDA-3] chunks processed: {i}")
    del chunk, cnt
    gc.collect()

counts = np.array(list(cust_counts.values()))
print(f"Customers seen: {len(counts):,}")
print(f"Statements per customer â€” min/median/mean/max: {counts.min()} / {np.median(counts):.1f} / {counts.mean():.2f} / {counts.max()}")
print(f"Date range (S_2): {min_date.date()} â†’ {max_date.date()}")

# histogram of statement counts
plt.figure(figsize=(6,3.5))
plt.hist(counts, bins=range(1, counts.max()+2))
plt.title("Statements per Customer (train)")
plt.xlabel("Statements"); plt.ylabel("Customers")
plt.show()



# EDA-4 â€” Schema + rough missingness on a sample
sample = pd.read_csv(f"{DATA_DIR}/train_data.csv", nrows=EDA_SAMPLE_ROWS)
print("Sample shape:", sample.shape)

# dtypes summary
dtype_counts = sample.dtypes.value_counts()
print("\nDtype counts:\n", dtype_counts)

# rough missingness %
na_pct = (sample.isna().sum() / len(sample) * 100).sort_values(ascending=False).head(25)
display(na_pct.to_frame("NA% (approx from sample)").round(2))


# EDA-5 â€” Categorical value counts (common: D_63, D_64)
cat_cols_to_peek = [c for c in ["D_63", "D_64"] if c in sample.columns]
for c in cat_cols_to_peek:
    vc = sample[c].value_counts(dropna=False).head(10)
    print(f"\nTop values for {c}:")
    display(vc)


# EDA-6 â€” Histograms for a few numeric columns (from sample)
num_cols = [c for c in sample.columns if c not in ("customer_ID","S_2") and np.issubdtype(sample[c].dtype, np.number)]
pick = num_cols[:6]  # first few numerics; change to your favorites

for col in pick:
    s = sample[col].dropna()
    if len(s) == 0:
        continue
    plt.figure(figsize=(6,3.5))
    plt.hist(s.values, bins=50)
    plt.title(f"{col} (sample)")
    plt.xlabel(col); plt.ylabel("Frequency")
    plt.show()


# EDA-7 â€” Correlation snapshot (Spearman) on a small numeric subset
sub = sample[num_cols].select_dtypes(include=[np.number]).copy()
sub = sub.iloc[:, :30]  # first 30 numeric cols to keep it light
sub = sub.fillna(sub.median(numeric_only=True))

corr = sub.corr(method="spearman").values
plt.figure(figsize=(5,4.5))
plt.imshow(corr, interpolation="nearest")
plt.title("Spearman Correlation (sample subset)")
plt.colorbar(fraction=0.046, pad=0.04)
plt.xticks([]); plt.yticks([])
plt.show()



import lightgbm as lgb
print("LightGBM version:", lgb.__version__)


# Cell 2 â€” AMEX metric + LightGBM callback
import numpy as np

def amex_metric(y_true, y_pred):
    """Official AMEX metric (vectorized)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)
    assert y_true.shape == y_pred.shape

    # Sort by prediction (desc)
    order = np.argsort(-y_pred, kind="mergesort")
    y_true = y_true[order]

    # Top 4% by weighted count (0 -> weight 20, 1 -> weight 1)
    weights = np.where(y_true == 0, 20, 1)
    cum_w = np.cumsum(weights)
    cutoff = int(0.04 * cum_w[-1])
    top_mask = cum_w <= cutoff
    top_four = y_true[top_mask].sum() / max(1, y_true.sum())

    # Normalized Gini
    def gini(a, p, w):
        order = np.argsort(-p, kind="mergesort")
        a, w = a[order], w[order]
        cum_w = np.cumsum(w)
        cum_y = np.cumsum(a * w)
        return (cum_y / cum_y[-1] - cum_w / cum_w[-1]).sum()

    den = gini(y_true, y_true, weights)
    g = gini(y_true, y_pred[order], weights) / (den if abs(den) > 1e-12 else 1e-12)

    return 0.5 * (g + top_four)

def lgb_amex_metric(preds, train_data):
    y = train_data.get_label()
    return ("amex", amex_metric(y, preds), True)



# Cell 2 â€” Two-pass, chunked â€œlast row per customerâ€� build (memory-safe)
import os, gc, numpy as np, pandas as pd

DATA_DIR = "/kaggle/input/amex-default-prediction"
WORK_DIR = "/kaggle/working"
os.makedirs(WORK_DIR, exist_ok=True)

# ---------- helpers ----------
def _hash_series_to_int32_pandas(s: pd.Series, modulo: int = 2048) -> pd.Series:
    s = s.astype("string").fillna("__NA__")
    h = pd.util.hash_pandas_object(s, index=False).astype("int64")
    h = (h & 0x7FFFFFFF) % modulo
    return h.astype("int32")

def _pass1_max_date(csv_path: str, chunksize: int = 2_000_000):
    """
    Pass 1: read only customer_ID,S_2 to find the max S_2 per customer.
    Returns a dict: cid -> max_date (pd.Timestamp)
    """
    cid_to_max = {}
    for chunk in pd.read_csv(csv_path, usecols=["customer_ID","S_2"], chunksize=chunksize):
        chunk["S_2"] = pd.to_datetime(chunk["S_2"])
        # reduce within chunk
        grp = chunk.groupby("customer_ID", sort=False)["S_2"].max()
        for cid, dt in grp.items():
            if (cid not in cid_to_max) or (dt > cid_to_max[cid]):
                cid_to_max[cid] = dt
        del chunk, grp
        gc.collect()
    return cid_to_max

def _pass2_collect_last_rows(csv_path: str, cid_to_max: dict, chunksize: int = 1_000_000):
    """
    Pass 2: scan full CSV; keep rows where S_2 == cid_to_max[cid].
    If multiple rows per (cid, S_2_max), keep the LAST (tail(1)) in that chunk.
    Returns a pandas DataFrame with one row per customer (may still need final groupby).
    """
    # probe full schema to keep all feature columns
    sample = pd.read_csv(csv_path, nrows=2000)
    cols = list(sample.columns)

    kept_parts = []
    for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=chunksize):
        chunk["S_2"] = pd.to_datetime(chunk["S_2"])
        # filter to rows that match that customer's max date
        # map lookup (vectorized): create a Series of max dates aligned to chunk cids
        max_dates = chunk["customer_ID"].map(cid_to_max)
        mask = (max_dates.notna()) & (chunk["S_2"].values == max_dates.values)
        if not mask.any():
            del chunk, max_dates
            gc.collect()
            continue
        sub = chunk.loc[mask].copy()
        # within this sub-chunk, keep last row per customer_ID
        sub = sub.sort_values(["customer_ID", "S_2"]).groupby("customer_ID", sort=False).tail(1)
        kept_parts.append(sub[cols])  # keep all columns
        del chunk, max_dates, sub
        gc.collect()

    if not kept_parts:
        return pd.DataFrame(columns=cols)

    df_last = pd.concat(kept_parts, axis=0, ignore_index=True)
    # In case same customer got split across chunks with same max date, finalize with tail(1)
    df_last = df_last.sort_values(["customer_ID", "S_2"]).groupby("customer_ID", sort=False).tail(1).reset_index(drop=True)
    return df_last

def _finalize_features_last(df_last: pd.DataFrame) -> pd.DataFrame:
    """
    Hash any non-numeric columns (except customer_ID, S_2), drop S_2.
    """
    for c in df_last.columns:
        if c in ("customer_ID", "S_2"):
            continue
        if not np.issubdtype(df_last[c].dtype, np.number):
            df_last[c] = _hash_series_to_int32_pandas(df_last[c], modulo=2048)
    return df_last.drop(columns=["S_2"])

def build_last_table(csv_path: str, label_df: pd.DataFrame | None = None, role: str = "train"):
    """
    role='train' â†’ returns train_df (merged with labels)
    role='test'  â†’ returns test_agg (customer_ID + features)
    """
    print(f"[{role}] pass 1: scanning max dates ...")
    cid_to_max = _pass1_max_date(csv_path)
    print(f"[{role}] unique customers found:", len(cid_to_max))

    print(f"[{role}] pass 2: collecting last rows ...")
    last_rows = _pass2_collect_last_rows(csv_path, cid_to_max)
    print(f"[{role}] last_rows shape:", last_rows.shape)

    print(f"[{role}] hashing categoricals & finalizing ...")
    last_feats = _finalize_features_last(last_rows)
    print(f"[{role}] finalized feature table:", last_feats.shape)

    if role == "train":
        assert label_df is not None, "label_df must be provided for role='train'"
        out = label_df.merge(last_feats, on="customer_ID", how="left")
        return out
    else:
        return last_feats

# ---------- TRAIN ----------
labels = pd.read_csv(f"{DATA_DIR}/train_labels.csv")  # customer_ID, target
train_df = build_last_table(f"{DATA_DIR}/train_data.csv", label_df=labels, role="train")

# ---------- TEST ----------
test_last = build_last_table(f"{DATA_DIR}/test_data.csv", label_df=None, role="test")

# Align columns to training features
feature_cols = [c for c in train_df.columns if c not in ("customer_ID","target")]
for col in feature_cols:
    if col not in test_last.columns:
        test_last[col] = np.nan
test_agg = test_last[["customer_ID"] + feature_cols]

print("Final shapes â†’ train_df:", train_df.shape, " | test_agg:", test_agg.shape)
display(train_df.head())
display(test_agg.head())

# Medians for imputation in Cells 3 & 4
TRAIN_FEATURES_ONLY = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
train_medians = TRAIN_FEATURES_ONLY.median(numeric_only=True)
print(f"Medians computed for {len(train_medians)} features.")



print(f"train_df: {train_df.shape}  |  test_agg: {test_agg.shape}")
display(train_df.head(3))


# Cell 3 â€” Train LightGBM (GPU) with callbacks early stopping
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np

print("LightGBM version:", lgb.__version__)

assert "train_df" in globals(), "Expected a DataFrame named train_df from your preprocessing step."

FEATURES = [c for c in train_df.columns if c not in ("customer_ID", "target")]
TARGET = "target"

X = train_df[FEATURES].replace([np.inf, -np.inf], np.nan).astype(np.float32)
# Use medians computed in Cell 2 if available; otherwise compute now
if "train_medians" not in globals():
    train_medians = X.median(numeric_only=True)
X = X.fillna(train_medians)
y = train_df[TARGET].astype(np.int8)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Features: {len(FEATURES)}")

train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set, free_raw_data=False)

params = {
    "objective": "binary",
    "metric": "binary_logloss",     # AMEX reported via feval
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 128,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.4,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "max_depth": -1,
    "lambda_l1": 1.0,
    "lambda_l2": 2.0,
    "max_bin": 255,
    "bin_construct_sample_cnt": 200_000,
    "device": "gpu",                # GPU
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    # If you hit OOM on GPU, try:
    # "force_col_wise": True,
}

# Use callbacks for early stopping & logging (compatible across LGBM versions)
callbacks = [
    lgb.early_stopping(stopping_rounds=200, verbose=True),
    lgb.log_evaluation(period=200),
]

model = lgb.train(
    params=params,
    train_set=train_set,
    num_boost_round=5000,
    valid_sets=[train_set, valid_set],
    valid_names=["train", "valid"],
    feval=lgb_amex_metric,          # from Cell 2
    callbacks=callbacks
)

# Validation AMEX
valid_pred = model.predict(X_valid, num_iteration=model.best_iteration or model.current_iteration())
print("AMEX (valid):", amex_metric(y_valid.values.astype(int), valid_pred))
print("Best iteration:", model.best_iteration or model.current_iteration())



print(f"Best iteration: {getattr(model, 'best_iteration', None) or model.current_iteration()}")
# Optional: quick top-20 feature importances
import pandas as pd
imp = pd.DataFrame({
    "feature": FEATURES,
    "gain": model.feature_importance(importance_type="gain")
}).sort_values("gain", ascending=False).head(20)
display(imp)


# Cell 4 â€” Inference & submission (GPU-trained model)

import numpy as np
import pandas as pd

# Sanity checks
assert "test_agg" in globals(), "Expected test_agg from Cell 2."
assert "FEATURES" in globals(), "Run Cell 3 first to define FEATURES."
assert "model" in globals(), "Run Cell 3 to train the model."
assert "train_medians" in globals(), "Expected train_medians from Cell 2/3."

# Align test columns to training FEATURES
for col in FEATURES:
    if col not in test_agg.columns:
        test_agg[col] = np.nan

X_test = test_agg[FEATURES].replace([np.inf, -np.inf], np.nan).astype(np.float32)
X_test = X_test.fillna(train_medians)

# Use best iteration if available, else current
best_it = getattr(model, "best_iteration", None) or getattr(model, "current_iteration", lambda: None)()
test_pred = model.predict(X_test, num_iteration=best_it)

# Build and save submission
DATA_DIR = "/kaggle/input/amex-default-prediction"
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv", usecols=["customer_ID"])
sub["prediction"] = test_pred

out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print("Saved:", out_path)

# Quick peek
display(sub.head())



# # CALCULATE SIZE OF EACH SEPARATE FILE
# def get_rows(customers, train, NUM_FILES = 20, verbose = ''):
#     chunk = len(customers)//NUM_FILES
#     if verbose != '':
#         print(f'We will split {verbose} data into {NUM_FILES} separate files.')
#         print(f'There will be {chunk} customers in each file (except the last file).')
#         print('Below are number of rows in each file:')
#     rows = []

#     for k in range(NUM_FILES):
#         if k==NUM_FILES-1: cc = customers[k*chunk:]
#         else: cc = customers[k*chunk:(k+1)*chunk]
#         s = train.loc[train.customer_ID.isin(cc)].shape[0]
#         rows.append(s)
#     if verbose != '': print( rows )
#     return rows

# if PROCESS_DATA:
#     NUM_FILES = 20
#     rows = get_rows(customers, train, NUM_FILES = NUM_FILES, verbose = 'train')


# def feature_engineer(train, PAD_CUSTOMER_TO_13_ROWS = True, targets = None):
        
#     # REDUCE STRING COLUMNS 
#     # from 64 bytes to 8 bytes, and 10 bytes to 3 bytes respectively
#     train['customer_ID'] = train['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
#     train.S_2 = cudf.to_datetime( train.S_2 )
#     train['year'] = (train.S_2.dt.year-2000).astype('int8')
#     train['month'] = (train.S_2.dt.month).astype('int8')
#     train['day'] = (train.S_2.dt.day).astype('int8')
#     del train['S_2']
        
#     # LABEL ENCODE CAT COLUMNS (and reduce to 1 byte)
#     # with 0: padding, 1: nan, 2,3,4,etc: values
#     d_63_map = {'CL':2, 'CO':3, 'CR':4, 'XL':5, 'XM':6, 'XZ':7}
#     train['D_63'] = train.D_63.map(d_63_map).fillna(1).astype('int8')

#     d_64_map = {'-1':2,'O':3, 'R':4, 'U':5}
#     train['D_64'] = train.D_64.map(d_64_map).fillna(1).astype('int8')
    
#     CATS = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_66', 'D_68']
#     OFFSETS = [2,1,2,2,3,2,3,2,2] #2 minus minimal value in full train csv
#     # then 0 will be padding, 1 will be NAN, 2,3,4,etc will be values
#     for c,s in zip(CATS,OFFSETS):
#         train[c] = train[c] + s
#         train[c] = train[c].fillna(1).astype('int8')
#     CATS += ['D_63','D_64']
    
#     # ADD NEW FEATURES HERE
#     # EXAMPLE: train['feature_189'] = etc etc etc
#     # EXAMPLE: train['feature_190'] = etc etc etc
#     # IF CATEGORICAL, THEN ADD TO CATS WITH: CATS += ['feaure_190'] etc etc etc
    
#     # REDUCE MEMORY DTYPE
#     SKIP = ['customer_ID','year','month','day']
#     for c in train.columns:
#         if c in SKIP: continue
#         if str( train[c].dtype )=='int64':
#             train[c] = train[c].astype('int32')
#         if str( train[c].dtype )=='float64':
#             train[c] = train[c].astype('float32')
            
#     # PAD ROWS SO EACH CUSTOMER HAS 13 ROWS
#     if PAD_CUSTOMER_TO_13_ROWS:
#         tmp = train[['customer_ID']].groupby('customer_ID').customer_ID.agg('count')
#         more = cupy.array([],dtype='int64') 
#         for j in range(1,13):
#             i = tmp.loc[tmp==j].index.values
#             more = cupy.concatenate([more,cupy.repeat(i,13-j)])
#         df = train.iloc[:len(more)].copy().fillna(0)
#         df = df * 0 - 1 #pad numerical columns with -1
#         df[CATS] = (df[CATS] * 0).astype('int8') #pad categorical columns with 0
#         df['customer_ID'] = more
#         train = cudf.concat([train,df],axis=0,ignore_index=True)
        
#     # ADD TARGETS (and reduce to 1 byte)
#     if targets is not None:
#         train = train.merge(targets,on='customer_ID',how='left')
#         train.target = train.target.astype('int8')
        
#     # FILL NAN
#     train = train.fillna(-0.5) #this applies to numerical columns
    
#     # SORT BY CUSTOMER THEN DATE
#     train = train.sort_values(['customer_ID','year','month','day']).reset_index(drop=True)
#     train = train.drop(['year','month','day'],axis=1)
    
#     # REARRANGE COLUMNS WITH 11 CATS FIRST
#     COLS = list(train.columns[1:])
#     COLS = ['customer_ID'] + CATS + [c for c in COLS if c not in CATS]
#     train = train[COLS]
    
#     return train


PROCESS_DATA = True
USE_GPU = True             # True: cuML RF on GPU, False: sklearn RF on CPU
MAKE_FEATURES = 'agg'      # 'last', 'agg', or 'flat13'
RANDOM_STATE = 26
TEST_SIZE = 0.25


def make_dataset_from_engineered(train_engineered, make=MAKE_FEATURES):
    has_target = 'target' in train_engineered.columns
    COLS = list(train_engineered.columns)
    CATS = COLS[1:1+11]
    ALL = COLS[1:]
    NUMS = [c for c in ALL if c not in CATS + (['target'] if has_target else [])]

    g = train_engineered.groupby('customer_ID')

    if make == 'last':
        df = g.tail(1)
        y = df['target'].values if has_target else None
        X = df.drop(columns=['target']) if has_target else df
        return X.set_index('customer_ID'), y, list(X.columns)

    elif make == 'agg':
        num_agg = g[NUMS].agg(['last','mean','std','min','max'])
        cat_last = g[CATS].agg('last')
        X = cat_last.join(num_agg)
        X.columns = [f"{a}__{b}" if isinstance(a, tuple) else a for a in X.columns]
        y = g['target'].agg('last').values if has_target else None
        return X, y, list(X.columns)

    elif make == 'flat13':
        df = train_engineered.copy()
        if has_target:
            df_target = df[['customer_ID','target']].groupby('customer_ID').tail(1).drop_duplicates().set_index('customer_ID')
            df = df.drop(columns=['target'])
        df['_row'] = (df.groupby('customer_ID').cumcount()).astype('int8')
        piv = df.pivot(index='customer_ID', columns='_row')
        piv.columns = [f'{c[0]}@t{int(c[1])}' for c in piv.columns]
        X = piv
        y = df_target.loc[X.index,'target'].values if has_target else None
        return X, y, list(X.columns)




import os, gc, sys, glob, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

# ---------------- CONFIG ----------------
INPUT_DIR  = "/kaggle/input/amex-default-prediction"
WORK_DIR   = "/kaggle/working"
TMP_DIR    = os.path.join(WORK_DIR, "lastsnap-temp")
CACHE_DIR  = os.path.join(WORK_DIR, "lastsnap-cache")
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

TRAIN_CSV  = os.path.join(INPUT_DIR, "train_data.csv")
TEST_CSV   = os.path.join(INPUT_DIR,  "test_data.csv")
LABELS_CSV = os.path.join(INPUT_DIR, "train_labels.csv")

# Cache files (reused across runs)
TRAIN_LAST_PQT = os.path.join(CACHE_DIR, "train_last_snapshot.parquet")
TEST_LAST_PQT  = os.path.join(CACHE_DIR, "test_last_snapshot.parquet")

# Chunking + split
CHUNKSIZE   = 1_000_000      # lower if you still see restarts: 700_000 or 500_000
VALID_SIZE  = 0.25
RANDOM_STATE = 26

# RandomForest (kept modest to stay RAM-safe)
RF_PARAMS = dict(
    n_estimators=400,
    max_depth=28,
    max_features="sqrt",
    min_samples_split=4,
    min_samples_leaf=1,
    bootstrap=True,
    n_jobs=-1,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

# Categorical maps / encodes (same meaning as your GPU pipeline)
CATS_BASE   = ['B_30','B_38','D_114','D_116','D_117','D_120','D_126','D_66','D_68']
OFFSETS     = [2,      1,      2,      2,      3,      2,      3,      2,      2]
D63_MAP     = {'CL':2, 'CO':3, 'CR':4, 'XL':5, 'XM':6, 'XZ':7}
D64_MAP     = {'-1':2, 'O':3, 'R':4, 'U':5}

# ---------------- Helpers ----------------
def _id_hex_to_int16(s):
    return int(str(s)[-16:], 16)

def amex_metric(y_true, y_prob):
    """
    Official AMEX metric implementation (vectorized, RAM-safe for our split sizes).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_prob)

    # normalize ranks and weights
    w = np.where(y_true == 0, 20, 1)

    # top 4% by prediction
    order = np.argsort(-y_pred)
    w_cum = np.cumsum(w[order]) / np.sum(w)
    top4 = y_true[order][w_cum <= 0.04]
    d = np.sum(top4) / np.sum(y_true)

    # gini
    def _gini(a, p, w):
        order = np.argsort(-p)
        a, w = a[order], w[order]
        cum_w = np.cumsum(w)
        cum_a = np.cumsum(a * w)
        sum_a = np.sum(a * w)
        g = np.sum(cum_a / sum_a * w) / np.sum(w) - (np.sum(cum_w / np.sum(w) * w) / np.sum(w))
        return g

    g = _gini(y_true, y_pred, w)
    gmax = _gini(y_true, y_true, w)
    return 0.5 * (g / gmax + d)

def feature_engineer_chunk(df, keep_time=True):
    """
    Lean CPU version of your feature_engineer WITHOUT padding.
    """
    df['customer_ID'] = df['customer_ID'].astype(str).str[-16:].apply(_id_hex_to_int16).astype('int64')

    df['S_2'] = pd.to_datetime(df['S_2'])
    df['year']  = (df['S_2'].dt.year - 2000).astype('int16')
    df['month'] = df['S_2'].dt.month.astype('int8')
    df['day']   = df['S_2'].dt.day.astype('int8')

    if 'D_63' in df.columns:
        df['D_63'] = df['D_63'].map(D63_MAP).fillna(1).astype('int16')
    if 'D_64' in df.columns:
        df['D_64'] = df['D_64'].map(D64_MAP).fillna(1).astype('int16')

    for c, s in zip(CATS_BASE, OFFSETS):
        if c in df.columns:
            x = pd.to_numeric(df[c], errors='coerce')
            x = (x + s).astype('float32')
            df[c] = x.fillna(1).astype('int16')

    skip = {'customer_ID','S_2','year','month','day','target'}
    for c in df.columns:
        if c in skip: continue
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast='integer')
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast='float')

    num_cols = [c for c in df.columns if c not in ['customer_ID','S_2','year','month','day','target']]
    df[num_cols] = df[num_cols].fillna(-0.5)

    if not keep_time:
        df = df.drop(columns=['S_2','year','month','day'])

    return df

def write_chunk_last_rows(csv_path, prefix, chunksize=CHUNKSIZE):
    # clean any previous temp parts for this prefix
    for p in glob.glob(os.path.join(TMP_DIR, f"{prefix}_last_part_*.parquet")):
        try: os.remove(p)
        except: pass

    i = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        fe = feature_engineer_chunk(chunk, keep_time=True)
        idx = fe.groupby('customer_ID')['S_2'].idxmax()
        last_rows = fe.loc[idx].copy()
        outp = os.path.join(TMP_DIR, f"{prefix}_last_part_{i}.parquet")
        last_rows.to_parquet(outp, index=False)
        print(f"[{prefix}] wrote {os.path.basename(outp)}  (rows={len(last_rows):,})")
        i += 1
        del chunk, fe, last_rows
        gc.collect()
    return i

def build_global_last(prefix):
    parts = sorted(glob.glob(os.path.join(TMP_DIR, f"{prefix}_last_part_*.parquet")))
    if not parts:
        raise FileNotFoundError(f"No parts found for prefix={prefix}")

    dfs = []
    for p in parts:
        dfp = pd.read_parquet(p)
        dfs.append(dfp)
    big = pd.concat(dfs, ignore_index=True)
    del dfs; gc.collect()

    idx = big.groupby('customer_ID')['S_2'].idxmax()
    last_global = big.loc[idx].copy()

    last_global = last_global.sort_values('customer_ID')
    last_global = last_global.drop(columns=['S_2','year','month','day'])
    last_global = last_global.set_index('customer_ID')

    # optional: clean temp
    for p in parts:
        try: os.remove(p)
        except: pass

    return last_global

# ---------------- Main ----------------
def main():
    # TRAIN last snapshot cache
    if os.path.exists(TRAIN_LAST_PQT):
        print(">> Loading cached TRAIN last snapshotâ€¦")
        X_train = pd.read_parquet(TRAIN_LAST_PQT).set_index('customer_ID')
    else:
        print(">> Processing TRAIN chunks to LAST rowsâ€¦")
        write_chunk_last_rows(TRAIN_CSV, prefix="train")
        print(">> Building global TRAIN last snapshotâ€¦")
        X_train = build_global_last(prefix="train")
        X_train.reset_index().to_parquet(TRAIN_LAST_PQT, index=False)
    print("   train per-customer shape:", X_train.shape)

    # Labels
    labels = pd.read_csv(LABELS_CSV)
    labels['customer_ID'] = labels['customer_ID'].astype(str).str[-16:].apply(_id_hex_to_int16).astype('int64')
    labels = labels.set_index('customer_ID')
    y = labels.reindex(X_train.index)['target'].astype('int8').values

    # Split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train, y, test_size=VALID_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Train RF
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_tr, y_tr)
    va_pred = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, va_pred)
    amex = amex_metric(y_va, va_pred)
    print(f"[Model] Validation ROC AUC = {auc:.5f} | AMEX = {amex:.5f}")

    # Save model + columns
    joblib.dump(model, os.path.join(WORK_DIR, "rf_model.joblib"))
    X_cols = X_train.columns.tolist()
    with open(os.path.join(WORK_DIR, "rf_columns.json"), "w") as f:
        json.dump(X_cols, f)
    print("Saved rf_model.joblib and rf_columns.json")

    # Save feature importances
    fi = pd.DataFrame({
        "feature": X_cols,
        "importance": model.feature_importances_.astype(np.float32)
    }).sort_values("importance", ascending=False)
    fi.to_csv(os.path.join(WORK_DIR, "feature_importances.csv"), index=False)
    print("Saved feature_importances.csv (top 10):")
    print(fi.head(10))

    # TEST last snapshot cache
    if os.path.exists(TEST_LAST_PQT):
        print("\n>> Loading cached TEST last snapshotâ€¦")
        X_test = pd.read_parquet(TEST_LAST_PQT).set_index('customer_ID')
    else:
        print("\n>> Processing TEST chunks to LAST rowsâ€¦")
        write_chunk_last_rows(TEST_CSV, prefix="test")
        print(">> Building global TEST last snapshotâ€¦")
        X_test = build_global_last(prefix="test")
        X_test.reset_index().to_parquet(TEST_LAST_PQT, index=False)
    print("   test per-customer shape:", X_test.shape)

    # Align columns
    missing = [c for c in X_cols if c not in X_test.columns]
    if missing:
        for c in missing:
            X_test[c] = 1 if c in (CATS_BASE + ['D_63','D_64']) else 0.0
    extra = [c for c in X_test.columns if c not in X_cols]
    if extra:
        X_test = X_test.drop(columns=extra)
    X_test = X_test[X_cols]

    # Predict test
    y_pred = model.predict_proba(X_test)[:, 1]
    # Convert numeric IDs back to 64-char hex strings (pad with zeros)
    customer_hex = [format(int(x), '064x') for x in X_test.index.values]
    
    submission = pd.DataFrame({
        "customer_ID": customer_hex,
        "prediction": y_pred.astype(np.float32)
    })

    submission.to_csv("submission.csv", index=False)
    print("\nâœ… Saved submission.csv")
    print("Artifacts:")
    print(" - rf_model.joblib")
    print(" - rf_columns.json")
    print(" - feature_importances.csv")
    print(" - submission.csv")
    print(" - Cached last snapshots in lastsnap-cache/")

if __name__ == "__main__":
    main()


import pandas as pd

results = []

# LightGBM (if present)
if 'valid_pred' in globals():
    results.append({
        "Model": "LightGBM (GPU)",
        "Valid AMEX": float(amex_metric(y_valid.values.astype(int), valid_pred)),
        "Notes": f"best_iteration={getattr(model,'best_iteration', None) or model.current_iteration()}"
    })

# Random Forest (if present)
if 'valid_pred_rf' in globals():
    results.append({
        "Model": "Random Forest" + (" (GPU/cuML)" if 'rf_gpu' in globals() else " (CPU/sklearn)"),
        "Valid AMEX": float(rf_valid_amex),
        "Notes": ""
    })

pd.DataFrame(results).sort_values("Valid AMEX", ascending=False).style.format({"Valid AMEX": "{:.6f}"})



import matplotlib.pyplot as plt

labels = [r["Model"] for r in results]
scores = [r["Valid AMEX"] for r in results]

plt.figure(figsize=(6,3.5))
plt.bar(labels, scores)
plt.ylabel("Valid AMEX")
plt.title("Model Comparison")
plt.xticks(rotation=10)
plt.show()

