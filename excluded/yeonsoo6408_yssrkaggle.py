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


import pandas as pd

# ì�¼ë¶€ íŒŒí‹°ì…˜ë§Œ ì‚¬ìš©í•´ì„œ sample ê´€ì°° 
sample_path = "/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id=0/part-0.parquet"

# ë�°ì�´í„° ë¶ˆëŸ¬ì˜¤ê¸° (ì²˜ì�Œ 100í–‰ë§Œ)
df = pd.read_parquet(sample_path)
df_sample = df.head(100)

# ì „ì²´ ì»¬ëŸ¼ ë³´ê¸°
print("ì „ì²´ ì»¬ëŸ¼ ìˆ˜:", df_sample.shape[1])
print("ì»¬ëŸ¼ ëª©ë¡�:\n", df_sample.columns.tolist())

# ë�°ì�´í„° ìƒ˜í”Œ í™•ì�¸
display(df_sample)

# ë�°ì�´í„° íƒ€ì�… í™•ì�¸
print("\nì»¬ëŸ¼ë³„ ë�°ì�´í„° íƒ€ì�…:")
print(df_sample.dtypes)


import pandas as pd
import glob

partition_paths = glob.glob('/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id=*/part-0.parquet')
nan_results = []

for path in partition_paths:
    df = pd.read_parquet(path)
    
    # NaN ë¹„ìœ¨ ê³„ì‚°
    nan_ratio = df.isnull().mean()
    nan_ratio.name = path.split('/')[-2]  # ì˜ˆ: partition_id=3
    nan_results.append(nan_ratio)

# íŒŒí‹°ì…˜ë³„ NaN ë¹„ìœ¨ ê²°ê³¼ë¥¼ í•˜ë‚˜ì�˜ DataFrameìœ¼ë¡œ í•©ì¹˜ê¸°
nan_df = pd.DataFrame(nan_results)

# í–‰ = íŒŒí‹°ì…˜, ì—´ = feature
nan_df.index.name = 'partition'
nan_df = nan_df.T  # feature ê¸°ì¤€ìœ¼ë¡œ ë³´ê¸° ì‰½ê²Œ ì „ì¹˜
display(nan_df)


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 18))
plt.imshow(nan_df.values, aspect='auto', cmap='Reds')
plt.colorbar(label='NaN Ratio')
plt.yticks(range(len(nan_df.index)), nan_df.index)
plt.xticks(range(len(nan_df.columns)), nan_df.columns, rotation=90)
plt.title("NaN Ratio by Feature and Partition")
plt.show()


# 1. partition_id=ìˆ«ì�� ìˆœì„œë¡œ ì»¬ëŸ¼ ì •ë ¬
nan_df_sorted = nan_df[sorted(
    nan_df.columns,
    key=lambda x: int(x.split('=')[1])  # '=' ë’¤ ìˆ«ì��ë¥¼ ê¸°ì¤€ìœ¼ë¡œ ì •ë ¬
)]

# 2. NaN ë¹„ìœ¨ì�´ 0ë³´ë‹¤ í�° featureë§Œ ì¶”ì¶œ
nan_nonzero = nan_df_sorted[nan_df_sorted.max(axis=1) > 0]

# 3. í™•ì�¸
display(nan_nonzero)


# =====================================
# NaN ì²˜ë¦¬ ê·œì¹™ ì �ìš© (ê°„ë‹¨ ë²„ì „, ë�°ëª¨)
# - nan_nonzero ê¸°ì¤€ìœ¼ë¡œ feature ë¶„ë¥˜ í›„ ë³€í™˜
# =====================================

import numpy as np

# ---- 1) Feature ë¶„ë¥˜ í•¨ìˆ˜ ----
def classify_nan_features(nan_df, sparse_max=0.05, partial_min=0.07, partial_max=0.30):
    """
    nan_df: index=feature, columns=partition_id=ìˆ«ì��, ê°’=NaN ë¹„ìœ¨
    """
    full_nan = []          # ì „ì²´ 100% NaN â†’ ë“œë��
    partition_100 = []     # ì�¼ë¶€ partition 100% NaN â†’ indicator ê¶Œì�¥
    partial_nan = []       # 7%~30% ì •ë�„ ê¾¸ì¤€í�ˆ NaN â†’ indicator ê¶Œì�¥
    sparse_nan = []        # 5% ì�´í•˜ í�¬ì†Œ NaN â†’ ë‹¨ìˆœ ëŒ€ì²´
    
    for feat in nan_df.index:
        ratios = nan_df.loc[feat].values
        overall = ratios.mean()
        
        if np.all(ratios >= 0.9999):   # ì „ì²´ 100% NaN
            full_nan.append(feat)
        elif np.any(ratios >= 0.9999): # ì�¼ë¶€ partitionì—�ì„œë§Œ 100% NaN
            partition_100.append(feat)
        elif overall <= sparse_max:
            sparse_nan.append(feat)
        elif partial_min <= overall <= partial_max:
            partial_nan.append(feat)
        else:
            # ì „ì²´ NaN ë¹„ìœ¨ì�´ í�¬ì§€ë§Œ 100%ëŠ” ì•„ë‹Œ ê²½ìš° â†’ partial ì·¨ê¸‰
            partial_nan.append(feat)
    
    return {
        "full_nan_drop": full_nan,
        "partition_100_indicator": partition_100,
        "partial_indicator": partial_nan,
        "sparse_impute_only": sparse_nan
    }

# ---- 2) ì‹¤ì œ NaN ì²˜ë¦¬ í•¨ìˆ˜ ----
def process_dataframe(df, classes, impute_strategy="zero"):
    """
    df: ì›�ë³¸ DataFrame
    classes: classify_nan_features ê²°ê³¼ dict
    impute_strategy: "zero" or "median"
    """
    df = df.copy()
    
    # ì™„ì „ NaN â†’ ë“œë��
    df.drop(columns=classes["full_nan_drop"], errors="ignore", inplace=True)
    
    # indicator ëŒ€ìƒ� í”¼ì²˜
    indicator_feats = classes["partition_100_indicator"] + classes["partial_indicator"]
    
    # indicator ì¶”ê°€
    for feat in indicator_feats:
        if feat in df.columns:
            df[f"{feat}_isnan"] = df[feat].isna().astype(np.int8)
    
    # impute ëŒ€ìƒ� = indicator_feats + sparse_impute_only
    impute_feats = indicator_feats + classes["sparse_impute_only"]
    
    for feat in impute_feats:
        if feat in df.columns:
            if impute_strategy == "zero":
                df[feat] = df[feat].fillna(0.0)
            elif impute_strategy == "median":
                med = df[feat].median(skipna=True)
                df[feat] = df[feat].fillna(med if not np.isnan(med) else 0.0)
    
    return df

# ---- 3) ì‹¤í–‰ ì˜ˆì‹œ ----

# Feature ë¶„ë¥˜
classes = classify_nan_features(nan_nonzero)

print("=== Feature ë¶„ë¥˜ ê²°ê³¼ ===")
for k, v in classes.items():
    print(f"{k}: {len(v)}ê°œ")

# ì˜ˆì‹œ: partition_id=0 ë�°ì�´í„° í•˜ë‚˜ ë¶ˆëŸ¬ì˜¤ê¸°
sample_path = "/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id=0/part-0.parquet"
df_raw = pd.read_parquet(sample_path)

print("ì›�ë³¸ shape:", df_raw.shape)

# NaN ì²˜ë¦¬ ì �ìš©
df_proc = process_dataframe(df_raw, classes, impute_strategy="zero")

print("ì²˜ë¦¬ í›„ shape:", df_proc.shape)
df_proc.head()



# indicator ë¶™ì�€ ì»¬ëŸ¼ í™•ì�¸
[col for col in df_proc.columns if col.endswith("_isnan")]


import pandas as pd

RESP_PATH = "/kaggle/input/jane-street-real-time-market-data-forecasting/responders.csv"

# í—¤ë�”ë§Œ ì¶œë ¥
hdr = pd.read_csv(RESP_PATH, nrows=0)
print("ì»¬ëŸ¼ ëª©ë¡�:", hdr.columns.tolist())

# ë�°ì�´í„° ì�¼ë¶€ í™•ì�¸
df_check = pd.read_csv(RESP_PATH, nrows=5)
display(df_check)


sample_path = "/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id=0/part-0.parquet"
df_check = pd.read_parquet(sample_path)
print(df_check.columns.tolist())
display(df_check.head())


# ================================================
# Step 1) X / y / w ë¶„ë¦¬
# df_proc: NaN ì²˜ë¦¬ ë��ë‚œ DataFrame (ì˜ˆ: partition0)
# ================================================

# 1. feature ì»¬ëŸ¼
feat_cols   = [c for c in df_proc.columns if c.startswith("feature_") and not c.endswith("_isnan")]
isnan_cols  = [c for c in df_proc.columns if c.endswith("_isnan")]
X_cols      = feat_cols + isnan_cols

# 2. íƒ€ê¹ƒ ì„ íƒ� (ì˜ˆ: responder_0 í•˜ë‚˜ë§Œ)
TARGET_COL = "responder_0"
y = df_proc[TARGET_COL].astype("float32")

# 3. ê°€ì¤‘ì¹˜ (weightê°€ ì�ˆìœ¼ë©´ í™œìš©)
w = df_proc["weight"].astype("float32") if "weight" in df_proc.columns else None

# 4. ìµœì¢… X í–‰ë ¬
X = df_proc[X_cols].astype("float32")

print(f"[X] {X.shape} (features={len(X_cols)})")
print(f"[y] {y.shape}, target={TARGET_COL}")
if w is not None:
    print(f"[w] {w.shape}")


# ================================================
# Step 2) ìƒ˜í”Œë§�
# - ì „ì²´ ë�°ì�´í„° ì¤‘ ì�¼ë¶€ë§Œ ë�œë�¤ ì„ íƒ�
# ================================================
SAMPLE_N = 100_000  # ì›�í•˜ëŠ” í�¬ê¸° ì¡°ì ˆ (ex: 50k, 100k)

N = len(X)
if N > SAMPLE_N:
    sample_idx = np.random.RandomState(42).choice(N, size=SAMPLE_N, replace=False)
else:
    sample_idx = np.arange(N)

X_s = X.iloc[sample_idx].reset_index(drop=True)
y_s = y.iloc[sample_idx].reset_index(drop=True)
w_s = w.iloc[sample_idx].reset_index(drop=True) if w is not None else None

print(f"[Sample] {len(X_s)} rows selected from {N}")


# Step 1) í•™ìŠµ/ê²€ì¦� ë¶„ë¦¬

from sklearn.model_selection import train_test_split

# ìƒ˜í”Œë§�ë�œ ë�°ì�´í„° (X_s, y_s, w_s) ì‚¬ìš©
X_tr, X_va, y_tr, y_va = train_test_split(
    X_s, y_s, test_size=0.2, random_state=42
)

w_tr = w_s.iloc[X_tr.index] if w_s is not None else None
w_va = w_s.iloc[X_va.index] if w_s is not None else None


# Step 2) í•™ìŠµ

import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# ê¸°ë³¸ íŒŒë�¼ë¯¸í„° (ë¹ ë¥´ê²Œ baseline ë§Œë“¤ê¸°)
model = lgb.LGBMRegressor(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# fit ì�¸ì�� êµ¬ì„±
fit_kwargs = {
    "X": X_tr, "y": y_tr,
    "eval_set": [(X_va, y_va)],
    "eval_metric": "l2",
    # verbose ëŒ€ì‹  callbackìœ¼ë¡œ ë¡œê·¸ ì œì–´
    "callbacks": [lgb.log_evaluation(period=50)]
}
if w_tr is not None:
    fit_kwargs["sample_weight"] = w_tr
    fit_kwargs["eval_sample_weight"] = [w_va]

# í•™ìŠµ
model.fit(**fit_kwargs)

# ê²€ì¦� MSE í™•ì�¸
pred_va = model.predict(X_va)
mse = mean_squared_error(y_va, pred_va, sample_weight=w_va) if w_va is not None else mean_squared_error(y_va, pred_va)
print(f"[Valid] MSE = {mse:.6f}")


# Step 3) Feature Importance ê³„ì‚°

# ì¤‘ìš”ë�„ ì¶”ì¶œ
importances = model.feature_importances_
fi = (
    pd.DataFrame({"feature": X_s.columns, "importance": importances})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

print("[Feature Importance] top 20")
display(fi.head(20))

# ìƒ�ìœ„ 20ê°œ feature ì�´ë¦„ ë½‘ê¸°
top20_features = fi.head(20)["feature"].tolist()

# ê·¸ 20ê°œë§Œ ë²ˆí˜¸ìˆœìœ¼ë¡œ ì •ë ¬
top20_sorted = sorted(top20_features, key=lambda x: int(x.split("_")[1]))
top20_sorted

# ìƒ�ìœ„ Nê°œ í”¼ì²˜ ì €ì�¥
#TOP_N = 80
#top_features = fi.head(TOP_N)["feature"].tolist()

# íŒŒì�¼ë¡œ ì €ì�¥ (ì„ íƒ�)
#fi.to_csv("/kaggle/working/feature_importance.csv", index=False)
#with open("/kaggle/working/top_features.txt", "w") as f:
    #for feat in top_features:
        #f.write(feat + "\n")


import matplotlib.pyplot as plt

# ìƒ�ìœ„ 20ê°œ feature importance ì‹œê°�í™”
top20 = fi.head(20)

plt.figure(figsize=(10, 6))
plt.barh(top20["feature"], top20["importance"], color="skyblue")
plt.gca().invert_yaxis()  # ë†’ì�€ importanceê°€ ìœ„ë¡œ ì˜¤ë�„ë¡�
plt.xlabel("Importance Score")
plt.title("Top 20 Feature Importances (LightGBM)")
plt.show()


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -----------------------------
# 1) ì‚¬ìš©í•  feature ì¤€ë¹„
# -----------------------------
top20_features = top20_sorted  # importance ê¸°ì¤€ ìƒ�ìœ„ 20ê°œ ë¦¬ìŠ¤íŠ¸
target = "responder_0"         # ì˜ˆì‹œ: responder_0

usecols = ["date_id", "time_id", "symbol_id", "weight"] + top20_features + [target]

# -----------------------------
# 2) ë�°ì�´í„° ë¡œë“œ
# -----------------------------
sample_path = "/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id=0/part-0.parquet"
df_raw = pd.read_parquet(sample_path, columns=usecols)

# NaN ì²˜ë¦¬ (ì�´ë¯¸ indicator ì²˜ë¦¬ë�œ ê²½ìš°ë©´ pass, ì•„ë‹ˆë©´ fillna)
df_raw = df_raw.fillna(0)

# -----------------------------
# 3) Train/Valid ë¶„ë¦¬
# -----------------------------
X = df_raw[top20_features]
y = df_raw[target]
w = df_raw["weight"]

X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(X, y, w, test_size=0.2, random_state=42)

# -----------------------------
# 4) ëª¨ë�¸ í•™ìŠµ
# -----------------------------
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_tr, y_tr,
    sample_weight=w_tr,
    eval_set=[(X_va, y_va)],
    eval_sample_weight=[w_va],
    eval_metric="l2"
)


# -----------------------------
# 5) ì„±ëŠ¥ í�‰ê°€
# -----------------------------
y_pred = model.predict(X_va)
mse = mean_squared_error(y_va, y_pred, sample_weight=w_va)
print(f"[Valid] MSE = {mse:.6f}")


from sklearn.metrics import mean_squared_error

# ê²€ì¦� ë�°ì�´í„° ì˜ˆì¸¡
y_pred = model.predict(X_va)

# ê°€ì¤‘ì¹˜ í�¬í•¨ MSE ê³„ì‚°
mse = mean_squared_error(y_va, y_pred, sample_weight=w_va)

print(f"[Valid] MSE = {mse:.6f}")


# responder 6ìœ¼ë¡œ ì„±ëŠ¥ ì·¤

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -----------------------------
# 1) ì‚¬ìš©í•  feature ì¤€ë¹„
# -----------------------------
top20_features = top20_sorted   # importance ê¸°ì¤€ ìƒ�ìœ„ 20ê°œ ë¦¬ìŠ¤íŠ¸
target = "responder_6"          # ì�´ë²ˆì—” responder_6

usecols = ["date_id", "time_id", "symbol_id", "weight"] + top20_features + [target]

# -----------------------------
# 2) ë�°ì�´í„° ë¡œë“œ
# -----------------------------
sample_path = "/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id=0/part-0.parquet"
df_raw = pd.read_parquet(sample_path, columns=usecols)

# NaN ì²˜ë¦¬
df_raw = df_raw.fillna(0)

# -----------------------------
# 3) Train/Valid ë¶„ë¦¬
# -----------------------------
X = df_raw[top20_features]
y = df_raw[target]
w = df_raw["weight"]

X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(X, y, w, test_size=0.2, random_state=42)

# -----------------------------
# 4) ëª¨ë�¸ í•™ìŠµ
# -----------------------------
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_tr, y_tr,
    sample_weight=w_tr,
    eval_set=[(X_va, y_va)],
    eval_sample_weight=[w_va],
    eval_metric="l2"
)

# -----------------------------
# 5) ì„±ëŠ¥ í�‰ê°€
# -----------------------------
y_pred = model.predict(X_va)
mse = mean_squared_error(y_va, y_pred, sample_weight=w_va)
print(f"[Valid - responder_6] MSE = {mse:.6f}")


# responder_6ì�„ ëŒ€ìƒ�ìœ¼ë¡œ ìƒ�ìœ„ 30ê°œ featureì�¨ì„œ í•™ìŠµ

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -----------------------------
# 1) ì‚¬ìš©í•  feature ì¤€ë¹„
# -----------------------------
top30_features = fi.head(30)["feature"].tolist()   # ì¤‘ìš”ë�„ ìƒ�ìœ„ 30ê°œ feature ë¦¬ìŠ¤íŠ¸
target = "responder_6"

usecols = ["date_id", "time_id", "symbol_id", "weight"] + top30_features + [target]

# -----------------------------
# 2) ë�°ì�´í„° ë¡œë“œ
# -----------------------------
sample_path = "/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id=0/part-0.parquet"
df_raw = pd.read_parquet(sample_path, columns=usecols)

# NaN ì²˜ë¦¬
df_raw = df_raw.fillna(0)

# -----------------------------
# 3) Train/Valid ë¶„ë¦¬
# -----------------------------
X = df_raw[top30_features]
y = df_raw[target]
w = df_raw["weight"]

X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(X, y, w, test_size=0.2, random_state=42)

# -----------------------------
# 4) ëª¨ë�¸ í•™ìŠµ
# -----------------------------
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_tr, y_tr,
    sample_weight=w_tr,
    eval_set=[(X_va, y_va)],
    eval_sample_weight=[w_va],
    eval_metric="l2"
)

# -----------------------------
# 5) ì„±ëŠ¥ í�‰ê°€
# -----------------------------
y_pred = model.predict(X_va)
mse = mean_squared_error(y_va, y_pred, sample_weight=w_va)
print(f"[Valid - responder_6 | top30 features] MSE = {mse:.6f}")


import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# -----------------------------
# 1) ì‚¬ìš©í•  feature ì¤€ë¹„
# -----------------------------
top30_features = fi.head(30)["feature"].tolist()   # importance ê¸°ì¤€ ìƒ�ìœ„ 30ê°œ
target = "responder_6"
usecols = ["date_id", "time_id", "symbol_id", "weight"] + top30_features + [target]

# -----------------------------
# 2) ë�°ì�´í„° ë¡œë“œ
# -----------------------------
sample_path = "/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet/partition_id=0/part-0.parquet"
df_raw = pd.read_parquet(sample_path, columns=usecols).fillna(0)

X = df_raw[top30_features].values
y = df_raw[target].values
w = df_raw["weight"].values

# -----------------------------
# 3) K-Fold ì„¤ì •
# -----------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mse_scores = []

# -----------------------------
# 4) Foldë³„ í•™ìŠµ & í�‰ê°€
# -----------------------------
for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    print(f"\n[Fold {fold+1}]")
    
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    w_tr, w_va = w[tr_idx], w[va_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42,
        n_jobs=-1,
    )
    
    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_va, y_va)],
        eval_sample_weight=[w_va],
        eval_metric="l2"
    )
    
    y_pred = model.predict(X_va)
    mse = mean_squared_error(y_va, y_pred, sample_weight=w_va)
    mse_scores.append(mse)
    print(f"[Fold {fold+1}] MSE = {mse:.6f}")

# -----------------------------
# 5) CV í�‰ê·  & í‘œì¤€í�¸ì°¨
# -----------------------------
print("\n===============================")
print(f"[CV ê²°ê³¼] í�‰ê·  MSE = {np.mean(mse_scores):.6f} | í‘œì¤€í�¸ì°¨ = {np.std(mse_scores):.6f}")


# (feature, importance) íŠœí”Œë¡œ ë³´ê¸°
top30_pairs = list(zip(fi.head(30)["feature"], fi.head(30)["importance"]))
for f, imp in top30_pairs:
    print(f"{f}: {imp}")


# ì¤‘ìš”ë�„ ìˆœ(= ëª¨ë�¸ì�´ ê°€ì�¥ ë§�ì�´ ì“´ ìˆœ) Top30 ëª©ë¡�
top30_importance = fi.head(30)["feature"].tolist()
print("[Top30 by importance]")
print(top30_importance)

# ë²ˆí˜¸ìˆœ ì •ë ¬í•œ Top30 ëª©ë¡�
top30_sorted = sorted(top30_importance, key=lambda x: int(x.split("_")[1]))
print("\n[Top30 by feature index]")
print(top30_sorted)

