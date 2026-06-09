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


import os
import gc
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import log_loss, roc_auc_score

import xgboost as xgb


DATA_DIR = "/kaggle/input/avazu-ctr-prediction"
TRAIN_PATH = os.path.join(DATA_DIR, "train.gz")
TEST_PATH = os.path.join(DATA_DIR, "test.gz")
SUB_PATH = os.path.join(DATA_DIR, "sampleSubmission.gz")

RANDOM_STATE = 42
SAMPLE_FRAC = 0.05    # 5% sample from the full 40M rows (~2M rows)
CHUNKSIZE = 1_000_000 # number of rows per chunk when reading

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 4)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything(RANDOM_STATE)

print("Using XGBoost version:", xgb.__version__)
print("Train path:", TRAIN_PATH)
print("Test path:", TEST_PATH)


sampled_chunks = []
total_rows = 0
sampled_rows = 0

for i, chunk in enumerate(pd.read_csv(TRAIN_PATH,
                                      compression="gzip",
                                      chunksize=CHUNKSIZE)):
    total_rows += len(chunk)
    # Sample a fixed fraction from each chunk for a balanced sample
    chunk_sample = chunk.sample(frac=SAMPLE_FRAC,
                                random_state=RANDOM_STATE)
    sampled_chunks.append(chunk_sample)
    sampled_rows += len(chunk_sample)

    print(f"Processed chunk {i + 1:02d}, "
          f"chunk rows = {len(chunk):,}, "
          f"sampled rows so far = {sampled_rows:,}")

train_df = pd.concat(sampled_chunks, ignore_index=True)
del sampled_chunks, chunk
gc.collect()

print("\nFull train (sampled) shape:", train_df.shape)
print(train_df.head())


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # Make sure hour is a zero-padded 8-char string
    hour_str = df["hour"].astype(str).str.zfill(8)

    # Build full datetime string: "20YY-MM-DD HH:00:00"
    dt_str = (
        "20"
        + hour_str.str[0:2] + "-"
        + hour_str.str[2:4] + "-"
        + hour_str.str[4:6] + " "
        + hour_str.str[6:8] + ":00:00"
    )
    dt = pd.to_datetime(dt_str, format="%Y-%m-%d %H:%M:%S")

    df["day_of_week"] = dt.dt.dayofweek.astype("int8")  # 0=Mon, 6=Sun
    df["hour_of_day"] = dt.dt.hour.astype("int8")       # 0..23
    # Use date as an integer (YYYYMMDD) to build time-based splits later
    df["date_int"] = (dt.dt.year * 10000 +
                      dt.dt.month * 100 +
                      dt.dt.day).astype("int32")

    # We won't keep raw hour anymore
    df = df.drop(columns=["hour"])
    return df

train_df = add_time_features(train_df)
print(train_df[["click", "day_of_week", "hour_of_day", "date_int"]].head())
print("Date range in sample:", train_df["date_int"].min(), "->", train_df["date_int"].max())

# Global CTR in sampled train
global_ctr = train_df["click"].mean()
print(f"Global CTR in sample: {global_ctr:.4f}")


EDA_N = 500_000
if len(train_df) > EDA_N:
    eda_df = train_df.sample(n=EDA_N, random_state=RANDOM_STATE)
else:
    eda_df = train_df.copy()

print("EDA sample shape:", eda_df.shape)


print("=== Global CTR on EDA sample ===")
global_ctr_eda = eda_df["click"].mean()
print(f"Global CTR (click=1 ratio): {global_ctr_eda:.4f}")

class_counts = eda_df["click"].value_counts().sort_index()
class_ratios = eda_df["click"].value_counts(normalize=True).sort_index()

summary_cls = pd.DataFrame({
    "count": class_counts,
    "ratio": class_ratios
})
summary_cls.index = ["no_click (0)", "click (1)"]
print(summary_cls)

# Plot class imbalance
fig, ax = plt.subplots()
sns.barplot(x=summary_cls.index, y=summary_cls["count"], ax=ax)
ax.set_title("Class Counts (EDA sample)")
ax.set_ylabel("Number of impressions")
ax.set_xlabel("Click label")
plt.tight_layout()
plt.show()


print("=== Feature cardinality (number of unique values) ===")
# Use a subset to keep it fast if you want
card_df = train_df.drop(columns=["click"]).nunique().sort_values(ascending=False)
card_df = card_df.to_frame(name="n_unique")
print(card_df)

# Show top 15 highest-cardinality features
display(card_df.head(15))

fig, ax = plt.subplots(figsize=(10, 6))
card_df.head(15).plot(kind="barh", ax=ax)
ax.invert_yaxis()
ax.set_title("Top 15 Features by Cardinality")
ax.set_xlabel("Number of unique values")
plt.tight_layout()
plt.show()


print("=== CTR by site_category ===")
plot_ctr_by_feature(eda_df, "site_category", top_n=20)

print("=== CTR by app_category ===")
plot_ctr_by_feature(eda_df, "app_category", top_n=20)


def top_ctr_table(df: pd.DataFrame, feature: str, top_n: int = 15, min_impressions: int = 1000):
    grouped = (
        df.groupby(feature)["click"]
        .agg(impressions="count", ctr="mean")
        .reset_index()
    )
    grouped = grouped[grouped["impressions"] >= min_impressions]
    grouped = grouped.sort_values("impressions", ascending=False).head(top_n)
    return grouped

print("=== Top site_domain by impressions (with CTR) ===")
top_sites = top_ctr_table(eda_df, "site_domain", top_n=15, min_impressions=2000)
display(top_sites)

print("=== Top app_domain by impressions (with CTR) ===")
top_apps = top_ctr_table(eda_df, "app_domain", top_n=15, min_impressions=2000)
display(top_apps)


heat_dw_hr = (
    eda_df
    .groupby(["day_of_week", "hour_of_day"])["click"]
    .mean()
    .unstack("hour_of_day")
    .sort_index()
)

plt.figure(figsize=(12, 4))
sns.heatmap(heat_dw_hr, annot=False, cmap="viridis")
plt.title("CTR by day_of_week x hour_of_day")
plt.ylabel("day_of_week (0=Mon)")
plt.xlabel("hour_of_day (0-23)")
plt.tight_layout()
plt.show()


heat_dev_hr = (
    eda_df
    .groupby(["device_type", "hour_of_day"])["click"]
    .mean()
    .unstack("hour_of_day")
    .sort_index()
)

plt.figure(figsize=(12, 4))
sns.heatmap(heat_dev_hr, annot=False, cmap="magma")
plt.title("CTR by device_type x hour_of_day")
plt.ylabel("device_type")
plt.xlabel("hour_of_day")
plt.tight_layout()
plt.show()


heat_conn_dw = (
    eda_df
    .groupby(["device_conn_type", "day_of_week"])["click"]
    .mean()
    .unstack("day_of_week")
    .sort_index()
)

plt.figure(figsize=(8, 4))
sns.heatmap(heat_conn_dw, annot=False, cmap="plasma")
plt.title("CTR by device_conn_type x day_of_week")
plt.ylabel("device_conn_type")
plt.xlabel("day_of_week (0=Mon)")
plt.tight_layout()
plt.show()


def plot_ctr_by_feature(df: pd.DataFrame, feature: str, top_n: int = 10):
    """
    Plots CTR (mean click) by a categorical or discrete feature,
    showing top_n most frequent categories.
    """
    tmp = df.copy()

    # Limit to most frequent categories if needed
    if tmp[feature].dtype == "object" or tmp[feature].nunique() > top_n:
        top_values = tmp[feature].value_counts().index[:top_n]
        tmp = tmp[tmp[feature].isin(top_values)]

    grouped = (
        tmp.groupby(feature)["click"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    fig, ax1 = plt.subplots()
    sns.barplot(data=grouped, x=feature, y="mean", ax=ax1)
    ax1.set_title(f"CTR by {feature}")
    ax1.set_ylabel("CTR")
    ax1.set_xlabel(feature)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print(grouped)


print("=== CTR by hour_of_day ===")
plot_ctr_by_feature(eda_df, "hour_of_day", top_n=24)

print("=== CTR by day_of_week ===")
plot_ctr_by_feature(eda_df, "day_of_week", top_n=7)

print("=== CTR by banner_pos ===")
plot_ctr_by_feature(eda_df, "banner_pos", top_n=10)

print("=== CTR by device_type ===")
plot_ctr_by_feature(eda_df, "device_type", top_n=10)

print("=== CTR by device_conn_type ===")
plot_ctr_by_feature(eda_df, "device_conn_type", top_n=10)


test_df = pd.read_csv(TEST_PATH, compression="gzip")
print("Raw test shape:", test_df.shape)

test_df = add_time_features(test_df)
print("Test with time features shape:", test_df.shape)

test_ids = test_df["id"].values


# Drop id from both; click is only in train
train_df = train_df.drop(columns=["id"])
test_df = test_df.drop(columns=["id"])

target_col = "click"

# We will NOT use 'date_int' as a model feature to avoid leaking the time split
numeric_cols = ["hour_of_day", "day_of_week"]  # can behave like numeric
time_split_col = "date_int"

# All other columns except target & date_int are treated as categorical
feature_cols = [c for c in train_df.columns if c not in [target_col]]
cat_cols = [c for c in feature_cols if c not in numeric_cols + [time_split_col]]

print("All feature columns:", feature_cols)
print("Numeric feature columns:", numeric_cols)
print("Categorical feature columns:", cat_cols)

# Ensure categorical columns are strings
for col in cat_cols:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

# Ordinal encode categorical features
enc = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

print("Fitting OrdinalEncoder on categorical features...")
train_df[cat_cols] = enc.fit_transform(train_df[cat_cols])
test_df[cat_cols] = enc.transform(test_df[cat_cols])

# Convert everything (features) to float32 for XGBoost GPU
for col in feature_cols:
    if col != time_split_col:  # we will use date_int only for splitting
        train_df[col] = train_df[col].astype("float32")
        test_df[col] = test_df[col].astype("float32")

# Keep labels and dates separately
y = train_df[target_col].values.astype("float32")
dates = train_df[time_split_col].values  # for time-based split

print("Train label distribution (click=1 ratio):", y.mean())
print("Unique dates in sample:", np.unique(dates))


unique_dates = np.sort(np.unique(dates))
max_date = unique_dates[-1]
print("Using last date for validation:", max_date)

val_mask = dates == max_date
train_mask = ~val_mask

train_idx = np.where(train_mask)[0]
val_idx = np.where(val_mask)[0]

print(f"Train rows: {train_idx.shape[0]:,}, Val rows: {val_idx.shape[0]:,}")

# Final feature set excludes target and date_int for modeling
model_feature_cols = [c for c in feature_cols if c != time_split_col]

X_train = train_df.loc[train_idx, model_feature_cols]
y_train = y[train_idx]

X_valid = train_df.loc[val_idx, model_feature_cols]
y_valid = y[val_idx]

print("X_train shape:", X_train.shape)
print("X_valid shape:", X_valid.shape)

# Free some memory
gc.collect()


dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

watchlist = [(dtrain, "train"), (dvalid, "valid")]

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",  # main metric for the competition
    "tree_method": "gpu_hist",  # GPU acceleration
    "predictor": "gpu_predictor",
    "learning_rate": 0.1,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "lambda": 10.0,
    "gamma": 0.0,
    "max_bin": 256
}

num_boost_round = 500
early_stopping_rounds = 30

print("Starting XGBoost training on GPU...")
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    evals=watchlist,
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=50
)

print("Best iteration:", model.best_iteration)


valid_pred_proba = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))

val_logloss = log_loss(y_valid, valid_pred_proba)
val_auc = roc_auc_score(y_valid, valid_pred_proba)

print(f"Validation Logloss: {val_logloss:.6f}")
print(f"Validation AUC:     {val_auc:.6f}")


dtest = xgb.DMatrix(test_df[model_feature_cols])
test_pred_proba = model.predict(
    dtest,
    iteration_range=(0, model.best_iteration + 1)
)

print("Test predictions shape:", test_pred_proba.shape)
print("Pred proba stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(
    test_pred_proba.min(), test_pred_proba.max(), test_pred_proba.mean()
))


submission = pd.read_csv(SUB_PATH, compression="gzip")
print("Sample submission shape:", submission.shape)
print(submission.head())

submission["click"] = test_pred_proba
submission.to_csv("submission.csv", index=False)

print("Saved submission.csv")


from scipy.stats import norm


def ab_test_ctr(
    df: pd.DataFrame,
    group_col: str,
    group_a,
    group_b,
    label_col: str = "click",
    alpha: float = 0.05
):
    """
    Two-proportion z-test for CTR difference between group_a and group_b.
    group_a = "treatment-like" group
    group_b = "control-like" group
    """
    df_ab = df[df[group_col].isin([group_a, group_b])].copy()

    a = df_ab[df_ab[group_col] == group_a]
    b = df_ab[df_ab[group_col] == group_b]

    n_a = len(a)
    n_b = len(b)
    k_a = a[label_col].sum()
    k_b = b[label_col].sum()

    p_a = k_a / n_a
    p_b = k_b / n_b
    diff = p_a - p_b

    # pooled proportion
    p_pool = (k_a + k_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))

    z = diff / se
    p_value = 2 * (1 - norm.cdf(abs(z)))  # two-sided

    # 95% CI for the difference
    z_alpha = norm.ppf(1 - alpha / 2)
    ci_low = diff - z_alpha * se
    ci_high = diff + z_alpha * se

    print(f"A/B on {group_col}: {group_a} (A) vs {group_b} (B)")
    print(f"Group A size: {n_a:,}, CTR_A: {p_a:.4f}")
    print(f"Group B size: {n_b:,}, CTR_B: {p_b:.4f}")
    print(f"Difference (A - B): {diff:.4f}")
    print(f"95% CI for diff: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"z-stat: {z:.3f}, p-value: {p_value:.4e}")
    if p_value < alpha:
        print(f"Result: statistically significant at alpha = {alpha}")
    else:
        print(f"Result: NOT statistically significant at alpha = {alpha}")

    return {
        "n_a": n_a,
        "n_b": n_b,
        "ctr_a": p_a,
        "ctr_b": p_b,
        "diff": diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "z": z,
        "p_value": p_value,
    }


print("device_conn_type value_counts:", eda_df["device_conn_type"].value_counts())

ab_result_conn = ab_test_ctr(
    eda_df,
    group_col="device_conn_type",
    group_a=0,
    group_b=2
)


print("banner_pos value_counts:", eda_df["banner_pos"].value_counts())

ab_result_banner = ab_test_ctr(
    eda_df,
    group_col="banner_pos",
    group_a=1,  # say position 1
    group_b=0   # baseline position 0
)




