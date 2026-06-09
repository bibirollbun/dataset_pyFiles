import os, gc, sys, math, json, warnings, textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, skew, kurtosis
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.6f}")


DATA_DIR = "/kaggle/input/playground-series-s5e10"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path  = os.path.join(DATA_DIR, "test.csv")
sub_path   = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(train_path), train_path
assert os.path.exists(test_path), test_path

# Efficient read (infers low_memory=False for safety)
train = pd.read_csv(train_path, low_memory=False)
test  = pd.read_csv(test_path, low_memory=False)
sample_sub = pd.read_csv(sub_path) if os.path.exists(sub_path) else None

train.shape, test.shape, None if sample_sub is None else sample_sub.shape


print("Train head:")
display(train.head(3))
print("\nTest head:")
display(test.head(3))


print("\nTrain info:")
train.info()
print("\nTest info:")
test.info()


print("\nMissing values (%):")
display(pd.DataFrame({
    "train_missing_%": train.isna().mean() * 100,
    "test_missing_%":  test.isna().mean() * 100
}).sort_index())


target_col = "accident_risk"
id_col = "id" if "id" in train.columns else None

train_cols = set(train.columns)
test_cols  = set(test.columns)
print("Columns only in train:", sorted(train_cols - test_cols))
print("Columns only in test:",  sorted(test_cols - train_cols))
print("Target present?", target_col in train.columns)


num_cols = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c]) and c not in [target_col]]
cat_cols = [c for c in train.columns if pd.api.types.is_object_dtype(train[c]) or str(train[c].dtype)=="category"]
bool_cols = [c for c in train.columns if train[c].dtype==bool]

cat_cols = sorted(list(set(cat_cols) | set(bool_cols) - {target_col}))

print(f"Numeric features ({len(num_cols)}):", num_cols)
print(f"Categorical features ({len(cat_cols)}):", cat_cols)


def dataset_health(df, name="df"):
    dup_rows = df.duplicated().sum()
    print(f"[{name}] rows: {len(df):,} | duplicate rows: {dup_rows:,}")
    if "id" in df.columns:
        dup_ids = df["id"].duplicated().sum()
        print(f"[{name}] unique ids: {df['id'].nunique():,} | duplicate ids: {dup_ids:,}")
    print(f"[{name}] memory usage: {df.memory_usage(deep=True).sum()/1024**2:,.2f} MB")

dataset_health(train, "train")
dataset_health(test,  "test")

# ID overlap (should ideally be disjoint)
if id_col:
    inter = set(train[id_col]).intersection(set(test[id_col]))
    print(f"Train/Test ID intersection: {len(inter):,}")


assert target_col in train.columns, "Target not found in train"
y = train[target_col]

print("Target summary stats:")
display(y.describe(percentiles=[0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]))


print("Skew:", skew(y))
print("Kurtosis:", kurtosis(y))


plt.figure(figsize=(8,5))
plt.hist(y, bins=50)
plt.title("accident_risk distribution")
plt.xlabel("accident_risk")
plt.ylabel("count")
plt.show()


def cat_cardinality_report(df, cats):
    rows = []
    for c in cats:
        vc = df[c].value_counts(dropna=False)
        rows.append({
            "col": c,
            "dtype": str(df[c].dtype),
            "unique": df[c].nunique(dropna=False),
            "top": vc.index[0],
            "top_count": int(vc.iloc[0]),
            "top_%": 100 * vc.iloc[0] / len(df)
        })
    return pd.DataFrame(rows).sort_values(["unique","col"])

cat_report = cat_cardinality_report(train, cat_cols)
display(cat_report)


def num_summary(df, nums):
    desc = df[nums].describe().T
    desc["missing_%"] = 100 * (1 - (desc["count"] / len(df)))
    # basic IQR outlier share
    def iqr_outlier_share(s):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        return ((s < lower) | (s > upper)).mean()
    desc["iqr_outlier_%"] = [iqr_outlier_share(df[c]) * 100 for c in desc.index]
    return desc

display(num_summary(train, num_cols))


schema = []
for c in sorted(set(train.columns).intersection(test.columns)):
    t_dtype = str(train[c].dtype)
    s_dtype = str(test[c].dtype)
    row = {"col": c, "train_dtype": t_dtype, "test_dtype": s_dtype, "dtype_match": t_dtype==s_dtype}
    if pd.api.types.is_numeric_dtype(train[c]):
        row["train_min"], row["train_max"] = train[c].min(), train[c].max()
        row["test_min"],  row["test_max"]  = test[c].min(),  test[c].max()
        row["range_overlap"] = not (row["test_max"] < row["train_min"] or row["test_min"] > row["train_max"])
    schema.append(row)

schema_df = pd.DataFrame(schema).sort_values(["dtype_match","col"], ascending=[True, True])
display(schema_df)


def ks_for_numeric(col):
    a, b = train[col].dropna(), test[col].dropna()
    if len(a)>0 and len(b)>0:
        stat, p = ks_2samp(a, b)
        return stat, p
    return np.nan, np.nan

def chi2_for_categorical(col, max_cats=200):
    a, b = train[col].astype("category"), test[col].astype("category")
    cats = list(set(a.cat.categories) | set(b.cat.categories))
    if len(cats) == 0 or len(cats)>max_cats:
        return np.nan, np.nan
    a_counts = a.value_counts().reindex(cats, fill_value=0).values
    b_counts = b.value_counts().reindex(cats, fill_value=0).values
    table = np.vstack([a_counts, b_counts])
    stat, p, _, _ = chi2_contingency(table)
    return stat, p

rows = []
for c in num_cols:
    stat, p = ks_for_numeric(c)
    rows.append({"col": c, "type": "numeric", "statistic": stat, "p_value": p})
for c in cat_cols:
    stat, p = chi2_for_categorical(c)
    rows.append({"col": c, "type": "categorical", "statistic": stat, "p_value": p})

drift_df = pd.DataFrame(rows).sort_values("p_value")
display(drift_df)


corr = train[num_cols + [target_col]].corr(numeric_only=True)
display(corr[target_col].sort_values(ascending=False).to_frame("corr_with_target"))


plt.figure(figsize=(6,5))
plt.imshow(corr, aspect='auto')
plt.title("Correlation matrix (numeric)")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)
plt.yticks(range(len(corr.index)), corr.index, fontsize=8)
plt.tight_layout()
plt.show()


def target_by_bins(df, col, ycol=target_col, bins=20):
    tmp = df[[col, ycol]].dropna().copy()
    tmp["bin"] = pd.qcut(tmp[col], q=bins, duplicates="drop")
    out = tmp.groupby("bin")[ycol].agg(["count","mean","std","min","max"])
    out["feature"] = col
    return out.reset_index()

bin_tables = []
for c in num_cols:
    bin_tables.append(target_by_bins(train, c, bins=20))
binned_df = pd.concat(bin_tables, ignore_index=True)
display(binned_df.head(40))


top_num = corr[target_col].drop(labels=[target_col]).abs().sort_values(ascending=False).index[:1].tolist()
if top_num:
    c = top_num[0]
    plt.figure(figsize=(7,5))
    plt.scatter(train[c], train[target_col], s=2, alpha=0.5)
    plt.title(f"{c} vs accident_risk")
    plt.xlabel(c); plt.ylabel("accident_risk")
    plt.show()


df_mi = train.copy()

encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    df_mi[c] = le.fit_transform(df_mi[c].astype(str))
    encoders[c] = le

features_for_mi = [c for c in df_mi.columns if c not in [target_col, id_col] and pd.api.types.is_numeric_dtype(df_mi[c])]
mi_vals = mutual_info_regression(df_mi[features_for_mi], df_mi[target_col], random_state=42)
mi_df = pd.DataFrame({"feature": features_for_mi, "mutual_info": mi_vals}).sort_values("mutual_info", ascending=False)
display(mi_df)


group_stats = []
for c in cat_cols:
    g = train.groupby(c)[target_col].agg(["count","mean","std","min","max"])
    g["col"] = c
    group_stats.append(g.reset_index())
group_stats_df = pd.concat(group_stats, axis=0, ignore_index=True)
display(group_stats_df.sort_values(["col","count"], ascending=[True, False]).head(50))


# Basic effect test (Kruskal, robust to non-normality)
test_rows = []
for c in cat_cols:
    # only keep top 30 categories to keep compute sane
    top_vals = train[c].value_counts().head(30).index
    samples = [train.loc[train[c]==val, target_col].values for val in top_vals]
    if len(samples)>=2:
        stat, p = stats.kruskal(*samples)
        test_rows.append({"col": c, "kruskal_stat": stat, "p_value": p, "n_groups": len(samples)})
effect_df = pd.DataFrame(test_rows).sort_values("p_value")
display(effect_df.head(20))


if not effect_df.empty:
    top_cat = effect_df.iloc[0]["col"]
    order = train[top_cat].value_counts().head(8).index
    means = train.groupby(top_cat)[target_col].mean().loc[order]
    plt.figure(figsize=(8,5))
    plt.bar(range(len(means)), means.values)
    plt.xticks(range(len(means)), means.index, rotation=45, ha="right")
    plt.title(f"Mean accident_risk by {top_cat} (top 8)")
    plt.ylabel("mean accident_risk")
    plt.tight_layout()
    plt.show()


# Pick two most correlated numeric features with target (excluding target)
ranked = corr[target_col].drop(labels=[target_col]).abs().sort_values(ascending=False).index.tolist()
pair = ranked[:2] if len(ranked)>=2 else ranked
if len(pair)==2:
    a, b = pair
    dfp = train[[a,b,target_col]].dropna().copy()
    # Bin both and compute target mean in grid
    dfp["A_bin"] = pd.qcut(dfp[a], 20, duplicates="drop")
    dfp["B_bin"] = pd.qcut(dfp[b], 20, duplicates="drop")
    pivot = dfp.pivot_table(index="A_bin", columns="B_bin", values=target_col, aggfunc="mean")
    plt.figure(figsize=(7,6))
    plt.imshow(pivot, aspect='auto', origin='lower')
    plt.title(f"Mean accident_risk across {a} x {b} bins")
    plt.colorbar()
    plt.xticks(range(len(pivot.columns)), range(len(pivot.columns)), rotation=90, fontsize=7)
    plt.yticks(range(len(pivot.index)), range(len(pivot.index)), fontsize=7)
    plt.tight_layout()
    plt.show()


# 1) Extremely high correlation with target?
high_corr = corr[target_col].drop(labels=[target_col])
suspects = high_corr[high_corr.abs() > 0.95].sort_values(ascending=False)
display(suspects.to_frame("corr_with_target"))


# 2) Data-copy check: exact duplicates within train on feature set (excluding id, target)
feature_cols = [c for c in train.columns if c not in [id_col, target_col] and c is not None]
dup_feats = train.duplicated(subset=feature_cols).sum()
print("Exact duplicate feature rows in train:", dup_feats)


probe = []
for c in num_cols:
    s = train[c].dropna()
    if (s<=0).any():
        continue
    # log transform
    s_log = np.log1p(s)
    probe.append({
        "feature": c,
        "orig_skew": float(skew(s)),
        "log1p_skew": float(skew(s_log))
    })
probe_df = pd.DataFrame(probe).sort_values("orig_skew", ascending=False)
display(probe_df.head(20))


# Very tiny linear probe (numeric only) to see approximate R2 (not for submission!)
from sklearn.linear_model import LinearRegression
tiny_feats = [c for c in num_cols if c != target_col]
X = train[tiny_feats].fillna(train[tiny_feats].median())
model = LinearRegression(n_jobs=None) if "n_jobs" in LinearRegression().get_params() else LinearRegression()
model.fit(X, y)
y_hat = model.predict(X)
print("Linear probe R^2 (in-sample, numeric only):", r2_score(y, y_hat))




