import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8")
sns.set_palette("deep")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 140)

# =========================
# CONFIG
# =========================
DATA_DIR = "/kaggle/input/playground-series-s5e11"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

ID_COL = "id"
TARGET_COL = "loan_paid_back"  # change here if needed

RANDOM_STATE = 42

print("Using data dir:", DATA_DIR)


train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("âœ… Data loaded")
print(f"Train shape: {train.shape}")
print(f"Test  shape: {test.shape}")

display(train.head())
display(test.head())


print("\n=== Train Info ===")
print(train.info())

print("\n=== Test Info ===")
print(test.info())

def memory_usage_mb(df, name="df"):
    mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"{name} memory usage: {mem:.2f} MB")

memory_usage_mb(train, "train")
memory_usage_mb(test, "test")


if TARGET_COL in train.columns:
    target_counts = train[TARGET_COL].value_counts(dropna=False).sort_index()
    target_ratio = train[TARGET_COL].value_counts(normalize=True).sort_index() * 100

    print("Target counts:")
    display(target_counts.to_frame(name="count"))

    print("\nTarget ratio (%):")
    display(target_ratio.round(3).to_frame(name="ratio_%"))

    plt.figure(figsize=(5, 4))
    sns.countplot(data=train, x=TARGET_COL)
    plt.title("Target class distribution")
    plt.tight_layout()
    plt.show()
else:
    print(f"â�Œ TARGET_COL='{TARGET_COL}' not in train. Please update TARGET_COL.")


def missing_values_table(df, name="df"):
    mis_count = df.isnull().sum()
    mis_pct = (mis_count / len(df)) * 100
    mis_df = pd.DataFrame(
        {"column": df.columns, "missing_count": mis_count, "missing_pct": mis_pct}
    )
    mis_df = mis_df[mis_df["missing_count"] > 0].sort_values(
        "missing_pct", ascending=False
    )
    print(f"\nğŸ“Š Missing values in {name}:")
    if mis_df.empty:
        print("No missing values.")
    else:
        display(mis_df)
    return mis_df

missing_train = missing_values_table(train, "train")
missing_test = missing_values_table(test, "test")

plt.figure(figsize=(10, 4))
sns.heatmap(train.isnull(), cbar=False, yticklabels=False)
plt.title("Missingness Heatmap â€“ Train")
plt.tight_layout()
plt.show()


dup_all = train.duplicated().sum()
print(f"Duplicate rows in train (all columns): {dup_all}")

if ID_COL in train.columns:
    dup_no_id = train.drop(columns=[ID_COL]).duplicated().sum()
    print(f"Duplicate rows in train (excluding {ID_COL}): {dup_no_id}")
else:
    print(f"{ID_COL} not found in train; skipping 'exclude id' duplicate check.")


def get_feature_types(df, target_col, id_col):
    feature_cols = [c for c in df.columns if c not in [target_col, id_col]]
    num_cols = []
    cat_cols = []
    for col in feature_cols:
        if df[col].dtype in ["int64", "float64"]:
            num_cols.append(col)
        else:
            cat_cols.append(col)
    return num_cols, cat_cols

num_cols, cat_cols = get_feature_types(train, TARGET_COL, ID_COL)

print(f"Numerical features ({len(num_cols)}):")
print(num_cols)
print("\nCategorical features ({len(cat_cols)}):")
print(cat_cols)


def numerical_detailed_stats(df, num_cols):
    rows = []
    n = len(df)
    for col in num_cols:
        s = df[col]
        stats = s.describe()
        zero_count = (s == 0).sum()
        neg_count = (s < 0).sum()
        missing_count = s.isnull().sum()
        row = {
            "column": col,
            "dtype": s.dtype,
            "count": stats["count"],
            "mean": stats["mean"],
            "std": stats["std"],
            "min": stats["min"],
            "25%": stats["25%"],
            "50%": stats["50%"],
            "75%": stats["75%"],
            "max": stats["max"],
            "skew": s.skew(),
            "kurtosis": s.kurtosis(),
            "missing_count": missing_count,
            "missing_pct": 100 * missing_count / n,
            "n_unique": s.nunique(dropna=True),
            "zero_count": zero_count,
            "zero_pct": 100 * zero_count / n,
            "negative_count": neg_count,
            "negative_pct": 100 * neg_count / n,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("column")

if num_cols:
    num_stats = numerical_detailed_stats(train, num_cols)
    print("ğŸ“ˆ Detailed numerical stats:")
    display(num_stats)
else:
    print("No numerical features found.")


def categorical_detailed_stats(df, cat_cols, max_top=5):
    rows = []
    n = len(df)
    for col in cat_cols:
        s = df[col]
        vc = s.value_counts(dropna=False)
        top_val = vc.index[0] if len(vc) > 0 else np.nan
        top_freq = vc.iloc[0] if len(vc) > 0 else np.nan
        missing_count = s.isnull().sum()
        rows.append(
            {
                "column": col,
                "dtype": s.dtype,
                "n_unique": s.nunique(dropna=True),
                "top_value": top_val,
                "top_freq": top_freq,
                "top_freq_pct": 100 * top_freq / n if n > 0 and not pd.isna(top_freq) else np.nan,
                "missing_count": missing_count,
                "missing_pct": 100 * missing_count / n,
            }
        )
    cat_stats = pd.DataFrame(rows).sort_values("column")
    print("ğŸ“Š Categorical stats summary:")
    display(cat_stats)

    # print sample value counts
    for col in cat_cols:
        print(f"\n--- Value counts for {col} (top {max_top}) ---")
        display(df[col].value_counts(dropna=False).head(max_top))

if cat_cols:
    categorical_detailed_stats(train, cat_cols)
else:
    print("No categorical features found.")


def outlier_report(df, num_cols):
    rows = []
    n = len(df)
    for col in num_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1 if Q3 > Q1 else 0

        if IQR == 0:
            iqr_outliers = pd.Series([False] * len(df))
        else:
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            iqr_outliers = (df[col] < lower) | (df[col] > upper)

        # Z-score
        mean = s.mean()
        std = s.std()
        if std == 0:
            z_outliers = pd.Series([False] * len(df))
        else:
            z = (df[col] - mean) / std
            z_outliers = z.abs() > 3

        row = {
            "column": col,
            "iqr_lower": (Q1 - 1.5 * IQR),
            "iqr_upper": (Q3 + 1.5 * IQR),
            "outlier_count_iqr": int(iqr_outliers.sum()),
            "outlier_pct_iqr": 100 * iqr_outliers.sum() / n,
            "outlier_count_z": int(z_outliers.sum()),
            "outlier_pct_z": 100 * z_outliers.sum() / n,
        }
        rows.append(row)
    out_df = pd.DataFrame(rows).sort_values("outlier_pct_iqr", ascending=False)
    return out_df

if num_cols:
    outliers_df = outlier_report(train, num_cols)
    print("ğŸš¨ Outlier report (sorted by IQR %):")
    display(outliers_df)
else:
    print("No numerical features, skipping outlier report.")


def check_suspicious_values(df):
    rules = []

    # generic helpers: only add rule if column exists
    if "age" in df.columns:
        rules.append(
            ("age", (df["age"] < 18) | (df["age"] > 100), "age outside [18, 100]")
        )

    if "credit_score" in df.columns:
        rules.append(
            (
                "credit_score",
                (df["credit_score"] < 300) | (df["credit_score"] > 900),
                "credit_score outside [300, 900]",
            )
        )

    if "interest_rate" in df.columns:
        rules.append(
            (
                "interest_rate",
                (df["interest_rate"] < 0) | (df["interest_rate"] > 100),
                "interest_rate outside [0, 100]",
            )
        )

    if "debt_to_income_ratio" in df.columns:
        rules.append(
            (
                "debt_to_income_ratio",
                (df["debt_to_income_ratio"] < 0) | (df["debt_to_income_ratio"] > 5),
                "debt_to_income_ratio <0 or >5",
            )
        )

    if "loan_amount" in df.columns:
        rules.append(
            (
                "loan_amount",
                df["loan_amount"] <= 0,
                "loan_amount <= 0",
            )
        )

    if "annual_income" in df.columns:
        rules.append(
            (
                "annual_income",
                df["annual_income"] <= 0,
                "annual_income <= 0",
            )
        )

    results = []
    n = len(df)
    for col, mask, desc in rules:
        count = int(mask.sum())
        pct = 100 * count / n
        results.append({"column": col, "rule": desc, "count": count, "pct": pct})

    if results:
        bad_df = pd.DataFrame(results).sort_values("pct", ascending=False)
        print("âš ï¸� Suspicious / bad-value checks (train):")
        display(bad_df)
    else:
        print("No domain rules matched existing columns (or no suspicious values).")

check_suspicious_values(train)


def low_variance_features(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    rows = []
    n = len(df)
    for col in df.columns:
        if col in exclude_cols:
            continue
        s = df[col]
        n_unique = s.nunique(dropna=True)
        unique_ratio = n_unique / max(n, 1)
        rows.append(
            {
                "column": col,
                "dtype": s.dtype,
                "n_unique": n_unique,
                "unique_ratio": unique_ratio,
            }
        )
    res = pd.DataFrame(rows).sort_values("unique_ratio")
    const = res[res["n_unique"] <= 1]
    near_const = res[(res["unique_ratio"] < 0.01) & (res["n_unique"] > 1)]

    print("Features sorted by uniqueness ratio:")
    display(res.head(20))

    print("\nğŸ”’ Constant features (n_unique <= 1):")
    display(const)

    print("\nâš ï¸� Near-constant features (unique_ratio < 1% & >1 unique):")
    display(near_const)

    return res, const, near_const

lv_res, const_res, near_const_res = low_variance_features(
    train, exclude_cols=[ID_COL, TARGET_COL]
)


if num_cols and TARGET_COL in train.columns:
    corr = train[num_cols + [TARGET_COL]].corr()

    # correlation with target
    target_corr = corr[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)
    print("ğŸ”— Correlation with target:")
    display(target_corr.to_frame("corr_with_target"))

    # highly correlated feature pairs
    pairs = []
    for i, c1 in enumerate(num_cols):
        for j, c2 in enumerate(num_cols):
            if j <= i:
                continue
            r = corr.loc[c1, c2]
            if abs(r) >= 0.9:
                pairs.append({"feature_1": c1, "feature_2": c2, "corr": r})

    if pairs:
        high_corr_pairs = pd.DataFrame(pairs).sort_values(
            "corr", key=lambda x: x.abs(), ascending=False
        )
        print("\nğŸ”¥ Highly correlated pairs (|r| >= 0.9):")
        display(high_corr_pairs)
    else:
        print("\nNo feature pairs with |corr| >= 0.9")

    # heatmap â€“ top features by |corr with target|
    top_n = min(20, len(num_cols))
    top_features = target_corr.abs().sort_values(ascending=False).head(top_n).index
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        train[list(top_features) + [TARGET_COL]].corr(),
        cmap="coolwarm",
        center=0,
        annot=False,
    )
    plt.title("Correlation heatmap â€“ top features + target")
    plt.tight_layout()
    plt.show()
else:
    print("Cannot compute correlations â€“ missing numeric columns or target.")


def plot_numeric_vs_target(df, num_cols, target_col, sample_size=50000):
    if target_col not in df.columns:
        print(f"Target {target_col} not found in df.")
        return
    if not num_cols:
        print("No numerical features.")
        return

    if len(df) > sample_size:
        plot_df = df.sample(sample_size, random_state=RANDOM_STATE)
    else:
        plot_df = df.copy()

    for col in num_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        sns.kdeplot(
            data=plot_df,
            x=col,
            hue=target_col,
            common_norm=False,
            fill=True,
            alpha=0.4,
            ax=axes[0],
        )
        axes[0].set_title(f"{col} â€“ KDE by {target_col}")

        sns.boxplot(data=plot_df, x=target_col, y=col, ax=axes[1])
        axes[1].set_title(f"{col} â€“ boxplot by {target_col}")

        plt.tight_layout()
        plt.show()

plot_numeric_vs_target(train, num_cols, TARGET_COL)


def categorical_target_analysis(df, cat_cols, target_col, max_categories=20):
    if target_col not in df.columns:
        print(f"Target {target_col} not found in df.")
        return
    if not cat_cols:
        print("No categorical features.")
        return

    for col in cat_cols:
        print(f"\n==== {col} vs {target_col} ====")
        grp = (
            df.groupby(col)[target_col]
            .agg(["count", "mean"])
            .rename(columns={"mean": "target_mean"})
            .sort_values("target_mean", ascending=False)
        )
        display(grp.head(20))

        top_cats = grp.sort_values("count", ascending=False).head(max_categories).index
        plot_df = df[df[col].isin(top_cats)]

        plt.figure(figsize=(10, 4))
        sns.barplot(
            data=plot_df,
            x=col,
            y=target_col,
            estimator=np.mean,
            order=top_cats,
        )
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{col} â€“ mean {target_col} by category (top {max_categories})")
        plt.tight_layout()
        plt.show()

categorical_target_analysis(train, cat_cols, TARGET_COL)


def compare_numeric_train_test(train, test, num_cols, sample_size=50000):
    if not num_cols:
        print("No numerical features.")
        return

    if len(train) > sample_size:
        train_s = train.sample(sample_size, random_state=RANDOM_STATE)
    else:
        train_s = train.copy()

    if len(test) > sample_size:
        test_s = test.sample(sample_size, random_state=RANDOM_STATE)
    else:
        test_s = test.copy()

    for col in num_cols:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(train_s[col], label="train", fill=True, alpha=0.4)
        sns.kdeplot(test_s[col], label="test", fill=True, alpha=0.4)
        plt.title(f"Train vs Test â€“ {col}")
        plt.legend()
        plt.tight_layout()
        plt.show()

def compare_categorical_train_test(train, test, cat_cols, max_categories=15):
    if not cat_cols:
        print("No categorical features.")
        return

    for col in cat_cols:
        plt.figure(figsize=(10, 4))

        train_prop = (
            train[col].value_counts(normalize=True)
            .head(max_categories)
            .sort_index()
        )
        test_prop = (
            test[col].value_counts(normalize=True)
            .head(max_categories)
            .sort_index()
        )

        all_idx = sorted(set(train_prop.index) | set(test_prop.index))
        train_prop = train_prop.reindex(all_idx).fillna(0)
        test_prop = test_prop.reindex(all_idx).fillna(0)

        x = np.arange(len(all_idx))
        width = 0.4

        plt.bar(x - width / 2, train_prop.values, width=width, label="train")
        plt.bar(x + width / 2, test_prop.values, width=width, label="test")

        plt.xticks(x, [str(i) for i in all_idx], rotation=45, ha="right")
        plt.title(f"Train vs Test â€“ {col} (proportions)")
        plt.legend()
        plt.tight_layout()
        plt.show()

print("ğŸ”� Comparing numerical distributions (train vs test)...")
compare_numeric_train_test(train, test, num_cols)

print("ğŸ”� Comparing categorical distributions (train vs test)...")
compare_categorical_train_test(train, test, cat_cols)


print("===== FINAL EDA SUMMARY =====")
print(f"Train rows: {len(train)}, columns: {train.shape[1]}")
print(f"Test  rows: {len(test)}, columns: {test.shape[1]}")

print(f"\nID column: {ID_COL}")
print(f"Target column: {TARGET_COL}")

if TARGET_COL in train.columns:
    tc = train[TARGET_COL].value_counts()
    tr = train[TARGET_COL].value_counts(normalize=True) * 100
    print("\nTarget distribution (count):", tc.to_dict())
    print("Target distribution (%):", tr.round(2).to_dict())

# Missing info
if not missing_train.empty:
    print("\nColumns with missing values in train:")
    print(
        missing_train[["column", "missing_pct"]]
        .sort_values("missing_pct", ascending=False)
        .to_dict(orient="records")
    )
else:
    print("\nNo missing values in train.")

# Duplicates
print(f"\nDuplicate rows (all columns): {dup_all}")
if ID_COL in train.columns:
    print(f"Duplicate rows (excluding {ID_COL}): {dup_no_id}")

# Outliers summary
if num_cols:
    many_outliers = outliers_df[outliers_df["outlier_pct_iqr"] > 5]
    print(
        f"\nOutlier summary (IQR): {len(many_outliers)} numeric features have > 5% outliers."
    )
    print(
        many_outliers[["column", "outlier_pct_iqr"]]
        .sort_values("outlier_pct_iqr", ascending=False)
        .to_dict(orient="records")
    )

# Constant / near-constant
print(f"\nConstant features: {len(const_res)}")
print(f"Near-constant features: {len(near_const_res)}")

print("\nCheck all plots & tables above for more detailed understanding.")
print("===== END OF EDA =====")

