import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

plt.style.use("default")
sns.set(font_scale=1.0)
plt.rcParams["figure.figsize"] = (10, 6)


# Competition data (S05E11)
COMP_TRAIN_PATH = "/kaggle/input/playground-series-s5e11/train.csv"
COMP_TEST_PATH  = "/kaggle/input/playground-series-s5e11/test.csv"

# Original dataset
ORIG_PATH       = "/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv"

train = pd.read_csv(COMP_TRAIN_PATH)
test  = pd.read_csv(COMP_TEST_PATH)
orig  = pd.read_csv(ORIG_PATH)

print("Competition train shape:", train.shape)
print("Competition test  shape:", test.shape)
print("Original dataset shape:", orig.shape)

display(train.head())
display(test.head())
display(orig.head())


def summarize_df(df, name="df"):
    print(f"===== SUMMARY: {name} =====")
    print("Shape:", df.shape)
    print("\nDtypes:")
    print(df.dtypes)
    
    print("\nMissing values (count):")
    print(df.isna().sum())
    
    print("\nMissing values (%):")
    print((df.isna().mean() * 100).round(3))
    
    print("\nDescriptive statistics (numeric):")
    display(df.describe().T)
    
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        print("\nCategorical columns:", cat_cols)
        for c in cat_cols:
            print(f"\nValue counts for '{c}':")
            display(df[c].value_counts(dropna=False).to_frame("count"))
    else:
        print("\nNo categorical columns detected.")
    print("=" * 50)

def get_num_cat_cols(df, target_col=None, ignore_cols=None):
    """Return numeric & categorical columns (excluding id + ignore_cols)."""
    if ignore_cols is None:
        ignore_cols = []
    ignore_cols = set(ignore_cols)
    ignore_cols.add("id")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target_col and target_col in num_cols:
        num_cols.remove(target_col)

    num_cols = [c for c in num_cols if c not in ignore_cols]
    cat_cols = [c for c in cat_cols if c not in ignore_cols]

    return num_cols, cat_cols

def plot_numeric_hist(df, cols, title_prefix="", bins=60):
    """Plot histograms for ALL numeric columns, using ALL rows."""
    if not cols:
        print(f"No numeric columns to plot for {title_prefix}")
        return
    
    for col in cols:
        plt.figure(figsize=(10,4))
        sns.histplot(df[col], bins=bins, kde=True)
        plt.title(f"{title_prefix}{col} distribution")
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()

def plot_categorical_counts(df, cols, title_prefix=""):
    """Plot counts for ALL categories (no top limit)."""
    if not cols:
        print(f"No categorical columns to plot for {title_prefix}")
        return
    
    for col in cols:
        vc = df[col].value_counts(dropna=False)
        
        plt.figure(figsize=(max(12, len(vc)/3), 5))
        sns.barplot(x=vc.index.astype(str), y=vc.values)
        plt.title(f"{title_prefix}{col} counts (ALL categories)")
        plt.xticks(rotation=90)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()



def correlation_heatmap(df, num_cols, title="Correlation Heatmap"):
    """Full correlation heatmap — may be large."""
    if not num_cols:
        print(f"No numeric columns for {title}")
        return
    
    corr = df[num_cols].corr()
    plt.figure(figsize=(min(0.8*len(num_cols)+4, 22), 16))
    sns.heatmap(corr, annot=False, cmap="viridis", center=0)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def target_numeric_relationships(df, target, num_cols):
    """Boxplots for target vs all numeric columns."""
    if not num_cols:
        print("No numeric columns for target analysis.")
        return
    
    for col in num_cols:
        plt.figure(figsize=(10,4))
        sns.boxplot(x=df[target], y=df[col])
        plt.title(f"{col} vs {target}")
        plt.tight_layout()
        plt.show()

def target_categorical_relationships(df, target, cat_cols):
    """Category → mean(target) for ALL categories."""
    if not cat_cols:
        print("No categorical columns for target vs categorical analysis.")
        return
    
    for col in cat_cols:
        agg = df.groupby(col)[target].agg(["mean", "count"]).sort_values("mean", ascending=False)
        
        print(f"\n=== {col} vs {target} (ALL CATEGORIES) ===")
        display(agg)
        
        plt.figure(figsize=(max(12, len(agg)/3), 5))
        sns.barplot(x=agg.index.astype(str), y=agg["mean"])
        plt.xticks(rotation=90)
        plt.title(f"Mean {target} by {col}")
        plt.tight_layout()
        plt.show()

def compare_num_distributions(df1, df2, cols, name1="df1", name2="df2", bins=60):
    """Compare numeric distributions using ALL rows."""
    if not cols:
        print(f"No numeric columns in common for {name1} vs {name2}.")
        return
    
    for col in cols:
        d1 = df1[col].dropna()
        d2 = df2[col].dropna()
        
        stat, pval = ks_2samp(d1, d2)
        print(f"=== {col} | {name1} vs {name2} ===")
        print(f"KS statistic: {stat:.4f}, p-value: {pval:.3e}")
        
        plt.figure(figsize=(10,4))
        sns.kdeplot(d1, label=name1, fill=True, alpha=0.4)
        sns.kdeplot(d2, label=name2, fill=True, alpha=0.4)
        plt.title(f"{col} distribution: {name1} vs {name2} (ALL ROWS)")
        plt.legend()
        plt.tight_layout()
        plt.show()

def compare_cat_distributions(df1, df2, cols, name1="df1", name2="df2"):
    """Compare categorical distributions using ALL categories."""
    if not cols:
        print(f"No categorical columns in common for {name1} vs {name2}.")
        return
    
    for col in cols:
        vc1 = df1[col].value_counts(normalize=True).rename(f"{name1}_pct")
        vc2 = df2[col].value_counts(normalize=True).rename(f"{name2}_pct")
        
        comparison = pd.concat([vc1, vc2], axis=1).fillna(0)
        comparison["diff"] = comparison[f"{name1}_pct"] - comparison[f"{name2}_pct"]

        print(f"\n=== {col} distribution (% share): {name1} vs {name2} (ALL categories) ===")
        display(comparison.sort_values("diff", key=abs, ascending=False))
        
        plt.figure(figsize=(max(12, len(comparison)/3), 5))
        comparison[[f"{name1}_pct", f"{name2}_pct"]].plot(kind="bar")
        plt.title(f"{col} distribution comparison (ALL CATEGORIES)")
        plt.xticks(rotation=90)
        plt.ylabel("Proportion")
        plt.tight_layout()
        plt.show()


summarize_df(train, "Competition Train")


summarize_df(test,  "Competition Test")


summarize_df(orig,  "Original Dataset")


TARGET_COMP = "loan_paid_back"   # competition target (train only)
TARGET_ORIG = "loan_paid_back"   # original target

if TARGET_COMP in train.columns:
    print("=== Competition Train Target Distribution ===")
    display(train[TARGET_COMP].value_counts().to_frame("count"))
    display(train[TARGET_COMP].value_counts(normalize=True).to_frame("proportion"))
    
    plt.figure()
    sns.countplot(x=TARGET_COMP, data=train)
    plt.title("Competition: loan_paid_back distribution")
    plt.tight_layout()
    plt.show()


if TARGET_ORIG in orig.columns:
    print("=== Original Dataset Target Distribution ===")
    display(orig[TARGET_ORIG].value_counts().to_frame("count"))
    display(orig[TARGET_ORIG].value_counts(normalize=True).to_frame("proportion"))
    
    plt.figure()
    sns.countplot(x=TARGET_ORIG, data=orig)
    plt.title("Original: loan_paid_back distribution")
    plt.tight_layout()
    plt.show()


num_cols_train, cat_cols_train = get_num_cat_cols(train, target_col=TARGET_COMP)
print("Numeric columns (train):", num_cols_train)
print("Categorical columns (train):", cat_cols_train)


# Univariate numeric distributions
plot_numeric_hist(train, num_cols_train, title_prefix="[Train]")


# Categorical counts
plot_categorical_counts(train, cat_cols_train, title_prefix="[Train] ")


# Correlation heatmap (numeric + target)
corr_cols_train = num_cols_train + [TARGET_COMP]
correlation_heatmap(train, corr_cols_train, title="Train Numeric Correlations (incl. target)")


# Target relationships - numeric
target_numeric_relationships(train, TARGET_COMP, num_cols_train)


# Target relationships - categorical
target_categorical_relationships(train, TARGET_COMP, cat_cols_train)


num_cols_test, cat_cols_test = get_num_cat_cols(test)
print("Numeric columns (test):", num_cols_test)
print("Categorical columns (test):", cat_cols_test)


# Univariate numeric distributions (test)
plot_numeric_hist(test, num_cols_test, title_prefix="[Test]")


# Categorical counts (test)
plot_categorical_counts(test, cat_cols_test, title_prefix="[Test]")


# Columns common between train and test (to check shift)
common_num_cols = sorted(list(set(num_cols_train).intersection(num_cols_test)))
common_cat_cols = sorted(list(set(cat_cols_train).intersection(cat_cols_test)))

print("Common numeric columns:", common_num_cols)
print("Common categorical columns:", common_cat_cols)


# Numeric distribution comparison: train vs test
compare_num_distributions(train, test, common_num_cols,
                          name1="train", name2="test",
                          bins=50)


# Categorical distribution comparison: train vs test
compare_cat_distributions(train, test, common_cat_cols,
                          name1="train", name2="test")


num_cols_orig, cat_cols_orig = get_num_cat_cols(orig, target_col=TARGET_ORIG)
print("Numeric columns (orig):", num_cols_orig)
print("Categorical columns (orig):", cat_cols_orig)


# Univariate numeric distributions
plot_numeric_hist(orig, num_cols_orig, title_prefix="[Orig]")


# Categorical counts
plot_categorical_counts(orig, cat_cols_orig, title_prefix="[Orig]")


# Correlation heatmap (original, numeric + target)
corr_cols_orig = num_cols_orig + [TARGET_ORIG]
correlation_heatmap(orig, corr_cols_orig, title="Original Numeric Correlations (incl. target)")


# Target relationships - numeric
target_numeric_relationships(orig, TARGET_ORIG, num_cols_orig)


# Target relationships - categorical
target_categorical_relationships(orig, TARGET_ORIG, cat_cols_orig)


# Overlapping columns (ignore id)
overlap_cols = sorted(list(set(train.columns).intersection(orig.columns)))
overlap_cols = [c for c in overlap_cols if c not in ["id"]]  # exclude id explicitly
print("Overlapping columns between original and competition train:", overlap_cols)


# Use the already-computed num/cat splits to define overlapping numeric/categorical
overlap_num_cols = [c for c in overlap_cols if (c in num_cols_train) and (c in num_cols_orig)]
overlap_cat_cols = [c for c in overlap_cols if (c in cat_cols_train) and (c in cat_cols_orig)]

print("Overlapping numeric columns:", overlap_num_cols)
print("Overlapping categorical columns:", overlap_cat_cols)


# Numeric: compare distributions
compare_num_distributions(orig, train, overlap_num_cols,
                          name1="original", name2="train",
                          bins=50)


# Categorical: compare distributions
compare_cat_distributions(orig, train, overlap_cat_cols,
                          name1="original", name2="train")


def summary_stats(df, cols, prefix):
    if not cols:
        return pd.DataFrame()
    return (df[cols]
            .agg(["mean", "std", "min", "max"])
            .T
            .add_prefix(prefix))

if overlap_num_cols:
    stats_orig = summary_stats(orig,  overlap_num_cols, "orig_")
    stats_train = summary_stats(train, overlap_num_cols, "train_")
    stats_compare = stats_orig.join(stats_train)
    stats_compare["mean_diff"] = stats_compare["train_mean"] - stats_compare["orig_mean"]
    stats_compare["std_ratio"] = stats_compare["train_std"] / stats_compare["orig_std"]
    print("Numeric summary comparison: original vs train")
    display(stats_compare)


# Correlation with target (original vs train)
def corr_with_target(df, num_cols, target, name):
    if not num_cols:
        return pd.DataFrame()
    corr = df[num_cols + [target]].corr()[target].drop(target)
    corr = corr.to_frame(name)
    return corr

if (TARGET_COMP in train.columns) and (TARGET_ORIG in orig.columns):
    common_target_num = sorted(list(set(num_cols_train).intersection(num_cols_orig)))
    if common_target_num:
        corr_orig  = corr_with_target(orig,  common_target_num, TARGET_ORIG,  "corr_orig")
        corr_train = corr_with_target(train, common_target_num, TARGET_COMP, "corr_train")
        corr_compare = corr_orig.join(corr_train)
        corr_compare["abs_diff"] = (corr_compare["corr_train"] - corr_compare["corr_orig"]).abs()
        print("Correlation with target (original vs train) for common numeric features")
        display(corr_compare.sort_values("abs_diff", ascending=False))


train_copy = train.copy()

# Encode categoricals
cat_cols = train_copy.select_dtypes('object').columns
encoder = OrdinalEncoder()
train_copy[cat_cols] = encoder.fit_transform(train_copy[cat_cols])

# Split
X = train_copy.drop('loan_paid_back', axis=1)
y = train_copy['loan_paid_back']

model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X, y)

importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False)


orig_copy = orig.copy()

# Encode categoricals
cat_cols = orig_copy.select_dtypes('object').columns
encoder = OrdinalEncoder()
orig_copy[cat_cols] = encoder.fit_transform(orig_copy[cat_cols])

# Split
Xo = orig_copy.drop('loan_paid_back', axis=1)
yo = orig_copy['loan_paid_back']

model2 = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
model2.fit(Xo, yo)

orig_importances = pd.Series(model2.feature_importances_, index=Xo.columns)
orig_importances.sort_values(ascending=False)




