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


# Imports and settings
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap

# Display options
pd.set_option('display.max_columns', 120)
pd.set_option('display.width', 120)

# Plotting style
sns.set_theme(style="whitegrid", context="notebook")

# Reproducibility
RANDOM_STATE = 42

# Helper: safe bar labels
def _autolabel(ax):
    for p in ax.patches:
        height = p.get_height()
        if np.isfinite(height):
            ax.annotate(f"{height:.1f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, rotation=0)

print({
    'python': sys.version.split()[0],
    'pandas': pd.__version__,
    'numpy': np.__version__,
    'seaborn': sns.__version__,
})


# Load dataset and quick overview
DATA_DIR = "/kaggle/input/playground-series-s5e11"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")

# Read CSV
try:
    df = pd.read_csv(TRAIN_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find {TRAIN_PATH}. Please verify the path.")

print("Shape:", df.shape)
df.head(3)


# Target column preparation and class balance
# Locate target
TARGET_CANDIDATES = ["loan_paid_back", "paid_back", "target", "defaulted", "loan_status"]
found = [c for c in TARGET_CANDIDATES if c in df.columns]
if not found:
    raise KeyError(f"None of the expected target columns found. Available columns: {list(df.columns)[:20]} ...")
TARGET = found[0]

# Coerce to binary numeric y
raw_y = df[TARGET]
if raw_y.dtype == 'O':
    mapping = {
        'yes': 1, 'y': 1, 'true': 1, 't': 1, 1: 1,
        'no': 0, 'n': 0, 'false': 0, 'f': 0, 0: 0
    }
    y = raw_y.astype(str).str.strip().str.lower().map(mapping)
else:
    y = raw_y.astype(float)

if y.isna().any():
    # Attempt to infer binary by comparing unique values
    uniq = sorted(raw_y.dropna().unique())
    print("Unmapped target values (showing up to 10):", uniq[:10])
    raise ValueError("Target could not be coerced to binary. Please inspect unique values above.")

# Attach back (clean target), keep original for reference
if TARGET != "loan_paid_back":
    print(f"INFO: Using '{TARGET}' as target (renaming to 'loan_paid_back' for convenience)")
    df = df.rename(columns={TARGET: 'loan_paid_back'})
    TARGET = 'loan_paid_back'

df[TARGET] = y.astype(int)

# Class balance
vc = df[TARGET].value_counts(dropna=False).sort_index()
vp = df[TARGET].value_counts(normalize=True, dropna=False).sort_index()
class_balance = pd.DataFrame({"count": vc, "percent": (vp*100).round(2)})
print(class_balance)

# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.countplot(data=df, x=TARGET, ax=ax[0])
ax[0].set_title("Class counts")
_autolabel(ax[0])

sns.barplot(x=class_balance.index, y=class_balance["percent"], ax=ax[1], palette="pastel")
ax[1].set_title("Class percentage")
for p, val in zip(ax[1].patches, class_balance["percent" ].values):
    ax[1].annotate(f"{val:.1f}%", (p.get_x()+p.get_width()/2., p.get_height()),
                   ha='center', va='bottom', fontsize=9)
plt.tight_layout()

# Baseline
majority_rate = vp.max()
print(f"Baseline (predict majority class): accuracy={majority_rate:.3f}")


# Missingness per column
na_counts = df.isna().sum()
na_pct = (na_counts / len(df) * 100).round(2)
missing = (
    pd.DataFrame({"missing_count": na_counts, "missing_percent": na_pct})
    .sort_values("missing_percent", ascending=False)
)

# Display top 30
missing.head(30)


# Plot missingness (top 20 with missing values)
miss_plot = missing[missing.missing_count > 0].head(20)
if not miss_plot.empty:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=miss_plot.reset_index(), x="missing_percent", y="index", palette="flare")
    plt.xlabel("Missing (%)")
    plt.ylabel("Feature")
    plt.title("Top missing features (up to 20)")
    plt.tight_layout()
else:
    print("No missing values detected.")


# Numeric columns and summary
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]
print(f"Numeric cols ({len(num_cols)}):", num_cols[:20], ("..." if len(num_cols) > 20 else ""))

df[num_cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T.head(20)


# Correlation of numeric features with target (Pearson)
if num_cols:
    corr = df[num_cols].corrwith(df[TARGET]).sort_values(key=lambda s: s.abs(), ascending=False)
    corr_df = corr.to_frame(name='corr_with_target')
    display(corr_df.head(20))

    # Plot top 10 by absolute correlation
    top_corr = corr.abs().sort_values(ascending=False).head(10).index
    plt.figure(figsize=(10, 5))
    sns.barplot(x=corr.loc[top_corr].values, y=top_corr, palette="crest")
    plt.title("Top numeric correlations with target (abs)")
    plt.xlabel("Pearson correlation")
    plt.ylabel("Feature")
    plt.tight_layout()
else:
    print("No numeric features detected.")


# Distributions for top numeric features (by correlation)
plot_cols = []
if num_cols:
    corr = df[num_cols].corrwith(df[TARGET]).abs().sort_values(ascending=False)
    plot_cols = list(corr.head(min(6, len(corr))).index)

if plot_cols:
    n = len(plot_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = axes.ravel()
    for i, col in enumerate(plot_cols):
        sns.histplot(data=df, x=col, hue=TARGET, multiple='stack', kde=True, ax=axes[i])
        axes[i].set_title("\n".join(wrap(f"{col} by {TARGET}", 30)))
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
else:
    print("No numeric columns to plot.")


# Identify categorical columns
cat_cols = [c for c in df.select_dtypes(include=['object', 'category', 'bool']).columns if c != TARGET]
print(f"Categorical cols ({len(cat_cols)}):", cat_cols[:20], ("..." if len(cat_cols) > 20 else ""))

# Cardinality
card = df[cat_cols].nunique(dropna=False).sort_values(ascending=False) if cat_cols else pd.Series(dtype=int)
card.head(20)


# Target rate by category (for low-cardinality features)
plots_done = 0
max_plots = 3
if cat_cols:
    global_mean = df[TARGET].mean()
    scores = []
    for col in cat_cols:
        nunq = df[col].nunique(dropna=False)
        if nunq <= 20 and nunq >= 2:
            grp = df.groupby(col, dropna=False)[TARGET].agg(['mean', 'count']).rename(columns={'mean': 'target_rate'})
            # Weighted variance of target_rate around global mean as a simple signal score
            score = ((grp['target_rate'] - global_mean)**2 * grp['count']).sum() / grp['count'].sum()
            scores.append((col, score, nunq))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Show top 10 scored categorical features
    pd.DataFrame(scores, columns=['feature', 'score', 'nunique']).head(10)

    # Plot top features
    for col, score, nunq in scores[:max_plots]:
        grp = df.groupby(col, dropna=False)[TARGET].mean().sort_values(ascending=False)
        plt.figure(figsize=(8, max(3, 0.35*len(grp))))
        sns.barplot(x=grp.values, y=grp.index, palette="mako")
        plt.title(f"{col} â€” target rate by category (score={score:.4f})")
        plt.xlabel("Mean loan_paid_back")
        plt.ylabel(col)
        plt.tight_layout()
        plots_done += 1

if plots_done == 0:
    print("No low-cardinality categorical features suitable for target-rate plotting were found.")


# Outlier counts per numeric feature (IQR rule)
outlier_stats = []
for col in num_cols:
    s = df[col].dropna()
    if s.empty:
        continue
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr
    count = ((s < lower) | (s > upper)).sum()
    outlier_stats.append((col, int(count), float(count)/len(s)))

outlier_df = pd.DataFrame(outlier_stats, columns=['feature', 'outlier_count', 'outlier_rate']).sort_values('outlier_rate', ascending=False)
outlier_df.head(20)


# Boxplots for top outlier-heavy features
top_outliers = outlier_df.head(3)['feature'].tolist() if not outlier_df.empty else []
if top_outliers:
    n = len(top_outliers)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 3))
    if n == 1:
        axes = [axes]
    for i, col in enumerate(top_outliers):
        sns.boxplot(data=df, x=TARGET, y=col, ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
else:
    print("No numeric features for outlier plotting.")


# Box/strip plots for top 3 numeric features by |corr|
plot_cols = []
if num_cols:
    corr = df[num_cols].corrwith(df[TARGET]).abs().sort_values(ascending=False)
    plot_cols = list(corr.head(min(3, len(corr))).index)

if plot_cols:
    fig, axes = plt.subplots(1, len(plot_cols), figsize=(5*len(plot_cols), 3))
    if len(plot_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, plot_cols):
        sns.boxplot(data=df, x=TARGET, y=col, ax=ax)
        sns.stripplot(data=df.sample(min(500, len(df)), random_state=RANDOM_STATE), x=TARGET, y=col, ax=ax, color='k', alpha=0.25, size=2)
        ax.set_title(f"{col} vs {TARGET}")
    plt.tight_layout()
else:
    print("No numeric features to visualize interactions.")


# Potential leakage by name
leaky_patterns = ["paid", "pay", "default", "status", "target"]
leaky_cols = [c for c in df.columns if any(p in c.lower() for p in leaky_patterns) and c != TARGET]
print("Columns with potentially leaky names:", leaky_cols)

# Near-perfect correlation with target (numeric only)
high_corr = []
for c in num_cols:
    corr = df[[c, TARGET]].dropna().corr().iloc[0,1]
    if pd.notna(corr) and abs(corr) > 0.95:
        high_corr.append((c, corr))
print("Near-perfect numeric correlations with target:", high_corr)

# Duplicate rows
dup_rows = int(df.duplicated().sum())
print("Duplicate rows:", dup_rows)

# ID columns and duplicate IDs
id_like = [c for c in df.columns if c.lower() in ("id",) or c.lower().endswith("_id") or c.lower().startswith("id_")]
dup_ids = {}
for c in id_like:
    dup_ids[c] = int(df[c].duplicated().sum())
print("ID-like columns:", id_like)
print("Duplicate counts for ID-like columns:", dup_ids)

