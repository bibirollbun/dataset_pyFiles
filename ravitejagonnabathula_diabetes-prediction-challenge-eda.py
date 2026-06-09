# Core Libraries
import os
import math
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


from statsmodels.stats.outliers_influence import variance_inflation_factor




# List dataset files
for dirname, _, filenames in os.walk('/kaggle/input/playground-series-s5e12'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train  = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


print("train.shape:",train.shape)
print("test.shape:",test.shape)


train.head()


train.describe()


train.isnull().sum()


train.duplicated().sum()



# ============================
# Base dtype detection
# ============================

target = "diagnosed_diabetes"   # change this to your target column

num_cols_raw  = train.select_dtypes(include='number').columns.tolist()
cat_cols_raw  = train.select_dtypes(include=['object', 'category']).columns.tolist()
bool_cols     = train.select_dtypes(include='bool').columns.tolist()
date_cols     = train.select_dtypes(include='datetime').columns.tolist()

# Remove id + target from automatic grouping
protected_cols = ["id", target]

# ============================
#  Detect binary 0/1 numeric columns
# ============================

binary_num_cols = [
    col for col in num_cols_raw
    if train[col].nunique() == 2 
       and col not in bool_cols 
       and col not in protected_cols
]

# ============================
#  Detect low-cardinality numeric categorical 
# (e.g., 0/1/2/3/4 codes)
# ============================

# Threshold â€“ can tune (20 is standard)
low_cardinality_threshold = 20

low_card_numeric_cols = [
    col for col in num_cols_raw
    if 2 < train[col].nunique() <= low_cardinality_threshold 
       and col not in binary_num_cols
       and col not in protected_cols
]

# ============================
#  Final Numerical Columns (True numeric features)
# ============================

true_num_cols = [
    col for col in num_cols_raw
    if col not in binary_num_cols 
    and col not in low_card_numeric_cols
    and col not in protected_cols
]

# ============================
#  Final Categorical Columns
# ============================

cat_cols = (
    cat_cols_raw +          # original object/category
    bool_cols +             # boolean flags
    binary_num_cols +       # 0/1
    low_card_numeric_cols   # 0/1/2/3/4 codes
)

# Remove duplicates
cat_cols = list(dict.fromkeys(cat_cols))

# ============================
#  Detect text columns (long strings)
# ============================

text_cols = [
    col for col in cat_cols
    if train[col].dtype == 'object'
       and train[col].astype(str).str.len().mean() > 50
]

# ============================
# STEP 7 â€” Detect high-cardinality categorical
# ============================

high_cardinality_threshold = 20

high_cardinality_cols = [
    col for col in cat_cols
    if train[col].nunique() > high_cardinality_threshold
]

low_cardinality_cols = [
    col for col in cat_cols
    if train[col].nunique() <= high_cardinality_threshold
]



print(f"True Numeric Columns ({len(true_num_cols)}):\n", true_num_cols, "\n")
print(f"Binary Categorical (0/1) ({len(binary_num_cols)}):\n", binary_num_cols, "\n")
print(f"Low-Cardinality Numeric Categories ({len(low_card_numeric_cols)}):\n", low_card_numeric_cols, "\n")
print(f"Categorical Columns ({len(cat_cols)}):\n", cat_cols, "\n")
print(f"Boolean Columns ({len(bool_cols)}):\n", bool_cols, "\n")
print(f"Text Columns ({len(text_cols)}):\n", text_cols, "\n")
print(f"Date Columns ({len(date_cols)}):\n", date_cols, "\n")
print(f"High Cardinality Categorical ({len(high_cardinality_cols)}):\n", high_cardinality_cols, "\n")
print(f"Low Cardinality Categorical ({len(low_cardinality_cols)}):\n", low_cardinality_cols, "\n")




# Class distribution
print("Class Distribution:")
print(train[target].value_counts())

print("\nClass Percentages:")
print(train[target].value_counts(normalize=True) * 100)

# Balanced vs imbalanced check
major_class_pct = train[target].value_counts(normalize=True).max() * 100

if major_class_pct > 70:
    print("\n Dataset is IMBALANCED.")
    print(f"Majority class = {major_class_pct:.2f}%")
else:
    print("\n Dataset is BALANCED.")
    print(f"Majority class = {major_class_pct:.2f}%")


# Class counts and percentages
class_counts = train[target].value_counts()
class_percent = train[target].value_counts(normalize=True) * 100

# Custom colors
colors = ['#FF6961', '#90EE90']   # red & light green

# Apply modern seaborn style
sns.set(style="whitegrid", font_scale=1.2)

# --- Create side-by-side plots ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# =====================
# Plot 1: Class Counts
# =====================
sns.barplot(
    x=class_counts.index.astype(str),
    y=class_counts.values,
    palette=colors,
    ax=axes[0],
    edgecolor="black"
)

axes[0].set_title("Class Distribution", fontsize=15, weight='bold')
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")

# Show values on top of bars
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + max(class_counts.values)*0.01, f"{v:,}", 
                 ha='center', fontsize=12, weight='bold')

# =====================
# Plot 2: Percentages
# =====================
sns.barplot(
    x=class_percent.index.astype(str),
    y=class_percent.values,
    palette=colors,
    ax=axes[1],
    edgecolor="black"
)

axes[1].set_title("Class Percentages (%)", fontsize=15, weight='bold')
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Percentage")

# Show percentage values
for i, v in enumerate(class_percent.values):
    axes[1].text(i, v + 1, f"{v:.2f}%", 
                 ha='center', fontsize=12, weight='bold')

plt.tight_layout()
plt.show()




for col in true_num_cols:
    if col in ("id", target):
        continue
        
    print("="*70)
    print(f"ğŸ“Œ Feature: {col}")
    print("="*70)
    
    # Basic Stats
    print(train[col].describe(), "\n")
    
    # Skewness
    print(f"Skewness: {train[col].skew():.3f}")
    
    # Zero-inflation
    print(f"Zero %: {(train[col] == 0).mean()*100:.2f}%")
    
    # Outlier Count using IQR
    Q1, Q3 = train[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers = ((train[col] < lower) | (train[col] > upper)).sum()
    print(f"Outliers: {outliers}")
    
    # Percentage of outliers
    outlier_pct = (outliers / len(train[col])) * 100
    print(f"Outlier %: {outlier_pct:.2f}%")
    
    # Range
    print(f"Min: {train[col].min()} | Max: {train[col].max()} | Mean: {train[col].mean():.2f} | Median: {train[col].median():.2f}")
    
    # =======================
    #  Dual Plots
    # =======================
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # -------------------------------
    # Plot 1: Histogram + KDE
    # -------------------------------
    sns.histplot(
        train[col],
        bins=40,
        kde=True,
        ax=axes[0],
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.5
    )
    
    axes[0].set_title(f"Distribution of {col}", fontsize=14)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Density")
    axes[0].grid(alpha=0.25)
    
    # -------------------------------
    # Plot 2:  Boxplot
    # -------------------------------
    sns.boxplot(
        x=train[col],
        ax=axes[1],
        color="#55C890",
        linewidth=1.2,
        fliersize=3
    )
    
    median = train[col].median()
    axes[1].axvline(median, color="#D62728", linestyle="--", linewidth=1.2)
    
    axes[1].text(
        median,
        0.05,
        f"Median: {median:.2f}",
        color="#D62728",
        fontsize=10,
        rotation=45,
        ha="left",
        weight="bold"
    )
    
    axes[1].set_title(f"Boxplot of {col}", fontsize=14)
    axes[1].set_xlabel("")
    axes[1].grid(axis="x", linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    plt.show()
 


# Thresholds (you can change)
rare_threshold = 0.01      # 1%
cardinality_threshold = 20 # more than 20 unique categories = high-cardinality

for col in cat_cols:

    print("="*70)
    print(f"Categorical Feature: {col}")
    print("="*70)

    # --- Value Counts ---
    vc = train[col].value_counts()
    print(" Value Counts:")
    print(vc, "\n")

    # --- Frequency (%) ---
    freq_pct = (train[col].value_counts(normalize=True) * 100).round(2)
    print(" Category Frequency (%):")
    print(freq_pct, "\n")

    # --- Rare Categories ---
    rare_cats = freq_pct[freq_pct < (rare_threshold * 100)].index.tolist()
    print(f" Rare Categories (< {rare_threshold*100}%): {rare_cats if rare_cats else 'None'}")

    # --- High Cardinality Check ---
    unique_count = train[col].nunique()
    if unique_count > cardinality_threshold:
        print(f" High Cardinality: YES ({unique_count} unique values)")
    else:
        print(f" High Cardinality: NO ({unique_count} unique values)")

    print("\n")

    # --- PLOTS ---
    num_categories = len(freq_pct)

    # Auto-adjust width based on number of categories
    plt.figure(figsize=( max(7, num_categories * 0.8), 5 ))

    # FREQUENCY PLOT (BAR)
    sns.barplot(x=freq_pct.index, y=freq_pct.values, palette="Set2")
    plt.title(f"Frequency (%) : {col}")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45, ha='right')
    for i, v in enumerate(freq_pct.values):
        plt.text(
            i, v + 0.5,                # position (x=index, y=value+offset)
            f"{v:.1f}%",               # format label
            ha='center', fontsize=9
        )

    plt.tight_layout()
    plt.show()


for col in true_num_cols:
    if col in ("id",target) :
        continue

    print("="*70)
    print(f"Numerical Feature vs Target: {col}")
    print("="*70)

    # --- Mean / Median Comparison ---
    means = train.groupby(target)[col].mean()
    medians = train.groupby(target)[col].median()

    print(" Mean by Target:")
    print(means, "\n")

    print(" Median by Target:")
    print(medians, "\n")

    # ---------------- PLOTS ----------------
    plt.figure(figsize=(14,5))

    # --- 1. BOX PLOT ---
    plt.subplot(1, 2, 1)
    sns.boxplot(x=train[target], y=train[col], palette="Set2")
    plt.title(f"Boxplot: {col} vs {target}")
    plt.xlabel("Target")
    plt.ylabel(col)

    # --- 2. KDE PLOT ---
    plt.subplot(1, 2, 2)
    for cls in sorted(train[target].unique()):
        sns.kdeplot(
            train[train[target] == cls][col],
            fill=True,
            label=f"{target} = {cls}",
            linewidth=1.5
        )

    plt.title(f"KDE: {col} by Target Class")
    plt.xlabel(col)
    plt.legend()

    plt.tight_layout()
    plt.show()


custom_palette = {0: "#FF6B6B", 1: "#4CAF50"}   # Red = default, Green = paid

for col in cat_cols:

    # Number of unique categories in this column
    num_categories = train[col].nunique()

    # Dynamic width: base width = 9, add 0.7 per category
    width = max(9, num_categories * 0.8)

    plt.figure(figsize=(width, 5))

    ax = sns.countplot(
        data=train,
        x=col,
        hue=target,
        palette=custom_palette,
        edgecolor='black',
        order=train[col].value_counts().index
    )
    
    plt.title(f'{col.replace("_", " ").title()} by Loan Repayment Status', fontsize=14)
    plt.xlabel(col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=25, ha='right')

    # -------- Add count labels above bars --------
    for bar in ax.patches:
        count = int(bar.get_height())
        ax.annotate(
            f'{count}',
            (bar.get_x() + bar.get_width() / 2, count),
            ha='center', va='bottom',
            fontsize=9
        )

    plt.legend(
        title='Loan Paid Back',
        labels=['No (0)', 'Yes (1)'],
        fontsize=10,
        title_fontsize=11
    )

    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()




"""

# Remove ID + target from numerical pairplot
for drop in ["id", target]:
    if drop in true_num_cols:
        true_num_cols.remove(drop)

# Build pairplot dataset
pairplot_df = train[true_num_cols + [target]]

# Pairplot
sns.pairplot(
    pairplot_df,
    hue=target,
    palette={0: "#FF6B6B", 1: "#4CAF50"},   # red/green
    diag_kind="kde",                        # KDE on diagonals
    corner=True,                            # show only lower triangle
    plot_kws={"alpha": 0.6, "s": 20},       # scatter styling
    diag_kws={"shade": True}                # KDE shading
)

plt.suptitle("Pairplot of Numerical Features", y=1.02, fontsize=14)
plt.show()

"""


for i in range(len(cat_cols)):
    for j in range(i+1, len(cat_cols)):

        col1 = cat_cols[i]
        col2 = cat_cols[j]

        print("="*80)
        print(f"Categorical vs Categorical: {col1}  Ã—  {col2}")
        print("="*80)

        # -------------------------------
        # CROSSTAB (counts)
        # -------------------------------
        ct = pd.crosstab(train[col1], train[col2])
        print("\n Crosstab (Counts):\n")
        print(ct)

        # -------------------------------
        # CROSSTAB (percentage row-wise)
        # -------------------------------
        ct_pct = pd.crosstab(train[col1], train[col2], normalize="index") * 100
        print("\n Crosstab (% by row):\n")
        print(ct_pct.round(2))

        # -------------------------------
        # HEATMAP (counts or %)
        # -------------------------------
        plt.figure(figsize=(max(8, ct.shape[1] * 1.2), max(6, ct.shape[0] * 0.7)))

        sns.heatmap(
            ct_pct,
            cmap="YlGnBu",
            annot=True,
            fmt=".1f",
            linewidths=0.5,
            cbar_kws={"label": "% Row Frequency"}
        )

        plt.title(f"Heatmap: {col1} vs {col2}", fontsize=14)
        plt.xlabel(col2, fontsize=12)
        plt.ylabel(col1, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.show()




print("\n===== FAST CORRELATION ANALYSIS =====\n")

# -------------------------------------------------------
# SAMPLE THE DATA (5k rows is enough for correlation)
# -------------------------------------------------------
sample_size = min(5000, len(train))
df_corr = train.sample(sample_size, random_state=42)

# -------------------------------------------------------
# SELECT NUMERICAL COLUMNS
# -------------------------------------------------------


# Remove ID + target
for drop in ["id", target]:
    if drop in true_num_cols:
        true_num_cols.remove(drop)

print(f"Using sample of {sample_size} rows for correlation.\n")

# -------------------------------------------------------
# PEARSON CORRELATION HEATMAP (with labels)
# -------------------------------------------------------
plt.figure(figsize=(12, 10))
corr_matrix = df_corr[true_num_cols].corr(method='pearson')

sns.heatmap(
    corr_matrix,
    cmap='coolwarm',
    annot=True,              # <<< LABELS ADDED HERE
    fmt=".2f",               # round to 2 decimals
    linewidths=0.4,
    square=False,
    cbar_kws={'shrink': 0.8}
)

plt.title("Pearson Correlation Heatmap (Linear Relationships)", fontsize=14)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# IDENTIFY HIGHLY CORRELATED PAIRS
# -------------------------------------------------------
threshold = 0.80
corr_pairs = (
    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
               .stack()
               .reset_index()
               .rename(columns={'level_0': 'Feature1', 'level_1': 'Feature2', 0: 'Correlation'})
)

high_corr = corr_pairs[abs(corr_pairs['Correlation']) > threshold]

print(" Highly Correlated Feature Pairs (|corr| > 0.80):\n")
print(high_corr if not high_corr.empty else "None found.\n")

# -------------------------------------------------------
# REDUNDANT FEATURE SUGGESTION (FAST)
# -------------------------------------------------------
redundant_features = set(high_corr["Feature2"].tolist())

print(" Suggested Redundant Features to Drop (if any):")
print(list(redundant_features) if redundant_features else "None")
print("\n")

# -------------------------------------------------------
# SPEARMAN CORRELATION (ONLY ON TOP 10 FEATURES) â€” FAST
# -------------------------------------------------------


top_corr_feats = (
    train[true_num_cols + [target]]
    .corr()[target]
    .abs()
    .sort_values(ascending=False)
    .head(10)
    .index
    .tolist()
)

if target in top_corr_feats:
    top_corr_feats.remove(target)

print(f"Using top correlated features for Spearman: {top_corr_feats}\n")

plt.figure(figsize=(10, 8))
spearman_matrix = df_corr[top_corr_feats].corr(method='spearman')

sns.heatmap(
    spearman_matrix,
    cmap='viridis',
    annot=True,              # <<< LABELS ADDED HERE
    fmt=".2f",               # rounded labels
    linewidths=0.4,
    square=False,
    cbar_kws={'shrink': 0.8}
)

plt.title("Spearman Rank Correlation Heatmap (Nonlinear Relationships)", fontsize=14)
plt.tight_layout()
plt.show()


# Remove id + target
for drop in ["id", target]:
    if drop in true_num_cols:
        true_num_cols.remove(drop)

outlier_summary = []

for col in true_num_cols:
    print("="*70)
    print(f"OUTLIER ANALYSIS FOR: {col}")
    print("="*70)

    # ------------------------------
    # Detect outliers using IQR
    # ------------------------------
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = train[(train[col] < lower) | (train[col] > upper)][col]
    outlier_count = outliers.count()
    outlier_pct = (outlier_count / len(train)) * 100

    print(f"Outlier Count: {outlier_count}")
    print(f"Outlier Percentage: {outlier_pct:.2f}%")
    print(f"IQR Lower Bound: {lower:.2f}")
    print(f"IQR Upper Bound: {upper:.2f}\n")

    # ------------------------------
    # Save summary for final table
    # ------------------------------
    outlier_summary.append({
        "Feature": col,
        "Outliers (%)": round(outlier_pct, 2),
        "Lower Bound": round(lower, 2),
        "Upper Bound": round(upper, 2),
    })

    # ------------------------------
    # Plot boxplot with boundaries
    # ------------------------------
    plt.figure(figsize=(7,4))
    sns.boxplot(x=train[col], color="#90EE90")
    plt.axvline(lower, color='red', linestyle='--', label="Lower Bound")
    plt.axvline(upper, color='orange', linestyle='--', label="Upper Bound")
    plt.title(f"Boxplot with Outlier Bounds: {col}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------
# Outlier Summary Table
# ------------------------------
outlier_df = pd.DataFrame(outlier_summary)
print("\n====== OUTLIER SUMMARY TABLE ======")
print(outlier_df)


print("\n===== MULTICOLLINEARITY CHECK =====\n")

# -------------------------------------------------------
# SAMPLE FOR SPEED
# -------------------------------------------------------
sample_size = min(5000, len(train))
df_corr = train.sample(sample_size, random_state=42)

# -------------------------------------------------------
# SELECT NUMERICAL COLUMNS
# -------------------------------------------------------


# Remove ID + target
for drop in ["id", target]:
    if drop in true_num_cols:
        true_num_cols.remove(drop)

# -------------------------------------------------------
# CORRELATION > 0.85 (REDUNDANT FEATURES)
# -------------------------------------------------------
corr_matrix = df_corr[true_num_cols].corr()

# Find highly correlated pairs
threshold = 0.85
corr_pairs = (
    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), 1).astype(bool))
               .stack()
               .reset_index()
               .rename(columns={'level_0': 'Feature1', 'level_1': 'Feature2', 0: 'Correlation'})
)

high_corr = corr_pairs[abs(corr_pairs['Correlation']) > threshold]

print(" Highly Correlated Pairs (|corr| > 0.85):\n")
print(high_corr if not high_corr.empty else "None found.")
print("\n")

# Proposed redundant features (drop second feature in each pair)
redundant_from_corr = list(high_corr["Feature2"].unique())

print(" Suggested Redundant Features (Correlation > 0.85):")
print(redundant_from_corr if redundant_from_corr else "None")
print("\n")

# -------------------------------------------------------
# VIF â€” VARIANCE INFLATION FACTOR
# -------------------------------------------------------
print(" VIF Analysis (Variance Inflation Factor):\n")

# Prepare dataframe for VIF
X = df_corr[true_num_cols].copy()
X = X.dropna()

vif_df = pd.DataFrame()
vif_df["Feature"] = X.columns
vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_df, "\n")

# -------------------------------------------------------
# REMOVE FEATURES WITH VIF > 10
# -------------------------------------------------------
high_vif = vif_df[vif_df["VIF"] > 10]["Feature"].tolist()

print(" Features With High VIF (> 10):")
print(high_vif if high_vif else "None")
print("\n")

