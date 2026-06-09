# Core Libraries
import os
import math
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
sns.set(style="whitegrid")

# Statistics and Diagnostics
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Preprocessing
from sklearn.preprocessing import LabelEncoder

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)

# Models
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Optimization
import optuna

# Explainability
import shap

# List dataset files
for dirname, _, filenames in os.walk('/kaggle/input/playground-series-s5e11'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train  = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


train.shape


train.info()


train.head()


train.describe()


train.isnull().sum()


train.duplicated().sum()


num_cols   = train.select_dtypes(include='number').columns.tolist()
cat_cols   = train.select_dtypes(include=['object', 'category']).columns.tolist()
bool_cols  = train.select_dtypes(include='bool').columns.tolist()
date_cols  = train.select_dtypes(include='datetime').columns.tolist()

# Optional: identify text-heavy columns
text_cols  = [col for col in cat_cols 
              if train[col].dtype == 'object' and train[col].str.len().mean() > 50]



print("Numerical Columns:", num_cols)
print("Categorical Columns:", cat_cols)
print("Boolean Columns:", bool_cols)
print("Datetime Columns:", date_cols)
print("Text Columns:", text_cols)



cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
#Threshold = the cutoff number of unique values after which a categorical column is considered high-cardinality.
threshold = 20

high_cardinality_cols = [col for col in cat_cols if train[col].nunique() > threshold]
low_cardinality_cols  = [col for col in cat_cols if train[col].nunique() <= threshold]

print("High Cardinality Categorical:", high_cardinality_cols)
print("Low Cardinality Categorical:", low_cardinality_cols)



# Target column : loan_paid_back


target = "loan_paid_back"   # change this to your target column

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

# Define color mapping
color_map = {
    '1.0': '#90EE90',
    '0.0': '#FF6961'
}

# --- Class Distribution ---
plt.figure(figsize=(6,4))
plt.bar(
    class_counts.index.astype(str),
    class_counts.values,
    linewidth=1.5,
    color=[color_map[i] for i in class_counts.index.astype(str)]
)
plt.title("Class Distribution", fontsize=14)
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# --- Class Percentages ---
plt.figure(figsize=(6,4))
plt.bar(
    class_percent.index.astype(str),
    class_percent.values,
    linewidth=1.5,
    color=[color_map[i] for i in class_percent.index.astype(str)]
)
plt.title("Class Percentages (%)", fontsize=14)
plt.xlabel("Class")
plt.ylabel("Percentage")
plt.show()



target = "loan_paid_back"

for col in num_cols:
    if col in ("id",target) :
        continue
        
    print("="*60)
    print(f"Feature: {col}")
    print("="*60)
    
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
    print(f"Min: {train[col].min()} | Max: {train[col].max()} | Mean: {train[col].mean()} | Medain: {train[col].median()}")
    
    # Plots
    plt.figure(figsize=(12,4))
    
    # Histogram + KDE (same plot)
    plt.subplot(1, 2, 1)
    sns.histplot(train[col], bins=30, kde=True, stat='density', color="#3498db")
    plt.title("Histogram + KDE")
    
    # Boxplot
    plt.subplot(1, 2, 2)

    sns.boxplot(
        x=train[col],
        color="#90EE90",
        linewidth=1.2,
        fliersize=3,
        saturation=0.9
    )

    plt.title(f"Boxplot: {col}", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    # Add median line
    median = train[col].median()
    plt.axvline(median, color='red', linestyle='--', linewidth=1)

    plt.text(median, 0.1, f"Median: {median:.2f}", color='red', fontsize=9, ha='left',rotation=45)


    
    plt.tight_layout()
    plt.show()



cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()

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




num_cols = train.select_dtypes(include='number').columns.tolist()

target = "loan_paid_back"

for col in num_cols:
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




# Remove ID + target from numerical pairplot
for drop in ["id", target]:
    if drop in num_cols:
        num_cols.remove(drop)

# Build pairplot dataset
pairplot_df = train[num_cols + [target]]

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



for i in range(len(cat_cols)):
    for j in range(i+1, len(cat_cols)):

        col1 = cat_cols[i]
        col2 = cat_cols[j]

        print("="*80)
        print(f"Categorical vs Categorical: {col1}  ×  {col2}")
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
num_cols = df_corr.select_dtypes(include='number').columns.tolist()

# Remove ID + target
for drop in ["id", "loan_paid_back"]:
    if drop in num_cols:
        num_cols.remove(drop)

print(f"Using sample of {sample_size} rows for correlation.\n")

# -------------------------------------------------------
# PEARSON CORRELATION HEATMAP (with labels)
# -------------------------------------------------------
plt.figure(figsize=(12, 10))
corr_matrix = df_corr[num_cols].corr(method='pearson')

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
# SPEARMAN CORRELATION (ONLY ON TOP 10 FEATURES) — FAST
# -------------------------------------------------------
target = 'loan_paid_back'

top_corr_feats = (
    train[num_cols + [target]]
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



cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
target = "loan_paid_back"

for col in cat_cols:
    for num in num_cols:
        plt.figure(figsize=(7,4))
        sns.boxplot(data=train, x=col, y=num)
        plt.title(f"{num} distribution across {col}")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()

        # Target mean vs numerical value grouped by category
        print(f"\nTarget mean of {target} by {col} for numerical variable {num}:")
        print(train.groupby(col)[num].mean())
        print("\n")



num_cols = train.select_dtypes(include="number").columns.tolist()

# Remove id + target
for drop in ["id", "loan_paid_back"]:
    if drop in num_cols:
        num_cols.remove(drop)

outlier_summary = []

for col in num_cols:
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
import pandas as pd
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
num_cols = df_corr.select_dtypes(include="number").columns.tolist()

# Remove ID + target
for drop in ["id", "loan_paid_back"]:
    if drop in num_cols:
        num_cols.remove(drop)

# -------------------------------------------------------
# CORRELATION > 0.85 (REDUNDANT FEATURES)
# -------------------------------------------------------
corr_matrix = df_corr[num_cols].corr()

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
# VIF — VARIANCE INFLATION FACTOR
# -------------------------------------------------------
print(" VIF Analysis (Variance Inflation Factor):\n")

# Prepare dataframe for VIF
X = df_corr[num_cols].copy()
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


train["is_train"] = 1
test["is_train"] = 0
test["loan_paid_back"] = np.nan
combined = pd.concat([train, test], ignore_index=True)
print("train_shape ",train.shape)
print("test_shape ",test.shape)
print("combined_shape ",combined.shape)
combined.head()


# Identify columns
num_cols = combined.select_dtypes(include='number').columns.tolist()
cat_cols = combined.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove ID column (never impute)
if 'id' in num_cols:
    num_cols.remove('id')

# Remove target from numerical imputations
if 'loan_paid_back' in num_cols:
    num_cols.remove('loan_paid_back')

# -------------------------------------------------------
# Handle Rare Categories (1% threshold)
# -------------------------------------------------------
rare_threshold = 0.01  # 1%

for col in cat_cols:
    freq = combined[col].value_counts(normalize=True)
    rare_cats = freq[freq < rare_threshold].index.tolist()

    if rare_cats:
        combined[col] = combined[col].replace(rare_cats, "Other")
        print(f"Replaced rare categories in {col}: {rare_cats}")

# -------------------------------------------------------
# Numerical → median (combined median)
# -------------------------------------------------------
for col in num_cols:
    median_val = combined[col].median()
    combined[col].fillna(median_val, inplace=True)

# -------------------------------------------------------
# Categorical → mode (combined mode)
# -------------------------------------------------------
for col in cat_cols:
    mode_val = combined[col].mode()[0]
    combined[col].fillna(mode_val, inplace=True)

# -------------------------------------------------------
# Check remaining missing values
# -------------------------------------------------------
print("Missing values after unified imputation:\n")
print(combined.isnull().sum())



combined.tail()


# 1. annual_income
# - Negative values are invalid and should be replaced with the median.
# - Valid range is > 0.
# - Extremely high values may exist but should be capped using IQR.
# - Optionally create an outlier flag for values exceeding the IQR upper bound.

# 2. loan_amount
# - Negative loan amounts are invalid and should be replaced with the median.
# - Valid range is > 0.
# - Extremely high values are possible but should be capped using IQR.
# - Optionally create an outlier flag for values exceeding the IQR upper bound.

# 3. debt_to_income_ratio
# - Negative values are invalid and should be set to 0.
# - The typical range is 0 to 1, but values up to 2 may occur.
# - Values above 2 are generally unrealistic and should be clipped.
# - Values greater than 1 can be flagged as high-risk.

# 4. credit_score
# - The valid range is 300 to 900.
# - Values below 300 or above 900 are invalid and should be clipped.
# - Create an outlier flag for values originally outside the valid range.

# 5. interest_rate
# - Negative values are invalid and should be set to 0.
# - Realistic domain range is 0 to 50.
# - Values above 50 should be clipped.
# - A flag can be created for values originally outside the valid range.
# - The feature is often skewed, so log-transforming may be beneficial.




# Identify numerical columns for combined
num_cols = combined.select_dtypes(include='number').columns.tolist()

# Remove ID and target
for drop in ["id", "loan_paid_back"]:
    if drop in num_cols:
        num_cols.remove(drop)

# ============================================================
# annual_income Processing (Rule-based)
# ============================================================

median_income = combined["annual_income"].median()

# Flag negative values
combined["annual_income_invalid_flag"] = np.where(combined["annual_income"] < 0, 1, 0)
combined.loc[combined["annual_income"] < 0, "annual_income"] = median_income

# IQR capping
Q1 = combined["annual_income"].quantile(0.25)
Q3 = combined["annual_income"].quantile(0.75)
IQR = Q3 - Q1
upper_cap = Q3 + 1.5 * IQR

combined["annual_income_outlier_flag"] = np.where(combined["annual_income"] > upper_cap, 1, 0)
combined["annual_income"] = np.clip(combined["annual_income"], 0, upper_cap)


# ============================================================
# loan_amount Processing (Rule-based)
# ============================================================

median_loan = combined["loan_amount"].median()

# Flag negative amounts
combined["loan_amount_invalid_flag"] = np.where(combined["loan_amount"] < 0, 1, 0)
combined.loc[combined["loan_amount"] < 0, "loan_amount"] = median_loan

# IQR capping
Q1 = combined["loan_amount"].quantile(0.25)
Q3 = combined["loan_amount"].quantile(0.75)
IQR = Q3 - Q1
upper_cap = Q3 + 1.5 * IQR

combined["loan_amount_outlier_flag"] = np.where(combined["loan_amount"] > upper_cap, 1, 0)
combined["loan_amount"] = np.clip(combined["loan_amount"], 0, upper_cap)


# ============================================================
# debt_to_income_ratio Processing (Rule-based)
# ============================================================

# Flag negatives
combined["dti_invalid_flag"] = np.where(combined["debt_to_income_ratio"] < 0, 1, 0)

# Clip values to [0, 2]
combined["debt_to_income_ratio"] = combined["debt_to_income_ratio"].clip(0, 2)

# Flag high-risk DTI (>1)
combined["dti_high_flag"] = np.where(combined["debt_to_income_ratio"] > 1, 1, 0)


# ============================================================
# credit_score Processing (Rule-based)
# ============================================================

# Flag outside valid domain
combined["credit_score_invalid_flag"] = np.where(
    (combined["credit_score"] < 300) | (combined["credit_score"] > 900), 1, 0
)

# Clip to valid range
combined["credit_score"] = combined["credit_score"].clip(300, 900)


# ============================================================
# interest_rate Processing (Rule-based)
# ============================================================

# Flag invalid interest rates
combined["interest_rate_invalid_flag"] = np.where(
    (combined["interest_rate"] < 0) | (combined["interest_rate"] > 50), 1, 0
)

# Clip to valid range
combined["interest_rate"] = combined["interest_rate"].clip(0, 50)



combined.columns


# =============================================================
# References:
# [R1] Löeffler & Posch – Credit Risk Modeling, Cambridge Univ Press
# [R2] Anderson – Credit Scoring & Its Applications
# [R3] Basel II/III Guidelines – Loan-to-Income, PD Models
# [R4] Federal Reserve – Consumer Credit Affordability Research
# [R5] Moody’s Analytics – Borrower Stress Indicator Research
# [R6] FICO – Score Reason Codes & Normalization Guidelines
# [R7] IMF – Loan Pricing & Default Modeling Framework
# [R8] Experian & TransUnion – Utilization × Score Risk Research
# =============================================================


# -----------------------------
# Affordability & Capacity
# -----------------------------
combined["income_loan_ratio"] = combined["annual_income"] / (combined["loan_amount"] + 1)      # R3
combined["loan_to_income_ratio"] = combined["loan_amount"] / (combined["annual_income"] + 1)    # R3
combined["affordability_index"] = (
    combined["annual_income"] /
    (combined["loan_amount"] * (1 + combined["interest_rate"]/100))
)                                                                                               # R7

# -----------------------------
# Pricing & Spread Indicators
# -----------------------------
combined["risk_margin"] = combined["interest_rate"] - (combined["credit_score"] / 100)          # R2
combined["interest_income_ratio"] = combined["interest_rate"] / (combined["annual_income"] + 1) # R4

# -----------------------------
# Stress & Leverage Indicators
# -----------------------------
combined["loan_interest_product"] = combined["loan_amount"] * combined["interest_rate"]          # R7
combined["dti_income_product"] = combined["debt_to_income_ratio"] * combined["annual_income"]    # R1
combined["stress_score"] = (
    combined["loan_amount"] *
    combined["interest_rate"] *
    combined["debt_to_income_ratio"]
)                                                                                               # R5

# -----------------------------
# Normalized & Transformed Features
# -----------------------------
combined["credit_score_norm"] = (combined["credit_score"] - 300) / 600                           # R6
combined["interest_rate_log"] = np.log1p(combined["interest_rate"])                              # R2

# -----------------------------
# Interaction Features
# -----------------------------
combined["score_interest_interaction"] = (
    combined["credit_score"] * combined["interest_rate"]
)                                                                                               # R7

combined["income_dti_interaction"] = (
    combined["annual_income"] * combined["debt_to_income_ratio"]
)                                                                                               # R8

combined["loan_dti_interaction"] = (
    combined["loan_amount"] * combined["debt_to_income_ratio"]
)                                                                                               # R1

# -----------------------------
# Utilization / Score Interaction
# -----------------------------
combined["credit_utilization_factor"] = (
    combined["debt_to_income_ratio"] * (900 - combined["credit_score"])
)                                                                                               # R8



combined.columns


# Identify categorical columns in combined
cat_cols = combined.select_dtypes(include=['object', 'category']).columns.tolist()

print("Categorical Columns:", cat_cols)


# Label encoding for tree-based models
for col in cat_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))

combined[cat_cols].head()



# Split combined back into train and test
final_train = combined[combined["is_train"] == 1].copy()
final_test  = combined[combined["is_train"] == 0].copy()

print("Train Shape:", final_train.shape)
print("Test Shape:", final_test.shape)



final_train.columns


# Columns to drop from model input
drop_cols = ["id", "is_train", "loan_paid_back"]

# Create feature matrix (X) by dropping unwanted columns
X = final_train.drop(columns=drop_cols)

# Target vector (y)
y = final_train["loan_paid_back"]

print("X shape:", X.shape)
print("y shape:", y.shape)



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)



X_test = final_test.drop(columns=["id", "is_train", "loan_paid_back"])

print("Test shape:", X_test.shape)



def evaluate_model(model, X_train, y_train, X_val, y_val, model_name="Model"):
    print(f"\n===== Evaluation for {model_name} =====")

    # -----------------------------
    # Predictions
    # -----------------------------
    val_pred_prob = model.predict_proba(X_val)[:, 1]
    val_pred = (val_pred_prob >= 0.5).astype(int)

    # -----------------------------
    # Metrics
    # -----------------------------
    accuracy = accuracy_score(y_val, val_pred)
    precision = precision_score(y_val, val_pred)
    recall = recall_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred)
    auc = roc_auc_score(y_val, val_pred_prob)

    # KS statistic
    fpr, tpr, _ = roc_curve(y_val, val_pred_prob)
    ks = max(tpr - fpr)

    # Gini coefficient
    gini = 2 * auc - 1

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"KS Statistic: {ks:.4f}")
    print(f"Gini Coefficient: {gini:.4f}")

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(y_val, val_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # -----------------------------
    # ROC Curve
    # -----------------------------
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.title(f"{model_name} - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -----------------------------
    # Precision-Recall Curve
    # -----------------------------
    precision_curve, recall_curve, _ = precision_recall_curve(y_val, val_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall_curve, precision_curve)
    plt.title(f"{model_name} - Precision Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.show()

    # -----------------------------
    # SHAP Explainability
    # -----------------------------
    print(f"\nGenerating SHAP values for {model_name}...")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)

        # Summary plot
        shap.summary_plot(shap_values, X_val, plot_type="dot", show=True)
    except Exception as e:
        print("\nSHAP could not be generated for this model:")
        print(e)

    # Return metrics for comparison table
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc,
        "KS": ks,
        "Gini": gini
    }







baseline_models = {
    "LightGBM": LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),
    
    "XGBoost": XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='auc',
        tree_method="hist",
        random_state=42,
        use_label_encoder=False
    ),
    
    "CatBoost": CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        verbose=False,
        random_seed=42,
        task_type="GPU" if False else "CPU"   # set GPU=True if available
    )
}



results = []

for model_name, model in baseline_models.items():
    print(f"\n===== Training {model_name} Baseline Model =====")

    model.fit(X_train, y_train)

    metrics = evaluate_model(
        model,
        X_train, y_train,
        X_val, y_val,
        model_name=model_name
    )
    
    results.append(metrics)



results_df = pd.DataFrame(results)
results_df.sort_values("AUC", ascending=False)





def objective_lgbm(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 0, 5),
        "lambda_l2": trial.suggest_float("lambda_l2", 0, 5),
        "verbosity": -1,
        "seed": 42
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data   = lgb.Dataset(X_val,   label=y_val)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=0)      # replaces verbose_eval
        ]
    )

    preds = model.predict(X_val)
    return roc_auc_score(y_val, preds)


study_lgb = optuna.create_study(direction="maximize")
study_lgb.optimize(objective_lgbm, n_trials=30)

best_lgb_params = study_lgb.best_params
best_lgb_params["objective"] = "binary"
best_lgb_params["metric"]     = "auc"
best_lgb_params["boosting_type"] = "gbdt"
best_lgb_params["verbosity"] = -1
best_lgb_params["seed"] = 42

print("Best LightGBM Params:", best_lgb_params)





def objective_xgb(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "tree_method": "hist",          # fast, CPU friendly
        "eval_metric": "auc",
        "random_state": 42,
        "use_label_encoder": False
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)

    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)


study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=30)

best_xgb_params = study_xgb.best_params
best_xgb_params["tree_method"] = "hist"
best_xgb_params["eval_metric"] = "auc"
best_xgb_params["random_state"] = 42
best_xgb_params["use_label_encoder"] = False

print("Best XGBoost Params:", best_xgb_params)





def objective_cat(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "iterations": 1200,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": False,
        "random_seed": 42,
        "task_type": "CPU"   # change to "GPU" if available
    }

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)


study_cat = optuna.create_study(direction="maximize")
study_cat.optimize(objective_cat, n_trials=30)

best_cat_params = study_cat.best_params
best_cat_params.update({
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "verbose": False,
    "random_seed": 42,
    "task_type": "CPU"
})

print("Best CatBoost Params:", best_cat_params)



# Final LightGBM params
lgb_params_final = {
    **best_lgb_params,
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 3000,
    "random_state": 42
}

best_lgb_model = lgb.LGBMClassifier(**lgb_params_final)
best_lgb_model.fit(X_train, y_train)



xgb_params_final = {
    **best_xgb_params,
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": 42,
    "use_label_encoder": False
}

best_xgb_model = XGBClassifier(**xgb_params_final)
best_xgb_model.fit(X_train, y_train)



cat_params_final = {
    **best_cat_params,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "verbose": False,
    "random_seed": 42,
    "task_type": "CPU"     # change to GPU if available
}

best_cat_model = CatBoostClassifier(**cat_params_final)
best_cat_model.fit(X_train, y_train)





all_results = []

# LightGBM
print("\n================ LIGHTGBM (Tuned) ================")
results_lgb = evaluate_model(
    best_lgb_model,
    X_train, y_train,
    X_val, y_val,
    model_name="LightGBM (Tuned)"
)
all_results.append(results_lgb)


# XGBoost
print("\n================ XGBOOST (Tuned) ================")
results_xgb = evaluate_model(
    best_xgb_model,
    X_train, y_train,
    X_val, y_val,
    model_name="XGBoost (Tuned)"
)
all_results.append(results_xgb)


# CatBoost
print("\n================ CATBOOST (Tuned) ================")
results_cat = evaluate_model(
    best_cat_model,
    X_train, y_train,
    X_val, y_val,
    model_name="CatBoost (Tuned)"
)
all_results.append(results_cat)


# Convert to dataframe for comparison
results_table = pd.DataFrame(all_results)
results_table = results_table.sort_values("AUC", ascending=False)

print("\n============== MODEL COMPARISON TABLE ==============\n")
print(results_table)



# Sort by AUC first, then KS, then Gini, then F1
results_table_sorted = (
    results_table
    .sort_values(["AUC", "KS", "Gini", "F1"], ascending=False)
    .reset_index(drop=True)
)

print("\n================ BEST MODEL (Auto-Selected) ================\n")
print(results_table_sorted.iloc[0])

best_model_name = results_table_sorted.iloc[0]["Model"]
print(f"\nBest model selected: {best_model_name}")



model_map = {
    "LightGBM (Tuned)": best_lgb_model,
    "XGBoost (Tuned)": best_xgb_model,
    "CatBoost (Tuned)": best_cat_model
}

best_model = model_map[best_model_name]



print(f"Retraining best model on full data: {best_model_name}")

if best_model_name == "LightGBM (Tuned)":
    final_model = lgb.LGBMClassifier(**lgb_params_final)
    final_model.fit(X, y)

elif best_model_name == "XGBoost (Tuned)":
    final_model = XGBClassifier(**xgb_params_final)
    final_model.fit(X, y)

elif best_model_name == "CatBoost (Tuned)":
    final_model = CatBoostClassifier(**cat_params_final)
    final_model.fit(X, y)

else:
    raise ValueError(f"Unknown best_model_name: {best_model_name}")





# Predict probabilities for the positive class (loan_paid_back = 1)
test_pred_prob = final_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "id": final_test["id"],
    "loan_paid_back": test_pred_prob
})




# Ensure only required columns
submission = submission[["id", "loan_paid_back"]]

# Round to 6 decimal places
submission["loan_paid_back"] = submission["loan_paid_back"].round(4)







submission["loan_paid_back"] = submission["loan_paid_back"].astype(float)



# Now save normally (index=False)
submission.to_csv("submission.csv", index=False)


# Check file size

print("File size (MB):", os.path.getsize("submission.csv") / (1024*1024))


submission.shape




os.listdir("/kaggle/working")





