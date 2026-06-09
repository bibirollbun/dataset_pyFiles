# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for seaborn 
sns.set_style("whitegrid")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# setup to avoid warnings, probably not recommended but keeps it simple 
import warnings

# Suppress pandas/seaborn deprecated option warning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*use_inf_as_na.*"
)

# Suppress seaborn categorical observed=False deprecation warning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*default of observed=False is deprecated.*"
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="When grouping with a length-1 list-like.*get_group.*"    
)


# Data directory: 
DATA_DIR = "/kaggle/input/playground-series-s5e12"

# import csv files for train and test 
train_path = os.path.join(DATA_DIR, "train.csv")
test_path  = os.path.join(DATA_DIR, "test.csv")

# dataframes for train and test 
train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

train_eda = train
test_eda  = test

ID_COL = "id"
TARGET_COL = "diagnosed_diabetes"

# -------------------------------
# 1) Column groups 
# -------------------------------
int_cols = [
    "age",
    "alcohol_consumption_per_week",
    "physical_activity_minutes_per_week",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "cholesterol_total",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
]

float_cols = [
    "diet_score",
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "bmi",
    "waist_to_hip_ratio",
]

cat_cols = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status",
]

bool_cols = [
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history",
]


# -------------------------------
# 3) Dataset overview
# -------------------------------
print("\n=== OVERVIEW ===")
print("Train shape:", train_eda.shape)
print("Test shape :", test_eda.shape)

# display(train_eda.head())
# display(train_eda.describe(include=[np.number]).T)

# print("\nDtypes:")
# display(train_eda.dtypes)

# -------------------------------
# 4) Missing value analysis
# -------------------------------
print("\n=== MISSING VALUES ===")
missing = (
    train_eda.isna()
    .sum()
    .to_frame("missing_count")
    .assign(missing_pct=lambda x: x["missing_count"] / len(train_eda))
    .sort_values("missing_pct", ascending=False)
)
display(missing[missing["missing_count"] > 0])

plt.figure(figsize=(10, 4))
sns.barplot(
    data=missing.reset_index(),
    x="index",
    y="missing_pct"
)
plt.xticks(rotation=90)
plt.title("Missing Value Percentage by Feature")
plt.ylabel("Percent Missing")
plt.xlabel("Feature")
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 5) Target distribution 
# -------------------------------
counts = (
    train_eda[TARGET_COL].astype(int)
    .value_counts(dropna=False)
    .reindex([0, 1], fill_value=0)  # explicit labels
)

positive_rate = train_eda[TARGET_COL].mean()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# ---- Left: text summary ----
axes[0].axis("off")
summary_text = (
    "Target distribution\n\n"    
    f"0 (No Diabetes): {counts.loc[0]:,}\n"
    f"1 (Diabetes): {counts.loc[1]:,}\n\n"
    f"Positive rate: {positive_rate:.3f}"
)

axes[0].text(
    0.05, 0.5, summary_text,
    fontsize=20,
    va="center"
)

# ---- Right: bar chart ----
sns.countplot(
    data=train_eda,
    x=TARGET_COL,
    ax=axes[1]
)
axes[1].set_title("Diagnosed Diabetes Distribution")
axes[1].set_xlabel("Diagnosed Diabetes")
axes[1].set_ylabel("Count")

plt.tight_layout()
# plt.savefig("target_distr_plot.png", dpi=300, bbox_inches="tight")
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

feature_list = ['age', 'bmi', 'physical_activity_minutes_per_week']

for x in feature_list:     
    # --- Configure these to match your dataset ---
    FEATURE_COL = x
    TARGET_COL = TARGET_COL          # change to "diagnosed_diabetes" if that's your column
    TARGET_LABELS = {0: "Non-Diabetic", 1: "Diabetic"}  # optional relabel
    
    # Optional: create a labeled target column for prettier legends
    plot_df = train_eda.copy()
    plot_df["Outcome_Label"] = plot_df[TARGET_COL].map(TARGET_LABELS).fillna(plot_df[TARGET_COL].astype(str))
    
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)

    if x == 'age':
        title = "Age"
    elif x == 'bmi':
        title = "BMI"
    elif x == 'physical_activity_minutes_per_week':
        title = 'Physical Activity'
    
    # ------------------------------------------------------------
    # (1) distribution: Histogram + KDE
    # ------------------------------------------------------------
    sns.histplot(
        data=plot_df,
        x=FEATURE_COL,
        bins=30,
        kde=True,
        ax=axes[0]
    )
    
    axes[0].set_title(f"{title} Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel(f"{title}")
    axes[0].set_ylabel("Count")
    
    # ------------------------------------------------------------
    # (2) Age vs Outcome: Boxplot
    # ------------------------------------------------------------
    sns.boxplot(
        data=plot_df,
        x=TARGET_COL,
        y=FEATURE_COL,
        ax=axes[1]
    )
    axes[1].set_title(f"{title} vs Diabetes Outcome (Boxplot)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Diabetes Outcome")
    axes[1].set_ylabel(f"{title}")
    
    # If you want the x-axis labeled as Non-Diabetic / Diabetic:
    axes[1].set_xticklabels([TARGET_LABELS.get(t.get_text(), t.get_text()) for t in axes[1].get_xticklabels()])
    
    # ------------------------------------------------------------
    # (3) Age distribution by Outcome: KDE overlap
    # ------------------------------------------------------------
    sns.kdeplot(
        data=plot_df,
        x=FEATURE_COL,
        hue="Outcome_Label",
        fill=True,
        common_norm=False,
        alpha=0.45,
        linewidth=2,
        ax=axes[2]
    )
    axes[2].set_title(f"{title} Distribution by Outcome (KDE)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel(f"{title}")
    axes[2].set_ylabel("Density")
    axes[2].legend(title="Diabetes Outcome",
        labels=["Diabetic", "Non-Diabetic"]
        )
    
    plt.suptitle(f"{title} — Overall Distribution and Outcome Differences", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()

    plt.savefig(f"{title}_overall", dpi=300, bbox_inches="tight")

    plt.show()



# -------------------------------
# 6) Numeric feature distributions + relationship to target
# -------------------------------
print("\n=================================== NUMERIC FEATURES ================================ \n\n")
for col in int_cols + float_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(
        data=train_eda,
        x=train_eda[col].dropna(),
        bins=30,
        kde=True,
        ax=axes[0]
    )
    axes[0].set_title(f"{col} Distribution")

    sns.boxplot(
        data=train_eda,
        x=TARGET_COL,
        y=col,
        ax=axes[1]
    )
    axes[1].set_title(f"{col} vs {TARGET_COL}")

    plt.tight_layout()
    
    plt.show()


# -------------------------------
# 7) Outlier assessment (IQR fraction)
# -------------------------------
print("\n=== OUTLIER FRACTION (IQR) ===\n\n")
outliers = []
for col in int_cols + float_cols:
    s = train_eda[col].dropna()
    if s.empty:
        outliers.append((col, np.nan))
        continue

    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1

    if IQR == 0:
        outliers.append((col, 0.0))
        continue

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    frac = ((train_eda[col] < lower) | (train_eda[col] > upper)).mean()
    outliers.append((col, float(frac)))

outlier_df = (
    pd.DataFrame(outliers, columns=["feature", "outlier_fraction"])
    .sort_values("outlier_fraction", ascending=False)
)
display(outlier_df)


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Normalize target once for safety
target_int = train_eda[TARGET_COL].astype(int)

print("\n=== CATEGORICAL FEATURES ===")

for col in cat_cols:
    # Order categories by frequency
    order = train_eda[col].value_counts().index

    # Compute target rate by category (label-safe)
    rate = (
        train_eda.assign(target_int=target_int)
        .groupby(col, observed=True)["target_int"]
        .mean()
        .reindex(order)
        .to_frame("diabetes_rate")
    )

    # -------------------------------
    # Side-by-side layout
    # -------------------------------
    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, max(4, 0.35 * len(order))),
        gridspec_kw={"width_ratios": [2, 1]}
    )

    # ---- Left: distribution ----
    sns.countplot(
        data=train_eda,
        y=col,
        order=order,
        ax=axes[0]
    )
    axes[0].set_title(f"{col} Distribution")
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel(col)

    # ---- Right: target rate table ----
    axes[1].axis("off")
    table = axes[1].table(
        cellText=rate.round(3).values,
        rowLabels=rate.index.astype(str),
        colLabels=["Diabetes Rate"],
        loc="center",
        cellLoc="center",
        colLoc="center"
    )
    table.scale(1, 1.4)
    axes[1].set_title("Diabetes Rate by Category")

    plt.tight_layout()
    plt.show()



# -------------------------------
# 9) Boolean feature prevalence + lift (pandas-based)
# -------------------------------
print("\n=== BOOLEAN FEATURES ===")

# Prevalence (True rate)
prevalence = train_eda[bool_cols].mean().sort_values(ascending=False).to_frame("prevalence")
display(prevalence)

# Lift vs target
lift = (
    train_eda.groupby(TARGET_COL)[bool_cols]
    .mean()
    .T
)
tmp = train_eda.copy()
tmp[TARGET_COL] = tmp[TARGET_COL].astype("int8")

lift = tmp.groupby(TARGET_COL)[bool_cols].mean().T

# Ensure both classes exist
for cls in [0, 1]:
    if cls not in lift.columns:
        lift[cls] = np.nan

lift["lift"] = lift[1] - lift[0]
display(lift.sort_values("lift", ascending=False))

# Optional: plot prevalence
plt.figure(figsize=(8, 3))
sns.barplot(
    data=prevalence.reset_index(),
    x="index",
    y="prevalence"
)
plt.title("Boolean Feature Prevalence (True Rate)")
plt.xlabel("Feature")
plt.ylabel("Prevalence")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Configuration ---
FEATURE_COL = "family_history_diabetes"
TARGET_COL = "diagnosed_diabetes"

LABEL_MAP = {0: "No Family History", 1: "Family History"}

sns.set_theme(style="whitegrid")

# --- Compute prevalence ---
prevalence = (
    train_eda.groupby(FEATURE_COL)[TARGET_COL]
      .mean()
      .reset_index()
)

prevalence[FEATURE_COL] = prevalence[FEATURE_COL].map(LABEL_MAP)

# --- Compute uplift (relative to baseline: No Family History) ---
baseline = prevalence.loc[
    prevalence[FEATURE_COL] == "No Family History", TARGET_COL
].values[0]

prevalence["uplift"] = prevalence[TARGET_COL] / baseline

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120)

# -----------------------------
# (1) Prevalence plot
# -----------------------------
sns.barplot(
    data=prevalence,
    x=FEATURE_COL,
    y=TARGET_COL,
    palette=["#1f77b4", "#d62728"],
    ax=axes[0]
)

axes[0].set_title("Diabetes Prevalence by Family History", fontweight="bold")
axes[0].set_xlabel("")
axes[0].set_ylabel("Prevalence")

for i, v in enumerate(prevalence[TARGET_COL]):
    axes[0].text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=11)

# -----------------------------
# (2) Uplift plot
# -----------------------------
sns.barplot(
    data=prevalence,
    x=FEATURE_COL,
    y="uplift",
    palette=["#1f77b4", "#d62728"],
    ax=axes[1]
)

axes[1].axhline(1.0, linestyle="--", color="gray", linewidth=1)
axes[1].set_title("Uplift Relative to No Family History", fontweight="bold")
axes[1].set_xlabel("")
axes[1].set_ylabel("Relative Risk")

for i, v in enumerate(prevalence["uplift"]):
    axes[1].text(i, v + 0.03, f"{v:.2f}×", ha="center", fontsize=11)

plt.suptitle(
    "Family History of Diabetes: Prevalence and Uplift",
    fontsize=14,
    fontweight="bold",
    y=1.05
)

plt.tight_layout()

plt.savefig(f"familiy_history_Prev_Uplift", dpi=300, bbox_inches="tight")
plt.show()




# -------------------------------
# 10) Correlation heatmap 
# -------------------------------
print("\n=================================== CORRELATION ANALYSIS ===================================\n\n")

corr_cols = int_cols + float_cols + bool_cols

corr = (
    train_eda[corr_cols + [TARGET_COL]]
    .corr(numeric_only=True)
    .dropna(axis=0, how="all")
    .dropna(axis=1, how="all")
)

# ---- Mask upper triangle (keep lower + diagonal) ----
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

# ---- Centered color normalization ----
import matplotlib.colors as mcolors
vmin = np.nanmin(corr.values)
vmax = np.nanmax(corr.values)

# Safety fallback (rare but avoids runtime warnings)
if not (vmin < 0 < vmax):
    vmin, vmax = -1.0, 1.0

norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# ---- Target correlation table ----
target_corr = (
    corr[TARGET_COL]
    .drop(TARGET_COL, errors="ignore")
    .sort_values(key=lambda s: s.abs(), ascending=False)
    .to_frame("corr_with_target")
)

# -------------------------------
# Side-by-side layout
# -------------------------------
fig, axes = plt.subplots(
    1, 2,
    figsize=(18, 12),
    gridspec_kw={"width_ratios": [3, 1]}
)

# ---- Left: heatmap ----
sns.heatmap(
    corr,
    mask=mask,
    cmap="coolwarm",
    norm=norm,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    ax=axes[0]
)
axes[0].set_title("Feature Correlation Matrix (Lower Triangle)")

# ---- Right: target correlation table ----
axes[1].axis("off")

table = axes[1].table(
    cellText=target_corr.round(3).values,
    rowLabels=target_corr.index.astype(str),
    colLabels=["Correlation with Target"],
    loc="center",
    cellLoc="center",
    colLoc="center"
)

# Increase text size
table.auto_set_font_size(False)
table.set_fontsize(14)   # try 11–14 depending on figure size
table.scale(1.2, 1.7)    # widen slightly, increase row height
# axes[1].set_title("Correlation with Target: \n(sorted by magnitude)")

plt.tight_layout()
plt.show()




# corr2 = corr.corr(numeric_only=True)[TARGET_COL].sort_values()
# corr2.plot(kind="barh", figsize=(6,4))

# Compute correlations with target
corr = (
   corr.corr(numeric_only=True)[TARGET_COL]
    .drop(TARGET_COL)
    .sort_values()
)

# Plot
plt.figure(figsize=(7, 5))
bars = plt.barh(
    corr.index,
    corr.values,
    color=["#d62728" if v > 0 else "#1f77b4" for v in corr.values]
)

# Reference line at zero
plt.axvline(0, color="gray", linewidth=1)

# Labels & title
plt.title("Feature Correlation with Diabetes Outcome", fontsize=12, weight="bold")
plt.xlabel("Pearson Correlation Coefficient")
plt.ylabel("Feature")

# Annotate values
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + (0.01 if width > 0 else -0.01),
        bar.get_y() + bar.get_height() / 2,
        f"{width:.2f}",
        va="center",
        ha="left" if width > 0 else "right",
        fontsize=9
    )

# Improve layout
plt.tight_layout()
plt.savefig("Feature_Correlation_plot.png", dpi=300, bbox_inches="tight")
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))

# downsample for speed
plot_df = train_eda.sample(n=50_000, random_state=42)

sns.scatterplot(
    data=plot_df,
    x="bmi",
    y="physical_activity_minutes_per_week",
    hue=TARGET_COL,
    palette={0: "#1f77b4", 1: "#d62728"},
    alpha=0.4,
    s=40
)

plt.title(
    "BMI vs Physical Activity by Diabetes Outcome",
    fontsize=16,
    fontweight="bold",
    pad=12
)
plt.xlabel("Body Mass Index (BMI)", fontsize=13)
plt.ylabel("Physical Activity (Minutes per Week)", fontsize=13)

plt.legend(
    title="Diabetes Outcome",
    labels=["Diabetic","Non-Diabetic"],
    fontsize=11,
    title_fontsize=11
)

plt.tight_layout()
plt.savefig("scatter_by_outcome.png", dpi=300, bbox_inches="tight")
plt.show()



import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# 3.7 Feature relationships (pairwise) - seaborn PairGrid
# -------------------------------
top_features = target_corr.head(4).index.tolist()

df_pair = train_eda[top_features + [TARGET_COL]].copy()
df_pair[TARGET_COL] = df_pair[TARGET_COL].astype(int)

# downsample for speed
df_pair = df_pair.sample(n=min(20000, len(df_pair)), random_state=42)

# Build PairGrid (seaborn alternative to pairplot)
g = sns.PairGrid(
    df_pair,
    vars=top_features,
    hue=TARGET_COL,
    corner=True,
    diag_sharey=False
)

# Off-diagonal scatter
g.map_lower(sns.scatterplot, alpha=0.35, s=12, linewidth=0)

# Diagonal distributions
g.map_diag(sns.histplot, bins=30, kde=False)

# Legend
g.add_legend(title=TARGET_COL)

# ---- Fix axes for binary features (0/1 only) ----
for i, yvar in enumerate(g.y_vars):
    for j, xvar in enumerate(g.x_vars):
        ax = g.axes[i, j]
        if ax is None:
            continue

        # Fix y-axis if y variable is binary
        if df_pair[yvar].nunique() == 2:
            ax.set_yticks([0, 1])
            ax.set_ylim(-0.05, 1.05)

        # Fix x-axis if x variable is binary
        if df_pair[xvar].nunique() == 2:
            ax.set_xticks([0, 1])
            ax.set_xlim(-0.05, 1.05)

plt.suptitle("Top Features vs Target (Pairwise)", y=1.02)
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import math

# -------------------------------
# 11) Train vs Test distribution check (numeric, grid)
# -------------------------------
print("\n=== TRAIN vs TEST DISTRIBUTION CHECK (numeric) ===")

num_cols = (int_cols + float_cols)[:6]

n_cols = 3
n_rows = math.ceil(len(num_cols) / n_cols)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(5 * n_cols, 4 * n_rows),
    sharey=False
)

axes = axes.flatten()

for ax, col in zip(axes, num_cols):
    sns.kdeplot(
        data=train_eda,
        x=col,
        fill=True,
        alpha=0.4,
        label="Train",
        ax=ax
    )
    sns.kdeplot(
        data=test_eda,
        x=col,
        fill=True,
        alpha=0.4,
        label="Test",
        ax=ax
    )
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Density")

# Remove unused axes
for ax in axes[len(num_cols):]:
    ax.axis("off")

# Single shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower right",
    ncol=2,
    frameon=False
)

fig.suptitle("Train vs Test Feature Distributions (Numeric)", y=1.02)
plt.tight_layout()
plt.show()


# -------------------------------
# 12) EDA summary (console)
# -------------------------------
print("\n=== EDA SUMMARY ===")
print(f"Rows (train): {len(train_eda):,}")
print(f"Rows (test) : {len(test_eda):,}")
print(f"Positive rate ({TARGET_COL}): {float(train_eda[TARGET_COL].mean()):.4f}")
print("\n\n Top 5 numeric abs correlations with target:")
display(target_corr.head(5))

