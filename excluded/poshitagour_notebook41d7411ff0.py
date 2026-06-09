# STEP 2: Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Make plots look clean
sns.set_style("whitegrid")

print("Libraries loaded successfully")



# STEP 3: Load the dataset (small sample)

print("Loading data...")

# Load first 50,000 rows only (safe for Kaggle)
train_data = pd.read_csv(
    "/kaggle/input/amex-default-prediction/train_data.csv",
    nrows=50000
)

# Load labels
train_labels = pd.read_csv(
    "/kaggle/input/amex-default-prediction/train_labels.csv"
)

print("Train data shape:", train_data.shape)
print("Train labels shape:", train_labels.shape)



# STEP 4: Merge train data with labels

print("Merging data...")

df = train_data.merge(
    train_labels,
    on="customer_ID",
    how="left"
)

print("Merged dataset shape:", df.shape)



# STEP 5: Quick data check

print("Column names:")
print(df.columns)

print("\nFirst 5 rows of the dataset:")
df.head()



# STEP 6: Check missing values

total_missing = df.isnull().sum().sum()
print("Total missing values in dataset:", total_missing)



# STEP 7: Analyse the target variable

target_distribution = df["target"].value_counts()
target_percentage = df["target"].value_counts(normalize=True) * 100

print("Target counts:")
print(target_distribution)

print("\nTarget percentage:")
print(target_percentage)



# STEP 8: Visualise default vs non-default

plt.figure(figsize=(6, 4))
sns.countplot(x="target", data=df, palette="viridis")

plt.title("Distribution of Credit Defaults")
plt.xlabel("Customer Status (0 = Paid, 1 = Default)")
plt.ylabel("Number of Customers")

plt.show()



# STEP 8 (UPGRADED): Professional default distribution chart

plt.figure(figsize=(7, 5))

ax = sns.countplot(
    x="target",
    data=df,
    hue="target",
    palette=["#1f77b4", "#d62728"],
    legend=False
)

# Titles and labels
plt.title("Credit Default Distribution", fontsize=14, weight="bold")
plt.xlabel("Customer Status (0 = Paid, 1 = Default)", fontsize=11)
plt.ylabel("Number of Customers", fontsize=11)

# Add numbers on top of bars
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha="center",
        va="bottom",
        fontsize=10
    )

sns.despine()
plt.tight_layout()
plt.show()



# STEP 8 (FINAL): Advanced & aesthetic default distribution chart

# Calculate percentages for labels
counts = df["target"].value_counts().sort_index()
percentages = df["target"].value_counts(normalize=True).sort_index() * 100

# Create figure
plt.figure(figsize=(8, 5))

ax = sns.barplot(
    x=["Paid (0)", "Default (1)"],
    y=counts.values,
    palette=["#4C72B0", "#DD8452"]
)

# Title and labels
plt.title("Customer Credit Default Distribution", fontsize=15, weight="bold", pad=15)
plt.ylabel("Number of Customers", fontsize=11)
plt.xlabel("Customer Status", fontsize=11)

# Add value + percentage labels
for i, value in enumerate(counts.values):
    ax.text(
        i,
        value + 500,
        f"{value:,}\n({percentages.iloc[i]:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=11,
        weight="bold"
    )

# Style cleanup
sns.despine(left=True, bottom=True)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

plt.show()



# STEP 9: Which customers default? (Spending behaviour comparison)

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="target",
    y="S_3",
    data=df,
    palette=["#4C72B0", "#DD8452"]
)

plt.title(
    "Spending Behaviour by Customer Default Status",
    fontsize=15,
    weight="bold",
    pad=15
)

plt.xlabel("Customer Status (0 = Paid, 1 = Default)", fontsize=11)
plt.ylabel("Spending Metric (S_3)", fontsize=11)

sns.despine()
plt.tight_layout()
plt.show()


# STEP 9 (EXECUTIVE VERSION): Who defaults? Spend vs Payment

plt.figure(figsize=(9, 6))

scatter = sns.scatterplot(
    data=df,
    x="S_3",
    y="P_2",
    hue="target",
    palette={0: "#4C72B0", 1: "#C44E52"},
    alpha=0.6
)

plt.title(
    "Customer Behaviour Segmentation: Spend vs Payment",
    fontsize=15,
    weight="bold",
    pad=15
)

plt.xlabel("Spending Behaviour (Higher = More Spend)", fontsize=11)
plt.ylabel("Payment Behaviour (Higher = Better Repayment)", fontsize=11)

# Improve legend
plt.legend(
    title="Customer Outcome",
    labels=["Paid (0)", "Defaulted (1)"],
    frameon=False
)

sns.despine()
plt.tight_layout()
plt.show()



# STEP 9 (FINAL): Executive segmentation – Default rate by customer type

# Create simple behavioural segments using medians
spend_median = df["S_3"].median()
payment_median = df["P_2"].median()

def classify_customer(row):
    if row["S_3"] <= spend_median and row["P_2"] > payment_median:
        return "Low Spend / High Payment (Safe)"
    elif row["S_3"] <= spend_median and row["P_2"] <= payment_median:
        return "Low Spend / Low Payment"
    elif row["S_3"] > spend_median and row["P_2"] > payment_median:
        return "High Spend / High Payment"
    else:
        return "High Spend / Low Payment (High Risk)"

df["Customer_Type"] = df.apply(classify_customer, axis=1)

# Calculate default rate per segment
segment_default_rate = (
    df.groupby("Customer_Type")["target"]
    .mean()
    .sort_values(ascending=False) * 100
)

# Plot
plt.figure(figsize=(9, 5))

ax = sns.barplot(
    x=segment_default_rate.values,
    y=segment_default_rate.index,
    palette="Reds_r"
)

plt.title(
    "Which Customer Types Are Most Likely to Default?",
    fontsize=15,
    weight="bold",
    pad=15
)

plt.xlabel("Default Rate (%)", fontsize=11)
plt.ylabel("Customer Behaviour Segment", fontsize=11)

# Add labels
for i, v in enumerate(segment_default_rate.values):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=11)

sns.despine()
plt.tight_layout()
plt.show()



# STEP 10: Early warning signal – Payment behaviour comparison

plt.figure(figsize=(9, 5))

# Calculate average payment behaviour by outcome
payment_by_target = df.groupby("target")["P_2"].mean()

ax = sns.barplot(
    x=["Paid (0)", "Defaulted (1)"],
    y=payment_by_target.values,
    palette=["#4C72B0", "#C44E52"]
)

plt.title(
    "Early Warning Signal: Decline in Payment Behaviour Before Default",
    fontsize=15,
    weight="bold",
    pad=15
)

plt.xlabel("Customer Outcome", fontsize=11)
plt.ylabel("Average Payment Behaviour (P_2)", fontsize=11)

# Add values on bars
for i, v in enumerate(payment_by_target.values):
    ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=11, weight="bold")

sns.despine()
plt.tight_layout()
plt.show()



# STEP 10 (FINAL): Early warning signal – Default rate by payment behaviour level

# Create payment behaviour bands (early warning levels)
df["Payment_Level"] = pd.qcut(
    df["P_2"],
    q=4,
    labels=[
        "Very Low Payment",
        "Low Payment",
        "Moderate Payment",
        "High Payment"
    ]
)

# Calculate default rate per payment level
payment_default_rate = (
    df.groupby("Payment_Level")["target"]
    .mean()
    .sort_values(ascending=False) * 100
)

# Plot
plt.figure(figsize=(9, 5))

ax = sns.barplot(
    x=payment_default_rate.values,
    y=payment_default_rate.index,
    palette="Reds_r"
)

plt.title(
    "Early Warning Signal: Default Risk by Payment Behaviour Level",
    fontsize=15,
    weight="bold",
    pad=15
)

plt.xlabel("Default Rate (%)", fontsize=11)
plt.ylabel("Payment Behaviour Level", fontsize=11)

# Add labels
for i, v in enumerate(payment_default_rate.values):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=11)

sns.despine()
plt.tight_layout()
plt.show()



# STEP 10 (ALTERNATIVE): Early warning funnel – Payment deterioration to default risk

# Create ordered payment behaviour bands (early to late warning)
df["Payment_Band"] = pd.qcut(
    df["P_2"],
    q=4,
    labels=[
        "Strong Payment Behaviour",
        "Moderate Payment Behaviour",
        "Weak Payment Behaviour",
        "Critical Payment Behaviour"
    ]
)

# Calculate default rate
funnel_data = (
    df.groupby("Payment_Band")["target"]
    .mean()
    .reindex([
        "Strong Payment Behaviour",
        "Moderate Payment Behaviour",
        "Weak Payment Behaviour",
        "Critical Payment Behaviour"
    ]) * 100
)

# Plot funnel-style horizontal bars
plt.figure(figsize=(9, 5))

ax = sns.barplot(
    x=funnel_data.values,
    y=funnel_data.index,
    palette=["#4CAF50", "#FFC107", "#FF9800", "#D32F2F"]
)

plt.title(
    "Early Warning Funnel: Payment Behaviour Deterioration and Default Risk",
    fontsize=15,
    weight="bold",
    pad=15
)

plt.xlabel("Default Rate (%)", fontsize=11)
plt.ylabel("Payment Behaviour Stage", fontsize=11)

# Add labels
for i, v in enumerate(funnel_data.values):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=11, weight="bold")

sns.despine()
plt.tight_layout()
plt.show()



# STEP 10 (FINAL Q2): Early warning signal – Risk escalation line chart

# Create ordered payment behaviour bands
df["Payment_Band"] = pd.qcut(
    df["P_2"],
    q=5,
    labels=[
        "Very Strong",
        "Strong",
        "Moderate",
        "Weak",
        "Critical"
    ]
)

# Calculate default rate per band
risk_curve = (
    df.groupby("Payment_Band")["target"]
    .mean()
    .reindex([
        "Very Strong",
        "Strong",
        "Moderate",
        "Weak",
        "Critical"
    ]) * 100
)

# Plot line chart
plt.figure(figsize=(9, 5))

plt.plot(
    risk_curve.index,
    risk_curve.values,
    marker="o",
    linewidth=3,
    color="#C44E52"
)

plt.fill_between(
    risk_curve.index,
    risk_curve.values,
    alpha=0.15,
    color="#C44E52"
)

plt.title(
    "Early Warning Signal: Default Risk Escalates as Payment Behaviour Deteriorates",
    fontsize=15,
    weight="bold",
    pad=15
)

plt.xlabel("Payment Behaviour (Best → Worst)", fontsize=11)
plt.ylabel("Default Rate (%)", fontsize=11)

# Add point labels
for i, v in enumerate(risk_curve.values):
    plt.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=11, weight="bold")

sns.despine()
plt.tight_layout()
plt.show()



# STEP 10 (CORRECTED): Early warning signal – intuitive risk escalation line chart

# Create payment behaviour bands (LOW P_2 = WORST payment)
df["Payment_Band"] = pd.qcut(
    df["P_2"],
    q=5,
    labels=[
        "Critical (Worst Payment)",
        "Weak",
        "Moderate",
        "Strong",
        "Very Strong (Best Payment)"
    ]
)

# Calculate default rate per band
risk_curve = (
    df.groupby("Payment_Band")["target"]
    .mean()
    .reindex([
        "Critical (Worst Payment)",
        "Weak",
        "Moderate",
        "Strong",
        "Very Strong (Best Payment)"
    ]) * 100
)

# Plot line chart
plt.figure(figsize=(9, 5))

plt.plot(
    risk_curve.index,
    risk_curve.values,
    marker="o",
    linewidth=3,
    color="#C44E52"
)

plt.fill_between(
    range(len(risk_curve)),
    risk_curve.values,
    alpha=0.15,
    color="#C44E52"
)

plt.title(
    "Early Warning Signal: Default Risk Decreases as Payment Behaviour Improves",
    fontsize=15,
    weight="bold",
    pad=15
)

plt.xlabel("Payment Behaviour (Worst → Best)", fontsize=11)
plt.ylabel("Default Rate (%)", fontsize=11)

# Add value labels
for i, v in enumerate(risk_curve.values):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=11, weight="bold")

sns.despine()
plt.tight_layout()
plt.show()



# Q3: Business impact of targeting high-risk groups

# Create risk tiers based on payment behaviour
df["Risk_Tier"] = pd.qcut(
    df["P_2"],
    q=3,
    labels=[
        "High Risk",
        "Medium Risk",
        "Low Risk"
    ]
)

# Calculate customer share and default share
summary = (
    df.groupby("Risk_Tier")
    .agg(
        Customers=("customer_ID", "count"),
        Defaults=("target", "sum")
    )
)

summary["Customer Share (%)"] = summary["Customers"] / summary["Customers"].sum() * 100
summary["Default Share (%)"] = summary["Defaults"] / summary["Defaults"].sum() * 100

# Reorder for presentation
summary = summary.loc[["High Risk", "Medium Risk", "Low Risk"]]

# Plot
plt.figure(figsize=(9, 5))

x = range(len(summary))

plt.bar(
    x,
    summary["Customer Share (%)"],
    width=0.4,
    label="Customer Share (%)",
    color="#4C72B0"
)

plt.bar(
    [i + 0.4 for i in x],
    summary["Default Share (%)"],
    width=0.4,
    label="Default Share (%)",
    color="#C44E52"
)

plt.xticks(
    [i + 0.2 for i in x],
    summary.index
)

plt.title(
    "Business Impact of Targeting High-Risk Customers",
    fontsize=15,
    weight="bold",
    pad=15
)

plt.ylabel("Percentage (%)", fontsize=11)
plt.xlabel("Customer Risk Segment", fontsize=11)

plt.legend(frameon=False)
sns.despine()
plt.tight_layout()
plt.show()



# STEP 12 (FIXED): Q3 – Business impact of targeting high-risk customers (Pareto curve)

# Create a risk score proxy (lower P_2 = higher risk)
df_q3 = df[["P_2", "target"]].dropna().copy()
df_q3["Risk_Score"] = -df_q3["P_2"]

# Sort customers by risk (highest risk first)
df_q3 = df_q3.sort_values("Risk_Score", ascending=False).reset_index(drop=True)

# Cumulative calculations (FIXED)
df_q3["Cumulative_Customers"] = (
    (df_q3.index + 1) / len(df_q3) * 100
)

df_q3["Cumulative_Defaults"] = (
    df_q3["target"].cumsum() / df_q3["target"].sum() * 100
)

# Plot Pareto curve
plt.figure(figsize=(9, 6))

plt.plot(
    df_q3["Cumulative_Customers"],
    df_q3["Cumulative_Defaults"],
    linewidth=3,
    color="#C44E52",
    label="Defaults Captured by Targeting"
)

# Reference diagonal (random targeting)
plt.plot(
    [0, 100],
    [0, 100],
    linestyle="--",
    color="grey",
    alpha=0.6,
    label="Random Targeting"
)

plt.title(
    "Business Impact of Targeting High-Risk Customers",
    fontsize=15,
    weight="bold",
    pad=15
)

plt.xlabel("Customers Targeted (%)", fontsize=11)
plt.ylabel("Defaults Captured (%)", fontsize=11)

plt.legend(frameon=False)
sns.despine()
plt.tight_layout()
plt.show()



# STEP 12 (GRAND VERSION): Lorenz-style risk concentration curve with impact shading

import numpy as np

# Prepare data
df_q3 = df[["P_2", "target"]].dropna().copy()
df_q3["Risk_Score"] = -df_q3["P_2"]  # higher score = higher risk

# Sort by risk (highest first)
df_q3 = df_q3.sort_values("Risk_Score", ascending=False).reset_index(drop=True)

# Cumulative shares
df_q3["cum_customers"] = (df_q3.index + 1) / len(df_q3)
df_q3["cum_defaults"] = df_q3["target"].cumsum() / df_q3["target"].sum()

# Convert to percentages
x = df_q3["cum_customers"] * 100
y = df_q3["cum_defaults"] * 100

# Plot
plt.figure(figsize=(10, 7))

# Targeting curve
plt.plot(
    x, y,
    color="#C62828",
    linewidth=3,
    label="Targeted Risk Strategy"
)

# Random baseline
plt.plot(
    [0, 100], [0, 100],
    linestyle="--",
    color="gray",
    linewidth=2,
    label="Untargeted (Random) Strategy"
)

# Impact area shading
plt.fill_between(
    x, y, x,
    where=(y > x),
    color="#C62828",
    alpha=0.25,
    label="Value Created by Targeting"
)

# Titles & labels
plt.title(
    "Business Impact of Targeting High-Risk Customers",
    fontsize=16,
    weight="bold",
    pad=20
)

plt.xlabel("Customers Targeted (%)", fontsize=12)
plt.ylabel("Defaults Captured (%)", fontsize=12)

plt.xlim(0, 100)
plt.ylim(0, 100)

plt.legend(frameon=False, fontsize=11)
sns.despine()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()



# Q3: Business impact of targeting high-risk groups
# Graph type: Stacked bar (default concentration)

# Create risk segments using payment behaviour
df_q3 = df[["P_2", "target"]].dropna().copy()

df_q3["Risk_Segment"] = pd.qcut(
    df_q3["P_2"],
    q=3,
    labels=["High Risk", "Medium Risk", "Low Risk"]
)

# Calculate default contribution by segment
default_contribution = (
    df_q3[df_q3["target"] == 1]
    .groupby("Risk_Segment")
    .size()
    / df_q3[df_q3["target"] == 1].shape[0]
    * 100
)

# Prepare data for stacked bar
segments = default_contribution.index.tolist()
values = default_contribution.values.tolist()

# Plot
plt.figure(figsize=(8, 4))

left = 0
colors = ["#C62828", "#F9A825", "#2E7D32"]

for segment, value, color in zip(segments, values, colors):
    plt.barh(
        y="Total Defaults",
        width=value,
        left=left,
        color=color,
        label=f"{segment} ({value:.1f}%)"
    )
    left += value

plt.title(
    "Business Impact of Targeting High-Risk Customers",
    fontsize=14,
    weight="bold",
    pad=12
)

plt.xlabel("Share of Total Defaults (%)")
plt.ylabel("")
plt.xlim(0, 100)

plt.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
sns.despine(left=True)
plt.tight_layout()
plt.show()



# STEP 0: Load data and create df

import pandas as pd

print("Loading data...")

# Load labels
train_labels = pd.read_csv("/kaggle/input/amex-default-prediction/train_labels.csv")

# Load a manageable sample of training data
train_data = pd.read_csv(
    "/kaggle/input/amex-default-prediction/train_data.csv",
    nrows=50000
)

# Merge to create final dataframe
df = train_data.merge(train_labels, on="customer_ID", how="left")

print("Data loaded successfully.")
print("df is now ready for analysis.")



# STEP 1: Dataset Validation - Structure & Size

print("STEP 1: DATASET STRUCTURE & SIZE CHECK\n")

rows, columns = df.shape
print(f"Number of rows (records): {rows}")
print(f"Number of columns (features): {columns}")

memory_usage = df.memory_usage(deep=True).sum() / (1024**2)
print(f"Approximate memory usage: {memory_usage:.2f} MB")



# STEP 2: Data Types & Schema Check

print("STEP 2: DATA TYPES & SCHEMA CHECK\n")

# Show data types
df_types = df.dtypes.value_counts()
print("Data type distribution:")
print(df_types)

print("\nSample of column data types:")
print(df.dtypes.head(10))

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

print(f"\nNumber of numeric columns: {len(numeric_cols)}")
print(f"Number of categorical columns: {len(categorical_cols)}")

print("\nCategorical columns (sample):")
print(list(categorical_cols[:10]))



# STEP 3: Missing Value Analysis (Data Quality Check)

print("STEP 3: MISSING VALUE ANALYSIS\n")

# Total missing values
total_missing = df.isnull().sum().sum()
print(f"Total missing values in dataset: {total_missing}")

# Missing values per column (top 10)
missing_by_column = (
    df.isnull()
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 10 columns with missing values:")
print(missing_by_column.head(10))

# Percentage missing for top columns
missing_percentage = (missing_by_column / len(df)) * 100

print("\nMissing percentage (Top 10 columns):")
print(missing_percentage.head(10).round(2))



# STEP 4: Target Variable Validation

print("STEP 4: TARGET VARIABLE CHECK\n")

# Count default vs non-default
target_counts = df["target"].value_counts()
target_percent = df["target"].value_counts(normalize=True) * 100

print("Target counts:")
print(target_counts)

print("\nTarget distribution (%):")
print(target_percent.round(2))



# STEP 5: Descriptive Statistics & Initial EDA

print("STEP 5: DESCRIPTIVE STATISTICS & INITIAL EDA\n")

key_features = ["P_2", "S_3", "B_1", "target"]

# Overall descriptive statistics
print("Overall Descriptive Statistics:")
print(df[key_features].describe().round(2))

# Descriptive statistics by default status
print("\nDescriptive Statistics by Default Status:")
grouped_stats = df[key_features].groupby("target").describe().round(2)
print(grouped_stats)



# STEP 6: Visual EDA – Clean, Business-Ready Charts (FIXED)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

# Convert target to string to avoid palette issues
df["target_str"] = df["target"].astype(str)

# -------- Chart 1: Payment Behaviour vs Default --------
plt.figure(figsize=(8, 5))

sns.boxplot(
    x="target_str",
    y="P_2",
    hue="target_str",
    data=df,
    palette={"0": "#2E7D32", "1": "#C62828"},
    legend=False
)

plt.title(
    "Payment Behaviour by Default Status",
    fontsize=14,
    weight="bold",
    pad=10
)

plt.xlabel("Customer Status (0 = Non-Default, 1 = Default)")
plt.ylabel("Payment Behaviour (P_2)")

plt.tight_layout()
plt.show()


# -------- Chart 2: Balance vs Default --------
plt.figure(figsize=(8, 5))

sns.boxplot(
    x="target_str",
    y="B_1",
    hue="target_str",
    data=df,
    palette={"0": "#2E7D32", "1": "#C62828"},
    legend=False
)

plt.title(
    "Outstanding Balance by Default Status",
    fontsize=14,
    weight="bold",
    pad=10
)

plt.xlabel("Customer Status (0 = Non-Default, 1 = Default)")
plt.ylabel("Balance Metric (B_1)")

plt.tight_layout()
plt.show()



# STEP 6 – Advanced Visual EDA
# Graph 1: Payment Behaviour Distribution by Default Status

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white", font_scale=1.1)

plt.figure(figsize=(9, 5))

sns.kdeplot(
    data=df[df["target"] == 0],
    x="P_2",
    fill=True,
    alpha=0.5,
    linewidth=2,
    label="Non-Default",
    color="#2E7D32"
)

sns.kdeplot(
    data=df[df["target"] == 1],
    x="P_2",
    fill=True,
    alpha=0.5,
    linewidth=2,
    label="Default",
    color="#C62828"
)

plt.title(
    "Payment Behaviour Distribution by Default Risk",
    fontsize=15,
    weight="bold",
    pad=12
)

plt.xlabel("Payment Behaviour (P₂)")
plt.ylabel("Density")
plt.legend(frameon=False)
sns.despine()
plt.tight_layout()
plt.show()



# STEP 6 – Advanced Visual EDA
# Graph 2: Default Risk Gradient vs Payment Behaviour

# Create payment behaviour bins
df["P2_bin"] = pd.qcut(df["P_2"], q=20)

# Calculate default rate per bin
risk_curve = df.groupby("P2_bin", observed=True)["target"].mean().reset_index()

# Extract midpoints for plotting
risk_curve["P2_mid"] = risk_curve["P2_bin"].apply(lambda x: x.mid)

plt.figure(figsize=(9, 5))

sns.scatterplot(
    x="P2_mid",
    y="target",
    data=risk_curve,
    s=80,
    color="#C62828"
)

sns.lineplot(
    x="P2_mid",
    y="target",
    data=risk_curve,
    linewidth=3,
    color="#C62828"
)

plt.title(
    "Default Risk Escalation as Payment Behaviour Deteriorates",
    fontsize=15,
    weight="bold",
    pad=12
)

plt.xlabel("Payment Behaviour (P₂)")
plt.ylabel("Default Rate")
sns.despine()
plt.tight_layout()
plt.show()



# STEP 6A: Ridgeline Plot (Joy Plot) – Payment Behaviour

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

# Prepare data
df_ridge = df[["P_2", "target"]].dropna()
df_ridge["target_label"] = df_ridge["target"].map({0: "Non-Default", 1: "Default"})

# Plot
plt.figure(figsize=(9, 5))

for i, label in enumerate(["Non-Default", "Default"]):
    subset = df_ridge[df_ridge["target_label"] == label]["P_2"]
    density = sns.kdeplot(
        subset,
        fill=True,
        linewidth=2,
        alpha=0.8,
        label=label
    )
    for artist in density.collections:
        artist.set_alpha(0.6)

plt.title(
    "Ridgeline View of Payment Behaviour by Risk Outcome",
    fontsize=15,
    weight="bold",
    pad=12
)

plt.xlabel("Payment Behaviour (P₂)")
plt.ylabel("")
plt.yticks([])
plt.legend(frameon=False)
sns.despine(left=True)
plt.tight_layout()
plt.show()



# STEP 6B: Hexbin Risk Landscape – Payment vs Balance

plt.figure(figsize=(8, 6))

plt.hexbin(
    df["P_2"],
    df["B_1"],
    C=df["target"],
    gridsize=40,
    cmap="inferno",
    reduce_C_function=np.mean
)

cb = plt.colorbar()
cb.set_label("Average Default Risk")

plt.title(
    "Customer Risk Landscape: Payment vs Balance",
    fontsize=15,
    weight="bold",
    pad=12
)

plt.xlabel("Payment Behaviour (P₂)")
plt.ylabel("Balance Metric (B₁)")
sns.despine()
plt.tight_layout()
plt.show()



# STEP 6B: Hexbin Risk Landscape – Payment vs Balance

plt.figure(figsize=(8, 6))

plt.hexbin(
    df["P_2"],
    df["B_1"],
    C=df["target"],
    gridsize=40,
    cmap="inferno",
    reduce_C_function=np.mean
)

cb = plt.colorbar()
cb.set_label("Average Default Risk")

plt.title(
    "Customer Risk Landscape: Payment vs Balance",
    fontsize=15,
    weight="bold",
    pad=12
)

plt.xlabel("Payment Behaviour (P₂)")
plt.ylabel("Balance Metric (B₁)")
sns.despine()
plt.tight_layout()
plt.show()



# STEP 6C: Styled Correlation Heatmap (Aesthetic)

features = ["P_2", "S_3", "B_1", "target"]
corr = df[features].corr()

plt.figure(figsize=(7, 5))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="magma",
    linewidths=1,
    linecolor="black",
    cbar_kws={"shrink": 0.8}
)

plt.title(
    "Correlation Structure of Key Risk Drivers",
    fontsize=14,
    weight="bold",
    pad=12
)

plt.tight_layout()
plt.show()



# STEP 6D: Risk Contour Map – Default Topography

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Prepare data
df_contour = df[["P_2", "B_1", "target"]].dropna()

# Only defaulters for risk surface
x = df_contour[df_contour["target"] == 1]["P_2"]
y = df_contour[df_contour["target"] == 1]["B_1"]

# Create grid
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]

# Kernel Density Estimation
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)

# Plot
plt.figure(figsize=(9, 6))
plt.contourf(xx, yy, f, levels=30, cmap="plasma")
plt.colorbar(label="Default Risk Density")

plt.title(
    "Default Risk Topography: Payment vs Balance",
    fontsize=16,
    weight="bold",
    pad=15
)

plt.xlabel("Payment Behaviour (P₂)")
plt.ylabel("Balance Metric (B₁)")
plt.tight_layout()
plt.show()



# STEP 6E: Feature–Risk Network Graph

import networkx as nx

# Select key features
features = ["P_2", "S_3", "B_1"]
corr = df[features + ["target"]].corr()["target"].drop("target")

# Create graph
G = nx.Graph()

# Add target node
G.add_node("Default Risk", size=3000)

# Add feature nodes
for feature in features:
    G.add_node(feature, size=1500)
    G.add_edge("Default Risk", feature, weight=abs(corr[feature]))

# Layout
pos = nx.spring_layout(G, seed=42)

# Plot
plt.figure(figsize=(8, 6))

# Nodes
nx.draw_networkx_nodes(
    G, pos,
    node_size=[G.nodes[n]["size"] for n in G.nodes],
    node_color=["#C62828" if n == "Default Risk" else "#1565C0" for n in G.nodes]
)

# Edges
nx.draw_networkx_edges(
    G, pos,
    width=[G[u][v]["weight"] * 10 for u, v in G.edges],
    alpha=0.7
)

# Labels
nx.draw_networkx_labels(G, pos, font_size=11, font_color="white")

plt.title(
    "Network View of Feature Influence on Default Risk",
    fontsize=15,
    weight="bold",
    pad=15
)

plt.axis("off")
plt.tight_layout()
plt.show()



# STEP 6F: Risk Galaxy – Artistic Scatter

plt.figure(figsize=(9, 6))

plt.scatter(
    df["P_2"],
    df["S_3"],
    c=df["target"],
    cmap="coolwarm",
    alpha=0.25,
    s=15
)

plt.title(
    "Customer Risk Galaxy: Payment vs Spend",
    fontsize=16,
    weight="bold",
    pad=15
)

plt.xlabel("Payment Behaviour (P₂)")
plt.ylabel("Spending Behaviour (S₃)")
plt.colorbar(label="Default Risk")

plt.grid(False)
plt.tight_layout()
plt.show()


