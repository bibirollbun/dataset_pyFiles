# =========================================================
# SETUP
# =========================================================
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Visualization settings
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("deep")

# Set display options for better readability
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

print("Libraries imported successfully.")


# =========================================================
# DATA LOADING
# =========================================================
# Define data path (update if needed)
DATA_PATH = "/kaggle/input/california-homelessness-prediction-challenge" 

# Load CSV files
train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))

# Print basic info
print("Data Loaded Successfully!\n")
print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"Sample submission shape: {sample_sub.shape}")

# Quick preview of each dataset
print("\nTrain (first five rows):")
display(train.head())
print("Test (first five rows):")
display(test.head())
print("Sample submission (first five rows):")
display(sample_sub.head())


# =========================================================
# INITIAL EXPLORATION
# =========================================================

# 1. View basic info
print("=== Basic Info ===")
train.info()
print("\n")

# 2. Describe numerical columns
print("=== Descriptive Statistics (Numeric Columns) ===")
display(train.describe().T.round(3))

# 3. Display column names and sample data types
print("=== Column Overview ===")
columns_overview = pd.DataFrame({
    "Column": train.columns,
    "Type": train.dtypes.astype(str),
    "Missing (%)": (train.isnull().mean() * 100).round(2)
})
display(columns_overview.head(15))


# =========================================================
# ID COLUMN ANALYSIS — SPLIT INTO COMPONENTS
# =========================================================

# Split ID into CountyCode and RegionNumber
id_split = train['ID'].str.split('_', expand=True)
id_split.columns = ['CountyCode', 'RegionNumber']

# Combine back for quick inspection
train_with_id_parts = pd.concat([train, id_split], axis=1)

# Confirm the split
train_with_id_parts[['ID', 'CountyCode', 'RegionNumber']].head(10)


# =========================================================
# COUNT UNIQUE COUNTY AND REGION VALUES
# =========================================================

print("Unique County Codes:", train_with_id_parts['CountyCode'].nunique())
print("Unique Region Numbers:", train_with_id_parts['RegionNumber'].nunique())

# View all county codes to understand dataset coverage
print("\nCounty Codes:", sorted(train_with_id_parts['CountyCode'].unique()))


# =========================================================
# VISUALIZE HOMELESSNESS RATE BY COUNTY
# =========================================================

plt.figure(figsize=(10, 5))
sns.boxplot(x='CountyCode', y='HOMELESS_RATE', data=train_with_id_parts, palette='Set2')
plt.title('Distribution of Homelessness Rate by County', fontsize=13)
plt.xlabel('County')
plt.ylabel('Homelessness Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# =========================================================
# DATA COMPLETENESS & DUPLICATE CHECK
# =========================================================

# 1. Missing value check
missing_summary = train.isnull().sum()
total_missing = missing_summary.sum()

if total_missing == 0:
    print("No missing values found in the dataset.")
else:
    print("Missing values detected — summary below:")
    display(missing_summary[missing_summary > 0])

# 2. Duplicate check
duplicate_count = train.duplicated().sum()
if duplicate_count == 0:
    print("No duplicate rows found in the dataset.")
else:
    print(f"Found {duplicate_count} duplicate rows.")



# =========================================================
# HISTOGRAMS FOR NUMERIC FEATURES
# =========================================================
import math

# Select numeric columns only
numeric_cols = train.select_dtypes(include='number').columns
graph_count = len(numeric_cols)

# Define grid layout
col_count = 4
row_count = math.ceil(graph_count / col_count)

# Create figure and axes
fig, axes = plt.subplots(row_count, col_count, figsize=(col_count * 3.5, row_count * 3.5))
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.3, hspace=0.4)
axes = axes.ravel()

# Loop through each numeric column
for i, col in enumerate(numeric_cols):
    axes[i].hist(train[col], bins=15, color='steelblue', edgecolor='black')
    axes[i].set_title(col, fontsize=9)
    axes[i].tick_params(axis='x', labelsize=8)
    axes[i].tick_params(axis='y', labelsize=8)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# =========================================================
# BOXPLOTS FOR NUMERIC FEATURES
# =========================================================

# Select numeric columns only
numeric_cols = train.select_dtypes(include='number').columns
graph_count = len(numeric_cols)

# Grid layout
col_count = 4
row_count = math.ceil(graph_count / col_count)

# Figure setup
fig, axes = plt.subplots(row_count, col_count, figsize=(col_count * 3.5, row_count * 3.5))
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.3, hspace=0.6)
axes = axes.ravel()

# Plot boxplots
for i, col in enumerate(numeric_cols):
    sns.boxplot(y=train[col], ax=axes[i], color='lightblue', fliersize=3, linewidth=0.8)
    axes[i].set_title(col, fontsize=9)
    axes[i].tick_params(axis='y', labelsize=8)
    axes[i].set_xlabel('')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# =========================================================
# CORRELATION MATRIX
# =========================================================

# Compute correlation matrix for numeric columns
corr_matrix = train.select_dtypes(include="number").corr()

# Sort features by correlation with target
target_col = "HOMELESS_RATE"
if target_col in corr_matrix.columns:
    target_corr = corr_matrix[target_col].sort_values(ascending=False)
    print("=== Top Features Correlated with HOMELESS_RATE ===")
    display(target_corr.head(10))
    display(target_corr.tail(10))

# Visualize correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm")
plt.title("Correlation Heatmap — All Numeric Features")
plt.tight_layout()
plt.show()



# =========================================================
# RELATIONSHIP WITH TARGET VARIABLE
# =========================================================

# Select top correlated features (excluding the target itself)
top_features = (
    corr_matrix[target_col]
    .drop(target_col)
    .abs()
    .sort_values(ascending=False)
    .head(4)
    .index.tolist()
)

print(f"Top correlated features with {target_col}: {top_features}")

# Pairwise scatterplots
sns.pairplot(
    data=train,
    x_vars=top_features,
    y_vars=[target_col],
    kind="reg",
    diag_kind=None,
    height=3,
    plot_kws={'line_kws': {'color': 'red', 'lw': 1}, 'scatter_kws': {'s': 20, 'alpha': 0.6}}
)
plt.suptitle(f"Top Correlated Features vs {target_col}", y=1.02, fontsize=13)
plt.show()


# =========================================================
# COUNTY-LEVEL ANALYSIS
# =========================================================

# Compute mean homelessness rate by county
county_summary = (
    train_with_id_parts.groupby("CountyCode")["HOMELESS_RATE"]
    .agg(["mean", "std", "min", "max", "count"])
    .sort_values(by="mean", ascending=False)
)

print("=== Average Homelessness Rate by County ===")
display(county_summary.round(4))

# Visualize variation by county
plt.figure(figsize=(10, 5))
sns.barplot(
    data=county_summary.reset_index(),
    x="CountyCode",
    y="mean",
    palette="viridis"
)
plt.title("Average Homelessness Rate by County", fontsize=13)
plt.xlabel("County")
plt.ylabel("Average Homelessness Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# =========================================================
# FEATURE INTERACTION EXPLORATION
# =========================================================

# Define possible pairs (examples — adjust if these columns exist)
interaction_pairs = [
    ("POVERTY_PCT", "RENT_BURDEN_PCT"),
    ("DISABILITY_POP_PCT", "UNEMPLOYMENT_PCT"),
    ("AGE_25_34_PCT", "NONFAMILY_SINGLE_FEMALE_PCT")
]

# Generate new interaction terms
for a, b in interaction_pairs:
    if a in train_with_id_parts.columns and b in train_with_id_parts.columns:
        new_col = f"{a}_x_{b}"
        train_with_id_parts[new_col] = train_with_id_parts[a] * train_with_id_parts[b]
        print(f"Created interaction feature: {new_col}")

# Check correlations of new features with target
interaction_corr = (
    train_with_id_parts[
        [col for col in train_with_id_parts.columns if "_x_" in col] + ["HOMELESS_RATE"]
    ].corr()["HOMELESS_RATE"].sort_values(ascending=False)
)

print("\n=== Correlation of Interaction Features with HOMELESS_RATE ===")
display(interaction_corr)


# =========================================================
# BASELINE MODEL SELECTION WITH SIMPLE TRANSFORMATIONS
# =========================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# --- Setup ---
target = "HOMELESS_RATE"
exclude_cols = ["ID", "CountyCode", "RegionNumber"]
features = [col for col in train_with_id_parts.columns if col not in exclude_cols + [target]]

X = train_with_id_parts[features]
y = train_with_id_parts[target]

# --- Split data ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Define models ---
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.001),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# --- Evaluate models ---
results = []
for name, model in models.items():
    pipe = Pipeline([
        ("yeojohnson", PowerTransformer(method="yeo-johnson")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    results.append({"Model": name, "MAE": mae, "RMSE": rmse})

results_df = pd.DataFrame(results).sort_values(by="MAE").reset_index(drop=True)
display(results_df.style.format({"MAE": "{:.6f}", "RMSE": "{:.6f}"}))

