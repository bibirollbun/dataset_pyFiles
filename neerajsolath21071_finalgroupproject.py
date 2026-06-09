import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# STEP 1 — Load a sampled version of train.csv using chunk loading (Kaggle path)

DATA_PATH = "/kaggle/input/microsoft-malware-prediction/train.csv"

chunksize = 500_000         # load 500k rows at a time
sample_frac = 0.10          # sample 10% of every chunk
random_state = 42

dfs = []
row_count = 0

for chunk in pd.read_csv(DATA_PATH, chunksize=chunksize):
    row_count += len(chunk)

    # sample inside the chunk
    chunk_sample = chunk.sample(frac=sample_frac, random_state=random_state)
    dfs.append(chunk_sample)

    print(f"Processed {row_count:,} rows so far...")

df = pd.concat(dfs, ignore_index=True)

print("\n✅ Finished sampling!")
print("Sample shape:", df.shape)
df.head()



# STEP 2 — Basic data overview

print("Shape:", df.shape)
print("\nColumn types:\n", df.dtypes.value_counts())

print("\nSample of dtypes:")
print(df.dtypes.head(10))

print("\nTarget distribution:")
print(df["HasDetections"].value_counts())
print("\nTarget proportions:")
print(df["HasDetections"].value_counts(normalize=True))

sns.countplot(x="HasDetections", data=df)
plt.title("Target distribution")
plt.show()



# STEP 3 — Missing values summary

missing = df.isnull().mean().sort_values(ascending=False) * 100
missing_df = missing.reset_index()
missing_df.columns = ['Feature', 'MissingPercent']

print(missing_df.head(15))  # top 15 missing features
print("\nTotal features with any missing data:", (missing > 0).sum())

# Plot top 20 missing
plt.figure(figsize=(10,6))
sns.barplot(
    x=missing_df['MissingPercent'].head(20),
    y=missing_df['Feature'].head(20),
    palette="viridis"
)
plt.title("Top 20 Features with Highest Missing Percent")
plt.xlabel("% Missing")
plt.ylabel("Feature")
plt.show()



# STEP 4 — Identify likely missingness types (MCAR, MAR, MNAR/NI)
# (This creates a dictionary you can use in your report)

missing_types = {
    "MCAR": [
        "Occasional OS telemetry fields that are missing at random",
    ],
    
    "MAR": [
        "Census columns depending on OS version",
        "Hardware info missing only on specific device classes",
        "Features missing because the machine lacks a setting but correlated with other features"
    ],
    
    "MNAR_or_NI": [
        "Columns missing because the machine does not have a specific hardware component",
        "Missing categorical fields where the missingness itself implies hardware absence",
    ]
}

missing_types



# STEP 5 — Drop columns with extreme missingness (>90%)

high_missing_cols = missing[missing > 90].index.tolist()
print("Dropping columns with >90% missing values:", high_missing_cols)

df = df.drop(columns=high_missing_cols)

print("\nNew shape after dropping high-missing columns:", df.shape)



# STEP 6 — Separate numerical and categorical features

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

print("Numeric columns:", len(numeric_cols))
print("Categorical columns:", len(categorical_cols))

# Make sure target stays numeric
numeric_cols = [col for col in numeric_cols if col != "HasDetections"]



# STEP 7 — Impute missing values

# Numeric: median fill
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Categorical: "Unknown" fill
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# Verify no missing values left
print("Remaining missing values:", df.isnull().sum().sum())



my_features = [
    "MachineIdentifier",
    "Processor",
    "IsSxsPassiveMode",
    "HasDetections",
    "OsPlatformSubRelease",          # <- Os not OS
    "OsBuild",                       # <- Os not OS
    "OsBuildLab",                    # <- Os
    "Census_ProcessorCoreCount",
    "Census_ProcessorManufacturerIdentifier",
    "Census_ProcessorModelIdentifier",
    "Census_PrimaryDiskTotalCapacity",
    "Census_PrimaryDiskTypeName"
]

for f in my_features:
    pct = df[f].isnull().mean() * 100
    print(f"{f}: {pct:.4f}% missing")



# Check if OsArchitecture exists, and if so, show missingness %
if "OsArchitecture" in df.columns:
    pct = df["OsArchitecture"].isnull().mean() * 100
    print(f"OsArchitecture: {pct:.4f}% missing")
else:
    print("Column 'OsArchitecture' not found in df.columns")



# ---- STEP: Define your assigned features ----

my_features = [
    "MachineIdentifier",
    "Processor",
    "IsSxsPassiveMode",
    "HasDetections",
    "OsPlatformSubRelease",
    "OsBuild",
    "OsBuildLab",
    "OsArchitecture",
    "Census_ProcessorCoreCount",
    "Census_ProcessorManufacturerIdentifier",
    "Census_ProcessorModelIdentifier",
    "Census_PrimaryDiskTotalCapacity",
    "Census_PrimaryDiskTypeName"
]

# ---- STEP: Helper to classify human-friendly variable type ----

def classify_variable_type(series):
    dt = series.dtype
    
    # IDs / identifiers (long strings with many unique values)
    if series.name.lower().endswith("identifier") or series.nunique() > 50000:
        return "Identifier / High Cardinality"
    
    # Numeric (int or float)
    if dt in ["int64", "float64"]:
        # binary detection
        if series.nunique() == 2:
            return "Binary Numeric"
        else:
            return "Numeric"
    
    # Categorical
    if dt == "object":
        # version strings often look like x.x.x.x
        sample = str(series.dropna().iloc[0])
        if "." in sample and any(char.isdigit() for char in sample):
            return "Version String / Categorical"
        return "Categorical"
    
    return str(dt)

# ---- STEP: Loop through your features and print summary ----

print("===== SUMMARY FOR MINAH'S FEATURES =====\n")

for f in my_features:
    print(f"Feature: {f}")
    
    if f not in df.columns:
        print("   ❌ Not found in df.columns\n")
        continue
    
    series = df[f]
    
    # Missingness %
    missing_pct = series.isnull().mean() * 100
    
    # Raw dtype
    raw_dtype = series.dtype
    
    # Human variable type
    var_type = classify_variable_type(series)
    
    print(f"   Exists: YES")
    print(f"   Missingness: {missing_pct:.4f}%")
    print(f"   Raw dtype: {raw_dtype}")
    print(f"   Variable Type: {var_type}\n")


