# -*- coding: utf-8 -*-
"""Microsoft Malware Prediction — Simplified Final Notebook"""

# ==============================
# Imports
# ==============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

# ==============================
# STEP 1 — Load training data
# ==============================

TRAIN_PATH = "/kaggle/input/microsoft-malware-prediction/train.csv"
TEST_PATH = "/kaggle/input/microsoft-malware-prediction/test.csv"

df = pd.read_csv(TRAIN_PATH)

# Optional: sample for faster training (comment out if you want full data)
df = df.sample(frac=0.2, random_state=42)

print("Train sample shape:", df.shape)

# ==============================
# STEP 2 — Basic missing summary and drop very sparse columns
# ==============================

missing = df.isnull().mean().sort_values(ascending=False) * 100
high_missing_cols = missing[missing > 90].index.tolist()

print("Dropping columns with >90% missing:", high_missing_cols)
df = df.drop(columns=high_missing_cols, errors="ignore")

print("Shape after dropping high-missing columns:", df.shape)

# ==============================
# STEP 3 — Feature engineering
# ==============================

# Helper: version major component
def extract_major(x):
    try:
        return int(str(x).split(".")[0])
    except:
        return -1

# Version majors
if "EngineVersion" in df.columns:
    df["EngineVersion_Major"] = df["EngineVersion"].apply(extract_major)
if "AppVersion" in df.columns:
    df["AppVersion_Major"] = df["AppVersion"].apply(extract_major)
if "AvSigVersion" in df.columns:
    df["AvSigVersion_Major"] = df["AvSigVersion"].apply(extract_major)

# CountryIdentifier → region
country_region_map = {
    1: "North America", 2: "Europe", 3: "Europe", 4: "South Asia",
    5: "Europe", 6: "South America", 7: "Europe", 8: "Middle East",
    9: "Europe", 10: "Europe", 11: "Oceania", 12: "Europe", 13: "East Asia",
    14: "Africa", 15: "Europe", 16: "Africa", 17: "East Asia",
    18: "Europe", 19: "Europe", 20: "East Asia", 21: "Middle East",
    22: "Europe", 23: "Europe", 24: "South Asia", 25: "South East Asia",
}

def map_country_to_region(x):
    return country_region_map.get(x, "Other")

if "CountryIdentifier" in df.columns:
    df["CountryRegion"] = df["CountryIdentifier"].apply(map_country_to_region)
    df = df.drop(columns=["CountryIdentifier"], errors="ignore")

# Modern OS flag based on OsBuild median
if "OsBuild" in df.columns:
    median_osbuild = df["OsBuild"].median()
    df["IsModernOS"] = (df["OsBuild"] >= median_osbuild).astype(int)
else:
    median_osbuild = None  # fallback if needed

# Basic security flags
if "SmartScreen" in df.columns:
    df["SmartScreenExists"] = (df["SmartScreen"].notna()).astype(int)
if "IsSxsPassiveMode" in df.columns:
    df["IsPassiveMode"] = (df["IsSxsPassiveMode"] == 1).astype(int)

# Group rare categories for some high-cardinality columns
def group_rare(series, threshold=0.005):
    freq = series.value_counts(normalize=True)
    rare_categories = freq[freq < threshold].index
    return series.apply(lambda x: "Other" if x in rare_categories else x)

high_card_cols = ["ProductName", "SmartScreen", "Census_OSBranch"]
for col in high_card_cols:
    if col in df.columns and df[col].dtype == "object":
        df[col] = group_rare(df[col])

# Save common ProductName values for test preprocessing
if "ProductName" in df.columns:
    common_values_ProductName = df["ProductName"].value_counts().head(10).index.tolist()
else:
    common_values_ProductName = []

# ==============================
# STEP 4 — Define numeric and categorical columns
# ==============================

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "HasDetections"]

categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
# Do not encode MachineIdentifier (keep it as ID only)
categorical_cols = [c for c in categorical_cols if c != "MachineIdentifier"]

print("Numeric columns:", len(numeric_cols))
print("Categorical columns:", len(categorical_cols))

# ==============================
# STEP 5 — Impute missing values in training data
# ==============================

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

print("Total remaining missing in df:", df.isnull().sum().sum())

# ==============================
# STEP 6 — Ordinal encode categorical variables
# ==============================

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)

df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

print("Categorical variables encoded.")

# ==============================
# STEP 7 — Train/test split for model evaluation
# ==============================

X = df.drop(columns=["HasDetections", "MachineIdentifier"])
y = df["HasDetections"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape, y_train.shape)
print("Validation shape:", X_val.shape, y_val.shape)

# ==============================
# STEP 8 — Model evaluation helper
# ==============================

def evaluate_model(name, model):
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)[:, 1]
        roc = roc_auc_score(y_val, y_proba)
    else:
        roc = np.nan

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": roc,
    }

# ==============================
# STEP 9 — Train baseline models
# ==============================

results = []

logreg = LogisticRegression(max_iter=200, n_jobs=-1)
results.append(evaluate_model("Logistic Regression", logreg))

tree = DecisionTreeClassifier(max_depth=12, random_state=42)
results.append(evaluate_model("Decision Tree", tree))

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
)
results.append(evaluate_model("Random Forest", rf))

xgb_best = XGBClassifier(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0,
    min_child_weight=1,
    reg_alpha=0.0,
    reg_lambda=1.0,
    eval_metric="auc",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)
results.append(evaluate_model("XGBoost (Tuned)", xgb_best))

print("\nSummary of model results:")
for r in results:
    print(r)

# ==============================
# STEP 10 — Retrain tuned XGBoost on full training data
# ==============================

X_full = df.drop(columns=["HasDetections", "MachineIdentifier"])
y_full = df["HasDetections"]

xgb_best.fit(X_full, y_full)

print("Tuned XGBoost retrained on full training sample.")

# ==============================
# STEP 11 — Load test data and save IDs
# ==============================

test_df = pd.read_csv(TEST_PATH)
print("Raw test shape:", test_df.shape)

# Save original MachineIdentifier as ID column
test_ids = test_df["MachineIdentifier"].copy()

# ==============================
# STEP 12 — Apply same cleaning / features to test data
# ==============================

# Drop same high-missing columns
test_df = test_df.drop(columns=high_missing_cols, errors="ignore")

# Version majors
if "EngineVersion" in test_df.columns:
    test_df["EngineVersion_Major"] = test_df["EngineVersion"].apply(extract_major)
if "AppVersion" in test_df.columns:
    test_df["AppVersion_Major"] = test_df["AppVersion"].apply(extract_major)
if "AvSigVersion" in test_df.columns:
    test_df["AvSigVersion_Major"] = test_df["AvSigVersion"].apply(extract_major)

# CountryIdentifier → region
if "CountryIdentifier" in test_df.columns:
    test_df["CountryRegion"] = test_df["CountryIdentifier"].apply(map_country_to_region)
    test_df = test_df.drop(columns=["CountryIdentifier"], errors="ignore")

# Modern OS flag (use same threshold if available)
if "OsBuild" in test_df.columns and median_osbuild is not None:
    test_df["IsModernOS"] = (test_df["OsBuild"] >= median_osbuild).astype(int)

# Basic security flags
if "SmartScreen" in test_df.columns:
    test_df["SmartScreenExists"] = (test_df["SmartScreen"].notna()).astype(int)
if "IsSxsPassiveMode" in test_df.columns:
    test_df["IsPassiveMode"] = (test_df["IsSxsPassiveMode"] == 1).astype(int)

# ProductName rare handling using training top values
if "ProductName" in test_df.columns and common_values_ProductName:
    test_df["ProductName"] = test_df["ProductName"].apply(
        lambda x: x if x in common_values_ProductName else "Other"
    )

# ==============================
# STEP 13 — Impute and encode test data
# ==============================

# Only keep numeric/categorical columns that exist in test_df
numeric_cols_test = [c for c in numeric_cols if c in test_df.columns]
categorical_cols_test = [c for c in categorical_cols if c in test_df.columns]

# Impute with training medians / "Unknown"
test_df[numeric_cols_test] = test_df[numeric_cols_test].fillna(
    df[numeric_cols_test].median()
)
test_df[categorical_cols_test] = test_df[categorical_cols_test].fillna("Unknown")

# Apply same encoder
test_df[categorical_cols_test] = encoder.transform(test_df[categorical_cols_test])

# ==============================
# STEP 14 — Align test features with training features and predict
# ==============================

# Drop ID column from feature matrix
X_test_full = test_df.drop(columns=["MachineIdentifier"], errors="ignore")

# Align columns
missing_cols_test = [c for c in X_full.columns if c not in X_test_full.columns]
extra_cols_test = [c for c in X_test_full.columns if c not in X_full.columns]

for col in missing_cols_test:
    X_test_full[col] = 0

X_test_full = X_test_full.drop(columns=extra_cols_test, errors="ignore")
X_test_full = X_test_full[X_full.columns]

print("Aligned test feature shape:", X_test_full.shape)

# Predict probabilities for HasDetections
test_preds = xgb_best.predict_proba(X_test_full)[:, 1]

# ==============================
# STEP 15 — Build submission.csv
# ==============================

submission = pd.DataFrame({
    "MachineIdentifier": test_ids,
    "HasDetections": test_preds,
})

submission.to_csv("submission.csv", index=False)
print("submission.csv created successfully.")


