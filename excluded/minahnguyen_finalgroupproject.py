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


# STEP 1 â€” Load a sampled version of train.csv using chunk loading (Kaggle path)

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

print("\nâœ… Finished sampling!")
print("Sample shape:", df.shape)
df.head()



# STEP 2 â€” Basic data overview

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



# STEP 3 â€” Missing values summary

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



# STEP 4 â€” Identify likely missingness types (MCAR, MAR, MNAR/NI)
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



# STEP 5 â€” Drop columns with extreme missingness (>90%)

high_missing_cols = missing[missing > 90].index.tolist()
print("Dropping columns with >90% missing values:", high_missing_cols)

df = df.drop(columns=high_missing_cols, errors="ignore")

print("\nNew shape after dropping high-missing columns:", df.shape)



# STEP 6 â€” Separate numerical and categorical features

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

print("Numeric columns:", len(numeric_cols))
print("Categorical columns:", len(categorical_cols))

# Make sure target stays numeric
numeric_cols = [col for col in numeric_cols if col != "HasDetections"]



# STEP 7 â€” Impute missing values

# Numeric: median fill
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Categorical: "Unknown" fill
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# Verify no missing values left
print("Remaining missing values:", df.isnull().sum().sum())



# STEP 8 â€” Feature Engineering: Extract major/minor versions

def extract_major(x):
    try:
        return int(str(x).split('.')[0])
    except:
        return -1

df["EngineVersion_Major"] = df["EngineVersion"].apply(extract_major)
df["AppVersion_Major"] = df["AppVersion"].apply(extract_major)
df["AvSigVersion_Major"] = df["AvSigVersion"].apply(extract_major)

print("Version-based features created.")
df[["EngineVersion", "EngineVersion_Major"]].head()



# STEP 8b â€” Feature Engineering: CountryIdentifier Region Grouping

# From Microsoft Malware Prediction data dictionary
country_region_map = {
    1: "North America", 2: "Europe", 3: "Europe", 4: "South Asia",
    5: "Europe", 6: "South America", 7: "Europe", 8: "Middle East",
    9: "Europe", 10: "Europe", 11: "Oceania", 12: "Europe", 13: "East Asia",
    14: "Africa", 15: "Europe", 16: "Africa", 17: "East Asia",
    18: "Europe", 19: "Europe", 20: "East Asia", 21: "Middle East",
    22: "Europe", 23: "Europe", 24: "South Asia", 25: "South East Asia",
    # unknown/rare/unassigned countries
}

def map_country_to_region(x):
    return country_region_map.get(x, "Other")

if "CountryIdentifier" in df.columns:
    df["CountryRegion"] = df["CountryIdentifier"].apply(map_country_to_region)
    print("Created CountryRegion grouped feature.")
else:
    print("CountryIdentifier not found in df.columns")



df = df.drop(columns=["CountryIdentifier"], errors="ignore")


# STEP 9 â€” Feature: Modern OS indicator using OsBuild

median_osbuild = df["OsBuild"].median()
df["IsModernOS"] = (df["OsBuild"] >= median_osbuild).astype(int)

print("IsModernOS feature created.")
df[["OsBuild", "IsModernOS"]].head()



# STEP 10 â€” Feature: Basic security-level indicators

df["SmartScreenExists"] = (df["SmartScreen"] != "Unknown").astype(int)
df["IsPassiveMode"] = (df["IsSxsPassiveMode"] == 1).astype(int)

print("Security features created.")
df[["SmartScreen", "SmartScreenExists", "IsSxsPassiveMode", "IsPassiveMode"]].head()



# STEP 11 â€” Rare category grouping for high-cardinality categorical features

def group_rare(series, threshold=0.005):
    freq = series.value_counts(normalize=True)
    rare_categories = freq[freq < threshold].index
    return series.apply(lambda x: "Other" if x in rare_categories else x)

high_card_cols = ["ProductName", "SmartScreen", "Census_OSBranch"]

for col in high_card_cols:
    if col in df.columns:
        df[col] = group_rare(df[col])
        print(f"Grouped rare categories in: {col}")



# STEP 12 â€” Encode categorical variables (Ordinal Encoding)

from sklearn.preprocessing import OrdinalEncoder

# Identify categorical columns again (after rare grouping)
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

print("Categorical variables encoded.")



# STEP 13 â€” Train/test split

X = df.drop("HasDetections", axis=1)
y = df["HasDetections"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape, y_train.shape)
print("Testing shape :", X_test.shape, y_test.shape)



# STEP 14 â€” Import models + define evaluation helper

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def evaluate_model(name, model):
    """
    Train model, make predictions, compute metrics.
    """
    print(f"\n===== {name} =====")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ROC-AUC (only if model has predict_proba)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
    else:
        roc = np.nan
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
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
        "ROC_AUC": roc
    }



# STEP 15 â€” Build and evaluate all models

results = []

# 1. Logistic Regression
logreg = LogisticRegression(max_iter=200, n_jobs=-1)
results.append(evaluate_model("Logistic Regression", logreg))

# 2. Decision Tree
tree = DecisionTreeClassifier(max_depth=12, random_state=42)
results.append(evaluate_model("Decision Tree", tree))

# 3. Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
results.append(evaluate_model("Random Forest", rf))

# 4. Baseline XGBoost
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)
results.append(evaluate_model("XGBoost", xgb))

# 5. Tuned XGBoost
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
    random_state=42
)
results.append(evaluate_model("XGBoost (Tuned)", xgb_best))

# Display all results
results



# STEP 16 â€” Confusion Matrix for Tuned XGBoost

from sklearn.metrics import ConfusionMatrixDisplay

# Get predictions for confusion matrix
y_pred_best = xgb_best.predict(X_test)

plt.figure(figsize=(6,5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_best, cmap="Blues", values_format='d')
plt.title("Confusion Matrix â€” XGBoost (Tuned)")
plt.show()



# STEP 17 â€” ROC Curve Comparison for All Models

from sklearn.metrics import RocCurveDisplay

plt.figure(figsize=(8,6))

models_for_plot = [
    ("Logistic Regression", logreg),
    ("Decision Tree", tree),
    ("Random Forest", rf),
    ("XGBoost", xgb),
    ("XGBoost (Tuned)", xgb_best)
]

for name, model in models_for_plot:
    y_proba = model.predict_proba(X_test)[:, 1]
    RocCurveDisplay.from_predictions(y_test, y_proba, name=name)

plt.title("ROC Curve Comparison â€” All Models")
plt.plot([0,1], [0,1], "k--")  # diagonal baseline
plt.grid(True)
plt.show()



# ============================================================
# FIX â€” Define common ProductName categories from training data
# ============================================================

common_values_ProductName = df["ProductName"].value_counts().head(10).index.tolist()
print("Common ProductName categories:", common_values_ProductName)



# ============================================================
# STEP A â€” Load Kaggle Test Data
# ============================================================

test_df = pd.read_csv("/kaggle/input/microsoft-malware-prediction/test.csv")
print("Test shape:", test_df.shape)



# ============================================================
# STEP B â€” Apply SAME cleaning + feature engineering to test_df
# ============================================================

# 1. Drop the same high-missing columns
test_df = test_df.drop(columns=high_missing_cols, errors="ignore")


# 2. Extract version major components
def extract_major_version(series):
    return series.astype(str).str.split(".", expand=True)[0].astype(float)

for col in ["EngineVersion", "AppVersion", "AvSigVersion"]:
    if col in test_df.columns:
        test_df[col + "_Major"] = extract_major_version(test_df[col])


# 3. Modern OS indicator
if "OsBuild" in test_df.columns:
    test_df["IsModernOS"] = (test_df["OsBuild"] >= 15000).astype(int)


# 4. SmartScreen flag
if "SmartScreen" in test_df.columns:
    test_df["SmartScreenExists"] = test_df["SmartScreen"].notnull().astype(int)


# 5. Passive mode flag
if "IsSxsPassiveMode" in test_df.columns:
    test_df["IsPassiveMode"] = (test_df["IsSxsPassiveMode"] == 1).astype(int)


# 6. Rare category grouping for ProductName
if "ProductName" in test_df.columns:
    test_df["ProductName"] = test_df["ProductName"].apply(
        lambda x: x if x in common_values_ProductName else "Other"
    )


# 7. CountryIdentifier â†’ CountryRegion
def map_country_to_region(x):
    return country_region_map.get(x, "Other")

if "CountryIdentifier" in test_df.columns:
    test_df["CountryRegion"] = test_df["CountryIdentifier"].apply(map_country_to_region)
    test_df = test_df.drop(columns=["CountryIdentifier"], errors="ignore")



# ============================================================
# FIX â€” Ensure numeric_cols & categorical_cols only include columns in test_df
# ============================================================

numeric_cols = [col for col in numeric_cols if col in test_df.columns]
categorical_cols = [col for col in categorical_cols if col in test_df.columns]



# ============================================================
# STEP C â€” Impute missing values
# ============================================================

test_df[numeric_cols] = test_df[numeric_cols].fillna(df[numeric_cols].median())
test_df[categorical_cols] = test_df[categorical_cols].fillna("Unknown")



# ============================================================
# STEP D â€” Apply SAME OrdinalEncoder as training
# ============================================================

test_df[categorical_cols] = encoder.transform(test_df[categorical_cols])



# ============================================================
# STEP E â€” Retrain tuned XGBoost on FULL cleaned training data
# ============================================================

X_full = df.drop(columns=["HasDetections"])
y_full = df["HasDetections"]

xgb_best.fit(X_full, y_full)



# ============================================================
# STEP F â€” Predict on Kaggle test set
# ============================================================
# ALIGN COLUMNS: âœ¨ THE MAGIC FIX âœ¨
missing_cols = [c for c in X_full.columns if c not in test_df.columns]
extra_cols   = [c for c in test_df.columns if c not in X_full.columns]

for col in missing_cols:
    test_df[col] = 0

test_df = test_df.drop(columns=extra_cols, errors='ignore')

test_df = test_df[X_full.columns]

# Now safe to predict
test_preds = xgb_best.predict_proba(test_df)[:, 1]

test_preds = xgb_best.predict_proba(test_df)[:, 1]



# ============================================================
# STEP G â€” Build submission.csv (FINAL OUTPUT)
# ============================================================

submission = pd.DataFrame({
    "MachineIdentifier": test_df["MachineIdentifier"],
    "HasDetections": test_preds
})

submission.to_csv("submission.csv", index=False)
print("ğŸ�‰ submission.csv created successfully! You may now download and submit it on Kaggle.")


