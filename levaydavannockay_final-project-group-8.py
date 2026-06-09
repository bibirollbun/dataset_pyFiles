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


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")


# load sample dataset
DATA_PATH = "/kaggle/input/microsoft-malware-prediction/train.csv"

chunksize = 500_000
sample_frac = 0.10
random_state = 42

dfs = []
row_count = 0

for chunk in pd.read_csv(DATA_PATH, chunksize=chunksize):
    row_count += len(chunk)
    chunk_sample = chunk.sample(frac=sample_frac, random_state=random_state)
    dfs.append(chunk_sample)
    print(f"Processed {row_count:,} rows so far...")

df = pd.concat(dfs, ignore_index=True)
print("Sample shape:", df.shape)
df.head()


# basic overview
print("Shape:", df.shape)
print("\nColumn types:\n", df.dtypes.value_counts())

print("\nTarget distribution:")
print(df["HasDetections"].value_counts(normalize=True))

sns.countplot(x="HasDetections", data=df)
plt.title("Target distribution")
plt.show()


# missing values
missing = df.isnull().mean().sort_values(ascending=False) * 100
missing_df = missing.reset_index()
missing_df.columns = ['Feature', 'MissingPercent']

print(missing_df.head(20))

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


# dropping high missing columns
high_missing_cols = missing[missing > 90].index.tolist()
print("Dropping columns with >90% missing values:", high_missing_cols)
df = df.drop(columns=high_missing_cols, errors="ignore")
print("New shape after dropping:", df.shape)


# seperate numerical & categorical

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != "HasDetections"]

print("Numeric cols:", len(numeric_cols))
print("Categorical cols:", len(categorical_cols))


# impute missing values

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[categorical_cols] = df[categorical_cols].fillna("Unknown")
print("Remaining missing values:", df.isnull().sum().sum())


# feature engineering

# extract major versions
def extract_major(x):
    try:
        return int(str(x).split('.')[0])
    except:
        return -1

for col in ["EngineVersion", "AppVersion", "AvSigVersion"]:
    if col in df.columns:
        df[col + "_Major"] = df[col].apply(extract_major)

# modern OS indicator
if "OsBuild" in df.columns:
    median_osbuild = df["OsBuild"].median()
    df["IsModernOS"] = (df["OsBuild"] >= median_osbuild).astype(int)

# security indicators
if "SmartScreen" in df.columns:
    df["SmartScreenExists"] = (df["SmartScreen"] != "Unknown").astype(int)
if "IsSxsPassiveMode" in df.columns:
    df["IsPassiveMode"] = (df["IsSxsPassiveMode"] == 1).astype(int)
def group_rare(series, threshold=0.005):
    freq = series.value_counts(normalize=True)
    rare_categories = freq[freq < threshold].index
    return series.apply(lambda x: "Other" if x in rare_categories else x)
high_card_cols = ["ProductName", "SmartScreen", "Census_OSBranch"]
for col in high_card_cols:
    if col in df.columns:
        df[col] = group_rare(df[col])


# encode categorical

categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df[categorical_cols] = encoder.fit_transform(df[categorical_cols])


# Train/Test Split

X = df.drop("HasDetections", axis=1)
y = df["HasDetections"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
print("Training shape:", X_train.shape)
print("Testing shape :", X_test.shape)


# evaluation

def evaluate_model(name, model):
    print(f"\n===== {name} =====")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:,1]
        roc = roc_auc_score(y_test, y_proba)
    else:
        roc = np.nan

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")

    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC_AUC": roc}


# Train & Evaluate Models
results = []

# Logistic Regression
results.append(evaluate_model("Logistic Regression", LogisticRegression(max_iter=200, n_jobs=-1)))

# Decision Tree
results.append(evaluate_model("Decision Tree", DecisionTreeClassifier(max_depth=12, random_state=42)))

# Random Forest
results.append(evaluate_model("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)))

# XGBoost
results.append(evaluate_model("XGBoost", XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, eval_metric="auc", tree_method="hist", n_jobs=-1, random_state=42)))

# Tuned XGBoost
results.append(evaluate_model("XGBoost (Tuned)", XGBClassifier(
    n_estimators=800, max_depth=6, learning_rate=0.05, subsample=0.9,
    colsample_bytree=0.9, gamma=0, min_child_weight=1,
    reg_alpha=0.0, reg_lambda=1.0, eval_metric="auc",
    tree_method="hist", n_jobs=-1, random_state=42
)))


# Display Results

results_df = pd.DataFrame(results)
results_df.sort_values(by="ROC_AUC", ascending=False)


# Define the tuned XGBoost model properly
best_model = XGBClassifier(
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

# Fit it on your training data
best_model.fit(X_train, y_train)


print("best_model is:", type(best_model))


# ============================
# 1. Load raw Kaggle test data
# ============================
test_df = pd.read_csv("/kaggle/input/microsoft-malware-prediction/test.csv")

# keep original IDs for submission
test_ids = test_df["MachineIdentifier"].copy()

# Drop very-high-missing columns (same as training)
test_df = test_df.drop(columns=high_missing_cols, errors="ignore")

# ==========================================
# 2. Handle missing values (same strategy)
# ==========================================

# Use only columns that exist in test_df
numeric_cols_test = [c for c in numeric_cols if c in test_df.columns]
categorical_cols_test = [c for c in categorical_cols if c in test_df.columns]

# Numerical: fill with TRAIN medians
test_df[numeric_cols_test] = test_df[numeric_cols_test].fillna(
    df[numeric_cols_test].median()
)

# Categorical: fill with "Unknown"
test_df[categorical_cols_test] = test_df[categorical_cols_test].fillna("Unknown")

# ============================
# 3. Feature engineering again
# ============================

def extract_major(x):
    try:
        return int(str(x).split(".")[0])
    except:
        return -1

for col in ["EngineVersion", "AppVersion", "AvSigVersion"]:
    if col in test_df.columns:
        test_df[col + "_Major"] = test_df[col].apply(extract_major)

if "OsBuild" in test_df.columns:
    median_osbuild = df["OsBuild"].median()  # use TRAIN median
    test_df["IsModernOS"] = (test_df["OsBuild"] >= median_osbuild).astype(int)

if "SmartScreen" in test_df.columns:
    test_df["SmartScreenExists"] = (test_df["SmartScreen"] != "Unknown").astype(int)

if "IsSxsPassiveMode" in test_df.columns:
    test_df["IsPassiveMode"] = (test_df["IsSxsPassiveMode"] == 1).astype(int)

# Group rare categories using TRAIN frequencies
def apply_train_rare_mapping(train_series, test_series, threshold=0.005):
    freq = train_series.value_counts(normalize=True)
    rare_cats = freq[freq < threshold].index
    return test_series.apply(lambda x: "Other" if x in rare_cats else x)

for col in high_card_cols:
    if col in test_df.columns and col in df.columns:
        test_df[col] = apply_train_rare_mapping(df[col], test_df[col])

# =========================================
# 4. Encode categoricals with SAME encoder
# =========================================

# Only encode columns the encoder was fit on and that exist in test_df
cat_for_encoder = [c for c in categorical_cols if c in test_df.columns]

test_df[cat_for_encoder] = encoder.transform(test_df[cat_for_encoder])

# ========================================
# 5. Align columns with training X matrix
# ========================================

# X was created as: X = df.drop("HasDetections", axis=1)
train_features = X.columns  # from earlier in your notebook

# Make sure test has all those columns (and in same order)
X_final_test = test_df[train_features]

# ============================
# 6. Predict & create submission
# ============================

final_predictions = best_model.predict_proba(X_final_test)[:, 1]

submission = pd.DataFrame({
    "MachineIdentifier": test_ids,       # <- original string IDs
    "HasDetections": final_predictions
})

submission.to_csv("submission.csv", index=False)
print("submission.csv saved!")


print(submission.shape)
print(submission.columns)
print(submission.head())

