# ============================================================
# #AutoML #AutoGluon #TabularData #BinaryClassification
# #Kaggle #Ensembling #Stacking #ModelSelection #MachineLearning
# ============================================================

# If running on Kaggle: you can keep this install line
!pip -q install -U "autogluon.tabular>=1.1.0"

import os, glob
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

# -------------------------
# Columns (as you specified)
# -------------------------
ID_COL = "id"
TARGET = "diagnosed_diabetes"

FEATURES = [
    "age",
    "alcohol_consumption_per_week",
    "physical_activity_minutes_per_week",
    "diet_score",
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "bmi",
    "waist_to_hip_ratio",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "cholesterol_total",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status",
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history",
]

CATEGORICAL = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status",
    "family_history_diabetes",
    "hypertension_history",
    "cardiovascular_history",
]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

# -------------------------
# Helper: auto-find train/test
# -------------------------
def find_csv_by_name(root="/kaggle/input", name="train.csv"):
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    return hits[0] if hits else None

train_path = find_csv_by_name(name="train.csv")
test_path  = find_csv_by_name(name="test.csv")

# If your dataset uses different filenames, set them manually:
train_path = "/kaggle/input/playground-series-s5e12/train.csv"
test_path  = "/kaggle/input/playground-series-s5e12/test.csv"

print("train_path:", train_path)
print("test_path :", test_path)

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

# -------------------------
# Basic sanity checks
# -------------------------
needed_train = [ID_COL, TARGET] + FEATURES
missing_train = [c for c in needed_train if c not in train.columns]
if missing_train:
    raise ValueError(f"Missing columns in train: {missing_train}")

needed_test = [ID_COL] + FEATURES
missing_test = [c for c in needed_test if c not in test.columns]
if missing_test:
    raise ValueError(f"Missing columns in test: {missing_test}")

# -------------------------
# Type casting (helps AutoML)
# -------------------------
def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # categoricals to "category"
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")

    # numerics to float (coerce errors -> NaN)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

train = cast_types(train)
test  = cast_types(test)

# Ensure target is 0/1 int (if it isn't already)
train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce").fillna(0).astype(int)

# Use only required columns (prevents leakage from extra cols)
train_ml = train[[ID_COL, TARGET] + FEATURES].copy()
test_ml  = test[[ID_COL] + FEATURES].copy()

# -------------------------
# AutoGluon Tabular AutoML
# -------------------------
# Notes:
# - If metric for the competition is unknown, ROC AUC is a solid default.
# - If leaderboard uses logloss, you can switch eval_metric="log_loss".
predictor = TabularPredictor(
    label=TARGET,
    eval_metric="roc_auc",
    path="ag_diabetes_autml",
    verbosity=2
)

predictor.fit(
    train_data=train_ml.drop(columns=[ID_COL]),
    presets="best_quality",   # strong ensemble (bagging + stacking), slower but usually best
    time_limit=60*60,         # 1 hour; adjust as needed
    num_bag_folds=5,
    num_stack_levels=1
)

# -------------------------
# Predict probabilities
# -------------------------
proba = predictor.predict_proba(test_ml.drop(columns=[ID_COL]))

# AutoGluon returns a DataFrame for binary classification with 2 columns (class labels).
# We want probability of class "1" (positive).
if isinstance(proba, pd.DataFrame):
    # pick column 1 if it exists, else take the "largest" label column
    if 1 in proba.columns:
        pred = proba[1].to_numpy()
    else:
        # fallback: choose the column that corresponds to positive class
        # (often '1', 'True', or the max label)
        col = sorted(proba.columns)[-1]
        pred = proba[col].to_numpy()
else:
    # fallback if returned as series/array
    pred = np.asarray(proba)

# -------------------------
# Save submission
# -------------------------
sub = pd.DataFrame({ID_COL: test_ml[ID_COL].values, TARGET: pred})
sub.to_csv("submission.csv", index=False)

print("✅ saved submission.csv")
print(sub.head())

# Optional: show leaderboard of models
lb = predictor.leaderboard(silent=True)
print("\n=== AutoGluon Leaderboard (top) ===")
print(lb.head(15))

