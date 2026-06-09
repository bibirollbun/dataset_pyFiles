# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Install YDF (Yggdrasil Decision Forests)
!pip install ydf -U -q

import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ydf  # The star of the show

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Global Config
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

SEED = 42
N_SPLITS = 5
TARGET = "diagnosed_diabetes"
ID_COL = "id"

# Set seeds for reproducibility
np.random.seed(SEED)

print(f"YDF Version: {ydf.__version__}")
print("Configuration set.")
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Load Data

train_path = "/kaggle/input/playground-series-s5e12/train.csv"
test_path = "/kaggle/input/playground-series-s5e12/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")


# Preprocessing: Handle "Zero" as Missing
# In diabetes data, 0 for BMI, Glucose, BP, etc. is physically impossible 
# and indicates missing data. We mark them as NaN so YDF handles them natively.

physio_cols = [
    "glucose", "blood_pressure", "skin_thickness", 
    "insulin", "bmi", "systolic_bp", "diastolic_bp",
    "triglycerides", "ldl_cholesterol", "hdl_cholesterol"
]

def clean_zeros(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            # Replace 0 with NaN
            df[c] = df[c].replace(0, np.nan)
    return df

print("Cleaning zeros in physiological columns...")
train = clean_zeros(train, physio_cols)
test = clean_zeros(test, physio_cols)


# Feature Engineering (Pruned)

def safe_div(a, b, eps=1e-6):
    return a / (b + eps)

def enhance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # --- Metabolic Ratios (High Value) ---
    if {"triglycerides", "hdl_cholesterol"}.issubset(df.columns):
        df["tg_hdl_ratio"] = safe_div(df["triglycerides"], df["hdl_cholesterol"])
        
    if {"ldl_cholesterol", "hdl_cholesterol"}.issubset(df.columns):
        df["ldl_hdl_ratio"] = safe_div(df["ldl_cholesterol"], df["hdl_cholesterol"])
        
    if {"cholesterol_total", "hdl_cholesterol"}.issubset(df.columns):
        df["non_hdl"] = df["cholesterol_total"] - df["hdl_cholesterol"]

    # --- Hemodynamics ---
    if {"systolic_bp", "diastolic_bp"}.issubset(df.columns):
        df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
        # Mean Arterial Pressure approximation
        df["map"] = df["diastolic_bp"] + (df["pulse_pressure"] / 3.0)

    # --- Body Composition Interaction ---
    if {"bmi", "waist_to_hip_ratio"}.issubset(df.columns):
        df["bmi_x_whr"] = df["bmi"] * df["waist_to_hip_ratio"]

    return df

print("Applying Feature Engineering...")
train_fe = enhance_features(train)
test_fe = enhance_features(test)

# Report added columns
new_cols = [c for c in train_fe.columns if c not in train.columns]
print(f"Added {len(new_cols)} features: {new_cols}")


# Prepare Data for YDF

# FIX: YDF requires Classification labels to be Integers (0, 1), not Floats (0.0, 1.0).
train_fe[TARGET] = train_fe[TARGET].astype(int)

X = train_fe.drop(columns=[ID_COL]) 
X_test = test_fe.drop(columns=[ID_COL])
test_ids = test[ID_COL]

print(f"Total features available: {X.shape[1] - 1}")


# Model Training: YDF with CV


kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_scores = []

print("\n========== YDF Training ==========")

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, X[TARGET]), 1):
    
    # Create Fold DataFrames
    train_fold = X.iloc[tr_idx].copy()
    valid_fold = X.iloc[va_idx].copy()
    
    # Define Learner
    learner = ydf.GradientBoostedTreesLearner(
        label=TARGET,
        task=ydf.Task.CLASSIFICATION,
        
        # Hyperparameters
        num_trees=1000,
        growing_strategy="BEST_FIRST_GLOBAL",
        early_stopping_num_trees_look_ahead=50,
    )
    
    # Train (Now strict-type safe)
    model = learner.train(train_fold, valid=valid_fold, verbose=0)
    
    # Predict
    val_pred = model.predict(valid_fold)
    test_pred = model.predict(X_test)
    
    # Store Predictions
    oof_preds[va_idx] = val_pred
    test_preds += test_pred / N_SPLITS
    
    # Evaluate
    auc = roc_auc_score(valid_fold[TARGET], val_pred)
    fold_scores.append(auc)
    print(f"Fold {fold} AUC: {auc:.5f}")

    # Feature Importance (Fold 1 only)
# --- Feature Importance (Fold 1 only) ---
    if fold == 1:
        print("\n--- YDF Variable Importance (Top 10) ---")
        imp = model.variable_importances()
        
        if 'SUM_SCORE' in imp:
            # We iterate by item to handle different ydf version return types safely
            for i, item in enumerate(imp['SUM_SCORE'][:10]):
                
                # Case 1: Item is a tuple (importance, name) -> older versions sometimes flip this
                if isinstance(item, tuple):
                    val1, val2 = item
                    # Check which one is the string (name) and which is the float (score)
                    if isinstance(val1, str):
                        print(f"{i+1}. {val1:<25} : {val2:.4f}")
                    else:
                        print(f"{i+1}. {val2:<25} : {val1:.4f}")
                        
                # Case 2: Item is an object (newer ydf versions)
                elif hasattr(item, "variable") and hasattr(item, "importance"):
                    # item.variable might be an object, get its name
                    name = item.variable.name if hasattr(item.variable, "name") else str(item.variable)
                    print(f"{i+1}. {name:<25} : {item.importance:.4f}")
                
                # Case 3: Fallback
                else:
                    print(f"{i+1}. {item}")
                    
        print("----------------------------------------\n")


# Results & Submission


cv_score = np.mean(fold_scores)
print(f"\nAverage CV AUC: {cv_score:.5f}")

sub = pd.DataFrame({
    ID_COL: test_ids,
    TARGET: test_preds
})

sub.to_csv("submission_y.csv", index=False)
print("Saved submission_y.csv")
display(sub.head())

