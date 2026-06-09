# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings("ignore")

import gc

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.feature_selection import SelectFromModel

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


INPUT_DIR = "/kaggle/input/playground-series-s5e12"

TARGET = "diagnosed_diabetes"
ID_COL = "id"  # name of the ID column in train/test

# Competition data
train = pd.read_csv(f"{INPUT_DIR}/train.csv")
test = pd.read_csv(f"{INPUT_DIR}/test.csv")
submission = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")

print("Shapes:")
print(f"  train: {train.shape}")
print(f"  test: {test.shape}")
print(f"Target column: {TARGET}")



# All columns except ID and target
base_cols = [c for c in train.columns if c not in [ID_COL, TARGET]]

print(f"Number of base features: {len(base_cols)}")
print("Example base features:", base_cols[:10])


# Clinical feature engineering


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a few simple, clinically meaningful features:
    - bmi_cat: BMI category (underweight, normal, overweight, obese)
    - bp_cat: blood pressure category based on systolic/diastolic BP
    - non_hdl: non-HDL cholesterol (total - HDL)
    """
    df = df.copy()
    
    # 1) BMI categories: 0=underweight, 1=normal, 2=overweight, 3=obese
    if "bmi" in df.columns:
        df["bmi_cat"] = pd.cut(
            df["bmi"],
            bins=[0, 18.5, 25, 30, 999],
            labels=[0, 1, 2, 3],
            include_lowest=True
        ).astype("int64")
    
    # 2) Blood pressure categories: 0=normal, 1=elevated/pre-hypertension, 2=high
    if {"systolic_bp", "diastolic_bp"}.issubset(df.columns):
        df["bp_cat"] = 0
        df.loc[
            (df["systolic_bp"] >= 140) | (df["diastolic_bp"] >= 90),
            "bp_cat"
        ] = 2
        df.loc[
            (
                (df["systolic_bp"] >= 120) & (df["systolic_bp"] < 140)
            ) | (
                (df["diastolic_bp"] >= 80) & (df["diastolic_bp"] < 90)
            ),
            "bp_cat"
        ] = 1
        df["bp_cat"] = df["bp_cat"].astype("int64")
    
    # 3) Non-HDL cholesterol: total cholesterol - HDL
    if {"cholesterol_total", "hdl_cholesterol"}.issubset(df.columns):
        df["non_hdl"] = df["cholesterol_total"] - df["hdl_cholesterol"]
    
    return df

# Apply to train and test
train_fe = add_clinical_features(train)
test_fe = add_clinical_features(test)

print("Train columns after feature engineering:", len(train_fe.columns))
print("Test columns after feature engineering:", len(test_fe.columns))



# Final feature matrices

extra_features = ["bmi_cat", "bp_cat", "non_hdl"]

# Some of these might not exist if the source columns were missing, so keep only those that are present
extra_features = [f for f in extra_features if f in train_fe.columns]

features = base_cols + extra_features

print(f"Total number of features used: {len(features)}")

X = train_fe[features].copy()
y = train_fe[TARGET].copy()
X_test = test_fe[features].copy()

print("X shape:", X.shape)
print("X_test shape:", X_test.shape)



# Handling missing values

# Fill numeric NaNs with median of training data
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

for col in num_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)



# Label encode only object (string) columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
print("Categorical (object) columns:", cat_cols)

for col in cat_cols:
    le = LabelEncoder()
    # fit on train
    X[col] = le.fit_transform(X[col].astype(str))
    # transform test with same mapping
    X_test[col] = le.transform(X_test[col].astype(str))



# Cross-validated XGBoost model

n_splits = 5  # you can change to 10 if your runtime is OK
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_pred = np.zeros(len(X))        # out-of-fold predictions for train
test_pred = np.zeros(len(X_test))  # averaged predictions for test

fold_auc_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    print(f"\n===== Fold {fold}/{n_splits} =====")
    
    X_trn, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_trn, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=2.0,       # L1 regularization
        reg_lambda=3.0,      # L2 regularization
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1            # use all CPU cores
    )
    
    model.fit(
        X_trn, y_trn,
        eval_set=[(X_trn, y_trn), (X_val, y_val)],
        verbose=100,
        early_stopping_rounds=200
    )
    
    # Best iteration chosen by early stopping
    best_iter = model.best_iteration
    print(f"Best iteration: {best_iter}")
    
    # OOF predictions for validation fold
    val_proba = model.predict_proba(X_val)[:, 1]
    oof_pred[val_idx] = val_proba
    
    # AUC on this fold
    fold_auc = roc_auc_score(y_val, val_proba)
    fold_auc_scores.append(fold_auc)
    print(f"Fold {fold} AUC: {fold_auc:.6f}")
    
    # Test predictions (average over folds)
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred += test_proba / n_splits  # accumulate average

# Overall CV AUC using all OOF predictions
cv_auc = roc_auc_score(y, oof_pred)
print("\n===== Cross-validation summary =====")
print("Fold AUCs:", [round(s, 6) for s in fold_auc_scores])
print(f"Mean AUC: {np.mean(fold_auc_scores):.6f}  |  Std: {np.std(fold_auc_scores):.6f}")
print(f"OOF CV AUC (all folds combined): {cv_auc:.6f}")



# submission file

# Attach the predictions to the submission template
submission[TARGET] = test_pred

# Name your output file (you can change this name)
OUTPUT_PATH = "submission_xgb_cv.csv"

# Save to CSV
submission.to_csv(OUTPUT_PATH, index=False)

print(f"Submission file saved as: {OUTPUT_PATH}")
print(f"Mean predicted probability: {submission[TARGET].mean():.5f}")
submission.head()



# Feature selection + CV model on selected features

from sklearn.feature_selection import SelectFromModel

# 8.1 Train a model on all data to get feature importance
fs_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.02,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=2.0,
    reg_lambda=3.0,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

fs_model.fit(X, y)

# 8.2 Use SelectFromModel to keep only important features
selector = SelectFromModel(
    fs_model,
    threshold="median",   # keep features with importance above the median
    prefit=True           # model is already fitted
)

X_fs = selector.transform(X)
X_test_fs = selector.transform(X_test)

# Get the names of selected features
feature_mask = selector.get_support()
selected_features = [f for f, keep in zip(features, feature_mask) if keep]

print(f"Original number of features: {len(features)}")
print(f"Selected number of features: {X_fs.shape[1]}")
print("Some selected features:", selected_features[:15])

# 8.3 Re-run CV with the reduced feature set
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_pred_fs = np.zeros(len(X_fs))
test_pred_fs = np.zeros(len(X_test_fs))
fold_auc_scores_fs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_fs, y), start=1):
    print(f"\n===== [FS] Fold {fold}/{n_splits} =====")
    
    X_trn, X_val = X_fs[train_idx], X_fs[val_idx]
    y_trn, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model_fs = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=2.0,
        reg_lambda=3.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )
    
    model_fs.fit(
        X_trn, y_trn,
        eval_set=[(X_trn, y_trn), (X_val, y_val)],
        verbose=100,
        early_stopping_rounds=200
    )
    
    val_proba = model_fs.predict_proba(X_val)[:, 1]
    oof_pred_fs[val_idx] = val_proba
    
    fold_auc = roc_auc_score(y_val, val_proba)
    fold_auc_scores_fs.append(fold_auc)
    print(f"[FS] Fold {fold} AUC: {fold_auc:.6f}")
    
    test_proba = model_fs.predict_proba(X_test_fs)[:, 1]
    test_pred_fs += test_proba / n_splits

# 8.4 Overall CV performance with feature selection
cv_auc_fs = roc_auc_score(y, oof_pred_fs)
print("\n===== [FS] Cross-validation summary =====")
print("Fold AUCs (FS):", [round(s, 6) for s in fold_auc_scores_fs])
print(f"Mean AUC (FS): {np.mean(fold_auc_scores_fs):.6f}  |  Std: {np.std(fold_auc_scores_fs):.6f}")
print(f"OOF CV AUC with feature selection: {cv_auc_fs:.6f}")



# =========================
# Part 9: Probability calibration (Isotonic Regression)
# =========================

from sklearn.isotonic import IsotonicRegression

# 9.1 Fit calibrator on OOF predictions vs true labels
calibrator = IsotonicRegression(out_of_bounds="clip")

calibrator.fit(oof_pred_fs, y)

# 9.2 Apply calibration to test predictions
test_pred_cal = calibrator.transform(test_pred_fs)

print("Before calibration:")
print(f"  Mean raw test prediction: {test_pred_fs.mean():.5f}")

print("After calibration:")
print(f"  Mean calibrated test prediction: {test_pred_cal.mean():.5f}")



# Optional: Save calibrated predictions as a new submission

submission_cal = submission.copy()
submission_cal[TARGET] = test_pred_cal

OUTPUT_PATH_CAL = "submission_xgb_fs_calibrated.csv"
submission_cal.to_csv(OUTPUT_PATH_CAL, index=False)

print(f"Calibrated submission file saved as: {OUTPUT_PATH_CAL}")
print(f"Mean calibrated prediction: {submission_cal[TARGET].mean():.5f}")
submission_cal.head()





