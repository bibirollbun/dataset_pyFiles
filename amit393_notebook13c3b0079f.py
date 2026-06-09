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


# =========================
# Kaggle Playground Series S5E12 - Diabetes Prediction
# =========================

from sklearn.preprocessing import PowerTransformer, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# =========================
# 1. Load Data
# =========================
print("Loading data...")
try:
    df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
    df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
    sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")
except FileNotFoundError:
    print("Error: Data files not found. Please ensure you're running in Kaggle environment.")
    print("Or update paths to your local data location.")
    raise

# Drop ID column
test_ids = df_test['id'].copy()
df_train = df_train.drop(columns=['id'])
df_test = df_test.drop(columns=['id'])

# =========================
# 2. Feature Engineering
# =========================
print("\nCreating features...")
for df in [df_train, df_test]:
    # Interaction features
    df["BMI_Age"] = df["bmi"] * df["age"]
    df["Cholesterol_BMI"] = df["cholesterol_total"] / (df["bmi"] + 1)
    df["Triglycerides_HDL"] = df["triglycerides"] / (df["hdl_cholesterol"] + 1)
    df["LDL_HDL_Ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1)
    df["Pulse_Pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["Screen_Sleep_Ratio"] = df["screen_time_hours_per_day"] / (df["sleep_hours_per_day"] + 0.1)
    
    # Additional risk factors
    df["Total_Risk_Score"] = (
        df["hypertension_history"] + 
        df["cardiovascular_history"] + 
        df["family_history_diabetes"]
    )
    df["BP_Product"] = df["systolic_bp"] * df["diastolic_bp"]
    df["Cholesterol_Risk"] = df["ldl_cholesterol"] - df["hdl_cholesterol"]

print(f"Total features after engineering: {df_train.shape[1] - 1}")

# =========================
# 3. Yeo-Johnson Transform
# =========================
print("\nApplying power transformations...")
skewed_cols = [
    "alcohol_consumption_per_week",
    "physical_activity_minutes_per_week",
    "cholesterol_total",
    "triglycerides",
]

pt = PowerTransformer(method="yeo-johnson")
df_train[skewed_cols] = pt.fit_transform(df_train[skewed_cols])
df_test[skewed_cols] = pt.transform(df_test[skewed_cols])

# =========================
# 4. Prepare Data
# =========================
TARGET = "diagnosed_diabetes"
X = df_train.drop(columns=[TARGET])
y = df_train[TARGET]

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print(f"\nCategorical columns: {len(cat_cols)}")
print(f"Numerical columns: {len(num_cols)}")

# Encode categorical
if len(cat_cols) > 0:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[cat_cols] = enc.fit_transform(X[cat_cols])
    df_test[cat_cols] = enc.transform(df_test[cat_cols])

# =========================
# 5. Initialize Stratified K-Fold & Arrays
# =========================
N_FOLDS = 5
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_xgb = np.zeros(len(X))

pred_lgb = np.zeros(len(df_test))
pred_cat = np.zeros(len(df_test))
pred_xgb = np.zeros(len(df_test))

# =========================
# 6. Model Training with Cross-Validation
# =========================
print(f"\n{'='*50}")
print(f"Training models with {N_FOLDS}-fold cross-validation...")
print(f"{'='*50}")

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n===== FOLD {fold} / {N_FOLDS} =====")
    X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
    X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]
    
    # LightGBM
    print("Training LightGBM...")
    lgb = LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        num_leaves=64,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        class_weight="balanced",
        verbosity=-1
    )
    lgb.fit(X_train, y_train)
    oof_lgb[val_idx] = lgb.predict_proba(X_valid)[:, 1]
    pred_lgb += lgb.predict_proba(df_test)[:, 1] / N_FOLDS
    lgb_score = roc_auc_score(y_valid, oof_lgb[val_idx])
    print(f"  LightGBM Fold {fold} ROC-AUC: {lgb_score:.5f}")
    
    # CatBoost
    print("Training CatBoost...")
    cat = CatBoostClassifier(
        iterations=1200,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False
    )
    cat.fit(X_train, y_train)
    oof_cat[val_idx] = cat.predict_proba(X_valid)[:, 1]
    pred_cat += cat.predict_proba(df_test)[:, 1] / N_FOLDS
    cat_score = roc_auc_score(y_valid, oof_cat[val_idx])
    print(f"  CatBoost Fold {fold} ROC-AUC: {cat_score:.5f}")
    
    # XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
        tree_method="hist",
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    oof_xgb[val_idx] = xgb.predict_proba(X_valid)[:, 1]
    pred_xgb += xgb.predict_proba(df_test)[:, 1] / N_FOLDS
    xgb_score = roc_auc_score(y_valid, oof_xgb[val_idx])
    print(f"  XGBoost Fold {fold} ROC-AUC: {xgb_score:.5f}")

# =========================
# 7. Model Performance Summary
# =========================
print(f"\n{'='*50}")
print("OUT-OF-FOLD VALIDATION SCORES")
print(f"{'='*50}")

model_scores = {
    "LightGBM": roc_auc_score(y, oof_lgb),
    "CatBoost": roc_auc_score(y, oof_cat),
    "XGBoost": roc_auc_score(y, oof_xgb),
}

scores_df = pd.DataFrame.from_dict(model_scores, orient="index", columns=["ROC-AUC"])
scores_df = scores_df.sort_values("ROC-AUC", ascending=False)
print(scores_df)

# =========================
# 8. Weighted Blending
# =========================
print(f"\n{'='*50}")
print("ENSEMBLE BLENDING")
print(f"{'='*50}")

oof_blend = 0.4 * oof_lgb + 0.35 * oof_cat + 0.25 * oof_xgb
pred_blend = 0.4 * pred_lgb + 0.35 * pred_cat + 0.25 * pred_xgb

blend_score = roc_auc_score(y, oof_blend)
print(f"Weighted Blend (0.4, 0.35, 0.25) ROC-AUC: {blend_score:.5f}")

# =========================
# 9. Level-2 Stacking with Logistic Regression
# =========================
print("\nTraining Level-2 Stacked Model...")
stack_train = np.column_stack([oof_lgb, oof_cat, oof_xgb])
stack_test = np.column_stack([pred_lgb, pred_cat, pred_xgb])

lvl2 = LogisticRegression(max_iter=2000, random_state=42)
lvl2.fit(stack_train, y)
oof_stack = lvl2.predict_proba(stack_train)[:, 1]
pred_stack = lvl2.predict_proba(stack_test)[:, 1]

stack_score = roc_auc_score(y, oof_stack)
print(f"Stacked Model ROC-AUC: {stack_score:.5f}")

# Print stacking weights
print(f"\nStacking Weights: LGB={lvl2.coef_[0][0]:.4f}, CAT={lvl2.coef_[0][1]:.4f}, XGB={lvl2.coef_[0][2]:.4f}")

# =========================
# 10. Choose Best Predictions
# =========================
print(f"\n{'='*50}")
print("FINAL MODEL SELECTION")
print(f"{'='*50}")

final_predictions = pred_stack if stack_score > blend_score else pred_blend
final_method = "Stacked" if stack_score > blend_score else "Blended"
final_score = max(stack_score, blend_score)

print(f"Using {final_method} predictions (ROC-AUC: {final_score:.5f})")

# =========================
# 11. ROC Curve Visualization
# =========================
plt.figure(figsize=(10, 6))

for name, oof_pred in [("LightGBM", oof_lgb), ("CatBoost", oof_cat), 
                        ("XGBoost", oof_xgb), ("Blended", oof_blend), 
                        ("Stacked", oof_stack)]:
    fpr, tpr, _ = roc_curve(y, oof_pred)
    auc_score = roc_auc_score(y, oof_pred)
    plt.plot(fpr, tpr, label=f"{name} (AUC: {auc_score:.4f})", linewidth=2)

plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves - All Models", fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# =========================
# 12. Create Submission File
# =========================
print(f"\n{'='*50}")
print("CREATING SUBMISSION")
print(f"{'='*50}")

submission = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': final_predictions
})

submission.to_csv('submission.csv', index=False)
print(f"Submission file created: submission.csv")
print(f"Shape: {submission.shape}")
print("\nFirst few predictions:")
print(submission.head(10))
print(f"\nPrediction statistics:")
print(submission['diagnosed_diabetes'].describe())

print("\n" + "="*50)
print("PROCESS COMPLETE!")
print("="*50)




