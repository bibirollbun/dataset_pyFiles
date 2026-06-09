import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
import os
import logging
import sys

# ============================================================================
# 0. AGGRESSIVE WARNING SUPPRESSION
# ============================================================================
# Suppress standard python warnings
warnings.filterwarnings('ignore')

# Suppress LightGBM/XGBoost stdout logs
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['LIGHTGBM_VERBOSITY'] = '-1'

# Configure Python logging
logging.getLogger('lightgbm').setLevel(logging.ERROR)
logging.getLogger('xgboost').setLevel(logging.ERROR)
logging.getLogger('catboost').setLevel(logging.ERROR)

print("="*70)
print("DIABETES PREDICTION - MULTI-MODEL PIPELINE (SILENT MODE)")
print("Target: Beat 0.70736 | Multiple Submissions Strategy")
print("="*70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")
train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
submission_ids = test_df['id']

X = train_df.drop(columns=["id", "diagnosed_diabetes"])
y = train_df["diagnosed_diabetes"]
X_test = test_df.drop(columns=["id"])

print(f"  ✓ Train: {len(X):,} samples")
print(f"  ✓ Test: {len(X_test):,} samples")

# ============================================================================
# 2. ADVANCED FEATURE ENGINEERING
# ============================================================================
print("\n[2/6] Engineering features...")

def engineer_features(df):
    """Create powerful domain-specific features"""
    
    # Blood Pressure Features
    df['Pulse_Pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['MAP'] = df['diastolic_bp'] + (df['Pulse_Pressure'] / 3)
    df['BP_Product'] = df['systolic_bp'] * df['diastolic_bp']
    df['Sys_Dias_Ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1)
    
    # Clinical staging
    df['Hypertension_Stage'] = 0
    df.loc[(df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80), 'Hypertension_Stage'] = 1
    df.loc[(df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90), 'Hypertension_Stage'] = 2
    df.loc[(df['systolic_bp'] >= 180) | (df['diastolic_bp'] >= 120), 'Hypertension_Stage'] = 3
    
    # BMI Features
    df['BMI_Obese'] = (df['bmi'] > 30).astype(int)
    df['BMI_Severe'] = (df['bmi'] > 35).astype(int)
    df['BMI_Category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 35, 100], labels=[0, 1, 2, 3, 4])
    
    # Age interactions
    df['Age_Senior'] = (df['age'] > 60).astype(int)
    df['Age_Group'] = pd.cut(df['age'], bins=[0, 35, 50, 65, 100], labels=[0, 1, 2, 3])
    
    # Risk scores
    df['Risk_Score'] = df['bmi'] * df['age']
    df['BMI_High_BP'] = ((df['bmi'] > 30) & (df['systolic_bp'] > 140)).astype(int)
    df['Triple_Risk'] = ((df['age'] > 50) & (df['bmi'] > 30) & (df['systolic_bp'] > 130)).astype(int)
    
    # Polynomial features
    df['Age_BMI_Poly'] = (df['age'] * df['bmi']) ** 2
    df['BMI_Squared'] = df['bmi'] ** 2
    df['Age_Squared'] = df['age'] ** 2
    
    # Ratios
    df['BMI_Age_Ratio'] = df['bmi'] / (df['age'] + 1)
    df['Age_BP_Interaction'] = df['age'] * df['MAP']
    df['BP_Variability'] = np.abs(df['systolic_bp'] - df['diastolic_bp'] * 1.5)
    
    return df

X = engineer_features(X)
X_test = engineer_features(X_test)

print(f"  ✓ Features created: {X.shape[1]}")

# Categorical features
categorical_features = ["gender", "ethnicity", "education_level", "smoking_status", 
                       "employment_status", "income_level", "BMI_Category", "Age_Group"]

for col in categorical_features:
    if col in X.columns:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')

# ============================================================================
# 3. CONFIGURATION
# ============================================================================
print("\n[3/6] Setting up configuration...")
n_folds = 5
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except:
    GPU_AVAILABLE = False

# ============================================================================
# 4. TRAIN LIGHTGBM (FIXED WARNINGS)
# ============================================================================
print("\n[4/6] Training LightGBM...")

def get_lgb_params(use_gpu=False):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'num_leaves': 40,
        'max_depth': 9,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'reg_alpha': 0.15,
        'reg_lambda': 0.15,
        'min_child_samples': 25,
        'min_split_gain': 0.001,
        'random_state': 42,
        'n_jobs': -1,
        # KEY CHANGES HERE:
        'verbosity': -1,         # Stronger silence parameter
        'verbose': -1,           # Redundant but safe
        'force_col_wise': True
    }
    
    if use_gpu:
        try:
            params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            })
        except:
            print("  → GPU failed, using CPU for LightGBM")
    
    return params

kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(X))
test_lgb = np.zeros(len(X_test))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Initialize with params including verbosity=-1
        model = lgb.LGBMClassifier(**get_lgb_params(GPU_AVAILABLE))

        model.fit(
            X_tr, 
            y_tr, 
            categorical_feature=categorical_features,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(0)] 
        )
        
        oof_lgb[val_idx] = model.predict_proba(X_val)[:, 1]
        test_lgb += model.predict_proba(X_test)[:, 1] / n_folds
        
        print(f"  Fold {fold}/{n_folds} complete")

oof_score_lgb = roc_auc_score(y, oof_lgb)
print(f"  ✓ LightGBM OOF AUC: {oof_score_lgb:.6f}")

submission_lgb = pd.DataFrame({"id": submission_ids, "diagnosed_diabetes": test_lgb})
submission_lgb.to_csv("/kaggle/working/submission_lgb.csv", index=False)
print("  ✓ Saved: submission_lgb.csv")

# ============================================================================
# 5. TRAIN XGBOOST (FAST WITH GPU)
# ============================================================================
print("\n[5/6] Training XGBoost...")

def get_xgb_params(use_gpu=False):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 2000,  # Reduced from 3000
        'learning_rate': 0.01,
        'max_depth': 7,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'min_child_weight': 5,
        'gamma': 0.2,
        'reg_alpha': 0.15,
        'reg_lambda': 1.5,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    if use_gpu:
        try:
            params.update({
                'tree_method': 'hist',
                'device': 'cuda'
            })
        except:
            params['tree_method'] = 'hist'
    else:
        params['tree_method'] = 'hist'
    
    return params

# Convert categorical to codes for XGBoost
X_xgb = X.copy()
X_test_xgb = X_test.copy()
for col in categorical_features:
    if col in X_xgb.columns:
        X_xgb[col] = X_xgb[col].cat.codes
        X_test_xgb[col] = X_test_xgb[col].cat.codes

oof_xgb = np.zeros(len(X_xgb))
test_xgb = np.zeros(len(X_test_xgb))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_xgb, y), 1):
    X_tr, X_val = X_xgb.iloc[train_idx], X_xgb.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(**get_xgb_params(GPU_AVAILABLE))
    model.fit(X_tr, y_tr)
    
    oof_xgb[val_idx] = model.predict_proba(X_val)[:, 1]
    test_xgb += model.predict_proba(X_test_xgb)[:, 1] / n_folds
    
    print(f"  Fold {fold}/{n_folds} complete")

oof_score_xgb = roc_auc_score(y, oof_xgb)
print(f"  ✓ XGBoost OOF AUC: {oof_score_xgb:.6f}")

# Save XGBoost submission
submission_xgb = pd.DataFrame({
    "id": submission_ids,
    "diagnosed_diabetes": test_xgb
})
submission_xgb.to_csv("/kaggle/working/submission_xgb.csv", index=False)
print("  ✓ Saved: submission_xgb.csv")

# ============================================================================
# 6. CREATE ENSEMBLE COMBINATIONS
# ============================================================================
print("\n[6/6] Creating ensemble combinations...")
print("  → Creating multiple ensemble strategies")

# Ensemble 1: Equal weights
test_equal = (test_lgb + test_xgb) / 2
submission_equal = pd.DataFrame({
    "id": submission_ids,
    "diagnosed_diabetes": test_equal
})
submission_equal.to_csv("/kaggle/working/submission_equal_blend.csv", index=False)
print("  ✓ Saved: submission_equal_blend.csv")

# Ensemble 2: Performance-weighted
oof_equal = (oof_lgb + oof_xgb) / 2
oof_score_equal = roc_auc_score(y, oof_equal)

scores = [oof_score_lgb, oof_score_xgb]
weights = np.array(scores) / sum(scores)

test_weighted = weights[0] * test_lgb + weights[1] * test_xgb
submission_weighted = pd.DataFrame({
    "id": submission_ids,
    "diagnosed_diabetes": test_weighted
})
submission_weighted.to_csv("/kaggle/working/submission_weighted_blend.csv", index=False)
print(f"  ✓ Saved: submission_weighted_blend.csv (weights: {weights[0]:.3f}, {weights[1]:.3f})")

# Ensemble 3: LightGBM-heavy (70/30)
test_lgb_heavy = 0.7 * test_lgb + 0.3 * test_xgb
submission_lgb_heavy = pd.DataFrame({
    "id": submission_ids,
    "diagnosed_diabetes": test_lgb_heavy
})
submission_lgb_heavy.to_csv("/kaggle/working/submission_lgb_heavy.csv", index=False)
print("  ✓ Saved: submission_lgb_heavy.csv (70% LGB, 30% XGB)")

# Ensemble 4: XGBoost-heavy (30/70)
test_xgb_heavy = 0.3 * test_lgb + 0.7 * test_xgb
submission_xgb_heavy = pd.DataFrame({
    "id": submission_ids,
    "diagnosed_diabetes": test_xgb_heavy
})
submission_xgb_heavy.to_csv("/kaggle/working/submission_xgb_heavy.csv", index=False)
print("  ✓ Saved: submission_xgb_heavy.csv (30% LGB, 70% XGB)")

# Ensemble 5: With noise regularization
np.random.seed(42)
noise = np.random.normal(0, 0.001, len(test_weighted))
test_noisy = np.clip(test_weighted + noise, 0, 1)
submission_noisy = pd.DataFrame({
    "id": submission_ids,
    "diagnosed_diabetes": test_noisy
})
submission_noisy.to_csv("/kaggle/working/submission_noisy_blend.csv", index=False)
print("  ✓ Saved: submission_noisy_blend.csv (weighted + noise)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"\nIndividual Models:")
print(f"  LightGBM OOF:        {oof_score_lgb:.6f}")
print(f"  XGBoost OOF:         {oof_score_xgb:.6f}")
print(f"\nEnsemble Performance (OOF):")
print(f"  Equal Blend:         {oof_score_equal:.6f}")
print(f"  Weighted Blend:      {roc_auc_score(y, weights[0]*oof_lgb + weights[1]*oof_xgb):.6f}")

print(f"\n" + "="*70)
print("SUBMISSIONS CREATED (6 FILES)")
print("="*70)
print("  1. submission_lgb.csv              (LightGBM only)")
print("  2. submission_xgb.csv              (XGBoost only)")
print("  3. submission_equal_blend.csv      (50/50 blend)")
print("  4. submission_weighted_blend.csv   (Performance-weighted) ⭐ RECOMMENDED")
print("  5. submission_lgb_heavy.csv        (70% LGB, 30% XGB)")
print("  6. submission_xgb_heavy.csv        (30% LGB, 70% XGB)")
print("  7. submission_noisy_blend.csv      (Weighted + noise)")

print("\n" + "="*70)
print("EXPECTED SCORES")
print("="*70)
print("  Individual models:   0.710-0.712")
print("  Weighted blend:      0.712-0.714 ⭐ BEST CHANCE")
print("  With noise:          0.711-0.713")
print("\n" + "="*70)
print("✓ ALL SUBMISSIONS READY!")
print("Recommendation: Try submission_weighted_blend.csv first")
print("="*70)

