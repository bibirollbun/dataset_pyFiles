"""
â•”â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•—
â•‘  DIABETES PREDICTION: ULTRA-OPTIMIZED ENSEMBLE MODEL v2.0               â•‘
â•‘  Target: 0.72+ AUC | Strategy: Advanced Feature Engineering + Stacking  â•‘
â•šâ•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ğŸš€ DIABETES PREDICTION: ULTRA-OPTIMIZED ENSEMBLE v2.0")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'n_folds': 5,
    'random_state': 42,
    'verbose': True,
    'use_quantile_transform': True,
    'use_pseudo_labeling': False,  # Can enable after first submission
    'optimize_weights': True
}

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n[1/9] Loading data...")
DATA_PATH = "/kaggle/input/playground-series-s5e12/"
train = pd.read_csv(DATA_PATH + "train.csv")
test = pd.read_csv(DATA_PATH + "test.csv")

print(f"âœ“ Train: {train.shape}, Test: {test.shape}")

# ============================================================================
# 2. ULTRA FEATURE ENGINEERING
# ============================================================================
print("\n[2/9] Ultra feature engineering...")

def create_ultra_features(df):
    """Comprehensive feature engineering with medical domain expertise"""
    df = df.copy()
    
    # === CORE HEALTH RATIOS ===
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 0.1)
    df['trig_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 0.1)
    df['non_hdl_chol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    df['atherogenic_index'] = np.log10(df['triglycerides'] / (df['hdl_cholesterol'] + 0.1))
    df['cholesterol_balance'] = df['hdl_cholesterol'] / (df['ldl_cholesterol'] + 0.1)
    
    # === BLOOD PRESSURE METRICS ===
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['map'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3
    df['bp_product'] = df['systolic_bp'] * df['diastolic_bp']
    df['systolic_ratio'] = df['systolic_bp'] / 120
    df['diastolic_ratio'] = df['diastolic_bp'] / 80
    df['bp_imbalance'] = df['systolic_bp'] / (df['diastolic_bp'] + 0.1)
    
    # === BMI & BODY COMPOSITION ===
    df['bmi_squared'] = df['bmi'] ** 2
    df['bmi_cubed'] = df['bmi'] ** 3
    df['bmi_waist_product'] = df['bmi'] * df['waist_to_hip_ratio']
    df['bmi_waist_ratio'] = df['bmi'] / (df['waist_to_hip_ratio'] + 0.1)
    df['body_comp_score'] = (df['bmi'] * 0.6) + (df['waist_to_hip_ratio'] * 100 * 0.4)
    
    # === BINARY RISK FLAGS ===
    df['obesity_flag'] = (df['bmi'] >= 30).astype(int)
    df['overweight_flag'] = ((df['bmi'] >= 25) & (df['bmi'] < 30)).astype(int)
    df['hypertension_flag'] = ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80)).astype(int)
    df['high_ldl_flag'] = (df['ldl_cholesterol'] > 130).astype(int)
    df['low_hdl_flag'] = (df['hdl_cholesterol'] < 40).astype(int)
    df['high_trig_flag'] = (df['triglycerides'] > 150).astype(int)
    df['central_obesity_flag'] = (df['waist_to_hip_ratio'] > 0.90).astype(int)
    
    # === LIFESTYLE RISK SCORES ===
    df['inactive_flag'] = (df['physical_activity_minutes_per_week'] < 150).astype(int)
    df['poor_diet_flag'] = (df['diet_score'] < 5).astype(int)
    df['poor_sleep_flag'] = ((df['sleep_hours_per_day'] < 7) | 
                             (df['sleep_hours_per_day'] > 9)).astype(int)
    df['excessive_screen_flag'] = (df['screen_time_hours_per_day'] > 7).astype(int)
    
    df['lifestyle_risk_score'] = (df['inactive_flag'] * 2 + 
                                  df['poor_diet_flag'] + 
                                  df['poor_sleep_flag'] + 
                                  df['excessive_screen_flag'])
    
    df['health_behavior_score'] = (df['diet_score'] + 
                                   (df['physical_activity_minutes_per_week'] / 30) + 
                                   df['sleep_hours_per_day'] - 
                                   (df['screen_time_hours_per_day'] / 2) - 
                                   df['alcohol_consumption_per_week'])
    
    # === AGE-BASED FEATURES ===
    df['age_squared'] = df['age'] ** 2
    df['age_log'] = np.log1p(df['age'])
    df['age_normalized'] = (df['age'] - 50) / 15
    df['senior_flag'] = (df['age'] >= 65).astype(int)
    df['middle_aged_flag'] = ((df['age'] >= 40) & (df['age'] < 65)).astype(int)
    df['young_flag'] = (df['age'] < 40).astype(int)
    
    # === METABOLIC SYNDROME SCORE ===
    df['metabolic_syndrome_score'] = (
        df['obesity_flag'] * 2 +
        df['hypertension_flag'] * 2 +
        df['high_trig_flag'] +
        df['low_hdl_flag'] +
        df['central_obesity_flag']
    )
    
    # === MEDICAL HISTORY COMPOSITE ===
    df['medical_risk_score'] = (
        df['family_history_diabetes'] * 3 +
        df['hypertension_history'] * 2 +
        df['cardiovascular_history'] * 2
    )
    
    # === CARDIOVASCULAR RISK ===
    df['cvd_risk_score'] = (
        (df['systolic_bp'] / 140) +
        (df['ldl_cholesterol'] / 130) +
        (df['bmi'] / 30) +
        (1 - df['hdl_cholesterol'] / 60)
    ) / 4
    
    # === ACTIVITY & SEDENTARY METRICS ===
    df['sedentary_ratio'] = df['screen_time_hours_per_day'] / \
                            ((df['physical_activity_minutes_per_week'] / 60) + 0.1)
    df['activity_per_day'] = df['physical_activity_minutes_per_week'] / 7
    df['activity_intensity'] = df['physical_activity_minutes_per_week'] / (df['age'] + 1)
    df['screen_activity_balance'] = df['screen_time_hours_per_day'] / (df['activity_per_day'] + 0.1)
    
    # === CRITICAL INTERACTIONS ===
    df['age_bmi'] = df['age'] * df['bmi']
    df['age_bmi_squared'] = df['age'] * (df['bmi'] ** 2)
    df['age_systolic'] = df['age'] * df['systolic_bp']
    df['age_family'] = df['age'] * df['family_history_diabetes']
    df['age_activity'] = df['age'] / (df['physical_activity_minutes_per_week'] + 1)
    df['bmi_systolic'] = df['bmi'] * df['systolic_bp']
    df['bmi_activity'] = df['bmi'] / (df['physical_activity_minutes_per_week'] + 1)
    df['bmi_ldl'] = df['bmi'] * df['ldl_cholesterol']
    df['age_metabolic'] = df['age'] * df['metabolic_syndrome_score']
    
    # === CHOLESTEROL INTERACTIONS ===
    df['ldl_age'] = df['ldl_cholesterol'] * df['age']
    df['hdl_activity'] = df['hdl_cholesterol'] * df['physical_activity_minutes_per_week']
    df['trig_bmi'] = df['triglycerides'] * df['bmi']
    df['chol_total_age'] = df['cholesterol_total'] * df['age']
    
    # === ADVANCED RATIOS ===
    df['risk_density'] = (df['metabolic_syndrome_score'] + 
                          df['medical_risk_score']) / (df['health_behavior_score'] + 0.1)
    df['health_age_ratio'] = df['health_behavior_score'] / (df['age'] + 1)
    df['bp_bmi_ratio'] = df['map'] / (df['bmi'] + 0.1)
    
    # === POLYNOMIAL FEATURES FOR TOP PREDICTORS ===
    df['family_squared'] = df['family_history_diabetes'] ** 2
    df['activity_squared'] = (df['physical_activity_minutes_per_week'] / 100) ** 2
    df['systolic_squared'] = (df['systolic_bp'] / 100) ** 2
    
    # === BINNED FEATURES ===
    df['age_bin'] = pd.cut(df['age'], bins=[0,35,50,65,100], labels=[0,1,2,3]).astype(float)
    df['bmi_bin'] = pd.cut(df['bmi'], bins=[0,18.5,25,30,100], labels=[0,1,2,3]).astype(float)
    df['activity_bin'] = pd.cut(df['physical_activity_minutes_per_week'], 
                                 bins=[0,50,100,150,200,1000], 
                                 labels=[0,1,2,3,4]).astype(float)
    
    # === NORMALIZED FEATURES ===
    df['systolic_normalized'] = (df['systolic_bp'] - 120) / 20
    df['diastolic_normalized'] = (df['diastolic_bp'] - 80) / 10
    df['bmi_normalized'] = (df['bmi'] - 25) / 5
    df['hdl_normalized'] = (df['hdl_cholesterol'] - 50) / 10
    df['ldl_normalized'] = (df['ldl_cholesterol'] - 100) / 20
    
    return df

train_fe = create_ultra_features(train)
test_fe = create_ultra_features(test)

print(f"âœ“ Created {train_fe.shape[1]} total features")

# ============================================================================
# 3. ENCODE CATEGORICALS
# ============================================================================
print("\n[3/9] Encoding categorical features...")

cat_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
            'smoking_status', 'employment_status']

for col in cat_cols:
    le = LabelEncoder()
    train_fe[col] = le.fit_transform(train_fe[col].astype(str))
    test_fe[col] = le.transform(test_fe[col].astype(str))

print(f"âœ“ Encoded {len(cat_cols)} categorical features")

# ============================================================================
# 4. QUANTILE TRANSFORMATION (Optional but powerful)
# ============================================================================
if CONFIG['use_quantile_transform']:
    print("\n[4/9] Applying quantile transformation...")
    
    feature_cols = [col for col in train_fe.columns if col not in ['id', 'diagnosed_diabetes']]
    
    # Select numerical features for transformation
    numeric_cols = train_fe[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    qt = QuantileTransformer(n_quantiles=1000, random_state=CONFIG['random_state'], output_distribution='normal')
    train_fe[numeric_cols] = qt.fit_transform(train_fe[numeric_cols])
    test_fe[numeric_cols] = qt.transform(test_fe[numeric_cols])
    
    print(f"âœ“ Transformed {len(numeric_cols)} numerical features")
else:
    print("\n[4/9] Skipping quantile transformation...")
    feature_cols = [col for col in train_fe.columns if col not in ['id', 'diagnosed_diabetes']]

# ============================================================================
# 5. PREPARE DATA
# ============================================================================
print("\n[5/9] Preparing datasets...")

X = train_fe[feature_cols].values
y = train_fe['diagnosed_diabetes'].values
X_test = test_fe[feature_cols].values

print(f"âœ“ X: {X.shape}, y: {y.shape}, X_test: {X_test.shape}")

# ============================================================================
# 6. OPTIMIZED MODEL PARAMETERS
# ============================================================================
print("\n[6/9] Configuring optimized models...")

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.015,
    'num_leaves': 95,
    'max_depth': 9,
    'min_child_samples': 25,
    'subsample': 0.75,
    'subsample_freq': 1,
    'colsample_bytree': 0.75,
    'reg_alpha': 0.3,
    'reg_lambda': 0.3,
    'min_split_gain': 0.01,
    'random_state': CONFIG['random_state'],
    'verbose': -1,
    'n_jobs': -1
}

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.015,
    'max_depth': 8,
    'min_child_weight': 5,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'reg_alpha': 0.3,
    'reg_lambda': 0.3,
    'gamma': 0.01,
    'random_state': CONFIG['random_state'],
    'tree_method': 'hist',
    'n_jobs': -1
}

cat_params = {
    'iterations': 3000,
    'learning_rate': 0.015,
    'depth': 8,
    'l2_leaf_reg': 5,
    'random_state': CONFIG['random_state'],
    'verbose': 0,
    'early_stopping_rounds': 150,
    'task_type': 'CPU',
    'border_count': 254
}

print("âœ“ Models configured with optimized hyperparameters")

# ============================================================================
# 7. CROSS-VALIDATION TRAINING
# ============================================================================
print(f"\n[7/9] Training ensemble with {CONFIG['n_folds']}-fold CV...")

skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, 
                      random_state=CONFIG['random_state'])

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

fold_scores = {'lgb': [], 'xgb': [], 'cat': []}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    if CONFIG['verbose']:
        print(f"\n{'â”€'*80}")
        print(f"FOLD {fold}/{CONFIG['n_folds']}")
        print(f"{'â”€'*80}")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # LightGBM
    if CONFIG['verbose']:
        print("  âš¡ LightGBM...")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    lgb_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=3000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(150, verbose=False)]
    )
    
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    test_lgb += lgb_model.predict(X_test) / CONFIG['n_folds']
    lgb_auc = roc_auc_score(y_val, oof_lgb[val_idx])
    fold_scores['lgb'].append(lgb_auc)
    if CONFIG['verbose']:
        print(f"     AUC: {lgb_auc:.6f}")
    
    # XGBoost
    if CONFIG['verbose']:
        print("  âš¡ XGBoost...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=3000,
        evals=[(dval, 'eval')],
        early_stopping_rounds=150,
        verbose_eval=False
    )
    
    oof_xgb[val_idx] = xgb_model.predict(dval)
    test_xgb += xgb_model.predict(xgb.DMatrix(X_test)) / CONFIG['n_folds']
    xgb_auc = roc_auc_score(y_val, oof_xgb[val_idx])
    fold_scores['xgb'].append(xgb_auc)
    if CONFIG['verbose']:
        print(f"     AUC: {xgb_auc:.6f}")
    
    # CatBoost
    if CONFIG['verbose']:
        print("  âš¡ CatBoost...")
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    
    oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    test_cat += cat_model.predict_proba(X_test)[:, 1] / CONFIG['n_folds']
    cat_auc = roc_auc_score(y_val, oof_cat[val_idx])
    fold_scores['cat'].append(cat_auc)
    if CONFIG['verbose']:
        print(f"     AUC: {cat_auc:.6f}")
    
    if CONFIG['verbose']:
        ensemble_val = (oof_lgb[val_idx] + oof_xgb[val_idx] + oof_cat[val_idx]) / 3
        print(f"  ğŸ�¯ Ensemble: {roc_auc_score(y_val, ensemble_val):.6f}")

# ============================================================================
# 8. OPTIMIZE ENSEMBLE WEIGHTS
# ============================================================================
print(f"\n[8/9] Optimizing ensemble weights...")

lgb_cv = roc_auc_score(y, oof_lgb)
xgb_cv = roc_auc_score(y, oof_xgb)
cat_cv = roc_auc_score(y, oof_cat)

if CONFIG['optimize_weights']:
    # Performance-based weighting
    scores = np.array([lgb_cv, xgb_cv, cat_cv])
    weights = scores / scores.sum()
else:
    weights = np.array([0.334, 0.333, 0.333])

oof_ensemble = oof_lgb * weights[0] + oof_xgb * weights[1] + oof_cat * weights[2]
test_ensemble = test_lgb * weights[0] + test_xgb * weights[1] + test_cat * weights[2]

ensemble_cv = roc_auc_score(y, oof_ensemble)

print(f"\n{'â•�'*80}")
print("ğŸ“Š FINAL CV RESULTS")
print(f"{'â•�'*80}")
print(f"LightGBM:  {lgb_cv:.6f} (Â±{np.std(fold_scores['lgb']):.6f})")
print(f"XGBoost:   {xgb_cv:.6f} (Â±{np.std(fold_scores['xgb']):.6f})")
print(f"CatBoost:  {cat_cv:.6f} (Â±{np.std(fold_scores['cat']):.6f})")
print(f"{'â”€'*80}")
print(f"ğŸ�† ENSEMBLE: {ensemble_cv:.6f}")
print(f"{'â”€'*80}")
print(f"Weights: LGB={weights[0]:.3f} | XGB={weights[1]:.3f} | CAT={weights[2]:.3f}")
print(f"{'â•�'*80}")

# ============================================================================
# 9. CREATE SUBMISSION
# ============================================================================
print(f"\n[9/9] Creating submission...")

submission = pd.DataFrame({
    'id': test_fe['id'],
    'diagnosed_diabetes': test_ensemble
})

submission.to_csv('submission_v2.csv', index=False)

print(f"\nâœ… Submission saved: submission_v2.csv")
print(f"\nğŸ“ˆ Prediction Statistics:")
print(f"   Min:  {test_ensemble.min():.6f}")
print(f"   Max:  {test_ensemble.max():.6f}")
print(f"   Mean: {test_ensemble.mean():.6f}")
print(f"   Std:  {test_ensemble.std():.6f}")

print(f"\n{'â•�'*80}")
print(f"ğŸ�¯ EXPECTED LB SCORE: ~{ensemble_cv:.5f}")
print(f"ğŸš€ IMPROVEMENT TARGET: +0.003 to +0.005")
print(f"{'â•�'*80}")

print("\nğŸ’¡ Next Steps for Further Improvement:")
print("   1. âœ“ Advanced feature engineering (DONE)")
print("   2. âœ“ Quantile transformation (DONE)")
print("   3. âœ“ Optimized hyperparameters (DONE)")
print("   4. â�­  Pseudo-labeling with high-confidence predictions")
print("   5. â�­  Neural network meta-learner")
print("   6. â�­  Hill climbing for weight optimization")
print("   7. â�­  Target encoding for categorical features")

print("\nâœ¨ TRAINING COMPLETE! âœ¨\n")

