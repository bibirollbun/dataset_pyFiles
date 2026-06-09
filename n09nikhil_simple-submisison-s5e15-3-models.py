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


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("✓ Libraries imported")


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

print(f"Competition train: {train.shape}")
print(f"Competition test: {test.shape}")

# Try to load original dataset (if available)
try:
    # Original dataset path - adjust if needed
    original = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_012_health_indicators_BRFSS2015.csv')
    print(f"✓ Original dataset loaded: {original.shape}")
    has_original = True
except:
    print("⚠ Original dataset not found - continuing without it")
    has_original = False


target = 'diagnosed_diabetes'

# Check distribution differences
print("\n" + "="*50)
print("TRAIN-TEST DISTRIBUTION ANALYSIS")
print("="*50)

X_train_raw = train.drop(['id', target], axis=1)
X_test_raw = test.drop(['id'], axis=1)

# Key features with distribution mismatch
check_features = ['physical_activity_minutes_per_week', 'age', 'bmi', 'systolic_bp']
for feat in check_features:
    if feat in X_train_raw.columns:
        train_mean = X_train_raw[feat].mean()
        test_mean = X_test_raw[feat].mean()
        diff_pct = abs(train_mean - test_mean) / train_mean * 100
        print(f"{feat:40s}: Train={train_mean:.2f}, Test={test_mean:.2f}, Diff={diff_pct:.1f}%")



def create_robust_features(df):
    """
    Create features that are ROBUST to distribution shifts
    Focus on:
    1. Ratios and relative measures (scale-invariant)
    2. Binary thresholds (absolute)
    3. Percentile-based features (distribution-independent)
    """
    df = df.copy()
    
    # === RATIO FEATURES (Scale-invariant) ===
    df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1)
    df['cholesterol_hdl_ldl_ratio'] = df['hdl_cholesterol'] / (df['ldl_cholesterol'] + 1)
    df['activity_per_bmi'] = df['physical_activity_minutes_per_week'] / (df['bmi'] + 1)
    df['waist_hip_ratio_risk'] = (df['waist_to_hip_ratio'] > 0.9).astype(int)
    
    # === CLINICAL THRESHOLDS (Absolute, distribution-independent) ===
    df['high_bp'] = ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80)).astype(int)
    df['stage2_hypertension'] = ((df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90)).astype(int)
    df['high_cholesterol'] = (df['cholesterol_total'] >= 200).astype(int)
    df['low_hdl'] = (df['hdl_cholesterol'] < 40).astype(int)
    df['high_ldl'] = (df['ldl_cholesterol'] >= 130).astype(int)
    df['high_triglycerides'] = (df['triglycerides'] >= 150).astype(int)
    df['obesity'] = (df['bmi'] >= 30).astype(int)
    df['overweight'] = ((df['bmi'] >= 25) & (df['bmi'] < 30)).astype(int)
    
    # === AGE RISK CATEGORIES ===
    df['age_risk_high'] = (df['age'] >= 60).astype(int)
    df['age_risk_medium'] = ((df['age'] >= 45) & (df['age'] < 60)).astype(int)
    
    # === METABOLIC SYNDROME (5 criteria) ===
    df['metabolic_syndrome_count'] = (
        (df['bmi'] >= 30).astype(int) +
        (df['triglycerides'] >= 150).astype(int) +
        (df['hdl_cholesterol'] < 40).astype(int) +
        ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80)).astype(int) +
        (df['waist_to_hip_ratio'] > 0.9).astype(int)
    )
    df['has_metabolic_syndrome'] = (df['metabolic_syndrome_count'] >= 3).astype(int)
    
    # === CARDIOVASCULAR RISK SCORE ===
    df['cardio_risk_score'] = (
        df['high_bp'] * 2 +
        df['high_cholesterol'] +
        df['obesity'] * 2 +
        df['age_risk_high'] +
        df['family_history_diabetes'] * 3 +  # Family history is strongest predictor
        df['hypertension_history'] +
        df['cardiovascular_history'] * 2
    )
    
    # === INTERACTION WITH TOP PREDICTOR (family_history) ===
    df['family_age_interaction'] = df['family_history_diabetes'] * df['age']
    df['family_bmi_interaction'] = df['family_history_diabetes'] * df['bmi']
    df['family_bp_interaction'] = df['family_history_diabetes'] * df['systolic_bp']
    
    # === LIFESTYLE RISK (Simpler version) ===
    df['poor_lifestyle'] = (
        (df['physical_activity_minutes_per_week'] < 150).astype(int) +
        (df['sleep_hours_per_day'] < 6).astype(int) +
        (df['screen_time_hours_per_day'] > 8).astype(int)
    )
    
    # === NORMALIZE ACTIVITY (ADDRESS DISTRIBUTION SHIFT) ===
    # Use rank-based normalization (robust to distribution shift)
    df['activity_rank'] = df['physical_activity_minutes_per_week'].rank(pct=True)
    df['activity_low'] = (df['activity_rank'] < 0.25).astype(int)
    df['activity_high'] = (df['activity_rank'] > 0.75).astype(int)
    
    return df

print("✓ Feature engineering function defined")
print("  Focus: Distribution-robust features")


X = train.drop(['id', target], axis=1)
y = train[target]
X_test = test.drop(['id'], axis=1)

# Apply feature engineering
print("\nApplying feature engineering...")
X_eng = create_robust_features(X)
X_test_eng = create_robust_features(X_test)

# If original dataset available, add it to training
if has_original:
    print("Processing original dataset...")
    # Map original dataset columns to competition format (adjust as needed)
    # This part depends on original dataset structure
    # For now, just combine if structures match
    try:
        orig_features = create_robust_features(original.drop([target], axis=1))
        X_combined = pd.concat([X_eng, orig_features], axis=0, ignore_index=True)
        y_combined = pd.concat([y, original[target]], axis=0, ignore_index=True)
        print(f"✓ Combined training size: {X_combined.shape}")
        X_eng = X_combined
        y = y_combined
    except:
        print("⚠ Could not combine datasets - using competition data only")

print(f"\n✓ Final training features: {X_eng.shape[1]}")
print(f"✓ Training samples: {len(X_eng):,}")



categorical_cols = X_eng.select_dtypes(include=['object']).columns.tolist()

if categorical_cols:
    print(f"\nEncoding {len(categorical_cols)} categorical features...")
    for col in categorical_cols:
        le = LabelEncoder()
        X_eng[col] = le.fit_transform(X_eng[col].astype(str))
        if col in X_test_eng.columns:
            X_test_eng[col] = le.transform(X_test_eng[col].astype(str))
    print("✓ Encoding complete")
else:
    print("✓ No categorical features to encode")



print("\n" + "="*80)
print("TRAINING OPTIMIZED LIGHTGBM MODEL")
print("="*80)

# Optimized parameters focusing on generalization
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,           # Slower learning
    'num_leaves': 31,                # Conservative (prevent overfitting)
    'max_depth': 6,                  # Not too deep
    'min_child_samples': 100,        # More samples per leaf (stronger regularization)
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,                # Strong L1 regularization
    'reg_lambda': 1.0,               # Strong L2 regularization
    'min_split_gain': 0.02,          # Higher threshold to split
    'max_bin': 255,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

# Use 5-fold CV (more stable than 10-fold for this size)
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_predictions = np.zeros(len(X_eng))
test_predictions = np.zeros(len(X_test_eng))
fold_scores = []
feature_importance = pd.DataFrame()

print(f"\nStarting {n_folds}-fold Cross-Validation...")
print("Focus: Generalization > Training accuracy\n")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_eng, y), 1):
    X_train, X_val = X_eng.iloc[train_idx], X_eng.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=3000,              # Allow more iterations
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=150),  # More patience
            lgb.log_evaluation(500)
        ]
    )
    
    oof_predictions[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_predictions += model.predict(X_test_eng, num_iteration=model.best_iteration) / n_folds
    
    fold_score = roc_auc_score(y_val, oof_predictions[val_idx])
    fold_scores.append(fold_score)
    
    # Store feature importance
    fold_importance = pd.DataFrame({
        'feature': X_eng.columns,
        'importance': model.feature_importance(importance_type='gain'),
        'fold': fold
    })
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
    
    print(f"Fold {fold} - Validation AUC: {fold_score:.6f} - Best iteration: {model.best_iteration}")

cv_score = roc_auc_score(y, oof_predictions)
print(f"\n{'='*80}")
print(f"CROSS-VALIDATION RESULTS")
print(f"{'='*80}")
print(f"Overall CV AUC: {cv_score:.6f}")
print(f"Mean Fold AUC: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
print(f"Best Fold: {max(fold_scores):.6f}")
print(f"Worst Fold: {min(fold_scores):.6f}")




print(f"\n{'='*80}")
print("TOP 20 MOST IMPORTANT FEATURES")
print(f"{'='*80}")

feature_importance_agg = (feature_importance.groupby('feature')['importance']
                          .mean()
                          .sort_values(ascending=False))

for i, (feat, imp) in enumerate(feature_importance_agg.head(20).items(), 1):
    print(f"{i:2d}. {feat:45s}: {imp:10.2f}")



submission = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': test_predictions
})

submission.to_csv('submission_robust.csv', index=False)

print(f"\n{'='*80}")
print("SUBMISSION CREATED")
print(f"{'='*80}")
print(f"Filename: submission_robust.csv")
print(f"Samples: {len(submission):,}")
print(f"Prediction range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")
print(f"Mean prediction: {test_predictions.mean():.4f} (train: {y.mean():.4f})")
print(f"Std prediction: {test_predictions.std():.4f}")

# Check if predictions are reasonable
if abs(test_predictions.mean() - y.mean()) < 0.05:
    print("✓ Prediction distribution matches training")
else:
    print("⚠ Prediction distribution differs from training (possible issue)")

print(f"\n{'='*80}")
print("EXPECTED IMPROVEMENTS:")
print(f"{'='*80}")
print("1. Robust features handle train-test distribution shift")
print("2. Strong regularization prevents overfitting")
print("3. Clinical thresholds are universally applicable")
print("4. Rank-based features are distribution-independent")
print("5. Single well-tuned model beats ensemble of overfit models")
print(f"\nTarget: 0.73+ on leaderboard (up from 0.697)")
print(f"{'='*80}")








