import numpy as np # linear algebra
import pandas as pd # data processing

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
train.info()
train.head()


test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
test.info()
test.head()


sample = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
sample.info()
sample.head()


# Target analysis
print(f"\nTarget distribution:")
print(train['diagnosed_diabetes'].value_counts())
print(f"Diabetes rate: {train['diagnosed_diabetes'].mean():.2%}")


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_data(df, is_train=True, le_dict=None):
    """Preprocess and encode categorical variables"""
    df = df.copy()
    
    # Categorical columns
    cat_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
                'smoking_status', 'employment_status']
    
    if is_train:
        # Label encoding for categorical variables
        le_dict = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le
        return df, le_dict
    else:
        # For test set, use fitted encoders
        for col in cat_cols:
            le = le_dict[col]
            df[col] = df[col].map(lambda s: le.classes_.tolist().index(s) if s in le.classes_ else -1)
        return df

print("\nPreprocessing data...")
train_processed, label_encoders = preprocess_data(train, is_train=True)
test_processed = preprocess_data(test, is_train=False, le_dict=label_encoders)



# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """Create domain-specific features for diabetes prediction"""
    df = df.copy()
    
    # 1. BMI categories (WHO classification)
    df['bmi_category'] = pd.cut(df['bmi'], 
                                 bins=[0, 18.5, 25, 30, 100],
                                 labels=[0, 1, 2, 3]).astype(int)
    
    # 2. Blood pressure categories
    df['bp_category'] = 0
    df.loc[(df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90), 'bp_category'] = 1
    
    # 3. Age groups (diabetes risk increases with age)
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(int)
    
    # 4. Cholesterol ratios (important for diabetes risk)
    df['cholesterol_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)
    
    # 5. Combined risk factors
    df['health_risk_score'] = (
        df['family_history_diabetes'] + 
        df['hypertension_history'] + 
        df['cardiovascular_history']
    )
    
    # 6. Lifestyle score
    df['lifestyle_score'] = (
        (df['physical_activity_minutes_per_week'] / 150) +  # WHO recommendation
        (df['sleep_hours_per_day'] / 8) +  # Optimal sleep
        df['diet_score'] - 
        (df['alcohol_consumption_per_week'] / 7) - 
        (df['screen_time_hours_per_day'] / 2)
    )
    
    # 7. Metabolic syndrome indicators
    df['metabolic_risk'] = (
        (df['bmi'] > 30).astype(int) +
        (df['waist_to_hip_ratio'] > 0.9).astype(int) +
        (df['triglycerides'] > 150).astype(int) +
        (df['hdl_cholesterol'] < 40).astype(int) +
        ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 85)).astype(int)
    )
    
    # 8. Interaction features
    df['bmi_age'] = df['bmi'] * df['age']
    df['bmi_waist'] = df['bmi'] * df['waist_to_hip_ratio']
    
    # 9. Polynomial features for key health indicators
    df['bmi_squared'] = df['bmi'] ** 2
    df['age_squared'] = df['age'] ** 2
    
    return df

print("\nEngineering features...")
train_fe = engineer_features(train_processed)
test_fe = engineer_features(test_processed)

print(f"Train shape after FE: {train_fe.shape}")
print(f"New features created: {train_fe.shape[1] - train.shape[1]}")


# ============================================================================
# PREPARE TRAIN/TEST DATA
# ============================================================================

# Features to use
feature_cols = [col for col in train_fe.columns if col not in ['id', 'diagnosed_diabetes']]

X = train_fe[feature_cols]
y = train_fe['diagnosed_diabetes']
X_test = test_fe[feature_cols]

print(f"\nFinal feature count: {len(feature_cols)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ============================================================================
# MODEL TRAINING
# ============================================================================

print("\n" + "="*80)
print("MODEL TRAINING & EVALUATION")
print("="*80)

models = {}
predictions = {}

# 1. Logistic Regression
print("\n[1/4] Logistic Regression...")
lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
lr_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"  CV ROC AUC: {lr_scores.mean():.5f} ± {lr_scores.std():.5f}")
lr.fit(X_scaled, y)
models['lr'] = lr
predictions['lr'] = lr.predict_proba(X_test_scaled)[:, 1]

# 2. Random Forest
print("\n[2/4] Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)
rf_scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"  CV ROC AUC: {rf_scores.mean():.5f} ± {rf_scores.std():.5f}")
rf.fit(X, y)
models['rf'] = rf
predictions['rf'] = rf.predict_proba(X_test)[:, 1]

# 3. Gradient Boosting
print("\n[3/4] Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
gb_scores = cross_val_score(gb, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"  CV ROC AUC: {gb_scores.mean():.5f} ± {gb_scores.std():.5f}")
gb.fit(X, y)
models['gb'] = gb
predictions['gb'] = gb.predict_proba(X_test)[:, 1]

# 4. XGBoost (if available)
try:
    import xgboost as xgb
    print("\n[4/4] XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc'
    )
    xgb_scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"  CV ROC AUC: {xgb_scores.mean():.5f} ± {xgb_scores.std():.5f}")
    xgb_model.fit(X, y)
    models['xgb'] = xgb_model
    predictions['xgb'] = xgb_model.predict_proba(X_test)[:, 1]
except:
    print("\n[4/4] XGBoost not available")


# ============================================================================
# ENSEMBLE PREDICTION
# ============================================================================

print("\n" + "="*80)
print("CREATING ENSEMBLE")
print("="*80)

# Weighted average based on CV performance
all_scores = {
    'lr': lr_scores.mean(),
    'rf': rf_scores.mean(),
    'gb': gb_scores.mean()
}

if 'xgb' in predictions:
    all_scores['xgb'] = xgb_scores.mean()

# Print model scores
print("\nModel Performance:")
for name, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name.upper()}: {score:.5f}")

# Weight by performance
total_score = sum(all_scores.values())
weights = {k: v/total_score for k, v in all_scores.items()}

print("\nEnsemble Weights:")
for name, weight in weights.items():
    print(f"  {name.upper()}: {weight:.4f}")

# Create ensemble
ensemble_pred = np.zeros(len(X_test))
for name, weight in weights.items():
    ensemble_pred += predictions[name] * weight


# ============================================================================
# GENERATE SUBMISSION
# ============================================================================

print("\n" + "="*80)
print("GENERATING SUBMISSION")
print("="*80)

submission = pd.DataFrame({
    'id': test_fe['id'],
    'diagnosed_diabetes': ensemble_pred
})

submission.to_csv('submission.csv', index=False)

print(f"\n✓ Submission saved: submission.csv")
print(f"  Shape: {submission.shape}")
print(f"\nPrediction Statistics:")
print(submission['diagnosed_diabetes'].describe())

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(submission['diagnosed_diabetes'], bins=50, edgecolor='black')
axes[0].set_xlabel('Predicted Probability')
axes[0].set_ylabel('Count')
axes[0].set_title('Prediction Distribution')

axes[1].hist(submission['diagnosed_diabetes'], bins=50, cumulative=True, density=True, edgecolor='black')
axes[1].set_xlabel('Predicted Probability')
axes[1].set_ylabel('Cumulative Probability')
axes[1].set_title('Cumulative Distribution')

plt.tight_layout()
plt.savefig('predictions.png', dpi=100)
plt.show()

