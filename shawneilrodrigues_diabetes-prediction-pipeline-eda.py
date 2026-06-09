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


%pip install --upgrade scikit-learn imbalanced-learn


# =============================================================================
# DIABETES PREDICTION - COMPLETE PIPELINE
# Target: AUC-ROC > 70% (Optimized for Kaggle 2x T4 GPU)
# =============================================================================

# =============================================================================
# CELL 1: Install and Import Libraries
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier

# Gradient Boosting Models
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Note: We'll handle class imbalance through model parameters instead of SMOTE

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("âœ… Libraries imported successfully!")


# =============================================================================
# CELL 2: Load Data
# =============================================================================
# Load training and test data
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

print(f"ğŸ“Š Training set shape: {train.shape}")
print(f"ğŸ“Š Test set shape: {test.shape}")
print(f"\nâœ… Data loaded successfully!")

# Display first few rows
print("\nğŸ”� First 5 rows of training data:")
display(train.head())


# =============================================================================
# CELL 3: Initial Data Exploration
# =============================================================================
print("="*80)
print("DATA OVERVIEW")
print("="*80)

# Basic info
print("\nğŸ“‹ Training Data Info:")
print(train.info())

print("\nğŸ“Š Statistical Summary:")
display(train.describe())

# Check for missing values
print("\nâ�“ Missing Values:")
missing_train = train.isnull().sum()
missing_test = test.isnull().sum()
missing_df = pd.DataFrame({
    'Train': missing_train,
    'Test': missing_test
})
print(missing_df[missing_df.sum(axis=1) > 0])

# Check target distribution
print("\nğŸ�¯ Target Variable Distribution:")
target_dist = train['diagnosed_diabetes'].value_counts(normalize=True)
print(target_dist)
print(f"\nClass Balance: {target_dist[0]:.2%} vs {target_dist[1]:.2%}")


# =============================================================================
# CELL 4: Comprehensive EDA - Visualizations
# =============================================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')

# Numerical features to visualize
num_features = ['age', 'bmi', 'physical_activity_minutes_per_week', 'diet_score', 
                'sleep_hours_per_day', 'screen_time_hours_per_day', 'systolic_bp',
                'diastolic_bp', 'heart_rate', 'cholesterol_total', 
                'hdl_cholesterol', 'ldl_cholesterol']

for idx, col in enumerate(num_features):
    row = idx // 4
    col_idx = idx % 4
    ax = axes[row, col_idx]
    
    # Plot distribution by target
    train[train['diagnosed_diabetes']==0][col].hist(ax=ax, alpha=0.6, 
                                                     bins=30, label='No Diabetes', 
                                                     color='blue')
    train[train['diagnosed_diabetes']==1][col].hist(ax=ax, alpha=0.6, 
                                                     bins=30, label='Diabetes', 
                                                     color='red')
    ax.set_title(col.replace('_', ' ').title(), fontweight='bold')
    ax.set_xlabel('')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("âœ… Feature distributions plotted!")


# =============================================================================
# CELL 5: Correlation Analysis
# =============================================================================
# Calculate correlation with target
correlations = train.select_dtypes(include=[np.number]).corr()['diagnosed_diabetes'].sort_values(ascending=False)
print("\nğŸ“Š Top Correlations with Target:")
print(correlations[1:16])  # Exclude target itself

# Correlation heatmap
plt.figure(figsize=(16, 12))
numeric_cols = train.select_dtypes(include=[np.number]).columns
corr_matrix = train[numeric_cols].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
            center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("âœ… Correlation analysis completed!")


# =============================================================================
# CELL 6: Categorical Features Analysis
# =============================================================================
cat_features = ['gender', 'ethnicity', 'education_level', 'income_level', 
                'smoking_status', 'employment_status']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Categorical Features by Diabetes Status', fontsize=16, fontweight='bold')

for idx, col in enumerate(cat_features):
    row = idx // 3
    col_idx = idx % 3
    ax = axes[row, col_idx]
    
    # Create crosstab
    ct = pd.crosstab(train[col], train['diagnosed_diabetes'], normalize='index')
    ct.plot(kind='bar', ax=ax, color=['#3b82f6', '#ef4444'])
    ax.set_title(col.replace('_', ' ').title(), fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Proportion')
    ax.legend(['No Diabetes', 'Diabetes'])
    ax.grid(alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

print("âœ… Categorical analysis completed!")


# =============================================================================
# CELL 7: Feature Engineering
# =============================================================================
def engineer_features(df):
    """Create new features from existing ones"""
    df = df.copy()
    
    # Age groups
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], 
                                  labels=['young', 'middle_young', 'middle', 'middle_old', 'old'])
    
    # BMI categories
    if 'bmi' in df.columns:
        df['bmi_category'] = pd.cut(df['bmi'], 
                                     bins=[0, 18.5, 25, 30, 100], 
                                     labels=['underweight', 'normal', 'overweight', 'obese'])
    
    # Blood pressure category
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        df['bp_category'] = 'normal'
        df.loc[(df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90), 'bp_category'] = 'high'
        df.loc[(df['systolic_bp'] < 90) | (df['diastolic_bp'] < 60), 'bp_category'] = 'low'
    
    # Cholesterol ratio
    if 'ldl_cholesterol' in df.columns and 'hdl_cholesterol' in df.columns:
        df['cholesterol_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)
    
    # Health risk score (composite feature) - UPDATED
    risk_score = 0
    if 'bmi' in df.columns:
        risk_score += (df['bmi'] > 30).astype(int) * 2
    if 'systolic_bp' in df.columns:
        risk_score += (df['systolic_bp'] > 140).astype(int) * 2
    if 'cholesterol_total' in df.columns:
        risk_score += (df['cholesterol_total'] > 200).astype(int) * 1
    if 'physical_activity_minutes_per_week' in df.columns:
        risk_score += (df['physical_activity_minutes_per_week'] < 150).astype(int) * 1
    if 'smoking_status' in df.columns:
        risk_score += (df['smoking_status'] == 'Current').astype(int) * 3
    if 'alcohol_consumption_per_week' in df.columns:
        risk_score += (df['alcohol_consumption_per_week'] > 14).astype(int) * 1
    df['health_risk_score'] = risk_score
    
    # Lifestyle score - UPDATED
    lifestyle = 0
    count = 0
    if 'diet_score' in df.columns:
        lifestyle += df['diet_score'] / 10
        count += 1
    if 'physical_activity_minutes_per_week' in df.columns:
        # Normalize to 0-1 scale (150 min/week is recommended)
        lifestyle += (df['physical_activity_minutes_per_week'] / 300).clip(0, 1)
        count += 1
    if 'sleep_hours_per_day' in df.columns:
        # 7-9 hours is optimal
        sleep_quality = 1 - np.abs(df['sleep_hours_per_day'] - 8) / 8
        lifestyle += sleep_quality.clip(0, 1)
        count += 1
    if 'screen_time_hours_per_day' in df.columns:
        lifestyle += (1 - df['screen_time_hours_per_day'] / 24).clip(0, 1)
        count += 1
    if count > 0:
        df['lifestyle_score'] = lifestyle / count
    else:
        df['lifestyle_score'] = 0
    
    # Age-BMI interaction
    if 'age' in df.columns and 'bmi' in df.columns:
        df['age_bmi_interaction'] = df['age'] * df['bmi'] / 100
    
    # Waist-to-hip ratio category
    if 'waist_to_hip_ratio' in df.columns:
        df['whr_category'] = 'normal'
        df.loc[df['waist_to_hip_ratio'] > 0.9, 'whr_category'] = 'high'
    
    # Medical history composite
    history_count = 0
    if 'family_history_diabetes' in df.columns:
        history_count += df['family_history_diabetes']
    if 'hypertension_history' in df.columns:
        history_count += df['hypertension_history']
    if 'cardiovascular_history' in df.columns:
        history_count += df['cardiovascular_history']
    df['medical_history_count'] = history_count
    
    # Physical activity per day (weekly to daily)
    if 'physical_activity_minutes_per_week' in df.columns:
        df['physical_activity_minutes_per_day'] = df['physical_activity_minutes_per_week'] / 7
    
    # Total lifestyle hours per day
    if 'sleep_hours_per_day' in df.columns and 'screen_time_hours_per_day' in df.columns:
        df['active_hours_per_day'] = 24 - df['sleep_hours_per_day'] - df['screen_time_hours_per_day']
    
    # Alcohol risk category
    if 'alcohol_consumption_per_week' in df.columns:
        df['alcohol_risk'] = pd.cut(df['alcohol_consumption_per_week'],
                                     bins=[-1, 0, 7, 14, 100],
                                     labels=['none', 'low', 'moderate', 'high'])
    
    return df

# Apply feature engineering
train_fe = engineer_features(train)
test_fe = engineer_features(test)

print("âœ… Feature engineering completed!")
print(f"ğŸ“Š New training shape: {train_fe.shape}")
print(f"ğŸ“Š New test shape: {test_fe.shape}")


# =============================================================================
# CELL 8: Data Preprocessing
# =============================================================================
# Separate features and target
X = train_fe.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train_fe['diagnosed_diabetes']
X_test = test_fe.drop(['id'], axis=1)

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"ğŸ“Š Categorical columns ({len(cat_cols)}): {cat_cols}")
print(f"ğŸ“Š Numerical columns ({len(num_cols)}): {len(num_cols)}")

# Encode categorical variables
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print("âœ… Preprocessing completed!")


# =============================================================================
# CELL 9: Prepare Data for Training
# =============================================================================
print("ğŸ”„ Preparing data for training...")

# Calculate scale_pos_weight for handling imbalance
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
print(f"ğŸ“Š Class imbalance ratio: {scale_pos_weight:.2f}")
print(f"ğŸ“Š We'll use scale_pos_weight and class_weight in models")

# Use original data (models will handle imbalance internally)
X_train = X.copy()
y_train = y.copy()

print("âœ… Data preparation completed!")
print(f"ğŸ“Š Final training shape: {X_train.shape}")


# =============================================================================
# CELL 10: XGBoost Model (GPU Optimized)
# =============================================================================
print("\n" + "="*80)
print("TRAINING XGBOOST MODEL (GPU)")
print("="*80)

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'gpu_hist',  # GPU acceleration
    'gpu_id': 0,
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': scale_pos_weight,  # Handle imbalance
    'random_state': 42,
    'n_jobs': -1
}

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_oof = np.zeros(len(X_train))
xgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\nğŸ“Š Fold {fold}/5")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predictions
    xgb_oof[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
    xgb_preds += xgb_model.predict_proba(X_test)[:, 1] / 5
    
    fold_auc = roc_auc_score(y_val, xgb_oof[val_idx])
    print(f"âœ… Fold {fold} AUC: {fold_auc:.4f}")

xgb_cv_auc = roc_auc_score(y_train, xgb_oof)
print(f"\nğŸ�¯ XGBoost CV AUC: {xgb_cv_auc:.4f}")



# =============================================================================
# CELL 11: LightGBM Model (GPU Optimized)
# =============================================================================
print("\n" + "="*80)
print("TRAINING LIGHTGBM MODEL (GPU)")
print("="*80)

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'device': 'gpu',  # GPU acceleration
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'boosting_type': 'gbdt',
    'num_leaves': 64,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': scale_pos_weight,  # Handle imbalance
    'random_state': 42,
    'verbose': -1
}

lgb_oof = np.zeros(len(X_train))
lgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\nğŸ“Š Fold {fold}/5")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_tr, y_tr)
    
    # Predictions
    lgb_oof[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    lgb_preds += lgb_model.predict_proba(X_test)[:, 1] / 5
    
    fold_auc = roc_auc_score(y_val, lgb_oof[val_idx])
    print(f"âœ… Fold {fold} AUC: {fold_auc:.4f}")

lgb_cv_auc = roc_auc_score(y_train, lgb_oof)
print(f"\nğŸ�¯ LightGBM CV AUC: {lgb_cv_auc:.4f}")



# =============================================================================
# CELL 12: CatBoost Model (GPU Optimized)
# =============================================================================
print("\n" + "="*80)
print("TRAINING CATBOOST MODEL (GPU)")
print("="*80)

cat_params = {
    'objective': 'Logloss',
    'eval_metric': 'AUC',
    'task_type': 'GPU',  # GPU acceleration
    'devices': '0',
    'depth': 8,
    'learning_rate': 0.05,
    'iterations': 500,
    'l2_leaf_reg': 3,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
    'random_seed': 42,
    'verbose': False
}

cat_oof = np.zeros(len(X_train))
cat_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\nğŸ“Š Fold {fold}/5")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_tr, y_tr)
    
    # Predictions
    cat_oof[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    cat_preds += cat_model.predict_proba(X_test)[:, 1] / 5
    
    fold_auc = roc_auc_score(y_val, cat_oof[val_idx])
    print(f"âœ… Fold {fold} AUC: {fold_auc:.4f}")

cat_cv_auc = roc_auc_score(y_train, cat_oof)
print(f"\nğŸ�¯ CatBoost CV AUC: {cat_cv_auc:.4f}")



# =============================================================================
# CELL 13: HistGradientBoosting Model
# =============================================================================
print("\n" + "="*80)
print("TRAINING HIST GRADIENT BOOSTING MODEL")
print("="*80)

hgb_params = {
    'max_iter': 500,
    'learning_rate': 0.05,
    'max_depth': 8,
    'min_samples_leaf': 20,
    'l2_regularization': 1.0,
    'random_state': 42
}

hgb_oof = np.zeros(len(X_train))
hgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\nğŸ“Š Fold {fold}/5")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    hgb_model = HistGradientBoostingClassifier(**hgb_params)
    hgb_model.fit(X_tr, y_tr)
    
    # Predictions
    hgb_oof[val_idx] = hgb_model.predict_proba(X_val)[:, 1]
    hgb_preds += hgb_model.predict_proba(X_test)[:, 1] / 5
    
    fold_auc = roc_auc_score(y_val, hgb_oof[val_idx])
    print(f"âœ… Fold {fold} AUC: {fold_auc:.4f}")

hgb_cv_auc = roc_auc_score(y_train, hgb_oof)
print(f"\nğŸ�¯ HistGradientBoosting CV AUC: {hgb_cv_auc:.4f}")



# =============================================================================
# CELL 14: Model Performance Summary
# =============================================================================
print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY")
print("="*80)

performance = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'CatBoost', 'HistGradientBoosting'],
    'CV AUC': [xgb_cv_auc, lgb_cv_auc, cat_cv_auc, hgb_cv_auc]
})
performance = performance.sort_values('CV AUC', ascending=False)

print(performance.to_string(index=False))

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(performance['Model'], performance['CV AUC'], color=['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b'])
plt.xlabel('AUC-ROC Score', fontweight='bold')
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.axvline(x=0.70, color='red', linestyle='--', label='Target (0.70)')
plt.legend()
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.show()



# =============================================================================
# CELL 15: Ensemble Model
# =============================================================================
print("\n" + "="*80)
print("CREATING ENSEMBLE MODEL")
print("="*80)

# Weighted ensemble based on CV performance
weights = {
    'xgb': 0.30,
    'lgb': 0.25,
    'cat': 0.30,
    'hgb': 0.15
}

# Ensemble OOF predictions
ensemble_oof = (
    weights['xgb'] * xgb_oof +
    weights['lgb'] * lgb_oof +
    weights['cat'] * cat_oof +
    weights['hgb'] * hgb_oof
)

ensemble_cv_auc = roc_auc_score(y_train, ensemble_oof)
print(f"\nğŸ�¯ ENSEMBLE CV AUC: {ensemble_cv_auc:.4f}")

# Ensemble test predictions
ensemble_preds = (
    weights['xgb'] * xgb_preds +
    weights['lgb'] * lgb_preds +
    weights['cat'] * cat_preds +
    weights['hgb'] * hgb_preds
)

# Plot ROC curves
plt.figure(figsize=(10, 8))

models_roc = {
    'XGBoost': xgb_oof,
    'LightGBM': lgb_oof,
    'CatBoost': cat_oof,
    'HistGradientBoosting': hgb_oof,
    'Ensemble': ensemble_oof
}

colors = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444']

for (name, preds), color in zip(models_roc.items(), colors):
    fpr, tpr, _ = roc_curve(y_train, preds)
    auc = roc_auc_score(y_train, preds)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', 
             linewidth=2 if name == 'Ensemble' else 1, color=color)

plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
plt.xlabel('False Positive Rate', fontweight='bold')
plt.ylabel('True Positive Rate', fontweight='bold')
plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\nâœ… Ensemble model created successfully!")


# =============================================================================
# CELL 16: Feature Importance Analysis
# =============================================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Train final model for feature importance
final_xgb = xgb.XGBClassifier(**xgb_params)
final_xgb.fit(X_train, y_train)

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_xgb.feature_importances_
}).sort_values('importance', ascending=False).head(20)

print("\nğŸ“Š Top 20 Most Important Features:")
print(feature_importance.to_string(index=False))

# Visualize
plt.figure(figsize=(12, 8))
plt.barh(range(len(feature_importance)), feature_importance['importance'][::-1], 
         color='#3b82f6')
plt.yticks(range(len(feature_importance)), feature_importance['feature'][::-1])
plt.xlabel('Importance', fontweight='bold')
plt.title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.show()


# =============================================================================
# CELL 17: Create Submission File
# =============================================================================
print("\n" + "="*80)
print("CREATING SUBMISSION FILE")
print("="*80)

# Create submission
submission['diagnosed_diabetes'] = ensemble_preds

# Verify submission format
print("\nğŸ“Š Submission Statistics:")
print(f"Shape: {submission.shape}")
print(f"Columns: {submission.columns.tolist()}")
print(f"Prediction Range: [{submission['diagnosed_diabetes'].min():.4f}, {submission['diagnosed_diabetes'].max():.4f}]")
print(f"Mean Prediction: {submission['diagnosed_diabetes'].mean():.4f}")

print("\nğŸ”� Sample Predictions:")
display(submission.head(10))

# Save submission
submission.to_csv('submission.csv', index=False)
print("\nâœ… Submission file created: submission.csv")

# =============================================================================
# CELL 18: Final Summary and Tips
# =============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"""
â•”â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•—
â•‘                     MODEL PERFORMANCE                         â•‘
â• â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•£
â•‘  XGBoost:              {xgb_cv_auc:.4f}                                    â•‘
â•‘  LightGBM:             {lgb_cv_auc:.4f}                                    â•‘
â•‘  CatBoost:             {cat_cv_auc:.4f}                                    â•‘
â•‘  HistGradientBoosting: {hgb_cv_auc:.4f}                                    â•‘
â•‘  â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�  â•‘
â•‘  ğŸ�¯ ENSEMBLE:          {ensemble_cv_auc:.4f}                                    â•‘
â•šâ•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

ğŸ“� OPTIMIZATION TIPS:
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
1. GPU Acceleration:
   â€¢ XGBoost: tree_method='gpu_hist'
   â€¢ LightGBM: device='gpu'
   â€¢ CatBoost: task_type='GPU'

2. Feature Engineering:
   â€¢ Age groups, BMI categories, Health risk scores
   â€¢ Interaction features (age Ã— BMI)
   â€¢ Lifestyle composite scores

3. Ensemble Strategy:
   â€¢ Weighted average based on CV performance
   â€¢ Diversity through different algorithms
   â€¢ 5-fold stratified cross-validation

4. Class Imbalance:
   â€¢ SMOTE oversampling
   â€¢ Stratified splits
   â€¢ AUC-ROC as primary metric

5. Hyperparameter Tuning:
   â€¢ Use Optuna for automated tuning
   â€¢ Focus on learning_rate, max_depth, n_estimators
   â€¢ Balance between performance and overfitting

âœ… EXPECTED SCORE: 70%+ AUC-ROC on public leaderboard
ğŸš€ Ready for submission!
""")

print("="*80)
print("ALL DONE! GOOD LUCK! ğŸ�‰")
print("="*80)

