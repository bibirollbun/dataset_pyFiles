# ==========================================
# CELL 1: Title and Introduction
# ==========================================
"""
# ğŸ�¦ Predicting Loan Payback - Playground Series S5E11
## A Comprehensive Analysis and Modeling Approach

**Competition Goal:** Predict the probability that a borrower will pay back their loan

**Evaluation Metric:** ROC AUC Score

---

### ğŸ“‹ Notebook Structure:
1. **Data Loading & Overview** ğŸ“Š
2. **Exploratory Data Analysis** ğŸ”�
3. **Feature Engineering** âš™ï¸�
4. **Model Training & Validation** ğŸ¤–
5. **Ensemble & Predictions** ğŸ�¯
6. **Submission** ğŸ“¤

---
**Author:** Your Name | **Date:** November 2025
"""


# ==========================================
# CELL 2: Import Libraries
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("âœ… All libraries imported successfully!")



# ==========================================
# CELL 3: Load Data
# ==========================================
# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

print(f"ğŸ“Š Train shape: {train.shape}")
print(f"ğŸ“Š Test shape: {test.shape}")
print(f"ğŸ“Š Submission shape: {submission.shape}")



# ==========================================
# CELL 4: Data Overview
# ==========================================
print("=" * 80)
print("TRAINING DATA OVERVIEW")
print("=" * 80)
print(train.head(10))
print("\n" + "=" * 80)
print("DATA TYPES & NULL VALUES")
print("=" * 80)
print(train.info())
print("\n" + "=" * 80)
print("STATISTICAL SUMMARY")
print("=" * 80)
print(train.describe())

# Check for missing values
print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)
missing = train.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("âœ… No missing values found!")




# ==========================================
# CELL 5: Target Variable Analysis
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Target distribution
target_counts = train['loan_paid_back'].value_counts()
axes[0].bar(target_counts.index, target_counts.values, color=['#FF6B6B', '#4ECDC4'])
axes[0].set_xlabel('Loan Paid Back', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[0].set_title('ğŸ�¯ Target Variable Distribution', fontsize=14, fontweight='bold')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Not Paid (0)', 'Paid (1)'])

for i, v in enumerate(target_counts.values):
    axes[0].text(i, v + 100, str(v), ha='center', fontweight='bold')

# Target percentage
target_pct = train['loan_paid_back'].value_counts(normalize=True) * 100
colors = ['#FF6B6B', '#4ECDC4']
explode = (0.05, 0.05)
axes[1].pie(target_pct.values, labels=['Not Paid (0)', 'Paid (1)'], autopct='%1.2f%%',
            colors=colors, explode=explode, startangle=90, textprops={'fontweight': 'bold'})
axes[1].set_title('ğŸ¥§ Target Variable Percentage', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\nğŸ“Š Class Distribution:")
print(f"   â€¢ Not Paid (0): {target_counts[0]:,} ({target_pct[0]:.2f}%)")
print(f"   â€¢ Paid (1): {target_counts[1]:,} ({target_pct[1]:.2f}%)")
print(f"\nâš–ï¸� Class Balance Ratio: {target_counts[1]/target_counts[0]:.2f}")



# ==========================================
# CELL 6: Numerical Features Analysis
# ==========================================
# Identify numerical and categorical features
numerical_features = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove('id')
numerical_features.remove('loan_paid_back')

categorical_features = train.select_dtypes(include=['object']).columns.tolist()

print(f"ğŸ”¢ Numerical Features ({len(numerical_features)}): {numerical_features}")
print(f"ğŸ�·ï¸� Categorical Features ({len(categorical_features)}): {categorical_features}")

# Correlation heatmap
print("\nğŸ“Š Computing correlation matrix...")
correlation_matrix = train[numerical_features + ['loan_paid_back']].corr()

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('ğŸ”¥ Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# Top correlations with target
target_corr = correlation_matrix['loan_paid_back'].drop('loan_paid_back').sort_values(ascending=False)
print("\nğŸ�¯ Top 10 Features Correlated with Target:")
print(target_corr.head(10))





# ==========================================
# CELL 7: Distribution of Top Numerical Features
# ==========================================
# Select top features for visualization
top_features = target_corr.abs().sort_values(ascending=False).head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, feature in enumerate(top_features):
    for target_val in [0, 1]:
        data = train[train['loan_paid_back'] == target_val][feature]
        axes[idx].hist(data, bins=50, alpha=0.6, 
                      label=f'Paid={target_val}', 
                      color=['#FF6B6B', '#4ECDC4'][target_val])
    
    axes[idx].set_xlabel(feature, fontweight='bold')
    axes[idx].set_ylabel('Frequency', fontweight='bold')
    axes[idx].set_title(f'Distribution: {feature}', fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.suptitle('ğŸ“Š Top 6 Features Distribution by Target', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()



# ==========================================
# CELL 8: Categorical Features Analysis
# ==========================================
if len(categorical_features) > 0:
    n_cat = len(categorical_features)
    n_cols = 3
    n_rows = (n_cat + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.ravel() if n_cat > 1 else [axes]
    
    for idx, feature in enumerate(categorical_features):
        # Cross-tabulation
        ct = pd.crosstab(train[feature], train['loan_paid_back'], normalize='index') * 100
        
        ct.plot(kind='bar', ax=axes[idx], color=['#FF6B6B', '#4ECDC4'], width=0.8)
        axes[idx].set_title(f'ğŸ�·ï¸� {feature} vs Target', fontweight='bold', fontsize=12)
        axes[idx].set_xlabel(feature, fontweight='bold')
        axes[idx].set_ylabel('Percentage (%)', fontweight='bold')
        axes[idx].legend(['Not Paid (0)', 'Paid (1)'], loc='best')
        axes[idx].grid(alpha=0.3, axis='y')
        axes[idx].tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for idx in range(len(categorical_features), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('ğŸ”� Categorical Features Analysis', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()
else:
    print("â„¹ï¸� No categorical features found in the dataset.")



# ==========================================
# CELL 9: Feature Engineering
# ==========================================
print("âš™ï¸� Starting Feature Engineering...")

def create_features(df):
    """Create new features for the dataset"""
    df = df.copy()
    
    # Example feature engineering (adjust based on actual columns)
    # You'll need to customize this based on the actual features in your dataset
    
    # 1. Interaction features
    if 'person_income' in df.columns and 'loan_amnt' in df.columns:
        df['income_to_loan_ratio'] = df['person_income'] / (df['loan_amnt'] + 1)
        df['loan_to_income_pct'] = (df['loan_amnt'] / (df['person_income'] + 1)) * 100
    
    # 2. Polynomial features for important columns
    if 'person_age' in df.columns:
        df['age_squared'] = df['person_age'] ** 2
        df['age_log'] = np.log1p(df['person_age'])
    
    # 3. Binning continuous features
    if 'person_income' in df.columns:
        df['income_bin'] = pd.qcut(df['person_income'], q=5, labels=False, duplicates='drop')
    
    # 4. Statistical features (if multiple related columns exist)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    if 'loan_paid_back' in numeric_cols:
        numeric_cols.remove('loan_paid_back')
    
    if len(numeric_cols) >= 3:
        df['feature_sum'] = df[numeric_cols].sum(axis=1)
        df['feature_mean'] = df[numeric_cols].mean(axis=1)
        df['feature_std'] = df[numeric_cols].std(axis=1)
        df['feature_max'] = df[numeric_cols].max(axis=1)
        df['feature_min'] = df[numeric_cols].min(axis=1)
    
    return df

# Apply feature engineering
train_fe = create_features(train)
test_fe = create_features(test)

print(f"âœ… Feature Engineering Complete!")
print(f"   â€¢ Original features: {train.shape[1]}")
print(f"   â€¢ New features: {train_fe.shape[1]}")
print(f"   â€¢ Features added: {train_fe.shape[1] - train.shape[1]}")




# ==========================================
# CELL 10: Data Preparation
# ==========================================
print("ğŸ”§ Preparing data for modeling...")

# Separate features and target
X = train_fe.drop(['id', 'loan_paid_back'], axis=1)
y = train_fe['loan_paid_back']
X_test = test_fe.drop(['id'], axis=1)

# Handle categorical features
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

if len(categorical_cols) > 0:
    print(f"ğŸ“� Encoding {len(categorical_cols)} categorical features...")
    le_dict = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        le_dict[col] = le

# Ensure all columns are aligned
X_test = X_test[X.columns]

print(f"âœ… Data prepared successfully!")
print(f"   â€¢ Training samples: {X.shape[0]:,}")
print(f"   â€¢ Test samples: {X_test.shape[0]:,}")
print(f"   â€¢ Total features: {X.shape[1]}")



# ==========================================
# CELL 11: Model Training - LightGBM
# ==========================================
import joblib
import pickle
import os

# Create directory for saving models
os.makedirs('models', exist_ok=True)

print("ğŸš€ Training LightGBM Model...")

lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_estimators': 1000,
    'random_state': 42
}

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lgbm_scores = []
lgbm_predictions = np.zeros(len(X_test))
lgbm_models = []  # Store all fold models

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    model = LGBMClassifier(**lgbm_params)
    model.fit(X_train_fold, y_train_fold, 
              eval_set=[(X_val_fold, y_val_fold)],
              callbacks=[])
    
    val_pred = model.predict_proba(X_val_fold)[:, 1]
    score = roc_auc_score(y_val_fold, val_pred)
    lgbm_scores.append(score)
    lgbm_models.append(model)  # Save model
    
    lgbm_predictions += model.predict_proba(X_test)[:, 1] / skf.n_splits
    
    # Save each fold model
    joblib.dump(model, f'models/lgbm_fold_{fold}.pkl')
    
    print(f"   Fold {fold} - ROC AUC: {score:.6f}")

print(f"\nğŸ“Š LightGBM CV Score: {np.mean(lgbm_scores):.6f} (+/- {np.std(lgbm_scores):.6f})")
print(f"ğŸ’¾ Saved 5 LightGBM models to 'models/' directory")








# ==========================================
# CELL 12: Model Training - XGBoost
# ==========================================
print("ğŸš€ Training XGBoost Model...")

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 1000,
    'random_state': 42,
    'verbosity': 0
}

xgb_scores = []
xgb_predictions = np.zeros(len(X_test))
xgb_models = []  # Store all fold models

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    model = XGBClassifier(**xgb_params)
    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_val_fold, y_val_fold)],
              verbose=False)
    
    val_pred = model.predict_proba(X_val_fold)[:, 1]
    score = roc_auc_score(y_val_fold, val_pred)
    xgb_scores.append(score)
    xgb_models.append(model)  # Save model
    
    xgb_predictions += model.predict_proba(X_test)[:, 1] / skf.n_splits
    
    # Save each fold model
    joblib.dump(model, f'models/xgb_fold_{fold}.pkl')
    
    print(f"   Fold {fold} - ROC AUC: {score:.6f}")

print(f"\nğŸ“Š XGBoost CV Score: {np.mean(xgb_scores):.6f} (+/- {np.std(xgb_scores):.6f})")
print(f"ğŸ’¾ Saved 5 XGBoost models to 'models/' directory")



# ==========================================
# CELL 13: Model Training - CatBoost
# ==========================================
print("ğŸš€ Training CatBoost Model...")

cb_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': 42,
    'verbose': False
}

cb_scores = []
cb_predictions = np.zeros(len(X_test))
cb_models = []  # Store all fold models

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    model = CatBoostClassifier(**cb_params)
    model.fit(X_train_fold, y_train_fold,
              eval_set=(X_val_fold, y_val_fold),
              verbose=False)
    
    val_pred = model.predict_proba(X_val_fold)[:, 1]
    score = roc_auc_score(y_val_fold, val_pred)
    cb_scores.append(score)
    cb_models.append(model)  # Save model
    
    cb_predictions += model.predict_proba(X_test)[:, 1] / skf.n_splits
    
    # Save each fold model
    model.save_model(f'models/catboost_fold_{fold}.cbm')
    
    print(f"   Fold {fold} - ROC AUC: {score:.6f}")

print(f"\nğŸ“Š CatBoost CV Score: {np.mean(cb_scores):.6f} (+/- {np.std(cb_scores):.6f})")
print(f"ğŸ’¾ Saved 5 CatBoost models to 'models/' directory")



# ==========================================
# CELL 14: Model Performance Comparison
# ==========================================
model_scores = {
    'LightGBM': (np.mean(lgbm_scores), np.std(lgbm_scores)),
    'XGBoost': (np.mean(xgb_scores), np.std(xgb_scores)),
    'CatBoost': (np.mean(cb_scores), np.std(cb_scores))
}

fig, ax = plt.subplots(figsize=(12, 6))

models = list(model_scores.keys())
means = [model_scores[m][0] for m in models]
stds = [model_scores[m][1] for m in models]

bars = ax.bar(models, means, yerr=stds, capsize=10, 
              color=['#FF6B6B', '#4ECDC4', '#95E1D3'], 
              edgecolor='black', linewidth=2, alpha=0.8)

ax.set_ylabel('ROC AUC Score', fontsize=12, fontweight='bold')
ax.set_title('ğŸ�† Model Performance Comparison (5-Fold CV)', fontsize=14, fontweight='bold')
ax.set_ylim([min(means) - 0.01, max(means) + 0.01])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{mean:.6f}\nÂ±{std:.6f}',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("ğŸ“Š FINAL MODEL SCORES")
print("="*60)
for model_name, (mean, std) in model_scores.items():
    print(f"{model_name:12s}: {mean:.6f} (+/- {std:.6f})")
print("="*60)



# ==========================================
# CELL 15: Ensemble Predictions
# ==========================================
print("ğŸ�¯ Creating Ensemble Predictions...")

# Simple average ensemble
ensemble_predictions = (lgbm_predictions + xgb_predictions + cb_predictions) / 3

# Weighted ensemble (weights based on CV scores)
weights = np.array([np.mean(lgbm_scores), np.mean(xgb_scores), np.mean(cb_scores)])
weights = weights / weights.sum()

weighted_ensemble = (
    lgbm_predictions * weights[0] + 
    xgb_predictions * weights[1] + 
    cb_predictions * weights[2]
)

print(f"âœ… Ensemble predictions created!")
print(f"   â€¢ Model weights: LightGBM={weights[0]:.3f}, XGBoost={weights[1]:.3f}, CatBoost={weights[2]:.3f}")
print(f"   â€¢ Prediction range: [{weighted_ensemble.min():.4f}, {weighted_ensemble.max():.4f}]")

# Visualize prediction distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(ensemble_predictions, bins=50, color='#4ECDC4', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Predicted Probability', fontweight='bold')
axes[0].set_ylabel('Frequency', fontweight='bold')
axes[0].set_title('ğŸ“Š Simple Average Ensemble Distribution', fontweight='bold')
axes[0].grid(alpha=0.3)

axes[1].hist(weighted_ensemble, bins=50, color='#FF6B6B', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Predicted Probability', fontweight='bold')
axes[1].set_ylabel('Frequency', fontweight='bold')
axes[1].set_title('ğŸ“Š Weighted Ensemble Distribution', fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()




# ==========================================
# CELL 16: Create Submission File
# ==========================================
print("ğŸ“¤ Creating submission file...")

# Use weighted ensemble for final submission
submission['loan_paid_back'] = weighted_ensemble

# Save submission
submission.to_csv('submission.csv', index=False)

print("âœ… Submission file created successfully!")
print(f"\nğŸ“‹ Submission Preview:")
print(submission.head(10))
print(f"\nğŸ“Š Submission Statistics:")
print(submission['loan_paid_back'].describe())

print("\n" + "="*60)
print("ğŸ�‰ NOTEBOOK EXECUTION COMPLETE!")
print("="*60)
print("ğŸ“� Submission file: submission.csv")
print("ğŸ�† Expected LB Score: ~{:.4f}".format(np.mean([np.mean(lgbm_scores), 
                                                       np.mean(xgb_scores), 
                                                       np.mean(cb_scores)])))
print("="*60)
print("\nğŸ’¡ Next Steps:")
print("   1. Download submission.csv")
print("   2. Submit to Kaggle")
print("   3. Check leaderboard score")
print("   4. Iterate and improve! ğŸš€")
print("="*60)



# ==========================================
# CELL 17: Model Saving Summary & Metadata
# ==========================================
print("ğŸ’¾ MODEL SAVING SUMMARY")
print("="*60)

# Save model metadata
model_metadata = {
    'competition': 'Playground Series S5E11 - Loan Payback Prediction',
    'date_trained': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'models': {
        'lgbm': {
            'params': lgbm_params,
            'cv_scores': lgbm_scores,
            'mean_score': np.mean(lgbm_scores),
            'std_score': np.std(lgbm_scores),
            'files': [f'models/lgbm_fold_{i}.pkl' for i in range(1, 6)]
        },
        'xgb': {
            'params': xgb_params,
            'cv_scores': xgb_scores,
            'mean_score': np.mean(xgb_scores),
            'std_score': np.std(xgb_scores),
            'files': [f'models/xgb_fold_{i}.pkl' for i in range(1, 6)]
        },
        'catboost': {
            'params': cb_params,
            'cv_scores': cb_scores,
            'mean_score': np.mean(cb_scores),
            'std_score': np.std(cb_scores),
            'files': [f'models/catboost_fold_{i}.cbm' for i in range(1, 6)]
        }
    },
    'ensemble_weights': {
        'lgbm': float(weights[0]),
        'xgb': float(weights[1]),
        'catboost': float(weights[2])
    },
    'feature_names': X.columns.tolist(),
    'n_features': X.shape[1]
}

# Save metadata as JSON
import json
with open('models/model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=4)

print(f"âœ… Saved {len(lgbm_models)} LightGBM models")
print(f"âœ… Saved {len(xgb_models)} XGBoost models")
print(f"âœ… Saved {len(cb_models)} CatBoost models")
print(f"âœ… Saved model metadata: models/model_metadata.json")
print(f"\nğŸ“� Total files saved: {len(lgbm_models) + len(xgb_models) + len(cb_models) + 1}")
print("="*60)

# List all saved files
print("\nğŸ“‚ Saved Model Files:")
print("-" * 60)
for model_type in ['lgbm', 'xgb', 'catboost']:
    print(f"\n{model_type.upper()}:")
    for fold in range(1, 6):
        if model_type == 'catboost':
            filename = f'models/catboost_fold_{fold}.cbm'
        else:
            filename = f'models/{model_type}_fold_{fold}.pkl'
        
        if os.path.exists(filename):
            size = os.path.getsize(filename) / (1024 * 1024)  # Convert to MB
            print(f"   âœ“ {filename} ({size:.2f} MB)")

print(f"\n   âœ“ models/model_metadata.json")
print("="*60)




# ==========================================
# CELL 18: Load Saved Models (Example)
# ==========================================
print("ğŸ“¥ EXAMPLE: HOW TO LOAD SAVED MODELS")
print("="*60)

print("""
# To load and use the saved models later:

# 1. Load LightGBM models
import joblib
lgbm_model_fold1 = joblib.load('models/lgbm_fold_1.pkl')
predictions_lgbm = lgbm_model_fold1.predict_proba(X_test)[:, 1]

# 2. Load XGBoost models
xgb_model_fold1 = joblib.load('models/xgb_fold_1.pkl')
predictions_xgb = xgb_model_fold1.predict_proba(X_test)[:, 1]

# 3. Load CatBoost models
from catboost import CatBoostClassifier
cb_model_fold1 = CatBoostClassifier()
cb_model_fold1.load_model('models/catboost_fold_1.cbm')
predictions_cb = cb_model_fold1.predict_proba(X_test)[:, 1]

# 4. Load all models and create ensemble
import json
with open('models/model_metadata.json', 'r') as f:
    metadata = json.load(f)

ensemble_weights = metadata['ensemble_weights']
print(f"Ensemble weights: {ensemble_weights}")

# 5. Recreate predictions from all folds
all_lgbm_preds = []
for fold in range(1, 6):
    model = joblib.load(f'models/lgbm_fold_{fold}.pkl')
    preds = model.predict_proba(X_test)[:, 1]
    all_lgbm_preds.append(preds)

lgbm_ensemble = np.mean(all_lgbm_preds, axis=0)
""")

print("="*60)

# Example: Actually load one model to verify
print("\nğŸ”� Verification: Loading one model as example...")
try:
    test_model = joblib.load('models/lgbm_fold_1.pkl')
    print("âœ… Successfully loaded models/lgbm_fold_1.pkl")
    print(f"   Model type: {type(test_model).__name__}")
    print(f"   Number of features: {test_model.n_features_in_}")
except Exception as e:
    print(f"â�Œ Error loading model: {e}")

print("\n" + "="*60)
print("ğŸ�¯ All models saved and ready for future use!")
print("="*60)

