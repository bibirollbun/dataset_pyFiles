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
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ===========================
# LOAD DATA
# ===========================
print("Loading data...")
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
test['accident_risk'] = 0.5

# Load original data for target encoding
orig = []
for k in [2, 10, 100]:
    df = pd.read_csv(f"/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_{k}k.csv")
    orig.append(df)
orig = pd.concat(orig, axis=0)
orig['id'] = np.arange(len(orig)) + test['id'].max() + 1
orig = orig[train.columns]

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Original data shape: {orig.shape}")

# Combine for feature engineering
combine = pd.concat([train, test, orig], axis=0, ignore_index=True)
print(f"Combined shape: {combine.shape}")

# ===========================
# FEATURE ENGINEERING
# ===========================
FEATURES = list(orig.columns[1:-1])
TARGET = orig.columns[-1]

print("\n" + "="*50)
print("CREATING BAYESIAN BASELINE FEATURE")
print("="*50)

# Optimal Bayesian solution feature
def f(X):
    """
    Bayesian formula for accident risk prediction
    Based on weighted combination of key risk factors
    """
    return \
    0.3 * X["curvature"] + \
    0.2 * (X["lighting"] == "night").astype(int) + \
    0.1 * (X["weather"] != "clear").astype(int) + \
    0.2 * (X["speed_limit"] >= 60).astype(int) + \
    0.1 * (X["num_reported_accidents"] > 2).astype(int)

def clip(f):
    """Apply truncated normal distribution clipping"""
    def clip_f(X):
        sigma = 0.05
        mu = f(X)
        a, b = -mu/sigma, (1-mu)/sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu*(Phi_b-Phi_a) + sigma*(phi_a-phi_b) + 1 - Phi_b
    return clip_f

# Create Bayesian feature
z = clip(f)(combine)
combine["y"] = z.values
FEATURES.append("y")

# Identify categorical and numerical features
CATS = []
NUMS = []
for c in FEATURES:
    if combine[c].dtype == 'object':
        CATS.append(c)
    else:
        NUMS.append(c)

print(f"\nCategorical features: {CATS}")
print(f"Numerical features: {NUMS}")

# ===========================
# MINIMAL HIGH-IMPACT FEATURES
# ===========================
print("\n" + "="*50)
print("ENGINEERING MINIMAL HIGH-IMPACT FEATURES")
print("="*50)

# Based on feature importance analysis, we use only the most impactful features

# Squared versions of top features
combine['curvature_sq'] = combine['curvature'] ** 2
combine['num_reported_accidents_sq'] = combine['num_reported_accidents'] ** 2

# Risk score composite feature
combine['risk_score'] = (
    combine['curvature'] * 0.3 +
    (combine['lighting'] == 'night').astype(int) * 0.2 +
    (combine['weather'] != 'clear').astype(int) * 0.1 +
    (combine['speed_limit'] >= 60).astype(int) * 0.2
)
combine['risk_score_sq'] = combine['risk_score'] ** 2

# Most important ratio feature
combine['curvature_div_num_reported_accidents'] = combine['curvature'] / (combine['num_reported_accidents'] + 1)

# Add engineered features to lists
FEATURES.extend(['curvature_sq', 'num_reported_accidents_sq', 'risk_score', 
                'risk_score_sq', 'curvature_div_num_reported_accidents'])
NUMS.extend(['curvature_sq', 'num_reported_accidents_sq', 'risk_score', 
            'risk_score_sq', 'curvature_div_num_reported_accidents'])

print(f"Total features after engineering: {len(FEATURES)}")

# Label encode categorical features
for c in CATS:
    combine[c], _ = combine[c].factorize()
    combine[c] = combine[c].astype('int32')

# Split back to train/test/orig
train = combine.iloc[:len(train)].copy()
test = combine.iloc[len(train):len(train) + len(test)].copy()
orig = combine.iloc[-len(orig):].copy()

# ===========================
# TARGET ENCODING
# ===========================
print("\n" + "="*50)
print("APPLYING TARGET ENCODING")
print("="*50)

# Target encode only the most valuable features
TE_FEATURES = ['y', 'curvature', 'num_reported_accidents', 'risk_score', 'risk_score_sq',
               'curvature_div_num_reported_accidents', 'curvature_sq', 'weather', 'holiday',
               'num_lanes', 'public_road', 'time_of_day', 'lighting', 'road_type']

TE = []
for c in TE_FEATURES:
    if c in FEATURES:
        tmp = orig.groupby(c)[TARGET].mean()
        n = f"TE_{c}"
        tmp.name = n
        train = train.merge(tmp, on=c, how='left')
        test = test.merge(tmp, on=c, how='left')
        TE.append(n)

# Fill NaN values with global mean
global_mean = orig[TARGET].mean()
for te_col in TE:
    train[te_col].fillna(global_mean, inplace=True)
    test[te_col].fillna(global_mean, inplace=True)

print(f"Created {len(TE)} target encoding features")
print(f"Total features: {len(FEATURES + TE)}")

# Mark categorical features for LightGBM
categorical_features = [i for i, col in enumerate(FEATURES + TE) if col in CATS]

# ===========================
# VISUALIZATION 1: BAYESIAN FORMULA AND RESIDUALS
# ===========================
print("\n" + "="*50)
print("VISUALIZING BAYESIAN PREDICTIONS AND RESIDUALS")
print("="*50)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Bayesian predictions vs actual
ax1 = axes[0, 0]
scatter = ax1.scatter(train['y'], train[TARGET], alpha=0.5, s=10)
ax1.plot([0, 1], [0, 1], 'r--', lw=2)
ax1.set_xlabel('Bayesian Predictions')
ax1.set_ylabel('Actual Target')
ax1.set_title('Bayesian Predictions vs Actual Values')
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals distribution
ax2 = axes[0, 1]
residuals = train[TARGET] - train['y']
ax2.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
ax2.axvline(x=0, color='red', linestyle='--', lw=2)
ax2.set_xlabel('Residuals (Actual - Bayesian)')
ax2.set_ylabel('Frequency')
ax2.set_title(f'Residual Distribution\nMean: {residuals.mean():.4f}, Std: {residuals.std():.4f}')
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals vs Bayesian predictions
ax3 = axes[1, 0]
ax3.scatter(train['y'], residuals, alpha=0.5, s=10)
ax3.axhline(y=0, color='red', linestyle='--', lw=2)
ax3.set_xlabel('Bayesian Predictions')
ax3.set_ylabel('Residuals')
ax3.set_title('Residuals vs Bayesian Predictions')
ax3.grid(True, alpha=0.3)

# Plot 4: Bayesian formula components
ax4 = axes[1, 1]
components = {
    'Curvature (30%)': 0.3 * train['curvature'].mean(),
    'Night Lighting (20%)': 0.2 * (train['lighting'] == train['lighting'].mode()[0]).mean(),
    'Bad Weather (10%)': 0.1 * (train['weather'] != train['weather'].mode()[0]).mean(),
    'High Speed (20%)': 0.2 * (train['speed_limit'] >= 60).mean(),
    'Many Accidents (10%)': 0.1 * (train['num_reported_accidents'] > 2).mean()
}
bars = ax4.bar(range(len(components)), list(components.values()))
ax4.set_xticks(range(len(components)))
ax4.set_xticklabels(list(components.keys()), rotation=45, ha='right')
ax4.set_ylabel('Average Contribution')
ax4.set_title('Bayesian Formula Components')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ===========================
# MODEL TRAINING WITH OPTIMIZED PARAMETERS
# ===========================
print("\n" + "="*50)
print("TRAINING LIGHTGBM WITH OPTIMIZED PARAMETERS")
print("="*50)

FOLDS = 7
SEED = 42

# Best parameters from Optuna optimization
best_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.0073315692,
    'num_leaves': 90,
    'max_depth': 12,
    'min_child_samples': 17,
    'min_child_weight': 0.0007127304,
    'subsample': 0.828766,
    'subsample_freq': 1,
    'colsample_bytree': 0.433137,
    'reg_alpha': 0.117006,
    'reg_lambda': 2.460118,
    'min_split_gain': 0.0,
    'max_bin': 227,
    'extra_trees': False,
    'path_smooth': 0.0,
    'verbosity': -1,
    'random_state': SEED,
    'n_jobs': -1,
    'force_col_wise': True
}

# Cross-validation
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
fold_scores = []
feature_importance_df = pd.DataFrame()

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(train)):
    print(f"\n{'='*25}")
    print(f"Fold {fold+1}/{FOLDS}")
    print(f"{'='*25}")
    
    # Prepare data - note the residual approach
    X_train = train.iloc[train_idx][FEATURES + TE]
    y_train = train.iloc[train_idx][TARGET] - train.iloc[train_idx]['y']  # Train on residuals
    
    X_valid = train.iloc[val_idx][FEATURES + TE]
    y_valid = train.iloc[val_idx][TARGET] - train.iloc[val_idx]['y']
    y_valid_baseline = train.iloc[val_idx]['y'].values
    
    X_test = test[FEATURES + TE]
    y_test_baseline = test['y'].values
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_features
    )
    
    valid_data = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=categorical_features,
        reference=train_data
    )
    
    # Train model
    model = lgb.train(
        params=best_params,
        train_set=train_data,
        num_boost_round=100000,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.log_evaluation(period=1000),
            lgb.early_stopping(stopping_rounds=200)
        ]
    )
    
    # Predictions - add back the Bayesian baseline
    oof_preds[val_idx] = model.predict(X_valid, num_iteration=model.best_iteration) + y_valid_baseline
    test_preds += (model.predict(X_test, num_iteration=model.best_iteration) + y_test_baseline) / FOLDS
    
    # Calculate fold score
    fold_rmse = np.sqrt(mean_squared_error(train.iloc[val_idx][TARGET], oof_preds[val_idx]))
    fold_scores.append(fold_rmse)
    print(f"Fold {fold+1} RMSE: {fold_rmse:.6f}")
    
    # Store feature importance
    importance = pd.DataFrame()
    importance['feature'] = FEATURES + TE
    importance['importance'] = model.feature_importance(importance_type='gain')
    importance['fold'] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, importance])

# Calculate overall CV score
cv_rmse = np.sqrt(mean_squared_error(train[TARGET].values, oof_preds))
print(f"\n{'='*50}")
print(f"CV SCORES BY FOLD:")
print(f"{'='*50}")
for i, score in enumerate(fold_scores):
    print(f"Fold {i+1}: {score:.6f}")
print(f"\nMean: {np.mean(fold_scores):.6f}")
print(f"Std:  {np.std(fold_scores):.6f}")
print(f"\nOVERALL CV RMSE: {cv_rmse:.6f}")
print(f"{'='*50}")

# ===========================
# VISUALIZATION 2: FEATURE IMPORTANCE
# ===========================
print("\n" + "="*50)
print("VISUALIZING FEATURE IMPORTANCE")
print("="*50)

# Calculate mean importance across folds
mean_importance = feature_importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Plot 1: Top 20 features bar plot
top_features = mean_importance.head(20)
ax1.barh(range(len(top_features)), top_features.values, color='skyblue')
ax1.set_yticks(range(len(top_features)))
ax1.set_yticklabels(top_features.index)
ax1.invert_yaxis()
ax1.set_xlabel('Average Gain')
ax1.set_title('Top 20 Features by Importance')
ax1.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, v in enumerate(top_features.values):
    ax1.text(v + max(top_features.values)*0.01, i, f'{v:.1f}', va='center')

# Plot 2: Feature importance by category
feature_categories = {
    'Original': [],
    'Bayesian': [],
    'Engineered': [],
    'Target Encoded': []
}

for feat in mean_importance.index:
    if feat.startswith('TE_'):
        feature_categories['Target Encoded'].append(mean_importance[feat])
    elif feat == 'y':
        feature_categories['Bayesian'].append(mean_importance[feat])
    elif feat in ['curvature_sq', 'num_reported_accidents_sq', 'risk_score', 
                  'risk_score_sq', 'curvature_div_num_reported_accidents']:
        feature_categories['Engineered'].append(mean_importance[feat])
    else:
        feature_categories['Original'].append(mean_importance[feat])

category_importance = {k: sum(v) for k, v in feature_categories.items()}
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
wedges, texts, autotexts = ax2.pie(category_importance.values(), labels=category_importance.keys(), 
                                    autopct='%1.1f%%', colors=colors, startangle=90)
ax2.set_title('Feature Importance by Category')

plt.tight_layout()
plt.show()

# Show top features with values
print("\nTop 15 Most Important Features:")
print("="*40)
for i, (feat, imp) in enumerate(mean_importance.head(15).items()):
    print(f"{i+1:2d}. {feat:35s} {imp:10.2f}")

# ===========================
# SAVE PREDICTIONS AND SUBMISSION
# ===========================
print("\n" + "="*50)
print("SAVING PREDICTIONS AND SUBMISSION")
print("="*50)

from datetime import datetime

# Generate timestamp and model name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = "LightGBM_BayesianResidual"
cv_score_str = f"{cv_rmse:.5f}".replace(".", "_")

# Save OOF predictions
oof_filename = f"oof_{model_name}_{timestamp}_CV{cv_score_str}.npy"
np.save(oof_filename, oof_preds)
print(f"OOF predictions saved: {oof_filename}")

# Save test predictions
test_filename = f"test_{model_name}_{timestamp}_CV{cv_score_str}.npy"
np.save(test_filename, test_preds)
print(f"Test predictions saved: {test_filename}")

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': test_preds
})

submission.to_csv('submission.csv', index=False)
print(f"\nSubmission saved: submission.csv")
print(f"Shape: {submission.shape}")
print(f"\nPrediction statistics:")
print(f"Mean: {test_preds.mean():.6f}")
print(f"Std:  {test_preds.std():.6f}")
print(f"Min:  {test_preds.min():.6f}")
print(f"Max:  {test_preds.max():.6f}")

# Final message
print("\n" + "="*50)
print("COMPLETE!")
print("="*50)
print(f"Model: LightGBM with Bayesian Residuals")
print(f"Features: Minimal High-Impact Set")
print(f"CV RMSE: {cv_rmse:.6f}")
print("\nFiles created:")
print(f"  - {oof_filename}")
print(f"  - {test_filename}")
print(f"  - submission.csv")
print("="*50)

