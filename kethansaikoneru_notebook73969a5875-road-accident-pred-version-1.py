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




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')


print("="*70)
print("STEP 1: LOADING AND EXPLORING DATA")
print("="*70)

# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTarget distribution:")
print(train['accident_risk'].describe())

# Check for missing values
print(f"\nMissing values in train:")
print(train.isnull().sum().sum())
print(f"Missing values in test:")
print(test.isnull().sum().sum())

# Identify column types
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()

if 'id' in num_cols:
    num_cols.remove('id')
if 'accident_risk' in num_cols:
    num_cols.remove('accident_risk')

print(f"\nCategorical columns: {len(cat_cols)}")
print(f"Numeric columns: {len(num_cols)}")

# Calculate correlations for numeric columns
if len(num_cols) > 0:
    correlations = train[num_cols + ['accident_risk']].corr()['accident_risk'].sort_values(ascending=False)
    print(f"\nTop 5 correlations with target:")
    print(correlations.head(6))

# Visualize target distribution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(train['accident_risk'], bins=50, edgecolor='black')
plt.title('Target Distribution')
plt.xlabel('accident_risk')

plt.subplot(1, 2, 2)
train['accident_risk'].plot(kind='box')
plt.title('Target Boxplot')
plt.tight_layout()
plt.savefig('target_distribution.png')
print("\n✓ Saved visualization: target_distribution.png")


print("\n" + "="*70)
print("STEP 2: ADVANCED FEATURE ENGINEERING")
print("="*70)

def create_advanced_features(df, is_train=True):
    """Create statistical and interaction features"""
    df = df.copy()
    
    # Get numeric columns
    num_cols_df = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in num_cols_df:
        num_cols_df.remove('id')
    if 'accident_risk' in num_cols_df:
        num_cols_df.remove('accident_risk')
    
    # Statistical aggregation features
    if len(num_cols_df) > 1:
        df['num_mean'] = df[num_cols_df].mean(axis=1)
        df['num_std'] = df[num_cols_df].std(axis=1)
        df['num_min'] = df[num_cols_df].min(axis=1)
        df['num_max'] = df[num_cols_df].max(axis=1)
        df['num_range'] = df['num_max'] - df['num_min']
        df['num_median'] = df[num_cols_df].median(axis=1)
        df['num_skew'] = df[num_cols_df].skew(axis=1)
        df['num_kurt'] = df[num_cols_df].kurtosis(axis=1)
        print(f"  ✓ Added 8 statistical features")
    
    return df

# Apply feature engineering
print("\nCreating features for training data...")
train_fe = create_advanced_features(train, is_train=True)

print("Creating features for test data...")
test_fe = create_advanced_features(test, is_train=False)

# Separate target and IDs
target = train_fe['accident_risk'].values
train_ids = train_fe['id'].values
test_ids = test_fe['id'].values

# Drop id and target
X_train = train_fe.drop(['id', 'accident_risk'], axis=1)
X_test = test_fe.drop(['id'], axis=1)

print(f"\nFeatures before encoding: {X_train.shape[1]}")



print("\n" + "="*70)
print("STEP 3: CATEGORICAL ENCODING")
print("="*70)

# Get categorical columns
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# 1. Frequency Encoding
print("\n1. Frequency Encoding:")
for col in cat_cols:
    freq = X_train[col].value_counts()
    X_train[f'{col}_freq'] = X_train[col].map(freq).fillna(0)
    X_test[f'{col}_freq'] = X_test[col].map(freq).fillna(0)
    print(f"  ✓ {col}")

# 2. Label Encoding
print("\n2. Label Encoding:")
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    le_dict[col] = le
    print(f"  ✓ {col}")

# 3. K-Fold Target Encoding (prevents overfitting)
print("\n3. K-Fold Target Encoding:")

def kfold_target_encoding(X_train, y_train, X_test, col, n_splits=5):
    """Target encoding with cross-validation to prevent leakage"""
    train_encoded = np.zeros(len(X_train))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Encode training data with CV
    for train_idx, val_idx in kf.split(X_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train[train_idx]
        
        # Calculate mean target per category
        target_means = pd.Series(y_tr, index=X_tr.index).groupby(X_tr[col]).mean()
        global_mean = y_tr.mean()
        
        # Map to validation set
        train_encoded[val_idx] = X_train.iloc[val_idx][col].map(target_means).fillna(global_mean)
    
    # Encode test data with full training data
    target_means_full = pd.Series(y_train, index=X_train.index).groupby(X_train[col]).mean()
    test_encoded = X_test[col].map(target_means_full).fillna(y_train.mean())
    
    return train_encoded, test_encoded

# Apply target encoding to each categorical column
for col in cat_cols:
    train_te, test_te = kfold_target_encoding(X_train, target, X_test, col)
    X_train[f'{col}_target_enc'] = train_te
    X_test[f'{col}_target_enc'] = test_te
    print(f"  ✓ {col}")

print(f"\nFinal feature count: {X_train.shape[1]}")

# Align columns
X_test = X_test[X_train.columns]
print("✓ Train and test features aligned")


print("\n" + "="*70)
print("STEP 4: MULTI-SEED MODEL TRAINING")
print("="*70)

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

def train_with_multiple_seeds(X, y, X_test, model_class, params, seeds, model_name):
    """Train same model with different random seeds"""
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(X))
    test_predictions = np.zeros(len(X_test))
    
    print(f"\nTraining {model_name} with {len(seeds)} seeds:")
    
    for seed in seeds:
        params['random_state'] = seed
        
        fold_oof = np.zeros(len(X))
        fold_test = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model = model_class(**params)
            model.fit(X_tr, y_tr)
            
            fold_oof[val_idx] = model.predict(X_val)
            fold_test += model.predict(X_test) / n_splits
        
        oof_predictions += fold_oof / len(seeds)
        test_predictions += fold_test / len(seeds)
        
        seed_rmse = np.sqrt(mean_squared_error(y, fold_oof))
        print(f"  Seed {seed}: {seed_rmse:.5f}")
    
    avg_cv = np.sqrt(mean_squared_error(y, oof_predictions))
    print(f"  Average CV: {avg_cv:.5f}")
    
    return oof_predictions, test_predictions, avg_cv

# Define seeds for diversity
seeds = [42, 123, 456, 789, 2024]

# Model 1: LightGBM
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': 7,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 2000,
    'verbose': -1,
    'random_state': 42
}

lgb_oof, lgb_test, lgb_cv = train_with_multiple_seeds(
    X_train, target, X_test, LGBMRegressor, lgb_params, seeds, "LightGBM"
)

# Model 2: CatBoost
cat_params = {
    'iterations': 2000,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3,
    'verbose': 0,
    'random_state': 42
}

cat_oof, cat_test, cat_cv = train_with_multiple_seeds(
    X_train, target, X_test, CatBoostRegressor, cat_params, seeds, "CatBoost"
)

# Model 3: XGBoost
xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.03,
    'max_depth': 6,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 2000,
    'verbosity': 0,
    'random_state': 42
}

xgb_oof, xgb_test, xgb_cv = train_with_multiple_seeds(
    X_train, target, X_test, XGBRegressor, xgb_params, seeds, "XGBoost"
)


print("\n" + "="*70)
print("STEP 5: PSEUDO-LABELING")
print("="*70)

# Average predictions from all models
avg_test_pred = (lgb_test + cat_test + xgb_test) / 3

# Select high-confidence predictions (very low or very high)
confidence_threshold = 0.15
confident_low = avg_test_pred < confidence_threshold
confident_high = avg_test_pred > (1 - confidence_threshold)
confident_mask = confident_low | confident_high

n_pseudo = confident_mask.sum()
pseudo_percent = (n_pseudo / len(X_test)) * 100

print(f"High-confidence predictions: {n_pseudo} ({pseudo_percent:.1f}%)")
print(f"  Low risk (<{confidence_threshold}): {confident_low.sum()}")
print(f"  High risk (>{1-confidence_threshold}): {confident_high.sum()}")

if n_pseudo > 100:  # Only if we have enough pseudo-labels
    # Combine original training with pseudo-labeled test data
    X_train_pseudo = pd.concat([X_train, X_test[confident_mask]], ignore_index=True)
    y_train_pseudo = np.concatenate([target, avg_test_pred[confident_mask]])
    
    print("\nRetraining LightGBM with pseudo-labels...")
    
    # Train on expanded dataset
    lgb_pseudo_model = LGBMRegressor(**lgb_params)
    lgb_pseudo_model.fit(X_train_pseudo, y_train_pseudo)
    lgb_pseudo_test = lgb_pseudo_model.predict(X_test)
    
    print("✓ Pseudo-labeled model trained")
else:
    print("Not enough confident predictions, skipping pseudo-labeling")
    lgb_pseudo_test = lgb_test.copy()


print("\n" + "="*70)
print("STEP 6: STACKING WITH META-MODEL")
print("="*70)

from sklearn.linear_model import Ridge

# Stack OOF predictions for training meta-model
stack_train = np.column_stack([lgb_oof, cat_oof, xgb_oof])

# Stack test predictions
stack_test = np.column_stack([lgb_test, cat_test, xgb_test])

print("Training Ridge meta-model...")
meta_model = Ridge(alpha=1.0)
meta_model.fit(stack_train, target)

# Predictions
stack_train_pred = meta_model.predict(stack_train)
stack_test_pred = meta_model.predict(stack_test)

# Evaluate
stack_cv = np.sqrt(mean_squared_error(target, stack_train_pred))

print(f"Stacking CV RMSE: {stack_cv:.5f}")
print(f"Meta-model coefficients:")
print(f"  LightGBM: {meta_model.coef_[0]:.4f}")
print(f"  CatBoost: {meta_model.coef_[1]:.4f}")
print(f"  XGBoost:  {meta_model.coef_[2]:.4f}")


print("\n" + "="*70)
print("STEP 7: RANK AVERAGING")
print("="*70)

def rank_average_predictions(pred_list):
    """Convert predictions to ranks and average them"""
    ranked_preds = []
    for pred in pred_list:
        # Convert to ranks (0 to 1 scale)
        ranks = rankdata(pred) / len(pred)
        ranked_preds.append(ranks)
    
    # Average the ranks
    return np.mean(ranked_preds, axis=0)

# Apply rank averaging
rank_avg_test = rank_average_predictions([
    lgb_test, 
    cat_test, 
    xgb_test, 
    lgb_pseudo_test,
    stack_test_pred
])

print("✓ Rank averaging applied to 5 predictions")
print(f"  Rank average stats: mean={rank_avg_test.mean():.5f}, std={rank_avg_test.std():.5f}")


print("\n" + "="*70)
print("STEP 8: ENSEMBLE WEIGHT OPTIMIZATION")
print("="*70)

from scipy.optimize import minimize

# Collect all OOF predictions
all_oof = np.column_stack([
    lgb_oof,
    cat_oof,
    xgb_oof,
    lgb_oof,  # Use same OOF for pseudo
    stack_train_pred,
    rank_average_predictions([lgb_oof, cat_oof, xgb_oof, lgb_oof, stack_train_pred])
])

# Collect all test predictions
all_test = np.column_stack([
    lgb_test,
    cat_test,
    xgb_test,
    lgb_pseudo_test,
    stack_test_pred,
    rank_avg_test
])

def weighted_rmse(weights, oof_preds, y_true):
    """RMSE for weighted ensemble"""
    weights = np.abs(weights)
    weights = weights / weights.sum()
    ensemble_pred = np.dot(oof_preds, weights)
    return np.sqrt(mean_squared_error(y_true, ensemble_pred))

print("Optimizing weights with scipy minimize...")

# Initial guess: equal weights
initial_weights = np.ones(all_oof.shape[1]) / all_oof.shape[1]

# Optimize
result = minimize(
    weighted_rmse,
    initial_weights,
    args=(all_oof, target),
    method='Nelder-Mead',
    options={'maxiter': 10000}
)

# Get optimal weights
optimal_weights = np.abs(result.x)
optimal_weights = optimal_weights / optimal_weights.sum()

# Calculate CV with optimal weights
optimal_cv = weighted_rmse(optimal_weights, all_oof, target)

print("\nOptimal Weights:")
model_names = ['LightGBM', 'CatBoost', 'XGBoost', 'Pseudo-LGB', 'Stacking', 'RankAvg']
for name, weight in zip(model_names, optimal_weights):
    print(f"  {name:12s}: {weight:.4f}")

print(f"\nOptimized Ensemble CV: {optimal_cv:.5f}")


print("\n" + "="*70)
print("STEP 9: FINAL PREDICTIONS")
print("="*70)

# Create final ensemble
final_predictions = np.dot(all_test, optimal_weights)

# Post-processing: Clip to training data range
train_min = target.min()
train_max = target.max()
final_predictions = np.clip(final_predictions, train_min, train_max)

print(f"Final prediction statistics:")
print(f"  Mean:   {final_predictions.mean():.5f}")
print(f"  Std:    {final_predictions.std():.5f}")
print(f"  Min:    {final_predictions.min():.5f}")
print(f"  Max:    {final_predictions.max():.5f}")
print(f"  Clipped to training range: [{train_min:.5f}, {train_max:.5f}]")

# ============================================================================
# STEP 10: Create and Save Submission
# ============================================================================
print("\n" + "="*70)
print("STEP 10: CREATE SUBMISSION FILE")
print("="*70)

submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': final_predictions
})

# Save main submission
submission.to_csv('submission.csv', index=False)
print("✓ Main submission saved: submission.csv")

# Also save individual model predictions for comparison
pd.DataFrame({'id': test_ids, 'accident_risk': lgb_test}).to_csv('submission_lgb.csv', index=False)
pd.DataFrame({'id': test_ids, 'accident_risk': cat_test}).to_csv('submission_cat.csv', index=False)
pd.DataFrame({'id': test_ids, 'accident_risk': xgb_test}).to_csv('submission_xgb.csv', index=False)
pd.DataFrame({'id': test_ids, 'accident_risk': stack_test_pred}).to_csv('submission_stack.csv', index=False)

print("✓ Individual model submissions also saved")




