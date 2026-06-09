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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTarget distribution:")
print(train['accident_risk'].describe())

# Check for missing values
print(f"\nMissing values in train:\n{train.isnull().sum()}")
print(f"\nMissing values in test:\n{test.isnull().sum()}")

# Basic EDA
print(f"\nFeature types:")
print(train.dtypes)

# Separate numeric and categorical columns
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")
print(f"\nCategorical columns: {categorical_cols}")

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
plt.show()

# Correlation with target (only numeric columns)
if 'accident_risk' in train.columns:
    correlations = train[numeric_cols].corr()['accident_risk'].sort_values(ascending=False)
    print(f"\nTop 10 correlations with target:")
    print(correlations.head(10))
    print(f"\nBottom 10 correlations with target:")
    print(correlations.tail(10))

# Check unique values in categorical columns
if len(categorical_cols) > 0:
    print(f"\nUnique values in categorical columns:")
    for col in categorical_cols[:5]:  # Show first 5
        print(f"{col}: {train[col].nunique()} unique values")
        if train[col].nunique() < 10:
            print(f"  Values: {train[col].unique()[:10]}")


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_features(df, is_train=True, top_features=None):
    """Create additional features"""
    df = df.copy()
    
    # Separate numeric and categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if 'id' in num_cols:
        num_cols.remove('id')
    if 'accident_risk' in num_cols:
        num_cols.remove('accident_risk')
    
    # Statistical features for numeric columns
    if len(num_cols) > 1:
        df['num_mean'] = df[num_cols].mean(axis=1)
        df['num_std'] = df[num_cols].std(axis=1)
        df['num_min'] = df[num_cols].min(axis=1)
        df['num_max'] = df[num_cols].max(axis=1)
        df['num_range'] = df['num_max'] - df['num_min']
        df['num_skew'] = df[num_cols].skew(axis=1)
    
    # Polynomial features - use same features for both train and test
    if top_features is not None and len(num_cols) > 0:
        for i, col1 in enumerate(top_features):
            if col1 not in df.columns:
                continue
                
            df[f'{col1}_squared'] = df[col1] ** 2
            df[f'{col1}_cubed'] = df[col1] ** 3
            df[f'{col1}_sqrt'] = np.sqrt(np.abs(df[col1]))
            
            for col2 in top_features[i+1:]:
                if col2 not in df.columns:
                    continue
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-5)
    
    return df

# Get numeric columns from train data
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
if 'id' in numeric_cols:
    numeric_cols.remove('id')
if 'accident_risk' in numeric_cols:
    numeric_cols.remove('accident_risk')

# Calculate correlations (only if we have numeric columns)
if len(numeric_cols) > 0:
    correlations = train[numeric_cols + ['accident_risk']].corr()['accident_risk'].sort_values(ascending=False)
    top_features = correlations.head(4).index.tolist()
    if 'accident_risk' in top_features:
        top_features.remove('accident_risk')
    top_features = [f for f in top_features if f in train.columns][:3]
    print(f"Top features for polynomial expansion: {top_features}")
else:
    top_features = None
    print("No numeric columns found for correlation analysis")

# Apply feature engineering with SAME top_features for both
print("\nApplying feature engineering to train...")
train_fe = create_features(train, is_train=True, top_features=top_features)
print(f"Train shape after FE: {train_fe.shape}")

print("\nApplying feature engineering to test...")
test_fe = create_features(test, is_train=False, top_features=top_features)
print(f"Test shape after FE: {test_fe.shape}")

# Prepare features and target
target = train_fe['accident_risk'].values
train_ids = train_fe['id'].values
test_ids = test_fe['id'].values

# Drop id and target
X_train = train_fe.drop(['id', 'accident_risk'], axis=1)
X_test = test_fe.drop(['id'], axis=1)

print(f"\nBefore alignment:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# IMPORTANT: Make sure test has same columns as train
# Add missing columns to test with zeros
missing_cols = set(X_train.columns) - set(X_test.columns)
if len(missing_cols) > 0:
    print(f"\nAdding {len(missing_cols)} missing columns to test:")
    for col in missing_cols:
        X_test[col] = 0
        print(f"  - {col}")

# Remove extra columns from test
extra_cols = set(X_test.columns) - set(X_train.columns)
if len(extra_cols) > 0:
    print(f"\nRemoving {len(extra_cols)} extra columns from test:")
    for col in extra_cols:
        print(f"  - {col}")
    X_test = X_test.drop(columns=extra_cols)

# Reorder test columns to match train
X_test = X_test[X_train.columns]

# Encode categorical variables
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
le_dict = {}

if len(cat_cols) > 0:
    print(f"\nEncoding {len(cat_cols)} categorical columns...")
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        le_dict[col] = le
        print(f"  - {col}: {len(le.classes_)} unique values")

print(f"\nFinal training shape: {X_train.shape}")
print(f"Final test shape: {X_test.shape}")
print(f"Columns match: {list(X_train.columns) == list(X_test.columns)}")
print(f"Shape match: {X_train.shape[1] == X_test.shape[1]}")

# Verify no issues
assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch!"
assert list(X_train.columns) == list(X_test.columns), "Column order mismatch!"
print("\n✓ Train and test datasets are perfectly aligned!")

# Display sample
print("\nSample of prepared data:")
print(X_train.head())


def get_cv_scores(model, X, y, n_splits=5):
    """Get CV scores and OOF predictions"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_tr, y_tr)
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        
        fold_score = np.sqrt(mean_squared_error(y_val, val_preds))
        cv_scores.append(fold_score)
        print(f"Fold {fold+1} RMSE: {fold_score:.5f}")
    
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    print(f"\nMean CV RMSE: {mean_cv:.5f} (+/- {std_cv:.5f})")
    
    return oof_preds, cv_scores, mean_cv


from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

# Model 1: LightGBM
print("=" * 50)
print("Training LightGBM")
print("=" * 50)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 7,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 1000,
    'random_state': 42,
    'verbose': -1
}

lgb_model = LGBMRegressor(**lgb_params)
lgb_oof, _, lgb_cv = get_cv_scores(lgb_model, X_train, target, n_splits=5)

# Model 2: CatBoost
print("\n" + "=" * 50)
print("Training CatBoost")
print("=" * 50)

cat_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'min_data_in_leaf': 20,
    'random_strength': 0.5,
    'bagging_temperature': 0.2,
    'random_seed': 42,
    'verbose': 0
}

cat_model = CatBoostRegressor(**cat_params)
cat_oof, _, cat_cv = get_cv_scores(cat_model, X_train, target, n_splits=5)

# Model 3: XGBoost
print("\n" + "=" * 50)
print("Training XGBoost")
print("=" * 50)

xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 1000,
    'random_state': 42,
    'verbosity': 0
}

xgb_model = XGBRegressor(**xgb_params)
xgb_oof, _, xgb_cv = get_cv_scores(xgb_model, X_train, target, n_splits=5)

# Model 4: Random Forest
print("\n" + "=" * 50)
print("Training Random Forest")
print("=" * 50)

rf_params = {
    'n_estimators': 500,
    'max_depth': 12,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

rf_model = RandomForestRegressor(**rf_params)
rf_oof, _, rf_cv = get_cv_scores(rf_model, X_train, target, n_splits=5)

# Model 5: Extra Trees
print("\n" + "=" * 50)
print("Training Extra Trees")
print("=" * 50)

et_params = {
    'n_estimators': 500,
    'max_depth': 12,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'max_features': 'sqrt',
    'random_state': 123,
    'n_jobs': -1
}

et_model = ExtraTreesRegressor(**et_params)
et_oof, _, et_cv = get_cv_scores(et_model, X_train, target, n_splits=5)

# Summary
print("\n" + "=" * 50)
print("MODEL SUMMARY")
print("=" * 50)
print(f"LightGBM CV: {lgb_cv:.5f}")
print(f"CatBoost CV: {cat_cv:.5f}")
print(f"XGBoost CV:  {xgb_cv:.5f}")
print(f"RandomForest CV: {rf_cv:.5f}")
print(f"ExtraTrees CV: {et_cv:.5f}")


import optuna
from optuna.samplers import TPESampler

def objective_lgb(trial):
    """Optimize LightGBM hyperparameters"""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'n_estimators': 1000,
        'random_state': 42,
        'verbose': -1
    }
    
    model = LGBMRegressor(**params)
    _, _, cv_score = get_cv_scores(model, X_train, target, n_splits=3)
    
    return cv_score

# Run optimization
print("Starting Optuna optimization for LightGBM...")
study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study.optimize(objective_lgb, n_trials=50, show_progress_bar=True)

print(f"\nBest CV Score: {study.best_value:.5f}")
print(f"Best Parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Train final model with best parameters
best_lgb_params = study.best_params
best_lgb_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 1000,
    'random_state': 42,
    'verbose': -1
})

optimized_lgb = LGBMRegressor(**best_lgb_params)
opt_lgb_oof, _, opt_lgb_cv = get_cv_scores(optimized_lgb, X_train, target, n_splits=5)

print(f"\nOptimized LightGBM CV: {opt_lgb_cv:.5f}")
print(f"Original LightGBM CV: {lgb_cv:.5f}")
print(f"Improvement: {lgb_cv - opt_lgb_cv:.5f}")


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def get_test_predictions(model, X_train, y_train, X_test, n_splits=5, model_name="Model"):
    """Get test predictions using CV averaging"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    test_preds = np.zeros(len(X_test))
    
    print(f"\nGenerating predictions for {model_name}...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        
        model.fit(X_tr, y_tr)
        fold_preds = model.predict(X_test)
        test_preds += fold_preds / n_splits
        print(f"  Fold {fold+1}/{n_splits} completed")
    
    return test_preds

# Generate predictions for all models
print("="*60)
print("GENERATING TEST PREDICTIONS FOR ALL MODELS")
print("="*60)

# LightGBM
from lightgbm import LGBMRegressor
lgb_model_final = LGBMRegressor(**lgb_params)
lgb_test = get_test_predictions(lgb_model_final, X_train, target, X_test, 
                                n_splits=5, model_name="LightGBM")

# CatBoost
from catboost import CatBoostRegressor
cat_model_final = CatBoostRegressor(**cat_params)
cat_test = get_test_predictions(cat_model_final, X_train, target, X_test,
                               n_splits=5, model_name="CatBoost")

# XGBoost
from xgboost import XGBRegressor
xgb_model_final = XGBRegressor(**xgb_params)
xgb_test = get_test_predictions(xgb_model_final, X_train, target, X_test,
                               n_splits=5, model_name="XGBoost")

# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_model_final = RandomForestRegressor(**rf_params)
rf_test = get_test_predictions(rf_model_final, X_train, target, X_test,
                              n_splits=5, model_name="RandomForest")

# Extra Trees
from sklearn.ensemble import ExtraTreesRegressor
et_model_final = ExtraTreesRegressor(**et_params)
et_test = get_test_predictions(et_model_final, X_train, target, X_test,
                              n_splits=5, model_name="ExtraTrees")

# Optimized LightGBM (if you ran Step 5 Optuna)
if 'best_lgb_params' in globals():
    optimized_lgb_final = LGBMRegressor(**best_lgb_params)
    opt_lgb_test = get_test_predictions(optimized_lgb_final, X_train, target, X_test,
                                       n_splits=5, model_name="Optimized LightGBM")
else:
    print("\nSkipping Optimized LightGBM (run Step 5 first)")
    opt_lgb_test = lgb_test.copy()  # Use regular LGB as fallback

print("\n" + "="*60)
print("SAVING INDIVIDUAL PREDICTIONS")
print("="*60)

# Save individual predictions
pd.DataFrame({'id': test_ids, 'accident_risk': lgb_test}).to_csv('lgb_submission.csv', index=False)
print("✓ Saved: lgb_submission.csv")

pd.DataFrame({'id': test_ids, 'accident_risk': cat_test}).to_csv('cat_submission.csv', index=False)
print("✓ Saved: cat_submission.csv")

pd.DataFrame({'id': test_ids, 'accident_risk': xgb_test}).to_csv('xgb_submission.csv', index=False)
print("✓ Saved: xgb_submission.csv")

pd.DataFrame({'id': test_ids, 'accident_risk': rf_test}).to_csv('rf_submission.csv', index=False)
print("✓ Saved: rf_submission.csv")

pd.DataFrame({'id': test_ids, 'accident_risk': et_test}).to_csv('et_submission.csv', index=False)
print("✓ Saved: et_submission.csv")

if 'best_lgb_params' in globals():
    pd.DataFrame({'id': test_ids, 'accident_risk': opt_lgb_test}).to_csv('opt_lgb_submission.csv', index=False)
    print("✓ Saved: opt_lgb_submission.csv")

print("\n✓ All individual model predictions saved successfully!")

# Quick stats
print("\n" + "="*60)
print("PREDICTION STATISTICS")
print("="*60)
print(f"LightGBM    - Mean: {lgb_test.mean():.5f}, Std: {lgb_test.std():.5f}")
print(f"CatBoost    - Mean: {cat_test.mean():.5f}, Std: {cat_test.std():.5f}")
print(f"XGBoost     - Mean: {xgb_test.mean():.5f}, Std: {xgb_test.std():.5f}")
print(f"RandomForest- Mean: {rf_test.mean():.5f}, Std: {rf_test.std():.5f}")
print(f"ExtraTrees  - Mean: {et_test.mean():.5f}, Std: {et_test.std():.5f}")
if 'best_lgb_params' in globals():
    print(f"Opt LightGBM- Mean: {opt_lgb_test.mean():.5f}, Std: {opt_lgb_test.std():.5f}")


from scipy.optimize import minimize, differential_evolution

# Stack OOF predictions
oof_predictions = np.column_stack([
    lgb_oof, cat_oof, xgb_oof, rf_oof, et_oof, opt_lgb_oof
])

# Stack test predictions
test_predictions = np.column_stack([
    lgb_test, cat_test, xgb_test, rf_test, et_test, opt_lgb_test
])

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def ensemble_rmse(weights, oof_preds, target):
    """Calculate RMSE for weighted ensemble"""
    weights = np.array(weights) / np.sum(weights)  # Normalize
    ensemble_pred = np.dot(oof_preds, weights)
    return rmse(target, ensemble_pred)

# Method 1: Scipy minimize
print("Optimizing weights with Scipy minimize...")
initial_weights = np.ones(oof_predictions.shape[1]) / oof_predictions.shape[1]
bounds = [(0, 1) for _ in range(oof_predictions.shape[1])]

result = minimize(
    ensemble_rmse,
    initial_weights,
    args=(oof_predictions, target),
    method='Nelder-Mead',
    bounds=bounds,
    options={'maxiter': 10000}
)

scipy_weights = result.x / np.sum(result.x)
scipy_score = ensemble_rmse(scipy_weights, oof_predictions, target)

print(f"\nScipy optimized weights:")
model_names = ['LightGBM', 'CatBoost', 'XGBoost', 'RandomForest', 'ExtraTrees', 'Opt_LGB']
for name, weight in zip(model_names, scipy_weights):
    print(f"  {name}: {weight:.4f}")
print(f"Ensemble CV RMSE: {scipy_score:.5f}")

# Method 2: Differential Evolution (more robust)
print("\nOptimizing weights with Differential Evolution...")

def de_objective(weights):
    return ensemble_rmse(weights, oof_predictions, target)

bounds_de = [(0, 1) for _ in range(oof_predictions.shape[1])]
result_de = differential_evolution(
    de_objective,
    bounds_de,
    seed=42,
    maxiter=1000,
    polish=True
)

de_weights = result_de.x / np.sum(result_de.x)
de_score = ensemble_rmse(de_weights, oof_predictions, target)

print(f"\nDifferential Evolution optimized weights:")
for name, weight in zip(model_names, de_weights):
    print(f"  {name}: {weight:.4f}")
print(f"Ensemble CV RMSE: {de_score:.5f}")

# Choose best weights
if de_score < scipy_score:
    best_weights = de_weights
    best_score = de_score
    method = "Differential Evolution"
else:
    best_weights = scipy_weights
    best_score = scipy_score
    method = "Scipy"

print(f"\nBest method: {method}")
print(f"Best Ensemble CV RMSE: {best_score:.5f}")


# Create optimized ensemble
final_ensemble_oof = np.dot(oof_predictions, best_weights)
final_ensemble_test = np.dot(test_predictions, best_weights)

# Verify OOF score
final_cv_score = rmse(target, final_ensemble_oof)
print(f"\nFinal Ensemble CV RMSE: {final_cv_score:.5f}")

# Compare with individual models
print("\nComparison:")
print(f"LightGBM:      {lgb_cv:.5f}")
print(f"CatBoost:      {cat_cv:.5f}")
print(f"XGBoost:       {xgb_cv:.5f}")
print(f"RandomForest:  {rf_cv:.5f}")
print(f"ExtraTrees:    {et_cv:.5f}")
print(f"Optimized LGB: {opt_lgb_cv:.5f}")
print(f"ENSEMBLE:      {final_cv_score:.5f} ⭐")

# Create submission
submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': final_ensemble_test
})

submission.to_csv('my_optimized_ensemble.csv', index=False)
print("\nSubmission saved as 'my_optimized_ensemble.csv'")



# Load your optimized ensemble from Step 8
my_ensemble = pd.read_csv('my_optimized_ensemble.csv')

print(f"My ensemble shape: {my_ensemble.shape}")
print(f"My ensemble stats - Mean: {my_ensemble['accident_risk'].mean():.5f}, Std: {my_ensemble['accident_risk'].std():.5f}")

# Option 1: Just submit your own ensemble (RECOMMENDED)
print("\n" + "="*60)
print("OPTION 1: Submit your own ensemble (RECOMMENDED)")
print("="*60)
my_ensemble.to_csv('submission.csv', index=False)
print("✓ Saved: submission.csv")
print("This is your best bet - it's based on diverse models with optimized weights")




