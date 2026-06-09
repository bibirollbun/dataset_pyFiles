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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")



def create_advanced_features(df, is_train=True):
    """Create domain-specific features for accident risk prediction"""
    df = df.copy()
    
    # ----- RISK INTERACTION FEATURES -----
    # High curvature at high speed = HIGH RISK
    df['curve_speed_risk'] = df['curvature'] * df['speed_limit']
    df['curve_per_speed'] = df['curvature'] / (df['speed_limit'] + 1)
    
    # Sharp curves with many lanes = complex navigation
    df['curve_lanes_interaction'] = df['curvature'] * df['num_lanes']
    
    # ----- VISIBILITY RISK FEATURES -----
    # Poor lighting conditions (night = 2, dim = 1, daylight = 0 after encoding)
    # We'll create this after encoding, but prepare for it
    
    # ----- WEATHER + LIGHTING COMBO -----
    # These will be created after encoding categoricals
    
    # ----- TIME-BASED RISK PATTERNS -----
    # Rush hour indicators (we'll encode these after categorical encoding)
    
    # ----- ACCIDENT DENSITY FEATURES -----
    # Historical accidents normalized by road characteristics
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    df['accidents_per_curve'] = df['num_reported_accidents'] / (df['curvature'] + 0.01)
    
    # Speed limit categories (use qcut for better distribution or handle edge cases)
    df['speed_category'] = pd.cut(df['speed_limit'], 
                                   bins=[-np.inf, 40, 60, np.inf], 
                                   labels=[0, 1, 2]).astype(int)
    
    # Curvature severity levels (handle edge cases with -inf and inf)
    df['curve_severity'] = pd.cut(df['curvature'], 
                                   bins=[-np.inf, 0.3, 0.6, np.inf], 
                                   labels=[0, 1, 2]).astype(int)
    
    # ----- POLYNOMIAL FEATURES (selective) -----
    df['curvature_squared'] = df['curvature'] ** 2
    df['speed_squared'] = df['speed_limit'] ** 2
    df['curvature_sqrt'] = np.sqrt(df['curvature'])
    
    # ----- STATISTICAL AGGREGATIONS -----
    numeric_cols = ['curvature', 'speed_limit', 'num_lanes', 'num_reported_accidents']
    df['numeric_mean'] = df[numeric_cols].mean(axis=1)
    df['numeric_std'] = df[numeric_cols].std(axis=1)
    df['numeric_max'] = df[numeric_cols].max(axis=1)
    df['numeric_min'] = df[numeric_cols].min(axis=1)
    df['numeric_range'] = df['numeric_max'] - df['numeric_min']
    
    return df


print("\n" + "="*60)
print("CREATING ADVANCED FEATURES")
print("="*60)

train_fe = create_advanced_features(train, is_train=True)
test_fe = create_advanced_features(test, is_train=False)

print(f"Train after initial FE: {train_fe.shape}")
print(f"Test after initial FE: {test_fe.shape}")

# Prepare features and target
target = train_fe['accident_risk'].values
train_ids = train_fe['id'].values
test_ids = test_fe['id'].values

X_train = train_fe.drop(['id', 'accident_risk'], axis=1)
X_test = test_fe.drop(['id'], axis=1)

# Encode categorical variables FIRST
cat_cols = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()
le_dict = {}

print(f"\nEncoding {len(cat_cols)} categorical columns...")
for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    le_dict[col] = le

# NOW create encoded-dependent features
print("\nCreating post-encoding interaction features...")

# Weather + Lighting risk (both are now numeric)
X_train['weather_lighting_risk'] = X_train['weather'] * X_train['lighting']
X_test['weather_lighting_risk'] = X_test['weather'] * X_test['lighting']

# Poor visibility score (higher = worse)
X_train['poor_visibility'] = (X_train['lighting'] >= 1).astype(int) * (X_train['weather'] >= 1).astype(int)
X_test['poor_visibility'] = (X_test['lighting'] >= 1).astype(int) * (X_test['weather'] >= 1).astype(int)

# Rush hour indicators (evening typically = 1 or specific value)
X_train['rush_hour_indicator'] = (X_train['time_of_day'] == 1).astype(int)
X_test['rush_hour_indicator'] = (X_test['time_of_day'] == 1).astype(int)

# Complex conditions: poor visibility + high speed
X_train['complex_condition'] = X_train['poor_visibility'] * (X_train['speed_limit'] > 60).astype(int)
X_test['complex_condition'] = X_test['poor_visibility'] * (X_test['speed_limit'] > 60).astype(int)

# Road type specific risks
X_train['highway_high_speed'] = (X_train['road_type'] == 0).astype(int) * (X_train['speed_limit'] > 60).astype(int)
X_test['highway_high_speed'] = (X_test['road_type'] == 0).astype(int) * (X_test['speed_limit'] > 60).astype(int)

print(f"\nFinal training shape: {X_train.shape}")
print(f"Final test shape: {X_test.shape}")
print(f"Total features: {X_train.shape[1]}")



print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Quick LightGBM to get feature importance
lgb_quick = LGBMRegressor(n_estimators=200, random_state=42, verbose=-1)
lgb_quick.fit(X_train, target)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': lgb_quick.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20))

# Select top features (keep top 80% of cumulative importance)
feature_importance['cumulative_importance'] = feature_importance['importance'].cumsum() / feature_importance['importance'].sum()
selected_features = feature_importance[feature_importance['cumulative_importance'] <= 0.95]['feature'].tolist()

print(f"\nSelected {len(selected_features)} features (95% cumulative importance)")
print(f"Removed {X_train.shape[1] - len(selected_features)} low-importance features")

# Filter datasets
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]


print("\n" + "="*60)
print("TRAINING BASE MODELS WITH SELECTED FEATURES")
print("="*60)

def get_cv_predictions(model, X, y, X_test, n_splits=5, model_name="Model"):
    """Get OOF and test predictions"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    cv_scores = []
    
    print(f"\nTraining {model_name}...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_tr, y_tr)
        
        # OOF predictions
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        
        # Test predictions
        test_preds += model.predict(X_test) / n_splits
        
        fold_score = np.sqrt(mean_squared_error(y_val, val_preds))
        cv_scores.append(fold_score)
        print(f"  Fold {fold+1} RMSE: {fold_score:.5f}")
    
    mean_cv = np.mean(cv_scores)
    print(f"  Mean CV RMSE: {mean_cv:.5f}")
    
    return oof_preds, test_preds, mean_cv

# Best parameters from your optimization
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.012,
    'num_leaves': 100,
    'max_depth': 11,
    'min_child_samples': 48,
    'subsample': 0.76,
    'colsample_bytree': 0.60,
    'reg_alpha': 0.002,
    'reg_lambda': 0.015,
    'n_estimators': 1500,
    'random_state': 42,
    'verbose': -1
}

cat_params = {
    'iterations': 1500,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'verbose': 0
}

xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.03,
    'max_depth': 6,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 1500,
    'random_state': 42,
    'verbosity': 0
}

# Train base models
lgb_oof, lgb_test, lgb_cv = get_cv_predictions(
    LGBMRegressor(**lgb_params),
    X_train_selected, target, X_test_selected,
    n_splits=5, model_name="LightGBM"
)

cat_oof, cat_test, cat_cv = get_cv_predictions(
    CatBoostRegressor(**cat_params),
    X_train_selected, target, X_test_selected,
    n_splits=5, model_name="CatBoost"
)

xgb_oof, xgb_test, xgb_cv = get_cv_predictions(
    XGBRegressor(**xgb_params),
    X_train_selected, target, X_test_selected,
    n_splits=5, model_name="XGBoost"
)


print("\n" + "="*60)
print("BUILDING STACKING ENSEMBLE")
print("="*60)

# Create meta-features from base model predictions
meta_train = pd.DataFrame({
    'lgb': lgb_oof,
    'cat': cat_oof,
    'xgb': xgb_oof,
    'lgb_cat_mean': (lgb_oof + cat_oof) / 2,
    'lgb_xgb_mean': (lgb_oof + xgb_oof) / 2,
    'cat_xgb_mean': (cat_oof + xgb_oof) / 2,
    'all_mean': (lgb_oof + cat_oof + xgb_oof) / 3,
    'lgb_cat_diff': np.abs(lgb_oof - cat_oof),
    'prediction_std': np.std([lgb_oof, cat_oof, xgb_oof], axis=0)
})

meta_test = pd.DataFrame({
    'lgb': lgb_test,
    'cat': cat_test,
    'xgb': xgb_test,
    'lgb_cat_mean': (lgb_test + cat_test) / 2,
    'lgb_xgb_mean': (lgb_test + xgb_test) / 2,
    'cat_xgb_mean': (cat_test + xgb_test) / 2,
    'all_mean': (lgb_test + cat_test + xgb_test) / 3,
    'lgb_cat_diff': np.abs(lgb_test - cat_test),
    'prediction_std': np.std([lgb_test, cat_test, xgb_test], axis=0)
})

print(f"Meta-features shape: {meta_train.shape}")

# Train meta-model (Light GBM works well as meta-learner)
meta_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.01,
    'num_leaves': 15,
    'max_depth': 4,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 500,
    'random_state': 42,
    'verbose': -1
}

stack_oof, stack_test, stack_cv = get_cv_predictions(
    LGBMRegressor(**meta_params),
    meta_train, target, meta_test,
    n_splits=5, model_name="Stacking Meta-Model"
)


print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

print("\nBase Models:")
print(f"  LightGBM:  {lgb_cv:.5f}")
print(f"  CatBoost:  {cat_cv:.5f}")
print(f"  XGBoost:   {xgb_cv:.5f}")
print(f"\nStacking:    {stack_cv:.5f} ⭐")

improvement = min(lgb_cv, cat_cv, xgb_cv) - stack_cv
print(f"\nImprovement: {improvement:.5f}")

# Create submission
submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': stack_test
})

submission.to_csv('stacking_submission.csv', index=False)
print("\n✓ Submission saved: stacking_submission.csv")
print(f"  Shape: {submission.shape}")
print(f"  Stats: Mean={submission['accident_risk'].mean():.5f}, Std={submission['accident_risk'].std():.5f}")




