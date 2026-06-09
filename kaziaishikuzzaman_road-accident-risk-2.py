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
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

print('All libraries imported successfully!')


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

print(f'Training Data Shape: {train.shape}')
print(f'Test Data Shape: {test.shape}')
print(f'\nTraining Data Head:')
print(train.head())
print(f'\nTest Data Head:')
print(test.head())
print(f'\nSample Submission Head:')
print(sample_submission.head())


# Detailed data exploration
print('Data Info:')
print(train.info())
print(f'\nDescriptive Statistics:')
print(train.describe())
print(f'\nMissing Values in Train:')
print(train.isnull().sum())
print(f'\nMissing Values in Test:')
print(test.isnull().sum())

# Identify column types
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols = [col for col in numerical_cols if col not in ['id', 'accident_risk']]

print(f'\nCategorical Columns: {categorical_cols}')
print(f'Numerical Columns: {numerical_cols}')


# Visualize target variable distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(train['accident_risk'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_title('Distribution of Accident Risk', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Accident Risk')
axes[0].set_ylabel('Frequency')

axes[1].boxplot(train['accident_risk'])
axes[1].set_title('Boxplot of Accident Risk', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Accident Risk')

plt.tight_layout()
plt.show()

print(f'Target Variable Statistics:')
print(train['accident_risk'].describe())


def create_features(df):
    """Create advanced features for accident risk prediction"""
    df_copy = df.copy()
    
    # ===== Interaction Features =====
    df_copy['speed_curvature_interaction'] = df_copy['speed_limit'] * df_copy['curvature']
    df_copy['lanes_curvature_interaction'] = df_copy['num_lanes'] * df_copy['curvature']
    df_copy['speed_lanes_ratio'] = df_copy['speed_limit'] / (df_copy['num_lanes'] + 1)
    df_copy['speed_num_accidents_interaction'] = df_copy['speed_limit'] * df_copy['num_reported_accidents']
    
    # ===== Non-linear Transformations =====
    df_copy['curvature_squared'] = df_copy['curvature'] ** 2
    df_copy['speed_squared'] = df_copy['speed_limit'] ** 2
    df_copy['log_speed'] = np.log1p(df_copy['speed_limit'])
    df_copy['log_accidents'] = np.log1p(df_copy['num_reported_accidents'])
    
    # ===== Accident Risk Features =====
    df_copy['accident_per_lane'] = (df_copy['num_reported_accidents'] + 1) / (df_copy['num_lanes'] + 1)
    df_copy['accident_rate_high'] = (df_copy['num_reported_accidents'] > df_copy['num_reported_accidents'].median()).astype(int)
    
    # ===== Boolean Flag Features =====
    df_copy['has_signs'] = df_copy['road_signs_present'].astype(int)
    df_copy['is_public'] = df_copy['public_road'].astype(int)
    df_copy['is_holiday'] = df_copy['holiday'].astype(int)
    df_copy['is_school_season'] = df_copy['school_season'].astype(int)
    
    # ===== Time-based Features =====
    time_of_day_map = {'morning': 1, 'afternoon': 2, 'evening': 3, 'night': 4}
    df_copy['time_numeric'] = df_copy['time_of_day'].map(time_of_day_map)
    
    # ===== Weather Risk Encoding =====
    weather_risk = {'clear': 1, 'rainy': 3, 'foggy': 4, 'snowy': 4}
    df_copy['weather_risk'] = df_copy['weather'].map(weather_risk)
    
    # ===== Lighting Encoding =====
    lighting_map = {'daylight': 1, 'dim': 2, 'night': 3}
    df_copy['lighting_numeric'] = df_copy['lighting'].map(lighting_map)
    
    # ===== Combined Risk Score =====
    df_copy['combined_risk_score'] = (df_copy['curvature'] * 0.3 + 
                                       (df_copy['weather_risk'] / 4) * 0.2 + 
                                       (df_copy['lighting_numeric'] / 3) * 0.2 + 
                                       (df_copy['num_reported_accidents'] / 7) * 0.3)
    
    # ===== Road Type Features =====
    road_type_map = {'urban': 1, 'rural': 2, 'highway': 3}
    df_copy['road_type_numeric'] = df_copy['road_type'].map(road_type_map)
    
    return df_copy

# Apply feature engineering
train_fe = create_features(train)
test_fe = create_features(test)

print('Features created successfully!')
print(f'Train shape after feature engineering: {train_fe.shape}')
print(f'Test shape after feature engineering: {test_fe.shape}')

new_features = [col for col in train_fe.columns if col not in train.columns]
print(f'\nNew features created ({len(new_features)}): {new_features}')


# Prepare features and target
X_train = train_fe.drop(['id', 'accident_risk'], axis=1)
y_train = train_fe['accident_risk']
X_test = test_fe.drop(['id'], axis=1)
test_ids = test_fe['id'].values

# Identify column types
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f'Categorical features: {categorical_features}')
print(f'Numerical features count: {len(numerical_features)}')
print(f'Total features: {len(X_train.columns)}')

# Create preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit preprocessor
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f'\nProcessed train shape: {X_train_processed.shape}')
print(f'Processed test shape: {X_test_processed.shape}')


# Split data for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_processed, y_train, test_size=0.2, random_state=42
)

print('Training individual models with optimized hyperparameters...')
print('='*60)

# ===== XGBoost =====
print('\n1. Training XGBoost...')
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)
xgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)
xgb_pred_val = xgb_model.predict(X_val)
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred_val))
xgb_mae = mean_absolute_error(y_val, xgb_pred_val)
xgb_r2 = r2_score(y_val, xgb_pred_val)
print(f'   XGBoost RMSE: {xgb_rmse:.6f}, MAE: {xgb_mae:.6f}, R²: {xgb_r2:.6f}')

# ===== LightGBM =====
print('\n2. Training LightGBM...')
lgb_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
print('\n2. Training LightGBM...')
lgb_model = LGBMRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
    random_state=42, n_jobs=-1, verbose=-1
)
lgb_model.fit(X_tr, y_tr)  # ✅ Simple fit - no callbacks!
lgb_pred_val = lgb_model.predict(X_val)
lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_pred_val))
lgb_mae = mean_absolute_error(y_val, lgb_pred_val)
lgb_r2 = r2_score(y_val, lgb_pred_val)
print(f'   LightGBM RMSE: {lgb_rmse:.6f}, MAE: {lgb_mae:.6f}, R²: {lgb_r2:.6f}')

# ===== CatBoost =====
print('\n3. Training CatBoost...')
cat_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    subsample=0.8,
    random_state=42,
    verbose=False
)
cat_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)
cat_pred_val = cat_model.predict(X_val)
cat_rmse = np.sqrt(mean_squared_error(y_val, cat_pred_val))
cat_mae = mean_absolute_error(y_val, cat_pred_val)
cat_r2 = r2_score(y_val, cat_pred_val)
print(f'   CatBoost RMSE: {cat_rmse:.6f}, MAE: {cat_mae:.6f}, R²: {cat_r2:.6f}')

# ===== Random Forest =====
print('\n4. Training Random Forest...')
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_tr, y_tr)
rf_pred_val = rf_model.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred_val))
rf_mae = mean_absolute_error(y_val, rf_pred_val)
rf_r2 = r2_score(y_val, rf_pred_val)
print(f'   Random Forest RMSE: {rf_rmse:.6f}, MAE: {rf_mae:.6f}, R²: {rf_r2:.6f}')

print('\n' + '='*60)
print('All models trained successfully!')


# Perform K-Fold cross-validation for more robust evaluation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

print('Performing 5-Fold Cross-Validation...')
print('='*60)

# XGBoost CV
xgb_cv_scores = cross_val_score(
    xgb_model, X_train_processed, y_train,
    cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1
)
xgb_cv_rmse = np.sqrt(-xgb_cv_scores)
print(f'\nXGBoost CV RMSE: {xgb_cv_rmse.mean():.6f} (+/- {xgb_cv_rmse.std():.6f})')
print(f'  Fold scores: {[f"{x:.6f}" for x in xgb_cv_rmse]}')

# LightGBM CV
lgb_cv_scores = cross_val_score(
    lgb_model, X_train_processed, y_train,
    cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1
)
lgb_cv_rmse = np.sqrt(-lgb_cv_scores)
print(f'\nLightGBM CV RMSE: {lgb_cv_rmse.mean():.6f} (+/- {lgb_cv_rmse.std():.6f})')
print(f'  Fold scores: {[f"{x:.6f}" for x in lgb_cv_rmse]}')

# CatBoost CV
cat_cv_scores = cross_val_score(
    cat_model, X_train_processed, y_train,
    cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1
)
cat_cv_rmse = np.sqrt(-cat_cv_scores)
print(f'\nCatBoost CV RMSE: {cat_cv_rmse.mean():.6f} (+/- {cat_cv_rmse.std():.6f})')
print(f'  Fold scores: {[f"{x:.6f}" for x in cat_cv_rmse]}')

# Random Forest CV
rf_cv_scores = cross_val_score(
    rf_model, X_train_processed, y_train,
    cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1
)
rf_cv_rmse = np.sqrt(-rf_cv_scores)
print(f'\nRandom Forest CV RMSE: {rf_cv_rmse.mean():.6f} (+/- {rf_cv_rmse.std():.6f})')
print(f'  Fold scores: {[f"{x:.6f}" for x in rf_cv_rmse]}')

print('\n' + '='*60)


# Create weighted ensemble based on CV performance
print('Creating Weighted Ensemble Model...')
print('='*60)

# Retrain models on full training data
print('\nRetraining models on full dataset...')

xgb_final = XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method='hist'
)
xgb_final.fit(X_train_processed, y_train, verbose=False)

lgb_final = LGBMRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
)
lgb_final.fit(X_train_processed, y_train)

cat_final = CatBoostRegressor(
    iterations=500, learning_rate=0.05, depth=6, subsample=0.8, random_state=42, verbose=False
)
cat_final.fit(X_train_processed, y_train, verbose=False)

rf_final = RandomForestRegressor(
    n_estimators=300, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1
)
rf_final.fit(X_train_processed, y_train)

# Generate predictions
xgb_pred_test = xgb_final.predict(X_test_processed)
lgb_pred_test = lgb_final.predict(X_test_processed)
cat_pred_test = cat_final.predict(X_test_processed)
rf_pred_test = rf_final.predict(X_test_processed)

# Calculate ensemble weights (inverse of CV RMSE)
weights = np.array([
    xgb_cv_rmse.mean(),
    lgb_cv_rmse.mean(),
    cat_cv_rmse.mean(),
    rf_cv_rmse.mean()
])
weights = 1 / weights
weights = weights / weights.sum()

print(f'\nEnsemble Weights:')
print(f'  XGBoost:      {weights[0]:.4f} ({weights[0]*100:.2f}%)')
print(f'  LightGBM:     {weights[1]:.4f} ({weights[1]*100:.2f}%)')
print(f'  CatBoost:     {weights[2]:.4f} ({weights[2]*100:.2f}%)')
print(f'  Random Forest: {weights[3]:.4f} ({weights[3]*100:.2f}%)')

# Create ensemble predictions
ensemble_pred = (
    weights[0] * xgb_pred_test +
    weights[1] * lgb_pred_test +
    weights[2] * cat_pred_test +
    weights[3] * rf_pred_test
)

print(f'\nEnsemble predictions shape: {ensemble_pred.shape}')


# CRITICAL: Create submission in correct format
print('\nCreating submission...')

submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': ensemble_pred
})

# Clip to [0, 1] range
submission['accident_risk'] = submission['accident_risk'].clip(0, 1)

# ✅ CORRECT PATH for Kaggle!
submission_path = '/kaggle/working/submission.csv'
submission.to_csv(submission_path, index=False)
print(f'✓ Submission saved to {submission_path}')


# Comprehensive visualization
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# 1. Model Comparison - RMSE
ax1 = fig.add_subplot(gs[0, 0])
models = ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest']
cv_rmse = [xgb_cv_rmse.mean(), lgb_cv_rmse.mean(), cat_cv_rmse.mean(), rf_cv_rmse.mean()]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
ax1.bar(models, cv_rmse, color=colors, alpha=0.8, edgecolor='black')
ax1.set_title('Cross-Validation RMSE Comparison', fontsize=11, fontweight='bold')
ax1.set_ylabel('RMSE')
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(cv_rmse):
    ax1.text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)

# 2. Ensemble Weights
ax2 = fig.add_subplot(gs[0, 1])
ax2.pie(weights, labels=models, autopct='%1.1f%%', colors=colors, startangle=90)
ax2.set_title('Ensemble Model Weights', fontsize=11, fontweight='bold')

# 3. Distribution of Test Predictions
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(ensemble_pred, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax3.set_title('Distribution of Ensemble Predictions', fontsize=11, fontweight='bold')
ax3.set_xlabel('Accident Risk')
ax3.set_ylabel('Frequency')

# 4. Validation Set Performance - XGBoost
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(y_val, xgb_pred_val, alpha=0.5, s=10)
ax4.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
ax4.set_title(f'XGBoost: Val RMSE={xgb_rmse:.4f}', fontsize=11, fontweight='bold')
ax4.set_xlabel('Actual')
ax4.set_ylabel('Predicted')

# 5. Validation Set Performance - LightGBM
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(y_val, lgb_pred_val, alpha=0.5, s=10, color='orange')
ax5.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
ax5.set_title(f'LightGBM: Val RMSE={lgb_rmse:.4f}', fontsize=11, fontweight='bold')
ax5.set_xlabel('Actual')
ax5.set_ylabel('Predicted')

# 6. Validation Set Performance - Ensemble
ax6 = fig.add_subplot(gs[1, 2])
ensemble_val = (weights[0] * xgb_pred_val + weights[1] * lgb_pred_val +
                 weights[2] * cat_pred_val + weights[3] * rf_pred_val)
ensemble_rmse_val = np.sqrt(mean_squared_error(y_val, ensemble_val))
ax6.scatter(y_val, ensemble_val, alpha=0.5, s=10, color='green')
ax6.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
ax6.set_title(f'Ensemble: Val RMSE={ensemble_rmse_val:.4f}', fontsize=11, fontweight='bold')
ax6.set_xlabel('Actual')
ax6.set_ylabel('Predicted')

# 7. Cross-Validation Score Distribution
ax7 = fig.add_subplot(gs[2, 0])
ax7.boxplot([xgb_cv_rmse, lgb_cv_rmse, cat_cv_rmse, rf_cv_rmse],
             labels=models)
ax7.set_title('Cross-Validation RMSE Distribution', fontsize=11, fontweight='bold')
ax7.set_ylabel('RMSE')
ax7.tick_params(axis='x', rotation=45)

# 8. Model Predictions Comparison
ax8 = fig.add_subplot(gs[2, 1])
ax8.hist(xgb_pred_test, bins=40, alpha=0.5, label='XGBoost', edgecolor='black')
ax8.hist(lgb_pred_test, bins=40, alpha=0.5, label='LightGBM', edgecolor='black')
ax8.hist(ensemble_pred, bins=40, alpha=0.5, label='Ensemble', edgecolor='black')
ax8.set_title('Prediction Distribution Comparison', fontsize=11, fontweight='bold')
ax8.set_xlabel('Accident Risk')
ax8.set_ylabel('Frequency')
ax8.legend()

# 9. Summary Statistics Table (as text)
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')
summary_text = f"""SUMMARY STATISTICS

Validation RMSE:
  XGB: {xgb_rmse:.6f}
  LGB: {lgb_rmse:.6f}
  CAT: {cat_rmse:.6f}
  RF:  {rf_rmse:.6f}

CV RMSE (Mean±Std):
  XGB: {xgb_cv_rmse.mean():.4f}±{xgb_cv_rmse.std():.4f}
  LGB: {lgb_cv_rmse.mean():.4f}±{lgb_cv_rmse.std():.4f}

Ensemble R²: {r2_score(y_val, ensemble_val):.4f}
"""
ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
plt.show()
print('Performance visualization saved!')


# Feature Importance Analysis - CORRECTED VERSION

# Get feature names from the original data
feature_names = []

# Add numerical feature names
numerical_features_list = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
feature_names.extend(numerical_features_list)

# Add categorical feature names
categorical_features_list = X_train.select_dtypes(include=['object']).columns.tolist()
feature_names.extend(categorical_features_list)

print(f'Total features in X_train: {len(X_train.columns)}')
print(f'Feature names created: {len(feature_names)}')

# Now let's get feature importances
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# ===== XGBoost Feature Importance =====
try:
    xgb_importances = xgb_final.feature_importances_
    if len(xgb_importances) == len(feature_names):
        xgb_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': xgb_importances
        }).sort_values('importance', ascending=False).head(15)
    else:
        # If lengths don't match, create generic feature names
        xgb_imp = pd.DataFrame({
            'feature': [f'Feature_{i}' for i in range(len(xgb_importances))],
            'importance': xgb_importances
        }).sort_values('importance', ascending=False).head(15)
    
    axes[0, 0].barh(range(len(xgb_imp)), xgb_imp['importance'], color='#FF6B6B')
    axes[0, 0].set_yticks(range(len(xgb_imp)))
    axes[0, 0].set_yticklabels(xgb_imp['feature'])
    axes[0, 0].set_title('XGBoost Feature Importance', fontsize=12, fontweight='bold')
    axes[0, 0].invert_yaxis()
except Exception as e:
    print(f"Error with XGBoost: {e}")

# ===== LightGBM Feature Importance =====
try:
    lgb_importances = lgb_final.feature_importances_
    if len(lgb_importances) == len(feature_names):
        lgb_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': lgb_importances
        }).sort_values('importance', ascending=False).head(15)
    else:
        lgb_imp = pd.DataFrame({
            'feature': [f'Feature_{i}' for i in range(len(lgb_importances))],
            'importance': lgb_importances
        }).sort_values('importance', ascending=False).head(15)
    
    axes[0, 1].barh(range(len(lgb_imp)), lgb_imp['importance'], color='#4ECDC4')
    axes[0, 1].set_yticks(range(len(lgb_imp)))
    axes[0, 1].set_yticklabels(lgb_imp['feature'])
    axes[0, 1].set_title('LightGBM Feature Importance', fontsize=12, fontweight='bold')
    axes[0, 1].invert_yaxis()
except Exception as e:
    print(f"Error with LightGBM: {e}")

# ===== CatBoost Feature Importance =====
try:
    cat_importances = cat_final.get_feature_importance()
    if len(cat_importances) == len(feature_names):
        cat_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': cat_importances
        }).sort_values('importance', ascending=False).head(15)
    else:
        cat_imp = pd.DataFrame({
            'feature': [f'Feature_{i}' for i in range(len(cat_importances))],
            'importance': cat_importances
        }).sort_values('importance', ascending=False).head(15)
    
    axes[1, 0].barh(range(len(cat_imp)), cat_imp['importance'], color='#45B7D1')
    axes[1, 0].set_yticks(range(len(cat_imp)))
    axes[1, 0].set_yticklabels(cat_imp['feature'])
    axes[1, 0].set_title('CatBoost Feature Importance', fontsize=12, fontweight='bold')
    axes[1, 0].invert_yaxis()
except Exception as e:
    print(f"Error with CatBoost: {e}")

# ===== Random Forest Feature Importance =====
try:
    rf_importances = rf_final.feature_importances_
    if len(rf_importances) == len(feature_names):
        rf_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_importances
        }).sort_values('importance', ascending=False).head(15)
    else:
        rf_imp = pd.DataFrame({
            'feature': [f'Feature_{i}' for i in range(len(rf_importances))],
            'importance': rf_importances
        }).sort_values('importance', ascending=False).head(15)
    
    axes[1, 1].barh(range(len(rf_imp)), rf_imp['importance'], color='#96CEB4')
    axes[1, 1].set_yticks(range(len(rf_imp)))
    axes[1, 1].set_yticklabels(rf_imp['feature'])
    axes[1, 1].set_title('Random Forest Feature Importance', fontsize=12, fontweight='bold')
    axes[1, 1].invert_yaxis()
except Exception as e:
    print(f"Error with Random Forest: {e}")

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print('Feature importance visualization saved!')


# Final comprehensive summary
print('\n' + '='*70)
print('FINAL MODEL PERFORMANCE SUMMARY'.center(70))
print('='*70)

print('\n1. INDIVIDUAL MODEL PERFORMANCE:')
print('-'*70)
print(f'\nValidation Set Metrics:')
print(f'  Model           RMSE      MAE       R² Score')
print(f'  {"─"*60}')
print(f'  XGBoost:        {xgb_rmse:.6f}  {xgb_mae:.6f}  {xgb_r2:.6f}')
print(f'  LightGBM:       {lgb_rmse:.6f}  {lgb_mae:.6f}  {lgb_r2:.6f}')
print(f'  CatBoost:       {cat_rmse:.6f}  {cat_mae:.6f}  {cat_r2:.6f}')
print(f'  Random Forest:  {rf_rmse:.6f}  {rf_mae:.6f}  {rf_r2:.6f}')

print('\n\n2. CROSS-VALIDATION PERFORMANCE (5-Fold):')
print('-'*70)
print(f'  Model           Mean RMSE    Std Dev     Min         Max')
print(f'  {"─"*60}')
print(f'  XGBoost:        {xgb_cv_rmse.mean():.6f}      {xgb_cv_rmse.std():.6f}      {xgb_cv_rmse.min():.6f}      {xgb_cv_rmse.max():.6f}')
print(f'  LightGBM:       {lgb_cv_rmse.mean():.6f}      {lgb_cv_rmse.std():.6f}      {lgb_cv_rmse.min():.6f}      {lgb_cv_rmse.max():.6f}')
print(f'  CatBoost:       {cat_cv_rmse.mean():.6f}      {cat_cv_rmse.std():.6f}      {cat_cv_rmse.min():.6f}      {cat_cv_rmse.max():.6f}')
print(f'  Random Forest:  {rf_cv_rmse.mean():.6f}      {rf_cv_rmse.std():.6f}      {rf_cv_rmse.min():.6f}      {rf_cv_rmse.max():.6f}')

print('\n\n3. ENSEMBLE MODEL CONFIGURATION:')
print('-'*70)
print(f'  Model           Weight      Percentage')
print(f'  {"─"*60}')
print(f'  XGBoost:        {weights[0]:.6f}      {weights[0]*100:.2f}%')
print(f'  LightGBM:       {weights[1]:.6f}      {weights[1]*100:.2f}%')
print(f'  CatBoost:       {weights[2]:.6f}      {weights[2]*100:.2f}%')
print(f'  Random Forest:  {weights[3]:.6f}      {weights[3]*100:.2f}%')

print('\n\n4. TEST SET PREDICTIONS SUMMARY:')
print('-'*70)
print(f'  Statistic                   Value')
print(f'  {"─"*60}')
print(f'  Number of predictions:     {len(ensemble_pred):>10}')
print(f'  Mean prediction:           {ensemble_pred.mean():>10.6f}')
print(f'  Std deviation:             {ensemble_pred.std():>10.6f}')
print(f'  Min prediction:            {ensemble_pred.min():>10.6f}')
print(f'  Max prediction:            {ensemble_pred.max():>10.6f}')
print(f'  Median prediction:         {np.median(ensemble_pred):>10.6f}')

print('\n\n5. OUTPUT FILES GENERATED:')
print('-'*70)
print('  ✓ submission.csv')
print('  ✓ feature_importance.png')

print('\n' + '='*70)
print('ANALYSIS COMPLETE'.center(70))
print('='*70 + '\n')

