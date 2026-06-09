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


# ğŸ“¦ Standard Libraries
import os
import sys
import gc
import math
import random
import warnings
from pathlib import Path

# ğŸ§® Data Manipulation
import numpy as np
import pandas as pd

# ğŸ“Š Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ğŸ§  Machine Learning
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.metrics import mean_squared_error, mean_absolute_error


# âš™ï¸� Models (optional â€” add or remove as needed)
from sklearn.linear_model import LogisticRegression , Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb 
import xgboost as xgb 
import catboost as cb 
from sklearn.model_selection import train_test_split ,GridSearchCV
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor 

# ğŸ§° Utility
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
np.random.seed(42)


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
print(train_df.info())




missing_train_total = train_df.isnull().sum().sum()
missing_test_total  = test_df.isnull().sum().sum()

print(f"ğŸ”¹ Missing values in Train set: {missing_train_total:,}")
print(f"ğŸ”¹ Missing values in Test set:  {missing_test_total:,}")

if missing_train_total == 0 and missing_test_total == 0:
    print("âœ… No missing values detected in either dataset.")
else:
    print("âš ï¸� Missing values found â€” further analysis required.")





def create_advanced_features(df):
    df_copy = df.copy()
    
    df_copy['speed_curvature'] = df_copy['speed_limit'] * df_copy['curvature']
    df_copy['lanes_speed'] = df_copy['num_lanes'] * df_copy['speed_limit']
    
    weather_risk = {'clear': 0, 'rainy': 1, 'foggy': 2}
    lighting_risk = {'daylight': 0, 'dim': 1, 'night': 2}
    time_risk = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}
    
    df_copy['weather_risk'] = df_copy['weather'].map(weather_risk)
    df_copy['lighting_risk'] = df_copy['lighting'].map(lighting_risk)
    df_copy['time_risk'] = df_copy['time_of_day'].map(time_risk)
    
    df_copy['environment_risk'] = (
        df_copy['weather_risk'] + 
        df_copy['lighting_risk'] + 
        df_copy['time_risk']
    )
    
    df_copy['complexity_score'] = (
        df_copy['curvature'] * df_copy['speed_limit'] * df_copy['num_lanes']
    ) / 100
    
    lighting_scores = {'daylight': 0, 'dim': 2, 'night': 3}
    weather_scores = {'clear': 0, 'rainy': 2, 'foggy': 3}
    df_copy['visibility_risk'] = (
        df_copy['lighting'].map(lighting_scores) + 
        df_copy['weather'].map(weather_scores)
    )
    
    time_scores = {'morning': 1, 'afternoon': 1.2, 'evening': 1.5, 'night': 2}
    df_copy['time_amplifier'] = df_copy['time_of_day'].map(time_scores)
    
    road_scores = {'urban': 1, 'rural': 1.5, 'highway': 2}
    df_copy['road_type_encoded'] = df_copy['road_type'].map(road_scores)
    
    df_copy['composite_risk'] = (
        df_copy['complexity_score'] *
        df_copy['visibility_risk'] *
        df_copy['time_amplifier'] *
        df_copy['road_type_encoded']
    ) / 10
    
    df_copy['peak_hour'] = (
        (df_copy['time_of_day'].isin(['morning', 'evening'])) & 
        (df_copy['holiday'] == 'False')
    ).astype(int)
    
    df_copy['high_risk_combo'] = (
        (df_copy['weather'].isin(['foggy', 'rainy'])) &
        (df_copy['lighting'].isin(['dim', 'night'])) &
        (df_copy['curvature'] > 0.5)
    ).astype(int)
    
    return df_copy

train_df = create_advanced_features(train_df)
test_df = create_advanced_features(test_df)

print(f"âœ… Train shape: {train_df.shape}")
print(f"âœ… Test shape:  {test_df.shape}")




features_to_use = [
    'road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting', 
    'weather', 'road_signs_present', 'public_road', 'time_of_day', 
    'holiday', 'school_season', 'num_reported_accidents',
    'speed_curvature', 'lanes_speed', 'weather_risk', 
    'lighting_risk', 'time_risk', 'environment_risk',
    'complexity_score', 'visibility_risk', 'time_amplifier',
    'road_type_encoded', 'composite_risk', 'peak_hour', 'high_risk_combo'
]

X = train_df[features_to_use].copy()
y = train_df['accident_risk']
X_test = test_df[features_to_use].copy()

categorical_features = [
    'road_type', 'lighting', 'weather', 'road_signs_present', 
    'public_road', 'time_of_day', 'holiday', 'school_season'
]

for col in ['road_signs_present', 'public_road', 'holiday', 'school_season']:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

print(f"âœ… Total features used: {len(features_to_use)}")
print(f"âœ… Categorical features: {categorical_features}")





from sklearn.preprocessing import LabelEncoder

def prepare_xgboost_features_simple(X, X_test=None):
    """
    Converts categorical features to numeric for XGBoost.

    Parameters:
        X (DataFrame): Training features
        X_test (DataFrame, optional): Test features
    
    Returns:
        X_xgb (DataFrame), X_test_xgb (DataFrame, optional)
    """
    X_xgb = X.copy()
    X_test_xgb = X_test.copy() if X_test is not None else None

    categorical_cols = [
        'road_type', 'lighting', 'weather', 'time_of_day', 
        'road_signs_present', 'public_road', 'holiday', 'school_season'
    ]

    for col in categorical_cols:
        le = LabelEncoder()
        X_xgb[col] = le.fit_transform(X_xgb[col].astype(str))

        if X_test_xgb is not None:
            unique_train = set(le.classes_)
            X_test_xgb[col] = X_test_xgb[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in unique_train else -1
            )

    if X_test_xgb is not None:
        return X_xgb, X_test_xgb
    return X_xgb






X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")


X_train_cat, X_val_cat, X_test_cat = X_train.copy(), X_val.copy(), X_test.copy()

X_train_xgb, X_val_xgb = prepare_xgboost_features_simple(X_train, X_val)
X_test_xgb = prepare_xgboost_features_simple(X_test)

X_train_lgb, X_val_lgb = prepare_xgboost_features_simple(X_train, X_val)
X_test_lgb = prepare_xgboost_features_simple(X_test)

cat_model = CatBoostRegressor(
    cat_features=categorical_features,
    iterations=800,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3,
    random_strength=0.5,
    bagging_temperature=0.8,
    early_stopping_rounds=50,
    verbose=False,
    random_state=42
)

xgb_model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.8,
    colsample_bylevel=0.8,
    reg_alpha=0.2,
    reg_lambda=0.3,
    gamma=0.1,
    eval_metric='rmse',
    early_stopping_rounds=50,
    verbosity=0,
    random_state=42
)

lgb_model = LGBMRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=45,
    subsample=0.85,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=0.3,
    min_child_samples=25,
    min_child_weight=0.001,
    verbose=-1,
    random_state=42
)

print("Training models...")

print("Training CatBoost...", end=" ")
cat_model.fit(X_train_cat, y_train, eval_set=[(X_val_cat, y_val)], verbose=False)
print("âœ“")

print("Training XGBoost...", end=" ")
xgb_model.fit(X_train_xgb, y_train, eval_set=[(X_val_xgb, y_val)], verbose=False)
print("âœ“")

print("Training LightGBM...", end=" ")
lgb_model.fit(X_train_lgb, y_train)
print("âœ“")

print("\nAll models trained successfully!")

cat_val_pred = cat_model.predict(X_val_cat)
xgb_val_pred = xgb_model.predict(X_val_xgb)
lgb_val_pred = lgb_model.predict(X_val_lgb)

cat_test_pred = cat_model.predict(X_test_cat)
xgb_test_pred = xgb_model.predict(X_test_xgb)
lgb_test_pred = lgb_model.predict(X_test_lgb)

print("Predictions ready for ensemble!")





print("Creating stacking ensemble...")

cat_train_pred = cat_model.predict(X_train_cat)
cat_val_pred = cat_model.predict(X_val_cat)
cat_test_pred = cat_model.predict(X_test_cat)

xgb_train_pred = xgb_model.predict(X_train_xgb)
xgb_val_pred = xgb_model.predict(X_val_xgb)
xgb_test_pred = xgb_model.predict(X_test_xgb)

lgb_train_pred = lgb_model.predict(X_train_lgb)
lgb_val_pred = lgb_model.predict(X_val_lgb)
lgb_test_pred = lgb_model.predict(X_test_lgb)

level1_train = np.column_stack([cat_train_pred, xgb_train_pred, lgb_train_pred])
level1_val = np.column_stack([cat_val_pred, xgb_val_pred, lgb_val_pred])
level1_test = np.column_stack([cat_test_pred, xgb_test_pred, lgb_test_pred])

print("Stacking feature shapes:")
print(f"Train: {level1_train.shape}")
print(f"Val: {level1_val.shape}")
print(f"Test: {level1_test.shape}")

params = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge = Ridge()
grid = GridSearchCV(ridge, param_grid=params, cv=5)
grid.fit(level1_train, y_train)

best_alpha = grid.best_params_['alpha']
print("Best alpha for Ridge:", best_alpha)

meta_model = Ridge(alpha=best_alpha)
meta_model.fit(level1_train, y_train)

stacking_val_pred = meta_model.predict(level1_val)
stacking_test_pred = meta_model.predict(level1_test)

print("Stacking ensemble training completed!")



cat_val_rmse = np.sqrt(mean_squared_error(y_val, cat_val_pred))
xgb_val_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
lgb_val_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_pred))
stack_val_rmse = np.sqrt(mean_squared_error(y_val, stacking_val_pred))

cat_mae = mean_absolute_error(y_val, cat_val_pred)
xgb_mae = mean_absolute_error(y_val, xgb_val_pred)
lgb_mae = mean_absolute_error(y_val, lgb_val_pred)

def combined_score(rmse, mae):
    return 0.7 * rmse + 0.3 * mae  # Give more importance to RMSE

cat_score = combined_score(cat_val_rmse, cat_mae)
xgb_score = combined_score(xgb_val_rmse, xgb_mae)
lgb_score = combined_score(lgb_val_rmse, lgb_mae)
stack_score = combined_score(stack_val_rmse, 0)  # Stacking already balanced

models_scores = {
    'CatBoost': cat_score,
    'XGBoost': xgb_score,
    'LightGBM': lgb_score,
    'Stacking': stack_score
}

weights = {}
total_weight = 0
for name, score in models_scores.items():
    weights[name] = np.exp(-score * 5)  # Exponential emphasizes differences
    total_weight += weights[name]

for name in weights:
    weights[name] /= total_weight

print("\nSmart Model Weights:")
for name, weight in weights.items():
    print(f"{name}: {weight:.3f}")

super_ensemble_val = (
    weights['CatBoost'] * cat_val_pred +
    weights['XGBoost'] * xgb_val_pred +
    weights['LightGBM'] * lgb_val_pred +
    weights['Stacking'] * stacking_val_pred
)

super_ensemble_test = (
    weights['CatBoost'] * cat_test_pred +
    weights['XGBoost'] * xgb_test_pred +
    weights['LightGBM'] * lgb_test_pred +
    weights['Stacking'] * stacking_test_pred
)

print("\nSuper Ensemble predictions are ready!")



super_ensemble_val_rmse = np.sqrt(mean_squared_error(y_val, super_ensemble_val))
super_ensemble_val_mae = mean_absolute_error(y_val, super_ensemble_val)

print("\n=== ğŸ§  SUPER ENSEMBLE PERFORMANCE ===")
print(f"Super Ensemble RMSE: {super_ensemble_val_rmse:.4f}")
print(f"Super Ensemble MAE:  {super_ensemble_val_mae:.4f}")

print("\n=== âš™ï¸� INDIVIDUAL MODEL PERFORMANCE ===")
print(f"CatBoost RMSE:  {cat_val_rmse:.4f}")
print(f"XGBoost RMSE:   {xgb_val_rmse:.4f}")
print(f"LightGBM RMSE:  {lgb_val_rmse:.4f}")
print(f"Stacking RMSE:  {stack_val_rmse:.4f}")

best_single_rmse = min(cat_val_rmse, xgb_val_rmse, lgb_val_rmse)
improvement = ((best_single_rmse - super_ensemble_val_rmse) / best_single_rmse) * 100

print("\n=== ğŸš€ IMPROVEMENT ===")
print(f"Improvement over best single model: {improvement:.2f}%")



lgb_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': lgb_model.feature_importances_
})
lgb_importance['importance'] /= lgb_importance['importance'].max()  # normalize

cat_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': cat_model.get_feature_importance()
})
cat_importance['importance'] /= cat_importance['importance'].max()

xgb_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
})
xgb_importance['importance'] /= xgb_importance['importance'].max()

combined_importance = (
    pd.merge(
        pd.merge(cat_importance, xgb_importance, on='feature', suffixes=('_cat', '_xgb')),
        lgb_importance, on='feature'
    )
)
combined_importance.rename(columns={'importance': 'importance_lgb'}, inplace=True)

combined_importance['importance_avg'] = (
    combined_importance['importance_cat'] +
    combined_importance['importance_xgb'] +
    combined_importance['importance_lgb']
) / 3

combined_importance = combined_importance.sort_values('importance_avg', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(
    data=combined_importance.head(15),
    x='importance_avg', y='feature',
    palette='viridis'
)
plt.title('ğŸ”� Top 15 Feature Importance - Super Ensemble (CatBoost + XGBoost + LightGBM)', fontsize=14)
plt.xlabel('Normalized Importance Score', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)
plt.tight_layout()
plt.show()

print("\nğŸ§  Top 10 Most Influential Features (Averaged Across Models):\n")
display(
    combined_importance.head(10)[[
        'feature', 
        'importance_avg', 
        'importance_cat', 
        'importance_xgb', 
        'importance_lgb'
    ]]
)


models_performance = {
    'CatBoost': cat_val_rmse,
    'XGBoost': xgb_val_rmse,
    'LightGBM': lgb_val_rmse,
    'Stacking': stack_val_rmse,
    'Super Ensemble': super_ensemble_val_rmse
}

plt.figure(figsize=(12, 6))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

plt.bar(models_performance.keys(), models_performance.values(), color=colors, alpha=0.8)
plt.ylabel('RMSE', fontsize=12)
plt.title('ğŸ“Š Model Performance Comparison (Lower is Better)', fontsize=14)
plt.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45)

for i, (model, rmse) in enumerate(models_performance.items()):
    plt.text(i, rmse + 0.001, f'{rmse:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()





plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.kdeplot(y_val, label='Actual', fill=True, alpha=0.5)
sns.kdeplot(cat_val_pred, label='CatBoost', fill=True, alpha=0.5)
plt.xlabel('Accident Risk')
plt.ylabel('Density')
plt.title('CatBoost vs Actual')
plt.legend()

plt.subplot(2, 3, 2)
sns.kdeplot(y_val, label='Actual', fill=True, alpha=0.5)
sns.kdeplot(xgb_val_pred, label='XGBoost', fill=True, alpha=0.5)
plt.xlabel('Accident Risk')
plt.ylabel('Density')
plt.title('XGBoost vs Actual')
plt.legend()

plt.subplot(2, 3, 3)
sns.kdeplot(y_val, label='Actual', fill=True, alpha=0.5)
sns.kdeplot(lgb_val_pred, label='LightGBM', fill=True, alpha=0.5)
plt.xlabel('Accident Risk')
plt.ylabel('Density')
plt.title('LightGBM vs Actual')
plt.legend()

plt.subplot(2, 3, 4)
sns.kdeplot(y_val, label='Actual', fill=True, alpha=0.5)
sns.kdeplot(stacking_val_pred, label='Stacking', fill=True, alpha=0.5)
plt.xlabel('Accident Risk')
plt.ylabel('Density')
plt.title('Stacking vs Actual')
plt.legend()

plt.subplot(2, 3, 5)
sns.kdeplot(y_val, label='Actual', fill=True, alpha=0.5)
sns.kdeplot(super_ensemble_val, label='Super Ensemble', fill=True, alpha=0.5)
plt.xlabel('Accident Risk')
plt.ylabel('Density')
plt.title('Super Ensemble vs Actual')
plt.legend()

plt.subplot(2, 3, 6)
sns.kdeplot(cat_val_pred, label='CatBoost', fill=True, alpha=0.3)
sns.kdeplot(xgb_val_pred, label='XGBoost', fill=True, alpha=0.3)
sns.kdeplot(lgb_val_pred, label='LightGBM', fill=True, alpha=0.3)
sns.kdeplot(super_ensemble_val, label='Super Ensemble', fill=True, alpha=0.3)
plt.xlabel('Accident Risk')
plt.ylabel('Density')
plt.title('All Models Distribution')
plt.legend()

plt.tight_layout()
plt.show()





feature_analysis_df = pd.DataFrame({
    'actual': y_val,
    'predicted': super_ensemble_val,  
    'error': super_ensemble_val - y_val, 
    'abs_error': np.abs(super_ensemble_val - y_val) 
})

important_features = ['speed_limit', 'curvature', 'num_lanes', 
                      'composite_risk', 'visibility_risk', 'complexity_score']

for feature in important_features:
    if feature in X_val.columns:
        feature_analysis_df[feature] = X_val[feature].values
    else:
        print(f"Warning: {feature} not found â€” switching to fallback features.")
        important_features = ['speed_limit', 'curvature', 'num_lanes', 
                              'weather_risk', 'time_risk', 'environment_risk']
        break

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, feature in enumerate(important_features[:6]):
    if feature in feature_analysis_df.columns:
        axes[i].scatter(feature_analysis_df[feature], feature_analysis_df['abs_error'], 
                        alpha=0.6, s=20, color='purple')
        z = np.polyfit(feature_analysis_df[feature], feature_analysis_df['abs_error'], 1)
        p = np.poly1d(z)
        axes[i].plot(feature_analysis_df[feature], p(feature_analysis_df[feature]), 
                     "r--", alpha=0.8, linewidth=2)
        
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Absolute Error')
        axes[i].set_title(f'Error vs {feature}\nSuper Ensemble')
        axes[i].grid(alpha=0.3)
    else:
        axes[i].text(0.5, 0.5, f'{feature} not available', 
                     ha='center', va='center', transform=axes[i].transAxes)
        axes[i].set_title(f'Missing: {feature}')

for i in range(len(important_features), 6):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

print("\nError Analysis Summary:")
for feature in important_features[:3]:  
    if feature in feature_analysis_df.columns:
        feature_analysis_df[f'{feature}_bin'] = pd.cut(feature_analysis_df[feature], bins=4)
        error_by_bin = feature_analysis_df.groupby(f'{feature}_bin')['abs_error'].mean()
        print(f"\n{feature} - Mean Absolute Error by range:")
        for bin_range, error in error_by_bin.items():
            print(f"  {bin_range}: {error:.4f}")




plt.figure(figsize=(20, 12))

plt.subplot(2, 4, 1)
plt.scatter(y_val, cat_val_pred, alpha=0.6, s=20, color='#FF6B6B')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.8)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'CatBoost\nRMSE: {cat_val_rmse:.4f}')
plt.grid(alpha=0.3)

plt.subplot(2, 4, 2)
plt.scatter(y_val, xgb_val_pred, alpha=0.6, s=20, color='#4ECDC4')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.8)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'XGBoost\nRMSE: {xgb_val_rmse:.4f}')
plt.grid(alpha=0.3)

plt.subplot(2, 4, 3)
plt.scatter(y_val, lgb_val_pred, alpha=0.6, s=20, color='#45B7D1')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.8)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'LightGBM\nRMSE: {lgb_val_rmse:.4f}')
plt.grid(alpha=0.3)

plt.subplot(2, 4, 4)
plt.scatter(y_val, super_ensemble_val, alpha=0.6, s=20, color='#96CEB4')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.8)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Super Ensemble\nRMSE: {super_ensemble_val_rmse:.4f}')
plt.grid(alpha=0.3)

plt.subplot(2, 4, 5)
cat_residuals = y_val - cat_val_pred
plt.scatter(cat_val_pred, cat_residuals, alpha=0.6, s=20, color='#FF6B6B')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('CatBoost Residuals')
plt.grid(alpha=0.3)

plt.subplot(2, 4, 6)
xgb_residuals = y_val - xgb_val_pred
plt.scatter(xgb_val_pred, xgb_residuals, alpha=0.6, s=20, color='#4ECDC4')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('XGBoost Residuals')
plt.grid(alpha=0.3)

plt.subplot(2, 4, 7)
lgb_residuals = y_val - lgb_val_pred
plt.scatter(lgb_val_pred, lgb_residuals, alpha=0.6, s=20, color='#45B7D1')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('LightGBM Residuals')
plt.grid(alpha=0.3)

plt.subplot(2, 4, 8)
super_ensemble_residuals = y_val - super_ensemble_val
plt.scatter(super_ensemble_val, super_ensemble_residuals, alpha=0.6, s=20, color='#96CEB4')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Super Ensemble Residuals')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()




plt.figure(figsize=(20, 6))

plt.subplot(1, 4, 1)
plt.hist(cat_test_pred, bins=30, alpha=0.7, edgecolor='black', color='#FF6B6B', label='CatBoost')
plt.xlabel('Predicted Accident Risk')
plt.ylabel('Frequency')
plt.title('CatBoost Test Predictions')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 4, 2)
plt.hist(xgb_test_pred, bins=30, alpha=0.7, edgecolor='black', color='#4ECDC4', label='XGBoost')
plt.xlabel('Predicted Accident Risk')
plt.ylabel('Frequency')
plt.title('XGBoost Test Predictions')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 4, 3)
plt.hist(lgb_test_pred, bins=30, alpha=0.7, edgecolor='black', color='#45B7D1', label='LightGBM')
plt.xlabel('Predicted Accident Risk')
plt.ylabel('Frequency')
plt.title('LightGBM Test Predictions')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 4, 4)
plt.hist(super_ensemble_test, bins=30, alpha=0.7, edgecolor='black', color='#96CEB4', label='Super Ensemble')
plt.xlabel('Predicted Accident Risk')
plt.ylabel('Frequency')
plt.title('Super Ensemble Test Predictions')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
sns.kdeplot(cat_test_pred, label='CatBoost', fill=True, alpha=0.3, color='#FF6B6B')
sns.kdeplot(xgb_test_pred, label='XGBoost', fill=True, alpha=0.3, color='#4ECDC4')
sns.kdeplot(lgb_test_pred, label='LightGBM', fill=True, alpha=0.3, color='#45B7D1')
sns.kdeplot(super_ensemble_test, label='Super Ensemble', fill=True, alpha=0.3, color='#96CEB4')
plt.xlabel('Predicted Accident Risk')
plt.ylabel('Density')
plt.title('Test Predictions Distribution - All Models')
plt.legend()
plt.grid(alpha=0.3)
plt.show()




submission_df = pd.DataFrame({
    'id': test_df['id'].values,
    'accident_risk': super_ensemble_test
})

submission_df['accident_risk'] = submission_df['accident_risk'].clip(0, 1)

submission_df.to_csv('/kaggle/working/submission.csv', index=False)

print("\nâœ… Super Ensemble predictions saved successfully!")
print(f"Prediction range: [{submission_df['accident_risk'].min():.3f}, {submission_df['accident_risk'].max():.3f}]")
print(f"Mean prediction: {submission_df['accident_risk'].mean():.3f}")





