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
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# =============================================
# 1. DATA LOADING
# =============================================

# Load the data
train_df = pd.read_csv('/kaggle/input/oemc-hackathon-global-precipitation-modeling/train.csv')
test_df = pd.read_csv('/kaggle/input/oemc-hackathon-global-precipitation-modeling/test.csv')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nTrain columns:", train_df.columns.tolist())
print("\nFirst few rows of training data:")
print(train_df.head())

# =============================================
# 2. DATA EXPLORATION
# =============================================

# Check data types and missing values
print("\n" + "="*50)
print("DATA INFO")
print("="*50)
print("\nTrain data info:")
print(train_df.info())
print("\nMissing values in train:")
print(train_df.isnull().sum())
print("\nMissing values in test:")
print(test_df.isnull().sum())

# Basic statistics
print("\n" + "="*50)
print("BASIC STATISTICS")
print("="*50)
print(train_df.describe())

# Target variable analysis
print("\n" + "="*50)
print("TARGET VARIABLE ANALYSIS")
print("="*50)
print(f"Min precipitation: {train_df['PRCP'].min()}")
print(f"Max precipitation: {train_df['PRCP'].max()}")
print(f"Mean precipitation: {train_df['PRCP'].mean():.2f}")
print(f"Median precipitation: {train_df['PRCP'].median():.2f}")

# Create log-transformed target
train_df['PRCP_LOG'] = np.log1p(train_df['PRCP'])

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Distribution of precipitation (original scale)
axes[0, 0].hist(train_df['PRCP'], bins=50, edgecolor='black')
axes[0, 0].set_xlabel('Precipitation (mm)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Precipitation (Original Scale)')

# Distribution of log-transformed precipitation
axes[0, 1].hist(train_df['PRCP_LOG'], bins=50, edgecolor='black')
axes[0, 1].set_xlabel('Log(1 + Precipitation)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Log-Transformed Precipitation')

# Correlation heatmap
feature_cols = ['TMIN_GEOM', 'TMAX_GEOM', 'PRCP_CHELSA', 'PRCP_ERA5', 
                'PRCP_TOTAL_IMERG', 'PRCP_LIQUID_IMERG', 'PRCP_ICE_IMERG']
corr_matrix = train_df[feature_cols + ['PRCP_LOG']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', ax=axes[1, 0], cmap='coolwarm')
axes[1, 0].set_title('Feature Correlation Matrix')

# Scatter plot: ERA5 vs Target
axes[1, 1].scatter(train_df['PRCP_ERA5'], train_df['PRCP'], alpha=0.5, s=1)
axes[1, 1].set_xlabel('ERA5 Precipitation')
axes[1, 1].set_ylabel('Observed Precipitation')
axes[1, 1].set_title('ERA5 vs Observed Precipitation')

plt.tight_layout()
plt.show()

# =============================================
# 3. FEATURE ENGINEERING
# =============================================

def create_features(df):
    """Create additional features for the model"""
    df = df.copy()
    
    # Extract temporal features from DATE if available
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'])
        df['year'] = df['DATE'].dt.year
        df['month'] = df['DATE'].dt.month
        
        # Seasonal features (sine/cosine encoding for cyclical nature)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Temperature-based features
    df['temp_range'] = df['TMAX_GEOM'] - df['TMIN_GEOM']
    df['temp_mean'] = (df['TMAX_GEOM'] + df['TMIN_GEOM']) / 2
    
    # Precipitation ensemble features
    prcp_cols = ['PRCP_CHELSA', 'PRCP_ERA5', 'PRCP_TOTAL_IMERG']
    df['prcp_mean'] = df[prcp_cols].mean(axis=1)
    df['prcp_std'] = df[prcp_cols].std(axis=1)
    df['prcp_max'] = df[prcp_cols].max(axis=1)
    df['prcp_min'] = df[prcp_cols].min(axis=1)
    
    # Ratios between different precipitation sources
    df['ratio_chelsa_era5'] = df['PRCP_CHELSA'] / (df['PRCP_ERA5'] + 1e-5)
    df['ratio_imerg_era5'] = df['PRCP_TOTAL_IMERG'] / (df['PRCP_ERA5'] + 1e-5)
    df['ratio_liquid_total'] = df['PRCP_LIQUID_IMERG'] / (df['PRCP_TOTAL_IMERG'] + 1e-5)
    
    # Log-transformed precipitation features (to handle skewness)
    for col in ['PRCP_CHELSA', 'PRCP_ERA5', 'PRCP_TOTAL_IMERG', 
                'PRCP_LIQUID_IMERG', 'PRCP_ICE_IMERG']:
        df[f'{col}_log'] = np.log1p(df[col])
    
    return df

# Apply feature engineering
train_df = create_features(train_df)
test_df = create_features(test_df)

print("\nFeatures after engineering:")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# =============================================
# 4. PREPARE DATA FOR MODELING
# =============================================

# Define feature columns
base_features = ['TMIN_GEOM', 'TMAX_GEOM', 'PRCP_CHELSA', 'PRCP_ERA5', 
                 'PRCP_TOTAL_IMERG', 'PRCP_LIQUID_IMERG', 'PRCP_ICE_IMERG']

engineered_features = ['month_sin', 'month_cos', 'temp_range', 'temp_mean',
                      'prcp_mean', 'prcp_std', 'prcp_max', 'prcp_min',
                      'ratio_chelsa_era5', 'ratio_imerg_era5', 'ratio_liquid_total',
                      'PRCP_CHELSA_log', 'PRCP_ERA5_log', 'PRCP_TOTAL_IMERG_log',
                      'PRCP_LIQUID_IMERG_log', 'PRCP_ICE_IMERG_log']

# You can choose to use all features or just base features
feature_cols = base_features + engineered_features

# Remove any features that don't exist in test
feature_cols = [f for f in feature_cols if f in test_df.columns]

print(f"\nUsing {len(feature_cols)} features for modeling")

# Prepare training data
X = train_df[feature_cols]
y = train_df['PRCP_LOG']
X_test = test_df[feature_cols]

# Handle missing values
X = X.fillna(X.mean())
X_test = X_test.fillna(X.mean())

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")
print(f"Test set size: {X_test.shape}")

# =============================================
# 5. MODEL TRAINING
# =============================================

# Dictionary to store models and scores
models = {}
scores = {}

print("\n" + "="*50)
print("MODEL TRAINING")
print("="*50)

# 1. Linear Regression (Baseline)
print("\n1. Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_val)
score_lr = r2_score(y_val, y_pred_lr)
print(f"   R² Score: {score_lr:.4f}")
models['LinearRegression'] = lr
scores['LinearRegression'] = score_lr

# 2. Ridge Regression
print("\n2. Ridge Regression...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_val)
score_ridge = r2_score(y_val, y_pred_ridge)
print(f"   R² Score: {score_ridge:.4f}")
models['Ridge'] = ridge
scores['Ridge'] = score_ridge

# 3. Random Forest
print("\n3. Random Forest...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)
score_rf = r2_score(y_val, y_pred_rf)
print(f"   R² Score: {score_rf:.4f}")
models['RandomForest'] = rf
scores['RandomForest'] = score_rf

# 4. XGBoost
print("\n4. XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_val)
score_xgb = r2_score(y_val, y_pred_xgb)
print(f"   R² Score: {score_xgb:.4f}")
models['XGBoost'] = xgb_model
scores['XGBoost'] = score_xgb

# 5. LightGBM
print("\n5. LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_val)
score_lgb = r2_score(y_val, y_pred_lgb)
print(f"   R² Score: {score_lgb:.4f}")
models['LightGBM'] = lgb_model
scores['LightGBM'] = score_lgb

# Display all scores
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
for model_name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name:20s}: R² = {score:.4f}")

# Select best model
best_model_name = max(scores, key=scores.get)
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name} with R² = {scores[best_model_name]:.4f}")

# =============================================
# 6. FEATURE IMPORTANCE (for tree-based models)
# =============================================

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*50)
    print("TOP 15 FEATURE IMPORTANCES")
    print("="*50)
    print(feature_importance.head(15))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top 20 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.show()

# =============================================
# 7. ENSEMBLE PREDICTIONS (Optional)
# =============================================

print("\n" + "="*50)
print("CREATING ENSEMBLE PREDICTIONS")
print("="*50)

# Simple averaging ensemble of top models
top_models = ['XGBoost', 'LightGBM', 'RandomForest']
ensemble_preds = np.zeros(len(X_test))

for model_name in top_models:
    if model_name in models:
        preds = models[model_name].predict(X_test)
        ensemble_preds += preds / len(top_models)
        print(f"Added {model_name} to ensemble")

# =============================================
# 8. CREATE SUBMISSION
# =============================================

print("\n" + "="*50)
print("CREATING SUBMISSION")
print("="*50)

# Create submission with best single model
submission = pd.DataFrame({
    'HACKATHON_ID': test_df['HACKATHON_ID'],
    'PRCP_LOG': best_model.predict(X_test)
})

# Save submission
submission.to_csv('submission_best_model.csv', index=False)
print(f"Submission saved: submission_best_model.csv")
print(f"Submission shape: {submission.shape}")
print("\nFirst few predictions:")
print(submission.head())

# Create ensemble submission
submission_ensemble = pd.DataFrame({
    'HACKATHON_ID': test_df['HACKATHON_ID'],
    'PRCP_LOG': ensemble_preds
})
submission_ensemble.to_csv('submission_ensemble.csv', index=False)
print(f"\nEnsemble submission saved: submission_ensemble.csv")

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)

