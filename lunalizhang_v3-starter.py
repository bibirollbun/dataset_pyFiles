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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import xgboost as xgb
import lightgbm as lgb
import warnings
import os
import pickle
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# =============================================
# SETUP MODEL DIRECTORY STRUCTURE
# =============================================

def setup_model_directory(model_name="model_1"):
    """Create directory structure for saving model outputs"""
    base_dir = f"/kaggle/working/{model_name}"
    
    # Create subdirectories
    dirs = {
        'base': base_dir,
        'submissions': f"{base_dir}/submissions",
        'models': f"{base_dir}/models",
        'features': f"{base_dir}/features",
        'plots': f"{base_dir}/plots",
        'logs': f"{base_dir}/logs"
    }
    
    for dir_name, dir_path in dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return dirs

# Setup directories
MODEL_DIRS = setup_model_directory("model_1")

# Create a log file
log_file = os.path.join(MODEL_DIRS['logs'], f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

def log_message(message, print_msg=True):
    """Log messages to file and optionally print"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    
    if print_msg:
        print(message)
    
    with open(log_file, 'a') as f:
        f.write(log_msg + '\n')

# =============================================
# 1. DATA LOADING
# =============================================

log_message("="*60)
log_message("STARTING PRECIPITATION PREDICTION PIPELINE")
log_message("="*60)

# Load the data
train_df = pd.read_csv('/kaggle/input/oemc-hackathon-global-precipitation-modeling/train.csv')
test_df = pd.read_csv('/kaggle/input/oemc-hackathon-global-precipitation-modeling/test.csv')

log_message(f"Train shape: {train_df.shape}")
log_message(f"Test shape: {test_df.shape}")
log_message(f"\nTrain columns: {train_df.columns.tolist()}")

# Save original test IDs for submission
test_ids = test_df['HACKATHON_ID'].copy()

# =============================================
# 2. DATA EXPLORATION & VISUALIZATION
# =============================================

log_message("\n" + "="*50)
log_message("DATA EXPLORATION")
log_message("="*50)

# Check data types and missing values
log_message("\nMissing values in train:")
log_message(str(train_df.isnull().sum()))
log_message("\nMissing values in test:")
log_message(str(test_df.isnull().sum()))

# Basic statistics
stats_df = train_df.describe()
stats_df.to_csv(os.path.join(MODEL_DIRS['features'], 'data_statistics.csv'))
log_message("\nBasic statistics saved to features/data_statistics.csv")

# Target variable analysis
log_message("\nTARGET VARIABLE ANALYSIS")
log_message(f"Min precipitation: {train_df['PRCP'].min()}")
log_message(f"Max precipitation: {train_df['PRCP'].max()}")
log_message(f"Mean precipitation: {train_df['PRCP'].mean():.2f}")
log_message(f"Median precipitation: {train_df['PRCP'].median():.2f}")
log_message(f"Std deviation: {train_df['PRCP'].std():.2f}")

# Create log-transformed target
train_df['PRCP_LOG'] = np.log1p(train_df['PRCP'])

# Enhanced Visualizations
fig, axes = plt.subplots(3, 3, figsize=(20, 15))

# 1. Distribution of precipitation (original scale)
axes[0, 0].hist(train_df['PRCP'], bins=100, edgecolor='black')
axes[0, 0].set_xlabel('Precipitation (mm)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Precipitation (Original Scale)')
axes[0, 0].axvline(train_df['PRCP'].median(), color='red', linestyle='--', label='Median')
axes[0, 0].legend()

# 2. Distribution of log-transformed precipitation
axes[0, 1].hist(train_df['PRCP_LOG'], bins=100, edgecolor='black')
axes[0, 1].set_xlabel('Log(1 + Precipitation)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Log-Transformed Precipitation')
axes[0, 1].axvline(train_df['PRCP_LOG'].median(), color='red', linestyle='--', label='Median')
axes[0, 1].legend()

# 3. Box plot of precipitation by month
if 'DATE' in train_df.columns:
    train_df['month'] = pd.to_datetime(train_df['DATE']).dt.month
    train_df.boxplot(column='PRCP_LOG', by='month', ax=axes[0, 2])
    axes[0, 2].set_xlabel('Month')
    axes[0, 2].set_ylabel('Log(1 + Precipitation)')
    axes[0, 2].set_title('Precipitation by Month')

# 4. Correlation heatmap
feature_cols = ['TMIN_GEOM', 'TMAX_GEOM', 'PRCP_CHELSA', 'PRCP_ERA5', 
                'PRCP_TOTAL_IMERG', 'PRCP_LIQUID_IMERG', 'PRCP_ICE_IMERG']
corr_matrix = train_df[feature_cols + ['PRCP_LOG']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', ax=axes[1, 0], cmap='coolwarm')
axes[1, 0].set_title('Feature Correlation Matrix')

# 5. Scatter: ERA5 vs Target
axes[1, 1].scatter(train_df['PRCP_ERA5'], train_df['PRCP'], alpha=0.3, s=1)
axes[1, 1].set_xlabel('ERA5 Precipitation')
axes[1, 1].set_ylabel('Observed Precipitation')
axes[1, 1].set_title('ERA5 vs Observed Precipitation')

# 6. Scatter: CHELSA vs Target
axes[1, 2].scatter(train_df['PRCP_CHELSA'], train_df['PRCP'], alpha=0.3, s=1)
axes[1, 2].set_xlabel('CHELSA Precipitation')
axes[1, 2].set_ylabel('Observed Precipitation')
axes[1, 2].set_title('CHELSA vs Observed Precipitation')

# 7. Scatter: IMERG Total vs Target
axes[2, 0].scatter(train_df['PRCP_TOTAL_IMERG'], train_df['PRCP'], alpha=0.3, s=1)
axes[2, 0].set_xlabel('IMERG Total Precipitation')
axes[2, 0].set_ylabel('Observed Precipitation')
axes[2, 0].set_title('IMERG Total vs Observed Precipitation')

# 8. Temperature vs Precipitation
axes[2, 1].scatter((train_df['TMAX_GEOM'] + train_df['TMIN_GEOM'])/2, 
                   train_df['PRCP_LOG'], alpha=0.3, s=1)
axes[2, 1].set_xlabel('Mean Temperature')
axes[2, 1].set_ylabel('Log(1 + Precipitation)')
axes[2, 1].set_title('Temperature vs Precipitation')

# 9. QQ plot for log-transformed target
from scipy import stats
stats.probplot(train_df['PRCP_LOG'], dist="norm", plot=axes[2, 2])
axes[2, 2].set_title('Q-Q Plot of Log-Transformed Precipitation')

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIRS['plots'], 'data_exploration.png'), dpi=100, bbox_inches='tight')
plt.show()

log_message("\nData exploration plots saved to plots/data_exploration.png")

# =============================================
# 3. ADVANCED FEATURE ENGINEERING
# =============================================

def create_features(df, is_train=True):
    """Create additional features for the model"""
    df = df.copy()
    
    # Extract temporal features from DATE if available
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'])
        df['year'] = df['DATE'].dt.year
        df['month'] = df['DATE'].dt.month
        df['day_of_year'] = df['DATE'].dt.dayofyear
        
        # Seasonal features (sine/cosine encoding for cyclical nature)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Season indicator
        df['season'] = df['month'].apply(lambda x: (x%12 + 3)//3)
    
    # Temperature-based features
    df['temp_range'] = df['TMAX_GEOM'] - df['TMIN_GEOM']
    df['temp_mean'] = (df['TMAX_GEOM'] + df['TMIN_GEOM']) / 2
    df['temp_std'] = df[['TMAX_GEOM', 'TMIN_GEOM']].std(axis=1)
    df['temp_product'] = df['TMAX_GEOM'] * df['TMIN_GEOM']
    
    # Precipitation ensemble features
    prcp_cols = ['PRCP_CHELSA', 'PRCP_ERA5', 'PRCP_TOTAL_IMERG']
    df['prcp_mean'] = df[prcp_cols].mean(axis=1)
    df['prcp_std'] = df[prcp_cols].std(axis=1)
    df['prcp_max'] = df[prcp_cols].max(axis=1)
    df['prcp_min'] = df[prcp_cols].min(axis=1)
    df['prcp_range'] = df['prcp_max'] - df['prcp_min']
    df['prcp_cv'] = df['prcp_std'] / (df['prcp_mean'] + 1e-5)  # Coefficient of variation
    
    # Weighted average (giving more weight to higher resolution data)
    df['prcp_weighted'] = (
        0.4 * df['PRCP_CHELSA'] +  # Highest resolution
        0.3 * df['PRCP_ERA5'] + 
        0.3 * df['PRCP_TOTAL_IMERG']
    )
    
    # Ratios between different precipitation sources
    df['ratio_chelsa_era5'] = df['PRCP_CHELSA'] / (df['PRCP_ERA5'] + 1e-5)
    df['ratio_imerg_era5'] = df['PRCP_TOTAL_IMERG'] / (df['PRCP_ERA5'] + 1e-5)
    df['ratio_liquid_total'] = df['PRCP_LIQUID_IMERG'] / (df['PRCP_TOTAL_IMERG'] + 1e-5)
    df['ratio_ice_total'] = df['PRCP_ICE_IMERG'] / (df['PRCP_TOTAL_IMERG'] + 1e-5)
    
    # Differences between sources (disagreement indicators)
    df['diff_chelsa_era5'] = df['PRCP_CHELSA'] - df['PRCP_ERA5']
    df['diff_imerg_era5'] = df['PRCP_TOTAL_IMERG'] - df['PRCP_ERA5']
    df['diff_chelsa_imerg'] = df['PRCP_CHELSA'] - df['PRCP_TOTAL_IMERG']
    
    # Log-transformed precipitation features (to handle skewness)
    for col in ['PRCP_CHELSA', 'PRCP_ERA5', 'PRCP_TOTAL_IMERG', 
                'PRCP_LIQUID_IMERG', 'PRCP_ICE_IMERG']:
        df[f'{col}_log'] = np.log1p(df[col])
    
    # Interaction features
    df['temp_prcp_interaction'] = df['temp_mean'] * df['prcp_mean']
    df['temp_range_prcp'] = df['temp_range'] * df['prcp_mean']
    
    # Extreme indicators
    df['is_freezing'] = (df['TMAX_GEOM'] < 0).astype(int)
    df['is_hot'] = (df['TMIN_GEOM'] > 25).astype(int)
    
    return df

# Apply feature engineering
log_message("\nApplying feature engineering...")
train_df = create_features(train_df, is_train=True)
test_df = create_features(test_df, is_train=False)

log_message(f"Features after engineering:")
log_message(f"Train shape: {train_df.shape}")
log_message(f"Test shape: {test_df.shape}")

# =============================================
# 4. PREPARE DATA FOR MODELING
# =============================================

# Define feature columns
base_features = ['TMIN_GEOM', 'TMAX_GEOM', 'PRCP_CHELSA', 'PRCP_ERA5', 
                 'PRCP_TOTAL_IMERG', 'PRCP_LIQUID_IMERG', 'PRCP_ICE_IMERG']

engineered_features = ['month_sin', 'month_cos', 'day_sin', 'day_cos',
                      'temp_range', 'temp_mean', 'temp_std', 'temp_product',
                      'prcp_mean', 'prcp_std', 'prcp_max', 'prcp_min', 
                      'prcp_range', 'prcp_cv', 'prcp_weighted',
                      'ratio_chelsa_era5', 'ratio_imerg_era5', 
                      'ratio_liquid_total', 'ratio_ice_total',
                      'diff_chelsa_era5', 'diff_imerg_era5', 'diff_chelsa_imerg',
                      'PRCP_CHELSA_log', 'PRCP_ERA5_log', 'PRCP_TOTAL_IMERG_log',
                      'PRCP_LIQUID_IMERG_log', 'PRCP_ICE_IMERG_log',
                      'temp_prcp_interaction', 'temp_range_prcp',
                      'is_freezing', 'is_hot']

# Combine all features
feature_cols = base_features + engineered_features

# Remove any features that don't exist in test
feature_cols = [f for f in feature_cols if f in test_df.columns]

log_message(f"\nUsing {len(feature_cols)} features for modeling")

# Save feature list
with open(os.path.join(MODEL_DIRS['features'], 'feature_list.json'), 'w') as f:
    json.dump(feature_cols, f, indent=2)
log_message("Feature list saved to features/feature_list.json")

# Prepare training data
X = train_df[feature_cols]
y = train_df['PRCP_LOG']
X_test = test_df[feature_cols]

# Handle missing values
X = X.fillna(X.mean())
X_test = X_test.fillna(X.mean())

# Save feature statistics for later use
feature_stats = pd.DataFrame({
    'mean': X.mean(),
    'std': X.std(),
    'min': X.min(),
    'max': X.max()
})
feature_stats.to_csv(os.path.join(MODEL_DIRS['features'], 'feature_stats.csv'))

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

log_message(f"\nTraining set size: {X_train.shape}")
log_message(f"Validation set size: {X_val.shape}")
log_message(f"Test set size: {X_test.shape}")

# =============================================
# 5. MODEL TRAINING WITH CROSS-VALIDATION
# =============================================

# Dictionary to store models and scores
models = {}
scores = {}
cv_scores = {}

log_message("\n" + "="*50)
log_message("MODEL TRAINING WITH CROSS-VALIDATION")
log_message("="*50)

# Setup k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

def train_and_evaluate(model, model_name, X_train, y_train, X_val, y_val, use_cv=True):
    """Train model and evaluate with optional cross-validation"""
    log_message(f"\nTraining {model_name}...")
    
    # Cross-validation
    if use_cv:
        cv_score = cross_val_score(model, X_train, y_train, cv=kfold, 
                                   scoring='r2', n_jobs=-1)
        log_message(f"   CV R² Scores: {cv_score}")
        log_message(f"   CV Mean R²: {cv_score.mean():.4f} (+/- {cv_score.std() * 2:.4f})")
        cv_scores[model_name] = cv_score.mean()
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Validate
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    
    log_message(f"   Validation R²: {r2:.4f}")
    log_message(f"   Validation RMSE: {rmse:.4f}")
    log_message(f"   Validation MAE: {mae:.4f}")
    
    return model, r2

# 1. Linear Regression (Baseline)
lr = LinearRegression()
lr, score_lr = train_and_evaluate(lr, 'LinearRegression', X_train, y_train, X_val, y_val, use_cv=False)
models['LinearRegression'] = lr
scores['LinearRegression'] = score_lr

# 2. Ridge Regression
ridge = Ridge(alpha=1.0)
ridge, score_ridge = train_and_evaluate(ridge, 'Ridge', X_train, y_train, X_val, y_val, use_cv=False)
models['Ridge'] = ridge
scores['Ridge'] = score_ridge

# 3. Random Forest
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf, score_rf = train_and_evaluate(rf, 'RandomForest', X_train, y_train, X_val, y_val, use_cv=False)
models['RandomForest'] = rf
scores['RandomForest'] = score_rf

# 4. Extra Trees
et = ExtraTreesRegressor(
    n_estimators=200,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
et, score_et = train_and_evaluate(et, 'ExtraTrees', X_train, y_train, X_val, y_val, use_cv=False)
models['ExtraTrees'] = et
scores['ExtraTrees'] = score_et

# 5. XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model, score_xgb = train_and_evaluate(xgb_model, 'XGBoost', X_train, y_train, X_val, y_val, use_cv=False)
models['XGBoost'] = xgb_model
scores['XGBoost'] = score_xgb

# 6. LightGBM
lgb_model = lgb.LGBMRegressor(
    n_estimators=200,
    max_depth=12,
    learning_rate=0.05,
    num_leaves=31,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_model, score_lgb = train_and_evaluate(lgb_model, 'LightGBM', X_train, y_train, X_val, y_val, use_cv=False)
models['LightGBM'] = lgb_model
scores['LightGBM'] = score_lgb

# Display all scores
log_message("\n" + "="*50)
log_message("MODEL COMPARISON")
log_message("="*50)
results_df = pd.DataFrame({
    'Model': list(scores.keys()),
    'Validation_R2': list(scores.values())
})
results_df = results_df.sort_values('Validation_R2', ascending=False)
log_message("\n" + str(results_df))

# Save results
results_df.to_csv(os.path.join(MODEL_DIRS['logs'], 'model_comparison.csv'), index=False)

# Select best model
best_model_name = max(scores, key=scores.get)
best_model = models[best_model_name]
log_message(f"\nBest model: {best_model_name} with R² = {scores[best_model_name]:.4f}")

# =============================================
# 6. SAVE MODELS
# =============================================

log_message("\n" + "="*50)
log_message("SAVING MODELS")
log_message("="*50)

# Save all models
for model_name, model in models.items():
    model_path = os.path.join(MODEL_DIRS['models'], f'{model_name}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    log_message(f"Saved {model_name} to models/{model_name}.pkl")

# Save best model separately
best_model_path = os.path.join(MODEL_DIRS['models'], 'best_model.pkl')
with open(best_model_path, 'wb') as f:
    pickle.dump(best_model, f)
log_message(f"Best model saved to models/best_model.pkl")

# Save model metadata
metadata = {
    'best_model': best_model_name,
    'best_score': scores[best_model_name],
    'all_scores': scores,
    'feature_count': len(feature_cols),
    'training_samples': X_train.shape[0],
    'validation_samples': X_val.shape[0],
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}
with open(os.path.join(MODEL_DIRS['models'], 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

# =============================================
# 7. FEATURE IMPORTANCE ANALYSIS
# =============================================

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    log_message("\n" + "="*50)
    log_message("TOP 15 FEATURE IMPORTANCES")
    log_message("="*50)
    log_message("\n" + str(feature_importance.head(15)))
    
    # Save feature importance
    feature_importance.to_csv(os.path.join(MODEL_DIRS['features'], 'feature_importance.csv'), index=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 10))
    top_features = feature_importance.head(25)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top 25 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIRS['plots'], 'feature_importance.png'), dpi=100, bbox_inches='tight')
    plt.show()

# =============================================
# 8. ENSEMBLE PREDICTIONS
# =============================================

log_message("\n" + "="*50)
log_message("CREATING ENSEMBLE PREDICTIONS")
log_message("="*50)

# Weighted ensemble based on performance
ensemble_models = ['XGBoost', 'LightGBM', 'RandomForest', 'ExtraTrees']
ensemble_weights = []
ensemble_preds = []

for model_name in ensemble_models:
    if model_name in models:
        weight = scores[model_name]
        ensemble_weights.append(weight)
        preds = models[model_name].predict(X_test)
        ensemble_preds.append(preds)
        log_message(f"Added {model_name} to ensemble with weight {weight:.4f}")

# Normalize weights
ensemble_weights = np.array(ensemble_weights)
ensemble_weights = ensemble_weights / ensemble_weights.sum()

# Weighted average
ensemble_final = np.average(ensemble_preds, axis=0, weights=ensemble_weights)

log_message(f"Ensemble created with {len(ensemble_models)} models")
log_message(f"Normalized weights: {ensemble_weights}")

# =============================================
# 9. CREATE AND SAVE SUBMISSIONS
# =============================================

log_message("\n" + "="*50)
log_message("CREATING SUBMISSIONS")
log_message("="*50)

# Create submission with best single model
submission_best = pd.DataFrame({
    'HACKATHON_ID': test_ids,
    'PRCP_LOG': best_model.predict(X_test)
})

# Create ensemble submission
submission_ensemble = pd.DataFrame({
    'HACKATHON_ID': test_ids,
    'PRCP_LOG': ensemble_final
})

# Create submission for each model
for model_name, model in models.items():
    submission_model = pd.DataFrame({
        'HACKATHON_ID': test_ids,
        'PRCP_LOG': model.predict(X_test)
    })
    submission_path = os.path.join(MODEL_DIRS['submissions'], f'submission_{model_name.lower()}.csv')
    submission_model.to_csv(submission_path, index=False)
    log_message(f"Saved {model_name} submission to submissions/submission_{model_name.lower()}.csv")

# Save main submissions
best_submission_path = os.path.join(MODEL_DIRS['submissions'], 'submission_best_model.csv')
ensemble_submission_path = os.path.join(MODEL_DIRS['submissions'], 'submission_ensemble.csv')

submission_best.to_csv(best_submission_path, index=False)
submission_ensemble.to_csv(ensemble_submission_path, index=False)

log_message(f"\nBest model submission saved to: {best_submission_path}")
log_message(f"Ensemble submission saved to: {ensemble_submission_path}")

# Display submission statistics
log_message("\nSubmission Statistics:")
log_message(f"Best Model - Mean: {submission_best['PRCP_LOG'].mean():.4f}, Std: {submission_best['PRCP_LOG'].std():.4f}")
log_message(f"Ensemble - Mean: {submission_ensemble['PRCP_LOG'].mean():.4f}, Std: {submission_ensemble['PRCP_LOG'].std():.4f}")

# =============================================
# 10. VALIDATION PREDICTIONS ANALYSIS
# =============================================

log_message("\n" + "="*50)
log_message("VALIDATION PREDICTIONS ANALYSIS")
log_message("="*50)

# Get predictions on validation set
val_predictions = {}
for model_name, model in models.items():
    val_predictions[model_name] = model.predict(X_val)

# Create validation ensemble
val_ensemble = np.average(
    [val_predictions[m] for m in ensemble_models if m in val_predictions],
    axis=0,
    weights=ensemble_weights
)

# Calculate ensemble validation score
ensemble_val_score = r2_score(y_val, val_ensemble)
log_message(f"Ensemble Validation R²: {ensemble_val_score:.4f}")

# Plot actual vs predicted for best model and ensemble
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Best single model
axes[0].scatter(y_val, val_predictions[best_model_name], alpha=0.5, s=1)
axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Log(1 + Precipitation)')
axes[0].set_ylabel('Predicted Log(1 + Precipitation)')
axes[0].set_title(f'{best_model_name} - Actual vs Predicted (R² = {scores[best_model_name]:.4f})')
axes[0].grid(True, alpha=0.3)

# Ensemble
axes[1].scatter(y_val, val_ensemble, alpha=0.5, s=1)
axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Log(1 + Precipitation)')
axes[1].set_ylabel('Predicted Log(1 + Precipitation)')
axes[1].set_title(f'Ensemble - Actual vs Predicted (R² = {ensemble_val_score:.4f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIRS['plots'], 'actual_vs_predicted.png'), dpi=100, bbox_inches='tight')
plt.show()

# =============================================
# FINAL SUMMARY
# =============================================

log_message("\n" + "="*60)
log_message("PIPELINE COMPLETE!")
log_message("="*60)
log_message(f"\nAll outputs saved to: /kaggle/working/model_1/")
log_message(f"Best Model: {best_model_name} (R² = {scores[best_model_name]:.4f})")
log_message(f"Ensemble R² on Validation: {ensemble_val_score:.4f}")
log_message(f"\nDirectory Structure:")
log_message(f"  - submissions/  : Contains all submission files")
log_message(f"  - models/       : Saved model pickle files")
log_message(f"  - features/     : Feature lists and importance")
log_message(f"  - plots/        : Visualization outputs")
log_message(f"  - logs/         : Training logs and results")
log_message(f"\nLog file saved to: {log_file}")

# Display directory contents
log_message("\nGenerated Files:")
for dir_name, dir_path in MODEL_DIRS.items():
    files = os.listdir(dir_path)
    if files:
        log_message(f"\n{dir_name}:")
        for file in files:
            file_path = os.path.join(dir_path, file)
            file_size = os.path.getsize(file_path) / 1024  # Size in KB
            log_message(f"  - {file} ({file_size:.1f} KB)")

log_message("\n" + "="*60)
log_message("Ready for submission!")
log_message("="*60)

