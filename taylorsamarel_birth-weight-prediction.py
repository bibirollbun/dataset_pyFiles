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


#!/usr/bin/env python3
"""
Birth Weight Prediction - FULLY WORKING VERSION
Handles all data types including categorical variables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import xgboost as xgb
import lightgbm as lgb

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error

import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("="*80)
print(" "*20 + "BIRTH WEIGHT PREDICTION - WORKING VERSION")
print("="*80)

# =============================================
# 1. LOAD DATA
# =============================================

print("\n" + "="*60)
print("1. LOADING DATA")
print("="*60)

# Load data
train_df = pd.read_csv('/kaggle/input/fall-25-birth-weight-prediction/baby-weights-dataset.csv')
test_df = pd.read_csv('/kaggle/input/fall-25-birth-weight-prediction/judge-without-labels.csv')

print(f"✓ Training data: {train_df.shape}")
print(f"✓ Test data: {test_df.shape}")

# =============================================
# 2. DATA PREPROCESSING
# =============================================

print("\n" + "="*60)
print("2. DATA PREPROCESSING")
print("="*60)

# Separate target
X = train_df.drop('BWEIGHT', axis=1)
y = train_df['BWEIGHT']
X_test = test_df.copy()

# Remove ID column if present
if 'ID' in X.columns:
    X = X.drop('ID', axis=1)
if 'ID' in X_test.columns:
    X_test = X_test.drop('ID', axis=1)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Identify column types
print("\nIdentifying column types...")
categorical_columns = []
numerical_columns = []

for col in X.columns:
    if X[col].dtype == 'object':
        categorical_columns.append(col)
    else:
        # Check if numeric column might be categorical
        unique_vals = X[col].nunique()
        if unique_vals < 10 and X[col].min() >= 0 and X[col].max() < 100:
            # Likely categorical encoded as numbers
            if unique_vals == 2:
                # Binary, keep as numeric
                numerical_columns.append(col)
            else:
                # Multi-class, treat as categorical
                categorical_columns.append(col)
        else:
            numerical_columns.append(col)

print(f"Numerical columns: {len(numerical_columns)}")
print(f"Categorical columns: {len(categorical_columns)}")

# Handle string categorical columns
string_categorical = [col for col in categorical_columns if X[col].dtype == 'object']
if string_categorical:
    print(f"\nEncoding {len(string_categorical)} string categorical columns...")
    
    for col in string_categorical:
        # Use label encoding for binary/ordinal
        le = LabelEncoder()
        
        # Fit on combined train and test to handle unseen categories
        combined = pd.concat([X[col], X_test[col]], axis=0)
        le.fit(combined.astype(str))
        
        # Transform
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        
        # Move to numerical after encoding
        numerical_columns.append(col)
        categorical_columns.remove(col)

# Handle missing values
print("\nHandling missing values...")
for col in X.columns:
    if X[col].isnull().sum() > 0:
        if col in numerical_columns:
            # Fill with median for numerical
            fill_value = X[col].median()
            X[col].fillna(fill_value, inplace=True)
            X_test[col].fillna(fill_value, inplace=True)
        else:
            # Fill with mode for categorical
            fill_value = X[col].mode()[0] if len(X[col].mode()) > 0 else 0
            X[col].fillna(fill_value, inplace=True)
            X_test[col].fillna(fill_value, inplace=True)

# Ensure all columns are numeric
print("\nConverting all columns to numeric...")
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# Fill any remaining NaN with 0
X = X.fillna(0)
X_test = X_test.fillna(0)

# Check for infinite values
X = X.replace([np.inf, -np.inf], 0)
X_test = X_test.replace([np.inf, -np.inf], 0)

print(f"✓ Preprocessing complete")
print(f"  Final X shape: {X.shape}")
print(f"  Final X_test shape: {X_test.shape}")

# =============================================
# 3. FEATURE ENGINEERING
# =============================================

print("\n" + "="*60)
print("3. FEATURE ENGINEERING")
print("="*60)

# Calculate correlations
correlations = X.corrwith(y).abs().sort_values(ascending=False)
top_features = correlations.head(10).index.tolist()

print("Top correlated features:")
for i, (feat, corr) in enumerate(correlations.head(10).items(), 1):
    print(f"  {i}. {feat}: {corr:.3f}")

# Add interaction features for top 3
print("\nAdding interaction features...")
top_3 = top_features[:3]
for i, feat1 in enumerate(top_3):
    for feat2 in top_3[i+1:]:
        interaction_name = f'{feat1}_x_{feat2}'
        X[interaction_name] = X[feat1] * X[feat2]
        X_test[interaction_name] = X_test[feat1] * X_test[feat2]
        print(f"  Added: {interaction_name}")

# Add polynomial features for top feature
if len(top_features) > 0:
    top_feat = top_features[0]
    X[f'{top_feat}_squared'] = X[top_feat] ** 2
    X_test[f'{top_feat}_squared'] = X_test[top_feat] ** 2
    X[f'{top_feat}_sqrt'] = np.sqrt(np.abs(X[top_feat]))
    X_test[f'{top_feat}_sqrt'] = np.sqrt(np.abs(X_test[top_feat]))
    print(f"  Added polynomial features for {top_feat}")

print(f"\n✓ Feature engineering complete")
print(f"  Total features: {X.shape[1]}")

# =============================================
# 4. MODEL TRAINING
# =============================================

print("\n" + "="*60)
print("4. MODEL TRAINING")
print("="*60)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=0.01, random_state=42),
    'ElasticNet': ElasticNet(alpha=0.01, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Extra Trees': ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)
}

# Train models
results = {}
predictions = {}

print("\nTraining models...")
print("-" * 50)

for name, model in models.items():
    try:
        print(f"{name}...", end=' ')
        
        # Use scaled data for linear models
        if name in ['Ridge', 'Lasso', 'ElasticNet']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            y_pred_test = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_pred_test = model.predict(X_test)
        
        # Ensure positive predictions
        y_pred = np.maximum(y_pred, 0.01)
        y_pred_test = np.maximum(y_pred_test, 0.01)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        # RMSLE
        try:
            rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
        except:
            rmsle = rmse
        
        results[name] = {
            'RMSE': rmse,
            'RMSLE': rmsle,
            'MAE': mae,
            'R2': r2
        }
        predictions[name] = y_pred_test
        
        print(f"RMSLE: {rmsle:.4f}, R²: {r2:.3f}")
        
    except Exception as e:
        print(f"Failed: {str(e)[:30]}")
        results[name] = {'RMSE': np.inf, 'RMSLE': np.inf, 'MAE': np.inf, 'R2': -1}
        predictions[name] = np.full(len(X_test), y.mean())

# Create results DataFrame
results_df = pd.DataFrame(results).T.sort_values('RMSLE')

print("\n" + "="*50)
print("MODEL PERFORMANCE SUMMARY")
print("="*50)
print(results_df)

# =============================================
# 5. ENSEMBLE PREDICTION
# =============================================

print("\n" + "="*60)
print("5. ENSEMBLE PREDICTION")
print("="*60)

# Get valid models
valid_models = results_df[results_df['RMSLE'] < np.inf].index.tolist()

if len(valid_models) > 0:
    print(f"\nCreating ensemble from {len(valid_models)} models")
    
    # Calculate weights based on performance
    weights = []
    for model in valid_models:
        weight = 1 / (results_df.loc[model, 'RMSLE'] + 0.001)
        weights.append(weight)
    
    weights = np.array(weights) / np.sum(weights)
    
    # Weighted average
    ensemble_preds = np.zeros(len(X_test))
    for model, weight in zip(valid_models, weights):
        ensemble_preds += predictions[model] * weight
        print(f"  {model}: weight={weight:.3f}")
    
    final_predictions = ensemble_preds
else:
    print("No valid models, using baseline")
    final_predictions = np.full(len(X_test), y.mean())

# Ensure reasonable bounds
final_predictions = np.clip(final_predictions, 0.5, 15.0)

print(f"\nPrediction statistics:")
print(f"  Mean: {final_predictions.mean():.2f}")
print(f"  Std: {final_predictions.std():.2f}")
print(f"  Min: {final_predictions.min():.2f}")
print(f"  Max: {final_predictions.max():.2f}")

# =============================================
# 6. CREATE SUBMISSION
# =============================================

print("\n" + "="*60)
print("6. CREATING SUBMISSION")
print("="*60)

submission = pd.DataFrame({
    'ID': range(1, len(final_predictions) + 1),
    'BWEIGHT': final_predictions
})

submission.to_csv('submission.csv', index=False)
print("✓ Submission saved to 'submission.csv'")
print("\nFirst 10 predictions:")
print(submission.head(10))

# =============================================
# 7. VISUALIZATION
# =============================================

print("\n" + "="*60)
print("7. VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Model performance
ax1 = axes[0, 0]
valid_results = results_df[results_df['RMSLE'] < np.inf]
if len(valid_results) > 0:
    ax1.bar(range(len(valid_results)), valid_results['RMSLE'].values)
    ax1.set_xticks(range(len(valid_results)))
    ax1.set_xticklabels(valid_results.index, rotation=45)
    ax1.set_ylabel('RMSLE')
    ax1.set_title('Model Performance')
    ax1.grid(True, alpha=0.3)

# Prediction distribution
ax2 = axes[0, 1]
ax2.hist(final_predictions, bins=50, edgecolor='black', alpha=0.7)
ax2.set_xlabel('Predicted Birth Weight')
ax2.set_ylabel('Frequency')
ax2.set_title('Prediction Distribution')
ax2.axvline(final_predictions.mean(), color='red', linestyle='--', 
           label=f'Mean: {final_predictions.mean():.2f}')
ax2.legend()

# Training vs Prediction distributions
ax3 = axes[1, 0]
ax3.hist(y, bins=50, alpha=0.5, label='Training', color='blue')
ax3.hist(final_predictions, bins=50, alpha=0.5, label='Predictions', color='orange')
ax3.set_xlabel('Birth Weight')
ax3.set_ylabel('Frequency')
ax3.set_title('Training vs Predictions')
ax3.legend()

# Feature importance (if RF was successful)
ax4 = axes[1, 1]
if 'Random Forest' in valid_models:
    rf_model = models['Random Forest']
    importances = rf_model.feature_importances_
    top_feat_idx = np.argsort(importances)[-10:]
    top_feat_names = [X.columns[i] for i in top_feat_idx]
    top_feat_imp = importances[top_feat_idx]
    
    ax4.barh(range(len(top_feat_imp)), top_feat_imp)
    ax4.set_yticks(range(len(top_feat_imp)))
    ax4.set_yticklabels(top_feat_names, fontsize=8)
    ax4.set_xlabel('Importance')
    ax4.set_title('Top 10 Feature Importances')
else:
    ax4.text(0.5, 0.5, 'Feature importance\nnot available', 
            ha='center', va='center', fontsize=12)
    ax4.set_title('Feature Importance')

plt.suptitle('Birth Weight Prediction Analysis', fontsize=14)
plt.tight_layout()
plt.show()

print("\n" + "="*80)
print(" " * 25 + "COMPLETE!")
print("="*80)

if len(valid_results) > 0:
    best_model = valid_results.index[0]
    print(f"\nBest Model: {best_model}")
    print(f"RMSLE: {valid_results.loc[best_model, 'RMSLE']:.4f}")
    print(f"R² Score: {valid_results.loc[best_model, 'R2']:.3f}")

print("\n✓ Submission file 'submission.csv' is ready for upload!")
print("="*80)

