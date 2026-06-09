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


# Podcast Listening Time Prediction - Kaggle Competition

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. LOAD DATA
print("Loading data...")
train_df = pd.read_csv('/kaggle/input/predict-podcast-listening-time/train.csv')
test_df = pd.read_csv('/kaggle/input/predict-podcast-listening-time/test.csv')
sample_submission = pd.read_csv('/kaggle/input/predict-podcast-listening-time/sample_submission.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")

# 2. EXPLORATORY DATA ANALYSIS (EDA)
print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Basic information about the dataset
print("\nTrain Dataset Info:")
print(train_df.info())

print("\n\nFirst 5 rows of training data:")
print(train_df.head())

print("\n\nColumn names:")
print(train_df.columns.tolist())

# Statistical summary
print("\n\nStatistical Summary of Numerical Features:")
print(train_df.describe())

# Target variable analysis
target_col = 'Listening_Time_minutes'
print(f"\n\nTarget Variable ({target_col}) Statistics:")
print(f"Mean: {train_df[target_col].mean():.2f}")
print(f"Median: {train_df[target_col].median():.2f}")
print(f"Std: {train_df[target_col].std():.2f}")
print(f"Min: {train_df[target_col].min():.2f}")
print(f"Max: {train_df[target_col].max():.2f}")

# Missing values
print("\n\nMissing Values in Training Data:")
missing_train = train_df.isnull().sum()
missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
if len(missing_train) > 0:
    print(missing_train)
    print(f"\nPercentage of missing values:")
    print((missing_train / len(train_df) * 100).round(2))
else:
    print("No missing values found!")

print("\n\nMissing Values in Test Data:")
missing_test = test_df.isnull().sum()
missing_test = missing_test[missing_test > 0].sort_values(ascending=False)
if len(missing_test) > 0:
    print(missing_test)
else:
    print("No missing values found!")

# Data types
print("\n\nData Types:")
print(train_df.dtypes.value_counts())

# Separate numerical and categorical columns
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

if 'id' in numerical_cols:
    numerical_cols.remove('id')
if target_col in numerical_cols:
    numerical_cols.remove(target_col)

print(f"\n\nNumerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# Visualizations
fig = plt.figure(figsize=(20, 15))

# Target distribution
plt.subplot(3, 3, 1)
train_df[target_col].hist(bins=50, edgecolor='black')
plt.title(f'Distribution of {target_col}')
plt.xlabel(target_col)
plt.ylabel('Frequency')

# Log-transformed target distribution
plt.subplot(3, 3, 2)
np.log1p(train_df[target_col]).hist(bins=50, edgecolor='black')
plt.title(f'Distribution of log({target_col})')
plt.xlabel(f'log({target_col})')
plt.ylabel('Frequency')

# Box plot of target
plt.subplot(3, 3, 3)
train_df.boxplot(column=target_col)
plt.title(f'Box Plot of {target_col}')

# Correlation heatmap (if numerical features exist)
if len(numerical_cols) > 0:
    plt.subplot(3, 3, 4)
    corr_matrix = train_df[numerical_cols + [target_col]].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()

# Top correlations with target
if len(numerical_cols) > 0:
    print("\n\nTop Features Correlated with Target:")
    correlations = train_df[numerical_cols].corrwith(train_df[target_col]).sort_values(ascending=False)
    print(correlations.head(10))

plt.tight_layout()
plt.show()

# Analyze categorical variables
if len(categorical_cols) > 0:
    print("\n\nCategorical Variables Analysis:")
    for col in categorical_cols[:5]:  # Analyze first 5 categorical columns
        print(f"\n{col}:")
        print(f"Unique values: {train_df[col].nunique()}")
        print(f"Most common values:")
        print(train_df[col].value_counts().head())
        
        # Plot mean target by category
        if train_df[col].nunique() < 20:
            plt.figure(figsize=(10, 6))
            train_df.groupby(col)[target_col].mean().sort_values().plot(kind='bar')
            plt.title(f'Mean {target_col} by {col}')
            plt.xlabel(col)
            plt.ylabel(f'Mean {target_col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

# 3. FEATURE ENGINEERING
print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

# Create copies for feature engineering
train_fe = train_df.copy()
test_fe = test_df.copy()

# Handle missing values (if any)
for col in numerical_cols:
    if train_fe[col].isnull().sum() > 0:
        median_value = train_fe[col].median()
        train_fe[col].fillna(median_value, inplace=True)
        test_fe[col].fillna(median_value, inplace=True)
        print(f"Filled missing values in {col} with median: {median_value}")

for col in categorical_cols:
    if train_fe[col].isnull().sum() > 0:
        mode_value = train_fe[col].mode()[0]
        train_fe[col].fillna(mode_value, inplace=True)
        test_fe[col].fillna(mode_value, inplace=True)
        print(f"Filled missing values in {col} with mode: {mode_value}")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_fe[col + '_encoded'] = le.fit_transform(train_fe[col].astype(str))
    test_fe[col + '_encoded'] = le.transform(test_fe[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded {col}: {train_fe[col].nunique()} unique values")

# Create some basic statistical features
if len(numerical_cols) > 1:
    train_fe['numerical_mean'] = train_fe[numerical_cols].mean(axis=1)
    train_fe['numerical_std'] = train_fe[numerical_cols].std(axis=1)
    train_fe['numerical_max'] = train_fe[numerical_cols].max(axis=1)
    train_fe['numerical_min'] = train_fe[numerical_cols].min(axis=1)
    
    test_fe['numerical_mean'] = test_fe[numerical_cols].mean(axis=1)
    test_fe['numerical_std'] = test_fe[numerical_cols].std(axis=1)
    test_fe['numerical_max'] = test_fe[numerical_cols].max(axis=1)
    test_fe['numerical_min'] = test_fe[numerical_cols].min(axis=1)
    
    print("\nAdded statistical features from numerical columns")

# 4. PREPARE DATA FOR MODELING
print("\n" + "="*50)
print("PREPARING DATA FOR MODELING")
print("="*50)

# Select features for modeling
feature_cols = [col for col in train_fe.columns if col not in ['id', target_col] + categorical_cols]
print(f"\nUsing {len(feature_cols)} features for modeling")

X = train_fe[feature_cols]
y = train_fe[target_col]
X_test = test_fe[feature_cols]

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 5. MODEL TRAINING
print("\n" + "="*50)
print("MODEL TRAINING")
print("="*50)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1),
    'CatBoost': CatBoostRegressor(iterations=100, random_state=42, verbose=0)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use scaled data for linear models
    if 'Linear' in name or 'Ridge' in name or 'Lasso' in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    results[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'model': model
    }
    
    print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# Find best model
best_model_name = min(results, key=lambda x: results[x]['RMSE'])
best_model = results[best_model_name]['model']
print(f"\n\nBest Model: {best_model_name} with RMSE: {results[best_model_name]['RMSE']:.4f}")

# 6. CROSS-VALIDATION FOR BEST MODEL
print("\n" + "="*50)
print("CROSS-VALIDATION")
print("="*50)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

if 'Linear' in best_model_name or 'Ridge' in best_model_name or 'Lasso' in best_model_name:
    cv_scores = cross_val_score(best_model, scaler.transform(X), y, 
                                cv=kfold, scoring='neg_mean_squared_error')
else:
    cv_scores = cross_val_score(best_model, X, y, 
                                cv=kfold, scoring='neg_mean_squared_error')

cv_rmse = np.sqrt(-cv_scores)
print(f"Cross-validation RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")

# 7. RETRAIN BEST MODEL ON FULL DATA
print("\n" + "="*50)
print("RETRAINING ON FULL DATA")
print("="*50)

if 'Linear' in best_model_name or 'Ridge' in best_model_name or 'Lasso' in best_model_name:
    best_model.fit(scaler.transform(X), y)
    predictions = best_model.predict(X_test_scaled)
else:
    best_model.fit(X, y)
    predictions = best_model.predict(X_test)

print(f"Predictions generated using {best_model_name}")

# 8. CREATE SUBMISSION FILE
print("\n" + "="*50)
print("CREATING SUBMISSION FILE")
print("="*50)

submission = pd.DataFrame({
    'id': test_df['id'],
    'Listening_Time_minutes': predictions
})

# Check for any negative predictions and clip them
if (predictions < 0).any():
    print(f"Found {(predictions < 0).sum()} negative predictions. Clipping to 0.")
    submission['Listening_Time_minutes'] = submission['Listening_Time_minutes'].clip(lower=0)

print("\nSubmission file statistics:")
print(submission['Listening_Time_minutes'].describe())

# Save submission
submission.to_csv('submission.csv', index=False)
print("\nSubmission file saved as 'submission.csv'")
print(f"Shape: {submission.shape}")
print("\nFirst 5 rows of submission:")
print(submission.head())

# 9. FEATURE IMPORTANCE (if applicable)
if hasattr(best_model, 'feature_importances_'):
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE")
    print("="*50)
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    plt.figure(figsize=(10, 8))
    feature_importance.head(20).plot(x='feature', y='importance', kind='barh')
    plt.title(f'Top 20 Feature Importances - {best_model_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

print("\n" + "="*50)
print("NOTEBOOK COMPLETE!")
print("="*50)

