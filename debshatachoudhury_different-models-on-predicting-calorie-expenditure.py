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


# ============================================
# Stage 1: Setup & Imports
# ============================================

# Core
import os
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling & Utilities
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor
from scipy.optimize import minimize

# Display and style
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid", palette="muted", font_scale=1.1)

# Verify input files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# ============================================
# Stage 2: Load Data
# ============================================

TRAIN_PATH = '/kaggle/input/playground-series-s5e5/train.csv'
TEST_PATH  = '/kaggle/input/playground-series-s5e5/test.csv'

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print("Train shape:", train.shape)
print("Test shape :", test.shape)
train.head()



# ============================================
# Stage 3: EDA — Structure, Summary, Missingness
# ============================================

# Basic structure
display(train.head())
display(train.describe(include='all').T)

# Dtypes and nulls
print("\nDtypes:")
print(train.dtypes)

print("\nMissing values per column:")
print(train.isna().sum())



# ============================================
# Stage 4: EDA — Distributions and Outliers
# ============================================

numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']

# Histograms
train[numeric_cols].hist(bins=30, figsize=(14, 8))
plt.suptitle("Numeric Distributions", y=1.02)
plt.show()

# Boxplots to inspect outliers
plt.figure(figsize=(14, 6))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 4, i)
    sns.boxplot(x=train[col], orient='h')
    plt.title(col)
plt.tight_layout()
plt.show()



# ============================================
# Stage 5: EDA — Correlations and Relationships
# ============================================

# Correlation heatmap for numeric features
plt.figure(figsize=(10, 8))
corr = train[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", square=True)
plt.title("Correlation Heatmap")
plt.show()

# Pairwise relationships on a sample to keep plots light
sample = train.sample(n=min(4000, len(train)), random_state=42)
pair_cols = ['Calories', 'Duration', 'Heart_Rate', 'Weight', 'Age']
sns.pairplot(sample[pair_cols], kind='scatter', plot_kws={'alpha': 0.5, 's': 12})
plt.suptitle("Pairwise Relationships (Sample)", y=1.02)
plt.show()

# Calories vs key drivers
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.scatterplot(data=sample, x='Duration', y='Calories', ax=axes[0], alpha=0.5)
axes[0].set_title("Calories vs Duration")

sns.scatterplot(data=sample, x='Heart_Rate', y='Calories', ax=axes[1], alpha=0.5)
axes[1].set_title("Calories vs Heart_Rate")

sns.violinplot(data=train, x='Sex', y='Calories', ax=axes[2])
axes[2].set_title("Calories by Sex")

plt.tight_layout()
plt.show()



# ============================================
# Stage 6: Feature Engineering
# ============================================

def add_features(df):
    df = df.copy()
    # Basics
    df['BMI'] = df['Weight'] / (df['Height'] / 100.0) ** 2

    # Mifflin-St Jeor BMR (male vs female)
    bmr_male   = (10 * df['Weight']) + (6.25 * df['Height']) - (5 * df['Age']) + 5
    bmr_female = (10 * df['Weight']) + (6.25 * df['Height']) - (5 * df['Age']) - 161
    df['BMR'] = np.where(df['Sex'] == 'Male', bmr_male, bmr_female)

    # Interactions
    df['Age_Weight']   = df['Age'] * df['Weight']
    df['Duration_HR']  = df['Duration'] * df['Heart_Rate']
    df['HR_Ratio']     = df['Heart_Rate'] / np.maximum(220 - df['Age'], 1)
    df['Temp_Diff']    = df['Body_Temp'] - 37
    df['Intensity']    = df['HR_Ratio'] * (df['Temp_Diff'] ** 2)

    # Duration indicators
    df['Very_Short_Duration'] = (df['Duration'] == 1).astype(int)
    df['Short_Duration']      = (df['Duration'] <= 4).astype(int)
    df['Long_Duration']       = (df['Duration'] >= 25).astype(int)

    # Polynomial
    df['Weight_Squared'] = df['Weight'] ** 2
    df['Age_Squared']    = df['Age'] ** 2

    return df

X = add_features(train.drop(['id', 'Calories'], axis=1))
y = train['Calories'].copy()
X_test = add_features(test.drop(['id'], axis=1))

# Encode 'Sex'
le = LabelEncoder()
X['Sex'] = le.fit_transform(X['Sex'])
X_test['Sex'] = le.transform(X_test['Sex'])

print("Engineered train shape:", X.shape)
display(X.head())



# ============================================
# Stage 7: Utilities — RMSLE and Weight Optimizer
# ============================================

def rmsle(y_true, y_pred):
    """
    Root Mean Squared Log Error with floor at 1 to avoid log issues.
    """
    y_pred = np.clip(y_pred, 1, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def optimize_weights(pred_matrix, y_true, init=None):
    """
    Given an OOF prediction matrix (n_samples x n_models) and y_true,
    find non-negative weights that sum to 1 minimizing RMSLE.
    """
    n_models = pred_matrix.shape[1]
    if init is None:
        init = np.ones(n_models) / n_models

    bounds = [(0, 1) for _ in range(n_models)]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    def objective(w):
        w = w / np.sum(w)
        blend = pred_matrix @ w
        return rmsle(y_true, blend)

    res = minimize(objective, init, method='SLSQP', bounds=bounds, constraints=constraints)
    w_opt = res.x / np.sum(res.x)
    return w_opt



import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from scipy.optimize import minimize

# RMSLE metric
def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 1, None)  # Ensure minimum 1 calorie
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Enhanced feature engineering
def add_features(df):
    # Basic features
    df['BMI'] = df['Weight'] / (df['Height']/100)**2
    df['BMR'] = (10 * df['Weight']) + (6.25 * df['Height']) - (5 * df['Age']) + 5
    df['BMR'] = df['BMR'].where(df['Sex'] == 'Male', 
                               (10 * df['Weight']) + (6.25 * df['Height']) - (5 * df['Age']) - 161)
    
    # Interaction features
    df['Age_Weight'] = df['Age'] * df['Weight']
    df['Duration_HR'] = df['Duration'] * df['Heart_Rate']
    df['HR_Ratio'] = df['Heart_Rate'] / (220 - df['Age'])
    df['Temp_Diff'] = df['Body_Temp'] - 37
    
    # Duration indicators
    df['Short_Duration'] = (df['Duration'] <= 4).astype(int)
    df['Very_Short_Duration'] = (df['Duration'] == 1).astype(int)
    
    # Age bins by gender
    df['Age_bin'] = pd.cut(df['Age'], bins=[0, 20, 30, 40, 50, 60, 100], labels=False)
    
    return df

X = add_features(train.drop(['id', 'Calories'], axis=1))
y = train['Calories']
X_test = add_features(test.drop('id', axis=1))

# Encode categorical features
le = LabelEncoder()
X['Sex'] = le.fit_transform(X['Sex'])
X_test['Sex'] = le.transform(X_test['Sex'])

# Initialize models (CPU-only version)
models = {
    'lgb': LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ),
    'xgb': XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.2,
        reg_lambda=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'  # Changed from gpu_hist to hist
    ),
    'ridge': Ridge(
        alpha=10.0,
        random_state=42
    )
}

# 5-Fold CV
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
oof_preds = {name: np.zeros(len(X)) for name in models}
test_preds = {name: np.zeros(len(X_test)) for name in models}
fold_scores = []

print("Running Optimized Ensemble CV...\n")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Scale features for Ridge
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train models
    print(f"Fold {fold}: Training LGBM...")
    models['lgb'].fit(X_train, y_train)
    
    print(f"Fold {fold}: Training XGBoost...")
    models['xgb'].fit(X_train, y_train)
    
    print(f"Fold {fold}: Training Ridge...")
    models['ridge'].fit(X_train_scaled, y_train)
    
    # Get predictions
    oof_preds['lgb'][val_idx] = models['lgb'].predict(X_val)
    oof_preds['xgb'][val_idx] = models['xgb'].predict(X_val)
    oof_preds['ridge'][val_idx] = models['ridge'].predict(X_val_scaled)
    
    # Test predictions
    X_test_scaled = scaler.transform(X_test)
    test_preds['lgb'] += models['lgb'].predict(X_test) / n_folds
    test_preds['xgb'] += models['xgb'].predict(X_test) / n_folds
    test_preds['ridge'] += models['ridge'].predict(X_test_scaled) / n_folds
    
    # Calculate fold RMSLE for each model
    fold_scores.append({
        'lgb': rmsle(y_val, oof_preds['lgb'][val_idx]),
        'xgb': rmsle(y_val, oof_preds['xgb'][val_idx]),
        'ridge': rmsle(y_val, oof_preds['ridge'][val_idx])
    })
    print(f"Fold {fold} RMSLE - LGB: {fold_scores[-1]['lgb']:.5f}, XGB: {fold_scores[-1]['xgb']:.5f}, Ridge: {fold_scores[-1]['ridge']:.5f}")

# Optimize ensemble weights
pred_matrix = np.column_stack([oof_preds[name] for name in models])

def objective(weights):
    weights = weights / np.sum(weights)
    blend = pred_matrix @ weights
    return rmsle(y, blend)

initial_weights = np.array([0.4, 0.4, 0.2])  # Start with reasonable defaults
bounds = [(0, 1) for _ in range(len(models))]
result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds,
                 constraints=({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}))

optimal_weights = result.x / np.sum(result.x)
print("\nOptimal Weights:", {name: f"{w:.3f}" for name, w in zip(models.keys(), optimal_weights)})

# Final predictions
final_test_preds = sum(test_preds[name] * w for name, w in zip(models.keys(), optimal_weights))
final_test_preds = np.clip(final_test_preds, 1, None)

# Calculate final scores
final_oof_preds = sum(oof_preds[name] * w for name, w in zip(models.keys(), optimal_weights))
final_rmsle = rmsle(y, final_oof_preds)

print("\nModel Performance:")
for name in models:
    mean_score = np.mean([fold[name] for fold in fold_scores])
    print(f"{name.upper():<6} | Mean RMSLE: {mean_score:.5f}")

print(f"\nEnsemble OOF RMSLE: {final_rmsle:.5f}")

# Save submission
submission = pd.DataFrame({'id': test['id'], 'Calories': final_test_preds})
submission.to_csv('submission.csv', index=False)
print("\nSubmission saved with optimized ensemble!")



import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.optimize import minimize

# RMSLE metric with clipping
def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 1, None)  # Ensure minimum 1 calorie
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Enhanced feature engineering
def add_features(df):
    # Basic physiological features
    df['BMI'] = df['Weight'] / (df['Height']/100)**2
    df['BMR'] = (10 * df['Weight']) + (6.25 * df['Height']) - (5 * df['Age']) + 5
    df['BMR'] = df['BMR'].where(df['Sex'] == 'Male', 
                               (10 * df['Weight']) + (6.25 * df['Height']) - (5 * df['Age']) - 161)
    
    # Interaction features
    df['Age_Weight'] = df['Age'] * df['Weight']
    df['Duration_HR'] = df['Duration'] * df['Heart_Rate']
    df['HR_Ratio'] = df['Heart_Rate'] / (220 - df['Age'])
    df['Temp_Diff'] = df['Body_Temp'] - 37
    df['Intensity'] = df['HR_Ratio'] * (df['Temp_Diff']**2)
    
    # Duration indicators
    df['Short_Duration'] = (df['Duration'] <= 4).astype(int)
    df['Very_Short_Duration'] = (df['Duration'] == 1).astype(int)
    df['Long_Duration'] = (df['Duration'] >= 25).astype(int)
    
    # Polynomial features
    df['Weight_Squared'] = df['Weight']**2
    df['Age_Squared'] = df['Age']**2
    
    return df

X = add_features(train.drop(['id', 'Calories'], axis=1))
y = train['Calories']
X_test = add_features(test.drop('id', axis=1))

# Encode categorical features
le = LabelEncoder()
X['Sex'] = le.fit_transform(X['Sex'])
X_test['Sex'] = le.transform(X_test['Sex'])

# Initialize models
models = {
    'lgb': LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ),
    'lasso': Lasso(
        alpha=0.0005,
        max_iter=5000,
        random_state=42,
        selection='random'
    )
}

# 5-Fold CV
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
oof_preds = {name: np.zeros(len(X)) for name in models}
test_preds = {name: np.zeros(len(X_test)) for name in models}
fold_scores = []

print("Running Lasso + LightGBM Ensemble CV...\n")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Scale features for Lasso
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train models with sample weights for duration bins
    weights = np.ones(len(X_train))
    weights[X_train['Duration'] <= 4] *= 3.5
    weights[X_train['Duration'] == 1] *= 5
    
    print(f"Fold {fold}: Training LightGBM...")
    models['lgb'].fit(X_train, y_train, sample_weight=weights)
    
    print(f"Fold {fold}: Training Lasso...")
    models['lasso'].fit(X_train_scaled, y_train, sample_weight=weights)
    
    # Get predictions
    oof_preds['lgb'][val_idx] = models['lgb'].predict(X_val)
    oof_preds['lasso'][val_idx] = models['lasso'].predict(X_val_scaled)
    
    # Test predictions
    X_test_scaled = scaler.transform(X_test)
    test_preds['lgb'] += models['lgb'].predict(X_test) / n_folds
    test_preds['lasso'] += models['lasso'].predict(X_test_scaled) / n_folds
    
    # Calculate fold RMSLE
    fold_scores.append({
        'lgb': rmsle(y_val, oof_preds['lgb'][val_idx]),
        'lasso': rmsle(y_val, oof_preds['lasso'][val_idx])
    })
    print(f"Fold {fold} RMSLE - LGB: {fold_scores[-1]['lgb']:.5f}, Lasso: {fold_scores[-1]['lasso']:.5f}")

# Optimize ensemble weights
pred_matrix = np.column_stack([oof_preds[name] for name in models])

def objective(weights):
    weights = weights / np.sum(weights)
    blend = pred_matrix @ weights
    return rmsle(y, blend)

initial_weights = np.array([0.7, 0.3])  # Favor LightGBM initially
bounds = [(0, 1) for _ in range(len(models))]
result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds,
                 constraints=({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}))

optimal_weights = result.x / np.sum(result.x)
print("\nOptimal Weights:", {name: f"{w:.3f}" for name, w in zip(models.keys(), optimal_weights)})

# Final predictions
final_test_preds = sum(test_preds[name] * w for name, w in zip(models.keys(), optimal_weights))
final_test_preds = np.clip(final_test_preds, 1, None)

# Calculate final scores
final_oof_preds = sum(oof_preds[name] * w for name, w in zip(models.keys(), optimal_weights))
final_rmsle = rmsle(y, final_oof_preds)

print("\nModel Performance:")
for name in models:
    mean_score = np.mean([fold[name] for fold in fold_scores])
    print(f"{name.upper():<6} | Mean RMSLE: {mean_score:.5f}")

print(f"\nEnsemble OOF RMSLE: {final_rmsle:.5f}")

# Save submission
submission = pd.DataFrame({'id': test['id'], 'Calories': final_test_preds})
submission.to_csv('submission.csv', index=False)
print("\nSubmission saved with optimized ensemble!")


