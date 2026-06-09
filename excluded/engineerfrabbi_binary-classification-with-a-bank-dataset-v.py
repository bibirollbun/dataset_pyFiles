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


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
import warnings
import os

# Set configurations
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("Libraries imported successfully!")


# Load the datasets
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
    sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
except:
    # If running locally, adjust the path
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    sample_submission = pd.read_csv('sample_submission.csv')

# Display the shape of the datasets
print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")

# Display the first few rows of the training data
train_df.head()


# Check data types and missing values
print("Data types and missing values in training data:")
print(train_df.info())

# Summary statistics
print("\nSummary statistics of numerical features:")
print(train_df.describe())

# Check for missing values
print("\nMissing values in training data:")
print(train_df.isnull().sum())

# Check for missing values in test data
print("\nMissing values in test data:")
print(test_df.isnull().sum())


# Check target variable distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='y', data=train_df)
plt.title('Distribution of Target Variable')
plt.xlabel('Subscription to Term Deposit')
plt.ylabel('Count')
plt.show()

# Calculate the percentage of each class
target_counts = train_df['y'].value_counts()
target_percentages = target_counts / len(train_df) * 100
print(f"Class distribution:\n{target_percentages}")


# Identify numerical and categorical features
numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()

# Remove target and id from numerical features
if 'y' in numerical_features:
    numerical_features.remove('y')
if 'id' in numerical_features:
    numerical_features.remove('id')

print(f"Numerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")


# Distribution of numerical features
plt.figure(figsize=(20, 15))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(4, 4, i)
    sns.histplot(train_df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()


# Box plots for numerical features vs target
plt.figure(figsize=(20, 15))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(4, 4, i)
    sns.boxplot(x='y', y=feature, data=train_df)
    plt.title(f'{feature} vs Target')
plt.tight_layout()
plt.show()


# Distribution of categorical features
plt.figure(figsize=(20, 15))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(3, 3, i)
    sns.countplot(y=feature, data=train_df)
    plt.title(f'Distribution of {feature}')
    plt.tight_layout()
plt.show()


# Categorical features vs target
plt.figure(figsize=(20, 15))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(3, 3, i)
    sns.countplot(y=feature, hue='y', data=train_df)
    plt.title(f'{feature} vs Target')
    plt.tight_layout()
plt.show()


# Correlation matrix for numerical features
plt.figure(figsize=(16, 12))
corr_matrix = train_df[numerical_features + ['y']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# Check for outliers in numerical features
plt.figure(figsize=(20, 15))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(4, 4, i)
    sns.boxplot(x=train_df[feature])
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()


# Create a copy of the original dataframes for feature engineering
train_fe = train_df.copy()
test_fe = test_df.copy()

# Feature engineering functions
def create_new_features(df):
    # Create interaction features
    if 'age' in df.columns and 'balance' in df.columns:
        df['age_balance_ratio'] = df['age'] / (df['balance'] + 1)  # Adding 1 to avoid division by zero
    
    if 'duration' in df.columns and 'campaign' in df.columns:
        df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
    
    # Create binning features for age
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], labels=['<30', '30-40', '40-50', '50-60', '>60'])
    
    # Create binning features for balance
    if 'balance' in df.columns:
        df['balance_group'] = pd.qcut(df['balance'].rank(method='first'), q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    # Create total contacts feature
    if 'campaign' in df.columns and 'previous' in df.columns:
        df['total_contacts'] = df['campaign'] + df['previous']
    
    # Create a feature for whether the customer was contacted before
    if 'previous' in df.columns:
        df['was_contacted_before'] = (df['previous'] > 0).astype(int)
    
    # Create a feature for the day of the week
    if 'day' in df.columns and 'month' in df.columns:
        # Map month to numerical value
        month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                     'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        df['month_num'] = df['month'].map(month_map)
        
        # Create a feature for the day of the year
        df['day_of_year'] = df['day'] + (df['month_num'] - 1) * 30  # Approximation
    
    return df

# Apply feature engineering
train_fe = create_new_features(train_fe)
test_fe = create_new_features(test_fe)

# Display the new features
print("New features created:")
new_features = set(train_fe.columns) - set(train_df.columns)
print(new_features)


# Update numerical and categorical features lists
numerical_features = train_fe.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = train_fe.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove target and id from numerical features
if 'y' in numerical_features:
    numerical_features.remove('y')
if 'id' in numerical_features:
    numerical_features.remove('id')

print(f"Updated numerical features: {numerical_features}")
print(f"Updated categorical features: {categorical_features}")


# Handle missing values
def handle_missing_values(df):
    # For numerical columns, fill with median
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # For categorical columns, fill with mode
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

# Apply missing value handling
train_processed = handle_missing_values(train_fe.copy())
test_processed = handle_missing_values(test_fe.copy())

# Check if there are any missing values left
print("Missing values in training data after preprocessing:")
print(train_processed.isnull().sum().sum())
print("\nMissing values in test data after preprocessing:")
print(test_processed.isnull().sum().sum())


# Encode categorical variables
# Create a label encoder for each categorical column
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    train_processed[col] = le.fit_transform(train_processed[col])
    test_processed[col] = le.transform(test_processed[col])
    label_encoders[col] = le

# Display the first few rows of the processed training data
train_processed.head()


# 1. Replace inf/-inf with NaN
train_processed[numerical_features] = train_processed[numerical_features].replace([np.inf, -np.inf], np.nan)
test_processed[numerical_features] = test_processed[numerical_features].replace([np.inf, -np.inf], np.nan)

# 2. Fill NaN with median (fit from train, apply on both)
medians = train_processed[numerical_features].median()
train_processed[numerical_features] = train_processed[numerical_features].fillna(medians)
test_processed[numerical_features] = test_processed[numerical_features].fillna(medians)

# 3. Scale numerical features
scaler = StandardScaler()
train_processed[numerical_features] = scaler.fit_transform(train_processed[numerical_features])
test_processed[numerical_features] = scaler.transform(test_processed[numerical_features])

# 4. Display first few rows
train_processed.head()


# Prepare data for modeling
X = train_processed.drop(['id', 'y'], axis=1)
y = train_processed['y']
X_test = test_processed.drop(['id'], axis=1)

# Split the data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")


# Define a function for cross-validation
def cross_validate_model(model, X, y, cv=5, random_state=RANDOM_SEED):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    roc_auc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        roc_auc = roc_auc_score(y_val_fold, y_pred_proba)
        roc_auc_scores.append(roc_auc)
        
        print(f"Fold {fold+1} ROC AUC: {roc_auc:.5f}")
    
    mean_roc_auc = np.mean(roc_auc_scores)
    std_roc_auc = np.std(roc_auc_scores)
    print(f"\nMean ROC AUC: {mean_roc_auc:.5f} ± {std_roc_auc:.5f}")
    
    return mean_roc_auc, std_roc_auc


# Model 1: LightGBM
print("LightGBM Model:")
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': RANDOM_SEED
}

lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_mean_score, lgb_std_score = cross_validate_model(lgb_model, X, y)

# Train on full training data and evaluate on validation set
lgb_model.fit(X_train, y_train)
lgb_val_pred = lgb_model.predict_proba(X_val)[:, 1]
lgb_val_score = roc_auc_score(y_val, lgb_val_pred)
print(f"Validation ROC AUC: {lgb_val_score:.5f}")


# Model 2: XGBoost
print("\nXGBoost Model:")
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'random_state': RANDOM_SEED,
    'use_label_encoder': False
}

xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_mean_score, xgb_std_score = cross_validate_model(xgb_model, X, y)

# Train on full training data and evaluate on validation set
xgb_model.fit(X_train, y_train)
xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]
xgb_val_score = roc_auc_score(y_val, xgb_val_pred)
print(f"Validation ROC AUC: {xgb_val_score:.5f}")


# Model 3: CatBoost
print("\nCatBoost Model:")
cb_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': RANDOM_SEED,
    'verbose': False
}

cb_model = cb.CatBoostClassifier(**cb_params)
cb_mean_score, cb_std_score = cross_validate_model(cb_model, X, y)

# Train on full training data and evaluate on validation set
cb_model.fit(X_train, y_train)
cb_val_pred = cb_model.predict_proba(X_val)[:, 1]
cb_val_score = roc_auc_score(y_val, cb_val_pred)
print(f"Validation ROC AUC: {cb_val_score:.5f}")


# Model 4: Logistic Regression
print("\nLogistic Regression Model:")
lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
lr_mean_score, lr_std_score = cross_validate_model(lr_model, X, y)

# Train on full training data and evaluate on validation set
lr_model.fit(X_train, y_train)
lr_val_pred = lr_model.predict_proba(X_val)[:, 1]
lr_val_score = roc_auc_score(y_val, lr_val_pred)
print(f"Validation ROC AUC: {lr_val_score:.5f}")


# Define the objective function for Optuna
def objective(trial, model_type='lgb'):
    if model_type == 'lgb':
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
            'verbose': -1,
            'random_state': RANDOM_SEED
        }
        model = lgb.LGBMClassifier(**params)
    
    elif model_type == 'xgb':
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'lambda': trial.suggest_float('lambda', 0, 10),
            'alpha': trial.suggest_float('alpha', 0, 10),
            'random_state': RANDOM_SEED,
            'use_label_encoder': False
        }
        model = xgb.XGBClassifier(**params)
    
    elif model_type == 'cb':
        params = {
            'iterations': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': RANDOM_SEED,
            'verbose': False
        }
        model = cb.CatBoostClassifier(**params)
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    roc_auc_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        roc_auc = roc_auc_score(y_val_fold, y_pred_proba)
        roc_auc_scores.append(roc_auc)
    
    return np.mean(roc_auc_scores)


# Tune LightGBM hyperparameters
print("Tuning LightGBM hyperparameters...")
lgb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
lgb_study.optimize(lambda trial: objective(trial, 'lgb'), n_trials=30, show_progress_bar=True)

print(f"Best LightGBM ROC AUC: {lgb_study.best_value:.5f}")
print("Best LightGBM parameters:")
for key, value in lgb_study.best_params.items():
    print(f"  {key}: {value}")


# Tune XGBoost hyperparameters
print("\nTuning XGBoost hyperparameters...")
xgb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
xgb_study.optimize(lambda trial: objective(trial, 'xgb'), n_trials=20, show_progress_bar=True)

print(f"Best XGBoost ROC AUC: {xgb_study.best_value:.5f}")
print("Best XGBoost parameters:")
for key, value in xgb_study.best_params.items():
    print(f"  {key}: {value}")


# Tune CatBoost hyperparameters
print("\nTuning CatBoost hyperparameters...")
cb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
cb_study.optimize(lambda trial: objective(trial, 'cb'), n_trials=5, show_progress_bar=True)

print(f"Best CatBoost ROC AUC: {cb_study.best_value:.5f}")
print("Best CatBoost parameters:")
for key, value in cb_study.best_params.items():
    print(f"  {key}: {value}")


# Train the best models with optimized parameters
# LightGBM
best_lgb_params = lgb_study.best_params
best_lgb_params.update({
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'random_state': RANDOM_SEED
})
best_lgb_model = lgb.LGBMClassifier(**best_lgb_params)
best_lgb_model.fit(X_train, y_train)
best_lgb_val_pred = best_lgb_model.predict_proba(X_val)[:, 1]
best_lgb_val_score = roc_auc_score(y_val, best_lgb_val_pred)
print(f"Best LightGBM Validation ROC AUC: {best_lgb_val_score:.5f}")

# XGBoost
best_xgb_params = xgb_study.best_params
best_xgb_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': RANDOM_SEED,
    'use_label_encoder': False
})
best_xgb_model = xgb.XGBClassifier(**best_xgb_params)
best_xgb_model.fit(X_train, y_train)
best_xgb_val_pred = best_xgb_model.predict_proba(X_val)[:, 1]
best_xgb_val_score = roc_auc_score(y_val, best_xgb_val_pred)
print(f"Best XGBoost Validation ROC AUC: {best_xgb_val_score:.5f}")

# CatBoost
best_cb_params = cb_study.best_params
best_cb_params.update({
    'iterations': 1000,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': RANDOM_SEED,
    'verbose': False
})
best_cb_model = cb.CatBoostClassifier(**best_cb_params)
best_cb_model.fit(X_train, y_train)
best_cb_val_pred = best_cb_model.predict_proba(X_val)[:, 1]
best_cb_val_score = roc_auc_score(y_val, best_cb_val_pred)
print(f"Best CatBoost Validation ROC AUC: {best_cb_val_score:.5f}")


# Plot ROC curves
plt.figure(figsize=(10, 8))
models = [
    ('LightGBM', best_lgb_val_pred),
    ('XGBoost', best_xgb_val_pred),
    ('CatBoost', best_cb_val_pred)
]

for name, y_pred in models:
    fpr, tpr, _ = roc_curve(y_val, y_pred)
    auc_score = roc_auc_score(y_val, y_pred)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.5f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()


# Feature importance for LightGBM
lgb_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=lgb_importance.head(15))
plt.title('LightGBM Feature Importance')
plt.tight_layout()
plt.show()


# Feature importance for XGBoost
xgb_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=xgb_importance.head(15))
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.show()


# Feature importance for CatBoost
cb_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_cb_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=cb_importance.head(15))
plt.title('CatBoost Feature Importance')
plt.tight_layout()
plt.show()


# Model comparison table
model_comparison = pd.DataFrame({
    'Model': ['LightGBM', 'XGBoost', 'CatBoost', 'Logistic Regression'],
    'CV Mean ROC AUC': [lgb_mean_score, xgb_mean_score, cb_mean_score, lr_mean_score],
    'CV Std ROC AUC': [lgb_std_score, xgb_std_score, cb_std_score, lr_std_score],
    'Validation ROC AUC': [lgb_val_score, xgb_val_score, cb_val_score, lr_val_score],
    'Tuned Validation ROC AUC': [best_lgb_val_score, best_xgb_val_score, best_cb_val_score, np.nan]
})

model_comparison = model_comparison.sort_values('Tuned Validation ROC AUC', ascending=False)
model_comparison


# Select the best model
best_model_name = model_comparison.iloc[0]['Model']
print(f"Best model: {best_model_name}")

if best_model_name == 'LightGBM':
    best_model = lgb.LGBMClassifier(**best_lgb_params)
elif best_model_name == 'XGBoost':
    best_model = xgb.XGBClassifier(**best_xgb_params)
elif best_model_name == 'CatBoost':
    best_model = cb.CatBoostClassifier(**best_cb_params)

# Train the best model on the full dataset
best_model.fit(X, y)


# Generate predictions on the test set
test_predictions = best_model.predict_proba(X_test)[:, 1]

# Create submission dataframe
submission = pd.DataFrame({
    'id': test_processed['id'],
    'y': test_predictions
})

# Display the first few rows of the submission
submission.head()


# Save the submission file
submission.to_csv('submission.csv', index=False)
print("Submission file saved successfully!")

# Check the distribution of predictions
plt.figure(figsize=(10, 6))
sns.histplot(submission['y'], kde=True)
plt.title('Distribution of Predictions')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.show()


# Additional: Ensemble approach
# Create an ensemble of the top models
print("Creating ensemble of top models...")

# Train the top models on the full dataset
best_lgb_model.fit(X, y)
best_xgb_model.fit(X, y)
best_cb_model.fit(X, y)

# Generate predictions from each model
lgb_test_pred = best_lgb_model.predict_proba(X_test)[:, 1]
xgb_test_pred = best_xgb_model.predict_proba(X_test)[:, 1]
cb_test_pred = best_cb_model.predict_proba(X_test)[:, 1]

# Create weighted ensemble (weights based on validation performance)
weights = [
    best_lgb_val_score,
    best_xgb_val_score,
    best_cb_val_score
]
weights = np.array(weights) / sum(weights)  # Normalize weights

ensemble_test_pred = (
    weights[0] * lgb_test_pred +
    weights[1] * xgb_test_pred +
    weights[2] * cb_test_pred
)

# Create ensemble submission
ensemble_submission = pd.DataFrame({
    'id': test_processed['id'],
    'y': ensemble_test_pred
})

# Save the ensemble submission
ensemble_submission.to_csv('ensemble_submission.csv', index=False)
print("Ensemble submission file saved successfully!")

# Display the first few rows of the ensemble submission
ensemble_submission.head()


# Compare individual model predictions with ensemble
plt.figure(figsize=(12, 8))
sns.kdeplot(test_predictions, label='Best Model')
sns.kdeplot(ensemble_test_pred, label='Ensemble')
plt.title('Comparison of Predictions: Best Model vs Ensemble')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.legend()
plt.show()


# Final summary
print("Final Summary:")
print(f"Best Model: {best_model_name}")
print(f"Best Validation ROC AUC: {model_comparison.iloc[0]['Tuned Validation ROC AUC']:.5f}")
print(f"Ensemble Weights - LightGBM: {weights[0]:.3f}, XGBoost: {weights[1]:.3f}, CatBoost: {weights[2]:.3f}")
print("\nSubmission files created:")
print("- submission.csv (Best Model)")
print("- ensemble_submission.csv (Ensemble)")
print("\nGood luck in the competition!")




