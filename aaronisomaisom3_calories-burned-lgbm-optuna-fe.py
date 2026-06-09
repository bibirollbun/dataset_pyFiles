# Install Optuna
!pip install optuna --quiet
!pip install optuna-integration[lightgbm] --quiet


# Author: Aaron Isom
# Kaggle Predict Calorie Expenditure
# LGBMRegressor and Optuna for hyperparameter tuning using RMSLE

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna
import warnings

from optuna.samplers import TPESampler
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, mean_squared_log_error

warnings.filterwarnings('ignore')

# RMSLE function
def rmsle(y_true, y_pred):
    # Ensure predictions are non-negative
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Create the RMSLE scorer
rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

# Load data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')

# Display first few rows
display(train_df.head(10))
display('Train Shape', train_df.shape)
display('Test Shape', test_df.shape)

display('Missing Train Values:', train_df.isnull().sum())
display('Missing Test Values:', test_df.isnull().sum())

# Describe the data
display(train_df.describe())

# Display information about dtypes
display('Train Data Info:', train_df.info())

# Preprocessing + Feature Engineering
for df in [train_df, test_df]:
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Intensity'] = df['Duration'] * df['Heart_Rate']
    df['Sex'] = df['Sex'].astype('category')
    df['log_Duration'] = np.log1p(df['Duration'])
    df['log_Intensity'] = np.log1p(df['Intensity'])
    df['log_BMI'] = np.log1p(df['BMI'])

#Clip the top 1% of Calories to reduce log-space distortion
cap = train_df['Calories'].quantile(0.99)
train_df['Calories'] = np.minimum(train_df['Calories'], cap)

X = train_df.drop('Calories', axis=1)
y = np.log1p(train_df['Calories'])  # Instead of raw Calories

# EDA
temp = X.copy()
temp['Sex'] = LabelEncoder().fit_transform(temp['Sex'])
temp['Calories'] = y
sns.heatmap(temp.corr(), annot=True, cmap='coolwarm')

cor_matrix = temp.corr()
low_corr_features = cor_matrix['Calories'].abs().sort_values()
print("Low correlation features:\n", low_corr_features[low_corr_features < 0.05])

# Distributions
features = X.columns.tolist()
for feature in features:
    plt.figure(figsize=(6, 4))
    sns.histplot(X[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()


# Optuna Tuning
def objective(trial):
  
    params = {
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 500, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 32, 512),
        'max_depth': trial.suggest_int('max_depth', 4, 16),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 80),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'max_bin': trial.suggest_int('max_bin', 64, 255),
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = LGBMRegressor(**params, early_stopping_round=100, random_state=42, n_jobs=-1, device='cpu', force_col_wise=True)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[LightGBMPruningCallback(trial, 'rmse')])
        y_pred = model.predict(X_valid)
        scores.append(rmsle(y_valid, y_pred))

    return np.mean(scores)  # Optuna will minimize this
  

# Uncomment to tune using Optuna
# Run Optuna trials for tuning
# study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
# study.optimize(objective, n_trials=100, show_progress_bar=True)
# print("Best Trial RMSLE:", study.best_value)
# print("Best Hyperparameters:", study.best_params)

# Train using Optuna to tune with best_params
# best_model = LGBMRegressor(**study.best_params, callbacks=[lgb.early_stopping(stopping_rounds=100)], random_state=42, n_jobs=-1, device='cpu', force_col_wise=True)

# Use final tuned best_params for submission
# Best Hyperparameters: {'n_estimators': 3442, 'learning_rate': 0.24728504318808653, 'num_leaves': 422, 'max_depth': 11, 'min_child_samples': 41, 'min_split_gain': 0.0011120339477239903, 'subsample': 0.7499527653010576, 'subsample_freq': 6, 'colsample_bytree': 0.8154215328620447, 'reg_alpha': 2.4881716445289936, 'reg_lambda': 6.3866492238889165, 'max_bin': 113}

best_params = {'n_estimators': 3442, 'learning_rate': 0.24728504318808653, 'num_leaves': 422, 'max_depth': 11, 'min_child_samples': 41, 'min_split_gain': 0.0011120339477239903, 
               'subsample': 0.7499527653010576, 'subsample_freq': 6, 'colsample_bytree': 0.8154215328620447, 'reg_alpha': 2.4881716445289936, 'reg_lambda': 6.3866492238889165, 'max_bin': 113}

# Define model
best_model = LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbose=-1, metric='rmse', objective='regression', force_col_wise=True)

# Fit model
best_model.fit(X, y)

# Cross-validated score (more realistic)
# cv_scores = cross_val_score(best_model, X, y, cv=5, scoring=rmsle)
# print(f"Cross-validated RMSLE score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Feature importances
importances = best_model.feature_importances_
sns.barplot(x=importances, y=features)
plt.title('Feature Importances')
plt.show()

# Predict on test set
final_preds = np.maximum(0, np.expm1(best_model.predict(test_df)))
final_preds = np.round(final_preds, 2)  # Helps with leaderboard noise sometimes

# Submission
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission['Calories'] = final_preds
submission.to_csv('submission.csv', index=False)
display(submission)
print('Submission file saved.')

