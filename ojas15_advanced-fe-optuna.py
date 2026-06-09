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


# --- 1. LIBRARIES ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import lightgbm as lgb
import optuna
import warnings
warnings.filterwarnings('ignore')

# --- 2. LOAD DATA (CORRECT PATHS) ---
print("Loading data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

# Store test_ids for submission
test_ids = test['id']

# Drop id from train and define target
train = train.drop('id', axis=1)
target = 'accident_risk'

print("Data loaded successfully.")


# --- 3. FEATURE ENGINEERING (ON COMBINED DATA) ---
print("Starting Feature Engineering...")
combined_df = pd.concat([train.drop(target, axis=1), test], ignore_index=True)
original_features = combined_df.columns.tolist()

# --- 3.1 Convert Booleans to Integers ---
bool_cols = ['road_signs_present', 'public_road']
for col in bool_cols:
    combined_df[col] = combined_df[col].astype(int)

# --- 3.2 Label Encode Categoricals ---
# We will encode them first to make interactions easier
categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
for col in categorical_cols:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col])

# --- 3.3 Create Interaction Features (THE GOLD) ---
# This is what Experts look for.
print("Creating interaction features...")
combined_df['weather_x_lighting'] = combined_df['weather'].astype(str) + '_' + combined_df['lighting'].astype(str)
combined_df['road_x_time'] = combined_df['road_type'].astype(str) + '_' + combined_df['time_of_day'].astype(str)
combined_df['lanes_x_speed'] = combined_df['num_lanes'].astype(str) + '_' + combined_df['speed_limit'].astype(str)
combined_df['road_x_signs'] = combined_df['road_type'].astype(str) + '_' + combined_df['road_signs_present'].astype(str)

# --- 3.4 Create Numerical Features ---
print("Creating numerical features...")
# Handle potential division by zero
combined_df['curvature_per_lane'] = combined_df['curvature'] / (combined_df['num_lanes'] + 1)
combined_df['curvature_x_speed'] = combined_df['curvature'] * combined_df['speed_limit']

# --- 3.5 Final Encoding of New Interaction Features ---
new_cat_cols = ['weather_x_lighting', 'road_x_time', 'lanes_x_speed', 'road_x_signs']
for col in new_cat_cols:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col])
    
# --- 4. SPLIT DATA AGAIN ---
X = combined_df.iloc[:len(train)]
X_test = combined_df.iloc[len(train):]
y = train[target]

# Define categorical features for LGBM
categorical_features = original_features + new_cat_cols
categorical_features = [col for col in categorical_features if col in X.columns and (X[col].dtype == 'object' or X[col].dtype == 'int64')]

print(f"Feature Engineering complete. New shape: {X.shape}")


# --- 5. OPTUNA HYPERPARAMETER TUNING ---
# We use a single split for fast tuning
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        'objective': 'regression_l1', # MAE is robust to outliers
        'metric': 'rmse',
        'n_estimators': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', -1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
    }
    
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)],
              categorical_feature=categorical_features)
    
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

print("\nStarting Optuna tuning...")
study = optuna.create_study(direction='minimize')
# Warm start with good default params
study.enqueue_trial({'learning_rate': 0.03, 'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'max_depth': -1})
study.optimize(objective, n_trials=50) 

print(f"Best trial RMSE: {study.best_value}")
print("Best params: ")
print(study.best_params)

# Get best params for full training
best_lgb_params = study.best_params
best_lgb_params['objective'] = 'regression_l1'
best_lgb_params['metric'] = 'rmse'
best_lgb_params['n_estimators'] = 7500 # Use more estimators for full train
best_lgb_params['n_jobs'] = -1
best_lgb_params['seed'] = 42


# --- 6. FULL MODEL TRAINING WITH KFOLD CV ---
N_SPLITS = 10
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
models = []

print("\n--- Starting Full Model Training (10 Folds) ---")
for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model = LGBMRegressor(**best_lgb_params)
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(200, verbose=False)],
              categorical_feature=categorical_features)
    
    val_preds = model.predict(X_val)
    oof_preds[val_index] = val_preds
    
    test_preds += model.predict(X_test) / N_SPLITS
    models.append(model)
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Fold {fold+1} RMSE: {fold_rmse}")

overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"\n--- Overall OOF RMSE: {overall_rmse} ---")

# --- 7. FEATURE IMPORTANCE ---
feature_importances = pd.DataFrame()
feature_importances['feature'] = X.columns
feature_importances['importance'] = np.mean([model.feature_importances_ for model in models], axis=0)
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 12))
sns.barplot(x='importance', y='feature', data=feature_importances.head(30))
plt.title('Top 30 Feature Importances (LGBM)')
plt.show()

# --- 8. CREATE SUBMISSION ---
submission_df = pd.DataFrame({'id': test_ids, 'accident_risk': test_preds})
# CRITICAL STEP: Clip predictions to be between 0 and 1
submission_df['accident_risk'] = submission_df['accident_risk'].clip(0, 1)
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully!")




