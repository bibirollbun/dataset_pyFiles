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

import warnings
warnings.filterwarnings('ignore')



# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import warnings
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from scipy.optimize import minimize

warnings.filterwarnings('ignore')



# Load the data
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


# Peek at the data
print("Train shape:", train.shape)
print("Test shape:", test.shape)


train.head()


train.columns


train.describe()


# Function to preprocess the data
def preprocess(train_df, test_df):
    # Create copies to avoid modifying originals
    train = train_df.copy()
    test = test_df.copy()
    
    # Drop id and non-numeric columns
    for df in [train, test]:
        if 'id' in df.columns:
            df.drop(columns=['id'], inplace=True)
        if 'Podcast_Name' in df.columns:
            df.drop(columns=['Podcast_Name', 'Episode_Title'], inplace=True)
    
    # Fill missing numerical values with median from train set
    train_medians = {}
    for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']:
        if col in train.columns:
            median_val = train[col].median()
            train_medians[col] = median_val
            train[col].fillna(median_val, inplace=True)
            if col in test.columns:
                test[col].fillna(median_val, inplace=True)
    
    # Handle Publication_Time - convert to numeric representation
    time_mapping = {
        'Morning': 0,
        'Afternoon': 1, 
        'Evening': 2,
        'Night': 3
    }
    for df in [train, test]:
        if 'Publication_Time' in df.columns:
            df['Publication_Hour'] = df['Publication_Time'].map(time_mapping)
            df.drop(columns=['Publication_Time'], inplace=True)
    
    # Encode categorical features
    # We need to ensure consistency between train and test
    label_encoders = {}
    
    for col in ['Genre', 'Publication_Day', 'Episode_Sentiment']:
        if col in train.columns and col in test.columns:
            # Combine unique values from both train and test
            unique_values = pd.concat([train[col], test[col]]).unique()
            
            # Create and fit encoder on all unique values
            le = LabelEncoder()
            le.fit(unique_values.astype(str))
            
            # Transform both train and test
            train[col] = le.transform(train[col].astype(str))
            test[col] = le.transform(test[col].astype(str))
            
            label_encoders[col] = le
    
    # Ensure all features are numeric
    train = train.select_dtypes(include=['int64', 'float64'])
    test = test.select_dtypes(include=['int64', 'float64'])
    
    return train, test


# Preprocess both train and test together to ensure consistent encoding
train_processed, test_processed = preprocess(train, test)


# Define target and features
X = train_processed.drop(columns=['Listening_Time_minutes'])
y = train_processed['Listening_Time_minutes']



# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Define objective function for Optuna
def objective(trial):
    """
    Objective function for Optuna to optimize.
    Defines the hyperparameter search space and returns the validation RMSE.
    """
    # Suggest hyperparameters
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'random_state': 42,
        'tree_method': 'hist'  # Faster training for large datasets
    }
    
    # Create and train model
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              early_stopping_rounds=50,
              verbose=False)
    
    # Make predictions
    y_pred = model.predict(X_valid)
    
    # Calculate RMSE
    rmse = mean_squared_error(y_valid, y_pred, squared=False)
    
    return rmse


# Create Optuna study 
study = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42)  # We want to minimize RMSE
)
# Run optimization
study.optimize(objective, n_trials=50, show_progress_bar=True)

# Print results
print("Best trial:")
trial = study.best_trial
print(f"  RMSE: {trial.value:.4f}")
print("  Best params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Visualize optimization
plot_optimization_history(study)
plot_param_importances(study)



# ------ ENSEMBLE SETUP ------ #
# 1. Your optimized XGBoost
xgb_model = xgb.XGBRegressor(
    **study.best_params,
    objective='reg:squarederror',
    random_state=42,
    tree_method='hist'
)

# 2. CatBoost with good defaults
cat_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    random_state=42,
    verbose=0
)

# 3. LightGBM with good defaults
lgbm_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=7,
    random_state=42,
    verbosity=-1
)


# ------ INDIVIDUAL MODEL EVALUATION ------ #
def evaluate_model(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_squared_error(y_valid, preds, squared=False)

print("Evaluating individual models...")
xgb_rmse = evaluate_model(xgb_model, X_train, y_train, X_valid, y_valid)
cat_rmse = evaluate_model(cat_model, X_train, y_train, X_valid, y_valid)
lgbm_rmse = evaluate_model(lgbm_model, X_train, y_train, X_valid, y_valid)

print(f"\nXGBoost RMSE: {xgb_rmse:.4f}")
print(f"CatBoost RMSE: {cat_rmse:.4f}")
print(f"LightGBM RMSE: {lgbm_rmse:.4f}")


# ------ SIMPLE AVERAGING ENSEMBLE ------ #
xgb_model.fit(X_train, y_train)
cat_model.fit(X_train, y_train)
lgbm_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_valid)
cat_pred = cat_model.predict(X_valid)
lgbm_pred = lgbm_model.predict(X_valid)

avg_pred = (xgb_pred + cat_pred + lgbm_pred) / 3
avg_rmse = mean_squared_error(y_valid, avg_pred, squared=False)
print(f"\nAverage Ensemble RMSE: {avg_rmse:.4f}")


# ------ WEIGHTED AVERAGE ENSEMBLE ------ #
def find_weights(weights):
    combined = weights[0]*xgb_pred + weights[1]*cat_pred + weights[2]*lgbm_pred
    return mean_squared_error(y_valid, combined)

result = minimize(find_weights, 
                 x0=[1/3, 1/3, 1/3],
                 bounds=[(0,1)]*3,
                 constraints=({'type': 'eq', 'fun': lambda w: sum(w)-1}))

optimal_weights = result.x
weighted_pred = (optimal_weights[0]*xgb_pred + 
                 optimal_weights[1]*cat_pred + 
                 optimal_weights[2]*lgbm_pred)

print(f"\nOptimal Weights - XGB: {optimal_weights[0]:.2f}, Cat: {optimal_weights[1]:.2f}, LGBM: {optimal_weights[2]:.2f}")
print(f"Weighted Ensemble RMSE: {mean_squared_error(y_valid, weighted_pred, squared=False):.4f}")


# ------ STACKING ENSEMBLE ------ #
stack_model = StackingRegressor(
    estimators=[
        ('xgb', xgb.XGBRegressor(**study.best_params)),
        ('cat', cat_model),
        ('lgbm', lgbm_model)
    ],
    final_estimator=Ridge(),
    cv=5
)

stack_model.fit(X_train, y_train)
stack_rmse = mean_squared_error(y_valid, stack_model.predict(X_valid), squared=False)
print(f"\nStacking Ensemble RMSE: {stack_rmse:.4f}")


# First add this import at the top of your notebook (with other imports)
import matplotlib.pyplot as plt

# Then your feature importance visualization code:
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
xgb.plot_importance(xgb_model)
plt.title("XGBoost Importance")

plt.subplot(1, 3, 2)
plt.barh(X.columns, cat_model.get_feature_importance())
plt.title("CatBoost Importance")

plt.subplot(1, 3, 3)
lgbm.plot_importance(lgbm_model, importance_type='gain')
plt.title("LightGBM Importance")

plt.tight_layout()
plt.show()


# ------ FINAL ENSEMBLE SUBMISSION ------ #
from datetime import datetime

# Train best ensemble on full data (example uses stacking)
final_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('cat', cat_model),
        ('lgbm', lgbm_model)
    ],
    final_estimator=Ridge(),
    cv=5
)

final_model.fit(X, y)  # Train on all data
test_preds = final_model.predict(test_processed)

# Create timestamped submission
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission = sample_submission.copy()
submission['Listening_Time_minutes'] = test_preds
submission.to_csv(f'ensemble_submission_{timestamp}.csv', index=False)

print(f"Final ensemble submission saved as: ensemble_submission_{timestamp}.csv")

