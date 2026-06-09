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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
import optuna
from joblib import parallel_backend
import multiprocessing


# Set up Optuna to use all available CPU cores
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress verbose output

# Load data 
df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv') 
# Define features and target
X = df.drop(['id', 'Calories'], axis=1)
y = df['Calories']


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing
numeric_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
categorical_features = ['Sex']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])



# Define Optuna objective function for hyperparameter tuning
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'n_jobs': -1  # Use all available cores for XGBoost,
        
    }
    
    model = XGBRegressor(**param, random_state=42)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Parallelize cross-validation with joblib
    with parallel_backend('loky', n_jobs=-1):
        scores = cross_val_score(pipeline, X_train, np.log1p(y_train), 
                               scoring='neg_mean_squared_log_error', 
                               cv=5)
    rmsle = np.sqrt(-scores.mean())
    return rmsle



# Run Optuna optimization with parallel trials
n_jobs = multiprocessing.cpu_count()
study = optuna.create_study(direction='minimize')
#Increase n_trials 
study.optimize(objective, n_trials=10, n_jobs=n_jobs)

# Get best parameters
best_params = study.best_params
best_params['n_jobs'] = -1  # Ensure final model uses all cores
print(best_params)
# Create final model with best parameters
final_model = XGBRegressor(**best_params, random_state=42)

# Create and fit pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', final_model)
])

# Fit model with parallel processing
with parallel_backend('loky', n_jobs=-1):
    pipeline.fit(X_train, np.log1p(y_train))



# Predict and evaluate
y_pred = pipeline.predict(X_test)
rmsle = np.sqrt(mean_squared_log_error(y_test, np.expm1(y_pred)))

print(f"Test RMSLE: {rmsle:.4f}")




# Feature importance
feature_names = (numeric_features + 
                [f"Sex_{cat}" for cat in pipeline.named_steps['preprocessor']
                .named_transformers_['cat'].categories_[0][1:]])
importances = pipeline.named_steps['model'].feature_importances_
feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Function to predict calories for new data
def predict_calories(new_data):
    return pipeline.predict(new_data)



new_data = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
predicted_calories = np.expm1(predict_calories(new_data))


print(predicted_calories)

