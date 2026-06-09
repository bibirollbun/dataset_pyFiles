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


df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
dftest = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


df.shape


dftest.shape


df.head()


dftest.head()


df.info()


df.isnull().sum()


dftest.isnull().sum()


df.duplicated().sum()


dftest.duplicated().sum()





idtest = dftest['id']#save id for submission


df.drop('id',axis = 1,inplace = True)
dftest.drop('id',axis = 1,inplace = True)



categorical = df.select_dtypes(include = ['object'] ).columns
numerical = df.select_dtypes(exclude = ['object']).columns

print("\nCategorical columns:",categorical.tolist())
print("\nNumerical columns:",numerical.tolist())



numerical = numerical.drop("Price", errors='ignore')



for column in categorical:
    print(f"\nValue counts in '{column}': \n{df[column].value_counts()}")



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer



# Define numerical pipeline
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define categorical pipeline
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False, drop='first'))
])

# Combine both pipelines into a ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numerical),
    ('cat', cat_pipeline, categorical)
])

# Prepare training data
x = df.drop('Price', axis=1)  
y = df['Price']

# Fit and transform training data
xprocessed = preprocessor.fit_transform(x)



# Transform test data
test_processed = preprocessor.transform(dftest)



print(xprocessed.shape, y.shape)  



import torch
print("Torch GPU Available:", torch.cuda.is_available())

import xgboost as xgb
print("XGBoost GPU Support:", xgb.XGBRegressor(tree_method="gpu_hist").booster == "gpu_hist")



import xgboost as xgb
print(xgb.__version__)
print(xgb.get_config())  # Check if GPU is enabled






import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb

# Prepare training & validation data
xtrain, xval, ytrain, yval = train_test_split(xprocessed, y, test_size=0.2, random_state=42)

# Store best models and RMSE scores
best_models = {}
rmse_scores = {}

def objective(trial, model_name):
    """Objective function for Optuna optimization"""
    if model_name == "CatBoost":
        bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "No"])
        
        params = {
            "iterations": trial.suggest_categorical("iterations", [1000, 1500]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 6, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),  # L2 Regularization
            "bootstrap_type": bootstrap_type,
            "task_type": "GPU"
        }
        
        # Add subsample only if bootstrap_type is Bernoulli
        if bootstrap_type == "Bernoulli":
            params["subsample"] = trial.suggest_float("subsample", 0.6, 0.9)

        model = CatBoostRegressor(**params, verbose=0)
    
    elif model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 300, 500]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 6, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),  # Regularization
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),  # L2 Regularization
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor",
            "gpu_id": 0
        }
        model = xgb.XGBRegressor(**params)
    
    elif model_name == "LightGBM":
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 300, 500]),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 100),
            "max_depth": trial.suggest_int("max_depth", -1, 15),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),  # Regularization
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),  # L2 Regularization
            "device": "gpu"
        }
        model = lgb.LGBMRegressor(**params)
    
    # Train model
    model.fit(xtrain, ytrain)
    
    # Evaluate RMSE on validation set
    ypred = model.predict(xval)
    rmse = mean_squared_error(yval, ypred, squared=False)
    
    return rmse

# Optimize each model
for model_name in ["CatBoost", "XGBoost", "LightGBM"]:
    print(f"\nðŸ”¹ Optimizing {model_name} using Optuna...")
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, model_name), n_trials=20, n_jobs=1)
    
    # Best model parameters
    best_params = study.best_params
    print(f"{model_name} Best Params: {best_params}")
    
    # Train the best model
    if model_name == "CatBoost":
        best_model = CatBoostRegressor(**best_params, task_type="GPU", verbose=0)
    elif model_name == "XGBoost":
        best_model = xgb.XGBRegressor(**best_params, tree_method="gpu_hist", predictor="gpu_predictor", gpu_id=0)
    elif model_name == "LightGBM":
        best_model = lgb.LGBMRegressor(**best_params, device="gpu")
    
    best_model.fit(xtrain, ytrain)
    best_models[model_name] = best_model
    
    # Evaluate final RMSE
    ypred = best_model.predict(xval)
    final_rmse = mean_squared_error(yval, ypred, squared=False)
    rmse_scores[model_name] = final_rmse
    print(f"{model_name} Final RMSE: {final_rmse}\n")

# Print final RMSE scores
print("\nðŸ”¹ RMSE scores of all models:")
for name, rmse in rmse_scores.items():
    print(f"{name}: {rmse}")



import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Define individual models with best hyperparameters
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.015241454296740045,
    num_leaves=39,
    max_depth=7,
    subsample=0.6019182665027505,
    reg_lambda=7.575371131413647
)

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.015241454296740045,
    max_depth=7,
    subsample=0.6019182665027505,
    reg_lambda=7.575371131413647,
    objective='reg:squarederror',
    random_state=42
)

cat_model = cb.CatBoostRegressor(
    bootstrap_type='Bernoulli',
    iterations=1000,
    learning_rate=0.010304980185474582,
    depth=6,
    l2_leaf_reg=2.5483643218760723,
    subsample=0.8670822285099169,
    verbose=0
)

# Fit models individually with early stopping
lgb_model.fit(xtrain, ytrain, eval_set=[(xval, yval)], eval_metric='rmse', callbacks=[lgb.early_stopping(50)])
xgb_model.fit(xtrain, ytrain, eval_set=[(xval, yval)], eval_metric='rmse', early_stopping_rounds=50, verbose=False)
cat_model.fit(xtrain, ytrain, eval_set=[(xval, yval)], early_stopping_rounds=50, verbose=False)

# Create Voting Regressor
voting_regressor = VotingRegressor(estimators=[
    ('lightgbm', lgb_model),
    ('xgboost', xgb_model),
    ('catboost', cat_model)
])

# Train Voting Regressor
voting_regressor.fit(xtrain, ytrain)

# Predict on validation set
val_predictions = voting_regressor.predict(xval)

# Calculate RMSE for validation set
rmse = np.sqrt(mean_squared_error(yval, val_predictions))
print(f"Voting Regressor RMSE: {rmse:.5f}")

# Make predictions on test data
test_predictions = voting_regressor.predict(test_processed)

# Prepare submission file
submission = pd.DataFrame({'id': idtest, 'Price': test_predictions})
submission.to_csv("submission.csv", index=False)

# Display first few rows
print(submission.head())


