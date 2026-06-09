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


# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
import optuna
from catboost import CatBoostRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor



# Load datasets
train = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/train.csv')
test = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/test.csv')


# Separate features and target
X = train.drop(columns=["target"])
y = train["target"]
test_id = test["id"]
X_test = test.drop(columns=["id"])


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


# --- Hyperparameter Tuning ---
def objective(trial):
    model_type = trial.suggest_categorical("model_type", ["lightgbm", "catboost", "xgboost", "svr"])
    
    if model_type == "lightgbm":
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        model = lgb.LGBMRegressor(**param, random_state=42)
    
    elif model_type == "catboost":
        param = {
            "iterations": trial.suggest_int("iterations", 500, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "depth": trial.suggest_int("depth", 5, 12),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }
        model = CatBoostRegressor(**param, silent=True, random_state=42)
    
    elif model_type == "xgboost":
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0),
        }
        model = XGBRegressor(**param, random_state=42)
    
    elif model_type == "svr":
        param = {
            "C": trial.suggest_float("C", 0.1, 10.0),
            "epsilon": trial.suggest_float("epsilon", 0.01, 1.0),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
        }
        model = SVR(**param)
    
    model.fit(X_scaled, y)
    preds = model.predict(X_scaled)
    rmse = np.sqrt(mean_squared_error(y, preds))
    return rmse


# Run Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200)
best_params = study.best_params
print("Best Parameters:", best_params)


# Train Final Models
final_models = []
if best_params["model_type"] == "lightgbm":
    final_model = lgb.LGBMRegressor(**{k: v for k, v in best_params.items() if k != "model_type"}, random_state=42)
elif best_params["model_type"] == "catboost":
    final_model = CatBoostRegressor(**{k: v for k, v in best_params.items() if k != "model_type"}, silent=True, random_state=42)
elif best_params["model_type"] == "xgboost":
    final_model = XGBRegressor(**{k: v for k, v in best_params.items() if k != "model_type"}, random_state=42)
elif best_params["model_type"] == "svr":
    final_model = SVR(**{k: v for k, v in best_params.items() if k != "model_type"})
final_model.fit(X_scaled, y)


# Stacking
stacking_model = StackingRegressor(
    estimators=[
        ("lightgbm", lgb.LGBMRegressor()),
        ("catboost", CatBoostRegressor(silent=True)),
        ("xgboost", XGBRegressor()),
        ("ridge", Ridge())
    ],
    final_estimator=LinearRegression()
)
stacking_model.fit(X_scaled, y)


# Make Predictions
predictions = stacking_model.predict(X_test_scaled)


# Create Submission File
submission = pd.DataFrame({"id": test_id, "target": predictions})

