import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from xgboost import XGBRegressor
import xgboost as xgb
import optuna
import lightgbm as lgb
from cuml.ensemble import RandomForestRegressor
import cudf
from sklearn.ensemble import StackingRegressor
from scipy.optimize import minimize


data = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
data.head()


X = data.drop(columns = ["id", "Calories"])
y = data["Calories"]


X["Sex_Num"] = X["Sex"].map({"male": 1, "female":0})
X = X.drop(columns = ["Sex"])


X.info()


X.columns


X["BMI"] = X["Weight"] / (X["Height"] ** 2)
X["Age_Adj_Weight"] = X["Weight"] * X["Age"]
X["HR/Duration"] = X["Heart_Rate"] / X["Duration"]
X["HR*Duration"] = X["Heart_Rate"] * X["Duration"]
X["Body_Temp_Dev"] = X["Body_Temp"] - 37
X["Body_Temp*HR"] = X["Body_Temp"] * X["Heart_Rate"]
X["Weight*Duration"] = X["Weight"] * X["Duration"]
X["MET"] = (X["Heart_Rate"] * X["Duration"]) / 100
X["Burn_Rate"] = X["Heart_Rate"] / X["Weight"]
X["Age_Adj_HR"] = (X["Heart_Rate"] / (220 - X["Age"])) * X["Duration"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 19)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.25, random_state = 19)


def get_rmsle(y_test, y_pred):
    return np.sqrt(np.mean((np.log1p(np.maximum(y_pred, 0)) - np.log1p(np.maximum(y_test, 0))) ** 2))


# Objective function for Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 8), 
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-4, 1, log=True), 
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'num_leaves': trial.suggest_int('num_leaves', 10, 50), 
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
        'device': 'gpu',
        'verbose': -1
    }
    
    # Initialize and train the model
    model = lgb.LGBMRegressor(
        objective = 'regression',
        metric = 'rmse',
        random_state = 19,
        **params
    )
    
    model.fit(
        X_train, np.log1p(np.maximum(y_train, 0)),
        eval_set = [(X_valid, np.log1p(np.maximum(y_valid, 0)))],
        eval_metric='rmse',
        callbacks = [lgb.early_stopping(stopping_rounds = 25, verbose = False)]
    )
    
    y_log_pred = model.predict(X_valid)
    y_pred = np.expm1(y_log_pred)
    result = get_rmsle(y_valid, y_pred)
    
    return result


study = optuna.create_study(direction = 'minimize')
study.optimize(objective, n_trials = 100)


# Print the best parameters and value
print("Best parameters:", study.best_params)
print("Best RMSLE:", study.best_value)


best_params = study.best_params
best_params.update({
    'device': 'gpu',
    'verbose': -1
})


optuna_lightgbm = lgb.LGBMRegressor(
    objective = 'regression',
    **best_params,
    metric = 'rmse',
    random_state = 19,
)

optuna_lightgbm.fit(X_train, np.log1p(np.maximum(y_train, 0)),
                    eval_set = [(X_valid, np.log1p(np.maximum(y_valid, 0)))],
                    eval_metric='rmse',
                    callbacks = [lgb.early_stopping(stopping_rounds = 25, verbose = False)])


y_log_pred = optuna_lightgbm.predict(X_test)
y_pred = np.expm1(y_log_pred)
result = get_rmsle(y_test, y_pred)
print("Test RMSLE:", result)


test_data = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

test_features = test_data.drop(columns = ["id"])

test_features["Sex_Num"] = test_features["Sex"].map({"male": 1, "female":0})
test_features = test_features.drop(columns = ["Sex"])

test_features["BMI"] = test_features["Weight"] / (test_features["Height"] ** 2)
test_features["Age_Adj_Weight"] = test_features["Weight"] * test_features["Age"]
test_features["HR/Duration"] = test_features["Heart_Rate"] / test_features["Duration"]
test_features["HR*Duration"] = test_features["Heart_Rate"] * test_features["Duration"]
test_features["Body_Temp_Dev"] = test_features["Body_Temp"] - 37
test_features["Body_Temp*HR"] = test_features["Body_Temp"] * test_features["Heart_Rate"]
test_features["Weight*Duration"] = test_features["Weight"] * test_features["Duration"]
test_features["MET"] = (test_features["Heart_Rate"] * test_features["Duration"]) / 100
test_features["Burn_Rate"] = test_features["Heart_Rate"] / test_features["Weight"]
test_features["Age_Adj_HR"] = (test_features["Heart_Rate"] / (220 - test_features["Age"])) * test_features["Duration"]


y_sub_log = optuna_lightgbm.predict(test_features)
y_sub = np.expm1(y_sub_log)
y_sub


sub6 = pd.DataFrame({"id": test_data["id"], "Calories": y_sub})
sub6.head()


sub6.to_csv("sub6.csv", index = False)


# Objective function for Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'tree_method': 'hist', 
        'device': 'cuda',
        'max_bin': trial.suggest_int('max_bin', 128, 512), 
        'eval_metric': 'rmse', 
        'early_stopping_rounds': 25
    }
    
    # Initialize and train the model
    model = XGBRegressor(
        objective = 'reg:squarederror',
        **params,
        random_state = 19
    )
    
    model.fit(
        X_train, np.log1p(np.maximum(y_train, 0)),
        eval_set = [(X_valid, np.log1p(np.maximum(y_valid, 0)))],
        verbose = False
    )
    
    y_log_pred = model.predict(X_valid)
    y_pred = np.expm1(y_log_pred)
    result = get_rmsle(y_valid, y_pred)
    
    return result


study = optuna.create_study(direction = 'minimize')
study.optimize(objective, n_trials = 100)


# Print the best parameters and value
print("Best parameters:", study.best_params)
print("Best RMSLE:", study.best_value)


best_params = study.best_params
best_params.update({
    'tree_method': 'hist',  
    'device': 'cuda', 
    'eval_metric': 'rmse', 
    'early_stopping_rounds': 25 
})


optuna_xgb = XGBRegressor(
    objective = 'reg:squarederror',
    **best_params,
    random_state = 19
)

optuna_xgb.fit(X_train, np.log1p(np.maximum(y_train, 0)),
               eval_set = [(X_valid, np.log1p(np.maximum(y_valid, 0)))],
               verbose = False)


y_log_pred = optuna_xgb.predict(X_test)
y_pred = np.expm1(y_log_pred)
result = get_rmsle(y_test, y_pred)
print("Test RMSLE:", result)


y_sub_log = optuna_xgb.predict(test_features)
y_sub = np.expm1(y_sub_log)
y_sub


sub4 = pd.DataFrame({"id": test_data["id"], "Calories": y_sub})
sub4.head()


sub4.to_csv("sub4.csv", index = False)


# Objective function for Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.5, 1.0),
        'n_bins': trial.suggest_int('n_bins', 16, 128),  # cuML-specific
    }

    # Convert data to cuDF for GPU
    X_train_cudf = cudf.from_pandas(X_train) if isinstance(X_train, pd.DataFrame) else cudf.DataFrame(X_train)
    y_train_cudf = cudf.Series(np.log1p(np.maximum(y_train, 0)))
    X_valid_cudf = cudf.from_pandas(X_valid) if isinstance(X_valid, pd.DataFrame) else cudf.DataFrame(X_valid)
    y_valid_cudf = cudf.Series(np.log1p(np.maximum(y_valid, 0)))
    
    # Initialize and train the model
    model = RandomForestRegressor(**params, output_type='numpy', n_streams=1)

    model.fit(X_train_cudf, y_train_cudf)
    
    y_log_pred = model.predict(X_valid_cudf)
    y_pred = np.expm1(y_log_pred)
    result = get_rmsle(y_valid, y_pred)
    
    return result


study = optuna.create_study(direction = 'minimize')
study.optimize(objective, n_trials = 50)


# Print the best parameters and value
print("Best parameters:", study.best_params)
print("Best RMSLE:", study.best_value)


best_params = study.best_params


optuna_rfr = RandomForestRegressor(**best_params, output_type = 'numpy', random_state = 19, n_streams = 1)
optuna_rfr.fit(X_train, np.log1p(np.maximum(y_train, 0)))


y_log_pred = optuna_rfr.predict(X_test)
y_pred = np.expm1(y_log_pred)
result = get_rmsle(y_test, y_pred)
print("Test RMSLE:", result)


# Get model params
lgbm_params = optuna_lightgbm.get_params()

xgb_params = optuna_xgb.get_params()
xgb_params.update({'early_stopping_rounds': None})

rfr_params = optuna_rfr.get_params()


# Define base models
lgbm_model = lgb.LGBMRegressor(**lgbm_params)

xgb_model = xgb.XGBRegressor(**xgb_params)

rfr_model = RandomForestRegressor(**rfr_params)


# Define stacking regressor
stacked_model = StackingRegressor(
    estimators = [('xgb', xgb_model), ('lgbm', lgbm_model), ('rfr', rfr_model)],
    final_estimator = LinearRegression(),
    cv = 5
)


# Train with log-transformed target for RMSLE
y_train_log = np.log1p(np.maximum(y_train, 0))
stacked_model.fit(X_train, y_train_log)


# Predict and calculate RMSLE
y_pred_log = stacked_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
rmsle = np.sqrt(np.mean((np.log1p(np.maximum(y_test, 0)) - np.log1p(np.maximum(y_pred, 0))) ** 2))
print(f"Stacked RMSLE: {rmsle}")


y_sub_log = stacked_model.predict(test_features)
y_sub = np.expm1(y_sub_log)
y_sub


sub7 = pd.DataFrame({"id": test_data["id"], "Calories": y_sub})
sub7.head()


sub7.to_csv("sub7.csv", index = False)


# Define stacking regressor
stacked_model2 = StackingRegressor(
    estimators = [('xgb', xgb_model), ('lgbm', lgbm_model)],
    final_estimator = LinearRegression(),
    cv = 5
)


# Train with log-transformed target for RMSLE
y_train_log = np.log1p(np.maximum(y_train, 0))
stacked_model2.fit(X_train, y_train_log)


# Predict and calculate RMSLE
y_pred_log = stacked_model2.predict(X_test)
y_pred = np.expm1(y_pred_log)
rmsle = np.sqrt(np.mean((np.log1p(np.maximum(y_test, 0)) - np.log1p(np.maximum(y_pred, 0))) ** 2))
print(f"Stacked RMSLE: {rmsle}")


y_sub_log = stacked_model2.predict(test_features)
y_sub = np.expm1(y_sub_log)
y_sub


sub8 = pd.DataFrame({"id": test_data["id"], "Calories": y_sub})
sub8.head()


sub8.to_csv("sub8.csv", index = False)


y_pred_xgb_log = optuna_xgb.predict(X_valid)
y_pred_lgb_log = optuna_lightgbm.predict(X_valid)
y_pred_rfr_log = optuna_rfr.predict(X_valid)
y_pred_xgb = np.expm1(y_pred_xgb_log)
y_pred_lgb = np.expm1(y_pred_lgb_log)
y_pred_rfr = np.expm1(y_pred_rfr_log)


def rmsle_weights(weights):
    y_pred = weights[0] * y_pred_xgb + weights[1] * y_pred_lgb + weights[2] * y_pred_rfr
    return np.sqrt(np.mean((np.log1p(np.maximum(y_valid, 0)) - np.log1p(np.maximum(y_pred, 0))) ** 2))


result = minimize(rmsle_weights, [1/3, 1/3, 1/3], constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1}, bounds=[(0, 1)]*3)
weights = result.x
print(f"Optimized weights: {weights}")


y_test_pred_xgb = np.expm1(optuna_xgb.predict(X_test))
y_test_pred_lgb = np.expm1(optuna_lightgbm.predict(X_test))
y_test_pred_rfr = np.expm1(optuna_rfr.predict(X_test))


y_test_pred = weights[0] * y_test_pred_xgb + weights[1] * y_test_pred_lgb + weights[2] * y_test_pred_rfr
rmsle = np.sqrt(np.mean((np.log1p(np.maximum(y_test, 0)) - np.log1p(np.maximum(y_test_pred, 0))) ** 2))
print(f"Weighted Ensemble RMSLE: {rmsle}")


y_sub_pred_xgb = np.expm1(optuna_xgb.predict(test_features))
y_sub_pred_lgb = np.expm1(optuna_lightgbm.predict(test_features))
y_sub_pred_rfr = np.expm1(optuna_rfr.predict(test_features))


y_sub = weights[0] * y_sub_pred_xgb + weights[1] * y_sub_pred_lgb + weights[2] * y_sub_pred_rfr
y_sub


sub9 = pd.DataFrame({"id": test_data["id"], "Calories": y_sub})
sub9.head()


sub9.to_csv("sub9.csv", index = False)




