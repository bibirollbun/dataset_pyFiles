import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')



train_data = train_df.copy()
test_data = test_df.copy()
TARGET = "accident_risk"
train_data['curvature'] = train_data['curvature'].round(2)
test_data['curvature'] = test_data['curvature'].round(2)
TARGET_ENCODED_FEATURES = train_data.columns.tolist()
TARGET_ENCODED_FEATURES.remove("id")
TARGET_ENCODED_FEATURES.remove("accident_risk")



import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import math
from sklearn.model_selection import KFold


def score_dataset(model):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = []
    for train_index, val_index in kf.split(train_data):
        X_train, X_val = train_data.iloc[train_index].copy(), train_data.iloc[val_index].copy()
        y_train, y_val = train_data.iloc[train_index].accident_risk.copy(), train_data.iloc[val_index].accident_risk.copy()
        for feature in TARGET_ENCODED_FEATURES:
            mapping = X_train.groupby(feature)[TARGET].mean()
            X_train[feature + "_target_encoded"] = X_train[feature].map(mapping).copy()
            X_val[feature + "_target_encoded"] = X_val[feature].map(mapping).copy()
            if feature != 'curvature':
                X_train[feature] = X_train[feature].astype('category')
                X_val[feature] = X_val[feature].astype('category')
        X_train = X_train.drop(TARGET, axis = 1)
        X_val = X_val.drop(TARGET, axis = 1)
        X_train = pd.get_dummies(X_train)
        X_val = pd.get_dummies(X_val)
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
        model.fit(X_train, y_train, eval_set = [(X_val, y_val)], verbose = False)
        prediction = model.predict(X_val)
        rmses.append(mean_squared_error(prediction, y_val, squared = False))
    return np.mean(rmses)


    
def objective(trial):
    xgb_params = dict(
        max_depth=trial.suggest_int("max_depth", 2, 15),
        learning_rate=trial.suggest_float("learning_rate", 1e-2, 1e-1, log=True),
        n_estimators=trial.suggest_int("n_estimators", 1000, 5000),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
        subsample=trial.suggest_float("subsample", 0.2, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1e2, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),
        max_bin=trial.suggest_int("max_bin", 128, 512),
        gamma=trial.suggest_float("gamma", 0, 10),
        n_jobs = -1,
        objective = "reg:squarederror",
        device = "cuda",
        random_state = 42,
        early_stopping_rounds = 100
    )
    xgb = XGBRegressor(**xgb_params)
    return score_dataset(xgb)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best params:", study.best_params)
print("Best RMSE:", study.best_value)


answer_model = XGBRegressor(
    **study.best_params,
    n_jobs=-1,
    objective="reg:squarederror",
    device="cuda",
    random_state=42
)


X_test = test_df.copy()
X_train = train_df.copy()
for feature in TARGET_ENCODED_FEATURES:
    mapping = X_train.groupby(feature)[TARGET].mean()
    X_train[feature + "_target_encoded"] = X_train[feature].map(mapping).copy()
    X_test[feature + "_target_encoded"] = X_test[feature].map(mapping).copy()
    if feature != 'curvature':
        X_train[feature] = X_train[feature].astype('category')
        X_test[feature] = X_test[feature].astype('category')
y_train = X_train.pop(TARGET)
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
answer_model.fit(X_train, y_train)


answer = answer_model.predict(X_test)
output = pd.DataFrame(
    {
        "id": test_df.id,
        "accdient_risk": answer.flatten()
    }
)
output.to_csv("output.csv", index = False)

