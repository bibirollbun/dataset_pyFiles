import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)


train_df = pd.read_csv('/kaggle/input/agriyield-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/agriyield-2025/test.csv')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
train_df.head()


print(train_df.isnull().sum())
print(test_df.isnull().sum())


temp = train_df.drop(columns=['field_id'])
correlation = temp.corr(method='pearson')
print(correlation['yield'].sort_values(ascending=False))
correlation = temp.corr(method='spearman')
print(correlation['yield'].sort_values(ascending=False))
correlation = temp.corr(method='kendall')
print(correlation['yield'].sort_values(ascending=False))


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df["ph_deviation"] = np.abs(df["soil_ph"] - 6.5)
    
    df["ph_category"] = pd.cut(
        df["soil_ph"],
        bins=[0, 6.0, 7.5, 14],
        labels=["acidic", "neutral", "alkaline"]
    )
    
    df["organic_log"] = np.log1p(df["organic_matter"])
    
    df["temp_rain_interaction"] = df["temperature"] * df["rainfall"]
    df["temp_humidity_interaction"] = df["temperature"] * df["humidity"]
    
    df["temp_sq"] = df["temperature"] ** 2
    df["humidity_sq"] = df["humidity"] ** 2
    
    df["rainfall_intensity"] = df["rainfall"] / 120
    
    df["rainfall_water_retention"] = df["rainfall"] * (1 - df["sand_pct"] / 100.0)
    
    df["ndvi_sq"] = df["ndvi"] ** 2
    df["ndvi_log"] = np.log1p(df["ndvi"])
    
    ph_dummies = pd.get_dummies(df["ph_category"], prefix="ph", drop_first=True)
    df = pd.concat([df.drop(columns=["ph_category"]), ph_dummies], axis=1)
    
    return df

train_df = transform_features(train_df)
test_df = transform_features(test_df)


X = train_df.drop(['field_id', 'yield'], axis=1)
y = train_df['yield']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_test = test_df.drop(['field_id'], axis=1)


import optuna
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

import os
os.environ["CATBOOST_LOGGING_LEVEL"] = "Silent"

cv = KFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial, model_type, X, y):
    if model_type == "xgb":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
        }
        model = XGBRegressor(**params)

    elif model_type == "lgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "verbose": -1,
        }
        model = LGBMRegressor(**params)

    elif model_type == "cat":
        params = {
            "iterations": trial.suggest_int("iterations", 500, 2000),
            "depth": trial.suggest_int("depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "random_seed": 42,
            "verbose": 0
        }
        model = CatBoostRegressor(**params)

    elif model_type == "rf":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "max_features": trial.suggest_categorical("max_features", [1.0, "sqrt", "log2"]),
            "random_state": 42,
        }
        model = RandomForestRegressor(**params)

    scores = cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=cv)
    return -scores.mean()


types = ["xgb", "lgbm", "cat", "rf"]
models = [XGBRegressor(), LGBMRegressor(), CatBoostRegressor(verbose=0), RandomForestRegressor()]

for i in range(4):
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, types[i], X_train, y_train), n_trials=30)
    print(f"Best params {types[i]}:", study.best_params)
    models[i].set_params(**study.best_params)


rmses = []

for i in range(4):
    model = models[i]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"RMSE on validation set: {rmse}")
    rmses.append(rmse)


from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)

estimators = [
    ("xgb", models[0]),
    ("lgbm", models[1]),
    ("cat", models[2]),
    ("rf", models[3]),
]

stack = StackingRegressor(
    estimators=estimators,
    final_estimator=LGBMRegressor(random_state=42),
    cv=cv,
    n_jobs=-1
)

stack.fit(X_train, y_train)
stack_preds = stack.predict(X_val)

stack_rmse = np.sqrt(mean_squared_error(y_val, stack_preds))
print("Stacking RMSE:", stack_rmse)

rmses.append(stack_rmse)
models.append(stack)


best_idx = min(enumerate(rmses), key=lambda x: x[1])[0]
pred_test = models[best_idx].predict(X_test)


submission = pd.read_csv('/kaggle/input/agriyield-2025/sample_submission.csv')
submission['yield'] = pred_test
submission.to_csv('submission.csv', index=False)
submission

