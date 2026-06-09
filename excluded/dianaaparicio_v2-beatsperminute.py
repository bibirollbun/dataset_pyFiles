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


# Librerías
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
import optuna



# Reading Data
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

X = train.drop(columns=["BeatsPerMinute", "id"])
y = train["BeatsPerMinute"]
X_test = test.drop(columns=["id"])


# Checking for missing values
print(train.isna().sum())
print()
print(test.isna().sum())


# Statistical description of the dataset
# For training dataset
train.describe()


# For testing
test.describe()


# Preprocesamiento
categorical_cols = X.select_dtypes(include=["object", "category"]).columns

# Target encoding
encoder = TargetEncoder(cols=categorical_cols)
X_enc = encoder.fit_transform(X, y)
X_test_enc = encoder.transform(X_test)

# Escalado
scaler = StandardScaler()
X_enc = scaler.fit_transform(X_enc)
X_test_enc = scaler.transform(X_test_enc)


# Selección de features con varianza baja
from sklearn.feature_selection import VarianceThreshold

# threshold controla qué tan "constante" debe ser una columna para eliminarla
selector = VarianceThreshold(threshold=1e-4)
X_enc = selector.fit_transform(X_enc)
X_test_enc = selector.transform(X_test_enc)

print(f"Features después de eliminar baja varianza: {X_enc.shape[1]}")


# Optuna para LightGBM
def objective(trial):
    params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 5000,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,            # sin límite
    "min_data_in_leaf": 10,     # permite hojas más pequeñas
    "min_gain_to_split": 0.0,   # no exige ganancia mínima
    "subsample": 0.8,
    "colsample_bytree": 0.8
}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    for train_idx, valid_idx in kf.split(X_enc):
        X_tr, X_val = X_enc[train_idx], X_enc[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)]
        )
        
        preds = model.predict(X_val)
        rmse_scores.append(mean_squared_error(y_val, preds, squared=False))
    
    return np.mean(rmse_scores)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1)
best_params = study.best_params



# Ensamble con LightGBM, XGBoost, CatBoost
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_enc))
test_preds = np.zeros(len(X_test_enc))

for train_idx, valid_idx in kf.split(X_enc):
    X_tr, X_val = X_enc[train_idx], X_enc[valid_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(**best_params, n_estimators=2000)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)]
    )
    lgb_preds_val = lgb_model.predict(X_val)
    lgb_preds_test = lgb_model.predict(X_test_enc)

    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=0)
    xgb_preds_val = xgb_model.predict(X_val)
    xgb_preds_test = xgb_model.predict(X_test_enc)

    # CatBoost
    cat_model = cb.CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=8, verbose=0, loss_function="RMSE")
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100)
    cat_preds_val = cat_model.predict(X_val)
    cat_preds_test = cat_model.predict(X_test_enc)

    # Ensamble (simple average)
    preds_val = (lgb_preds_val + xgb_preds_val + cat_preds_val) / 3
    preds_test = (lgb_preds_test + xgb_preds_test + cat_preds_test) / 3

    oof_preds[valid_idx] = preds_val
    test_preds += preds_test / kf.n_splits

cv_rmse = mean_squared_error(y, oof_preds, squared=False)
print("CV RMSE:", cv_rmse)


#Generar y validar submission

assert len(test_preds) == len(test), f"Error: {len(test_preds)} preds vs {len(test)} filas en test"

submission = pd.DataFrame({
    "id": test["id"].values,        
    "BeatsPerMinute": test_preds
})

expected_cols = ["id", "BeatsPerMinute"]
assert list(submission.columns) == expected_cols, f"Error: columnas {list(submission.columns)} no coinciden con {expected_cols}"

print("Shape submission:", submission.shape)
print(submission.head())

submission.to_csv("submission.csv", index=False)
print("Archivo 'submission.csv' generado correctamente")


