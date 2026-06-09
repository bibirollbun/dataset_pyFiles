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
import warnings
warnings.simplefilter("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import make_scorer
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
y = train["Calories"]
numeric_cols = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train["Sex_encoded"] = encoder.fit_transform(train["Sex"])
test["Sex_encoded"] = encoder.transform(test["Sex"])

def add_features(df, cols):
    df_copy = df.copy()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            feature1 = cols[i]
            feature2 = cols[j]
            df_copy[f"{feature1}_x_{feature2}"] = df_copy[feature1] * df_copy[feature2]
    return df_copy
added_train = add_features(train, numeric_cols)
added_test = add_features(test, numeric_cols)
new_train = added_train.drop(["id", "Sex", "Calories"], axis=1)
new_test = added_test.drop(["id", "Sex"], axis=1)
x = new_train
x_test = new_test
y = np.log1p(y)
folds = 5
kf = KFold(n_splits=folds, shuffle=True, random_state=42)

base_models = {
    "catboost": CatBoostRegressor(random_seed=42, cat_features=["Sex_encoded"], task_type="GPU"),
    "xgboost": XGBRegressor(random_state=42, n_estimators=1000, learning_rate=0.05, tree_method='hist', task_type="GPU"),
    "lightgbm": LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.05)
}


oof_predictions = np.zeros((len(train), len(base_models)))
test_predictions = np.zeros((len(test), len(base_models)))
model_names = list(base_models.keys())


print("\n-------base models learning-----------\n")
for i, (name, model) in enumerate(base_models.items()):
    print(f"Training {name}")
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    for fold, (train_idx, val_idx) in enumerate(kf.split(x, y)):
        x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
        x_val, y_val = x.iloc[val_idx], y.iloc[val_idx]

        if name != "lightgbm":
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)],verbose=100)

        else:
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)])

        oof_preds[val_idx] = model.predict(x_val)
        test_preds += model.predict(x_test) / folds

    oof_predictions[:, i] = oof_preds
    test_predictions[:, i] = test_preds
    rmsle_oof = np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(oof_preds)))
    print(f"{name} OOF RMSLE: {rmsle_oof:.4f}")

print("\n-------meta model learning-----------\n")
meta_model = Ridge(random_state=42)

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

grid_search = GridSearchCV(meta_model, param_grid, cv=5, scoring=rmsle_scorer, n_jobs=-1)
grid_search.fit(oof_predictions, y)

print("Best parameters found by grid search:", grid_search.best_params_)
best_meta_model = grid_search.best_estimator_


final_predictions = best_meta_model.predict(test_predictions)
final_predictions = np.expm1(final_predictions)
final_predictions = np.clip(final_predictions, 1, 314)
submission["Calories"] = final_predictions


submission.head(10)


submission.to_csv("final_submission.csv",index=False)

