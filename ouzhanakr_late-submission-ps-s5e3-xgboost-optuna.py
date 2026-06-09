# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import os, gc, warnings
warnings.filterwarnings("ignore")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


train.info()


train.head()


train.drop('id',axis=1,inplace=True)
test.drop('id', axis=1, inplace=True)


target = 'rainfall'
features = [i for i in train.columns if i != target]



train["day_sin"] = np.sin(2*np.pi*(train["day"]/31.0))
train["day_cos"] = np.cos(2*np.pi*(train["day"]/31.0))
test["day_sin"]  = np.sin(2*np.pi*(test["day"]/31.0))
test["day_cos"]  = np.cos(2*np.pi*(test["day"]/31.0))

rad_tr = np.deg2rad(train["winddirection"]);  rad_te = np.deg2rad(test["winddirection"])
train["wind_u"] = train["windspeed"] * np.sin(rad_tr); test["wind_u"] = test["windspeed"] * np.sin(rad_te)
train["wind_v"] = train["windspeed"] * np.cos(rad_tr); test["wind_v"] = test["windspeed"] * np.cos(rad_te)

train["temp_range"] = train["maxtemp"] - train["mintemp"]
test["temp_range"]  = test["maxtemp"]  - test["mintemp"]
train["temp_mean2"] = (train["maxtemp"] + train["mintemp"]) / 2.0
test["temp_mean2"]  = (test["maxtemp"]  + test["mintemp"])  / 2.0

train["dewpoint_dep"] = train["temparature"] - train["dewpoint"]
test["dewpoint_dep"]  = test["temparature"]  - test["dewpoint"]

train["humid_x_cloud"] = train["humidity"] * train["cloud"]
test["humid_x_cloud"]  = test["humidity"]  * test["cloud"]
train["humid_x_sun"] = train["humidity"] * train["sunshine"]
test["humid_x_sun"]  = test["humidity"]  * test["sunshine"]

train["press_x_temp"] = train["pressure"] * train["temparature"]
test["press_x_temp"]  = test["pressure"]  * test["temparature"]


# def objective(trial):
#     params = {
#         'n_estimators':     trial.suggest_int('n_estimators', 500, 20000),
#         "learning_rate":    trial.suggest_float("learning_rate", 0.0001, 0.3, log=True),
#         "max_depth":        trial.suggest_int("max_depth", 3, 15),
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
#         "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
#         "gamma":            trial.suggest_float("gamma", 0.0, 2.0),
#         "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
#         "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
#         "random_state":     42,
#         "eval_metric":      "auc",
#         "n_jobs":           -1,
#         "tree_method":      "hist"
#     }

#     fold_aucs = []
#     for train_idx, val_idx in kf.split(X,y):
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#         model = XGBClassifier(**params)
#         model.fit(
#             X_train,y_train,
#             eval_set=[(X_val, y_val)],
#             verbose=False,
#             early_stopping_rounds = 200
#         )
        
#         preds = model.predict_proba(X_val)[:,1]
#         fold_aucs.append(roc_auc_score(y_val, preds))

#         return np.mean(fold_aucs)


# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials= 300, show_progress_bar=True)
# print("Best AUC:", study.best_value)
# print("Best Params:", study.best_params)


# best_params = study.best_params
# best_params.update({
#     "random_state": 42,
#     "n_jobs": -1,
#     "eval_metric": "auc",
#     "tree_method": "hist"
# })


# best_params


best_params = {
    'n_estimators': 4566,
    'learning_rate': 0.0018675923793610339,
    'max_depth': 9,
    'min_child_weight': 12,
    'subsample': 0.7005904576317113,
    'colsample_bytree': 0.7071009956786711,
    'gamma': 1.9683373965863564,
    'reg_alpha': 0.03889642010212889,
    'reg_lambda': 0.0822784758697506,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'auc',
    'tree_method': 'hist'
}


test.shape


kf = KFold(n_splits=5, shuffle=True, random_state=42)
X = train.drop(columns=[target])
y = train[target]


oof = np.zeros(len(X))
test_pred = np.zeros(len(test))
scores = []


for fold, (train_idx, val_idx) in enumerate(kf.split(X,y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(**best_params)
    model.fit(
        X_train,y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=300
    )
    preds_val = model.predict_proba(X_val)[:,1]
    oof[val_idx] = preds_val
    test_pred += model.predict_proba(test)[:,1]/ 5

    score = roc_auc_score(y_val, preds_val)
    scores.append(score)
    print(f"AUC: {score:.5f}")
print("\nFold AUC:", [f"{s:.4f}" for s in scores])
print(f"OOF AUC mean: {np.mean(scores):.5f}")


sub["rainfall"] = test_pred
sub.to_csv("submission.csv", index=False)


sub


ls

