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
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


print(train.describe())


print(train.select_dtypes('object'))
print(train.columns)


train = train.fillna(0)
test = test.fillna(0)

target = train["diagnosed_diabetes"]
features = train.drop(['diagnosed_diabetes','id'], axis=1)
upd_test = test.drop('id',axis=1)

cat_cols = features.select_dtypes('object').columns.tolist()


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(pd.concat([features[cat_cols], upd_test[cat_cols]], axis = 0))

encoded_train = encoder.transform(features[cat_cols])
encoded_test = encoder.transform(upd_test[cat_cols])

encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(cat_cols), index=features.index)
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(cat_cols), index = upd_test.index)

features_final = pd.concat([features.drop(cat_cols, axis = 1), encoded_train_df.reset_index(drop=True)], axis = 1)
test_final = pd.concat([upd_test.drop(cat_cols, axis=1), encoded_test_df.reset_index(drop=True)], axis = 1)
print(features.shape, target.shape)
x_train, x_val, y_train, y_val = train_test_split(features_final, target, test_size= 0.2, random_state=42)



import optuna
from sklearn.metrics import accuracy_score
import lightgbm as lgb

def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 10)
    }
    train_data = lgb.Dataset(x_train, label = y_train)
    valid_data = lgb.Dataset(x_val, label = y_val)
    model = lgb.train(
        params,
        train_data,
        valid_sets = [valid_data],
        num_boost_round = 500,
        callbacks = [
        lgb.early_stopping(stopping_rounds = 50, 
        verbose = False,)]
    )
    preds = model.predict(x_val)
    preds = [1 if x > 0.5 else 0 for x in preds]
    return accuracy_score(y_val, preds)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=70)
print(study.best_value)
print(study.best_params)


best_params = study.best_params
model = lgb.LGBMClassifier(**best_params)
model.fit(x_train, y_train)


y_pred = model.predict_proba(x_val)[:,1]


from sklearn.metrics import roc_auc_score
print("Validation ROC AUC:", roc_auc_score(y_val, y_pred))



test_preds = model.predict_proba(test_final)[:,1]
print(test_preds)


submission = pd.DataFrame({
'id':test.id,
'diagnosed_diabetes':test_preds,})
submission.to_csv("submission.csv", index=False)

