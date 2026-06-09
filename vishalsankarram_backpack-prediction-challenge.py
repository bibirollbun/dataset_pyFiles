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


train_df_1=pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv",on_bad_lines="skip")
train_df_2=pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv",on_bad_lines="skip")
train_df= pd.concat([train_df_1, train_df_2], ignore_index=True)
test_df=pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv",on_bad_lines="skip")
print(train_df.shape)
print(test_df.shape)


train_df['Waterproof'] = train_df['Waterproof'].map({'No': 0, 'Yes': 1})
train_df['Laptop Compartment'] = train_df['Laptop Compartment'].map({'No': 0, 'Yes': 1})
print(train_df.shape)


test_df['Waterproof'] = test_df['Waterproof'].map({'No': 0, 'Yes': 1})
test_df['Laptop Compartment'] = test_df['Laptop Compartment'].map({'No': 0, 'Yes': 1})
print(test_df.shape)


train_df = train_df.drop('id', axis=1)


print(train_df.columns)


import xgboost as xgb
from sklearn.model_selection import train_test_split


X=train_df.drop('Price', axis=1)
Y=train_df['Price']
final_test_X=test_df


categorical_columns = ['Brand', 'Material', 'Size', 'Style', 'Color']
for col in categorical_columns:
    X[col] = X[col].astype('category')


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


train_dmatrix = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
test_dmatrix = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)


import optuna
from sklearn.metrics import mean_squared_error
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "max_depth": trial.suggest_int("max_depth", 6, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2),
    }
    model=xgb.train(params, train_dmatrix, num_boost_round=200)
    preds=model.predict(test_dmatrix)
    return mean_squared_error(y_test, preds)
study=optuna.create_study(direction="minimize")
study.optimize(objective,n_trials=50,n_jobs=-1)


best_score = study.best_trial.value
best_params = study.best_trial.params


params = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "max_depth": best_params["max_depth"],
    "learning_rate": best_params["learning_rate"],
    "subsample": best_params["subsample"],
    "colsample_bytree": best_params["colsample_bytree"],
    "min_child_weight": best_params["min_child_weight"],
    "gamma": best_params["gamma"],
    "reg_alpha": best_params["reg_alpha"],
    "reg_lambda": best_params["reg_lambda"],
    "eval_metric": "rmse",
    "tree_method": "hist"
}


model=xgb.train(params, train_dmatrix, num_boost_round=200)


categorical_columns=['Brand', 'Material', 'Size', 'Style', 'Color']
for col in categorical_columns:
    test_df[col] = test_df[col].astype('category')
dtest=xgb.DMatrix(test_df.drop('id', axis=1), enable_categorical=True)
predictions = model.predict(dtest)


final_ans = pd.DataFrame({
    "id": test_df["id"],
    "Price": predictions
})
final_ans.to_csv("sample_submission.csv", index=False)

