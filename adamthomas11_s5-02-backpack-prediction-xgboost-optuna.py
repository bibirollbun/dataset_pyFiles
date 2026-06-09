# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import make_scorer

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
extra_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
sub_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

df = pd.concat([train_df, extra_df], ignore_index=True)


def preprocess_baseline(data):
    for i in data.columns:
        if (i != 'Price') & (i != 'Weight Capacity (kg)') & (i != 'Compartments') & (i != 'id'):
            data[i] = data[i].fillna("unknown")
    
    # Select categorical columns for get_dummies
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=False)
    
    return data

df = preprocess_baseline(df)
df = df.astype({col: 'int' for col in df.select_dtypes(include='bool').columns})

df.head()


train_df, temp_df = train_test_split(df, test_size=0.2, random_state=0)  
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=0) 

# Print dataset sizes
print(f"Train set: {len(train_df)} rows")
print(f"Validation set: {len(val_df)} rows")
print(f"Test set: {len(test_df)} rows")



X = df.drop(columns=['id', 'Price']) 
y = df['Price'] 


pip install optuna-integration[xgboost]


import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, make_scorer

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# RMSE scorer
def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Optuna objective function
def objective(trial):
    # Define model parameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'early_stopping_rounds': 25
    }

    # Instantiate the model
    model = xgb.XGBRegressor(**params, random_state=0)

    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    
    scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = model.predict(X_val)
        scores.append(rmse_scorer(y_val, y_pred))

    return np.mean(scores)

# Run Optuna optimization
#study = optuna.create_study(direction='minimize')
#study.optimize(objective, n_trials=100)

# Best parameters from Optuna
#best_params = study.best_params
#print("Best parameters:", best_params)

best_params = {'n_estimators': 791, 
               'learning_rate': 0.017532081941933595, 
               'max_depth': 6, 
               'min_child_weight': 9, 
               'subsample': 0.601017883750796, 
               'colsample_bytree': 0.8352989442660304, 
               'gamma': 0.48629059342560066, 
               'reg_alpha': 1.0788797333993974, 
               'reg_lambda': 0.9600302448385072}

# Train the final model on the full training data
final_model = xgb.XGBRegressor(**best_params, random_state=0)
final_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

# Evaluate on the test set
y_pred = final_model.predict(X_test)
test_rmse = rmse_scorer(y_test, y_pred)

print("Final Test RMSE:", test_rmse)



sub_df.head()


ids = sub_df['id']
sub_df = preprocess_baseline(sub_df)
sub_df = sub_df.drop(columns=['id'])
sub_df = sub_df.astype({col: 'int' for col in sub_df.select_dtypes(include='bool').columns})

pred = final_model.predict(sub_df)

submission = pd.DataFrame({'id': ids, 'Price': pred})
print(submission.head())

submission.to_csv('submission.csv', index=False)



