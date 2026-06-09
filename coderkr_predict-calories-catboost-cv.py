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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import optuna
import warnings
warnings.filterwarnings('ignore')

# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

# Check target
TARGET = 'Calories'

# Drop ID column, store for later use
train_ids = train['id']
test_ids = test['id']
train = train.drop(columns=['id'])
test = test.drop(columns=['id'])

# Separate features and target
X = train.drop(columns=[TARGET])
y = train[TARGET]
X_test = test.copy()



# CatBoost handles categorical features internally, so just mark it
categorical_features = ['Sex']

# Convert to string for CatBoost if not already
X['Sex'] = X['Sex'].astype(str)
X_test['Sex'] = X_test['Sex'].astype(str)



def add_features(df):
    df = df.copy()
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['HR_per_duration'] = df['Heart_Rate'] / df['Duration']
    df['Temp_HR_ratio'] = df['Body_Temp'] / df['Heart_Rate']
    df['Work'] = df['Duration'] * df['Heart_Rate']
    return df

X = add_features(X)
X_test = add_features(X_test)



def objective(trial):
    params = {
        'iterations': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
        'loss_function': 'RMSE',
        'task_type': 'CPU',
        'verbose': 0,
        'early_stopping_rounds': 50
    }

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    # Bin the target for stratification
    bins = pd.qcut(y, q=10, labels=False)

    for train_idx, valid_idx in kf.split(X, bins):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=(X_valid, y_valid),
                  cat_features=categorical_features)

        preds = model.predict(X_valid)
        score = mean_squared_error(y_valid, preds, squared=False)
        scores.append(score)

    return np.mean(scores)

# Uncomment to run tuning (takes time)
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=30)
# best_params = study.best_params



params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3.0,
    'random_strength': 0.5,
    'loss_function': 'RMSE',
    'task_type': 'CPU',
    'verbose': 0,
    'early_stopping_rounds': 50
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

# Bin the target for stratified regression
bins = pd.qcut(y, q=10, labels=False)

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, bins)):
    print(f"Fold {fold+1}")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=(X_valid, y_valid),
              cat_features=categorical_features)

    oof_preds[valid_idx] = model.predict(X_valid)
    test_preds += model.predict(X_test) / kf.n_splits

rmse = mean_squared_error(y, oof_preds, squared=False)
print(f"OOF RMSE: {rmse:.4f}")



submission['Calories'] = test_preds
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")



import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Load Data
# -------------------------------
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

# -------------------------------
# Basic Prep
# -------------------------------
TARGET = 'Calories'
categorical_features = ['Sex']

train_ids = train['id']
test_ids = test['id']
X = train.drop(columns=['id', TARGET])
y = train[TARGET]
X_test = test.drop(columns=['id'])

X['Sex'] = X['Sex'].astype(str)
X_test['Sex'] = X_test['Sex'].astype(str)

# -------------------------------
# Feature Engineering
# -------------------------------
def add_features(df):
    df = df.copy()
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['HR_per_duration'] = df['Heart_Rate'] / df['Duration']
    df['Temp_HR_ratio'] = df['Body_Temp'] / df['Heart_Rate']
    df['Work'] = df['Duration'] * df['Heart_Rate']
    df['WH_ratio'] = df['Weight'] / df['Height']
    df['Age_Duration'] = df['Age'] * df['Duration']
    df['HR_temp_sum'] = df['Heart_Rate'] + df['Body_Temp']
    return df

X = add_features(X)
X_test = add_features(X_test)

# -------------------------------
# Stratified CV Setup
# -------------------------------
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
bins = pd.qcut(y, q=10, labels=False)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

# -------------------------------
# CatBoost Parameters (Optimized)
# -------------------------------
params = {
    'iterations': 2000,
    'learning_rate': 0.025,
    'depth': 7,
    'l2_leaf_reg': 5,
    'random_strength': 0.8,
    'bagging_temperature': 0.2,
    'loss_function': 'RMSE',
    'early_stopping_rounds': 100,
    'verbose': 0,
    'task_type': 'CPU'
}

# -------------------------------
# Training with CV
# -------------------------------
for fold, (train_idx, valid_idx) in enumerate(kf.split(X, bins)):
    print(f"Fold {fold+1}")
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=(X_valid, y_valid),
              cat_features=categorical_features)

    oof_preds[valid_idx] = model.predict(X_valid)
    test_preds += model.predict(X_test) / kf.n_splits

rmse = mean_squared_error(y, oof_preds, squared=False)
print(f"\n✅ Final CV RMSE: {rmse:.5f}")

# -------------------------------
# Submission
# -------------------------------
submission['Calories'] = test_preds
submission.to_csv('submission.csv', index=False)
print("Submission saved as 'submission.csv'")


