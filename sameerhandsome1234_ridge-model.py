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


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col='id')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge


categorical_cols = train.select_dtypes(include="object").columns
numerical_cols = train.select_dtypes(include=["int64", "float64"]).columns

print("Categorical Columns:", list(categorical_cols))
print("Numerical Columns:", list(numerical_cols))


categorical_col = train.select_dtypes('object').columns.tolist()
for col in categorical_col:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


bool_col = train.select_dtypes('bool').columns.tolist()
train[bool_col] = train[bool_col].astype(int)
test[bool_col] = test[bool_col].astype(int)


X = train.drop(columns=['accident_risk'])
y = train['accident_risk']

X_test = test.copy()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def objective(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 100.0, log=True)
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
    solver = trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr", "sag", "saga"])
    
    model = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver=solver, random_state=42)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    
    for train_idx, val_idx in kf.split(X_scaled):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_scores.append(rmse)
    
    return np.mean(rmse_scores)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("\nâœ… Best Parameters:", study.best_params)
print("ğŸ�† Best RMSE:", study.best_value)


best_model = Ridge(**study.best_params, random_state=42)
best_model.fit(X_scaled, y)

import pickle
with open("ridge_best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nğŸ�¯ Model saved as 'ridge_best_model.pkl'")

