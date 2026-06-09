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
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load the dataset
train_df = pd.read_csv("/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/train.csv")  # Replace with your training dataset path
test_df = pd.read_csv("/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/test.csv")  # Replace with your test dataset path

# Separate features and target
X = train_df.drop(columns=["target"])
y = train_df["target"]
X_test = test_df.drop(columns=["id"])

# Add min and max columns row-wise
X['row_min'] = X.min(axis=1)
X['row_max'] = X.max(axis=1)
X_test['row_min'] = X_test.min(axis=1)
X_test['row_max'] = X_test.max(axis=1)

# Define the hyperparameter space
space = {
    'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400]),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'max_depth': hp.choice('max_depth', [3, 6, 9, 12]),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'gamma': hp.uniform('gamma', 0, 5)
}

# Define the objective function
def objective(params):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = XGBRegressor(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            gamma=params['gamma'],
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse_scores.append(mean_squared_error(y_val, y_pred))

    avg_mse = np.mean(mse_scores)
    return {'loss': avg_mse, 'status': STATUS_OK}

# Run hyperparameter optimization
trials = Trials()
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

# Train the best model
best_model = XGBRegressor(
    n_estimators=[100, 200, 300, 400][best_params['n_estimators']],
    learning_rate=best_params['learning_rate'],
    max_depth=[3, 6, 9, 12][best_params['max_depth']],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    gamma=best_params['gamma'],
    random_state=42
)
best_model.fit(X, y)

# Make predictions on the test dataset
predictions = best_model.predict(X_test)

# Save predictions to a CSV file
output_df = pd.DataFrame({
    "id": test_df["id"],  # Replace "id" with your test dataset ID column
    "target": predictions
})
output_df.to_csv("predictions3.csv", index=False)

print("Best hyperparameters:", best_params)
print("Predictions saved to predictions3.csv")





