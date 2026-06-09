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


import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler


# Load data
df_train = pd.read_csv("/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/train.csv")
df_test = pd.read_csv("/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/test.csv")

# Separate features and target
X = df_train.drop(columns=["target"])
y = df_train["target"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Define Gradient Boosting and hyperparameter grid
model = GradientBoostingRegressor(random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
}

# KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(model, param_grid, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
grid.fit(X_scaled, y)


# Best model and results
best_model = grid.best_estimator_
print(f"Best params: {grid.best_params_}")
print(f"Best cross-validated MSE: {-grid.best_score_}")

# Prepare test data and make predictions
X_test = df_test.drop(columns="id")
X_test_scaled = scaler.transform(X_test)
df_test["target"] = best_model.predict(X_test_scaled)

# Save predictions
df_test[["id", "target"]].to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")

