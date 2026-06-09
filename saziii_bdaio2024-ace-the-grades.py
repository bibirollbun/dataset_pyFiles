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
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load datasets
train_data = pd.read_csv('/kaggle/input/bdaio-2024-ace-the-grades/bdaio-2024-ace-the-grades/train.csv')
test_data = pd.read_csv('/kaggle/input/bdaio-2024-ace-the-grades/bdaio-2024-ace-the-grades/test.csv')

# Target variable
y = train_data['Test_Score']

# Basic features
features = [
    'Study_Hours', 'Sleep_Hours', 'Class_Attendance', 'Homework_Completed',
    'Participation', 'IQ', 'Internet_Usage_Hours'
]

# Feature engineering
def add_features(df):
    df['Study_per_Sleep'] = df['Study_Hours'] / (df['Sleep_Hours'] + 1)
    df['Homework_per_Internet'] = df['Homework_Completed'] / (df['Internet_Usage_Hours'] + 1)
    df['IQ_x_Participation'] = df['IQ'] * df['Participation']
    df['Study_x_Attendance'] = df['Study_Hours'] * df['Class_Attendance']
    return df

X = add_features(train_data[features].copy())
X_test = add_features(test_data[features].copy())

# Add new feature names
features += ['Study_per_Sleep', 'Homework_per_Internet', 'IQ_x_Participation', 'Study_x_Attendance']

# Validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Hyperparameter tuning (basic randomized search)
param_grid = {
    'n_estimators': [300, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

xgb_model = XGBRegressor(random_state=42, n_jobs=-1)

search = RandomizedSearchCV(
    xgb_model, param_distributions=param_grid,
    scoring='neg_root_mean_squared_error',  # you can change this
    n_iter=20, cv=3, verbose=1, n_jobs=-1
)
search.fit(X_train, y_train)

print("Best Parameters:", search.best_params_)

# Best model
best_model = search.best_estimator_

# Evaluate on validation set
val_preds = best_model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
print(f"Validation RMSE: {rmse:.4f}")

# Plot feature importance
plot_importance(best_model)
plt.tight_layout()
plt.show()

# Retrain on full data
full_X = pd.concat([X_train, X_val])
full_y = pd.concat([y_train, y_val])
best_model.fit(full_X, full_y)

# Predict on test data
test_preds = best_model.predict(X_test)

# Submission
submission = pd.DataFrame({
    'ID': test_data['ID'],
    'Predicted': test_preds
})
submission.to_csv('submission.csv', index=False)


