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


import kagglehub

# Download later version
path = kagglehub.dataset_download("mdmub0587/older-dataset-for-dont-overfit-ii-challenge")

print("Path to dataset files:", path)


#try grid search this case
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

train_data = pd.read_csv('/kaggle/input/older-dataset-for-dont-overfit-ii-challenge/train.csv')
test_data = pd.read_csv('/kaggle/input/older-dataset-for-dont-overfit-ii-challenge/test.csv')
x_train = train_data.drop (columns=['id', 'target'])
y_train = train_data['target']
x_test = test_data.drop(columns=['id'])


# scaling the data
scaler = StandardScaler() 
x_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)


# Grid search method
param_grid = {
    'C': [0.001, 0.01, 0.05, 0.1, 0.2],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
model = LogisticRegression(random_state=42, max_iter=2000)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42), 
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(x_scaled, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")


# use the best model to predict

best_model = grid_search.best_estimator_
test_predictions = best_model.predict_proba(x_test_scaled)[:, 1]

submission = pd.DataFrame({
    'id': test_data['id'],
    'target': test_predictions
})

submission.to_csv('grid_search_submission.csv', index=False)

