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

train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
train


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor

y = train_data["Price"]

features = ["Brand", "Material", "Size", "Compartments", "Laptop Compartment", "Waterproof", "Style", "Weight Capacity (kg)"]

X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

X = X.fillna(0)
X_test = X_test.fillna(0)

model = HistGradientBoostingRegressor(early_stopping=True, random_state=42)
params = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_iter': [100, 200],
    'max_depth': [None, 10, 20],
    'l2_regularization': [0.0, 1.0, 10.0]        
}

grid_obj = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=3,            
    n_jobs=-1,       
    scoring='neg_root_mean_squared_error',
    verbose=2
)
grid_obj = grid_obj.fit(X, y)
model = grid_obj.best_estimator_
print("Best parameters found:", grid_obj.best_params_)
model.fit(X, y)
predictions = model.predict(X_test)
# 'l2_regularization': 0.0, 'learning_rate': 0.05, 'max_depth': 10, 'max_iter': 100
output = pd.DataFrame({'id': test_data.id, 'Price': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

