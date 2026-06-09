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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error



train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
train.head()



train['Size'] = pd.to_numeric(train['Size'], errors='coerce')


if train['Size'].isnull().all():
    train.drop(columns=['Size'], inplace=True)  


train.fillna(method='ffill', inplace=True)
test.fillna(method='ffill', inplace=True)


print(train.isnull().sum())


categorical_cols = ['Brand', 'Material', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
data = pd.get_dummies(train, columns=categorical_cols, drop_first=True)



X = data.drop(columns=['id', 'Price'])
y = np.log1p(data['Price'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_grid = {
    'n_estimators': [500, 1000],
    'learning_rate': [0.05, 0.1],
    'max_depth': [6, 8],
    'random_state': [42]
}
xgb_model = XGBRegressor(enable_categorical=True)
grid_search = GridSearchCV(xgb_model, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=2)
grid_search.fit(X_train, y_train)


best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(X_test)
y_pred = np.expm1(y_pred)


rmse = np.sqrt(mean_squared_error(np.expm1(y_test), y_pred))
print(f"Optimized RMSE: {rmse:.2f}")


test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


test_data.drop(columns=['Size'], inplace=True, errors='ignore')
test_data.fillna(method='ffill', inplace=True)


test_data = pd.get_dummies(test_data, columns=categorical_cols, drop_first=True)



missing_cols = set(X.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0  # Add missing columns with default values
test_data = test_data[X.columns]  # Ensure correct column order



test_predictions = best_xgb.predict(test_data)
test_predictions = np.expm1(test_predictions)  # Reverse log transformation



if 'id' in test_data.columns:
    submission = pd.DataFrame({'id': test_data['id'], 'Price': test_predictions})
else:
    submission = pd.DataFrame({'id': range(300000, 300000 + len(test_data)), 'Price': test_predictions})

submission.to_csv("submission.csv", index=False)



