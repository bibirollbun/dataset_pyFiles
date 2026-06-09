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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")  

train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

num_cols = ['Compartments', 'Weight Capacity (kg)']
train[num_cols] = train[num_cols].fillna(train[num_cols].median())
test[num_cols] = test[num_cols].fillna(train[num_cols].median())

cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
train[cat_cols] = train[cat_cols].fillna('Unknown')
test[cat_cols] = test[cat_cols].fillna('Unknown')

train['is_large_capacity'] = (train['Weight Capacity (kg)'] > 20).astype(int)
test['is_large_capacity'] = (test['Weight Capacity (kg)'] > 20).astype(int)

color_counts = train['Color'].value_counts().to_dict()
train['color_freq'] = train['Color'].map(color_counts)
test['color_freq'] = test['Color'].map(color_counts).fillna(0)

target_col = 'Price'
X = train.drop(columns=['id', target_col])
y = train[target_col]
test_data = test.drop(columns=['id'])

X = pd.get_dummies(X)
test_data = pd.get_dummies(test_data)
X, test_data = X.align(test_data, join='left', axis=1, fill_value=0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [6, 10],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

xgb = XGBRegressor(random_state=42, n_jobs=-1)

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=5, 
    verbose=1
)

grid.fit(X_train, y_train)

print("die besten Parameter gefunden:")
print(grid.best_params_)


best_model = grid.best_estimator_
val_preds = best_model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
print("RMSE auf validation set (mit optimisierte Modell):", rmse)


final_preds = best_model.predict(test_data)

submission = pd.DataFrame({
    'id': test['id'],
    'Price': final_preds
})
submission.to_csv('submission.csv', index=False)


