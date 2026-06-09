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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error



train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')



train['Compartments_per_kg'] = train['Compartments'] / train['Weight Capacity (kg)']
test['Compartments_per_kg'] = test['Compartments'] / test['Weight Capacity (kg)']



X = train.drop(columns=['id', 'Price'])
y = train['Price']
X_test = test.drop(columns=['id'])

categorical_cols = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

X = X.fillna(X.mean(numeric_only=True))
X_test = X_test.fillna(X_test.mean(numeric_only=True))



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



param_grid = {
    'n_estimators': [100],
    'max_depth': [None, 10],
    'min_samples_split': [2]
}


grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=2, scoring='neg_root_mean_squared_error', verbose=1)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)



best_model = grid.best_estimator_

y_pred = best_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("Improved RMSE:", rmse)



preds = best_model.predict(X_test)
submission['Price'] = preds
submission.to_csv('submission.csv', index=False)
submission.head()


