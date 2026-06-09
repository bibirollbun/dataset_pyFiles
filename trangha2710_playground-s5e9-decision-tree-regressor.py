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
import seaborn as sns

train_set = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
train_set.info()


train_set['id'] = train_set['id'].astype(str)
train_set.head(10)


train_set.isnull().sum()


train_set.describe()


from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error


X = train_set.drop(['id','BeatsPerMinute'],axis = 1)
y = train_set['BeatsPerMinute']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


base_tree = DecisionTreeRegressor(random_state=42)
param_grid = {
    'max_depth': [3, 4, 5, 6, 8, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': [None, 'sqrt', 'log2']
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV
gs = GridSearchCV(
    estimator=base_tree,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  
    cv=cv,
    n_jobs=-1,
    verbose=1
)


gs.fit(X_train, y_train)
best_tree = gs.best_estimator_

y_pred = best_tree.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (mse)**(1/2)
print(f"Root Mean Squared Error: {rmse:.4f}")


test_set = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
test_set['id'] = test_set['id'].astype(str)
X_test_final = test_set.drop('id', axis = 1)
predictions = best_tree.predict(X_test_final)


#extract submission.csv
submission = pd.DataFrame({
    "id": test_set["id"],
    "BeatsPerMinute": predictions
})
submission.to_csv("submission.csv", index=False)

