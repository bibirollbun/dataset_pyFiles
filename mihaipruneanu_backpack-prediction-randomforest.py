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

PATH = '/kaggle/input/playground-series-s5e2/'
train_main  = pd.read_csv(PATH + 'train.csv')
train_extra = pd.read_csv(PATH + 'training_extra.csv')
test_df     = pd.read_csv(PATH + 'test.csv')

train_df = pd.concat([train_main, train_extra], ignore_index=True)


#drop ID column
train_df = train_df.drop('id', axis=1)
test_ids = test_df['id']  
test_df = test_df.drop('id', axis=1)

#separate target
y = train_df['Price']
X = train_df.drop('Price', axis=1)

#handle missing values(fill categorical with "Unknown", numeric with median)
X = X.fillna(value={col: "Unknown" for col in X.select_dtypes('object').columns})
X = X.fillna(value={col: X[col].median() for col in X.select_dtypes('number').columns})

test_df = test_df.fillna(value={col: "Unknown" for col in test_df.select_dtypes('object').columns})
test_df = test_df.fillna(value={col: test_df[col].median() for col in test_df.select_dtypes('number').columns})

#one-hot encode categorical variables
X = pd.get_dummies(X)
test_df = pd.get_dummies(test_df)
# define the test matrix and ID vector



X, test_df = X.align(test_df, join='left', axis=1, fill_value=0)



print("Train shape:", X.shape)
print("Test shape:", test_df.shape)



print("Missing in train:", X.isnull().sum().sum())
print("Missing in test:", test_df.isnull().sum().sum())



print("Train columns:", X.columns.tolist()[:10])  



print(y.describe())



X.head()



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)



from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

y_pred = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f"RMSE: {rmse:.2f}")



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_estimators': [50,100],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [1.0, 'sqrt']
}

rf = RandomForestRegressor(random_state=42)

#randomized search
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=5,  
    cv=2,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

#evaluate
y_pred = best_model.predict(X_val)

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))


print(f"Best RMSE: {rmse:.2f}")
print("Best parameters:", random_search.best_params_)



import pandas as pd


preds = model.predict(test_df)

submission = pd.DataFrame({
    'id':    test_ids,
    'Price': preds
})

submission.to_csv('submission_rf.csv', index=False)
print("submission_rf.csv", submission.shape)




