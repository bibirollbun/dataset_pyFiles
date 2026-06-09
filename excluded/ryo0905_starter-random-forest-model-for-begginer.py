# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

train.head()


test.head()


sample_submission.head()


train.info()


test.info()


train.describe(include='all')


# Dummy variables for categorical variables
train_dummies = pd.get_dummies(train)
test_dummies = pd.get_dummies(test)

# Split data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(train_dummies.drop('Calories', axis=1), train_dummies['Calories'], test_size=0.3, random_state=42)

X_train.head()


# Create model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on validation set
y_pred = model.predict(X_val)

# Calculate RMSE
rmse = mean_squared_error(y_val, y_pred, squared=False)

rmse



y_test = model.predict(test_dummies)

submission = pd.DataFrame({'id': test_dummies['id'], 'Calories': y_test})
submission.to_csv('submission.csv', index=False)

submission

