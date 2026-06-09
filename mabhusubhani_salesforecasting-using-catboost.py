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
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error


## Loading Data
# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


# Fill Missing Values
train['num_sold'].fillna(train['num_sold'].median(), inplace=True)


# Feature Engineering
def date_feature_engineering(df):
    df['date'] = pd.to_datetime(df['date'])
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    df['DayOfWeek'] = df['date'].dt.dayofweek
    df['WeekOfYear'] = df['date'].dt.isocalendar().week
    return df



train = date_feature_engineering(train)
test = date_feature_engineering(test)


# Drop the original date column
train.drop('date', axis=1, inplace=True)
test.drop('date', axis=1, inplace=True)



# Define Categorical Columns
categorical_cols = ['country', 'store', 'product']


# Prepare Features and Target
X = train.drop(['id', 'num_sold'], axis=1)
y = train['num_sold']
X_test = test.drop('id', axis=1)


# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# Initialize CatBoost Regressor
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=8,
    cat_features=categorical_cols,
    loss_function='MAPE',
    eval_metric='MAPE',
    verbose=100,
    random_seed=42
)


# Train the Model
model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)



# Evaluate the Model
y_pred = model.predict(X_val)
mape = mean_absolute_percentage_error(y_val, y_pred)
print(f"Validation MAPE: {mape:.5f}")


# Predict on Test Data
test_predictions = model.predict(X_test)


# Create Submission File
submission = pd.DataFrame({'id': test['id'], 'num_sold': test_predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file created!")


submission.head()



















