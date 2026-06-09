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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error
from lightgbm.callback import early_stopping

# Load the datasets
train_path = '/kaggle/input/playground-series-s5e1/train.csv'  # Replace with your actual file path
test_path = '/kaggle/input/playground-series-s5e1/test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Convert 'date' to datetime format
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

# Fill missing values in 'num_sold'
train_df['num_sold'] = train_df['num_sold'].fillna(train_df['num_sold'].median())

# Extract date-related features
for df in [train_df, test_df]:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday

# Encode categorical features
label_encoders = {}
for col in ['country', 'store', 'product']:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

# Define features and target
features = ['country', 'store', 'product', 'year', 'month', 'day', 'weekday']
target = 'num_sold'

# Split the training data for validation
X_train, X_val, y_train, y_val = train_test_split(
    train_df[features], train_df[target], test_size=0.2, random_state=42
)

# Train a LightGBM model
lgb_model = LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42)
callbacks = [early_stopping(stopping_rounds=50, verbose=True)]
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="mape", callbacks=callbacks)

# Evaluate the model on validation data
y_val_pred = lgb_model.predict(X_val)
mape = mean_absolute_percentage_error(y_val, y_val_pred)
print(f"Validation MAPE (LightGBM): {mape:.4f}")

# Make predictions on the test set
test_df['num_sold'] = lgb_model.predict(test_df[features])

# Prepare the submission file
submission_df = test_df[['id', 'num_sold']]
submission_df.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")


