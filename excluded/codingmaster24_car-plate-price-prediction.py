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


train = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv")
test = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv")
sample_submission = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv")


train.head()


test.head()


sample_submission.head()


import pandas as pd
import numpy as np
import re
import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from lightgbm import early_stopping as lgb_early_stopping
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping
from lightgbm.callback import log_evaluation


# Feature Engineering

def extract_plate_features(plate):
    letters = re.findall(r"[A-Za-zА-Яа-я]+", plate)  # Extract letters
    numbers = re.findall(r"\d+", plate)  # Extract numbers
    letter_part = "".join(letters) if letters else ""  # Combine letters
    number_part = int("".join(numbers)) if numbers else 0  # Combine numbers
    return letter_part, number_part

train[['plate_letters', 'plate_numbers']] = train['plate'].apply(lambda x: pd.Series(extract_plate_features(str(x))))
test[['plate_letters', 'plate_numbers']] = test['plate'].apply(lambda x: pd.Series(extract_plate_features(str(x))))


# Convert letters to categorical numerical values
le = LabelEncoder()
train['plate_letters'] = le.fit_transform(train['plate_letters'])
test['plate_letters'] = le.transform(test['plate_letters'])


# Convert date to numerical features
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])
train['year'] = train['date'].dt.year
test['year'] = test['date'].dt.year
train['month'] = train['date'].dt.month
test['month'] = test['date'].dt.month
train['day'] = train['date'].dt.day
test['day'] = test['date'].dt.day


# Drop unnecessary columns
train.drop(columns=['id', 'plate', 'date'], inplace=True)
test.drop(columns=['id', 'plate', 'date'], inplace=True)


# Ensure test has same features as train
missing_cols = set(train.columns) - set(test.columns)
for col in missing_cols:
    test[col] = 0  # Fill missing columns with default values

test = test[train.drop(columns=['price']).columns]  # Ensure column order matches train


# Splitting data
X = train.drop(columns=['price'])
y = train['price']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Model Training
models = {
    "CatBoost": CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.05, verbose=200),
    "LightGBM": LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31),
    "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6)
}

predictions = {}

for name, model in models.items():
    print(f"Training {name}...")
    if name == "LightGBM":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="rmse", callbacks=[lgb_early_stopping(100), log_evaluation(200)])
    else:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=200)
    predictions[name] = model.predict(test)



# Ensemble: Averaging Model Predictions
final_prediction = (predictions['CatBoost'] + predictions['LightGBM'] + predictions['XGBoost']) / 3


# Submission
submission = sample_submission.copy()
submission['price'] = final_prediction
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")


submission




