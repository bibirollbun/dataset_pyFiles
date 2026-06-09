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



## 1. Setup and Imports 
!pip install xgboost scikit-learn pandas numpy matplotlib seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
## 2. Data Loading 
train_df = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv")
test_df = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv")
sample_submission = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv")
train_df.head()
train_df.info()
train_df.describe()
## 3. Feature Engineering 
# Extract region code (last 2–3 digits)
train_df['region'] = train_df['plate'].str.extract(r"(\d{2,3})$")
# Extract letters and numbers separately
train_df['letters'] = train_df['plate'].str.findall(r"[A-Z]").str.join('')
train_df['numbers'] = train_df['plate'].str.extract(r"(\d{3})")

# Datetime features
train_df['date'] = pd.to_datetime(train_df['date'])
train_df['year']  = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day']   = train_df['date'].dt.day
train_df['hour']  = train_df['date'].dt.hour

# Palindrome flags
train_df['is_palindrome_letters'] = train_df['letters'].apply(lambda x: x == x[::-1])
train_df['is_palindrome_numbers'] = train_df['numbers'].apply(lambda x: x == x[::-1])

# Unique character counts
train_df['unique_letters_count'] = train_df['letters'].apply(lambda x: len(set(x)))
train_df['unique_numbers_count'] = train_df['numbers'].apply(lambda x: len(set(x)))

# Split numbers into individual digits and derive stats
digits = train_df['numbers'].apply(lambda s: list(map(int, s)))
train_df[['d1', 'd2', 'd3']] = pd.DataFrame(digits.tolist(), index=train_df.index)
train_df['sum_numbers']      = train_df['d1'] + train_df['d2'] + train_df['d3']
train_df['product_numbers']  = train_df['d1'] * train_df['d2'] * train_df['d3']

# Flag specific letter combos
for combo in ['BOP', 'XAM']:
    train_df[f'is_{combo.upper()}'] = (train_df['letters'] == combo)

# Convert numbers string to integer
train_df['numbers_int'] = train_df['numbers'].astype(int)
import sys
sys.path.append('/kaggle/input/russian-car-plates-prices-prediction')

from supplemental_russian import GOVERNMENT_CODES

def get_gov_tags(row):
    for (letters, num_range, region), (_, is_forbidden, has_advantage, level) in GOVERNMENT_CODES.items():
        if row['letters'] == letters and str(row['region']) == region:
            number = int(row['numbers'])
            if num_range[0] <= number <= num_range[1]:
                return pd.Series([is_forbidden, has_advantage, level])
    return pd.Series([0, 0, 0])
    
train_df[['is_forbidden', 'has_advantage', 'significance_level']] = train_df.apply(get_gov_tags, axis=1)
train_df
## 4. Train-Test Split
X = train_df.drop(columns=['price', 'id', 'plate', 'date'])
y = np.log1p(train_df['price'])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)
## 5. Baseline Feature Aggregations 
for col in ['letters', 'numbers', 'region']:
    agg = X_train.join(train_df[['price']]).groupby(col)['price'].mean()
    X_train[f'{col}_mean_price'] = X_train[col].map(agg)

    X_val = X_val.merge(
        X_train[[col, f'{col}_mean_price']].drop_duplicates(),
        on=col, how='left'
    )

    X_val[f'{col}_mean_price'] = X_val[f'{col}_mean_price'] \
        .fillna(X_train[f'{col}_mean_price'].mean())

X_train
## 6. Model Training with GridSearchCV 
X_train.columns
features = ['year', 'month', 'day', 'hour',
       'is_palindrome_letters', 'is_palindrome_numbers',
       'unique_letters_count', 'unique_numbers_count', 'd1', 'd2', 'd3',
       'sum_numbers', 'product_numbers', 'is_BOP', 'is_XAM', 'numbers_int',
       'is_forbidden', 'has_advantage', 'significance_level',
       'letters_mean_price', 'numbers_mean_price', 'region_mean_price']
# SMAPE definition

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff  = np.abs(y_true - y_pred)
    return np.mean(diff / denom) * 100

# Scorer (note: greater_is_better=False)
smape_scorer = make_scorer(lambda yt, yp: smape(np.expm1(yt), np.expm1(yp)), greater_is_better=False)

param_grid = {
    'n_estimators': [50, 100, 500],
    'max_depth':    [3, 5, 10],
    'learning_rate':[0.05, 0.1, 0.2]
}

model = XGBRegressor(random_state=42)
grid  = GridSearchCV(model, param_grid, scoring=smape_scorer, cv=3, verbose=1, n_jobs=-1)
grid.fit(X_train[features], y_train)

print("Best SMAPE (inverted):", -grid.best_score_)
print("Best params:", grid.best_params_)

# Retrain final model
best_model = XGBRegressor(**grid.best_params_, random_state=42)
best_model.fit(X_train[features], y_train)
## 7. Evaluation 
# Predictions and inverse transform
y_val_pred_log = best_model.predict(X_val[features])
y_val_pred     = np.expm1(y_val_pred_log)
y_true         = np.expm1(y_val)

mae   = mean_absolute_error(y_true, y_val_pred)
rmse  = np.sqrt(mean_squared_error(y_true, y_val_pred))
smape_score = smape(y_true, y_val_pred)

print(f"SMAPE: {smape_score:.2f}%")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Feature importance
importances = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
importances.head(10)
## 8. Submission 
test_df = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv")
test_df.head()
# Extract region code (last 2–3 digits)
test_df['region'] = test_df['plate'].str.extract(r"(\d{2,3})$")
# Extract letters and numbers separately
test_df['letters'] = test_df['plate'].str.findall(r"[A-Z]").str.join('')
test_df['numbers'] = test_df['plate'].str.extract(r"(\d{3})")

# Datetime features
test_df['date'] = pd.to_datetime(test_df['date'])
test_df['year']  = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day']   = test_df['date'].dt.day
test_df['hour']  = test_df['date'].dt.hour

# Palindrome flags
test_df['is_palindrome_letters'] = test_df['letters'].apply(lambda x: x == x[::-1])
test_df['is_palindrome_numbers'] = test_df['numbers'].apply(lambda x: x == x[::-1])

# Unique character counts
test_df['unique_letters_count'] = test_df['letters'].apply(lambda x: len(set(x)))
test_df['unique_numbers_count'] = test_df['numbers'].apply(lambda x: len(set(x)))

# Split numbers into individual digits and derive stats
digits = test_df['numbers'].apply(lambda s: list(map(int, s)))
test_df[['d1', 'd2', 'd3']] = pd.DataFrame(digits.tolist(), index=test_df.index)
test_df['sum_numbers']      = test_df['d1'] + test_df['d2'] + test_df['d3']
test_df['product_numbers']  = test_df['d1'] * test_df['d2'] * test_df['d3']

# Flag specific letter combos
for combo in ['BOP', 'XAM']:
    test_df[f'is_{combo.upper()}'] = (test_df['letters'] == combo)

# Convert numbers string to integer
test_df['numbers_int'] = test_df['numbers'].astype(int)

test_df[['is_forbidden', 'has_advantage', 'significance_level']] = test_df.apply(get_gov_tags, axis=1)

for col in ['letters', 'numbers', 'region']:
    test_df = test_df.merge(
        X_train[[col, f'{col}_mean_price']].drop_duplicates(),
        on=col, how='left'
    )

    test_df[f'{col}_mean_price'] = test_df[f'{col}_mean_price'] \
        .fillna(X_train[f'{col}_mean_price'].mean())
# Predict and save
test_X = test_df[features]
test_pred_log = best_model.predict(test_X)
test_pred     = np.expm1(test_pred_log).astype(int)

submission = pd.DataFrame({
    'id': test_df['id'],
    'price': test_pred
})
submission.to_csv('submission.csv', index=False)


