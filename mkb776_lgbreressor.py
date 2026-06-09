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
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt



train_df = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
train_df.head(2)


test_df = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')
test_df.head(2)


# Drop unnecessary column (if exists)
train_df.drop(columns=['sale_nbr'], inplace=True, errors='ignore')
test_df.drop(columns=['sale_nbr'], inplace=True, errors='ignore')

# Fill missing categoricals
for col in ['subdivision', 'submarket']:
    train_df[col] = train_df[col].fillna("Missing")
    test_df[col] = test_df[col].fillna("Missing")

# Convert sale_date to datetime format
train_df['sale_date'] = pd.to_datetime(train_df['sale_date'], errors='coerce')
test_df['sale_date'] = pd.to_datetime(test_df['sale_date'], errors='coerce')

# Optional: Drop rows where sale_date couldn't be parsed
train_df.dropna(subset=['sale_date'], inplace=True)
test_df.dropna(subset=['sale_date'], inplace=True)

# Feature engineering: extract date parts
train_df['sale_year'] = train_df['sale_date'].dt.year
train_df['sale_month'] = train_df['sale_date'].dt.month
test_df['sale_year'] = test_df['sale_date'].dt.year
test_df['sale_month'] = test_df['sale_date'].dt.month

# Drop original date column
train_df.drop(columns=['sale_date'], inplace=True)
test_df.drop(columns=['sale_date'], inplace=True)

# Log-transform target
train_df['log_price'] = np.log1p(train_df['sale_price'])

# Separate features and target
X = train_df.drop(columns=['id', 'sale_price', 'log_price'])
y = train_df['log_price']
X_test = test_df.drop(columns=['id'])


# Encode categoricals safely
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy='mean'), num_cols),
    ("cat", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
])

X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)

test_ids = test_df['id']


# Best tuned parameters
alpha_lower = 0.03
alpha_upper = 0.97

common_params = {
    'objective': 'quantile',
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'num_leaves': 64,
    'min_child_samples': 20,
    'reg_alpha': 2.0,
    'reg_lambda': 2.0,
    'random_state': 42
}

# Train lower and upper models
model_lower = LGBMRegressor(**common_params, alpha=alpha_lower)
model_upper = LGBMRegressor(**common_params, alpha=alpha_upper)

model_lower.fit(X_processed, y)
model_upper.fit(X_processed, y)


# Predict in log space
y_lower_log = model_lower.predict(X_test_processed)
y_upper_log = model_upper.predict(X_test_processed)

# Inverse log
y_lower = np.expm1(y_lower_log)
y_upper = np.expm1(y_upper_log)

# Fix edge cases where lower > upper
y_lower = np.minimum(y_lower, y_upper)
y_upper = np.maximum(y_lower, y_upper)

# Final submission format
submission_df = pd.DataFrame({
    'id': test_ids,
    'pi_lower': y_lower,
    'pi_upper': y_upper
})

# Save submission to Kaggle-recognized output folder
submission_df.to_csv("/kaggle/working/final_submission.csv", index=False)
print("✅ Submission saved to /kaggle/working/final_submission.csv")
print(os.listdir("/kaggle/working"))




