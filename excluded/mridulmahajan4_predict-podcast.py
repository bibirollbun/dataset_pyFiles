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


# ğŸ“š Import libraries
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ğŸ”¥ Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

# ğŸ§¹ Drop 'id' column
train = train.drop(['id'], axis=1)
test = test.drop(['id'], axis=1)

# ğŸ�·ï¸� Handle categorical features
categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

for col in categorical_cols:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

# ğŸ�¯ Separate features and target
X = train.drop('Listening_Time_minutes', axis=1)
y = train['Listening_Time_minutes']
X_test = test.copy()

# âš™ï¸� Handle missing values correctly
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Add 'Unknown' to categories
for col in categorical_cols:
    X[col] = X[col].cat.add_categories('Unknown')
    X_test[col] = X_test[col].cat.add_categories('Unknown')

# Now safely fill missing
X[categorical_cols] = X[categorical_cols].fillna('Unknown')
X[numerical_cols] = X[numerical_cols].fillna(0)

X_test[categorical_cols] = X_test[categorical_cols].fillna('Unknown')
X_test[numerical_cols] = X_test[numerical_cols].fillna(0)


# ğŸ”¥ Split into training and validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# ğŸ”¥ LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_cols)

# âš™ï¸� LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': -1,
    'seed': 42,
    'verbose': -1
}

# ğŸš€ Train LightGBM model
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, valid_data],
    num_boost_round=10000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100)
    ]
)


# ğŸ�¯ Predict on test set
predictions = model.predict(X_test)

# ğŸ“� Prepare submission
sample_submission['Listening_Time_minutes'] = predictions
sample_submission.to_csv('submission.csv', index=False)

print("âœ… Submission file created successfully! Ready to submit ğŸš€")





