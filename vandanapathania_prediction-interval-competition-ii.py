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
import lightgbm as lgb

# 1. Load Data
train = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
test = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')
sample_submission = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/sample_submission.csv')

# Step 3: Select features
target = 'sale_price'
id_col = 'id'
features = [col for col in train.columns if col in test.columns and col != id_col]
X = train[features].copy()
y = train[target].copy()
X_test = test[features].copy()

# Step 4: Fill missing values
for col in X.columns:
    if X[col].dtype in ['float64', 'int64']:
        median = X[col].median()
        X[col] = X[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)
    else:
        X[col] = X[col].fillna('missing')
        X_test[col] = X_test[col].fillna('missing')

# Step 5: One-hot encode categorical columns
cat_cols = X.select_dtypes(include=['object']).columns
for c in cat_cols:
  X[c]      = X[c].astype('category')
  X_test[c] = X_test[c].astype('category')

# Step 6: Train LightGBM quantile regression models
# Lower quantile (5%)
model_lower = lgb.LGBMRegressor(objective='quantile', alpha=0.05, n_estimators=200, random_state=42)
model_lower.fit(X, y, categorical_feature=list(cat_cols))
pi_lower = model_lower.predict(X_test)

# Upper quantile (95%)
model_upper = lgb.LGBMRegressor(objective='quantile', alpha=0.95, n_estimators=200, random_state=42)
model_upper.fit(X, y, categorical_feature=list(cat_cols))
pi_upper = model_upper.predict(X_test)

# Step 7: Prepare submission file
submission = pd.DataFrame({
    'id': test[id_col],
    'pi_lower': pi_lower,
    'pi_upper': pi_upper
})
submission.to_csv('submission.csv', index=False, float_format='%.4f')
print('Submission saved to submission.csv')




