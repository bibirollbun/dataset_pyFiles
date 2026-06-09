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
from sklearn.ensemble import RandomForestRegressor

# 1. Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

# 2. Target column
target = 'Price'

# 3. Separate features and target
X = train.drop(columns=['id', target])
y = train[target]
X_test = test.drop(columns=['id'])

# 4. Identify numeric and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# 5. Fill missing numeric values with median
for col in num_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# 6. Fill missing categorical values with 'Unknown'
for col in cat_cols:
    X[col] = X[col].fillna('Unknown')
    X_test[col] = X_test[col].fillna('Unknown')

# 7. One-hot encode categorical variables
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

# 8. Align columns to have the same features in train and test
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# 9. Train model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# 10. Predict on test set
preds = model.predict(X_test)

# 11. Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'Price': preds
})
submission.to_csv('submission.csv', index=False)

print("Baseline submission created successfully!")


