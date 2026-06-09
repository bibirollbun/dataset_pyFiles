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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


# Load training data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')

# Load test data
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')



# print(train_df.isna().sum())
# print(test_df.isna().sum())

# Impute missing values with the median for numerical columns
train_df.fillna(train_df.median(), inplace=True)
test_df.fillna(test_df.median(), inplace=True)


# Convert 'day' to datetime
train_df['day'] = pd.to_datetime(train_df['day'])
test_df['day'] = pd.to_datetime(test_df['day'])

# Extract additional features
train_df['year'] = train_df['day'].dt.year
train_df['month'] = train_df['day'].dt.month
train_df['day_of_year'] = train_df['day'].dt.dayofyear

test_df['year'] = test_df['day'].dt.year
test_df['month'] = test_df['day'].dt.month
test_df['day_of_year'] = test_df['day'].dt.dayofyear



# Features
remove_features = ['rainfall','id', "day"]
features = [c for c in train_df.columns if not c in remove_features]
print(features)

# Target
target = 'rainfall'

X = train_df[features]
y = train_df[target]



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)


y_val_prob = model.predict_proba(X_val_scaled)[:, 1]  # Probability of the positive class
roc_auc = roc_auc_score(y_val, y_val_prob)
print(f'Validation ROC AUC: {roc_auc:.4f}')


# check the data value
print(pd.DataFrame({"id": test_df["id"].values[0:10], "rainfall": y_val_prob[0:10]}))


# Prepare test data
X_test = test_df[features]
X_test_scaled = scaler.transform(X_test)

# Predict probabilities for the test set
test_prob = model.predict_proba(X_test_scaled)[:, 1]

# Prepare submission
submission = pd.DataFrame({'id': test_df['id'], 'rainfall': test_prob})
submission.to_csv('submission.csv', index=False)


submission.head()

