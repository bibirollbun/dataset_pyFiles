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


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


print(train.shape)
print(test.shape)


train.isnull().sum()


test.isnull().sum()


train.duplicated().sum()


test.duplicated().sum()


train.info()


test.info()


train["Sex"].value_counts()


test["Sex"].value_counts()


train.head(5)


test.head(5)


for col in train:
    print(f"{col} has {train[col].nunique()}")


submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
submission.head(5)


print(f"The columns of train dataset is {train.columns}")
print(f"The columns of test dataset is {test.columns}")
print(f"The columns of submission dataset is {submission.columns}")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error

# Feature Engineering: Convert categorical variables
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

# Features and target variable
X_train = train.drop(['id', 'Calories'], axis=1)
y_train = train['Calories']
X_test = test.drop(['id'], axis=1)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on test set
predictions = model.predict(X_test_scaled)

# Prepare submission DataFrame
submission = pd.DataFrame({'id': test['id'], 'Calories': predictions})

# Save the submission file
submission.to_csv('submission.csv', index=False)

# Optionally, calculate RMSLE on train/test splits (if validation data is available)
y_pred_train = model.predict(X_train_scaled)
rmsle = np.sqrt(mean_squared_log_error(y_train, y_pred_train))
print(f"RMSLE on training data: {rmsle}")


