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


df_train  = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")


df_test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


df_main = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


df_train.head()


df_test.head()


df_main.head()


df_train.describe()


df_train.info()


df_train.tail()


df_train.isnull().sum()


df_train.duplicated().sum()


df_test.describe()


df_test.info()


df_test.tail()


df_test.sample(5)


df_test.isnull().sum()


df_test.duplicated().sum()


df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)


df_train.shape,df_test.shape


df_train.corr()


df_train.dtypes


y = df_train['BeatsPerMinute']
X = df_train.drop(columns=['BeatsPerMinute'])


X_test = df_test


X


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


df_train.columns


# Import libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

# Features and target
X = train.drop(columns=["BeatsPerMinute"])
y = train["BeatsPerMinute"]

# Train a simple Random Forest model
model = RandomForestRegressor(random_state=42, n_jobs=-1)
model.fit(X, y)

# Predict on test set
preds = model.predict(test)

# Save submission file
sample["BeatsPerMinute"] = preds
sample.to_csv("submission.csv", index=False)

print("✅ submission.csv created successfully!")
print(sample.head())





