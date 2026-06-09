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


import matplotlib.pyplot as plt
import seaborn as sns
import supplemental_english  # Importing supplemental data (if needed)
import supplemental_russian

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Load datasets
train = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv", index_col=0)
test = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv", index_col=0)
train


test


def preprocess(df):
    df["date"] = pd.to_datetime(df["date"])  # Convert date to datetime format
    df["year"] = df["date"].dt.year          #Extract the year
    df["month"] = df["date"].dt.month        #Extract the months
    df["day"] = df["date"].dt.day            #Extract the day
    df["weekday"] = df["date"].dt.weekday  # Day of the week (0=Monday)
    
    # Extracting features from the plate (region code)
    df["region_code"] = df["plate"].str.extract(r'(\d{2,3})$')  # Extracting numeric region codes
    df["region_code"] = df["region_code"].astype(float)  # Convert to numeric
    
    # Drop unnecessary columns
    df.drop(columns=["plate", "date"], inplace=True)
    return df


train = preprocess(train)
test = preprocess(test)


sns.lineplot(data=train, x="region_code", y="price")


sns.lineplot(data=test, x="year", y="region_code", hue="month", style="month")


train.isnull().sum()


test.isnull().sum()


test1 = test.drop(columns='price')


X= train.iloc[:, [1,2,3,4,5]]
y= train.iloc[:, 0]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


import lightgbm as lgb
model = lgb.LGBMRegressor()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


y_pred


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


test_predictions = model.predict(test1)
test_predictions


submission = pd.DataFrame({"id": test.index, "price": test_predictions})
submission.to_csv("submission.csv", index=False)

print("File is submited sucessfully")

