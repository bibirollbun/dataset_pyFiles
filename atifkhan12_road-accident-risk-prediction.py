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
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,mean_absolute_percentage_error


test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
train=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')


print("Train dataset shape: ",train.shape)
print("Test dataset shape: ",test.shape)


train.head()


train.tail()


train.isnull().sum()


train.duplicated().sum()


train.info()


train.describe()


train["curvature"].plot(kind="hist")


train["speed_limit"].plot(kind="hist")


sns.barplot(x=train["road_type"],y=train["accident_risk"])
plt.title("accident rist by road_type")
plt.show()


sns.barplot(x=train["lighting"],y=train["accident_risk"])
plt.title("accident rist on lighting")
plt.show()


sns.barplot(x=train["weather"],y=train["accident_risk"])
plt.title("accident rist on weather")
plt.show()


sns.barplot(x="time_of_day",y="num_reported_accidents",data=train)
plt.show()


sns.barplot(x="speed_limit",y="num_reported_accidents",data=train)
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(train['accident_risk'],
             bins=50,
             kde=True)

plt.title('Accident Risk Distribution')
plt.xlabel('Accident Risk')
plt.ylabel('Frequency')
plt.show()


le = LabelEncoder()
Label_encoder={}
for col in train.columns:
    if train[col].dtype == 'object':
        train[col] = le.fit_transform(train[col])
        test[col] = le.transform(test[col])
        Label_encoder[col]=le


X = train.drop(columns=["id","accident_risk"],axis=1)
y = train['accident_risk']


train_X,test_X,train_y,test_y = train_test_split(X,
                                                  y, 
                                                  test_size=0.2, 
                                                  random_state=42)


dt = DecisionTreeRegressor()
dt.fit(train_X, train_y)
predictions = dt.predict(test_X)

print("r2_score: ",r2_score(test_y, predictions))
print("mean_squared_error: ",mean_squared_error(test_y, predictions))
print("mean_absolute_error: ",mean_absolute_error(test_y, predictions))


rf = RandomForestRegressor(n_estimators=100,
                              random_state=42)
rf.fit(train_X, train_y)
predictions = rf.predict(test_X)

print("r2_score: ",r2_score(test_y, predictions))
print("mean_squared_error: ",mean_squared_error(test_y, predictions))
print("mean_absolute_error: ",mean_absolute_error(test_y, predictions))


lg = LinearRegression()
lg.fit(train_X, train_y)
predictions = lg.predict(test_X)

print("r2_score: ",r2_score(test_y, predictions))
print("mean_squared_error: ",mean_squared_error(test_y, predictions))
print("mean_absolute_error: ",mean_absolute_error(test_y, predictions))


submission=rf.predict(test.drop("id",axis=1))
sample_submission=pd.DataFrame({
    "id":test["id"],
    "submission":submission
})
sample_submission.to_csv("saample_submission.csv")


sample_submission.head()

