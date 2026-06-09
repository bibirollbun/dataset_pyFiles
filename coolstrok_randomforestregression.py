# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


trainingDF = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
testDF = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


trainingDF.head()


testDF.head()


trainingDF.info()


testDF.info()


trainingDF.isnull().sum()


testDF.isnull().sum()


# Preprocessing the data as we have a lot of missing values in the training and testing dataset
trainingDF["Episode_Length_minutes"] = trainingDF["Episode_Length_minutes"].fillna(trainingDF["Episode_Length_minutes"].mean())
testDF["Episode_Length_minutes"] = testDF["Episode_Length_minutes"].fillna(testDF["Episode_Length_minutes"].mean())

trainingDF["Guest_Popularity_percentage"] = trainingDF["Guest_Popularity_percentage"].fillna(trainingDF["Guest_Popularity_percentage"].mean())
testDF["Guest_Popularity_percentage"] = testDF["Guest_Popularity_percentage"].fillna(testDF["Guest_Popularity_percentage"].mean())

trainingDF["Number_of_Ads"] = trainingDF["Number_of_Ads"].fillna(trainingDF["Number_of_Ads"].mean())


print(trainingDF.isnull().sum())
print(testDF.isnull().sum())


print(trainingDF.info())
print(testDF.info())


for column in trainingDF.columns:
    if trainingDF[column].dtype == "object":
        le = LabelEncoder()
        trainingDF[column] = le.fit_transform(trainingDF[column])

for column in testDF.columns:
    if testDF[column].dtype == "object":
        le = LabelEncoder()
        testDF[column] = le.fit_transform(testDF[column])


print(trainingDF.info())
print(testDF.info())


X_train = trainingDF.drop(columns = "Listening_Time_minutes")
Y_train = trainingDF["Listening_Time_minutes"]

xgbr = XGBRegressor(n_estimators = 100, learning_rate = 0.1, max_depth = 6, random_state = 42)
xgbr.fit(X_train, Y_train)


Y_pred = xgbr.predict(testDF)

outputDF = pd.DataFrame({"id": testDF["id"], "Listening_Time_minutes": Y_pred})
outputDF.head()


outputDF.to_csv("submission.csv", index = False)




