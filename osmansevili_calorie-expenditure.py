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


import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

sample = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')



train.head()


train.shape


train.info()


train.isnull().sum()


train.describe()


test.shape


test.info()


test.isnull().sum()


plt.hist(train["Calories"], bins=50, color="skyblue", edgecolor="black")
plt.title("Calories")
plt.xlabel("Calories burned")
plt.ylabel("Count")
plt.show()


num_cols = train.select_dtypes(include=['int64','float64']).columns
plt.figure(figsize=(8,6))
sns.heatmap(train[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation")
plt.show()


# Body Mass Index
train["BMI"] = train["Weight"] / ((train["Height"] / 100) ** 2)
test["BMI"] = test["Weight"] / ((test["Height"] / 100) ** 2)

# Duration relative to heart rate (efficiency measure)
train["Duration_per_Heart"] = train["Duration"] / train["Heart_Rate"]
test["Duration_per_Heart"] = test["Duration"] / test["Heart_Rate"]

# Exercise intensity: interaction between heart rate and body temperature
train["Intensity"] = train["Heart_Rate"] * train["Body_Temp"]
test["Intensity"] = test["Heart_Rate"] * test["Body_Temp"]

# Body temperature per minute of exercise
train["Temp_per_Minute"] = train["Body_Temp"] / train["Duration"]
test["Temp_per_Minute"] = test["Body_Temp"] / test["Duration"]


le = LabelEncoder()
train["Sex"] = le.fit_transform(train["Sex"])   # e.g., female=0, male=1 (alphabetical order)
test["Sex"]  = le.transform(test["Sex"])


x = train.drop(columns=["id", "Calories"])   # keep engineered features
y = train["Calories"]


x_test = test.drop(columns=["id"])


x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.20, random_state=42)


g=GaussianNB()
b=BernoulliNB()


g.fit(x_train,y_train)


gtahmin=g.predict(x_test)


accuracy_score(y_test, gtahmin)


confusion_matrix(y_test, gtahmin)


submission = pd.DataFrame({
    'ID': y_test,
    'Class': gtahmin
})


submission




