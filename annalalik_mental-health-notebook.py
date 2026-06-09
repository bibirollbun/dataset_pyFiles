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


train_data = pd.read_csv("/kaggle/input/playground-series-s4e11/train.csv")
train_data.head()



test_data = pd.read_csv("/kaggle/input/playground-series-s4e11/test.csv")
test_data.head()


print(train_data.isna().sum())


train_data["Profession"] = train_data["Profession"].fillna("Unknown")
test_data["Profession"] = test_data["Profession"].fillna("Unknown")


train_data["Academic Pressure"] = train_data["Academic Pressure"].fillna(train_data["Academic Pressure"].mode()[0])
test_data["Academic Pressure"] = test_data["Academic Pressure"].fillna(test_data["Academic Pressure"].mode()[0])


train_data["Work Pressure"] = train_data["Work Pressure"].fillna(train_data["Work Pressure"].mean())
test_data["Work Pressure"] = test_data["Work Pressure"].fillna(test_data["Work Pressure"].mean())


train_data["CGPA"] = train_data["CGPA"].fillna(train_data["CGPA"].mean())
test_data["CGPA"] = test_data["CGPA"].fillna(test_data["CGPA"].mean())


train_data["Study Satisfaction"] = train_data["Study Satisfaction"].fillna(train_data["Study Satisfaction"].mode()[0])
test_data["Study Satisfaction"] = test_data["Study Satisfaction"].fillna(test_data["Study Satisfaction"].mode()[0])


train_data["Job Satisfaction"] = train_data["Job Satisfaction"].fillna(train_data["Job Satisfaction"].mode()[0])
test_data["Job Satisfaction"] = test_data["Job Satisfaction"].fillna(test_data["Job Satisfaction"].mode()[0])


train_data["Dietary Habits"] = train_data["Dietary Habits"].fillna(train_data["Dietary Habits"].mode()[0])
test_data["Dietary Habits"] = test_data["Dietary Habits"].fillna(test_data["Dietary Habits"].mode()[0])


train_data["Degree"] = train_data["Degree"].fillna(train_data["Degree"].mode()[0])
test_data["Degree"] = test_data["Degree"].fillna(test_data["Degree"].mode()[0])


train_data["Financial Stress"] = train_data["Financial Stress"].fillna(train_data["Financial Stress"]).mode()[0];
test_data["Financial Stress"] = test_data["Financial Stress"].fillna(test_data["Financial Stress"]).mode()[0];


train_data.describe()
print(train_data.isna().sum())


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.boxplot(x=train_data["Depression"], y=train_data["Age"])
plt.title("Age & depression risk")
plt.show()

sns.boxplot(x=train_data["Depression"], y=train_data["Work Pressure"])
plt.title("Work Pressure & depression risk")
plt.show()

sns.boxplot(x=train_data["Depression"], y=train_data["Academic Pressure"])
plt.title("Academic Pressure & depression risk")
plt.show()

sns.boxplot(x=train_data["Depression"], y=train_data["Job Satisfaction"])
plt.title("Job Satisfaction & depression risk")
plt.show()

sns.boxplot(x=train_data["Depression"], y=train_data["Study Satisfaction"])
plt.title("Study Satisfaction & depression risk")
plt.show()

sns.boxplot(x=train_data["Depression"], y=train_data["Work/Study Hours"])
plt.title("Work/Study Hours & depression risk")
plt.show()

sns.boxplot(x=train_data["Depression"], y=train_data["Financial Stress"])
plt.title("Financial Stress & depression risk")
plt.show()

sns.boxplot(x=train_data["Depression"], y=train_data["Have you ever had suicidal thoughts ?"])
plt.title("Suicidal thoughts & depression risk")
plt.show()



print(train_data.isna().sum())


test_data.head()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Wybór cech
features = ["Age", "Academic Pressure", "Work Pressure", "Financial Stress", "Work/Study Hours"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
y = train_data["Depression"]

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'id': test_data.id, 'Depression': predictions})
output.to_csv('submission_mental_health.csv', index=False)

output_file = pd.read_csv("/kaggle/working/submission_mental_health.csv")
output_file.head()


importances = model.feature_importances_
feature_names = X.columns

# Wykres cech o największym wpływie
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feature_names)
plt.title("The most important features that have impact on depression")
plt.show()

