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


# Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Read "train.csv" file
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission_file = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


# See informations
train.info()


sns.scatterplot(x = "curvature",
                y = "accident_risk",
                data = train)
plt.title("Curvature vs. Accident Risk")
plt.show()


train["road_type"].value_counts().sort_index().plot(kind = "bar")
plt.title("The amount of accidents by road type")
plt.xlabel("Road type")
plt.ylabel("Number of accidents")
plt.show()


train["weather"].value_counts().sort_index().plot(kind = "bar")
plt.title("The amount of accidents by weather")
plt.xlabel("Weather")
plt.ylabel("Number of accidents")
plt.show()


train["lighting"].value_counts().sort_index().plot(kind = "bar")
plt.title("The amount of accidents by lighting")
plt.xlabel("Lighting")
plt.ylabel("Number of accidents")
plt.show()


train["time_of_day"].value_counts().sort_index().plot(kind = "bar")
plt.title("The amount of accidents by time of the day")
plt.xlabel("Time of the day")
plt.ylabel("Number of accidents")
plt.show()


sns.scatterplot(x = "speed_limit",
                y = "accident_risk",
                data = train)
plt.title("Speed limit vs accident risk")
plt.show()


train["road_signs_present"].value_counts().plot(kind = "pie", autopct='%1.1f%%')
plt.title("Accident Risk According to the Roadsign's Existence")
plt.show()


train["public_road"].value_counts().plot(kind = "pie", autopct='%1.1f%%')
plt.title("Accident Risk According to the Road (Public Road/Not)")
plt.show()


train["holiday"].value_counts().plot(kind = "pie", autopct='%1.1f%%')
plt.title("Accident Risk According to the Day (Holiday/Not)")
plt.show()


train["school_season"].value_counts().plot(kind = "pie", autopct='%1.1f%%')
plt.title("Accident Risk According to the Season (School Season/Not)")
plt.show()


# Create a histogram for "accident_risk" column
train["accident_risk"].hist(bins = 100)


# Change boolean data from train and test dataset to integer type
train[train.select_dtypes('bool').columns] = train.select_dtypes('bool').astype(int)
test[test.select_dtypes('bool').columns] = test.select_dtypes('bool').astype(int)


# Select all object-typed columns and separate them into their new own columns (boolean)
categorical_columns = ['road_type', 'lighting', 'weather', 'time_of_day']
train_data = pd.get_dummies(train, columns=categorical_columns)
test_data = pd.get_dummies(test, columns=categorical_columns)


train_data


# Drop 'id' column only if it exists
if 'id' in train_data.columns:
    train.drop('id',axis=1,inplace=True)
if 'id' in test_data.columns:
    test.drop('id',axis=1,inplace=True)


# Create the heatmap to see the correlations between all the columns of train_data
sns.heatmap(train_data.corr(), cmap='PRGn')


# Select features and target
X = train_data.drop('accident_risk', axis=1) # Features
y = train_data['accident_risk'] # Target (the objective), what we want to predict

from sklearn.model_selection import train_test_split
# "test_size=0.2" means 20% data will be used for testing and 80% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Import sklearn libraries to create the models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris


# Data scaling, fit and transform the training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Define random forest model
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# Evaluate regression model
# RMSE indicates the average difference between the predicted and actual values.
# A lower RMSE means better performance
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"Random Forest RMSE: {rmse:.4f}")


# Define KNN model
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)


# Evaluate regression model
# RMSE indicates the average difference between the predicted and actual values.
# A lower RMSE means better performance
rmse2 = np.sqrt(mean_squared_error(y_test, y_pred_knn))
print(f"KNN Regressor RMSE: {rmse2:.4f}")


# Choose the smaller RMSE value between two models
test_pred = rf.predict(test_data) # Random Forest model


test_pred


# Write the prediction results based on the given format
submission_file['accident_risk']=test_pred.round(3)


# Create the .csv file from it
submission_file.to_csv('submission_accident_risk.csv', index=False)
print("Prediction file successfully created!")

