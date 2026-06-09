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


import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_log_error


train_data = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train_data_missing = train_data.isna().mean() * 100
test_data_missing = test_data.isna().mean() * 100

print("Percentage missing value in Train Data")
print(train_data_missing)

print("\n Percentage missing value in Test Data")
print(test_data_missing)


train_data_noid = train_data.drop(['id'], axis=1)


train_data_noid.describe()


# Checking outlier
kolom = train_data_noid.drop(['Sex'], axis=1).columns
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12,8))

for i, ax in enumerate(axs.flatten()):
    ax.boxplot(train_data_noid[kolom[i]])
    ax.set_xlabel(f'{kolom[i]}')
    ax.set_ylabel('Outlier')
    
plt.tight_layout()
plt.show()


le = LabelEncoder()
train_data_noid['Sex'] = le.fit_transform(train_data_noid['Sex'])
test_data['Sex'] = le.fit_transform(test_data['Sex'])


train_data.columns


def new_feature(df):
    df['BMI'] = df['Weight']/((df['Height']/100)**2)
    df['Heart Rate per Minute'] = df['Heart_Rate']/df['Duration'] 
    df['Weight x Duration'] = df['Weight']*df['Duration']
    df['Heart Rate x Body Temp'] = df['Heart_Rate']*df['Body_Temp']


new_feature(train_data_noid)
new_feature(test_data)


train_data_noid


test_data


train_data_noid.describe()


test_data.describe()


def feature_scaling(feature):
    scaler = MinMaxScaler()
    normalized_feature = scaler.fit_transform(feature)
    return normalized_feature


feature_scaling(train_data_noid)


feature_scaling(test_data)


X_train = train_data_noid.drop(['Calories'], axis=1)
y_train = train_data_noid.Calories


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)


log_reg = LogisticRegression(max_iter=50, random_state=0)
log_reg.fit(X_train, y_train)


y_pred_log = log_reg.predict(X_test)


mae_log_reg = mean_squared_log_error(y_test, y_pred_log)
rmsle_log_reg = math.sqrt(mae_log_reg)

print(f'Root Mean Square Log Error: {rmsle_log_reg}')


randomforest = RandomForestRegressor(max_depth=5, random_state=0)
randomforest.fit(X_train, y_train)


y_pred_rf = randomforest.predict(X_test)


mae_rf = mean_squared_log_error(y_test, y_pred_rf)
rmsle_rf = math.sqrt(mae_rf)

print(f'Root Mean Square Log Error: {rmsle_rf}')


knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train, y_train)


y_pred_knn = knn_regressor.predict(X_test)


mae_knn = mean_squared_log_error(y_test, y_pred_knn)
rmsle_knn = math.sqrt(mae_knn)

print(f'Root Mean Square Log Error: {rmsle_knn}')


id_pred = test_data['id']


prediction = knn_regressor.predict(test_data.drop(['id'], axis=1))


submission = pd.DataFrame({"id" : id_pred, "Calories" : prediction})


submission


submission.to_csv("submission.csv",index=False)

