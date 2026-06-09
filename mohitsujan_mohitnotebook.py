# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
test_data = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")
print(train_data.head())
print(test_data.head())


def prepare(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    return df


final_train_data = prepare(train_data)
final_test_data = prepare(test_data)
X_df = final_train_data[['hour', 'day', 'month', 'year', 'dayofweek']]
y_df = final_train_data['count']
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.025, random_state=62)
model = RandomForestRegressor(max_depth=30,n_estimators=155,max_features='sqrt', max_samples=0.8, random_state=62,n_jobs=-1,)
model.fit(X_train, y_train)
val_preds = model.predict(X_test)
val_score = np.sqrt(mean_squared_log_error(y_test, val_preds))
print(f"RMSLE: {val_score}")


test_features = test_data[['hour', 'day', 'month', 'year', 'dayofweek']]
test_prediction = model.predict(test_features)


submission = pd.DataFrame({'datetime': test_data['datetime'],'count': test_prediction})
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")


plt.figure(figsize=(15, 8))
plt.plot(train_data['datetime'], train_data['count'], label="Data")
plt.xlabel("years")
plt.ylabel("No of bikes")
plt.title("Bike Sharing Demand")
plt.legend()
plt.show()

