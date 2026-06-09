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
import sklearn
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


train = pd.read_csv(r'/kaggle/input/playground-series-s5e4/train.csv')
train.head()


train.info()


traindf = train.drop(columns=['Podcast_Name','Episode_Title','Publication_Day','Genre','Publication_Time'])
traindf.head()


sentiment_mapping = {'Neutral': 0, 'Negative': 1, 'Positive': 2}
traindf['Sentiment_Label'] = traindf['Episode_Sentiment'].map(sentiment_mapping)
traindf.head()


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Drop rows with missing target values
traindf = traindf.dropna(subset=['Listening_Time_minutes'])

# Separate features and target variable
X = traindf.drop(columns=['Listening_Time_minutes', 'Episode_Sentiment'])
y = traindf['Listening_Time_minutes']

# Handle missing values in features
X = X.fillna(X.mean())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost regressor
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
# Calculate Root Mean Squared Error
rmse = sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")




