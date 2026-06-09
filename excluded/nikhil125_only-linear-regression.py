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

import warnings
warnings.filterwarnings('ignore')

# Your code that might generate warnings goes here


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


df0=pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
testdf=pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


df0.info()


df0['Guest_Popularity_percentage'].fillna(df0['Guest_Popularity_percentage'].mean(), inplace=True)
df0['Episode_Length_minutes'].fillna(df0['Episode_Length_minutes'].mean(), inplace=True)
df0['Number_of_Ads'].fillna(df0['Number_of_Ads'].mode()[0], inplace=True)

testdf['Guest_Popularity_percentage'].fillna(testdf['Guest_Popularity_percentage'].mean(), inplace=True)
testdf['Episode_Length_minutes'].fillna(testdf['Episode_Length_minutes'].mean(), inplace=True)
testdf['Number_of_Ads'].fillna(testdf['Number_of_Ads'].mode()[0], inplace=True)

categorical_cols = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

df0 = pd.get_dummies(df0, columns=categorical_cols, drop_first=True)
testdf = pd.get_dummies(testdf, columns=categorical_cols, drop_first=True)


# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Prepare the data
X = df0.drop(['Listening_Time_minutes', 'Episode_Title', 'id'], axis=1)
testdf = testdf.drop([ 'Episode_Title', 'id'], axis=1)
y = df0['Listening_Time_minutes']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
final_predictions = model.predict(testdf)

# Evaluate the model using RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')


df_subm=pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
df_subm['Listening_Time_minutes'] = final_predictions
df_subm.set_index('id', inplace=True)
df_subm.to_csv('submission_01_01.csv')
df_subm.head()

