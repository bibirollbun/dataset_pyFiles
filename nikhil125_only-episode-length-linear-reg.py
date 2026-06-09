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

import warnings
warnings.filterwarnings('ignore')

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df=pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
testdf=pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


df=df[["Episode_Length_minutes","Listening_Time_minutes"]]
df=df.dropna()
X=df[["Episode_Length_minutes"]]
y=df["Listening_Time_minutes"]
testdf=testdf[["Episode_Length_minutes"]]

median_episode_length_df = df['Episode_Length_minutes'].median()
testdf['Episode_Length_minutes'].fillna(median_episode_length_df, inplace=True)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred2 = model.predict(testdf)
# Evaluate the model using RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')


t_id = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")["id"]
y_pred2 = pd.Series(y_pred2)
submission_df = pd.concat([t_id, y_pred2], axis=1)
submission_df.columns = ['id', 'Listening_Time_minutes']
submission_df.to_csv("submission.csv", index=False)

