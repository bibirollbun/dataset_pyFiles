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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


submission_path = '/kaggle/input/playground-series-s5e5/sample_submission.csv'
train_path = '/kaggle/input/playground-series-s5e5/train.csv'
test_path = '/kaggle/input/playground-series-s5e5/test.csv'


submission_data = pd.read_csv(submission_path)
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)


train_data.shape, test_data.shape, submission_data.shape


train_data


train_data.isna().sum()


train_data.describe()


df = train_data


sex = {'male' : 1 , 'female' : 0}


df['Sex'] = df['Sex'].map(sex)


df.sample(5)


df['height_m'] = df['Height'] / 100
df['BMI'] = df['Weight'] / (df['height_m'] ** 2)


df['Cardio_Load'] = df['Heart_Rate'] * df['Duration']
df['Fever_Flag'] = (df['Body_Temp'] > 37.5).astype(int)


df2 = df.drop(['height_m','Height','Weight','Heart_Rate','Duration','Body_Temp'], axis=1)


corr = df2.corr()
sns.heatmap(corr, annot=True)


df3 = df2.drop(['id'], axis=1)


corr = df3.corr()
sns.heatmap(corr, annot= True)


X = df3.drop(['Calories'], axis=1)
y = df3['Calories']


corr = X.corr()
sns.heatmap(corr, annot=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_model.score(X_test, y_test) #highest: 0.9681
y_pred = lr_model.predict(X_test)
print(f"LinearRegression â†’ RÂ²: {r2_score(y_test, y_pred):.4f}, RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")

