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

import seaborn as sns
import matplotlib.pyplot as plt

from category_encoders import TargetEncoder

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
import joblib


df_train=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


df_train.head(20)


df_train.info()


df_train.shape


df_train.isnull().sum()


df_test.isnull().sum()


df_train=df_train.drop('id', axis=1)
test_id=df_test['id']
df_test=df_test.drop('id', axis=1)


# Histogram of Listening Time
plt.figure(figsize=(10, 6))
sns.histplot(df_train['Listening_Time_minutes'], bins=20, kde=True)
plt.title('Distribution of Listening Times')
plt.xlabel('Listening Time (minutes)')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df_train.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()


# Boxplot of Listening Time by Genre
plt.figure(figsize=(12, 6))
sns.boxplot(x='Genre', y='Listening_Time_minutes', data=df_train)
plt.title('Listening Time by Genre')
plt.xticks(rotation=45)
plt.grid()
plt.show()


# Handling missing values in train dataset
imputer = SimpleImputer(strategy='mean')
df_train['Episode_Length_minutes'] = imputer.fit_transform(df_train[['Episode_Length_minutes']])
df_train['Guest_Popularity_percentage'] = imputer.fit_transform(df_train[['Guest_Popularity_percentage']])
df_train['Number_of_Ads'] = imputer.fit_transform(df_train[['Number_of_Ads']])


# Handling missing values in train dataset
imputer = SimpleImputer(strategy='mean')
df_test['Episode_Length_minutes'] = imputer.fit_transform(df_test[['Episode_Length_minutes']])
df_test['Guest_Popularity_percentage'] = imputer.fit_transform(df_test[['Guest_Popularity_percentage']])


# Encoding categorical variables
label_encoders = {}
for column in ['Podcast_Name','Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']:
    le = LabelEncoder()
    df_train[column] = le.fit_transform(df_train[column])
    label_encoders[column] = le


# Splitting the dataset
X = df_train.drop('Listening_Time_minutes', axis=1)
y = df_train['Listening_Time_minutes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Training the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)


# Evaluating the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# Calculate R² Score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2}")


# Saving the model
joblib.dump(model, 'podcast_model.pkl')


for column, le in label_encoders.items():
    joblib.dump(le, f'{column}_encoder.pkl')

