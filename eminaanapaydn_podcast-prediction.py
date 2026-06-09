# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Read data
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


# Extract info about data
print("Train Data:")
print(train.info())
print(train.describe())
print(train.isnull().sum())
print()
print("Test Data:")
print(test.info())
print(test.describe())
print(test.isnull().sum())


# Dropping unnecassary columns
train = train.drop(columns=['id', 'Podcast_Name', 'Episode_Title', 'Publication_Day', 'Publication_Time'])
test = test.drop(columns=['id', 'Podcast_Name', 'Episode_Title', 'Publication_Day', 'Publication_Time'])


# Handling missing values
train["Number_of_Ads"] = train["Number_of_Ads"].fillna(train["Number_of_Ads"].median())

train["Episode_Length_minutes"] = train.groupby("Genre")["Episode_Length_minutes"].transform(lambda x: x.fillna(x.median()))
test["Episode_Length_minutes"] = test.groupby("Genre")["Episode_Length_minutes"].transform(lambda x: x.fillna(x.median()))

train["Guest_Popularity_percentage"] = train["Guest_Popularity_percentage"].fillna(0)
test["Guest_Popularity_percentage"] = test["Guest_Popularity_percentage"].fillna(0)


# Checking again
print("Train Data:")
print(train.info())
print(train.describe())
print(train.isnull().sum())
print()
print("Test Data:")
print(test.info())
print(test.describe())
print(test.isnull().sum())


# EDA
plt.figure(figsize=(10, 6))
sns.countplot(data=train, x='Genre', palette='viridis')
plt.xticks(rotation=90)
plt.title('Podcast Genre Distribution')
plt.show()


plt.figure(figsize=(8, 6))
sns.countplot(data=train, x='Episode_Sentiment', palette='coolwarm')
plt.title('Episode Sentiment Distribution')
plt.show()


plt.figure(figsize=(8, 6))
sns.scatterplot(data=train, x='Episode_Length_minutes', y='Listening_Time_minutes', alpha=0.5)
plt.title('Podcast Length vs Listening Time')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(data=train, x='Episode_Length_minutes')
plt.title('Podcast Length Outlier Values')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(data=train, x='Listening_Time_minutes')
plt.title('Listening Time Outlier Values')
plt.show()


# Selecting categorical columns to encode
cat_col = train.select_dtypes(include='object').columns.tolist()
num_col = train.select_dtypes(include=['int64', 'float64']).columns.tolist()


print(train['Genre'].unique())
print(train['Genre'].nunique())
print(train['Episode_Sentiment'].unique())
print(train['Episode_Sentiment'].nunique())


# OHE for Genre
train = pd.get_dummies(train, columns=['Genre'], drop_first=True, dtype=int)
test = pd.get_dummies(test, columns=['Genre'], drop_first=True, dtype=int)


# Label encoding for Episode_Sentiment
le = LabelEncoder()

train['Episode_Sentiment'] = le.fit_transform(train['Episode_Sentiment'])
test['Episode_Sentiment'] = le.transform(test['Episode_Sentiment'])


# Define features and target
y = train['Listening_Time_minutes']
X = train.drop('Listening_Time_minutes', axis=1)


# Split data to train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=44)


# Define model
xgb = XGBRegressor(random_state=44)

# Fit model
model = xgb.fit(X_train, y_train)


# Predict the outputs for validation
y_val_pred = model.predict(X_val)


# Display model validation score and train score
print(f"Train Score: {model.score(X_train, y_train)}")
print(f"Validation Score: {model.score(X_val, y_val)}")


# Calculate RMSE for validation
mse_val = mean_squared_error(y_val, y_val_pred)
rmse_val = np.sqrt(mse_val)

print(f"Validation RMSE: {rmse_val}")


# Predict the outputs for test 
y_test_pred = model.predict(test)


# Create a dataframe for submission
submission = pd.DataFrame({
    'id': sample_sub['id'],
    'Listening_Time_minutes': y_test_pred
})

submission.to_csv('submission.csv', index=False)

