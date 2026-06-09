# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
sns.set_palette('rocket')

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train_data.head()


categorical = ['road_type', 'lighting', 'weather', 'time_of_day']
for column in categorical:
    print(f'Train: {train_data[column].unique()}\tTest: {test_data[column].unique()}')


train_data.isna().sum().sum()


test_data.isna().sum().sum()


(train_data == np.inf).sum()


sns.histplot(data=train_data, x=train_data['accident_risk']);


time_of_day_counts = train_data['time_of_day'].value_counts().reset_index()


columns = ['road_signs_present', 'num_lanes', 'holiday', 'school_season']

fig, axes = plt.subplots(1, len(columns), figsize=(16, 4), sharey=True)
for i, col in enumerate(columns):
    sns.violinplot(data=train_data, x=col, y='accident_risk', inner='quartile', ax=axes[i])
    if i != 0:
        axes[i].set_ylabel("")
axes[0].set_ylabel("Accident risk")
plt.show();


sns.barplot(x='time_of_day', y='count', data=time_of_day_counts);


sns.barplot(x='weather', y='accident_risk', data=train_data.groupby('weather')['accident_risk'].mean().reset_index());


accident_risk_by_lighting = train_data.groupby('lighting')['accident_risk'].mean().reset_index()
accident_risk_by_weather = train_data.groupby('weather')['accident_risk'].median().reset_index()
accident_risk_by_weather


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.violinplot(data=train_data, x='weather', y='accident_risk', inner='quartile', ax=axes[0])
sns.barplot(data=train_data['weather'].value_counts().reset_index(), x='weather', y='count', ax=axes[1])

plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(data=accident_risk_by_lighting, x='lighting', y='accident_risk', ax=axes[0])
sns.barplot(data=train_data.groupby('road_type')['accident_risk'].mean().reset_index(), x='road_type', y='accident_risk', ax=axes[1])

plt.tight_layout()
plt.show()


len(train_data)


fig, axes = plt.subplots(1, 3, figsize=(14, 6))

sns.boxplot(data=train_data, x='num_reported_accidents', y='accident_risk', ax=axes[0])
sns.regplot(data=train_data, x='num_reported_accidents', y='accident_risk', ax=axes[1])
sns.barplot(data=train_data['num_reported_accidents'].value_counts().reset_index(), x='num_reported_accidents', y='count', ax=axes[2])
plt.tight_layout()
plt.show();


train_data.head()


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.barplot(data=train_data['speed_limit'].value_counts().reset_index(), x='speed_limit', y='count', ax=axes[0])
sns.barplot(data=train_data.groupby('speed_limit')['accident_risk'].mean().reset_index(), x='speed_limit', y='accident_risk', ax=axes[1]);


categorical = ['road_type', 'lighting', 'weather', 'time_of_day']


categorical = ['road_type', 'lighting', 'weather', 'time_of_day']
onehot_encoded_cols_train = pd.get_dummies(train_data[categorical])
onehot_encoded_cols_test = pd.get_dummies(test_data[categorical])
train_data = train_data.drop(categorical, axis=1)
test_data = test_data.drop(categorical, axis=1)


onehot_encoded_cols_train.head()


# Check if onehot encoding created the same categories in both train and test sets
all([col in onehot_encoded_cols_test.columns for col in onehot_encoded_cols_train.columns])


train_data = pd.concat([train_data, onehot_encoded_cols_train], axis=1)
test_data = pd.concat([test_data, onehot_encoded_cols_test], axis=1)


len(train_data.columns), len(test_data.columns)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

rf_model = RandomForestRegressor()
train_data = train_data[:1000]
print(len(train_data))
X = train_data.drop('accident_risk', axis=1).values
y = train_data['accident_risk'].values
print(X.shape, y.shape)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)


rf_model.fit(X, y)


preds = rf_model.predict(X)
mse = (np.sum((preds - y) ** 2)) / len(y)
mse

