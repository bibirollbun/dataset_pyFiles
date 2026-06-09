import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
paths = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        paths.append(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install py7zr


import py7zr
for filenames in paths:
    with py7zr.SevenZipFile(filenames, mode='r') as z_ref:
        z_ref.extractall(path='/kaggle/working')


%%time
data = pd.read_csv("/kaggle/working/train.csv")
data["date"] =  pd.to_datetime(data["date"])


monthly_counts_data = data['date'].dt.to_period('M').value_counts().sort_index()

plt.figure(figsize=(12, 6))
monthly_counts_data.plot(kind='bar', color='skyblue')
plt.title('Count of Data for Each Month for All data', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


%%time
THRESHOLD_TRAIN_DATE = pd.to_datetime("2017-07-01")
THRESHOLD_TEST_DATE = pd.to_datetime("2017-08-01")

data[data["date"] < THRESHOLD_TRAIN_DATE].to_csv("/kaggle/working/train_data.csv")
data[(data["date"] >= THRESHOLD_TRAIN_DATE)&(data["date"] < THRESHOLD_TEST_DATE)].to_csv("/kaggle/working/test_data.csv")

%xdel data


%%time
df = pd.read_csv("/kaggle/working/train_data.csv")
test_df = pd.read_csv("/kaggle/working/test_data.csv")


df["date"] = pd.to_datetime(df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])


monthly_counts = df['date'].dt.to_period('M').value_counts().sort_index().reset_index()
monthly_counts_test = test_df['date'].dt.to_period('M').value_counts().sort_index().reset_index()


monthly_counts['type'] = 'Train'
monthly_counts_test['type'] = 'Test'
combined = pd.concat([monthly_counts, monthly_counts_test])

plt.figure(figsize=(12, 6))
sns.barplot(data=combined, x='date', y='count', hue='type', palette=['blue', 'red'])
plt.title('Count of Data for Each Month (Train vs Test)', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Dataset')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


df.head()


holiday_events = pd.read_csv("/kaggle/working/holidays_events.csv")
print(holiday_events.shape)
holiday_events.head()


items = pd.read_csv("/kaggle/working/items.csv")
print(items.shape)
items.head()


oil = pd.read_csv("/kaggle/working/oil.csv")
print(oil.shape)
oil.head()


stores = pd.read_csv("/kaggle/working/stores.csv")
print(stores.shape)
stores.head()


from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


df['onpromotion'] = df['onpromotion'].fillna(0).astype(int)
test_df['onpromotion'] = test_df['onpromotion'].fillna(0).astype(int)


df['unit_sales'] = df['unit_sales'].apply(lambda x: max(0, x))
X = df[['store_nbr', 'item_nbr', 'onpromotion']]
y = df['unit_sales']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

dummy_model = DummyRegressor(strategy="mean")
dummy_model.fit(X_train, y_train)

y_pred_val = dummy_model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"Baseline RMSE (Dummy Regressor - Mean): {rmse}")


df['day_of_week'] = df['date'].dt.dayofweek
test_df['day_of_week'] = test_df['date'].dt.dayofweek

df['week_of_year'] = df['date'].dt.isocalendar().week
test_df['week_of_year'] = test_df['date'].dt.isocalendar().week

df['month'] = df['date'].dt.month
test_df['month'] = test_df['date'].dt.month

df['year'] = df['date'].dt.year
test_df['year'] = test_df['date'].dt.year

df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
test_df['is_weekend'] = test_df['day_of_week'].isin([5, 6]).astype(int)


df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)


df.to_csv('train_filtered_features.csv', index=False)
test_df.to_csv('test_filtered_features.csv', index=False)

