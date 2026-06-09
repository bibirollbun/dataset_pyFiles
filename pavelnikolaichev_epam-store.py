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
dtype_dict={"id":np.uint32,
            "store_nbr":np.uint8,
            "item_nbr":np.uint32,
            "unit_sales":np.float32
           }
data = pd.read_csv("/kaggle/working/train.csv", dtype=dtype_dict)
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
dtype_dict={"id":np.uint32,
            "store_nbr":np.uint8,
            "item_nbr":np.uint32,
            "unit_sales":np.float32
           }

df = pd.read_csv("/kaggle/working/train_data.csv", dtype=dtype_dict)
test_df = pd.read_csv("/kaggle/working/test_data.csv", dtype=dtype_dict)


df["date"] = pd.to_datetime(df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])


df.isna().sum()


# Assuming onpromotion na = -1
df.fillna(-1, inplace=True)


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
holiday_events['date'] = pd.to_datetime(holiday_events['date'])
print(holiday_events.shape)
holiday_events.head()


df = df.merge(holiday_events[['date', 'type', 'locale']], on='date', how='left')
test_df = test_df.merge(holiday_events[['date', 'type', 'locale']], on='date', how='left')

df['is_holiday'] = df['type'].notnull().astype(int)
test_df['is_holiday'] = test_df['type'].notnull().astype(int)

df['type'] = df['type'].fillna('None')
test_df['type'] = test_df['type'].fillna('None')
df['locale'] = df['locale'].fillna('None')
test_df['locale'] = test_df['locale'].fillna('None')


holiday_sales = df.groupby('type')['unit_sales'].sum()

plt.figure(figsize=(10, 6))
sns.barplot(x=holiday_sales.index, y=holiday_sales.values, palette='viridis')
plt.title('Unit Sales by Holiday Type')
plt.xlabel('Holiday Type')
plt.ylabel('Total Unit Sales')
plt.show()

locale_sales = df.groupby('locale')['unit_sales'].sum()

plt.figure(figsize=(10, 6))
sns.barplot(x=locale_sales.index, y=locale_sales.values, palette='viridis')
plt.title('Unit Sales by Locale')
plt.xlabel('Locale')
plt.ylabel('Total Unit Sales')
plt.show()

%xdel holiday_sales
%xdel locale_sales

holiday_sales = df.groupby('type')['unit_sales'].mean()

plt.figure(figsize=(10, 6))
sns.barplot(x=holiday_sales.index, y=holiday_sales.values, palette='viridis')
plt.title('Unit Sales by Holiday Type')
plt.xlabel('Holiday Type')
plt.ylabel('Mean Unit Sales')
plt.show()

locale_sales = df.groupby('locale')['unit_sales'].mean()

plt.figure(figsize=(10, 6))
sns.barplot(x=locale_sales.index, y=locale_sales.values, palette='viridis')
plt.title('Unit Sales by Locale')
plt.xlabel('Locale')
plt.ylabel('Mean Unit Sales')
plt.show()

%xdel holiday_sales
%xdel locale_sales


avg_sales_holiday = df[df['is_holiday'] == 1]['unit_sales'].mean()
avg_sales_non_holiday = df[df['is_holiday'] == 0]['unit_sales'].mean()

plt.figure(figsize=(8, 6))
sns.barplot(x=['Holiday', 'Non-Holiday'], y=[avg_sales_holiday, avg_sales_non_holiday], palette='viridis')
plt.title('Average Unit Sales: Holiday vs. Non-Holiday')
plt.xlabel('Day Type')
plt.ylabel('Average Unit Sales')
plt.show()

%xdel avg_sales_holiday
%xdel avg_sales_non_holiday


# # too lazy for using sklearn dummy/one-hot encoding - UPD: forgot that catboost does brrrr
# types = ['National', 'Regional', 'Local', 'None']
# for t in types:
#     train_data[f'is_{t.lower()}_holiday'] = (train_data['type'] == t).astype(int)
#     test_data[f'is_{t.lower()}_holiday'] = (test_data['type'] == t).astype(int)


items = pd.read_csv("/kaggle/working/items.csv")
print(items.shape)
items.head()


df = df.merge(items, on='item_nbr', how='left')
test_df = test_df.merge(items, on='item_nbr', how='left')


family_sales = df.groupby('family')['unit_sales'].sum()

plt.figure(figsize=(12, 8))
sns.barplot(x=family_sales.index, y=family_sales.values, palette='viridis')
plt.title('Total Unit Sales by Item Family')
plt.xlabel('Item Family')
plt.ylabel('Total Unit Sales')
plt.xticks(rotation=90)
plt.show()

%xdel family_sales


class_sales = df.groupby('class')['unit_sales'].sum()

top = class_sales.nlargest(15) # Top 15
plt.figure(figsize=(12, 8))
sns.barplot(x=top.index, y=top.values, palette='viridis')
plt.show()
%xdel class_sales, top


oil = pd.read_csv("/kaggle/working/oil.csv")
print(oil.shape)
oil.head()


oil.isna().sum()


oil['date'] = pd.to_datetime(oil['date'])


fig, axis = plt.subplots(figsize=(12, 6))

oil[(oil["date"] >= pd.to_datetime("2012-12-01"))&(oil["date"] < pd.to_datetime("2017-08-01"))].fillna(0).plot(x='date', y='dcoilwtico', kind='bar', color='skyblue', ax=axis)
axis.set_title('Oil Prices Data for Each day', fontsize=14)
axis.set_xlabel('Day', fontsize=12)
axis.set_ylabel('Price', fontsize=12)
# axis.set_xticklabels(axis.get_xticklabels(), rotation=45)
axis.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


fig, axis = plt.subplots(figsize=(12, 6))

df.groupby('date')['unit_sales'].mean().plot(x='date', y='unit_sales', kind='bar', color='skyblue', ax=axis)
axis.set_title('Unit sales Data for Each day (no separation by items)', fontsize=14)
axis.set_xlabel('Day', fontsize=12)
axis.set_ylabel('Price', fontsize=12)
# axis.set_xticklabels(axis.get_xticklabels(), rotation=45)
axis.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


%xdel oil


stores = pd.read_csv("/kaggle/working/stores.csv")
print(stores.shape)
stores.head()


df = df.merge(stores, on='store_nbr', how='left')
test_df = test_df.merge(stores, on='store_nbr', how='left')
# Another crash


def nwrmsle(y_true, y_pred, W=None):
    y_true = y_true.clip(0, y_true.max()) # To prevent log <= 0
    y_pred = y_pred.clip(0, y_pred.max()) 
    
    if W is None:
        W = np.ones_like(y_true)
    
    weighted_msle = np.nansum(W * ((np.log1p(y_pred) - np.log1p(y_true)) ** 2)) / np.sum(W)
    
    return np.sqrt(weighted_msle)


df.isna().sum()


test_df.isna().sum()


df['year'] = df['date'].dt.year.astype(np.uint16)
df['month'] = df['date'].dt.month.astype(np.uint8)
df['day'] = df['date'].dt.day.astype(np.uint8)
df['dayofweek'] = df['date'].dt.dayofweek


from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=10)


!pip install catboost


# from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool

# # Just in case
# # df.fillna(0, inplace=True)

# # df['date'] = pd.to_datetime(df['date'])
# # df['year'] = df['date'].dt.year
# # df['month'] = df['date'].dt.month
# # df['day'] = df['date'].dt.day
# # df['dayofweek'] = df['date'].dt.dayofweek
# # df['weekofyear'] = df['date'].dt.isocalendar().week

# # features = ['store_nbr', 'item_nbr', 'year', 'month', 'day', 'dayofweek', 'weekofyear',] #  'family', 'class', 'perishable', 'cluster'
# # target = 'unit_sales'

# # cat_features = ['family', 'class', 'perishable', 'cluster']
# # cat_features = []

# # Perishable columns have weights, which have been described in dataset description
df['weight'] = df['perishable'].apply(lambda x: 1.25 if x == 1 else 1.0)
# features = ['store_nbr', 'item_nbr', 'year', 'month', 'day', 'dayofweek', 'onpromotion',] #  'family', 'class', 'perishable', 'cluster'
# cat_features = ['onpromotion']
# target = 'unit_sales'
# _, X_val, _, y_val = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# # train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
# # val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

nwrmsle_scorer = make_scorer(nwrmsle, greater_is_better=False, W=None) # no perishables (. UPD: Lazy to add weights (

params = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2],
    'iterations': [500, 1000], # early stopping goes brr, so I assume no need to make it too complex
    'l2_leaf_reg': [1, 3, 5]
}

model = CatBoostRegressor(loss_function='RMSE', verbose=100, random_state=42)

grid_search = GridSearchCV(estimator=model, param_grid=params, scoring=nwrmsle_scorer, cv=tscv, n_jobs=-1, verbose=2) # 'neg_mean_squared_log_error'
features = ['store_nbr', 'item_nbr', 'year', 'month', 'day', 'dayofweek', 'onpromotion','family', 'class', 'perishable', 'cluster', 'weight'] #  'family', 'class', 'perishable', 'cluster'
cat_features = ['onpromotion', 'family', 'class', 'perishable']
target = 'unit_sales'

# batch_size = 300000
# for i in range(0, len(df), batch_size):
#     batch = df[i:i + batch_size]
grid_search.fit(df[features], df[target], cat_features=cat_features)

print("Best parameters:", grid_search.best_params_)
print("Best NWRMSLE score:", -grid_search.best_score_)

# best_model = grid_search.best_estimator_
# best_model.fit(train_data[features], train_data[target], cat_features=cat_features)


# test_df['date'] = pd.to_datetime(test_df['date'])
test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['dayofweek'] = test_df['date'].dt.dayofweek

test_df['weight'] = test_df['perishable'].apply(lambda x: 1.25 if x == 1 else 1.0)

# Merge with stores and items
# test_df = test_df.merge(stores, on='store_nbr', how='left')
# test_df = test_df.merge(items, on='item_nbr', how='left')

# Handle missing values
test_df.fillna(0, inplace=True)

test_predictions = grid_search.best_estimator_.predict(test_df[features])

# Save predictions
submission = pd.DataFrame({'id': test_df['id'], 'unit_sales': test_predictions})
submission.to_csv('submission.csv', index=False)




