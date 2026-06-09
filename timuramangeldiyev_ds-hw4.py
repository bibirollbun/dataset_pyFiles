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
THRESHOLD_TRAIN_DATE_LOW = pd.to_datetime("2016-07-01")
THRESHOLD_TRAIN_DATE = pd.to_datetime("2017-07-01")
THRESHOLD_TEST_DATE = pd.to_datetime("2017-08-01")

data[(data["date"] >= THRESHOLD_TRAIN_DATE_LOW)&(data["date"] < THRESHOLD_TRAIN_DATE)].to_csv("/kaggle/working/train_data.csv")
data[(data["date"] >= THRESHOLD_TRAIN_DATE)&(data["date"] < THRESHOLD_TEST_DATE)].to_csv("/kaggle/working/test_data.csv")

%xdel data


%reset -f


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


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


df.shape


test_df.shape


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


df


df.info()


df.isna().sum() / len(df)


df['onpromotion'].value_counts()


df['onpromotion'] = df['onpromotion'].fillna('unknown')


test_df['onpromotion'] = test_df['onpromotion'].fillna('unknown')


df['item_nbr'].value_counts()


items


items.info()


items['family'].value_counts()


items.isna().sum()/len(items)


df = df.merge(items, on="item_nbr", how="left")


df


test_df = test_df.merge(items, on="item_nbr", how="left")


df.isna().sum()/len(df)


df['store_nbr'].value_counts()


stores.info()


stores['cluster'].value_counts()


df = df.merge(stores, on="store_nbr", how="left")
test_df = test_df.merge(stores, on="store_nbr", how="left")


df.head()


df.isna().sum()/len(df)


oil['date'] = pd.to_datetime(oil['date'])


oil.info()


oil.isna().sum()/len(oil)


oil


oil["year"] = oil["date"].dt.year
oil["month"] = oil["date"].dt.month

monthly_oil = (
    oil.groupby(["year", "month"])["dcoilwtico"]
    .mean()
    .reset_index()
)


monthly_oil['month-year'] = pd.to_datetime(monthly_oil.year.astype(str) + '/' + monthly_oil.month.astype(str) + '/01')


import seaborn as sns
sns.lineplot(data=monthly_oil, y='dcoilwtico', x='month-year')


monthly_oil[monthly_oil['year'] >= 2016]


df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
test_df["year"] = test_df["date"].dt.year
test_df["month"] = test_df["date"].dt.month


df = df.merge(monthly_oil, on=["year", "month"], how="left")
test_df = test_df.merge(monthly_oil, on=["year", "month"], how="left")


df.head()


df.isna().sum() / len(df)


holiday_events


holiday_events['date'] = pd.to_datetime(holiday_events['date'])


holiday_events["is_holiday"] = 1
holidays_national = holiday_events[holiday_events["locale"]=="National"][["date","is_holiday"]]
holidays_national = holidays_national.drop_duplicates("date")


df = df.merge(holidays_national, on="date", how="left")
test_df = test_df.merge(holidays_national, on="date", how="left")

df["is_holiday"] = df["is_holiday"].fillna(0)
test_df["is_holiday"] = test_df["is_holiday"].fillna(0)


df['is_holiday'] = (df['is_holiday'] == 1)
test_df["is_holiday"] = (test_df["is_holiday"] == 1)


df['is_holiday'].value_counts()


df.isna().sum() / len(df)


df['is_weekend'] = df['date'].dt.dayofweek >= 5
test_df['is_weekend'] = test_df['date'].dt.dayofweek >= 5



df


df.columns


numeric = ['class', 'dcoilwtico', 'perishable']
categorical = ['onpromotion', 'family', 'class', 'type', 'cluster', 'is_holiday', 'is_weekend']
label = 'unit_sales'


X_train = df[numeric + categorical]
y_train = df[label]
X_test = test_df[numeric + categorical]
y_test = test_df[label]


X_train.to_csv("/kaggle/working/X_train.csv")
y_train.to_csv("/kaggle/working/y_train.csv")
X_test.to_csv("/kaggle/working/X_test.csv")
y_test.to_csv("/kaggle/working/y_test.csv")


%reset -f


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


X_train = pd.read_csv("/kaggle/working/X_train.csv")
y_train = pd.read_csv("/kaggle/working/y_train.csv")
X_test = pd.read_csv("/kaggle/working/X_test.csv")
y_test = pd.read_csv("/kaggle/working/y_test.csv")



y_train = y_train['unit_sales']
y_test = y_test['unit_sales']


X_train


from tqdm import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV

import lightgbm as lgb


numeric = ['class', 'dcoilwtico', 'perishable']
categorical = ['onpromotion', 'family', 'class', 'type', 'cluster', 'is_holiday', 'is_weekend']
label = 'unit_sales'
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ]
)


def normalized_weighted_rmsle(y_true, y_pred, weights):
    log_errors = np.log1p(y_pred) - np.log1p(y_true)
    squared_log_errors = log_errors ** 2
    weighted_squared_log_errors = weights * squared_log_errors
    sum_weighted_errors = np.sum(weighted_squared_log_errors)
    sum_weights = np.sum(weights)
    normalized_weighted_mean = sum_weighted_errors / sum_weights
    return np.sqrt(normalized_weighted_mean)
weights_train = np.where(X_train['perishable'] == 1, 1.25, 1)
weights_test = np.where(X_test['perishable'] == 1, 1.25, 1)


min(y_train)


np.sum(y_train < 0) / len(y_train)


y_test = np.maximum(y_test, 0)
y_train = np.maximum(y_train, 0)


dummy_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DummyRegressor(strategy='mean'))
])
dummy_reg.fit(X_train, y_train)

y_train_pred_dummy = dummy_reg.predict(X_train)
y_test_pred_dummy = dummy_reg.predict(X_test)

wrmsle_train_dummy = normalized_weighted_rmsle(y_train, y_train_pred_dummy, weights_train)
wrmsle_test_dummy = normalized_weighted_rmsle(y_test, y_test_pred_dummy, weights_test)


print(f"Train Normalized Weighted RMSLE: {wrmsle_train_dummy}, Test Normalized Weighted RMSLE: {wrmsle_test_dummy}")



lgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', lgb.LGBMRegressor(objective='regression', boosting_type='gbdt', learning_rate=0.1, 
                                    num_leaves=30, n_estimators=200, random_state=42, n_jobs=-1))
])

lgb_pipeline.fit(X_train, y_train)

y_train_pred_lgb = lgb_pipeline.predict(X_train)
y_test_pred_lgb = lgb_pipeline.predict(X_test)

wrmsle_train_lgb = normalized_weighted_rmsle(y_train, y_train_pred_lgb, weights_train)
wrmsle_test_lgb = normalized_weighted_rmsle(y_test, y_test_pred_lgb, weights_test)

print(f"Train Normalized Weighted RMSLE: {wrmsle_train_lgb}, Test Normalized Weighted RMSLE: {wrmsle_test_lgb}")





