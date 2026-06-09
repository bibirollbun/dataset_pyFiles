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
for dirname, _, filenames in os.walk('/kaggle/input/favorita-grocery-sales-forecasting'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        paths.append(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install -q py7zr


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


usecols = ["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]
dtypes = {
    "date": 'str',
    "store_nbr": 'int',
    "item_nbr": 'int',
    "unit_sales": 'float',
    "onpromotion": 'O'
}


%%time
df = pd.read_csv(
    "/kaggle/input/zipped-favorita-grocery-sales/train_data/train_data.csv", 
    usecols=usecols, 
    dtype=dtypes, 
    parse_dates=['date']
)
test_df = pd.read_csv(
    "/kaggle/input/zipped-favorita-grocery-sales/test_data/test_data.csv", 
    usecols=usecols, 
    dtype=dtypes, 
    parse_dates=['date']
)


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


THRESHOLD_TRAIN_DATE_LOWER = pd.to_datetime("2017-01-01")
df = df[df["date"] >= THRESHOLD_TRAIN_DATE_LOWER]
df = df.reset_index(drop=True)
df.head()


holiday_events = pd.read_csv("/kaggle/working/holidays_events.csv")
print(holiday_events.shape)
holiday_events.head()


holiday_events = holiday_events[holiday_events.transferred == False]
holiday_events = holiday_events[holiday_events.type != "Work Day"]


holiday_events['date'] = pd.to_datetime(holiday_events['date'])
holiday_events = holiday_events.rename(columns={"type": "holiday_type"})


df = df.merge(holiday_events[['date', 'holiday_type', 'locale']], on='date', how='left')
test_df = test_df.merge(holiday_events[['date', 'holiday_type', 'locale']], on='date', how='left')


test_df['is_holiday'] = test_df['holiday_type'].notna().astype(int)
df['is_holiday'] = df['holiday_type'].notna().astype(int)

df.head()


items = pd.read_csv("/kaggle/working/items.csv")
print(items.shape)
items.head()


df = df.merge(items, on='item_nbr', how='left')
test_df = test_df.merge(items, on='item_nbr', how='left')


oil = pd.read_csv("/kaggle/working/oil.csv")
print(oil.shape)
oil.head()


oil['date'] = pd.to_datetime(oil['date'])


df_dates = df[['date']]
test_df_dates = test_df[['date']]


df_dates = df_dates.merge(oil, on='date', how='left', copy=False)
test_df_dates = test_df_dates.merge(oil, on='date', how='left', copy=False)


df_dates = df_dates.sort_values('date')
test_df_dates = test_df_dates.sort_values('date')


df_dates.dcoilwtico = df_dates.dcoilwtico.ffill()
df_dates.dcoilwtico = df_dates.dcoilwtico.bfill()

test_df_dates.dcoilwtico = test_df_dates.dcoilwtico.ffill()
test_df_dates.dcoilwtico = test_df_dates.dcoilwtico.bfill()


df_dates = df_dates.sort_index()
test_df_dates = test_df_dates.sort_index()


df['oil_price'] = df_dates.dcoilwtico
test_df['oil_price'] = test_df_dates.dcoilwtico


df.head()


stores = pd.read_csv("/kaggle/working/stores.csv")
print(stores.shape)
stores.head()


stores = stores.rename(columns={"type": "store_type"})


df = df.merge(stores, on='store_nbr', how='left')
test_df = test_df.merge(stores, on='store_nbr', how='left')


weights = test_df.perishable.astype(float).values * 0.25 + 1
weights


def compute_log(arr):
    x = np.ma.log(arr)
    return x.filled(0)


def nwrmsle(y_pred, y_true, weights):
    return np.sqrt(np.sum(weights * (compute_log(y_pred + 1) - compute_log(y_true + 1)) ** 2) / np.sum(weights))


from sklearn.dummy import DummyRegressor

dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(df, df['unit_sales'].values)

dummy_prediction = dummy_regr.predict(test_df)


dummy_score = nwrmsle(dummy_prediction, test_df['unit_sales'].values, weights)
dummy_score


df.head()


df.isna().sum()


df['holiday_type'] = df.holiday_type.fillna('Not holiday')
test_df['holiday_type'] = test_df.holiday_type.fillna('Not holiday')

df['locale'] = df.locale.fillna('None')
test_df['locale'] = test_df.locale.fillna('None')


df.isna().sum()


df['holiday_type'].unique()


holiday_type_mapping = {"Transfer": 2, "Holiday": 2, "Additional": 2, "Event": 1, "Not holiday": 0}


df['holiday_type_encoded'] = df['holiday_type'].map(holiday_type_mapping)
test_df['holiday_type_encoded'] = test_df['holiday_type'].map(holiday_type_mapping)

df.head()


df.locale.unique()


holiday_locale_mapping = {"National": 3, "Regional": 2, "Local": 1, "None": 0}


df['holiday_locale_encoded'] = df['locale'].map(holiday_type_mapping)
test_df['holiday_locale_encoded'] = test_df['locale'].map(holiday_type_mapping)

df.head()


df['onpromotion'] = df['onpromotion'].map({"False": 0, "True": 1})
test_df['onpromotion'] = test_df['onpromotion'].map({"False": 0, "True": 1})


df.nunique()


ONE_HOT_FEATURES = ['family', 'city', 'state', 'store_type', 'cluster']
NUMERICAL_FEATURES = ['oil_price']


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

preprocessor = ColumnTransformer(
    [
        ('scaler', StandardScaler(), NUMERICAL_FEATURES),
        ('one_hot', OneHotEncoder(handle_unknown='ignore'), ONE_HOT_FEATURES),
        ('imputer', SimpleImputer(strategy="median"), 
         ['onpromotion', 'is_holiday', 'perishable', 'holiday_type_encoded', 'holiday_locale_encoded']
        )
    ]
)


# Columns that will be used during training

X_train = df[[
    'onpromotion', 'is_holiday', 'perishable', 
    'oil_price', 'holiday_type_encoded', 'holiday_locale_encoded',
    'family', 'city', 'state', 'store_type', 'cluster'
]]
X_test = test_df[[
    'onpromotion', 'is_holiday', 'perishable', 
    'oil_price', 'holiday_type_encoded', 'holiday_locale_encoded',
    'family', 'city', 'state', 'store_type', 'cluster'
]]


X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)


X_train_preprocessed.shape, X_test_preprocessed.shape


y_train = df['unit_sales'].values
y_test = test_df['unit_sales'].values


from sklearn.linear_model import LinearRegression

simple_regression = LinearRegression()
simple_regression = simple_regression.fit(X_train_preprocessed, y_train)

simple_regression_prediction = simple_regression.predict(X_test_preprocessed)

simple_regression_score = nwrmsle(y_pred=simple_regression_prediction, y_true=y_test, weights=weights)
simple_regression_score


from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge = ridge.fit(X_train_preprocessed, y_train)

ridge_prediction = ridge.predict(X_test_preprocessed)

ridge_score = nwrmsle(y_pred=ridge_prediction, y_true=y_test, weights=weights)
ridge_score


from sklearn.model_selection import train_test_split
X_train_sp, X_val_sp, y_train_sp, y_val_sp = train_test_split(
    X_train_preprocessed, y_train, test_size=0.15, random_state=42
)


import lightgbm as lgb

train_data = lgb.Dataset(X_train_sp, label=y_train_sp)
val_data = lgb.Dataset(X_val_sp, label=y_val_sp, reference=train_data)


params = {
    'num_leaves': 80,
    'objective': 'regression',
    'min_data_in_leaf': 200,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'metric': 'l2',
    'num_threads': 16
}


result = lgb.train(
    params, 
    train_data, 
    num_boost_round=500,
    valid_sets=[train_data, val_data]
)


lgbm_prediction = result.predict(X_test_preprocessed)
lgbm_score = nwrmsle(y_pred=lgbm_prediction, y_true=y_test, weights=weights)
lgbm_score




