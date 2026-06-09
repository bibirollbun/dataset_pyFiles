import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error 


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

data[  data["date"] < THRESHOLD_TRAIN_DATE].to_csv("/kaggle/working/train_data.csv")
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


df.info()
print('\n null vals')
df.isnull().any()


test_df.info()
print('\n null vals')
test_df.isnull().any()


print('Holidays')
display(holiday_events.info())
display(holiday_events.isnull().any())

print('\n items')
display(items.info())
display(items.isnull().any())

print('\n oil')
display(oil.info())
display(oil.isnull().any())

print('\nstores')
display(stores.info())
display(stores.isnull().any())


df["date"].dt.year.value_counts(sort = False).plot.bar()
plt.show()
df2016 = df[df["date"].dt.year == 2016]
df2016["date"].dt.month.value_counts(sort = False).plot.bar()
plt.show()


%xdel df2016


df = df[df["date"] >= pd.to_datetime("2017-01-01")]
df.memory_usage().sum()


import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

store_counts = stores.groupby(['state', 'city' ]).size().reset_index(name='store_count')

fig = px.treemap(
    store_counts,
    path=['state', 'city' ],  
    values='store_count',   
    title='Treemap of Store Counts Across States and Cities',
    color='store_count',    
    color_continuous_scale='Viridis' 
)

fig.show(renderer='iframe')




%xdel store_counts


type_counts = stores['type'].value_counts()
type_counts.plot(kind='bar', color='skyblue', edgecolor='black', figsize=(8, 6))
plt.title('Number of Stores Across Different Types', fontsize=14)
plt.xlabel('Store Type', fontsize=12)
plt.ylabel('Number of Stores', fontsize=12)
plt.show()
%xdel type_counts


type_counts = stores['cluster'].value_counts()
type_counts.plot(kind='bar', color='skyblue', edgecolor='black', figsize=(8, 6))
plt.title('Number of Stores Across Different clusters', fontsize=14)
plt.xlabel('Store cluster', fontsize=12)
plt.ylabel('Number of Stores', fontsize=12)
plt.show()
%xdel type_counts


df = df.merge(stores , on = 'store_nbr' , how = 'left')
display(df.head())


sales_by_state_city = df.groupby(['state', 'city' ])['unit_sales'].sum().reset_index(name='sales_by_state_city')

fig = px.treemap(
    sales_by_state_city,
    path=['state', 'city' ],  
    values='sales_by_state_city',   
    title='Treemap of unit sales Across States and Cities',
    color='sales_by_state_city',    
    color_continuous_scale='Viridis' 
)

fig.show(renderer='iframe')

%xdel sales_by_state_city


type_counts = df.groupby(['type'])['unit_sales'].sum()
type_counts.plot(kind='bar', color='skyblue', edgecolor='black', figsize=(8, 6) )
plt.title('Number of unit sales Across Different Types', fontsize=14)
plt.xlabel('Store Type', fontsize=12)
plt.ylabel('unit sales', fontsize=12)
plt.show()


type_counts = df.groupby(['cluster'])['unit_sales'].sum()
type_counts.plot(kind='bar', color='skyblue', edgecolor='black', figsize=(8, 6) )
plt.title('Number of unit sales Across Different Clusters', fontsize=14)
plt.xlabel('Store Cluster', fontsize=12)
plt.ylabel('unit sales', fontsize=12)
plt.show()

%xdel type_counts




test_df = test_df.merge(stores , on = 'store_nbr' , how = 'left')


%xdel stores


holidays_counts = holiday_events['type'].value_counts()
holidays_counts.plot(kind='bar', color='skyblue', edgecolor='black', figsize=(8, 6))
plt.title('Number of different holiday types ', fontsize=14)
plt.xlabel('type', fontsize=12)
plt.ylabel('Number', fontsize=12)
plt.show()
%xdel holidays_counts


holiday_events['is_holiday'] = holiday_events['type'] == 'Holiday'
holiday_events.drop(columns=['type'], inplace=True)
holiday_events = holiday_events[holiday_events['is_holiday']]



display(holiday_events.head(20))


holidays_counts = holiday_events['locale'].value_counts()
holidays_counts.plot(kind='bar', color='skyblue', edgecolor='black', figsize=(8, 6))
plt.title('Number of different holiday locale types ', fontsize=14)
plt.xlabel('locale', fontsize=12)
plt.ylabel('Number', fontsize=12)
plt.show()
%xdel holidays_counts


national_holidays = holiday_events[holiday_events['locale'] == 'National']
local_holidays = holiday_events[holiday_events['locale'] == 'Local']
regional_holidays = holiday_events[holiday_events['locale'] == 'Regional']
%xdel holiday_events



print("National Holidays Count:")
print(national_holidays['locale'].value_counts())

print("\nLocal Holidays Count:")
print(local_holidays['locale'].value_counts())

print("\nRegional Holidays Count:")
print(regional_holidays['locale'].value_counts())


df['date'] = pd.to_datetime(df['date'])

national_holidays['date'] = pd.to_datetime(national_holidays['date'])
regional_holidays['date'] = pd.to_datetime(regional_holidays['date'])
local_holidays['date'] = pd.to_datetime(local_holidays['date'])

regional_holidays_dict = regional_holidays.set_index('date')['locale_name'].to_dict()
local_holidays_dict = local_holidays.set_index('date')['locale_name'].to_dict()

def merge_holidays(data):
    data['is_holiday'] = data['date'].isin(national_holidays['date'])
    
    data['is_holiday'] |= data['date'].map(regional_holidays_dict) == data['state']

    
    data['is_holiday'] |= data['date'].map(local_holidays_dict) == data['city']

merge_holidays(df)
display(df.head(10))

merge_holidays(test_df)
display(test_df.head(10))


item_family_counts = items['family'].value_counts()
item_family_counts.plot(kind='bar', color='skyblue', edgecolor='black', figsize=(8, 6))
plt.title('number of items in different family', fontsize=14)
plt.xlabel('family', fontsize=12)
plt.ylabel('number', fontsize=12)
plt.show()
%xdel item_family_counts


df = df.merge(items , on = 'item_nbr' , how = 'left')
display(df.head())


test_df = test_df.merge(items , on = 'item_nbr' , how = 'left')
display(test_df.head())


family_counts = df.groupby(['family'])['unit_sales'].sum()
family_counts.plot(kind='bar', color='skyblue', edgecolor='black', figsize=(8, 6) )
plt.title('Number of unit sales Across Different item family', fontsize=14)
plt.xlabel('item family ', fontsize=12)
plt.ylabel('unit sales', fontsize=12)
plt.show()



%xdel items


oil['date'] = pd.to_datetime(oil['date']) 
oil_2016 = oil[oil['date'].dt.year == 2016]  

plt.figure(figsize=(10, 5))
plt.plot(oil_2016['date'], oil_2016['dcoilwtico'], marker='o', linestyle='-', label='Oil Price')
plt.title('Oil Prices in 2016')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


%xdel oil_2016


oil["dcoilwtico"].ffill(inplace=True)
print(oil.isnull().sum())


df = df.merge(oil, how="left", on="date")
test_df = test_df.merge(oil, how="left", on="date")
display(df.head())
display(test_df.head())


df["dcoilwtico"].ffill(inplace=True)
test_df["dcoilwtico"].ffill(inplace=True)
df["dcoilwtico"].bfill(inplace=True)
test_df["dcoilwtico"].bfill(inplace=True)


df["unit_sales"].corr(df["dcoilwtico"])


display(df.isnull().sum())
display(test_df.isnull().sum())


dummy_regressor = DummyRegressor(strategy="mean")
dummy_regressor.fit(df, df['unit_sales'].values)

display(mean_squared_error(dummy_regressor.predict(test_df) , test_df['unit_sales']))


df.head()


from sklearn.preprocessing import LabelEncoder

categorical_cols = ['store_nbr', 'city', 'state', 'type', 'cluster', 'family', 'is_holiday']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le 
    test_df[col] = le.fit_transform(test_df[col])




df['date'] = pd.to_datetime(df['date']) 
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek

test_df['date'] = pd.to_datetime(test_df['date']) 
test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['day_of_week'] = test_df['date'].dt.dayofweek


df = df.drop(['id', 'Unnamed: 0', 'date'], axis=1)
test_df = test_df.drop(['id', 'Unnamed: 0', 'date'], axis=1)

df['onpromotion'] = df['onpromotion'].astype(bool)


X_train = df.drop(['unit_sales'], axis=1)  
Y_train = df['unit_sales']  

X_test = test_df.drop(['unit_sales'], axis=1) 
Y_test = test_df['unit_sales']  




%xdel df 
    
%xdel test_df



train_data = lgb.Dataset(X_train, label=Y_train)
test_data = lgb.Dataset(X_test, label=Y_test, reference=train_data)


params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

lgbm_model = lgb.train(
    params, 
    train_data, 
    num_boost_round=1000,
    valid_sets=[train_data, test_data]
)



Y_pred = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration)
rmse = np.sqrt(mean_squared_error(Y_test,Y_pred))  
print(f"RMSE: {rmse}")

