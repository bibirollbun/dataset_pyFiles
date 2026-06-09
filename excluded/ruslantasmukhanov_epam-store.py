import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import sys

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import make_scorer
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint, uniform


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


!pip install py7zr --quiet


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
THRESHOLD_TRAIN_DATE_FROM = pd.to_datetime("2017-01-01")
THRESHOLD_TRAIN_DATE = pd.to_datetime("2017-07-01")
THRESHOLD_TEST_DATE = pd.to_datetime("2017-08-01")

data[(THRESHOLD_TRAIN_DATE_FROM <= data["date"])&(data["date"] < THRESHOLD_TRAIN_DATE)].to_csv("/kaggle/working/train_data.csv")
data[(data["date"] >= THRESHOLD_TRAIN_DATE)&(data["date"] < THRESHOLD_TEST_DATE)].to_csv("/kaggle/working/test_data.csv")

%xdel data


%%time
train_df = pd.read_csv("/kaggle/working/train_data.csv")
test_df = pd.read_csv("/kaggle/working/test_data.csv")


train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])


# import sys
# import gc

# # Collect garbage to remove unreferenced objects
# gc.collect()

# # List all objects and their sizes in memory
# for var_name, var_obj in globals().items():
#     try:
#         if sys.getsizeof(var_obj) / (1024 ** 2) > 100:
#             print(f"{var_name}: {sys.getsizeof(var_obj) / (1024 ** 2):.2f} MB")
#     except TypeError:
#         pass


monthly_counts = train_df['date'].dt.to_period('M').value_counts().sort_index().reset_index()
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


train_df.head()


holiday_events = pd.read_csv("/kaggle/working/holidays_events.csv")
print(holiday_events.shape)
holiday_events.head()


holiday_events.isna().sum()


items = pd.read_csv("/kaggle/working/items.csv")
print(items.shape)
items.head()


items.isna().sum()


oil = pd.read_csv("/kaggle/working/oil.csv")
print(oil.shape)
oil.head()


oil.isna().sum()


oil = oil.dropna(subset=['dcoilwtico'])


oil.isna().sum()


stores = pd.read_csv("/kaggle/working/stores.csv")
print(stores.shape)
stores.head()


stores.isna().sum()


train_df.dtypes


print("Number of NaN in train_df['unit_sales']:", train_df['unit_sales'].isna().sum())
print("Number of NaN in train_df['unit_sales']:", test_df['unit_sales'].isna().sum())


print(train_df.shape, test_df.shape)
print("train/test ratio:", train_df.shape[0] / test_df.shape[0])


train_df.info()
train_df.isnull().sum()
train_df.describe()


item_counts = train_df['item_nbr'].value_counts()
item_bins = pd.cut(item_counts, bins=10).value_counts().sort_index()

plt.figure(figsize=(12, 6))
item_bins.plot(kind='bar', color='skyblue', edgecolor='black')

plt.title('Distribution of Item Sales (Grouped)', fontsize=16)
plt.xlabel('Number of Sales (Binned)', fontsize=14)
plt.ylabel('Number of Items', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# converting to datetime
holiday_events['date'] = pd.to_datetime(holiday_events['date'])
oil['date'] = pd.to_datetime(oil['date'])


print(holiday_events.dtypes)
print("-------")
print(oil.dtypes)


train_df = train_df.merge(items, on='item_nbr', how='left')
test_df = test_df.merge(items, on='item_nbr', how='left')

print("Train shape after merging items:", train_df.shape)
print("Test shape after merging items:", test_df.shape)


train_df = train_df.merge(stores, on='store_nbr', how='left')
test_df = test_df.merge(stores, on='store_nbr', how='left')


oil.isnull().sum()


oil = oil.sort_values('date').fillna(method='ffill')


train_df = train_df.merge(oil, on='date', how='left')
test_df = test_df.merge(oil, on='date', how='left')


train_df.isna().sum()


train_df.info()


holiday_events


holidays = holiday_events


holiday_counts = holidays['locale'].value_counts()

plt.figure(figsize=(8, 5))
sns.barplot(x=holiday_counts.index, y=holiday_counts.values, palette="viridis")

plt.title("Number of Holidays by Type", fontsize=16)
plt.xlabel("Holiday Type", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()


holidays = holiday_events

# national holidays
holiday_national = holidays[holidays['locale'] == 'National'].copy()
holiday_national['is_holiday_national'] = 1
holiday_national = holiday_national[['date', 'is_holiday_national', 'type', 'transferred']]
holiday_national.rename(columns={'type': 'type_national'}, inplace=True)

# regional holidays
holiday_regional = holidays[holidays['locale'] == 'Regional'].copy()
holiday_regional['is_holiday_regional'] = 1
holiday_regional = holiday_regional[['date', 'locale_name', 'is_holiday_regional', 'type', 'transferred']]
holiday_regional.rename(columns={'type': 'type_regional'}, inplace=True)


# local holidays
holiday_local = holidays[holidays['locale'] == 'Local'].copy()
holiday_local['is_holiday_local'] = 1
holiday_local = holiday_local[['date', 'locale_name', 'is_holiday_local', 'type', 'transferred']]
holiday_local.rename(columns={'type': 'type_local'}, inplace=True)



holiday_local.rename(columns={'transferred': 'transferred_local'}, inplace=True)


assert(holidays.shape[0] == holiday_national.shape[0] + holiday_regional.shape[0] + holiday_local.shape[0])


print(holiday_national.dtypes)
print(holiday_regional.dtypes)
print(holiday_local.dtypes)


# merging local holidays
train_df = train_df.merge(
    holiday_local, 
    left_on=['date', 'city'],
    right_on=['date', 'locale_name'],
    how='left'
)
test_df = test_df.merge(
    holiday_local, 
    left_on=['date', 'city'],
    right_on=['date', 'locale_name'],
    how='left'
)
train_df['is_holiday_local'] = train_df['is_holiday_local'].fillna(0)
test_df['is_holiday_local'] = test_df['is_holiday_local'].fillna(0)


# merging national holidays
train_df = train_df.merge(holiday_national, on='date', how='left')
test_df  = test_df.merge(holiday_national, on='date', how='left')
train_df['is_holiday_national'] = train_df['is_holiday_national'].fillna(0)
test_df['is_holiday_national'] = test_df['is_holiday_national'].fillna(0)


train_df[train_df["is_holiday_national"] == 1].shape


# merging regional holidays
train_df = train_df.merge(
    holiday_regional, 
    left_on=['date', 'state'],
    right_on=['date', 'locale_name'],
    how='left'
)
test_df = test_df.merge(
    holiday_regional, 
    left_on=['date', 'state'],
    right_on=['date', 'locale_name'],
    how='left'
)

train_df['is_holiday_regional'] = train_df['is_holiday_regional'].fillna(0)
test_df['is_holiday_regional'] = test_df['is_holiday_regional'].fillna(0)


train_df.dtypes


# Average unit sales by state
state_sales = train_df.groupby('state')['unit_sales'].mean().reset_index().sort_values(by='unit_sales', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=state_sales, x='state', y='unit_sales', palette='coolwarm')
plt.title('Average Unit Sales by State')
plt.xlabel('State')
plt.ylabel('Average Unit Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Average unit sales by city
city_sales = train_df.groupby('city')['unit_sales'].mean().reset_index().sort_values(by='unit_sales', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=city_sales, x='city', y='unit_sales', palette='coolwarm')
plt.title('Average Unit Sales by City')
plt.xlabel('City')
plt.ylabel('Average Unit Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


state_mapping = {
    'Pichincha': 0,
    'Pastaza': 2
}

city_mapping = {
    'Quito': 0,
    'Cayambe': 0,
    'Riobamba': 2,
    'Salinas': 2,
    'Santa Dormingo': 2,
    'Ibarra': 2,
    'Playas': 2,
    'Latacunga': 2,
    'Puyo': 2
}

train_df['state'] = train_df['state'].astype('category')
test_df['state'] = test_df['state'].astype('category')

train_df['city'] = train_df['city'].astype('category')
test_df['city'] = test_df['city'].astype('category')

train_df['state_class'] = train_df['state'].map(state_mapping).fillna(1).astype('uint8')
test_df['state_class'] = test_df['state'].map(state_mapping).fillna(1).astype('uint8')

train_df['city_class'] = train_df['city'].map(city_mapping).fillna(1).astype('uint8')
test_df['city_class'] = test_df['city'].map(city_mapping).fillna(1).astype('uint8')

train_df.drop(columns=['state', 'city'], inplace=True)
test_df.drop(columns=['state', 'city'], inplace=True)

gc.collect()


train_df.head()


drop_columns = ['locale_name_x', 'locale_name_y', 'transferred_local', 'transferred_x', 'transferred_y', 'type_local', 'type_national', 'type_regional', 'id', 'Unnamed: 0']

train_df.drop(columns=drop_columns, inplace=True)
test_df.drop(columns=drop_columns, inplace=True)


gc.collect()


test_df.head()


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_col='date'):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col])
        X['year'] = X[self.date_col].dt.year
        X['month'] = X[self.date_col].dt.month
        X['day'] = X[self.date_col].dt.day
        X['day_of_week'] = X[self.date_col].dt.dayofweek
        X['day_of_year'] = X[self.date_col].dt.dayofyear
        X['is_weekend'] = (X[self.date_col].dt.dayofweek >= 5).astype('uint8')
        X.drop(columns=[self.date_col], inplace=True)
        return X

    def get_feature_names_out(self, input_features=None):
        return ['year', 'month', 'day', 'day_of_week', 'day_of_year', 'is_weekend']


X_train = train_df.drop(columns=['unit_sales'])
y_train = train_df['unit_sales']

X_val = test_df.drop(columns=['unit_sales'])
y_val = test_df['unit_sales']


X_train.dtypes


def convert_categorical_features(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    for col in cat_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(-1)
        else:
            df[col] = df[col].fillna("missing")
        
        if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            continue
        
        if pd.api.types.is_float_dtype(df[col]):
            all_integers = df[col].apply(lambda x: x.is_integer() if not pd.isna(x) else True).all()
            if all_integers:
                df[col] = df[col].astype('uint16')
            else:
                df[col] = df[col].astype(str)
        else:
            df[col] = df[col].astype(str)
    
    return df


# dropping train and test data becaues I have no memory...
del train_df
del test_df

gc.collect()


numerical_features = [
    'store_nbr', 'item_nbr', 'dcoilwtico', 
    'year','month','day','day_of_week','day_of_year' 
    # these ones will be added after date extracting in pipeline
]

categorical_features = [
    'onpromotion', 'family', 'class', 'state_class', 'city_class', 'type',
    'is_holiday_local', 'is_holiday_national', 'is_holiday_regional',
    'cluster', 'perishable',
    'is_weekend' # this one will be added after date extracting in pipeline
]


X_train = convert_categorical_features(X_train, categorical_features[:-1])
X_val = convert_categorical_features(X_val, categorical_features[:-1])

gc.collect()


numeric_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)


model_dummy = DummyRegressor(strategy='mean')

pipeline_dummy = Pipeline([
    ('date_feats', DateFeatureExtractor(date_col='date')),
    ('preprocessor', preprocessor),
    ('regressor', model_dummy)
])

# handling cases where log function takes negative or zero values
def safe_log1p(x):
    return np.log1p(np.clip(x, 1e-15, None))

def safe_expm1(x):
    return np.expm1(np.clip(x, None, 709))

# wrapping with log transform
pipeline_dummy = TransformedTargetRegressor(
    regressor=pipeline_dummy,
    func=safe_log1p,
    inverse_func=safe_expm1
)


model_lgb = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42
)

pipeline_lgb = Pipeline([
    ('date_feats', DateFeatureExtractor(date_col='date')),
    ('preprocessor', preprocessor),
    ('regressor', model_lgb)
])

# wrapping with log transformer on target field
pipeline_lgb = TransformedTargetRegressor(
    regressor=pipeline_lgb,
    func=safe_log1p,
    inverse_func=safe_expm1
)


preprocessor_cat = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features)
    ],
    remainder='passthrough'
)


feature_extraction_pipeline = Pipeline([
    ('date_feats', DateFeatureExtractor(date_col='date')),
    ('preprocessor', preprocessor_cat)
])


X_temp = feature_extraction_pipeline.named_steps['date_feats'].transform(X_train.head())
feature_extraction_pipeline.named_steps['preprocessor'].fit(X_temp)
col_names_for_preprocessor = X_temp.columns
feature_names = feature_extraction_pipeline.named_steps['preprocessor'].get_feature_names_out(col_names_for_preprocessor)

categorical_indices = [i for i, feature in enumerate(feature_names) 
                      if any(cat_feat in feature for cat_feat in categorical_features)]

print("Categorical feature indices:", categorical_indices)


model_cat = CatBoostRegressor(
    iterations=250,
    learning_rate=0.1,
    random_seed=42,
    cat_features=categorical_indices,
    verbose=False
)

pipeline_cat = Pipeline([
    ('date_feats', DateFeatureExtractor(date_col='date')),
    ('preprocessor', preprocessor_cat),
    ('regressor', model_cat)
])

pipeline_cat = TransformedTargetRegressor(
    regressor=pipeline_cat,
    func=safe_log1p,
    inverse_func=safe_expm1
)


pipeline_dummy.fit(X_train, y_train)


pipeline_lgb.fit(X_train, y_train)


# pipeline_cat.fit(X_train, y_train)


dummy_val_preds = pipeline_dummy.predict(X_val)


lgb_val_preds = pipeline_lgb.predict(X_val)


# cat_val_preds = pipeline_cat.predict(X_val)


def nwrmsle(y_true, y_pred, weights):
    # ensuring no negative predictions are seen
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    y_true = np.clip(y_true, a_min=0, a_max=None)
    
    log_diff = np.log1p(y_pred) - np.log1p(y_true)
    weighted_squared_log_diff = weights * (log_diff ** 2)
    score = np.sqrt(np.sum(weighted_squared_log_diff) / np.sum(weights))
    
    return score


perishable_weights_val = np.where(X_val['perishable'] == 1, 1.25, 1.00)


dummy_val_score = nwrmsle(y_val.values, dummy_val_preds, perishable_weights_val)
print("Dummy NWRMSLE Score:", dummy_val_score)


lgb_val_score = nwrmsle(y_val.values, lgb_val_preds, perishable_weights_val)
print("LGB NWRMSLE Score:", lgb_val_score)




