import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pickle
import os

from IPython.display import display

from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error



df_sales = pd.read_csv("sales.csv", index_col='Unnamed: 0')
df_online = pd.read_csv('online.csv', index_col='Unnamed: 0')
df_markdowns = pd.read_csv('markdowns.csv', index_col='Unnamed: 0')
df_price_history = pd.read_csv('price_history.csv', index_col='Unnamed: 0')
df_discounts_history = pd.read_csv('discounts_history.csv', index_col='Unnamed: 0')
df_actual_matrix = pd.read_csv('actual_matrix.csv', index_col='Unnamed: 0')
df_catalog = pd.read_csv('catalog.csv', index_col='Unnamed: 0')
df_stores = pd.read_csv('stores.csv', index_col='Unnamed: 0')
df_test = pd.read_csv('test.csv', sep = ';')
df_sample_submission = pd.read_csv('sample_submission.csv')


print('df_sales')
display(df_sales.head(2))

print('df_online')
display(df_online.head(2))

print('df_markdowns')
display(df_markdowns.head(2))

print('df_price_history')
display(df_price_history.head(2))

print('df_discounts_history')
display(df_discounts_history.head(2))

print('df_actual_matrix')
display(df_actual_matrix.head(2))

print('df_catalog')
display(df_catalog.head(2))

print('df_stores')
display(df_stores.head(2))

print('df_test')
display(df_test.head(2))

print('df_sample_submission')
display(df_sample_submission.head(2))


print('df_sales')
display(df_sales.shape)

print('df_online')
display(df_online.shape)

print('df_markdowns')
display(df_markdowns.shape)

print('df_price_history')
display(df_price_history.shape)

print('df_discounts_history')
display(df_discounts_history.shape)

print('df_actual_matrix')
display(df_actual_matrix.shape)

print('df_catalog')
display(df_catalog.shape)

print('df_stores')
display(df_stores.shape)

print('df_test')
display(df_test.shape)

print('df_sample_submission')
display(df_sample_submission.shape)


df_sales.store_id.unique()


# Merge sales data with catalog and store info

data = df_sales.merge(df_catalog, on="item_id", how="left").merge(df_stores, on="store_id", how="left")
data.head()


# Add online sales data

data = data.merge(df_online[["date", "item_id", "store_id", "quantity"]], 
                  on=["date", "item_id", "store_id"], 
                  how="left", 
                  suffixes=("", "_online"))
data.head()


# Add markdown data

data = data.merge(df_markdowns[["date", "item_id", "store_id", "quantity", "price"]], 
                  on=["date", "item_id", "store_id"], 
                  how="left", 
                  suffixes=("", "_markdown"))
data.head()


# Add price history
data = data.merge(df_price_history[["date", "item_id", "store_id", "price"]], 
                  on=["date", "item_id", "store_id"], 
                  how="left", 
                  suffixes=("", "_price_history"))

# Add discounts history
data = data.merge(df_discounts_history[["date", "item_id", "store_id", "promo_type_code"]], 
                  on=["date", "item_id", "store_id"], 
                  how="left")

# Add actual matrix data
data = data.merge(df_actual_matrix[["item_id", "store_id"]], on=["item_id", "store_id"], how="inner")

data.head()


data.columns


df_train = data[['date', 'item_id', 'store_id', 'quantity']]
df_train.head()


df_train.info()


df_train.isnull().sum()


# Feature Engineering

# Convert date to datetime and extract features
df_train['date'] = pd.to_datetime(df_train['date'])
df_train['year'] = df_train['date'].dt.year
df_train['month'] = df_train['date'].dt.month
df_train['day'] = df_train['date'].dt.day
df_train['day_of_week'] = df_train['date'].dt.dayofweek

del df_train['date']

df_train.head()


df_train.info()


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to the column
df_train['item_id'] = label_encoder.fit_transform(df_train['item_id'])


df_train.info()


# Prepare Data for Modeling
X = df_train.drop('quantity', axis = 1)
y = df_train['quantity']



# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)



df_test.head()


# Feature Engineering for test data

# Convert date to datetime and extract features
df_test['date'] = pd.to_datetime(df_test['date'])
df_test['year'] = df_test['date'].dt.year
df_test['month'] = df_test['date'].dt.month
df_test['day'] = df_test['date'].dt.day
df_test['day_of_week'] = df_test['date'].dt.dayofweek

# Apply LabelEncoder to the column
#df_test['item_id'] = label_encoder.transform(df_test['item_id'])

del df_test['date']

df_test.head()


# Ensure unseen labels in test data are handled

def safe_transform(encoder, data):
    # Get the known classes
    known_classes = set(encoder.classes_)
    # Replace unseen labels with a default value or NaN
    data = np.where(data.isin(known_classes), data, 'unknown')
    # Fit the LabelEncoder with the new "unknown" class if necessary
    if 'unknown' not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, 'unknown')
    return encoder.transform(data)

# Apply the LabelEncoder safely to the test data
df_test['item_id'] = safe_transform(label_encoder, df_test['item_id'])



df_test.info()


df_test2 = df_test.copy(deep = True)


del df_test['row_id']


# Get prediction of quantity
y_pred = model.predict(df_test)


df_sample_submission['quantity'] = y_pred

df_sample_submission.to_csv('submit.csv', index=False)

df_sample_submission.head()

