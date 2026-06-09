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


!pip install category_encoders
!pip install tensorflow keras

import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from sklearn.metrics import mean_squared_error
!apt-get install p7zip-full -y


import os
os.listdir('/kaggle/working')


import os
os.listdir('/kaggle/input/data-zip')


train = pd.read_csv('../input/data-zip/train.csv', low_memory=False)


print(train.columns)


!7z x /kaggle/input/favorita-grocery-sales-forecasting/stores.csv.7z -o/kaggle/working/


!7z x /kaggle/input/favorita-grocery-sales-forecasting/holidays_events.csv.7z -o/kaggle/working/
!7z x /kaggle/input/favorita-grocery-sales-forecasting/items.csv.7z -o/kaggle/working/
!7z x /kaggle/input/favorita-grocery-sales-forecasting/test.csv.7z -o/kaggle/working/


holidays_events = pd.read_csv('../working/holidays_events.csv')
items = pd.read_csv('../working/items.csv')
test = pd.read_csv('../working/test.csv')
stores = pd.read_csv('../working/stores.csv')


print(holidays_events.columns)
print(items.columns)
print(test.columns)
print(stores.columns)


print("\nHolidays_event:")
print(holidays_events.head())

print("\nItems:")
print(items.head())

print("\nTest:")
print(test.head())

print("\nStores:")
print(stores.head())


train_items = pd.merge(train, items, how='left', on=['item_nbr'])
print(train_items.head())


print(train_items.head())


mask=train_items['family'] == "LIQUOR,WINE,BEER"
train_items_formatted=train_items.loc[mask]


train_items_formatted.head()


train_items_formatted.to_csv('/kaggle/working/train_items_formatted.csv')


train_items_stores = pd.merge(train_items_formatted, stores, how='left', on=['store_nbr'])
print(train_items_stores.head())


train_items_stores_holidays_events = pd.merge(train_items_stores, holidays_events, how='left', on=['date'])
train_items_stores_holidays_events.head()


unique_values_type_y = train_items_stores_holidays_events['type_y'].unique()
unique_values_transferred = train_items_stores_holidays_events['transferred'].unique()
unique_values_onpromotion = train_items_stores_holidays_events['onpromotion'].unique()


print("\nUnique_values_type_y:")
print(unique_values_type_y)

print("\nTransferred:")
print(unique_values_transferred)

print("\nOnpromotion:")
print(unique_values_onpromotion)

print(len(train_items_stores_holidays_events))


filtered_train_items_stores_holidays_events = train_items_stores_holidays_events[
    (train_items_stores_holidays_events["city"].isin(["Guayaquil", "Quito"]))]
print(len(filtered_train_items_stores_holidays_events))

unique_values_type_y = filtered_train_items_stores_holidays_events['type_y'].unique()
unique_values_transferred = filtered_train_items_stores_holidays_events['transferred'].unique()
unique_values_onpromotion = filtered_train_items_stores_holidays_events['onpromotion'].unique()
unique_values_locale = filtered_train_items_stores_holidays_events['locale'].unique()
unique_values_locale_name = filtered_train_items_stores_holidays_events['locale_name'].unique()
unique_values_description = filtered_train_items_stores_holidays_events['description'].unique()



print("\nUnique_values_type_y:")
print(unique_values_type_y)

print("\nTransferred:")
print(unique_values_transferred)

print("\nOnpromotion:")
print(unique_values_onpromotion)

print("\nLocale:")
print(unique_values_locale)

print("\nLocale_name:")
print(unique_values_locale_name)

print("\nDescription:")
print(unique_values_description)


#filtered_train_items_stores_holidays_events.head()
print(filtered_train_items_stores_holidays_events.info())


# Replace NaN values
filtered_train_items_stores_holidays_events = (
    filtered_train_items_stores_holidays_events.copy()
)
# Convert 'date' to datetime
filtered_train_items_stores_holidays_events['date'] = pd.to_datetime(filtered_train_items_stores_holidays_events['date'])

# Feature Engineering: Extract useful features from 'date'
filtered_train_items_stores_holidays_events['year'] = filtered_train_items_stores_holidays_events['date'].dt.year
filtered_train_items_stores_holidays_events['month'] = filtered_train_items_stores_holidays_events['date'].dt.month
filtered_train_items_stores_holidays_events['day'] = filtered_train_items_stores_holidays_events['date'].dt.day
filtered_train_items_stores_holidays_events['weekday'] = filtered_train_items_stores_holidays_events['date'].dt.weekday

filtered_train_items_stores_holidays_events.fillna({"type_y":"Normal day"}, inplace=True)  # Replace NaN in type_y with "Unknown"
filtered_train_items_stores_holidays_events.fillna({"locale":"Unknown"}, inplace=True)  # Replace NaN in onpromotion with False
filtered_train_items_stores_holidays_events.fillna({"locale_name":"Unknown"}, inplace=True)  # No holiday name
filtered_train_items_stores_holidays_events.fillna({"description":"No Holiday"}, inplace=True)  # No holiday description
filtered_train_items_stores_holidays_events.fillna({"onpromotion": False}, inplace=True)  # Replace NaN in onpromotion with False
filtered_train_items_stores_holidays_events.fillna({"transferred": False}, inplace=True)  # Replace NaN in transferred with False
filtered_train_items_stores_holidays_events['transferred'] = filtered_train_items_stores_holidays_events['transferred'].astype(int)



# Encode categorical variables
encoder = LabelEncoder()
encoded_city = encoder.fit_transform(filtered_train_items_stores_holidays_events['city'])
encoded_type_y = encoder.fit_transform(filtered_train_items_stores_holidays_events['type_y'])

filtered_train_items_stores_holidays_events["encoded_city"] = encoded_city
filtered_train_items_stores_holidays_events["encoded_type_y"] = encoded_type_y


unique_values_transferred = filtered_train_items_stores_holidays_events['city'].unique()

print(unique_values_transferred)


filtered_train_items_stores_holidays_events=filtered_train_items_stores_holidays_events.groupby(['date', 'year', 'month', 'day', 'weekday', 'encoded_city', 'encoded_type_y', 'transferred'], as_index=False).agg({"unit_sales": "sum"})



filtered_train_items_stores_holidays_events.head()


# Normalize/Scale 'unit_sales' and other numeric features
scaler = MinMaxScaler()
filtered_train_items_stores_holidays_events[['unit_sales']] = scaler.fit_transform(filtered_train_items_stores_holidays_events[['unit_sales']])

print(filtered_train_items_stores_holidays_events.head())


# Select features and target
X = filtered_train_items_stores_holidays_events[['year', 'month', 'day', 'weekday', 'encoded_city', 'encoded_type_y', 'transferred']]
y = filtered_train_items_stores_holidays_events['unit_sales']


# Convert data into sequences for LSTM
sequence_length = 10  # Adjust the sequence length based on your needs
generator = TimeseriesGenerator(X.values, y.values, length=sequence_length, batch_size=32)


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# Build the LSTM Model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(sequence_length, X_train.shape[1])))
model.add(Dropout(0.2))
model.add(Dense(1))  # Output layer for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(generator, epochs=50, verbose=1)

# Evaluate the model
X_test_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))  # Adjust dimensions
loss = model.evaluate(X_test_reshaped, y_test, verbose=0)
print(f'Model Loss: {loss}')


import matplotlib.pyplot as plt

# Assuming 'date' column is already in datetime format
plt.figure(figsize=(12, 6))

# Plot training data
plt.plot(filtered_train_items_stores_holidays_events['date'][:len(X_train)], y_train, label="Train Data", color='blue')

# Plot test data
plt.plot(filtered_train_items_stores_holidays_events['date'][len(X_train):], y_test, label="Test Data", color='red')

plt.xlabel('Date')
plt.ylabel('Unit Sales')
plt.title('Train vs Test Data')
plt.legend()
plt.show()



train_items_stores_holidays_events.to_csv('/kaggle/working/train_items_stores_holidays_events.csv')

