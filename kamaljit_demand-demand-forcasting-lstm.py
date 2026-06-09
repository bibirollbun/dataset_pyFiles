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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# read training data
train = pd.read_csv("/kaggle/input/demand-forecasting-kernels-only/train.csv")
# read test data
test = pd.read_csv("/kaggle/input/demand-forecasting-kernels-only/test.csv")


# shape of the data
train.shape, test.shape


# null check
train.isnull().sum()


# duplicate check
train.duplicated().sum()


train.head()


# information of the dataset
train.info()


# change the data type of date feature
train['date'] = pd.to_datetime(train['date'])
# create year feature
train["year"] = train['date'].dt.year
# create month feature
train["month"] = train['date'].dt.month


# drop date feature
train.drop(columns="date", axis=1, inplace=True)


# top 5 rows
train.head()


train['store'].value_counts()


# this data is for these many years
train['year'].value_counts()


# sale distribution of years
sns.countplot(data=train, x='year')
plt.title("Sales distribution for years")
plt.xlabel("Years")
plt.ylabel("Count of frequency")
plt.plot()


# sale distribution of years
sns.lineplot(data=train, x='year', y = "sales")
plt.title("Sales distribution for years")
plt.xlabel("Years")
plt.ylabel("Count of frequency")
plt.plot()


# sale distribution of years
sns.lineplot(data=train, x='year', y = "sales", hue="store")
plt.title("Sales distribution for years")
plt.xlabel("Years")
plt.ylabel("Count of frequency")
plt.plot()


# pair plot of the dataset
sns.pairplot(train)
plt.plot()


corr_matrix = train.corr()
corr_matrix


# correlation with sotore
corr_matrix['store']


# heatmap of the corr_matirx
sns.heatmap(corr_matrix)
plt.show()


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split


# Load your dataset
data = pd.read_csv("/kaggle/input/demand-forecasting-kernels-only/train.csv")


data.head()


data.shape


# change the data type of date feature
data['date'] = pd.to_datetime(data['date'])
# create year feature
data["year"] = data['date'].dt.year
# create month feature
data["month"] = data['date'].dt.month
# day
# data["day"] = data['date'].dt.day


data.head()


# Filter data for a specific store (e.g., store 1)
store_id = 1
store_data = data[data["store"] == store_id]


# Aggregate sales by year and month
store_data = store_data.groupby(["year", "month"])["sales"].sum().reset_index()


store_data.shape


# Create a date column for easier time series handling
store_data["date"] = pd.to_datetime(store_data["year"].astype(str) + "-" + store_data["month"].astype(str))


# Sort by date
store_data = store_data.sort_values("date")


# Set date as the index
store_data.set_index("date", inplace=True)


# Use only the 'sales' column for forecasting
sales_data = store_data[["sales"]]


# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(sales_data)


# Prepare the data for LSTM
def create_dataset(data, time_step):
    X = []
    y = []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])  # Input sequence
        y.append(data[i + time_step, 0])      # Target value
    return np.array(X), np.array(y)


X, y = create_dataset(data=scaled_data, time_step=12)


# Reshape X to be [samples, time steps, features] (required for LSTM)
X = X.reshape(X.shape[0], X.shape[1], 1)


# split the data inot training and testing sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size

X_train, X_test = X[0: train_size], X[train_size:len(X)]
y_train, y_test = y[0: train_size], y[train_size:len(y)]


# # Build the LSTM model
time_step = 12
model = Sequential()
model.add(LSTM(500, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(l2=0.01), input_shape=(time_step, 1)))  # First LSTM layer
model.add(LSTM(500, return_sequences=False, kernel_regularizer=tf.keras.regularizers.L2(l2=0.01)))  # Second LSTM layer
model.add(Dense(250, kernel_regularizer=tf.keras.regularizers.L2(l2=0.01)))  # Dense layer
model.add(Dense(1))   # Output layer


# model summary
model.summary()


# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")


# Train the model
history = model.fit(X_train, y_train, validation_split=0.25, batch_size=16, epochs=50, verbose=1)


pd.DataFrame(history.history).plot()
plt.plot()


# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# Inverse transform the predictions to original scale
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))


# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(sales_data.index, scaler.inverse_transform(scaled_data), label="Actual Sales", alpha=0.7)
plt.plot(sales_data.index[time_step:train_size + time_step], train_predict[:train_size], label="Train Predictions", alpha=0.7)
plt.plot(sales_data.index[train_size + time_step:train_size + time_step + len(test_predict)], test_predict, label="Test Predictions", alpha=0.7)
plt.title(f"Store {store_id} Sales Forecasting Using LSTM")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()


# Evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, test_predict)
mse = mean_squared_error(y_test, test_predict)
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")


pd.read_csv("/kaggle/input/demand-forecasting-kernels-only/sample_submission.csv")


test.head()


# change date to datetime data type
test['date'] = pd.to_datetime(test['date'])
# create year feature
test['year'] = test['date'].dt.year
# create month feature
test['month'] = test['date'].dt.month


test.head()










