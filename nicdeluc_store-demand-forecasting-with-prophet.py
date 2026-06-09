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


import logging

# Get the logger for cmdstanpy
logger = logging.getLogger('cmdstanpy')

# Set the level to WARNING to hide informational messages
logger.setLevel(logging.WARNING)


train_df = pd.read_csv("/kaggle/input/demand-forecasting-kernels-only/train.csv")
X_test = pd.read_csv("/kaggle/input/demand-forecasting-kernels-only/test.csv", index_col="id").reset_index(drop=True)


print("Train set size:", train_df.shape)
train_df.head()


print("Test set size:", X_test.shape)
X_test.head()


print(train_df.info())
print("\n")
print(X_test.info())


# Delete duplicates of the train set
train_df.drop_duplicates(inplace=True)
print(train_df.shape)


train_df["date"] = pd.to_datetime(train_df["date"])
X_test["date"] = pd.to_datetime(X_test["date"])


# Get the number of stores and items
n_stores = train_df["store"].nunique()
n_items = train_df["item"].nunique()


# Construct a train and test sets for each store and each item (total: 500)
train_sets = {}
test_sets = {} 

for store in range(1, n_stores + 1):
    for item in range(1, n_items + 1):

        train_sets[(store, item)] = train_df[["date", "sales"]][(train_df["store"] == store) & (train_df["item"] == item)].copy()
        train_sets[(store, item)].rename(columns={"date": "ds", "sales": "y"}, inplace=True)
        
        test_sets[(store, item)] = X_test[["date"]][(X_test["store"] == store) & (X_test["item"] == item)].copy()
        test_sets[(store, item)].rename(columns={"date": "ds"}, inplace=True)


import random
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline


# Take a random sample of keys of the train_sets dictionary
key_sample = random.sample(list(train_sets.keys()), 4)


# Plot the time series
fig, axes = plt.subplots(4, 1, figsize=(15, 15))

for ax, key in zip(axes.flatten(), key_sample):
    store, item = key
    train_sets[key].set_index('ds').plot(ax=ax, title=f"Sales of item {item} in store {store}")

plt.tight_layout()
plt.show()


from prophet import Prophet


models = {}

# Train the models and store them in a dictionary
for key in train_sets.keys():
    models[key] = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    models[key].fit(train_sets[key])


separate_forecasts = {}

# Make predictions for each model and test set
for (store, item), model in models.items():
    future_df = test_sets[(store, item)]
    
    prediction = model.predict(future_df[["ds"]])
    
    # Select only the columns we need (date and prediction)
    forecast = prediction[["ds", "yhat"]].copy()
    forecast["store"] = store
    forecast["item"] = item

    separate_forecasts[(store, item)] = forecast


# Plot the predictions with their original time series
fig, axes = plt.subplots(4, 1, figsize=(15, 15), sharex=True)

for ax, key in zip(axes.flatten(), key_sample):
    store, item = key
    
    historical_data = train_sets[key]
    forecast_data = models[key].predict(test_sets[key]) 

    # Plot historical data 
    historical_data.plot(kind='line', x='ds', y='y', color='black',
                         ax=ax, label='Historical Data')
    
    # Plot the forecasts
    forecast_data.plot(kind='line', x='ds', y='yhat', color='blue',
                       ax=ax, label='Forecast')

    ax.set_title(f"Sales of item {item} in store {store}")
    ax.legend()

plt.tight_layout()
plt.show()


# Join all the predictions into a single data frame
forecasts = pd.concat(list(separate_forecasts.values())).rename(columns={"ds": "date", "yhat": "sales"})


final_forecasts = pd.merge(X_test, forecasts, how="left")
final_forecasts["id"] = final_forecasts.index                # Need this column for submission
final_forecasts


sample_submission = pd.read_csv("/kaggle/input/demand-forecasting-kernels-only/sample_submission.csv")
sample_submission


submission = final_forecasts[["id", "sales"]].to_csv("submission.csv", index=False)

