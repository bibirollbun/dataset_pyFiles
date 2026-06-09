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


pd.set_option('display.max_columns', None)


train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
train.head()


print(train.shape)
print(test.shape)


print(train.dtypes)


categorical_cols = train.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"--- {col} ---")
    print(train[col].value_counts())
    print("\n")


train.describe() # for non-object cols


import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(train.isnull(), cbar=False, cmap='viridis')
print(train.isnull().sum())


id_test = test["id"]
train = train.dropna()
train = train.drop("id", axis = 1)
test = test.drop("id", axis = 1)
dict_country = {"Kenya": 0, "Italy": 1, "Canada": 2, "Singapore": 3, "Finland": 4, "Norway": 5}
dict_store = {"Discount Stickers": 0, "Stickers for Less": 1, "Premium Sticker Mart": 2}
dict_product = {"Holographic Goose": 0, "Kaggle": 1, "Kaggle Tiers": 2, "Kerneler": 3, "Kerneler Dark Mode": 4}
train["country"] = train["country"].map(dict_country)
train["store"] = train["store"].map(dict_store)
train["product"] = train["product"].map(dict_product)
test["country"] = test["country"].map(dict_country)
test["store"] = test["store"].map(dict_store)
test["product"] = test["product"].map(dict_product)
print(train.head())
print(test.head())


print(train.shape)
print(train.dtypes)


train["num_sold"].hist(bins=100)
plt.title("Original")
plt.show()


np.log1p(train["num_sold"].copy()).hist(bins=100)
plt.title("log(1+x)")
plt.show()


np.sqrt(train["num_sold"].copy()).hist(bins=100)
plt.title("sqrt(x)")
plt.show()


train["log_num_sold"] = np.log1p(train["num_sold"])


train["date"] = pd.to_datetime(train["date"])
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day

reference_date = pd.Timestamp("2009-12-31")
train['time_from_2009_12_31'] = (train['date'] - reference_date).dt.days


test["date"] = pd.to_datetime(test["date"])
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day

reference_date = pd.Timestamp("2009-12-31")
test['time_from_2009_12_31'] = (test['date'] - reference_date).dt.days


top_50_peaks = train.nlargest(50, 'num_sold')
print(top_50_peaks)


for product in train['product'].unique():
    subset = train[train['product'] == product]
    plt.figure(figsize=(8, 5))
    
    # Plot the data
    plt.plot(
        subset['time_from_2009_12_31'],
        subset['num_sold'],
        label=f'Product {product}',
        linewidth=0.2
    )

    for i in range(1, 8):  # Loop to create lines for 365, 365*2, ..., 365*4
        plt.axvline(x=365 * i, color='red', linestyle='--', linewidth=0.8, label=f'x = {365 * i}' if i == 1 else None)
        
    # Add title, labels, and legend
    plt.title(f'Product {product}: Number Sold vs. Time')
    plt.xlabel('Time from 2009-12-31 (days)')
    plt.ylabel('Number Sold')
    plt.legend()
    plt.grid(True)
    plt.show()


import math

def calculate_k(row):
    t = row['time_from_2009_12_31']
    if row['product'] == 0:
        return 0.2 * np.cos(2 * math.pi * t / 365)
    elif row['product'] == 1:
        return 0.8 * np.sin(2 * math.pi * (t - 365) / 730)
    elif row['product'] == 2:
        return 0.8 * np.sin(2 * math.pi * (t + 183) / 730)
    elif row['product'] == 3:
        return -0.8 * np.sin(2 * math.pi * t / 365)
    elif row['product'] == 4:
        return 0.9 * np.sin(2 * math.pi * t / 365)
    else:
        return 0


train['sell_multiplier'] = train.apply(calculate_k, axis=1)
train['is_month_end'] = train['day'].isin({27, 28, 29, 30, 31, 1, 2, 3, 4}).astype(int)
train['is_end_start_year'] = (
    ((train['month'] == 12) & (train['day'].isin({29, 30, 31}))) |  # End of the year
    ((train['month'] == 1) & (train['day'].isin({1, 2, 3})))       # Start of the year
).astype(int)

test['sell_multiplier'] = test.apply(calculate_k, axis=1)
test['is_month_end'] = test['day'].isin({27, 28, 29, 30, 31, 1, 2, 3, 4}).astype(int)
test['is_end_start_year'] = (
    ((test['month'] == 12) & (test['day'].isin({29, 30, 31}))) |  # End of the year
    ((test['month'] == 1) & (test['day'].isin({1, 2, 3})))       # Start of the year
).astype(int)


train = train.drop(columns = ["date", "year", "month", "day", "num_sold"])
test = test.drop(columns = ["date", "year", "month", "day"])


train["time_from_2009_12_31"] = train["time_from_2009_12_31"]
test["time_from_2009_12_31"] = test["time_from_2009_12_31"]


print(train.dtypes)
print(test.dtypes)


# experiment with feature selection
def select_important_features(df, columns = ['country', 'store', 'product', 'is_month_end', 'sell_multiplier', 'is_end_start_year', 'time_from_2009_12_31']):
    return df[columns]


import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


y = train["log_num_sold"]
X = select_important_features(train)

X = X[sorted(X.columns)]
test = test[sorted(test.columns)]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=114514)


model = xgb.XGBRegressor(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=7,
    random_state=114514,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse:.4f}")


#import optuna
#import xgboost as xgb
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
#import numpy as np
#
## Define the objective function for Optuna
#def objective(trial):
#    param = {
#        'n_estimators': trial.suggest_int('n_estimators', 200, 300),
#        'learning_rate': trial.suggest_float('learning_rate', 0.07, 0.2),
#        'max_depth': 7,
#        'random_state': 114514
#    }
#
#    # Split data
#    y = train["log_num_sold"]
#    X = select_important_features(train)
#
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=114514)
#
#    # Create and train the model
#    model = xgb.XGBRegressor(**param)
#    model.fit(X_train, y_train)

#    # Predict and calculate RMSE
#    y_pred = model.predict(X_test)
#    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#    print(f"Trial {trial.number}: RMSE = {rmse:.4f}")
#    return rmse
#
## Create the Optuna study
#study = optuna.create_study(direction='minimize')
#study.optimize(objective, n_trials=50)
#
## Print the best parameters and score
#print("Best Parameters:", study.best_params)
#print(f"Best RMSE: {study.best_value:.4f}")



prediction = np.exp(model.predict(test)) - 1
print(prediction)


submission = pd.DataFrame({'id': id_test, 'num_sold': prediction})
submission.to_csv("submission.csv", index=False)

