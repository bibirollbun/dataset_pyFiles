import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_percentage_error

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# %% DATA
train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv", index_col='id')


#%% INFOS & BASIC ETL
train.info()

train.isna().mean()
test.isna().mean()

train.dropna(inplace=True)


#%% ETL DATE
def data_convert(x, y):
    
    # convert date to datetime
    x['date'] = pd.to_datetime(x['date'])
    y['date'] = pd.to_datetime(y['date'])
    
    # create new feature w/ datatime
    x['year'] = x['date'].dt.year
    x['month'] = x['date'].dt.month
    x['day_of_weekday'] = x['date'].dt.weekday
    
    y['year'] = y['date'].dt.year
    y['month'] = y['date'].dt.month
    y['day_of_weekday'] = y['date'].dt.weekday 
    
data_convert(train, test)


# %% ONE-HOT ENCODING 

def onehotencoding(x, columns):
    return pd.get_dummies(x, columns=columns, drop_first=True, dtype="int")

# AplicaÃ§Ã£o no train e test
columns_to_encode = ['country', 'store', 'product']
train = onehotencoding(train, columns_to_encode)
test = onehotencoding(test, columns_to_encode)


# %% SPLIT DATA TO TRAIN AND TEST MODELS

# ALMOST 80% TO TRAIN AND 20% TO TEST
to_train = train.iloc[:180000,:] 
to_test = train.iloc[180000:,:]

X_train =to_train.drop(['date', 'num_sold'], axis=1)
y_train = to_train['num_sold']
X_test = to_test.drop(['date', 'num_sold'], axis=1)
y_test = to_test['num_sold']


# %% XGBREGRESSOR

# params
param_combinations = [
    {'n_estimators': 50, 'learning_rate': 0.01, 'max_depth': 3},
    {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 5},
    {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 7},
    {'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 10},
    {'n_estimators': 100, 'learning_rate': 0.2, 'max_depth': 6}
]

# initialize a variable to store the best model and the smallest MAPE
best_model = None
best_mape = float('inf') 

# loop to test each combination of parameters
for params in param_combinations:
    print(f'Testando com: {params}')
    
    # initialize the model with current parameters
    model = XGBRegressor(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth']
    )
    
    # train the model
    model.fit(X_train, y_train)
    
    # make predictions
    y_pred = model.predict(X_test)
    
    # calculate MAPE
    mape = (abs(y_test - y_pred) / y_test).mean() * 100
    print(f'MAPE: {mape:.2f}%')
    
    # if the current MAPE is the lowest so far, update the best model
    if mape < best_mape:
        best_mape = mape
        best_model = model

# after the loop, best_model will have the model with the best parameters
print(f'\Best model with better MAPE: {best_mape:.2f}%')


#%% PLOTS
y_pred = best_model.predict(X_test)
to_test['num_sold_pred'] = y_pred
monthly_data = to_test.resample('M', on='date').agg({
    'num_sold': 'sum',         
    'num_sold_pred': 'sum'}).reset_index()

plt.figure(figsize=(10, 5))
plt.plot(monthly_data['date'], monthly_data['num_sold'], label='Real', linewidth=2)
plt.plot(monthly_data['date'], monthly_data['num_sold_pred'], label='Predict', linewidth=1.5, linestyle="--")
plt.title('Monthly Sales: Actual vs Forecast', size=20)
plt.legend()
plt.xlabel('DATA')
plt.ylabel('NUM SOLD')
plt.show()


# %% PREDICT IN REAL TEST DATASET
predict = best_model.predict(test.drop(columns="date"))
submission = pd.DataFrame({"id":test.index, "num_sold":predict})
submission.to_csv("/kaggle/working/submission_xgb_kaggle.csv", index=False)

