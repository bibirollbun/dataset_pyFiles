!pip install --pre pycaret


import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt

from catboost import CatBoostRegressor
from sklearn.linear_model import BayesianRidge, Ridge
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from pycaret.regression import setup, compare_models

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


#Loading data
calendar= pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv")
inventory= pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv")
train=pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv")
test=pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv")
#solution=pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')
#test_weights=pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')


#Check NA
calendar.isna().sum()


#Check reason for NA
calendar.head()


#Check holiday type, type of holiday might affect type of products bought?
calendar['holiday_name'].value_counts()


#Check inventory NA
inventory.isna().sum()


#Check inventory head
inventory.head()


#Check train NA
train.isna().sum()


#Drop train NA
train=train.dropna(subset=["sales"])


#Check train head
train.head()


test.head()


train=train.drop(['availability'], axis=1)


#Merge inventory
inventory=inventory.drop(['name','warehouse'], axis=1)

train = pd.merge(train, inventory,
              how = 'left',
              on = ['unique_id']
             )

test = pd.merge(test, inventory,
              how = 'left',
              on = ['unique_id']
             )

train.head()


#Merge calendar
train = pd.merge(train, calendar,
              how = 'left',
              on = ['date','warehouse']
             )

test = pd.merge(test, calendar,
              how = 'left',
              on = ['date','warehouse']
             )

train.head()


#Determining max discount applied
train['disc_applied'] = train[['type_0_discount','type_1_discount','type_2_discount','type_3_discount','type_4_discount','type_5_discount','type_6_discount']].max(axis=1)
test['disc_applied'] = test[['type_0_discount','type_1_discount','type_2_discount','type_3_discount','type_4_discount','type_5_discount','type_6_discount']].max(axis=1)

#Determining type of discount applied
columns = ['type_0_discount','type_1_discount','type_2_discount','type_3_discount','type_4_discount','type_5_discount','type_6_discount']

values = train[columns].to_numpy()
row_max = np.max(values, axis=1, keepdims=True)
max_mask = (values == row_max) & (np.sum(values == row_max, axis=1, keepdims=True) == 1)
train[columns] = max_mask.astype(int)


values = test[columns].to_numpy()
row_max = np.max(values, axis=1, keepdims=True)
max_mask = (values == row_max) & (np.sum(values == row_max, axis=1, keepdims=True) == 1)
test[columns] = max_mask.astype(int)


#Encoding product category, holiday name, and warehouse
encoder=OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(train[['warehouse','holiday_name','L1_category_name_en','L2_category_name_en','L4_category_name_en']])

encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['warehouse','holiday_name','L1_category_name_en','L2_category_name_en','L4_category_name_en']))
train = pd.concat([train.drop(columns=['warehouse','holiday_name','L1_category_name_en','L2_category_name_en','L4_category_name_en']), encoded_df], axis=1)

#Dropping L3, unable to encode
train.drop(columns = ['holiday_name_nan','L3_category_name_en'], inplace=True)
#train.iloc[:, 20:180] = train.iloc[:, 20:180].astype(int)
train.head()


#Clear memory
import gc
del calendar, inventory, columns,values, row_max, max_mask
gc.collect()


#Encode for test
test = pd.concat([test.drop(columns=['warehouse','holiday_name','L1_category_name_en','L2_category_name_en','L4_category_name_en']), encoded_df], axis=1)
test.drop(columns = ['holiday_name_nan','L3_category_name_en'], inplace=True)

del encoded_df, encoded


#Add day of week, date, month, year
train['date'] = pd.to_datetime(train['date'])

train['day_of_week'] = train['date'].dt.dayofweek
train['Year'] = train['date'].dt.year 
train['Month'] = train['date'].dt.month 
train['Day'] = train['date'].dt.day 

train.drop(columns = ['date'], inplace=True)

test['date'] = pd.to_datetime(test['date'])

test['day_of_week'] = test['date'].dt.dayofweek
test['Year'] = test['date'].dt.year 
test['Month'] = test['date'].dt.month 
test['Day'] = test['date'].dt.day 

test_date=test['date']
test.drop(columns = ['date'], inplace=True)


#No na values
train.isnull().sum().sum()


missing_columns = [col for col in train.columns if col not in test.columns]

for col in missing_columns:
    test[col] = 0

test.drop(columns = ['sales'], inplace=True)


#Split X and y
#lgbm=lgb()

y_train=train['sales']
X_train=train.drop(columns=['sales'])
train_data = lgb.Dataset(X_train, label=y_train)
del train, missing_columns, X_train, y_train


#Train
params={'learning_rate': 0.021796506746095975,
 'num_leaves': 93,
 'max_depth': 10,
 'min_child_samples': 25,
 'subsample': 0.7057135664023435,
 'colsample_bytree': 0.8528497905459008,
 'reg_alpha': 0.036786449788597686,
 'reg_lambda': 0.3151110021900479,
 'num_boost_round': 11000,
 'objective': 'regression',
 'metric': 'mae',
 'boosting_type': 'gbdt',
 'verbose': -1}

model=lgb.train(params=params, train_set=train_data)


#Test and submit
pred = model.predict(test, num_iteration=model.best_iteration)


test['sales_pred']=pred
test['date'] = test_date
test=test.dropna(subset=["unique_id"])
test['id']=test['unique_id'].astype(int).astype(str) + "_" + test['date'].astype(str)
test[['id','sales_pred']].to_csv("submission.csv",index=False)

