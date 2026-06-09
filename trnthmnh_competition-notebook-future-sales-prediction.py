# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install --upgrade googletrans==4.0.0-rc1


import pandas as pd
import numpy as np
from googletrans import Translator


# Loading the dataset
#1.sales train data
sales_train_data = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
#2.test data
test_data= pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
#3.items data
items_data = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
#4.item category data
item_categories_data = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
#5.shop data
shops_data = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')



#Delacring the translator and creating a user defined function
translator = Translator()

def translate_text(text):
    return translator.translate(text, src='ru', dest='en').text


item_categories_data['item_category_name'] = item_categories_data['item_category_name'].apply(translate_text)
shops_data['shop_name'] = shops_data['shop_name'].apply(translate_text)

print(item_categories_data.head())
print(shops_data.head())


sales_train_data.info()


sales_train_data.head()


sales_train_data['date'] = pd.to_datetime(sales_train_data['date'], format='%d.%m.%Y')
sales_train_data['month'] = sales_train_data['date'].dt.month
sales_train_data['year'] = sales_train_data['date'].dt.year


monthly_sales = sales_train_data.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].sum().reset_index()
monthly_sales.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)


test_data['date_block_num'] = 34
train_test = pd.concat([monthly_sales, test_data.drop(columns=['ID'])], ignore_index=True)
train_test = pd.merge(train_test, items_data, on='item_id', how='left')
train_test = pd.merge(train_test, item_categories_data, on='item_category_id', how='left')
train_test = pd.merge(train_test, shops_data, on='shop_id', how='left')


train_test


train_test['month'] = train_test['date_block_num'] % 12


for col in ['shop_id', 'item_id', 'item_category_id']:
    train_test[col] = train_test[col].astype('category').cat.codes
    
def lag_feature(df, lags, col):
    tmp = df[['date_block_num', 'shop_id', 'item_id', col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num', 'shop_id', 'item_id', col + '_lag_' + str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return df


train_test = lag_feature(train_test, [1, 2, 3, 6, 12], 'item_cnt_month')
train_test.fillna(0, inplace=True)


# Convert object columns to category
for col in ['item_name', 'item_category_name', 'shop_name']:
    train_test[col] = train_test[col].astype('category').cat.codes
    
train_set = train_test[train_test.date_block_num < 34].drop(['item_cnt_month'], axis=1)
y_train = train_test[train_test.date_block_num < 34]['item_cnt_month']
test_set = train_test[train_test.date_block_num == 34].drop(['item_cnt_month'], axis=1)


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb


X_train, X_true, y_train, y_true = train_test_split(train_set, y_train, test_size=0.2, random_state=42)


dtrain = xgb.DMatrix(X_train, label=y_train)
dtrue = xgb.DMatrix(X_true, label=y_true)
params = {'objective': 'reg:squarederror', 'max_depth': 6, 'eta': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8}
xgb_model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtrue, 'validation')])
y_pred_xgb = xgb_model.predict(xgb.DMatrix(X_true))
rmse_xgb = np.sqrt(mean_squared_error(y_true, y_pred_xgb))
print(f'XGBoost RMSE: {rmse_xgb}')


# Random Forest model
rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_true_rf = rf_model.predict(X_true)
rmse_rf = np.sqrt(mean_squared_error(y_true, y_true_rf))
print(f'Random Forest RMSE: {rmse_rf}')


lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_true = lgb.Dataset(X_true, label=y_true, reference=lgb_train)
params = {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9}
lgb_model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_true])
y_true_lgb = lgb_model.predict(X_true, num_iteration=lgb_model.best_iteration)
rmse_lgb = np.sqrt(mean_squared_error(y_true, y_true_lgb))
print(f'LightGBM RMSE: {rmse_lgb}')


# Predictions
y_test_pred = lgb_model.predict(test_set, num_iteration=lgb_model.best_iteration)
# Clip predictions as per competition requirements
y_test_pred = np.clip(y_test_pred, 0, 20)




# Create submission file
submission = pd.DataFrame({
    'ID': test_data['ID'],
    'item_cnt_month': y_test_pred
})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!!!")

