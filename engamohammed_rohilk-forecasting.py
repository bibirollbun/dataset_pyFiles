import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns 
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict,KFold
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as SME
from tqdm import tqdm 

from xgboost import XGBRegressor
import optuna



calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv')
inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
sales_train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv')
sales_test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv')
solution = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')
weight_test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')


weight_test


sales_test


sales_train


sales_train.info(),sales_test.info(),inventory.info(),calendar.info()


sales_train['discount'] = np.max(sales_train.iloc[:,7:], axis=1)
sales_test['discount'] = np.max(sales_test.iloc[:,5:], axis=1)
sales_train.date = pd.to_datetime(sales_train.date)
sales_test.date = pd.to_datetime(sales_test.date)
calendar.date = pd.to_datetime(calendar.date)


sales_train.drop(['type_0_discount','type_1_discount','type_2_discount','type_3_discount','type_4_discount','type_5_discount','type_6_discount'], axis=1, inplace=True)
sales_test.drop(['type_0_discount','type_1_discount','type_2_discount','type_3_discount','type_4_discount','type_5_discount','type_6_discount'], axis=1, inplace=True)


# sales_train.warehouse.unique()
# np.sort(sales_train.date.unique())
sales_train.unique_id.unique().shape
np.sort(sales_train.date.unique())


grouped_warehouse_with_date=sales_train.groupby(['date','warehouse'])['sales'].agg('sum').reset_index()
plt.figure(figsize=(10,6))
sns.lineplot(data=grouped_warehouse_with_date, x='date', y= 'sales', hue='warehouse')
plt.show()


grouped_warehouse_with_date=sales_train.groupby(['date','warehouse'])['total_orders'].agg('sum').reset_index()
plt.figure(figsize=(10,6))
sns.lineplot(data=grouped_warehouse_with_date, x='date', y= 'total_orders', hue='warehouse')
plt.show()


plt.figure(figsize=(10,6))
sns.boxplot(data=sales_train, x='warehouse', y='sales')
plt.show()


plt.figure(figsize=(10,6))
sns.boxplot(data=sales_train, x='warehouse', y='total_orders')
plt.show()


monthly_sales_train=sales_train.set_index('date')['sales'].resample('M').sum()
plt.figure(figsize=(10,6))
monthly_sales_train.plot()
plt.title('Monthly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.grid()


monthly_total_orders_train=sales_train.set_index('date')['total_orders'].resample('M').sum()
plt.figure(figsize=(10,6))
monthly_sales_train.plot()
plt.title('Monthly Total Orders Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.grid()


decom= seasonal_decompose(monthly_sales_train, period=12)
plt.figure(figsize=(10,6))
decom.plot()
plt.show()


sales_train_with_calendar=pd.merge(sales_train,calendar,how='left',on=['date','warehouse'])
sales_test_with_calendar=pd.merge(sales_test,calendar,how='left',on=['date','warehouse'])


sales_train_with_calendar = sales_train_with_calendar.drop(['holiday_name','availability'],axis=1)
sales_test_with_calendar = sales_test_with_calendar.drop(['holiday_name'],axis=1)


corr = sales_train_with_calendar[['sell_price_main',
                                  'sales','discount','holiday','shops_closed','winter_school_holidays','school_holidays']].corr()
plt.figure(figsize=(5,5))
sns.heatmap(corr,annot=True, fmt='.2f' )
plt.show()


sales_train_with_calendar=sales_train_with_calendar.sort_values(by=['unique_id','date'], ascending=[True,True]).fillna(method='ffill').sort_index()
sales_train_with_calendar


sales_train_with_calendar.isnull().sum()


train_data=pd.concat([sales_train_with_calendar.drop(['warehouse'],axis=1),pd.get_dummies(sales_train_with_calendar.warehouse)], axis=1)
test_data= pd.concat([sales_test_with_calendar.drop(['warehouse'],axis=1),pd.get_dummies(sales_test_with_calendar.warehouse)], axis=1)


train_data = pd.merge(train_data, weight_test,how='left', on='unique_id' )
train_data


y = train_data.sales
x= train_data.drop(['sales'], axis=1)


x['year'] = x.date.dt.year
x['month'] = x.date.dt.month
x['day'] = x.date.dt.day
x['week'] = x.date.dt.isocalendar().week
x['weekday'] = x.date.dt.dayofweek 


test_data['year'] = test_data.date.dt.year
test_data['month'] =test_data.date.dt.month
test_data['day'] = test_data.date.dt.day
test_data['week'] = test_data.date.dt.isocalendar().week
test_data['weekday'] = test_data.date.dt.dayofweek 


x = x.drop(['date'],axis=1)
test_data = test_data.drop(['date'],axis=1)
x= x*1
test_data= test_data*1


weight = x.pop('weight')



scaler = StandardScaler()
scaler.fit(x)


scaled_x= scaler.transform(x)
scaled_test = scaler.transform(test_data)


def weighted_mae(y_true, y_pred, weights):
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)


def optimizer(trial):
    param ={
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'max_depth': trial.suggest_int('max_depth', 3,15),
        'min_child_weight': trial.suggest_float('min_child_weight', 2,15),
        "learning_rate" : trial.suggest_float('learning_rate',1e-4, 0.5),
        'subsample': trial.suggest_float('subsample', 0.2, 1),
        'gamma': trial.suggest_float("gamma", 1e-4, 1.0),
        "colsample_bytree" : trial.suggest_float('colsample_bytree',0.2,1),
        "colsample_bylevel" : trial.suggest_float('colsample_bylevel',0.2,1),
        "colsample_bynode" : trial.suggest_float('colsample_bynode',0.2,1),
        'tree_method': 'gpu_hist',  # Use GPU
        'n_jobs': -1
        
    }

    xgb_opt = XGBRegressor(**param, eval_metric= 'mae')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    wmae_scores = []
    for train_idx, val_idx in kf.split(x):
        X_train, X_val = x.iloc[train_idx], x.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_train, w_val = weight.iloc[train_idx], weight.iloc[val_idx]
      
        xgb_opt.fit(X_train, y_train, 
                   eval_set=[(X_val, y_val)], 
                early_stopping_rounds=50,
                   verbose=False)
        y_pred = xgb_opt.predict(X_val)

         # Compute WMAE for this fold
        fold_wmae = weighted_mae(y_val, y_pred, w_val)
        wmae_scores.append(fold_wmae)

    return np.mean(wmae_scores)  # Return the average WMAE across folds




case = optuna.create_study(direction='minimize')
# case.optimize(optimizer, n_trials=100, timeout=3600)



best_param={'n_estimators': 1013,
 'max_depth': 14,
 'min_child_weight': 8.492474568003953,
 'learning_rate': 0.4663010727367519,
 'subsample': 0.7211936416529738,
 'gamma': 0.388619651133159,
 'colsample_bytree': 0.38884284955603254,
 'colsample_bylevel': 0.45250780165909454,
 'colsample_bynode': 0.7849769577316856}


model = XGBRegressor(**best_param, tree_method='gpu_hist', n_jobs=-1)


model.fit(scaled_x,y)  #eval_set=[(X_val, y_val)], 
               


predictions = model.predict(scaled_test)
predictions


solution['sales_hat']= predictions
solution.to_csv('submission.csv', index=False)
solution







