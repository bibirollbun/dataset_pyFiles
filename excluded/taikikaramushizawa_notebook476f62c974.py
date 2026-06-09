import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import xgboost as xgb
import csv
import pickle
from datetime import datetime
import os
import glob
import optuna # Hyperparameter Optimizer


# '''inputs list:
# /kaggle/input/rossmann-store-sales/sample_submission.csv
# /kaggle/input/rossmann-store-sales/store.csv
# /kaggle/input/rossmann-store-sales/train.csv
#     Store:int, DayOfWeek:int, Date:YYYY-MM-DD, Sales:int, Customers:int, Open:0|1, Promo:0|1, StateHoliday:a|b|c|0, SchoolHoliday:0|1 
# /kaggle/input/rossmann-store-sales/test.csv
# '''

train = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv', dtype={'StateHoliday':str}, parse_dates=[2])
test = pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv', dtype={'StateHoliday':str}, parse_dates=[3])
store = pd.read_csv("../input/rossmann-store-sales/store.csv")

# print(train['Store'].unique())
# print(test['Store'].unique())


def process_train_test(pd):
    # fillnaでOpenに0をセット
    # 学習に使えるようYYYY-MM-DDフォーマットを変形
    pd.fillna({'Open':0}, inplace=True)
    pd['week_of_year'] = pd['Date'].dt.isocalendar().week
    pd['year'] = pd['Date'].dt.isocalendar().year
    pd['month'] = pd['Date'].dt.month
    pd['day'] = pd['Date'].dt.day


process_train_test(train)
print(train.isnull().sum().sum())
train.head()


process_train_test(test)
print(test.isnull().sum())
test.head()


store['PromoInterval'] = store['PromoInterval'].str[0] # パターンが決まっており、先頭一文字で分類可能のため加工
store.fillna(0,inplace = True)
print(store.isnull().sum().sum())
store.head()


# SQLでいうleft (inner) joinさせる感じ
train = train.merge(store,how='left',on='Store')
print(train.shape)
print("train missing value ",train.isnull().sum().sum())


test = test.merge(store,how='left',on='Store')
print(test.shape)
print("test missing value ",test.isnull().sum().sum())


# 年度切りそろえ
train['year'] = train['year'] - 2013
test['year'] = test['year'] - 2013

train['Promo2SinceYear'] = train['Promo2SinceYear'] - 2008
train['Promo2SinceYear'] = train['Promo2SinceYear'].apply(lambda x:0 if x < 0 else x)
test['Promo2SinceYear'] = test['Promo2SinceYear'] - 2008
test['Promo2SinceYear'] = test['Promo2SinceYear'].apply(lambda x:0 if x < 0 else x)


# NaN防止
def int_try_execpt(x):
    try:
        return int(x)
    except ValueError:
        return 0


# 距離をざっくり桁数に変換。log(0)でinfになることを防止するため1を加算して桁を一つ切り落としている
train['CompetitionDistance'] = np.log(np.array(train['CompetitionDistance'].apply(int_try_execpt)) + 1) /10
test['CompetitionDistance'] = np.log(np.array(test['CompetitionDistance'].apply(int_try_execpt)) + 1) /10


from isoweek import Week

#おおよそ直近半年間(25週間)の販促は売り上げに影響する。逆も然り
def hasPromo2weeks(pd):
    result = []
    for index, row in pd.loc[:,['Date','Promo2SinceWeek','Promo2SinceYear']].iterrows():
        if index % 100000 == 0:
            print("processing row ",index)
        if row[2] == 0:
            weeks_since_promo2 = 0
        else:
            #isoweek to return the date of Monday, i.e. 2010-03-29
            start_promo2 = Week(int(row[2]), int(row[1])).monday()
            weeks_since_promo2 = (row[0].date() - start_promo2).days // 7
            #if promotion happen in current year, it will result negative, however the final result is 0, should try 1?
            #Because if haspromo2week is 0, latestpromo2months will be 0 also, row 1017204
            if weeks_since_promo2 < 0:
                weeks_since_promo2 = 0
        result.append(min(weeks_since_promo2, 25))   
    return result


train['hasPromo2weeks'] = hasPromo2weeks(train)
print("Complete : train dataset")
test['hasPromo2weeks'] = hasPromo2weeks(test)
print("Complete : test dataset")


#To encode promote state holiday, store type and assortment 
def abc2int(pd):
    d = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
    return pd.map(d)


for col in ['StateHoliday','StoreType','Assortment']:
    train[col] = abc2int(train[col]).fillna(0)
    test[col] = abc2int(test[col]).fillna(0)


def PromoInterval2int(pd):
    d = {'0': 0, 'J': 1, 'F': 2, 'M': 3}
    return pd.map(d)


train['PromoInterval'] = PromoInterval2int(train['PromoInterval']).fillna(0)
test['PromoInterval'] = PromoInterval2int(test['PromoInterval']).fillna(0)


# 直近に販促を行ったのはいつか(25週間以内)
def latest_promo2_months(pd):
    result = []
    for index, row in pd.loc[:,['Date','hasPromo2weeks','PromoInterval']].iterrows():
        if index % 100000 == 0:
            print("processing row ",index)
            
        if row[1] == 0:
            weeks_since_latest_promo2 =  0
        elif row[2] == 0:
            weeks_since_latest_promo2 =  0
        else:
            if row[0].month < row[2]:
                latest_promo2_start_year = row[0].year - 1
                latest_promo2_start_month = row[2] + 12 - 3
            else:
                latest_promo2_start_year = row[0].year
                latest_promo2_start_month = ((row[0].month - row[2]) // 3) * 3 + row[2]

            latest_promo2_start_day = datetime(year=int(latest_promo2_start_year),
                                               month=int(latest_promo2_start_month),
                                               day=1)
            weeks_since_latest_promo2 = (row[0] - latest_promo2_start_day).days // 30
        result.append(weeks_since_latest_promo2) 
    return result


train['latest_promo2_months'] = latest_promo2_months(train)
print("complete : train dataset")
test['latest_promo2_months'] = latest_promo2_months(test)
print("complete : test dataset")


train_inds = np.where((train['Open'] ==1) & (train['Sales'] >0) )[0]
len(train_inds)

train_x, train_y = train.copy().drop(columns = ['Sales','Date','Customers','Store']).iloc[train_inds,:], train.copy()['Sales'][train_inds]
train_x.shape, train_y.shape

# print(f"train:\n{train_x.head()}\n test:\n{test_x.head()}")


test_inds = np.where(test['Open'] ==1)[0]
test_inds0 = np.where(test['Open'] ==0)[0]
print("Test open stores: ",len(test_inds))
print("Test closed stores: ",len(test_inds0))
test_x = test.copy().drop(columns = ['Date','Id','Store'])
test_x.shape


print(train_x.columns)
print(test_x.columns)


# def objective(trial):
#     '''
#     Hyperparameter Optimize by optuna
#     '''
#     param = {
#         'objective' : 'reg:squarederror',
#         'max_depth' : trial.suggest_int('max_depth', 8, 20),
#         'learning_rate' : trial.suggest_float('learning_rate', 0.1, 0.6),
#         'n_estimators': trial.suggest_int('n_estimators', 600, 900),
#         'random_state' : 42,
#         'early_stopping_rounds' : 10,
#         'min_child_weight' : trial.suggest_float('min_child_weight',1.0,5.0),
#         'subsample' : trial.suggest_float('subsample',0.0,1.0),
#         'colsample_bytree' : trial.suggest_float('colsample_bytree',0.4,0.6)
#     }
#     rmspes = []

#     #K-Fold オブジェクト
#     kf = KFold(n_splits=4,shuffle=True, random_state=100)

#     #K-Fold CV
#     for train_index, valid_index in kf.split(train_x):
#         x_train_cv, y_train_cv = train_x.iloc[train_index], train_y.iloc[train_index]
#         x_valid_cv, y_valid_cv = train_x.iloc[valid_index], train_y.iloc[valid_index]
#         model = XGBRegressor(**param)

#         model.fit(x_train_cv,
#                   y_train_cv,
#                   eval_set = [(x_train_cv, y_train_cv),(x_valid_cv, y_valid_cv)],
#                   verbose = 100,
#                  )
#         y_pred_valid = model.predict(x_valid_cv)

#         y_valid_cv.replace({0: 1}, inplace=True)

#         # RMSPE
#         temp_rmspe_valid = np.sqrt(np.mean(((y_pred_valid - y_valid_cv) / y_valid_cv)**2))*100
#         rmspes.append(temp_rmspe_valid)
#         print(f"{temp_rmspe_valid=}")
#         print(f"{rmspes=}")
#     return np.mean(rmspes)

# study = optuna.create_study(direction='minimize', study_name="distributed-study")
# study.optimize(objective, n_trials=3, n_jobs=-1) # multithread

# print('Number of finished trials:', len(study.trials))
# print('Best trial:', study.best_trial.params)

# # Trial 2 finished with value: 17.584629303402636 and parameters: {'max_depth': 11, 'learning_rate': 0.10374726495013334, 'n_estimators': 764, 'min_child_weight': 1.4920271473117688, 'subsample': 0.9067913883591385, 'colsample_bytree': 0.5873785759052303}. Best is trial 2 with value: 17.584629303402636.


# # モデルの学習を行う
model = XGBRegressor(objective='reg:squarederror',max_depth=11,learning_rate=0.1, subsample=0.9,n_estimators=700, random_state=42, min_child_weight=1.5,colsample_bytree=0.59)

model.fit(train_x, train_y)

pred = model.predict(test_x)
# y_pred += list(pred)

print(pred)


id_sales = pd.DataFrame({
    "Id": range(1,len(pred)+1),
    "Sales":  pred
})

# locを使ってDFにSales行を追加
test.loc[test_inds0, 'Sales'] = 0
test.loc[test_inds,'Sales'] = id_sales

result = pd.DataFrame({"Id": test["Id"],'Sales': test["Sales"]})
# print(result)
# print(result.loc[479]) # Open=0の日。Salesが0か確認用

result.to_csv("submission.csv", index=False)

