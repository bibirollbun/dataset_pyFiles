import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# from sklearn.preprocessing import TargetEncoder
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.model_selection import KFold

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)


def convert_date(dataframe):
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    dataframe['day_of_week'] = dataframe.apply(lambda x: 'Monday' if x['date'].dayofweek == 0 else 'Tuesday' if x['date'].dayofweek == 1 else 'Wednesday' if x['date'].dayofweek == 2 else 'Thursday' if x['date'].dayofweek == 3 else 'Friday' if x['date'].dayofweek == 4 else 'Saturday' if x['date'].dayofweek == 5 else 'Sunday', axis=1)
    temp_df=pd.get_dummies(dataframe['day_of_week'],drop_first=True,dtype=float)
    #
    dataframe=pd.concat([dataframe.drop(columns=['day_of_week']),temp_df],axis=1)
    print(dataframe)
    dataframe['is_weekend'] = dataframe.apply(lambda x: 1 if x['date'].dayofweek >= 5 else 0, axis=1)
    dataframe['year'] = dataframe.apply(lambda x: x['date'].year, axis=1)
    dataframe['month'] = dataframe.apply(lambda x: x['date'].month, axis=1)
    dataframe['num_of_week']=dataframe.apply(lambda x: x['date'].week, axis=1)
    dataframe['day'] = dataframe.apply(lambda x: x['date'].day, axis=1)
    dataframe['day_of_year'] = dataframe.apply(lambda x: x['date'].dayofyear, axis=1)
    dataframe['season']=dataframe.apply(lambda x: 'Spring' if ((x['month']>=2)&(x['month']<=4)) else 'Summer' if ((x['month']>=5)&(x['month']<=7)) else 'Autumn' if ((x['month']>=8)&(x['month']<=10)) else 'Winter', axis=1)
    dataframe['group'] = (dataframe['year'] - 2020) * 48 + dataframe['month'] * 4 + dataframe['day'] // 7
    
    print(dataframe)
    print(dataframe.columns.values)
    #Encoding
    dataframe['year_sin']=np.sin(2*np.pi*dataframe['year']/np.max(dataframe['year']))
    dataframe['year_cos']=np.cos(2*np.pi*dataframe['year']/np.max(dataframe['year']))
    dataframe['month_sin']=np.sin(2*np.pi*dataframe['month']/12)
    dataframe['month_cos']=np.cos(2*np.pi*dataframe['month']/12)
    dataframe['day_sin']=np.sin(2*np.pi*dataframe['day']/31)
    dataframe['day_cos']=np.cos(2*np.pi*dataframe['day']/31)
    dataframe['day_sin4'] = np.sin(dataframe['day_of_year'] * (8 * np.pi /  365.0))
    dataframe['day_cos4'] = np.cos(dataframe['day_of_year'] * (8 * np.pi /  365.0))
    dataframe['day_sin3'] = np.sin(dataframe['day_of_year'] * (6 * np.pi /  365.0))
    dataframe['day_cos3'] = np.cos(dataframe['day_of_year'] * (6 * np.pi /  365.0))
    dataframe['day_sin2'] = np.sin(dataframe['day_of_year'] * (4 * np.pi /  365.0))
    dataframe['day_cos'] = np.cos(dataframe['day_of_year'] * (4 * np.pi /  365.0))
    dataframe['day_sin'] = np.sin(dataframe['day_of_year'] * (2 * np.pi /  365.0))
    dataframe['day_cos2'] = np.cos(dataframe['day_of_year'] * (2 * np.pi /  365.0))
    dataframe['day_sin_0.5'] = np.sin(dataframe['day_of_year'] * (1 * np.pi /  365.0))
    dataframe['day_cos_0.5'] = np.cos(dataframe['day_of_year'] * (1 * np.pi /  365.0))    
        
    # dataframe['Group'] = (dataframe['year'] - 2010) * 48 + dataframe['month'] * 4 +dataframe['day'] // 7
    dataframe['day_of_week_numeric']= dataframe.apply(lambda x: x['date'].dayofweek,axis=1) 
 
 
 
     
    for country in countries_unique:
        get_plot_group_by_country(f'{country}',dataframe)
 
 
 
    
    dataframe.drop(columns=['date'],inplace=True)
    # dataframe['year'] = dataframe.year.astype('category')
    return dataframe


data_train=pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
sample=pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
data_test=pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

gdp_per_capita_df = pd.read_csv("/kaggle/input/world-gdpgdp-gdp-per-capita-and-annual-growths/gdp_per_capita.csv")
# print(gdp_per_capita_df['Country Name'].unique())
print(gdp_per_capita_df.loc[gdp_per_capita_df['Country Name']=='Canada'])
# print(gdp_per_capita_df['Country Name'])
# print(gdp_per_capita_df.index.values)

# data_train['gdp']=data_train.apply(lambda x: )
store_unique=data_train['store'].unique()
product_unique=data_train['product'].unique()
countries_unique=data_train['country'].unique()
submission_ids=data_test['id']


print(data_train)


def get_plot_by_country(country_name):
    data_train['date'] = pd.DatetimeIndex(data_train['date'])
    temp_data = data_train[['date', 'country', 'num_sold', 'product']].loc[
        data_train['country'] == f"{country_name}"].groupby(
        by=['product', pd.Grouper(key='date', freq='M')]).sum().reset_index()
    plt.figure(figsize=(20, 5))

    product_id = 0
    plt.plot(temp_data.loc[temp_data['product'] == product_unique[product_id]]['date'],
             temp_data.loc[temp_data['product'] == product_unique[product_id]]['num_sold'], label=f'{product_unique[product_id]}')
    plt.plot(temp_data.loc[temp_data['product'] == product_unique[product_id + 1]]['date'],
             temp_data.loc[temp_data['product'] == product_unique[product_id + 1]]['num_sold'],
             label=f'{product_unique[product_id + 1]}')
    plt.plot(temp_data.loc[temp_data['product'] == product_unique[product_id + 2]]['date'],
             temp_data.loc[temp_data['product'] == product_unique[product_id + 2]]['num_sold'],
             label=f'{product_unique[product_id + 2]}')
    plt.plot(temp_data.loc[temp_data['product'] == product_unique[product_id + 3]]['date'],
             temp_data.loc[temp_data['product'] == product_unique[product_id + 3]]['num_sold'],
             label=f'{product_unique[product_id + 3]}')
    # print(temp_data)
    plt.grid()
    plt.legend()
    plt.title(f"{country_name}")
    plt.show()


def get_plot_by_product(product_name):
    data_train['date'] = pd.DatetimeIndex(data_train['date'])
    temp_data = data_train[['date', 'country', 'num_sold', 'product']].loc[
        data_train['product'] == f"{product_name}"].groupby(
        by=['country', pd.Grouper(key='date', freq='M')]).sum().reset_index()
    plt.figure(figsize=(20, 5))

    country_id = 0
    plt.plot(temp_data.loc[temp_data['country'] == countries_unique[country_id]]['date'],
             (temp_data.loc[temp_data['country'] == countries_unique[country_id]]['num_sold']-temp_data.loc[temp_data['country'] == countries_unique[country_id]]['num_sold'].max())/temp_data.loc[temp_data['country'] == countries_unique[country_id]]['num_sold'].max(), label=f'{countries_unique[country_id]}')
    plt.plot(temp_data.loc[temp_data['country'] == countries_unique[country_id + 1]]['date'],
             (temp_data.loc[temp_data['country'] == countries_unique[country_id + 1]]['num_sold']-temp_data.loc[temp_data['country'] == countries_unique[country_id+1]]['num_sold'].max())/temp_data.loc[temp_data['country'] == countries_unique[country_id+1]]['num_sold'].max(),
             label=f'{countries_unique[country_id + 1]}')
    plt.plot(temp_data.loc[temp_data['country'] == countries_unique[country_id + 2]]['date'],
             (temp_data.loc[temp_data['country'] == countries_unique[country_id + 2]]['num_sold']-temp_data.loc[temp_data['country'] == countries_unique[country_id+2]]['num_sold'].max())/temp_data.loc[temp_data['country'] == countries_unique[country_id+2]]['num_sold'].max(),
             label=f'{countries_unique[country_id + 2]}')
    plt.plot(temp_data.loc[temp_data['country'] == countries_unique[country_id + 3]]['date'],
             (temp_data.loc[temp_data['country'] == countries_unique[country_id + 3]]['num_sold']-temp_data.loc[temp_data['country'] == countries_unique[country_id+3]]['num_sold'].max())/temp_data.loc[temp_data['country'] == countries_unique[country_id+3]]['num_sold'].max(),
             label=f'{countries_unique[country_id + 3]}')
    plt.plot(temp_data.loc[temp_data['country'] == countries_unique[country_id + 4]]['date'],
             (temp_data.loc[temp_data['country'] == countries_unique[country_id + 4]]['num_sold']-temp_data.loc[temp_data['country'] == countries_unique[country_id+4]]['num_sold'].max())/temp_data.loc[temp_data['country'] == countries_unique[country_id+4]]['num_sold'].max(),
             label=f'{countries_unique[country_id + 4]}')
    plt.plot(temp_data.loc[temp_data['country'] == countries_unique[country_id + 5]]['date'],
             (temp_data.loc[temp_data['country'] == countries_unique[country_id + 5]]['num_sold']-temp_data.loc[temp_data['country'] == countries_unique[country_id+5]]['num_sold'].max())/temp_data.loc[temp_data['country'] == countries_unique[country_id+5]]['num_sold'].max(),
             label=f'{countries_unique[country_id + 5]}')
    # print(temp_data)
    plt.grid()
    plt.legend()
    plt.title(f"{product_name}")
    plt.show()


def get_plot_group_by_country(country_name,data_train):
    data_train['date'] = pd.DatetimeIndex(data_train['date'])
    temp_data = data_train[['date', 'country', 'group', 'product']].loc[
        data_train['country'] == f"{country_name}"].groupby(
        by=['product', pd.Grouper(key='date', freq='M')]).sum().reset_index()
    plt.figure(figsize=(20, 5))

    product_id = 0
    plt.plot(temp_data.loc[temp_data['product'] == product_unique[product_id]]['date'],
             temp_data.loc[temp_data['product'] == product_unique[product_id]]['group'], label=f'{product_unique[product_id]}')
    plt.plot(temp_data.loc[temp_data['product'] == product_unique[product_id + 1]]['date'],
             temp_data.loc[temp_data['product'] == product_unique[product_id + 1]]['group'],
             label=f'{product_unique[product_id + 1]}')
    plt.plot(temp_data.loc[temp_data['product'] == product_unique[product_id + 2]]['date'],
             temp_data.loc[temp_data['product'] == product_unique[product_id + 2]]['group'],
             label=f'{product_unique[product_id + 2]}')
    plt.plot(temp_data.loc[temp_data['product'] == product_unique[product_id + 3]]['date'],
             temp_data.loc[temp_data['product'] == product_unique[product_id + 3]]['group'],
             label=f'{product_unique[product_id + 3]}')
    # print(temp_data)
    plt.grid()
    plt.legend()
    plt.title(f"{country_name}")
    plt.show()


for country in countries_unique:
    get_plot_by_country(f'{country}')




data_train.drop(columns=['id'],inplace=True)
data_test.drop(columns=['id'],inplace=True)

data_train['num_sold']=data_train['num_sold'].apply(lambda x: np.log1p(x))
# print(data_train)
data_train=convert_date(data_train)

data_test=convert_date(data_test)


print('data train\n\n')
print(data_train)
print('\n')

print(data_train.columns.values)
print('data_test\n\n')
print(data_test)
print('\n')
print(data_test.columns.values)






weekday_ratios=data_train.groupby(by='day_of_week_numeric')['num_sold'].mean()/data_train.groupby(by='day_of_week_numeric')['num_sold'].mean().mean()
print(weekday_ratios)

week_ratios=data_train.groupby(by='num_of_week')['num_sold'].mean()/data_train.groupby(by='num_of_week')['num_sold'].mean().mean()
print(week_ratios)

month_ratios=data_train.groupby(by='month')['num_sold'].mean()/data_train.groupby(by='month')['num_sold'].mean().mean()
print(month_ratios)

country_ratios=data_train.groupby(by='country')['num_sold'].mean()/data_train.groupby(by='country')['num_sold'].mean().mean()
print(country_ratios)

product_ratios=data_train.groupby(by='product')['num_sold'].mean()/data_train.groupby(by='product')['num_sold'].mean().mean()
print(product_ratios)

store_ratios=data_train.groupby(by='store')['num_sold'].mean()/data_train.groupby(by='store')['num_sold'].mean().mean()
print(store_ratios)





ohe=OneHotEncoder()
nan_count=data_train.isna().sum()
data_train.dropna(inplace=True)
nan_count=data_train.isna().sum()

nan_count=data_test.isna().sum()


data_train.drop(columns=['day_of_week_numeric','num_of_week','year','month','day'])



print(data_train)



data_train=pd.concat([pd.get_dummies(data_train.drop(columns=['day_of_week_numeric','num_of_week','year','month','day']),drop_first=True,dtype=float),data_train[['country','product','store','day_of_week_numeric','num_of_week','year','month','day']]],axis=1)
print(data_train)
# data_test=pd.get_dummies(data_test,drop_first=True,dtype=float)
# data_test.drop(columns=['day_of_week_numeric','num_of_week','year','month','day'],inplace=True)


# data_train=pd.get_dummies(data_train,drop_first=True,dtype=float)
# # data_test=pd.get_dummies(data_test,drop_first=True,dtype=float)
# print(data_train)
# le = LabelEncoder()
# le.fit(data_train['country'])
# data_train['country']=le.transform(data_train['country'])
# data_test['country']=le.transform(data_test['country'])

# le.fit(data_train['store'])
# data_train['store']=le.transform(data_train['store'])
# data_test['store']=le.transform(data_test['store'])

# le.fit(data_train['product'])
# data_train['product']=le.fit_transform(data_train['product'])
# data_test['product']=le.fit_transform(data_test['product'])



print(data_train)


import sklearn
sklearn.metrics.get_scorer_names()



# from sklearn.metrics import mean_absolute_percentage_error, make_scorer
# mape_scorer = make_scorer(mean_absolute_percentage_error)


# reg:squarederror
# neg_mean_squared_error
# for LGBM just type: regression as loss function instead of mape



# X_train, X_test, y_train, y_test = data_train.drop(columns=['num_sold']),data_test,data_train['num_sold'],sample['num_sold']
model1 = XGBRegressor(objective='reg:absoluteerror')
# model.fit(X_train, y_train)
# import lightgbm as lgb
# from lightgbm import LGBMRegressor
# # model = LGBMRegressor(metric='mape',verbose=-1)

# from sklearn.tree import DecisionTreeRegressor

# # model=DecisionTreeRegressor()
# distributions = {
#     # 'n_estimators': [5,10,15,30,50,100,150],
#     # 'boosting_type': ['gbdt'],#,'dart','rf'
#     'max_leaves': [15,31,50,100,200,500,1000],
#     'max_depth': [-1,15,30,60,100,200],
#     'n_estimators': [10,50,100,200,500,1000],
#                      # 'reg_lambda': [0.1,0.3,0.5,0.7,1],
#                      #       'reg_alpha': [0.1,0.3,0.5,0.7,1],
#     # 'colsample_bytree':[0.1,0.3,0.5,0.7,1],
#     #     'colsample_bylevel':[0.1,0.3,0.5,0.7,1],
#     #     'colsample_bynode':[0.1,0.3,0.5,0.7,1],
#     #                         # 'eta':[0.3,0.1,0.5,0.8],
#     # 'learning_rate': [0.05, 0.1, 0.15, 0.20,0.5,1],
#     # 'min_child_weight': [1, 2, 3, 4]
#     }
# clf = RandomizedSearchCV(model, distributions)
    
# search = clf.fit(X_train, y_train)

# print('Search is finished. Best parameters: \n')
# print(search.best_params_)
# model=search.best_estimator_



def get_mape(y_true,y_pred):
    MAPE_test=MAPE(y_true,y_pred)
    print(f'MAPE test:{MAPE_test}')
    return MAPE_test


def eval_MAPE_with_ratios(data_test,preds_test,y_test):
    best_MAPE=1
    best_preds=[]
    test_df=pd.concat([data_test,pd.Series(preds_test,name='num_sold',index=data_test.index.values)],axis=1)
    # print(test_df)
    print('Model results without Ratios:')
    MAPE_test=get_mape(y_test,preds_test)

    if MAPE_test<best_MAPE:
        best_MAPE=MAPE_test
        best_preds=0
        
    preds_test=test_df.apply(lambda x: x['num_sold']*weekday_ratios[x['day_of_week_numeric']]*week_ratios[x['num_of_week']]*month_ratios[x['month']],axis=1)
    # print(preds_test)
    
    print('\n\nRatios: day_of_week, num_of_week, month')
    MAPE_test=get_mape(y_test,preds_test)

    if MAPE_test<best_MAPE:
        best_MAPE=MAPE_test
        best_preds=1
        
    preds_test=test_df.apply(lambda x: x['num_sold']*weekday_ratios[x['day_of_week_numeric']]*week_ratios[x['num_of_week']]*month_ratios[x['month']]*country_ratios[x['country']],axis=1)
    
    print('\nRatios: day_of_week, num_of_week, month, COUNTRY')
    MAPE_test=get_mape(y_test,preds_test)

    if MAPE_test<best_MAPE:
        best_MAPE=MAPE_test
        best_preds=2
        
    preds_test=test_df.apply(lambda x: x['num_sold']*weekday_ratios[x['day_of_week_numeric']]*week_ratios[x['num_of_week']]*month_ratios[x['month']]*product_ratios[x['product']],axis=1)
    
    print('\nRatios: day_of_week, num_of_week, month, PRODUCT')
    MAPE_test=get_mape(y_test,preds_test)

    if MAPE_test<best_MAPE:
        best_MAPE=MAPE_test
        best_preds=3
        
    preds_test=test_df.apply(lambda x: x['num_sold']*weekday_ratios[x['day_of_week_numeric']]*week_ratios[x['num_of_week']]*month_ratios[x['month']]*store_ratios[x['store']],axis=1)
    
    print('\nRatios: day_of_week, num_of_week, month, STORE')
    MAPE_test=get_mape(y_test,preds_test)
    
    if MAPE_test<best_MAPE:
        best_MAPE=MAPE_test
        best_preds=4
        
    preds_test=test_df.apply(lambda x: x['num_sold']*weekday_ratios[x['day_of_week_numeric']]*week_ratios[x['num_of_week']]*month_ratios[x['month']]*country_ratios[x['country']]*product_ratios[x['product']],axis=1)
    
    print('\nRatios: day_of_week, num_of_week, month, COUNTRY,PRODUCT')
    MAPE_test=get_mape(y_test,preds_test)

    if MAPE_test<best_MAPE:
        best_MAPE=MAPE_test
        best_preds=5
        
    preds_test=test_df.apply(lambda x: x['num_sold']*weekday_ratios[x['day_of_week_numeric']]*week_ratios[x['num_of_week']]*month_ratios[x['month']]*country_ratios[x['country']]*store_ratios[x['store']],axis=1)
    
    print('\nRatios: day_of_week, num_of_week, month, COUNTRY,STORE')
    MAPE_test=get_mape(y_test,preds_test)

    if MAPE_test<best_MAPE:
            best_MAPE=MAPE_test
            best_preds=6

    return best_preds
        


best_ratios=0
ratios=pd.Series(['No ratios','day_of_week, num_of_week, month','+COUNTRY','+STORE','+PRODUCT','+COUNTRY,STORE','+COUNTRY,PRODUCT','+STORE,PRODUCT'])
y=data_train['num_sold']
data_train.drop(columns=['num_sold'],inplace=True)
MAPE_train_arr,MAPE_test_arr=[],[]
kf = KFold(n_splits=20)

best_score=100
for i,(train_index, test_index) in enumerate(kf.split(data_train)):
    
    X_train, X_test = data_train.iloc[train_index].drop(columns=['country','product','store','day_of_week_numeric','num_of_week','year','month','day']), data_train.iloc[test_index].drop(columns=['country','product','store','day_of_week_numeric','num_of_week','year','month','day'])
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    
    X_test_with_info=data_train.iloc[test_index]
    
    distributions = {
    'n_estimators': [10,50,100,200,300],
    'max_depth': [9,15,20,30] ,
                     # 'lambda':[0.1,0.5,1],
                           # 'alpha':[0.1,0.5,1],
                           #  'eta':[0.3,0.1,0.5,0.8],
    'reg_lambda': [0,0.1,0.3,0.5,0.7,1],
    'reg_alpha': [0,0.1,0.3,0.5,0.7,0.9,1],
    'learning_rate': [0.1, 0.15, 0.20,0.5],
    'min_child_weight': [1, 2, 4,10,20,30],
    'colsample_bytree':[0,0.1,0.3,0.5,0.7,1],
    'colsample_bylevel':[0,0.1,0.3,0.5,0.7,1],
    'colsample_bynode':[0,0.1,0.3,0.5,0.7,1],
    }
    

    
    clf = RandomizedSearchCV(model1, distributions,
            verbose=0,scoring='neg_mean_absolute_error')
    search = clf.fit(X_train, y_train)
    
    model=search.best_estimator_
    
    print('Search is finished. Best parameters: \n')
    print(search.best_params_)
    # print(best_params)
            
    y_pred = model.predict(X_train)
    MAPE_train = MAPE(y_train, y_pred)

    y_pred = model.predict(X_test)
    MAPE_test = MAPE(y_test, y_pred)

    MAPE_train_arr.append(MAPE_train)
    MAPE_test_arr.append(MAPE_test)

    print(f'\n FOLD-{i}. MAPE train: {MAPE_train}, MAPE test: {MAPE_test} \n')

    
    best_ratios=eval_MAPE_with_ratios(X_test_with_info,y_pred,y_test)

    if MAPE_test<best_score:
        best_score=MAPE_test
        best_model=model
        print(f' FOLD-{i}. Model has best score. Best ratios is {ratios[best_ratios]}')


model1=best_model



import lightgbm as lgb
from lightgbm import LGBMRegressor
model2 = LGBMRegressor(metric='mape',verbose=-1)



best_ratios=0
ratios=pd.Series(['No ratios','day_of_week, num_of_week, month','+COUNTRY','+STORE','+PRODUCT','+COUNTRY,STORE','+COUNTRY,PRODUCT','+STORE,PRODUCT'])
# y=data_train['num_sold']
# data_train.drop(columns=['num_sold'],inplace=True)
MAPE_train_arr,MAPE_test_arr=[],[]
kf = KFold(n_splits=20)

best_score=100
for i,(train_index, test_index) in enumerate(kf.split(data_train)):
    
    X_train, X_test = data_train.iloc[train_index].drop(columns=['country','product','store','day_of_week_numeric','num_of_week','year','month','day']), data_train.iloc[test_index].drop(columns=['country','product','store','day_of_week_numeric','num_of_week','year','month','day'])
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    
    X_test_with_info=data_train.iloc[test_index]
    
    distributions = {
        "n_estimators":  [10,50,100,200,300],
        "learning_rate": [0.1, 0.15, 0.20],
        "num_leaves": [20,60,100,200,500,1000],
        "max_depth": [9,15,20,30],
        "min_data_in_leaf": [200,600,1000,2000,5000],
        "lambda_l1": [0,10,30,50,70,100],
        "lambda_l2": [0,10,30,50,70,100],
        "min_gain_to_split": [0,3,5,9,12,15,20]
    }
    

    
    clf = RandomizedSearchCV(model2, distributions,
            verbose=0,scoring='neg_mean_absolute_percentage_error')
    search = clf.fit(X_train, y_train)
    
    model=search.best_estimator_
    
    print('Search is finished. Best parameters: \n')
    print(search.best_params_)
    # print(best_params)
            
    y_pred = model.predict(X_train)
    MAPE_train = MAPE(y_train, y_pred)

    y_pred = model.predict(X_test)
    MAPE_test = MAPE(y_test, y_pred)

    MAPE_train_arr.append(MAPE_train)
    MAPE_test_arr.append(MAPE_test)

    print(f'\n FOLD-{i}. MAPE train: {MAPE_train}, MAPE test: {MAPE_test} \n')

    
    best_ratios=eval_MAPE_with_ratios(X_test_with_info,y_pred,y_test)

    if MAPE_test<best_score:
        best_score=MAPE_test
        best_model=model
        print(f' FOLD-{i}. Model has best score. Best ratios is {ratios[best_ratios]}')


model2=best_model



from sklearn.linear_model import LinearRegression

# y=data_train['num_sold']
# data_train.drop(columns=['num_sold'],inplace=True)
MAPE_train_arr,MAPE_test_arr=[],[]
kf = KFold(n_splits=20)

best_score=100
for i,(train_index, test_index) in enumerate(kf.split(data_train)):
    
    X_train, X_test = data_train.iloc[train_index].drop(columns=['country','product','store','day_of_week_numeric','num_of_week','year','month','day']), data_train.iloc[test_index].drop(columns=['country','product','store','day_of_week_numeric','num_of_week','year','month','day'])
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    
    X_test_with_info=data_train.iloc[test_index]

    pred1=model1.predict(X_train)
    pred2=model2.predict(X_train)

    
    model=LinearRegression()
    model.fit(np.column_stack((pred1,pred2)),y_train)

    y_pred=model.predict(np.column_stack((pred1,pred2)))
            
    MAPE_train = MAPE(y_train, y_pred)

    pred1=model1.predict(X_test)
    pred2=model2.predict(X_test)

    
    y_pred = model.predict(np.column_stack((pred1,pred2)))
    MAPE_test = MAPE(y_test, y_pred)

    MAPE_train_arr.append(MAPE_train)
    MAPE_test_arr.append(MAPE_test)

    print(f'\n FOLD-{i}. MAPE train: {MAPE_train}, MAPE test: {MAPE_test} \n')

    
    best_ratios=eval_MAPE_with_ratios(X_test_with_info,y_pred,y_test)

    if MAPE_test<best_score:
        best_score=MAPE_test
        best_model=model
        print(f' FOLD-{i}. Model has best score. Best ratios is {ratios[best_ratios]}')


model=best_model



X_train, y_train, y_test = data_train.drop(columns=['country','product','store','day_of_week_numeric','num_of_week','year','month','day']),y,sample['num_sold']



X_test=pd.get_dummies(data_test.drop(columns=['day_of_week_numeric','num_of_week','month','year','day']),drop_first=True,dtype=float)
X_test_with_info=data_test


pred1=model1.predict(X_train)
pred2=model2.predict(X_train)

preds_train = model.predict(np.column_stack((pred1,pred2)))


pred1=model1.predict(X_test)
pred2=model2.predict(X_test)

preds_test = model.predict(np.column_stack((pred1,pred2)))

test_df=pd.concat([X_test_with_info,pd.Series(preds_test,name='num_sold',index=data_test.index.values)],axis=1)

if best_ratios==1:
    preds_test=test_df.apply(lambda x: x['num_sold']*weekday_ratios[x['day_of_week_numeric']]*week_ratios[x['num_of_week']]*month_ratios[x['month']],axis=1)
elif best_ratios==2:
       preds_test=test_df.apply(lambda x: x['num_sold']*weekday_ratios[x['day_of_week_numeric']]*week_ratios[x['num_of_week']]*month_ratios[x['month']]*country_ratios[x['country']],axis=1)
elif best_ratios==3:
            preds_test=test_df.apply(lambda x: x['num_sold']*weekday_ratios[x['day_of_week_numeric']]*week_ratios[x['num_of_week']]*month_ratios[x['month']]*product_ratios[x['product']],axis=1)
elif best_ratios==4:
        preds_test=test_df.apply(lambda x: x['num_sold']*weekday_ratios[x['day_of_week_numeric']]*week_ratios[x['num_of_week']]*month_ratios[x['month']]*store_ratios[x['store']],axis=1)
elif best_ratios==5:
    preds_test=test_df.apply(lambda x: x['num_sold']*weekday_ratios[x['day_of_week_numeric']]*week_ratios[x['num_of_week']]*month_ratios[x['month']]*country_ratios[x['country']]*product_ratios[x['product']],axis=1)
elif best_ratios==6:
    preds_test=test_df.apply(lambda x: x['num_sold']*weekday_ratios[x['day_of_week_numeric']]*week_ratios[x['num_of_week']]*month_ratios[x['month']]*country_ratios[x['country']]*store_ratios[x['store']],axis=1)
elif best_ratios==7:
    preds_test=test_df.apply(lambda x: x['num_sold']*weekday_ratios[x['day_of_week_numeric']]*week_ratios[x['num_of_week']]*month_ratios[x['month']]*product_ratios[x['product']]*store_ratios[x['store']],axis=1)
else: preds_test=preds_test 
    



# test_df=pd.concat([data_test,pd.Series(preds_test,name='num_sold')],axis=1)
# print(test_df)
# print('Model results without Ratios:')
# get_mape(y_train,preds_train,y_test,preds_test)

# preds_test=test_df.apply(lambda x: x['num_sold']*weekday_ratios[x['day_of_week_numeric']]*week_ratios[x['num_of_week']]*month_ratios[x['month']],axis=1)
# print(preds_test)

# print('\n\nRatios: day_of_week, num_of_week, month')
# get_mape(y_train,preds_train,y_test,preds_test)

# preds_test=test_df.apply(lambda x: x['num_sold']*weekday_ratios[x['day_of_week_numeric']]*week_ratios[x['num_of_week']]*month_ratios[x['month']]*country_ratios[x['country']],axis=1)

# print('Ratios: day_of_week, num_of_week, month, COUNTRY')
# get_mape(y_train,preds_train,y_test,preds_test)

# preds_test=test_df.apply(lambda x: x['num_sold']*weekday_ratios[x['day_of_week_numeric']]*week_ratios[x['num_of_week']]*month_ratios[x['month']]*product_ratios[x['product']],axis=1)

# print('Ratios: day_of_week, num_of_week, month, PRODUCT')
# get_mape(y_train,preds_train,y_test,preds_test)

# preds_test=test_df.apply(lambda x: x['num_sold']*weekday_ratios[x['day_of_week_numeric']]*week_ratios[x['num_of_week']]*month_ratios[x['month']]*store_ratios[x['store']],axis=1)

# print('Ratios: day_of_week, num_of_week, month, STORE')
# get_mape(y_train,preds_train,y_test,preds_test)


# preds_test=test_df.apply(lambda x: x['num_sold']*weekday_ratios[x['day_of_week_numeric']]*week_ratios[x['num_of_week']]*month_ratios[x['month']]*country_ratios[x['country']]*product_ratios[x['product']],axis=1)

# print('Ratios: day_of_week, num_of_week, month, COUNTRY,PRODUCT')
# get_mape(y_train,preds_train,y_test,preds_test)

# preds_test=test_df.apply(lambda x: x['num_sold']*weekday_ratios[x['day_of_week_numeric']]*week_ratios[x['num_of_week']]*month_ratios[x['month']]*country_ratios[x['country']]*store_ratios[x['store']],axis=1)

# print('Ratios: day_of_week, num_of_week, month, COUNTRY,STORE')
# get_mape(y_train,preds_train,y_test,preds_test)



# preds_test=test_df.apply(lambda x: x['num_sold']*weekday_ratios[x['day_of_week_numeric']]*week_ratios[x['num_of_week']]*month_ratios[x['month']]*product_ratios[x['product']]*store_ratios[x['store']],axis=1)

# print('Ratios: day_of_week, num_of_week, month, PRODUCT,STORE')

# get_mape(y_train,preds_train,y_test,preds_test)



# country_ratios=data_train.groupby(by='country')['num_sold'].mean()/data_train.groupby(by='country')['num_sold'].mean().mean()
# print(country_ratios)

# product_ratios=data_train.groupby(by='product')['num_sold'].mean()/data_train.groupby(by='product')['num_sold'].mean().mean()
# print(product_ratios)

# store_ratios=data_train.groupby(by='store')['num_sold'].mean()/data_train.groupby(by='store')['num_sold'].mean().mean()
# print(store_ratios
      
# print(weekday_ratios)

# week_ratios=data_train.groupby(by='num_of_week')['num_sold'].mean()/data_train.groupby(by='num_of_week')['num_sold'].mean().mean()
# print(week_ratios)

# month_ratios=data_train.groupby(by='month')['num_sold'].mean()/data_train.groupby(by='month')['num_sold'].mean().mean()
# print(month_ratios)



print(preds_test)


print(X_train.dtypes)


# feature_important = model.get_booster().get_score(importance_type='weight')



# print(feature_important)


# keys = list(feature_important.keys())
# values = list(feature_important.values())

# data = pd.DataFrame(data=(values-np.min(values))/(np.max(values)-np.min(values)), index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
# print(data.shape[0])
# data.nlargest(40, columns="score").plot(kind='barh', figsize = (20,10)) ## plot top 40 features



MAPE_train=MAPE(y_train,preds_train)
MAPE_test=MAPE(y_test,preds_test)
print(f'MAPE train:{MAPE_train}')
print(f'MAPE test:{MAPE_test}')



submission=pd.concat([submission_ids,pd.Series(np.expm1(preds_test),name='num_sold')],names=['id','num_sold'],axis=1)
print(submission)
submission.to_csv('submission.csv',index=False)

