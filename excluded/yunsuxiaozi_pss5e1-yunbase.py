source_file_path = '/kaggle/input/yunbase/Yunbase/baseline.py'
target_file_path = '/kaggle/working/baseline.py'
with open(source_file_path, 'r', encoding='utf-8') as file:
    content = file.read()
with open(target_file_path, 'w', encoding='utf-8') as file:
    file.write(content)


!pip install -q --requirement /kaggle/input/yunbase/Yunbase/requirements.txt  \
--no-index --find-links file:/kaggle/input/yunbase/


from baseline import Yunbase
import holidays
import pandas as pd#read csv,parquet
import numpy as np#for scientific computation of matrices
import warnings#avoid some negligible errors
#The filterwarnings () method is used to set warning filters, which can control the output method and level of warning information.
warnings.filterwarnings('ignore')

import random#provide some function to generate random_seed.
#set random seed,to make sure model can be recurrented.
def seed_everything(seed):
    np.random.seed(seed)#numpy's random seed
    random.seed(seed)#python built-in random seed
seed_everything(seed=2025)


print("deal with cool start")
from sklearn.linear_model import LinearRegression

train=pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
num_sold_mean_df=train.groupby(['country','store','product'])['num_sold'].mean().reset_index()
nan_group=num_sold_mean_df[num_sold_mean_df['num_sold'].isna()].reset_index(drop=True)


stores=train['store'].unique()
products=train['product'].unique()

for i in range(len(nan_group)):
    test_c,test_s,test_p=nan_group.loc[i]['country'],nan_group.loc[i]['store'],nan_group.loc[i]['product']
    train_X,train_y,test_X=[],[],[]
    #训练数据是用其他商店的其他商品预测这个商品
    for store in stores:
        X,y=[],[]
        for product in products:
            if product!=test_p:
                X.append(train[(train['country']==test_c)&(train['store']==store)&(train['product']==product)]['num_sold'].values)
            else:
                y.append(train[(train['country']==test_c)&(train['store']==store)&(train['product']==product)]['num_sold'].values)
        X=list(np.array(X).T)
        y=list(np.array(y).T)
        if store!=test_s:#不是测试数据        
            train_X+=X
            train_y+=y
        else:
            test_X+=X
    train_X,train_y,test_X=np.array(train_X),np.array(train_y),np.array(test_X)
    nan_row1=list(np.where(np.isnan(train_X).any(axis=1))[0])
    nan_row2=list(np.where(np.isnan(train_y).any(axis=1))[0])
    nan_rows = list(set(nan_row1+nan_row2 ))
    # 删除包含NaN值的行
    train_X = np.delete(train_X, nan_rows, axis=0)
    train_y = np.delete(train_y, nan_rows, axis=0)
    test_X = np.nan_to_num(test_X, nan=0.0)
    model=LinearRegression()
    model.fit(train_X,train_y)
    test_preds=model.predict(test_X)
    train.loc[(train['country']==test_c)&(train['store']==test_s)&(train['product']==test_p),'num_sold']=test_preds
train.head()


#drop nan or zero value
train['num_sold']=train['num_sold'].fillna(0)
train=train[train['num_sold']!=0]
#weight
train['weight']=train['num_sold'].apply(lambda x:1/np.log1p(x))
test=pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
train.head()


import requests
def get_gdp_per_capita(country, year):
    alpha3= {'China':'CN','Canada':'CAN','Argentina':'ARG',
            'Russian':'RUS','Germany':'DE','Czechia':'CZ',
            'Hungary':'HU','Japan': 'JPN','Italy':'ITA',
            'Estonia':'EST','Spain':'ESP','Singapore':'SGP',
            'Norway':'NOR','Kenya':'KEN','Finland':'FIN', 
           }
    url = "https://api.worldbank.org/v2/country/{0}/indicator/NY.GDP.PCAP.CD?date={1}&format=json".format(
        alpha3[country], year)
    response = requests.get(url).json()
    return response[1][0]['value']

countrys=[]
years=[]
gdps = []
for country in ['Canada','Finland', 'Italy', 'Kenya', 'Norway', 'Singapore']:
    for year in range(2010, 2020):
        countrys.append(country)
        years.append(year)
        gdps.append(get_gdp_per_capita(country, year))
gdp_df=pd.DataFrame({"country":countrys,'year':years,'gdp':gdps})
gdp_df.head()


def FE(df):
    #meaningless
    df.drop(['id'],axis=1,inplace=True)

    print("< holiday feature >")

    country2idx={'Canada':0, 'Finland':1, 'Italy':2, 'Kenya':3, 'Norway':4, 'Singapore':5}
    holidays_models=[holidays.Canada(),holidays.Finland(),holidays.Italy(),
     holidays.Kenya(),holidays.Norway(),holidays.Singapore()]
    df['countryidx']=df['country'].map(country2idx)
    countryidx,date=df['countryidx'].values,df['date'].values
    holiday_name=[]
    for i in range(len(date)):
        holiday_name.append(holidays_models[countryidx[i]].get(date[i]))
        
    df['holiday_name']=holiday_name
    df['is_holiday']=(df['holiday_name'].apply(lambda x:bool(x is not None))).astype(np.int8)

    df['next_day_is_holiday']=df.groupby(['country','store','product'])['is_holiday'].shift(-1).fillna(0)
    df['last_day_is_holiday']=df.groupby(['country','store','product'])['is_holiday'].shift(1).fillna(0)

    print("< date feature >")
    df['date_copy']=df['date']
    df['date_copy']=pd.to_datetime(df['date_copy'])
    
    df['dayofyear']=df['date_copy'].dt.dayofyear
    df['sin_dayofyear']=np.sin(2*np.pi*df['dayofyear']/365)
    df['cos_dayofyear']=np.cos(2*np.pi*df['dayofyear']/365)
    
    df['dayofweek']=df['date_copy'].dt.dayofweek
    df['weekday'] = df['date_copy'].dt.weekday
    df['weekend']=(df['dayofweek']>4).astype(np.int8)
    df['sin_dayofweek']=np.sin(2*np.pi*df['dayofweek']/7)
    df['cos_dayofweek']=np.cos(2*np.pi*df['dayofweek']/7)

    df['weekofyear'] = df['date_copy'].dt.isocalendar().week
    df['sin_weekofyear']=np.sin(2*np.pi*df['weekofyear']/52)
    df['cos_weekofyear']=np.cos(2*np.pi*df['weekofyear']/52)

    df['year']=df['date_copy'].dt.year
    df['quarter']=df['date_copy'].dt.quarter
    df['sin_quarter']=np.sin(2*np.pi*df['quarter']/4)
    df['cos_quarter']=np.cos(2*np.pi*df['quarter']/4)
    
    df['month']=df['date_copy'].dt.month
    df['is_month_start'] = df['date_copy'].dt.is_month_start
    df['is_month_end'] = df['date_copy'].dt.is_month_end
    df['sin_month']=np.sin(2*np.pi*df['month']/12)
    df['cos_month']=np.cos(2*np.pi*df['month']/12)
    
    df['day']=df['date_copy'].dt.day
    df['dayofmonth']=df['day']//10
    df['sin_day']=np.sin(2*np.pi*df['day']/30)
    df['cos_day']=np.cos(2*np.pi*df['day']/30)
    
    df['next_day_is_weekend']=df.groupby(['country','store','product'])['weekend'].shift(-1).fillna(0)
    df['last_day_is_weekend']=df.groupby(['country','store','product'])['weekend'].shift(1).fillna(0)

    print("< gdp feature >")
    df=df.merge(gdp_df,on=['country','year'],how='left')

    print("< category features >")
    df['month_country'] = df['month'].astype(str) + "_" + df['country']
    df['month_store'] = df['month'].astype(str) + "_" + df['store']
    df['month_product'] = df['month'].astype(str) + "_" + df['product']

    df['month_country_store'] = df['month'].astype(str) + "_" + df['country'] + "_" + df['store']
    df['month_country_product'] = df['month'].astype(str) + "_" + df['country'] + "_" + df['product']
    df['month_store_product'] = df['month'].astype(str) + "_" + df['store'] + "_" + df['product']

    print("< store weight >")
    store2weight={'Discount Stickers':0.184716,'Premium Sticker Mart':0.441564,'Stickers for Less':0.373720}
    df['storeweight']=df['store'].apply(lambda x:store2weight[x])
    
    df.drop(['date_copy'],axis=1,inplace=True)

    return df
train=FE(train)
test=FE(test)
train.head()


import datetime as dt
# Creating variables for analysis
analysis = train.copy()
# date column is separated for each element
analysis['date'] = pd.to_datetime(analysis['date'])
analysis['day'] = analysis['date'].dt.day
analysis['week'] = analysis['date'].dt.dayofweek
analysis['month'] = analysis['date'].dt.month
analysis['year'] = analysis['date'].dt.year
analysis['day_of_year'] = analysis['date'].dt.dayofyear
analysis['time_no'] = (
    analysis['date'] - dt.datetime(2017, 1, 1)) // dt.timedelta(days=1)
analysis.loc[analysis['date'] > dt.datetime(2020, 2, 29), 'time_no'] -= 1
date_columns = ['date', 'day', 'week', 'month', 'year', 'time_no']

uniques = {}
for column in analysis.columns:
    uniques[column] = analysis[column].unique().tolist()

import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(1, 3, figsize=(20, 5))

# First plot
grouped_data = analysis.groupby(['date', 'store'])['num_sold'].sum().reset_index()
for store in uniques['store']:
    store_data = grouped_data[grouped_data['store'] == store]
    axs[0].plot(store_data['date'], store_data['num_sold'], ".", label=store)
axs[0].legend()
axs[0].set_title('Aggregated Sales Over Time Per Store')

# Second plot
for store in uniques['store']:
    store_data = grouped_data[grouped_data['store'] == store]
    axs[1].plot(store_data['date'], store_data['num_sold'] /
             store_data['num_sold'].sum(), ".", label=store)
axs[1].legend()
axs[1].set_title('Aggregated Sales after Normalization Over Time Per Store')

# Third plot
for store in uniques['store']:
    store_data = grouped_data[grouped_data['store'] == store]['num_sold']
    sum_store=store_data.sum()
    axs[2] = sns.kdeplot(store_data/sum_store, label=store, ax=axs[2])
axs[2].legend()
axs[2].set_title('Normalized Sales Distribution Across Stores')

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1, 3, figsize=(20, 5))

# First plot
grouped_data_country = analysis.groupby(['date', 'country'])['num_sold'].sum().reset_index()

for country in uniques['country']:
    country_data = grouped_data_country[grouped_data_country['country'] == country]
    axs[0].plot(country_data['date'], country_data['num_sold'], ".", label=country)
axs[0].legend()
axs[0].set_title('Aggregated Sales Over Time Per Country')

# Second plot
for country in uniques['country']:
    country_data = grouped_data_country[grouped_data_country['country'] == country]
    axs[1].plot(country_data['date'], country_data['num_sold']/country_data['num_sold'].sum(), ".", label=country)
axs[1].legend()
axs[1].set_title('Aggregated Sales after Normalization Over Time Per Country')

# Third plot
for country in uniques['country']:
    country_data = grouped_data_country[grouped_data_country['country']== country]
    sum_country = country_data['num_sold'].sum()
    axs[2] = sns.kdeplot(country_data['num_sold']/sum_country, label=country)
axs[2].legend()
axs[2].set_title('Normalized Sales Distribution Across Countries')
plt.tight_layout()
plt.show()


df= analysis.groupby(['date', 'country','year'])[['num_sold']].sum().reset_index()
df=df.merge(gdp_df,on=['country','year'],how='left')

fig, axs = plt.subplots(1, 2, figsize=(15, 5))

for country in uniques['country']:
    country_data = df[df['country'] == country]
    axs[0].plot(country_data['date'], country_data['num_sold']/country_data['gdp'], ".", label=country)
axs[0].legend()
axs[0].set_title('Aggregated Sales divided by GDP Over Time Per Country')

# Third plot
for country in uniques['country']:
    country_data = df[df['country'] == country]
    axs[1] = sns.kdeplot(country_data['num_sold']/country_data['gdp'], label=country)
axs[1].legend()
axs[1].set_title(
    'Normalized Sales Distribution  divided by GDP Across Countries')
plt.tight_layout()
plt.show()


from  lightgbm import LGBMRegressor,LGBMClassifier,log_evaluation,early_stopping
category_cols=['country','store', 'product','countryidx', 'holiday_name',  'dayofyear', 
'sin_dayofyear', 'cos_dayofyear','dayofweek', 'weekday', 'weekend', 
 'sin_dayofweek', 'cos_dayofweek','weekofyear', 'sin_weekofyear',
 'cos_weekofyear', 'quarter','sin_quarter', 'cos_quarter', 'month', 
'sin_month', 'cos_month', 'day', 'dayofmonth', 'sin_day', 'cos_day', 'month_country',
'month_store', 'month_product', 'month_country_store', 'month_country_product', 'month_store_product']
yunbase=Yunbase(num_folds=1,
                  models=[(LGBMRegressor(boosting_type='gbdt',n_estimators=256,
                                         importance_type='gain',random_state=2024,
                                         num_leaves=64),'lgb'),
                         ],
                  FE=None,
                  seed=2024,
                  objective='regression',
                  metric='mape',
                  target_col='num_sold',
                  one_hot_max=-1,
                  use_high_corr_feat=False,
                  exp_mode=True,
                  plot_feature_importance=True,
               )
#预测的是num_sold/gdp试试
train['num_sold']=train['num_sold']/train['gdp']
test_preds=yunbase.purged_cross_validation(train,test,
                                category_cols=category_cols,train_test_gap=0,
                                date_col='date',test_date_range=365*3,
                                use_seasonal_features=False,
                                timestep='day',
                               )


# lb score with GDP:0.06184515293984288
# pb score with GDP:0.06248815862780902
valid_targets=np.load("/kaggle/working/Yunbase_info/lgb_seed2024_fold0_target.npy")
valid_preds=np.load("/kaggle/working/Yunbase_info/lgb_seed2024_fold0_valid_pred.npy")
valid_gdp=train[-len(valid_preds):]['gdp'].values
lb_split=len(valid_preds)//3
print(f"lb score with GDP:{yunbase.Metric(valid_targets[:lb_split]*valid_gdp[:lb_split],valid_preds[:lb_split]*valid_gdp[:lb_split])}")
print(f"pb score with GDP:{yunbase.Metric(valid_targets[lb_split:]*valid_gdp[lb_split:],valid_preds[lb_split:]*valid_gdp[lb_split:])}")


from  lightgbm import LGBMRegressor,LGBMClassifier,log_evaluation,early_stopping
yunbase=Yunbase(num_folds=1,
                  models=[(LGBMRegressor(boosting_type='gbdt',n_estimators=256,
                                         importance_type='gain',random_state=2024,
                                         num_leaves=64),'lgb'),
                         ],
                  FE=None,
                  seed=2024,
                  objective='regression',
                  metric='mape',
                  target_col='num_sold',
                  one_hot_max=8,
                  use_high_corr_feat=False,
                  exp_mode=True,
                  plot_feature_importance=True,
               )
test_preds=yunbase.purged_cross_validation(train.drop(['gdp'],axis=1),test.drop(['gdp'],axis=1),
                                category_cols=category_cols,train_test_gap=0,
                                date_col='date',test_date_range=365*3,
                                use_seasonal_features=False,
                                timestep='day',
                               )


# lb score with GDP:0.062136163864673424
# pb score with GDP:0.062293033808977716
valid_targets=np.load("/kaggle/working/Yunbase_info/lgb_seed2024_fold0_target.npy")
valid_preds=np.load("/kaggle/working/Yunbase_info/lgb_seed2024_fold0_valid_pred.npy")
valid_gdp=train[-len(valid_preds):]['gdp'].values
lb_split=len(valid_preds)//3
print(f"lb score with GDP:{yunbase.Metric(valid_targets[:lb_split]*valid_gdp[:lb_split],valid_preds[:lb_split]*valid_gdp[:lb_split])}")
print(f"pb score with GDP:{yunbase.Metric(valid_targets[lb_split:]*valid_gdp[lb_split:],valid_preds[lb_split:]*valid_gdp[lb_split:])}")


# from  lightgbm import LGBMRegressor,LGBMClassifier,log_evaluation,early_stopping
# from catboost import CatBoostRegressor,CatBoostClassifier
# from xgboost import XGBRegressor,XGBClassifier
# yunbase=Yunbase(num_folds=1,
#                   models=[(LGBMRegressor(n_estimators=256,random_state=2025),'lgb'),
#                           (CatBoostRegressor(n_estimators=256,random_state=2025),'cat'),
#                           (XGBRegressor(n_estimators=256,enable_categorical=True,
#                                         random_state=2025),'xgb')
#                          ],
#                   FE=None,
#                   seed=2025,
#                   objective='regression',
#                   metric='mape',
#                   target_col='num_sold',
#                   one_hot_max=8,
#                   use_high_corr_feat=False,
#                   exp_mode=True,
#                   plot_feature_importance=True,
#                )
# test_preds=yunbase.purged_cross_validation(train,test,
#                                 category_cols=category_cols,train_gap_each_fold=0,
#                                 date_col='date',test_date_range=1,
#                                 only_inference=True,
#                                 use_seasonal_features=False,
#                                 timestep='day',
#                                )


from  lightgbm import LGBMRegressor,LGBMClassifier,log_evaluation,early_stopping
from catboost import CatBoostRegressor,CatBoostClassifier
from xgboost import XGBRegressor,XGBClassifier
yunbase=Yunbase(num_folds=5,
                  models=[(LGBMRegressor(n_estimators=300,random_state=2025),'lgb'),
                          (CatBoostRegressor(n_estimators=300,random_state=2025),'cat'),
                          (XGBRegressor(n_estimators=300,enable_categorical=True,
                                        random_state=2025),'xgb')
                         ],
                  FE=None,
                  seed=2025,
                  objective='regression',
                  metric='mape',
                  group_col='year',
                  target_col='num_sold',
                  one_hot_max=8,
                  use_high_corr_feat=False,
                  exp_mode=True,
                  plot_feature_importance=True,
               )
yunbase.fit(train.drop(['gdp'],axis=1))
test_preds=yunbase.predict(test.drop(['gdp'],axis=1))


sub=pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")
sub[yunbase.target_col]=test_preds*test['gdp'].values
sub.loc[sub['id']%45==0,'num_sold']*=0.9
sub.to_csv("post_process.csv",index=None)
sub.head()

