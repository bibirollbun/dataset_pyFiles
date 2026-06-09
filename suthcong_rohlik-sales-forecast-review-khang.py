import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv').drop(['warehouse','product_unique_id'],axis=1)
calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv', parse_dates=['date'])
train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv', parse_dates=['date'])


# Chỉ cần lấy các id trong test_set để train và predict
test_id = test.unique_id.unique()

train = train[train['unique_id'].isin(test_id)]
train = train.drop(columns=['availability'])
train = train.sort_values(by='date').reset_index(drop=True)
train.head(10)


# Số ngày bán tối đa
train.date.max() - train.date.min()


train.unique_id.value_counts()   # 3625 (sản phẩm) 


# Thay vì drop thẳng dataframe gốc thì nên tạo df mới để xử lý
# df=df.drop(['type_0_discount','type_1_discount','type_2_discount','type_3_discount','type_4_discount','type_5_discount','type_6_discount','availability','sell_price_main','sell_price_main','warehouse','total_orders'],axis=1)

train_data = train[['unique_id','date','sales']].copy()
train_data.head(10)


# Biểu đồ này k thể hiện giá trị gì lắm
sns.histplot(data=df,x='unique_id',binwidth=1)


# count=0
# d=0
# i=0
# start=0
# data=[]

# while(1):
#     if(df.at[i,'unique_id']-df.at[i+1,'unique_id']!=0):
#         count=count+1
#     if(count==5):
#         break
#     i=i+1
# df_5id=df.iloc[:i+1]


# Muốn lấy 5 unique đầu tiên thì đơn giản chỉ cần
print('5 ids:',test_id[:5])
df_5id = train_data[train_data['unique_id'].isin(test_id[:5])]
df_5id


data_grouped = df_5id.groupby('unique_id').count()
data_grouped['sales'].plot.bar()
print('------------------Số data mỗi unique_id')


i = test_id[0]
i


# before
df_5id[df_5id['unique_id']==i]


# after
date_range = pd.date_range(start=df_5id['date'].min(), end=df_5id['date'].max(), freq='D')

dftmp=df_5id[df_5id['unique_id']==i]
dftmp = dftmp.set_index('date').reindex(date_range)
dftmp['unique_id'] = dftmp['unique_id'].fillna(i)
dftmp['sales'] = dftmp['sales'].fillna(0)
dftmp.reset_index()


data=[]

# chỉ gồm 3 cột unique, date và sales nên chỉ cần fill giá trị sales = 0
for _,i in enumerate(test_id[:5]):
    dftmp=df_5id[df_5id['unique_id']==i]
    dftmp = dftmp.set_index('date').reindex(date_range)
    dftmp['unique_id'] = dftmp['unique_id'].fillna(i)
    dftmp['sales'] = dftmp['sales'].fillna(0)
    dftmp = dftmp.reset_index()
    data.append(dftmp)


df1=pd.concat(data)
df1


data_grouped = df1.groupby('unique_id').count()
data_grouped['sales'].plot.bar()
print('------------------Số data mỗi unique_id')


# count=0
# d=0
# i=0
# start=0
# data=[]
# date_range = pd.date_range(start=df_5id['date'].min(), end=df_5id['date'].max(), freq='D')
# #df1.set_index('date')

# # Hạn chế dùng while khi xử lý vì tốn thời gian debug (trừ khi cần thiết)
# # while(1):
# #     if(df.at[i,'unique_id']-df.at[i+1,'unique_id']!=0):
# #         count=count+1
# #         dftmp=df.iloc[start:i]
# #         dftmp = dftmp.set_index('date').reindex(date_range)
# #         dftmp['unique_id'] = dftmp['unique_id'].fillna(df.at[i,'unique_id'])
# #         dftmp['sales'] = dftmp['sales'].fillna(0)
# #         start=i+1
# #         data.append(dftmp)
# #         #df1=pd.concat([df1,dftmp])
# #     if(count==5):
# #         break
# #     i=i+1
# df1=pd.concat(data)
# df1['date']=df1.index
# df1['index_num'] = range(1, len(df1) + 1)
# df1 = df1.reset_index(drop=True)
# df1=df1.drop(['index_num'],axis=1)
# df1 = df1[['date', 'unique_id', 'sales']]


df1.rename(columns={'date': 'ds','sales':'y'}, inplace=True)
train = df1.loc[df1['ds'] < '2024-05-20']
valid = df1.loc[(df1['ds'] >= '2024-05-20') & (df1['ds'] <= '2024-06-02')]
h = valid['ds'].nunique()
a=df1['unique_id'].unique()
a


set_dark_theme()
fig, axs = plt.subplots(4, 1, figsize=(7, 5), sharex=True)
data.iloc[:, :4].plot(
    legend   = True,
    subplots = True, 
    title    = 'Sales of store 2',
    ax       = axs, 
)
for ax in axs:
    ax.axvline(pd.to_datetime(end_train) , color='white', linestyle='--', linewidth=1.5)
    ax.axvline(pd.to_datetime(end_val) , color='white', linestyle='--', linewidth=1.5)
fig.tight_layout()
plt.show()


from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

model = StatsForecast(models=[AutoARIMA(season_length=7)], freq='D', n_jobs=-1)
model.fit(train)


p = model.predict(h=h, level=[90])
p = p.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()
wmape_ = wmape(p['y'].values, p['AutoARIMA'].values)
print(f'WMAPE: {wmape_:.2%}\n')
p


from sklearn.metrics import mean_absolute_error
wmae=mean_absolute_error(p['y'].values, p['AutoARIMA'].values)
print('WMAE: ', wmae)


fig,ax = plt.subplots(5,1, figsize=(16,12))
a=df1['unique_id'].unique()
for ax_, family in enumerate(a):
    p.loc[p['unique_id'] == family].plot(x='ds', y='y', ax=ax[ax_], label='y', title=family, linewidth=2)
    p.loc[p['unique_id'] == family].plot(x='ds', y='AutoARIMA', ax=ax[ax_], label='AutoARIMA')
    ax[ax_].set_xlabel('Date')
    ax[ax_].set_ylabel('Sales')
    ax[ax_].fill_between(p.loc[p['unique_id'] == family, 'ds'].values,
                         p.loc[p['unique_id'] == family, 'AutoARIMA-lo-90'], 
                         p.loc[p['unique_id'] == family, 'AutoARIMA-hi-90'], 
                         alpha=0.2,
                         color='orange')
    ax[ax_].set_title(f'{family} - Orange band: 90% confidence interval')
    ax[ax_].legend()
fig.tight_layout()


print(p.to_string())


df1=df1.drop(['date'],axis=1)
print(df1.to_string())


dft=df.iloc[:34]
dft = dft.set_index('date').reindex(date_range)
print(dft.to_string())


x=df.iloc[:1131]
print(x.to_string())


df1=df.iloc[:34]
date_range = pd.date_range(start=train['date'].min(), end=train['date'].max(), freq='D')
train_full = df1.set_index('date').reindex(date_range)
#train_full['index'] = range(1, len(train_full) + 1)
#train_full.set_index('index')
train_full


date_range


train.sort_values(by=['unique_id','date'])


test=pd.read_csv('sales_test.csv')
test.head()


test=test.drop(['type_0_discount','type_1_discount','type_2_discount','type_3_discount','type_4_discount','type_5_discount','type_6_discount','sell_price_main','sell_price_main','warehouse','total_orders'],axis=1)


test=test.sort_values(by='unique_id')
test=test.reset_index(drop=True)
test.head()





df.sort_values(by=['date'])


print(df.to_string())


df=df.sort_values(by=['date'])
print(df.to_string())


df['day_diff'] = df['date'].diff().dt.days
df['day_diff'].iloc[1:].nunique()



print(df.to_string())


sns.distplot(df['sales'])
plt.show()


sns.boxplot(df['sales'])
plt.show()


df['year'] = pd.DatetimeIndex(df['date']).year
df['month'] = pd.DatetimeIndex(df['date']).month
df['day'] = pd.DatetimeIndex(df['date']).day
data_grouped = df.groupby('year').mean()
data_grouped['sales'].plot.bar()
plt.show()


sns.lineplot(x=df['date'], y=df['sales'], color='dodgerblue')

plt.show()


fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True, figsize=(16,12))

sns.lineplot(df['date'], df['sales'], color='dodgerblue', ax=ax[0])
ax[0].set_title('sale', fontsize=14)

resampled_df = df[['date','sales']].resample('7D', on='date').mean().reset_index(drop=False)
sns.lineplot(resampled_df['date'], resampled_df['sales'], color='dodgerblue', ax=ax[1])
ax[1].set_title('Weekly sales', fontsize=14)

resampled_df = df[['date','sales']].resample('M', on='date').mean().reset_index(drop=False)
sns.lineplot(resampled_df['date'], resampled_df['sales'], color='dodgerblue', ax=ax[2])
ax[2].set_title('Monthly sales', fontsize=14)
plt.show()


from statsmodels.tsa.stattools import adfuller
print('sales: p_value = \t',adfuller(df['sales'].values)[1])


from statsmodels.graphics.tsaplots import plot_acf

plot_acf(df['sales'], lags=100)
plt.show()


test=pd.read_csv('sales_test.csv')
#test.head()
test=test.drop(['type_0_discount','type_1_discount','type_2_discount','type_3_discount','type_4_discount','type_5_discount','type_6_discount','sell_price_main','warehouse','total_orders'],axis=1)
test.head()


from datetime import datetime, date

test['date'] = pd.to_datetime(test['date'], format = '%Y/%m/%d')
test.head(17)


d=0
i=0
while(test.at[i,'unique_id']-test.at[i+1,'unique_id']==0):
    d=d+1
    i=i+1
print(d)


test=test.drop(['type_0_discount','type_1_discount','type_2_discount','type_3_discount','type_4_discount','type_5_discount','type_6_discount','sell_price_main','sell_price_main','warehouse','total_orders'],axis=1)
test.head()


from datetime import datetime, date

test['date'] = pd.to_datetime(test['date'], format = '%Y/%m/%d')
test.head()


test=test.sort_values(by=['unique_id'])


print(test.to_string())


pip install statsforecast --user


from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA




