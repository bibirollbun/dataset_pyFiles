import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


pip install statsforecast


inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv').drop(['warehouse','product_unique_id'],axis=1)
calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv', parse_dates=['date'])
train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv', parse_dates=['date'])
submit = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')


test_id = test.unique_id.unique()

train = train[train['unique_id'].isin(test_id)]
train = train.drop(columns=['availability'])
train = train.sort_values(by='date').reset_index(drop=True)
train.head(10)


train.date.max() - train.date.min()


train.unique_id.value_counts()   # 3625 (sản phẩm) 


train_data = train[['unique_id','date','sales']].copy()
train_data.head(10)


from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA



def process_df(train_data,i):
    dftmp=train_data[train_data['unique_id']==i]
    dftmp = dftmp.set_index('date').reindex(date_range)
    dftmp['unique_id'] = dftmp['unique_id'].fillna(i)
    dftmp['sales'] = dftmp['sales'].fillna(0)
    dftmp.reset_index()
    dftmp['date'] = dftmp.index
    dftmp['indexi'] = range(1, len(dftmp) + 1)
    dftmp = dftmp.reset_index(drop=True)
    dftmp = dftmp.drop(['indexi'],axis=1)
    dftmp = dftmp[['date', 'unique_id', 'sales']]
    dftmp.rename(columns={'date': 'ds','sales':'y'}, inplace=True)
    return dftmp


#p = model.predict(h=14, level=[90])
#p['unique_id'] = p['unique_id'].astype(int)
#p['solution_id'] = p['unique_id'].astype(str)+'_'+p['ds'].astype(str)
#p


date_range = pd.date_range(start=train_data['date'].min(), end=train_data['date'].max(), freq='D')
for _,i in enumerate(test_id[0:500]):
    print(f'Predict for UniqueID {i}...')
    dftmp=process_df(train_data,i)
    model = StatsForecast(models=[AutoARIMA(season_length=7)], freq='D', n_jobs=-1)
    model.fit(dftmp)
    p = model.predict(h=14, level=[90])
    p['unique_id'] = p['unique_id'].astype(int)
    p['solution_id'] = p['unique_id'].astype(str)+'_'+p['ds'].astype(str)
    solution_id_to_sales_hat = p.set_index('solution_id')['AutoARIMA']
    # Update df1['sales_hat'] based on the mapping, leaving unmatched rows intact
    submit['sales_hat'] = submit['id'].map(solution_id_to_sales_hat).fillna(submit['sales_hat'])


submit.head(30)


submit.to_csv('submit.csv',index=False)







