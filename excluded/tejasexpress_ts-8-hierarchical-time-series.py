!pip install hierarchicalforecast statsforecast datasetsforecast


import numpy as np
import pandas as pd
import os


from datasetsforecast.hierarchical import HierarchicalData
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import  BottomUp, TopDown, MiddleOut, MinTrace, ERM
from statsforecast.core import StatsForecast
from statsforecast.models import AutoARIMA, Naive
from hierarchicalforecast.evaluation import HierarchicalEvaluation

from sklearn.metrics import mean_squared_error as mse

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight') 



# general settings
class CFG:
    data_folder = '../input/tsdata-1/'
    img_dim1 = 20
    img_dim2 = 10

# adjust the parameters for displayed figures    
plt.rcParams.update({'figure.figsize': (CFG.img_dim1,CFG.img_dim2)})   


def my_rmse(x,y):
    return(np.round( np.sqrt(mse(x.values,y.values)) ,4))


# sales data calendar_df = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv', parse_dates=['date'])
calendar_df = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv', parse_dates=['date'])
calendar_df = calendar_df.loc[:, ['date', 'wm_yr_wk', 'd']]
df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_evaluation.csv')
df = df.loc[df.item_id=='FOODS_3_819']
df_T = df.melt(id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])
df_T.drop(columns=['id'], inplace=True)

sales_df = df_T.merge(calendar_df, left_on='variable', right_on='d', how='left')
sales_df.rename(columns={'value': 'sales_qty'}, inplace=True)
df = sales_df.loc[sales_df.date >= '2014-01-01', ['date', 'store_id', 'sales_qty']]
df['state_id'] = df['store_id'].str[:2]

df.head(3)


# create the long format matrix: individual stores
df_ind = df.groupby(['date', 'store_id'])[['sales_qty']].sum()
df_ind.reset_index(inplace=True)
df_ind = df_ind.T.reset_index(drop=True).T
df_ind.columns = ['ds', 'unique_id', 'sales']

# create the long format matrix: state level
df_sta = df.groupby(['date', 'state_id'])[['sales_qty']].sum()
df_sta.reset_index(inplace=True)
df_sta.columns = ['ds', 'unique_id', 'sales']

# create the long format matrix: total level
df_tot = df.groupby(['date'])[['sales_qty']].sum()
df_tot.reset_index(inplace=True)
df_tot['unique_id'] = 'Total'
df_tot.columns = ['ds', 'sales', 'unique_id' ]


# combine all three
dfx = pd.concat([df_ind, df_sta, df_tot], axis = 0)
print(df_ind.shape, df_sta.shape, df_tot.shape, dfx.shape)

# format
xset = set(dfx.unique_id)
dfx.columns = ['ds','unique_id', 'y']
dfx['ds'] = pd.to_datetime(dfx['ds'])
dfx.head(10)


S = np.zeros((len(xset), len([f for f in xset if '_' in f])))


# rows / columns
list1 = ['Total', 'CA','CA_1','CA_2','CA_3','CA_4','TX','TX_1','TX_2','TX_3','WI','WI_1','WI_2','WI_3']
list2 = ['CA_1','CA_2','CA_3','CA_4','TX_1','TX_2','TX_3','WI_1','WI_2','WI_3']
S = pd.DataFrame(0, index=list1, columns=list2)
S.columns.name = 'unique_id'


# encode the hierarchical structure
S.loc['Total'] = 1
S.loc['CA'][['CA_1','CA_2','CA_3', 'CA_4']] = 1
S.loc['TX'][['TX_1','TX_2','TX_3']] = 1
S.loc['WI'][['WI_1','WI_2','WI_3']] = 1
for x in S.columns:
    S.loc[x][x]= 1
S = S.astype(int)
S


tags = {}
tags['Country'] = np.array(['Total'], dtype=object)
tags['Country/State'] = np.array(['CA', 'TX', 'WI'], dtype=object)
tags['Country/State/Store'] = np.array(['CA_1', 'CA_2', 'CA_3', 'CA_4',  
                                        'TX_1', 'TX_2', 'TX_3',
                                        'WI_1', 'WI_2', 'WI_3'], dtype=object)
tags


horizon = 7 

x_test = dfx.groupby('unique_id').tail(horizon)
x_train = dfx.drop(x_test.index)
x_test = x_test.set_index('unique_id')
x_train = x_train.set_index('unique_id')


!python --version


pip install pyyaml numpy pandas torch scikit-learn transformers==4.40.1 datasets==2.18.0 accelerate==0.28.0


import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm

context_length = 28
prediction_length = horizon  # same as 7
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    'Maple728/TimeMoE-50M',
    device_map=device,
    trust_remote_code=True,
)

model.eval()

# Collect all series
grouped = x_train.reset_index().groupby('unique_id')
preds = []

for uid, group in tqdm(grouped):
    group = group.sort_values('ds')
    series = pd.to_numeric(group['y'], errors='coerce').dropna().values.astype(np.float32)
    
    if len(series) < context_length:
        continue  # skip short series

    context = series[-context_length:]
    context = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)  # [1, context_length]

    # normalize
    mean, std = context.mean(), context.std()
    normed_context = (context - mean) / std

    with torch.no_grad():
        output = model.generate(normed_context, max_new_tokens=prediction_length)
    
    normed_pred = output[:, -prediction_length:]
    pred = normed_pred * std + mean
    pred = pred.squeeze(0).cpu().numpy()

    # build forecast DataFrame
    ds_future = pd.date_range(group['ds'].max() + pd.Timedelta(1, 'D'), periods=prediction_length)
    pred_df = pd.DataFrame({'ds': ds_future, 'unique_id': uid, 'y_hat': pred})
    preds.append(pred_df)

# Combine predictions
x_hat = pd.concat(preds)
x_hat.columns = ['ds', 'unique_id', 'y']


xmat = pd.merge(left = x_test, right = x_hat, on = ['ds', 'unique_id'])
xmat.reset_index(inplace=True)  # ensures all index columns become normal columns

xmat.head(3)



print(xmat.columns)


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse_ = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return round(rmse, 4), round(mse_, 4), round(mae, 4)



# Merge test and forecast
xmat = pd.merge(left=x_test.reset_index(), right=x_hat, on=['ds', 'unique_id'])
xmat.rename(columns={'y_x': 'y', 'y_y': 'pred'}, inplace=True)

rmse, mse_, mae = calc_metrics(xmat['y'], xmat['pred'])
print(f"Overall RMSE: {rmse}, MSE: {mse_}, MAE: {mae}")

# Metrics per hierarchy level
for k in tags.keys():
    idx = xmat['unique_id'].isin(tags[k])
    rmse, mse_, mae = calc_metrics(xmat.loc[idx, 'y'], xmat.loc[idx, 'pred'])
    print(f"{k} RMSE: {rmse}, MSE: {mse_}, MAE: {mae}")





pip install nixtla


from nixtla import NixtlaClient

nixtla_client = NixtlaClient(api_key='nixak-Lrnl94MsYJ5TBxKWHuKeBgI9o0MBpvB1uRAkurieWQ4QAREBF7tAoezyOqLclS2TimYf9IlDqt9lD2Eq')
nixtla_client.validate_api_key()



print("x_train shape:", x_train.shape)
print("x_train columns:", x_train.columns)
print(x_train.head())



from tqdm import tqdm
import pandas as pd
from nixtla import NixtlaClient
import traceback

nixtla_client = NixtlaClient(api_key='nixak-Lrnl94MsYJ5TBxKWHuKeBgI9o0MBpvB1uRAkurieWQ4QAREBF7tAoezyOqLclS2TimYf9IlDqt9lD2Eq')  # replace with your API key
nixtla_client.validate_api_key()
x_train_debug = x_train.reset_index()
grouped = x_train_debug.groupby('unique_id')

horizon = 7
nixtla_preds = []

for uid, group in tqdm(grouped):
    try:
        if 'y' not in group.columns or group['y'].isnull().all():
            print(f"Skipping {uid}: 'y' column missing or all NaN")
            continue

        df_nixtla = group[['ds', 'y']].rename(columns={'ds': 'timestamp', 'y': 'value'})
        df_nixtla['value'] = pd.to_numeric(df_nixtla['value'], errors='coerce')
        df_nixtla = df_nixtla.dropna(subset=['value'])
        df_nixtla['timestamp'] = pd.to_datetime(df_nixtla['timestamp'])

        df_nixtla.set_index('timestamp', inplace=True)
        df_nixtla = df_nixtla.asfreq('D')
        df_nixtla['value'] = df_nixtla['value'].ffill()
        df_nixtla.reset_index(inplace=True)

        if df_nixtla.empty or df_nixtla['value'].isnull().all():
            print(f"Skipping {uid}: cleaned data is empty or all null")
            continue

        forecast_df = nixtla_client.forecast(
            df=df_nixtla,
            h=horizon,
            time_col='timestamp',
            target_col='value',
            model='timegpt-1'
        )

        print(f"{uid} forecast_df.columns:", forecast_df.columns.tolist())  # Debug

        if 'timestamp' in forecast_df.columns and 'TimeGPT' in forecast_df.columns:
            forecast_df = forecast_df.rename(columns={'timestamp': 'ds', 'TimeGPT': 'y'})
            forecast_df['unique_id'] = uid
            nixtla_preds.append(forecast_df[['ds', 'unique_id', 'y']])
        else:
            print(f"Skipping {uid}: forecast_df missing expected columns")

    except Exception as e:
        print(f"\nSkipping {uid}: {type(e).__name__}: {e}")
        traceback.print_exc()



x_hat = pd.concat(nixtla_preds, ignore_index=True)


xmat = pd.merge(x_test.reset_index(), x_hat, on=['ds', 'unique_id'])
xmat.rename(columns={'y_x': 'y', 'y_y': 'pred'}, inplace=True)

# Metrics
rmse, mse_, mae = calc_metrics(xmat['y'], xmat['pred'])
print(f"Overall RMSE: {rmse}, MSE: {mse_}, MAE: {mae}")

for k in tags.keys():
    idx = xmat['unique_id'].isin(tags[k])
    rmse, mse_, mae = calc_metrics(xmat.loc[idx, 'y'], xmat.loc[idx, 'pred'])
    print(f"{k} RMSE: {rmse}, MSE: {mse_}, MAE: {mae}")


