import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
from datetime import datetime

asset_details = pd.read_csv('../input/g-research-crypto-forecasting/asset_details.csv')


import warnings
warnings.filterwarnings("ignore")


asset_details


# dict_ticker = {'Bitcoin Cash':'BCH',
# 'Binance Coin':'BNB',
# 'Bitcoin':'BTC',
# 'EOS.IO':'EOS',
# 'Ethereum Classic':'ETC',
# 'Ethereum':'ETH',
# 'Litecoin':'LTC',
# 'Monero':'XMR',
# 'TRON':'TRX',
# 'Stellar':'XLM',
# 'Cardano':'ADA',
# 'IOTA':'IOTA',
# 'Maker':'MKR',
# 'Dogecoin':'DOGE'}

# chunks = []

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     if dirname == '/kaggle/input/g-research-binance-1m-data-btc':
#         for filename in filenames:
#             ticker = filename.split('-')[0]
#             full_name = [i for i in dict_ticker if dict_ticker[i]==ticker][0]
#             Asset_ID = asset_details[asset_details.Asset_Name == full_name].Asset_ID.values[0]
            
#             data = pd.read_parquet(dirname+'/'+filename, engine='pyarrow')
#             data = data.reset_index()
#             data = data.rename(columns={'open_time':'timestamp','open':'Open','high':'High','low':'Low','close':'Close','quote_asset_volume':'Volume','number_of_trades':'Count'})
#             data.timestamp = (pd.to_datetime(data.timestamp).astype(int)/ 10**9).astype(int)

#             #definition not okay so far
#             data['VWAP'] = (data.Low * data.taker_buy_base_asset_volume + data.High * data.taker_buy_quote_asset_volume)/(data.taker_buy_base_asset_volume + data.taker_buy_quote_asset_volume)
            
#             data = data.drop('volume',axis=1)
            
#             data['Asset_ID'] = Asset_ID
            
#             price_column = 'Close'

#             data['Time'] = pd.to_datetime(data['timestamp'], unit='s')
#             data.sort_values(by='Time', inplace=True)
#             data.set_index(keys='Time', inplace=True)
#             data['p1'] = data[price_column].shift(freq='-1T')
#             data['p16'] = data[price_column].shift(freq='-16T')
#             data['r'] = np.log(data.p16/data.p1)
#             data.drop(['p1', 'p16'], axis=1, inplace=True)
#             data.reset_index(inplace=True)

#             chunks.append(data)

# data = pd.concat(chunks)
# data.sort_values(by='timestamp', inplace=True)


dict_ticker = {'Bitcoin Cash':'BCH',
'Binance Coin':'BNB',
'Bitcoin':'BTC',
'EOS.IO':'EOS',
'Ethereum Classic':'ETC',
'Ethereum':'ETH',
'Litecoin':'LTC',
'Monero':'XMR',
'TRON':'TRX',
'Stellar':'XLM',
'Cardano':'ADA',
'IOTA':'IOTA',
'Maker':'MKR',
'Dogecoin':'DOGE'}


for dirname, _, filenames in os.walk('/kaggle/input'):
    if dirname == '/kaggle/input/g-research-binance-1m-data-btc':
        for filename in filenames:
            ticker = filename.split('-')[0]
            full_name = [i for i in dict_ticker if dict_ticker[i]==ticker][0]
            Asset_ID = asset_details[asset_details.Asset_Name == full_name].Asset_ID.values[0]
            
            data = pd.read_parquet(dirname+'/'+filename, engine='pyarrow')
            data = data.reset_index()
            data = data.rename(columns={'open_time':'timestamp','open':'Open','high':'High','low':'Low','close':'Close','quote_asset_volume':'Volume','number_of_trades':'Count'})
            data.timestamp = (pd.to_datetime(data.timestamp).astype(int)/ 10**9).astype(int)

            #definition not okay so far
            data['VWAP'] = (data.Low * data.taker_buy_base_asset_volume + data.High * data.taker_buy_quote_asset_volume)/(data.taker_buy_base_asset_volume + data.taker_buy_quote_asset_volume)
            
            data = data.drop('volume',axis=1)
            
            data['Asset_ID'] = Asset_ID
            
            price_column = 'Close'

            data['Time'] = pd.to_datetime(data['timestamp'], unit='s')
            data.sort_values(by='Time', inplace=True)
            data.set_index(keys='Time', inplace=True)
            data['p1'] = data[price_column].shift(freq='-1T')
            data['p16'] = data[price_column].shift(freq='-16T')
            data['r'] = np.log(data.p16/data.p1)
            data.drop(['p1', 'p16'], axis=1, inplace=True)
            data.reset_index(inplace=True)


data.sort_values(by='timestamp', inplace=True)


data.head()


data['w'] = data['Asset_ID'].map(asset_details.set_index(keys='Asset_ID')['Weight'])
weight_sum = asset_details.Weight.sum()

data['weighted_asset_r'] = data.w * data.r
time_group = data.groupby('Time')

m = time_group['weighted_asset_r'].sum() / time_group['w'].sum()

data.set_index(keys=['Time'], inplace=True)
data['m'] = m
data.reset_index(inplace=True)

data['m2'] = data.m ** 2
data['mr'] = data.r * data.m

ids = list(asset_details.Asset_ID)

chunks = []
for id in ids:
    # type: pd.DataFrame
    asset = data[data.Asset_ID == id].copy()
    asset.sort_values(by='Time', inplace=True)
    asset.set_index(keys='Time', inplace=True)
    asset['mr_rolling'] = asset['mr'].rolling(window='3750T', min_periods=3750).mean()
    asset['m2_rolling'] = asset['m2'].rolling(window='3750T', min_periods=3750).mean()
    asset.reset_index(inplace=True)
    chunks.append(asset)
    debug = 1

data = pd.concat(chunks)
data.sort_values(by='Time', inplace=True)
data['beta'] = data['mr_rolling'] / data['m2_rolling']

data['Target'] = data['r'] - data['beta'] * data['m']


data.head()


len(data)


data = data.drop(['taker_buy_base_asset_volume','taker_buy_quote_asset_volume','r','w','weighted_asset_r','m','m2','mr','mr_rolling','m2_rolling','beta','Time'],axis=1)

col_ordered = ['timestamp','Asset_ID','Count','Open','High','Low','Close','Volume','VWAP','Target']
data = data[col_ordered]

dtype0 = {'Asset_ID': 'int8', 'Count': 'int32', 'Count': 'int32',
       'Open': 'float32', 'High': 'float32', 'Low': 'float32', 'Close': 'float32',
       'Volume': 'float32', 'VWAP': 'float32'}

data = data.astype(dtype0)


data.head()


data.to_parquet('add_train.parquet')

