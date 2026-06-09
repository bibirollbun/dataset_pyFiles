import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import time
import os


def totimestamp(s):
    return int(time.mktime(datetime.strptime(s, "%d/%m/%Y").timetuple()))


data_folder = "/kaggle/input/g-research-crypto-forecasting/"
crypto_df = pd.read_csv(data_folder + 'train.csv')
asset_details = pd.read_csv(data_folder + 'asset_details.csv')
crypto_df['datetime'] = pd.to_datetime(crypto_df['timestamp'], unit='s')

print("Loaded training data shape:", crypto_df.shape)


btc = crypto_df[crypto_df['Asset_ID'] == 1].set_index('timestamp')
eth = crypto_df[crypto_df['Asset_ID'] == 6].set_index('timestamp')


btc_candle = btc.iloc[-200:]
fig = go.Figure(data=[go.Candlestick(
    x=pd.to_datetime(btc_candle.index, unit='s'),
    open=btc_candle['Open'], high=btc_candle['High'],
    low=btc_candle['Low'], close=btc_candle['Close']
)])
fig.update_layout(title='BTC Candlestick (Recent 200 mins)', xaxis_title='Time', yaxis_title='Price (USD)')
fig.show()


btc = btc.reindex(range(btc.index[0], btc.index[-1]+60, 60), method='pad')
eth = eth.reindex(range(eth.index[0], eth.index[-1]+60, 60), method='pad')

plt.figure(figsize=(14, 5))
plt.plot(pd.to_datetime(btc.index, unit='s'), btc['Close'], label='BTC')
plt.plot(pd.to_datetime(eth.index, unit='s'), eth['Close'], label='ETH')
plt.title("Bitcoin vs Ethereum: Close Price History")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()


def log_return(series, periods=1):
    return np.log(series).diff(periods=periods)

btc['log_return'] = log_return(btc['Close'])
eth['log_return'] = log_return(eth['Close'])

plt.figure(figsize=(14, 5))
plt.plot(pd.to_datetime(btc.index, unit='s'), btc['log_return'], label='BTC Log Return', alpha=0.7)
plt.plot(pd.to_datetime(eth.index, unit='s'), eth['log_return'], label='ETH Log Return', alpha=0.7)
plt.title("Log Returns Over Time")
plt.xlabel("Time")
plt.ylabel("Log Return")
plt.legend()
plt.show()


btc_mini = btc.loc[totimestamp('01/06/2021'):totimestamp('01/07/2021')]
eth_mini = eth.loc[totimestamp('01/06/2021'):totimestamp('01/07/2021')]

btc_log = log_return(btc_mini['Close']).dropna()
eth_log = log_return(eth_mini['Close']).dropna()
rolling_corr = btc_log.rolling(1000).corr(eth_log)

plt.figure(figsize=(14, 4))
plt.plot(pd.to_datetime(btc_log.index, unit='s'), rolling_corr)
plt.title("BTC-ETH Rolling Correlation (June 2021)")
plt.xlabel("Time")
plt.ylabel("Correlation")
plt.show()


all_returns = pd.DataFrame()

for asset_id in asset_details['Asset_ID']:
    asset = crypto_df[crypto_df['Asset_ID'] == asset_id].set_index('timestamp')
    asset = asset.reindex(range(asset.index[0], asset.index[-1] + 60, 60), method='pad')

    lret = log_return(asset['Close']).dropna()
    lret.name = asset_details.loc[asset_details['Asset_ID'] == asset_id, 'Asset_Name'].values[0]

    all_returns = pd.concat([all_returns, lret], axis=1)

all_returns.dropna(inplace=True)

corr_matrix = all_returns.corr()

plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
plt.colorbar()
plt.title("Log Return Correlation Matrix Across All Assets")
plt.tight_layout()
plt.show()





