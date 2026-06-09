import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb


data_folder = '../input/g-research-crypto-forecasting/'
crypto_df = pd.read_csv(data_folder + 'train.csv')
crypto_df['datetime'] = pd.to_datetime(crypto_df['timestamp'], unit='s')


def totimestamp(s):
    return int(time.mktime(datetime.strptime(s, "%d/%m/%Y").timetuple()))

def log_return(series, periods=1):
    return np.log(series).diff(periods=periods)

def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])

def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']


def prepare_features(asset_id, name):
    asset = crypto_df[crypto_df['Asset_ID'] == asset_id].set_index('timestamp')
    asset = asset.reindex(range(asset.index[0], asset.index[-1]+60, 60), method='pad')

    df = pd.DataFrame(index=asset.index)
    df[f'{name}_logret_1'] = log_return(asset['VWAP'], periods=1)
    df[f'{name}_logret_5'] = log_return(asset['VWAP'], periods=5)
    df[f'{name}_abs_logret_1'] = df[f'{name}_logret_1'].abs()
    df[f'{name}_upper_shadow'] = upper_shadow(asset)
    df[f'{name}_lower_shadow'] = lower_shadow(asset)
    df[f'{name}_target'] = asset['Target']

    return df.dropna()

btc_df = prepare_features(1, 'BTC')
eth_df = prepare_features(6, 'ETH')


combined_df = pd.concat([btc_df, eth_df], axis=1).dropna()
X = combined_df.drop(columns=['BTC_target', 'ETH_target'])
y_btc = combined_df['BTC_target']
y_eth = combined_df['ETH_target']


split_time = totimestamp('01/06/2021')
X_train = X.loc[X.index < split_time]
X_test = X.loc[X.index >= split_time]
y_btc_train = y_btc.loc[X.index < split_time]
y_btc_test = y_btc.loc[X.index >= split_time]
y_eth_train = y_eth.loc[X.index < split_time]
y_eth_test = y_eth.loc[X.index >= split_time]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


ridge_btc = Ridge(alpha=1.0)
ridge_btc.fit(X_train_scaled, y_btc_train)
y_pred_btc = ridge_btc.predict(X_test_scaled)
print("BTC Ridge Corr:", np.corrcoef(y_pred_btc, y_btc_test)[0, 1])


ridge_eth = Ridge(alpha=1.0)
ridge_eth.fit(X_train_scaled, y_eth_train)
y_pred_eth = ridge_eth.predict(X_test_scaled)
print("ETH Ridge Corr:", np.corrcoef(y_pred_eth, y_eth_test)[0, 1])


train_data = lgb.Dataset(X_train, label=y_btc_train)
test_data = lgb.Dataset(X_test, label=y_btc_test)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1
}

lgb_model = lgb.train(params, train_data, valid_sets=[test_data])
y_pred_lgb = lgb_model.predict(X_test)
print("BTC LightGBM Corr:", np.corrcoef(y_pred_lgb, y_btc_test)[0, 1])


plt.figure(figsize=(14, 4))
plt.plot(y_btc_test.index[:500], y_btc_test[:500], label='True BTC Returns', alpha=0.7)
plt.plot(y_btc_test.index[:500], y_pred_btc[:500], label='Ridge BTC Predictions', alpha=0.7)
plt.plot(y_btc_test.index[:500], y_pred_lgb[:500], label='LGB BTC Predictions', alpha=0.7)
plt.title("BTC: True vs Predicted Returns (First 500 samples)")
plt.legend()
plt.xlabel("Timestamp")
plt.ylabel("Return")
plt.grid(True)
plt.tight_layout()
plt.show()




