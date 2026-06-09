import numpy as np 
import pandas as pd
pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import time
import datetime

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def encode_position(x, y, z):
    return x | (y << 16) | (z << 32)

def decode_position(encoded_value):
    x = encoded_value & 0x3FF  # Mask for the first 10 bits
    y = (encoded_value >> 16) & 0x3FF # Shift right by 10 bits and mask
    z = (encoded_value >> 32) & 0x3FF # Shift right by 20 bits and mask
    return x, y, z

x, y, z = 5, 200, 700
encoded = encode_position(x, y, z)
decoded_x, decoded_y, decoded_z = decode_position(encoded)

print(f"Original: x={x}, y={y}, z={z}") # Original: x=5, y=200, z=700
print(f"Encoded: {encoded}") # Encoded: 717317
print(f"Decoded: x={decoded_x}, y={decoded_y}, z={decoded_z}") # Decoded: x=5, y=200, z=700


from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
# X, y = ...
# regr = MultiOutputRegressor(Ridge(random_state=123)).fit(X, y)
# regr.predict(X[[0]])


df_train = pd.read_csv("/kaggle/input/minecraft-positions-predictor/train.csv", sep=",")
df_test = pd.read_csv("/kaggle/input/minecraft-positions-predictor/test.csv", sep=",")
df_test_data = pd.read_csv("/kaggle/input/minecraft-positions-predictor/test_data.csv", sep=",")
df_entity = pd.read_csv("/kaggle/input/minecraft-positions-predictor/entity_id_data.csv", sep=",")


df_train.head()


df_train.dtypes


plt.figure(figsize=(12,2))
sns.countplot(data=df_train, x="id", color='#30a2da')
plt.axhline(np.median(df_train["id"].value_counts()), color='red', linestyle="--", linewidth=1)
plt.title(f"Number of data points per `id`, median: {round(np.median(df_train['id'].value_counts()))}")
plt.show()


def get_total_seconds(t: pd.Timestamp) -> float:
    """
    convert timestamp to second
    Args:
        t (pd.Timestamp): _description_

    Returns:
        float: _description_
    """
    return t.second + t.microsecond/1000000

df_train["time"] = pd.to_datetime(df_train["timestamp"])
df_train["time_seconds"] = df_train["time"].apply(lambda x: get_total_seconds(x))
df_train[df_train["id"]==1].head()


fig, ax = plt.subplots(3,1, figsize=(12,3), sharex=True)#, gridspec_kw = {'hspace':.1})

# select first id
tmp = df_train[df_train["id"]==1].sort_values(by='time_seconds', ascending=True)

ax[0].plot(tmp["time_seconds"], tmp["x"], label = 'x coordinate')
ax[1].plot(tmp["time_seconds"], tmp["y"], label = 'y coordinate')
ax[2].plot(tmp["time_seconds"], tmp["z"], label = 'z coordinate')
ax[2].set_xlabel("time [seconds]")
_= [ax[i].legend(loc='upper left') for i in range(0,3)]
plt.tight_layout()
plt.show()


from statsmodels.tsa.stattools import grangercausalitytests
maxlag=12
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


# If a given p-value is < significance level (0.05), then, the corresponding X series (column) causes the Y (row).
tmp2 = tmp[["x", "y", "z", "time_seconds"]].copy()
tmp2.index = tmp2.time_seconds
tmp2 = tmp2.drop(columns=["time_seconds"])
tmp2 = tmp2.rename(columns={"x":"coord_x", "y":"coord_y", "z":"coord_z"})
grangers_causation_matrix(tmp2, variables = tmp2.columns)


from statsmodels.tsa.stattools import adfuller

def check_stationarity(data: pd.DataFrame, col: str):
    result = adfuller(data[col])
    print(f'ADF Test for {col}')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    if result[1] <= 0.05:
        print("Data is stationary")
    else:
        print("Data is non-stationary")
    print("")

for col in tmp2.columns:
    check_stationarity(data=tmp2, col=col)


# 1st difference
tmp2_differenced = tmp2.diff().dropna()

for col in tmp2_differenced.columns:
    check_stationarity(data=tmp2_differenced, col=col)


# 2nd difference
tmp2_differenced_2 = tmp2_differenced.diff().dropna()

for col in tmp2_differenced_2.columns:
    check_stationarity(data=tmp2_differenced_2, col=col)


from statsmodels.tsa.api import VAR

nobs = 100
x_train, x_test = tmp2_differenced_2.iloc[0:-nobs], tmp2_differenced_2.iloc[-nobs:]
print(x_train.shape, x_test.shape)

x_train_vals = x_train.copy()
x_train_vals.index = range(0, len(x_train_vals))

model = VAR(x_train_vals)
for i in range(1,10):
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')


x = model.select_order(maxlags=12)
x.summary()


# best model
model_fitted = model.fit(8)
model_fitted.summary()


# Get the lag order
lag_order = model_fitted.k_ar

forecast_input = x_test.values[-lag_order:]
forecast_input

# Forecast
fc = model_fitted.forecast(y=forecast_input, steps=nobs)
fc_df = pd.DataFrame(fc, index=x_test.index, columns=x_test.columns + '_2d')
fc_df


for col in x_test.columns:
    plt.figure(figsize=(10, 2))
    # plt.plot(train_data[col], label='Training Data')
    plt.plot(x_test[col], label='test Data')
    plt.plot(fc_df[f"{col}_2d"], label='Predictions')
    plt.title(f'VAR Model Predictions vs Actual Data for {col}: differentiated 2 times')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(loc='upper left')
    plt.show()


# TO DO: Invert the double diiferenciation to get the real forecast
# TO DO: investigate skforecaster to include exog variables




