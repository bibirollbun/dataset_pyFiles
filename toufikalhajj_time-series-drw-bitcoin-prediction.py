import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
import warnings
from scipy import stats
import statsmodels.api as sm
warnings.filterwarnings('ignore')
plt.style.use('seaborn-poster')


df = pd.read_csv('/kaggle/input/bitcoin-historical-data/btcusd_1-min_data.csv')


df.head()


df.tail()


df.Timestamp = pd.to_datetime(df.Timestamp, unit='s')
df.index = df.Timestamp


df = df.resample('D').mean()
df = df[df['Timestamp'] >= '2020-03-01']
df_month = df.resample('M').mean()
df_year = df.resample('A-DEC').mean()
df_Q = df.resample('Q-DEC').mean()


df.head()


df.tail()


import matplotlib.pyplot as plt


fig = plt.figure(figsize=[18, 10])
plt.suptitle('Bitcoin Exchange Rates - Mean USD', fontsize=22, fontweight='bold')


plt.subplot(221)
plt.plot(df['Close'], '-', label='By Days', color='dodgerblue', linewidth=2)
plt.title('Bitcoin by Day', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(222)
plt.plot(df_month['Close'], '-', label='By Months', color='tomato', linewidth=2)
plt.title('Bitcoin by Month', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(223)
plt.plot(df_Q['Close'], '-', label='By Quarters', color='seagreen', linewidth=2)
plt.title('Bitcoin by Quarter', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(224)
plt.plot(df_year['Close'], '-', label='By Year', color='purple', linewidth=2)
plt.title('Bitcoin by Year', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)


plt.tight_layout(rect=[0, 0, 1, 0.96])


plt.show()


import matplotlib.pyplot as plt
import statsmodels.api as sm

plt.figure(figsize=[17, 10])

result = sm.tsa.seasonal_decompose(df_month['Close'], model='multiplicative', period=12)

result.plot()
plt.suptitle('Seasonal Decomposition of Bitcoin Monthly Close Prices', fontsize=18, fontweight='bold')
plt.subplots_adjust(top=0.92)

for ax in plt.gcf().get_axes():
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

adf_result = sm.tsa.stattools.adfuller(df_month['Close'])

print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")

if adf_result[1] < 0.05:
    print("Result: The time series is stationary (reject H0).")
else:
    print("Result: The time series is NOT stationary (fail to reject H0).")

plt.show()



from scipy import stats
import statsmodels.api as sm

df_month['Close_box'], lmbda = stats.boxcox(df_month['Close'])

adf_result = sm.tsa.stattools.adfuller(df_month['Close_box'])

print(f"Box-Cox Lambda: {lmbda:.4f}")
print(f"ADF Statistic (Box-Cox transformed): {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")

if adf_result[1] < 0.05:
    print("Result: The transformed series is stationary (reject H0).")
else:
    print("Result: The transformed series is NOT stationary (fail to reject H0).")



import statsmodels.api as sm


df_month['Close_box_diff'] = df_month['Close_box'] - df_month['Close_box'].shift(12)

adf_result = sm.tsa.stattools.adfuller(df_month['Close_box_diff'].dropna())

print(f"ADF Statistic (Box-Cox transformed & seasonally differenced): {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")

if adf_result[1] < 0.05:
    print("Result: The transformed & differenced series is stationary (reject H0).")
else:
    print("Result: The transformed & differenced series is NOT stationary (fail to reject H0).")



# Second difference: difference of the seasonally differenced series by lag 1
df_month['Close_box_diff2'] = df_month['Close_box_diff'] - df_month['Close_box_diff'].shift(1)

# Drop initial NaNs due to differencing (first 13 rows)
diff2_series = df_month['Close_box_diff2'].dropna()

# Augmented Dickey-Fuller test on the twice differenced series
adf_result = sm.tsa.stattools.adfuller(diff2_series)
p_value = adf_result[1]

if p_value < 0.05:
    stationarity_message = "The series is stationary (p < 0.05)."
else:
    stationarity_message = "The series is NOT stationary (p >= 0.05)."

# Seasonal decomposition on the twice differenced series (additive model)
result = sm.tsa.seasonal_decompose(diff2_series, model='additive', period=12)

# Plotting decomposition results in 4 stacked plots
fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

result.observed.plot(ax=axes[0], color='blue', label='Observed')
axes[0].set_title('Observed')
axes[0].legend(loc='upper left')

result.trend.plot(ax=axes[1], color='orange', label='Trend')
axes[1].set_title('Trend')
axes[1].legend(loc='upper left')

result.seasonal.plot(ax=axes[2], color='green', label='Seasonal')
axes[2].set_title('Seasonal')
axes[2].legend(loc='upper left')

result.resid.plot(ax=axes[3], color='red', label='Residual')
axes[3].set_title('Residual')
axes[3].legend(loc='upper left')

plt.suptitle('Seasonal Decomposition of Close Price (Box-Cox Transformed & 2nd Differenced)', fontsize=16)
plt.subplots_adjust(top=0.9)  # Title spacing

# Show Dickey-Fuller p-value and stationarity message below plots
plt.figtext(0.1, 0.01, f'Dickey-Fuller p-value: {p_value:.5f}\n{stationarity_message}', fontsize=12)

plt.show()


import matplotlib.pyplot as plt
import statsmodels.api as sm


series = df_month['Close_box_diff2'].dropna()

plt.figure(figsize=(15, 8))

# Plot ACF
ax1 = plt.subplot(211)
sm.graphics.tsa.plot_acf(series.values.squeeze(), lags=15, ax=ax1)
ax1.set_title('Autocorrelation Function (ACF)', fontsize=14)
ax1.set_xlabel('Lags', fontsize=12)
ax1.set_ylabel('ACF', fontsize=12)
ax1.grid(True)

# Plot PACF
ax2 = plt.subplot(212)
sm.graphics.tsa.plot_pacf(series.values.squeeze(), lags=15, ax=ax2, method='ywm')  # method='ywm' is a stable default
ax2.set_title('Partial Autocorrelation Function (PACF)', fontsize=14)
ax2.set_xlabel('Lags', fontsize=12)
ax2.set_ylabel('PACF', fontsize=12)
ax2.grid(True)

plt.tight_layout()
plt.suptitle('ACF and PACF for Close Price (Box-Cox Transformed & 2nd Differenced)', fontsize=16, y=1.02)

plt.show()



import warnings
import statsmodels.api as sm
from itertools import product


# (p, q) non-seasonal AR and MA orders
p = range(0, 3)
q = range(0, 3)

# (P, Q) seasonal AR and MA orders
P = range(0, 2)
Q = range(0, 2)

# Fixed differencing parameters
d = 1    # regular differencing order 
D = 1    # seasonal differencing order (1 to handle yearly seasonality)

seasonal_period = 12  # Monthly data seasonality

# Generate all parameter combinations for order and seasonal_order
parameters = product(p, q, P, Q)
parameters_list = list(parameters)

best_aic = float("inf")
best_model = None
best_param = None
results = []

warnings.filterwarnings('ignore')  # Hide convergence warnings during fitting

print("Starting SARIMAX grid search...")

for param in parameters_list:
    try:
        order = (param[0], d, param[1])
        seasonal_order = (param[2], D, param[3], seasonal_period)

        model = sm.tsa.statespace.SARIMAX(df_month['Close_box'], 
                                          order=order,
                                          seasonal_order=seasonal_order,
                                          enforce_stationarity=False,
                                          enforce_invertibility=False).fit(disp=False)
        
        aic = model.aic

        results.append((param, aic))

        if aic < best_aic:
            best_aic = aic
            best_model = model
            best_param = param

    except Exception as e:
        # Skip invalid parameter combos silently or print minimal info
        # print(f"Skipping {param} due to error: {e}")
        continue

print(f"Grid search complete!")
print(f"Best SARIMAX model AIC: {best_aic:.3f}")
print(f"Best parameters: order={best_param[:2]}, seasonal_order=({best_param[2]}, {D}, {best_param[3]}, {seasonal_period})")



import matplotlib.pyplot as plt
import statsmodels.api as sm

plt.figure(figsize=(15, 7))

# Plot residuals time series
ax1 = plt.subplot(211)
best_model.resid[13:].plot(ax=ax1, title='Residuals')
ax1.set_ylabel('Residuals')

# Plot ACF of residuals
ax2 = plt.subplot(212)
sm.graphics.tsa.plot_acf(best_model.resid[13:].dropna().values.squeeze(), lags=12, ax=ax2)
ax2.set_title('ACF of Residuals')

plt.tight_layout()

# Dickey-Fuller test for residuals (checking stationarity/white noise)
adf_pvalue = sm.tsa.stattools.adfuller(best_model.resid[13:].dropna())[1]
print(f"Dickey–Fuller test p-value for residuals: {adf_pvalue:.6f}")

plt.show()



def invboxcox(y,lmbda):
   if lmbda == 0:
      return(np.exp(y))
   else:
      return(np.exp(np.log(lmbda*y+1)/lmbda))


from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


df_month2 = df_month[['Close']].copy()


start_date = datetime(2023, 12, 30)
end_date = datetime(2028, 3, 31)
date_list = pd.date_range(start=start_date, end=end_date, freq='M')


future = pd.DataFrame(index=date_list, columns=df_month.columns)


df_month2 = pd.concat([df_month2, future])


forecast_values = best_model.predict(start=len(df_month2)-len(future), end=len(df_month2)-1)

df_month2.loc[date_list, 'forecast'] = invboxcox(forecast_values, lmbda)


plt.figure(figsize=(15, 7))
df_month2['Close'].plot(label='Actual Close Price')
df_month2['forecast'].plot(color='r', ls='--', label='Forecasted Close Price')
plt.legend()
plt.title('Bitcoin Close Price Forecast')
plt.ylabel('Price (USD)')
plt.show()



