# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# linear algebra
# data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you construct a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import numpy as np
import pandas as pd
import os
for dirname, __var, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
dataset = pd.read_csv('/kaggle/input/orange-2/train.csv')
print(dataset.head())
dataset.info()


testing = pd.read_csv('/kaggle/input/orange-2/test.csv')
testing.head()


# --- Import necessary libraries ---
# --- Load dataset ---
# --- Convert Date to datetime and sort ---
# --- Interpolate numeric columns only (prevents FutureWarning) ---
# --- Check for missing values after interpolation ---
# --- Filter for AAPL and MSFT ---
# --- Plot Close prices for AAPL & MSFT ---
# --- Decomposition: Additive and Multiplicative for AAPL ---
# --- Decomposition: Additive and Multiplicative for MSFT ---
# --- Plot AAPL Decomposition ---
# AAPL Additive
# MSFT Multiplicative

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
dataset = pd.read_csv('/kaggle/input/orange-2/train.csv')
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset.sort_values(by='Date')
dataset[dataset.select_dtypes(include='number').columns] = dataset.select_dtypes(include='number').interpolate(method='linear')
print('âœ… Missing values after interpolation:\n', dataset.isnull().sum())
aapl = dataset[dataset['Company'] == 'AAPL'].set_index('Date')
msft = dataset[dataset['Company'] == 'MSFT'].set_index('Date')
plt.figure(figsize=(10, 5))
plt.plot(aapl['Close'], label='AAPL Close', color='blue')
plt.plot(msft['Close'], label='MSFT Close', color='orange')
plt.title('Cleaned Stock Prices - AAPL vs MSFT')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
decompose_aapl_add = seasonal_decompose(aapl['Close'], model='additive', period=30)
decompose_aapl_mult = seasonal_decompose(aapl['Close'], model='multiplicative', period=30)
decompose_msft_add = seasonal_decompose(msft['Close'], model='additive', period=30)
decompose_msft_mult = seasonal_decompose(msft['Close'], model='multiplicative', period=30)
figure, axes = plt.subplots(4, 2, figsize=(15, 10))
figure.suptitle('Seasonal Decomposition: AAPL & MSFT (Additive vs Multiplicative)', fontsize=16)
decompose_aapl_add.observed.plot(ax=axes[0, 0], title='AAPL Additive - Observed')
decompose_aapl_add.trend.plot(ax=axes[1, 0], title='AAPL Additive - Trend')
decompose_aapl_add.seasonal.plot(ax=axes[2, 0], title='AAPL Additive - Seasonal')
decompose_aapl_add.resid.plot(ax=axes[3, 0], title='AAPL Additive - Residual')
decompose_msft_mult.observed.plot(ax=axes[0, 1], title='MSFT Multiplicative - Observed')
decompose_msft_mult.trend.plot(ax=axes[1, 1], title='MSFT Multiplicative - Trend')
decompose_msft_mult.seasonal.plot(ax=axes[2, 1], title='MSFT Multiplicative - Seasonal')
decompose_msft_mult.resid.plot(ax=axes[3, 1], title='MSFT Multiplicative - Residual')
plt.tight_layout()
plt.show()


# --- Import libraries ---
# --- Load and preprocess data ---
# Interpolate numeric missing values only
# --- Filter AAPL and MSFT ---
# --- Compute SMA and EWMA for AAPL ---
# --- Compute SMA and EWMA for MSFT ---
# --- Plot for AAPL ---
# --- Plot for MSFT ---

import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('/kaggle/input/orange-2/train.csv')
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset.sort_values('Date')
dataset[dataset.select_dtypes(include='number').columns] = dataset.select_dtypes(include='number').interpolate(method='linear')
aapl = dataset[dataset['Company'] == 'AAPL'].set_index('Date')
msft = dataset[dataset['Company'] == 'MSFT'].set_index('Date')
aapl['SMA_10'] = aapl['Close'].rolling(window=10).mean()
aapl['SMA_50'] = aapl['Close'].rolling(window=50).mean()
aapl['EWMA_10'] = aapl['Close'].ewm(span=10, adjust=False).mean()
aapl['EWMA_50'] = aapl['Close'].ewm(span=50, adjust=False).mean()
msft['SMA_10'] = msft['Close'].rolling(window=10).mean()
msft['SMA_50'] = msft['Close'].rolling(window=50).mean()
msft['EWMA_10'] = msft['Close'].ewm(span=10, adjust=False).mean()
msft['EWMA_50'] = msft['Close'].ewm(span=50, adjust=False).mean()
plt.figure(figsize=(12, 6))
plt.plot(aapl['Close'], label='AAPL Close', color='black', linewidth=1)
plt.plot(aapl['SMA_10'], label='SMA 10', color='blue', linestyle='--')
plt.plot(aapl['SMA_50'], label='SMA 50', color='orange', linestyle='--')
plt.plot(aapl['EWMA_10'], label='EWMA 10', color='green')
plt.plot(aapl['EWMA_50'], label='EWMA 50', color='red')
plt.title('AAPL - SMA vs EWMA (10 & 50 Days)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
plt.figure(figsize=(12, 6))
plt.plot(msft['Close'], label='MSFT Close', color='black', linewidth=1)
plt.plot(msft['SMA_10'], label='SMA 10', color='blue', linestyle='--')
plt.plot(msft['SMA_50'], label='SMA 50', color='orange', linestyle='--')
plt.plot(msft['EWMA_10'], label='EWMA 10', color='green')
plt.plot(msft['EWMA_50'], label='EWMA 50', color='red')
plt.title('MSFT - SMA vs EWMA (10 & 50 Days)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(decompose_aapl_add.trend, label='AAPL Trend', color='blue')
plt.plot(decompose_msft_add.trend, label='MSFT Trend', color='orange')
plt.title('Trend Comparison: AAPL vs MSFT')
plt.xlabel('Date')
plt.ylabel('Trend Component (Adjusted Close)')
plt.legend()
plt.show()


# --- Import necessary libraries ---
# --- Load dataset (update path if needed) ---
# --- Convert Date to datetime and clean ---
# --- Interpolate numeric missing values only ---
# --- Filter for Apple (AAPL) only ---
# --- Plot the time series ---
# --- Apply Augmented Dickey-Fuller (ADF) Test ---
# --- Display results ---
# --- Interpretation helper ---

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
dataset = pd.read_csv('/kaggle/input/orange-2/train.csv')
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset.sort_values(by='Date')
dataset = dataset.infer_objects(copy=False)
dataset.interpolate(method='linear', inplace=True)
aapl = dataset[dataset['Company'] == 'AAPL'].set_index('Date')
plt.figure(figsize=(10, 5))
plt.plot(aapl['Close'], color='blue')
plt.title('Apple (AAPL) Closing Price Time Series')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()
result = adfuller(aapl['Close'].dropna())
print(' Augmented Dickey-Fuller Test Results for AAPL')
print(f'Test Statistic: {result[0]:.4f}')
print(f'P-value: {result[1]:.4f}')
print('Critical Values:')
for key, value in result[4].items():
    print(f'   {key}: {value:.4f}')
if result[1] < 0.05:
    print('\n The series is stationary (reject null hypothesis of unit root).')
else:
    print('\n The series is NOT stationary (fail to reject null hypothesis).')


# --- Import required libraries ---
# --- Load dataset ---
# --- Convert Date to datetime and sort ---
# --- Filter for Apple (AAPL) ---
# --- Plot the original series ---
# --- Apply Log Transformation ---
# --- Apply First-Order Differencing ---
# --- Apply ADF Test again ---
# --- Interpretation helper ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
dataset = pd.read_csv('/kaggle/input/orange-2/train.csv')
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset.sort_values(by='Date')
dataset = dataset.infer_objects(copy=False)
dataset.interpolate(method='linear', inplace=True)
aapl = dataset[dataset['Company'] == 'AAPL'].set_index('Date')
aapl_close = aapl['Close']
plt.figure(figsize=(10, 5))
plt.plot(aapl_close, color='blue')
plt.title('Apple (AAPL) - Original Closing Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()
aapl_log = np.log(aapl_close)
plt.figure(figsize=(10, 5))
plt.plot(aapl_log, color='green')
plt.title('AAPL Log-Transformed Closing Price')
plt.xlabel('Date')
plt.ylabel('Log(Close)')
plt.show()
aapl_log_diff = aapl_log.diff().dropna()
plt.figure(figsize=(10, 5))
plt.plot(aapl_log_diff, color='purple')
plt.title('AAPL - Log Transformed & First-Order Differenced')
plt.xlabel('Date')
plt.ylabel('Differenced Log(Close)')
plt.show()
result_diff = adfuller(aapl_log_diff)
print(' ADF Test Results after Log + First Differencing')
print(f'Test Statistic: {result_diff[0]:.4f}')
print(f'P-value: {result_diff[1]:.4f}')
print('Critical Values:')
for key, value in result_diff[4].items():
    print(f'   {key}: {value:.4f}')
if result_diff[1] < 0.05:
    print('\n The differenced log series is stationary (reject null hypothesis).')
else:
    print('\n The series is still not stationary (fail to reject null hypothesis).')


# Task 2.3: Illustrating the role of stationarity in ARIMA modeling
# The demonstration uses Apple's (AAPL) stock prices before and after differencing.
# --- Step 1: Load and preprocess the dataset ---
# --- Step 2: Ensure date formatting and handle missing values ---
# --- Step 3: Extract AAPL closing price and evaluate stationarity ---
# --- Step 4: Apply first-order differencing and re-evaluate ---
# --- Step 5: Plot original vs differenced series ---
# --- Step 6: Visualize ACF and PACF for differenced data ---

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Define helper visualization functions
def visualize_acf(series, ax=None, lags=40):
    """Plot the Autocorrelation Function (ACF) for the given time series."""
    plot_acf(series.dropna(), ax=ax, lags=lags)

def visualize_pacf(series, ax=None, lags=40):
    """Plot the Partial Autocorrelation Function (PACF) for the given time series."""
    plot_pacf(series.dropna(), ax=ax, lags=lags)

# Load and prepare dataset
data_frame = pd.read_csv('/kaggle/input/orange-2/train.csv')
data_frame['Date'] = pd.to_datetime(data_frame['Date'])
data_frame = data_frame.sort_values(by='Date')
data_frame = data_frame.infer_objects(copy=False)
data_frame.interpolate(method='linear', inplace=True)

# Focus on Apple's (AAPL) closing price data
aapl_series = data_frame[data_frame['Company'] == 'AAPL'].set_index('Date')['Close'].dropna()

# Conduct ADF test before differencing
adf_before = adfuller(aapl_series)
print('ADF Statistic (Before Differencing):', adf_before[0])
print('p-value:', adf_before[1])

# Apply first-order differencing and test again
aapl_diff_series = aapl_series.diff().dropna()
adf_after = adfuller(aapl_diff_series)
print('\nADF Statistic (After Differencing):', adf_after[0])
print('p-value:', adf_after[1])

# Plot original and differenced time series
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(aapl_series, color='steelblue')
plt.title('AAPL Closing Price (Non-Stationary)')
plt.subplot(2, 1, 2)
plt.plot(aapl_diff_series, color='seagreen')
plt.title('AAPL After First Differencing (Stationary)')
plt.tight_layout()
plt.show()

# Plot ACF and PACF for the differenced series
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
visualize_acf(aapl_diff_series, ax=axes[0])
visualize_pacf(aapl_diff_series, ax=axes[1])
axes[0].set_title('ACF After Differencing')
axes[1].set_title('PACF After Differencing')
plt.show()



# Task 2.5: Train ARIMA models for AAPL Open/Close and produce multi-day forecasts
# The following cell corrects previous undefined `arima()` usage by employing
# the `ARIMA` class from statsmodels. It also ensures series are clean prior to modeling.

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Load and prepare dataset
data_frame = pd.read_csv('/kaggle/input/orange-2/train.csv')
data_frame['Date'] = pd.to_datetime(data_frame['Date'])
data_frame = data_frame.sort_values(by='Date')
data_frame = data_frame.infer_objects(copy=False)
data_frame.interpolate(method='linear', inplace=True)

# Isolate Apple (AAPL) records and select series of interest
aapl = data_frame[data_frame['Company'] == 'AAPL'].set_index('Date')
aapl_close = aapl['Close'].dropna()
aapl_open = aapl['Open'].dropna()

# Visualize recent Close price behavior
plt.figure(figsize=(10, 5))
plt.plot(aapl_close, label='AAPL Close Price')
plt.title('AAPL Close Price - Original Series')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Define ARIMA order (p, d, q) â€” chosen manually here for demonstration
order = (5, 1, 0)

# Build and fit ARIMA models for Close and Open series
model_close = ARIMA(aapl_close, order=order)
model_open  = ARIMA(aapl_open,  order=order)

fitted_close = model_close.fit()
fitted_open  = model_open.fit()

# Forecast horizon
forecast_days = 240

# Produce forecasts
forecast_close = fitted_close.forecast(steps=forecast_days)
forecast_open  = fitted_open.forecast(steps=forecast_days)

# Construct a DataFrame for forecasted values with a proper date index
forecast_index = pd.date_range(start=aapl.index[-1] + pd.Timedelta(days=1),
                               periods=forecast_days, freq='D')
forecast_dataset = pd.DataFrame({
    'Date': forecast_index,
    'Forecast_Open': np.asarray(forecast_open),
    'Forecast_Close': np.asarray(forecast_close)
})

print('\nğŸ“ˆ Forecast Sample:')
print(forecast_dataset.head())

# Plot last observed window against forecasted values
recent_span = 240
observed_window = aapl_close[-recent_span:] if len(aapl_close) >= recent_span else aapl_close

plt.figure(figsize=(12, 6))
plt.plot(observed_window.index, observed_window.values, label='Actual Close (Recent)', color='blue')
plt.plot(forecast_dataset['Date'], forecast_dataset['Forecast_Close'], label='Forecast Close (Next 240 days)', color='red')
plt.title('AAPL Close Price Forecast (ARIMA)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Optional evaluation: compute RMSE over the last 30 known points (if available)
if len(aapl_close) >= 60:  # ensure sufficient history for train/test split
    training = aapl_close[:-30]
    testing  = aapl_close[-30:]
    eval_model = ARIMA(training, order=order).fit()
    prediction = eval_model.forecast(steps=30)
    rmse = np.sqrt(mean_squared_error(testing.values, np.asarray(prediction)))
    print(f'âœ… RMSE (Last 30 Days): {rmse:.2f}')
else:
    print('â„¹ï¸� Not enough historical data to compute a meaningful 30-day RMSE (need >= 60 observations).')



# --- Import libraries ---
# --- Load dataset ---
# --- Convert Date and clean ---
# --- Focus on Google (GOOGL) ---
# --- Compute percentiles for price states ---
# --- Define a function to classify stock price states ---
# Low
# Medium
# High
# --- Apply classification ---
# --- Display counts of each state ---
# --- Visualize stock price with colored states ---
# --- Display sample data ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('/kaggle/input/orange-2/train.csv')
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset.sort_values(by='Date')
dataset = dataset.infer_objects(copy=False)
dataset.interpolate(method='linear', inplace=True)
googl = dataset[dataset['Company'] == 'GOOGL'].set_index('Date')
low_threshold = googl['Close'].quantile(0.33)
high_threshold = googl['Close'].quantile(0.66)

def price_state(price):
    if price <= low_threshold:
        return 'L'
    elif price <= high_threshold:
        return 'M'
    else:
        return 'H'
googl['State'] = googl['Close'].apply(price_state)
print('ğŸ“Š Price State Distribution:')
print(googl['State'].value_counts())
plt.figure(figsize=(12, 6))
plt.plot(googl.index, googl['Close'], label='GOOGL Close Price', color='gray')
plt.scatter(googl.index, googl['Close'], c=googl['State'].map({'L': 'blue', 'M': 'orange', 'H': 'red'}), label='Price State', s=25)
plt.title('GOOGL Stock Price States (Low, Medium, High)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
print('\nğŸ§¾ Sample Data with Price States:')
print(googl[['Close', 'State']].head(10))


# Q4.2 â€” Transition matrix for GOOGL price states (day-to-day)
# ---- Load & preprocess (same as Q4.1) ----
# ---- Filter GOOGL and compute thresholds ----
# ---- Build transitions: state_t -> state_t+1 ----
# next day
# ---- Count transitions ----
# ---- Convert counts to probabilities (row-normalize) ----
# ---- Display results ----
# ---- Plot heatmap of transition probabilities ----
# Annotate with probabilities

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('/kaggle/input/orange-2/train.csv')
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset.sort_values('Date')
dataset = dataset.infer_objects(copy=False)
dataset[dataset.select_dtypes(include='number').columns] = dataset.select_dtypes(include='number').interpolate(method='linear')
googl = dataset[dataset['Company'] == 'GOOGL'].set_index('Date').sort_index()
low_threshold = googl['Close'].quantile(0.33)
high_threshold = googl['Close'].quantile(0.66)

def price_state(price):
    if price <= low_threshold:
        return 'L'
    elif price <= high_threshold:
        return 'M'
    else:
        return 'H'
googl['State'] = googl['Close'].apply(price_state)
states = googl['State']
next_states = states.shift(-1)
transitions = pd.DataFrame({'from': states[:-1], 'to': next_states[:-1]})
transition_counts = pd.crosstab(transitions['from'], transitions['to']).reindex(index=['L', 'M', 'H'], columns=['L', 'M', 'H'], fill_value=0)
transition_matrix = transition_counts.div(transition_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
print('Transition Counts (from -> to):\n')
print(transition_counts)
print('\nTransition Probability Matrix (rows sum to 1):\n')
print(transition_matrix.round(4))
figure, axis = plt.subplots(figsize=(6, 5))
cax = axis.imshow(transition_matrix.values, vmin=0, vmax=1, cmap='Blues')
axis.set_xticks([0, 1, 2])
axis.set_yticks([0, 1, 2])
axis.set_xticklabels(['L', 'M', 'H'])
axis.set_yticklabels(['L', 'M', 'H'])
axis.set_xlabel('Next Day State')
axis.set_ylabel('Current Day State')
axis.set_title('GOOGL Daily State Transition Probabilities')
for index in range(3):
    for index2 in range(3):
        text = f'{transition_matrix.values[index, index2]:.2f}'
        axis.text(index2, index, text, ha='center', va='center', color='black', fontsize=12)
figure.colorbar(cax, fraction=0.046, pad=0.04, label='Probability')
plt.tight_layout()
plt.show()


# Q4.3 â€” Expected days to reach H starting from L (analytic + Monte Carlo)
# --- Load & preprocess (same pipeline as Q4.1 / Q4.2) ---
# --- Filter GOOGL and compute thresholds & states ---
# --- Build transition counts and probabilities (rows = current state) ---
# DataFrame
# --- Analytic expected hitting time: solve for E[L], E[M] with E[H]=0 ---
# Order of states: ['L','M','H']
# Build A = I - P_sub, b = 1
# 2x2
# Solve A x = b for x = expected times from non-target states
# Singular matrix (rare); fallback to pseudo-inverse
# --- Monte Carlo simulation to estimate expected hitting time from L to H ---
# Run many trials

import pandas as pd
import numpy as np
dataset = pd.read_csv('/kaggle/input/orange-2/train.csv')
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset.sort_values('Date')
dataset = dataset.infer_objects(copy=False)
dataset[dataset.select_dtypes(include='number').columns] = dataset.select_dtypes(include='number').interpolate(method='linear')
googl = dataset[dataset['Company'] == 'GOOGL'].set_index('Date').sort_index()
low_threshold = googl['Close'].quantile(0.33)
high_threshold = googl['Close'].quantile(0.66)

def price_state(price):
    if price <= low_threshold:
        return 'L'
    elif price <= high_threshold:
        return 'M'
    else:
        return 'H'
googl['State'] = googl['Close'].apply(price_state)
transitions = pd.DataFrame({'from': googl['State'][:-1].values, 'to': googl['State'][1:].values})
counts = pd.crosstab(transitions['from'], transitions['to']).reindex(index=['L', 'M', 'H'], columns=['L', 'M', 'H'], fill_value=0)
p_val = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
print('Transition Probability Matrix (P):')
print(p_val.round(4))
print()
non_target = ['L', 'M']
p_val_sub = p_val.loc[non_target, non_target].values
a_val = np.eye(len(non_target)) - p_val_sub
b_val = np.ones(len(non_target))
try:
    e_val_non_target = np.linalg.solve(a_val, b_val)
except np.linalg.LinAlgError:
    e_val_non_target = np.linalg.pinv(a_val).dot(b_val)
e_val = {'L': e_val_non_target[0], 'M': e_val_non_target[1], 'H': 0.0}
print('Analytic expected days to reach H (hitting time):')
print(f"  E[L -> H] = {e_val['L']:.4f} days")
print(f"  E[M -> H] = {e_val['M']:.4f} days")
print()
import random

def simulate_one(start_state, p_val_dataset, target='H', max_steps=10000):
    state = start_state
    steps = 0
    while state != target and steps < max_steps:
        probs = p_val_dataset.loc[state].values
        states = p_val_dataset.columns.tolist()
        state = random.choices(states, weights=probs, k=1)[0]
        steps += 1
    return steps
n_val_trials = 5000
results = []
for __var in range(n_val_trials):
    t_val = simulate_one('L', p_val, target='H')
    results.append(t_val)
results = np.array(results)
mean_sim = results.mean()
median_sim = np.median(results)
pct = np.percentile(results, 90)
print(f'Monte Carlo estimate (n={n_val_trials}):')
print(f'  Mean steps from L to H = {mean_sim:.4f}')
print(f'  Median = {median_sim:.0f}')
print(f'  90th percentile = {pct:.0f}')
print()
print('Note: If many trials hit max_steps (indicating possible non-reachability or near-zero prob), increase max_steps or check chain connectivity.')


import pandas as pd
import numpy as np
import random

# --- Load and preprocess data ---
df = pd.read_csv('/kaggle/input/orange-2/train.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').infer_objects(copy=False)
df[df.select_dtypes(include='number').columns] = df.select_dtypes(include='number').interpolate(method='linear')

# --- Focus on GOOGL and compute daily percentage change ---
googl = df[df['Company'] == 'GOOGL'].set_index('Date').sort_index()
googl = googl[['Close']].copy()
googl['Pct_Change'] = googl['Close'].pct_change()
googl['Is_Crash'] = googl['Pct_Change'] <= -0.10

# --- Define thresholds for Low/Medium/High price bands ---
low_thr = googl['Close'].quantile(0.33)
high_thr = googl['Close'].quantile(0.66)

def classify_state(price):
    if price <= low_thr:
        return 'L'
    elif price <= high_thr:
        return 'M'
    else:
        return 'H'

# --- Assign base states and override crashes ---
googl['State'] = googl['Close'].apply(classify_state)
googl.loc[googl['Is_Crash'], 'State'] = 'C'

# --- Build transition probability matrix ---
current = googl['State']
next_state = current.shift(-1)
transitions = pd.DataFrame({'from': current[:-1].values, 'to': next_state[:-1].values})

states = ['L', 'M', 'H', 'C']
count_matrix = pd.crosstab(transitions['from'], transitions['to']).reindex(index=states, columns=states, fill_value=0)
prob_matrix = count_matrix.div(count_matrix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

# Make 'C' absorbing
prob_matrix.loc['C'] = 0.0
prob_matrix.loc['C', 'C'] = 1.0

# âœ… FIXED: Now prob_matrix exists
P_arr = prob_matrix.loc[states, states].values  # shape (4,4)

# Cumulative distribution per row for fast sampling
cdf_matrix = np.cumsum(P_arr, axis=1)

print("âœ… prob_matrix defined and P_arr computed successfully!")
print(prob_matrix)



# --- Q5.1: Investor Retention during Stable Macroeconomic Conditions ---
# --- Load dataset ---
# --- Convert Date to datetime and sort ---
# --- Interpolate missing numeric values ---
# --- Check available macro columns ---
# --- Ensure required macro variables exist ---
# --- Define Stable Macroeconomic Conditions ---
# --- Compute Average Trading Volume per Company during Stable Conditions ---
# --- Identify Company with Highest Investor Retention ---
# --- Visualization ---

import pandas as pd
dataset = pd.read_csv('/kaggle/input/orange-2/train.csv')
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset = dataset.sort_values('Date')
dataset = dataset.infer_objects(copy=False)
dataset.interpolate(method='linear', inplace=True)
print('Available columns:', dataset.columns.tolist())
if not {'Inflation', 'Interest'}.issubset(dataset.columns):
    raise value_error("Dataset must include 'Inflation' and 'Interest' columns for this task.")
median_inflation = dataset['Inflation'].median()
median_interest = dataset['Interest'].median()
stable_mask = (dataset['Inflation'] < median_inflation) & (dataset['Interest'] < median_interest)
stable_periods = dataset[stable_mask]
avg_volume_stable = stable_periods.groupby('Company')['Volume'].mean().sort_values(ascending=False)
print('\nğŸ“Š Average Trading Volume during Stable Conditions:')
print(avg_volume_stable)
highest_retention_company = avg_volume_stable.idxmax()
print(f'\nğŸ�† Company with Highest Investor Retention during Stable Conditions: {highest_retention_company}')
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
avg_volume_stable.plot(kind='bar', color=['steelblue', 'orange', 'green'])
plt.title('Average Trading Volume During Stable Macroeconomic Conditions')
plt.xlabel('Company')
plt.ylabel('Average Volume')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()


# --- Download 3 companies' stock data ---
# auto_adjust=False to keep actual open/close values
# Keep only necessary columns and flatten if MultiIndex
# ensure single-level columns
# --- Combine all into one DataFrame ---
# --- Ensure columns are numeric ---
# --- Compute Value = Volume Ã— (Close - Open) ---
# --- Compute 30-day rolling mean and std for each company ---
# --- Compute average std per company (lower std = more consistent) ---

import yfinance as yf
import pandas as pd
companies = ['GOOGL', 'AAPL', 'MSFT']
dataset_1 = []
for company in companies:
    dataset = yf.download(company, start='2020-01-01', end='2024-12-31', auto_adjust=False)
    dataset = dataset[['Open', 'Close', 'Volume']].copy()
    dataset.columns = ['Open', 'Close', 'Volume']
    dataset['Company'] = company
    dataset_1.append(dataset)
combined = pd.concat(dataset_1).dropna().reset_index()
combined['Open'] = pd.to_numeric(combined['Open'], errors='coerce')
combined['Close'] = pd.to_numeric(combined['Close'], errors='coerce')
combined['Volume'] = pd.to_numeric(combined['Volume'], errors='coerce')
combined['Value'] = (combined['Volume'] * (combined['Close'] - combined['Open'])).astype(float)
rolling_stats = []
for company, group in combined.groupby('Company'):
    buffer = group.copy()
    buffer['Rolling_Mean'] = buffer['Value'].rolling(30).mean()
    buffer['Rolling_Std'] = buffer['Value'].rolling(30).std()
    rolling_stats.append(buffer)
rolling_stats = pd.concat(rolling_stats)
avg_std = rolling_stats.groupby('Company')['Rolling_Std'].mean().sort_values()
print('\nğŸ“Š Average 30-Day Rolling Std (Consistency Measure):')
print(avg_std)
print(f'\nğŸ�† Most consistent value: {avg_std.idxmin()}')


# --- Download data for 3 companies ---
# âœ… Flatten MultiIndex columns if present
# --- Combine all into one DataFrame ---
# --- âœ… Calculate daily returns safely ---
# ensure Series
# --- Define Gain/Loss state ---
# --- Function to compute transition matrix ---
# --- Compute transition matrix for each company ---
# Most likely transition

import yfinance as yf
import pandas as pd
companies = ['GOOGL', 'AAPL', 'MSFT']
dataset_1 = []
for company in companies:
    dataset = yf.download(company, start='2020-01-01', end='2024-12-31', auto_adjust=False)
    dataset.columns = [column[0] if isinstance(column, tuple) else column for column in dataset.columns]
    dataset = dataset[['Close']].copy()
    dataset['Company'] = company
    dataset_1.append(dataset)
combined = pd.concat(dataset_1).dropna().reset_index()
returns_list = []
for company in companies:
    buffer = combined[combined['Company'] == company].copy()
    buffer['Return'] = buffer['Close'].pct_change().squeeze()
    returns_list.append(buffer)
combined = pd.concat(returns_list)
combined['State'] = combined['Return'].apply(lambda feature: 'Gain' if feature > 0 else 'Loss')

def transition_matrix(states):
    transitions = pd.crosstab(index=states[:-1], columns=states[1:], normalize='index')
    return transitions.fillna(0)
for company in companies:
    sub = combined[combined['Company'] == company]
    matrix = transition_matrix(sub['State'].values)
    print(f'\nğŸ”¹ {company} Return Transition Matrix:')
    print(matrix)
    most_likely = matrix.stack().idxmax()
    prob = matrix.stack().max()
    print(f'â�¡ï¸� Most likely transition for {company}: {most_likely[0]} â†’ {most_likely[1]} ({prob:.2f} probability)')


# Your Answerimport numpy as np
# --- Download stock data (example: AAPL, GOOGL, MSFT) ---
# flatten columns
# Combine all company data
# --- Calculate Log Returns ---
# --- Compute 10-day Rolling Volatility (std of log returns) ---
# --- Compute Proxy for Retention = Avg Trading Volume (10-day rolling mean) ---
# --- Display sample results ---

import pandas as pd
import yfinance as yf
companies = ['AAPL', 'GOOGL', 'MSFT']
dataset_1 = []
for company in companies:
    dataset = yf.download(company, start='2020-01-01', end='2024-12-31', auto_adjust=False)
    dataset.columns = [column[0] if isinstance(column, tuple) else column for column in dataset.columns]
    dataset['Company'] = company
    dataset_1.append(dataset)
combined = pd.concat(dataset_1).reset_index()
combined['Log_Return'] = np.log(combined['Close'] / combined['Close'].shift(1))
combined['Volatility'] = combined.groupby('Company')['Log_Return'].transform(lambda feature: feature.rolling(10).std())
combined['Retention_Proxy'] = combined.groupby('Company')['Volume'].transform(lambda feature: feature.rolling(10).mean())
print(combined[['Date', 'Company', 'Log_Return', 'Volatility', 'Retention_Proxy']].dropna().head(15))


# --- Download AAPL data ---
# --- Compute Log Returns and Volatility (10-day rolling std of log returns) ---
# --- Create Proxy Macroeconomic Data (for demo) ---
# In real use, you'd merge with actual inflation & interest data
# 2% - 6%
# 3% - 7%
# --- Drop missing volatility values ---
# --- Define features and target ---
# --- Add constant term for regression ---
# --- Fit OLS Regression Model ---
# --- Display summary ---

import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
dataset = yf.download('AAPL', start='2020-01-01', end='2024-12-31')[['Close', 'Volume']]
dataset = dataset.dropna().reset_index()
dataset['Log_Return'] = np.log(dataset['Close'] / dataset['Close'].shift(1))
dataset['Volatility'] = dataset['Log_Return'].rolling(window=10).std()
np.random.seed(42)
dataset['Inflation'] = np.random.uniform(2, 6, len(dataset))
dataset['Interest_Rate'] = np.random.uniform(3, 7, len(dataset))
dataset['Log_Volume'] = np.log(dataset['Volume'])
dataset = dataset.dropna()
feature_1 = dataset[['Inflation', 'Interest_Rate', 'Log_Volume']]
target_1 = dataset['Volatility']
feature_1 = sm.add_constant(feature_1)
estimator = sm.OLS(target_1, feature_1).fit()
print(estimator.summary())


# Get residuals and fitted values
# --- Histogram of residuals ---
# --- Q-Q Plot ---
# --- Residuals vs Fitted Plot ---

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
residuals = estimator.resid
fitted = estimator.fittedvalues
plt.figure(figsize=(6, 4))
sns.histplot(residuals, bins=30, kde=True)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
plt.figure(figsize=(6, 4))
stats.probplot(residuals, dist='norm', plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()
plt.figure(figsize=(6, 4))
sns.scatterplot(x=fitted, y=residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()


# --- Generate submission.csv from ARIMA model forecast ---
import pandas as pd

# === Create submission.csv using Test.csv ===

# Load test file (240 rows expected)
test_df = pd.read_csv("/kaggle/input/orange-2/test.csv")

# Ensure it has a Date column (if not, use index or forecast_index)
if "Date" not in test_df.columns:
    test_df["Date"] = forecast_dataset["Date"].values[:len(test_df)]

# Use your forecasted values for the submission
submission = pd.DataFrame({
    "date": test_df["Date"],
    "forecasted_open": forecast_dataset["Forecast_Open"].values[:len(test_df)],
    "forecasted_close": forecast_dataset["Forecast_Close"].values[:len(test_df)]
})

# Save to submission.csv
submission.to_csv("/kaggle/working/submission.csv", index=False)

print("âœ… submission.csv created successfully with", len(submission), "rows")
display(submission.head())









