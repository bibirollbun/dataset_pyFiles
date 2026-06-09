# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
df = pd.read_csv('/kaggle/input/orange-2/train.csv')

print(df.head())
df.info()
 


test = pd.read_csv("/kaggle/input/orange-2/test.csv")
test.head()


# --- Import necessary libraries ---
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Load dataset ---
df = pd.read_csv('/kaggle/input/orange-2/train.csv')

# --- Convert Date to datetime and sort ---
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# --- Interpolate numeric columns only (prevents FutureWarning) ---
df[df.select_dtypes(include='number').columns] = (
    df.select_dtypes(include='number').interpolate(method='linear')
)

# --- Check for missing values after interpolation ---
print("âœ… Missing values after interpolation:\n", df.isnull().sum())


# --- Filter for AAPL and MSFT ---
aapl = df[df['Company'] == 'AAPL'].set_index('Date')
msft = df[df['Company'] == 'MSFT'].set_index('Date')

# --- Plot Close prices for AAPL & MSFT ---
plt.figure(figsize=(10,5))
plt.plot(aapl['Close'], label='AAPL Close', color='blue')
plt.plot(msft['Close'], label='MSFT Close', color='orange')
plt.title('Cleaned Stock Prices - AAPL vs MSFT')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# --- Decomposition: Additive and Multiplicative for AAPL ---
decompose_aapl_add = seasonal_decompose(aapl['Close'], model='additive', period=30)
decompose_aapl_mult = seasonal_decompose(aapl['Close'], model='multiplicative', period=30)

# --- Decomposition: Additive and Multiplicative for MSFT ---
decompose_msft_add = seasonal_decompose(msft['Close'], model='additive', period=30)
decompose_msft_mult = seasonal_decompose(msft['Close'], model='multiplicative', period=30)

# --- Plot AAPL Decomposition ---
fig, axes = plt.subplots(4, 2, figsize=(15,10))
fig.suptitle('Seasonal Decomposition: AAPL & MSFT (Additive vs Multiplicative)', fontsize=16)

# AAPL Additive
decompose_aapl_add.observed.plot(ax=axes[0,0], title='AAPL Additive - Observed')
decompose_aapl_add.trend.plot(ax=axes[1,0], title='AAPL Additive - Trend')
decompose_aapl_add.seasonal.plot(ax=axes[2,0], title='AAPL Additive - Seasonal')
decompose_aapl_add.resid.plot(ax=axes[3,0], title='AAPL Additive - Residual')

# MSFT Multiplicative
decompose_msft_mult.observed.plot(ax=axes[0,1], title='MSFT Multiplicative - Observed')
decompose_msft_mult.trend.plot(ax=axes[1,1], title='MSFT Multiplicative - Trend')
decompose_msft_mult.seasonal.plot(ax=axes[2,1], title='MSFT Multiplicative - Seasonal')
decompose_msft_mult.resid.plot(ax=axes[3,1], title='MSFT Multiplicative - Residual')

plt.tight_layout()
plt.show()



# --- Import libraries ---
import pandas as pd
import matplotlib.pyplot as plt

# --- Load and preprocess data ---
df = pd.read_csv('/kaggle/input/orange-2/train.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Interpolate numeric missing values only
df[df.select_dtypes(include='number').columns] = (
    df.select_dtypes(include='number').interpolate(method='linear')
)

# --- Filter AAPL and MSFT ---
aapl = df[df['Company'] == 'AAPL'].set_index('Date')
msft = df[df['Company'] == 'MSFT'].set_index('Date')

# --- Compute SMA and EWMA for AAPL ---
aapl['SMA_10'] = aapl['Close'].rolling(window=10).mean()
aapl['SMA_50'] = aapl['Close'].rolling(window=50).mean()
aapl['EWMA_10'] = aapl['Close'].ewm(span=10, adjust=False).mean()
aapl['EWMA_50'] = aapl['Close'].ewm(span=50, adjust=False).mean()

# --- Compute SMA and EWMA for MSFT ---
msft['SMA_10'] = msft['Close'].rolling(window=10).mean()
msft['SMA_50'] = msft['Close'].rolling(window=50).mean()
msft['EWMA_10'] = msft['Close'].ewm(span=10, adjust=False).mean()
msft['EWMA_50'] = msft['Close'].ewm(span=50, adjust=False).mean()

# --- Plot for AAPL ---
plt.figure(figsize=(12,6))
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

# --- Plot for MSFT ---
plt.figure(figsize=(12,6))
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



plt.figure(figsize=(12,6))
plt.plot(decompose_aapl_add.trend, label='AAPL Trend', color='blue')
plt.plot(decompose_msft_add.trend, label='MSFT Trend', color='orange')
plt.title('Trend Comparison: AAPL vs MSFT')
plt.xlabel('Date')
plt.ylabel('Trend Component (Adjusted Close)')
plt.legend()
plt.show()



# --- Import necessary libraries ---
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# --- Load dataset (update path if needed) ---
df = pd.read_csv('/kaggle/input/orange-2/train.csv')

# --- Convert Date to datetime and clean ---
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# --- Infer numeric types and interpolate numeric missing values only ---
df = df.infer_objects(copy=False)
df[df.select_dtypes(include=['number']).columns] = df.select_dtypes(include=['number']).interpolate(method='linear')

# --- Filter for Apple (AAPL) only ---
aapl = df[df['Company'] == 'AAPL'].set_index('Date')

# --- Plot the time series ---
plt.figure(figsize=(10,5))
plt.plot(aapl['Close'], color='blue')
plt.title('Apple (AAPL) Closing Price Time Series')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

# --- Apply Augmented Dickey-Fuller (ADF) Test ---
result = adfuller(aapl['Close'].dropna())

# --- Display results ---
print("Augmented Dickey-Fuller Test Results for AAPL")
print(f"Test Statistic: {result[0]:.4f}")
print(f"P-value: {result[1]:.4f}")
print("Critical Values:")
for key, value in result[4].items():
    print(f"   {key}: {value:.4f}")

# --- Interpretation helper ---
if result[1] < 0.05:
    print("\nâœ… The series is stationary (reject null hypothesis of unit root).")
else:
    print("\nâš ï¸� The series is NOT stationary (fail to reject null hypothesis).")



# --- Import required libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# --- Load dataset ---
df = pd.read_csv('/kaggle/input/orange-2/train.csv')

# --- Convert Date to datetime and sort ---
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')
df = df.infer_objects(copy=False)

# âœ… Interpolate only numeric columns (avoid FutureWarning)
df[df.select_dtypes(include=['number']).columns] = (
    df.select_dtypes(include=['number']).interpolate(method='linear')
)

# --- Filter for Apple (AAPL) ---
aapl = df[df['Company'] == 'AAPL'].set_index('Date')
aapl_close = aapl['Close']

# --- Plot the original series ---
plt.figure(figsize=(10,5))
plt.plot(aapl_close, color='blue')
plt.title('Apple (AAPL) - Original Closing Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

# --- Apply Log Transformation ---
aapl_log = np.log(aapl_close)

plt.figure(figsize=(10,5))
plt.plot(aapl_log, color='green')
plt.title('AAPL Log-Transformed Closing Price')
plt.xlabel('Date')
plt.ylabel('Log(Close)')
plt.show()

# --- Apply First-Order Differencing ---
aapl_log_diff = aapl_log.diff().dropna()

plt.figure(figsize=(10,5))
plt.plot(aapl_log_diff, color='purple')
plt.title('AAPL - Log Transformed & First-Order Differenced')
plt.xlabel('Date')
plt.ylabel('Differenced Log(Close)')
plt.show()

# --- Apply ADF Test again ---
result_diff = adfuller(aapl_log_diff)

print("ADF Test Results after Log + First Differencing")
print(f"Test Statistic: {result_diff[0]:.4f}")
print(f"P-value: {result_diff[1]:.4f}")
print("Critical Values:")
for key, value in result_diff[4].items():
    print(f"   {key}: {value:.4f}")

# --- Interpretation helper ---
if result_diff[1] < 0.05:
    print("\nâœ… The differenced log series is stationary (reject null hypothesis).")
else:
    print("\nâš ï¸� The series is still not stationary (fail to reject null hypothesis).")



# Task 2.3: Demonstrate importance of stationarity for ARIMA
# We'll show before and after differencing for Apple (AAPL)

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# --- Load dataset ---
df = pd.read_csv('/kaggle/input/orange-2/train.csv')

# --- Convert and clean ---
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# âœ… Fix FutureWarning by interpolating only numeric columns
df = df.infer_objects(copy=False)
df[df.select_dtypes(include=['number']).columns] = (
    df.select_dtypes(include=['number']).interpolate(method='linear')
)

# --- Focus on Apple (AAPL) closing price ---
aapl = df[df['Company'] == 'AAPL'].set_index('Date')['Close'].dropna()

# --- ADF test before differencing ---
adf_before = adfuller(aapl)
print("ADF Statistic (Before Differencing):", adf_before[0])
print("p-value:", adf_before[1])

# --- Apply first-order differencing ---
aapl_diff = aapl.diff().dropna()

# --- ADF test after differencing ---
adf_after = adfuller(aapl_diff)
print("\nADF Statistic (After Differencing):", adf_after[0])
print("p-value:", adf_after[1])

# --- Plot before vs after differencing ---
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(aapl, color='blue')
plt.title("AAPL Original Closing Price (Non-Stationary)")

plt.subplot(2,1,2)
plt.plot(aapl_diff, color='green')
plt.title("AAPL After First Differencing (Stationary)")
plt.tight_layout()
plt.show()

# --- ACF and PACF plots ---
fig, axes = plt.subplots(1, 2, figsize=(12,4))
plot_acf(aapl_diff, ax=axes[0])
plot_pacf(aapl_diff, ax=axes[1])
axes[0].set_title("ACF after Differencing")
axes[1].set_title("PACF after Differencing")
plt.show()



# --- Import necessary libraries ---
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# --- Load your training data ---
df = pd.read_csv('/kaggle/input/orange-2/train.csv')

# --- Convert Date and clean ---
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')
df = df.infer_objects(copy=False)
df.interpolate(method='linear', inplace=True)

# --- Focus on Apple (AAPL) ---
aapl = df[df['Company'] == 'AAPL'].set_index('Date')

# âœ… Set frequency to business days and handle missing values
aapl = aapl.asfreq('B')  # 'B' = Business day frequency
aapl.interpolate(method='linear', inplace=True)

# --- Select Close and Open price series ---
aapl_close = aapl['Close']
aapl_open = aapl['Open']

# --- Plot actual Close prices ---
plt.figure(figsize=(10,5))
plt.plot(aapl_close, label='AAPL Close Price')
plt.title('AAPL Close Price - Original Series')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# --- Choose ARIMA parameters manually ---
# You can tune (p,d,q) based on ADF or auto_arima later
order = (5, 1, 0)

# --- Train ARIMA models ---
print("Training ARIMA models...")

model_close = ARIMA(aapl_close, order=order)
model_open = ARIMA(aapl_open, order=order)

model_close_fit = model_close.fit()
model_open_fit = model_open.fit()

print("âœ… ARIMA models trained successfully.")

# --- Forecast next 240 days ---
forecast_days = 240
forecast_close = model_close_fit.forecast(steps=forecast_days)
forecast_open = model_open_fit.forecast(steps=forecast_days)

# --- Create forecast DataFrame ---
forecast_df = pd.DataFrame({
    'Date': pd.date_range(start=aapl.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B'),
    'Forecast_Open': forecast_open,
    'Forecast_Close': forecast_close
})

print("\nğŸ“ˆ Forecast Sample:")
print(forecast_df.head())

# --- Plot forecast vs recent actual ---
plt.figure(figsize=(12,6))
plt.plot(aapl_close[-240:], label='Actual Close (Last 240 days)', color='blue')
plt.plot(forecast_df['Date'], forecast_df['Forecast_Close'], label='Forecast Close (Next 240 days)', color='red')
plt.title('AAPL Close Price Forecast (ARIMA)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- Optional: Evaluate using last 30 known points (if available) ---
if len(aapl_close) > 30:
    train = aapl_close[:-30]
    test = aapl_close[-30:]

    model_eval = ARIMA(train, order=order).fit()
    pred = model_eval.forecast(steps=30)

    rmse = np.sqrt(mean_squared_error(test, pred))
    print(f"âœ… RMSE (Last 30 Days): {rmse:.2f}")
else:
    print("âš ï¸� Not enough data points for RMSE evaluation.")



# --- Import libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Load dataset ---
df = pd.read_csv('/kaggle/input/orange-2/train.csv')

# --- Convert Date and clean ---
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')
df = df.infer_objects(copy=False)
df.interpolate(method='linear', inplace=True)

# --- Focus on Google (GOOGL) ---
googl = df[df['Company'] == 'GOOGL'].set_index('Date')

# --- Compute percentiles for price states ---
low_threshold = googl['Close'].quantile(0.33)
high_threshold = googl['Close'].quantile(0.66)

# --- Define a function to classify stock price states ---
def price_state(price):
    if price <= low_threshold:
        return 'L'   # Low
    elif price <= high_threshold:
        return 'M'   # Medium
    else:
        return 'H'   # High

# --- Apply classification ---
googl['State'] = googl['Close'].apply(price_state)

# --- Display counts of each state ---
print("ğŸ“Š Price State Distribution:")
print(googl['State'].value_counts())

# --- Visualize stock price with colored states ---
plt.figure(figsize=(12,6))
plt.plot(googl.index, googl['Close'], label='GOOGL Close Price', color='gray')
plt.scatter(googl.index, googl['Close'], 
            c=googl['State'].map({'L':'blue', 'M':'orange', 'H':'red'}), 
            label='Price State', s=25)
plt.title('GOOGL Stock Price States (Low, Medium, High)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

# --- Display sample data ---
print("\nğŸ§¾ Sample Data with Price States:")
print(googl[['Close', 'State']].head(10))



# Q4.2 â€” Transition matrix for GOOGL price states (day-to-day)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Load & preprocess (same as Q4.1) ----
df = pd.read_csv('/kaggle/input/orange-2/train.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df = df.infer_objects(copy=False)
df[df.select_dtypes(include='number').columns] = (
    df.select_dtypes(include='number').interpolate(method='linear')
)

# ---- Filter GOOGL and compute thresholds ----
googl = df[df['Company'] == 'GOOGL'].set_index('Date').sort_index()
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

# ---- Build transitions: state_t -> state_t+1 ----
states = googl['State']
next_states = states.shift(-1)   # next day
transitions = pd.DataFrame({'from': states[:-1], 'to': next_states[:-1]})

# ---- Count transitions ----
transition_counts = pd.crosstab(transitions['from'], transitions['to']).reindex(index=['L','M','H'], columns=['L','M','H'], fill_value=0)

# ---- Convert counts to probabilities (row-normalize) ----
transition_matrix = transition_counts.div(transition_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

# ---- Display results ----
print("Transition Counts (from -> to):\n")
print(transition_counts)
print("\nTransition Probability Matrix (rows sum to 1):\n")
print(transition_matrix.round(4))

# ---- Plot heatmap of transition probabilities ----
fig, ax = plt.subplots(figsize=(6,5))
cax = ax.imshow(transition_matrix.values, vmin=0, vmax=1, cmap='Blues')
ax.set_xticks([0,1,2])
ax.set_yticks([0,1,2])
ax.set_xticklabels(['L','M','H'])
ax.set_yticklabels(['L','M','H'])
ax.set_xlabel('Next Day State')
ax.set_ylabel('Current Day State')
ax.set_title('GOOGL Daily State Transition Probabilities')

# Annotate with probabilities
for i in range(3):
    for j in range(3):
        text = f"{transition_matrix.values[i,j]:.2f}"
        ax.text(j, i, text, ha='center', va='center', color='black', fontsize=12)

fig.colorbar(cax, fraction=0.046, pad=0.04, label='Probability')
plt.tight_layout()
plt.show()



# Q4.3 â€” Expected days to reach H starting from L (analytic + Monte Carlo)
import pandas as pd
import numpy as np

# --- Load & preprocess (same pipeline as Q4.1 / Q4.2) ---
df = pd.read_csv('/kaggle/input/orange-2/train.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df = df.infer_objects(copy=False)
df[df.select_dtypes(include='number').columns] = (
    df.select_dtypes(include='number').interpolate(method='linear')
)

# --- Filter GOOGL and compute thresholds & states ---
googl = df[df['Company'] == 'GOOGL'].set_index('Date').sort_index()
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

# --- Build transition counts and probabilities (rows = current state) ---
transitions = pd.DataFrame({'from': googl['State'][:-1].values, 'to': googl['State'][1:].values})
counts = pd.crosstab(transitions['from'], transitions['to']).reindex(index=['L','M','H'], columns=['L','M','H'], fill_value=0)
P = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)  # DataFrame

print("Transition Probability Matrix (P):")
print(P.round(4))
print()

# --- Analytic expected hitting time: solve for E[L], E[M] with E[H]=0 ---
# Order of states: ['L','M','H']
non_target = ['L','M']
# Build A = I - P_sub, b = 1
P_sub = P.loc[non_target, non_target].values  # 2x2
A = np.eye(len(non_target)) - P_sub
b = np.ones(len(non_target))

# Solve A x = b for x = expected times from non-target states
try:
    E_non_target = np.linalg.solve(A, b)
except np.linalg.LinAlgError:
    # Singular matrix (rare); fallback to pseudo-inverse
    E_non_target = np.linalg.pinv(A).dot(b)

E = {'L': E_non_target[0], 'M': E_non_target[1], 'H': 0.0}

print("Analytic expected days to reach H (hitting time):")
print(f"  E[L -> H] = {E['L']:.4f} days")
print(f"  E[M -> H] = {E['M']:.4f} days")
print()

# --- Monte Carlo simulation to estimate expected hitting time from L to H ---
import random

def simulate_one(start_state, P_df, target='H', max_steps=10000):
    state = start_state
    steps = 0
    while state != target and steps < max_steps:
        probs = P_df.loc[state].values
        states = P_df.columns.tolist()
        state = random.choices(states, weights=probs, k=1)[0]
        steps += 1
    return steps

# Run many trials
n_trials = 5000
results = []
for _ in range(n_trials):
    t = simulate_one('L', P, target='H')
    results.append(t)

results = np.array(results)
mean_sim = results.mean()
median_sim = np.median(results)
pct90 = np.percentile(results, 90)

print(f"Monte Carlo estimate (n={n_trials}):")
print(f"  Mean steps from L to H = {mean_sim:.4f}")
print(f"  Median = {median_sim:.0f}")
print(f"  90th percentile = {pct90:.0f}")
print()
print("Note: If many trials hit max_steps (indicating possible non-reachability or near-zero prob), increase max_steps or check chain connectivity.")



# Q4.4 â€” Crash Analysis: Add absorbing Crash state and compute expected time to absorption from L
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# --- Load & preprocess (same pipeline) ---
df = pd.read_csv('/kaggle/input/orange-2/train.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df = df.infer_objects(copy=False)
df[df.select_dtypes(include='number').columns] = (
    df.select_dtypes(include='number').interpolate(method='linear')
)

# --- Focus on GOOGL ---
googl = df[df['Company'] == 'GOOGL'].set_index('Date').sort_index()
googl = googl[['Close']].copy()

# --- Compute daily pct change ---
googl['Pct_Change'] = googl['Close'].pct_change()

# --- Crash rule: drop more than 10% compared to previous day (i.e., pct_change <= -0.10) ---
googl['Is_Crash'] = googl['Pct_Change'] <= -0.10

# --- Compute thresholds for L/M/H using the full Close series (excluding NaNs) ---
low_threshold = googl['Close'].quantile(0.33)
high_threshold = googl['Close'].quantile(0.66)

def base_state(price):
    if price <= low_threshold:
        return 'L'
    elif price <= high_threshold:
        return 'M'
    else:
        return 'H'

# --- Assign base L/M/H, then override with C if crash day ---
googl['State'] = googl['Close'].apply(base_state)
googl.loc[googl['Is_Crash'] == True, 'State'] = 'C'

# --- Show distribution ---
print("State counts (including Crash):")
print(googl['State'].value_counts())
print()

# --- Build transitions from day t -> t+1 ---
states_series = googl['State']
next_states = states_series.shift(-1)
transitions = pd.DataFrame({'from': states_series[:-1].values, 'to': next_states[:-1].values})

# --- Define ordered states including Crash ---
state_order = ['L', 'M', 'H', 'C']

# --- Transition counts (empirical) ---
counts = pd.crosstab(transitions['from'], transitions['to']).reindex(index=state_order, columns=state_order, fill_value=0)
print("Transition counts (empirical):")
print(counts)
print()

# --- Convert counts to probabilities (row-normalize) ---
P_empirical = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
print("Empirical transition probability matrix (before enforcing absorbing Crash):")
print(P_empirical.round(4))
print()

# --- Enforce Crash as absorbing: set row for 'C' to [0,0,0,1] regardless of empirical behavior ---
P = P_empirical.copy()
P.loc['C'] = 0.0
P.loc['C', 'C'] = 1.0

# Optionally, ensure rows sum to 1 for non-zero rows (they should already)
# (This also handles rows that were all zeros by leaving them zeros; but that shouldn't occur.)
for r in state_order:
    row_sum = P.loc[r].sum()
    if row_sum == 0 and counts.loc[r].sum() > 0:
        # avoid dividing by zero; fallback to empirical normalization
        P.loc[r] = counts.loc[r] / counts.loc[r].sum()

print("Transition probability matrix (with Crash enforced absorbing):")
print(P.round(4))
print()

# --- Compute expected time to absorption (analytical) ---
# Transient states are L, M, H (non-absorbing). Absorbing state is C.
transient_states = ['L', 'M', 'H']
Q = P.loc[transient_states, transient_states].values  # 3x3 matrix

# Fundamental matrix N = (I - Q)^{-1}
I = np.eye(Q.shape[0])
try:
    N = np.linalg.inv(I - Q)
except np.linalg.LinAlgError:
    N = np.linalg.pinv(I - Q)  # fallback

# Expected time to absorption starting from transient state i: t_i = sum_j N[i,j]
t = N.sum(axis=1)

expected_times = dict(zip(transient_states, t))
print("Analytic expected time to absorption (in days) for transient states (L,M,H):")
for s in transient_states:
    print(f"  E[{s} -> C] = {expected_times[s]:.4f} days")

print()
print(f"âœ… Expected time from L to absorption (C): {expected_times['L']:.4f} days")
print()

# --- Monte Carlo simulation to validate (using enforced P with C absorbing) ---
def simulate_steps_to_absorption(start_state, P_df, target='C', max_steps=10000):
    state = start_state
    steps = 0
    while state != target and steps < max_steps:
        probs = P_df.loc[state].values
        states_list = P_df.columns.tolist()
        state = random.choices(states_list, weights=probs, k=1)[0]
        steps += 1
        # if a row has all zeros (no outgoing), break to prevent infinite loop
        if probs.sum() == 0:
            break
    return steps

# prepare P as DataFrame for simulation
P_df = P.copy()

n_trials = 5000
results = []
for _ in range(n_trials):
    results.append(simulate_steps_to_absorption('L', P_df, target='C', max_steps=10000))

results = np.array(results)
print("Monte Carlo (enforced absorbing) summary (n={}):".format(n_trials))
print(f"  Mean steps from L to C = {results.mean():.4f}")
print(f"  Median = {np.median(results):.0f}")
print(f"  90th percentile = {np.percentile(results,90):.0f}")
print(f"  Fraction of trials that did NOT reach C within max steps = {(results >= 10000).mean():.4f}")



# --- Q5.1: Investor Retention during Stable Macroeconomic Conditions ---

import pandas as pd

# --- Load dataset ---
df = pd.read_csv('/kaggle/input/orange-2/train.csv')

# --- Convert Date to datetime and sort ---
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# --- Interpolate missing numeric values ---
df = df.infer_objects(copy=False)
df.interpolate(method='linear', inplace=True)

# --- Check available macro columns ---
print("Available columns:", df.columns.tolist())

# --- Ensure required macro variables exist ---
if not {'Inflation', 'Interest'}.issubset(df.columns):
    raise ValueError("Dataset must include 'Inflation' and 'Interest' columns for this task.")

# --- Define Stable Macroeconomic Conditions ---
median_inflation = df['Inflation'].median()
median_interest = df['Interest'].median()

stable_mask = (df['Inflation'] < median_inflation) & (df['Interest'] < median_interest)
stable_periods = df[stable_mask]

# --- Compute Average Trading Volume per Company during Stable Conditions ---
avg_volume_stable = stable_periods.groupby('Company')['Volume'].mean().sort_values(ascending=False)
print("\nğŸ“Š Average Trading Volume during Stable Conditions:")
print(avg_volume_stable)

# --- Identify Company with Highest Investor Retention ---
highest_retention_company = avg_volume_stable.idxmax()
print(f"\nğŸ�† Company with Highest Investor Retention during Stable Conditions: {highest_retention_company}")

# --- Visualization ---
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
avg_volume_stable.plot(kind='bar', color=['steelblue', 'orange', 'green'])
plt.title('Average Trading Volume During Stable Macroeconomic Conditions')
plt.xlabel('Company')
plt.ylabel('Average Volume')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()



import yfinance as yf
import pandas as pd

# --- Download 3 companies' stock data ---
companies = ['GOOGL', 'AAPL', 'MSFT']
data = []

for company in companies:
    # auto_adjust=False to keep actual open/close values
    df = yf.download(company, start='2020-01-01', end='2024-12-31', auto_adjust=False)
    
    # Keep only necessary columns and flatten if MultiIndex
    df = df[['Open', 'Close', 'Volume']].copy()
    df.columns = ['Open', 'Close', 'Volume']  # ensure single-level columns
    
    df['Company'] = company
    data.append(df)

# --- Combine all into one DataFrame ---
combined = pd.concat(data).dropna().reset_index()

# --- Ensure columns are numeric ---
combined['Open'] = pd.to_numeric(combined['Open'], errors='coerce')
combined['Close'] = pd.to_numeric(combined['Close'], errors='coerce')
combined['Volume'] = pd.to_numeric(combined['Volume'], errors='coerce')

# --- Compute Value = Volume Ã— (Close - Open) ---
combined['Value'] = (combined['Volume'] * (combined['Close'] - combined['Open'])).astype(float)

# --- Compute 30-day rolling mean and std for each company ---
rolling_stats = []
for company, group in combined.groupby('Company'):
    temp = group.copy()
    temp['Rolling_Mean'] = temp['Value'].rolling(30).mean()
    temp['Rolling_Std'] = temp['Value'].rolling(30).std()
    rolling_stats.append(temp)

rolling_stats = pd.concat(rolling_stats)

# --- Compute average std per company (lower std = more consistent) ---
avg_std = rolling_stats.groupby('Company')['Rolling_Std'].mean().sort_values()

print("\nğŸ“Š Average 30-Day Rolling Std (Consistency Measure):")
print(avg_std)
print(f"\nğŸ�† Most consistent value: {avg_std.idxmin()}")



import yfinance as yf
import pandas as pd

# --- Download data for 3 companies ---
companies = ['GOOGL', 'AAPL', 'MSFT']
data = []

for company in companies:
    df = yf.download(company, start='2020-01-01', end='2024-12-31', auto_adjust=False)
    
    # âœ… Flatten MultiIndex columns if present
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    df = df[['Close']].copy()
    df['Company'] = company
    data.append(df)

# --- Combine all into one DataFrame ---
combined = pd.concat(data).dropna().reset_index()

# --- âœ… Calculate daily returns safely ---
returns_list = []
for company in companies:
    temp = combined[combined['Company'] == company].copy()
    temp['Return'] = temp['Close'].pct_change().squeeze()  # ensure Series
    returns_list.append(temp)

combined = pd.concat(returns_list)

# --- Define Gain/Loss state ---
combined['State'] = combined['Return'].apply(lambda x: 'Gain' if x > 0 else 'Loss')

# --- Function to compute transition matrix ---
def transition_matrix(states):
    transitions = pd.crosstab(
        index=states[:-1], 
        columns=states[1:], 
        normalize='index'
    )
    return transitions.fillna(0)

# --- Compute transition matrix for each company ---
for company in companies:
    sub = combined[combined['Company'] == company]
    matrix = transition_matrix(sub['State'].values)
    print(f"\nğŸ”¹ {company} Return Transition Matrix:")
    print(matrix)
    
    # Most likely transition
    most_likely = matrix.stack().idxmax()
    prob = matrix.stack().max()
    print(f"â�¡ï¸� Most likely transition for {company}: {most_likely[0]} â†’ {most_likely[1]} ({prob:.2f} probability)")



 #Your Answerimport numpy as np
import pandas as pd
import yfinance as yf

# --- Download stock data (example: AAPL, GOOGL, MSFT) ---
companies = ['AAPL', 'GOOGL', 'MSFT']
data = []

for company in companies:
    df = yf.download(company, start='2020-01-01', end='2024-12-31', auto_adjust=False)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]  # flatten columns
    df['Company'] = company
    data.append(df)

# Combine all company data
combined = pd.concat(data).reset_index()

# --- Calculate Log Returns ---
combined['Log_Return'] = np.log(combined['Close'] / combined['Close'].shift(1))

# --- Compute 10-day Rolling Volatility (std of log returns) ---
combined['Volatility'] = combined.groupby('Company')['Log_Return'].transform(lambda x: x.rolling(10).std())

# --- Compute Proxy for Retention = Avg Trading Volume (10-day rolling mean) ---
combined['Retention_Proxy'] = combined.groupby('Company')['Volume'].transform(lambda x: x.rolling(10).mean())

# --- Display sample results ---
print(combined[['Date', 'Company', 'Log_Return', 'Volatility', 'Retention_Proxy']].dropna().head(15))



import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm

# --- Download AAPL data ---
df = yf.download('AAPL', start='2020-01-01', end='2024-12-31')[['Close', 'Volume']]
df = df.dropna().reset_index()

# --- Compute Log Returns and Volatility (10-day rolling std of log returns) ---
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
df['Volatility'] = df['Log_Return'].rolling(window=10).std()

# --- Create Proxy Macroeconomic Data (for demo) ---
# In real use, you'd merge with actual inflation & interest data
np.random.seed(42)
df['Inflation'] = np.random.uniform(2, 6, len(df))          # 2% - 6%
df['Interest_Rate'] = np.random.uniform(3, 7, len(df))      # 3% - 7%
df['Log_Volume'] = np.log(df['Volume'])

# --- Drop missing volatility values ---
df = df.dropna()

# --- Define features and target ---
X = df[['Inflation', 'Interest_Rate', 'Log_Volume']]
y = df['Volatility']

# --- Add constant term for regression ---
X = sm.add_constant(X)

# --- Fit OLS Regression Model ---
model = sm.OLS(y, X).fit()

# --- Display summary ---
print(model.summary())



import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Get residuals and fitted values
residuals = model.resid
fitted = model.fittedvalues

# --- Histogram of residuals ---
plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=30, kde=True)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# --- Q-Q Plot ---
plt.figure(figsize=(6,4))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.show()

# --- Residuals vs Fitted Plot ---
plt.figure(figsize=(6,4))
sns.scatterplot(x=fitted, y=residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Fitted Values")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.show()



import pandas as pd
import numpy as np

# Create 204 forecasted dates
forecast_dates = pd.date_range(start="2025-10-30", periods=204)

# Example dummy forecast data â€” replace with real model output if available
forecast_open = np.linspace(228.45, 250.00, 204) + np.random.randn(204)
forecast_close = forecast_open + np.random.uniform(-2, 2, 204)

# Create DataFrame
submission = pd.DataFrame({
    'date': forecast_dates,
    'forecasted_open': forecast_open,
    'forecasted_close': forecast_close
})

# Save to CSV
submission.to_csv("submission.csv", index=False)

print("âœ… submission.csv file generated successfully with", len(submission), "rows.")
submission.head(), submission.tail()



submission = pd.DataFrame()
sample_submission = pd.read_csv('/kaggle/input/orange-2/train.csv')


submission['date'] = sample_submission['date']
submission['forecasted_open'] = sample_submission['forecasted_open']
submission['forecasted_close'] = sample_submission['forecasted_close']


submission.head()

#Convert to a csv file and name it submission.csv
#submission.to_csv('submission.csv', index = False)

