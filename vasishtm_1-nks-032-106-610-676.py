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


# --- PREPROCESSING CELL (Version 2 - Run this ONCE) ---
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

print(f"Original df columns: {df.columns.to_list()}")

df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date').sort_index()

df[['Inflation', 'Interest']] = df[['Inflation', 'Interest']].interpolate(method='linear')
df[['Inflation', 'Interest']] = df[['Inflation', 'Interest']].bfill().ffill()
print("Cleaned 'Inflation' and 'Interest' columns in main 'df'.")

print("Creating clean time series variables...")
aapl_close_clean = df[df['Company'] == 'AAPL']['Close'].asfreq('D').interpolate(method='time').dropna()
msft_close_clean = df[df['Company'] == 'MSFT']['Close'].asfreq('D').interpolate(method='time').dropna()

aapl_stationary = np.log(aapl_close_clean).diff().dropna()

googl_raw_close = df[df['Company'] == 'GOOGL']['Close'].sort_index()
googl_ffill_close = googl_raw_close.asfreq('D').ffill().dropna()
print("Clean variables created: 'aapl_close_clean', 'msft_close_clean', 'googl_ffill_close', 'aapl_stationary'")
print("Preprocessing complete. 'df' is now fully clean.")


print("--- Decomposing AAPL ---")
aapl_mult = seasonal_decompose(aapl_close_clean, model='multiplicative', period=30)
aapl_mult.plot().suptitle('AAPL Multiplicative Decomposition (30-day)', y=1.02)
plt.tight_layout()
plt.show()

print("--- Decomposing MSFT ---")
msft_mult = seasonal_decompose(msft_close_clean, model='multiplicative', period=30)
msft_mult.plot().suptitle('MSFT Multiplicative Decomposition (30-day)', y=1.02)
plt.tight_layout()
plt.show()


aapl_sma_10 = aapl_close_clean.rolling(window=10).mean()
aapl_ewma_10 = aapl_close_clean.ewm(span=10, adjust=False).mean()
aapl_sma_50 = aapl_close_clean.rolling(window=50).mean()
aapl_ewma_50 = aapl_close_clean.ewm(span=50, adjust=False).mean()
print("AAPL Moving Averages Calculated.")

msft_sma_10 = msft_close_clean.rolling(window=10).mean()
msft_ewma_10 = msft_close_clean.ewm(span=10, adjust=False).mean()
msft_sma_50 = msft_close_clean.rolling(window=50).mean()
msft_ewma_50 = msft_close_clean.ewm(span=50, adjust=False).mean()
print("MSFT Moving Averages Calculated.")

plt.figure(figsize=(14, 7))
plt.plot(aapl_close_clean, label='AAPL Close Price', color='black', alpha=0.3)
plt.plot(aapl_sma_10, label='10-day SMA', color='blue', linestyle='--')
plt.plot(aapl_ewma_10, label='10-day EWMA', color='blue', linestyle='-')
plt.plot(aapl_sma_50, label='50-day SMA', color='red', linestyle='--')
plt.plot(aapl_ewma_50, label='50-day EWMA', color='red', linestyle='-')
plt.title('AAPL Close Price: SMA vs. EWMA')
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(msft_close_clean, label='MSFT Close Price', color='black', alpha=0.3)
plt.plot(msft_sma_10, label='10-day SMA', color='blue', linestyle='--')
plt.plot(msft_ewma_10, label='10-day EWMA', color='blue', linestyle='-')
plt.plot(msft_sma_50, label='50-day SMA', color='red', linestyle='--')
plt.plot(msft_ewma_50, label='50-day EWMA', color='red', linestyle='-')
plt.title('MSFT Close Price: SMA vs. EWMA')
plt.legend()
plt.show()


print("Running ADF Test on the clean 'Close' price series (aapl_close_clean)...")
adf_result = adfuller(aapl_close_clean, autolag='AIC')

print(f'\n--- Results for AAPL Raw Close Price ---')
print(f'ADF Test Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')
print('Critical Values:')
for key, value in adf_result[4].items():
    print(f'\t{key}: {value}')

print("\n--- Interpretation ---")
print("Null Hypothesis (H0): The series is NON-STATIONARY.")
if adf_result[1] > 0.05:
    print(f"Result: The p-value ({adf_result[1]:.4f}) is > 0.05. We FAIL to reject the H0.")
    print("Conclusion: The raw AAPL 'Close' price series is NON-STATIONARY.")
else:
    print(f"Result: The p-value ({adf_result[1]:.4f}) is <= 0.05. We REJECT the H0.")
    print("Conclusion: The raw AAPL 'Close' price series is STATIONARY.")


print("Running ADF Test on the *transformed* (log-differenced) series (aapl_stationary)...")
adf_result_transformed = adfuller(aapl_stationary, autolag='AIC')

print(f'\n--- Results for Transformed (Log-Differenced) AAPL ---')
print(f'Transformed ADF Test Statistic: {adf_result_transformed[0]}')
print(f'Transformed p-value: {adf_result_transformed[1]}')
print('Critical Values:')
for key, value in adf_result_transformed[4].items():
    print(f'\t{key}: {value}')

print("\n--- Interpretation ---")
print("Null Hypothesis (H0): The series is NON-STATIONARY.")
if adf_result_transformed[1] > 0.05:
    print(f"Result: The p-value ({adf_result_transformed[1]}) is > 0.05. We FAIL to reject the H0.")
    print("Conclusion: The transformed series is NON-STATIONARY.")
else:
    print(f"Result: The p-value ({adf_result_transformed[1]}) is <= 0.05. We REJECT the H0.")
    print("Conclusion: The transformed series is STATIONARY.")

print("\n--- Has it improved? ---")
print("Yes, dramatically. The original series was non-stationary (p-value ~0.98),")
print("but the log-differenced series is strongly stationary (p-value is near 0).")


!pip install pmdarima


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

try:
    import pmdarima as pm
    print(f"Successfully imported pmdarima version: {pm.__version__}")
except ImportError:
    print("--- ERROR ---: pmdarima not found. Ensure '!pip install pmdarima' ran successfully.")
    raise
except ValueError as e:
     print(f"--- FAILED TO IMPORT PMDARIMA (ValueError) ---: {e}")
     print("This indicates a numpy/environment incompatibility. Restart runtime after install.")
     raise

from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

try:
    train_df = pd.read_csv("/kaggle/input/orange-2/train.csv")
    test_df = pd.read_csv("/kaggle/input/orange-2/test.csv")
    
    aapl_train_df = train_df[train_df['Company'] == 'AAPL'].sort_values('Date').copy()
    aapl_train_df['Date'] = pd.to_datetime(aapl_train_df['Date'])
    aapl_train_df = aapl_train_df.set_index('Date')

    log_aapl_open = np.log(aapl_train_df['Open']).replace([np.inf, -np.inf], 0).fillna(0)
    log_aapl_close = np.log(aapl_train_df['Close']).replace([np.inf, -np.inf], 0).fillna(0)

    print("Log-transformed 'Open' and 'Close' series prepared (using raw trading days).")

except NameError:
    print("--- FATAL ERROR ---: 'df' not found. Please re-run Preprocessing Cell (or Cell 2).")
    raise
except Exception as e:
    print(f"Error preparing data: {e}")
    raise

print(f"\n--- Finding Best SARIMA Model for Log-Prices (m=5) ---")
print("Running auto_arima for log('Close') price (This may take a minute)...")
model_log_close = pm.auto_arima(log_aapl_close, 
                                 start_p=1, start_q=1,
                                 test='adf', max_p=3, max_q=3,
                                 m=5, seasonal=True, 
                                 start_P=0, D=None, 
                                 d=None, 
                                 stepwise=True, 
                                 suppress_warnings=True,
                                 with_intercept=True, 
                                 error_action='ignore',
                                 trend=None,
                                 information_criterion='aic')

print("\nRunning auto_arima for log('Open') price (This may take a minute)...")
model_log_open = pm.auto_arima(log_aapl_open, 
                                start_p=1, start_q=1,
                                test='adf', max_p=3, max_q=3,
                                m=5, seasonal=True, 
                                start_P=0, D=None, 
                                d=None, 
                                stepwise=True, 
                                suppress_warnings=True,
                                with_intercept=True, 
                                error_action='ignore',
                                trend=None,
                                information_criterion='aic')

print("\n--- Best log('Close') SARIMA Model Found: ---")
print(model_log_close.summary())
print("\n--- Best log('Open') SARIMA Model Found: ---")
print(model_log_open.summary())

try:
    n_periods_to_forecast = len(test_df)
    forecast_index = pd.to_datetime(test_df['Date'])
    print(f"\nLoaded 'test.csv'. Need to forecast {n_periods_to_forecast} days.")
except Exception as e:
    n_periods_to_forecast = 240
    last_date_log = log_aapl_close.index[-1]
    forecast_index = pd.date_range(start=last_date_log + pd.Timedelta(days=1), periods=n_periods_to_forecast, freq='D')
    print(f"Error loading test.csv: {e}. Forecasting 240 periods.")

print("\nGenerating forecasts on log scale...")
log_close_forecast = model_log_close.predict(n_periods=n_periods_to_forecast)
log_open_forecast = model_log_open.predict(n_periods=n_periods_to_forecast)

print("Converting forecasts back to original price scale using np.exp()...")
final_close_forecast_values = np.exp(log_close_forecast)
final_open_forecast_values = np.exp(log_open_forecast)

forecast_df = pd.DataFrame({
    'date': forecast_index, 
    'forecasted_open': final_open_forecast_values.values, 
    'forecasted_close': final_close_forecast_values.values 
})
forecast_df = forecast_df.set_index('date')

print("\n--- Final Forecast DataFrame Head (for submission.csv) ---")
print(forecast_df.head())

print("\n--- Plotting Final Forecast ---")
plt.figure(figsize=(15, 8))
plt.plot(aapl_train_df['Close'].iloc[-500:], label='Historical Close (last 500d)', color='blue')
plt.plot(aapl_train_df['Open'].iloc[-500:], label='Historical Open (last 500d)', color='green', alpha=0.7)
plt.plot(forecast_df['forecasted_close'], label='Final Forecasted Close', linestyle='--', color='blue')
plt.plot(forecast_df['forecasted_open'], label='Final Forecasted Open', linestyle='--', color='green')
plt.title(f'AAPL: FINAL Log-Transformed auto_arima SARIMA (m=5) Forecast')
plt.legend()
plt.show()


googl_raw_close = df[df['Company'] == 'GOOGL']['Close'].sort_index()
googl_ffill_close = googl_raw_close.asfreq('D').ffill().dropna()
print("Created daily, forward-filled series for GOOGL.")
low_threshold = googl_ffill_close.quantile(0.33)
high_threshold = googl_ffill_close.quantile(0.66)

print(f"\n--- Google (GOOGL) Price State Thresholds ---")
print(f"Low (L) cutoff: Prices <= {low_threshold:.2f}")
print(f"Medium (M) cutoff: Prices > {low_threshold:.2f} and <= {high_threshold:.2f}")
print(f"High (H) cutoff: Prices > {high_threshold:.2f}")

def classify_state(price):
    if price <= low_threshold:
        return 'L'
    elif price <= high_threshold:
        return 'M'
    else:
        return 'H'

googl_states = googl_ffill_close.apply(classify_state)

print("\n--- Resulting State Series (Head) ---")
print(googl_states.head())
print("\nState definitions and 'googl_states' series created successfully.")


import pandas as pd

try:
    states_df = pd.DataFrame({
        'from_state': googl_states.shift(1),
        'to_state': googl_states
    }).dropna()
    
    transition_counts = pd.crosstab(states_df['from_state'], states_df['to_state'])
    
    transition_matrix = transition_counts.apply(lambda r: r / r.sum(), axis=1)
    
    print("--- Google (GOOGL) State Transition Matrix (Probabilities) ---")
    
    state_order = ['L', 'M', 'H']
    transition_matrix = transition_matrix.reindex(index=state_order, columns=state_order, fill_value=0)
    
    print(transition_matrix)

except NameError:
    print("--- ERROR ---: 'googl_states' not found. Run new Q4.1 first.")


import numpy as np
import pandas as pd

try:
    transient_states = ['L', 'M']
    Q = transition_matrix.loc[transient_states, transient_states]
    
    I = np.eye(len(transient_states))
    I_minus_Q = I - Q
    b = np.ones(len(transient_states))
    E = np.linalg.solve(I_minus_Q, b)
    
    expected_days = pd.Series(E, index=transient_states)
    
    print("--- Expected Days to Reach State H ---")
    print(expected_days)
    
    print(f"\nResult: The expected number of days to reach state H starting from state L is: {expected_days['L']:.2f} days.")

except NameError:
    print("--- ERROR ---: 'transition_matrix' not found. Run new Q4.2 first.")


import pandas as pd
import numpy as np

try:
    googl_pct_change = googl_ffill_close.pct_change()
    
    is_crash = (googl_pct_change < -0.10)
    
    googl_states_crash = googl_states.copy()
    
    googl_states_crash[is_crash] = 'C'
    
    print(f"--- Crash Analysis ---")
    print(f"Total crash days found: {is_crash.sum()}") 

    states_df_crash = pd.DataFrame({
        'from_state': googl_states_crash.shift(1),
        'to_state': googl_states_crash
    }).dropna()
    
    transition_counts_crash = pd.crosstab(states_df_crash['from_state'], states_df_crash['to_state'])
    
    all_states = ['L', 'M', 'H', 'C']
    transition_counts_crash = transition_counts_crash.reindex(index=all_states, columns=all_states, fill_value=0)

    transition_counts_crash.loc['C'] = 0  
    transition_counts_crash.loc['C', 'C'] = 1  
    
    transition_matrix_crash = transition_counts_crash.apply(lambda r: r / r.sum(), axis=1)
    
    transition_matrix_crash = transition_matrix_crash.fillna(0)
    transition_matrix_crash.loc['C', 'C'] = 1.0

    print("\n--- New 4x4 Transition Matrix (with Absorbing Crash State) ---")
    print(transition_matrix_crash)

    transient_rows = ['L', 'M', 'H']
    Q = transition_matrix_crash.loc[transient_rows, transient_rows]
    
    I = np.eye(len(transient_rows))
    I_minus_Q = I - Q
    b = np.ones(len(transient_rows))
    
    E_absorption = np.linalg.solve(I_minus_Q, b)
    
    expected_days_to_crash = pd.Series(E_absorption, index=transient_rows)
    
    print("\n--- Expected Days to Reach a 'Crash' (Absorption) ---")
    print(expected_days_to_crash)
    
    print(f"\nResult: The expected time to absorption (Crash) starting from state L is: {expected_days_to_crash['L']:.2f} days.")

except NameError:
    print("--- ERROR ---: Required variables not found. Run new Q4.1 first.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


median_inflation = df['Inflation'].median()
median_interest = df['Interest'].median()

print(f"--- Stable Conditions Defined ---")
print(f"Median Inflation: {median_inflation:.2f}")
print(f"Median Interest: {median_interest:.2f}")

stable_df = df[
    (df['Inflation'] < median_inflation) &
    (df['Interest'] < median_interest)
].copy()

print(f"Found {len(stable_df)} rows representing stable periods.")

companies = ['GOOGL', 'AAPL', 'MSFT']
stable_retention = stable_df[stable_df['Company'].isin(companies)]

retention_by_company = stable_retention.groupby('Company')['Volume'].mean().sort_values(ascending=False)

print("\n--- Investor Retention (Avg. Volume in Stable Periods) ---")
print(retention_by_company)

winner = retention_by_company.idxmax()
print(f"\n--- Result ---")
print(f"The company with the highest retention is: {winner}")


import pandas as pd
import numpy as np

try:
    aapl_df = df[df['Company'] == 'AAPL']
    msft_df = df[df['Company'] == 'MSFT']
    googl_df = df[df['Company'] == 'GOOGL']
    
    aapl_day_value = aapl_df['Volume'] * (aapl_df['Close'] - aapl_df['Open'])
    msft_day_value = msft_df['Volume'] * (msft_df['Close'] - msft_df['Open'])
    googl_day_value = googl_df['Volume'] * (googl_df['Close'] - googl_df['Open'])
    
    print("Calculated 'Volume * (Close - Open)' for each company.")
    window = 30
    aapl_volatility = aapl_day_value.rolling(window=window).std()
    msft_volatility = msft_day_value.rolling(window=window).std()
    googl_volatility = googl_day_value.rolling(window=window).std()
    
    print(f"Calculated 30-day rolling standard deviation (volatility) of this value.")

    avg_volatility = pd.Series({
        'AAPL': aapl_volatility.mean(),
        'MSFT': msft_volatility.mean(),
        'GOOGL': googl_volatility.mean()
    })
    
    avg_volatility = avg_volatility.sort_values(ascending=True)
    
    print("\n--- Consistency of 'Volume * (Close - Open)' Value ---")
    print("(Measured by avg. 30-day rolling standard deviation - lower is more consistent)\n")
    print(avg_volatility)
    
    most_consistent = avg_volatility.idxmin()
    lowest_std = avg_volatility.min()
    
    print(f"\n--- Result ---")
    print(f"The company with the MOST CONSISTENT value is: {most_consistent}")
    print(f"(Lowest avg. rolling std. dev: {lowest_std:,.2f})")

except NameError:
    print("--- ERROR ---")
    print("The variable 'df' was not found.")
    print("Please make sure you have run your PREPROCESSING cell first.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


import pandas as pd
import numpy as np

def get_return_transition_matrix(close_series):
    """
    Calculates the Gain/Loss transition matrix for a given stock's close price.
    """
    returns = close_series.pct_change()
    
    states = np.where(returns > 0, 'Gain', 'Loss')
    states_series = pd.Series(states, index=close_series.index)
    
    transitions_df = pd.DataFrame({
        'from_state': states_series.shift(1),
        'to_state': states_series
    })
    
    transitions_df = transitions_df.dropna()
    
    counts = pd.crosstab(transitions_df['from_state'], transitions_df['to_state'])
    
    matrix = counts.apply(lambda r: r / r.sum(), axis=1)
    
    matrix = matrix.reindex(index=['Gain', 'Loss'], columns=['Gain', 'Loss'], fill_value=0)
    
    return matrix

try:
    aapl_close_raw = df[df['Company'] == 'AAPL']['Close']
    msft_close_raw = df[df['Company'] == 'MSFT']['Close']
    googl_close_raw = df[df['Company'] == 'GOOGL']['Close']
    
    aapl_matrix = get_return_transition_matrix(aapl_close_raw)
    msft_matrix = get_return_transition_matrix(msft_close_raw)
    googl_matrix = get_return_transition_matrix(googl_close_raw)
    
    print("\n--- Apple (AAPL) Return Transitions ---")
    print(aapl_matrix)
    print(f"Most likely transition: {aapl_matrix.stack().idxmax()} (Probability: {aapl_matrix.stack().max():.2%})")
    
    print("\n--- Microsoft (MSFT) Return Transitions ---")
    print(msft_matrix)
    print(f"Most likely transition: {msft_matrix.stack().idxmax()} (Probability: {msft_matrix.stack().max():.2%})")
    
    print("\n--- Google (GOOGL) Return Transitions ---")
    print(googl_matrix)
    print(f"Most likely transition: {googl_matrix.stack().idxmax()} (Probability: {googl_matrix.stack().max():.2%})")
    
    print("\n--- Result ---")
    print(f"The most likely transition for AAPL is {' -> '.join(aapl_matrix.stack().idxmax())}.")
    print(f"The most likely transition for MSFT is {' -> '.join(msft_matrix.stack().idxmax())}.")
    print(f"The most likely transition for GOOGL is {' -> '.join(googl_matrix.stack().idxmax())}.")

except NameError:
    print("--- ERROR ---")
    print("The variable 'df' was not found.")
    print("Please make sure you have run your PREPROCESSING cell first.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


aapl_reg_df = df[df['Company'] == 'AAPL'].copy()

aapl_log_returns = np.log(aapl_reg_df['Close']).diff()

window = 10
aapl_reg_df['Volatility'] = aapl_log_returns.rolling(window=window).std()

aapl_reg_df['log_Volume'] = np.log(aapl_reg_df['Volume'])

aapl_reg_df = aapl_reg_df.dropna() 

print(f"--- AAPL Regression DataFrame Created ('aapl_reg_df') ---")
print(aapl_reg_df[['Close', 'Volatility', 'log_Volume', 'Inflation', 'Interest']].head())


import statsmodels.api as sm

try:
    X = aapl_reg_df[['log_Volume', 'Inflation', 'Interest']]
    
    y = aapl_reg_df['Volatility']

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    print("\n--- OLS Regression Results ---")
    print(model.summary())

except NameError:
    print("--- ERROR ---")
    print("The variable 'aapl_reg_df' was not found.")
    print("Please make sure you have run the cell for Q6.1 first.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

try:
    residuals = model.resid
    predicted = model.fittedvalues
    
    print("Extracted residuals and predicted values from the model.")

    print("Plotting diagnostics for Normality...")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=30, edgecolor='k')
    plt.title('Histogram of Residuals')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    plt.show()

    print("Plotting diagnostics for Homoscedasticity...")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(predicted, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs. Predicted Values')
    plt.xlabel('Predicted Volatility')
    plt.ylabel('Residuals')
    plt.show()

except NameError:
    print("--- ERROR ---")
    print("The variable 'model' was not found.")
    print("Please make sure you have run the cell for Q6.2 first.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


import pandas as pd
import numpy as np

try:
    submission_df = forecast_df.copy()
    index_name = submission_df.index.name
    submission_df = submission_df.reset_index()
    date_column_name = index_name if index_name is not None else 'index'
    submission_df.rename(columns={date_column_name: 'date'}, inplace=True)
    submission_df = submission_df[['date', 'forecasted_open', 'forecasted_close']]
    file_name = 'submission.csv'
    submission_df.to_csv(file_name, index=False)
    print(f"--- Successfully created '{file_name}' ---")
    print("This file is now ready for submission.")
    print("\nHere is a preview of the file:")
    print(submission_df.head())

except NameError:
    print("--- ERROR ---")
    print("The variable 'forecast_df' was not found.")
    print("Please go back and re-run the cell for Q3.1, then run this cell again.")
except KeyError as e:
    print(f"--- ERROR ---")
    print(f"A KeyError occurred during column selection: {e}")
    print("This might indicate an issue with the forecast_df structure.")
    print("Please check the output of forecast_df.head() in cell 3.1")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

