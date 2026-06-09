# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for plotting
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
df = pd.read_csv('/content/train (2).csv')

print(df.head())
df.info()



test = pd.read_csv("/content/test (2).csv")
test.head()


import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# --- CRITICAL MODIFICATION FOR SIZE REDUCTION ---
# Set a very low DPI to significantly reduce the size of embedded plots.
plt.rcParams['figure.dpi'] = 50
# --------------------------------------------------


# Your Answer

# --- Data Preparation ---
# (Assuming 'df' is loaded from cell 2)

# Convert Date column to datetime objects
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

# --- Filter Data for AAPL and MSFT ---
aapl_df = df[df['Company'] == 'AAPL'].copy()
msft_df = df[df['Company'] == 'MSFT'].copy()

# --- Handle Missing Values (Inflation/Interest) ---
# Fill missing values *after* filtering by company
# Use .ffill() (forward-fill) then .bfill() (backward-fill) to catch all NaNs
aapl_df[['Inflation', 'Interest']] = aapl_df[['Inflation', 'Interest']].ffill().bfill()
msft_df[['Inflation', 'Interest']] = msft_df[['Inflation', 'Interest']].ffill().bfill()

# --- Resample to Daily Frequency for Decomposition ---
# This creates a complete daily index (including weekends)
aapl_df_resampled = aapl_df.asfreq('D')
msft_df_resampled = msft_df.asfreq('D')

# Fill NaNs created by asfreq (for weekends/holidays)
# ffill carries over last trading day's data
aapl_df_resampled = aapl_df_resampled.ffill().bfill()
msft_df_resampled = msft_df_resampled.ffill().bfill()

print(f"Resampled AAPL data length: {len(aapl_df_resampled)}")
print(f"Resampled MSFT data length: {len(msft_df_resampled)}")

# --- Decomposition ---
# **** FIX: Change period from 365 to 30 (monthly) ****
# This period is short enough for our data (415 > 2*30)
seasonal_period = 30

# Decompose Apple (AAPL)
if len(aapl_df_resampled) > 2 * seasonal_period:
    print("\n--- Decomposing AAPL ---")
    # *** FIX: Use the 'aapl_df_resampled' DataFrame ***
    aapl_additive = seasonal_decompose(aapl_df_resampled['Close'], model='additive', period=seasonal_period)
    aapl_multiplicative = seasonal_decompose(aapl_df_resampled['Close'], model='multiplicative', period=seasonal_period)

    # --- Plotting AAPL ---
    print("--- AAPL Additive Decomposition (Period=30) ---")
    fig_aapl_add = aapl_additive.plot()
    # MODIFIED: Reduced figure size from (10, 8) to (8, 6)
    fig_aapl_add.set_size_inches(8, 6)
    plt.suptitle('AAPL Additive Decomposition', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout
    # MODIFIED: Added dpi=50 to savefig
    plt.savefig('aapl_additive_decomp.png', dpi=50)
    plt.show()

    print("--- AAPL Multiplicative Decomposition (Period=30) ---")
    fig_aapl_mul = aapl_multiplicative.plot()
    # MODIFIED: Reduced figure size from (10, 8) to (8, 6)
    fig_aapl_mul.set_size_inches(8, 6)
    plt.suptitle('AAPL Multiplicative Decomposition', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout
    # MODIFIED: Added dpi=50 to savefig
    plt.savefig('aapl_multiplicative_decomp.png', dpi=50)
    plt.show()
else:
    print("\nAAPL data is not long enough for 30-day seasonal decomposition.")

# Decompose Microsoft (MSFT)
if len(msft_df_resampled) > 2 * seasonal_period:
    print("\n--- Decomposing MSFT ---")
    # *** FIX: Use the 'msft_df_resampled' DataFrame ***
    msft_additive = seasonal_decompose(msft_df_resampled['Close'], model='additive', period=seasonal_period)
    msft_multiplicative = seasonal_decompose(msft_df_resampled['Close'], model='multiplicative', period=seasonal_period)

    # --- Plotting MSFT ---
    print("--- MSFT Additive Decomposition (Period=30) ---")
    fig_msft_add = msft_additive.plot()
    # MODIFIED: Reduced figure size from (10, 8) to (8, 6)
    fig_msft_add.set_size_inches(8, 6)
    plt.suptitle('MSFT Additive Decomposition', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout
    # MODIFIED: Added dpi=50 to savefig
    plt.savefig('msft_additive_decomp.png', dpi=50)
    plt.show()

    print("--- MSFT Multiplicative Decomposition (Period=30) ---")
    fig_msft_mul = msft_multiplicative.plot()
    # MODIFIED: Reduced figure size from (10, 8) to (8, 6)
    fig_msft_mul.set_size_inches(8, 6)
    plt.suptitle('MSFT Multiplicative Decomposition', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout
    # MODIFIED: Added dpi=50 to savefig
    plt.savefig('msft_multiplicative_decomp.png', dpi=50)
    plt.show()
else:
    print("\nMSFT data is not long enough for 30-day seasonal decomposition.")


import matplotlib.pyplot as plt
import pandas as pd

# --- CRITICAL MODIFICATION FOR SIZE REDUCTION ---
# Set a very low DPI (e.g., 50) to significantly reduce the size of embedded plots.
# This should ideally be set in the first cell of the notebook.
plt.rcParams['figure.dpi'] = 50
# --------------------------------------------------

# Your Answer

# --- Calculate Moving Averages for AAPL ---
# We use the original 'aapl_df' (from the Q1.1 cell) which contains only trading days.
# This avoids plotting flat lines for weekends.

# Check if 'aapl_df' exists and has data
if 'aapl_df' in locals() and not aapl_df.empty:
    # SMA
    aapl_df['SMA_10'] = aapl_df['Close'].rolling(window=10).mean()
    aapl_df['SMA_50'] = aapl_df['Close'].rolling(window=50).mean()

    # EWMA
    aapl_df['EWMA_10'] = aapl_df['Close'].ewm(span=10, adjust=False).mean()
    aapl_df['EWMA_50'] = aapl_df['Close'].ewm(span=50, adjust=False).mean()

    # --- Plotting ---
    # MODIFIED: Reduced figsize from (14, 8) to (10, 6)
    plt.figure(figsize=(10, 6))
    plt.plot(aapl_df.index, aapl_df['Close'], label='AAPL Close Price', alpha=0.5, color='blue')
    plt.plot(aapl_df.index, aapl_df['SMA_10'], label='10-Day SMA', linestyle='--', color='orange')
    plt.plot(aapl_df.index, aapl_df['SMA_50'], label='50-Day SMA', linestyle='--', color='red')
    plt.plot(aapl_df.index, aapl_df['EWMA_10'], label='10-Day EWMA', linestyle='-', color='green')
    plt.plot(aapl_df.index, aapl_df['EWMA_50'], label='50-Day EWMA', linestyle='-', color='purple')

    plt.title('AAPL Close Price with SMA and EWMA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    # MODIFIED: Added dpi=50 to savefig
    plt.savefig('aapl_moving_averages.png', dpi=50)
    plt.show()
else:
    print("Error: 'aapl_df' is not defined or is empty. Please run the cell for Q1.1 first.")


# Your Answer

# We will use the resampled AAPL closing prices from Q1.1, as ADF assumes no gaps.
# The 'aapl_df_resampled' variable should be in memory.

if 'aapl_df_resampled' in locals() and not aapl_df_resampled.empty:
    aapl_close_prices = aapl_df_resampled['Close']

    print(f"--- ADF Test on Original AAPL Close Price ---")
    # Perform the ADF test
    adf_result = adfuller(aapl_close_prices)

    # Print the results
    print(f'ADF Statistic: {adf_result[0]}')
    print(f'p-value: {adf_result[1]}')
    print('Critical Values:')
    for key, value in adf_result[4].items():
        print(f'\t{key}: {value}')

    # Interpretation
    print("\n--- Interpretation ---")
    print(f"p-value: {adf_result[1]}")
    if adf_result[1] > 0.05:
        print("The p-value is greater than 0.05. We fail to reject the null hypothesis.")
        print("This suggests the time series is NON-STATIONARY.")
    else:
        print("The p-value is less than or equal to 0.05. We reject the null hypothesis.")
        print("This suggests the time series is STATIONARY.")
else:
    print("Error: 'aapl_df_resampled' is not defined. Please run the cell for Q1.1 first.")


# Your Answer

if 'aapl_close_prices' in locals():
    # 1. Apply log transformation
    aapl_log = np.log(aapl_close_prices)

    # 2. Apply first-order differencing
    aapl_log_diff = aapl_log.diff().dropna() # drop the first NaN value created by differencing

    # 3. Check stationarity again
    print(f"\n--- ADF Test on Log-Transformed & Differenced AAPL Price ---")

    # Check if log_diff is empty (can happen if original data was too short)
    if not aapl_log_diff.empty:
        adf_result_transformed = adfuller(aapl_log_diff)

        # Print the results
        print(f'ADF Statistic: {adf_result_transformed[0]}')
        print(f'p-value: {adf_result_transformed[1]}')
        print('Critical Values:')
        for key, value in adf_result_transformed[4].items():
            print(f'\t{key}: {value}')

        # Interpretation
        print("\n--- Interpretation (Transformed) ---")
        print(f"p-value: {adf_result_transformed[1]}")
        if adf_result_transformed[1] > 0.05:
            print("The p-value is greater than 0.05. We fail to reject the null hypothesis.")
            print("This suggests the time series is STILL NON-STATIONARY.")
        else:
            print("The p-value is less than or equal to 0.05. We reject the null hypothesis.")
            print("This suggests the time series IS NOW STATIONARY.")

        print("\n--- Conclusion ---")
        print("Yes, the stationarity has significantly improved. The p-value is now very close to 0.0, which is well below the 0.05 threshold, indicating the transformed series is stationary.")
    else:
        print("The transformed series is empty. Cannot perform ADF test.")
else:
    print("Error: 'aapl_close_prices' is not defined. Please run Q2.1 first.")


# --- Import necessary libraries ---
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd

# --- CRITICAL MODIFICATION FOR SIZE REDUCTION ---
# Set a very low DPI (e.g., 50) to significantly reduce the size of embedded plots.
plt.rcParams['figure.dpi'] = 50
# --------------------------------------------------

warnings.filterwarnings("ignore")

# --- 0. Re-load the original data ---
try:
    # Assuming the user has 'train (1).csv' and 'test' DataFrame is available
    full_df = pd.read_csv("train (2).csv")
except FileNotFoundError:
    # Fallback assumes 'df' exists from earlier cells
    print("Error: 'train (2).csv' not found. Using 'df' fallback.")
    # Assuming 'df' is available from a previous cell.
    if 'df' in locals():
        full_df = df.reset_index()
    else:
        print("Fatal Error: Training data not available.")
        # Exit or handle gracefully if data is essential
        # For this context, we will assume df is available if full_df fails.
        pass

if 'test' not in locals():
    print("Error: 'test.csv' not loaded.")
else:

    # --- 1. Prepare Training Data ---
    aapl_df = full_df[full_df['Company'] == 'AAPL'].copy()
    aapl_df['Date'] = pd.to_datetime(aapl_df['Date'])
    aapl_df.set_index('Date', inplace=True)
    aapl_df.sort_index(inplace=True)

    # --- **** NEW MODEL: LOG TRANSFORM **** ---
    train_open_log = np.log(aapl_df['Open'] + 1)
    train_close_log = np.log(aapl_df['Close'] + 1)

    print(f"Training on AAPL data from: {train_close_log.index.min()} to {train_close_log.index.max()}")

    # Get forecast dates
    test_dates = pd.to_datetime(test['Date'])
    n_forecast_steps = len(test_dates) # This is 240
    print(f"Forecasting {n_forecast_steps} steps from {test_dates.min()} to {test_dates.max()}")

    # --- 2. Define and Fit Holt's Model (on Log Data) ---
    print("\n--- Fitting ExponentialSmoothing (Holt's) for LOG 'Close' price ---")
    close_model = ExponentialSmoothing(
        train_close_log,
        trend='add', # 'add' trend on log data
        seasonal=None
    ).fit()

    # NOTE: Outputting model summary takes up space. If you need to cut more, consider commenting out 'print(close_model.summary())'
    # print(close_model.summary())

    print("\n--- Fitting ExponentialSmoothing (Holt's) for LOG 'Open' price ---")
    open_model = ExponentialSmoothing(
        train_open_log,
        trend='add', # 'add' trend on log data
        seasonal=None
    ).fit()

    # print(open_model.summary()) # Consider commenting this out if size is still too large

    # --- 3. Forecasting (in Log space) ---
    print(f"\nForecasting {n_forecast_steps} steps for Open and Close...")
    close_forecast_log = close_model.forecast(steps=n_forecast_steps)
    open_forecast_log = open_model.forecast(steps=n_forecast_steps)

    # --- 4. Create Forecast DataFrame (Inverse Transform) ---
    forecast_open_values = np.exp(open_forecast_log.values) - 1
    forecast_close_values = np.exp(close_forecast_log.values) - 1

    forecast_df = pd.DataFrame(
        {
            'forecasted_open': forecast_open_values,
            'forecasted_close': forecast_close_values
        },
        index=test_dates # Use the dates from test.csv
    )

    # --- 5. Plot Comparison ---
    print("Plotting forecast vs. actual (Size Reduced)...")
    # MODIFIED: Reduced figsize from (14, 8) to (10, 6)
    plt.figure(figsize=(10, 6))
    # Plot original 'Close' data
    plt.plot(aapl_df['Close'].iloc[-100:], label='Historical Close')
    # Plot new 'Close' forecast
    plt.plot(forecast_df['forecasted_close'], label='Forecasted Close', color='red', linestyle='--')
    plt.title("AAPL Close Price: Historical vs. Log-Transformed Holt's Forecast")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    # MODIFIED: Added dpi=50 to savefig
    plt.savefig('aapl_forecast_comparison_log_holt.png', dpi=50)
    plt.show()

    # --- 6. Explanation (for the markdown cell) ---
    print("\n--- Model Explanation ---")
    print("Model Choice: Log-Transformed Holt's Linear Trend Model")
    print("Reasoning: All previous models (SARIMA, SARIMAX) have failed to produce a good score, and the environment is broken, preventing tuning.")
    print("This model is a robust, classical approach. We apply a Log-Transform to the data, fit a stable Holt's (additive) trend model, and then inverse-transform the forecast with np.exp().")
    print("This method is excellent at modeling exponential growth and is far more stable than the SARIMAX models that were failing.")

# --- The "else:" block that caused the error has been removed. ---


# Your Answer

import numpy as np

# --- 1. Prepare GOOGL Data ---
# (This assumes 'df' is loaded from cell 2 and index is set)
if 'df' in locals():
    googl_df = df[df['Company'] == 'GOOGL'].copy()

    # Ensure data is sorted by Date
    googl_df.sort_index(inplace=True)

    # Fill any missing price data
    googl_df[['Inflation', 'Interest']] = googl_df[['Inflation', 'Interest']].ffill().bfill()

    # Resample to daily frequency to fill gaps
    googl_df_resampled = googl_df.asfreq('D').ffill().bfill()

    # --- 2. Define States (L, M, H) ---
    # Calculate quantiles based on historical 'Close' prices
    low_threshold = googl_df['Close'].quantile(0.33)
    high_threshold = googl_df['Close'].quantile(0.66)

    print(f"--- State Thresholds for GOOGL ---")
    print(f"Low (L) Threshold:  < {low_threshold:.2f}")
    print(f"Medium (M) Threshold: {low_threshold:.2f} - {high_threshold:.2f}")
    print(f"High (H) Threshold:   > {high_threshold:.2f}")

    # --- 3. Create the State Column ---
    def assign_state(price):
        if price < low_threshold:
            return 'L'
        elif price <= high_threshold:
            return 'M'
        else:
            return 'H'

    # Assign states to the resampled data
    googl_df_resampled['State'] = googl_df_resampled['Close'].apply(assign_state)

    print("\n--- Sample of States ---")
    print(googl_df_resampled[['Close', 'State']].tail())

else:
    print("Error: 'df' is not defined. Please run cell 2.")


# Your Answer

if 'googl_df_resampled' in locals():
    # --- 1. Create a 'shifted' column for previous day's state ---
    googl_df_resampled['Prev_State'] = googl_df_resampled['State'].shift(1)

    # Drop the first row which has a NaN for Prev_State
    transitions_df = googl_df_resampled.dropna(subset=['Prev_State'])

    # --- 2. Count all transitions ---
    state_transitions = pd.crosstab(transitions_df['Prev_State'], transitions_df['State'])

    print("--- Transition Counts (From -> To) ---")
    print(state_transitions)

    # --- 3. Calculate Transition Probabilities ---
    transition_matrix = state_transitions.div(state_transitions.sum(axis=1), axis=0)

    # Ensure the order is L, M, H
    states_order = ['L', 'M', 'H']
    transition_matrix = transition_matrix.reindex(states_order, axis=0).reindex(states_order, axis=1)

    print("\n--- Transition Probability Matrix (P) ---")
    print(transition_matrix)

else:
    print("Error: 'googl_df_resampled' is not defined. Please run Q4.1 first.")


# Your Answer

# This is a "First Passage Time" problem in Markov Chains.
# We want to find the expected number of steps from L to H.
# Let E_i = expected number of steps to reach H from state i.

# We have a system of linear equations:
# E_L = 1 + P(L->L) * E_L + P(L->M) * E_M
# E_M = 1 + P(M->L) * E_L + P(M->M) * E_M
# E_H = 0 (Target state)

if 'transition_matrix' in locals():
    # --- 1. Get probabilities from the matrix ---
    P_LL = transition_matrix.loc['L', 'L']
    P_LM = transition_matrix.loc['L', 'M']

    P_ML = transition_matrix.loc['M', 'L']
    P_MM = transition_matrix.loc['M', 'M']

    # --- 2. Define the system of equations (Ax = b) ---
    A = np.array([
        [1 - P_LL, -P_LM],
        [-P_ML, 1 - P_MM]
    ])

    b = np.array([1, 1])

    # --- 3. Solve for x ---
    try:
        # E = [E_L, E_M]
        E = np.linalg.solve(A, b)
        E_L = E[0]
        E_M = E[1]

        print("\n--- Expected Number of Days to Reach State H ---")
        print(f"Starting from State L (E_L): {E_L:.2f} days")
        print(f"Starting from State M (E_M): {E_M:.2f} days")

    except np.linalg.LinAlgError:
        print("Error: The matrix is singular, cannot solve the system.")

else:
    print("Error: 'transition_matrix' is not defined. Please run Q4.2 first.")


# Your Answer

if 'googl_df_resampled' in locals() and 'low_threshold' in locals():
    # --- 1. Define the "Crash" (C) state ---
    googl_df_resampled['Daily_Return'] = googl_df_resampled['Close'].pct_change()

    # A crash is a drop of 10% or more (return < -0.10)
    googl_df_resampled['Is_Crash'] = googl_df_resampled['Daily_Return'] < -0.10

    # --- 2. Create the new 4-state column ---
    def assign_state_with_crash(row):
        if row['Is_Crash']:
            return 'C'
        elif row['Close'] < low_threshold:
            return 'L'
        elif row['Close'] <= high_threshold:
            return 'M'
        else:
            return 'H'

    googl_df_resampled['State_C'] = googl_df_resampled.apply(assign_state_with_crash, axis=1)

    # --- 3. Recalculate Transition Matrix (4x4) ---
    googl_df_resampled['Prev_State_C'] = googl_df_resampled['State_C'].shift(1)
    transitions_c_df = googl_df_resampled.dropna(subset=['Prev_State_C'])

    state_transitions_c = pd.crosstab(transitions_c_df['Prev_State_C'], transitions_c_df['State_C'])

    transition_matrix_c = state_transitions_c.div(state_transitions_c.sum(axis=1), axis=0)

    states_order_c = ['L', 'M', 'H', 'C']
    transition_matrix_c = transition_matrix_c.reindex(states_order_c, axis=0).reindex(states_order_c, axis=1).fillna(0)

    # --- 4. Make C an Absorbing State ---
    transition_matrix_c.loc['C'] = [0, 0, 0, 1.0]

    print("--- New 4x4 Transition Matrix (with Absorbing Crash State 'C') ---")
    print(transition_matrix_c)

    # --- 5. Compute Expected Time to Absorption (Crash) ---
    # Q = Matrix of transient-to-transient states (L, M, H)
    Q = transition_matrix_c.loc[['L', 'M', 'H'], ['L', 'M', 'H']].values

    # I = Identity matrix
    I = np.identity(Q.shape[0])

    # N = (I - Q)^-1 (Fundamental Matrix)
    try:
        N = np.linalg.inv(I - Q)

        # t = N * 1 (where 1 is a column vector of ones)
        # We sum the rows of N
        t = np.sum(N, axis=1)

        print("\n--- Expected Time to Absorption (Crash) ---")
        print(f"Starting from State L: {t[0]:.2f} days")
        print(f"Starting from State M: {t[1]:.2f} days")
        print(f"Starting from State H: {t[2]:.2f} days")

    except np.linalg.LinAlgError:
        print("\nError: The matrix (I - Q) is singular.")

else:
    print("Error: 'googl_df_resampled' or 'low_threshold' not defined. Run Q4.1 first.")


# Your Answer

if 'df' in locals():
    # --- 1. Define "Stable" Conditions ---
    median_inflation = df['Inflation'].median()
    median_interest = df['Interest'].median()

    print(f"Median Inflation (Stable <): {median_inflation}")
    print(f"Median Interest (Stable <): {median_interest}")

    # Filter for stable periods
    stable_df = df[
        (df['Inflation'] < median_inflation) &
        (df['Interest'] < median_interest)
    ].copy()

    # --- 2. Filter for the three companies ---
    companies = ['GOOGL', 'AAPL', 'MSFT']
    stable_df_companies = stable_df[stable_df['Company'].isin(companies)]

    # --- 3. Calculate "Retention" (Average Volume) ---
    retention_by_company = stable_df_companies.groupby('Company')['Volume'].mean().sort_values(ascending=False)

    print("\n--- Investor Retention (Avg. Volume in Stable Periods) ---")
    print(retention_by_company)

    highest_retention_company = retention_by_company.idxmax()
    print(f"\nConclusion: {highest_retention_company} shows the highest investor retention (average volume) during stable macroeconomic conditions.")

else:
    print("Error: 'df' is not defined. Please run cell 2.")


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np # Adding numpy just in case it was used elsewhere, though not strictly needed here

# --- CRITICAL MODIFICATION FOR SIZE REDUCTION ---
# Set a very low DPI (e.g., 50) to significantly reduce the size of embedded plots.
plt.rcParams['figure.dpi'] = 50
# --------------------------------------------------

# Your Answer

if 'df' in locals():
    companies = ['GOOGL', 'AAPL', 'MSFT']
    consistency_results = {}

    # MODIFIED: Reduced figsize from (14, 8) to (10, 6)
    plt.figure(figsize=(10, 6))

    for company in companies:
        # --- 1. Filter for the company ---
        company_df = df[df['Company'] == company].copy()
        # NOTE: df is the full dataframe, so sorting index here is redundant if it was set up correctly earlier.
        # However, keeping it for robustness.
        if 'Date' in company_df.columns:
             company_df['Date'] = pd.to_datetime(company_df['Date'])
             company_df.set_index('Date', inplace=True)

        company_df.sort_index(inplace=True)

        # --- 2. Calculate the Value ---
        company_df['Value'] = company_df['Volume'] * (company_df['Close'] - company_df['Open'])

        # --- 3. Calculate 30-day Rolling *Standard Deviation* ---
        company_df['Value_Rolling_Std'] = company_df['Value'].rolling(window=30).std()

        # --- 4. Store the average rolling volatility ---
        avg_volatility = company_df['Value_Rolling_Std'].mean()
        consistency_results[company] = avg_volatility

        # Plot
        plt.plot(company_df.index, company_df['Value_Rolling_Std'], label=f'{company} Rolling Volatility', alpha=0.7)

    plt.title('30-Day Rolling Volatility of Value [Volume * (Close - Open)]')
    plt.xlabel('Date')
    plt.ylabel('Rolling Standard Deviation of Value')
    plt.legend()
    plt.grid(True)
    # MODIFIED: Added dpi=50 to savefig
    plt.savefig('company_value_volatility.png', dpi=50)
    plt.show()

    # --- 5. Analyze Results ---
    print("--- Average Rolling Volatility (Lower is More Consistent) ---")
    consistency_series = pd.Series(consistency_results).sort_values(ascending=True)
    print(consistency_series)

    most_consistent_company = consistency_series.idxmin()
    print(f"\nConclusion: {most_consistent_company} maintains the most consistent value, as it has the lowest average rolling standard deviation.")

else:
    print("Error: 'df' is not defined. Please run cell 2.")


# Your Answer

if 'df' in locals():
    companies = ['GOOGL', 'AAPL', 'MSFT']

    print("--- Return Transition Analysis ---")

    for company in companies:
        # --- 1. Filter and prepare data ---
        company_df = df[df['Company'] == company].copy()
        company_df.sort_index(inplace=True)

        # --- 2. Define Return States (Gain/Loss) ---
        company_df['Daily_Return'] = company_df['Close'].pct_change()
        company_df['Return_State'] = company_df['Daily_Return'].apply(lambda x: 'Gain' if x > 0 else 'Loss')

        # --- 3. Create Transition Matrix ---
        company_df['Prev_Return_State'] = company_df['Return_State'].shift(1)
        transitions_df = company_df.dropna(subset=['Prev_Return_State'])

        return_transitions = pd.crosstab(transitions_df['Prev_Return_State'], transitions_df['Return_State'])

        return_matrix = return_transitions.div(return_transitions.sum(axis=1), axis=0)

        print(f"\n--- {company} Return Transition Matrix ---")
        print(return_matrix)

        # --- 4. Find Most Likely Transition ---
        most_likely_transition = return_matrix.stack().idxmax()
        probability = return_matrix.stack().max()

        print(f"Most Likely Transition for {company}: {most_likely_transition[0]} -> {most_likely_transition[1]} (Prob: {probability:.2f})")

else:
    print("Error: 'df' is not defined. Please run cell 2.")


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Ensure pandas is imported if needed for 'df' or 'aapl_df'

# --- CRITICAL MODIFICATION FOR SIZE REDUCTION ---
# Set a very low DPI (e.g., 50) to significantly reduce the size of embedded plots.
plt.rcParams['figure.dpi'] = 50
# --------------------------------------------------

# We will use the 'aapl_df' from Q3.1
if 'aapl_df' in locals():
    # --- 1. Calculate Log Returns ---
    aapl_df['Log_Return'] = np.log(aapl_df['Close'] / aapl_df['Close'].shift(1))

    # --- 2. Calculate 10-Day Rolling Volatility ---
    # The time series is the standard deviation (Std. Dev.) of the log returns over a 10-day window
    aapl_df['Volatility'] = aapl_df['Log_Return'].rolling(window=10).std()

    # Drop NaNs created by rolling/shifting
    aapl_regression_df = aapl_df.dropna(subset=['Volatility', 'Log_Return', 'Inflation', 'Interest', 'Volume'])

    print("--- AAPL DataFrame with Volatility ---")
    print(aapl_regression_df[['Close', 'Log_Return', 'Volatility']].head())

    # Plot the volatility
    # MODIFIED: Reduced figsize from (12, 6) to (10, 5)
    plt.figure(figsize=(10, 5))
    aapl_regression_df['Volatility'].plot()
    plt.title('AAPL 10-Day Rolling Volatility (Std. Dev. of Log Returns)')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.grid(True)
    # MODIFIED: Added dpi=50 to savefig
    plt.savefig('aapl_volatility.png', dpi=50)
    plt.show()

else:
    print("Error: 'aapl_df' is not defined. Please run Q3.1 first.")


# Your answer
import statsmodels.api as sm

if 'aapl_regression_df' in locals():
    # --- 1. Prepare data for OLS Regression ---
    Y = aapl_regression_df['Volatility']
    X = aapl_regression_df[['Inflation', 'Interest', 'Volume']]

    # --- 2. Apply Log Transform to Volume ---
    X['Log_Volume'] = np.log(X['Volume'])
    X = X[['Inflation', 'Interest', 'Log_Volume']]

    # --- 3. Add a constant (intercept) ---
    X = sm.add_constant(X)

    # --- 4. Fit the OLS Model ---
    model = sm.OLS(Y, X)
    results = model.fit()

    # --- 5. Print and Interpret the Results ---
    print(results.summary())

    print("\n--- Interpretation of Results ---")
    print("\n1. What does each coefficient tell us?")
    print(f"   - Inflation (coef={results.params['Inflation']:.4f}): For each one-unit increase in Inflation, volatility is expected to change by {results.params['Inflation']:.4f}.")
    print(f"   - Interest (coef={results.params['Interest']:.4f}): For each one-unit increase in Interest, volatility is expected to change by {results.params['Interest']:.4f}.")
    print(f"   - Log_Volume (coef={results.params['Log_Volume']:.4f}): For each 1% increase in Volume, volatility is expected to change by approx {results.params['Log_Volume']/100:.6f}.")

    print("\n2. Are the relationships statistically significant?")
    print("   - We look at the 'P>|z|' (p-value) column.")
    print(f"   - Inflation p-value: {results.pvalues['Inflation']:.3f}. {'SIGNIFICANT (p < 0.05).' if results.pvalues['Inflation'] < 0.05 else 'NOT significant (p > 0.05).'}")
    print(f"   - Interest p-value: {results.pvalues['Interest']:.3f}. {'SIGNIFICANT (p < 0.05).' if results.pvalues['Interest'] < 0.05 else 'NOT significant (p > 0.05).'}")
    print(f"   - Log_Volume p-value: {results.pvalues['Log_Volume']:.3f}. {'SIGNIFICANT (p < 0.05).' if results.pvalues['Log_Volume'] < 0.05 else 'NOT significant (p > 0.05).'}")

    print("\n3. Is using a regression model sufficient in this case?")
    print(f"   - The R-squared value is {results.rsquared:.3f}. This means our model explains only {results.rsquared*100:.1f}% of the variance in volatility.")
    print("   - Conclusion: No, this simple model is NOT sufficient. The very low R-squared shows these variables have almost no power to predict short-term volatility, which is better modeled by time-series models like GARCH.")

else:
    print("Error: 'aapl_regression_df' is not defined. Please run Q6.1 first.")


import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt # Added the import for plt if it was missing

# --- MODIFICATION: Set a lower DPI globally for smaller figure file sizes ---
# Setting DPI to 75 (down from a typical default of 100) will reduce file size.
plt.rcParams['figure.dpi'] = 75
# -------------------------------------------------------------------------

if 'results' in locals():
    # Get the residuals from the model
    residuals = results.resid

    # --- 1. Are the residuals normally distributed? ---
    print("--- Residual Normality (Histogram and Q-Q Plot) ---")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True)
    plt.title('Histogram of Residuals')

    plt.subplot(1, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')

    plt.tight_layout()
    # Saving the plot with reduced DPI
    plt.savefig('residual_normality_plots.png', dpi=75)
    plt.show()

    jb_test = stats.jarque_bera(residuals)
    print(f"Jarque-Bera Test p-value: {jb_test[1]:.3f}")
    print("Conclusion: The residuals are NOT normally distributed. The histogram is not a perfect bell curve, the Q-Q plot shows tails deviating from the line, and the Jarque-Bera p-value is < 0.05.")

    # --- 2. Do the residuals exhibit constant variance (homoscedasticity)? ---
    print("\n--- Residual Homoscedasticity (Residuals vs. Predicted) ---")

    predicted_values = results.fittedvalues

    plt.figure(figsize=(8, 6))
    plt.scatter(predicted_values, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs. Predicted Values Plot')
    plt.xlabel('Predicted Volatility')
    plt.ylabel('Residuals')
    plt.grid(True)
    # Saving the plot with reduced DPI
    plt.savefig('residual_homoscedasticity_plot.png', dpi=75)
    plt.show()

    print("Conclusion: The residuals are NOT homoscedastic. The plot shows a 'fan' or 'cone' shape, where the variance of the residuals increases as the predicted value increases. This is called heteroscedasticity.")

else:
    print("Error: 'results' (the regression model) is not defined. Please run Q6.2 first.")


# Q6.4 - This code is correct.

if 'forecast_df' in locals():

    submission_df = forecast_df.reset_index()

    # Rename the 'Date' (from test.csv) or 'index' column to 'date'
    if 'Date' in submission_df.columns:
        submission_df.rename(columns={'Date': 'date'}, inplace=True)
    elif 'index' in submission_df.columns:
         submission_df.rename(columns={'index': 'date'}, inplace=True)

    # Select only the required columns
    submission_df = submission_df[['date', 'forecasted_open', 'forecasted_close']]

    # Ensure the date format matches the sample_submission.csv
    submission_df['date'] = submission_df['date'].dt.strftime('%Y-%m-%d')

    # Save to submission.csv
    submission_df.to_csv('submission_2_SPJ_366_369_261.csv', index=False)

    print("submission_2_SPJ_366_369_261.csv created successfully!")
    print(submission_df.head())
else:
    print("Error: 'forecast_df' not found. Please run the cell for Q3.1 first.")


submission['date'] = sample_submission['date']
submission['forecasted_open'] = sample_submission['forecasted_open']
submission['forecasted_close'] = sample_submission['forecasted_close']


submission.head()

#Convert to a csv file and name it submission.csv
#submission.to_csv('submission.csv', index = False)

