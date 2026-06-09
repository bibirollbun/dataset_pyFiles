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








"""
Step 1: Data Loading (Kaggle Version)
NO Google Drive - Use Kaggle data path!
"""

import os
import pandas as pd
import numpy as np

# ===================================
# Kaggle data path (ì��ë�™ ì—°ê²°ë�¨)
# ===================================
data_path = '/kaggle/input/m5-forecasting-accuracy/'

print("=" * 60)
print("Step 1: Data Loading")
print("=" * 60)

# Check available files
print("\nğŸ“� Available files:")
try:
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {file} ({file_size:.2f} MB)")
except Exception as e:
    print(f"â�Œ Error listing files: {e}")
    print("Make sure you joined the M5 Competition!")

# Load sales data (evaluation version)
print("\nğŸ“Š Loading sales data...")
sales = pd.read_csv(data_path + 'sales_train_evaluation.csv')
print(f"âœ… Sales data loaded! Shape: {sales.shape}")
print(f"   Products: {sales.shape[0]:,}")
print(f"   Columns: {sales.shape[1]}")

# Load calendar data
print("\nğŸ“… Loading calendar data...")
calendar = pd.read_csv(data_path + 'calendar.csv')
print(f"âœ… Calendar data loaded! Shape: {calendar.shape}")

# Load prices data
print("\nğŸ’° Loading prices data...")
prices = pd.read_csv(data_path + 'sell_prices.csv')
print(f"âœ… Prices data loaded! Shape: {prices.shape}")

# Extract day columns (d_1, d_2, ..., d_1941)
day_columns = [col for col in sales.columns if col.startswith('d_')]
print(f"\nğŸ“Š Analysis period: {len(day_columns)} days")
print(f"   From: {day_columns[0]} to {day_columns[-1]}")

# Calculate daily total sales (aggregate all products)
print("\nğŸ”„ Aggregating to daily total sales...")
daily_sales = sales[day_columns].sum(axis=0)
daily_sales.index = range(1, len(daily_sales) + 1)

print(f"\nâœ… Daily sales calculated!")
print(f"   - Average daily sales: {daily_sales.mean():,.0f} units")
print(f"   - Max daily sales: {daily_sales.max():,.0f} units")
print(f"   - Min daily sales: {daily_sales.min():,.0f} units")
print(f"   - Total days: {len(daily_sales)}")

# Basic info
print("\nğŸ“‹ Sales DataFrame Info:")
print(sales.info(verbose=False, memory_usage=False))

print("\n" + "=" * 60)
print("âœ… Step 1 Completed!")
print("ğŸ“� Next: Step 2 - Exploratory Data Analysis")
print("=" * 60)


"""
Step 2: Time Series Visualization (EDA)
- Check overall sales trend
- Identify temporal patterns
- Visual inspection of seasonality/trend
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Font settings (English only)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Graph style settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)

print("=" * 60)
print("Step 2: Time Series Visualization (EDA)")
print("=" * 60)

# 1. Calculate total daily sales
print("\n[1] Calculating total daily sales...")

# Select columns starting with 'd_' (sales data)
day_columns = [col for col in sales.columns if col.startswith('d_')]
print(f"Analysis period: {len(day_columns)} days")

# Daily total sales across all products
daily_sales = sales[day_columns].sum(axis=0)
daily_sales.index = range(1, len(daily_sales) + 1)

print(f"Daily average sales: {daily_sales.mean():,.0f} units")
print(f"Daily maximum sales: {daily_sales.max():,.0f} units")
print(f"Daily minimum sales: {daily_sales.min():,.0f} units")

# 2. Visualize overall sales trend
print("\n[2] Visualizing overall sales trend...")

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# 2-1. Overall period sales trend
axes[0].plot(daily_sales.index, daily_sales.values, linewidth=1, alpha=0.7)
axes[0].set_title('Daily Sales Trend - Full Period', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Day')
axes[0].set_ylabel('Total Sales')
axes[0].grid(True, alpha=0.3)

# Display statistics
mean_val = daily_sales.mean()
axes[0].axhline(y=mean_val, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:,.0f}')
axes[0].legend()

# 2-2. Recent 100 days zoom-in
axes[1].plot(daily_sales.index[-100:], daily_sales.values[-100:], linewidth=2, color='orange')
axes[1].set_title('Recent 100 Days Sales (Zoomed)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Day')
axes[1].set_ylabel('Total Sales')
axes[1].grid(True, alpha=0.3)

# 2-3. Moving averages (7-day, 28-day)
ma_7 = daily_sales.rolling(window=7).mean()
ma_28 = daily_sales.rolling(window=28).mean()

axes[2].plot(daily_sales.index, daily_sales.values, linewidth=0.5, alpha=0.3, label='Original')
axes[2].plot(ma_7.index, ma_7.values, linewidth=2, label='7-day MA', color='blue')
axes[2].plot(ma_28.index, ma_28.values, linewidth=2, label='28-day MA', color='red')
axes[2].set_title('Trend Analysis with Moving Averages', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Day')
axes[2].set_ylabel('Total Sales')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Visualization completed!")

# 3. Mapping with date information
print("\n[3] Mapping date information...")

# Extract date information from calendar data
calendar['date'] = pd.to_datetime(calendar['date'])
calendar['year'] = calendar['date'].dt.year
calendar['month'] = calendar['date'].dt.month
calendar['day_of_week'] = calendar['date'].dt.dayofweek
calendar['week_of_year'] = calendar['date'].dt.isocalendar().week

print(f"Analysis period: {calendar['date'].min()} ~ {calendar['date'].max()}")
print(f"Total: {len(calendar)} days")

# 4. Day-of-week sales pattern analysis
print("\n[4] Analyzing sales pattern by day of week...")

# Calculate sales by day of week
weekday_sales = {}
for i, row in calendar.iterrows():
    d_col = row['d']
    if d_col in day_columns:
        weekday = row['day_of_week']
        if weekday not in weekday_sales:
            weekday_sales[weekday] = []
        weekday_sales[weekday].append(daily_sales[i+1])

# Calculate average
weekday_avg = {k: np.mean(v) for k, v in weekday_sales.items()}

# Visualization
fig, ax = plt.subplots(figsize=(12, 6))
weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
values = [weekday_avg[i] for i in range(7)]

bars = ax.bar(weekdays, values, color=sns.color_palette("husl", 7), alpha=0.8)
ax.set_title('Average Sales by Day of Week', fontsize=14, fontweight='bold')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Average Sales')
ax.grid(True, alpha=0.3, axis='y')

# Display values
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:,.0f}',
            ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("Day-of-week analysis completed!")

# 5. Monthly sales trend
print("\n[5] Analyzing monthly sales trend...")

# Calculate monthly sales
monthly_sales = {}
for i, row in calendar.iterrows():
    d_col = row['d']
    if d_col in day_columns:
        year_month = f"{row['year']}-{row['month']:02d}"
        if year_month not in monthly_sales:
            monthly_sales[year_month] = []
        monthly_sales[year_month].append(daily_sales[i+1])

# Calculate average
monthly_avg = {k: np.mean(v) for k, v in sorted(monthly_sales.items())}

# Visualization
fig, ax = plt.subplots(figsize=(15, 6))
months = list(monthly_avg.keys())
values = list(monthly_avg.values())

ax.plot(range(len(months)), values, marker='o', linewidth=2, markersize=6)
ax.set_title('Monthly Average Sales Trend', fontsize=14, fontweight='bold')
ax.set_xlabel('Year-Month')
ax.set_ylabel('Average Daily Sales')
ax.set_xticks(range(0, len(months), 3))
ax.set_xticklabels(months[::3], rotation=45, ha='right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Monthly analysis completed!")

# 6. Basic statistics summary
print("\n" + "=" * 60)
print("Time Series Statistics Summary")
print("=" * 60)

print(f"""
Sales Statistics:
  - Mean: {daily_sales.mean():,.2f}
  - Median: {daily_sales.median():,.2f}
  - Std Dev: {daily_sales.std():,.2f}
  - Min: {daily_sales.min():,.0f}
  - Max: {daily_sales.max():,.0f}

Observations:
  - Overall trend: {'Increasing' if daily_sales.iloc[-100:].mean() > daily_sales.iloc[:100].mean() else 'Decreasing'}
  - Volatility: {'High' if daily_sales.std() / daily_sales.mean() > 0.2 else 'Moderate'}
  - Highest sales day: {weekdays[max(weekday_avg, key=weekday_avg.get)]}
  - Lowest sales day: {weekdays[min(weekday_avg, key=weekday_avg.get)]}
""")

print("=" * 60)
print("Step 2 Completed!")
print("Next Step: Step 3 - Time Series Decomposition")
print("=" * 60)


"""
Step 3: Time Series Decomposition
- Separate Trend component
- Separate Seasonality component
- Extract Residual (stationary component)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

print("=" * 60)
print("Step 3: Time Series Decomposition")
print("=" * 60)

# 1. Prepare time series data
print("\n[1] Preparing time series data...")

# Convert to pandas Series with proper index
ts_data = pd.Series(daily_sales.values, index=pd.date_range(start='2011-01-29', periods=len(daily_sales), freq='D'))

print(f"Time series length: {len(ts_data)} days")
print(f"Start date: {ts_data.index[0]}")
print(f"End date: {ts_data.index[-1]}")

# 2. Perform decomposition (Additive model)
print("\n[2] Performing time series decomposition (Additive model)...")
print("   This may take a moment...")

# Additive decomposition: Y(t) = Trend(t) + Seasonal(t) + Residual(t)
# Period = 7 (weekly seasonality)
decomposition = seasonal_decompose(ts_data, model='additive', period=7, extrapolate_trend='freq')

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

print("Decomposition completed!")

# 3. Visualize decomposition results
print("\n[3] Visualizing decomposition results...")

fig, axes = plt.subplots(4, 1, figsize=(15, 12))

# Original
axes[0].plot(ts_data.index, ts_data.values, linewidth=1, color='black')
axes[0].set_title('Original Time Series', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Sales')
axes[0].grid(True, alpha=0.3)

# Trend
axes[1].plot(trend.index, trend.values, linewidth=2, color='blue')
axes[1].set_title('Trend Component', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Trend')
axes[1].grid(True, alpha=0.3)

# Seasonal
axes[2].plot(seasonal.index, seasonal.values, linewidth=1, color='green')
axes[2].set_title('Seasonal Component (Weekly)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Seasonality')
axes[2].grid(True, alpha=0.3)

# Residual
axes[3].plot(residual.index, residual.values, linewidth=0.5, color='red', alpha=0.7)
axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[3].set_title('Residual Component (Should be Stationary)', fontsize=14, fontweight='bold')
axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Date')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Decomposition visualization completed!")

# 4. Analyze each component
print("\n[4] Analyzing each component...")

print("\n--- Trend Component ---")
print(f"Mean: {trend.mean():,.2f}")
print(f"Std: {trend.std():,.2f}")
print(f"Min: {trend.min():,.2f}")
print(f"Max: {trend.max():,.2f}")
print(f"Trend direction: {'Increasing' if trend.iloc[-100:].mean() > trend.iloc[:100].mean() else 'Decreasing'}")

print("\n--- Seasonal Component ---")
print(f"Mean: {seasonal.mean():,.2f}")
print(f"Std: {seasonal.std():,.2f}")
print(f"Min: {seasonal.min():,.2f}")
print(f"Max: {seasonal.max():,.2f}")
print(f"Range: {seasonal.max() - seasonal.min():,.2f}")

print("\n--- Residual Component ---")
print(f"Mean: {residual.mean():,.2f}")
print(f"Std: {residual.std():,.2f}")
print(f"Min: {residual.min():,.2f}")
print(f"Max: {residual.max():,.2f}")

# 5. Visualize weekly seasonality pattern
print("\n[5] Analyzing weekly seasonality pattern...")

# Extract one week of seasonal pattern
weekly_pattern = seasonal[:7]

fig, ax = plt.subplots(figsize=(10, 6))
weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
ax.plot(weekdays, weekly_pattern.values, marker='o', linewidth=2, markersize=10, color='green')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.set_title('Weekly Seasonal Pattern', fontsize=14, fontweight='bold')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Seasonal Effect')
ax.grid(True, alpha=0.3)

# Add values on points
for i, (day, val) in enumerate(zip(weekdays, weekly_pattern.values)):
    ax.text(i, val, f'{val:.0f}', ha='center', va='bottom' if val > 0 else 'top')

plt.tight_layout()
plt.show()

print("Weekly pattern analysis completed!")

# 6. Check residual distribution
print("\n[6] Checking residual distribution...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram
axes[0].hist(residual.dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0].set_title('Residual Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Residual Value')
axes[0].set_ylabel('Frequency')
axes[0].grid(True, alpha=0.3)

# Box plot
axes[1].boxplot(residual.dropna(), vert=True)
axes[1].set_title('Residual Box Plot', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Residual Value')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("Residual distribution check completed!")

# 7. Summary statistics
print("\n" + "=" * 60)
print("Decomposition Summary")
print("=" * 60)

original_var = ts_data.var()
trend_var = trend.var()
seasonal_var = seasonal.var()
residual_var = residual.var()

total_var = trend_var + seasonal_var + residual_var
trend_pct = (trend_var / total_var) * 100
seasonal_pct = (seasonal_var / total_var) * 100
residual_pct = (residual_var / total_var) * 100

print(f"""
Variance Decomposition:
  - Original variance: {original_var:,.2f}
  - Trend variance: {trend_var:,.2f} ({trend_pct:.1f}%)
  - Seasonal variance: {seasonal_var:,.2f} ({seasonal_pct:.1f}%)
  - Residual variance: {residual_var:,.2f} ({residual_pct:.1f}%)

Observations:
  - Trend explains {trend_pct:.1f}% of total variation
  - Seasonality explains {seasonal_pct:.1f}% of total variation
  - Residual explains {residual_pct:.1f}% of total variation

Key Insights:
  - {'Strong' if trend_pct > 50 else 'Moderate' if trend_pct > 20 else 'Weak'} trend component
  - {'Strong' if seasonal_pct > 20 else 'Moderate' if seasonal_pct > 5 else 'Weak'} seasonal component
  - Residual mean â‰ˆ {residual.mean():.2f} (should be close to 0)
  - Residual std = {residual.std():.2f}
""")

print("=" * 60)
print("Step 3 Completed!")
print("Next Step: Step 4 - Stationarity Test")
print("=" * 60)

# Save components for later use
print("\nSaving components for next steps...")
trend_component = trend
seasonal_component = seasonal
residual_component = residual
print("Components saved as: trend_component, seasonal_component, residual_component")


"""
Step 4: Stationarity Test
- ADF Test (Augmented Dickey-Fuller Test)
- KPSS Test
- Visual inspection
- Confirm if residual is stationary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Step 4: Stationarity Test")
print("=" * 70)

# 1. ADF Test function
def adf_test(timeseries, name=''):
    """
    Augmented Dickey-Fuller Test
    H0 (Null Hypothesis): Time series has unit root (Non-stationary)
    H1 (Alternative Hypothesis): Time series is stationary

    If p-value < 0.05: Reject H0 â†’ Stationary
    If p-value >= 0.05: Fail to reject H0 â†’ Non-stationary
    """
    print(f"\n{'=' * 70}")
    print(f"ADF Test: {name}")
    print('=' * 70)

    result = adfuller(timeseries.dropna(), autolag='AIC')

    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.3f}')

    if result[1] <= 0.05:
        print(f"\nâœ… Result: STATIONARY (p-value = {result[1]:.6f} < 0.05)")
        print("   â†’ Reject null hypothesis")
        print("   â†’ Time series does NOT have unit root")
    else:
        print(f"\nâ�Œ Result: NON-STATIONARY (p-value = {result[1]:.6f} >= 0.05)")
        print("   â†’ Fail to reject null hypothesis")
        print("   â†’ Time series has unit root")

    return result[1]  # Return p-value

# 2. KPSS Test function
def kpss_test(timeseries, name=''):
    """
    KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin)
    H0 (Null Hypothesis): Time series is stationary
    H1 (Alternative Hypothesis): Time series is non-stationary

    If p-value < 0.05: Reject H0 â†’ Non-stationary
    If p-value >= 0.05: Fail to reject H0 â†’ Stationary
    """
    print(f"\n{'=' * 70}")
    print(f"KPSS Test: {name}")
    print('=' * 70)

    result = kpss(timeseries.dropna(), regression='c', nlags="auto")

    print(f'KPSS Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Critical Values:')
    for key, value in result[3].items():
        print(f'   {key}: {value:.3f}')

    if result[1] >= 0.05:
        print(f"\nâœ… Result: STATIONARY (p-value = {result[1]:.6f} >= 0.05)")
        print("   â†’ Fail to reject null hypothesis")
        print("   â†’ Time series is stationary")
    else:
        print(f"\nâ�Œ Result: NON-STATIONARY (p-value = {result[1]:.6f} < 0.05)")
        print("   â†’ Reject null hypothesis")
        print("   â†’ Time series is non-stationary")

    return result[1]  # Return p-value

print("\n[1] Testing Original Time Series...")
print("=" * 70)

# 3. Test Original Data
adf_p_original = adf_test(ts_data, 'Original Time Series')
kpss_p_original = kpss_test(ts_data, 'Original Time Series')

# 4. Test Trend Component
print("\n\n[2] Testing Trend Component...")
print("=" * 70)
adf_p_trend = adf_test(trend_component, 'Trend Component')
kpss_p_trend = kpss_test(trend_component, 'Trend Component')

# 5. Test Seasonal Component
print("\n\n[3] Testing Seasonal Component...")
print("=" * 70)
adf_p_seasonal = adf_test(seasonal_component, 'Seasonal Component')
kpss_p_seasonal = kpss_test(seasonal_component, 'Seasonal Component')

# 6. Test Residual Component (MOST IMPORTANT!)
print("\n\n[4] Testing Residual Component (MOST IMPORTANT!)...")
print("=" * 70)
print("â­� This should be STATIONARY for modeling!")
adf_p_residual = adf_test(residual_component, 'Residual Component')
kpss_p_residual = kpss_test(residual_component, 'Residual Component')

# 7. Visual inspection of Residual
print("\n\n[5] Visual Inspection of Residual Component...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Time series plot
axes[0, 0].plot(residual_component.index, residual_component.values, linewidth=0.5, alpha=0.7)
axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[0, 0].set_title('Residual Time Series', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Residual Value')
axes[0, 0].grid(True, alpha=0.3)

# Rolling mean and std
rolling_mean = residual_component.rolling(window=30).mean()
rolling_std = residual_component.rolling(window=30).std()

axes[0, 1].plot(residual_component.index, residual_component.values, linewidth=0.5, alpha=0.3, label='Original')
axes[0, 1].plot(rolling_mean.index, rolling_mean.values, linewidth=2, label='Rolling Mean (30-day)', color='blue')
axes[0, 1].plot(rolling_std.index, rolling_std.values, linewidth=2, label='Rolling Std (30-day)', color='red')
axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[0, 1].set_title('Rolling Statistics (Check for Stationarity)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Value')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Distribution
axes[1, 0].hist(residual_component.dropna(), bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_title('Residual Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Residual Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# ACF plot (Autocorrelation)
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residual_component.dropna(), lags=40, ax=axes[1, 1], alpha=0.05)
axes[1, 1].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Visual inspection completed!")

# 8. Summary table
print("\n" + "=" * 70)
print("STATIONARITY TEST SUMMARY")
print("=" * 70)

summary_data = {
    'Component': ['Original', 'Trend', 'Seasonal', 'Residual'],
    'ADF p-value': [adf_p_original, adf_p_trend, adf_p_seasonal, adf_p_residual],
    'KPSS p-value': [kpss_p_original, kpss_p_trend, kpss_p_seasonal, kpss_p_residual]
}

summary_df = pd.DataFrame(summary_data)

# Add interpretation
def interpret_stationarity(adf_p, kpss_p):
    if adf_p < 0.05 and kpss_p >= 0.05:
        return 'âœ… STATIONARY'
    elif adf_p >= 0.05 and kpss_p < 0.05:
        return 'â�Œ NON-STATIONARY'
    elif adf_p < 0.05 and kpss_p < 0.05:
        return 'âš ï¸� MIXED RESULT'
    else:
        return 'âš ï¸� UNCERTAIN'

summary_df['Result'] = summary_df.apply(lambda row: interpret_stationarity(row['ADF p-value'], row['KPSS p-value']), axis=1)

print("\n")
print(summary_df.to_string(index=False))

print("\n" + "=" * 70)
print("KEY FINDINGS:")
print("=" * 70)

# Check if residual is stationary
residual_stationary = (adf_p_residual < 0.05) and (kpss_p_residual >= 0.05)

if residual_stationary:
    print("\nâœ… GREAT NEWS! Residual component is STATIONARY!")
    print("   â†’ Ready for time series modeling (ARIMA, etc.)")
    print("   â†’ Mean and variance are constant over time")
    print("   â†’ No trend or seasonality remaining")
else:
    print("\nâš ï¸� WARNING! Residual component may NOT be fully stationary!")
    print("   â†’ May need additional preprocessing")
    print("   â†’ Consider differencing or other transformations")

print("\n" + "=" * 70)
print("INTERPRETATION GUIDE:")
print("=" * 70)
print("""
ADF Test:
  - p-value < 0.05 â†’ Stationary (Reject H0: has unit root)
  - p-value >= 0.05 â†’ Non-stationary (Fail to reject H0)

KPSS Test:
  - p-value >= 0.05 â†’ Stationary (Fail to reject H0: is stationary)
  - p-value < 0.05 â†’ Non-stationary (Reject H0)

For modeling, we need:
  âœ… ADF p-value < 0.05 AND KPSS p-value >= 0.05
""")

print("\n" + "=" * 70)
print("âœ… Step 4 Completed!")
print("ğŸ“� Next Step: Step 5 - Train/Test Split & Data Preparation")
print("=" * 70)


"""
Step 5: Train/Test Split & Data Preparation
- Split data into training and testing sets
- Prepare components for modeling
- Validate split strategy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("Step 5: Train/Test Split & Data Preparation")
print("=" * 70)

# 1. Check data length
print("\n[1] Checking data dimensions...")

total_days = len(ts_data)
print(f"Total days in dataset: {total_days}")
print(f"Start date: {ts_data.index[0]}")
print(f"End date: {ts_data.index[-1]}")

# 2. Define split point
print("\n[2] Defining train/test split...")

# Train/Test split ratio
train_ratio = 0.8  # 80% for training, 20% for testing

# Calculate split point
train_end = int(total_days * train_ratio)
test_days = total_days - train_end

print(f"\nğŸ“Š Split Strategy:")
print(f"   Split ratio: {train_ratio*100:.0f}% train / {(1-train_ratio)*100:.0f}% test")
print(f"   Total days available: {total_days} days")
print(f"   Training set: Day 1 ~ Day {train_end} ({train_end} days, {train_end/total_days*100:.1f}%)")
print(f"   Testing set: Day {train_end + 1} ~ Day {total_days} ({test_days} days, {test_days/total_days*100:.1f}%)")

# Validation check
if train_end >= total_days:
    raise ValueError(f"train_end ({train_end}) must be less than total_days ({total_days})")
if test_days <= 0:
    raise ValueError(f"Test set is empty!")

# 3. Split all components
print("\n[3] Splitting all components...")

# Original data
train_original = ts_data[:train_end]
test_original = ts_data[train_end:]

# Trend component
train_trend = trend_component[:train_end]
test_trend = trend_component[train_end:]

# Seasonal component
train_seasonal = seasonal_component[:train_end]
test_seasonal = seasonal_component[train_end:]

# Residual component (THIS IS WHAT WE'LL MODEL!)
train_residual = residual_component[:train_end]
test_residual = residual_component[train_end:]

print(f"âœ… Train set size: {len(train_original)} days")
print(f"âœ… Test set size: {len(test_original)} days")

# 4. Visualize split
print("\n[4] Visualizing train/test split...")

fig, axes = plt.subplots(4, 1, figsize=(15, 12))

# Original
axes[0].plot(train_original.index, train_original.values, label='Train', linewidth=1, color='blue')
axes[0].plot(test_original.index, test_original.values, label='Test', linewidth=1, color='orange')
axes[0].axvline(x=train_original.index[-1], color='red', linestyle='--', linewidth=2, label='Split Point')
axes[0].set_title('Original Time Series - Train/Test Split', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Sales')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Trend
axes[1].plot(train_trend.index, train_trend.values, label='Train', linewidth=2, color='blue')
axes[1].plot(test_trend.index, test_trend.values, label='Test', linewidth=2, color='orange')
axes[1].axvline(x=train_trend.index[-1], color='red', linestyle='--', linewidth=2, label='Split Point')
axes[1].set_title('Trend Component - Train/Test Split', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Trend')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Seasonal
axes[2].plot(train_seasonal.index, train_seasonal.values, label='Train', linewidth=1, color='blue')
axes[2].plot(test_seasonal.index, test_seasonal.values, label='Test', linewidth=1, color='orange')
axes[2].axvline(x=train_seasonal.index[-1], color='red', linestyle='--', linewidth=2, label='Split Point')
axes[2].set_title('Seasonal Component - Train/Test Split', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Seasonality')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Residual (MOST IMPORTANT!)
axes[3].plot(train_residual.index, train_residual.values, label='Train', linewidth=0.5, color='blue', alpha=0.7)
axes[3].plot(test_residual.index, test_residual.values, label='Test', linewidth=0.5, color='orange', alpha=0.7)
axes[3].axvline(x=train_residual.index[-1], color='red', linestyle='--', linewidth=2, label='Split Point')
axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.3)
axes[3].set_title('Residual Component (Stationary) - Train/Test Split', fontsize=14, fontweight='bold')
axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Date')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("âœ… Visualization completed!")

# 5. Validate stationarity of training residual
print("\n[5] Validating stationarity of training residual...")

from statsmodels.tsa.stattools import adfuller, kpss

# ADF test on training residual
adf_result = adfuller(train_residual.dropna(), autolag='AIC')
print(f"\nADF Test on Training Residual:")
print(f"   ADF Statistic: {adf_result[0]:.6f}")
print(f"   p-value: {adf_result[1]:.6f}")
if adf_result[1] < 0.05:
    print(f"   âœ… Training residual is STATIONARY (p < 0.05)")
else:
    print(f"   âš ï¸� Training residual may not be stationary (p >= 0.05)")

# KPSS test on training residual
kpss_result = kpss(train_residual.dropna(), regression='c', nlags="auto")
print(f"\nKPSS Test on Training Residual:")
print(f"   KPSS Statistic: {kpss_result[0]:.6f}")
print(f"   p-value: {kpss_result[1]:.6f}")
if kpss_result[1] >= 0.05:
    print(f"   âœ… Training residual is STATIONARY (p >= 0.05)")
else:
    print(f"   âš ï¸� Training residual may not be stationary (p < 0.05)")

# 6. Compare statistics between train and test
print("\n[6] Comparing statistics between train and test sets...")

stats_comparison = pd.DataFrame({
    'Component': ['Original', 'Trend', 'Seasonal', 'Residual'],
    'Train Mean': [
        train_original.mean(),
        train_trend.mean(),
        train_seasonal.mean(),
        train_residual.mean()
    ],
    'Test Mean': [
        test_original.mean(),
        test_trend.mean(),
        test_seasonal.mean(),
        test_residual.mean()
    ],
    'Train Std': [
        train_original.std(),
        train_trend.std(),
        train_seasonal.std(),
        train_residual.std()
    ],
    'Test Std': [
        test_original.std(),
        test_trend.std(),
        test_seasonal.std(),
        test_residual.std()
    ]
})

print("\n")
print(stats_comparison.to_string(index=False))

# 7. Prepare data summary
print("\n" + "=" * 70)
print("DATA PREPARATION SUMMARY")
print("=" * 70)

# Check if test set exists
if len(test_original) > 0:
    test_range_str = f"{test_original.index[0]} to {test_original.index[-1]}"
else:
    test_range_str = "No test data (train_end >= total_days)"

print(f"""
ğŸ“Š Dataset Information:
   - Total days: {total_days}
   - Training days: {train_end} ({train_end/total_days*100:.1f}%)
   - Testing days: {total_days - train_end} ({(total_days-train_end)/total_days*100:.1f}%)
   
ğŸ“… Date Ranges:
   - Training: {train_original.index[0]} to {train_original.index[-1]}
   - Testing: {test_range_str}

âœ… Components Split:
   - Original (full data with trend + seasonality)
   - Trend (long-term pattern)
   - Seasonal (weekly pattern)
   - Residual (stationary component for modeling) â­�

ğŸ�¯ Modeling Strategy:
   1. Model the TRAINING RESIDUAL (stationary)
   2. Generate predictions for TEST period
   3. Add back SEASONAL component
   4. Add back TREND component
   5. Get final forecast = Residual_pred + Seasonal + Trend
""")

print("=" * 70)
print("âœ… Step 5 Completed!")
print("ğŸ“� Next Step: Step 6 - Time Series Modeling (ARIMA)")
print("=" * 70)

# 8. Save split data for modeling
print("\n[8] Saving split data for next steps...")

# Create a dictionary to store all splits
split_data = {
    'train_original': train_original,
    'test_original': test_original,
    'train_trend': train_trend,
    'test_trend': test_trend,
    'train_seasonal': train_seasonal,
    'test_seasonal': test_seasonal,
    'train_residual': train_residual,
    'test_residual': test_residual
}

print("âœ… Split data saved!")
print("\nAvailable variables:")
print("   - train_original, test_original")
print("   - train_trend, test_trend")
print("   - train_seasonal, test_seasonal")
print("   - train_residual, test_residual â­� (for modeling)")

print("\n" + "=" * 70)
print("ğŸ�¯ Ready for modeling!")
print("=" * 70)


"""
Step 6: ARIMA Modeling & Forecasting
- Find optimal ARIMA parameters
- Train model on residual component
- Generate predictions
- Reconstruct forecast with trend + seasonal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Step 6: ARIMA Modeling & Forecasting")
print("=" * 70)

# 1. ACF and PACF plots to determine ARIMA parameters
print("\n[1] Analyzing ACF and PACF for parameter selection...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# ACF plot
plot_acf(train_residual.dropna(), lags=40, ax=axes[0], alpha=0.05)
axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Lags')
axes[0].set_ylabel('ACF')

# PACF plot
plot_pacf(train_residual.dropna(), lags=40, ax=axes[1], alpha=0.05)
axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Lags')
axes[1].set_ylabel('PACF')

plt.tight_layout()
plt.show()

print("âœ… ACF/PACF plots completed!")
print("""
Parameter Selection Guide:
- ACF cuts off at lag q â†’ MA(q)
- PACF cuts off at lag p â†’ AR(p)
- Both decay gradually â†’ ARMA(p,q)
- d = 0 (residual is already stationary)
""")

# 2. Try multiple ARIMA models and select best one
print("\n[2] Testing multiple ARIMA models...")

# Define candidate models
candidate_models = [
    (0, 0, 0),  # Baseline (White Noise)
    (1, 0, 0),  # AR(1)
    (2, 0, 0),  # AR(2)
    (0, 0, 1),  # MA(1)
    (0, 0, 2),  # MA(2)
    (1, 0, 1),  # ARMA(1,1)
    (2, 0, 1),  # ARMA(2,1)
    (1, 0, 2),  # ARMA(1,2)
    (2, 0, 2),  # ARMA(2,2)
]

results = []

print("\nTesting models...")
for order in candidate_models:
    try:
        model = ARIMA(train_residual.dropna(), order=order)
        fitted_model = model.fit()
        aic = fitted_model.aic
        bic = fitted_model.bic
        results.append({
            'order': order,
            'AIC': aic,
            'BIC': bic
        })
        print(f"   ARIMA{order}: AIC={aic:.2f}, BIC={bic:.2f}")
    except Exception as e:
        print(f"   ARIMA{order}: Failed ({str(e)[:50]})")

# Convert to DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('AIC')

print("\nğŸ“Š Model Comparison (sorted by AIC):")
print(results_df.to_string(index=False))

# Select best model (lowest AIC)
best_order = results_df.iloc[0]['order']
print(f"\nâœ… Best model selected: ARIMA{best_order}")
print(f"   AIC: {results_df.iloc[0]['AIC']:.2f}")
print(f"   BIC: {results_df.iloc[0]['BIC']:.2f}")

# 3. Fit the best ARIMA model
print("\n[3] Fitting best ARIMA model...")

final_model = ARIMA(train_residual.dropna(), order=best_order)
fitted_final_model = final_model.fit()

print("âœ… Model fitting completed!")
print(f"\nModel Summary:")
print(fitted_final_model.summary())

# 4. Generate predictions for test period
print("\n[4] Generating predictions for test period...")

# Predict residual for test period
n_forecast = len(test_residual)
forecast_residual = fitted_final_model.forecast(steps=n_forecast)

print(f"âœ… Generated {n_forecast} forecasts")
print(f"   Mean prediction: {forecast_residual.mean():.2f}")
print(f"   Std prediction: {forecast_residual.std():.2f}")

# 5. Reconstruct final forecast
print("\n[5] Reconstructing final forecast...")

# Get test seasonal and trend components
test_seasonal_values = test_seasonal.values
test_trend_values = test_trend.values

# Reconstruct: Final = Residual_pred + Seasonal + Trend
final_forecast = forecast_residual.values + test_seasonal_values + test_trend_values

print("âœ… Final forecast reconstructed!")
print(f"""
Reconstruction:
  Residual prediction: mean={forecast_residual.mean():.2f}
  + Seasonal component: mean={test_seasonal.mean():.2f}
  + Trend component: mean={test_trend.mean():.2f}
  = Final forecast: mean={final_forecast.mean():.2f}
""")

# 6. Visualize predictions
print("\n[6] Visualizing predictions...")

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# 6-1. Residual: Actual vs Predicted
axes[0].plot(train_residual.index[-100:], train_residual.values[-100:], 
             label='Train (last 100 days)', linewidth=1, color='blue', alpha=0.5)
axes[0].plot(test_residual.index, test_residual.values, 
             label='Test Actual', linewidth=2, color='green')
axes[0].plot(test_residual.index, forecast_residual.values, 
             label='Test Predicted', linewidth=2, color='red', linestyle='--')
axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
axes[0].axvline(x=train_residual.index[-1], color='red', linestyle='--', alpha=0.5)
axes[0].set_title('Residual Component: Actual vs Predicted', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Residual')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 6-2. Components stacking
test_dates = test_original.index
axes[1].plot(test_dates, test_trend_values, label='Trend', linewidth=2)
axes[1].plot(test_dates, test_trend_values + test_seasonal_values, 
             label='Trend + Seasonal', linewidth=2)
axes[1].plot(test_dates, final_forecast, label='Trend + Seasonal + Residual (Final)', 
             linewidth=2, linestyle='--')
axes[1].set_title('Forecast Components Reconstruction', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Value')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 6-3. Final forecast vs actual
axes[2].plot(train_original.index[-100:], train_original.values[-100:], 
             label='Train (last 100 days)', linewidth=1, color='blue', alpha=0.5)
axes[2].plot(test_original.index, test_original.values, 
             label='Test Actual', linewidth=2, color='green')
axes[2].plot(test_original.index, final_forecast, 
             label='Forecast', linewidth=2, color='red', linestyle='--')
axes[2].axvline(x=train_original.index[-1], color='red', linestyle='--', alpha=0.5, 
                label='Train/Test Split')
axes[2].set_title('Final Forecast vs Actual Sales', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Sales')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("âœ… Visualization completed!")

# 7. Evaluate forecast accuracy
print("\n[7] Evaluating forecast accuracy...")

# Calculate metrics
mae = mean_absolute_error(test_original.values, final_forecast)
rmse = np.sqrt(mean_squared_error(test_original.values, final_forecast))
mape = np.mean(np.abs((test_original.values - final_forecast) / test_original.values)) * 100

# Baseline (naive forecast - last value)
naive_forecast = np.full(len(test_original), train_original.values[-1])
mae_naive = mean_absolute_error(test_original.values, naive_forecast)
rmse_naive = np.sqrt(mean_squared_error(test_original.values, naive_forecast))

print(f"""
ğŸ“Š Forecast Accuracy Metrics:

ARIMA Model:
  - MAE (Mean Absolute Error): {mae:,.2f}
  - RMSE (Root Mean Squared Error): {rmse:,.2f}
  - MAPE (Mean Absolute Percentage Error): {mape:.2f}%

Baseline (Naive):
  - MAE: {mae_naive:,.2f}
  - RMSE: {rmse_naive:,.2f}

Improvement:
  - MAE improvement: {(1 - mae/mae_naive)*100:.1f}%
  - RMSE improvement: {(1 - rmse/rmse_naive)*100:.1f}%
""")

# 8. Residual diagnostics
print("\n[8] Model diagnostics...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Residuals
model_residuals = fitted_final_model.resid

# 8-1. Residuals plot
axes[0, 0].plot(model_residuals)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_title('Model Residuals', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].grid(True, alpha=0.3)

# 8-2. Residuals distribution
axes[0, 1].hist(model_residuals, bins=30, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=0, color='red', linestyle='--')
axes[0, 1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Residual Value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 8-3. ACF of residuals
plot_acf(model_residuals, lags=40, ax=axes[1, 0], alpha=0.05)
axes[1, 0].set_title('ACF of Model Residuals', fontsize=12, fontweight='bold')

# 8-4. Q-Q plot
from scipy import stats
stats.probplot(model_residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("âœ… Model diagnostics completed!")

# 9. Summary
print("\n" + "=" * 70)
print("STEP 6 SUMMARY")
print("=" * 70)

print(f"""
âœ… Model Selection:
   - Best model: ARIMA{best_order}
   - AIC: {results_df.iloc[0]['AIC']:.2f}
   - BIC: {results_df.iloc[0]['BIC']:.2f}

âœ… Forecasting:
   - Forecast horizon: {n_forecast} days
   - Method: Decomposition + ARIMA + Reconstruction

âœ… Accuracy:
   - MAE: {mae:,.2f}
   - RMSE: {rmse:,.2f}
   - MAPE: {mape:.2f}%
   - Better than naive by {(1 - mae/mae_naive)*100:.1f}%

âœ… Components:
   - Residual predicted by ARIMA{best_order}
   - Seasonal component added back
   - Trend component added back
   - Final forecast = Residual_pred + Seasonal + Trend
""")

print("=" * 70)
print("âœ… Step 6 Completed!")
print("ğŸ“� Next: Interpret results and create final report")
print("=" * 70)

# Save results for later use
forecast_results = {
    'test_actual': test_original.values,
    'forecast': final_forecast,
    'forecast_residual': forecast_residual.values,
    'test_seasonal': test_seasonal.values,
    'test_trend': test_trend.values,
    'mae': mae,
    'rmse': rmse,
    'mape': mape,
    'best_model': best_order
}

print("\nâœ… Results saved in 'forecast_results' dictionary")


import pandas as pd

# ì˜ˆì‹œ ìƒ�ì„±
submission = pd.DataFrame({
    'id': [f'item_{i+1}' for i in range(len(final_forecast))],
    'F1': final_forecast  # ì˜ˆì¸¡ë�œ íŒ�ë§¤ëŸ‰
})

submission.to_csv('submission.csv', index=False)
print("âœ… submission.csv created successfully!")


