import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.fft import fft, ifft


df = pd.read_csv('/kaggle/input/follow-the-trend-time-series-forecasting/train.csv')

# Convert Sales_Date to datetime
df['Sales_Date'] = pd.to_datetime(df['Sales_Date'])

# Set Sales_Date as the index
df.set_index('Sales_Date', inplace=True)

# Inspect the data
print(df.head())
print(df.info())
print(df.isnull().sum())


# Generating descriptive statistics of the DataFrame
df.describe()


unique_dates = df.index.unique()
print(f"Number of unique dates: {len(unique_dates)}")


print("Unique Products:", df['Product'].nunique())
print("Unique Countries:", df['Country'].nunique())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df['Country'].value_counts().plot(kind='barh', ax=axes[0], color='skyblue', title='Sales by Country')
df['Product'].value_counts().plot(kind='barh', ax=axes[1], color='salmon', title='Sales by Product')

plt.tight_layout()
plt.show()



# Sales distribution by Country
plt.figure(figsize=(10, 6))
sns.boxplot(x='Country', y='Sales_Qty', data=df)
plt.title('Sales Distribution by Country')
plt.xticks(rotation=90)
plt.show()

# Sales distribution by Product
plt.figure(figsize=(10, 6))
sns.boxplot(x='Product', y='Sales_Qty', data=df)
plt.title('Sales Distribution by Product')
plt.xticks(rotation=90)
plt.show()


# total sales quantity for each date, summed across all Country and Product combinations
# Aggregate sales by date (sum)
df_aggregated = df.groupby(df.index)['Sales_Qty'].sum()  

# Plot the aggregated sales trend
plt.figure(figsize=(12, 6))
plt.plot(df_aggregated.index, df_aggregated.values, label='Aggregated Sales Quantity', color='blue')
plt.title('Overall Sales Trend Over Time (Aggregated)')
plt.xlabel('Date')
plt.ylabel('Sales Quantity')
plt.legend()
plt.grid()
plt.show()


# Decompose the time series
decomposition = seasonal_decompose(df_aggregated, model='additive', period=12)

# Create a larger figure and adjust spacing
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.5)  # Adjust spacing between subplots

# Graph 1: Observed (Original Series)
ax1.plot(decomposition.observed, label='Observed')
ax1.set_ylabel('Value')
ax1.set_title('Original Time Series (Observed)')
ax1.legend()

# Graph 2: Trend
ax2.plot(decomposition.trend, label='Trend', color='orange')
ax2.set_ylabel('Value')
ax2.set_title('Trend')
ax2.legend()

# Graph 3: Seasonal
ax3.plot(decomposition.seasonal, label='Seasonal', color='green')
ax3.set_ylabel('Value')
ax3.set_title('Seasonality')
ax3.legend()

# Graph 4: Residual
ax4.plot(decomposition.resid, label='Residual', color='red')
ax4.set_ylabel('Value')
ax4.set_title('Residuals')
ax4.legend()

# Display the figure
plt.show()


# Extracting month and year from the index
df['month'] = df.index.month
df['year'] = df.index.year

# Grouping data by month and year
df_plot = df.groupby(['month', 'year'])['Sales_Qty'].sum().reset_index()

# Defining color palette
np.random.seed(42)
years = df_plot['year'].unique()
colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'black']

# Plotting
plt.figure(figsize=(16, 12))
for i, y in enumerate(years):
    if i < len(colors):
        plt.plot('month', 'Sales_Qty', data=df_plot[df_plot['year'] == y], color=colors[i], label=y)
        plt.text(df_plot[df_plot['year'] == y]['month'].max() + 0.1, 
                 df_plot[df_plot['year'] == y]['Sales_Qty'].values[-1], 
                 str(y), fontsize=12, color=colors[i])

plt.gca().set(ylabel='Sales Quantity', xlabel='Month')
plt.yticks(fontsize=12, alpha=.7)
plt.title("Seasonal Plot - Monthly Sales Quantity", fontsize=20)
plt.ylabel('Sales Quantity')
plt.xlabel('Month')
plt.legend(title="Year", loc='upper left')
plt.show()


window_size = 6  # Example: 3-month rolling average
rolling_mean = df_aggregated.rolling(window=window_size).mean()

plt.figure(figsize=(12, 6))
plt.plot(df_aggregated, label='Original')
plt.plot(rolling_mean, label=f'{window_size}-Month Rolling Mean', color='red')
plt.title('Rolling Mean of Sales Quantity')
plt.xlabel('Date')
plt.ylabel('Sales Quantity')
plt.legend()
plt.show()


# a) Drop rows with NaNs 
df_cleaned = df.dropna()

# sophisticated methods like interpolation.  Here's an example with forward fill:
df_filled = df.fillna(method='ffill')  # Forward fill

df_aggregated = df_filled['Sales_Qty'].resample('MS').sum() #resampling after filling nan values
# 2. Detrend the Data (using the cleaned or filled data)
df_detrended = df_aggregated.diff().dropna()  # First-order differencing


# 3. Compute the Fourier Transform (on the detrended data)
N = len(df_detrended)
T = 1  # Time interval 
yf = fft(df_detrended.values)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

# 4. Analyze the Frequency Spectrum
plt.figure(figsize=(12, 6))
plt.plot(xf, np.abs(yf[0:N//2]))
plt.title('Frequency Spectrum of Detrended Sales Quantity')
plt.xlabel('Frequency (cycles/month)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

# ... (Rest of the analysis: find peaks, interpret, etc.)


#Lag analysis (Is past sales impacting future sales?):

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df['Sales_Qty'], lags=20)  # Adjust lags as needed
plt.show()

