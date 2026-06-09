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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Visualization configuration
plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)


print("Loading data...")
train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
print(f"Data loaded successfully. Shape: {train.shape}")
train = train[:262943]
# Initial data overview
print("\n=== Initial Data Overview ===")
display(train.head())



print("\nData types:")
print(train.dtypes.value_counts())



print("\nDescriptive statistics:")
train.describe().T # Transposed for better readability


print("\n=== Missing Value Analysis ===")
missing_data = train.isnull().sum().sort_values(ascending=False)
missing_percent = (missing_data / len(train) * 100).round(2)
missing_report = pd.concat([missing_data, missing_percent], axis=1)
missing_report.columns = ['Missing Count', 'Missing %']
display(missing_report[missing_report['Missing Count'] > 0])


print("\n=== Target Variable Analysis ===")
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Distribution plot
sns.histplot(train['label'], bins=100, kde=True, ax=ax[0])
ax[0].set_title('Target Distribution')

# Stationarity test
adf_test = adfuller(train['label'].dropna())
print(f"\nADF Test Results:\n- ADF Statistic: {adf_test[0]:.4f}\n- p-value: {adf_test[1]:.4f}")

# Descriptive stats
print("\nTarget Statistics:")
print(train['label'].describe().T)


print("\n=== Temporal Analysis ===")
train['timestamp'] = pd.to_datetime(train.index)

# Time series visualization
fig, ax = plt.subplots(2, 1, figsize=(16, 10))
train['label'].plot(ax=ax[0], title='Target Time Series')

# Seasonality decomposition
train['hour'] = train['timestamp'].dt.hour
train['day_of_week'] = train['timestamp'].dt.dayofweek
train['month'] = train['timestamp'].dt.month

train.groupby('hour')['label'].mean().plot(kind='bar', ax=ax[1], 
                                          title='Average Target by Hour of Day')
plt.tight_layout()


print("\n=== Explicit Feature Analysis ===")
explicit_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']

# Correlation matrix
plt.figure(figsize=(10, 8))
corr_matrix = train[explicit_features + ['label']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            annot_kws={'size': 10}, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.show()


print("\n=== Anonymous Feature Analysis ===")
X_cols = [col for col in train.columns if col.startswith('X')]

# Top correlated features
corr_with_target = train[X_cols].corrwith(train['label']).abs().sort_values(ascending=False)
print("\nTop 10 Features by Correlation Magnitude:")
print(corr_with_target.head(10))

# Visualize top correlations
top_features = corr_with_target.head(5).index
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, feature in zip(axes, top_features):
    sns.scatterplot(x=train[feature], y=train['label'], ax=ax, alpha=0.3)
    ax.set_title(f'{feature} (corr: {corr_with_target[feature]:.2f})')
plt.tight_layout()


print("\n=== Advanced Time Series Analysis ===")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# ACF/PACF plots
plot_acf(train['label'].dropna(), lags=50, ax=axes[0], title='ACF')
plot_pacf(train['label'].dropna(), lags=50, ax=axes[1], title='PACF')
plt.tight_layout()

# Rolling statistics
rolling_window = 60  # 60-minute window
train['rolling_mean'] = train['label'].rolling(window=rolling_window).mean()
train['rolling_std'] = train['label'].rolling(window=rolling_window).std()

plt.figure(figsize=(16, 6))
plt.plot(train['label'], label='Raw Target', alpha=0.5)
plt.plot(train['rolling_mean'], label=f'{rolling_window}-min MA', color='red')
plt.plot(train['rolling_std'], label=f'{rolling_window}-min StdDev', color='green')
plt.title('Rolling Statistics Analysis')
plt.legend()
plt.show()


print("\n=== Outlier Analysis ===")
fig, axes = plt.subplots(4, 1, figsize=(16, 12))  

# Target outliers
sns.boxplot(x=train['label'], ax=axes[0])
axes[0].set_title('Target Outliers')

# Feature outliers
for i, feature in enumerate(explicit_features[:3]):
    sns.boxplot(x=train[feature], ax=axes[i + 1])
    axes[i + 1].set_title(f'{feature} Outliers')

plt.tight_layout()
plt.show()


