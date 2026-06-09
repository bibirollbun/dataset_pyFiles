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


import time
# load training data
t1 = time.time()
df_train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
t2 = time.time()
print('Elapsed time [s]:', np.round(t2-t1, 2))


# preview
df_train.head(2)


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# train.parquet
## The training dataset containing all historical market data along with the corresponding labels.

### timestamp: The timestamp index representing the minute associated with each row.
### bid_qty: The total quantity buyers are willing to purchase at the best (highest) bid price at the given timestamp.
### ask_qty: The total quantity sellers are offering to sell at the best (lowest) ask price at the given timestamp.
### buy_qty: The total trading quantity executed at the best ask price during the given minute.
### sell_qty: The total trading quantity executed at the best bid price during the given minute.
### volume: The total traded volume during the minute.
### X_{1,...,780}: A set of anonymized market features derived from proprietary data sources.
### label: The target variable representing the anonymized market price movement to be predicted.

# structure details
# df_train.info(verbose=True, show_counts=True)

# main features
features_main = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']

# anonymized features
features_x = ['X' + str(i) for i in range(1,890+1)]

# target
target = 'label'


import warnings
warnings.filterwarnings('ignore')


print ("Rows     : " ,df_train.shape[0])
print ("Columns  : " ,df_train.shape[1])
print ("\nMissing values :  ", df_train.isnull().any())
print ("\nUnique values :  \n",df_train.nunique())
     

df_train_non_indexed=df_train.copy()


df_train.reset_index(drop=False)


from datetime import datetime


df_train.index


label_data=df_train['label']
label_data.head(2)


import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt


label_data.plot(grid=True)


import matplotlib.pyplot as plt
import pandas as pd

# # Ensure the index is a DatetimeIndex
# df_train.index = pd.to_datetime(df_train.index)

# # Loop from March to December 2023
# for month in range(3, 13):  # 3 to 12 inclusive
#     month_str = f'2023-{month:02d}'  # e.g., '2023-03', '2023-04', etc.
#     df_month = df_train.loc[month_str]

#     if 'label' in df_month.columns:
#         df_label_month = df_month['label']

#         plt.figure(figsize=(10, 4))
#         df_label_month.plot(grid=True, title=f'Label Plot - {month_str}');
#         plt.xlabel('Date')
#         plt.ylabel('Label')
#         plt.tight_layout()
#         plt.show()
#     else:
#         print(f"'label' column not found for {month_str}")



# import matplotlib.pyplot as plt
# import pandas as pd

# # Ensure datetime index
# df_train.index = pd.to_datetime(df_train.index)

# # Define simplified ranges
# ranges = {
#     '[-2.5, 2.5]': lambda x: (x >= -2.5) & (x <= 2.5),
#     '(-5, -2.5) ∪ (2.5, 5]': lambda x: ((x > 2.5) & (x <= 5)) | ((x > -5) & (x < -2.5)),
#     'Extreme (< -5 or > 5)': lambda x: (x <= -5) | (x > 5)
# }

# # Loop through March to December 2023
# # for month in range(3, 13):
#     month_str = f'2023-{month:02d}'
#     df_month = df_train.loc[month_str]

#     if 'label' not in df_month.columns:
#         print(f"'label' column not found in {month_str}")
#         continue

#     df_label = df_month['label']

#     # --- Plot Setup ---
#     plt.figure(figsize=(14, 5))

#     # Line plot
#     plt.subplot(1, 2, 1)
#     df_label.plot(grid=True, title=f'Label Time Series - {month_str}')
#     plt.xlabel('Date')
#     plt.ylabel('Label')

#     # Box plot
#     plt.subplot(1, 2, 2)
#     box_data = []

#     for label_range, condition in ranges.items():
#         filtered = df_label[condition(df_label)]
#         box_data.append(filtered)

#     plt.boxplot(box_data, labels=ranges.keys(), showfliers=False)
#     plt.title(f'Label Box Plot by Group - {month_str}')
#     plt.ylabel('Label')

#     plt.tight_layout()
#     plt.show()



df=df_train.reset_index()
df.head()



# You processed it and saved the needed output, now delete 'df_train' as it is no longer needed
del df_train

# Import garbage collector and free memory
import gc
gc.collect()


import plotly.express as px
fig = px.line(df, x='index', y='label', title='label with Slider')

fig.update_xaxes(rangeslider_visible=True)
fig.show()


# import plotly.express as px
# fig = px.line(df, x='index', y='volume', title='volume with Slider')

# fig.update_xaxes(rangeslider_visible=True)
# fig.show()


df.columns


# fig = px.line(df, x='index', y='bid_qty', title='bid qty with Slider')

# fig.update_xaxes(
#     rangeslider_visible=True,
#     rangeselector=dict(
#         buttons=list([
#             dict(count=1, label="1m", step="month", stepmode="backward"),
#             dict(count=2, label="2m", step="month", stepmode="backward"),
#             dict(count=3, label="3m", step="month", stepmode="backward"),
#             dict(step="all")
#         ])
#     )
# )
# fig.show()
     


features_main


# # Ensure datetime index
# df.index = pd.to_datetime(df.index)

# # Slice date range and select main features
# df_filtered = df['2023':'2024'][features_main].copy()
# df_filtered['month'] = df_filtered.index.to_period('M')  # or use .month for numeric

# # Group by month and get descriptive stats
# monthly_stats = df_filtered.groupby('month').describe()

# print(monthly_stats)

# df.head()


df = df.set_index('index')


# Rolling sum with a 5-row window
window = 5
rolling_sum = df.rolling(window=window, min_periods=window).sum()

# Identify rows to keep: first row, plus every minute % 5 == 0 after the first row
to_show = (df.index == df.index[0]) | ((df.index.minute % 5 == 0) & (df.index != df.index[0]))

# Combine: for the first row, show original; for 5-minute marks, show rolling sum
result = df.copy()
result[to_show] = rolling_sum[to_show]
result.iloc[0] = df.iloc[0]  # Ensure first row is as-is

# print(result[to_show])



df.shape


# You processed it and saved the needed output, now delete 'large_df' as it is no longer needed
del df

# Import garbage collector and free memory
import gc
gc.collect()


result[to_show].tail()


train=result[to_show]


features_main


# import matplotlib.pyplot as plt

# # Replace 'features_main' with your actual feature list
# # Example: features_main = ['X1', 'X2', 'X3', ..., 'Xn']
# # And train is your DataFrame

# axes = train[features_main].hist(
#     bins=20, 
#     figsize=(15, 15)
# )

# for ax in axes.flatten():
#     ax.set_xlim(0, 1000)  # Set x-axis from 0 to 1
#     ax.set_xlabel("Value (0-100)")

# plt.tight_layout()
# plt.show()



train[['label']].plot(kind='density')


pd.plotting.lag_plot(train['label'],lag=1)



# You processed it and saved the needed output, now delete 'large_df' as it is no longer needed
del train

# Import garbage collector and free memory
gc.collect()


pd.plotting.lag_plot(result[to_show]['label'],lag=85)


pd.plotting.lag_plot(result[to_show]['label'],lag=13)


result[to_show].index
df_reset = result[to_show].reset_index(drop=False)



df_reset.tail(1)


multi_data = df_reset[features_main]
multi_data.plot(subplots=True)


df_reset.tail(1)


df_reset.index


df_reset.loc['2023-03':'2024-02', ['bid_qty', 'ask_qty']].plot(figsize=(15,8), linewidth=3, fontsize=15)

import matplotlib.pyplot as plt
plt.xlabel('year_month', fontsize=20)
plt.show()


df_reset.isnull().any()


# Get the count of nulls in each column
null_counts = df_reset.isnull().sum()

# Filter columns where count of nulls > 0
cols_with_nulls = null_counts[null_counts > 0]

print(cols_with_nulls)



g = sns.pairplot(df_reset[features_main])


df_resets=df_reset[features_main].corr(method='pearson')
df_resets


g = sns.heatmap(df_resets,  vmax=.6, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', cmap='coolwarm')
g.figure.set_size_inches(10,10)
    
plt.show()


df_resets.columns


# Filter for March 2023
march1 = result[to_show].loc['2023-03-01':'2023-03-02']['label']



pd.plotting.autocorrelation_plot(march1)

