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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime as datetime

warnings.filterwarnings("ignore")


df_train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")


df_train.head()


df_train.info()


df_train = df_train.drop('id', axis=1)
df_train.head()


#Convert the date column in datetime format
df_train['date'] = pd.to_datetime(df_train['date'])


df_train.plot(color = 'green')


plt.figure(figsize=(12,6))
df_train.groupby('date')['num_sold'].sum().plot(xlabel = 'date',
                                                color = 'green',
                                                ylabel = 'number of Produce Sold',
                                                title = 'Total sales Over Time')
plt.grid()
plt.show()


df_train['date'] = pd.to_datetime(df_train['date'])

plt.figure(figsize=(12,6))
df_train.groupby('date')['num_sold'].sum().plot(xlabel = 'date',
                                                ylabel = 'Number of Produce Sold',
                                                xlim=['2014-05-01', '2017-1-05'],
                                                ylim = [0, 130000],
                                                color = 'green',
                                                title = 'Sales of last two years')
plt.grid()
plt.show()


# set the date column as the index of DataFrame,

df_train = df_train.set_index('date')
df_train


df_train.index


# .loc = Label based indexing [Use .loc when you know the labels of rows or columns.]

# .iloc = integer based indexing [Use .iloc when you know the positions of rows or columns.]

df_train.loc['2010-01-01': '2010-01-02']


index = df_train.loc['2010-01-01': '2011-01-02'].index
numsold_open = df_train.loc['2010-01-01':'2011-01-02']['num_sold']


numsold_open


figure,axis = plt.subplots()
plt.tight_layout()
#preventing overlapping
figure.autofmt_xdate()
axis.plot(index, numsold_open)


""" First when you see the dataframe column date always having object datatype you have to convert 
that column in to datetime format usinf fpoollowing one line """
# df_train['date'] = pd.to_datetime(df_train['date'])


#DateTime Index
df_train.info()


df_train = df_train.reset_index()
df_train


df_train.info()


df_train = df_train.set_index('date', drop=True)


df_train


from datetime import datetime
datetime(2025,1,17)


datetime.now()


date = datetime(2025,1,17)


date


date.day


date.weekday()


date.year


date.month


df_train.head()


# Minimum number of sales in each year
df_train.resample(rule='A').min()


#Rule A:- Year end frequency
df_train.resample(rule='A').min()['num_sold'].plot()


# Maximim number of sales in each year
df_train.resample(rule='A').max()


df_train.resample(rule='A').max()['num_sold'].plot()


#Rule QS:- Quarterly start frequency
df_train.resample(rule='QS')['num_sold'].plot()
plt.grid()
plt.show()


#Rule BA:- Business end frequency
df_train.resample(rule='BA')['num_sold'].plot()
plt.grid()
plt.show()


#Rule BQS:- Business Qurterly frequency
df_train.resample(rule='BQS')['num_sold'].plot()
plt.grid()
plt.show()


df_train['num_sold'].resample(rule='A').mean().plot(kind='bar')


# ruleM = Monthly
df_train['num_sold'].resample(rule='M').max().plot(kind='bar', figsize=(15, 6))


df_train.head()


df_train["num_sold"].rolling(5).mean().head(10)


df_train['num_sold: 30 days of rolling']=df_train['num_sold'].rolling(30).mean()


df_train.head(35)


# Plot with colors
df_train[['num_sold', 'num_sold: 30 days of rolling']].plot(
    figsize=(12, 5),
    color=['blue', 'yellow'],  
    linewidth=2,            
    alpha=0.9                
)

plt.title("Sales and Rolling Sales Trend", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Number Sold", fontsize=12)
plt.legend(['Num Sold', '30-Day Rolling Average'])
plt.grid(True, linestyle='--', alpha=0.5)  

plt.show()


## In the above plot the yellow line is more smoother version of blue line.


df_train['num_sold: 10 days of rolling'] = df_train['num_sold'].rolling(window=10, min_periods=1).mean()
df_train['num_sold: 30 days of rolling'] = df_train['num_sold'].rolling(window=30, min_periods=1).mean()
df_train['num_sold: 50 days of rolling'] = df_train['num_sold'].rolling(window=50, min_periods=1).mean()


df_train[["num_sold", "num_sold: 10 days of rolling"]].plot(figsize=(12, 5))


df_train[['num_sold', 'num_sold: 10 days of rolling', 'num_sold: 30 days of rolling', 'num_sold: 50 days of rolling']].plot(figsize=(12, 6),
                                                                                                                            color=['blue', 'orange', 'green', 'red'],
                                                                                                                            alpha=0.9)

plt.xlabel("Date", fontsize=16)
plt.ylabel("Sales", fontsize=16)
plt.title("Comparison between Sales and different rolling windows", fontsize=16)
plt.legend(['Actual Sales', '10-Days Rolling Average', '30-Days Rolling Average', '50-Days Rolling Average'])
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


"""The .expanding() function in pandas is used to calculate cumulative statistics 
(such as cumulative sum, mean, etc.) across the data. Unlike rolling windows (like .rolling()), 
which operate on a fixed-size window, .expanding() operates on an increasing window size, starting 
from the first data point and expanding until the current one."""

df_train['Cumulative Moving Average'] = df_train['num_sold'].expanding().mean()
df_train['Cumulative Maximum'] = df_train['num_sold'].expanding().max()
df_train['Cumulative Minimum'] = df_train['num_sold'].expanding().min()

df_train[['Cumulative Moving Average','Cumulative Maximum', 'Cumulative Minimum']].plot(figsize=(12, 7),
                                                                                        ylim=(-1000, 8000),
                                                                                        color=['green', 'blue', 'red'],
                                                                                        linewidth=3,
                                                                                        alpha=0.9)

plt.xlabel("Date", fontsize=14)
plt.ylabel("Sales", fontsize=14)
plt.title("Cumulative Moving Average", fontsize=14)
plt.legend(['Cumulative Moving Average', 'cumulative Maximum', 'Cumulative Minimum'])
plt.grid(visible=True, linestyle="--", alpha=0.7)                                             
plt.show                                                                                          


"""As we can see above plot the green line which is Cumulative moving Average is not increasing as 
time passes whixh basically means the sales is nor increasing significantly"""


# EMA foe sales of Sticker
# Let's put smoothing factor alpha=0.1
df_train['EMA_0.1'] = df_train['num_sold'].ewm(alpha=0.1, adjust=False).mean() 
df_train[['num_sold', 'EMA_0.1']].plot(figsize=(12,6),
                                       linewidth=2,
                                       alpha=0.6)

plt.xlabel("Dates", fontsize=15)
plt.ylabel("Sales", fontsize=15)
plt.grid()
plt.show()


# EMA foe sales of Sticker
# Let's put smoothing factor alpha=0.6
df_train['EMA_0.6'] = df_train['num_sold'].ewm(alpha=0.6, adjust=False).mean() 
df_train[['num_sold', 'EMA_0.6']].plot(figsize=(12,6),
                                       color=['blue','red'],
                                       linewidth=2,
                                       alpha=0.6)

plt.xlabel("Dates", fontsize=15)
plt.ylabel("Sales", fontsize=15)
plt.grid()
plt.show()


# As we can see as we change the "alpha" that is smoothing factor the Exponetial Moving average also changes.


df_train['EMA_5Days'] = df_train['num_sold'].ewm(span=4).mean()
df_train[['num_sold', 'EMA_5Days']].plot(figsize=(12,6),
                                         color=['orange', 'green'])




