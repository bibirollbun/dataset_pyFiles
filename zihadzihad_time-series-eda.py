## Install Pandas Data Reader
!pip install pandas-datareader


!pip install yfinance
!pip install fix-yahoo-finance


from pandas_datareader import data as pdr
import pandas as pd
from datetime import datetime
import yfinance as yf



import yfinance as yf

tesla = yf.Ticker("NVDA")
tesla_data = tesla.history(period='5y', interval='1d')
tesla_data


type(tesla_data)


tesla_data.tail()


df_tesla = tesla_data.copy()
df_tesla.plot()


df_tesla['High'].plot()


df_tesla['Low'].plot()


df_tesla[['Open','Close']].plot(figsize=(12,4))


## Xlimit and y limit
df_tesla['High'].plot(xlim=['2020-01-01', '2021-09-01'], figsize=(12,4))


df_tesla['High'].plot(xlim=['2020-01-01', '2021-09-01'], ylim=[0,300], figsize=(12,4))


## lets apply some color
df_tesla['High'].plot(xlim=['2020-01-01', '2021-09-01'], ylim=[0,300], figsize=(12,4),c='green')


df_tesla.index


print(df_tesla.loc['2021-01-01':'2022-09-01'].index)
index = df_tesla.loc['2021-01-01':'2022-09-01'].index
print(df_tesla.loc['2021-01-01':'2022-09-01']['Open'])
share_open = df_tesla.loc['2021-01-01':'2022-09-01']['Open']


import matplotlib.pyplot as plt
%matplotlib inline


figure,axis = plt.subplots()
plt.tight_layout()
figure.autofmt_xdate()
axis.plot(index,share_open)


df_tesla.info()


## Datetime index
df_tesla.reset_index().info()


df_tesla = df_tesla.reset_index()
pd.to_datetime(df_tesla['Date'])


df_tesla.set_index('Date',drop=True)


df_tesla


from datetime import datetime
datetime.now()


def add_num(num1, num2):
    return num1 + num2


num1=20
num2=10
start_time=datetime.now()
add_num(num1,num2)
end_time=datetime.now()
print(end_time-start_time)


date = datetime.now()
date.weekday()


sticker_sale = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')


sticker_sale.head()


import pandas as pd


sticker_sale['date'] = pd.to_datetime(sticker_sale['date'])


sticker_sale = sticker_sale.set_index('date',drop=True)


sticker_sale.info()


sticker_sale.resample(rule='A').min()


sticker_sale.resample(rule='A').max()['num_sold'].plot()


sticker_sale['num_sold'].resample(rule='BA').mean().plot(kind='bar')


sticker_sale['num_sold'].rolling(10).mean()

