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


df_cal = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv")
df_inv = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv")
df_train = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv")
df_test = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv")
df_sub = pd.read_csv("/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv")


df_train.tail()


len(df_train["unique_id"].unique())


df_train["warehouse"].unique()


print(df_train["unique_id"].min(),df_train["unique_id"].max(),df_test["unique_id"].min(),df_test["unique_id"].max())


# predict 2024-06-03 to 2024-06-16

df_sub[lambda df: df['id'].str[0:2] == "2_"]


import datetime

df_sub['unique_id']=df_sub.apply(lambda row: row['id'].split("_")[0], axis=1) 
df_sub['date']=df_sub.apply(lambda row: row['id'].split("_")[1], axis=1) 
df_sub['dayofweek']=df_sub.apply(lambda row: datetime.datetime.strptime(row['date'], '%Y-%m-%d').weekday(), axis=1) 
df_train['dayofweek']=df_train.apply(lambda row: datetime.datetime.strptime(row['date'], '%Y-%m-%d').weekday(), axis=1) 
df_sub.head()


df_train["date"].max()


df_train[df_train["date"]=="2024-06-02"][['unique_id','sales']]


df_train[df_train["date"]>='2024-05-01'].groupby(['unique_id','dayofweek'])['sales'].mean()
#df_train[df_train["date"]>='2024-05-01'].groupby(['unique_id','dayofweek'])['sales'].mean().to_dict()
#{(1, 0): 186.6775,
# (1, 1): 135.93,
# (1, 2): 128.68,
 #(1, 3): 193.038,
 #(1, 4): 146.166,
 #(1, 5): 113.704,


# use mean last month of sales by day of week
forecast = df_train[df_train["date"]>='2024-05-01'].groupby(['unique_id','dayofweek'])['sales'].mean().to_dict()
df_sub["sales_hat"].astype(float)
for i in df_sub.index:
    if (int(df_sub["unique_id"].loc[i]),df_sub["dayofweek"].loc[i]) in forecast.keys():
        df_sub["sales_hat"].at[i] = forecast[(int(df_sub["unique_id"].loc[i]),df_sub["dayofweek"].loc[i])]
    else:
        df_sub["sales_hat"].at[i] = 0.0
df_sub.tail()


import matplotlib.pyplot as plt
f = df_train[(df_train["unique_id"]==4572) & (df_train["date"]>='2024-05-01')][['date','sales']].sort_values(by=['date'])
x = np.array(f["date"])
y = np.array(f["sales"])
fig, axs = plt.subplots(1, 1, figsize=(12, 4.7), layout='constrained')
axs.plot(x,y,marker='x')
plt.show()


df_sub[["id","sales_hat"]].to_csv("/kaggle/working/submission.csv", index=False)

