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


train=pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
train.head()


train.info()



train.head()


train.isnull().sum()


pip install autoviz


from autoviz import data_cleaning_suggestions


data_cleaning_suggestions(train)


train = train.dropna(subset=['num_sold'])


train.info()


test=pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
test.head()


#data_cleaning_suggestions(test)


import datetime as dt


train["date"]= pd.to_datetime(train["date"])
test["date"]= pd.to_datetime(test["date"])


train.info()
test.info()


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.lineplot(data=train.groupby('date')['num_sold'].sum().reset_index(), x='date', y='num_sold')
plt.title('Total Products Sold Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Products Sold')

plt.show()


plt.figure(figsize=(12, 6))
train.groupby('country')['num_sold'].sum().plot(kind='bar')
plt.title('Total Products Sold by Country')
plt.xlabel('Country')
plt.ylabel('Number of Products Sold')
plt.grid()
plt.show()



plt.figure(figsize=(12, 6))
train.groupby('store')['num_sold'].sum().plot(kind='bar')
plt.title('Total Products Sold by Store Type')
plt.xlabel('Store Type')
plt.ylabel('Number of Products Sold')
plt.grid()
plt.show()


train['date'] = pd.to_datetime(train['date'], errors='coerce')
# Seasonal trends (e.g., monthly sales) visualization
train['month'] = train['date'].dt.month
train['month'] = train['date'].dt.month
plt.figure(figsize=(12, 6))
train.groupby('month')['num_sold'].sum().plot()
plt.title('Seasonal Trends: Products Sold by Month')
plt.xlabel('Month')
plt.ylabel('Number of Products Sold')
plt.grid()
plt.show()



plt.figure(figsize=(12, 6))
(train.groupby(['date', 'store'])['num_sold'].sum().groupby('store').mean()).plot(kind='bar')
plt.title('Average Daily Sales per Store Type')
plt.xlabel('Store Type')
plt.ylabel('Average Daily Products Sold')
plt.grid()
plt.show()



X=train.drop("num_sold",axis=1)
y= train["num_sold"]


final=pd.concat([X,test],axis=0)
final.info()


final=final.drop("month",axis=1)


final.info()


final['date'] = pd.to_datetime(final['date'], errors='coerce')


final["year"] = final['date'].dt.year
final["month"]= final["date"].dt.month
final["day"]=final["date"].dt.day


final.head()






final.head()


produ= pd.get_dummies(final["product"])
produ=produ.astype(int)
produ.head()



final.info()


new_final=pd.concat([final,produ],axis=1)


new_final.head()


new_final.country.unique()


country=pd.get_dummies(new_final.country)
country=country.astype(int)
country.head()


final=pd.concat([new_final,country],axis=1)


final.head()


final.store.unique()


store=pd.get_dummies(final.store)
store=store.astype(int)
store.head()


final=pd.concat([final,store],axis=1)


final.head()


final.drop(columns=["date","country","store","product"],axis=1,inplace=True)
final.head()


from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(final)



from sklearn.model_selection import train_test_split
X_tarin,y_train,X_test,y_test= train_test_split(final,y,test_size=0.2,random_state=42)




