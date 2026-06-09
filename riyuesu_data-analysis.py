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

train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")

# 基本情報
print(train.info())
print(train.describe())

# 欠損値確認
print(train.isnull().sum())



import matplotlib.pyplot as plt

train['datetime'] = pd.to_datetime(train['datetime'])
train.set_index('datetime', inplace=True)

# 1日単位の利用者数推移
train['count'].resample('D').sum().plot(figsize=(15,5), title='Daily count')
plt.show()

# 1時間単位の利用者数推移（サンプル期間）
train['count'].iloc[:500].plot(title='Hourly count (sample)')
plt.show()



# 曜日別平均利用者数
train.groupby(train.index.weekday)['count'].mean().plot(kind='bar', title='Average count by weekday')
plt.show()

# 時間別平均利用者数
train.groupby(train.index.hour)['count'].mean().plot(kind='bar', title='Average count by hour')
plt.show()

# 季節別平均利用者数
train.groupby('season')['count'].mean().plot(kind='bar', title='Average count by season')
plt.show()

# 祝日・平日別平均利用者数
train.groupby('holiday')['count'].mean().plot(kind='bar', title='Average count by holiday')
plt.show()



import seaborn as sns

numeric_cols = ['temp','atemp','humidity','windspeed','count']
sns.pairplot(train[numeric_cols])
plt.show()

# 相関行列
corr = train[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()



train['count'].hist(bins=50)
plt.title('Count distribution')
plt.show()

# log変換後
import numpy as np
np.log1p(train['count']).hist(bins=50)
plt.title('Log1p count distribution')
plt.show()



# データをロードし直すか、事前にcasualとregisteredが利用可能な状態にしておく必要があります
# ここでは、元のtrainデータにcasualとregistered列が存在することを前提とします。

# 1-A. 時間別（Hour）平均利用者数：利用者タイプ別
plt.figure(figsize=(10, 5))
train.groupby(train.index.hour)['registered'].mean().plot(kind='bar', label='Registered', color='skyblue')
train.groupby(train.index.hour)['casual'].mean().plot(kind='bar', label='Casual', color='salmon', alpha=0.7)
plt.title('Average Count by Hour (Split by User Type)')
plt.legend()
plt.xticks(rotation=0)
plt.show()

# 1-B. 曜日別（Weekday）平均利用者数：利用者タイプ別
plt.figure(figsize=(10, 5))
train.groupby(train.index.weekday)['registered'].mean().plot(kind='bar', label='Registered', color='skyblue')
train.groupby(train.index.weekday)['casual'].mean().plot(kind='bar', label='Casual', color='salmon', alpha=0.7)
plt.title('Average Count by Weekday (Split by User Type)')
plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
plt.legend()
plt.show()

# 1-C. 勤務日 (Workingday) 別平均利用者数：利用者タイプ別
plt.figure(figsize=(6, 5))
train.groupby('workingday')[['registered', 'casual']].mean().plot(kind='bar', ax=plt.gca())
plt.title('Average Count by Workingday (Split by User Type)')
plt.xticks(ticks=[0, 1], labels=['Holiday/Weekend', 'Workingday'], rotation=0)
plt.show()


# 2-A. 気温（Temp）と利用者数の散布図：利用者タイプ別
plt.figure(figsize=(12, 6))
# Registered
plt.subplot(1, 2, 1)
sns.scatterplot(x='temp', y='registered', data=train, alpha=0.5)
plt.title('Registered vs Temp')
# Casual
plt.subplot(1, 2, 2)
sns.scatterplot(x='temp', y='casual', data=train, alpha=0.5)
plt.title('Casual vs Temp')
plt.show()

# 2-B. 天候（Weather）別平均利用者数：利用者タイプ別
plt.figure(figsize=(10, 5))
weather_map = {1: 'Good', 2: 'Mist/Cloud', 3: 'Light Rain/Snow', 4: 'Heavy Rain/Snow'}
train_weather_mean = train.groupby('weather')[['registered', 'casual']].mean()
train_weather_mean.plot(kind='bar', ax=plt.gca())
plt.title('Average Count by Weather (Split by User Type)')
plt.xticks(ticks=train_weather_mean.index, labels=[weather_map[i] for i in train_weather_mean.index], rotation=45)
plt.show()

