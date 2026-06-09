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

train = pd.read_csv('/kaggle/input/tabular-playground-series-sep-2022/train.csv')
train.describe()


train.columns





train['row_id'].value_counts()


train['row_id'].value_counts()


import pandas as pd

# データ読み込み（ファイルパスは適宜変更）
train = pd.read_csv('/kaggle/input/tabular-playground-series-sep-2022/train.csv')


train['row_id'].value_counts()


train['date'].value_counts()


train.groupby('date')['num_sold'].mean()





train.pivot_table('num_sold', index = 'date', columns = 'country')


train['date'] = pd.to_datetime(train['date'])
train['dayofweek'] = train['date'].dt.dayofweek  # 月:0〜日:6
train['day_type'] = pd.cut(train['dayofweek'], [0, 4, 6], labels=['Weekday', 'Weekend'])





train['day_type'] = train['dayofweek'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')


train = pd.get_dummies(train, columns=['country', 'store', 'product'])


train.columns





train.isnull().sum().sum()





train.groupby('dayofweek')['num_sold'].mean().plot(kind='bar', title='Average Sales by Day of Week')





import pandas as pd

train = pd.read_csv('/kaggle/input/tabular-playground-series-sep-2022/train.csv')
train.describe()


##　カラムを表示する





train.columns





train.pivot_table('num_sold', index = 'date', columns = 'country')





train['date'] = pd.to_datetime(train['date'])
train['dayofweek'] = train['date'].dt.dayofweek
train['day_type'] = train['dayofweek'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')





train.groupby('dayofweek')['num_sold'].mean().plot(kind='bar', title='Average Sales by Day of Week')




