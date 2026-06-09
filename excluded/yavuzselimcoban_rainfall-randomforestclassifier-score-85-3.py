# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
df_train.drop(['id','day'],axis =1,inplace = True)
df_test = df_test.dropna()
day_id = df_test['id']
df_test.drop(['id','day'],axis =1,inplace=True)
df_train.head(3)


# by converting the pressure to cmHG, I aim to reduce the possibility of incorrect coefficients.

df_train['cmHg'] = df_train['pressure']*75006156130264*10**-17
df_train.drop('pressure',axis = 1 ,inplace=True)
df_test['cmHg'] = df_test['pressure']*75006156130264*10**-17
df_test.drop('pressure',axis = 1 ,inplace=True)
df_test.head(3)


df_train['fark'] = df_train['maxtemp'] - df_train['mintemp'] 
df_train['fark2'] = df_train['dewpoint'] - df_train['fark']
df_test['fark'] = df_test['maxtemp'] - df_test['mintemp'] 
df_test['fark2'] = df_test['dewpoint'] - df_test['fark']
df_train.head(3)


df_train = df_train[df_train['temparature'] <= df_train['maxtemp']]
df_train = df_train[df_train['temparature'] >= df_train['mintemp']]
df_train.head(3)


plt.plot(df_train.index,df_train['fark'])
plt.show()


# Does not seem to be a directly effective featur
df_train = df_train[df_train['fark'] < 9]


plt.plot(df_train.index,df_train['fark2'])
plt.show()


df_train = df_train[df_train['fark2'] > -5]


plt.plot(df_train.index,df_train['windspeed'])
plt.show()


plt.plot(df_train.index,df_train['winddirection'])
plt.show()


# it matches with 'fark'


plt.plot(df_train.index,df_train['sunshine'])
plt.show()


plt.plot(df_train.index,df_train['humidity'])
plt.show()


df_train = df_train[df_train['humidity'] > 50]


y= df_train['rainfall']
x= df_train.drop('rainfall',axis = 1)
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 42 ,train_size = 0.8)
rfc = RandomForestClassifier()
model = rfc.fit(x_train,y_train)
model.score(x_test,y_test)


data_predict = pd.DataFrame()
a = model.predict(df_test)
data_predict['id'] = day_id
data_predict['predict'] = a
data_predict




