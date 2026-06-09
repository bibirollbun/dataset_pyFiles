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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


dataset_path = '/kaggle/input/playground-series-s5e10/train.csv'

df = pd.read_csv(dataset_path)

df.head()


#shape of dataset
df.shape


df.columns.to_list()


#replace columns that have true and false as categories to numirique values
df.replace({True:1, False:0}, inplace=True)


#
df.head()


#descovering the unique values in the categorical columns 
print("road_type:", df.road_type.unique(),
"\nlighting:", df.lighting.unique(),
"\nweather:", df.weather.unique(),
"\ntime_of_day:",df.time_of_day.unique())


road_type=['urban', 'rural','highway'] 
lighting=['daylight' ,'dim', 'night'] 
weather=['rainy', 'clear', 'foggy'] 
time_of_day=['afternoon', 'evening', 'morning']


#df["road_type"] = df["road_type"].replace({'urban':0, 'rural':1 , 'highway':2})


def convert_columns(data):
    df = data
    columns1 = [
    road_type,
    lighting,
    weather,
    time_of_day]
    columns2 = [
    'road_type',
    'lighting',
    'weather',
    'time_of_day']
    for c, co in zip(columns1, columns2):
        
        df[co] = df[co].replace({c[0]:0, c[1]:1 , c[2]:2})
    


convert_columns(df)


df.head()


#rechicking unique values
print("road_type:", df.road_type.unique(),
"\nlighting:", df.lighting.unique(),
"\nweather:", df.weather.unique(),
"\ntime_of_day:",df.time_of_day.unique())


#the dataset contians some unnecessery columns like id, it will be better to drop it first

df.drop(['id', 'num_reported_accidents'], axis=1, inplace = True)


df.head()


df.describe()








#now let's specifie our trainset and targetset; it's clear that accident_rsik 
#column is the target one and the other will be used for training
X = df[df.columns.to_list()[:-1]]

y = df.accident_risk 



X.columns


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size = 20, random_state = 42)


# from sklearn.linear_model import LinearRegression

# model = LinearRegression()

from sklearn.ensemble import RandomForestRegressor

model  = RandomForestRegressor(n_estimators=100, criterion='absolute_error')

model.fit(X_train, Y_train)


y_pred = model.predict(X_test)


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score

print("mse = ", mean_squared_error(Y_test, y_pred))
print("/n rmse = ", np.sqrt(mes))
print("/n mae = ", mean_absolute_error(Y_test, y_pred))
print("/n r2_s = ", r2_score(Y_test, y_pred))




y_pred, Y_test


df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


df_test = df_test[df_test.columns.to_list()[:-1]]


df_test.head()


df_test.replace({False:0, True:1}, inplace = True)


df_test['road_type']


convert_columns(df_test)





# df_test.head()


df_pre = df_test.iloc[:, 1:]


scaler = StandardScaler()
df_pre = scaler.fit_transform(df_pre)


predictions = model.predict(df_pre)


predictions


df = pd.DataFrame({
                'id':df_test['id'],
                'accident_risk':predictions})


df.head()


df.to_csv('submission.csv', index = False)




