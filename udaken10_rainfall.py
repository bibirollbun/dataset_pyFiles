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
import datetime
from datetime import datetime
import numpy as np


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')
sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

train['expected_day'] = (train.index.values) % 365 + 1

train['day'] = train['expected_day']
train.drop('expected_day', axis=1, inplace=True)

train_1 = train[0:365]
train_2 = train[365:730]
train_3 = train[730:1095]
train_4 = train[1095:1460]
train_5 = train[1460:1825]
train_6 = train[1825:2190]

train_1['day_of_year'] = train_1['day']
train_1

from datetime import datetime

train_1['date'] = train_1['day_of_year'].apply(lambda x: datetime.strptime(str(x), '%j'))

train_2['day_of_year'] = train_2['day']
train_3['day_of_year'] = train_3['day']
train_4['day_of_year'] = train_4['day']
train_5['day_of_year'] = train_5['day']
train_6['day_of_year'] = train_6['day']
train_2['date'] = train_2['day_of_year'].apply(lambda x: datetime.strptime(str(x), '%j'))
train_3['date'] = train_3['day_of_year'].apply(lambda x: datetime.strptime(str(x), '%j'))
train_4['date'] = train_4['day_of_year'].apply(lambda x: datetime.strptime(str(x), '%j'))
train_5['date'] = train_5['day_of_year'].apply(lambda x: datetime.strptime(str(x), '%j'))
train_6['date'] = train_6['day_of_year'].apply(lambda x: datetime.strptime(str(x), '%j'))

train_2['date'] = train_2['date'].apply(lambda x: x.replace(year=x.year + 1))
train_3['date'] = train_3['date'].apply(lambda x: x.replace(year=x.year + 2))
train_4['date'] = train_4['date'].apply(lambda x: x.replace(year=x.year + 3))
train_5['date'] = train_5['date'].apply(lambda x: x.replace(year=x.year + 4))
train_6['date'] = train_6['date'].apply(lambda x: x.replace(year=x.year + 5))

train_data = pd.concat([train_1, train_2, train_3, train_4, train_5, train_6])
train_data


train_data.set_index('date', inplace=True)
train_data.index = pd.to_datetime(train_data.index)
train_data.head()

target = train_data.rainfall


test_1 = test[0:365]
test_2 = test[365:730]

test_1['date'] = test_1['day'].apply(lambda x: datetime.strptime(str(x), '%j'))
test_2['date'] = test_2['day'].apply(lambda x: datetime.strptime(str(x), '%j'))

test_1['date'] = test_1['date'].apply(lambda x: x.replace(year=x.year + 6))
test_2['date'] = test_2['date'].apply(lambda x: x.replace(year=x.year + 7))

test_data = pd.concat([test_1, test_2])

test_data.set_index('date', inplace=True)

test_data.index = pd.to_datetime(test_data.index)

test_data['day_of_year'] = test_data['day']
test_data


train_data['date'] = train_data.index
test_data['date'] = test_data.index


train_data['month'] = train_data['date'].dt.month
test_data['month'] = test_data['date'].dt.month


train_data['month'].astype(int)
test_data['month'].astype(int)


# seasonというカラムを作るmonthが１２，1,2の時は、１とし、３，４の時は２とし５、６，７，８，９の時は３とし、１０、１１の時は４とする

def assign_season(month):
  if month in [12, 1, 2]:
    return 1
  elif month in [3, 4]:
    return 2
  elif month in [5, 6, 7, 8, 9]:
    return 3
  elif month in [10, 11]:
    return 4
  else:
    return np.nan # handle cases outside the defined months

train_data['season'] = train_data['month'].apply(assign_season)
test_data['season'] = test_data['month'].apply(assign_season)


def feature_cook(df):
    df['cloud*windspeed / pressure'] = df['cloud']*df['windspeed'] / df['pressure']
    # df['sin(winddiretion)*windspeed'] = np.sin(df['winddirection'])*df['windspeed']
    df['humidity*(mintemp-dewpoint)'] = df['humidity']*(df['mintemp'] - df['dewpoint'])
    df['humidity*cloud'] = df['humidity']*df['cloud']
    df['1/(sunshine**2)'] = 1/(df['sunshine']**2 + 0.001)
    df['cloud**2'] = df['cloud']**2
    df['cloud-sunshine'] = df['cloud'] - df['sunshine']
    df['winddirection_bin'] = np.sin(2*np.pi*pd.cut(df['winddirection'], bins=8, labels=False))
    df['sin(winddiretion)*windspeed'] = np.sin(2*np.pi*(df['winddirection_bin']))*df['windspeed']

    df['temp_diff']=df['maxtemp']-df['mintemp']
    
    #気圧の変化
    df['pressure_change'] = df['pressure'].diff()

    #湿度の変化
    # df['humidity_change'] = df['humidity'].diff()

    #雲量の変化
    # df['cloud_change'] = df['cloud'].diff()
    # df['month']=df['day']//31
    df['sin_day']=np.sin(2*np.pi*df['day']/365)
    df['cos_day']=np.cos(2*np.pi*df['day']/365)

    for c in [
        #'pressure', 
        'maxtemp', 
        #'temparature', 'mintemp',
        'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection','windspeed']:
        for gap in [1]:
            df[c+f"_shift{gap}"]=df[c].shift(gap)
            df[c+f"_diff{gap}"]=df[c].diff(gap)

    df.fillna(method='bfill', inplace=True) #NaNを後ろの値で埋める
    df.fillna(method='ffill', inplace=True) #NaNを前の値で埋める
    return df


feature_cook(train_data)
feature_cook(test_data)


# rainfallごとにmonthをかぞえ、上位4つに''rain_season' ＝１というフラグを立てます

# Group by month and count occurrences of rainfall
monthly_rainfall = train_data.groupby('month')['rainfall'].sum()

# Get the top 4 months with the highest rainfall
top_4_months = monthly_rainfall.nlargest(4).index

# Create the 'rain_season' flag
train_data['rain_season'] = 0  # Initialize the flag to 0 for all rows
test_data['rain_season'] = 0
train_data.loc[train_data['month'].isin(top_4_months), 'rain_season'] = 1
test_data.loc[test_data['month'].isin(top_4_months), 'rain_season'] = 1


train_data.columns


col_to_drop = ['day', 'date',	'pressure',	
               #'maxtemp',	
               'temparature',	'mintemp',	'dewpoint',	'humidity',	'cloud',	'sunshine',	'winddirection',	'windspeed',	'day_of_year']


train_df = train_data.drop(col_to_drop, axis=1)
test_df = test_data.drop(col_to_drop, axis=1)


train_df.columns



from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split


X = train_df.drop(columns=['rainfall'])
y = train_df['rainfall']
X_test = test_df


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Reshape Input for CNN (adding a channel dimension)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))


model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-5, verbose=1)


history = model.fit(
    X_train, y_train,
    epochs=500, batch_size=32, validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr], verbose=1
)


test_preds = model.predict(X_test_scaled).flatten()

test_preds


sub['rainfall'] = test_preds
sub.to_csv('submission.csv',index = False)

sub




