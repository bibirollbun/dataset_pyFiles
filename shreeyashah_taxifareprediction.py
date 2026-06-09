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
import matplotlib.pyplot as plt
import seaborn as sns

import warnings 
warnings.filterwarnings("ignore")


df_train =  pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv', nrows = 2_000_000)
df_train.head()


df_train = df_train.iloc[:,1:]


df_train.info()


df_train.shape


df_train.describe()


df_train.isna().sum()


df_train = df_train.dropna()


#sns.boxplot(df['fare_amount'])


df_train = df_train[(df_train['fare_amount'] > 1) & (df_train['fare_amount'] < 100)]


df_train = df_train[(df_train['passenger_count'] >= 1) & (df_train['passenger_count'] <= 6)]


df_train = df_train[
    (df_train['pickup_longitude'] > -75) & (df_train['pickup_longitude'] < -72) &
    (df_train['dropoff_longitude'] > -75) & (df_train['dropoff_longitude'] < -72) &
    (df_train['pickup_latitude'] > 40) & (df_train['pickup_latitude'] < 42) &
    (df_train['dropoff_latitude'] > 40) & (df_train['dropoff_latitude'] < 42)
]


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # kilometers


df_train['distance_km'] = haversine(
    df_train['pickup_latitude'], df_train['pickup_longitude'],
    df_train['dropoff_latitude'], df_train['dropoff_longitude']
)

df_train = df_train[(df_train['distance_km'] > 0.1) & (df_train['distance_km'] < 30)]


#sns.boxplot(df['fare_amount'])


#sns.distplot(df['fare_amount'], kde=True, bins=30)


X= df_train.iloc[:,1:]
y = df_train.iloc[:,0]


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100,max_depth =10 ,random_state=2,bootstrap=True, max_samples=5000,n_jobs=1)
#rf.fit(X.iloc[:,1:],y)
#cv = cross_val_score(rf, X.iloc[:,1:],y,cv=10)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
#br.fit(X.iloc[:,1:],y)



from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
sr = SVR(kernel = 'rbf')
br = BaggingRegressor(n_estimators = 10, estimator=sr, max_samples=5000, bootstrap=True )
#br.fit(X.iloc[:,1:],y)


from sklearn.ensemble import VotingRegressor
vr = VotingRegressor([('lr', lr), ('rf', rf),('svr',br)], verbose=True, n_jobs=-1)
vr.fit(X.iloc[:,1:],y)


df_test = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv',parse_dates=["pickup_datetime"])
df_test.head()


key = df_test['key']
df_test = df_test.iloc[:,2:]


df_test['distance_km'] = haversine(
    df_test['pickup_latitude'], df_test['pickup_longitude'],
    df_test['dropoff_latitude'], df_test['dropoff_longitude']
)


y_pred = vr.predict(df_test)


results = pd.DataFrame({'key':key,'fare_amount':y_pred})
print(results)


results.to_csv('submission.csv', index=False)

