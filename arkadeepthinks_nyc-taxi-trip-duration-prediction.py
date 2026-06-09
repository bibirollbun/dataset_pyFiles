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


df=pd.read_csv("/kaggle/input/nyc-taxi-trip-duration/train.zip")
df.shape


print(f'Number of rows: {df.shape[0]}')
print(f'Number of columns: {df.shape[1]}')
df.iloc[1,:]


df.isnull().sum()


df.dtypes


# converting strings to datetime features
df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
df['dropoff_datetime'] = pd.to_datetime(df.dropoff_datetime)

# Converting yes/no flag to 1 and 0
df['store_and_fwd_flag'] = 1 * (df.store_and_fwd_flag.values == 'Y')
df['check_trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).map(lambda x: x.total_seconds())

duration_difference = df[np.abs(df['check_trip_duration'].values  - df['trip_duration'].values) > 1]
duration_difference.shape


df['trip_duration'].describe()/3600 # Trip duration in hours


import matplotlib.pyplot as plt
import seaborn as sns
df['log_trip_duration']=np.log(df['trip_duration'].values + 1)
sns.distplot(df['log_trip_duration'], kde =False, bins = 200)
plt.show()


features=['passenger_count','vendor_id','store_and_fwd_flag']
for feature in features:
    print(df[feature].value_counts())


df['pickup_datetime'].min(),df['pickup_datetime'].max()


df['day_of_week']=df['pickup_datetime'].dt.weekday
df['hour_of_day']=df['pickup_datetime'].dt.hour


import matplotlib.pyplot as plt
plt.figure(figsize=(22,8))

#Passenger Count
plt.subplot(121)
sns.countplot(x='day_of_week',data=df)
plt.xlabel('Week Day')
plt.ylabel('Total number of Pickups')

#vendor_id
plt.subplot(122)
sns.countplot(x='hour_of_day',data=df)
plt.xlabel("Hour of Day")
plt.ylabel("Total number of pickups")


sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2,2,figsize=(20,10), sharex=False, sharey = False)
sns.despine(left=True)
sns.distplot(df['pickup_latitude'].values, label = 'pickup_latitude',color="b",bins = 100, ax=axes[0,0])
sns.distplot(df['pickup_longitude'].values, label = 'pickup_longitude',color="r",bins =100, ax=axes[1,0])
sns.distplot(df['dropoff_latitude'].values, label = 'dropoff_latitude',color="b",bins =100, ax=axes[0,1])
sns.distplot(df['dropoff_longitude'].values, label = 'dropoff_longitude',color="r",bins =100, ax=axes[1,1])
plt.setp(axes, yticks=[])
plt.tight_layout()
plt.show()



df = df.loc[(df.pickup_latitude > 40.6) & (df.pickup_latitude < 40.9)]
df = df.loc[(df.dropoff_latitude>40.6) & (df.dropoff_latitude < 40.9)]
df = df.loc[(df.dropoff_longitude > -74.05) & (df.dropoff_longitude < -73.7)]
df = df.loc[(df.pickup_longitude > -74.05) & (df.pickup_longitude < -73.7)]
df_data_new = df.copy()
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2,2,figsize=(10, 10), sharex=False, sharey = False)#
sns.despine(left=True)
sns.distplot(df_data_new['pickup_latitude'].values, label = 'pickup_latitude',color="b",bins = 100, ax=axes[0,0])
sns.distplot(df_data_new['pickup_longitude'].values, label = 'pickup_longitude',color="r",bins =100, ax=axes[0,1])
sns.distplot(df_data_new['dropoff_latitude'].values, label = 'dropoff_latitude',color="b",bins =100, ax=axes[1, 0])
sns.distplot(df_data_new['dropoff_longitude'].values, label = 'dropoff_longitude',color="r",bins =100, ax=axes[1, 1])
plt.setp(axes, yticks=[])
plt.tight_layout()

plt.show()


df.columns


summary_wdays_avg_duration = pd.DataFrame(df.groupby(['day_of_week'])['trip_duration'].median())
summary_wdays_avg_duration.reset_index(inplace = True)
summary_wdays_avg_duration['unit']=1

sns.lineplot(summary_wdays_avg_duration, x="day_of_week", y="trip_duration")


summary_hourly_avg_duration = pd.DataFrame(df.groupby(['hour_of_day'])['trip_duration'].median())
summary_hourly_avg_duration.reset_index(inplace = True)
summary_hourly_avg_duration['unit']=1

sns.lineplot(data=summary_hourly_avg_duration, x="hour_of_day", y="trip_duration")


plt.figure(figsize=(22, 6))
sns.boxplot(x="vendor_id", y="trip_duration", data=df)
plt.show()


plt.figure(figsize=(22,6))
df_sub=df[df['trip_duration']<50000]
sns.boxplot(x='vendor_id',y='trip_duration',data=df_sub)
plt.show()


summary_wdays_avg_duration = pd.DataFrame(df.groupby(['vendor_id','day_of_week'])['trip_duration'].mean())
summary_wdays_avg_duration.reset_index(inplace = True)

sns.lineplot(summary_wdays_avg_duration,x='day_of_week',y='trip_duration',hue='vendor_id')


summary_wdays_avg_duration = pd.DataFrame(df.groupby(['vendor_id','day_of_week'])['trip_duration'].median())
summary_wdays_avg_duration.reset_index(inplace = True)
summary_wdays_avg_duration['unit']=1
sns.lineplot(summary_wdays_avg_duration, x='day_of_week', y='trip_duration', hue='vendor_id')


df.passenger_count.value_counts()


df.passenger_count.value_counts()
plt.figure(figsize=(22,6))
df_sub=df[df['trip_duration']<10000]
sns.boxplot(x="passenger_count",y="trip_duration",data=df_sub)
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

plt.figure(figsize=(12, 6))

# drop unwanted columns
df1 = df.drop(['id', 'pickup_datetime', 'dropoff_datetime',
              'passenger_count', 'check_trip_duration', 'log_trip_duration'],
             axis=1, errors='ignore')

# convert all to string before factorizing
corr = df1.apply(lambda x: pd.factorize(x.astype(str))[0]).corr()

# heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
            linewidths=.2, cmap="YlGnBu")
plt.title("Feature Correlation Heatmap")
plt.show()



%matplotlib inline
import numpy as np 
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor



df.columns


# converting strings to datetime features
df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
df['dropoff_datetime'] = pd.to_datetime(df.dropoff_datetime)


# Log transform the Y values
df_y = np.log1p(df['trip_duration'])

# Add some datetime features
df.loc[:, 'pickup_weekday'] = df['pickup_datetime'].dt.weekday
df.loc[:, 'pickup_hour_weekofyear'] = df['pickup_datetime'].dt.isocalendar().week
df.loc[:, 'pickup_hour'] = df['pickup_datetime'].dt.hour
df.loc[:, 'pickup_minute'] = df['pickup_datetime'].dt.minute
df.loc[:, 'pickup_dt'] = (df['pickup_datetime'] - df['pickup_datetime'].min()).dt.total_seconds()
df.loc[:, 'pickup_week_hour'] = df['pickup_weekday'] * 24 + df['pickup_hour']



#displacement
y_dist= df['pickup_longitude'] - df['dropoff_longitude']
x_dist = df['pickup_latitude'] - df['dropoff_latitude']

#square distance
df['dist_sq'] = (y_dist ** 2) + (x_dist ** 2)

#distance
df['dist_sqrt'] = df['dist_sq'] ** 0.5



def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def direction_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


df['haversine_distance'] = haversine_array(df['pickup_latitude'].values, 
                                                     df['pickup_longitude'].values, 
                                                     df['dropoff_latitude'].values, 
                                                     df['dropoff_longitude'].values)


df['direction'] = direction_array(df['pickup_latitude'].values, 
                                          df['pickup_longitude'].values, 
                                          df['dropoff_latitude'].values, 
                                          df['dropoff_longitude'].values)



df.columns


### Binned Coordinates ###
df['pickup_latitude_round3'] = np.round(df['pickup_latitude'],3)
df['pickup_longitude_round3'] = np.round(df['pickup_longitude'],3)

df['dropoff_latitude_round3'] = np.round(df['dropoff_latitude'],3)
df['dropoff_longitude_round3'] = np.round(df['dropoff_longitude'],3)


df.vendor_id.value_counts()


df['vendor_id']=df['vendor_id']-1


df.isnull().sum()


df=df.drop(['trip_duration','check_trip_duration','log_trip_duration','pickup_datetime','dropoff_datetime','store_and_fwd_flag'],axis=1)


df.drop('id', axis=1, inplace=True)


df.columns


df.head()


from sklearn.metrics import mean_squared_error as mse
from math import sqrt


from sklearn.model_selection import train_test_split as tts
xtrain,x_test,y_train,y_test=tts(df,df_y,test_size=1/3, random_state=0)


mean_pred = np.repeat(y_train.mean(),len(y_test))

sqrt(mse(y_test, mean_pred))


from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error
from math import sqrt
import numpy as np

# model and KFold setup
model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=11)

# Define RMSE scorer (since scikit-learn doesn't have RMSE directly)
rmse_scorer = make_scorer(mean_squared_error, squared=False)

# Perform cross-validation
lr_scores = cross_val_score(model, X=df, y=df_y, cv=kf, scoring=rmse_scorer)

print("RMSE for each fold:", lr_scores)
print("Average RMSE:", np.mean(lr_scores))



from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from math import sqrt
import numpy as np

# Define model
model = DecisionTreeRegressor(min_samples_leaf=25, min_samples_split=25, random_state=42)

# Define KFold
kf = KFold(n_splits=5, shuffle=True, random_state=11)

# Define RMSE scorer
rmse_scorer = make_scorer(mean_squared_error, squared=False)

# Perform cross-validation
dt_scores = cross_val_score(model, X=df, y=df_y, cv=kf, scoring=rmse_scorer)

print("RMSE for each fold:", dt_scores)
print("Average RMSE:", np.mean(dt_scores))



 results_df=pd.DataFrame({'linear_regression_scores':lr_scores,'dt_scores':dt_scores})


results_df.plot(y=['linear_regression_scores','dt_scores'], kind="bar",legend=False)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np

# Define parameters
rf_params = {
    'random_state': 0,
    'n_estimators': 8,
    'max_depth': 6,
    'n_jobs': -1,
    'min_samples_split': 43
}

# Initialize model
rf_model = RandomForestRegressor(**rf_params)

# Define KFold
kf = KFold(n_splits=5, shuffle=True, random_state=11)

# Define RMSE scorer
rmse_scorer = make_scorer(mean_squared_error, squared=False)

# Perform cross-validation
rf_scores = cross_val_score(rf_model, X=df, y=df_y, scoring=rmse_scorer,cv=kf)

# Print results
print("RMSE for each fold:", rf_scores)
print("Average RMSE:", np.mean(rf_scores))



#Splitting the data into df and Validation set
xtrain, xtest, ytrain, ytest = train_test_split(df,df_y,test_size=1/3, random_state=0)


dtrain=xgb.DMatrix(xtrain,label=ytrain)
dvalid=xgb.DMatrix(xtest,label=ytest)

watchlist=[(dtrain,'train'),(dvalid,'valid')]


xgb_params = {}
xgb_params["objective"] = "reg:linear"
xgb_params['eval_metric'] = "rmse"
xgb_params["eta"] = 0.05
xgb_params["min_child_weight"] = 10
xgb_params["subsample"] = 0.9
xgb_params["colsample_bytree"] = 0.7
xgb_params["max_depth"] = 5
xgb_params['silent'] = 1
xgb_params["seed"] = 2019
xgb_params["nthread"] = -1
xgb_params["lambda"] = 2

xgb_model = xgb.train(xgb_params, dtrain, 10000, watchlist, early_stopping_rounds=50,
      maximize=False, verbose_eval=20)
print('Modeling RMSE %.5f' % xgb_model.best_score)


 xgb.plot_importance(xgb_model, max_num_features=28, height=0.7)


xgb_params['n_estimators'] = xgb_model.best_iteration
xgb_model_final = xgb.XGBRegressor()
kf = KFold(n_splits=5, shuffle=True, random_state=11)
rmse_scorer = make_scorer(mean_squared_error, squared=False)
xgb_scores = cross_val_score(xgb_model_final, X=df, y=df_y, cv=kf, scoring=rmse_scorer)
print("RMSE for each fold:", xgb_scores)
print("Average RMSE:", np.mean(xgb_scores))
print("Standard Deviation:", np.std(xgb_scores))


results_final=pd.DataFrame({'LR':lr_scores,'DT':dt_scores,"RF":rf_scores,"XGB":xgb_scores})
results_final.plot(y=['LR','DT','RF','XGB'],kind="bar")
plt.title("Comparison of RMSE scores of each model")




