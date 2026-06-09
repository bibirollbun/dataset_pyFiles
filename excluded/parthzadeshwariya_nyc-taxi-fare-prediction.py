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


%config Completer.use_jedi = False


!wc -l /kaggle/input/new-york-city-taxi-fare-prediction/train.csv


!head /kaggle/input/new-york-city-taxi-fare-prediction/train.csv


selected_cols = 'fare_amount,pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count'.split(',')
selected_cols


dtypes = {
     'fare_amount': 'float32',
     'pickup_datetime': 'float32',
     'pickup_longitude': 'float32',
     'pickup_latitude': 'float32',
     'dropoff_longitude': 'float32',
     'dropoff_latitude': 'float32',
     'passenger_count': 'uint8'
}


import random
import pandas as pd
import numpy as np
random.seed(42)


sample_frac = 0.01
def skip_row(row_idx):
    if row_idx==0:
        return False
    return random.random() > sample_frac # as the random chooses random number uniformly, there is exactly 1% chance that random.random() will give less than sample_frac


df = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv', parse_dates=['pickup_datetime'], usecols=selected_cols, dtype=dtypes, skiprows=skip_row)


df.head()


df.isna().sum()


test_df = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv',parse_dates=['pickup_datetime'] ,dtype=dtypes)
test_df.head()


df.info()


df.describe()
# pay attention to min,max values of lat and long
# pay attention to max values of passenger counts


df['pickup_datetime'].min(), df['pickup_datetime'].max()


test_df.describe()
# what we can do is train our model on the data that lies in the range of test data


# TODO: EDA and graphs
# ask questions:
'''
1. what is the busiest day of the week?
2. what is the busiest time of the day?
3. in which month are fares the highest?
4. which pickup locations have the highest fares?
5. which drop locations have the highest dares?
6. what is the average ride distance
'''


from sklearn.model_selection import train_test_split


train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
len(train_df), len(val_df)


print(train_df.isna().sum())
print(val_df.isna().sum())
# no null values


train_df.columns


input_cols = [ 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
target_col = 'fare_amount'


train_inputs = train_df[input_cols]
train_targets = train_df[target_col]


val_inputs = val_df[input_cols]
val_targets = val_df[target_col]


test_inputs = test_df[input_cols]


# training hardcoded or baseline models


class MeanRegressor:
    def fit(self, inputs, targets):
        self.mean = targets.mean()

    def predict(self, inputs):
        return np.full(inputs.shape[0], self.mean) # returns np.arr of length inputs.shape[0] and with all the values equal to self.mean


mean_model = MeanRegressor()


mean_model.fit(train_inputs, train_targets)


train_pred = mean_model.predict(train_inputs)


val_pred = mean_model.predict(val_inputs)


from sklearn.metrics import mean_squared_error


def rmse(targets, preds):
    return mean_squared_error(targets, preds, squared=False)


train_rmse = rmse(train_targets, train_pred)
train_rmse


val_rmse = rmse(val_targets, val_pred)
val_rmse


from sklearn.linear_model import LinearRegression


linear_model = LinearRegression()


linear_model.fit(train_inputs, train_targets)


train_preds = linear_model.predict(train_inputs)
train_preds


rmse(train_targets, train_preds) # no meaning of training this type of models as dumb hardcoded models have the same amount of error


val_preds = linear_model.predict(val_inputs)
rmse(val_targets, val_preds)
# this type of model is useless, it is may be due to poor feature engineering, as the lat and long make no sense to model and also the pickup date and time, duration of the ride also matters


# for kaggle competitions submit your models everyday in order to check the score is improving or not
# make a function to create submission file as we will be submitting various files


def predict_submit(model, test_inputs, fname):
    test_preds = model.predict(test_inputs)
    sub_df = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv')
    sub_df['fare_amount'] = test_preds
    sub_df.to_csv(fname, index=None)
    return sub_df



# predict_submit(linear_model, 'linear_model.csv')


def add_dateparts(df, col):
    df[col + '_year'] = df[col].dt.year
    df[col + '_month'] = df[col].dt.month
    df[col + '_day'] = df[col].dt.day
    df[col + '_weekday'] = df[col].dt.weekday
    df[col + '_hour'] = df[col].dt.hour


add_dateparts(train_df, 'pickup_datetime')


add_dateparts(val_df, 'pickup_datetime')


add_dateparts(test_df, 'pickup_datetime')


# using haversine distance to calculate the distance between given latitude and longitude
import numpy as np
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the greate circle distance between two points
    in the earth (specified in decimal degrees)

    all args must be of equal length
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))

    km = 6367 * c
    return km


def trip_dist(df):
    df['trip_distance'] = haversine_np(df['pickup_longitude'],
                                       df['pickup_latitude'],
                                       df['dropoff_longitude'],
                                       df['dropoff_latitude'])


trip_dist(train_df)
trip_dist(val_df)


trip_dist(test_df)


# creative feature engineering:
'''
creative feature engineering is a lot more effective than excessive hyperparameter tuning. 
Just on or two good feature improve the model's performance drastically

we'll add the distance from drop location of the popular landmarkss in NYC
- JFK airport
- LGA airport
- EWR airport
- Times Square
- Met Museum
- World Trade Center
'''


jfk_lonlat = -73.7781, 40.6413
lga_lonlat = -73.8740, 40.7769
ewr_lonlat = -74.1745, 40.6895
met_lonlat = -73.9632, 40.7794
wtc_lonlat = -74.0099, 40.7126


def add_landmark_dropoff_distance(df, landmark_name, landmark_lonlat):
    lon, lat = landmark_lonlat
    df[landmark_name+ '_drop_distance'] = haversine_np(lon, lat, df['dropoff_longitude'], df['dropoff_latitude'])


def add_landmarks(a_df):
    landmarks = [('jfk', jfk_lonlat), ('lga', lga_lonlat), ('ewr', ewr_lonlat), ('met', met_lonlat), ('wtc', wtc_lonlat)]
    for name, lonlat in landmarks:
        add_landmark_dropoff_distance(a_df, name, lonlat)


add_landmarks(train_df)
add_landmarks(val_df)
add_landmarks(test_df)


# outlier removal
'''
fare amount = 1 to 500
longitudes = -75 to -72
latitudes = 40 to 42
passenger_count = 1 to 6
'''


def remove_outliers(df):
    return df[(df['fare_amount'] >= 1.) &
              (df['fare_amount'] <= 500.) &
              (df['pickup_longitude'] >= -75.) &
              (df['pickup_longitude'] <= -72.) &
              (df['dropoff_longitude'] >= -75.) &
              (df['dropoff_longitude'] <= -72.) &
              (df['dropoff_latitude'] >= 40.) &
              (df['dropoff_latitude'] <= 42.) &
              (df['pickup_latitude'] >= 40.) &
              (df['pickup_latitude'] <= 42.) &
              (df['passenger_count'] <= 6.) &
              (df['passenger_count'] >= 1.)]


train_df = remove_outliers(train_df)


val_df = remove_outliers(val_df)


train_df.to_parquet('train.parquet')


val_df.to_parquet('val.parquet')


train_df.columns


input_cols = ['pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
       'pickup_datetime_year', 'pickup_datetime_month', 'pickup_datetime_day',
       'pickup_datetime_weekday', 'pickup_datetime_hour', 'trip_distance',
       'jfk_drop_distance', 'lga_drop_distance', 'ewr_drop_distance',
       'met_drop_distance', 'wtc_drop_distance']
target_col = 'fare_amount'


train_inputs = train_df[input_cols]
train_targets = train_df[target_col]


val_inputs = val_df[input_cols]
val_targets = val_df[target_col]


test_inputs = test_df[input_cols]


def evaluate(model):
    train_preds = model.predict(train_inputs)
    train_rmse = mean_squared_error(train_targets, train_preds, squared=False)
    val_preds = model.predict(val_inputs)
    val_rmse = mean_squared_error(val_targets, val_preds, squared=False)
    return train_rmse, val_rmse, train_preds, val_preds


from sklearn.linear_model import Ridge


model1 = Ridge(random_state=42, alpha=0.9)


model1.fit(train_inputs, train_targets)


train_rmse, val_rmse, _, _ = evaluate(model1)
print(f'train_rmse: {train_rmse}')
print(f'val_rmse: {val_rmse}')


predict_submit(model1, test_inputs, 'ridge_submission.csv')


from sklearn.ensemble import RandomForestRegressor
model2 = RandomForestRegressor(random_state=42, n_jobs=-1, max_depth=10, n_estimators=100)


%%time
model2.fit(train_inputs, train_targets)


train_rmse, val_rmse, _, _ = evaluate(model2)
print(f'train_rmse: {train_rmse}')
print(f'val_rmse: {val_rmse}')
predict_submit(model2, test_inputs, 'randomforest_submission.csv')


from xgboost import XGBRegressor
model3 = XGBRegressor(max_depth=5, objective='reg:squarederror', n_estimators=200, random_state=42, n_jobs=-1)


%%time
model3.fit(train_inputs, train_targets)


train_rmse, val_rmse, _, _ = evaluate(model3)
print(f'train_rmse: {train_rmse}')
print(f'val_rmse: {val_rmse}')
predict_submit(model3, test_inputs, 'XGRegressor_submission.csv')


'''
hyperparameter tuning:
- tune the most impactful hyperparameter eg. n_estimator
- with the best value of first hyperparameter tune the next
- continue
'''


import matplotlib.pyplot as plt

def test_params(ModelClass, **params):
    model= ModelClass(**params).fit(train_inputs, train_targets)
    train_rmse = mean_squared_error(model.predict(train_inputs), train_targets, squared=False)
    val_rmse = mean_squared_error(model.predict(val_inputs), val_targets, squared=False)
    return train_rmse, val_rmse

def test_params_and_plot(ModelClass, param_name, param_values, **other_params):
    train_errors, val_errors = [], []
    for value in param_values:
        params = dict(other_params)
        params[param_name] = value
        train_rmse, val_rmse = test_params(ModelClass, **params)
        train_errors.append(train_rmse)
        val_errors.append(val_rmse)

    plt.figure(figsize=(10,6))
    plt.title('Overfitting Curve: ' + param_name)
    plt.plot(param_values, train_errors, 'b-o')
    plt.plot(param_values, val_errors, 'r-o')
    plt.xlabel(param_name)
    plt.ylabel('RMSE')


best_params = {
    'random_state': 42,
    'n_jobs': -1,
    'objective': 'reg:squarederror',
    'learning_rate': 0.05
}


%%time
test_params_and_plot(XGBRegressor, 'n_estimators', [100, 200, 400, 500], **best_params)


best_params['n_estimators'] = 500


test_params_and_plot(XGBRegressor, 'max_depth', [3,5,7,9], **best_params)


best_params['max_depth'] = 9


test_params_and_plot(XGBRegressor, 'learning_rate', [0.01 ,0.05, 0.1, 0.15, 0.2, 0.25, 0.3], **best_params)


best_params['learning_rate'] = 0.05


test_params_and_plot(XGBRegressor, 'subsample', [0.01 ,0.05, 0.08, 0.1], **best_params)


best_params['subsample'] = 0.1


test_params_and_plot(XGBRegressor, 'colsample_bytree', [0.1, 0.2, 0.5, 0.7, 0.9], **best_params)


best_params['colsample_bytree'] = 0.8


best_params


xgb_final_model = XGBRegressor(objective='reg:squarederror',
                               n_jobs=-1,
                               random_state=42,
                               n_estimators=500,
                               max_depth=8,
                               learning_rate=0.08,
                               subsample=0.7,
                               colsample_bytree=0.7)


xgb_final_model.fit(train_inputs, train_targets)


train_rmse, val_rmse, _, _ = evaluate(xgb_final_model)
print(f'train_rmse: {train_rmse}')
print(f'val_rmse: {val_rmse}')
predict_submit(xgb_final_model, test_inputs, 'XGBfinal_submission.csv')


# the training is done only on 1% of data, we can do on more amount of data
# we can also do ensembling also

