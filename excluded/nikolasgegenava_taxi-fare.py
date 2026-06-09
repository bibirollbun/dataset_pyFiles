import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings 
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv', nrows=500000)
df_train.head()


df_test = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')
df_test.head()


!pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip


import pandas_profiling

profile = pandas_profiling.ProfileReport(df_train, title="Training Dataset", minimal=True, progress_bar=False)
profile.to_notebook_iframe()


df_train.isna().sum()


df_train = df_train.dropna()


df_train.isna().sum()


plt.figure(figsize = (10, 6))
sns.distplot(df_train['fare_amount'])


df_train = df_train[df_train['fare_amount'] > 0]


plt.figure(figsize = (10, 6))
sns.distplot(df_train['fare_amount'])


df_train['passenger_count'].value_counts().plot.bar(color = 'b', edgecolor = 'k');
plt.title('Passenger Counts'); 
plt.xlabel('Number of Passengers'); 
plt.ylabel('Count');


df_train = df_train[df_train['passenger_count'] != 0]


df = df_train.copy()
df['key'] = pd.to_datetime(df['key'])

df['year'] = df['key'].dt.year
df['month'] = df['key'].dt.month
df['day'] = df['key'].dt.day
df['hour'] = df['key'].dt.hour
df['minute'] = df['key'].dt.minute
df['second'] = df['key'].dt.second
df['day_of_week'] = df['key'].dt.weekday


df.head()


df.drop(['key', 'pickup_datetime'], axis=1, inplace=True)


plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='year', y='fare_amount', marker='o', linewidth=2)
plt.xlabel('Year')
plt.ylabel('Average Fare Amount')
plt.title('Change in Fare Amount Over the Years')
plt.grid(True)


sns.boxplot(df['fare_amount'])


df = df[df['fare_amount'].between(left = 2.5, right = 100)]


df['fare-bin'] = pd.cut(df['fare_amount'], bins = list(range(0, 50, 5))).astype(str)

df.loc[df['fare-bin'] == 'nan', 'fare-bin'] = '[45+]'

df.loc[df['fare-bin'] == '(5, 10]', 'fare-bin'] = '(05, 10]'

df['fare-bin'].value_counts().sort_index().plot.bar(color = 'b', edgecolor = 'k');
plt.title('Fare Binned');


palette = sns.color_palette('Paired', 10)
color_mapping = {fare_bin: palette[i] for i, fare_bin in enumerate(df['fare-bin'].unique())}
color_mapping


df['color'] = df['fare-bin'].map(color_mapping)


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

df['haversine'] = df.apply(
    lambda row: haversine_distance(row['pickup_latitude'], row['pickup_longitude'],
                                   row['dropoff_latitude'], row['dropoff_longitude']), axis=1)

df.head()


subset = df.sample(100000, random_state=42)

plt.figure(figsize = (10, 6))

for f, grouped in subset.groupby('fare-bin'):
    sns.kdeplot(grouped['haversine'], label = f'{f}', color = list(grouped['color'])[0]);
    
plt.title('Distribution of Haversine Distance by Fare Bin');


sns.boxplot(df['haversine'])
plt.title('Haversine Distance Boxplot')
plt.show()


df = df[df['haversine'] <= 2000]


sns.boxplot(df['haversine'])
plt.title('Haversine Distance Boxplot')
plt.show()


df.head()


df['abs_lat_diff'] = (df['dropoff_latitude'] - df['pickup_latitude']).abs()
df['abs_lon_diff'] = (df['dropoff_longitude'] - df['pickup_longitude']).abs()


df.drop(['day', 'hour', 'minute', 'second', 'day_of_week'], axis=1)


important = ['abs_lat_diff', 'abs_lon_diff', 'passenger_count', 'haversine', 'year']


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_train, X_val, y_train, y_val = train_test_split(df, np.array(df['fare_amount']), 
                                                      stratify = df['fare-bin'],
                                                      random_state = 42, test_size = 0.2)


models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor(),

}

results = []

for name, model in tqdm(models.items(), desc="Training Models", leave=True):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])
    pipeline.fit(X_train[important], y_train)
    y_pred = pipeline.predict(X_val[important])
    
    results.append({
        'Model': name,
        'MAE': mean_absolute_error(y_val, y_pred),
        'MSE': mean_squared_error(y_val, y_pred),
        'R2': r2_score(y_val, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_val, y_pred))
    })

results_df = pd.DataFrame(results)


results_df


from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor(n_estimators = 20, max_depth = 20, 
                                      max_features = None, oob_score = True, 
                                      bootstrap = True, verbose = 1, n_jobs = -1)

random_forest.fit(X_train[['haversine', 'abs_lat_diff', 'abs_lon_diff', 'passenger_count']], y_train)


def metrics(train_pred, valid_pred, y_train, y_valid):
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
    
    return train_rmse, valid_rmse

def evaluate(model, features, X_train, X_valid, y_train, y_valid):
    train_pred = model.predict(X_train[features])
    valid_pred = model.predict(X_valid[features])
    
    train_rmse, valid_rmse = metrics(train_pred, valid_pred,
                                                             y_train, y_valid)
    
    print(f'Training RMSE = {round(train_rmse, 2)}')
    print(f'Val RMSE = {round(valid_rmse, 2)}')


evaluate(random_forest, ['haversine', 'abs_lat_diff', 'abs_lon_diff', 'passenger_count'],
         X_train, X_val, y_train, y_val)


def model_rf(X_train, X_valid, y_train, y_valid, features,
             model = RandomForestRegressor(n_estimators = 20, max_depth = 20,
                                           n_jobs = -1),
             return_model = False):
    
    model.fit(X_train[features], y_train)
    
    evaluate(model, features, X_train, X_valid, y_train, y_valid)
    
    if return_model:
        return model


df.head()


model_rf(X_train, X_val, y_train, y_val, 
                   features = ['abs_lat_diff', 'abs_lon_diff', 'haversine', 'passenger_count',
                               'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'year', 'month'])

