import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
from sklearn import cluster
from sklearn import feature_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def add_osrm_features(df, df_osrm):
    
    """The function that adds 3 columns from the right table to the left table"""
    
    # Merge the tables using the 'id' key 
    merged_df = df.merge(df_osrm[['id', 'total_distance', 'total_travel_time', 'number_of_steps']], on = 'id', how = 'left')
    
    return merged_df 

def add_holiday_features(df, df_holidays):
    
    """ The function that returns an updated table with trip data, including an added column `pickup_holiday` """
    
    # Let's create a list of holiday dates
    df_holidays_list = list(df_holidays['date'].values)
    
    # Let's convert the 'pickup_date' feature to the "str" format
    df['pickup_date_str'] = df['pickup_date'].astype('str')
    
    # Let's create a column 'pickup_holiday', where 1 indicates that the date is in the `df_holidays_list`, and 0 otherwise
    df['pickup_holiday']  = df['pickup_date_str'].apply(lambda x: 1 if x in df_holidays_list else 0)
    
    # Let's remove the 'pickup_date_str' feature
    df = df.drop(axis = 1, columns = 'pickup_date_str')
    
    return df 

def add_datetime_features(df):
    
    """The function that takes a table with trip data as input and returns the same table with three additional columns"""
    
    df['pickup_date'] = df['pickup_datetime'].dt.date
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_day_of_week'] = df['pickup_datetime'].dt.dayofweek
    
    return df

def get_haversine_distance(lat1, lng1, lat2, lng2):
    
    """The function for calculating distance using the Haversine's formula (in kilometers)"""
    
    # Converting angles to radians
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    # The radius of the Earth in kilometers
    EARTH_RADIUS = 6371 
    # Сalculating the shortest distance \( h \) using the Haversine's formula
    lat_delta = lat2 - lat1
    lng_delta = lng2 - lng1
    d = np.sin(lat_delta * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng_delta * 0.5) ** 2
    h = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def get_angle_direction(lat1, lng1, lat2, lng2):
    
    """The function to calculate the direction angle of movement (in degrees)"""
    
    # Converting angles to radians
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    # Calculating the direction angle **alpha** using the bearing angle formula
    lng_delta_rad = lng2 - lng1
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    alpha = np.degrees(np.arctan2(y, x))
    return alpha

def add_geographical_features(df):
    
    """The function takes a table with trip data as input and returns an updated table with two added columns"""
    
    # Сonverting our features-coordinates into NumPy-arrays
    lat1 = df['pickup_latitude'].to_numpy()
    lng1 = df['pickup_longitude'].to_numpy()
    lat2 = df['dropoff_latitude'].to_numpy()
    lng2 = df['dropoff_longitude'].to_numpy()
    
    df['haversine_distance'] = pd.Series(get_haversine_distance(lat1, lng1, lat2, lng2))
    df['direction'] = pd.Series(get_angle_direction(lat1, lng1, lat2, lng2))
    
    return df


def add_cluster_features(df, clustering_method):
    
    """The function that labels trip numbers by cluster numbers"""
    
    predictions = clustering_method.labels_

    # Save predictions in DataFrame
    df['geo_cluster'] = predictions
    
    return df

def add_weather_features(df_taxi, weather_df):
    
    """The function adds 5 features from the weather table to the taxi-trips table"""
    
    # We will keep only the features of interest and, in addition, the key features
    weather_df = weather_df[['temperature', 'visibility', 'wind speed', 'precip', 'events', 'date', 'hour']]
    
    df_taxi['pickup_date'] = taxi_data["pickup_date"].astype('str')
    
    # Let's add the features we are interested in to our table on the left using keys
    merged_df = df_taxi.merge(weather_df, left_on = ['pickup_date', 'pickup_hour'], right_on = ['date', 'hour'], how = 'left')
    
    # We will remove key and no longer needed features
    merged_df = merged_df.drop(axis=1, columns=['date', 'hour'])
    
    return merged_df 

def fill_null_weather_data(df_taxi):
    
    """The function for filling missing data with the median value"""
    
    cols_list = ['temperature', 'visibility', 'wind speed', 'precip']
    
    # For each feature from the `cols_list`:
    for col in cols_list:
        
        # We will group the data by the `pickup_date` column and fill the missing values with the median value for each group
        df_taxi[col] = df_taxi[col].fillna(df_taxi.groupby('pickup_date')[col].transform('median')) 
    
    values = {
        'total_distance': df_taxi['total_distance'].median(),
        'total_travel_time': df_taxi['total_travel_time'].median(),
        'number_of_steps': df_taxi['number_of_steps'].median(),
        'events': 'None'
    }
    
    # Filling in missing values according to the specified dictionary
    df_taxi = df_taxi.fillna(values)
    
    return df_taxi


# Look at the data
taxi_data = pd.read_csv("/kaggle/input/nyc-taxi-trip-duration/train.zip")
print('Train data shape: {}'.format(taxi_data.shape))
taxi_data.head()


# Convert the features 'pickup_datetime' and 'dropoff_datetime' to the `datetime` data type with the format "year-month-day hour:minute:second"
taxi_data['pickup_datetime'] = pd.to_datetime(taxi_data['pickup_datetime'])
taxi_data['dropoff_datetime'] = pd.to_datetime(taxi_data['dropoff_datetime'])


taxi_data.isnull().mean() * 100


taxi_data.info()


taxi_data = add_datetime_features(taxi_data)


# Load the table of public holidays
holiday_data = pd.read_csv('/kaggle/input/holiday-data/holiday_data.csv', sep=';')


# Add a binary feature 'pickup_holiday'
taxi_data = add_holiday_features(taxi_data, holiday_data)


# Load OSRM data
osrm_data = pd.read_csv('/kaggle/input/osrm-data-train/osrm_data_train.csv')


# Add three new features 'total_distance', 'total_travel_time', 'number_of_steps'
taxi_data = add_osrm_features(taxi_data, osrm_data)


# Add two new 'haversine_distance' and 'direction'
taxi_data = add_geographical_features(taxi_data)


# We create a training dataset from the geographical coordinates of all points
coords = np.hstack((taxi_data[['pickup_latitude', 'pickup_longitude']],
                    taxi_data[['dropoff_latitude', 'dropoff_longitude']]))

# Training the clustering algorithm
kmeans = cluster.KMeans(n_clusters=10, n_init=10, random_state=42)
kmeans.fit(coords)


# Add a feature indicating cluster membership 'geo_cluster'
taxi_data = add_cluster_features(taxi_data, kmeans)


# Add 5 new weather-related features by date
weather_data = pd.read_csv('/kaggle/input/weather-data/weather_data.csv')
taxi_data = add_weather_features(taxi_data, weather_data)


# Let's add 5 new weather-related features by date
taxi_data = fill_null_weather_data(taxi_data)


avg_speed = taxi_data['total_distance'] / taxi_data['trip_duration'] * 3.6
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(x=avg_speed.index, y=avg_speed, ax=ax)
ax.set_xlabel('Index')
ax.set_ylabel('Average speed');


# We will keep only those trips where "total_distance" does not exceed 24 h and the average speed does not exceed 300 km/h
taxi_data_new = taxi_data[(taxi_data['trip_duration'] <= 24*60*60)&(taxi_data['total_distance'] / taxi_data['trip_duration'] * 3.6 < 300)]

# Let's update of the table numeration
taxi_data_new = taxi_data_new.reset_index(drop=True)

print(f'Total number of detected outliers: {taxi_data.shape[0] - taxi_data_new.shape[0]}')


taxi_data_new["trip_duration_log"] = np.log(taxi_data_new["trip_duration"] + 1)
display(taxi_data_new)


# Set the histogram parameters
fig = plt.figure(figsize=(10, 8))
sns.histplot(
    data=taxi_data_new,
    x='trip_duration_log',
    bins=70,
    kde=True,
);

fig = plt.figure(figsize=(10, 8))
boxplot = sns.boxplot(
    data=taxi_data_new,
    x='trip_duration_log',
    orient='h',
    width=0.9
)
boxplot.set_title('Boxplot distribution for the target feature trip_duration_log');
boxplot.set_xlabel('Trip_duration_log');
boxplot.grid()

from scipy.stats import normaltest
import numpy as np

#  Checking the target feature `'trip_duration_log'` for normality using the D’Agostino test at a significance level of 0.05
statistic, p_value = normaltest(taxi_data_new['trip_duration_log'])

print(f"Statistic: {statistic:.2f}")
print(f"P-value: {p_value:.2f}")

if p_value < 0.05:
    print("❌ The distribution is not normal (we reject H₀)")
else:
    print("✅ The distribution may be normal (we do not reject H₀))")



plt.figure(figsize=(12, 6))
countplot = sns.countplot(x="pickup_hour", hue="pickup_hour", data=taxi_data_new, edgecolor="black")
countplot.set_title('Pickup_hour countplot');
plt.show()


fig = plt.figure(figsize=(12, 6))
barplot = sns.barplot(
    data=taxi_data_new,
    x='pickup_hour',
    y='trip_duration',
    hue='pickup_hour',
    estimator='median'
)
barplot.set_title('Barplot of median trip duration by hour of the day');


plt.figure(figsize=(12, 6))
countplot = sns.countplot(x="pickup_day_of_week", hue="pickup_day_of_week", data=taxi_data_new, edgecolor="navy")
countplot.set_title('Pickup_day_of_week countplot');
plt.show()


fig = plt.figure(figsize=(12, 6))
barplot = sns.barplot(
    data=taxi_data_new,
    x='pickup_day_of_week',
    y='trip_duration',
    hue='pickup_day_of_week',
    estimator='median',
    palette='Spectral'
)
barplot.set_title('Median trip duration by number of the day barplot');


pickup_hour_days_df = taxi_data_new.groupby(['pickup_hour', 'pickup_day_of_week'])['trip_duration'].median()

fig = plt.figure(figsize=(12, 6))
heatmap = sns.heatmap(data=pickup_hour_days_df.unstack(), cmap='coolwarm')
heatmap.set_title('Pickup_hour/days heatmap', fontsize=16);


train_data = taxi_data_new.copy()
drop_columns = ['id', 'dropoff_datetime', 'pickup_datetime', 'pickup_date']
train_data = train_data.drop(drop_columns, axis=1)
print('Shape of data: {}'.format(train_data.shape))


# Let's encode the binary features "vendor_id" and "store_and_fwd_flag" using LabelEncoder()
le = LabelEncoder()
train_data["vendor_id"] = le.fit_transform(train_data["vendor_id"])
train_data["store_and_fwd_flag"] = le.fit_transform(train_data["store_and_fwd_flag"])
train_data


one_hot_encoder_cols = ['pickup_day_of_week', 'geo_cluster', 'events']
# Declare the encoder
one_hot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
# We train and immediately apply the transformation to the dataset, then convert the result into an array
data_onehot_array = one_hot_encoder.fit_transform(train_data[one_hot_encoder_cols]).toarray()
# We obtain the encoded column names
column_names = one_hot_encoder.get_feature_names_out(one_hot_encoder_cols)
# Creating a DataFrame from the encoded features
data_onehot = pd.DataFrame(data_onehot_array, columns=column_names)

# Let's add the encoded columns to the train_data data-frame
train_data = pd.concat(
    [train_data.reset_index(drop=True).drop(one_hot_encoder_cols, axis=1), data_onehot], 
    axis=1
    )


X = train_data.drop(['trip_duration', 'trip_duration_log'], axis=1)
y = train_data['trip_duration']
y_log = train_data['trip_duration_log']


X_train, X_valid, y_train_log, y_valid_log = model_selection.train_test_split(
    X, y_log, 
    test_size=0.33, 
    random_state=42
)


from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(f_regression, k=25)
selector.fit(X_train, y_train_log)

best_features = list(selector.get_feature_names_out())

# Select the top 25 features for further prediction
X_train = X_train[best_features]
X_valid = X_valid[best_features]

print(f'The best features according to the SelectKBest method: {best_features}')


from sklearn.preprocessing import MinMaxScaler

MinMax_scaler = MinMaxScaler()
MinMax_scaler.fit(X_train)
X_train_scaled_array = MinMax_scaler.transform(X_train)
X_valid_scaled_array = MinMax_scaler.transform(X_valid)

X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=X_train.columns)
X_valid_scaled = pd.DataFrame(X_valid_scaled_array, columns=X_valid.columns)


from sklearn.linear_model import LinearRegression

lin_reg = linear_model.LinearRegression()
lin_reg.fit(X_train_scaled, y_train_log)

y_train_log_predict = lin_reg.predict(X_train_scaled)
y_valid_log_predict = lin_reg.predict(X_valid_scaled)

RMSLE_train = (metrics.mean_squared_error(y_train_log, y_train_log_predict))**0.5
RMSLE_valid = (metrics.mean_squared_error(y_valid_log, y_valid_log_predict))**0.5
print(f'RMSLE for the simple linear regression model on the training set: {round((RMSLE_train),2)}')
print(f'RMSLE for the simple linear regression model on the valid set: {round((RMSLE_valid),2)}')


# Creating a polynomial feature generator
poly = preprocessing.PolynomialFeatures(degree=2, include_bias=False)
poly.fit(X_train_scaled)
# Generating polynomial features for the training set
X_train_poly = poly.transform(X_train_scaled)
# Generating polynomial features for the valid set
X_valid_poly = poly.transform(X_valid_scaled)

# Creat LinearRegression object
lr_model_poly = linear_model.LinearRegression()
# Train the model using the OLS method
lr_model_poly.fit(X_train_poly, y_train_log)
y_train_predict_poly = lr_model_poly.predict(X_train_poly)
y_valid_predict_poly = lr_model_poly.predict(X_valid_poly)
 
# We calculate RMSLE for two samples
RMSLE_train_poly = (metrics.mean_squared_error(y_train_log, y_train_predict_poly))**0.5
RMSLE_valid_poly = (metrics.mean_squared_error(y_valid_log, y_valid_predict_poly))**0.5
print(f'RMSLE for the polinomial model regression on the training set: {round((RMSLE_train_poly),2)}')
print(f'RMSLE for the polinomial model regression on the valid set: {round((RMSLE_valid_poly),2)}')


# Сreate an instance of a linear regression model with L2 regularization.
ridge_lr_poly = linear_model.Ridge(alpha=1)

# Fitting model
ridge_lr_poly.fit(X_train_poly, y_train_log)
# Make a prediction on the training set
y_train_predict_poly_ridge = ridge_lr_poly.predict(X_train_poly)
# Make a prediction on the validation set
y_valid_predict_poly_ridge = ridge_lr_poly.predict(X_valid_poly)

# We calculate RMSLE for two samples
RMSLE_train_poly_ridge = (metrics.mean_squared_error(y_train_log, y_train_predict_poly_ridge))**0.5
RMSLE_valid_poly_ridge = (metrics.mean_squared_error(y_valid_log, y_valid_predict_poly_ridge))**0.5
print(f'RMSLE for the polinomial model regression on the training set: {round((RMSLE_train_poly_ridge),2)}')
print(f'RMSLE for the polinomial model regression on the valid set: {round((RMSLE_valid_poly_ridge),2)}')


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import hyperopt
from hyperopt import hp, fmin, tpe, Trials

# Let's define the hyperparameter search space
space={
       'max_depth' : hp.quniform('max_depth', 7, 13, 1),
       'min_samples_leaf': hp.quniform('min_samples_leaf', 2, 9, 1),
        'max_leaf_nodes': hp.quniform('max_leaf_nodes', 5, 25, 2)
      }
# Fixing random_state
random_state = 42

def hyperopt_DTR(params, cv=5, X=X_train_scaled, y=y_train_log, random_state=random_state):
    # The function receives a combination of hyperparameters in "params"
    params = {
               'max_depth': int(params['max_depth']), 
               'min_samples_leaf': int(params['min_samples_leaf']),
                'max_leaf_nodes': int(params['max_leaf_nodes'])
              }
  
    # Let's use this combination to build the model
    model = DecisionTreeRegressor(**params, random_state=random_state)

    # fitting the model
    model.fit(X, y)
    score = (metrics.mean_squared_error(y, model.predict(X)))**0.5
    
    return -score


trials = Trials() # Logging ours results

best=fmin(hyperopt_DTR, 
          space=space, 
          algo=tpe.suggest, 
          max_evals=80, 
          trials=trials, 
          rstate=np.random.default_rng(random_state), 
          verbose=True
         )
print("The best parameters: {}".format(best))

# Let's train the model and make a prediction
DSM_model = DecisionTreeRegressor(
    random_state=random_state, 
    max_leaf_nodes=int(best['max_leaf_nodes']),
    max_depth=int(best['max_depth']),
    min_samples_leaf=int(best['min_samples_leaf'])
)

DSM_model.fit(X_train_scaled, y_train_log)
y_train_pred = DSM_model.predict(X_train_scaled)
y_valid_pred = DSM_model.predict(X_valid_scaled)

# We calculate RMSLE for two samples on the best parametrs
RMSLE_train_DSM = (metrics.mean_squared_error(y_train_log, y_train_pred))**0.5
RMSLE_valid_DSM = (metrics.mean_squared_error(y_valid_log, y_valid_pred))**0.5

print(f'RMSLE for the DecisionTreeRegressor model on the training set: {round((RMSLE_train_DSM),2)}')
print(f'RMSLE for the DecisionTreeRegressor model on the valid set: {round((RMSLE_valid_DSM),2)}')


# Build a RandomForestRegressor model
from sklearn.ensemble import RandomForestRegressor

model_RFR = RandomForestRegressor(
    n_estimators=200, 
    max_depth=12,       
    random_state=42,       
    criterion='squared_error',
    min_samples_split=20
)

# Fitting model
model_RFR.fit(X_train_scaled, y_train_log)

# Make a prediction on the training set
y_train_pred = model_RFR.predict(X_train_scaled)
# Make a prediction on the validation set
y_valid_pred = model_RFR.predict(X_valid_scaled)

# We calculate RMSLE for two samples on the best parametrs
RMSLE_train_RFR = (metrics.mean_squared_error(y_train_log, y_train_pred))**0.5
RMSLE_valid_RFR = (metrics.mean_squared_error(y_valid_log, y_valid_pred))**0.5

print(f'RMSLE for the DecisionTreeRegressor model on the training set: {round((RMSLE_train_RFR),2)}')
print(f'RMSLE for the DecisionTreeRegressor model on the valid set: {round((RMSLE_valid_RFR),2)}')


from sklearn.ensemble import GradientBoostingRegressor

# Build a GradientBoostingRegressor model
model_GBR = GradientBoostingRegressor(
        learning_rate=0.5,
        n_estimators=100,
        max_depth=6, 
        min_samples_split=30,
        random_state=42
)

# Fitting model
model_GBR.fit(X_train_scaled, y_train_log)

# Make a prediction on the training set
y_train_pred = model_GBR.predict(X_train_scaled)
# Make a prediction on the validation set
y_valid_pred = model_GBR.predict(X_valid_scaled)

# We calculate RMSLE for two samples on the best parametrs
RMSLE_train_GBR = (metrics.mean_squared_error(y_train_log, y_train_pred))**0.5
RMSLE_valid_GBR = (metrics.mean_squared_error(y_valid_log, y_valid_pred))**0.5

print(f'RMSLE for the DecisionTreeRegressor model on the training set: {round((RMSLE_train_GBR),2)}')
print(f'RMSLE for the DecisionTreeRegressor model on the valid set: {round((RMSLE_valid_GBR),2)}')


# Now let's look at the importance of the features
fig, ax = plt.subplots(figsize=(20, 7)) 
feature = X_train_scaled.columns 
feature_importances = model_GBR.feature_importances_ 

sns.barplot(x=feature, y=feature_importances, ax=ax);
ax.set_title('Bar plot feature importances')
ax.set_xlabel('Features')
ax.set_ylabel('Importances')
ax.xaxis.set_tick_params(rotation=65);


# Now create a submission for the test dataset
test_data = pd.read_csv("/kaggle/input/nyc-taxi-trip-duration/test.zip")
osrm_data_test = pd.read_csv("/kaggle/input/osrm-data-test/osrm_data_test.csv")
test_id = test_data['id']

test_data['pickup_datetime']=pd.to_datetime(test_data['pickup_datetime'],format='%Y-%m-%d %H:%M:%S')
test_data = add_datetime_features(test_data)
test_data = add_holiday_features(test_data, holiday_data)
test_data = add_osrm_features(test_data, osrm_data_test)
test_data = add_geographical_features(test_data)

# We create a training dataset from the geographical coordinates of all points
coords = np.hstack((test_data[['pickup_latitude', 'pickup_longitude']],
                    test_data[['dropoff_latitude', 'dropoff_longitude']]))

# Training the clustering algorithm
kmeans = cluster.KMeans(n_clusters=10,n_init=10, random_state=42)
kmeans.fit(coords)

test_data = add_cluster_features(test_data, kmeans)
test_data = add_weather_features(test_data, weather_data)
test_data = fill_null_weather_data (test_data)

test_data['vendor_id'] = test_data['vendor_id'].apply(lambda x: 0 if x == 1 else 1)
test_data['store_and_fwd_flag'] = test_data['store_and_fwd_flag'].apply(lambda x: 0 if x == 'N' else 1)
test_data_onehot = one_hot_encoder.fit_transform(test_data[one_hot_encoder_cols]).toarray()
column_names = one_hot_encoder.get_feature_names_out(one_hot_encoder_cols)
test_data_onehot = pd.DataFrame(test_data_onehot, columns=column_names)

test_data = pd.concat(
    [test_data.reset_index(drop=True).drop(one_hot_encoder_cols, axis=1), test_data_onehot], 
    axis=1
)

X_test = test_data[best_features]
X_test_scaled = MinMax_scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Make a prediction on the test dataset
y_test_pred_log = model_GBR.predict(X_test_scaled)
y_test_predict = np.exp(y_test_pred_log) -1

submission = pd.DataFrame({'id': test_id, 'trip_duration': y_test_predict})

display(submission)

