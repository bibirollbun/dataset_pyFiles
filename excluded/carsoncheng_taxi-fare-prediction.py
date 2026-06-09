import pandas as pd
#df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")
df = pd.read_csv("/kaggle/input/taxi-fare-reduced/taxi_fare_reduced.csv").drop(columns=["Unnamed: 0", "key", "pickup_datetime"])


df.head()


y = df['fare_amount']
X = df.drop(columns=['fare_amount'])


X[pd.isnull(X).any(axis=1)] # no missing data


# split the data again
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(len(X_train), len(X_val), len(X_test))


train_df = pd.concat([X_train, y_train], axis=1)


import matplotlib.pyplot as plt
import seaborn as sns

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('KDE Plots of Location Coordinates', fontsize=16)

# Plot each column
sns.kdeplot(train_df['pickup_longitude'], ax=axes[0, 0], fill=True)
axes[0, 0].set_title('Pickup Longitude')
axes[0, 0].set_xlabel('Longitude')
axes[0, 0].set_ylabel('Density')

sns.kdeplot(train_df['pickup_latitude'], ax=axes[0, 1], fill=True)
axes[0, 1].set_title('Pickup Latitude')
axes[0, 1].set_xlabel('Latitude')
axes[0, 1].set_ylabel('Density')

sns.kdeplot(train_df['dropoff_longitude'], ax=axes[1, 0], fill=True)
axes[1, 0].set_title('Dropoff Longitude')
axes[1, 0].set_xlabel('Longitude')
axes[1, 0].set_ylabel('Density')

sns.kdeplot(train_df['dropoff_latitude'], ax=axes[1, 1], fill=True)
axes[1, 1].set_title('Dropoff Latitude')
axes[1, 1].set_xlabel('Latitude')
axes[1, 1].set_ylabel('Density')

plt.tight_layout()
plt.show()


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#coord_features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
scaler = StandardScaler()
pca = ___________ # YOUR CODE HERE
X_train_reduced = pca.fit_transform(scaler.fit_transform(X_train))
X_train_reduced
sns.scatterplot(x=X_train_reduced[:,0], y=X_train_reduced[:,1])





def evaluate(y_train, train_preds, y_val, val_preds):
    print(f"Training MSE: {mean_squared_error(y_train, train_preds)}")
    print(f"Training MAE: {mean_absolute_error(y_train, train_preds)}")
    print(f"Training R^2 score: {r2_score(y_train, train_preds)}")
    print(f"Validation MSE: {mean_squared_error(y_val, val_preds)}")
    print(f"Validation MAE: {mean_absolute_error(y_val, val_preds)}")
    print(f"Validation R^2 score: {r2_score(y_val, val_preds)}")


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
reg = LinearRegression() # define the model using the constructor
reg.fit(X_train, y_train) # fit the model
### model evaluation
train_preds = reg.predict(X_train) # evaluate the model
val_preds = reg.predict(X_val) # evaluate the model
evaluate(y_train, train_preds, y_val, val_preds)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
reg = RandomForestRegressor() # define the model using the constructor
reg.fit(X_train, y_train) # fit the model
### model evaluation
train_preds = reg.predict(X_train) # evaluate the model
val_preds = reg.predict(X_val) # evaluate the model
evaluate(y_train, train_preds, y_val, val_preds)


# distance calculation function from https://www.kaggle.com/code/lukajincharadze/ml-midterm
# engineer the h_distance feature (haversine distance)
# the geometric distance between two locations on the earth (based on two longitude/latitude pairs)
import numpy as np
def h_distance(df):
    lat1, lon1, lat2, lon2 = df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude']
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of Earth in Kilometers
    r = 6371

    d = c * r
    return d


# feature engineering: 
X_train['h_distance'] = h_distance(X_train)
X_val['h_distance'] = h_distance(X_val)
X_test['h_distance'] = h_distance(X_test)


X_train


X_train[X_train == 0.0] = np.nan
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]
X_val[X_val == 0.0] = np.nan
X_val = X_val.dropna()
y_val = y_val.loc[X_val.index]
X_test[X_test == 0.0] = np.nan
X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]





from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
reg = LinearRegression() # define the model using the constructor
reg.fit(X_train.drop(columns=['h_distance']), y_train) # fit the model
### model evaluation
train_preds = reg.predict(X_train.drop(columns=['h_distance'])) # evaluate the model
val_preds = reg.predict(X_val.drop(columns=['h_distance'])) # evaluate the model
evaluate(y_train, train_preds, y_val, val_preds)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
reg = RandomForestRegressor() # define the model using the constructor
reg.fit(X_train.drop(columns=['h_distance']), y_train) # fit the model
### model evaluation
train_preds = reg.predict(X_train.drop(columns=['h_distance'])) # evaluate the model
val_preds = reg.predict(X_val.drop(columns=['h_distance'])) # evaluate the model
evaluate(y_train, train_preds, y_val, val_preds)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
reg = LinearRegression() # define the model using the constructor
reg.fit(X_train, y_train) # fit the model
### model evaluation
train_preds = reg.predict(X_train) # evaluate the model
val_preds = reg.predict(X_val) # evaluate the model
evaluate(y_train, train_preds, y_val, val_preds)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
reg = RandomForestRegressor() # define the model using the constructor
reg.fit(X_train, y_train) # fit the model
### model evaluation
train_preds = reg.predict(X_train) # evaluate the model
val_preds = reg.predict(X_val) # evaluate the model
evaluate(y_train, train_preds, y_val, val_preds)







