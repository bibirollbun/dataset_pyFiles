
import random 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 


from sklearn.metrics import r2_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split



import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


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



train = pd.read_csv("/kaggle/input/nyc-taxi-trip-duration/train.zip")
test = pd.read_csv("/kaggle/input/nyc-taxi-trip-duration/test.zip")
submission = pd.read_csv("/kaggle/input/nyc-taxi-trip-duration/sample_submission.zip")
train.head(3)



test.head()


submission.head()




len(train),len(test),len(submission)



# check the null values
train.isnull().sum()




# check duplicate value: (zero duplicate value)
train.duplicated().sum().item()



train.info()



# description of the data:
train.describe()




# min, mean, max of target variable: (second):
""" 
- MIN: 1sec how?? (Maybe Outlier)
- MEDIAN: 662 means 50% trip_duration is less then 662s or 11.03 minutes
- MAX: 3526282/(60*60*24) = 40.83 days
**Here, For finding meadian, we sort the data in assending order**
- FROM MEDIAN MAX, May be our data is Right Squwed.
"""

print("min value: {}".format(train["trip_duration"].min()))
print("max value: {}".format(train["trip_duration"].max()))
print("median value: {}".format(train["trip_duration"].median()))
print("standard deviation: {}".format(train["trip_duration"].std()))
print("variance: {}".format(train["trip_duration"].var()))





# skewness of target variables: +ve values (r8 skwed)
print("skewness of target variables: {}".format(train["trip_duration"].skew()))




""" 
- It's very hard to visulize such big value 
- Soln, Create a new column and convert the second into minutes and hours
"""
#boxplot:
sns.boxplot(data=train,x="trip_duration")


train.columns , test.columns, submission.columns



train["trip_duration_min"] = train["trip_duration"]/60
train["trip_duration_hours"] = train["trip_duration"]/(60*60)




#now check the ouliers:
""" 
- Higly outlier
- Without Xlim: X-axis range(0~1000)hours
- Q3+1.5IQR: X-axis range (0-0.5)hours
- **Decition: Should be use Mathematical Transformation.**
- **For, we have also test data, so don't directly use fit_trainsfrom method**
- **Otherwise, lemda value will not be good for small amount of test data**
"""

plt.xlim(0,2)
sns.boxplot(data=train,x="trip_duration_hours")



!pip install geodatasets


import geodatasets
import geopandas as gpd

geodatasets.data



nyc_map = gpd.read_file(geodatasets.get_path("nybb"))
nyc_map.crs


nyc_map[:3]


nyc_map.plot(edgecolor="black")



#in train dataset, the value measurement of log,lat = EPSC:4326
#so, eitar Coordiate Reference System(crs) change kora jabe na. while making the data

drop_data = gpd.GeoDataFrame(
    data=train,
   geometry= gpd.points_from_xy(x=train["pickup_longitude"],y=train["pickup_latitude"]),
   crs="EPSG:4326" 
)



drop_data.crs



# chnage crs of drop_data now
nyc_map = nyc_map.to_crs(drop_data.crs)




# pick up distance --> overlap into NCY--Map: 
# outside of the map--->is outliers.
fig,ax = plt.subplots(figsize=(10,6))
plt.xlim(-74.3,-73.65)
plt.ylim(40.5,40.92)
nyc_map.plot(column="BoroName",ax=ax,edgecolor="black",
             legend=True,legend_kwds={"loc":"center left"})
drop_data.plot(ax=ax,color="black",markersize=1,alpha=0.5)
leg = ax.get_legend()
if leg:
    leg.set_bbox_to_anchor((1, 0.5))
plt.show()


""" 
New York City's latitude and longtitude cooridnates are 40.730610, -73.935242.,
visulize only trip that is occur within New York City
"""
# show only the trip pickup points:

plt.figure(figsize=(10,6))
sns.scatterplot(data=train,x="pickup_longitude",y="pickup_latitude",s=1,alpha=0.5)
plt.xlim(-74.04,-73.75)
plt.ylim(40.6,40.9)
plt.show()





def haversine_formula(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dphi = lat2 - lat1
    dlambda = lon2 - lon1

    a = np.sin(dphi / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Earth radius in km
    R = 6371.0
    d = R * c
    return d

lat1 = 5.06995833
lon1 = 6.96471111
lat2 = 12.66002222
lon2 = 8.599 

distance_km = haversine_formula(lat1, lon1, lat2, lon2)
print(distance_km)



submission.columns



# adding a new column name distance: 
train["distance"] = haversine_formula(
    lat1=train["pickup_latitude"],
    lon1=train["pickup_longitude"],
    lat2=train["dropoff_latitude"],
    lon2=train["dropoff_longitude"])


test["distance"] = haversine_formula(
    lat1=test["pickup_latitude"],
    lon1=test["pickup_longitude"],
    lat2=test["dropoff_latitude"],
    lon2=test["dropoff_longitude"])



train.columns



# visulize distance:
plt.figure(figsize=(10,4))
plt.xlabel("Distance")
plt.ylabel("Trip Duration In Hours")
# alpha=density of points
plt.scatter(x=train["distance"],y=train["trip_duration_hours"],alpha=0.2)
plt.tight_layout()
plt.grid(True)
plt.show()


train.columns


test.columns


from sklearn.ensemble import IsolationForest

isoF = IsolationForest(
    n_estimators=100,
    random_state=42,
    contamination=0.01,
    verbose=True
)

# Fit only on training data
isoF.fit(train[["distance", "trip_duration"]])

# Predict anomalies
train["anomaly"] = isoF.predict(train[["distance", "trip_duration"]])





#if isolation==1 then anomaly other Not anomaly:
train[:2]



""" 
Here, 
- Blue color is not: Anomaly
- green is: Anomaly 
"""

plt.figure(figsize=(10,4))
plt.scatter(train[train["anomaly"] == 1]["distance"],
                train[train["anomaly"] == 1]["trip_duration"],
                c="blue",alpha=0.3)

plt.scatter(train[train["anomaly"] == -1]["distance"],
                train[train["anomaly"] == -1]["trip_duration"],
                c="green",alpha=0.5)
plt.ylim(0,4000)
plt.xlim(0,200)
plt.show()



#seperate anomaly data:
train = train[train["anomaly"]==1]


len(train)



# plot after removing anomaly: 
plt.xlabel("Distance in Kilometer")
plt.ylabel("Time is second")
sns.scatterplot(data=train,y="trip_duration",x="distance",alpha=0.3)




train["pickup_datetime"][:1]



date_time_train = pd.to_datetime(train["pickup_datetime"][:1])
date_time_train 


date_time_test = pd.to_datetime(test["pickup_datetime"][:1])
date_time_test



#pick up date,day,
date_time_train.dt.day,date_time_train.dt.month,date_time_train.dt.year



train["dropoff_datetime"] = pd.to_datetime(train["dropoff_datetime"])
train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"])


test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])
#test["dropoff_datetime"] = pd.to_datetime(test["dropoff_datetime"])

print(train.info())




# store only pickup date:
train["date"] = train["pickup_datetime"].dt.date
test["date"] = test["pickup_datetime"].dt.date



#pick up hour,minute and second:
date_time_train.dt.hour, date_time_train.dt.minute,date_time_train.dt.second



train["date"].dtype



train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
train["date"].dtype,test["date"].dtype



""" 
- The question is: in one day there are not only one trip. How lineplot() find the distance?
- sns.lineplot() average all the distance travel in same day.
"""
# visulize distance by date:
plt.figure(figsize=(10,6))
sns.lineplot(data=train,x="date",y="distance",errorbar=None) #errorbar=ci=Confidence Inverval
plt.tight_layout()
plt.grid("True")
plt.show()




# visulization: trip duration vs date:
plt.figure(figsize=(10,6))
sns.lineplot(data=train,x="date",y="trip_duration",errorbar=None) #errorbar=ci=Confidence Inverval
plt.tight_layout()
plt.grid("True")
plt.show()




# Insert new column name: 
print(train["pickup_datetime"].dt.hour[0])
print(train["pickup_datetime"].dt.day_of_week[5])

train["pickup_day"] = train["pickup_datetime"].dt.day_of_week
train["pickup_hour"] = train["pickup_datetime"].dt.hour


test["pickup_day"] = test["pickup_datetime"].dt.day_of_week
test["pickup_hour"] = test["pickup_datetime"].dt.hour




# <----------------------Time: 24 Hours + Total Trip--------------------------->
""" 
23 -> 11.PM
0 -> 12.00 PM 

---High Traffic---
17 -> 5.00 PM 
18 -> 6.00 PM
**Work Finish Time**

---Low Traffic---
4 -> 4.00 AM
5 -> 5.00 AM
**Time To Sleep**
"""
# pickhour VS number of trip count:
plt.figure(figsize=(10,6))
plt.xlabel("Hour")
plt.ylabel("Total Number Of Trip")
sns.countplot(data=train,x="pickup_hour")
plt.show()



#<---------------------------Day: + Total Trip Count -------------------------->
""" 
Observation:
    - Off day: Friday and Saturday: Traffic High
"""
train['pickup_day_name'] = train['pickup_day'].map({
    0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
    3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
})

test['pickup_day_name'] = test['pickup_day'].map({
    0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
    3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
})

plt.figure(figsize=(10,5))
weekday_labels = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
sns.countplot(x='pickup_day_name', data=train, order=weekday_labels)
plt.title('Trips by Day of the Week')
plt.xlabel('Day of Week')
plt.ylabel('Trip Count')
plt.show()




#<-----------------------Trip Duration VS Hour-------------------------------->
""" 
**Observation:**
The graph shows that the average trip duration increases from around 6 am to 3 pm. 
This means that the trips themselves are taking longer during that period, 
not necessarily that the number of trips is higher.
"""

hourlyAvgDuration = train.groupby('pickup_hour')["trip_duration"].mean()

plt.figure(figsize=(10,6))
sns.lineplot(x=hourlyAvgDuration.index,y=hourlyAvgDuration.values,marker='o')
plt.xlabel("time: Hour")
plt.ylabel("Average Trip Duration(Second)")
plt.xticks(hourlyAvgDuration.index)
plt.grid(True)
plt.tight_layout()
plt.show()





#<-----------------------Trip Duration VS Day of Week-------------------------------->
avg_by_day = train.groupby("pickup_day_name")["trip_duration"].mean()
plt.figure(figsize=(10,6))
sns.lineplot(x=avg_by_day.index,y=avg_by_day.values,marker='o')
plt.xlabel("Weekly")
plt.ylabel("Average Trip Duration(Second)")
plt.xticks(avg_by_day.index)
plt.grid(True)
plt.tight_layout()
plt.show()




# <---------------------Create flag based on week and pick our----------------------------->
""" 
Time: 7<=x<=9 or 16<=x<=19 then, 1 otherwize 0
0 -> Monday 
1 -> Thues
0: 'Monday'
1: 'Tuesday'
2: 'Wednesday'
3: 'Thursday'
4: 'Friday'
5: 'Saturday'
6: 'Sunday'
(5,6->weekend)
"""

# rush_hour, weekend
train['rush_hour'] = train['pickup_hour'].apply(
    lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0)

train['weekend'] = train['pickup_day'].apply(
    lambda x: 1 if x >= 5 else 0)

# for test data:
test['rush_hour'] = test['pickup_hour'].apply(
    lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0)

test['weekend'] = test['pickup_day'].apply(
    lambda x: 1 if x >= 5 else 0)




#%pip install prophet


# Time Serics Forecasting with: **Prophet**
from prophet import Prophet



""" 
**New Work City divided into five boroughs:**
- the Bronx
- Brooklyn
- Manhattan
- Queens 
- Staten Island
**Conculation: the number of cluster should be 5**
"""
from sklearn.cluster import KMeans 



# pickup cluster:
kmeans_pickup = KMeans(n_clusters=5,random_state=42,verbose=False)
kmeans_pickup.fit(train[["pickup_latitude","pickup_longitude"]])



kmeans_pickup.inertia_



# Expriment what should be number of cluster:
# wcss -> Within Cluster Some of Squared Distance
# randomstate is must---> Cause, kmean-select centriod randomly
wcss = []
for i in range(1,16):
    kmean = KMeans(n_clusters=i,random_state=42)
    kmean.fit_transform(train[["pickup_latitude","pickup_longitude"]])
    wcss.append(kmean.inertia_)
wcss 



#Elbow Curve: 
# akdom beci nile overfit korbe
# best value between: (6 to 8)
#
plt.title("Elbo Method -> Elbo Curve")
plt.grid(True)
sns.lineplot(x=np.arange(start=1,stop=16,step=1),y=wcss)
plt.show()



#<----------kmean dropout cluster------------------->
kmeans_dropout = KMeans(n_clusters=5,random_state=42,verbose=False)
kmeans_dropout.fit(train[["dropoff_latitude","dropoff_longitude"]])




train_pickup_cluster = kmeans_pickup.predict(train[["pickup_latitude","pickup_longitude"]])
train_dropout_cluster = kmeans_dropout.predict(train[["dropoff_latitude","dropoff_longitude"]])



# for test data:
test_dropout_cluster = kmeans_dropout.predict(test[["dropoff_latitude","dropoff_longitude"]])
test_pickup_cluster = kmeans_pickup.predict(test[["pickup_latitude","pickup_longitude"]])




#for train dataset:
train["pickup_cluster"] = train_pickup_cluster
train["dropout_cluster"] = train_dropout_cluster

#for test dataset:
test["dropout_cluster"] = test_dropout_cluster
test["pickup_cluster"] = test_pickup_cluster



train["pickup_cluster"].unique() , test["pickup_cluster"].unique()



# we have many values in train data, what's why we have 5 clusters.
# But, in our test data our we have less value, that's why we have 4 clusters.
train["dropout_cluster"].unique(), test["dropout_cluster"].unique() 



train["manhattan_distance"] = np.abs(train["pickup_latitude"] - train["dropoff_latitude"]) + np.abs(
    train["pickup_longitude"]- train["dropoff_longitude"])

test["manhattan_distance"] = np.abs(test["pickup_latitude"] - test["dropoff_latitude"]) + np.abs(
    test["pickup_longitude"]- test["dropoff_longitude"])




""" 
- But we need number:
"""
pickup_cluster = pd.get_dummies(train["pickup_cluster"])
pickup_cluster[:5]



pickup_cluster_train = pd.get_dummies(train["pickup_cluster"],prefix="pickup_cluster")
pickup_cluster_test = pd.get_dummies(test["pickup_cluster"],prefix="pickup_cluster")




pickup_cluster_one_train = pd.get_dummies(train["pickup_cluster"],dtype=float,prefix="pickup_cluster")
pickup_cluster_one_test = pd.get_dummies(test["pickup_cluster"],dtype=float,prefix="pickup_cluster")


# test data mayn't have same number of cluster: 
full_columns = [f"pickup_cluster_{i}" for i in range(5)]  # ['pickup_cluster_0', ..., 'pickup_cluster_4']

# Reindex both to add missing columns with 0
pickup_cluster_one_train = pickup_cluster_one_train.reindex(columns=full_columns, fill_value=0.0)
pickup_cluster_one_test = pickup_cluster_one_test.reindex(columns=full_columns, fill_value=0.0)





# one of other columns: 
dropout_cluster_one_train = pd.get_dummies(train["dropout_cluster"],dtype=float,prefix="dropout_cluster")
store_and_fwd_flag_one_train = pd.get_dummies(train["store_and_fwd_flag"],dtype=float,prefix="store_and_fwd_flag")
passenger_count_one_train = pd.get_dummies(train["passenger_count"],dtype=float,prefix="passenger_count")
vendor_id_one_train = pd.get_dummies(train["vendor_id"],dtype=float,prefix="vendor_id")

# for test:
# one of other columns: 
dropout_cluster_one_test = pd.get_dummies(test["dropout_cluster"],dtype=float,prefix="dropout_cluster")
store_and_fwd_flag_one_test = pd.get_dummies(test["store_and_fwd_flag"],dtype=float,prefix="store_and_fwd_flag")
passenger_count_one_test = pd.get_dummies(test["passenger_count"],dtype=float,prefix="passenger_count")
vendor_id_one_test = pd.get_dummies(test["vendor_id"],dtype=float,prefix="vendor_id")


# same for test data: 
# Define all possible columns (for 5 clusters)
full_columns = [f"dropout_cluster_{i}" for i in range(5)]  # ['pickup_cluster_0', ..., 'pickup_cluster_4']

# Reindex both to add missing columns with 0
dropout_cluster_one_train = dropout_cluster_one_train.reindex(columns=full_columns, fill_value=0.0)
dropout_cluster_one_test = dropout_cluster_one_test.reindex(columns=full_columns, fill_value=0.0)


dropout_cluster_one_train.columns


dropout_cluster_one_test.columns


train.info()


""" 
"vendor_id",
"store_and_fwd_flag", -->object
"passenger_count",
"""
# remove unwanted columns:
train_df = train.drop(columns=["pickup_cluster","dropout_cluster",
                               "pickup_day_name",
                               "pickup_datetime","dropoff_datetime","date",
                               "id","store_and_fwd_flag",
                               "trip_duration_min","trip_duration_hours","anomaly"],axis=1)

test_df = test.drop(columns=["pickup_cluster","dropout_cluster",
                             "pickup_day_name",
                             "pickup_datetime","date",
                             "id","store_and_fwd_flag"],axis=1)

train_df.dtypes


train_df.describe()



train_df = train_df.join(pickup_cluster_one_train)
test_df = test_df.join(pickup_cluster_one_test)

train_df = train_df.join(dropout_cluster_one_train)
test_df = test_df.join(dropout_cluster_one_test)

train_df = train_df.join(store_and_fwd_flag_one_train)
test_df = test_df.join(store_and_fwd_flag_one_test)





#train_df = train_df.join(passenger_count_one)



#train_df = train_df.join(vendor_id_one)


train_df.columns


test_df.columns



# seperate X,y for tranning data:
X  = train_df.drop(columns=["trip_duration"])
y = train["trip_duration"]


X.columns


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)



print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")



from sklearn.linear_model import SGDRegressor

regressor = SGDRegressor(verbose=False,eta0=0.00001,learning_rate="adaptive")
regressor.fit(X_train,y_train)



from sklearn.metrics import r2_score
y_pred = regressor.predict(X_test)
r2_score(y_test,y_pred)


X_train.columns


test_df.columns


y_pred = regressor.predict(test_df)


y_pred = regressor.predict(X_train)
r2_score(y_train,y_pred)


# import optuna 
# import lightgbm as lgb
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.model_selection import cross_val_score

# def Objective(trial):
#     classifier_name = trial.suggest_categorical(
#         "regressor", 
#         ['RandomForestRegressor', 'GradientBoostingRegressor', 'LGBMRegressor']
#     )

#     if classifier_name == 'RandomForestRegressor':
#         n_estimators = trial.suggest_int("n_estimators", 50, 300)
#         max_depth = trial.suggest_int("max_depth", 3, 20)
#         min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
#         min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
#         bootstrap = trial.suggest_categorical("bootstrap", [True, False])
#         model = RandomForestRegressor(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             bootstrap=bootstrap,
#             random_state=42
#         )

#     elif classifier_name == 'GradientBoostingRegressor':
#         n_estimators = trial.suggest_int("n_estimators", 50, 300)
#         learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
#         max_depth = trial.suggest_int("max_depth", 3, 20)
#         min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
#         min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
#         model = GradientBoostingRegressor(
#             n_estimators=n_estimators,
#             learning_rate=learning_rate,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             random_state=42
#         )
    
#     elif classifier_name == 'LGBMRegressor':
#         n_estimators = trial.suggest_int("n_estimators", 100, 500)
#         learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
#         num_leaves = trial.suggest_int("num_leaves", 2, 256)
#         max_depth = trial.suggest_int("max_depth", 3, 20)
#         model = lgb.LGBMRegressor(
#             n_estimators=n_estimators,
#             learning_rate=learning_rate,
#             num_leaves=num_leaves,
#             max_depth=max_depth,
#             random_state=42
#         )

#     #cross val score: 
#     score = cross_val_score(
#         model,
#         X_train,
#         y_train,
#         cv=3,
#         scoring="r2" 
#     ).mean()

#     return score



#!pip install 'lightgbm[scikit-learn]'



import lightgbm as lgb 

# <-------------------Model with LGBMRegressor--------------------------->
classifier = lgb.LGBMRegressor(n_estimators=300,
                  learning_rate=0.085,
                  num_leaves=248,
                  max_depth=10,
                  n_jobs=-1,
                  verbose=1
            )

classifier.fit(X_train,y_train)




y_pred = classifier.predict(X_train)
r2_score(y_train,y_pred)



y_pred = classifier.predict(X_test)
r2_score(y_test,y_pred)



submission.columns



# y_pred = classifier.predict(test_df)
# r2_score(submission["trip_duration"],y_pred)


