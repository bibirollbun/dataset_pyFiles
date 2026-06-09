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


# Import required libraries 
import matplotlib.pyplot as plt 
import seaborn as sns 


from pathlib import Path 
from zipfile import ZipFile 


# Path for the dataset 
train_path = "/kaggle/input/nyc-taxi-trip-duration/train.zip"
test_path = "/kaggle/input/nyc-taxi-trip-duration/test.zip"


# output path for the zip files
def extract_zip_file(filepath):
    output_path = Path('/kaggle/working/') / 'raw'
    output_path.mkdir(parents = True, exist_ok = True)
    with ZipFile(file = filepath) as f:
        f.extractall(path = output_path)


# extract the train file 
extract_zip_file(train_path)

# extract the test file 
extract_zip_file(test_path)


# Read the csv files

train_df = pd.read_csv("/kaggle/working/raw/train.csv")
test_df = pd.read_csv("/kaggle/working/raw/test.csv")


print(f"The shape of the train file is : {train_df.shape}")
print(f"The shape of the test file is : {test_df.shape}")


train_df.sample(25)


# Check for duplicates in train_df

train_df.isnull().sum()


# Check duplicates in test_df 

test_df.isnull().sum()


# Columns to apply describe function 
cols_to_describe_num = train_df.columns[[5, 6, 7, 8, 10]]

cols_to_describe_num


train_df.columns[4:].to_list() + train_df.columns[[1]].to_list()


# Check duplicates values in this selected columns from data 
train_df.duplicated(subset = (train_df.columns[[1]].to_list() + train_df.columns[4:].to_list())).sum()


# Use describe function on "cols_to_describe_num"
train_df[cols_to_describe_num].describe()


# Calculate Q3 value 
(1.075000e+03 * 60) / 3600

# The Q3 of the data is around 17-18 minutes which shows that people in NYC prefer shorter cab trips.


# Statistical summary for the categorical columns 
cols_to_describe_cat = [col for col in train_df.columns if col not in cols_to_describe_num]
cols_to_describe_cat = np.array(cols_to_describe_cat)[[1, 4, 5]]

cols_to_describe_cat


# value counts of the each categorical columns 

for col in cols_to_describe_cat:
    print(f" The unique value in : {col} are ---> ", np.sort(train_df[col].unique()))
    print(train_df[col].value_counts())
    print("*" * 50, end = '\n')


# Check data type of train data 

train_df.dtypes


# information about the train_df 

train_df.info(memory_usage = 'deep', show_counts = False)


# Convert the big sized integer oclumns to small size 

#train_df['vendor_id'] = train_df['vendor_id'].astype(np.int8)

# train_df['passenger_count'] = train_df['vendor_id'].astype(np.int32)


train_df.info(memory_usage = 'deep', show_counts = False)


# train_df columns 
train_df.columns


train_df.dtypes


# Statistical description of the target column 

target_col = 'trip_duration'

train_df[target_col].describe().reset_index()


# Number of trips in the target column that have durations less than 1 min or 60 sec..

train_df[target_col].loc[train_df[target_col] <= 60].size


# distribution of target column of trips less than a minute

sns.violinplot(x= train_df[target_col].loc[train_df[target_col] <= 60])


# trip durations equal to 1 sec

train_df.loc[train_df[target_col] == 1].size


# Convert the trip duration col to minutes for easier understanding 

target_col_minutes = train_df[target_col] / 60

target_col_minutes.describe().reset_index()


# distribution

import matplotlib.pyplot as plt  
import seaborn as sns

sns.kdeplot(target_col_minutes)


(60000 / 60) / 24


# box plot 

sns.boxplot(target_col_minutes)


# extreme points in the data (results in hours)

target_col_minutes[target_col_minutes > 5000] / 60


# extreme points in the complete data 

train_df[target_col_minutes > 5000]


# drop the extreme value points from the data 
target_col_minutes.drop(index = target_col_minutes[target_col_minutes > 5000].index, inplace = True)


sns.kdeplot(target_col_minutes)


1400 / 60


target_col_minutes.describe().reset_index()


# box plot

sns.boxplot(target_col_minutes)


target_col_hour = train_df['trip_duration'] / 3600

target_col_hour


sns.kdeplot(target_col_hour)


sns.boxplot(target_col_hour)


# calculate the upper limit based on IQR approach 

Q1_target, Q3_target = target_col_hour.quantile([0.25, 0.75])

IQR = Q3_target - Q1_target


print(f'Q1 = {Q1_target * 60:.2f} minutes    ', f'Q3 = {Q3_target * 60:.2f} minutes')
print(f'IQR = {IQR * 60:.2f} minutes')

upper_bound = Q3_target + (1.5 * IQR)

print(f'Upper bound = {upper_bound * 60:.2f} minutes')


# distribution of pickup latitude and longitude of above upper limit 
x_temp = train_df.loc[(target_col_hour > upper_bound),:].copy()

x_temp.loc[:,'trip_duration'] = x_temp['trip_duration'] / 3600


sns.boxplot(x_temp,y = 'trip_duration')


# remove the extreme points

drop_indices = x_temp[x_temp['trip_duration'] > 100].index

drop_indices


# drop extreme points 
x_temp.drop(index = drop_indices, inplace = True)


sns.boxplot(x_temp, y = 'trip_duration')


# boxplot of latitude 

fig = plt.figure(figsize = (12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x_temp, y = 'pickup_latitude')
plt.subplot(1, 2, 2)
sns.boxplot(x_temp, y = 'dropoff_latitude')


# boxplot of longitude 

fig = plt.figure(figsize = (12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x_temp, y = 'pickup_longitude')
plt.subplot(1, 2, 2)
sns.boxplot(x_temp, y = 'dropoff_longitude')


# class to detect outliers 

def detect_outliers(data, columns, fold = 1.5):
    data_temp = data.copy()
    new_df = pd.DataFrame(columns = columns)
    for col in columns:
        Q1, Q3 = data_temp[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        upper_bound = Q3 + (fold * IQR)
        lower_bound = Q1 - (fold * IQR)
        filter_data = data_temp.loc[(data_temp[col] <= lower_bound) | (data_temp[col] >= upper_bound)]
        new_df = pd.concat([new_df, filter_data])

    return new_df


detect_outliers(x_temp, columns = ['pickup_latitude'])


detect_outliers(x_temp, columns = ['dropoff_latitude'], fold = 5)


x_temp['passenger_count'].value_counts()


sns.countplot(x_temp, x = 'passenger_count')


# passenger_count and vendor_id

pd.crosstab(x_temp['passenger_count'], columns = x_temp['vendor_id'], normalize = 'columns') * 100


# trip duration summary

x_temp.pivot_table(index = 'passenger_count', columns = 'vendor_id', values = 'trip_duration', aggfunc = ['min','max','median','mean'])


sns.countplot(x_temp, x = 'vendor_id')


# avg trip duration in hours based on passenger count 

sns.barplot(x_temp, x = 'passenger_count', y = 'trip_duration', hue = 'vendor_id')


# aggregate stats
target_col_minutes.agg(func = ['mean', 'median', 'min', 'max'])


# print the percentile value in the data 
percentiles = np.arange(0.9, 1, 0.01)
res_list = []
for quant in percentiles:
    res = target_col_minutes.quantile(quant)
    res_list.append(res)
    print(f'The value for percentile = {quant * 100:.2f} : {res :.2f} minutes')


# plot 
plt.plot(percentiles * 100, res_list)
plt.xlabel('Percentile')
plt.ylabel('Value')


# print the percentile value in the data 
percentiles = np.arange(0.99, 1, 0.001)
res_list = []
for quant in percentiles:
    res = target_col_minutes.quantile(quant)
    res_list.append(res)
    print(f'The value for percentile = {quant * 100:.2f} : {res :.2f} minutes')


# plot 
plt.plot(percentiles * 100, res_list)
plt.xlabel('Percentile')
plt.ylabel('Value')


# print the percentile value in the data 
percentiles = np.arange(0.998, 1, 0.0001)
res_list = []
for quant in percentiles:
    res = target_col_minutes.quantile(quant)
    res_list.append(res)
    print(f'The value for percentile = {quant * 100:.2f} : {res :.2f} minutes')


# plot 
plt.figure(figsize = (8, 4))
plt.plot(np.round((percentiles * 100),1), res_list)
plt.xlabel('Percentile')
plt.ylabel('Value')


# number of data points above or equal to the 100 minutes mark 
time_ranges = np.arange(100, 1100, 100)
prev_val = 0
for time in time_ranges:
    new_val = target_col_minutes[target_col_minutes >= time].size
    print(f'{time} = {new_val},   diff = {np.abs(new_val - prev_val)}')
    prev_val = new_val


train_df['trip_duration'] / 60


normal_duration_df = train_df.loc[(train_df['trip_duration'] / 60) < 100, :].copy()

normal_duration_df



# number of rows in data with trip duration higher than 100 min 

(target_col_minutes > 100).sum()


!mkdir /kaggle/working/data-without-outliers


# save the data where the trip duration (target column) is till 100 min
df_path = ('/kaggle/working/data-without-outliers/train.csv')

normal_duration_df.to_csv(df_path)


# drop and pickups of points where time was more than 100

extreme_time_duration_df = train_df.loc[(train_df['trip_duration'] / 60) >= 100, :]

extreme_time_duration_df


# Border of new york 
latitude_coord = train_df['pickup_latitude'].quantile([0.01, 0.99]).sort_values(ascending = False).values + np.array([0.05,-0.05])

longitude_coord = train_df['pickup_longitude'].quantile([0.01, 0.99]).sort_values(ascending = False).values + np.array([0.1,-0.1])


# Heatmap of pickups and dropoffs where trip duration was greater than 100 min

fig,ax = plt.subplots(nrows = 1, ncols = 2, sharey = True, figsize = (15, 6))

ax[0].set_facecolor('white')
ax[0].scatter(extreme_time_duration_df['pickup_latitude'], extreme_time_duration_df['pickup_longitude'], s = 0.1, color = 'navy')
ax[0].set_ylim(longitude_coord)
ax[0].set_xlim(latitude_coord)
ax[0].set_title('Heatmap for Pickups')
ax[0].set_xlabel('Latitude')
ax[0].set_ylabel('Longitude')



ax[1].set_facecolor('white')
ax[1].scatter(extreme_time_duration_df['dropoff_latitude'], extreme_time_duration_df['dropoff_longitude'], s = 0.1, color = 'navy')
ax[1].set_ylim(longitude_coord)
ax[1].set_xlim(latitude_coord)
ax[1].set_title('Heatmap for Dropoffs')
ax[1].set_xlabel('Latitude')
ax[1].set_ylabel('Longitude')


# Heatmap of pickups and dropoffs where trip duration was less than 100 min

fig,ax = plt.subplots(nrows = 1, ncols = 2, sharey = True, figsize = (15, 6))

ax[0].set_facecolor('white')
ax[0].scatter(normal_duration_df['pickup_latitude'], normal_duration_df['pickup_longitude'], s = 0.1, color = 'navy')
ax[0].set_ylim(longitude_coord)
ax[0].set_xlim(latitude_coord)
ax[0].set_title('Heatmap for Pickups')
ax[0].set_xlabel('Latitude')
ax[0].set_ylabel('Longitude')



ax[1].set_facecolor('white')
ax[1].scatter(normal_duration_df['dropoff_latitude'], normal_duration_df['dropoff_longitude'], s = 0.1, color = 'navy')
ax[1].set_ylim(longitude_coord)
ax[1].set_xlim(latitude_coord)
ax[1].set_title('Heatmap for Dropoffs')
ax[1].set_xlabel('Latitude')
ax[1].set_ylabel('Longitude')


# distribution of target col when max time duration is less than or equals 100 min

sns.boxplot(target_col_minutes[target_col_minutes <= 100])


# passenger counts for normal_duration_df
normal_duration_df['passenger_count'].unique()


# passenger counts for extreme_time_duration_df

extreme_time_duration_df['passenger_count'].unique()


# Let's divide the particular trip duration with the 60 min
normal_duration_df.loc[:, 'trip_duration'] = normal_duration_df['trip_duration'] / 60

extreme_time_duration_df.loc[:, 'trip_duration'] = extreme_time_duration_df['trip_duration'] / 60


# boxplot based on passenger_count 

sns.boxplot(normal_duration_df, x = 'passenger_count', y = 'trip_duration')


sns.boxplot(extreme_time_duration_df, x = 'passenger_count', y = 'trip_duration')


sns.kdeplot(normal_duration_df.loc[normal_duration_df['passenger_count'] == 0, :], x = 'trip_duration', label = 'zero count')
sns.kdeplot(normal_duration_df.loc[~(normal_duration_df['passenger_count'] == 0), :], x = 'trip_duration', label = 'non-zero count')

plt.legend()
plt.show()


passenger_count_range = list(range(0,7))

sns.kdeplot(normal_duration_df.loc[normal_duration_df['passenger_count'].isin(passenger_count_range), :], x = 'trip_duration', hue = 'passenger_count', palette = 'viridis')


# kdeplot for distribution based on passenger_count (standardized)

passenger_count_grp = normal_duration_df.groupby('passenger_count')

# standardized each distribution

temp_df = pd.DataFrame()
temp_df['passenger_count'] = normal_duration_df['passenger_count'].copy()

temp_df['trip_duration'] = passenger_count_grp['trip_duration'].transform(lambda x: (x - x.mean()) / x.std())

passenger_count_range = list(range(1, 7))

sns.kdeplot(temp_df.loc[temp_df['passenger_count'].isin(passenger_count_range),:], x = 'trip_duration', hue = 'passenger_count', palette = 'viridis')


# remove extreme values 
extreme_time_duration_df = extreme_time_duration_df.loc[extreme_time_duration_df['trip_duration'] < 10000,:]


# boxplot based on passenger count 

sns.boxplot(extreme_time_duration_df, x = 'passenger_count', y = 'trip_duration')



normal_duration_df['trip_duration'].skew()


# apply box cox transformation 

from sklearn.preprocessing import PowerTransformer

pt_target = PowerTransformer(method = 'box-cox', standardize = False)

# transform output column

trip_duration_trans = pt_target.fit_transform(normal_duration_df[['trip_duration']])

sns.kdeplot(trip_duration_trans)


train_df['id'].head()


# remove the id tag from the values
def remove_id_tag(data):
    data['id'] = data['id'].str.replace('id', '')
    return data['id']


# remove tag from train data 
remove_id_tag(train_df)


# remove tag from the test data
remove_id_tag(test_df)


# convert id column to int
train_df['id'] = train_df['id'].astype('int')
test_df['id'] = test_df['id'].astype('int')


# test_id column is required
def test_id_column(data):
    if data.shape[0] == data['id'].nunique():
        new_data = data.drop(columns = 'id')
        return new_data
    else:
        return data


train_df_new = test_id_column(train_df)
test_df_new = test_id_column(test_df)


train_df_new


test_df_new


# categories in the vendor_id column 
train_df_new['vendor_id'].value_counts().sort_index()


# pie chart showing distribution 
fig = plt.figure(figsize = (12, 6))
plt.pie(train_df_new['vendor_id'].value_counts().sort_index(), autopct = '%.2f%%',explode = [0.2,0], shadow = True, labels = ['Vendor 1', 'Vendor 2'], labeldistance = 0.2)
plt.title('Distribution of Rides according to vendor_id')
plt.show()


# change the column to datetime
train_df_new['pickup_datetime'] = pd.to_datetime(train_df_new['pickup_datetime'])
test_df_new['pickup_datetime'] = pd.to_datetime(test_df_new['pickup_datetime'])


train_df_new.shape


train_df_new['pickup_datetime'].dt.weekday


# feature extraction 
def make_datetime_features(data,column_type):
    data[f'{column_type}_hour'] = data[f'{column_type}_datetime'].dt.hour
    data[f'{column_type}_date'] = data[f'{column_type}_datetime'].dt.day
    data[f'{column_type}_month'] = data[f'{column_type}_datetime'].dt.month
    data[f'{column_type}_day'] = data[f'{column_type}_datetime'].dt.weekday
    data[f'is_weekend'] = data.apply(lambda row: row[f'{column_type}_day'] >= 5, axis = 1).astype('int')
    return data


# datetime features for pickup column in train data
make_datetime_features(train_df_new, 'pickup')


# datetime features for pickup column in test data 
make_datetime_features(test_df_new, 'pickup')


train_df_new.shape


train_df_new.head()


# save the data 
# make the dir as datetime 
save_path = Path('/kaggle/working/') / 'datetime'
save_path.mkdir(parents = True, exist_ok = True)

# save the train file 
train_df_new.to_csv(save_path / 'train.csv')

# save the test file 
test_df_new.to_csv(save_path / 'test.csv')


# time range for the train data 
train_df_new['pickup_datetime'].max() - train_df_new['pickup_datetime'].min()


# time range for the test data
test_df_new['pickup_datetime'].max() - test_df_new['pickup_datetime'].min()


# time range for the train and test data
data_temp = {'Opening_date': [train_df_new['pickup_datetime'].min(), test_df_new['pickup_datetime'].min()],
            'CLosing_date': [train_df_new['pickup_datetime'].max(), test_df_new['pickup_datetime'].min()]}

pd.DataFrame(data_temp, index = ['train', 'test']).T


# total pickups each day

train_df_new['pickup_datetime'].dt.date.value_counts().sort_index()


# pickup patterns for train and test data
plt.plot(train_df_new['pickup_datetime'].dt.date.value_counts().sort_index(), color = 'green', label = 'train')
plt.plot(test_df_new['pickup_datetime'].dt.date.value_counts().sort_index(), color = 'red', label = 'test')
plt.legend()
plt.show()


# pickup patterns for train and test data (normalized) --> equalize the scales
plt.plot(train_df_new['pickup_datetime'].dt.date.value_counts(normalize = True).sort_index(), color = 'green',alpha = 0.7, label = 'train')
plt.plot(test_df_new['pickup_datetime'].dt.date.value_counts(normalize = True).sort_index(), color = 'red', label = 'test')
plt.legend()
plt.show()


train_df_new.dtypes


# avg daily pickups

train_df_new['pickup_datetime'].dt.date.value_counts().mean()


# avg pickups daily across all months

month_group = train_df_new.groupby(by = 'pickup_month')

daily_pickups_mean = month_group['pickup_date'].value_counts().mean()

daily_pickups_mean


# taxi trips instead of travel bans

ban_df = train_df_new.loc[(train_df_new['pickup_date'].isin(list(range(20,31))) & (train_df_new['pickup_month'] == 1))]

ban_df['pickup_date'].value_counts().sort_index().plot()


# no. of days when pickups were below the daily avg

unique_dates = train_df_new['pickup_datetime'].dt.date.value_counts()

below_daily_avg_count = unique_dates[unique_dates < daily_pickups_mean].size

percentage_below_avg = below_daily_avg_count / unique_dates.index.size

print(f'The number of days where number of pickups is below the daily avg is {below_daily_avg_count} which is {(percentage_below_avg * 100):.2f}% of the total days')


from IPython.display import HTML

def horizontal(dfs):
    html = '<div style="display:flex">'

    for df in dfs:
        html += '<div style="margin-right:32px">'
        html += df.to_html()
        html += '</div>'
    html += '</div>'

    display(HTML(html))


temp_df = unique_dates[unique_dates < daily_pickups_mean].reset_index().sort_values('pickup_datetime')

temp_df['pickup_datetime'] = pd.to_datetime(temp_df['pickup_datetime'])
temp_df['month'] = temp_df['pickup_datetime'].dt.month

dfs_to_display = []

for month in np.sort(train_df_new['pickup_month'].unique()):
    filter_month = temp_df['month'] == month
    dfs_to_display.append(temp_df.loc[filter_month,['pickup_datetime','count']].reset_index(drop = True))


horizontal(dfs_to_display)


ban_df['pickup_date'].value_counts().sort_values()


train_df_new['pickup_month'].unique()


# number of pickups on travel ban days (23rd & 24th Jan,2016)

ban_pickups = ban_df['pickup_date'].value_counts().sort_values()


print(f'Number of Pickups on banned days are {ban_pickups[[23,24]].sum()}')


# borders of new york 

latitude_coord = train_df_new['pickup_latitude'].quantile([0.01, 0.99]).sort_values(ascending = False).values + np.array([0.05, -0.05])

longitude_coord = train_df_new['pickup_longitude'].quantile([0.01, 0.99]).sort_values(ascending = False).values + np.array([0.1, -0.1])



longitude_coord


latitude_coord


# Heatmap of pickups and dropoffs during the travel ban
ban_df = ban_df.loc[ban_df['pickup_date'].isin([23,24])]

fig,ax = plt.subplots(nrows = 1, ncols = 2, sharey = True, figsize = (15, 6))

ax[0].set_facecolor('white')
ax[0].scatter(ban_df['pickup_latitude'], ban_df['pickup_longitude'], s = 0.1, color = 'navy')
ax[0].set_ylim(longitude_coord)
ax[0].set_xlim(latitude_coord)
ax[0].set_title('Heatmap for Pickups')
ax[0].set_xlabel('Latitude')
ax[0].set_ylabel('Longitude')



ax[1].set_facecolor('white')
ax[1].scatter(ban_df['dropoff_latitude'], ban_df['dropoff_longitude'], s = 0.1, color = 'navy')
ax[1].set_ylim(longitude_coord)
ax[1].set_xlim(latitude_coord)
ax[1].set_title('Heatmap for Dropoffs')
ax[1].set_xlabel('Latitude')
ax[1].set_ylabel('Longitude')


# count of total pickups each month

sns.countplot(train_df_new, x = 'pickup_month')


# pickups month wise in sorted order 

train_df_new['pickup_month'].value_counts().sort_values(ascending = False).plot(kind = 'bar')


# rush hours for taxi pickups

sns.countplot(train_df_new, x = 'pickup_hour')


# rush hours divided into 3 clusters. (high rush, avg rush, low rush)

cluster_input = train_df_new['pickup_hour'].value_counts().sort_index().reset_index()


cluster_input.plot(x = 'pickup_hour', y = 'count', kind = 'scatter')


cluster_input


from scipy.cluster.hierarchy import dendrogram, linkage

clustering = linkage(y = cluster_input.values, method = 'single')

dendrogram(clustering)

plt.show()



from sklearn.cluster import AgglomerativeClustering, KMeans

agglo = AgglomerativeClustering(n_clusters = 2, linkage = 'single', metric = 'euclidean')

agglo


# plot the clusters

predictions = agglo.fit_predict(cluster_input)

sns.scatterplot(cluster_input, x = 'pickup_hour', y = 'count', hue = np.where(predictions == 0, 'High Rush Hour', 'Low Rush Hour'))

plt.show()


from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


# elbow plot for the clusters along with silhouette score 

n_clusters = [1, 2, 3, 4, 5,6]
inertia_kmeans = []

for n in n_clusters:
    kmeans = KMeans(n_clusters = n, n_init = 10, max_iter = 50)
    # fit the data
    predictions = kmeans.fit_predict(cluster_input)
    # append the wcss in the list 
    inertia_kmeans.append(kmeans.inertia_)
    # calculate the silhouette score 
    if n < 2:
        continue
    else:
        score = silhouette_score(X = cluster_input, labels = predictions)
        print(f'The silhouette score for {n} clusters is {score}')


plt.plot(n_clusters, inertia_kmeans, color = 'red')
plt.title('Elbow Curve')
plt.xlabel('No. of Clusters')
plt.ylabel('WCSS')
plt.show()


# train kmeans for 2 cluster 

kmeans = KMeans(n_clusters = 2, n_init = 10, max_iter = 50)

predictions = kmeans.fit_predict(cluster_input)


# get the centroids and plot on the curve
cluster_centroids = kmeans.cluster_centers_

# plot the scatter chart
sns.scatterplot(cluster_input, x = 'pickup_hour', y = 'count', hue = np.where(predictions == 0, 'High Rush Hour', 'Low Rush Hour'))


plt.scatter(cluster_centroids[:,0], cluster_centroids[:,1], marker = '+', s = 60, c = 'k', label = 'Avg Pickups')
plt.legend()
plt.show()


cluster_centroids = kmeans.cluster_centers_

cluster_centroids


# get the centroids and plot on the curve
cluster_centroids = kmeans.cluster_centers_

# plot the scatter chart
sns.scatterplot(cluster_input, x = 'pickup_hour', y = 'count', hue = np.where(predictions == 0, 'High Rush Hour', 'Low Rush Hour'))


plt.scatter(cluster_centroids[:,0], cluster_centroids[:,1], marker = '+', s = 60, c = 'k', label = 'Avg Pickups')
plt.axhline(y = cluster_centroids[:,1].mean(), linestyle = '--', color= 'red', label = 'Avg')
plt.legend()
plt.show()


# we can also try double groupby applications

weekend_grp = train_df_new.groupby('is_weekend')

weekend_grp['pickup_day'].value_counts().groupby('is_weekend').mean()


# pickup patterns on hour of the day for weekdays and weekends

def calculate_day_avg(group):
    value_counts = group['pickup_hour'].value_counts()
    number_of_days = group['pickup_day'].nunique()
    group_avg = value_counts / number_of_days
    return group_avg


weekend_grp.apply(calculate_day_avg).unstack(level = 0).plot(kind = 'bar')
plt.show()


train_df_new.dtypes


# Heatmap of pickups and dropoffs during the weekdays
temp_df = train_df_new.loc[train_df_new['is_weekend'] == 0, :]

fig,ax = plt.subplots(nrows = 1, ncols = 2, sharey = True, figsize = (15, 6))

ax[0].set_facecolor('white')
ax[0].scatter(temp_df['pickup_latitude'], temp_df['pickup_longitude'], s = 0.05, color = 'navy')
ax[0].set_ylim(longitude_coord)
ax[0].set_xlim(latitude_coord)
ax[0].set_title('Heatmap for Pickups')
ax[0].set_xlabel('Latitude')
ax[0].set_ylabel('Longitude')



ax[1].set_facecolor('white')
ax[1].scatter(temp_df['dropoff_latitude'], temp_df['dropoff_longitude'], s = 0.05, color = 'navy')
ax[1].set_ylim(longitude_coord)
ax[1].set_xlim(latitude_coord)
ax[1].set_title('Heatmap for Dropoffs')
ax[1].set_xlabel('Latitude')
ax[1].set_ylabel('Longitude')


# Heatmap of pickups and dropoffs during the weekend
temp_df = train_df_new.loc[train_df_new['is_weekend'] == 1, :]

fig,ax = plt.subplots(nrows = 1, ncols = 2, sharey = True, figsize = (15, 6))

ax[0].set_facecolor('white')
ax[0].scatter(temp_df['pickup_latitude'], temp_df['pickup_longitude'], s = 0.05, color = 'navy')
ax[0].set_ylim(longitude_coord)
ax[0].set_xlim(latitude_coord)
ax[0].set_title('Heatmap for Pickups')
ax[0].set_xlabel('Latitude')
ax[0].set_ylabel('Longitude')



ax[1].set_facecolor('white')
ax[1].scatter(temp_df['dropoff_latitude'], temp_df['dropoff_longitude'], s = 0.05, color = 'navy')
ax[1].set_ylim(longitude_coord)
ax[1].set_xlim(latitude_coord)
ax[1].set_title('Heatmap for Pickups')
ax[1].set_xlabel('Latitude')
ax[1].set_ylabel('Longitude')


# Heatmap of pickups during the weekdays & weekend
temp_df1 = train_df_new.loc[train_df_new['is_weekend'] == 0, :]
temp_df2 = train_df_new.loc[train_df_new['is_weekend'] == 1, :]

fig,ax = plt.subplots(nrows = 1, ncols = 2, sharey = True, figsize = (15, 6))

ax[0].set_facecolor('white')
ax[0].scatter(temp_df1['pickup_latitude'], temp_df1['pickup_longitude'], s = 0.05, color = 'navy')
ax[0].set_ylim(longitude_coord)
ax[0].set_xlim(latitude_coord)
ax[0].set_title('Heatmap for Pickups on weekdays')
ax[0].set_xlabel('Latitude')
ax[0].set_ylabel('Longitude')



ax[1].set_facecolor('white')
ax[1].scatter(temp_df2['pickup_latitude'], temp_df2['pickup_longitude'], s = 0.05, color = 'navy')
ax[1].set_ylim(longitude_coord)
ax[1].set_xlim(latitude_coord)
ax[1].set_title('Heatmap for Pickups on weekends')
ax[1].set_xlabel('Latitude')
ax[1].set_ylabel('Longitude')


# Heatmap of dropoff during the weekdays & weekend
temp_df1 = train_df_new.loc[train_df_new['is_weekend'] == 0, :]
temp_df2 = train_df_new.loc[train_df_new['is_weekend'] == 1, :]

fig,ax = plt.subplots(nrows = 1, ncols = 2, sharey = True, figsize = (15, 6))

ax[0].set_facecolor('white')
ax[0].scatter(temp_df1['dropoff_latitude'], temp_df1['dropoff_longitude'], s = 0.05, color = 'navy')
ax[0].set_ylim(longitude_coord)
ax[0].set_xlim(latitude_coord)
ax[0].set_title('Heatmap for Dropoff on weekdays')
ax[0].set_xlabel('Latitude')
ax[0].set_ylabel('Longitude')



ax[1].set_facecolor('white')
ax[1].scatter(temp_df2['dropoff_latitude'], temp_df2['dropoff_longitude'], s = 0.05, color = 'navy')
ax[1].set_ylim(longitude_coord)
ax[1].set_xlim(latitude_coord)
ax[1].set_title('Heatmap for Dropoff on weekends')
ax[1].set_xlabel('Latitude')
ax[1].set_ylabel('Longitude')


# distribution of the column 
sns.kdeplot(train_df_new['pickup_latitude'], label = 'pickup')
sns.kdeplot(train_df_new['dropoff_latitude'], label = 'dropoff')
plt.legend()
plt.show()


# boxplot 

sns.boxplot(train_df_new[['pickup_latitude', 'dropoff_latitude']])
plt.show()


# get the min, max, median, mean

train_df_new[['pickup_latitude', 'dropoff_latitude']].agg(['min','max', 'mean', 'median', 'std'])


def compare_df_size(old_df, new_df):
    old_df_shape = old_df.shape
    new_df_shape = new_df.shape
    percentage_change = ((old_df_shape[0] - new_df_shape[0]) / old_df_shape[0]) * 100
    print(f'The shape of old dataframe is {old_df_shape}')
    print(f'The shape of old dataframe is {new_df_shape}')
    print(f'The difference of rows is {old_df_shape[0] - new_df_shape[0]}')
    print(f'The percentage of outliers removed are {percentage_change:.2f}')


def remove_outliers(dataframe, column_names, fold = 1.5):
    filtered_df = dataframe.copy()
    for column_name in column_names:
        Q1, Q3 = dataframe[column_name].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        upper_bound = Q3 + (fold * IQR)
        lower_bound = Q1 - (fold * IQR)
        filtered_df = filtered_df.loc[(dataframe[column_name] >= lower_bound) & (dataframe[column_name] <= upper_bound), :].copy()
    compare_df_size(old_df = dataframe, new_df = filtered_df)
    return filtered_df


# remove the outliers from the pickup latitude / dropoff latitude

df_without_outliers = remove_outliers(dataframe = train_df_new, column_names = ['pickup_latitude', 'dropoff_latitude',
                                                                               'pickup_longitude', 'dropoff_longitude'], fold = 3)


# test the normality of columns
from scipy.stats import jarque_bera

def test_for_normality(data, column_name):
    alpha = 0.05
    _,p_val = jarque_bera(data[column_name].values)

    if p_val >= alpha:
        print(f'Fail to reject the H0', f'The {column_name} is normality distributed')
    else:
        print(f'Reject the H0', f'The {column_name} is not normality distributed')


train_df_new.columns[4:8]


column_to_test_for_normality = train_df_new.columns[4:8]


for col in column_to_test_for_normality:
    test_for_normality(train_df_new, column_name = col)
    print('*' * 20, end = '\n')


# distribution of the column 

sns.kdeplot(df_without_outliers['pickup_latitude'], label = 'pickup')
sns.kdeplot(df_without_outliers['dropoff_latitude'], label = 'dropoff')
plt.legend()
plt.show()


# boxplot 

sns.boxplot(df_without_outliers[['pickup_latitude', 'dropoff_latitude']])
plt.show()


# distribution 

sns.kdeplot(train_df_new['pickup_longitude'], label = 'pickup')
sns.kdeplot(train_df_new['dropoff_longitude'], label = 'dropoff')
plt.legend()
plt.show()


# boxplot 

sns.boxplot(train_df_new[['pickup_longitude', 'dropoff_longitude']])
plt.show()


# get the min, max, median, mean

train_df_new[['pickup_longitude', 'dropoff_longitude']].agg(['min','max', 'mean', 'median'])


# distribution of the column 

sns.kdeplot(df_without_outliers['pickup_longitude'], label = 'pickup')
sns.kdeplot(df_without_outliers['dropoff_longitude'], label = 'dropoff')
plt.legend()
plt.show()


# boxplot 

sns.boxplot(df_without_outliers[['pickup_longitude', 'dropoff_longitude']])
plt.show()


!pip install feature-engine


train_df_new.columns[4:8]


1 - 0.9982


100 * 60


def plot_boxplots(data, columns):
    div, rem = divmod(len(columns), 2)
    number_of_rows = div + rem
    number_of_columns = 2
    fig = plt.figure(figsize = (15, 8))
    for ind, col in enumerate(columns):
        plt.subplot(number_of_rows, number_of_columns, ind + 1)
        sns.boxplot(data, y = col, whis = 3)
        plt.tight_layout()

    plt.show()


def plot_kdeplots(data, columns):
    div, rem = divmod(len(columns), 2)
    number_of_rows = div + rem
    number_of_columns = 2
    fig = plt.figure(figsize = (15, 8))
    for ind, col in enumerate(columns):
        plt.subplot(number_of_rows, number_of_columns, ind + 1)
        sns.kdeplot(data,x = col)
        plt.tight_layout()

    plt.show()


normal_duration_df.head()


normal_duration_df['trip_duration'].max()


normal_duration_df.columns


df_subset = normal_duration_df.iloc[:, 5:9]

df_subset


latitude_columns = normal_duration_df.columns[normal_duration_df.columns.str.contains('latitude')]

longitude_columns = normal_duration_df.columns[normal_duration_df.columns.str.contains('longitude')]


# boxplot for latitude columns

plot_boxplots(df_subset, latitude_columns)


normal_duration_df.loc[(df_subset['pickup_latitude'] > 50)]


# boxplot for longitude columns

plot_boxplots(df_subset, longitude_columns)


remove_outliers(df_subset, df_subset.columns.to_list(), fold = 1.5)


# 0.1 and 99.9 percentile values

df_subset.quantile([0.001, 0.999])


# statistical summary of trip durations without removal of outliers

summary_original = normal_duration_df['trip_duration'].agg(func = ['min', 'max', 'median', 'mean'])
summary_original.name = 'before_removal'

summary_original


# distribution of passengers count before outlier removal 

normal_duration_df['passenger_count'].value_counts()


# data row where passenger count is greater than 6

normal_duration_df[normal_duration_df['passenger_count'] > 6]


from feature_engine.outliers import OutlierTrimmer

trimmer = OutlierTrimmer(capping_method = 'quantiles', tail = 'both', fold = 0.001, variables = df_subset.columns.to_list())


# remove outliers

df_without_outliers = trimmer.fit_transform(normal_duration_df)

compare_df_size(normal_duration_df, df_without_outliers)


trimmer.left_tail_caps_


trimmer.right_tail_caps_


df_without_outliers.columns


# statistical summary of trip durations after removal of outliers

summary_after_removal = df_without_outliers['trip_duration'].agg(func = ['min', 'max', 'median', 'mean'])
summary_after_removal.name = 'after_removal'

summary_after_removal


# merge the two results

pd.concat([summary_original, summary_after_removal], axis = 1)


# plot the two results

pd.concat([summary_original, summary_after_removal], axis = 1).iloc[1:, :].plot(kind = 'bar')
plt.show()


# distribution of passenger count after removal of outliers

df_without_outliers['passenger_count'].value_counts()


# boxplot for latitude columns

plot_boxplots(df_without_outliers, latitude_columns)


# boxplot for longitude columns

plot_boxplots(df_without_outliers, longitude_columns)


# distribution of the output column before and after removal of outliers

sns.kdeplot(normal_duration_df, x = 'trip_duration', label = 'before removal of outliers')

sns.kdeplot(df_without_outliers, x = 'trip_duration', label = 'after removal of outliers')
plt.legend()
plt.show()


from sklearn.preprocessing import PowerTransformer


latitude_columns.to_list() + longitude_columns.to_list()


df_without_outliers[df_subset.columns].agg(['min', 'max', 'mean', 'median'])


df_without_outliers.loc[(df_without_outliers['pickup_latitude'] == df_without_outliers['pickup_latitude'].min()), :]


!mkdir /kaggle/working/final_data


# save the data 

df_without_outliers.to_csv('/kaggle/working/final_data/train.csv')


# check 

final_df = df_without_outliers.copy()


final_df


# kdeplot for the targert column 

sns.kdeplot(final_df, x = 'trip_duration')
plt.show()


# apply yeo-johnson transformation 
from sklearn.preprocessing import PowerTransformer

pt_target = PowerTransformer(standardize = False)

# transform the output column

trip_duration_trans = pt_target.fit_transform(final_df[['trip_duration']])

sns.kdeplot(trip_duration_trans)
plt.show()


# apply yeo-johnson transformation 
from sklearn.preprocessing import PowerTransformer

pt_target = PowerTransformer(standardize = True)

# transform the output column

trip_duration_trans = pt_target.fit_transform(final_df[['trip_duration']])

sns.kdeplot(trip_duration_trans)
plt.show()


pd.DataFrame(trip_duration_trans, columns = ['target']).skew()


np.mean(trip_duration_trans), np.std(trip_duration_trans)


final_df['passenger_count'].value_counts()


# impact of 0 passenger in data

final_df.loc[final_df['passenger_count'] == 0, 'trip_duration'].agg(func = ['min', 'max', 'median', 'mean', 'std'])


# boxplot 

sns.boxplot(final_df.loc[(final_df['passenger_count'] == 0), 'trip_duration'].reset_index(drop = True))
plt.ylabel('Trip Duration(minutes')
plt.show()


sns.boxplot(final_df, x = 'passenger_count', y = 'trip_duration')
plt.show()


# vendor distribution for 0 passengers

final_df.loc[(final_df['passenger_count'] == 0), 'vendor_id'].value_counts().plot(kind = 'pie', autopct = '%.2f%%')
plt.show()


# hypothesis test for impact of passenger count on trip duration 

from scipy.stats import f_oneway, levene

# test for homoscedasticity or similar variances

arr_count_1 = final_df.loc[(final_df['passenger_count'] == 1), 'trip_duration'].values
arr_count_2 = final_df.loc[(final_df['passenger_count'] == 2), 'trip_duration'].values
arr_count_3 = final_df.loc[(final_df['passenger_count'] == 3), 'trip_duration'].values
arr_count_4 = final_df.loc[(final_df['passenger_count'] == 4), 'trip_duration'].values
arr_count_5 = final_df.loc[(final_df['passenger_count'] == 5), 'trip_duration'].values
arr_count_6 = final_df.loc[(final_df['passenger_count'] == 6), 'trip_duration'].values

leven_stat, levene_p_val = levene(arr_count_1, arr_count_2, arr_count_3, arr_count_4, arr_count_5, arr_count_6)

alpha = 0.05

print(levene_p_val)

if levene_p_val > alpha:
    print('Fail to reject the H0, the variance of samples are similar')
else:
    print('Reject the H0, the variance are different')


# std of trip duration based on passenger count 

final_df.groupby('passenger_count')['trip_duration'].std()


final_df.shape


# remove the rows of data where passenger count is 8

final_df = final_df.drop(final_df[final_df['passenger_count'] == 8].index)

final_df.shape


# unqiue values in the passenger count column 

np.sort(final_df['passenger_count'].unique())


sns.countplot(final_df, x = 'store_and_fwd_flag')


# statistical summary for target column

final_df.groupby('store_and_fwd_flag')['trip_duration'].agg(func = ['min', 'max', 'median', 'mean', 'std']).T.plot(kind = 'bar')
plt.show()


sns.boxplot(final_df, x = 'store_and_fwd_flag', y = 'trip_duration')


final_df['store_and_fwd_flag'].value_counts(normalize = True) * 100




