import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import missingno as msno 
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import LabelEncoder


# Set working directory and read data
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/bike-sharing-demand/train.csv')
test = pd.read_csv('../input/bike-sharing-demand/test.csv')


train.shape


test.shape


train.head()


test.head()


train.dtypes


test.dtypes


# Introduce variables to combine train and test
test['registered'] = 0
test['casual'] = 0
test['count'] = 0
data = pd.concat([train, test], axis=0)


data.shape


# Visualise missing values as bar chart
msno.bar(data,figsize=(8,4))


# Create a grid of 4 rows and 2 columns
fig, axes = plt.subplots(4, 2, figsize=(8, 10))
axes = axes.flatten()

# Define the columns to create histograms for
columns = ['season', 'weather', 'holiday', 'workingday', 'temp', 'atemp', 'humidity', 'windspeed']

# Generate histograms
for i, col in enumerate(columns):
    sn.histplot(data[col], ax=axes[i])
    axes[i].set_title(f'Histogram of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('frequency')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# Convert specific columns to categorical
for col in ['season', 'weather', 'holiday', 'workingday']:
    data[col] = data[col].astype('category')


# Extract hour, day, and year from datetime
data['hour'] = pd.to_datetime(data['datetime']).dt.hour.astype('category')
data['day'] = pd.to_datetime(data['datetime']).dt.day
data['month'] = pd.to_datetime(data['datetime']).dt.month.astype('category')
data['year'] = pd.to_datetime(data['datetime']).dt.year.astype('category')
data['day_name'] = pd.to_datetime(data['datetime']).dt.day_name()


# Split back into train and test
train = data[data['datetime'].str.slice(8, 10).astype(int) < 20]
test = data[data['datetime'].str.slice(8, 10).astype(int) > 19]


# Create boxplot
plt.figure(figsize=(8, 4))
sn.boxplot([train['count'][train['hour'] == hour] for hour in sorted(train['hour'].unique())])
plt.title('Boxplot of total users')
plt.xlabel('hour')
plt.ylabel('total users')
plt.show()


# Create a grid of 2 rows and 1 column
fig, axes = plt.subplots(2, 1, figsize=(8, 6))
axes = axes.flatten()

# Define the columns to create boxplots for
columns = ['casual', 'registered']

# Generate boxplots
for i, col in enumerate(columns):
    sn.boxplot([train[col][train['hour'] == hour] for hour in sorted(train['hour'].unique())], ax=axes[i])
    axes[i].set_title(f'Boxplot of {col} users')
    axes[i].set_xlabel('hour')
    axes[i].set_ylabel(f'{col} users')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# Create a grid of 3 rows and 1 column
fig, axes = plt.subplots(3, 1, figsize=(8, 9))
axes = axes.flatten()

# Define the columns to create boxplots for
columns = ['count', 'casual', 'registered']

# Generate boxplots
for i, col in enumerate(columns):
    sn.boxplot([np.log1p(train[col])[train['hour'] == hour] for hour in sorted(train['hour'].unique())], ax = axes[i])
    axes[i].set_xlabel('hour')
    if col == 'count':
        axes[i].set_title('Boxplot of total users')
        axes[i].set_ylabel('total users')
    else:
        axes[i].set_title(f'Boxplot of {col} users')
        axes[i].set_ylabel(f'{col} users')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# Create a grid of 3 rows and 1 column
fig, axes = plt.subplots(3, 1, figsize=(8, 10))
axes = axes.flatten()

# Define the columns to create boxplots for
columns = ['count', 'casual', 'registered']

# Generate boxplots
for i, col in enumerate(columns):
    sn.boxplot(train, x = 'day_name', y = col, ax = axes[i])
    axes[i].set_xlabel('day_name')
    if col == 'count':
        axes[i].set_title('Boxplot of total users')
        axes[i].set_ylabel('total users')
    else:
        axes[i].set_title(f'Boxplot of {col} users')
        axes[i].set_ylabel(f'{col} users')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# Create a grid of 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes = axes.flatten()

# Define the columns to create boxplots for
columns = ['count', 'casual', 'registered']

# Generate boxplots
for i, col in enumerate(columns):
    sn.boxplot(train, x = 'year', y = col, ax = axes[i])
    axes[i].set_xlabel('year')
    if col == 'count':
        axes[i].set_title('Boxplot of total users')
        axes[i].set_ylabel('total users')
    else:
        axes[i].set_title(f'Boxplot of {col} users')
        axes[i].set_ylabel(f'{col} users')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# Create a grid of 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes = axes.flatten()

# Define the columns to create boxplots for
columns = ['count', 'casual', 'registered']

# Generate boxplots
for i, col in enumerate(columns):
    sn.boxplot(train, x = 'weather', y = col, ax = axes[i])
    axes[i].set_xlabel('weather')
    if col == 'count':
        axes[i].set_title('Boxplot of total users')
        axes[i].set_ylabel('total users')
    else:
        axes[i].set_title(f'Boxplot of {col} users')
        axes[i].set_ylabel(f'{col} users')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# Define the columns for correlation analysis
df_corr = train[['registered', 'casual', 'count', 'temp', 'atemp', 'humidity', 'windspeed']]

# Compute the correlation matrix
correlation_matrix = df_corr.corr()

# Create correlation heatmap
mask = np.array(correlation_matrix)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(16,8)
sn.heatmap(correlation_matrix, mask=mask,vmax=.8, square=True,annot=True)


# Use decision tree for binning - registered users
tree_reg = DecisionTreeRegressor(max_depth=4)
tree_reg.fit(train[['hour']], train['registered'])
fig, axes = plt.subplots(1,1,figsize = (10,10), dpi=300)
plot_tree(tree_reg, feature_names=['hour'], filled=True)
plt.show()



# Initialise the column with 0
data['hour_bin_registered'] = 0

# Assign values to hour bins for registered users
data.loc[pd.to_numeric(data['hour']) <= 7, 'hour_bin_registered'] = 1
data.loc[pd.to_numeric(data['hour']) == 8, 'hour_bin_registered'] = 2
data.loc[pd.to_numeric(data['hour']) == 9, 'hour_bin_registered'] = 3
data.loc[(pd.to_numeric(data['hour']) > 9) & (pd.to_numeric(data['hour']) < 18), 'hour_bin_registered'] = 4
data.loc[(pd.to_numeric(data['hour']) == 18) | (pd.to_numeric(data['hour']) == 19), 'hour_bin_registered'] = 5
data.loc[(pd.to_numeric(data['hour']) == 20) | (pd.to_numeric(data['hour']) == 21), 'hour_bin_registered'] = 6
data.loc[pd.to_numeric(data['hour']) >= 22, 'hour_bin_registered'] = 7


# Use decision tree for binning - casual users
tree_cas = DecisionTreeRegressor(max_depth=4)
tree_cas.fit(train[['hour']], train['casual'])
fig, axes = plt.subplots(1,1,figsize = (10,10), dpi=300)
plot_tree(tree_cas, feature_names=['hour'], filled=True)
plt.show()



# Initialise the column with 0
data['hour_bin_casual'] = 0

# Assign values to hour bins for casual users
data.loc[pd.to_numeric(data['hour']) <= 7, 'hour_bin_casual'] = 1
data.loc[(pd.to_numeric(data['hour']) == 8) | (pd.to_numeric(data['hour']) == 9), 'hour_bin_casual'] = 2
data.loc[pd.to_numeric(data['hour']) == 10, 'hour_bin_casual'] = 3
data.loc[(pd.to_numeric(data['hour']) > 10) & (pd.to_numeric(data['hour']) < 20), 'hour_bin_casual'] = 4
data.loc[(pd.to_numeric(data['hour']) == 20) | (pd.to_numeric(data['hour']) == 21), 'hour_bin_casual'] = 5
data.loc[pd.to_numeric(data['hour']) >= 22, 'hour_bin_casual'] = 6


# Use decision tree for binning - registered users
temp_tree_reg = DecisionTreeRegressor(max_depth=3)
temp_tree_reg.fit(train[['temp']], train['registered'])
fig, axes = plt.subplots(1, 1, figsize = (10,10), dpi = 300)
plot_tree(temp_tree_reg, feature_names = ['temp'], filled = True)
plt.show()


# Initialise the column with 0
data['temp_bin_registered'] = 0

# Assign values to temp bins for registered users
data.loc[data['temp'] <= 11.07, 'temp_bin_registered'] = 1
data.loc[(data['temp'] > 11.07) & (data['temp'] <= 12.71), 'temp_bin_registered'] = 2
data.loc[(data['temp'] > 12.71) & (data['temp'] <= 19.27), 'temp_bin_registered'] = 3
data.loc[(data['temp'] > 19.27) & (data['temp'] <= 22.55), 'temp_bin_registered'] = 4
data.loc[(data['temp'] > 22.55) & (data['temp'] <= 28.29), 'temp_bin_registered'] = 5
data.loc[(data['temp'] > 28.29) & (data['temp'] <= 29.93), 'temp_bin_registered'] = 6
data.loc[(data['temp'] > 29.93) & (data['temp'] <= 30.75), 'temp_bin_registered'] = 7
data.loc[data['temp'] > 30.75, 'temp_bin_registered'] = 8


# Use decision tree for binning - casual users
temp_tree_casual = DecisionTreeRegressor(max_depth=3)
temp_tree_casual.fit(train[['temp']], train['casual'])
fig, axes = plt.subplots(1, 1,figsize = (10, 10), dpi = 300)
plot_tree(temp_tree_casual, feature_names = ['temp'], filled = True)
plt.show()


# Initialise the column with 0
data['temp_bin_casual'] = 0

# Assign values to temp bins for casual users
data.loc[data['temp'] <= 12.71, 'temp_bin_casual'] = 1
data.loc[(data['temp'] > 12.71) & (data['temp'] <= 15.17), 'temp_bin_casual'] = 2
data.loc[(data['temp'] > 15.17) & (data['temp'] <= 19.27), 'temp_bin_casual'] = 3
data.loc[(data['temp'] > 19.27) & (data['temp'] <= 23.37), 'temp_bin_casual'] = 4
data.loc[(data['temp'] > 23.37) & (data['temp'] <= 29.11), 'temp_bin_casual'] = 5
data.loc[(data['temp'] > 29.11) & (data['temp'] <= 29.93), 'temp_bin_casual'] = 6
data.loc[(data['temp'] > 29.93) & (data['temp'] <= 32.29), 'temp_bin_casual'] = 7
data.loc[data['temp'] > 32.29, 'temp_bin_casual'] = 8


# Initialise the column with 0
data['qtr_bin'] = 0

# Assign values to quarter bins
data.loc[(data['year'] == 2011) & (pd.to_numeric(data['month']) <= 3), 'qtr_bin'] = 1
data.loc[(data['year'] == 2011) & (pd.to_numeric(data['month']) > 3) & (pd.to_numeric(data['month']) <= 6), 'qtr_bin'] = 2
data.loc[(data['year'] == 2011) & (pd.to_numeric(data['month']) > 6) & (pd.to_numeric(data['month']) <= 9), 'qtr_bin'] = 3
data.loc[(data['year'] == 2011) & (pd.to_numeric(data['month']) > 9), 'qtr_bin'] = 4
data.loc[(data['year'] == 2012) & (pd.to_numeric(data['month']) <= 3), 'qtr_bin'] = 5
data.loc[(data['year'] == 2012) & (pd.to_numeric(data['month']) > 3) & (pd.to_numeric(data['month']) <= 6), 'qtr_bin'] = 6
data.loc[(data['year'] == 2012) & (pd.to_numeric(data['month']) > 6) & (pd.to_numeric(data['month']) <= 9), 'qtr_bin'] = 7
data.loc[(data['year'] == 2012) & (pd.to_numeric(data['month']) > 9), 'qtr_bin'] = 8


# Initialise the column with 0
data['day_type_bin'] = ''

# Assign values to date type bins
data.loc[(data['holiday'] == 1), 'day_type_bin'] = 'holiday'
data.loc[(data['holiday'] == 0) & (data['workingday'] == 1), 'day_type_bin'] = 'working day'
data.loc[(data['holiday'] == 0) & (data['workingday'] == 0), 'day_type_bin'] = 'weekend'


# Initialise the column with 0
data['weekend'] = 0

# Assign values
data.loc[(data['day_name'] == 'Saturday') | (data['day_name'] == 'Sunday'), 'weekend'] = 1


wind_0 = data[data['windspeed'] == 0]
wind_1 = data[data['windspeed'] != 0]

rf_wind = RandomForestRegressor(n_estimators=250, random_state=415)
rf_wind.fit(wind_1[['season', 'weather', 'humidity', 'temp', 'year']], wind_1['windspeed'])
wind_0['windspeed'] = rf_wind.predict(wind_0[['season', 'weather', 'humidity', 'temp', 'year']])
data = pd.concat([wind_0, wind_1])


for col in ['season', 'holiday', 'workingday', 'weather', 'hour', 'day_name', 'hour_bin_registered']:
    data[col] = LabelEncoder().fit_transform(data[col])


train['log_registered'] = np.log1p(train['registered'])
train['log_casual'] = np.log1p(train['casual'])


rf_reg = RandomForestRegressor(n_estimators=250, random_state=415)
rf_reg.fit(train[['hour', 'workingday', 'temp', 'humidity', 'season', 'weather']], train['log_registered'])

rf_cas = RandomForestRegressor(n_estimators=250, random_state=415)
rf_cas.fit(train[['hour', 'workingday', 'temp', 'humidity', 'season', 'weather']], train['log_casual'])


test['log_registered'] = rf_reg.predict(test[['hour', 'workingday', 'temp', 'humidity', 'season', 'weather']])
test['log_casual'] = rf_cas.predict(test[['hour', 'workingday', 'temp', 'humidity', 'season', 'weather']])

test['registered'] = np.expm1(test['log_registered'])
test['casual'] = np.expm1(test['log_casual'])
test['count'] = test['registered'] + test['casual']


submission = test[['datetime', 'count']]
submission.to_csv("submit.csv", index=False)

