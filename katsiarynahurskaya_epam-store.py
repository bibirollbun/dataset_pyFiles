import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
paths = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        paths.append(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install py7zr


import py7zr
for filenames in paths:
    with py7zr.SevenZipFile(filenames, mode='r') as z_ref:
        z_ref.extractall(path='/kaggle/working')


%%time
data = pd.read_csv("/kaggle/working/train.csv")
data["date"] =  pd.to_datetime(data["date"])


monthly_counts_data = data['date'].dt.to_period('M').value_counts().sort_index()

plt.figure(figsize=(12, 6))
monthly_counts_data.plot(kind='bar', color='skyblue')
plt.title('Count of Data for Each Month for All data', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


%%time
THRESHOLD_TRAIN_DATE = pd.to_datetime("2017-07-01")
THRESHOLD_TEST_DATE = pd.to_datetime("2017-08-01")

data[data["date"] < THRESHOLD_TRAIN_DATE].to_csv("/kaggle/working/train_data.csv")
data[(data["date"] >= THRESHOLD_TRAIN_DATE)&(data["date"] < THRESHOLD_TEST_DATE)].to_csv("/kaggle/working/test_data.csv")

%xdel data


%%time
df = pd.read_csv("/kaggle/working/train_data.csv")
test_df = pd.read_csv("/kaggle/working/test_data.csv")


df["date"] = pd.to_datetime(df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])


monthly_counts = df['date'].dt.to_period('M').value_counts().sort_index().reset_index()
monthly_counts_test = test_df['date'].dt.to_period('M').value_counts().sort_index().reset_index()


monthly_counts['type'] = 'Train'
monthly_counts_test['type'] = 'Test'
combined = pd.concat([monthly_counts, monthly_counts_test])

plt.figure(figsize=(12, 6))
sns.barplot(data=combined, x='date', y='count', hue='type', palette=['blue', 'red'])
plt.title('Count of Data for Each Month (Train vs Test)', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Dataset')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


df.head()


print(df.shape)
print(test_df.shape)


holiday_events = pd.read_csv("/kaggle/working/holidays_events.csv")
print(holiday_events.shape)
holiday_events.head()


# Display unique values for the 'type', 'locale', and 'transferred' columns
print("Unique values in 'type':", holiday_events['type'].unique())
print("Unique values in 'locale':", holiday_events['locale'].unique())
print("Unique values in 'transferred':", holiday_events['transferred'].unique())


# Plot the distribution of the types of the events
type_counts = holiday_events['type'].value_counts()

plt.figure(figsize=(10, 6))
type_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Event Types', fontsize=16)
plt.xlabel('Event Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Plot the distribution of the levels of events
locale_counts = holiday_events['locale'].value_counts()

plt.figure(figsize=(8, 5))
locale_counts.plot(kind='bar', color='orange', edgecolor='black')
plt.title('Distribution of Locale Types', fontsize=16)
plt.xlabel('Locale', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Extract the unique years
holiday_events['date'] = pd.to_datetime(holiday_events['date'])
unique_years = holiday_events['date'].dt.year.unique()
print("Unique years in the 'date' column:", unique_years)


# Count the number of transferred and non-transferred holidays
transferred_counts = holiday_events['transferred'].value_counts()
print("Holidays not transferred (False):", transferred_counts.get(False, 0))
print("Holidays transferred (True):", transferred_counts.get(True, 0))


# Check for NaN values in the dataset holiday_events
nan_values = holiday_events.isna().sum()
print("Number of NaN values in each column:")
print(nan_values)


# Add a flag indicating whether a date is a holiday to the training and test sets
# Extract unique holiday dates from holiday_events dataset
holiday_dates = holiday_events['date'].unique()
# Add a flag for holidays using .isin()
df['is_holiday'] = df['date'].isin(holiday_dates).astype(int)
test_df['is_holiday'] = test_df['date'].isin(holiday_dates).astype(int)

print("Training set with holiday flag:\n", df.head())
print("\nTest set with holiday flag:\n", test_df.head())


# Group by is_holiday to calculate average sales
holiday_sales = df.groupby('is_holiday')['unit_sales'].mean()

plt.figure(figsize=(6, 4))
holiday_sales.plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Average Sales: Holiday vs Non-Holiday', fontsize=16)
plt.xlabel('Is Holiday', fontsize=12)
plt.ylabel('Average Sales', fontsize=12)
plt.xticks([0, 1], labels=['Non-Holiday', 'Holiday'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Categorize holidays by their type and consider their impact on sales
# Create a dictionary mapping dates to their event types
event_type_map = holiday_events.set_index('date')['type'].to_dict()

df['event_type'] = df['date'].map(event_type_map).fillna('No Holiday')
test_df['event_type'] = test_df['date'].map(event_type_map).fillna('No Holiday')

print("Training set with holiday types:\n", df.head())
print("\nTest set with holiday types:\n", test_df.head())


# Use the locale column to analyze regional effects
# Create a dictionary mapping dates to their locale
locale_map = holiday_events.set_index('date')['locale'].to_dict()

df['locale_impact'] = df['date'].map(locale_map).fillna('None')
test_df['locale_impact'] = test_df['date'].map(locale_map).fillna('None')

print("Training set with locale impact:\n", df.head())
print("\nTest set with locale impact:\n", test_df.head())


items = pd.read_csv("/kaggle/working/items.csv")
print(items.shape)
items.head()


# Group items by 'family' and 'class' and count the number of items in each group
family_class_group = items.groupby(['family', 'class']).size().reset_index(name='item_count')

# Analyze trends by family
family_trends = family_class_group.groupby('family')['item_count'].sum().reset_index()
family_trends = family_trends.sort_values(by='item_count', ascending=False)

# Plot the item count by family
plt.figure(figsize=(20, 6))
plt.bar(family_trends['family'], family_trends['item_count'], color='skyblue', edgecolor='black')
plt.title('Item Count by Family', fontsize=16)
plt.xlabel('Family', fontsize=12)
plt.ylabel('Item Count', fontsize=12)
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Group items by family and class and calculate the count of items
family_class_group = items.groupby(['family', 'class']).size().unstack(fill_value=0)

# Plot a stacked bar chart for family and class distribution
family_class_group.T.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='tab20')
plt.title('Class Distribution Within Families', fontsize=16)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Item Count', fontsize=12)
plt.legend(title='Family', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Plot heatmap
import seaborn as sns
top_families = family_class_group.sum(axis=1).nlargest(12).index
heatmap_data = family_class_group.loc[top_families]

plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=False, fmt='d', cbar=True)
plt.title('Class Distribution Heatmap for Top 12 Families', fontsize=16)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Family', fontsize=12)
plt.show()


# Map the weights of the perishable column to the training and test datasets using the item_nbr column
weights_map = items.set_index('item_nbr')['perishable'].apply(lambda x: 1.25 if x == 1 else 1.0).to_dict()

df['weight'] = df['item_nbr'].map(weights_map)
test_df['weight'] = test_df['item_nbr'].map(weights_map)

print("Training set weights:\n", df[['item_nbr', 'weight']].head())
print("\nTest set weights:\n", test_df[['item_nbr', 'weight']].head())


oil = pd.read_csv("/kaggle/working/oil.csv")
print(oil.shape)
oil.head()


oil.describe()


print("Missing values in dcoilwtico:", oil['dcoilwtico'].isna().sum())


oil['date'] = pd.to_datetime(oil['date'])
# Fill missing values in dcoilwtico using forward and backward fill
oil['dcoilwtico'] = oil['dcoilwtico'].ffill().bfill()
print("Missing values in dcoilwtico after filling:", oil['dcoilwtico'].isna().sum())


oil_price_map = oil.set_index('date')['dcoilwtico'].to_dict() # Create a dictionary mapping date to dcoilwtico

# Map oil prices to the training and test datasets based on the date
df['oil_price'] = df['date'].map(oil_price_map)
test_df['oil_price'] = test_df['date'].map(oil_price_map)

# Check for missing oil prices in training and test datasets
missing_train = df['oil_price'].isna().sum()
missing_test = test_df['oil_price'].isna().sum()

print("Missing oil prices in training set:", missing_train)
print("Missing oil prices in test set:", missing_test)


# Find dates in df that are missing in the oil dataset
missing_dates_train = df[~df['date'].isin(oil['date'])]['date'].unique()
print("Missing dates in training set:", missing_dates_train)

# Find dates in test_df that are missing in the oil dataset
missing_dates_test = test_df[~test_df['date'].isin(oil['date'])]['date'].unique()
print("Missing dates in test set:", missing_dates_test)


all_dates = pd.date_range(start=min(df['date'].min(), test_df['date'].min()), 
                          end=max(df['date'].max(), test_df['date'].max())) # Create a full date range from the training and test sets

oil = oil.set_index('date').reindex(all_dates).reset_index() # Reindex the oil dataset to include all dates, and fill missing values
oil.rename(columns={'index': 'date'}, inplace=True)

oil['dcoilwtico'] = oil['dcoilwtico'].ffill().bfill() # Fill missing oil prices


oil_price_map = oil.set_index('date')['dcoilwtico'].to_dict() # Create a dictionary mapping date to dcoilwtico

# Map oil prices to the training and test datasets based on the date
df['oil_price'] = df['date'].map(oil_price_map)
test_df['oil_price'] = test_df['date'].map(oil_price_map)

# Check for missing oil prices in training and test datasets
missing_train = df['oil_price'].isna().sum()
missing_test = test_df['oil_price'].isna().sum()

print("Missing oil prices in training set:", missing_train)
print("Missing oil prices in test set:", missing_test)


stores = pd.read_csv("/kaggle/working/stores.csv")
print(stores.shape)
stores.head()


print(stores.info())
print("Unique cities:", stores['city'].unique())
print("Unique states:", stores['state'].unique())
print("Unique store types:", stores['type'].unique())
print("Unique clusters:", stores['cluster'].unique())


# Create mapping dictionaries for each relevant column
city_map = stores.set_index('store_nbr')['city'].to_dict()
state_map = stores.set_index('store_nbr')['state'].to_dict()
type_map = stores.set_index('store_nbr')['type'].to_dict()
cluster_map = stores.set_index('store_nbr')['cluster'].to_dict()

# Add the mapped columns to the training dataset
df['city'] = df['store_nbr'].map(city_map)
df['state'] = df['store_nbr'].map(state_map)
df['store_type'] = df['store_nbr'].map(type_map)
df['cluster'] = df['store_nbr'].map(cluster_map)

# Add the mapped columns to the test dataset
test_df['city'] = test_df['store_nbr'].map(city_map)
test_df['state'] = test_df['store_nbr'].map(state_map)
test_df['store_type'] = test_df['store_nbr'].map(type_map)
test_df['cluster'] = test_df['store_nbr'].map(cluster_map)

print("Training set with store information:\n", df.head())
print("\nTest set with store information:\n", test_df.head())


# Group sales by city
city_sales = df.groupby('city')['unit_sales'].sum().sort_values(ascending=False)

plt.figure(figsize=(14, 6))
city_sales.plot(kind='bar', color='skyblue')
plt.title('Sales by City', fontsize=16)
plt.xlabel('City', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Group sales by state
state_sales = df.groupby('state')['unit_sales'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
state_sales.plot(kind='bar', color='orange')
plt.title('Sales by State', fontsize=16)
plt.xlabel('State', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Group sales by store type
type_sales = df.groupby('store_type')['unit_sales'].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
type_sales.plot(kind='bar', color='green')
plt.title('Sales by Store Type', fontsize=16)
plt.xlabel('Store Type', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Group sales by cluster
cluster_sales = df.groupby('cluster')['unit_sales'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
cluster_sales.plot(kind='bar', color='purple')
plt.title('Sales by Store Cluster', fontsize=16)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Label encode 'city', 'state', 'store_type', and 'cluster'
from sklearn.preprocessing import LabelEncoder

le_city = LabelEncoder()
le_state = LabelEncoder()
le_type = LabelEncoder()
le_cluster = LabelEncoder()

df['city_encoded'] = le_city.fit_transform(df['city'])
df['state_encoded'] = le_state.fit_transform(df['state'])
df['store_type_encoded'] = le_type.fit_transform(df['store_type'])
df['cluster_encoded'] = le_cluster.fit_transform(df['cluster'])

test_df['city_encoded'] = le_city.transform(test_df['city'])
test_df['state_encoded'] = le_state.transform(test_df['state'])
test_df['store_type_encoded'] = le_type.transform(test_df['store_type'])
test_df['cluster_encoded'] = le_cluster.transform(test_df['cluster'])

print("Encoded training set:\n", df[['city', 'city_encoded']].head())
print("\nEncoded test set:\n", test_df[['city', 'city_encoded']].head())


# Add regional sales as a new feature for EDA
regional_sales_map = df.groupby('state')['unit_sales'].sum().to_dict()
df['regional_sales'] = df['state'].map(regional_sales_map)
test_df['regional_sales'] = test_df['state'].map(regional_sales_map)


# Group by store type and cluster to calculate average sales per store type and per cluster as new features for EDA
avg_sales_per_type = df.groupby('store_type')['unit_sales'].mean()
avg_sales_per_cluster = df.groupby('cluster')['unit_sales'].mean()

type_avg_sales_map = avg_sales_per_type.to_dict()
cluster_avg_sales_map = avg_sales_per_cluster.to_dict()

df['avg_sales_per_type'] = df['store_type'].map(type_avg_sales_map)
df['avg_sales_per_cluster'] = df['cluster'].map(cluster_avg_sales_map)

test_df['avg_sales_per_type'] = test_df['store_type'].map(type_avg_sales_map)
test_df['avg_sales_per_cluster'] = test_df['cluster'].map(cluster_avg_sales_map)

print("Training set with new features:\n", df[['store_type', 'avg_sales_per_type']].head())
print("\nTest set with new features:\n", test_df[['store_type', 'avg_sales_per_type']].head())


df.head()


test_df.head()


df.info()


print("Unique onpromotion:", df['onpromotion'].unique())


# Count the values for False, True, and NaN
onpromotion_counts = df['onpromotion'].value_counts(dropna=False)

# Display the counts
print("Counts of onpromotion categories in the training set:")
print(onpromotion_counts)

# If working with test_df as well
test_onpromotion_counts = test_df['onpromotion'].value_counts(dropna=False)
print("\nCounts of onpromotion categories in the test set:")
print(test_onpromotion_counts)


df['onpromotion'] = df['onpromotion'].fillna(False).astype(int) 
test_df['onpromotion'] = test_df['onpromotion'].astype(int)

print("Transformed 'onpromotion' column in training set:\n", df['onpromotion'].value_counts())
print("\nTransformed 'onpromotion' column in test set:\n", test_df['onpromotion'].value_counts())


# Group sales by onpromotion status
promotion_sales = df.groupby('onpromotion')['unit_sales'].mean()

plt.figure(figsize=(6, 4))
promotion_sales.plot(kind='bar', color=['skyblue', 'orange'][:len(promotion_sales)])
plt.title('Average Sales by Promotion Status', fontsize=16)
plt.xlabel('Promotion Status', fontsize=12)
plt.ylabel('Average Sales', fontsize=12)
plt.xticks([0, 1][:len(promotion_sales)], labels=['Not on Promotion', 'On Promotion'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Label encode 'event_type' in the training and test sets
le_event = LabelEncoder()
df['event_type_encoded'] = le_event.fit_transform(df['event_type'].fillna('None'))
test_df['event_type_encoded'] = le_event.transform(test_df['event_type'].fillna('None'))

print("Transformed 'event_type' column in training set:\n", df[['event_type', 'event_type_encoded']].head())


# Look at the values of 'locale_impact' column in the training set
locale_impact_counts = df['locale_impact'].value_counts(dropna=False)
print(locale_impact_counts)


# Label encode 'locale_impact' in the training and test sets
le_locale = LabelEncoder()
df['locale_impact_encoded'] = le_locale.fit_transform(df['locale_impact'].fillna('None'))
test_df['locale_impact_encoded'] = le_locale.transform(test_df['locale_impact'].fillna('None'))

print("Transformed 'locale_impact' column in training set:\n", df[['locale_impact', 'locale_impact_encoded']].head())


df_sample = df.sample(frac=0.05, random_state=42) # Sample 5% of the training data
test_df_sample = test_df.sample(frac=0.05, random_state=42) # Sample 5% of the test data

print("Original training data shape:", df.shape)
print("Sampled training data shape:", df_sample.shape)
print("Original test data shape:", test_df.shape)
print("Sampled test data shape:", test_df_sample.shape)


# Select features for the baseline `DummyModel`
baseline_features = [
    'onpromotion',           # Direct impact on sales
    'is_holiday',            # Holidays often affect sales patterns
    'oil_price',             # Economic indicator
    'weight',                # Weight factor for perishable items
    'regional_sales',        # Aggregate sales data for regional trends
    'avg_sales_per_type',    # Store type-based sales averages
    'avg_sales_per_cluster'  # Cluster-level sales averages
]


# Prepare X_train and X_test
X_train = df_sample[baseline_features]
X_test = test_df_sample[baseline_features]


# Prepare y_train and y_test
y_train = df_sample['unit_sales']
y_test = test_df_sample['unit_sales']


# Implement DummyModel
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

dummy_model = DummyRegressor(strategy='mean')

dummy_model.fit(X_train, y_train)

y_pred_dummy = dummy_model.predict(X_test)

baseline_rmse = mean_squared_error(y_test, y_pred_dummy, squared=False)
print("Baseline Model RMSE:", baseline_rmse)


# Plot the distribution of unit_sales
plt.hist(y_train, bins=50, alpha=0.7, label="Training Set")
plt.hist(y_test, bins=50, alpha=0.7, label="Test Set")
plt.title("Distribution of Unit Sales")
plt.xlabel("Unit Sales")
plt.ylabel("Frequency")
plt.legend()
plt.show()

print("Mean of unit_sales:", y_train.mean())
print("Standard deviation of unit_sales:", y_train.std())
print("Median of unit_sales:", y_train.median())


# Select features for LGBMRegressor and CatBoost models
features = [
    'onpromotion', 'is_holiday', 'oil_price', 'weight',
    'city_encoded', 'state_encoded', 'store_type_encoded', 'cluster_encoded',
    'event_type_encoded', 'locale_impact_encoded',
    'regional_sales', 'avg_sales_per_type', 'avg_sales_per_cluster'
]


X_train = df_sample[features]
y_train = df_sample['unit_sales']  
X_test = test_df_sample[features]
y_test = test_df_sample['unit_sales']  

weights = test_df_sample['weight'] # Extract weights for evaluation


# Initialize the LightGBM Regressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

lgbm = LGBMRegressor(
    random_state=42,
    objective='regression',
    n_estimators=500,
    learning_rate=0.1,
    max_depth=10,
    num_leaves=31
)

lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_test)

rmse_lgbm = mean_squared_error(y_test, y_pred_lgbm, squared=False)
print("LightGBM RMSE:", rmse_lgbm)


def nwrmsle(y_true, y_pred, weights):
    log_diff = np.log1p(y_pred) - np.log1p(y_true)
    weighted_log_diff = (weights * (log_diff ** 2)).sum()
    normalization = weights.sum()
    return np.sqrt(weighted_log_diff / normalization)

nwrmsle_lgbm = nwrmsle(y_test, y_pred_lgbm, weights)
print("LightGBM NWRMSLE:", nwrmsle_lgbm)


from sklearn.model_selection import GridSearchCV

param_grid = {
    'num_leaves': [31, 50],
    'max_depth': [-1, 10],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 500]
}

grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)
print("Best parameters for LightGBM:", grid_search.best_params_)


from catboost import CatBoostRegressor

catboost = CatBoostRegressor(loss_function='RMSE', logging_level='Silent', random_state=42)

param_grid = {
    'depth': [6, 10, 15],
    'learning_rate': [0.01, 0.1, 0.2],
    'iterations': [500, 1000, 1500],
    'l2_leaf_reg': [1, 3, 5]
}

grid_search_cb = GridSearchCV(estimator=catboost, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_search_cb.fit(X_train, y_train)

best_catboost = grid_search_cb.best_estimator_

y_pred_catboost = best_catboost.predict(X_test)
y_pred_catboost = np.expm1(y_pred_catboost)  # Reverse log-transform


