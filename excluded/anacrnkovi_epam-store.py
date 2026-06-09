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


holiday_events = pd.read_csv("/kaggle/working/holidays_events.csv")
print(holiday_events.shape)
holiday_events.head()


# Ensure date columns are datetime
holiday_events["date"] = pd.to_datetime(holiday_events["date"])

# Create a flag for holidays in the training dataset
df["is_holiday"] = df["date"].isin(holiday_events["date"]).astype(int)

# Create a flag for holidays in the test dataset
test_df["is_holiday"] = test_df["date"].isin(holiday_events["date"]).astype(int)

# Display the updated training dataset
df[["date", "is_holiday"]].head()



# Create a dictionary to map dates to holiday types
holiday_type_map = dict(zip(holiday_events['date'], holiday_events['type']))

# Map the holiday type to the training and test datasets
df['type'] = df['date'].map(holiday_type_map).fillna('None')
test_df['type'] = test_df['date'].map(holiday_type_map).fillna('None')

# Display a few rows to confirm
display(df[['date', 'type']].head())



# Group by holiday type to calculate average sales
holiday_impact = df.groupby('type')['unit_sales'].mean().reset_index()
display(holiday_impact)



# Create a dictionary to map dates to holiday locales
holiday_locale_map = dict(zip(holiday_events['date'], holiday_events['locale']))

# Map the holiday locale to the training and test datasets
df['locale'] = df['date'].map(holiday_locale_map).fillna('None')
test_df['locale'] = test_df['date'].map(holiday_locale_map).fillna('None')

# Group by locale to calculate average sales
locale_impact = df.groupby('locale')['unit_sales'].mean().reset_index()
display(locale_impact)



items = pd.read_csv("/kaggle/working/items.csv")
print(items.shape)
items.head()


# Load items.csv
items = pd.read_csv("/kaggle/working/items.csv")

# Create dictionaries for memory-efficient mapping
family_map = dict(zip(items['item_nbr'], items['family']))
class_map = dict(zip(items['item_nbr'], items['class']))
perishable_map = dict(zip(items['item_nbr'], items['perishable']))

# Map family, class, and perishable to df and test_df
df['family'] = df['item_nbr'].map(family_map)
df['class'] = df['item_nbr'].map(class_map)
df['perishable'] = df['item_nbr'].map(perishable_map)

test_df['family'] = test_df['item_nbr'].map(family_map)
test_df['class'] = test_df['item_nbr'].map(class_map)
test_df['perishable'] = test_df['item_nbr'].map(perishable_map)

# Display a few rows to confirm
df[['item_nbr', 'family', 'class', 'perishable']].head()



# Group by family and calculate average sales
family_trends = df.groupby('family')['unit_sales'].mean().reset_index()

# Group by class and calculate average sales
class_trends = df.groupby('class')['unit_sales'].mean().reset_index()

# Display the top 5 families and classes with the highest sales
display(family_trends.sort_values(by='unit_sales', ascending=False).head())
display(class_trends.sort_values(by='unit_sales', ascending=False).head())



oil = pd.read_csv("/kaggle/working/oil.csv")
print(oil.shape)
oil.head()


# Create a dictionary for oil prices
oil['date'] = pd.to_datetime(oil['date'])
oil_price_map = dict(zip(oil['date'], oil['dcoilwtico']))

# Map oil prices to the training and test datasets
df['oil_price'] = df['date'].map(oil_price_map)
test_df['oil_price'] = test_df['date'].map(oil_price_map)

# Display a few rows to confirm
print(df[['date', 'oil_price']].head())



# Drop rows where oil_price is NaN for correlation analysis
oil_sales_correlation = df[['oil_price', 'unit_sales']].dropna().corr()

# Display correlation results
print("Correlation between oil price and unit sales:")
print(oil_sales_correlation)

# Visualize the trend of oil price vs. sales (optional if memory allows)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.scatter(df['oil_price'], df['unit_sales'], alpha=0.5)
plt.title('Oil Price vs Unit Sales')
plt.xlabel('Oil Price')
plt.ylabel('Unit Sales')
plt.grid()
plt.show()



stores = pd.read_csv("/kaggle/working/stores.csv")
print(stores.shape)
stores.head()


import pandas as pd

# Load the stores data and create mappings
stores = pd.read_csv("/kaggle/working/stores.csv")
city_map = dict(zip(stores['store_nbr'], stores['city']))
state_map = dict(zip(stores['store_nbr'], stores['state']))
type_map = dict(zip(stores['store_nbr'], stores['type']))
cluster_map = dict(zip(stores['store_nbr'], stores['cluster']))

# Define a function to map attributes in chunks
def process_chunk(input_path, output_path, city_map, state_map, type_map, cluster_map):
    chunk_size = 50000  # Smaller chunks to save memory
    with pd.read_csv(input_path, chunksize=chunk_size) as reader:
        for i, chunk in enumerate(reader):
            # Map store attributes to the chunk
            chunk['city'] = chunk['store_nbr'].map(city_map)
            chunk['state'] = chunk['store_nbr'].map(state_map)
            chunk['store_type'] = chunk['store_nbr'].map(type_map)
            chunk['cluster'] = chunk['store_nbr'].map(cluster_map)
            
            # Append or create a new file
            if i == 0:
                chunk.to_csv(output_path, index=False)
            else:
                chunk.to_csv(output_path, mode='a', header=False, index=False)

# Process the training data
process_chunk("/kaggle/working/train_data.csv", "/kaggle/working/updated_train_data.csv", 
              city_map, state_map, type_map, cluster_map)

# Process the test data
process_chunk("/kaggle/working/test_data.csv", "/kaggle/working/updated_test_data.csv", 
              city_map, state_map, type_map, cluster_map)



import pandas as pd

# Define a function to process chunks and calculate regional sales
def calculate_regional_sales(input_path, output_path):
    chunk_size = 50000  # Process small chunks to save memory
    regional_data = {}  # Dictionary to hold aggregated sales by city and state
    
    # Read data in chunks
    with pd.read_csv(input_path, chunksize=chunk_size) as reader:
        for chunk in reader:
            # Group by city and state to calculate sales for the chunk
            grouped = chunk.groupby(['city', 'state'])['unit_sales'].sum()
            
            # Aggregate the results into the dictionary
            for (city, state), sales in grouped.items():
                if (city, state) not in regional_data:
                    regional_data[(city, state)] = 0
                regional_data[(city, state)] += sales
    
    # Convert the aggregated results to a DataFrame
    regional_df = pd.DataFrame.from_dict(regional_data, orient='index', columns=['total_sales'])
    regional_df.reset_index(inplace=True)
    regional_df.rename(columns={'level_0': 'city', 'level_1': 'state'}, inplace=True)
    
    # Save the results to disk
    regional_df.to_csv(output_path, index=False)

# Process training data for regional sales
calculate_regional_sales("/kaggle/working/updated_train_data.csv", "/kaggle/working/regional_sales.csv")

# Load and display the results to confirm
regional_sales = pd.read_csv("/kaggle/working/regional_sales.csv")
print(regional_sales.head())



# Define a function to process chunks and calculate sales by store type
def calculate_store_type_sales(input_path, output_path):
    chunk_size = 50000  # Process small chunks to save memory
    store_type_data = {}  # Dictionary to hold aggregated sales by store type
    
    # Read data in chunks
    with pd.read_csv(input_path, chunksize=chunk_size) as reader:
        for chunk in reader:
            # Group by store_type to calculate sales for the chunk
            grouped = chunk.groupby('store_type')['unit_sales'].sum()
            
            # Aggregate the results into the dictionary
            for store_type, sales in grouped.items():
                if store_type not in store_type_data:
                    store_type_data[store_type] = 0
                store_type_data[store_type] += sales
    
    # Convert the aggregated results to a DataFrame
    store_type_df = pd.DataFrame.from_dict(store_type_data, orient='index', columns=['total_sales'])
    store_type_df.reset_index(inplace=True)
    store_type_df.rename(columns={'index': 'store_type'}, inplace=True)
    
    # Save the results to disk
    store_type_df.to_csv(output_path, index=False)

# Process training data for store type sales
calculate_store_type_sales("/kaggle/working/updated_train_data.csv", "/kaggle/working/store_type_sales.csv")

# Load and display the results to confirm
store_type_sales = pd.read_csv("/kaggle/working/store_type_sales.csv")
print(store_type_sales.head())



# Define a function to process chunks and calculate sales by cluster
def calculate_cluster_sales(input_path, output_path):
    chunk_size = 50000  # Process small chunks to save memory
    cluster_data = {}  # Dictionary to hold aggregated sales by cluster
    
    # Read data in chunks
    with pd.read_csv(input_path, chunksize=chunk_size) as reader:
        for chunk in reader:
            # Group by cluster to calculate sales for the chunk
            grouped = chunk.groupby('cluster')['unit_sales'].sum()
            
            # Aggregate the results into the dictionary
            for cluster, sales in grouped.items():
                if cluster not in cluster_data:
                    cluster_data[cluster] = 0
                cluster_data[cluster] += sales
    
    # Convert the aggregated results to a DataFrame
    cluster_df = pd.DataFrame.from_dict(cluster_data, orient='index', columns=['total_sales'])
    cluster_df.reset_index(inplace=True)
    cluster_df.rename(columns={'index': 'cluster'}, inplace=True)
    
    # Save the results to disk
    cluster_df.to_csv(output_path, index=False)

# Process training data for cluster sales
calculate_cluster_sales("/kaggle/working/updated_train_data.csv", "/kaggle/working/cluster_sales.csv")

# Load and display the results to confirm
cluster_sales = pd.read_csv("/kaggle/working/cluster_sales.csv")
print(cluster_sales.head())



import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load training and test datasets
train_data = pd.read_csv("/kaggle/working/updated_train_data.csv", usecols=['unit_sales'])
test_data = pd.read_csv("/kaggle/working/updated_test_data.csv", usecols=['unit_sales'])

# Calculate the mean of unit_sales from the training set
mean_sales = train_data['unit_sales'].mean()

# Predict the mean for all test data
test_data['baseline_prediction'] = mean_sales

# Evaluate the baseline model
mae = mean_absolute_error(test_data['unit_sales'], test_data['baseline_prediction'])
mse = mean_squared_error(test_data['unit_sales'], test_data['baseline_prediction'])

print(f"Baseline Model Performance:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")

# Display the first few predictions
print(test_data[['unit_sales', 'baseline_prediction']].head())



from sklearn.preprocessing import StandardScaler

# Process numeric features one by one
def process_numeric_feature(input_path, output_path, feature, is_training=True):
    scaler = StandardScaler()  # Initialize scaler
    chunk_size = 50000  # Small chunk size
    
    with pd.read_csv(input_path, chunksize=chunk_size) as reader:
        for i, chunk in enumerate(reader):
            # Scale the feature
            chunk[feature] = scaler.fit_transform(chunk[[feature]]) if is_training else scaler.transform(chunk[[feature]])
            
            # Save to file
            if i == 0:
                chunk[[feature]].to_csv(output_path, index=False)
            else:
                chunk[[feature]].to_csv(output_path, mode='a', header=False, index=False)

# Example: Process 'oil_price'
process_numeric_feature("/kaggle/working/updated_train_data.csv", "/kaggle/working/train_oil_price.csv", 'oil_price', is_training=True)
process_numeric_feature("/kaggle/working/updated_test_data.csv", "/kaggle/working/test_oil_price.csv", 'oil_price', is_training=False)
from sklearn.preprocessing import OneHotEncoder

# Process categorical features one by one
def process_categorical_feature(input_path, output_path, feature, is_training=True):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)  # Initialize encoder
    chunk_size = 50000  # Small chunk size

    with pd.read_csv(input_path, chunksize=chunk_size) as reader:
        for i, chunk in enumerate(reader):
            # Encode the feature
            encoded = encoder.fit_transform(chunk[[feature]]) if is_training else encoder.transform(chunk[[feature]])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([feature]), index=chunk.index)
            
            # Save to file
            if i == 0:
                encoded_df.to_csv(output_path, index=False)
            else:
                encoded_df.to_csv(output_path, mode='a', header=False, index=False)

# Example: Process 'holiday_type'
process_categorical_feature("/kaggle/working/updated_train_data.csv", "/kaggle/working/train_holiday_type.csv", 'holiday_type', is_training=True)
process_categorical_feature("/kaggle/working/updated_test_data.csv", "/kaggle/working/test_holiday_type.csv", 'holiday_type', is_training=False)


