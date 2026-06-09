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


historic = pd.read_csv("/kaggle/input/bpl-ai4good-wheat-price-forecasting/train.csv")
historic.head()


historic[(historic.year == 2022) & (historic.month == 3)].price_usd.value_counts()


historic.market.value_counts()


test = pd.read_csv("/kaggle/input/bpl-ai4good-wheat-price-forecasting/test.csv")


def get_market(string):
    return string.split("_")[0]
    
markets = test.market_year_month.apply(get_market)


pd.Series(markets).value_counts()


historic_markets = pd.unique(historic.market)
historic_markets[:10]


test_markets = pd.unique(pd.Series(markets))
len(historic_markets), len(test_markets)


missing_markets = set(test_markets) - set(historic_markets)
print(missing_markets)


historic.info()


historic = historic.dropna(subset=['price_usd'])


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

# Select only numerical columns
numeric_cols = historic.select_dtypes(include=['number']).columns

# Determine the number of plots needed
num_features = len(numeric_cols)
num_rows = math.ceil(num_features / 3)

# Create subplots
fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
axes = axes.flatten()  # Flatten the axes array for easy iteration

# Plot each feature
for i, col in enumerate(numeric_cols):
    sns.histplot(historic[col], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


historic[historic.price_localcurrency > 20000].head()


historic_markets = pd.unique(historic.market)
missing_markets = set(test_markets) - set(historic_markets)
print(missing_markets)


historic_2022 = historic[historic.year >= 2022]
historic_2022.head()


historic_2022_gbycommodity = historic_2022.groupby("commodity")


import matplotlib.pyplot as plt

# Iterate through each group in the grouped DataFrame (historic_2022_gbycommodity)
for commodity, group in historic_2022_gbycommodity:
    # Calculate the mean of price_usd for each month (1, 2, 3)
    mean_prices = group.groupby('month')['price_usd'].mean()
    if commodity == "Wheat flour (imported)": continue
    # Plot the mean price for each month
    plt.plot(mean_prices.index, mean_prices.values, label=commodity)

# Customize the plot
plt.title('Mean Price USD per Month for Each Commodity')
plt.xlabel('Month')
plt.ylabel('Mean Price USD')
plt.legend(title='Commodity')
plt.grid(True)
plt.xticks([1, 2, 3], ['Month 1', 'Month 2', 'Month 3'])  # X-axis labels for months
plt.tight_layout()
plt.show()


historic_2021 = historic[historic.year == 2021]
historic_2021_gbycommodity = historic_2021.groupby("commodity")

# Iterate through each group in the grouped DataFrame (historic_2022_gbycommodity)
for commodity, group in historic_2021_gbycommodity:
    # Calculate the mean of price_usd for each month (1, 2, 3)
    mean_prices = group.groupby('month')['price_usd'].mean()[:3]
    if commodity == "Wheat flour (imported)": continue
    # Plot the mean price for each month
    plt.plot(mean_prices.index, mean_prices.values, label=commodity)

# Customize the plot
plt.title('Mean Price USD per Month for Each Commodity')
plt.xlabel('Month')
plt.ylabel('Mean Price USD')
plt.legend(title='Commodity')
plt.grid(True)
plt.xticks([1, 2, 3], ['Month 1', 'Month 2', 'Month 3'])  # X-axis labels for months
plt.tight_layout()
plt.show()


historic_2021 = historic[historic.year == 2020]
historic_2021_gbycommodity = historic_2021.groupby("commodity")

# Iterate through each group in the grouped DataFrame (historic_2022_gbycommodity)
for commodity, group in historic_2021_gbycommodity:
    # Calculate the mean of price_usd for each month (1, 2, 3)
    mean_prices = group.groupby('month')['price_usd'].mean()[:3]
    if commodity == "Wheat flour (imported)": continue
    # Plot the mean price for each month
    plt.plot(mean_prices.index, mean_prices.values, label=commodity)

# Customize the plot
plt.title('Mean Price USD per Month for Each Commodity')
plt.xlabel('Month')
plt.ylabel('Mean Price USD')
plt.legend(title='Commodity')
plt.grid(True)
plt.xticks([1, 2, 3], ['Month 1', 'Month 2', 'Month 3'])  # X-axis labels for months
plt.tight_layout()
plt.show()


historic.commodity.value_counts()


# Drop rows where commodity doesn't include both "wheat" and "flour"
historic = historic[historic['commodity'].str.contains('wheat', case=False) & historic['commodity'].str.contains('flour', case=False)]
historic.commodity.value_counts()


historic.info()


import seaborn as sns
import matplotlib.pyplot as plt

# Plot the distribution of years (assuming 'year' is the column containing the year data)
sns.histplot(historic['year'], kde=False, bins=30)  # You can adjust the number of bins if needed

# Customize the plot
plt.title('Distribution of Years')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()


historic_after2018 = historic[historic.year >= 2018]


import seaborn as sns
import matplotlib.pyplot as plt

# Plot the distribution of price_usd for each commodity
for commodity, group in historic_after2018.groupby('commodity'):
    plt.figure(figsize=(8, 5))
    sns.histplot(group['price_usd'], kde=True, bins=30)
    plt.title(f'Price USD Distribution for {commodity}')
    plt.xlabel('Price USD')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Filter records after 2018
historic_afterJune2019 = historic[(historic['year'] > 2019) | ((historic['year'] == 2019) & (historic['month'] > 6))]

# Group by commodity, year, and month and calculate the mean of price_usd
monthly_avg_price = historic_after_2018.groupby(['commodity', 'year', 'month'])['price_usd'].mean().reset_index()

# Create a new 'year_month' column for easier plotting
monthly_avg_price['year_month'] = monthly_avg_price['year'].astype(str) + '-' + monthly_avg_price['month'].astype(str).str.zfill(2)

# Plot monthly time series for each commodity group
for commodity, group in monthly_avg_price.groupby('commodity'):
    plt.figure(figsize=(10, 6))
    group.plot(x='year_month', y='price_usd', marker='o', linestyle='-', title=f'Monthly Time Series for {commodity}', legend=False)
    plt.xlabel('Year-Month')
    plt.ylabel('Average Price USD')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Plot lon against lat for the filtered data
plt.figure(figsize=(8, 6))
plt.scatter(historic_afterJune2019['latitude'], historic_afterJune2019['longitude'], alpha=0.6, color='blue')

# Customize the plot
plt.title('Geographical Distribution: lon vs lat')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.tight_layout()
plt.show()


historic = historic.dropna()


historic = historic[historic['commodity'] == "Wheat flour"]


check_2022Mar(historic)


historic.info()


from sklearn.cluster import KMeans

# Selecting the lat and lon columns for clustering
coords = historic[['latitude', 'longitude']]

# Apply KMeans clustering (let's say we want to create 5 clusters, you can adjust the number of clusters)
kmeans = KMeans(n_clusters=5, random_state=42)
historic['cluster'] = kmeans.fit_predict(coords)

# Check the cluster assignments (optional)
print(historic[['latitude', 'longitude', 'cluster']].head())

# Filter records after June 2019
year = 2017
historic_after_june_2019 = historic[(historic['year'] > year) | ((historic['year'] == year) & (historic['month'] > 6))]

# Group by cluster, commodity, year, and month, and calculate the mean of price_usd
monthly_avg_price = historic_after_june_2019.groupby(['cluster', 'commodity', 'year', 'month'])['price_usd'].mean().reset_index()

# Create a new 'year_month' column for easier plotting (optional)
monthly_avg_price['year_month'] = monthly_avg_price['year'].astype(str) + '-' + monthly_avg_price['month'].astype(str).str.zfill(2)


pip install basemap --quiet


import matplotlib.pyplot as plt

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Scatter plot of latitude vs longitude, coloring by cluster
scatter = plt.scatter(historic['longitude'], historic['latitude'], c=historic['cluster'], cmap='tab10', s=10)

# Add title and labels
plt.title('Clustered Latitude and Longitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Add a colorbar to indicate cluster numbers
plt.colorbar(scatter, label='Cluster')

# Show the plot
plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm

# Create a figure and axis
fig, ax = plt.subplots(1, figsize=(10, 6))

# Create a Basemap instance
m = Basemap(projection='cyl', resolution='c', ax=ax)

# Draw map features
m.drawcoastlines()
m.drawcountries()

# Create a colormap with 5 distinct colors (for clusters 0-4)
cmap = cm.get_cmap('tab10', 10)  # 'tab10' has 10 distinct colors, we use the first 5

# Plot clustered latitudes and longitudes with different colors for each cluster
for cluster in historic['cluster'].unique():
    cluster_data = historic[historic['cluster'] == cluster]
    # Scatter the points with the corresponding color
    ax.scatter(cluster_data['longitude'], cluster_data['latitude'], 
               c=[cmap(cluster)]*cluster_data.shape[0], 
               s=1, label=f'Cluster {cluster}')
    
# Add legend and title
plt.legend(title='Clusters', loc='upper left', markerscale=5)
plt.title('Clustered Latitude and Longitude on Basemap')

# Show the map
plt.show()


# Plot the monthly time series for each commodity with each cluster as a separate line
for commodity, commodity_group in monthly_avg_price.groupby('commodity'):
    plt.figure(figsize=(12, 6))
    
    # Loop through each cluster and plot the time series for that cluster
    for cluster, cluster_group in commodity_group.groupby('cluster'):
        # Create a new column 'date' for plotting, combining year and month
        cluster_group['date'] = pd.to_datetime(cluster_group[['year', 'month']].assign(day=1))
        
        # Plot each cluster's price_usd time series
        plt.plot(cluster_group['date'], cluster_group['price_usd'], marker='o', label=f'Cluster {cluster}')
    
    # Customize the plot
    plt.title(f'Monthly Time Series for {commodity}')
    plt.xlabel('Date')
    plt.ylabel('Average Price USD')
    plt.xticks(rotation=45)
    plt.legend(title='Clusters')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


historic_after_june_2019 = historic_after_june_2019[historic_after_june_2019['commodity'] == "Wheat flour"]


historic_after_june_2019.info()


historic_after_june_2019.head()


from sklearn.cluster import KMeans

# Selecting the lat and lon columns for clustering
coords = historic[['latitude', 'longitude']]

# Apply KMeans clustering (let's say we want to create 5 clusters, you can adjust the number of clusters)
kmeans = KMeans(n_clusters=5, random_state=42)
historic['cluster'] = kmeans.fit_predict(coords)

# Check the cluster assignments (optional)
print(historic[['latitude', 'longitude', 'cluster']].head())

# Filter records after June 2019
year = 2017
historic_after_june_2019 = historic[(historic['year'] > year) | ((historic['year'] == year) & (historic['month'] > 6))]

historic_after_june_2019.head()


def check_2022Mar(df, year):
    return df[(df.year == year) & (df.month == 3)].shape[0]


check_2022Mar(historic_after_june_2019)


historic_after_june_2019.info()


historic_after_june_2019['conversion_rate'] = historic_after_june_2019['price_localcurrency'] / historic_after_june_2019['price_usd']


check_2022Mar(historic_after_june_2019)


historic_after_june_2019.head()


columns_to_drop = ["commodity", "unit", "currency", "date", "market_year_month", "price_localcurrency"]
historic_after_june_2019 = historic_after_june_2019.drop(columns=columns_to_drop, errors='ignore')


check_2022Mar(historic_after_june_2019)


from sklearn.preprocessing import MinMaxScaler

# Select columns that start with "igc"
igc_columns = [col for col in historic_after_june_2019.columns if col.startswith("igc")] + ['conversion_rate']

# Filter out values greater than 330
# for col in igc_columns:
#     historic_after_june_2019 = historic_after_june_2019[historic_after_june_2019[col] <= 330]

# Apply Min-Max scaling to transform values between 0 and 10
scaler = MinMaxScaler(feature_range=(0, 10))
historic_after_june_2019[igc_columns] = scaler.fit_transform(historic_after_june_2019[igc_columns])


check_2022Mar(historic_after_june_2019, 2022)


historic_after_june_2019['year'] = historic_after_june_2019['year'] - 2017


2022 - 2017


check_2022Mar(historic_after_june_2019, 4)


historic_after_june_2019.head()


import matplotlib.pyplot as plt
import seaborn as sns

# Select columns that start with "igc"
igc_columns = [col for col in historic_after_june_2019.columns if col.startswith("igc")]

# Plot distributions
plt.figure(figsize=(12, 6))
for col in igc_columns:
    sns.histplot(historic_after_june_2019[col], kde=True, label=col, alpha=0.6)

plt.legend()
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Distribution of IGC Columns")
plt.show()


historic_after_june_2019.info()


train_raw = historic_after_june_2019.reset_index(drop=True)
train_raw.to_csv("train_raw.csv", index=False)


import pandas as pd
import numpy as np

train_raw = pd.read_csv("/kaggle/input/cleaned-data/train_raw.csv")
train_raw.head()


train_raw.info()


from sklearn.preprocessing import LabelEncoder

label_encoders = {}

for col in ['admin0', 'admin1', 'admin2', 'market']:
    le = LabelEncoder()
    train_raw[col] = le.fit_transform(train_raw[col])
    label_encoders[col] = le


train_raw.head()


admin1_mapping = {index: label for index, label in enumerate(label_encoders['admin1'].classes_)}
print(dict(list(admin1_mapping.items())[:10]))


import math

def dist(coor1, coor2):
    '''Haversine distance between two pairs of (latitude, longitude) coordinates. '''
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = math.radians(coor1[0]), math.radians(coor1[1])
    lat2, lon2 = math.radians(coor2[0]), math.radians(coor2[1])
    
    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Calculate the distance
    distance = R * c
    return distance


def findNextMonth(Xt, df=train_raw):
    market, year, month = Xt.market, Xt.year, Xt.month
    if month == 12: year, month = year+1, 1
    else: month += 1
    exact_match = df[(df['market'] == market) & 
                  (df['year'] == year) & 
                  (df['month'] == month)]
    if len(exact_match) == 0: return findNextMonth_bestMatch(Xt)
    return exact_match.iloc[0]

def findNextMonth_bestMatch(Xt, df=train_raw):
    '''Given all markets and one record, find the best match based on closest
        Haversine distance. '''
    lat, lon, year, month = Xt.latitude, Xt.longitude, Xt.year, Xt.month
    
    # Filter the DataFrame for the correct year and month
    if month == 12: year, month = year+1, 1
    else: month += 1
    
    filtered_df = df[(df['year'] == year) & (df['month'] == month)]
    
    # Calculate distances for all rows using a vectorized approach
    distances = np.array([dist((lat, lon), (row_lat, row_lon)) for row_lat, row_lon in zip(filtered_df['latitude'], filtered_df['longitude'])])

    # Find the index of the closest row
    closest_index = np.argmin(distances)
    # Return the closest row from the DataFrame
    return filtered_df.iloc[closest_index]


starting_months = [7, 8, 9, 10, 11, 12, 1, 2]
start_year = 2017
years = list(np.array([2017, 2018, 2019, 2020]) - start_year)
end_year = 2021 - start_year
starting_months_2021 = [7, 8, 9, 10, 11]
window = 4 # use a window of 4 prev months to predict
iy = 5 # the fifth month is y


def buildXY(X1, df=train_raw, window=4):
    X_list = [X1]
    for _ in range(window-1): X_list.append(findNextMonth(X_list[-1]))
    X = pd.DataFrame(X_list)
    y = findNextMonth(X_list[-1])[['price_usd', 'igc_wheat', 'igc_maize', 'igc_rice', 'igc_barley', 'conversion_rate']]
    return (X, y)


train_raw.info()


def buildX(year, month, df=train_raw):
    return df[(df['year'] == year) & (df['month'] == month)]


X1df = buildX(0, 7)
X1df.head()


X, y = buildXY(X1df.iloc[0])
X.head()


y


from tqdm import tqdm


data = []

for year in years:
    for month in starting_months:
        if year == 2017 and month <= 2: continue
        X = buildX(year, month)
        print(f'Year {year + 2017} Month {month}')
        data_XY = [buildXY(x) for _, x in tqdm(X.iterrows())]
        data.extend(data_XY)

for month in starting_months_2021: 
    print(end_year + 2017, month)
    X = buildX(end_year, month)
    data_XY = [buildXY(x) for _, x in tqdm(X.iterrows())]
    data.extend(data_XY)


len(data)


data[0]


data[-1]


batch_size = 32


class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data = [(torch.tensor(X.values, dtype=torch.float32), torch.tensor(y.values, dtype=torch.float32))
                     for X, y in data_list]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Create dataset and dataloader
dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


with open("dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)


# Load the dataset
with open("/kaggle/input/cleaned-data/dataset.pkl", "rb") as f:
    loaded_dataset = pickle.load(f)

# Recreate DataLoader
dataloader = DataLoader(loaded_dataset, batch_size=batch_size, shuffle=True)


# Iterate through it
first_batch = next(iter(dataloader))
X_batch, y_batch = first_batch
print(X_batch.shape, y_batch.shape)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim

import pandas as pd
import numpy as np

import math
import time
import pickle
from tqdm import tqdm

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed) 


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)  # Apply positional encoding

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, output_dim, dropout=0.2):
        super(TimeSeriesTransformer, self).__init__()
        
        # Linear embedding for input features
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Fully connected output layer
        self.fc = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)  # Project input features to d_model
        x = self.positional_encoding(x)  # Add positional encoding
        x = self.transformer_encoder(x)  # Apply Transformer Encoder
        x = self.fc(x[:, -1, :])  # Take the last time step's representation
        return x

# Example usage
batch_size = 32
time_steps = X_batch.shape[1]
features = X_batch.shape[2]
output_dim = y_batch.shape[1]

# Initialize the model
model = TimeSeriesTransformer(input_dim=features, d_model=128, num_heads=8, num_layers=4, output_dim=output_dim)

model


# Step 1: Split the DataLoader into training and validation sets using the 0.2 threshold
train_size = int(0.8 * len(dataloader.dataset))  # 80% for training
val_size = len(dataloader.dataset) - train_size  # 20% for validation

train_dataset, val_dataset = random_split(dataloader.dataset, [train_size, val_size])

# Create the corresponding DataLoader for train and validation sets
train_loader = DataLoader(train_dataset, batch_size=dataloader.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=dataloader.batch_size, shuffle=False)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = RMSELoss()


model.eval()  # Set the model to evaluation mode

# Calculate validation loss without any training
val_loss_before = 0
with torch.no_grad():  # No gradient calculation during validation
    for X_val, y_val in val_loader:
        output = model(X_val)  # Get predictions
        loss = criterion(output, y_val)  # Compute loss
        val_loss_before += loss.item()

print(f"Validation Loss before training: {val_loss_before / len(val_loader):.4f}")


# Step 3: Train the model for 1 epoch
model.train()  # Set the model to training mode

start_time = time.time() 

for batch_idx, (X_batch, y_batch) in tqdm(enumerate(train_loader)):
    optimizer.zero_grad()  # Clear the gradients before each backprop

    # Forward pass
    output = model(X_batch)  # Get model predictions for current batch

    # Compute the loss
    loss = criterion(output, y_batch)

    # Backward pass
    loss.backward()  # Compute gradients
    optimizer.step()  # Update model weights

end_time = time.time()  # End the timer

# Calculate the time taken for 1 epoch
epoch_time = end_time - start_time
print(f"Time taken for one epoch: {epoch_time:.2f} seconds")

# Step 4: Optionally, you can validate the model after training for 1 epoch
model.eval()  # Set the model to evaluation mode

with torch.no_grad():  # No gradient calculation needed during validation
    val_loss = 0
    for X_val, y_val in val_loader:
        output = model(X_val)  # Get model predictions for validation batch
        loss = criterion(output, y_val)
        val_loss += loss.item()

    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")


def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, report_every_n_epochs=5):
    for epoch in range(1, num_epochs + 1):
        # Step 1: Train the model for one epoch
        model.train()  # Set the model to training mode

        start_time = time.time()  # Start the timer

        for batch_idx, (X_batch, y_batch) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()  # Clear the gradients before each backprop

            # Forward pass
            output = model(X_batch)  # Get model predictions for the current batch

            # Compute the loss
            loss = criterion(output, y_batch)

            # Backward pass
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model weights

        end_time = time.time()  # End the timer

        # Calculate the time taken for 1 epoch
        epoch_time = end_time - start_time

        if epoch % report_every_n_epochs == 0:
            print(f"\nEpoch {epoch}/{num_epochs} - Time taken: {epoch_time:.2f} seconds")

        # Step 2: Validate the model after training
        model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # No gradient calculation needed during validation
        val_loss = 0
        for X_val, y_val in val_loader:
            output = model(X_val)  # Get model predictions for validation batch
            loss = criterion(output, y_val)
            val_loss += loss.item()

        print(f"Final Validation Loss: {val_loss / len(val_loader):.4f}")


train(model=model, 
      train_loader=train_loader, 
      val_loader=val_loader, 
      optimizer=optimizer, 
      criterion=criterion, 
      num_epochs=5, 
      report_every_n_epochs=1)


# Save the model's state_dict
torch.save(model.state_dict(), 'model_RMSE.pth')  # Save to the current directory
print("Model saved!")


import pandas as pd
import numpy as np


test = pd.read_csv("/kaggle/input/bpl-ai4good-wheat-price-forecasting/test.csv")
test.head()


train_raw = pd.read_csv("/kaggle/input/cleaned-data/train_raw.csv")
train_raw.head()


market_encoder = label_encoders["market"]


markets = market_encoder.inverse_transform(train_raw['market'])
unique_markets = set(pd.unique(markets))


test_markets = set(test.market_year_month.apply(lambda x: x.split("_")[0]))


test_markets - unique_markets


market = pd.Series(test.market_year_month.apply(lambda x: x.split("_")[0]))
market[market == 'National Average']


test.iloc[127]


coordinates = {
    'Bihar': (25.0961, 85.3131),
    'Kohima': (25.6700, 94.1190),
    'Maharashtra': (19.0760, 72.8777),
    'Sherpur Sadar': (24.0833, 88.5833),
    'Tehran Market': (35.6892, 51.3890)
}


'Bihar' in coordinates.keys()


market_encoder = label_encoders["market"]


def find_coor(market, df=train_raw):
    markets = df[df.market == market]
    return markets.latitude.values[0], markets.longitude.values[0]

def initial_test_data(testX):
    start_year = 2021 - 2017
    start_month = 11
    lat, lon = 0, 0
    market = testX.split("2022")[0][:-1]
    original_market = market
    if market == 'National Average': 
        return original_market, market, start_year, start_month, lat, lon
    if market in coordinates.keys(): 
        lat, lon = coordinates[market][0], coordinates[market][1]
    else: 
        market = market_encoder.transform([market])[0]
        lat, lon = find_coor(market)
    return original_market, market, start_year, start_month, lat, lon


test_df = pd.DataFrame(list(test.market_year_month.apply(initial_test_data)), 
                       columns=['market_str', 'market', 'year', 'month', 'latitude', 'longitude'])
test_df.head()


test_df.info()


test_df = test_df[test_df.market != 'National Average']
test_df.info()


test_df.to_csv("test.csv", index=False)


def buildXY_test(X1, df=train_raw, window=5):
    X_list = [X1]
    for _ in range(window-1): X_list.append(findNextMonth(X_list[-1]))
    X = pd.DataFrame(X_list[1:])
    return X


buildXY_test(test_df.iloc[0])


test_original = test.drop([127, 128], axis=0).reset_index(drop=True)
test_markets = test_original.market_year_month.apply(lambda x: x.split("_")[0])


type(test_markets)


from tqdm import tqdm


# Load the trained model
model_path = '/kaggle/input/cleaned-data/model_RMSE.pth'
model.load_state_dict(torch.load('/kaggle/input/cleaned-data/model.pth'))
model.eval()  # Set model to evaluation mode

# Assume test_df is provided and buildXY function exists
y_cols = ['price_usd', 'igc_wheat', 'igc_maize', 'igc_rice', 'igc_barley', 'conversion_rate']
predictions = pd.DataFrame(columns = ['market', 'year' ,'month'] + y_cols)  # To store April, May, and June predictions

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="predicting"):
    # Get initial X from buildXY
    market = row.market_str
    row = row.drop("market_str")
    
    X = buildXY_test(row)  # X is a DataFrame with 4 months of data

    year = 2022
    for month in [4, 5, 6]:
        # Convert X to tensor
        X_tensor = torch.tensor(X.values, dtype=torch.float32).unsqueeze(0)  # Add batch dim

        # Predict next time step
        with torch.no_grad():
            y_pred = model(X_tensor).squeeze(0).numpy()  # Get prediction as NumPy array

        X.loc[len(X)] = X.iloc[-1]

        # Update the last row with y_pred values
        X.iloc[-1, 8:13] = y_pred[:5]  # Replace indices 8 to 12
        X.iloc[-1, 14] = y_pred[5]      # Replace index 14
        
        # Drop the first row
        X = X.iloc[1:].reset_index(drop=True)
        result = [market, year, month]
        result.extend(y_pred)
        predictions.loc[len(predictions)] = result # append the prediction result for (market, month) to predictions df

predictions.head()


predictions.to_csv("predictions.csv", index=False)


# Create 'market_year_month' by concatenating 'market', 'year', and 'month' with underscores
predictions['market_year_month'] = predictions['market'].astype(str) + "_" + \
                                   predictions['year'].astype(str) + "_" + \
                                   predictions['month'].astype(str)
# Select the required columns for submission
submission = predictions[['market_year_month', 'price_usd']]
submission = submission[submission['market_year_month'].isin(test['market_year_month'])]
submission = submission.drop_duplicates()
# Calculate the average of the 'price_usd' column
avg_price = submission['price_usd'].mean()

# Create two new rows with 'market_year_month' and 'price_usd' values
new_rows = pd.DataFrame({
    'market_year_month': ['National Average_2022_4', 'National Average_2022_5'],
    'price_usd': [avg_price, avg_price]
})

# Insert the new rows after index 126
submission = pd.concat([submission.iloc[:127], new_rows, submission.iloc[127:]], ignore_index=True)
# Reindex the entire dataframe after insertion
submission = submission.reset_index(drop=True)

# Save to CSV without index
submission.to_csv("submission.csv", index=False)

print("submission.csv saved successfully!")


submission.shape


submission.head(20)




