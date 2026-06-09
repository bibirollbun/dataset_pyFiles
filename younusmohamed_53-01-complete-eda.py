import ast
import folium
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from collections import Counter
from folium.plugins import MarkerCluster

# Configure plots for inline display (if using Jupyter)
%matplotlib inline


# Load CSV files
train_df = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")
taxonomy_df = pd.read_csv("/kaggle/input/birdclef-2025/taxonomy.csv")
sample_submission = pd.read_csv("/kaggle/input/birdclef-2025/sample_submission.csv")

# Load location metadata from the text file
with open("/kaggle/input/birdclef-2025/recording_location.txt", "r") as f:
    recording_location = f.read()

print("Train data shape:", train_df.shape)
print("Taxonomy data shape:", taxonomy_df.shape)
print("Sample submission shape:", sample_submission.shape)


# Check for missing values
print(train_df.info())
print('\n',train_df.isnull().sum())

# Check for duplicates
print('\n',"Duplicates in train.csv:", train_df.duplicated().sum())


plt.figure(figsize=(12,6))
train_df['primary_label'].value_counts().plot(kind='bar')
plt.title("Distribution of Primary Species Labels")
plt.xlabel("Species")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(8,6))
sns.histplot(train_df['rating'], bins=30, kde=True)
plt.title("Distribution of Audio Quality Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()


# Compute correlation matrix (extend list if additional numerical features are added)
corr_matrix = train_df[['rating']].copy()
print(corr_matrix.corr())

sns.heatmap(corr_matrix.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


avg_rating_by_species = train_df.groupby('primary_label')['rating'].mean().sort_values(ascending=False)
plt.figure(figsize=(12,6))
avg_rating_by_species.plot(kind='bar')
plt.title("Average Rating by Species")
plt.xlabel("Species")
plt.ylabel("Average Rating")
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(8,6))
train_df['type'].value_counts().plot(kind='bar')
plt.title("Distribution of Recording Types")
plt.xlabel("Type")
plt.ylabel("Count")
plt.show()


# Convert secondary_labels from string to list and flatten the results
secondary_labels_list = []
for label in train_df['secondary_labels']:
    try:
        parsed = ast.literal_eval(label) if label and label != "[]" else []
    except:
        parsed = []
    secondary_labels_list.extend(parsed)

secondary_counter = Counter(secondary_labels_list)
print("Most common secondary labels:", secondary_counter.most_common(10))


plt.figure(figsize=(8,6))
taxonomy_df['class_name'].value_counts().plot(kind='bar')
plt.title("Distribution of Taxonomic Classes")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()


merged_df = pd.merge(train_df, taxonomy_df, left_on="primary_label", right_on="primary_label", how="left")
print("Merged data shape:", merged_df.shape)
print("Missing taxonomy info:", merged_df['class_name'].isnull().sum())


def count_files(directory):
    return sum([len(files) for _, _, files in os.walk(directory)])

print("Train Audio files:", count_files("/kaggle/input/birdclef-2025/train_audio"))
print("Train Soundscapes files:", count_files("/kaggle/input/birdclef-2025/train_soundscapes"))
print("Test Soundscapes files:", count_files("/kaggle/input/birdclef-2025/test_soundscapes"))


plt.figure(figsize=(8, 6))
sns.boxplot(x=train_df['rating'])
plt.title("Boxplot of Ratings")
plt.xlabel("Rating")
plt.show()

print("Rating Summary Statistics:")
print(train_df['rating'].describe())

# Calculate the interquartile range (IQR)
Q1 = train_df['rating'].quantile(0.25)
Q3 = train_df['rating'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f"Rating outlier bounds: Lower = {lower_bound:.2f}, Upper = {upper_bound:.2f}")

# Identify potential rating outliers
rating_outliers = train_df[(train_df['rating'] < lower_bound) | (train_df['rating'] > upper_bound)]
print("Number of rating outliers:", rating_outliers.shape[0])
print("Sample rating outliers:")
print(rating_outliers[['rating']].head())


# Remove rows with missing coordinate values
location_data = train_df.dropna(subset=['latitude', 'longitude'])

# Scatterplot of recording locations
plt.figure(figsize=(8, 6))
sns.scatterplot(x='longitude', y='latitude', data=location_data, alpha=0.5)
plt.title("Scatter Plot of Recording Locations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Calculate IQR for latitude
lat_Q1 = location_data['latitude'].quantile(0.25)
lat_Q3 = location_data['latitude'].quantile(0.75)
lat_IQR = lat_Q3 - lat_Q1
lat_lower_bound = lat_Q1 - 1.5 * lat_IQR
lat_upper_bound = lat_Q3 + 1.5 * lat_IQR

# Calculate IQR for longitude
lon_Q1 = location_data['longitude'].quantile(0.25)
lon_Q3 = location_data['longitude'].quantile(0.75)
lon_IQR = lon_Q3 - lon_Q1
lon_lower_bound = lon_Q1 - 1.5 * lon_IQR
lon_upper_bound = lon_Q3 + 1.5 * lon_IQR

print(f"Latitude bounds: {lat_lower_bound:.2f} - {lat_upper_bound:.2f}")
print(f"Longitude bounds: {lon_lower_bound:.2f} - {lon_upper_bound:.2f}")

# Identify geographical outliers
geo_outliers = location_data[
    (location_data['latitude'] < lat_lower_bound) | (location_data['latitude'] > lat_upper_bound) |
    (location_data['longitude'] < lon_lower_bound) | (location_data['longitude'] > lon_upper_bound)
]
print("Number of geographical outliers:", geo_outliers.shape[0])
print("Sample geographical outliers:")
print(geo_outliers[['latitude', 'longitude']].head())


# Calculate the center of the map using the mean coordinates
mean_lat = location_data['latitude'].mean()
mean_lon = location_data['longitude'].mean()

# Initialize the folium map
m = folium.Map(location=[mean_lat, mean_lon], zoom_start=8)

# Add a marker cluster to group nearby points
marker_cluster = MarkerCluster().add_to(m)

# Add markers for each recording location with a popup showing the primary species label
for idx, row in location_data.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Species: {row['primary_label']}"
    ).add_to(marker_cluster)

# Optionally, save the map to an HTML file
m.save("recording_locations_map.html")
print("Interactive map saved as 'recording_locations_map.html'.")


species_counts = train_df['primary_label'].value_counts()
plt.figure(figsize=(12, 6))
species_counts.plot(kind='bar')
plt.title("Distribution of Primary Species Labels")
plt.xlabel("Species")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()

print("Species count statistics:")
print(species_counts.describe())


collection_counts = train_df['collection'].value_counts()
plt.figure(figsize=(8, 6))
collection_counts.plot(kind='bar')
plt.title("Distribution of Recordings by Collection Source")
plt.xlabel("Collection")
plt.ylabel("Count")
plt.show()

print("Collection counts:")
print(collection_counts)


sample_submission.to_csv('submission.csv')







