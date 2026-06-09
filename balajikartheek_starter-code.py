# Data manipulation
import pandas as pd  
import numpy as np   

# Database/SQL
import sqlite3       
from pandasql import sqldf 

import matplotlib.pyplot as plt
import seaborn as sns

# Interactive plots 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Time series/trend analysis
from statsmodels.tsa.seasonal import seasonal_decompose

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# External data sources
import kagglehub

# System Utilities
import os               
import re                
from tqdm import tqdm   
import warnings
warnings.filterwarnings('ignore')


# Download datasets
meta_kaggle_data_path = kagglehub.dataset_download("kaggle/meta-kaggle")
meta_kaggle_code_path = kagglehub.dataset_download("kaggle/meta-kaggle-code")

print(f"Meta Kaggle Data path: {meta_kaggle_data_path}")
print(f"Meta Kaggle Code path: {meta_kaggle_code_path}")


import kagglehub
import os

# Download datasets
meta_kaggle_path = kagglehub.dataset_download("kaggle/meta-kaggle")
meta_kaggle_code_path = kagglehub.dataset_download("kaggle/meta-kaggle-code")

print(f"Meta Kaggle Data path: {meta_kaggle_path}")
print(f"Meta Kaggle Code path: {meta_kaggle_code_path}\n")

# List top-level directories and key files
def list_sample(path, sample_size=8):
    print(f"\n{path.split('/')[-1].upper()} STRUCTURE:")
    items = os.listdir(path)
    print(f"Total items: {len(items)}")
    print(f"Sample items:")
    for item in items[:sample_size]:
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            print(f"ğŸ“� {item}/ [DIR]")
        else:
            print(f"ğŸ“„ {item} ({round(os.path.getsize(item_path)/1024**2, 2)} MB)")




list_sample(meta_kaggle_code_path)


list_sample(meta_kaggle_path)


# Load and preview key CSV files from Meta Kaggle
def load_and_preview_csv(folder_path, file_name, nrows=5):
    file_path = os.path.join(folder_path, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, nrows=nrows)
        print(f"\nPreview of '{file_name}':")
        print(df.head(nrows))
    else:
        print(f"{file_name} not found in {folder_path}")


# Preview some important CSVs
print("\nPREVIEW OF KEY Data FILES:\n")
load_and_preview_csv(meta_kaggle_path, "Users.csv")
load_and_preview_csv(meta_kaggle_path, "Organizations.csv")
load_and_preview_csv(meta_kaggle_path, "UserOrganizations.csv")
load_and_preview_csv(meta_kaggle_path, "UserAchievements.csv")
load_and_preview_csv(meta_kaggle_path, "UserFollowers.csv")


# Load datasets
def load_data(path):
    print(f'Reading the Data: {path}')
    return pd.read_csv(path)

users = load_data(os.path.join(meta_kaggle_path, "Users.csv"))
orgs = load_data(os.path.join(meta_kaggle_path, "Organizations.csv"))
user_orgs = load_data(os.path.join(meta_kaggle_path, "UserOrganizations.csv"))
# achievements = load_data(os.path.join(meta_kaggle_path, "UserAchievements.csv")) # This is 7.33GB
followers = load_data(os.path.join(meta_kaggle_path, "UserFollowers.csv"))


!pip install dask


from dask import dataframe as dd

file_path = os.path.join(meta_kaggle_path, "UserAchievements.csv")

achievements = dd.read_csv(file_path)

achievements.head()


from dask import dataframe as dd
import os

# Define paths
meta_kaggle_path = "./meta-kaggle"  # Update this path if needed
csv_file = (os.path.join(meta_kaggle_path, "UserAchievements.csv"))
parquet_file = os.path.join(meta_kaggle_path, "UserAchievements.parquet")


# # Read CSV with Dask
# ddf = dd.read_csv(csv_file)

# # Save as Parquet
# ddf.to_parquet(parquet_file)


users.head()


# Clean RegisterDate and JoinDate for time-based analysis
users['RegisterDate'] = pd.to_datetime(users['RegisterDate'], errors='coerce')
user_orgs['JoinDate'] = pd.to_datetime(user_orgs['JoinDate'], errors='coerce')


# 1. Top Performing Users
top_users = users.sort_values(by='PerformanceTier', ascending=False).head(10)
print("\nTop Users by Performance Tier:")
print(top_users[['DisplayName', 'RegisterDate', 'PerformanceTier', 'Country']].to_string())


# 2. Most Active Organizations
org_user_counts = user_orgs.groupby('OrganizationId').size().reset_index(name='MemberCount')
top_orgs = org_user_counts.merge(orgs[['Id', 'Name']], left_on='OrganizationId', right_on='Id')
top_orgs = top_orgs.sort_values(by='MemberCount', ascending=False).head(10)
print("\nTop Organizations by Members:")
print(top_orgs[['Name', 'MemberCount']].to_string())


# 3. User Achievements
user_achievements = achievements.groupby('UserId')['Points'].sum().reset_index()
top_achievers = user_achievements.merge(users[['Id', 'DisplayName']], left_on='UserId', right_on='Id')
top_achievers = top_achievers.sort_values(by='Points', ascending=False).head(10)
print("\nTop Achievers by Points:")
print(top_achievers[['DisplayName', 'Points']].to_string())


# 4. User Followers
follower_counts = followers.groupby('FollowingUserId').size().reset_index(name='FollowerCount')
top_followed = follower_counts.merge(users[['Id', 'DisplayName']], left_on='FollowingUserId', right_on='Id')
top_followed = top_followed.sort_values(by='FollowerCount', ascending=False).head(10)
print("\nMost Followed Users:")
print(top_followed[['DisplayName', 'FollowerCount']].to_string())


# Visualization: Top Organizations
plt.figure(figsize=(10,6))
sns.barplot(x='MemberCount', y='Name', data=top_orgs, palette="viridis")
plt.title("Top Organizations by Member Count")
plt.xlabel("Number of Members")
plt.ylabel("Organization")
plt.tight_layout()
plt.show()


# Visualization: Top Achievers
plt.figure(figsize=(10,6))
sns.barplot(x='Points', y='DisplayName', data=top_achievers, palette="coolwarm")
plt.title("Top Achievers by Points")
plt.xlabel("Total Achievement Points")
plt.ylabel("User")
plt.tight_layout()
plt.show()


users['RegisterYear'] = pd.to_datetime(users['RegisterDate']).dt.year
performance_trend = users.groupby('RegisterYear')['PerformanceTier'].value_counts().unstack(fill_value=0)
performance_trend.plot(kind='line', title="User Performance Tier Growth Over Time")


org_members = user_orgs.groupby('OrganizationId').size().reset_index(name='MemberCount')
top_orgs = org_members.merge(orgs[['Id', 'Name']], left_on='OrganizationId', right_on='Id')
top_orgs.sort_values(by='MemberCount', ascending=False).head(10).plot.bar(x='Name', y='MemberCount', title="Top Organizations by Members")


top_users = users[users['PerformanceTier'] >= 4]  # Gold or Silver+
country_counts = top_users['Country'].value_counts().head(10)
country_counts.plot(kind='barh', title="Top Countries by High Performing Users")


users['RegYear'] = pd.to_datetime(users['RegisterDate']).dt.year
retention = users.groupby('RegYear').size()
retention.plot(kind='line', title="User Registration Trend Over Years")


# Count users per country
country_counts = users['Country'].value_counts().reset_index()
country_counts.columns = ['Country', 'UserCount']

# Keep top N countries (e.g., top 10)
top_countries = country_counts.head(10)
print(top_countries)


# Sample lat/lon for top countries
country_coords = {
    'United States': [37.0902, -95.7129],
    'India': [20.5937, 78.9629],
    'Brazil': [-14.2350, -51.9253],
    'Russia': [61.5240, 105.3188],
    'China': [35.8617, 104.1954],
    'Germany': [51.1657, 10.4515],
    'Japan': [36.204823, 138.252930],
    'Canada': [56.1304, -106.3468],
    'France': [46.2276, 2.2137],
    'United Kingdom': [55.3781, -3.4360],
    'Australia': [-25.2744, 133.7751]
}

# Map coordinates to each row
top_countries['lat'] = top_countries['Country'].map(lambda x: country_coords.get(x, [None, None])[0])
top_countries['lon'] = top_countries['Country'].map(lambda x: country_coords.get(x, [None, None])[1])

print(top_countries)


import folium
world_map = folium.Map()
# Add markers for each country with top users
for _, row in top_countries.iterrows():
    folium.Marker([row['lat'], row['lon']], popup=row['Country']).add_to(world_map)
world_map.save("kaggle_users_map.html")
print("Map saved as 'kaggle_users_map.html'")


from IPython.display import IFrame

# Show the map in an IFrame
display(IFrame(src='kaggle_users_map.html', width=800, height=500))




