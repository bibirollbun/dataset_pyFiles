import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans  # FIXED
from sklearn.metrics import silhouette_score

# Load the dataset
file_path = "/kaggle/input/cities/Cities.csv"  # Ensure the correct file path
df = pd.read_csv(file_path)

# Display first few rows
print("Initial Data Sample:")
print(df.head())

# Data Cleaning
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Display after cleaning
print("Data after Cleaning:")
print(df.head())

# Data Processing
le = LabelEncoder()
df['State'] = le.fit_transform(df['State'].astype(str))  # Convert to string before encoding

# Feature Selection
features = ['CityID', 'State']
X = df[features]

# Data Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Output confirmation
print("Data Processing and Scaling Done.")

# Clustering Model
kmeans = KMeans(n_clusters=5, random_state=42)  # Initialize KMeans with 5 clusters
df['Cluster'] = kmeans.fit_predict(X_scaled)  # Assign clusters to the dataset

# Model Evaluation
silhouette_avg = silhouette_score(X_scaled, df['Cluster'])
print(f'Silhouette Score: {silhouette_avg}')  # Print how well the clustering performed

# Display sample output
print("Clustering Done. Sample Data:")
print(df.head())

# Visualization of clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['CityID'], y=df['State'], hue=df['Cluster'], palette='viridis')
plt.title('City Clustering')
plt.show()



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

