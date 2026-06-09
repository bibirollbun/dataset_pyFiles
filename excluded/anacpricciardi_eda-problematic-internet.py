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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Reload the train.csv file
train_path = '/kaggle/input/child-mind-institute-problematic-internet-use/train.csv'
train_df = pd.read_csv(train_path)

# Display general information about the dataset to confirm successful loading
train_info = train_df.info()
train_info




# Step 1: Distribution of the target variable (sii)
sii_distribution = train_df['sii'].value_counts(normalize=True).sort_index()

# Plot the distribution of 'sii'
plt.figure(figsize=(8, 5))
sns.barplot(x=sii_distribution.index, y=sii_distribution.values, palette='viridis')
plt.title('Distribution of Severity Impairment Index (sii)', fontsize=14)
plt.xlabel('sii (Severity Level)', fontsize=12)
plt.ylabel('Proportion of Participants', fontsize=12)
plt.xticks(ticks=[0, 1, 2, 3], labels=['None', 'Mild', 'Moderate', 'Severe'])
plt.show()



# Step 2: Missing data visualization
missing_data = train_df.isnull().mean().sort_values(ascending=False)

# Plot the missing data rates
plt.figure(figsize=(12, 6))
sns.barplot(x=missing_data.index[:30], y=missing_data.values[:30], palette='coolwarm')
plt.title('Top 30 Features with Highest Missing Rates', fontsize=14)
plt.ylabel('Proportion of Missing Values', fontsize=12)
plt.xticks(rotation=90)
plt.show()







# Step 3: Correlation with sii
# Ensure 'id' and 'sii' are dropped only if they exist in the DataFrame
numeric_features = train_df.select_dtypes(include=['float64', 'int64']).copy()
columns_to_drop = [col for col in ['id', 'sii'] if col in numeric_features.columns]
numeric_features = numeric_features.drop(columns=columns_to_drop, errors='ignore')

# Compute correlations with 'sii' safely
correlations = numeric_features.corrwith(train_df['sii']).sort_values(ascending=False)

# Plot the top 10 correlations with 'sii'
plt.figure(figsize=(10, 6))
sns.barplot(x=correlations.index[:10], y=correlations.values[:10], palette='magma')
plt.title('Top 10 Features Correlated with sii', fontsize=14)
plt.ylabel('Correlation Coefficient', fontsize=12)
plt.xticks(rotation=45)
plt.show()




# Step 4: Advanced Insight - Feature Interaction Analysis
# Explore potential interactions between top correlated features
# Select the top 2 positively and negatively correlated features
positive_features = correlations.head(2).index.tolist()
negative_features = correlations.tail(2).index.tolist()
selected_features = positive_features + negative_features

# Pairplot to visualize potential interactions
sns.pairplot(train_df, vars=selected_features, hue='sii', palette='coolwarm', corner=True)
plt.suptitle('Feature Interactions and sii', y=1.02, fontsize=16)
plt.show()


# Step 5: Advanced Insight - Clustering based on correlated features
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Scale the data for clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numeric_features[selected_features].dropna())

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_clusters = kmeans.fit_predict(scaled_features)

# Add clusters to the dataset
clustered_data = train_df.dropna(subset=selected_features).copy()
clustered_data['Cluster'] = kmeans_clusters

# Visualize clusters with a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=clustered_data[selected_features[0]],
    y=clustered_data[selected_features[1]],
    hue=clustered_data['Cluster'],
    palette='tab10',
    style=clustered_data['sii']
)
plt.title('Clusters based on Key Features', fontsize=14)
plt.xlabel(selected_features[0], fontsize=12)
plt.ylabel(selected_features[1], fontsize=12)
plt.legend(title='Cluster/SII')
plt.show()





