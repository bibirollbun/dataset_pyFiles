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


from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Read and Combine data
df_train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
df_origin = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')
df_train = df_train.drop('id', axis=1)
df_combined = pd.concat([df_train, df_origin], ignore_index=True)
print(f"   - Combination complete! Full dataset size: {df_combined.shape}")


# Grouping
print("\n2. Grouping songs into 'Slow', 'Medium', and 'Fast' based on BPM...")

# Define the bin edges and corresponding labels
# Bins: [0, 85), [85, 145), [145, max+1)
bin_edges = [0, 85, 145, df_combined['BeatsPerMinute'].max() + 1]
bin_labels = ['Slow', 'Medium', 'Fast']

# Use pd.cut to create a new 'BPM_Bin' column
df_combined['BPM_Bin'] = pd.cut(
    df_combined['BeatsPerMinute'],
    bins=bin_edges,
    labels=bin_labels,
    right=False  
)
print("   - Grouping complete!")

# (Optional) Print the number of songs in each group
print("\n--- Number of songs in each BPM group ---")
print(df_combined['BPM_Bin'].value_counts())


# --- Step 3: Calculate the mean of each feature for each group ---
print("\n3. Calculating the mean of each feature within each group...")
feature_means_by_bin = df_combined.drop('BeatsPerMinute', axis=1).groupby('BPM_Bin').mean()


print("\n--- Mean Feature Values for Each BPM Group ---")
print(feature_means_by_bin.T)


def calculate_eta_squared(df, feature_col, group_col):
    """Manually calculate Eta Squared (effect size) for ANOVA"""
    
    groups = [group_data[feature_col].values for name, group_data in df.groupby(group_col)]
    
    # Calculate the overall mean
    overall_mean = df[feature_col].mean()
    
    # Calculate the sum of squares between groups (SS_between)
    ss_between = sum(len(group) * (group.mean() - overall_mean)**2 for group in groups)
    
    # Calculate the total sum of squares (SS_total)
    ss_total = sum((x - overall_mean)**2 for x in df[feature_col])
    
    # Calculate Eta Squared
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    return eta_squared


anova_results = []
features_to_test = [col for col in df_combined.columns if col not in ['BeatsPerMinute', 'BPM_Bin']]

for feature in features_to_test:
    
    groups = [
        df_combined[feature][df_combined['BPM_Bin'] == 'Slow'],
        df_combined[feature][df_combined['BPM_Bin'] == 'Medium'],
        df_combined[feature][df_combined['BPM_Bin'] == 'Fast']
    ]
    
    # One-way ANOVA
    f_statistic, p_value = f_oneway(*groups)
    eta_sq = calculate_eta_squared(df_combined, feature, 'BPM_Bin')
    anova_results.append({
        'Feature': feature,
        'F-statistic': f_statistic,
        'p-value': p_value,
        'Eta Squared (η²)': eta_sq
    })
    
results_df = pd.DataFrame(anova_results)
results_df['Statistically Significant (p < 0.05)?'] = results_df['p-value'] < 0.05


print("\n--- ANOVA Test Results ---")
print(results_df.to_string())


df_combined = pd.concat([df_train, df_origin], ignore_index=True)
X = df_combined.drop('BeatsPerMinute', axis=1)
y = df_combined['BeatsPerMinute']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


k = 6
print(f"\nPerforming K-Means clustering on the full dataset with k={k}...")
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(X_scaled)


df_combined['Cluster'] = cluster_labels
print("   - Clustering complete! 'Cluster' column has been added to the DataFrame.")


print("\nAnalyzing clustering results...")
# Calculate the size of each cluster
cluster_sizes = df_combined['Cluster'].value_counts().sort_index()
cluster_analysis = df_combined.groupby('Cluster').mean()
cluster_analysis['ClusterSize'] = cluster_sizes


print(cluster_analysis.T)

