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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')


print("=" * 60)
print("STEP 1: LOADING AND EXPLORING DATA")
print("=" * 60)

# Load dataset
df = pd.read_csv('/kaggle/input/penguin-clustering-analysis/penguins.csv')

print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head(10))

print(f"\nData types:")
print(df.dtypes)

print(f"\nMissing values:")
print(df.isnull().sum())

print(f"\nBasic statistics:")
print(df.describe())


print("\n" + "=" * 60)
print("STEP 2: DATA PREPROCESSING")
print("=" * 60)

# Remove rows with missing values
df_clean = df.dropna()
print(f"\nDataset shape after removing NaN: {df_clean.shape}")

# Select features for clustering
features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
X = df_clean[features].copy()

print(f"\nSelected features: {features}")
print(f"\nFeature statistics before normalization:")


print("\n" + "=" * 60)
print("STEP 3: DATA NORMALIZATION")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nData normalized using StandardScaler")
print(f"\nNormalized data statistics:")
print(f"Mean: {X_scaled.mean(axis=0)}")
print(f"Std: {X_scaled.std(axis=0)}")


print("\n" + "=" * 60)
print("STEP 4: DETERMINING OPTIMAL NUMBER OF CLUSTERS (ELBOW METHOD)")
print("=" * 60)

inertias = []
silhouette_scores = []
davies_bouldin_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans_temp.labels_))
    davies_bouldin_scores.append(davies_bouldin_score(X_scaled, kmeans_temp.labels_))

print(f"\nInertia values for different k:")
for k, inertia in zip(K_range, inertias):
    print(f"  k={k}: {inertia:.2f}")

print(f"\nSilhouette scores for different k:")
for k, score in zip(K_range, silhouette_scores):
    print(f"  k={k}: {score:.4f}")

print(f"\nDavies-Bouldin scores for different k:")
for k, score in zip(K_range, davies_bouldin_scores):
    print(f"  k={k}: {score:.4f}")

# Visualize elbow curve and silhouette scores
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Elbow curve
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)', fontsize=11)
axes[0].set_ylabel('Inertia', fontsize=11)
axes[0].set_title('Elbow Method For Optimal k', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Silhouette score
axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (k)', fontsize=11)
axes[1].set_ylabel('Silhouette Score', fontsize=11)
axes[1].set_title('Silhouette Score vs k', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Davies-Bouldin Index (lower is better)
axes[2].plot(K_range, davies_bouldin_scores, 'ro-', linewidth=2, markersize=8)
axes[2].set_xlabel('Number of Clusters (k)', fontsize=11)
axes[2].set_ylabel('Davies-Bouldin Index', fontsize=11)
axes[2].set_title('Davies-Bouldin Index vs k\n(Lower is Better)', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('elbow_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nOptimal k is likely 3 (based on elbow method and silhouette score)")



print("\n" + "=" * 60)
print("STEP 5: APPLYING K-MEANS WITH OPTIMAL K=3")
print("=" * 60)

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

print(f"\nClusters assigned to each sample: {len(clusters)} samples")
print(f"Cluster distribution:")
unique, counts = np.unique(clusters, return_counts=True)
for cluster_id, count in zip(unique, counts):
    print(f"  Cluster {cluster_id}: {count} samples ({count/len(clusters)*100:.1f}%)")


print("\n" + "=" * 60)
print("STEP 6: CLUSTERING EVALUATION METRICS")
print("=" * 60)

# Calculate metrics
inertia = kmeans.inertia_
silhouette_avg = silhouette_score(X_scaled, clusters)
davies_bouldin = davies_bouldin_score(X_scaled, clusters)
calinski_harabasz = calinski_harabasz_score(X_scaled, clusters)

print(f"\nInertia (Sum of Squared Errors): {inertia:.2f}")
print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"  (Range: -1 to 1, closer to 1 is better)")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
print(f"  (Minimum value indicates better separation)")
print(f"Calinski-Harabasz Index: {calinski_harabasz:.2f}")
print(f"  (Higher values indicate better-defined clusters)")


print("\n" + "=" * 60)
print("STEP 7: DETAILED CLUSTER ANALYSIS")
print("=" * 60)

df_clean['Cluster'] = clusters

for i in range(optimal_k):
    print(f"\n--- CLUSTER {i} ---")
    cluster_data = df_clean[df_clean['Cluster'] == i][features]
    print(cluster_data.describe())


print("\n" + "=" * 60)
print("STEP 8: VISUALIZATION")
print("=" * 60)

# Color palette
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# 2D scatter plots for different feature pairs
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

feature_pairs = [
    ('culmen_length_mm', 'culmen_depth_mm'),
    ('culmen_length_mm', 'flipper_length_mm'),
    ('culmen_length_mm', 'body_mass_g'),
    ('culmen_depth_mm', 'flipper_length_mm'),
    ('culmen_depth_mm', 'body_mass_g'),
    ('flipper_length_mm', 'body_mass_g')
]

for idx, (feat1, feat2) in enumerate(feature_pairs):
    ax = axes[idx // 3, idx % 3]
    for i in range(optimal_k):
        mask = clusters == i
        ax.scatter(df_clean[mask][feat1], df_clean[mask][feat2], 
                  label=f'Cluster {i}', alpha=0.6, s=50, color=colors[i])
    
    # Plot centroids
    centroid_feat1 = scaler.inverse_transform(kmeans.cluster_centers_)[:, features.index(feat1)]
    centroid_feat2 = scaler.inverse_transform(kmeans.cluster_centers_)[:, features.index(feat2)]
    ax.scatter(centroid_feat1, centroid_feat2, marker='X', s=300, 
              color='black', edgecolors='white', linewidths=2, label='Centroids')
    
    ax.set_xlabel(feat1, fontsize=10)
    ax.set_ylabel(feat2, fontsize=10)
    ax.set_title(f'{feat1} vs {feat2}', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cluster_scatter_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Box plots for each feature by cluster
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, feature in enumerate(features):
    ax = axes[idx]
    data_to_plot = [df_clean[df_clean['Cluster'] == i][feature].values for i in range(optimal_k)]
    bp = ax.boxplot(data_to_plot, labels=[f'C{i}' for i in range(optimal_k)], patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel(feature, fontsize=11)
    ax.set_title(f'{feature} Distribution by Cluster', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('cluster_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# Heatmap of cluster centroids
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids_original, columns=features)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(centroids_df.T, annot=True, fmt='.1f', cmap='YlOrRd', 
            xticklabels=[f'Cluster {i}' for i in range(optimal_k)],
            cbar_kws={'label': 'Feature Value'}, ax=ax)
ax.set_title('Cluster Centroids Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('cluster_centroids_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nAll visualizations saved successfully!")

