import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import warnings

# Suppress FutureWarning related to use_inf_as_na deprecation in seaborn/pandas
warnings.filterwarnings("ignore", category=FutureWarning, message=".*use_inf_as_na.*")

# Suppress the specific UserWarning about figure layout change in seaborn
warnings.filterwarnings("ignore", category=UserWarning, message=".*figure layout has changed to tight.*")


# ## Stratified K-Fold Cross-Validation Setup

# Define data path and dataset
DATA_PATH = '/kaggle/input/ariel-data-challenge-2025'
DATASET = 'train'

# Load the planet_ids from star info CSV, converting to int index
planet_ids = pd.read_csv(f'{DATA_PATH}/{DATASET}_star_info.csv', index_col='planet_id').index.astype(int)

# Create DataFrame of planet_ids for fold assignment
df = pd.DataFrame({'planet_id': planet_ids})

# Initialize KFold with 5 splits, shuffling enabled for randomness and reproducibility
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Assign fold numbers to planet_ids by iterating over each fold's validation indices
folds = []
for fold_index, (_, val_idx) in enumerate(kf.split(planet_ids)):
    for i in val_idx:
        folds.append({'planet_id': planet_ids[i], 'fold': fold_index})

folds_df = pd.DataFrame(folds)

# Save fold assignments to CSV for easy reuse
folds_df.to_csv('planet_kfolds.csv', index=False)


# ## PCA Visualization of Fold Assignments

# Load star_info features for all planets (including planets in our folds)
star_info = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train_star_info.csv')

# Load fold assignments created earlier
folds = pd.read_csv('/kaggle/working/planet_kfolds.csv')

# Merge star_info and folds on planet_id to align features with fold labels
df = pd.merge(star_info, folds, on='planet_id')

# Extract feature matrix and fold labels
features = df.drop(columns=['planet_id', 'fold']).values
fold_labels = df['fold'].values
planet_ids = df['planet_id'].values


# Reduce dimensionality to 2 principal components for visualization
pca = PCA(n_components=2)
components = pca.fit_transform(features)


# Plot each fold with distinct color to visualize distribution
plt.figure(figsize=(8, 6))
colors = plt.get_cmap('Set1')
for fold in range(5):
    idx = (fold_labels == fold)
    plt.scatter(
        components[idx, 0], components[idx, 1],
        color=colors(fold), label=f'Fold {fold}', alpha=0.8, edgecolor='none', s=40
    )
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.legend(title='Fold')
plt.title('PCA of Star Info Colored by Fold Assignment')
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Exploratory Data Analysis of Star Features

# Load starinfo dataset containing relevant astrophysical parameters
starinfo = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train_star_info.csv')

# Selected features for analysis
features = ['Rs', 'Ms', 'Ts', 'Mp', 'P', 'sma', 'i']

# Standardize features to zero mean and unit variance for clustering and PCA
scaler = StandardScaler()
scaled_features = scaler.fit_transform(starinfo[features])


# Pairplot shows feature distributions and pairwise relationships with KDE diagonals
sns.pairplot(starinfo[features], diag_kind='kde', plot_kws={'alpha':0.5})
plt.suptitle("Pairwise Distributions of Starinfo Parameters", y=1.02)
plt.show()


# Apply PCA and print explained variance ratio for dimensionality reduction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")


# Use KMeans to find clusters in the data, selecting 3 clusters based on domain knowledge/elbow method
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(pca_result)
starinfo['cluster'] = cluster_labels


# Visualize clusters in 2D PCA space
plt.figure(figsize=(8,6))
for cluster in range(n_clusters):
    plt.scatter(
        pca_result[cluster_labels == cluster, 0],
        pca_result[cluster_labels == cluster, 1],
        label=f'Cluster {cluster}', alpha=0.5
    )
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of starinfo features with KMeans Clusters')
plt.legend()
plt.grid(True)
plt.show()


# Silhouette score measures cluster separation quality; higher is better
from sklearn.metrics import silhouette_score
score = silhouette_score(pca_result, cluster_labels)
print(f"Silhouette score: {score:.3f}")


# ## Alternative Clustering Methods Exploration
from sklearn.cluster import AgglomerativeClustering

# Agglomerative clustering with 5 clusters as another method
agg_model = AgglomerativeClustering(n_clusters=5)
agg_labels = agg_model.fit_predict(scaled_features)

plt.figure(figsize=(8,6))
for lbl in np.unique(agg_labels):
    plt.scatter(
        pca_result[agg_labels == lbl, 0],
        pca_result[agg_labels == lbl, 1],
        label=f'Cluster {lbl}', alpha=0.6,
    )
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Agglomerative Clustering Visualization')
plt.legend()
plt.grid(True)
plt.show()



# ## 3D PCA Cluster Visualization

from mpl_toolkits.mplot3d import Axes3D  # Required for 3D projection

# Perform PCA with 3 components for 3D plotting
pca_3d = PCA(n_components=3)
pca_3d_result = pca_3d.fit_transform(scaled_features)
print(f"PCA Explained Variance Ratio (3D): {pca_3d.explained_variance_ratio_}")

# Visualize clusters in 3D PCA space
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
for cluster in range(n_clusters):
    mask = cluster_labels == cluster
    ax.scatter(
        pca_3d_result[mask, 0],
        pca_3d_result[mask, 1],
        pca_3d_result[mask, 2],
        label=f"Cluster {cluster}", alpha=0.5, s=35
    )
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
ax.set_title('3D PCA of starinfo features with KMeans Clusters')
ax.legend()
plt.tight_layout()
plt.show()


# ## Analysis of Prediction Errors Relative to Star Features

# Load model predictions and ground truth spectra
predictions = np.load('/kaggle/input/predictions-0/predictions.npy')
train = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train.csv')

# Extract spectral columns (wavelength data)
spectrum_col_names = train.columns.drop('planet_id')

# Construct ground truth spectra matrix ordered by planet_ids
ground_truth = np.vstack([
    train.loc[train['planet_id'] == pid, spectrum_col_names].values.flatten().astype(float)
    for pid in planet_ids
])

# Calculate mean squared error per planet
mse_per_planet = np.mean((predictions - ground_truth) ** 2, axis=1)



# Load starinfo and select features in correct planet_id order
features = ['Rs', 'Ms', 'Ts', 'Mp', 'P', 'sma', 'i']
starinfo = pd.read_csv('/kaggle/input/ariel-data-challenge-2025/train_star_info.csv')
if 'planet_id' in starinfo.columns:
    starinfo = starinfo.set_index('planet_id')
star_features = starinfo.loc[planet_ids, features].values


# Plot scatter of each feature against prediction error (MSE)
for i, feat in enumerate(features):
    plt.figure()
    plt.scatter(starinfo[feat], mse_per_planet, alpha=0.6)
    plt.xlabel(feat)
    plt.ylabel("Per-planet MSE")
    plt.title(f"Prediction Error vs {feat}")
    plt.ylim(top=0.0002)
    plt.grid(True)
    plt.show()


# Binned boxplots of error by quartiles of each feature
df = pd.DataFrame({f:starinfo[f] for f in features})
df['mse_error'] = mse_per_planet

for feat in features:
    df['bin'] = pd.qcut(df[feat], 4, labels=False)
    plt.figure()
    sns.boxplot(x=df['bin'], y=df['mse_error'])
    plt.xlabel(f"{feat} (binned)")
    plt.ylabel("Per-planet MSE")
    plt.title(f"Error by {feat} Quartile")
    plt.show()


# Print correlation coefficients between features and error
for i, feat in enumerate(features):
    corr = np.corrcoef(starinfo[feat], mse_per_planet)[0,1]
    print(f"Correlation of {feat} with error: {corr:.3f}")



# Additional visualization: Planet mass vs Stellar mass
plt.figure(figsize=(6, 6))
plt.scatter(starinfo['Ms'], starinfo['Mp'], alpha=0.6, edgecolor='k')
plt.xlabel('Stellar Mass (Ms)')
plt.ylabel('Planet Mass (Mp)')
plt.title('Planet Mass vs. Stellar Mass')
plt.grid(True)
plt.show()

