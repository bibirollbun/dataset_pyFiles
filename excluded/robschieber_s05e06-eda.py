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


import os
import tensorflow as tf

device_name = tf.test.gpu_device_name()

is_gpu = False

if "GPU" not in device_name:
    print("GPU device not found")
    is_gpu = True

try:
    import cudf
except Exception:
    print('error loading cuda cudf extensions, using gpu is reccomended for this nb')

try:
    import cuml
    from cuml.preprocessing import StandardScaler
    
except Exception:
    print('error loading cuda StandardScalar, using sklearn')
    from sklearn.preprocessing import StandardScaler


import datetime
import pandas as pd
import seaborn as sns
import kagglehub
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning, message="'force_all_finite' was renamed to 'ensure_all_finite'")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
np.random.seed(42)

playground_series_s5e6_path = kagglehub.competition_download('playground-series-s5e6')
original_dataset_path = kagglehub.dataset_download("gdabhishek/fertilizer-prediction")

original_data_path = os.path.join(original_dataset_path, "Fertilizer Prediction.csv")
train_data_path = os.path.join(playground_series_s5e6_path, "train.csv")
test_data_path = os.path.join(playground_series_s5e6_path, "test.csv")



scaler = StandardScaler()
le = LabelEncoder()


CROP_CONFIG = {
    'Sugarcane': 1,
    'Millets': 2,
    'Barley': 3,
    'Paddy': 4,
    'Pulses': 5, 
    'Tobacco': 6,
    'Ground Nuts': 7, 
    'Maize': 8, 
    'Cotton': 9, 
    'Wheat': 10,
    'Oil seeds': 11
}

SOIL_CONFIG = {
    'Clayey': 25.0,
    'Sandy': 5.0,
    'Red': 10.0,
    'Loamy': 15.0,
    'Black': 20.0
}

def clean_data(data):
    data = data.rename({"Temparature": "Temperature",
                      "Soil Type": "Soil_Type",
                      "Crop Type": "Crop_Type",
                      "Humidity ": "Humidity",
                      "Fertilizer Name": "Fertilizer_Name" }, axis=1)
    return data
    
def get_data(path, rowcount=-1, crop_config=CROP_CONFIG, soil_config=SOIL_CONFIG):

  data = pd.read_csv(path)
  data = clean_data(data)
  is_train = ('Fertilizer_Name' in data.columns)

  if is_train:
    data['label'] = le.fit_transform(data['Fertilizer_Name'])

  if rowcount >0:
    print(f'resampling to {rowcount} rows')
    data = resample(data, n_samples=rowcount, random_state=42)

  data['Soil_Type'] = data['Soil_Type'].map(soil_config)
  data['Crop_Type'] = data['Crop_Type'].map(crop_config)

  return data


raw_train_data = pd.read_csv(train_data_path)
train_data = get_data(train_data_path)
original_data = get_data(original_data_path)
train_data = pd.concat([train_data.reset_index(drop=True), original_data.reset_index(drop=True) ])
test_data = get_data(test_data_path)

X = train_data.drop(['id', 'Fertilizer_Name', 'label'], axis=1)
X_test = test_data.drop(['id'], axis=1)
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
train_data.head()


import cuml
from cuml import KMeans
from cuml.decomposition import PCA


def get_kmeans(X_scaled):
  pca = PCA(n_components = 2)
  pca_projection = pca.fit_transform( X_scaled)

  pca_df = pd.DataFrame(pca_projection)
  pca_df.columns = ['PCA_0', 'PCA_1']

  #X_scaled = cudf.DataFrame(X_scaled)
  kmeans = KMeans(n_clusters=7)
  #kmeans_float.fit(X_scaled)
  k_means_projection = kmeans.fit_transform(pca_projection, y=None, convert_dtype=False, sample_weight=None)
  pca_df['KMEANS_Label'] = kmeans.labels_

  return pca_df

pca_df_train = get_kmeans(X_scaled)
pca_df_test = get_kmeans(X_test_scaled)

df_train = pd.concat([train_data.reset_index(drop=True), pca_df_train], axis=1)
df_test = pd.concat([test_data.reset_index(drop=True), pca_df_test], axis=1)

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA_0', y='PCA_1', hue='KMEANS_Label',  data=df_train, palette='tab10', alpha=0.9)
plt.title(f'PCA Visualization - Train Dataset')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA_0', y='PCA_1', hue='KMEANS_Label',  data=df_test, palette='tab10', alpha=0.9)
plt.title(f'PCA Visualization - Test Dataset')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py

pca_df = pca_df_test
#X_scaled = cudf.DataFrame(X_scaled)
pca_projection = pca_df[['PCA_0','PCA_1']].to_numpy()
kmeans = KMeans(n_clusters=7)
#kmeans_float.fit(X_scaled)
k_means_projection = kmeans.fit_transform(pca_projection, y=None, convert_dtype=False, sample_weight=None)


x_min, x_max = pca_projection[:, 0].min() - 1, pca_projection[:, 0].max() + 1
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
y_min, y_max = pca_projection[:, 1].min() - 1, pca_projection[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])


# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(pca_projection[:, 0], pca_projection[:, 1], "k.", markersize=2)

# Plot the centroids as a white X
centroids = np.array(kmeans.cluster_centers_)
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.title(
    "K-means clustering on the fertilizer dataset (PCA-reduced data)\n"
    "Centroids are marked with white cross"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


import cuml

# train clustering
reducer = cuml.UMAP(n_components=2, random_state=42,
               n_neighbors=30, min_dist=0.1)
train_embedding = reducer.fit_transform(X_scaled).to_numpy()

# apply hdb scan clustering
train_clusterer = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=100, prediction_data=True)
train_clusterer.fit(train_embedding)
train_cluster_labels = train_clusterer.labels_
train_soft_clusters = cuml.cluster.hdbscan.all_points_membership_vectors(train_clusterer)
train_num_clusters = max(train_cluster_labels)
train_pred_labels, train_pred_probs = cuml.cluster.hdbscan.approximate_predict(train_clusterer, train_embedding)

train_umap_df = pd.DataFrame(train_embedding, columns=['UMAP1', 'UMAP2'])
train_umap_df['Pred_Label'] = train_pred_labels
train_umap_df['Pred_Prob'] = train_pred_probs
train_umap_df['Cluster_Label'] = train_cluster_labels

for idx, col in enumerate([f"Prob_Cluster_{i}" for i in range(train_num_clusters)]):
    train_umap_df[col] = np.round(train_soft_clusters[:,idx],3)

train_umap_df['Is_Noise'] = (train_umap_df['Pred_Label'] == -1).astype(int)

#test clustering
test_embedding = reducer.transform(X_test_scaled).to_numpy()
test_clusterer = cuml.cluster.hdbscan.HDBSCAN(min_cluster_size=100, prediction_data=True)
test_clusterer.fit(test_embedding)
test_cluster_labels = test_clusterer.labels_
test_soft_clusters = cuml.cluster.hdbscan.all_points_membership_vectors(test_clusterer)
test_num_clusters = max(test_cluster_labels)
test_pred_labels, test_pred_probs = cuml.cluster.hdbscan.approximate_predict(test_clusterer, test_embedding)

test_umap_df = pd.DataFrame(test_embedding, columns=['UMAP1', 'UMAP2'])
test_umap_df['Pred_Label'] = test_pred_labels
test_umap_df['Pred_Prob'] = test_pred_probs
test_umap_df['Cluster_Label'] = test_cluster_labels

for idx, col in enumerate([f"Prob_Cluster_{i}" for i in range(train_num_clusters)]):
    test_umap_df[col] = np.round(test_soft_clusters[:,idx],3)

test_umap_df['Is_Noise'] = (test_umap_df['Pred_Label'] == -1).astype(int)




plt.figure(figsize=(10, 8))
sns.scatterplot(x='UMAP1', y='UMAP2', hue='Cluster_Label',  
                data=train_umap_df, palette='tab10', alpha=0.9)
plt.title(f'UMAP Visualization - Train Data ({len(train_umap_df)} samples)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
sns.scatterplot(x='UMAP1', y='UMAP2', hue='Cluster_Label',  
                data=test_umap_df, palette='tab10', alpha=0.9)
plt.title(f'UMAP Visualization - Test Data ({len(test_umap_df)} samples)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# T-SNE+K-Means

tsne_subset_size = 20000  # Subset size for t-SNE computation
perp = 30  # Perplexity for t-SNE

subset_indices = np.random.choice(X_scaled.shape[0], tsne_subset_size, replace=False)

X_subset = X_scaled.iloc[subset_indices]

# Run t-SNE
tsne = cuml.TSNE(n_components=2, random_state=42, perplexity=perp, n_iter=500)
projection = tsne.fit_transform(X_subset).to_numpy()

# Run K-means to identify 7 Fertilizers
kmeans = KMeans(n_clusters=7, random_state=42, n_init='auto')
tsne_clusters = kmeans.fit_predict(projection)

# Build tsne_df
tsne_df = pd.DataFrame(projection, columns=['TSNE1', 'TSNE2'])
tsne_df['Cluster'] = tsne_clusters
tsne_df['Crop_Type'] = train_data['Crop_Type'].iloc[subset_indices].values
tsne_df['Soil_Type'] = train_data['Soil_Type'].iloc[subset_indices].values
tsne_df['Fertilizer_Name'] = train_data['Fertilizer_Name'].iloc[subset_indices].values

plt.figure(figsize=(10, 8))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Cluster', data=tsne_df, palette='tab10', alpha=0.6)
plt.title(f't-SNE Visualization (Perplexity = {perp})')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Fertilizer_Name', data=tsne_df, palette='tab10', alpha=0.6)
plt.title(f't-SNE Visualization - Fertilizer Name (Perplexity = {perp})')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Visualize t-SNE with Soil Type coloring
plt.figure(figsize=(10, 8))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Crop_Type', data=tsne_df, palette='tab10', alpha=0.6)
plt.title(f't-SNE Visualization - Crop Type (Perplexity = {perp})')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Visualize t-SNE with Soil Type coloring
plt.figure(figsize=(10, 8))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Soil_Type', data=tsne_df, palette='tab10', alpha=0.6)
plt.title(f't-SNE Visualization - Soil Type (Perplexity = {perp})')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Run HDBSCAN on the t-SNE projection
clusterer = cuml.HDBSCAN(min_cluster_size=10, prediction_data=True).fit(projection)
hdbscan_clusters = clusterer.labels_  # Cluster labels (-1 for noise points)
soft_clusters = cuml.cluster.hdbscan.all_points_membership_vectors(clusterer)  # Membership probabilities

# Update tsne_df with HDBSCAN cluster labels
tsne_df['Cluster'] = hdbscan_clusters

# Analyze HDBSCAN clusters
num_clusters = len(set(hdbscan_clusters)) - (1 if -1 in hdbscan_clusters else 0)
num_noise_points = list(hdbscan_clusters).count(-1)
print(f"\nHDBSCAN Fond {num_clusters} Clusters")

# Map clusters to Crop Type
cluster_to_crop = tsne_df[tsne_df['Cluster'] != -1]\
    .groupby('Cluster')['Crop_Type'].agg(lambda x: x.mode()[0])


# Soil Type distribution within each HDBSCAN cluster
soil_distribution = tsne_df[tsne_df['Cluster'] != -1]\
    .groupby(['Cluster', 'Soil_Type']).size().unstack(fill_value=0)
soil_proportions = soil_distribution.div(soil_distribution.sum(axis=1), axis=0)
#print("\nSoil Type Proportions Within Each HDBSCAN Cluster (Excluding Noise Points):")
#print(soil_proportions)

cluster_to_crop.head()


# Visualize t-SNE with HDBSCAN clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='TSNE1', y='TSNE2', hue='Cluster', style='Cluster', data=tsne_df,
    palette='tab20', alpha=0.6, legend='full'
)
plt.title(f't-SNE Visualization (Perplexity = {perp}) - HDBSCAN Clusters - -1 is Noise ')
plt.legend(ncol=4, bbox_to_anchor=(1.05, 1), loc='upper left', title='Cluster')
#plt.tight_layout()
plt.show()


