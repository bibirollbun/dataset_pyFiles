import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import cudf
from cuml.cluster import HDBSCAN
import umap
from cuml.cluster import KMeans
import cupy as cp
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from cuml.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture


df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet',
                    engine = 'pyarrow')
df.head()


df.shape


df = df.loc[:, ~df.isin([np.inf, -np.inf]).any()] # any() checks atleast one element
df.shape


X = df.drop(columns = ['label'])
y = df['label']


df_sample = X.sample(n = 30000, random_state = 42)
features = df_sample.values
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
kmeans = KMeans(n_clusters = 5, random_state = 42)
cluster_labels = kmeans.fit_predict(features)
df_sample['cluster'] = cluster_labels


umap_model = umap.UMAP(n_components = 2)
X_umap = umap_model.fit_transform(features)


plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=df_sample['cluster'],
    cmap='Dark2',        
    alpha=0.7,
    s=10,
    vmin=0,
    vmax=4              
)
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('K-Means Clusters Visualized with UMAP (sampled 30k points)')
plt.colorbar(scatter, label='Cluster', ticks=range(5))
plt.show()


gdf = cudf.DataFrame.from_pandas(pd.DataFrame(features))
clusterer = HDBSCAN(min_cluster_size = 7)
hdb_labels = clusterer.fit_predict(gdf)
df_sample['hdbscan_cluster'] = hdb_labels.to_numpy()
#get() method transfers the data from GPU memory to host memory as a NumPy array.


plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    X_umap[:, 0], X_umap[:, 1],
    c=df_sample['hdbscan_cluster'],  
    cmap='Dark2', s=10, alpha=0.7
)
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('K-Means Clusters on UMAP-Reduced Data')
plt.colorbar(scatter, label='Cluster')
plt.show()


agglo = AgglomerativeClustering(n_clusters = 40)
agglo_labels = agglo.fit_predict(gdf)
df_sample['agglo_cluster'] = agglo_labels.to_numpy()


plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    X_umap[:, 0], X_umap[:, 1],
    c=df_sample['agglo_cluster'],  
    cmap='Dark2', s=10, alpha=0.7
)
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('K-Means Clusters on UMAP-Reduced Data')
plt.colorbar(scatter, label='Cluster')
plt.show()


first_10 = df.iloc[:, :10]

# Compute min, max, median for each column
summary = pd.DataFrame({
    'min': first_10.min(),
    'max': first_10.max(),
    'median': first_10.median()
})

print(summary)


negative_cols = [col for col in X.columns if (X[col] < 0).any()]
print(negative_cols)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


del df


import joblib 

scalers = {}  # Dictionary to store the scaler for each column
X_train_scaled = X_train.copy()

for col in X_train.columns:
    if (X_train[col] < 0).any():
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled[[col]] = scaler.fit_transform(X_train[[col]])
    scalers[col] = scaler

joblib.dump(scalers, 'column_scalers.pkl')


first_10 =X_train_scaled.iloc[:, :10]

# Compute min, max, median for each column
summary = pd.DataFrame({
    'min': first_10.min(),
    'max': first_10.max(),
    'median': first_10.median()
})

print(summary)


X_test_scaled = X_test.copy()
for col in X_test.columns:
    X_test_scaled[[col]] = scalers[col].transform(X_test[[col]])


first_10 =X_test_scaled.iloc[:, :10]

# Compute min, max, median for each column
summary = pd.DataFrame({
    'min': first_10.min(),
    'max': first_10.max(),
    'median': first_10.median()
})

print(summary)


X_train_scaled_cudf = cudf.DataFrame.from_pandas(X_train_scaled)

kmeans_gpu = KMeans(n_clusters=5, random_state=42)
k_labels = kmeans_gpu.fit_predict(X_train_scaled_cudf)

X_train_scaled["k_labels"] = k_labels.to_numpy()

# Save the trained GPU KMeans model
joblib.dump(kmeans_gpu, "kmeans_model_gpu.pkl")


X_test_scaled_cudf = cudf.DataFrame.from_pandas(X_test_scaled)


k_labels_test = kmeans_gpu.predict(X_test_scaled_cudf)

X_test_scaled["k_labels"] = k_labels_test.to_numpy()


X_train_scaled.to_parquet("X_train_scaled.parquet", index=False)
y_train.to_frame().to_parquet("y_train.parquet", index=False)

X_test_scaled.to_parquet("X_test_scaled.parquet", index=False)
y_test.to_frame().to_parquet("y_test.parquet", index=False)

