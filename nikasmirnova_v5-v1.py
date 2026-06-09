from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np
import pandas as pd
from torch import nn


class EmbNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("tiny_vit_5m_224.dist_in22k_ft_in1k", pretrained=True, num_classes=0)

    def forward(self, image):
        x = self.model(image)
        return x


def generate_submit(pred_cluster):
    import hashlib
    sub = pd.DataFrame()
    sub['id'] = np.arange(len(pred_cluster))
    sub['target'] = pred_cluster
    hsh = hashlib.sha256(sub.to_csv(index=False).encode('utf-8')).hexdigest()[:8]
    submit_path = f"submit_{hsh}.csv"
    print(f"SUBMIT_NAME: {submit_path}")
    print(sub.head(10))
    sub.to_csv(submit_path, index = None)


X_1 = np.load('/kaggle/input/neoai-2025-cluster-pictures/data_1.npz')
X_1 = X_1.f.arr_0
X_2 = np.load('/kaggle/input/neoai-2025-cluster-pictures/data_2.npz')
X_2 = X_2.f.arr_0

km = KMeans(32)
X = np.concatenate((X_1.reshape((X_1.shape[0], X_1.shape[1] * X_1.shape[2])), X_2.reshape((X_2.shape[0], X_2.shape[1] * X_2.shape[2]))), 1)
pred_cluster = km.fit_predict(X)

generate_submit(pred_cluster)


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import hashlib
import timm
import torch.nn as nn

class EmbNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("tiny_vit_5m_224.dist_in22k_ft_in1k", pretrained=True, num_classes=0)

    def forward(self, image):
        x = self.model(image)
        return x

def generate_submit(pred_cluster):
    sub = pd.DataFrame()
    sub['id'] = np.arange(len(pred_cluster))
    sub['target'] = pred_cluster
    hsh = hashlib.sha256(sub.to_csv(index=False).encode('utf-8')).hexdigest()[:8]
    submit_path = f"submit_{hsh}.csv"
    print(f"SUBMIT_NAME: {submit_path}")
    print(sub.head(10))
    sub.to_csv(submit_path, index=None)

X_1 = np.load('/kaggle/input/neoai-2025-cluster-pictures/data_1.npz')
X_1 = X_1.f.arr_0
X_2 = np.load('/kaggle/input/neoai-2025-cluster-pictures/data_2.npz')
X_2 = X_2.f.arr_0

# Average augmentations
X_1_mean = np.mean(X_1, axis=1)
X_2_mean = np.mean(X_2, axis=1)

# Concatenation and normalization
X = np.concatenate((X_1_mean, X_2_mean), axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans Clustering with additional parameters
km = KMeans(n_clusters=32, random_state=42, n_init=100, max_iter=500, init='k-means++')
pred_cluster_km = km.fit_predict(X_scaled)
score_km = silhouette_score(X_scaled, pred_cluster_km)
print(f"KMeans Silhouette Score: {score_km}")
generate_submit(pred_cluster_km)



!pip install hdbscan


from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from hdbscan import HDBSCAN

hdbscan = HDBSCAN(min_cluster_size=5, metric='euclidean')
pred_cluster_hdbscan = hdbscan.fit_predict(X_scaled)
# Handle the case where all labels are -1 (noise)
if len(set(pred_cluster_hdbscan)) > 1:
    score_hdbscan = silhouette_score(X_scaled, pred_cluster_hdbscan)
    print(f"HDBSCAN Silhouette Score: {score_hdbscan}")
else:
    print("HDBSCAN resulted in all noise points.")

bisecting_kmeans = KMeans(n_clusters=32, random_state=42)
pred_cluster_bisecting = bisecting_kmeans.fit_predict(X_scaled)
score_bisecting = silhouette_score(X_scaled, pred_cluster_bisecting)

print(f"Bisecting KMeans Silhouette Score: {score_bisecting}")



generate_submit(pred_cluster_hdbscan)
generate_submit(pred_cluster_bisecting)


# Spectral Clustering with Varying Metrics
spectral_rbf = SpectralClustering(n_clusters=32, affinity='rbf', random_state=42)
pred_cluster_spectral_rbf = spectral_rbf.fit_predict(X_scaled)
score_spectral_rbf = silhouette_score(X_scaled, pred_cluster_spectral_rbf)

print(f"Spectral Clustering (RBF) Silhouette Score: {score_spectral_rbf}")

spectral_knn = SpectralClustering(n_clusters=32, affinity='nearest_neighbors', random_state=42)
pred_cluster_spectral_knn = spectral_knn.fit_predict(X_scaled)
score_spectral_knn = silhouette_score(X_scaled, pred_cluster_spectral_knn)

print(f"Spectral Clustering (KNN) Silhouette Score: {score_spectral_knn}")

generate_submit(pred_cluster_spectral_rbf)
generate_submit(pred_cluster_spectral_knn)


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
import numpy as np

connectivity = kneighbors_graph(X_scaled, n_neighbors=X_scaled.shape[0], mode='connectivity', include_self=True)

agglomerative = AgglomerativeClustering(n_clusters=32, connectivity=connectivity)
pred_cluster_agglomerative = agglomerative.fit_predict(X_scaled)
score_agglomerative = silhouette_score(X_scaled, pred_cluster_agglomerative)

print(f"Agglomerative Clustering Silhouette Score: {score_agglomerative}")

# Generate submissions
def generate_submit(predictions):
    print("Generated submission for predictions.")

generate_submit(pred_cluster_agglomerative)


spectral = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', random_state=42)
pred_cluster_spectral = spectral.fit_predict(X_scaled)

score_spectral = silhouette_score(X_scaled, pred_cluster_spectral)

print(f"Spectral Clustering Silhouette Score: {score_spectral}")


generate_submit(pred_cluster_spectral)


