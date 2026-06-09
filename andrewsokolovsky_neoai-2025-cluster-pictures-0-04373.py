!pip install umap-learn


import numpy as np
import pandas as pd
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import timm
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline


np.random.seed(42)
torch.manual_seed(42)


class EmbNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("tiny_vit_5m_224.dist_in22k_ft_in1k", pretrained=True, num_classes=0)

    def forward(self, image):
        x = self.model(image)
        return x


print("Loading data...")
X_1 = np.load('/kaggle/input/neoai-2025-cluster-pictures/data_1.npz')['arr_0']
X_2 = np.load('/kaggle/input/neoai-2025-cluster-pictures/data_2.npz')['arr_0']

print(f"Size X_1: {X_1.shape}")
print(f"Size X_2: {X_2.shape}")
assert X_1.shape[0] == X_2.shape[0], "The number of rows in X_1 and X_2 must match"


def generate_submit(pred_cluster):
    sub = pd.DataFrame()
    sub['id'] = np.arange(len(pred_cluster))
    sub['target'] = pred_cluster
    hsh = hashlib.sha256(sub.to_csv(index=False).encode('utf-8')).hexdigest()[:8]
    submit_path = f"submit_{hsh}.csv"
    print(f"File name: {submit_path}")
    print("First 10 lines of the result:")
    print(sub.head(10))
    sub.to_csv(submit_path, index=None)
    return submit_path

def prepare_pseudo_images(X_1, X_2):
    n_samples = X_1.shape[0]
    
    X_1_flat = X_1.reshape(n_samples, -1) 
    X_2_flat = X_2.reshape(n_samples, -1) 
    X = np.concatenate([X_1_flat, X_2_flat], axis=1) 
    
    target_size = 3 * 224 * 224  
    current_size = X.shape[1]
    
    X_interpolated = np.zeros((n_samples, target_size))
    for i in range(n_samples):
        x = np.linspace(0, 1, current_size)
        y = X[i]
        f = interp1d(x, y, kind='linear')
        x_new = np.linspace(0, 1, target_size)
        X_interpolated[i] = f(x_new)
    
    X_images = X_interpolated.reshape(n_samples, 3, 224, 224)
    
    images = torch.tensor(X_images, dtype=torch.float32)
    
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images = images.permute(0, 2, 3, 1) 
    images = transform(images.permute(0, 3, 1, 2))  
    return images

def balance_clusters(X, clusters, min_size=50):
    cluster_counts = pd.Series(clusters).value_counts()
    small_clusters = cluster_counts[cluster_counts < min_size].index
    
    if len(small_clusters) == 0:
        return clusters
    
    nn = NearestNeighbors(n_neighbors=1)
    valid_idx = np.isin(clusters, small_clusters, invert=True)
    nn.fit(X[valid_idx])
    
    for cl in small_clusters:
        small_idx = clusters == cl
        if small_idx.sum() == 0:
            continue
        distances, indices = nn.kneighbors(X[small_idx])
        new_labels = clusters[valid_idx][indices.flatten()]
        clusters[small_idx] = new_labels
    
    return clusters


print("Preparing pseudo-images for EmbNet...")
images = prepare_pseudo_images(X_1, X_2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmbNet().to(device).eval()

dataset = TensorDataset(images)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

embeddings = []
with torch.no_grad():
    for batch in dataloader:
        batch_emb = model(batch[0].to(device))
        embeddings.append(batch_emb.cpu().numpy())

embeddings = np.concatenate(embeddings, axis=0)
print(f"Embedding dimensions from EmbNet: {embeddings.shape}")


scaler = RobustScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

reducer = umap.UMAP(n_components=32, n_neighbors=100, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(embeddings_scaled)
print(f"Size after UMAP: {X_umap.shape}")

umap_2d = umap.UMAP(n_components=2, n_neighbors=100, min_dist=0.1, random_state=42)
X_2d = umap_2d.fit_transform(embeddings_scaled)

kmeans_temp = KMeans(n_clusters=32, n_init=50, random_state=42)
temp_labels = kmeans_temp.fit_predict(X_umap)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=temp_labels, palette='tab20', s=50)
plt.title("2D visualization of embeddings after UMAP with clusters")
plt.savefig("umap_visualization_clusters.png")
plt.close()


kf = KFold(n_splits=3, shuffle=True, random_state=42)
sil_scores = []
db_scores = []
ch_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_umap)):
    print(f"Fold processing {fold + 1}...")
    X_train, X_val = X_umap[train_idx], X_umap[val_idx]
    
    kmeans = KMeans(n_clusters=32, n_init=50, random_state=42)
    train_labels = kmeans.fit_predict(X_train)
    
    if len(np.unique(train_labels)) > 1:
        sil = silhouette_score(X_train, train_labels)
        db = davies_bouldin_score(X_train, train_labels)
        ch = calinski_harabasz_score(X_train, train_labels)
        sil_scores.append(sil)
        db_scores.append(db)
        ch_scores.append(ch)
        print(f"FOLD {fold + 1}: Silhouette={sil:.4f}, Davies-Bouldin={db:.4f}, Calinski-Harabasz={ch:.4f}")
    else:
        print(f"FOLD {fold + 1}: Not enough clusters to evaluate")

print(f"Mean Silhouette Score: {np.mean(sil_scores):.4f} Â± {np.std(sil_scores):.4f}")
print(f"Mean Davies-Bouldin Score: {np.mean(db_scores):.4f} Â± {np.std(db_scores):.4f}")
print(f"Mean Calinski-Harabasz Score: {np.mean(ch_scores):.4f} Â± {np.std(ch_scores):.4f}")



print("Clustering on all data...")
kmeans = KMeans(n_clusters=32, n_init=100, max_iter=1000, random_state=42)
clusters = kmeans.fit_predict(X_umap)

clusters = balance_clusters(X_umap, clusters, min_size=50)

sil_score = silhouette_score(X_umap, clusters) if len(np.unique(clusters)) > 1 else -1
db_score = davies_bouldin_score(X_umap, clusters) if len(np.unique(clusters)) > 1 else -1
ch_score = calinski_harabasz_score(X_umap, clusters) if len(np.unique(clusters)) > 1 else -1
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Score: {db_score:.4f}")
print(f"Calinski-Harabasz Score: {ch_score:.4f}")

cluster_counts = pd.Series(clusters).value_counts()
print("Distribution of cluster sizes:")
print(cluster_counts)

plt.figure(figsize=(10, 6))
plt.hist(clusters, bins=32, edgecolor='black')
plt.title("Distribution of cluster sizes")
plt.xlabel("cluster")
plt.ylabel("Number of samples")
plt.savefig("cluster_size_histogram.png")
plt.close()


print("generation submit...")
submit_path = generate_submit(clusters)

