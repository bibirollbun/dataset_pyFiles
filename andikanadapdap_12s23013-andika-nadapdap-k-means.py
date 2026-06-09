import numpy as np                  # linear algebra
import pandas as pd                 # data processing, CSV I/O
import os

# Lihat semua file di /kaggle/input
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Ganti path ini jika nama folder di atas berbeda
DATA_PATH = '/kaggle/input/penguin-clustering-analysis/penguins.csv'

df = pd.read_csv(DATA_PATH)
print(df.shape)
df.head()


# Info dan statistik awal
df.info()
df.describe(include='all')


# Simpan kolom species (kalau ada) hanya untuk interpretasi, bukan fitur clustering
label_col = None
if 'species' in df.columns:
    label_col = 'species'
    species = df['species']
    df_features = df.drop(columns=['species'])
else:
    df_features = df.copy()

# Pisahkan kolom numerik & kategorik
numeric_cols = df_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df_features.select_dtypes(include=['object', 'category']).columns.tolist()

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# Untuk tugas ini kita fokus ke fitur morfologi numerik saja
X_num = df_features[numeric_cols]

# Tangani missing value secara sederhana: impute dengan mean
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_num)

# Standardisasi fitur
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print("Shape after preprocessing:", X_scaled.shape)


import matplotlib.pyplot as plt
import seaborn as sns

# Histogram fitur numerik
X_num.hist(bins=20, figsize=(12, 8))
plt.tight_layout()
plt.show()


# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(X_num.corr(), annot=True, fmt=".2f", cmap="viridis")
plt.title("Correlation heatmap of numeric features")
plt.show()


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sse = []
silhouette_scores = []
K_RANGE = range(2, 10)   # mulai dari 2 cluster

for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)                     # SSE / inertia
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot Elbow
plt.figure(figsize=(6, 4))
plt.plot(list(K_RANGE), sse, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("SSE (inertia)")
plt.title("Elbow Method for K-Means")
plt.show()

# Plot Silhouette
plt.figure(figsize=(6, 4))
plt.plot(list(K_RANGE), silhouette_scores, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette score vs k")
plt.show()

for k, s in zip(K_RANGE, silhouette_scores):
    print(f"k={k}: silhouette={s:.3f}")


best_k = 3  # ubah jika kamu memilih k lain

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

df['cluster'] = cluster_labels
df.head()


# Ringkasan ukuran tiap cluster
cluster_counts = df['cluster'].value_counts().sort_index()
print("Jumlah anggota tiap cluster:")
print(cluster_counts)

# Rata-rata fitur numerik per cluster
cluster_means = df.groupby('cluster')[numeric_cols].mean()
cluster_means


if label_col is not None:
    ctab = pd.crosstab(df['cluster'], species)
    print("Crosstab cluster vs species:")
    print(ctab)


from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=df['cluster'],
    palette='tab10'
)
plt.title("Clusters visualized in 2D PCA space")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title='Cluster')
plt.show()

