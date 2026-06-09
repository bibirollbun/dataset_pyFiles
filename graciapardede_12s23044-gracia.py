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
# You can also w# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')



# ===== 2. BACA DATA =====

# Baca file penguins.csv
if os.path.exists('/kaggle/input'):
    data = pd.read_csv('/kaggle/input/penguin-clustering-analysis/penguins.csv')
else:
    data = pd.read_csv('penguins.csv')

print(f"✓ Data berhasil dibaca!")
print(f"  Total: {data.shape[0]} baris, {data.shape[1]} kolom")
print(f"\nContoh data:")
print(data.head())
print(f"\nKolom yang ada:")
print(data.columns.tolist())



# ===== 3. EDA =====

print("\n=== INFO DATASET ===")
print(data.info())

print("\n=== STATISTIK ===")
print(data.describe())

print("\n=== CEK MISSING VALUES ===")
missing = data.isnull().sum()
print(missing[missing > 0])

# Visualisasi
numerik = data.select_dtypes(include=[np.number]).columns.tolist()
if 'id' in numerik:
    numerik.remove('id')

n_cols = min(len(numerik), 5)
plt.figure(figsize=(15, 4))
for i, col in enumerate(numerik[:n_cols], 1):
    plt.subplot(1, n_cols, i)
    plt.hist(data[col].dropna(), bins=20, edgecolor='black', color='skyblue')
    plt.title(col)
plt.tight_layout()
plt.show()

print("✓ EDA selesai!")



# ===== 4. PREPROCESSING =====

# Simpan ID untuk submission
if 'id' in data.columns:
    ids = data['id'].copy()
    X = data.drop('id', axis=1)
else:
    ids = pd.Series(range(len(data)))
    X = data.copy()

print(f"\n=== PREPROCESSING ===")
print(f"Data awal: {X.shape}")

# Ubah kategorikal jadi angka
categorical_cols = X.select_dtypes(include=['object']).columns
print(f"Kolom kategorikal: {categorical_cols.tolist()}")

for col in categorical_cols:
    X[col] = X[col].astype('category').cat.codes

# Isi missing values
X = X.fillna(X.median())

print(f"✓ Preprocessing selesai!")
print(f"  Data siap cluster: {X.shape[0]} baris, {X.shape[1]} kolom")
print(f"  Fitur: {X.columns.tolist()}")



# ===== 5. NORMALISASI =====

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n✓ Data sudah dinormalisasi!")
print(f"  Mean: {X_scaled.mean():.4f} (mendekati 0 ✓)")
print(f"  Std: {X_scaled.std():.4f} (mendekati 1 ✓)")



# ===== 6. CARI K OPTIMAL =====

K_range = range(2, 11)
inertia = []
silhouette = []
davies_bouldin = []
calinski = []

print("\n=== MENCARI K OPTIMAL ===")
for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(X_scaled, labels))
    davies_bouldin.append(davies_bouldin_score(X_scaled, labels))
    calinski.append(calinski_harabasz_score(X_scaled, labels))
    
    print(f"K={k}: Silhouette={silhouette[-1]:.3f}, DB={davies_bouldin[-1]:.3f}")

print("✓ Selesai!")



# ===== 7. PLOT GRAFIK =====

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(K_range, inertia, 'bo-', linewidth=2, markersize=8)
axes[0,0].set_title('Elbow Method', fontweight='bold')
axes[0,0].set_xlabel('K')
axes[0,0].set_ylabel('Inertia')
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(K_range, silhouette, 'go-', linewidth=2, markersize=8)
axes[0,1].axhline(0.5, color='r', linestyle='--', alpha=0.7, label='Good (>0.5)')
axes[0,1].set_title('Silhouette Score', fontweight='bold')
axes[0,1].set_xlabel('K')
axes[0,1].set_ylabel('Silhouette')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(K_range, davies_bouldin, 'ro-', linewidth=2, markersize=8)
axes[1,0].axhline(1.0, color='r', linestyle='--', alpha=0.7, label='Good (<1.0)')
axes[1,0].set_title('Davies-Bouldin Index', fontweight='bold')
axes[1,0].set_xlabel('K')
axes[1,0].set_ylabel('Davies-Bouldin')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(K_range, calinski, 'mo-', linewidth=2, markersize=8)
axes[1,1].set_title('Calinski-Harabasz Score', fontweight='bold')
axes[1,1].set_xlabel('K')
axes[1,1].set_ylabel('Calinski-Harabasz')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Pilih K optimal
optimal_k = 3  

print(f"\n✓ K optimal dipilih: {optimal_k}")



# ===== 8. TRAINING MODEL =====

print(f"\n=== TRAINING K-MEANS ===")
print(f"Training dengan K={optimal_k} pada SEMUA data...")

model = KMeans(
    n_clusters=optimal_k,
    init='k-means++',
    random_state=42,
    n_init=20,
    max_iter=300
)

model.fit(X_scaled)
labels = model.predict(X_scaled)  # Cluster untuk SEMUA data

print(f"✓ Model sudah dilatih!")
print(f"  Inertia: {model.inertia_:.2f}")
print(f"  Iterasi: {model.n_iter_}")



# ===== 9. EVALUASI MODEL (VERSI SINGKAT) =====

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# hitung metrik evaluasi utama
sse = model.inertia_
sil = silhouette_score(X_scaled, labels)
db  = davies_bouldin_score(X_scaled, labels)
ch  = calinski_harabasz_score(X_scaled, labels)

print("=== Evaluasi Model K-Means ===")
print(f"SSE (Inertia)        : {sse:.2f}")
print(f"Silhouette Score     : {sil:.4f}")
print(f"Davies-Bouldin Index : {db:.4f}")
print(f"Calinski-Harabasz    : {ch:.2f}")

print("\nInterpretasi singkat:")
print("- SSE kecil  → cluster relatif kompak.")
print("- Silhouette mendekati/di atas 0.5 → pemisahan cluster cukup baik.")
print("- Davies-Bouldin mendekati 0 dan < 1 → cluster cukup terpisah.")
print("- Calinski-Harabasz besar → antar-cluster terpisah, intra-cluster rapat.")



# ===== 10. VISUALISASI =====

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(model.cluster_centers_)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', 
                      alpha=0.6, edgecolors='k', s=50)
plt.title(f'Hasil K-Means (K={optimal_k})', fontsize=13, fontweight='bold')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', 
                      alpha=0.5, edgecolors='k', s=50)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='*', 
            s=500, edgecolors='black', linewidths=2, label='Centroids')
plt.title('Clustering + Centroids', fontsize=13, fontweight='bold')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Visualisasi selesai!")


