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


# Klasterisasi Pinguin (K-Means + Evaluasi)
# 
# - Dataset: /kaggle/input/penguin-clustering-analysis/penguins.csv  
# - Fitur untuk klaster: culmen_length, culmen_depth, flipper_length, body_mass  
# - Algoritma utama: K-Means  
# - Penentuan K: Elbow (SSE) + Silhouette 


# %% 
# # Import library yang dibutuhkan untuk klasterisasi dan visualisasi.
import numpy as np # buat operasi matematika & array
import pandas as pd # buat baca & olah data tabel (CSV → DataFrame)
import matplotlib.pyplot as plt # buat bikin grafik
import seaborn as sns  # buat visualisasi data yang lebih rapi

from sklearn.preprocessing import StandardScaler # buat normalisasi data
from sklearn.cluster import KMeans # buat algoritma K-Means
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances
)
from sklearn.decomposition import PCA

sns.set(style="whitegrid", font_scale=1.1)
pd.set_option("display.max_columns", None)



import warnings
warnings.filterwarnings("ignore")


# %% 
# fungsi metrik klaster

def dunn_index(X, labels):
    """Dunn index: min jarak antar klaster / max diameter klaster."""
    u = np.unique(labels)
    if len(u) < 2:
        return np.nan
    D = pairwise_distances(X)
    intra = []
    for k in u:
        idx = np.where(labels == k)[0]
        if len(idx) <= 1:
            intra.append(0.0)
        else:
            intra.append(D[np.ix_(idx, idx)].max())
    max_intra = np.max(intra)
    inter = []
    for i, ki in enumerate(u):
        for kj in u[i+1:]:
            idx_i = np.where(labels == ki)[0]
            idx_j = np.where(labels == kj)[0]
            inter.append(D[np.ix_(idx_i, idx_j)].min())
    min_inter = np.min(inter)
    return min_inter / max_intra


def print_cluster_metrics(X, labels, name="Model"):
    """Cetak metrik internal: silhouette, CH, DB, Dunn."""
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    dunn = dunn_index(X, labels)
    print(f"=== {name} ===")
    print(f"Silhouette       : {sil:.4f}")
    print(f"Calinski-Harabasz: {ch:.2f}")
    print(f"Davies-Bouldin   : {db:.4f}")
    print(f"Dunn index       : {dunn:.4f}\n")



# %% 
# load data kaggle
df = pd.read_csv("/kaggle/input/penguin-clustering-analysis/penguins.csv")
print("shape:", df.shape)
df.head()



import warnings
warnings.filterwarnings("ignore")



# %% 
# info awal & missing value
df.info()
print("\nMissing per kolom:")
print(df.isna().sum())



# %% 
# distribusi fitur numerik
num_cols = ["culmen_length_mm", "culmen_depth_mm",
            "flipper_length_mm", "body_mass_g"]

plt.figure(figsize=(12, 8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()



# %% 
# preprocessing & pemilihan fitur

df_clean = df.copy()
df_clean["sex"] = df_clean["sex"].replace(".", np.nan)  # sex untuk analisis saja

# hapus baris semua numerik NaN
df_clean = df_clean.dropna(subset=num_cols, how="all")

# buang outlier flipper_length_mm (rentang wajar 150–260 mm)
df_clean = df_clean[
    (df_clean["flipper_length_mm"] > 150) &
    (df_clean["flipper_length_mm"] < 260)
]

# imputasi median numerik
for col in num_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

print("shape setelah cleaning:", df_clean.shape)
print("missing numerik:\n", df_clean[num_cols].isna().sum())



# %% 
# standardisasi fitur numerik (wajib sebelum K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[num_cols].values)
pd.DataFrame(X_scaled, columns=num_cols).head()



# %% 
# cari K optimal: Elbow (SSE) + Silhouette

K_range = range(2, 9)
inertias, sil_scores = [], []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)                  # SSE
    sil_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(list(K_range), inertias, marker="o")
plt.xlabel("K"); plt.ylabel("SSE"); plt.title("Elbow Method")

plt.subplot(1, 2, 2)
plt.plot(list(K_range), sil_scores, marker="o")
plt.xlabel("K"); plt.ylabel("Silhouette"); plt.title("Silhouette vs K")

plt.tight_layout()
plt.show()

for k, s, sse in zip(K_range, sil_scores, inertias):
    print(f"K={k}: silhouette={s:.4f}, SSE={sse:.2f}")



# %% 
# model utama: K-Means dengan K=3 (berdasarkan grafik + 3 spesies pinguin)

best_k = 3
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
labels_km = kmeans.fit_predict(X_scaled)
df_clean["cluster_kmeans"] = labels_km

print("jumlah anggota cluster:\n", df_clean["cluster_kmeans"].value_counts(), "\n")
print_cluster_metrics(X_scaled, labels_km, name="K-Means (K=3)")
print("SSE (inertia) K=3:", kmeans.inertia_)



# %% 
# bandingkan metrik untuk K=2,3,4

for k in [2, 3, 4]:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    print(f"Evaluasi K={k}")
    print_cluster_metrics(X_scaled, labels, name=f"K-Means (K={k})")
    print("SSE:", km.inertia_)
    print("-"*40)



# %% 
# profil rata-rata tiap cluster & distribusi sex

cluster_profile = df_clean.groupby("cluster_kmeans")[num_cols].mean()
display(cluster_profile)

print("\nproporsi sex per cluster:")
print(df_clean.groupby("cluster_kmeans")["sex"].value_counts(normalize=True))



# %% 
# PCA 2D untuk visualisasi cluster

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df_clean["pca1"] = X_pca[:, 0]
df_clean["pca2"] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_clean, x="pca1", y="pca2",
                hue="cluster_kmeans", palette="Set1")
plt.title("K-Means (K=3) di ruang PCA")
plt.legend(title="cluster")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_clean, x="pca1", y="pca2", hue="sex")
plt.title("PCA diwarnai berdasarkan sex")
plt.show()



# %% 
# pairplot (opsional, bisa di-skip kalau terlalu lama)

sns.pairplot(df_clean, vars=num_cols, hue="cluster_kmeans", corner=True)
plt.suptitle("Pairplot fitur numerik per cluster", y=1.02)
plt.show()


