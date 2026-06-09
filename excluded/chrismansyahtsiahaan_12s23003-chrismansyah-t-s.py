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
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings

# Konfigurasi agar output bersih
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

# Path file spesifik Kaggle
FILE_PATH = '/kaggle/input/penguin-clustering-analysis/penguins.csv'

# Membaca Data
try:
    df = pd.read_csv(FILE_PATH)
    print(" Dataset berhasil dimuat!")
    print(f"Dimensi Data: {df.shape}")
except FileNotFoundError:
    print(f" Error: File tidak ditemukan di {FILE_PATH}. Cek kembali path input.")

# Menampilkan 5 baris pertama
df.head()


# Cek Missing Values
print("Missing values sebelum handling:\n", df.isnull().sum())

# Handling Missing Values (Drop baris yang kosong karena jumlahnya sedikit)
df_clean = df.dropna().reset_index(drop=True)
print(f"\nJumlah data setelah pembersihan: {len(df_clean)}")

# Feature Selection: Memilih hanya kolom numerik yang relevan untuk clustering
# Kita mengabaikan 'rowid' atau 'sex' untuk algoritma jarak murni
features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
X = df_clean[features]

# Standardisasi (Wajib untuk K-Means agar varians data setara)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData features (terstandarisasi) siap digunakan.")


inertia = []
sil_scores = []
K_range = range(2, 10) # Mencoba k dari 2 sampai 9

for k in K_range:
    # n_init=10 diset eksplisit untuk menghindari warning future version
    kmeans = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Visualisasi Grafik
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Elbow (Inertia)
color = 'tab:blue'
ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
ax1.set_ylabel('Inertia (Sum of Squared Error)', color=color, fontsize=12)
ax1.plot(K_range, inertia, marker='o', color=color, linewidth=2, label='Inertia')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, axis='x')

# Plot Silhouette Score
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Silhouette Score', color=color, fontsize=12)
ax2.plot(K_range, sil_scores, marker='s', linestyle='--', color=color, linewidth=2, label='Silhouette Score')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Metode Penentuan K Optimal: Elbow Method & Silhouette Analysis', fontsize=14)
plt.show()

print("Analisis Grafik:")
print("- Elbow Curve mulai melandai (siku) di k=3.")
print("- Silhouette Score cukup tinggi di k=2 dan k=3.")
print("- Keputusan: Kita pilih k=3 (sesuai jumlah spesies penguin pada umumnya: Adelie, Chinstrap, Gentoo).")


k_optimal = 3
kmeans_final = KMeans(n_clusters=k_optimal, init='random', n_init=10, random_state=42)
clusters = kmeans_final.fit_predict(X_scaled)

# Menyimpan hasil klaster ke dataframe utama
df_clean['Cluster_Label'] = clusters


# Menghitung metrik evaluasi
score_sil = silhouette_score(X_scaled, clusters)
score_db = davies_bouldin_score(X_scaled, clusters)

print(f"=== Evaluasi Model K-Means (k={k_optimal}) ===")
print(f"1. Silhouette Score     : {score_sil:.4f} (Semakin dekat ke 1 semakin baik)")
print(f"2. Davies-Bouldin Index : {score_db:.4f}  (Semakin dekat ke 0 semakin baik)")


# Menggunakan PCA untuk mereduksi 4 dimensi fitur menjadi 2 dimensi agar bisa digambar
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
# Scatter plot hasil klaster
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_clean['Cluster_Label'], palette='viridis', s=100, style=df_clean['Cluster_Label'])
# Menandai Centroid
centers_pca = pca.transform(kmeans_final.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=300, marker='*', label='Centroids')

plt.title(f'Visualisasi Cluster Penguin (PCA Projection)\nk={k_optimal}', fontsize=14)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

