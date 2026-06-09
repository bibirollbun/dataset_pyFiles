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


# 1. SETUP & LOAD DATA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# Path file menggunakan lokasi Kaggle Input yang spesifik
FILE_PATH = "/kaggle/input/penguin-clustering-analysis/penguins.csv"

# Load dataset
try:
    df = pd.read_csv(FILE_PATH)
    print("Dataset berhasil dimuat!")
    print(f"Dimensi Data: {df.shape}")
    print("\nData Head:")
    print(df.head())
except FileNotFoundError:
    print(f"ERROR: File tidak ditemukan di path: {FILE_PATH}. Pastikan Anda berada di lingkungan Kaggle.")


# 2. DATA PREPROCESSING & FEATURE SELECTION

# Cek Missing Values
print("Missing values sebelum handling:\n", df.isnull().sum())

# Drop baris dengan missing values dan reset index
df_clean = df.dropna().reset_index(drop=True)
print(f"\nJumlah data setelah pembersihan: {len(df_clean)}")

# Pilih fitur numerik untuk clustering
features = [
    'culmen_length_mm',
    'culmen_depth_mm',
    'flipper_length_mm',
    'body_mass_g'
]

X = df_clean[features]

# Standardisasi (Scaling) data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData features (terstandarisasi) siap digunakan.")


# 3. MENENTUKAN JUMLAH KLASTER OPTIMAL

inertia = []
sil_scores = []
K_range = range(2, 10)

for k in K_range:
    # Perbaikan: n_jobs=-1 dihapus dari konstruktor KMeans
    model = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)
    model.fit(X_scaled)
    inertia.append(model.inertia_)
    sil_scores.append(silhouette_score(X_scaled, model.labels_))

# Visualisasi Elbow & Silhouette
fig, ax1 = plt.subplots(figsize=(12, 6))

# Elbow (Inertia)
ax1.plot(K_range, inertia, marker='o', linewidth=2, color="blue", label="Inertia")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia (Sum of Squared Errors)", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")

# Silhouette Score
ax2 = ax1.twinx()
ax2.plot(K_range, sil_scores, marker='s', linestyle='--', linewidth=2, color="red", label="Silhouette Score")
ax2.set_ylabel("Silhouette Score (Max = 1)", color="red")
ax2.tick_params(axis='y', labelcolor="red")

plt.title("Metode Elbow & Silhouette Score untuk Optimal K")
plt.grid(True, linestyle=':')
plt.show()

print("\nAnalisis Grafik:")
print("- Elbow cenderung membentuk 'siku' di k=3.")
print("- Silhouette Score mencapai puncaknya di k=2 atau k=3.")
print("- Memilih k=3 karena ini sesuai dengan tiga spesies penguin yang diketahui.")


# 4. IMPLEMENTASI FINAL K-MEANS

k_optimal = 3
# Menggunakan n_init=10 untuk hasil yang robust
kmeans_final = KMeans(n_clusters=k_optimal, init='random', n_init=10, random_state=42)

clusters = kmeans_final.fit_predict(X_scaled)

# Tambahkan label klaster ke DataFrame yang bersih
df_clean["Cluster_Label"] = clusters

print(f"Klastering final dengan k={k_optimal} telah dilakukan.")
print("\nJumlah data per klaster:")
print(df_clean["Cluster_Label"].value_counts())


# 5. EVALUASI MODEL

score_sil = silhouette_score(X_scaled, clusters)
score_db = davies_bouldin_score(X_scaled, clusters)
inertia_final = kmeans_final.inertia_

print(f"=== Evaluasi Clustering (k={k_optimal}) ===")
print(f"Sum of Squared Errors (Inertia) : {inertia_final:.2f}")
print(f"Silhouette Score                : {score_sil:.4f} (semakin dekat 1 semakin baik)")
print(f"Davies-Bouldin Index            : {score_db:.4f} (semakin dekat 0 semakin baik)")


# 6. VISUALISASI PCA

# Reduksi dimensi menjadi 2 komponen
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))

# Scatter plot dari hasil klastering
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=df_clean["Cluster_Label"],
    palette="viridis",
    s=100,
    legend="full"
)

# Proyeksikan Centroids ke ruang PCA
centers_pca = pca.transform(kmeans_final.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=300, marker='X', label='Centroids', edgecolors='black')

plt.title("Visualisasi Cluster Penguin (PCA)")
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend(title='Cluster')
plt.grid(True, alpha=0.5)
plt.show()

