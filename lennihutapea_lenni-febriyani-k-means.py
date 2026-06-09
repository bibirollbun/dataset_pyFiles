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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Library Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Konfigurasi agar output bersih & rapi
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load Dataset
try:
    df = pd.read_csv('/kaggle/input/penguin-clustering-analysis/penguins.csv')
except:
    df = pd.read_csv('penguins.csv')

print("Dataset berhasil dimuat!")
df.head()


## 2. Data Cleaning (Handling Missing Values)

# Cek data kosong
print(f"Missing values awal: {df.isnull().sum().sum()}")

# Strategi: Saya memilih mengisi data kosong (Imputasi) daripada membuangnya (Drop),
# agar jumlah data tetap maksimal (344 baris) dan analisis lebih akurat.

numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

# Isi angka dengan Mean
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# Isi kategori dengan Modus
for col in categorical_cols:
    if len(df[col].mode()) > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print(f"Missing values setelah cleaning: {df.isnull().sum().sum()}")
print(f"Total Data Siap Pakai: {len(df)} baris")


## 3. Preprocessing

# Memilih fitur fisik + sex
features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']
X_raw = df[features].copy()

# Encoding: Mengubah Sex (Male/Female) menjadi angka
le = LabelEncoder()
X_raw['sex'] = le.fit_transform(X_raw['sex'].astype(str))

# Scaling: Wajib untuk K-Means agar varians data setara
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print("Data berhasil distandarisasi dan siap untuk modeling.")


## 4. Menentukan Jumlah Klaster Optimal (K)

inertia = []
sil_scores = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plotting Grafik Analisis
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Elbow Method
ax[0].plot(K_range, inertia, marker='o', color='#003f5c')
ax[0].set_title('Elbow Method (Cari Siku)')
ax[0].set_xlabel('Jumlah Klaster (K)')
ax[0].set_ylabel('Inertia')
ax[0].axvline(x=3, color='red', linestyle='--')

# Silhouette Score
ax[1].plot(K_range, sil_scores, marker='o', color='#7a5195')
ax[1].set_title('Silhouette Score (Makin Tinggi Makin Bagus)')
ax[1].set_xlabel('Jumlah Klaster (K)')
ax[1].set_ylabel('Score')
ax[1].axvline(x=3, color='red', linestyle='--')

plt.show()


## 5. Modeling & Evaluasi Mendalam (PCA Visualization)

# Jalankan K-Means Final
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)
df['Cluster'] = clusters

# --- Evaluasi Metrik Lengkap ---
score_sil = silhouette_score(X_scaled, clusters)
score_db = davies_bouldin_score(X_scaled, clusters)

print(f"=== Evaluasi Model (K=3) ===")
print(f"Silhouette Score      : {score_sil:.4f} (Mendekati 1 = Sangat Baik)")
print(f"Davies-Bouldin Index  : {score_db:.4f}  (Mendekati 0 = Sangat Baik)")

# --- Visualisasi PCA (Principal Component Analysis) ---
# Teknik ini memadatkan 4 dimensi fitur menjadi 2 dimensi agar bisa digambar
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centers_pca = pca.transform(kmeans_final.cluster_centers_)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', s=100, alpha=0.8)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=300, marker='*', label='Centroids')
plt.title('Visualisasi Cluster Penguin (Proyeksi PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

# Tampilkan statistik rata-rata untuk interpretasi
print("\nStatistik Rata-rata per Klaster (Gunakan ini untuk analisis):")
display(df.groupby('Cluster')[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']].mean())

