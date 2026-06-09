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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# Konfigurasi tampilan agar rapi
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
warnings.filterwarnings('ignore') # Mengabaikan warning minor agar notebook bersih

print("Libraries imported successfully.")


# Load dataset
# Pastikan path file sesuai dengan lokasi di Kaggle
try:
    df = pd.read_csv('/kaggle/input/penguin-clustering-analysis/penguins.csv')
except:
    df = pd.read_csv('penguins.csv') # Fallback untuk lokal

# Menampilkan informasi dasar
print(f"Dataset Shape: {df.shape}")
print("\nInfo Dataset:")
df.info()

# Menampilkan 5 baris pertama
display(df.head())


# Visualisasi Boxplot untuk mendeteksi Outlier pada fitur numerik
features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']

plt.figure(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i+1)
    sns.boxplot(data=df, y=col, color='skyblue')
    plt.title(f'Distribusi {col}', fontsize=12)
plt.tight_layout()
plt.show()

# Cek data ekstrim secara spesifik
print("Mengecek data aneh (Outliers):")
print(df[(df['flipper_length_mm'] > 4000) | (df['flipper_length_mm'] < 0)])


# 1. Drop Missing Values
df_clean = df.dropna().copy()

# 2. Hapus Outlier Ekstrim (Error Input)
# Menghapus data dengan flipper_length > 300mm (mustahil untuk penguin)
# Menghapus data negatif
df_clean = df_clean[(df_clean['flipper_length_mm'] > 0) & (df_clean['flipper_length_mm'] < 300)]

# 3. Membersihkan kolom 'sex' dari nilai yang tidak valid (seperti '.')
df_clean = df_clean[df_clean['sex'] != '.']

print(f"Ukuran data setelah cleaning: {df_clean.shape}")


# Encoding kolom 'sex' (Male=1, Female=0)
le = LabelEncoder()
df_clean['sex_encoded'] = le.fit_transform(df_clean['sex'])

# Memilih fitur untuk clustering
cluster_features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex_encoded']
X = df_clean[cluster_features]

# Standard Scaling (Mean=0, Std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Mengubah kembali ke DataFrame untuk kemudahan visualisasi nanti (opsional tapi rapi)
X_scaled_df = pd.DataFrame(X_scaled, columns=cluster_features)
display(X_scaled_df.head())


inertia = []
silhouette_scores = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plotting Hasil Evaluasi
fig, ax1 = plt.subplots(1, 2, figsize=(15, 5))

# Grafik Elbow
ax1[0].plot(K_range, inertia, marker='o', linestyle='--', color='teal')
ax1[0].set_title('Elbow Method (Inertia)')
ax1[0].set_xlabel('Number of Clusters (k)')
ax1[0].set_ylabel('Inertia')

# Grafik Silhouette
ax1[1].plot(K_range, silhouette_scores, marker='o', linestyle='--', color='purple')
ax1[1].set_title('Silhouette Score Analysis')
ax1[1].set_xlabel('Number of Clusters (k)')
ax1[1].set_ylabel('Silhouette Score')

plt.show()


# Penerapan K-Means dengan k=3
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean['cluster'] = kmeans_final.fit_predict(X_scaled)

# Menghitung pusat cluster (centroid) dalam skala asli
numeric_cols = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
cluster_summary = df_clean.groupby('cluster')[numeric_cols].mean()

print("Rata-rata karakteristik setiap cluster:")
display(cluster_summary)


# Visualisasi Distribusi Cluster
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_clean, x='culmen_length_mm', y='culmen_depth_mm', 
                hue='cluster', palette='viridis', s=100, alpha=0.8)
plt.title('Visualisasi Cluster: Culmen Length vs Depth', fontsize=15)
plt.xlabel('Culmen Length (mm)')
plt.ylabel('Culmen Depth (mm)')
plt.legend(title='Cluster ID')
plt.show()

# Pairplot Komprehensif
sns.pairplot(df_clean, vars=numeric_cols, hue='cluster', palette='viridis', corner=True)
plt.suptitle('Pairplot Fitur Berdasarkan Cluster', y=1.02, fontsize=16)
plt.show()




