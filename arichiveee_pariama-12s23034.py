import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# ------------------------------------------------------------
# 0. CEK FILE DI FOLDER INPUT (opsional, biar kelihatan rapi)
# ------------------------------------------------------------
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Memuat file 'penguins.csv' ke dalam DataFrame
penguin_df = pd.read_csv('/kaggle/input/penguin-clustering-analysis/penguins.csv')

print("Ukuran data:", penguin_df.shape)
penguin_df.head()



# Cek jumlah missing value per kolom
print("\nJumlah missing value per kolom:")
print(penguin_df.isna().sum())

# Di sini aku menggunakan strategi dropna untuk menyederhanakan.
# (Kalau mau lebih advanced bisa pakai imputasi mean.)
penguin_df = penguin_df.dropna()
print("\nUkuran data setelah drop NaN:", penguin_df.shape)

penguin_df.head()


# Memilih fitur numerik utama untuk clustering:
# - culmen_length_mm
# - culmen_depth_mm
# - flipper_length_mm
# - body_mass_g
X = penguin_df[['culmen_length_mm',
                'culmen_depth_mm',
                'flipper_length_mm',
                'body_mass_g']].copy()

# Scatter plot awal untuk melihat pola data (2 dimensi dulu)
plt.figure(figsize=(10,5))
plt.scatter(X['culmen_length_mm'], X['culmen_depth_mm'])
plt.xlabel('culmen_length_mm')
plt.ylabel('culmen_depth_mm')
plt.title('Sebaran Data Penguin (sebelum clustering)')
plt.show()


# K-Means sensitif terhadap skala fitur, jadi perlu distandarisasi.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


inertias = []
sil_scores = []
K_range = range(2, 11)   # coba dari 2 sampai 10 cluster

for k in K_range:
    kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_tmp = kmeans_tmp.fit_predict(X_scaled)
    
    inertias.append(kmeans_tmp.inertia_)               # SSE / inertia
    sil_scores.append(silhouette_score(X_scaled, labels_tmp))

# Plot Elbow Method
plt.figure(figsize=(10,4))
plt.plot(list(K_range), inertias, marker='o')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Inertia (SSE)')
plt.title('Elbow Method')
plt.grid(True)
plt.show()

# Plot Silhouette Score
plt.figure(figsize=(10,4))
plt.plot(list(K_range), sil_scores, marker='o')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score untuk berbagai nilai k')
plt.grid(True)
plt.show()

# Dari dua grafik di atas, biasanya k=3 terlihat cukup masuk akal
# (bisa kamu sesuaikan kalau grafikmu menunjukkan hal lain).
k_optimal = 3
print("\nK terpilih (k_optimal):", k_optimal)


kmeans_final = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Simpan label cluster ke dataframe asli
penguin_df['Cluster_Label'] = cluster_labels

print("\nJumlah data per cluster:")
print(penguin_df['Cluster_Label'].value_counts())

# ---- Metrik evaluasi klasterisasi ----
inertia_final = kmeans_final.inertia_
silhouette_final = silhouette_score(X_scaled, cluster_labels)
ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
db_score = davies_bouldin_score(X_scaled, cluster_labels)

print("\n=== METRIK KLASTERISASI (MODEL FINAL) ===")
print("Inertia (SSE):", inertia_final)
print("Silhouette Score:", silhouette_final)
print("Calinski-Harabasz Index:", ch_score)
print("Davies-Bouldin Index:", db_score)


plt.figure(figsize=(10,5))

for c in range(k_optimal):
    subset = penguin_df[penguin_df['Cluster_Label'] == c]
    plt.scatter(subset['culmen_length_mm'],
                subset['culmen_depth_mm'],
                label=f'Cluster {c}',
                alpha=0.7)

# menandai centroid juga (di space asli, bukan scaled)
centers_scaled = kmeans_final.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)

plt.scatter(centers_original[:, 0],
            centers_original[:, 1],
            marker='X',
            s=200,
            edgecolor='black',
            label='Centroid')

plt.xlabel('culmen_length_mm')
plt.ylabel('culmen_depth_mm')
plt.title('Hasil Clustering K-Means pada Data Penguin')
plt.legend()
plt.show()


# MEMBUAT FILE SUBMISSION UNTUK KAGGLE

submission_data = pd.DataFrame({
    'Id': penguin_df.index.values,
    'Cluster_Label': penguin_df['Cluster_Label']
})

submission_data.head()

submission_data.to_csv('submission.csv', index=False)

print("File 'submission.csv' berhasil dibuat.")

