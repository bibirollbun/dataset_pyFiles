import os                                 # cek file path
import warnings                           # sembunyikan warning agar tampilan rapi
warnings.filterwarnings("ignore")

import numpy as np                        # operasi numerik
import pandas as pd                       # manipulasi tabel (DataFrame)
import matplotlib.pyplot as plt           # plotting dasar
import seaborn as sns                     # plotting statistik
sns.set(style="whitegrid", rc={"figure.figsize":(9,6)})  # atur gaya grafik

# sklearn: scaling, kmeans, pca, metrik evaluasi
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score,
                             calinski_harabasz_score,
                             davies_bouldin_score,
                             adjusted_rand_score)   # ARI jika ada label 'species'
# scipy (untuk Dunn index) akan dipakai di section tersendiri


path = "/mnt/data/penguins.csv"                   # path file yang kamu upload
if not os.path.exists(path):                      # jika path tidak ada, coba fallback Kaggle
    path = "/kaggle/input/penguin-clustering-analysis/penguins.csv"

print("Membaca file dari:", path)                 # tampilkan path yang dipakai
df = pd.read_csv(path)                            # baca CSV ke DataFrame
print("Ukuran dataset (baris, kolom):", df.shape) # info dimensi
print("Kolom dataset:", df.columns.tolist())      # tampilkan nama kolom
display(df.head())                                # tampilkan 5 baris pertama


candidate_features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
available = [c for c in candidate_features if c in df.columns]  # hanya yang ada di file

print("Fitur kandidat tersedia:", available)      # tampilkan fitur yang bisa dipakai
print("\nJumlah missing pada fitur terpilih:")
print(df[available].isnull().sum())               # jumlah NaN per kolom

# Buat salinan data bersih: hapus baris yang ada NaN di fitur penting
df_clean = df.dropna(subset=available).copy()     # .copy() agar aman untuk assign kolom baru
print(f"\nBaris sebelum dropna: {len(df)}, setelah dropna: {len(df_clean)}")


display(df_clean[available].describe())

corr = df_clean[available].corr()                 # matriks korelasi antar fitur
print("\nKorelasinya:")
display(corr)

# Visualisasi heatmap korelasi untuk melihat hubungan fitur
plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap Korelasi Fitur")
plt.show()

# Keputusan fitur: untuk dataset penguin, ke-4 fitur ini informatif -> gunakan semua yang tersedia
features = available.copy()
print("Fitur final yang dipakai untuk clustering:", features)


X = df_clean[features].values                     # matriks fitur asli
scaler = StandardScaler()                         # inisialisasi scaler
X_scaled = scaler.fit_transform(X)                # transformasi (mean=0, std=1)

# Simpan versi scaled ke dataframe agar mudah diinspeksi (opsional)
for i, col in enumerate(features):
    df_clean[col + "_scaled"] = X_scaled[:, i]


from scipy.spatial.distance import cdist

def dunn_index(X, labels):
    # Jika hanya 1 cluster atau dataset kosong, kembalikan nan
    labels_unique = np.unique(labels)
    if len(labels_unique) < 2:
        return np.nan

    # Intra-cluster diameter: max pairwise distance setiap cluster, ambil max dari semua
    diameters = []
    for lab in labels_unique:
        pts = X[labels == lab]
        if len(pts) < 2:
            diameters.append(0.0)
        else:
            d = cdist(pts, pts)
            diameters.append(d.max())
    max_diameter = np.max(diameters) if len(diameters) > 0 else 0.0

    # Inter-cluster: hitung jarak minimal antar dua cluster (min pairwise)
    min_inter = np.inf
    for i in range(len(labels_unique)):
        for j in range(i+1, len(labels_unique)):
            a = X[labels == labels_unique[i]]
            b = X[labels == labels_unique[j]]
            if len(a) == 0 or len(b) == 0:
                continue
            dij = cdist(a, b).min()
            if dij < min_inter:
                min_inter = dij

    if max_diameter == 0:
        return np.nan
    return min_inter / max_diameter


results = []
K_range = range(2, 11)                        # mulai k=2 karena silhouette tidak didefinisikan untuk k=1

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)  # n_init eksplisit untuk kestabilan
    labs = km.fit_predict(X_scaled)                       # latih K-Means pada data scaled
    inertia = km.inertia_                                 # SSE (sum squared error)
    sil = silhouette_score(X_scaled, labs)                # silhouette
    ch = calinski_harabasz_score(X_scaled, labs)          # Calinski-Harabasz
    db = davies_bouldin_score(X_scaled, labs)             # Davies-Bouldin (lebih kecil lebih baik)
    dunn = dunn_index(X_scaled, labs)                     # Dunn index (lebih besar lebih baik)
    results.append({'k':k, 'inertia':inertia, 'silhouette':sil,
                    'calinski':ch, 'davies':db, 'dunn':dunn})

metrics_df = pd.DataFrame(results)             # ringkasan metrik per k
display(metrics_df)


plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(metrics_df['k'], metrics_df['inertia'], marker='o'); plt.title('Elbow (Inertia)'); plt.xlabel('k'); plt.grid(True)
plt.subplot(1,3,2)
plt.plot(metrics_df['k'], metrics_df['silhouette'], marker='o'); plt.title('Silhouette Score'); plt.xlabel('k'); plt.grid(True)
plt.subplot(1,3,3)
plt.plot(metrics_df['k'], metrics_df['dunn'], marker='o'); plt.title('Dunn Index'); plt.xlabel('k'); plt.grid(True)
plt.suptitle('Evaluasi metrik vs jumlah cluster (k)')
plt.show()

# Rekomendasi sederhana: pilih k dengan silhouette tertinggi, cross-check dengan Dunn/inertia
best_by_sil = metrics_df.loc[metrics_df['silhouette'].idxmax(), 'k']
best_by_dunn = metrics_df.loc[metrics_df['dunn'].idxmax(), 'k']
print("Rekomendasi k -> silhouette:", best_by_sil, ", dunn:", best_by_dunn)

# Ambil k_final: jika sama antara silhouette & dunn, pilih itu; jika tidak, pakai silhouette
k_final = int(best_by_sil)
print("Memilih k_final =", k_final)


kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init=50)  # n_init lebih besar agar stabil
labels_final = kmeans_final.fit_predict(X_scaled)
df_clean.loc[:, 'Cluster'] = labels_final  # simpan label cluster pada dataframe

# Hitung centroid di skala asli (kembalikan dari skala)
centroids_scaled = kmeans_final.cluster_centers_
centroids_orig = scaler.inverse_transform(centroids_scaled)
centroid_df = pd.DataFrame(centroids_orig, columns=features)
print("Centroid (skala asli):")
display(centroid_df)


inertia_final = kmeans_final.inertia_
sil_final = silhouette_score(X_scaled, labels_final) if len(np.unique(labels_final))>1 else np.nan
ch_final = calinski_harabasz_score(X_scaled, labels_final) if len(np.unique(labels_final))>1 else np.nan
db_final = davies_bouldin_score(X_scaled, labels_final) if len(np.unique(labels_final))>1 else np.nan
dunn_final = dunn_index(X_scaled, labels_final)

print(f"Inertia (SSE): {inertia_final:.2f}")
print(f"Silhouette: {sil_final:.4f}")
print(f"Calinski-Harabasz: {ch_final:.4f}")
print(f"Davies-Bouldin: {db_final:.4f}")
print(f"Dunn index: {dunn_final:.4f}")

# Jika kolom 'species' ada, hitung Adjusted Rand Index (ARI) untuk membandingkan hasil cluster dengan label asli
if 'species' in df_clean.columns:
    true_labels = pd.factorize(df_clean['species'])[0]
    ari_val = adjusted_rand_score(true_labels, labels_final)
    print("Adjusted Rand Index (ARI) dibanding species:", round(ari_val,4))
else:
    print("Kolom 'species' tidak tersedia -> ARI tidak dihitung.")

# Ringkasan ukuran cluster dan rata-rata fitur tiap cluster (skala asli)
print("\nJumlah titik per cluster:")
print(df_clean['Cluster'].value_counts().sort_index())

print("\nRata-rata fitur per cluster (skala asli):")
display(df_clean.groupby('Cluster')[features].mean())

# Tulis interpretasi singkat per cluster (bahasa manusia)
for c in sorted(df_clean['Cluster'].unique()):
    m = df_clean.loc[df_clean['Cluster']==c, features].mean()
    print(f"\nInterpretasi Cluster {c}:")
    print(f" - culmen_length ≈ {m['culmen_length_mm']:.2f} mm, culmen_depth ≈ {m['culmen_depth_mm']:.2f} mm")
    print(f" - flipper_length ≈ {m['flipper_length_mm']:.2f} mm, body_mass ≈ {m['body_mass_g']:.2f} g")
    # contoh penjelasan: paruh panjang vs pendek, tubuh besar vs kecil
    if m['culmen_length_mm'] > df_clean['culmen_length_mm'].mean():
        pl = "paruh relatif panjang"
    else:
        pl = "paruh relatif pendek"
    if m['body_mass_g'] > df_clean['body_mass_g'].mean():
        tm = "tubuh relatif besar"
    else:
        tm = "tubuh relatif kecil"
    print(f" -> Kesimpulan cepat: {pl}, {tm}.")


# 10a - Scatter Culmen length vs depth (gunakan data asli agar menyebar)
plt.figure(figsize=(10,7))
plt.scatter(df_clean['culmen_length_mm'], df_clean['culmen_depth_mm'],
            c=df_clean['Cluster'], cmap='viridis', s=70, edgecolor='k', alpha=0.9)
plt.xlabel('Culmen Length (mm)')
plt.ylabel('Culmen Depth (mm)')
plt.title(f'Visualisasi Cluster (k={k_final}) - Culmen length vs depth')
plt.colorbar(label='Cluster ID')
plt.show()



# 10b - Pairplot lengkap: melihat distribusi fitur antar cluster
plot_cols = features + ['Cluster']
sns.pairplot(df_clean[plot_cols], hue='Cluster', palette='viridis', diag_kind='kde', plot_kws={'s':40, 'edgecolor':'k'})
plt.suptitle("Pairplot Fitur Berdasarkan Cluster", y=1.02)
plt.show()


# Section 11 - Simpan hasil agar bisa diunduh / disubmit
metrics_df = metrics_df.copy()
metrics_df.to_csv('/kaggle/working/cluster_metrics_per_k.csv', index=False)  # metrik tiap k
df_clean.to_csv('/kaggle/working/penguins_clustered.csv', index=False)       # hasil final
centroid_df.to_csv('/kaggle/working/centroids.csv', index=False)            # centroid skala asli

print("File tersimpan di /kaggle/working/: cluster_metrics_per_k.csv, penguins_clustered.csv, centroids.csv")


