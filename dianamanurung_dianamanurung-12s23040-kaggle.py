# Import Library yang dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Konfigurasi visualisasi
sns.set(style="whitegrid")

# Load Dataset (Gunakan path yang sudah kita temukan tadi)
df = pd.read_csv('/kaggle/input/penguin-clustering-analysis/penguins.csv')

# Tampilkan 5 baris data teratas
print("Dataset Shape:", df.shape)
df.head()


# Cek Missing Values
print("Jumlah Missing Values:\n", df.isnull().sum())

# Cek Statistik Deskriptif
# Perhatikan nilai MAX pada flipper_length_mm (5000) dan MIN (-132) -> INI ANOMALI
print("\nStatistik Deskriptif:")
display(df[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']].describe())

# Visualisasi Outlier menggunakan Boxplot
plt.figure(figsize=(10, 4))
sns.boxplot(x=df['flipper_length_mm'], color='red')
plt.title("Deteksi Outlier Ekstrem pada Panjang Sirip (Sebelum Cleaning)")
plt.show()


# 1. Menghapus baris dengan Missing Values (NaN)
df_clean = df.dropna().copy()

# 2. Menghapus Outlier Ekstrem
# Kita memfilter data agar hanya menyisakan nilai yang masuk akal secara biologis
# (Sirip harus > 0 mm dan < 300 mm)
df_clean = df_clean[
    (df_clean['flipper_length_mm'] > 0) & 
    (df_clean['flipper_length_mm'] < 300)
]

print("Ukuran Data Setelah Cleaning:", df_clean.shape)
print("Status: Outlier (5000mm dan -132mm) berhasil dihapus.")


# Memilih fitur numerik untuk clustering
features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
X = df_clean[features]

# Standard Scaling (Mean=0, Std=1)
# Agar berat badan (ribuan) tidak mendominasi panjang paruh (puluhan)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data berhasil dinormalisasi.")
print("Contoh 5 data pertama (Scaled):\n", X_scaled[:5])


inertia = []
silhouette_scores = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Visualisasi Evaluasi (Dual Axis Plot)
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot Inertia (Elbow)
ax1.set_xlabel('Jumlah Cluster (k)')
ax1.set_ylabel('Inertia (Elbow)', color='tab:blue')
ax1.plot(K_range, inertia, 'o-', color='tab:blue', label='Inertia')
ax1.grid(True)

# Plot Silhouette Score
ax2 = ax1.twinx()
ax2.set_ylabel('Silhouette Score', color='tab:orange')
ax2.plot(K_range, silhouette_scores, 'o--', color='tab:orange', label='Silhouette Score')

plt.title('Evaluasi Jumlah Cluster Terbaik: K=3 Terlihat Paling Optimal')
plt.show()


# Modeling dengan K=3 (Optimal)
k_optimal = 3
kmeans_final = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans_final.fit_predict(X_scaled)

# Reduksi Dimensi ke 2D menggunakan PCA untuk keperluan visualisasi
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plotting Hasil Clustering
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=X_pca[:, 0], 
    y=X_pca[:, 1], 
    hue=df_clean['Cluster'], 
    palette='viridis', 
    s=100, 
    alpha=0.8
)
plt.title(f'Visualisasi Cluster Penguin (K={k_optimal}) dengan PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster ID')
plt.show()


# Mengelompokkan rata-rata fitur berdasarkan Cluster
cluster_summary = df_clean.groupby('Cluster')[features].mean()

print("--- RATA-RATA KARAKTERISTIK PER CLUSTER ---")
print(cluster_summary)

print("\n--- KESIMPULAN ANALISIS ---")
print("Berdasarkan hasil clustering, populasi penguin terbagi menjadi 3 kelompok distinktif:")
print("1. Cluster 0: Ukuran tubuh KECIL, Paruh PENDEK (Kemungkinan Adelie).")
print("2. Cluster 1: Ukuran tubuh BESAR, Sirip PANJANG (Kemungkinan Gentoo).")
print("3. Cluster 2: Ukuran tubuh SEDANG, Paruh PANJANG (Kemungkinan Chinstrap).")
print("\nMetode K-Means terbukti efektif memisahkan spesies berdasarkan fitur fisik setelah pembersihan outlier.")

