# CELL 1 — Import Library & Setup (tanpa warning)
import warnings                                                        # Untuk mematikan warning kosmetik
warnings.filterwarnings("ignore")                                      # ← Hilangkan semua RuntimeWarning
import seaborn as sns, pandas as pd, matplotlib.pyplot as plt          # Library utama
from sklearn.preprocessing import StandardScaler                       # Standarisasi data
from sklearn.cluster import KMeans                                     # Algoritma K-Means

df = sns.load_dataset('penguins') \
     .rename(columns={'bill_length_mm':'culmen_length_mm', 
                      'bill_depth_mm':'culmen_depth_mm'})   
df.head()                                                              


# CELL 2 — Pembersihan & Standarisasi Data
features = ['culmen_length_mm', 'culmen_depth_mm', 
            'flipper_length_mm', 'body_mass_g']                        # 4 fitur untuk clustering

df = df.dropna(subset=features).reset_index(drop=True)                # Hapus NaN hanya di fitur numerik
X = df[features]                                                       # Ambil data fitur
X_scaled = StandardScaler().fit_transform(X)                          # Standarisasi

print(f"Data siap → {X_scaled.shape[0]} penguin (bersih total)")


# CELL 3 — Elbow Method: Cari Jumlah Cluster Terbaik
inertia = []                                                           # List untuk menyimpan nilai inertia
for k in range(1, 10):                                                 # Coba k dari 1 sampai 9
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1,10), inertia, 'o-', color='#3498db', markersize=8)   # Plot garis + titik
plt.title('Elbow Method', fontsize=14, fontweight='bold')              # Judul metode Elbow
plt.xlabel('Jumlah Cluster (k)')                                       # Label sumbu X
plt.ylabel('Inertia')                                                  # Label sumbu Y
plt.grid(alpha=0.3)                                                    # Grid tipis
plt.show()                                                             # Tampilkan grafik


# CELL 4 — Training Model Final dengan k = 3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)              # Buat model akhir dengan 3 cluster
df['cluster'] = kmeans.fit_predict(X_scaled)                           # Latih + langsung beri label cluster

print("Clustering selesai!")                                           # Konfirmasi training sukses
print("\nDistribusi cluster:")                                         # Spasi biar rapi
print(df['cluster'].value_counts().sort_index())                       # Tampilkan jumlah penguin per cluster


# CELL 5 — Visualisasi Hasil + Simpan Submission
plt.figure(figsize=(9,6))                                              # Ukuran grafik ideal
plt.scatter(df['culmen_length_mm'], df['culmen_depth_mm'], 
            c=df['cluster'], cmap='rainbow', s=70)                     # Warna = cluster, ukuran titik sedang
plt.title('Hasil Clustering Penguin (k=3)', fontsize=16)               # Judul utama
plt.xlabel('Culmen Length (mm)')                                       # Label X
plt.ylabel('Culmen Depth (mm)')                                        # Label Y
plt.grid(alpha=0.3)                                                    # Grid tipis
plt.show()                                                             # Tampilkan grafik

df[['cluster']].to_csv('submission.csv', index=False)                  # Simpan hanya kolom cluster




