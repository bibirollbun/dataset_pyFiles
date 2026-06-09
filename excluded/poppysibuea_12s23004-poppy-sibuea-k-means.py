# 1. Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# Agar visualisasi lebih rapi
plt.style.use('ggplot')

# 2. Load Data Otomatis (Anti-Error Path)
file_path = ''
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename == 'penguins.csv':
            file_path = os.path.join(dirname, filename)

if file_path == '':
    print("Error: File penguins.csv tidak ditemukan!")
else:
    print(f"File ditemukan di: {file_path}")
    df = pd.read_csv(file_path)
    
    print("\nData Awal:")
    print(df.head())

    # 3. Data Cleaning
    # Hapus baris kosong (NaN)
    df_clean = df.dropna().copy()
    
    # === BAGIAN BARU: HAPUS OUTLIER (DATA SAMPAH) ===
    # Kita buang data yang siripnya > 4000 mm (tidak masuk akal)
    # Kita juga buang data yang siripnya <= 0 (error negatif)
    df_clean = df_clean[df_clean['flipper_length_mm'] < 4000]
    df_clean = df_clean[df_clean['flipper_length_mm'] > 0]
    
    print(f"Sisa data bersih setelah hapus outlier: {len(df_clean)} baris")
    # =======================================================
    
    # 4. Seleksi Fitur
    # Kita hanya ambil kolom angka (numerik) dari data yang SUDAH BERSIH
    X = df_clean.select_dtypes(include=[np.number])
    
    # Tampilkan fitur yang dipakai
    print("\nFitur yang digunakan untuk Clustering:")
    print(X.head())

    # 5. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("\nData berhasil di-scale dan siap untuk K-Means!")


# --- TAHAP 2: Menentukan Jumlah Cluster (Re-Run dengan Data Bersih) ---

# Import ulang Library agar aman
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

inertia = []
silhouette_scores = []
K_range = range(2, 10)

print("Sedang menghitung ulang dengan data bersih...")

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled) # X_scaled ini sudah versi bersih
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Visualisasi
plt.figure(figsize=(15, 5))

# Elbow
plt.subplot(1, 2, 1)
plt.plot(K_range, inertia, marker='o', linestyle='--', color='blue')
plt.title('Elbow Method (Data Bersih)')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Inertia')
plt.grid(True)

# Silhouette
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, marker='o', linestyle='--', color='green')
plt.title('Silhouette Score (Data Bersih)')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Silhouette Coefficient')
plt.grid(True)

plt.tight_layout()
plt.show()


# --- TAHAP 3 & 4: FINAL CLUSTERING & SUBMISSION ---

# Import library (jaga-jaga kalau sesi terputus)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Jalankan K-Means dengan K=3 (Sesuai jumlah spesies asli)
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
# Ingat: Kita fit ke data yang SUDAH DIBERSIHKAN
clusters = kmeans_final.fit_predict(X_scaled)

# 2. Masukkan label hasil ke dataframe bersih
df_clean['Cluster'] = clusters

# 3. Visualisasi Scatter Plot (Hasil Akhir)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_clean, 
    x='flipper_length_mm', 
    y='body_mass_g', 
    hue='Cluster', 
    palette='viridis', 
    s=100, 
    alpha=0.8
)
plt.title('Hasil Akhir Clustering Penguin (3 Cluster)')
plt.xlabel('Panjang Sirip (mm)')
plt.ylabel('Berat Badan (g)')
plt.legend(title='Cluster ID')
plt.show()

# 4. Interpretasi Data (Statistik Rata-rata)
# Ini bahan untuk kamu menulis analisis di laporan/Readme
print("\n=== PROFIL RATA-RATA TIAP CLUSTER ===")
numeric_cols = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
display(df_clean.groupby('Cluster')[numeric_cols].mean())

# 5. Buat File Submission (CSV)
# Kita ambil index asli sebagai ID
submission = df_clean.reset_index()[['index', 'Cluster']]
submission.columns = ['id', 'Cluster'] # Sesuaikan header

# Simpan ke file
submission.to_csv('submission_final.csv', index=False)
print("\n✅ SELESAI! File 'submission_final.csv' berhasil dibuat.")
print("Silakan download dari panel Output di sebelah kanan.")

