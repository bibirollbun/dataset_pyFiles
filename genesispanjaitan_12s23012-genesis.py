# ==========================================================================================
# BAGIAN 1: IMPORT LIBRARY
# ==========================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

# Mengatur gaya visualisasi
sns.set(style="whitegrid")

# ==========================================================================================
# BAGIAN 2: MENCARI & MEMUAT DATA SECARA OTOMATIS
# ==========================================================================================
# Kita cari file 'penguins.csv' di folder input Kaggle secara otomatis
# agar tidak terjadi error "File Not Found".

target_file = "penguins.csv"
file_path = None

print("Sedang mencari dataset...")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename == target_file:
            file_path = os.path.join(dirname, filename)
            print(f"--> File DITEMUKAN di: {file_path}")
            break
    if file_path: break

if file_path:
    df = pd.read_csv(file_path)
    print("\nData berhasil dimuat!")
    print(df.head())
else:
    # Error handling jika file benar-benar tidak ada
    raise FileNotFoundError("File 'penguins.csv' tidak ditemukan! Pastikan Anda sudah Add Data.")

# ==========================================================================================
# BAGIAN 3: PRE-PROCESSING (PEMBERSIHAN DATA)
# ==========================================================================================

# [1] Pemilihan Fitur (Hanya ambil kolom numerik)
features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
X = df[features].copy()

# [2] Mengatasi Missing Values (NaN)
# Mengisi data kosong dengan nilai rata-rata (mean)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# [3] Standardisasi Data (Scaling)
# Penting agar berat badan (ribuan gram) tidak mendominasi fitur lain (puluhan mm)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ==========================================================================================
# BAGIAN 4: MENENTUKAN K OPTIMAL (ELBOW & SILHOUETTE)
# ==========================================================================================
inertia = []
silhouette_scores = []
K_range = range(2, 10)

print("\nMenghitung K Optimal (Elbow Method & Silhouette Score)...")

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Visualisasi Grafik
fig, ax1 = plt.subplots(figsize=(12, 6))

# Grafik Elbow (Inertia)
ax1.plot(K_range, inertia, 'bo-', label='Inertia (Elbow)')
ax1.set_xlabel('Jumlah Klaster (K)')
ax1.set_ylabel('Inertia (SSE)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Grafik Silhouette Score
ax2 = ax1.twinx()
ax2.plot(K_range, silhouette_scores, 'rs--', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Evaluasi Jumlah Klaster: Elbow vs Silhouette')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

plt.show()

# ==========================================================================================
# BAGIAN 5: TRAINING MODEL FINAL
# ==========================================================================================
# Kita gunakan K=3 sesuai pengetahuan biologi (3 spesies penguin)
# Meskipun Silhouette score mungkin menyarankan K=2, K=3 lebih masuk akal secara konteks.
final_k = 3

print(f"\nMelakukan Clustering Final dengan K={final_k}...")
kmeans_final = KMeans(n_clusters=final_k, init='random', n_init=10, random_state=42)
clusters = kmeans_final.fit_predict(X_scaled)

# Simpan hasil ke DataFrame
df['Cluster_Labels'] = clusters

# ==========================================================================================
# BAGIAN 6: EVALUASI & VISUALISASI HASIL
# ==========================================================================================
# Hitung skor final
final_score = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score Final (K={final_k}): {final_score:.4f}")

# Visualisasi Scatter Plot (Perbaikan dari error sebelumnya)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='culmen_length_mm', y='flipper_length_mm', 
                hue='Cluster_Labels', palette='viridis', s=100, alpha=0.8)

plt.title(f'Hasil Clustering Penguin (K={final_k})\nCulmen Length vs Flipper Length')
plt.xlabel('Culmen Length (mm)')
plt.ylabel('Flipper Length (mm)')
plt.legend(title='Cluster ID')
plt.show()

# Tampilkan karakteristik rata-rata tiap klaster
print("\n=== KARAKTERISTIK RATA-RATA TIAP KLASTER ===")
print(df.groupby('Cluster_Labels')[features].mean().round(2))
print("\nTips Interpretasi:")
print("- Lihat klaster dengan 'body_mass_g' terbesar -> Kemungkinan Gentoo Penguin.")
print("- Lihat perbedaan 'culmen_depth' dan 'culmen_length' untuk membedakan Adelie & Chinstrap.")

# ==========================================================================================
# BAGIAN 7: SIMPAN SUBMISSION
# ==========================================================================================
submission = pd.DataFrame()
submission['id'] = df.index  # Gunakan index sebagai ID
submission['label'] = clusters

submission.to_csv('submission.csv', index=False)
print("\nSUKSES: File 'submission.csv' berhasil disimpan!")

