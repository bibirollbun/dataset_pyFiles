import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import warnings

warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '1'

%matplotlib inline


file_path = 'penguins.csv'

if os.path.exists(file_path):
    data = pd.read_csv(file_path)
    print(f"Data berhasil dimuat dari: {file_path}")
else:
    # Cek path alternatif (biasanya struktur di Kaggle)
    kaggle_path = '/kaggle/input/penguin-clustering-analysis/penguins.csv'
    if os.path.exists(kaggle_path):
        data = pd.read_csv(kaggle_path)
        print(f"Data berhasil dimuat dari: {kaggle_path}")
    else:
        print("ERROR: File 'penguins.csv' tidak ditemukan. Pastikan Anda sudah mengupload file tersebut.")

# Tampilkan 5 data teratas
if 'data' in locals():
    display(data.head())


features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']

X = data[features]

X = X.dropna()

print(f"Jumlah data setelah pembersihan: {len(X)} baris")
X.head()


# Inisialisasi Scaler
scaler = StandardScaler()

# Transformasi data
X_scaled = scaler.fit_transform(X)

# ubah kembali ke DataFrame agar mudah dilihat 
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
X_scaled_df.describe()


inertia = []
range_values = range(1, 10)

for k in range_values:
    kmeans = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plotting Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(range_values, inertia, marker='o', linestyle='--')
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('Inertia (Error)')
plt.title('Metode Elbow untuk Menentukan K Optimal')
plt.grid(True)
plt.show()


kmeans = KMeans(n_clusters=3, init='random', n_init=10, random_state=42)
kmeans.fit(X_scaled)

labels = kmeans.labels_
X_result = X.copy()
X_result['Cluster'] = labels

print("Jumlah penguin per cluster:")
print(X_result['Cluster'].value_counts())


plt.figure(figsize=(10, 6))

# Plot data penguin berdasarkan cluster
plt.scatter(X_result['culmen_length_mm'], X_result['culmen_depth_mm'], 
            c=X_result['Cluster'], cmap='viridis', s=50, alpha=0.7, label='Penguin')

plt.xlabel('Panjang Paruh (mm)')
plt.ylabel('Tebal Paruh (mm)')
plt.title('Hasil Clustering Penguin (K=3)')
plt.colorbar(label='Cluster')
plt.show()


# Simpan ke CSV tanpa index
X_result.to_csv('submission_penguin.csv', index=False)
print("Berhasil menyimpan file 'submission_penguin.csv'")

