NIM = "12S23035"
Nama = "Julius Sinaga"


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Agar plot muncul di notebook
%matplotlib inline


# Membaca dataset
data = pd.read_csv('/kaggle/input/penguin/penguins.csv')
data.head()


# Mengisi Missing Values
# A. Isi kolom angka dengan Rata-rata (Mean)
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# B. Isi kolom teks/kategori dengan Modus (Nilai terbanyak)
cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Simpan ke variabel data_clean agar cell di bawahnya tetap jalan
data_clean = data.copy()

# Encoding Data
features = data_clean.copy()
features_encoded = pd.get_dummies(features, drop_first=True)
features_encoded.head()


# Melakukan standarisasi (Scaling)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(features_encoded)

# Melihat statistik data setelah discaling
pd.DataFrame(data_scaled, columns=features_encoded.columns).describe()


# Mencari K optimal dengan Elbow Method
sse = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='random', random_state=42)
    kmeans.fit(data_scaled)
    sse.append(kmeans.inertia_)

# Plotting Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, sse, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia (SSE)')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()


# Fit K-Means dengan K=3 (atau sesuaikan dengan hasil Elbow Anda)
optimal_k = 3 
kmeans = KMeans(n_clusters=optimal_k, init='random', random_state=42)

# Melakukan prediksi klaster
clusters = kmeans.fit_predict(data_scaled)

# Menambahkan hasil klaster ke dataframe asli untuk analisis
data_clean['Cluster'] = clusters
print(data_clean['Cluster'].value_counts())


# Visualisasi Klaster
# Kita ambil 2 kolom pertama dari data yang sudah di-encode untuk sumbu x dan y
x_axis = data_scaled[:, 0] # Fitur pertama
y_axis = data_scaled[:, 1] # Fitur kedua

plt.figure(figsize=(10, 6))
plt.scatter(x_axis, y_axis, c=clusters, cmap='viridis', marker='o', edgecolor='k')

# Menandai Centroid
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')

plt.title(f'Visualisasi Klaster Penguin (K={optimal_k})')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')
plt.legend()
plt.show()


# Membuat file submission
submission = pd.DataFrame({
    'id': data_clean.index,
    'cluster': clusters
})

# Menyimpan ke file CSV
submission.to_csv('submission.csv', index=False)

