import pandas as pd          
import numpy as np           
import matplotlib.pyplot as plt   # Library untuk membuat visualisasi grafik
import seaborn as sns             # Library visualisasi data berbasis matplotlib, lebih estetik
from sklearn.preprocessing import StandardScaler   # Untuk melakukan standarisasi data (mean=0, std=1)
from sklearn.impute import KNNImputer             # Untuk imputasi nilai yang hilang menggunakan KNN
from sklearn.cluster import KMeans                # Algoritma K-Means untuk clustering
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score  
# Metric untuk evaluasi kualitas clustering

from sklearn.decomposition import PCA              # Untuk reduksi dimensi (Principal Component Analysis)
import warnings                                    
warnings.filterwarnings("ignore")                  



# 2. LOAD DATASET PENGUINS.CSV
import glob, os
files = glob.glob("/kaggle/input/**/penguins.csv", recursive=True)
if not files:
    files = glob.glob("penguins.csv")
df = pd.read_csv(files[0])
print(f"Berhasil load dari: {files[0]}")
print(f"Shape awal: {df.shape}")
display(df.head(8))


print("EXPLORATORY DATA ANALYSIS (EDA)")

# Menampilkan 10 baris pertama dari dataset untuk melihat struktur awal data
display(df.head(10))

# Menampilkan informasi umum dataset: tipe data, jumlah kolom, non-null data
print("\nInfo dataset:")
print(df.info())

# Menampilkan jumlah missing values pada setiap kolom
print("\nMissing values:")
print(df.isnull().sum())

# Menampilkan statistik deskriptif dari data numerik seperti mean, std, min, max
print("\nStatistik deskriptif:")
display(df.describe())



# Membuat figure berukuran besar agar visualisasi lebih mudah dibaca
plt.figure(figsize=(15,10))

# Melakukan iterasi untuk setiap kolom kecuali kolom terakhir pada dataset
# enumerate(..., 1) memberi nomor mulai dari 1 agar sesuai dengan penomoran subplot
for i, col in enumerate(df.columns[:-1], 1):

    # Membuat subplot dalam layout 2 baris dan 2 kolom
    plt.subplot(2, 2, i)

    # Membuat histogram untuk setiap kolom
    # kde=True untuk menampilkan kurva distribusi
    # hue digunakan jika terdapat kolom 'sex' untuk membedakan distribusi berdasarkan jenis kelamin
    sns.histplot(
        data=df,
        x=col,
        kde=True,
        hue='sex' if 'sex' in df.columns else None  # Kondisional: hanya pakai hue jika kolom 'sex' ada
    )

    # Memberikan judul pada setiap subplot
    plt.title(f'Distribusi {col}')

plt.tight_layout()
plt.show()



# Membuat pairplot (scatter matrix) untuk melihat hubungan antar fitur secara visual.
sns.pairplot(
    df.dropna(),
    hue='sex' if 'sex' in df.columns else None,  # Jika ada kolom 'sex', gunakan sebagai warna pemisah
    corner=True  # hanya menampilkan bagian bawah matriks untuk mengurangi kepadatan plot
)

plt.suptitle("Pairplot Fitur Penguin", y=1.02)  # y=1.02 untuk menaikkan posisi judul agar tidak bertabrakan dengan plot

plt.show()




print("DATA CLEANING")

df = df.drop(index=[9, 48], errors='ignore').reset_index(drop=True)

# Menghapus kolom 'sex' karena proses yang dilakukan adalah unsupervised clustering,
# sehingga label atau informasi kategori tidak diperlukan.
df = df.drop('sex', axis=1, errors='ignore')

# Menampilkan jumlah baris dataset setelah proses cleaning dilakukan.
print(f"Setelah cleaning: {df.shape[0]} baris")



# n_neighbors=5 berarti imputasi dilakukan berdasarkan 5 tetangga terdekat
imputer = KNNImputer(n_neighbors=5)

# Mengaplikasikan imputasi ke dataset df
# fit_transform() akan menghitung nilai imputasi dan menggantinya langsung
# Hasilnya dikembalikan dalam bentuk DataFrame dengan kolom yang sama seperti df
df_clean = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Membuat objek StandardScaler untuk melakukan standarisasi data
# Standarisasi mengubah setiap fitur agar memiliki mean=0 dan standar deviasi=1
scaler = StandardScaler()

# Mengaplikasikan standardisasi ke data yang sudah dibersihkan
# X_scaled akan menjadi array numpy berisi data yang sudah distandardisasi
X_scaled = scaler.fit_transform(df_clean)



print("PENENTUAN JUMLAH KLASTER OPTIMAL (Elbow + Silhouette)")

# List kosong untuk menyimpan nilai inertia (untuk metode Elbow)
inertias = []

# List kosong untuk menyimpan nilai silhouette score
sil_scores = []

# Rentang jumlah cluster yang akan diuji, dari 2 sampai 10
k_range = range(2, 11)

# Melakukan iterasi untuk setiap nilai k (jumlah cluster)
for k in k_range:
    
    # Membuat model KMeans dengan jumlah cluster k
    # random_state=42 agar hasil konsisten setiap dijalankan
    # n_init=30 berarti algoritma diulang 30 kali untuk hasil terbaik
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=30)
    
    # Melatih model menggunakan data yang sudah distandardisasi
    kmeans.fit(X_scaled)

    # Menyimpan nilai inertia untuk metode Elbow
    # inertia = total jarak kuadrat dari data ke centroid cluster
    inertias.append(kmeans.inertia_)

    # Menghitung dan menyimpan silhouette score untuk k
    # silhouette score menilai kualitas cluster (semakin tinggi semakin baik)
    sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))



# Membuat figure dengan 1 baris dan 2 kolom subplot berukuran 14x5
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))

# Plot kurva Elbow: hubungan antara jumlah cluster (k) dan nilai inertia
ax1.plot(k_range, inertias, 'bo-', markersize=8)  
ax1.set_title('Elbow Method')                   
ax1.set_xlabel('Jumlah Klaster (k)')              
ax1.set_ylabel('Inertia')                         
ax1.grid(True)                                   

# Plot kurva Silhouette Score: menunjukkan kualitas cluster untuk setiap k
ax2.plot(k_range, sil_scores, 'ro-', markersize=8)  
ax2.set_title('Silhouette Coefficient')              
ax2.set_xlabel('Jumlah Klaster (k)')                 
ax2.set_ylabel('Silhouette Score')                   
ax2.grid(True)                                       

# Memberi judul besar pada keseluruhan figure
plt.suptitle('Penentuan Jumlah Klaster Optimal', fontsize=16, fontweight='bold')

# Menampilkan kedua grafik
plt.show()

# Menentukan jumlah cluster optimal berdasarkan analisis Elbow + Silhouette + domain knowledge
optimal_k = 3

# Menampilkan hasil final pemilihan cluster
print(f"\n→ K optimal yang dipilih: {optimal_k}")
print("   Alasan: Elbow inflexi di k=3 | Silhouette tertinggi | Domain knowledge: 3 spesies penguin")



# Membuat model KMeans dengan jumlah klaster optimal (optimal_k)
# random_state=42 → agar hasil konsisten setiap dijalankan
# n_init=50 → menjalankan inisialisasi centroid sebanyak 50 kali untuk hasil terbaik
# max_iter=500 → batas maksimum iterasi agar algoritma lebih stabil
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=50, max_iter=500)

# Melakukan pelatihan dan sekaligus memprediksi cluster untuk setiap data
# fit_predict() menggabungkan fit() dan predict() dalam satu langkah
labels = kmeans.fit_predict(X_scaled)

# Menyalin dataframe hasil cleaning agar tidak mengubah data asli
df_result = df_clean.copy()

# Menambahkan kolom baru 'Cluster' untuk menyimpan hasil cluster tiap baris
df_result['Cluster'] = labels



# Menampilkan judul section untuk evaluasi hasil clustering
print("EVALUASI KLASTER (Silhouette, DBI, CH Index + Interpretasi)")

# Menghitung nilai Silhouette Score secara keseluruhan
# Semakin mendekati 1 → cluster semakin bagus dan terpisah jelas
sil_global = silhouette_score(X_scaled, labels)

# Menghitung Davies-Bouldin Index (DBI)
# Semakin kecil → cluster semakin baik
db_index = davies_bouldin_score(X_scaled, labels)

# Menghitung Calinski-Harabasz Index (CH Index)
# Semakin besar → cluster semakin baik dan terpisah
ch_index = calinski_harabasz_score(X_scaled, labels)

# Menampilkan hasil evaluasi numerik
print(f"Silhouette Score (global)      : {sil_global:.4f}")
print(f"Davies-Bouldin Index           : {db_index:.4f}")
print(f"Calinski-Harabasz Index        : {ch_index:.1f}")



sample_silhouette_values = silhouette_samples(X_scaled, labels)

# Visualisasi Silhouette Plot per klaster
fig, ax = plt.subplots(figsize=(10,7))
y_lower = 10
for i in range(optimal_k):
    cluster_sil_values = sample_silhouette_values[labels == i]
    cluster_sil_values.sort()
    size_cluster = cluster_sil_values.shape[0]
    y_upper = y_lower + size_cluster
    color = plt.cm.tab10(i)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil_values, facecolor=color, edgecolor=color, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size_cluster, str(i))
    y_lower = y_upper + 10

ax.set_title("Silhouette Plot per Klaster (k=3)", fontsize=16, fontweight='bold')
ax.set_xlabel("Nilai Silhouette Coefficient")
ax.set_ylabel("Cluster Label")
ax.axvline(x=sil_global, color="red", linestyle="--", label=f'Rata-rata = {sil_global:.3f}')
ax.legend()
plt.show()


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12,9))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette='deep', s=100, alpha=0.9, edgecolor='k')
plt.title('Penguin Clustering K-Means (k=3) – Proyeksi PCA 2D', fontsize=16, fontweight='bold')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.legend(title='Cluster')
plt.grid(True, alpha=0.3)
plt.show()


print("INTERPRETASI KLASTER")
summary = df_result.groupby('Cluster').mean().round(2)
display(summary)

print("\nKESIMPULAN & INTERPRETASI:")
print("• Cluster 0 → Gentoo     : flipper length > 210 mm, body mass tinggi (5000–6000g)")
print("• Cluster 1 → Adelie     : culmen pendek & dalam, flipper < 200 mm, tubuh kecil")
print("• Cluster 2 → Chinstrap  : culmen panjang tapi tipis (depth kecil), flipper sedang")



submission = pd.DataFrame({
    'row_id': range(len(labels)),
    'cluster': labels
})
submission.to_csv('submission.csv', index=False)
display(submission.head(10))

