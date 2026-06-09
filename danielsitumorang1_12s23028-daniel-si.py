Nim  = "12S23028"
Nama = "Daniel Situmorang"


# Cell 1: Setup dan Import Libraries
# Import library yang diperlukan untuk K-Means clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Import K-Means dan preprocessing tools dari scikit-learn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Import berbagai metrik evaluasi untuk clustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')


# Cell 2: Load dan Eksplorasi Data
# Load dataset penguins dari kompetisi Kaggle
df = pd.read_csv('/kaggle/input/penguin-clustering-analysis/penguins.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Tampilkan sample data untuk memahami struktur dataset
print("\nSample data:")
print(df.head())

# Info missing values
print("\nMissing values:")
print(df.isnull().sum())


# Cell 3: Pemilihan Fitur dan Data Cleaning
# Memilih 4 fitur morfologi yang relevan untuk clustering penguin:
# - culmen_length_mm: panjang paruh (indikator feeding behavior)
# - culmen_depth_mm: kedalaman paruh (indikator diet specialization) 
# - flipper_length_mm: panjang sirip (indikator swimming ability)
# - body_mass_g: massa tubuh (indikator ukuran keseluruhan)
selected_features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']

# Data cleaning: hapus missing values untuk analisis clustering yang akurat
df_clean = df[selected_features].dropna()
print(f"Data: {len(df)} → {len(df_clean)} (after cleaning)")

# Visualisasi distribusi fitur untuk memahami karakteristik data
# Penting untuk identifikasi outliers dan normalitas distribusi
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Distribusi Fitur Penguin', fontsize=14)

for i, col in enumerate(selected_features):
    ax = axes[i//2, i%2]
    ax.hist(df_clean[col], bins=20, alpha=0.7, edgecolor='black')
    ax.set_title(col)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Analisis korelasi antar fitur untuk memahami hubungan variabel
# Korelasi tinggi dapat mempengaruhi hasil clustering
plt.figure(figsize=(8, 6))
correlation_matrix = df_clean.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f')
plt.title('Matriks Korelasi Fitur')
plt.show()


# Cell 4: Standardisasi Data
# PREPROCESSING DATA (persiapan untuk K-Means)
# Standardisasi sangat penting untuk K-Means karena:
# 1. Algoritma K-Means sensitif terhadap skala data
# 2. Fitur dengan nilai besar (body_mass_g) bisa mendominasi
# 3. Standardisasi membuat semua fitur memiliki kontribusi yang sama
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)
X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features)

# Visualisasi perbandingan before vs after standardization
# Menunjukkan efek standardisasi pada distribusi data
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Perbandingan Sebelum vs Sesudah Standardisasi', fontsize=14)

for i, col in enumerate(selected_features):
    ax = axes[i//2, i%2]
    # Data asli (skala berbeda-beda)
    ax.hist(df_clean[col], alpha=0.6, label='Sebelum', bins=20, color='red')
    # Data setelah standardisasi (mean=0, std=1)
    ax.hist(X_scaled_df[col], alpha=0.6, label='Sesudah', bins=20, color='blue')
    ax.set_title(col)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Cell 5: Menentukan Jumlah Klaster Optimal
# Menggunakan multiple metode untuk menentukan jumlah cluster terbaik:
# 1. Elbow Method - mencari "siku" pada grafik WCSS vs k
# 2. Silhouette Coefficient - mengukur kualitas clustering
# 3. Calinski-Harabasz Index - rasio dispersi antar vs dalam cluster
k_range = range(2, 11)
inertias = []  # Within-Cluster Sum of Squares (WCSS)
silhouette_scores = []  # Silhouette coefficient untuk setiap k
calinski_scores = []   # Calinski-Harabasz index untuk setiap k

# Loop untuk menguji berbagai nilai k dan menghitung metrik evaluasi
for k in k_range:
    # Fitting K-Means untuk nilai k tertentu
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Hitung berbagai metrik untuk evaluasi k optimal
    inertias.append(kmeans.inertia_)  # WCSS (semakin kecil semakin baik)
    silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))  # Range [-1,1], semakin tinggi semakin baik
    calinski_scores.append(calinski_harabasz_score(X_scaled, cluster_labels))  # Semakin tinggi semakin baik

# Visualisasi hasil evaluasi untuk menentukan k optimal
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Elbow Method - cari titik "siku" untuk k optimal
ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('k')
ax1.set_ylabel('WCSS (Inertia)')
ax1.set_title('Elbow Method')
ax1.grid(True)

# Plot 2: Silhouette Analysis - k dengan silhouette score tertinggi
ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('k')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')
ax2.grid(True)

# Identifikasi dan tandai k optimal berdasarkan silhouette score
best_k = k_range[np.argmax(silhouette_scores)]
ax2.scatter([best_k], [max(silhouette_scores)], color='red', s=100, zorder=5)
ax2.annotate(f'Best k = {best_k}', xy=(best_k, max(silhouette_scores)), 
            xytext=(best_k+0.5, max(silhouette_scores)-0.02),
            arrowprops=dict(arrowstyle='->', color='red'))

# Plot 3: Calinski-Harabasz Index - validasi tambahan untuk k optimal
ax3.plot(k_range, calinski_scores, 'go-', linewidth=2, markersize=8)
ax3.set_xlabel('k')
ax3.set_ylabel('Calinski-Harabasz Index')
ax3.set_title('Calinski-Harabasz Index')
ax3.grid(True)

plt.tight_layout()
plt.show()

# Tentukan k optimal berdasarkan analisis di atas
optimal_k = best_k
print(f"K optimal: {optimal_k}")
print(f"Silhouette Score: {max(silhouette_scores):.3f}")

# Tabel ringkasan untuk dokumentasi dan analisis
print(f"\nTabel evaluasi k:")
print(f"{'k':<3} | {'WCSS':<8} | {'Silhouette':<10}")
print("-"*25)
for i, k in enumerate(k_range):
    print(f"{k:<3} | {inertias[i]:<8.1f} | {silhouette_scores[i]:<10.3f}")


# Cell 6: Implementasi K-Means Final
# Menggunakan k optimal yang telah ditentukan dari analisis sebelumnya
# Parameter:
# - n_clusters: jumlah cluster optimal dari analisis
# - random_state: untuk reproducible results  
# - n_init: jumlah inisialisasi untuk hasil yang stabil
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Analisis distribusi hasil clustering
# Penting untuk memahami seberapa seimbang pembagian cluster
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()

print(f"Distribusi cluster:")
for i, count in enumerate(cluster_counts):
    print(f"Cluster {i}: {count} data ({count/len(cluster_labels)*100:.1f}%)")

# Tambahkan hasil clustering ke dataframe untuk analisis lanjutan
# Ini memungkinkan analisis karakteristik setiap cluster
df_with_clusters = df_clean.copy()
df_with_clusters['Cluster'] = cluster_labels


# Cell 7: Visualisasi Hasil Clustering
# VISUALISASI HASIL
# PCA untuk visualisasi 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(15, 5))

# Plot 1: Hasil clustering
plt.subplot(1, 3, 1)
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i in range(optimal_k):
    mask = cluster_labels == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=colors[i], label=f'Klaster {i}', alpha=0.7, s=50)

# Centroids
centroids_pca = pca.transform(kmeans_final.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
           c='black', marker='X', s=200, linewidths=3, label='Centroids')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('K-Means Clustering (PCA)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Distribusi cluster
plt.subplot(1, 3, 2)
cluster_counts.plot(kind='bar', color=colors[:optimal_k])
plt.title('Distribusi Klaster')
plt.xlabel('Klaster')
plt.ylabel('Jumlah Data')
plt.xticks(rotation=0)

# Plot 3: Feature comparison
plt.subplot(1, 3, 3)
for cluster in range(optimal_k):
    mask = cluster_labels == cluster
    plt.scatter(df_clean.iloc[mask, 0], df_clean.iloc[mask, 2], 
               c=colors[cluster], label=f'Klaster {cluster}', alpha=0.7)

plt.xlabel('Culmen Length (mm)')
plt.ylabel('Flipper Length (mm)')
plt.title('Culmen vs Flipper Length')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Cell 8: Evaluasi Model
# Menggunakan multiple metrics untuk evaluasi comprehensive:

# Internal validation metrics (tidak butuh ground truth)
wcss = kmeans_final.inertia_  # Within-Cluster Sum of Squares
sil_score = silhouette_score(X_scaled, cluster_labels)  # Silhouette coefficient
ch_score = calinski_harabasz_score(X_scaled, cluster_labels)  # Calinski-Harabasz index
db_score = davies_bouldin_score(X_scaled, cluster_labels)  # Davies-Bouldin index

# Tampilkan hasil evaluasi dengan interpretasi
print(f"WCSS: {wcss:.2f}")  # Semakin kecil semakin baik (kompaknya cluster)
print(f"Silhouette Score: {sil_score:.3f}")  # Range [-1,1], >0.5 = good clustering
print(f"Calinski-Harabasz: {ch_score:.2f}")  # Semakin tinggi semakin baik (separasi cluster)
print(f"Davies-Bouldin: {db_score:.3f}")  # Semakin rendah semakin baik (kompaknya cluster)

# Interpretasi kualitas clustering berdasarkan silhouette score
if sil_score > 0.5:
    quality = "BAIK"  # Cluster well-separated
elif sil_score > 0.25:
    quality = "SEDANG"  # Reasonable clustering  
else:
    quality = "KURANG"  # Poor clustering with overlaps

print(f"Kualitas Clustering: {quality}")

# Silhouette analysis per sample untuk analisis detail
# Menunjukkan distribusi silhouette coefficient dalam setiap cluster
from sklearn.metrics import silhouette_samples
plt.figure(figsize=(8, 6))

sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)
y_lower = 10

# Plot silhouette untuk setiap cluster
for i in range(optimal_k):
    # Ambil silhouette values untuk cluster i
    cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    cluster_silhouette_values.sort()
    
    size_cluster_i = cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = colors[i]
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values,
                     facecolor=color, edgecolor=color, alpha=0.7)
    
    # Label cluster di tengah-tengah
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

# Garis rata-rata silhouette score untuk referensi
plt.axvline(x=sil_score, color="red", linestyle="--", 
           label=f'Avg: {sil_score:.3f}')
plt.xlabel('Silhouette Coefficient')
plt.ylabel('Cluster Label')
plt.title('Silhouette Analysis')
plt.legend()
plt.show()


# Cell 9: Analisis Karakteristik Klaster
# ANALISIS KARAKTERISTIK SETIAP KLASTER (interpretasi mendalam)
# Menganalisis profil morfologi setiap cluster untuk memahami pola

# Hitung rata-rata setiap fitur per cluster
cluster_profiles = df_with_clusters.groupby('Cluster')[selected_features].mean().round(2)
cluster_profiles  # Tampilkan tabel karakteristik

# Visualisasi karakteristik relatif setiap cluster
# Normalisasi terhadap overall mean untuk melihat perbedaan relatif
plt.figure(figsize=(10, 6))
overall_means = df_clean[selected_features].mean()
normalized_means = cluster_profiles / overall_means  # Nilai >1 = above average, <1 = below average

sns.heatmap(normalized_means.T, 
            annot=True, 
            fmt='.2f',
            cmap='RdYlBu_r',
            center=1,  # Center pada nilai 1 (rata-rata)
            cbar_kws={'label': 'Relative to Overall Mean'})
plt.title('Karakteristik Relatif Setiap Klaster')
plt.xlabel('Klaster')
plt.ylabel('Fitur')
plt.show()

# Interpretasi detail karakteristik setiap cluster
print("\nInterpretasi karakteristik:")
for i in range(optimal_k):
    cluster_data = df_with_clusters[df_with_clusters['Cluster'] == i]
    cluster_means = cluster_data[selected_features].mean()
    
    print(f"\nCluster {i} ({len(cluster_data)} penguin):")
    
    # Analisis setiap fitur relatif terhadap rata-rata keseluruhan
    for feature in selected_features:
        mean_val = cluster_means[feature]
        overall_mean = overall_means[feature]
        
        # Klasifikasi ukuran berdasarkan perbandingan dengan rata-rata
        if mean_val > overall_mean * 1.1:  # 10% di atas rata-rata
            status = "BESAR"
        elif mean_val < overall_mean * 0.9:  # 10% di bawah rata-rata
            status = "KECIL"
        else:
            status = "SEDANG"
            
        print(f"   {feature}: {mean_val:.1f} ({status})")
    
    # Kategori penguin berdasarkan massa tubuh sebagai indikator utama
    body_mass = cluster_means['body_mass_g']
    if body_mass > overall_means['body_mass_g'] * 1.15:
        category = "Penguin BESAR"  # Heavy penguins
    elif body_mass < overall_means['body_mass_g'] * 0.85:
        category = "Penguin KECIL"  # Light penguins
    else:
        category = "Penguin SEDANG"  # Medium-sized penguins
    
    print(f"   Kategori: {category}")


# Export dataset dengan cluster assignments untuk dokumentasi
df_with_clusters.to_csv('penguins_clustered.csv', index=False)

# Summary hasil clustering
print(f"K optimal: {optimal_k}, Silhouette: {sil_score:.3f}")
print("File saved: penguins_clustered.csv")


