# Import library yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Set style untuk visualisasi
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

kaggle_path = '/kaggle/input/penguin-clustering-analysis/'


# Load dataset
df = pd.read_csv(kaggle_path + 'penguins.csv')

# Tampilkan informasi dasar dataset
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())


# Membuat copy dataset untuk preprocessing
df_clean = df.copy()

# Menghapus baris dengan missing values
print(f"Jumlah baris sebelum cleaning: {len(df_clean)}")
df_clean = df_clean.dropna()
print(f"Jumlah baris setelah cleaning: {len(df_clean)}")

# Menghapus outlier yang jelas terlihat (nilai negatif dan ekstrem)
# Berdasarkan eksplorasi data, terdapat nilai flipper_length_mm yang negatif
df_clean = df_clean[
    (df_clean['culmen_length_mm'] > 0) & 
    (df_clean['culmen_depth_mm'] > 0) & 
    (df_clean['flipper_length_mm'] > 0) & 
    (df_clean['flipper_length_mm'] < 300) &  # Remove extreme outlier
    (df_clean['body_mass_g'] > 0)
]

print(f"Jumlah baris setelah remove outlier: {len(df_clean)}")

# Encode variabel kategorikal 'sex'
le = LabelEncoder()
df_clean['sex_encoded'] = le.fit_transform(df_clean['sex'])

print("\nEncoding untuk sex:")
for i, label in enumerate(le.classes_):
    print(f"{label}: {i}")

print("\nData setelah preprocessing:")
print(df_clean.head())


# Visualisasi distribusi fitur
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribusi Fitur Pinguin', fontsize=16, fontweight='bold')

features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex_encoded']

for idx, feature in enumerate(features):
    row = idx // 3
    col = idx % 3
    axes[row, col].hist(df_clean[feature], bins=30, edgecolor='black', alpha=0.7)
    axes[row, col].set_title(f'Distribusi {feature}')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')

# Remove empty subplot
fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.show()


# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df_clean[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex_encoded']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, fmt='.2f')
plt.title('Correlation Matrix - Penguin Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nCorrelation Matrix:")
print(correlation_matrix)


# Pairplot untuk melihat hubungan antar fitur
features_for_plot = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
sns.pairplot(df_clean[features_for_plot], diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Pairplot - Penguin Morphological Features', y=1.02, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# Memilih fitur untuk clustering
# Menggunakan semua fitur numerik untuk analisis yang komprehensif
features_for_clustering = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex_encoded']
X = df_clean[features_for_clustering].values

print(f"Shape data untuk clustering: {X.shape}")
print(f"Features yang digunakan: {features_for_clustering}")

# Normalisasi data menggunakan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData setelah normalisasi (5 baris pertama):")
print(X_scaled[:5])
print("\nMean setelah scaling (seharusnya mendekati 0):")
print(X_scaled.mean(axis=0))
print("\nStd setelah scaling (seharusnya mendekati 1):")
print(X_scaled.std(axis=0))


# Metode Elbow untuk menentukan jumlah cluster optimal
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot Elbow Method
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Elbow curve
ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
ax1.set_title('Elbow Method untuk Menentukan Optimal k', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Silhouette Score
ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Score untuk Berbagai k', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nInertia untuk setiap k:")
for k, inertia in zip(K_range, inertias):
    print(f"k={k}: {inertia:.2f}")

print("\nSilhouette Score untuk setiap k:")
for k, score in zip(K_range, silhouette_scores):
    print(f"k={k}: {score:.4f}")

optimal_k = K_range[silhouette_scores.index(max(silhouette_scores))]
print(f"\nOptimal k berdasarkan Silhouette Score: {optimal_k}")


# Implementasi K-Means dengan jumlah cluster optimal
# Berdasarkan analisis, kita akan menggunakan k=3 (umumnya ada 3 spesies pinguinn)
optimal_k = 3

# Inisialisasi dan fit model K-Means
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Tambahkan label cluster ke dataframe
df_clean['cluster'] = cluster_labels

print(f"K-Means Clustering dengan k={optimal_k}")
print(f"\nDistribusi cluster:")
print(df_clean['cluster'].value_counts().sort_index())
print(f"\nPersentase distribusi cluster:")
print(df_clean['cluster'].value_counts(normalize=True).sort_index() * 100)


# Fungsi untuk menghitung Dunn Index
def dunn_index(X, labels):
    """
    Menghitung Dunn Index untuk evaluasi kualitas clustering.
    Dunn Index = min(inter-cluster distance) / max(intra-cluster distance)
    Nilai lebih tinggi menunjukkan clustering yang lebih baik.
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Hitung intra-cluster distances (diameter maksimum dari setiap cluster)
    intra_dists = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            distances = cdist(cluster_points, cluster_points, metric='euclidean')
            intra_dists.append(np.max(distances))
    
    max_intra_dist = np.max(intra_dists) if intra_dists else 1
    
    # Hitung inter-cluster distances (jarak minimum antar cluster)
    inter_dists = []
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i+1:]:
            cluster1 = X[labels == label1]
            cluster2 = X[labels == label2]
            distances = cdist(cluster1, cluster2, metric='euclidean')
            inter_dists.append(np.min(distances))
    
    min_inter_dist = np.min(inter_dists) if inter_dists else 1
    
    return min_inter_dist / max_intra_dist

# Evaluasi menggunakan berbagai metrik
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
dunn_idx = dunn_index(X_scaled, cluster_labels)

print("="*60)
print("EVALUASI MODEL CLUSTERING")
print("="*60)
print(f"\n1. Silhouette Score: {silhouette_avg:.4f}")
print("   Range: [-1, 1], Semakin tinggi semakin baik")
print("   > 0.5: Struktur cluster yang kuat")
print("   0.25-0.5: Struktur cluster yang cukup")
print("   < 0.25: Struktur cluster yang lemah")

print(f"\n2. Davies-Bouldin Index: {davies_bouldin:.4f}")
print("   Semakin rendah semakin baik (minimum 0)")
print("   Mengukur rata-rata similarity antar cluster")

print(f"\n3. Calinski-Harabasz Score: {calinski_harabasz:.2f}")
print("   Semakin tinggi semakin baik")
print("   Ratio variance between-cluster dan within-cluster")

print(f"\n4. Dunn Index: {dunn_idx:.4f}")
print("   Semakin tinggi semakin baik")
print("   Mengukur kompaktness dan separasi cluster")

print(f"\n5. Sum of Squared Errors (Inertia): {kmeans_final.inertia_:.2f}")
print("   Semakin rendah semakin baik")
print("   Total jarak kuadrat dari setiap titik ke centroid cluster-nya")

print("\n" + "="*60)


# Visualisasi cluster dalam 2D menggunakan fitur-fitur penting
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Visualisasi Clustering Pinguinn - Berbagai Perspektif Fitur', 
             fontsize=16, fontweight='bold')

feature_pairs = [
    ('culmen_length_mm', 'culmen_depth_mm'),
    ('flipper_length_mm', 'body_mass_g'),
    ('culmen_length_mm', 'flipper_length_mm'),
    ('culmen_depth_mm', 'body_mass_g'),
    ('culmen_length_mm', 'body_mass_g'),
    ('culmen_depth_mm', 'flipper_length_mm')
]

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for idx, (feat1, feat2) in enumerate(feature_pairs):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    for cluster in range(optimal_k):
        cluster_data = df_clean[df_clean['cluster'] == cluster]
        ax.scatter(cluster_data[feat1], cluster_data[feat2], 
                  c=colors[cluster], label=f'Cluster {cluster}',
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(feat1, fontsize=10)
    ax.set_ylabel(feat2, fontsize=10)
    ax.set_title(f'{feat1} vs {feat2}', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Visualisasi 3D clustering
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(15, 5))

# Plot 1: Culmen dimensions + Flipper length
ax1 = fig.add_subplot(131, projection='3d')
for cluster in range(optimal_k):
    cluster_data = df_clean[df_clean['cluster'] == cluster]
    ax1.scatter(cluster_data['culmen_length_mm'], 
               cluster_data['culmen_depth_mm'],
               cluster_data['flipper_length_mm'],
               c=colors[cluster], label=f'Cluster {cluster}',
               alpha=0.6, s=50)
ax1.set_xlabel('Culmen Length (mm)')
ax1.set_ylabel('Culmen Depth (mm)')
ax1.set_zlabel('Flipper Length (mm)')
ax1.set_title('3D View: Culmen & Flipper', fontweight='bold')
ax1.legend()

# Plot 2: Culmen length + Body mass + Flipper length
ax2 = fig.add_subplot(132, projection='3d')
for cluster in range(optimal_k):
    cluster_data = df_clean[df_clean['cluster'] == cluster]
    ax2.scatter(cluster_data['culmen_length_mm'], 
               cluster_data['body_mass_g'],
               cluster_data['flipper_length_mm'],
               c=colors[cluster], label=f'Cluster {cluster}',
               alpha=0.6, s=50)
ax2.set_xlabel('Culmen Length (mm)')
ax2.set_ylabel('Body Mass (g)')
ax2.set_zlabel('Flipper Length (mm)')
ax2.set_title('3D View: Size Features', fontweight='bold')
ax2.legend()

# Plot 3: All size features
ax3 = fig.add_subplot(133, projection='3d')
for cluster in range(optimal_k):
    cluster_data = df_clean[df_clean['cluster'] == cluster]
    ax3.scatter(cluster_data['flipper_length_mm'], 
               cluster_data['body_mass_g'],
               cluster_data['culmen_depth_mm'],
               c=colors[cluster], label=f'Cluster {cluster}',
               alpha=0.6, s=50)
ax3.set_xlabel('Flipper Length (mm)')
ax3.set_ylabel('Body Mass (g)')
ax3.set_zlabel('Culmen Depth (mm)')
ax3.set_title('3D View: Body Features', fontweight='bold')
ax3.legend()

plt.tight_layout()
plt.show()


# Analisis karakteristik setiap cluster
print("="*80)
print("KARAKTERISTIK SETIAP CLUSTER")
print("="*80)

for cluster in range(optimal_k):
    print(f"\n{'='*80}")
    print(f"CLUSTER {cluster}")
    print(f"{'='*80}")
    cluster_data = df_clean[df_clean['cluster'] == cluster]
    print(f"Jumlah individu: {len(cluster_data)}")
    print(f"\nStatistik Deskriptif:")
    print(cluster_data[features_for_clustering].describe())
    print(f"\nDistribusi Jenis Kelamin:")
    print(cluster_data['sex'].value_counts())


# Box plot untuk perbandingan fitur antar cluster
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Perbandingan Distribusi Fitur Antar Cluster', fontsize=16, fontweight='bold')

numerical_features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']

for idx, feature in enumerate(numerical_features):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    df_clean.boxplot(column=feature, by='cluster', ax=ax)
    ax.set_title(f'Distribusi {feature} per Cluster', fontweight='bold')
    ax.set_xlabel('Cluster')
    ax.set_ylabel(feature)

plt.tight_layout()
plt.show()


print("="*80)
print("INTERPRETASI HASIL CLUSTERING")
print("="*80)

print("\n1. METODE YANG DIGUNAKAN:")
print("   - Algoritma: K-Means Clustering")
print(f"   - Jumlah cluster optimal: {optimal_k}")
print("   - Fitur yang digunakan: culmen_length_mm, culmen_depth_mm,")
print("     flipper_length_mm, body_mass_g, sex_encoded")
print("   - Preprocessing: StandardScaler normalization")

print("\n2. METRIK EVALUASI:")
print(f"   - Silhouette Score: {silhouette_avg:.4f}")
print(f"   - Davies-Bouldin Index: {davies_bouldin:.4f}")
print(f"   - Calinski-Harabasz Score: {calinski_harabasz:.2f}")
print(f"   - Dunn Index: {dunn_idx:.4f}")
print(f"   - Sum of Squared Errors: {kmeans_final.inertia_:.2f}")

print("\n3. METODE PENENTUAN JUMLAH CLUSTER:")
print("   - Elbow Method: Mengidentifikasi 'elbow point' pada grafik inertia")
print("   - Silhouette Analysis: Memilih k dengan silhouette score tertinggi")
print("   - Domain Knowledge: Menggunakan pengetahuan bahwa ada 3 spesies penguin")

print("\n4. INTERPRETASI CLUSTER:")
print("   Berdasarkan analisis karakteristik morfologis, 3 cluster yang teridentifikasi")
print("   kemungkinan merepresentasikan 3 spesies penguin yang berbeda:")
for cluster in range(optimal_k):
    cluster_data = df_clean[df_clean['cluster'] == cluster]
    avg_flipper = cluster_data['flipper_length_mm'].mean()
    avg_culmen_length = cluster_data['culmen_length_mm'].mean()
    avg_culmen_depth = cluster_data['culmen_depth_mm'].mean()
    avg_mass = cluster_data['body_mass_g'].mean()
    
    print(f"\n   Cluster {cluster} ({len(cluster_data)} individu):")
    print(f"   - Rata-rata Flipper Length: {avg_flipper:.2f} mm")
    print(f"   - Rata-rata Culmen Length: {avg_culmen_length:.2f} mm")
    print(f"   - Rata-rata Culmen Depth: {avg_culmen_depth:.2f} mm")
    print(f"   - Rata-rata Body Mass: {avg_mass:.2f} g")

print("\n5. KESIMPULAN:")
print("   - K-Means berhasil mengidentifikasi 3 kelompok penguin yang distinct")
print("   - Clustering menunjukkan separasi yang baik berdasarkan metrik evaluasi")
print("   - Fitur morfologis (terutama flipper length dan culmen dimensions)")
print("     menjadi pembeda utama antar cluster")
print("   - Hasil clustering konsisten dengan ekspektasi 3 spesies penguin")

print("\n" + "="*80)


# Implementasi DBSCAN untuk perbandingan
from sklearn.neighbors import NearestNeighbors

# Menentukan epsilon menggunakan k-distance graph
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, -1], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel('Data Points sorted by distance', fontsize=12)
plt.ylabel('5th Nearest Neighbor Distance', fontsize=12)
plt.title('K-Distance Graph untuk Menentukan Epsilon (DBSCAN)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Implementasi DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Evaluasi DBSCAN
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"\nDBSCAN Results:")
print(f"Estimated number of clusters: {n_clusters_dbscan}")
print(f"Estimated number of noise points: {n_noise}")

if n_clusters_dbscan > 1:
    # Hitung silhouette score hanya untuk non-noise points
    mask = dbscan_labels != -1
    if sum(mask) > 0:
        dbscan_silhouette = silhouette_score(X_scaled[mask], dbscan_labels[mask])
        print(f"DBSCAN Silhouette Score: {dbscan_silhouette:.4f}")
        print(f"\nPerbandingan:")
        print(f"K-Means Silhouette Score: {silhouette_avg:.4f}")
        print(f"DBSCAN Silhouette Score: {dbscan_silhouette:.4f}")
else:
    print("DBSCAN tidak menemukan cukup cluster untuk evaluasi")


# Membuat submission file untuk Kaggle
df_submission = pd.read_csv(kaggle_path + 'penguins.csv')

# Buat submission dataframe dengan cluster assignment
submission_df = pd.DataFrame({
    'id': range(len(df_submission)),
    'cluster': df_clean['cluster'].mode()[0]  # Default cluster
})

# Assign cluster untuk data yang valid (ada di df_clean)
for idx, row in df_submission.iterrows():
    if not pd.isna([row['culmen_length_mm'], row['culmen_depth_mm'], 
                    row['flipper_length_mm'], row['body_mass_g']]).any():
        match = df_clean[
            (df_clean['culmen_length_mm'] == row['culmen_length_mm']) &
            (df_clean['culmen_depth_mm'] == row['culmen_depth_mm']) &
            (df_clean['flipper_length_mm'] == row['flipper_length_mm']) &
            (df_clean['body_mass_g'] == row['body_mass_g'])
        ]
        if len(match) > 0:
            submission_df.loc[idx, 'cluster'] = int(match.iloc[0]['cluster'])

# Simpan ke CSV
submission_df.to_csv('submission.csv', index=False)

print("✅ Submission file berhasil dibuat!")
print(f"\nPreview:\n{submission_df.head(10)}")
print(f"\nTotal: {len(submission_df)} | Distribusi: {submission_df['cluster'].value_counts().sort_index().to_dict()}")

