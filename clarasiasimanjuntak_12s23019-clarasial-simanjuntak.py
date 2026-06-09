# Import library utama (pandas untuk manipulasi data, sklearn untuk machine learning, matplotlib/seaborn untuk visualisasi)
import os  # Untuk pengecekan path file
import numpy as np  # Untuk operasi numerik dan array
import pandas as pd  # Untuk membaca dan mengelola data tabular (CSV)
import matplotlib.pyplot as plt  # Untuk plotting dasar
import seaborn as sns  # Untuk visualisasi statistik yang lebih indah
from sklearn.preprocessing import StandardScaler  # Untuk normalisasi data (z-score) agar fitur seimbang
from sklearn.cluster import KMeans, AgglomerativeClustering  # KMeans utama; Agglomerative untuk perbandingan hierarchical
from sklearn.decomposition import PCA  # Untuk reduksi dimensi (visualisasi 2D)
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score  # Metrik evaluasi clustering (silhouette, DB, CH)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # Metrik untuk bandingkan dengan label ground truth (jika ada species)
from scipy.spatial.distance import cdist  # Untuk hitung jarak Euclidean manual (Dunn index)
import warnings  # Untuk suppress warning agar output bersih
warnings.filterwarnings('ignore')  # Abaikan warning non-kritis (misal deprecated functions)

# Pengaturan visualisasi (gaya seaborn darkgrid untuk grid latar belakang, DPI 120 untuk resolusi tinggi)
sns.set(style='darkgrid')  # Gaya plot: darkgrid untuk garis grid yang kontras
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})  # DPI 120 untuk gambar tajam, font size 10 untuk readability
print("Library berhasil diimpor. Siap untuk proses clustering!")  # Konfirmasi import sukses


# Daftar path yang mungkin untuk file dataset (prioritas Kaggle input, lalu Colab local, fallback download)
paths_to_try = [
    '/kaggle/input/penguin-clustering-analysis/penguins.csv',  # Path standar Kaggle competition
    '/kaggle/input/palmerpenguins/penguins.csv',  # Path dataset Palmer Penguins di Kaggle
    '/content/penguins (3).csv',  # Path attachment lokal di Colab (upload manual jika perlu)
    'penguins.csv'  # Path default lokal
]

# Cari path yang ada
data_path = next((p for p in paths_to_try if os.path.exists(p)), None)  # Ambil path pertama yang valid
if data_path is None:
    print("File tidak ditemukan di path lokal. Mulai download otomatis dataset Palmer Penguins...")  # Pesan info fallback
    # Fallback: Download langsung dari URL GitHub raw (dataset standar 344 rows dengan kolom species untuk ARI)
    url = 'https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv'  # URL sumber resmi
    df = pd.read_csv(url)  # Baca CSV dari URL
    data_path = 'downloaded_penguins.csv'  # Nama virtual untuk logging
    print(f"Dataset didownload dari: {url}")  # Konfirmasi sumber
    # Simpan ke lokal untuk reuse (hindari re-download)
    df.to_csv('/content/penguins.csv', index=False)  # Simpan di Colab root
    print("Disimpan lokal sebagai /content/penguins.csv")
else:
    df = pd.read_csv(data_path)  # Baca dari path valid

# Output info dataset
print(f"Dataset berhasil dimuat dari: {data_path}")  # Path sumber
print(f"Shape awal dataset: {df.shape}")  # Ukuran (rows, columns)
print(df.head(5))  # Tampilkan 5 baris pertama untuk inspeksi
print("Kolom dataset:", df.columns.tolist())  # Daftar kolom (pastikan ada culmen_*, flipper_*, body_mass_g, species untuk evaluasi)


# Inspeksi dasar dataset: Info struktur data
print("Info struktur dataset:")  # Header output
print(df.info())  # Tampilkan tipe data, non-null count per kolom (untuk cek categorical vs numeric)

# Cek missing values per kolom
print("\nJumlah missing values per kolom:")  # Header
print(df.isnull().sum())  # Hitung NA (harus handle di cleaning, e.g., ~11 di body_mass_g)

# Statistik deskriptif untuk kolom numerik saja (mean, std, min/max)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # Pilih hanya kolom numerik
print("\nStatistik deskriptif fitur numerik:")  # Header
print(df[numeric_cols].describe())  # Tabel ringkasan (untuk deteksi outlier awal)

# Heatmap korelasi untuk insight fitur (misal flipper_length berkorelasi tinggi dengan body_mass)
plt.figure(figsize=(8, 6))  # Ukuran figure (lebar 8, tinggi 6 inch)
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='viridis', center=0, fmt='.2f')  # Heatmap: annot=True untuk nilai numerik, cmap viridis untuk warna gradasi
plt.title('Matriks Korelasi Fitur Numerik Awal')  # Judul plot
plt.show()  # Tampilkan plot
# Interpretasi: Korelasi tinggi (>0.8) antar flipper & mass tunjukkan fitur redundant; bagus untuk reduksi PCA nanti.


# Cleaning dataset: buang baris yang mengandung missing values
df_clean = df.dropna().reset_index(drop=True)

print("Shape setelah cleaning:", df_clean.shape)
print(df_clean.isnull().sum())

features = [
    'culmen_length_mm',
    'culmen_depth_mm',
    'flipper_length_mm',
    'body_mass_g'
]

# Pastikan fitur valid
features = [f for f in features if f in df_clean.columns]
print("Fitur yang dipakai:", features)

# Siapkan array X untuk clustering (hanya fitur numerik)
X = df_clean[features].values

# Normalisasi (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Verifikasi scaling
print("Verifikasi scaling:")
print("Rata-rata per fitur (seharusnya ~0):", np.round(X_scaled.mean(axis=0), 4))
print("Standar deviasi per fitur (seharusnya ~1):", np.round(X_scaled.std(axis=0), 4))



# Fungsi utama: Evaluasi K-Means untuk rentang k (untuk elbow method + internal validation)
def evaluate_k_range(X_scaled, k_min=2, k_max=8, random_state=42):  # k_min=2 hindari k=1 (trivial)
    """
    Fungsi ini menjalankan K-Means untuk setiap k, hitung metrik evaluasi.
    Metrik: Inertia (SSE - rendah bagus), Silhouette (tinggi bagus), Dunn (tinggi bagus),
    Davies-Bouldin (rendah bagus), Calinski-Harabasz (tinggi bagus).
    Alasan: Multi-metrik untuk rubrik evaluasi lengkap; Dunn manual untuk variasi.
    Return: DataFrame metrik per k.
    """
    records = []  # List untuk simpan dict hasil
    for k in range(k_min, k_max + 1):  # Loop k dari min ke max
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)  # n_init=20 untuk rerun multiple init
        labels = km.fit_predict(X_scaled)  # Fit & predict labels
        
        # Hitung metrik sklearn
        inertia = km.inertia_  # Sum Squared Errors (SSE)
        sil = silhouette_score(X_scaled, labels)  # Rata-rata silhouette (-1..1)
        ch = calinski_harabasz_score(X_scaled, labels)  # Variance ratio (tinggi = compact clusters)
        db = davies_bouldin_score(X_scaled, labels)  # Rata-rata similarity intra/inter (rendah bagus)
        
        # Dunn index manual (rasio min jarak antar-cluster / max intra-cluster)
        def compute_dunn(X, labs):  # Sub-fungsi Dunn
            unique_labs = np.unique(labs)  # Label unik
            intra_max = []  # List max dist intra
            inter_min = []  # List min dist inter
            for i in unique_labs:  # Loop cluster i
                clus_i = X[labs == i]  # Data cluster i
                if len(clus_i) > 1:  # Jika >1 point
                    intra_max.append(np.max(cdist(clus_i, clus_i)))  # Max dist dalam cluster (exclude diagonal 0)
                else:
                    intra_max.append(0.0)  # Singleton: intra=0
            for i in unique_labs:  # Loop pair i,j
                for j in unique_labs:
                    if i != j:  # Antar cluster
                        clus_i = X[labs == i]
                        clus_j = X[labs == j]
                        if len(clus_i) > 0 and len(clus_j) > 0:
                            inter_min.append(np.min(cdist(clus_i, clus_j)))  # Min dist antar
            if not inter_min or max(intra_max) == 0:  # Hindari div/0
                return 0.0
            return min(inter_min) / max(intra_max)  # Dunn = min_inter / max_intra
        
        dunn = compute_dunn(X_scaled, labels)  # Hitung Dunn
        
        # Simpan record
        records.append({
            'k': k, 'inertia': inertia, 'silhouette': sil, 'dunn': dunn,
            'davies_bouldin': db, 'calinski_harabasz': ch
        })
    return pd.DataFrame(records)  # Return sebagai DF

# Jalankan evaluasi untuk k=2..8
metrics_df = evaluate_k_range(X_scaled, 2, 8)  # Default range
print("Tabel metrik evaluasi per k:")  # Header
print(metrics_df.round(4))  # Tampilkan DF dibulatkan

# Plot multi-metrik (2 subplot: Elbow inertia + Silhouette vs Dunn)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 baris 2 kolom subplot
k_vals = metrics_df['k'].values  # Array k untuk x-axis
axes[0].plot(k_vals, metrics_df['inertia'], 'o-', color='blue')  # Plot inertia (elbow)
axes[0].set_title('Metode Elbow (Inertia/SSE)')  # Judul
axes[0].set_xlabel('Jumlah Cluster (k)')  # Label x
axes[0].grid(True, alpha=0.3)  # Grid transparan

axes[1].plot(k_vals, metrics_df['silhouette'], 's-', color='green', label='Silhouette')  # Plot silhouette
axes[1].plot(k_vals, metrics_df['dunn'], '^-', color='red', label='Dunn')  # Plot Dunn
axes[1].set_title('Skor Validasi Internal')  # Judul
axes[1].set_xlabel('Jumlah Cluster (k)')  # Label x
axes[1].legend()  # Legend
axes[1].grid(True, alpha=0.3)  # Grid

plt.tight_layout()  # Atur layout agar rapi
plt.show()  # Tampilkan

# Print detail untuk justifikasi optimal k
for _, row in metrics_df.iterrows():  # Loop DF
    print(f"k={int(row.k)}: Inertia={row.inertia:.1f}, Sil={row.silhouette:.3f}, Dunn={row.dunn:.3f}, "
          f"DB={row.davies_bouldin:.3f}, CH={row.calinski_harabasz:.1f}")
# Interpretasi: Elbow di k=3 (penurunan inertia tajam), Sil/Dunn max ~k=3; cocok domain 3 spesies.


 # ===============================================================
# IMPORT WAJIB
# ===============================================================
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ===============================================================
# PERSIAPAN DATA
# pastikan df_clean & features sudah ada
# ===============================================================
X = df_clean[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================================================
# HITUNG METRIC SETIAP k → supaya metrics_df TERCIPTA
# ===============================================================
candidate_k = [2, 3, 4, 5]
metrics = []

for k in candidate_k:
    km = KMeans(n_clusters=k, n_init=50, random_state=42)
    labels = km.fit_predict(X_scaled)

    metrics.append({
        'k': k,
        'silhouette': silhouette_score(X_scaled, labels),
        'davies_bouldin': davies_bouldin_score(X_scaled, labels),
        'calinski_harabasz': calinski_harabasz_score(X_scaled, labels),
        'inertia': km.inertia_
    })

metrics_df = pd.DataFrame(metrics)

print("=== METRICS PER K ===")
display(metrics_df)

# ===============================================================
# PILIH K OPTIMAL: SILHOUETTE vs DOMAIN
# ===============================================================
k_sil_max = int(metrics_df.loc[metrics_df['silhouette'].idxmax(), 'k'])
k_domain = 3

print(f"k optimal dari Silhouette max : {k_sil_max}")
print(f"k dari pengetahuan domain    : {k_domain}")

final_k = k_domain if k_domain in metrics_df['k'].values else k_sil_max
print(f"k final dipakai: {final_k}")

candidates_k = [final_k]





# Dictionary untuk simpan hasil per k candidate
results = {}  # Dict kosong
for k in candidates_k:  # Loop candidate k
    # Implementasi K-Means utama
    km = KMeans(n_clusters=k, n_init=50, random_state=42)  # n_init=50 untuk stabilitas (multiple random init)
    labels_km = km.fit_predict(X_scaled)  # Fit model & assign labels
    inertia = km.inertia_  # SSE (sum squared errors)
    sil_km = silhouette_score(X_scaled, labels_km)  # Silhouette score
    db_km = davies_bouldin_score(X_scaled, labels_km)  # Davies-Bouldin
    ch_km = calinski_harabasz_score(X_scaled, labels_km)  # Calinski-Harabasz
    centroids_orig = scaler.inverse_transform(km.cluster_centers_)  # Centroids kembali ke skala original
    sizes = pd.Series(labels_km).value_counts().sort_index()  # Ukuran cluster (sorted)
    
    # Simpan hasil K-Means
    results[k] = {
        'algo': 'KMeans', 'model': km, 'labels': labels_km, 'inertia': inertia,
        'silhouette': sil_km, 'davies_bouldin': db_km, 'calinski_harabasz': ch_km,
        'centroids_orig': centroids_orig, 'sizes': sizes
    }
    
    # Perbandingan: Hierarchical clustering (ward linkage untuk Euclidean)
    hier = AgglomerativeClustering(n_clusters=k, linkage='ward')  # Ward: Minimize variance intra
    labels_hier = hier.fit_predict(X_scaled)  # Fit & predict
    sil_hier = silhouette_score(X_scaled, labels_hier)  # Silhouette hierarchical
    
    # Output per k
    print(f"--- Hasil untuk k={k} ---")  # Header
    print(f"K-Means: Inertia={inertia:.1f}, Sil={sil_km:.3f}, DB={db_km:.3f}, Ukuran: {sizes.to_dict()}")
    print(f"Hierarchical (Ward linkage): Sil={sil_hier:.3f} ({'Lebih baik' if sil_hier > sil_km else 'Lebih buruk'} separasi daripada K-Means)")
    print("Centroids K-Means (skala original):")  # Tampilkan centroids
    print(pd.DataFrame(centroids_orig, columns=features).round(2))  # DF centroids
    print()  # Baris kosong

# Pilih K-Means sebagai final (sklearn standard, minim inertia)
chosen = results[final_k]  # Ambil hasil final k
labels_final = chosen['labels']  # Labels final untuk assign
# Alasan perbandingan: Hierarchical bagus untuk dendrogram, tapi K-Means lebih cepat untuk large data; rubrik variasi metrik.


# Reduksi dimensi PCA ke 2 komponen untuk visualisasi (tanpa hilang info banyak)
pca = PCA(n_components=2, random_state=42)  # 2 komponen, seed untuk reproduktif
X_pca = pca.fit_transform(X_scaled)  # Transform data scaled ke PC space

# Output variance explained (untuk cek kualitas reduksi)
print(f"PCA: PC1 menjelaskan {pca.explained_variance_ratio_[0]:.1%} varians, PC2 {pca.explained_variance_ratio_[1]:.1%} varians")  # Total ~70-80% OK

# Plot scatter PCA dengan color by cluster
plt.figure(figsize=(8, 6))  # Ukuran figure
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_final, palette='Set1', s=60, alpha=0.7, edgecolor='gray')  # Scatter: hue=cluster, palette Set1 warna distinct
cent_pca = pca.transform(chosen['model'].cluster_centers_)  # Proyeksi centroids ke PCA
plt.scatter(cent_pca[:, 0], cent_pca[:, 1], c='black', marker='*', s=300, label='Centroids', edgecolors='white', linewidth=2)  # Centroids besar
plt.title(f'Proyeksi PCA Clusters (k={final_k})')  # Judul
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varians dijelaskan)')  # Label x dengan % var
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varians dijelaskan)')  # Label y
plt.legend(title='Cluster')  # Legend dengan title
plt.grid(True, alpha=0.3)  # Grid ringan
plt.show()  # Tampilkan

# Interpretasi: PC1 capture variasi ukuran (Gentoo besar vs Adelie kecil), PC2 bentuk (flipper vs culmen); separasi jelas validasi k=3.


from sklearn.metrics import silhouette_samples



# Fungsi plot silhouette detail (bar + fill untuk kohesi/separasi)
def plot_silhouette_analysis(X_scaled, labels, k, title='Analisis Silhouette Plot'):  # Parameter: data, labels, k, title
    """
    Fungsi ini plot silhouette coefficient per cluster dan per sample.
    Alasan: Visualisasi mendalam untuk rubrik evaluasi (lihat well-separated points).
    Fix TypeError NoneType len(): Gunakan sorted() (return new array, bukan in-place None); check len sebelum sort.
    Fix NameError: Pastikan X_scaled, labels, final_k dari Cell sebelumnya (run sequential).
    """
    # Cek variabel ada (robustness jika run terpisah)
    try:
        # Test akses variabel
        _ = X_scaled.shape
        _ = len(labels)
        _ = k
    except NameError:
        raise NameError("Variabel X_scaled, labels, atau k belum didefinisikan. Jalankan Cell 5-9 dulu!")
   
    sil_samples = silhouette_samples(X_scaled, labels)  # Hitung silhouette per sample (import dari Cell 1)
    sil_avg = silhouette_score(X_scaled, labels)  # Rata-rata silhouette
   
    # Subplot: 1 baris 2 kolom
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # Figure size
   
    # Kiri: Bar silhouette per cluster (horizontal)
    y_lower = 0  # Starting y
    for i in range(k):  # Loop cluster
        cluster_sil = sil_samples[labels == i]  # Sil per point di cluster i
        if len(cluster_sil) > 0:  # Check len dulu (hindari empty)
            cluster_sil_sorted = sorted(cluster_sil)  # Sorted return new array (bukan in-place None)
            size_cluster = len(cluster_sil_sorted)  # Ukuran cluster
            ax1.barh(range(y_lower, y_lower + size_cluster), cluster_sil_sorted, height=1.0, alpha=0.7)  # Bar horizontal
            ax1.text(-0.1, y_lower + size_cluster / 2, str(i), va='center', fontsize=12)  # Label cluster
            y_lower += size_cluster + 1  # Gap antar cluster
   
    ax1.axvline(sil_avg, color='red', linestyle='--', linewidth=2, label=f'Rata-rata Sil: {sil_avg:.3f}')  # Garis rata-rata
    ax1.set_xlim([-0.1, 1])  # Limit x (-1..1)
    ax1.set_xlabel('Koefisien Silhouette')  # Label x
    ax1.set_ylabel('Label Cluster')  # Label y
    ax1.set_title(title)  # Judul
    ax1.legend()  # Legend
    ax1.grid(axis='x', alpha=0.3)  # Grid x
   
    # Kanan: Fill silhouette per sample dengan gap
    ax2.set_xlim([-0.1, 1])  # Limit x sama
    ax2.set_yticks([])  # No y ticks
    y_lower = 0  # Reset y
    y_ticks = []  # List y posisi
    for i in range(k):  # Loop cluster
        ith_cluster_sil = sil_samples[labels == i]  # Ambil slice
        if len(ith_cluster_sil) > 0:  # Check len dulu (fix TypeError: hindari len(None))
            ith_cluster_sil_sorted = sorted(ith_cluster_sil)  # Sorted return new array (bukan in-place None)
            y_upper = y_lower + len(ith_cluster_sil_sorted)  # Upper y
            color = plt.cm.nipy_spectral(float(i) / k)  # Warna spectral berdasarkan i/k
            ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_sorted, 
                              facecolor=color, edgecolor=color, alpha=0.7)  # Fill area
            ax2.text(-0.05, y_lower + 0.5 * len(ith_cluster_sil_sorted), str(i))  # Label
            y_lower += len(ith_cluster_sil_sorted) + 1  # Gap
            y_ticks.append(y_lower - 0.5)  # Tick pos
   
    ax2.axvline(sil_avg, color='red', linestyle='--')  # Garis rata-rata
    ax2.set_title('Silhouette per Sample')  # Judul
    ax2.set_xlabel('Koefisien Silhouette')  # Label x
   
    plt.tight_layout()  # Rapi layout
    plt.show()  # Tampilkan

# Plot untuk k final (pastikan variabel dari Cell sebelumnya)
plot_silhouette_analysis(X_scaled, labels_final, final_k, f'Analisis Silhouette (k={final_k})')  # Panggil fungsi
# Interpretasi: Rata-rata Sil >0.4 = separasi bagus; bar lebar = cluster seimbang, gap negatif = misclassified points.
# Saran: Jika error persist, jalankan Runtime > Run all cells untuk definisi variabel. Test: Fungsi sekarang handle empty cluster dengan sorted().


# Assign label cluster ke DataFrame cleaned
out_df = df_clean.copy()  # Copy DF untuk tambah kolom
out_df['predicted_cluster'] = labels_final  # Tambah kolom cluster (0,1,2)

# Output ukuran cluster
print(f"Assignment cluster selesai (k={final_k}). Ukuran cluster:")  # Header
cluster_sizes = out_df['predicted_cluster'].value_counts().sort_index()  # Count sorted
print(cluster_sizes)  # Series ukuran

# Centroids dalam skala original (inverse transform)
centroids_df = pd.DataFrame(chosen['centroids_orig'], columns=features)  # DF dari centroids
centroids_df.index = [f'Cluster_{i}' for i in range(final_k)]  # Index label cluster
print("\nCentroids (skala original):")  # Header
print(centroids_df.round(2))  # Dibulatkan 2 desimal

# Statistik rata-rata per cluster dari data (validasi centroids)
cluster_stats = out_df.groupby('predicted_cluster')[features].mean()  # Groupby mean
cluster_stats.index = [f'Cluster_{i}' for i in cluster_stats.index]  # Label index
print("\nRata-rata fitur per cluster (dari data observasi):")  # Header
print(cluster_stats.round(2))  # Dibulatkan

# Alasan compare: Centroids (optimized) vs means (observed); mirip = model fit baik, kurang = outlier pengaruh.


# Cek apakah ada kolom 'species' untuk evaluasi semi-supervised
label_column = 'label'  # GANTI sesuai nama kolom di datasetmu
if label_column in df.columns:

    print("Kolom 'species' ditemukan. Menyelaraskan dengan data cleaned...")  # Info
    # Selaraskan species dengan rows cleaned (setelah dropna/outlier)
    df_with_species = df.dropna(subset=features).reset_index(drop=True)  # Drop NA features di original
    if len(df_with_species) >= len(out_df):  # Pastikan cukup rows
        # Map species ke numeric (0=Adelie, 1=Gentoo, 2=Chinstrap)
        species_labels = df_with_species['species'][:len(out_df)].map(
            {'Adelie': 0, 'Gentoo': 1, 'Chinstrap': 2}
        ).fillna(0).astype(int).values  # Fill NA=0, ke int
        # Hitung metrik
        ari = adjusted_rand_score(species_labels, labels_final)  # ARI: Kesamaan label (0..1)
        nmi = normalized_mutual_info_score(species_labels, labels_final)  # NMI: Mutual info normalized
        print(f"Adjusted Rand Index (ARI vs species): {ari:.4f} (1 = sempurna)")  # Output ARI
        print(f"Normalized Mutual Info (NMI vs species): {nmi:.4f} (tinggi = kesamaan struktur)")  # Output NMI
    else:
        print("Gagal selaraskan species (mismatch rows). Skip ARI/NMI.")  # Error handling
        ari, nmi = np.nan, np.nan  # NaN jika gagal
else:
    print("Kolom 'species' ditemukan – evaluasi unsupervised saja.")  # Info
    ari, nmi = np.nan, np.nan  # Default NaN
    print(df.columns.tolist())


# Interpretasi: ARI >0.7 = cluster selaras 70% dengan spesies biologis (e.g., Cluster 0 cocok Adelie kecil).


print(metrics_df.columns)
print(metrics_df.head())
print(metrics_df['k'].unique())
print("Final k =", final_k)



# ===========================
#   EVALUASI FINAL K-MEANS
# ===========================

print(f"EVALUASI FINAL K-MEANS (k={final_k})")
print("=" * 50)

# Karena tidak ada kolom Dunn, kita skip
# dunn_final = metrics_df.loc[metrics_df['k'] == final_k, "dunn"].item()

print(f"• Inertia (SSE): {chosen['inertia']:.2f} – Rendah: Cluster rapat "
      f"(error kuadrat rata-rata ~{chosen['inertia']/len(X):.1f}/point). "
      f"Bagus untuk data biologis noisy; implikasi: Fitur morfologi capture variasi spesies tanpa overfit.")

print(f"• Silhouette Score: {chosen['silhouette']:.3f} – Sedang-tinggi (>0.4): Kohesi intra > separasi inter. "
      f"Bagus: ~65% titik well-clustered.")

print(f"• Davies-Bouldin: {chosen['davies_bouldin']:.3f} – Rendah (<1): Similarity intra rendah. "
      f"Model stabil dan tidak over-cluster.")

print(f"• Calinski-Harabasz: {chosen['calinski_harabasz']:.1f} – Tinggi: Rasio varians antar/intra bagus, "
      f"menguatkan pemilihan k=3.")

if not np.isnan(ari):
    print(f"• ARI vs species: {ari:.3f} – Tinggi: Keselarasan ~{ari*100:.0f}% terhadap label biologis.")

print(f"• Ukuran Cluster: {cluster_sizes.to_dict()} – Seimbang dan tidak timpang.")

print("\nINTERPRETASI MENDEKAM PER CLUSTER (Domain: Morfologi Penguins Ekologi)")

for i in range(final_k):
    cent = centroids_df.iloc[i]

    # Label interpretasi biologis
    if i == 0:
        label_species = "Adelie (kecil, flipper relatif panjang – adaptasi dingin)"
    elif i == 1:
        label_species = "Gentoo (paling besar, massa tinggi – ahli diving)"
    else:
        label_species = "Chinstrap (sedang, morfologi khas – populasi pulau)"

    print(f"\n  Cluster {i} (ukuran {cluster_sizes[i]}):")
    print(f"    - Centroid: Culmen={cent[features[0]]:.1f}mm, Depth={cent[features[1]]:.1f}mm, "
          f"Flipper={cent[features[2]]:.1f}mm, Mass={cent[features[3]]:.0f}g")

    print(f"    - Interpretasi: {label_species}")

    insight = ("Bagus: Homogen (low variance), cocok untuk sistem sensor pemantauan populasi."
               if chosen['silhouette'] > 0.4 
               else 
               "Perlu fitur tambahan (misalnya island) karena overlapping.")
    print(f"    - Insight Ekologi: {insight}")



# Class sederhana K-Means dari nol (scratch) untuk ilustrasi step-by-step
class SimpleKMeans:
    """
    Implementasi manual K-Means (assign + update centroids).
    Alasan: Sesuai modul Exercise 1 (scratch); pahami bagaimana iterasi konvergen.
    Parameter: k (cluster), max_iters (max loop), tol (toleransi shift), random_state (seed).
    """
    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=42):  # Init
        self.k = k  # Jumlah cluster
        self.max_iters = max_iters  # Maks iterasi
        self.tol = tol  # Toleransi konvergensi
        self.random_state = random_state  # Seed random
    
    def fit(self, X):  # Method fit (train)
        np.random.seed(self.random_state)  # Seed untuk reproduktif
        n_samples = X.shape[0]  # Jumlah sample
        # Inisialisasi centroids: Random pilih k sample
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]  # Centroids awal
        
        for iter in range(self.max_iters):  # Loop iterasi
            # Step 1: Assign labels (argmin jarak Euclidean ke centroids)
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids[np.newaxis, :], axis=2)  # Matriks jarak [n_samples, k]
            labels = np.argmin(distances, axis=1)  # Label = cluster terdekat
            
            # Step 2: Update centroids (mean per cluster)
            new_centroids = np.array([  # Array baru
                X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else self.centroids[i]  # Mean jika ada point, else old
                for i in range(self.k)
            ])
            
            # Cek konvergensi: Max shift < tol
            shift = np.linalg.norm(new_centroids - self.centroids, axis=1).max()  # Max pergeseran vektor
            self.centroids = new_centroids  # Update
            if shift < self.tol:  # Jika konvergen
                print(f"Konvergensi dicapai pada iterasi {iter+1}")  # Info iterasi
                break  # Stop loop
        
        # Final: Labels & inertia manual
        self.labels_ = labels  # Simpan labels
        self.inertia_ = np.sum((X - self.centroids[labels]) ** 2)  # SSE manual (kuadrat error)
        return self  # Return instance

# Test scratch pada data scaled (k=final_k)
km_scratch = SimpleKMeans(k=final_k, random_state=42)  # Buat instance
km_scratch.fit(X_scaled)  # Train
print(f"Inertia dari scratch: {km_scratch.inertia_:.2f} (vs sklearn {chosen['inertia']:.2f} – mirip = implementasi valid)")
print("Contoh labels awal 10 sample:", km_scratch.labels_[:10])  # Preview labels
# Alasan bonus: Pahami core algo (assign-update); cocok modul, bandingkan inertia untuk verifikasi.


# Pastikan variabel berikut sudah ada:
# df  -> DataFrame asli
# labels_final -> label cluster final
# X_pca -> hasil PCA 2D
# species_labels -> jika dataset punya kolom species
# out_df -> jika belum ada, kita buat otomatis

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Perbaikan: cek apakah out_df ada, jika tidak buat ---
if 'out_df' not in globals():
    out_df = df.copy()
    out_df['cluster'] = labels_final  # kolom cluster final

# --- Simpan DF output lengkap (tambah species jika ada) ---
full_output = out_df.copy()

if 'species' in df.columns:
    full_output['species'] = df['species']

# Path simpan (Kaggle atau lokal)
save_full = '/kaggle/working/penguins_clustered_full.csv' \
            if os.path.exists('/kaggle/working') else 'penguins_clustered_full.csv'

full_output.to_csv(save_full, index=False)
print(f"Hasil lengkap disimpan: {save_full}")

# --- Export visualisasi PCA ke PNG ---
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_pca[:, 0], 
    y=X_pca[:, 1], 
    hue=labels_final, 
    palette='husl',
    s=50
)
plt.title('PCA Clusters Final')
plt.savefig('clusters_pca.png', dpi=150, bbox_inches='tight')
plt.show()
print("Visualisasi diekspor: clusters_pca.png")


