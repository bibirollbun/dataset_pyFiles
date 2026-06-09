# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# =========================================
#  MEMUAT SEMUA CSV AIRBNB (COLAB VERSION)
# =========================================
import pandas as pd
from pathlib import Path
from IPython.display import display

# 1ï¸�âƒ£ Ganti sesuai lokasi folder hasil unzip
BASE_PATH = Path('/kaggle/input/airbnb-recruiting-new-user-bookings')

# 2ï¸�âƒ£ Enam file CSV yang akan dipelajari
files = {
    'train_users'          : '/kaggle/input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip',
    'test_users'           : '/kaggle/input/airbnb-recruiting-new-user-bookings/test_users.csv.zip',
    'sessions'             : '/kaggle/input/airbnb-recruiting-new-user-bookings/sessions.csv.zip',
    'countries'            : '/kaggle/input/airbnb-recruiting-new-user-bookings/countries.csv.zip',
    'age_gender_bkts'      : '/kaggle/input/airbnb-recruiting-new-user-bookings/age_gender_bkts.csv.zip',
    'sample_submission_NDF': '/kaggle/input/airbnb-recruiting-new-user-bookings/sample_submission_NDF.csv.zip'
}

# 3ï¸�âƒ£ Fungsi pemuat CSV
def load_csv(csv_path):
    """Muat satu berkas CSV dan kembalikan DataFrame."""
    return pd.read_csv(csv_path)

# 4ï¸�âƒ£ Loop: tampilkan ringkasan tiap DataFrame
for name, fname in files.items():
    print(f"\n========== {name.upper()} ==========")
    df = load_csv(BASE_PATH / fname)

    # Ukuran & tipe data
    print("ğŸ“Œ Ukuran data :", df.shape)
    print("ğŸ“Œ Tipe data   :\n", df.dtypes)

    # Lihat 5 baris pertama
    print("ğŸ“Œ 5 baris pertama:")
    display(df.head())



# Asumsikan df_train sudah diload sebelumnya
df_train = load_csv(files['train_users'])

# (a) Deskripsi struktur dan dimensi
print("Jumlah baris dan kolom:", df_train.shape)
print("\nTipe data masing-masing kolom:")
print(df_train.dtypes)

print("\nContoh data:")
display(df_train.head())



# Cek missing
missing = df_train.isnull().sum()
print("Missing values:\n", missing[missing > 0])

# Cek duplikasi
print("Duplikasi ID:", df_train.duplicated(subset='id').sum())

# Statistik kolom numerik
display(df_train[['age']].describe())

# Visual outlier usia
import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x=df_train['age'])
plt.title("Boxplot Usia")
plt.show()



# =========================================
# ğŸ“¦ Import Library (pastikan sudah di awal notebook)
import pandas as pd
import numpy as np
from pathlib import Path

# =========================================
# ğŸ“¥ Load Dataset Airbnb
# Corrected path and removed .zip extension
BASE = Path('/kaggle/input/airbnb-recruiting-new-user-bookings/')
df_train = pd.read_csv(BASE / '/kaggle/input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip')

# =========================================
# ğŸ”¹ 1. Tangani Outlier pada Kolom Usia
df_train['age'] = df_train['age'].apply(
    lambda x: x if pd.notnull(x) and 18 <= x <= 100 else np.nan
)

# =========================================
# ğŸ”¹ 2. Imputasi Numerik (Median)
num_median = df_train['age'].median()
df_train['age'].fillna(num_median, inplace=True)

# =========================================
# ğŸ”¹ 3. Imputasi Categorical
cat_cols = ['gender', 'signup_method', 'language', 'affiliate_channel']
df_train[cat_cols] = df_train[cat_cols].fillna('unknown')

# =========================================
# ğŸ”¹ 4. Hapus Duplikasi (jika ada)
df_train.drop_duplicates(inplace=True)

# Cek hasil akhir
print("âœ… Ukuran data setelah pembersihan:", df_train.shape)
df_train[cat_cols + ['age']].head()


# %% â”€â”€â”€ MEMBANGUN df_clean (CLEANED DATAFRAME) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import pandas as pd
import numpy as np  # perlu untuk None â†’ np.nan secara eksplisit (opsional tapi direkomendasikan)

# Salin data mentah
df_clean = df_train.copy()

# 1) Buang pengguna tanpa booking (NDF) agar fokus pada perilaku pemesan
df_clean = df_clean[df_clean['country_destination'] != 'NDF'].copy()

# 2) Validasi & filter usia (15â€“100); sisanya â†’ NaN
df_clean['age'] = df_clean['age'].apply(
    lambda x: x if pd.notnull(x) and 15 <= x <= 100 else np.nan
)

# 3) Imputasi usia dengan median
median_age = df_clean['age'].median()
df_clean['age'].fillna(median_age, inplace=True)

# 4) Imputasi sederhana untuk beberapa kolom kategorikal
cat_missing = ['gender', 'first_browser', 'first_affiliate_tracked']
df_clean[cat_missing] = df_clean[cat_missing].fillna('Unknown')

# 5) Pastikan tidak ada baris duplikat
df_clean.drop_duplicates(inplace=True)

# Cek hasil akhir
print("âœ… df_clean siap dipakai. Dimensi:", df_clean.shape)



from sklearn.preprocessing import LabelEncoder, StandardScaler

df_clean = df_train.copy()

# Drop NDF (tidak booking)
df_clean = df_clean[df_clean['country_destination'] != 'NDF']

# Filter usia
df_clean = df_clean[(df_clean['age'] >= 15) & (df_clean['age'] <= 100)]

# Isi missing
for col in ['gender', 'first_browser', 'first_affiliate_tracked']:
    df_clean[col] = df_clean[col].fillna('Unknown')

# Encoding gender
le_gender = LabelEncoder()
df_clean['gender_enc'] = le_gender.fit_transform(df_clean['gender'])

# Normalisasi usia
scaler = StandardScaler()
df_clean['age_scaled'] = scaler.fit_transform(df_clean[['age']])



# ğŸ“¦ Impor pustaka visualisasi
import seaborn as sns
import matplotlib.pyplot as plt

# Pastikan df_clean sudah ada di memori
# -------------------------------------

# Distribusi usia
sns.histplot(df_clean['age'], bins=30)
plt.title("Distribusi Usia")
plt.show()

# Gender vs Tujuan
plt.figure(figsize=(8,4))
sns.countplot(x='gender', hue='country_destination', data=df_clean)
plt.title("Gender vs Country Destination")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# %% [EDA â€“ Distribusi Usia & Korelasi Fitur]
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Daftar fitur kategorikal yang akan dianalisis
enc_cols = ['gender', 'signup_method', 'language',
            'affiliate_channel', 'first_browser',
            'first_device_type', 'first_affiliate_tracked']

# =====================================
# ğŸ”¹ Distribusi Usia Pengguna
plt.figure(figsize=(6, 4))
sns.histplot(df_clean['age'], bins=30, kde=True)
plt.title("Distribusi Usia Pengguna Airbnb")
plt.xlabel("Usia")
plt.ylabel("Jumlah")
plt.tight_layout()
plt.show()

# =====================================
# ğŸ”¹ Korelasi Usia dengan Fitur Kategorikal (Encoded)

# Siapkan DataFrame baru untuk korelasi
df_corr = df_clean[['age']].copy()

# Konversi semua fitur kategorikal ke kode integer (category â†’ code)
for col in enc_cols:
    df_corr[col] = df_clean[col].astype('category').cat.codes

# Hitung korelasi antar semua fitur
corr = df_corr.corr()

# Plot heatmap korelasi
plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='Blues', square=True, linewidths=0.5)
plt.title("ğŸ”— Matriks Korelasi: Usia + Fitur Kategorikal (Encoded)")
plt.tight_layout()
plt.show()



# ========================================
# ğŸ“¦ Import Library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ========================================
# ğŸ“� Definisikan Path dan File
# BASE_PATH = Path('/kaggle/input/airbnb-recruiting-new-user-bookings') # This path is incorrect
files = {
    'train_users': '/kaggle/input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip', # Corrected file path
}

# ========================================
# ğŸ“¥ Fungsi Load CSV
def load_csv(csv_path):
    # Assuming the file is a regular CSV, not zipped
    return pd.read_csv(csv_path)

# ========================================
# ğŸ“¥ Load Dataset
df_train = load_csv(files['train_users'])

# ========================================
# ğŸ§¹ Data Cleaning Minimal
df_clean = df_train.copy()

# Ubah ke format datetime
df_clean['date_account_created'] = pd.to_datetime(df_clean['date_account_created'], errors='coerce')
df_clean['date_first_booking'] = pd.to_datetime(df_clean['date_first_booking'], errors='coerce')

# Hapus baris yang tidak valid
df_clean = df_clean.dropna(subset=['date_account_created'])

# ========================================
# ğŸ“ˆ (a) Tren Pembuatan Akun per Bulan
df_clean['dac_month'] = df_clean['date_account_created'].dt.to_period('M')

monthly_accounts = (
    df_clean.groupby('dac_month')
            .size()
            .to_timestamp()
            .sort_index()
)

plt.figure(figsize=(10,4))
sns.lineplot(x=monthly_accounts.index, y=monthly_accounts.values)
plt.title("ğŸ“ˆ Tren Jumlah Akun Baru Airbnb per Bulan")
plt.xlabel("Bulan")
plt.ylabel("Jumlah Akun")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ========================================
# ğŸ“Š (b) Tren Booking Pertama (Top 5 Negara)
booked = df_clean.dropna(subset=['date_first_booking']).copy()
top5 = booked['country_destination'].value_counts().head(5).index.tolist()

booked_top = booked[booked['country_destination'].isin(top5)].copy()
booked_top['dfb_month'] = booked_top['date_first_booking'].dt.to_period('M')

monthly_bookings = (
    booked_top.groupby(['dfb_month', 'country_destination'])
              .size()
              .unstack(fill_value=0)
              .to_timestamp()
              .sort_index()
)

plt.figure(figsize=(10,5))
monthly_bookings.plot(ax=plt.gca(), linewidth=2)
plt.title("ğŸ“Š Tren Booking Pertama per Bulan (5 Negara Teratas)")
plt.xlabel("Bulan")
plt.ylabel("Jumlah Booking")
plt.legend(title="Negara")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ========================================
# ğŸ“¦ Import Library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ========================================
# ğŸ“Œ Pastikan df_clean sudah ada dan lengkap
# Jika belum, jalankan pembersihan berikut:
df_clean = df_train.copy()

# Tambah kolom penting & drop baris tidak lengkap
df_clean = df_clean[
    (df_clean['age'] >= 15) & (df_clean['age'] <= 100)
].copy()

for col in ['gender', 'signup_method', 'language', 'affiliate_channel',
            'first_browser', 'first_device_type', 'first_affiliate_tracked']:
    df_clean[col] = df_clean[col].fillna('Unknown')

df_clean = df_clean.dropna(subset=['age', 'signup_flow'])

# ========================================
# 1ï¸�âƒ£ Pilih fitur numerik & kategorikal
num_cols = ['age', 'signup_flow']
cat_cols = ['gender', 'signup_method', 'language',
            'affiliate_channel', 'first_browser',
            'first_device_type', 'first_affiliate_tracked']

features = num_cols + cat_cols

# ========================================
# 2ï¸�âƒ£ Preprocessing Pipeline (Scaling + One-Hot)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

X_proc = preprocessor.fit_transform(df_clean[features])

# ========================================
# 3ï¸�âƒ£ Clustering: MiniBatch KMeans
k = 4
mbk = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024, max_iter=300)
df_clean['cluster'] = mbk.fit_predict(X_proc)

# ========================================
# 4ï¸�âƒ£ Evaluasi Cepat: Silhouette Score
score = silhouette_score(X_proc, df_clean['cluster'], sample_size=10000, random_state=42)
print(f"âœ… Silhouette Score (sample 10k): {score:.3f}")



# ---------------------------------------
# 5. Profil klaster: fitur numerik
# ---------------------------------------
num_summary = df_clean.groupby('cluster')[num_cols].mean().round(2)
print("ğŸ“Œ Rata-rata fitur numerik per klaster:")
display(num_summary)

# ---------------------------------------
# 6. Profil klaster: fitur kategorikal
# ---------------------------------------
def top_categories(df, column, n=3):
    return (
        df.groupby('cluster')[column]
        .value_counts(normalize=True)
        .groupby(level=0)
        .head(n)
        .mul(100).round(1)
        .unstack()
    )

for col in cat_cols:
    print(f"\nğŸ“Œ Distribusi {col.upper()} per Klaster (Top {k})")
    display(top_categories(df_clean, col, n=3))



# ---------------------------------------
# 7. Visualisasi hasil klaster dalam 2D
# ---------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_proc)

df_clean['pca_x'] = X_pca[:, 0]
df_clean['pca_y'] = X_pca[:, 1]

plt.figure(figsize=(7,5))
sns.scatterplot(data=df_clean, x='pca_x', y='pca_y',
                hue='cluster', palette='tab10', s=30)
plt.title("Visualisasi Klaster (PCA 2D)")
plt.xlabel("PC-1")
plt.ylabel("PC-2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()



# ===============================
# ğŸ�·ï¸� Peta nomor klaster â†’ nama segmen
cluster_names = {
    0: 'Young Explorers',
    1: 'Budget Solo',
    2: 'Home Comfort Seekers',
    3: 'Global Frequent Flyers'
}

# ===============================
# ğŸ”� Tambahkan kolom label klaster
df_clean['cluster_label'] = df_clean['cluster'].map(cluster_names)

# Jika ada klaster yang tidak terpetakan, beri label fallback
df_clean['cluster_label'] = df_clean['cluster_label'].fillna(
    df_clean['cluster'].apply(lambda x: f"Cluster_{x}")
)

# ===============================
# ğŸ”� Contoh output
print("Contoh label klaster:")
display(df_clean[['cluster', 'cluster_label']].head())



# ==========================================
# ğŸ“¦ Import Library
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA

# ==========================================
# ğŸ”¹ 0) Load & Clean Data â†’ df_clean
BASE = Path('/kaggle/input/airbnb-recruiting-new-user-bookings')
df_train = pd.read_csv(BASE / '/kaggle/input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip')

df_clean = (
    df_train
      # a. Hanya yang melakukan booking (bukan NDF)
      .loc[df_train['country_destination'] != 'NDF']
      # b. Validasi usia antara 15â€“100
      .assign(age=lambda d: d['age'].where(d['age'].between(15, 100)))
)

# Imputasi
df_clean['age'].fillna(df_clean['age'].median(), inplace=True)
for col in ['gender', 'first_browser', 'first_affiliate_tracked']:
    df_clean[col] = df_clean[col].fillna('Unknown')
df_clean.drop_duplicates(inplace=True)

print("âœ… df_clean shape:", df_clean.shape)

# ==========================================
# ğŸ”¹ 1) Preprocessing: Scaling & Encoding
num_cols = ['age', 'signup_flow']
cat_cols = ['gender', 'signup_method', 'language',
            'affiliate_channel', 'first_browser',
            'first_device_type', 'first_affiliate_tracked']
features = num_cols + cat_cols

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

X_proc = preprocessor.fit_transform(df_clean[features])

# ==========================================
# ğŸ”¹ 2) Clustering: MiniBatchKMeans (k=4)
k = 4
mbk = MiniBatchKMeans(n_clusters=k, random_state=42,
                      batch_size=1024, max_iter=300, n_init=10)
labels = mbk.fit_predict(X_proc)
df_clean['cluster'] = labels

# ==========================================
# ğŸ”¹ 3) Evaluasi Silhouette Score
sil_global = silhouette_score(
    X_proc, labels,
    sample_size=min(10000, X_proc.shape[0]),
    random_state=42
)
print(f"\nâœ… Global Silhouette Score: {sil_global:.3f}")

# Per sampel
sample_sil_vals = silhouette_samples(X_proc, labels)
df_clean['sil_val'] = sample_sil_vals

print("\nğŸ“Š Silhouette Rata-rata per Klaster:")
display(df_clean.groupby('cluster')['sil_val'].mean().round(3))

# ==========================================
# ğŸ”¹ 4) Silhouette Plot per Klaster
plt.figure(figsize=(6,4))
y_lower = 10
for i in range(k):
    vals = sample_sil_vals[labels == i]
    vals.sort()
    size = len(vals)
    y_upper = y_lower + size
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.7)
    plt.text(-0.05, y_lower + size / 2, f'Cluster {i}')
    y_lower = y_upper + 10

plt.axvline(x=sil_global, color='red', linestyle='--')
plt.xlabel("Nilai Silhouette")
plt.ylabel("Index Sampel")
plt.title("ğŸ“ˆ Silhouette Plot per Klaster")
plt.tight_layout()
plt.show()

# ==========================================
# ğŸ”¹ 5) PCA untuk Visualisasi 2D
coords = PCA(n_components=2, random_state=42).fit_transform(X_proc)
plt.figure(figsize=(6,5))
sns.scatterplot(x=coords[:,0], y=coords[:,1],
                hue=labels, palette='tab10', s=25)
plt.title("ğŸ§© Visualisasi Klaster (PCA 2D)")
plt.xlabel("PCâ€‘1")
plt.ylabel("PCâ€‘2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()



# ======================================================
# ğŸ“‹ Fungsi Laporan Lengkap Hasil Klastering
# ======================================================
def full_cluster_report(df, sil_global, sil_per, k_best=2):
    """
    Cetak Insight Evaluasi Klastering, Insight per Klaster (2 terbaik + 1 terburuk),
    Rekomendasi, dan Kesimpulan berbasis Silhouette Score.
    
    Parameters:
    - df : DataFrame dengan kolom 'cluster'
    - sil_global : float, silhouette score keseluruhan
    - sil_per : pd.Series, silhouette rata-rata per cluster (index: cluster)
    - k_best : int, jumlah klaster terbaik yang ingin ditampilkan
    """
    # Validasi input
    assert isinstance(sil_per, pd.Series), "sil_per harus pd.Series"
    assert 'cluster' in df.columns, "'cluster' tidak ditemukan di DataFrame"

    # Urutkan silhouette per klaster
    ranked = sil_per.sort_values(ascending=False)
    top_clusters = ranked.head(k_best).index.tolist()
    worst_cluster = ranked.idxmin()

    # -----------------------------------
    # 1) Insight Evaluasi Klastering
    print("="*75)
    print("ğŸ“Š INSIGHT EVALUASI KLASTERING")
    print(f"â€¢ Global Silhouette Score (k={df['cluster'].nunique()}): {sil_global:.3f}")
    print("â€¢ Rata-rata silhouette per klaster:")
    for c, s in sil_per.items():
        print(f"   - Cluster {c}: {s:.3f}")

    # -----------------------------------
    # 2) Insight Klaster (terbaik & terburuk)
    print("\nğŸ“Œ INSIGHT PER KLASTER")
    for i, c in enumerate(top_clusters, 1):
        size = df[df['cluster'] == c].shape[0]
        print(f"  {i}. Cluster {c} â†’ silhouette {sil_per[c]:.3f} "
              f"({size:,} pengguna) â€” segmen paling terdefinisi.")

    size_worst = df[df['cluster'] == worst_cluster].shape[0]
    print(f"  {i+1}. Cluster {worst_cluster} â†’ silhouette {sil_per[worst_cluster]:.3f} "
          f"({size_worst:,} pengguna) â€” segmen paling tumpang-tindih.\n")

    # -----------------------------------
    # 3) Rekomendasi
    print("ğŸ’¡ REKOMENDASI")
    print("â€¢ Uji jumlah klaster alternatif (mis. k = 3 atau 5) dan bandingkan Silhouette Score.")
    print("â€¢ Tambahkan fitur perilaku seperti frekuensi sesi, waktu booking, atau asal negara.")
    print("â€¢ Gunakan teknik reduksi dimensi non-linear seperti t-SNE atau UMAP.")
    print("â€¢ Eksplorasi algoritme lain: DBSCAN, Hierarchical Clustering.\n")

    # -----------------------------------
    # 4) Kesimpulan
    print("ğŸ“� KESIMPULAN")
    print("â€¢ Segmentasi saat ini belum optimal (silhouette global masih rendah).")
    print("â€¢ Hanya sebagian klaster yang terdefinisi dengan baik; sisanya cenderung tumpang tindih.")
    print("â€¢ Diperlukan optimasi fitur dan jumlah klaster untuk mendapatkan segmentasi yang bermakna.")
    print("â€¢ Klaster yang jelas akan membantu personalisasi layanan & strategi pemasaran Airbnb.")
    print("="*75)

# ======================================================
# ğŸ“Š Jalankan Fungsi Laporan
sil_per_cluster = df_clean.groupby('cluster')['sil_val'].mean().round(3)

full_cluster_report(
    df=df_clean,
    sil_global=sil_global,       # sudah dihitung sebelumnya
    sil_per=sil_per_cluster
)


