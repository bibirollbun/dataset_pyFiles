import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import lightgbm as lgb # LightGBM

# --- 1. Pemuatan Data ---
# Koreksi path file sesuai dengan struktur Kaggle
try:
    df_train = pd.read_csv('/kaggle/input/kom-1338-data-mining-prediksi-status-kelulusan/training.csv')
    df_test = pd.read_csv('/kaggle/input/kom-1338-data-mining-prediksi-status-kelulusan/testing.csv')
    sample_submission = pd.read_csv('/kaggle/input/kom-1338-data-mining-prediksi-status-kelulusan/sample_submission.csv')
except FileNotFoundError:
    print("Pastikan file CSV berada di path yang benar atau sesuaikan path jika dijalankan lokal.")
    # Untuk pengujian lokal jika file ada di direktori yang sama:
    # df_train = pd.read_csv('training.csv')
    # df_test = pd.read_csv('testing.csv')
    # sample_submission = pd.read_csv('sample_submission.csv')
    # Jika tetap error, pastikan file CSV ada.
    exit()


print("Data Training:")
print(df_train.head())
print(f"\nUkuran data training: {df_train.shape}")

print("\nData Testing:")
print(df_test.head())
print(f"\nUkuran data testing: {df_test.shape}")

# --- 2. Eksplorasi Data Awal (Ringkas) ---
print("\nInfo Data Training:")
df_train.info()

# Cek nilai null (sepertinya tidak ada berdasarkan deskripsi, tapi baik untuk dicek)
print("\nNilai Null di Training:")
print(df_train.isnull().sum().sum())
print("\nNilai Null di Testing:")
print(df_test.isnull().sum().sum())

# Distribusi Target
print("\nDistribusi Target:")
print(df_train['Target'].value_counts(normalize=True))

# --- 3. Pra-pemrosesan Data ---

# Pisahkan fitur (X) dan target (y)
X_train_full = df_train.drop(['sample_id', 'Target'], axis=1)
y_train_full = df_train['Target']
X_test = df_test.drop('sample_id', axis=1)
test_sample_ids = df_test['sample_id'] # Simpan sample_id untuk submission

# Encoding Target
le = LabelEncoder()
y_train_full_encoded = le.fit_transform(y_train_full)
# Kelas yang di-encode: le.classes_ akan memberitahu kita urutannya

# Identifikasi fitur kategorikal dan numerik
# Berdasarkan deskripsi dan inspeksi nama kolom
# Kolom-kolom pertama cenderung kategorikal, sisanya numerik (seperti unit kurikulum, GDP, dll.)
# Kita perlu hati-hati di sini. Dari dataset paper aslinya, beberapa kolom angka sebenarnya adalah kode kategori.
# Mari kita anggap semua yang bukan float adalah kategori untuk saat ini, atau kita bisa merujuk ke paper
# Untuk pendekatan yang lebih aman, kita bisa lihat jumlah unique values.
# Fitur dengan banyak unique values dan tipe integer/float cenderung numerik.
# Fitur dengan sedikit unique values dan tipe integer cenderung kategorikal.

# Berdasarkan nama dan deskripsi paper, berikut pembagiannya (ini penting!)
categorical_features_names = [
    'Marital status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification', 'Nacionality',
    "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation", 'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International'
]

# Semua fitur yang bukan kategorikal dan bukan 'Target' atau 'sample_id' adalah numerik
numerical_features_names = [col for col in X_train_full.columns if col not in categorical_features_names]

print(f"\nFitur Kategorikal ({len(categorical_features_names)}): {categorical_features_names}")
print(f"Fitur Numerik ({len(numerical_features_names)}): {numerical_features_names}")

# Pastikan semua fitur kategorikal yang teridentifikasi ada di X_train_full.columns
missing_cat_in_df = [f for f in categorical_features_names if f not in X_train_full.columns]
if missing_cat_in_df:
    print(f"WARNING: Fitur kategorikal berikut tidak ditemukan di dataframe: {missing_cat_in_df}")
    # Hapus dari daftar jika memang tidak ada
    categorical_features_names = [f for f in categorical_features_names if f in X_train_full.columns]

# Membuat preprocessor
# Untuk fitur numerik: StandardScaler
# Untuk fitur kategorikal: OneHotEncoder (handle_unknown='ignore' penting untuk test set)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features_names),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_names)
    ],
    remainder='passthrough' # Jika ada kolom yang tidak terdaftar, biarkan saja (sebaiknya tidak ada)
)

# --- 4. Pemilihan Model dan Pembuatan Pipeline ---
# Kita akan menggunakan LightGBM
lgbm_model = lgb.LGBMClassifier(random_state=42)

# Buat pipeline lengkap
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', lgbm_model)])

# --- 5. Pelatihan Model dengan Seluruh Data Training ---
print("\nMelatih model dengan seluruh data training...")
pipeline.fit(X_train_full, y_train_full_encoded)
print("Pelatihan model selesai.")

# --- 6. Prediksi pada Data Test ---
print("\nMembuat prediksi pada data test...")
test_predictions_encoded = pipeline.predict(X_test) # Jika pakai gridsearch, ganti pipeline -> best_pipeline

# Ubah kembali hasil prediksi dari numerik ke label string asli
test_predictions_labels = le.inverse_transform(test_predictions_encoded)
print("Prediksi selesai.")

# --- 7. Pembuatan File Submission ---
submission_df = pd.DataFrame({
    'sample_id': test_sample_ids,
    'Target': test_predictions_labels
})

submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print("\nFile submission.csv telah dibuat.")
print(submission_df.head())

# Verifikasi format submission
print(f"\nUkuran file submission: {submission_df.shape}")
print(f"Contoh format file submission (sample_submission.csv):")
print(sample_submission.head())
print(f"Ukuran sample_submission: {sample_submission.shape}")

if submission_df.shape[0] == sample_submission.shape[0] and submission_df['sample_id'].equals(sample_submission['sample_id']):
    print("\nFormat sample_id pada file submission sudah sesuai dengan sample_submission.csv.")
else:
    print("\nPERINGATAN: Format sample_id pada file submission TIDAK SESUAI dengan sample_submission.csv.")
    print(f"Jumlah baris submission: {submission_df.shape[0]}, Jumlah baris sample: {sample_submission.shape[0]}")

