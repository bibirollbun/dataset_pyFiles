import os
import pandas as pd
import numpy as np
import h5py # Untuk membaca file HDF5
import cv2 # Untuk mendecode byte string gambar dari HDF5
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- 0. PATHS DAN KONSTANTA ---
KAGGLE_INPUT_DIR = '/kaggle/input/isic-2024-challenge'
TRAIN_META_PATH = os.path.join(KAGGLE_INPUT_DIR, 'train-metadata.csv')
TRAIN_IMAGE_HDF5_PATH = os.path.join(KAGGLE_INPUT_DIR, 'train-image.hdf5')
GROUP_COL = 'patient_id'
IMAGE_SIZE = 128 # Ukuran standar untuk EfficientNetB0 (bisa disesuaikan: 224, 256, dst.)


df_train = pd.read_csv(TRAIN_META_PATH)
df_train['target'] = df_train['target'].astype(float)

print("--- Pemeriksaan Dataset: ISIC 2024 Metadata ---")
print(f"Total Sampel Pelatihan (Lesi): {len(df_train)}")
print(f"Total Pasien Unik: {df_train['patient_id'].nunique()}")

# 1. Analisis Ketidakseimbangan Kelas (Class Imbalance)
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df_train)
plt.title('Distribusi Target (Malignant vs. Benign)')
plt.xticks([0, 1], ['0: Benign (Jinak)', '1: Malignant (Ganas)'])
plt.xlabel('Diagnosis')
plt.ylabel('Jumlah Lesi')
plt.show()

malignant_count = df_train['target'].sum()
total_count = len(df_train)
imbalance_ratio = malignant_count / total_count * 100

print(f"\nTotal Kasus Malignant (Target=1): {int(malignant_count):,}")
print(f"Persentase Kasus Malignant: {imbalance_ratio:.2f}%")

# 2. Analisis Distribusi Pasien per Lesi (Grouping)
lesions_per_patient = df_train.groupby('patient_id')['isic_id'].count().sort_values(ascending=False)

print("\n--- Analisis Grouping Pasien ---")
print(f"Lesi Maksimum per Pasien: {lesions_per_patient.max()}")
print(f"Lesi Rata-rata per Pasien: {lesions_per_patient.mean():.2f}")
print("5 Pasien dengan Lesi Terbanyak:")
print(lesions_per_patient.head(5).to_string())

# 3. Distribusi Salah Satu Fitur Numerik Penting (Age)
plt.figure(figsize=(8, 4))
sns.histplot(df_train['age_approx'].dropna(), kde=True, bins=30)
plt.title('Distribusi Usia Pasien')
plt.xlabel('Usia')
plt.ylabel('Frekuensi')
plt.show()

# 4. Korelasi Fitur dengan Target (Menggunakan Fitur Tabular Sederhana)
# Hanya untuk melihat tren, tanpa preprocessing
plt.figure(figsize=(10, 6))
sns.boxplot(x='sex', y='age_approx', hue='target', data=df_train)
plt.title('Usia vs. Jenis Kelamin berdasarkan Target')
plt.xlabel('Jenis Kelamin')
plt.ylabel('Usia')
plt.show()


# --- KONFIGURASI UNTUK SAMPLING ---
# Definisikan CFG (Configuration) untuk memudahkan penyesuaian rasio sampling
class CFG:
    # Perkiraan rasio yang wajar untuk undersampling kelas 0 (Benign)
    # Ini akan mengambil 20% dari sampel Benign (400,000 * 0.2 = 80,000)
    neg_sample = 0.2  
    
    # Perkiraan rasio yang wajar untuk oversampling kelas 1 (Malignant)
    # Ini akan menggandakan sampel Malignant (400 * 200 = 80,000)
    # (Perhatikan: Anda harus menghitung frac agar total kelas 1 menjadi sekitar 80,000)
    # Kita gunakan nilai yang cukup besar untuk oversampling
    pos_sample_multiplier = 200 # Misal, target 80.000 sampel Malignant. Jika ada 393, maka 80000/393 ~ 200
    seed = 42
    
# Hitung rasio oversampling berdasarkan perkiraan jumlah kasus Malignant
malignant_count_approx = df_train.query("target==1").shape[0]
target_pos_samples = df_train.query("target==0").shape[0] * CFG.neg_sample # Target jumlah sampel positif

# Hitung frac untuk oversampling agar jumlah sampel positif mendekati sampel negatif
# Jika hasil perkalian dengan frac melebihi 1.0, berarti replace=True akan digunakan
CFG.pos_sample_frac = target_pos_samples / malignant_count_approx

# --- Terapkan Sampling ---
print("\n==============================================")
print("SAMPLING UNTUK MENGATASI CLASS IMBALANCE")
print("==============================================")

print("Class Distribution Before Sampling (%):")
display(df_train.target.value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

# 1. Undersample kelas 0 (Benign)
positive_df = df_train.query("target==0").sample(frac=CFG.neg_sample, random_state=CFG.seed).reset_index(drop=True)

# 2. Oversample kelas 1 (Malignant) dengan penggantian (replace=True)
# Menggunakan frac=CFG.pos_sample_frac akan mencoba mencapai keseimbangan 1:1
negative_df = df_train.query("target==1").sample(
    frac=CFG.pos_sample_frac, 
    replace=True, 
    random_state=CFG.seed
).reset_index(drop=True)

# 3. Gabungkan dan Acak
df_train_sampled = pd.concat([positive_df, negative_df], axis=0).sample(frac=1.0, random_state=CFG.seed).reset_index(drop=True)

print("\nClass Distribution After Sampling (%):")
display(df_train_sampled.target.value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

# Perbarui df_train ke versi yang sudah di-sampling
df_train = df_train_sampled


# --- PENYESUAIAN SETELAH SAMPLING ---

# Perbarui array target dan groups dari DataFrame yang sudah di-sampling
y_target = df_train['target'].values
groups = df_train[GROUP_COL].values


# 1. Analisis Ketidakseimbangan Kelas (Class Imbalance)
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df_train)
plt.title('Distribusi Target (Malignant vs. Benign)')
plt.xticks([0, 1], ['0: Benign (Jinak)', '1: Malignant (Ganas)'])
plt.xlabel('Diagnosis')
plt.ylabel('Jumlah Lesi')
plt.show()

malignant_count = df_train['target'].sum()
total_count = len(df_train)
imbalance_ratio = malignant_count / total_count * 100

print(f"\nTotal Kasus Malignant (Target=1): {int(malignant_count):,}")
print(f"Persentase Kasus Malignant: {imbalance_ratio:.2f}%")


#MEMILIH FITUR YANG DIMASUKKAN KE MODEL DARI CATBOOST
import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df_train['target'] = df_train['target'].astype(float)

# Hapus kolom yang tidak akan digunakan sebagai fitur (ID, target, diagnosis spesifik)
# Kita akan fokus pada fitur-fitur yang juga ada di test-metadata.csv
cols_to_drop = ['isic_id', 'target', 'lesion_id', 'iddx_full', 'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5', 
                'mel_mitotic_index', 'mel_thick_mm', 'tbp_lv_dnn_lesion_confidence', 
                'attribution', 'copyright_license']
X = df_train.drop(columns=cols_to_drop, errors='ignore')
y = df_train['target']

# Pisahkan fitur numerik dan kategorikal
categorical_features_indices = [i for i, col in enumerate(X.columns) if X[col].dtype == 'object']
numerical_features_indices = [i for i, col in enumerate(X.columns) if X[col].dtype != 'object']

# --- 2. Preprocessing Minimal untuk CatBoost ---

# CatBoost dapat menangani NaN dan fitur kategorikal secara langsung, 
# tetapi kita akan mengisi NaN pada fitur numerik untuk konsistensi.
for i in numerical_features_indices:
    col = X.columns[i]
    X[col].fillna(X[col].median(), inplace=True)

# Ganti NaN di kolom kategorikal dengan 'Unknown'
for i in categorical_features_indices:
    col = X.columns[i]
    X[col].fillna('Unknown', inplace=True)

# Pisahkan data untuk validasi
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- 3. Latih Model CatBoost untuk Mendapatkan Feature Importance ---

# Inisialisasi CatBoost Classifier
model_cb = CatBoostClassifier(
    iterations=100,             # Jumlah iterasi yang rendah (hanya untuk feature selection)
    learning_rate=0.1,
    loss_function='Logloss',    # Cocok untuk klasifikasi biner
    random_seed=42,
    verbose=0,                  # Matikan output pelatihan
    cat_features=X.columns[categorical_features_indices].tolist()
)

print("Memulai pelatihan CatBoost untuk Feature Importance...")
model_cb.fit(X_train, y_train, eval_set=(X_val, y_val))
print("Pelatihan selesai.")


# --- 4. Tampilkan Skor Feature Importance ---

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model_cb.get_feature_importance()
}).sort_values(by='Importance', ascending=False)

print("\n==============================================")
print("TOP FEATURE IMPORTANCE (Metrik Klasifikasi):")
print("==============================================")
print(feature_importances.head(10).to_string(index=False))


# --- 5. Tentukan Fitur Akhir ---

# Fitur yang akan Anda gunakan untuk model EfficientNet Fusion adalah 
# 8-10 fitur teratas dengan skor Importance tertinggi (misalnya, Importance > 2.0)
FINAL_TABULAR_FEATURES = feature_importances[feature_importances['Importance'] > 2.0]['Feature'].tolist()

print("\n==============================================")
print("FITUR TABULAR AKHIR YANG DIREKOMENDASIKAN:")
print("==============================================")
print(FINAL_TABULAR_FEATURES)


# 1. Analisis Ketidakseimbangan Kelas (Class Imbalance)
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df_train)
plt.title('Distribusi Target (Malignant vs. Benign)')
plt.xticks([0, 1], ['0: Benign (Jinak)', '1: Malignant (Ganas)'])
plt.xlabel('Diagnosis')
plt.ylabel('Jumlah Lesi')
plt.show()

malignant_count = df_train['target'].sum()
total_count = len(df_train)
imbalance_ratio = malignant_count / total_count * 100

print(f"\nTotal Kasus Malignant (Target=1): {int(malignant_count):,}")
print(f"Persentase Kasus Malignant: {imbalance_ratio:.2f}%")


# --- DAFTAR FITUR BERDASARKAN HASIL CATBOOST ANDA (Disesuaikan) ---

# Fitur Numerik: Semua fitur tbp_lv_..., clin_size_long_diam_mm, dan age_approx
TOP_NUMERICAL_FEATURES = [
    'clin_size_long_diam_mm',
    'tbp_lv_H',
    'tbp_lv_minorAxisMM',
    'tbp_lv_deltaB',
    'tbp_lv_perimeterMM',
    'tbp_lv_radial_color_std_max',
    'tbp_lv_nevi_confidence',
    'tbp_lv_deltaA',
    'tbp_lv_deltaLBnorm',
    'tbp_lv_stdLExt',
    'tbp_lv_y',
    'tbp_lv_deltaLB',
    'tbp_lv_B',
    'tbp_lv_z',
    'age_approx', 
    # Perhatikan: Fitur seperti 'tbp_lv_norm_color', 'tbp_lv_Hext', dll. dari daftar lama 
    # telah DIHILANGKAN karena tidak ada di daftar baru Anda.
]

# Fitur Kategorikal: Fitur kontekstual pasien (tidak ada di daftar baru Anda, tapi WAJIB ada)
CATEGORICAL_FEATURES = [
    'sex', 
    'anatom_site_general'
]

# patient_id TIDAK DIMASUKKAN di sini. Kolom ini hanya untuk Grouping (K-Fold).
GROUP_COL = 'patient_id'


# Imputasi untuk fitur Numerik: Isi NaN dengan nilai median
for col in TOP_NUMERICAL_FEATURES:
    df_train.loc[:, col] = df_train[col].fillna(df_train[col].median())

# Imputasi untuk fitur Kategorikal: Isi NaN dengan 'unknown'
for col in CATEGORICAL_FEATURES:
    df_train.loc[:, col] = df_train[col].fillna('unknown')


# --- Fitur yang sudah Anda definisikan ---
ALL_CHECK_FEATURES = TOP_NUMERICAL_FEATURES + CATEGORICAL_FEATURES

print("--- VERIFIKASI AKHIR DATA CLEANING ---")

# 1. Pilih hanya kolom yang Anda gunakan
df_check = df_train[ALL_CHECK_FEATURES]

# 2. Hitung jumlah NaN di setiap kolom
missing_values_count = df_check.isnull().sum()

# 3. Filter dan tampilkan hanya kolom yang masih memiliki NaN
remaining_nan = missing_values_count[missing_values_count > 0]

if remaining_nan.empty:
    print("Semua fitur yang digunakan TIDAK memiliki nilai NaN. Data bersih!")
else:
    print("PERINGATAN: Nilai NaN masih tersisa di fitur berikut:")
    print(remaining_nan)


# 1. Analisis Ketidakseimbangan Kelas (Class Imbalance)
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df_train)
plt.title('Distribusi Target (Malignant vs. Benign)')
plt.xticks([0, 1], ['0: Benign (Jinak)', '1: Malignant (Ganas)'])
plt.xlabel('Diagnosis')
plt.ylabel('Jumlah Lesi')
plt.show()

malignant_count = df_train['target'].sum()
total_count = len(df_train)
imbalance_ratio = malignant_count / total_count * 100

print(f"\nTotal Kasus Malignant (Target=1): {int(malignant_count):,}")
print(f"Persentase Kasus Malignant: {imbalance_ratio:.2f}%")


# --- 1. DATA LOADER: CUSTOM KERAS SEQUENCE (MENGATASI HDF5 & RAM LIMIT) ---
class ISICDataGenerator(Sequence):
    """Data Generator untuk memuat gambar dari HDF5 dan metadata tabular batch demi batch."""
    
    def __init__(self, df, tabular_data, hdf5_file_path, image_ids, batch_size=32, shuffle=True):
        self.df = df
        self.tabular_data = tabular_data # Array metadata yang sudah di-encode
        self.image_ids = image_ids      # Daftar isic_id sesuai urutan df/tabular_data
        self.hdf5_file_path = hdf5_file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
        # Buka file HDF5 SEKALI saat inisialisasi
        self.hdf5_file = h5py.File(self.hdf5_file_path, 'r')
        
    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        """Mengatur ulang indeks setelah setiap epoch (jika shuffle=True)."""
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Mendapatkan batch data (Gambar dan Tabular) untuk indeks batch tertentu."""
        # Ambil indeks sampel untuk batch ini
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Inisialisasi array untuk output batch
        X_img_batch = np.empty((self.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        X_tab_batch = np.empty((self.batch_size, self.tabular_data.shape[1]), dtype=np.float32)
        y_batch = np.empty((self.batch_size, 1), dtype=np.float32)

        # Loop untuk memuat data di batch ini
        for i, idx in enumerate(indexes):
            
            # --- 1. Ambil Data Gambar (dari HDF5) ---
            current_isic_id = self.image_ids.iloc[idx]
            
            # Akses byte string gambar menggunakan ISIC ID sebagai key
            byte_string = self.hdf5_file[current_isic_id][()] 
            
            # Decode dan Preprocessing Gambar
            nparr = np.frombuffer(byte_string, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[...,::-1] # BGR to RGB
            
            # Resize dan Normalisasi
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            X_img_batch[i,] = image.astype(np.float32) / 255.0
            
            # --- 2. Ambil Data Tabular ---
            X_tab_batch[i,] = self.tabular_data[idx]
            
            # --- 3. Ambil Target ---
            y_batch[i,] = self.df.loc[idx, 'target']
        
        # Output model Fusion: Dua input, satu output target
        return {'image_input': X_img_batch, 'tabular_input': X_tab_batch}, y_batch

    def __del__(self):
        """Pastikan file HDF5 tertutup saat objek generator dihancurkan."""
        self.hdf5_file.close()
        print("File HDF5 ditutup.")


# --- 2. PREPROCESSING METADATA ---
df_train['target'] = df_train['target'].astype(float)
df_train['isic_id'] = df_train['isic_id'].astype(str) # Pastikan ISIC ID adalah string

# Terapkan Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), TOP_NUMERICAL_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
    ],
    remainder='drop' 
)
X_tabular = preprocessor.fit_transform(df_train)

# Persiapan untuk K-Fold
y_target = df_train['target'].values
groups = df_train[GROUP_COL].values
isic_ids = df_train['isic_id']
TABULAR_SHAPE = (X_tabular.shape[1],)
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

print(f"   Data Tabular Akhir Bersih. Shape: {X_tabular.shape}")


# 1. Analisis Ketidakseimbangan Kelas (Class Imbalance)
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df_train)
plt.title('Distribusi Target (Malignant vs. Benign)')
plt.xticks([0, 1], ['0: Benign (Jinak)', '1: Malignant (Ganas)'])
plt.xlabel('Diagnosis')
plt.ylabel('Jumlah Lesi')
plt.show()

malignant_count = df_train['target'].sum()
total_count = len(df_train)
imbalance_ratio = malignant_count / total_count * 100

print(f"\nTotal Kasus Malignant (Target=1): {int(malignant_count):,}")
print(f"Persentase Kasus Malignant: {imbalance_ratio:.2f}%")


# --- 3. K-FOLD SETUP ---
N_SPLITS = 5 
sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


# --- 4. MODEL DEFINITION: EFFICIENTNET FUSION ---
def create_fusion_model(input_shape_img, input_shape_tabular):
    
    # Jalur 1: Input Gambar (EfficientNet)
    input_img = Input(shape=input_shape_img, name='image_input')
    cnn_backbone = EfficientNetB0(
        include_top=False, weights='imagenet', input_tensor=input_img, pooling='avg'
    )
    cnn_features = cnn_backbone.output 
    
    # Jalur 2: Input Tabular (Metadata)
    input_tabular = Input(shape=input_shape_tabular, name='tabular_input')
    tabular_features = Dense(32, activation='relu')(input_tabular)
    tabular_features = Dropout(0.2)(tabular_features)

    # Fusion
    merged_features = Concatenate()([cnn_features, tabular_features])
    
    # Klasifikasi Akhir
    final_output = Dense(64, activation='relu')(merged_features)
    final_output = Dropout(0.3)(final_output)
    output = Dense(1, activation='sigmoid', name='output')(final_output)
    
    model = Model(inputs=[input_img, input_tabular], outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy']
    )
    return model

print("\n4. MODEL DEFINITION")
# Buat instance model
model = create_fusion_model(IMAGE_SHAPE, TABULAR_SHAPE)
print("   Model EfficientNet Fusion siap dibuat.")


# # --- 5. TRAINING LOOP (ITERASI K-FOLD) ---
# print("\n5. MEMULAI TRAINING LOOP (Grouped K-Fold)")
# EPOCHS = 10  # ubah ke >1 biar kelihatan perubahan AUC
# BATCH_SIZE = 32

# history_all = []  # simpan semua history tiap fold

# for fold, (train_idx, val_idx) in enumerate(sgkf.split(df_train, y_target, groups)):
#     print(f"\n==================== FOLD {fold+1}/{N_SPLITS} ====================")
    
#     df_train_fold = df_train.iloc[train_idx].reset_index(drop=True)
#     X_tab_train_fold = X_tabular[train_idx]
    
#     df_val_fold = df_train.iloc[val_idx].reset_index(drop=True)
#     X_tab_val_fold = X_tabular[val_idx]
    
#     isic_ids_train = df_train_fold['isic_id']
#     isic_ids_val = df_val_fold['isic_id']

#     train_generator = ISICDataGenerator(
#         df_train_fold, X_tab_train_fold, TRAIN_IMAGE_HDF5_PATH, isic_ids_train, BATCH_SIZE
#     )
#     val_generator = ISICDataGenerator(
#         df_val_fold, X_tab_val_fold, TRAIN_IMAGE_HDF5_PATH, isic_ids_val, BATCH_SIZE, shuffle=False
#     )
    
#     history = model.fit(
#         train_generator,
#         epochs=EPOCHS, 
#         validation_data=val_generator,
#         verbose=1
#     )
    
#     history_all.append(history.history)
    
#     val_loss, val_auc, val_acc = model.evaluate(val_generator, verbose=0)
#     print(f"FOLD {fold+1} - Validation AUC: {val_auc:.4f}")



# # Simpan bobot setelah semua fold
# model.save_weights('final_isic_fusion_weights.h5')
# print("\nPelatihan Selesai. Model weights saved.")

