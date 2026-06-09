import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pandas as pd

# Load application data
df = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')

# Display first few rows
df.head()



# Menampilkan 5 baris pertama dataset
print("5 Baris Pertama Dataset Home Credit:")
print(df.head())

# Menampilkan informasi dasar tentang dataset
print("\nInformasi Dataset:")
print(df.info())

# Menampilkan jumlah baris dan kolom
print(f"\nJumlah Baris: {df.shape[0]}, Jumlah Kolom: {df.shape[1]}")

# Menampilkan tipe data setiap kolom
print("\nTipe Data Setiap Kolom:")
print(df.dtypes)

# Menampilkan jumlah nilai yang hilang per kolom
print("\nJumlah Nilai yang Hilang per Kolom:")
print(df.isnull().sum())


import matplotlib.pyplot as plt

# Menghitung persentase nilai yang hilang per kolom
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)

# Visualisasi nilai yang hilang
plt.figure(figsize=(12, 8))
missing_percent.plot(kind='bar')
plt.title('Persentase Nilai yang Hilang per Kolom')
plt.ylabel('Persentase (%)')
plt.show()

print("\nKolom dengan nilai yang hilang lebih dari 50%:")
print(missing_percent[missing_percent > 40])


# Drop kolom dengan missing > 60%
threshold = 0.6  # 60%
df_drop = df.loc[:, df.isnull().mean() < threshold]
print(f'Shape sebelum drop kolom: {df.shape}')
print(f'Shape setelah drop kolom: {df_drop.shape}')


from sklearn.impute import SimpleImputer

# Imputasi untuk kolom numerik
numerical_cols = df_drop.select_dtypes(include=['int64', 'float64']).columns
numerical_cols = [col for col in numerical_cols if col not in ['TARGET', 'SK_ID_CURR']]

# Menggunakan median untuk imputasi (karena lebih robust terhadap outlier)
imputer_num = SimpleImputer(strategy='median')
df_drop[numerical_cols] = imputer_num.fit_transform(df_drop[numerical_cols])


# Imputasi untuk kolom kategorikal
categorical_cols = df_drop.select_dtypes(include=['object']).columns

# Menggunakan modus untuk imputasi
imputer_cat = SimpleImputer(strategy='most_frequent')
df_drop[categorical_cols] = imputer_cat.fit_transform(df_drop[categorical_cols])


# Memeriksa apakah masih ada missing values
print("Jumlah missing values:")
print(df_drop.isnull().sum().sum())


# Fungsi untuk analisis outlier
def analyze_outliers(df_drop, columns, threshold=1.5):
    outlier_report = {}
    
    for col in columns:
        Q1 = df_drop[col].quantile(0.25)
        Q3 = df_drop[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = df_drop[(df_drop[col] < lower_bound) | (df_drop[col] > upper_bound)]
        
        outlier_report[col] = {
            'n_outliers': len(outliers),
            'pct_outliers': f"{len(outliers)/len(df)*100:.2f}%",
            'min_val': df_drop[col].min(),
            'max_val': df_drop[col].max(),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    return pd.DataFrame(outlier_report).T

key_columns = [
    'AMT_INCOME_TOTAL', 
    'AMT_CREDIT', 
    'AMT_ANNUITY',
    'AMT_GOODS_PRICE',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED'
]

# Analisis outlier
outlier_analysis = analyze_outliers(df, key_columns)
print("Analisis Outlier Awal:")
display(outlier_analysis)


import numpy as np
def winsorize_data(df, columns, threshold=1.5):
    """
    Mengganti outlier dengan nilai batas menggunakan pendekatan winsorizing
    threshold: 3.0 untuk lebih longgar (3 x IQR)
    """
    df_clean = df.copy()
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Capping
        df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
        df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
    
    return df_clean

# Terapkan winsorizing
df_no_outliers= winsorize_data(df_drop, key_columns, threshold=3.0)


import seaborn as sns

# Analisis setelah penanganan
print("\nAnalisis Setelah Penanganan Outlier:")
outlier_analysis_clean = analyze_outliers(df_drop, key_columns)
display(outlier_analysis_clean)

# Visualisasi perbandingan
fig, axes = plt.subplots(len(key_columns), 2, figsize=(15, 20))
for i, col in enumerate(key_columns):
    # Sebelum
    sns.boxplot(x=df_drop[col], ax=axes[i, 0])
    axes[i, 0].set_title(f'Original {col}')
    
    # Sesudah
    sns.boxplot(x=df_no_outliers[col], ax=axes[i, 1])
    axes[i, 1].set_title(f'Cleaned {col}')
    
plt.tight_layout()
plt.show()

# Periksa ukuran dataset
print(f"\nDimensi dataset sebelum cleaning: {df_drop.shape}")
print(f"Dimensi dataset setelah cleaning: {df_no_outliers.shape}")


# Verifikasi untuk satu kolom contoh
col = key_columns[0]  # ambil kolom pertama

Q1 = df_drop[col].quantile(0.25)
Q3 = df_drop[col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3.0 * IQR
upper_bound = Q3 + 3.0 * IQR

# Cek apakah masih ada nilai di luar bounds
outliers_exist = ((df_no_outliers[col] < lower_bound) | (df_no_outliers[col] > upper_bound)).any()
print(f"Apakah masih ada outlier di kolom {col} setelah winsorizing? {outliers_exist}")

# Cek nilai min/max sebelum dan sesudah
print(f"\nKolom: {col}")
print(f"Sebelum - Min: {df_drop[col].min()}, Max: {df_drop[col].max()}")
print(f"Sesudah - Min: {df_no_outliers[col].min()}, Max: {df_no_outliers[col].max()}")


from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Memisahkan kolom kategorikal
categorical_cols = df_no_outliers.select_dtypes(include=['object']).columns

# One-Hot Encoding untuk kolom dengan cardinality rendah (<10 nilai unik)
low_cardinality_cols = [col for col in categorical_cols if df_no_outliers[col].nunique() < 10]

# Label Encoding untuk kolom dengan cardinality tinggi
high_cardinality_cols = [col for col in categorical_cols if df_no_outliers[col].nunique() >= 10]

# Melakukan One-Hot Encoding
df_encoded = pd.get_dummies(df_no_outliers, columns=low_cardinality_cols, drop_first=True)

# Melakukan Label Encoding untuk kolom dengan cardinality tinggi
label_encoder = LabelEncoder()
for col in high_cardinality_cols:
    df_encoded[col] = label_encoder.fit_transform(df_encoded[col].astype(str))

print("\nShape setelah encoding:", df_encoded.shape)


from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Memisahkan fitur dan target
X = df_encoded.drop(columns=['TARGET', 'SK_ID_CURR'])
y = df_encoded['TARGET']

# Daftar kolom numerik untuk diskalakan
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Standarisasi (Z-score normalization)
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Alternatif: Normalisasi Min-Max
# minmax_scaler = MinMaxScaler()
# X[numeric_cols] = minmax_scaler.fit_transform(X[numeric_cols])

print("\nContoh data setelah standarisasi:")
print(X[numeric_cols].head())


# Membuat fitur baru: Rasio Kredit terhadap Pendapatan
df_encoded['CREDIT_INCOME_RATIO'] = df_encoded['AMT_CREDIT'] / df_encoded['AMT_INCOME_TOTAL']

# Membuat fitur baru: Rasio Anuitas terhadap Pendapatan
df_encoded['ANNUITY_INCOME_RATIO'] = df_encoded['AMT_ANNUITY'] / df_encoded['AMT_INCOME_TOTAL']

# Membuat fitur baru: Usia dalam tahun (dari DAYS_BIRTH)
df_encoded['AGE_YEARS'] = abs(df_encoded['DAYS_BIRTH']) / 365

# Membuat fitur baru: Lama bekerja dalam tahun (dari DAYS_EMPLOYED)
df_encoded['WORKING_YEARS'] = abs(df_encoded['DAYS_EMPLOYED']) / 365

# Membuat fitur baru: Persentase hari bekerja dari usia
df_encoded['WORKING_LIFE_RATIO'] = abs(df_encoded['DAYS_EMPLOYED']) / abs(df_encoded['DAYS_BIRTH'])

# Memeriksa fitur baru
print("\nFitur baru yang dibuat:")
new_features = ['CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'AGE_YEARS', 'WORKING_YEARS', 'WORKING_LIFE_RATIO']
print(df_encoded[new_features].describe())


df_encoded.head()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Memisahkan fitur dan target
X = df_encoded.drop('TARGET', axis=1)
y = df_encoded['TARGET']

# Menampilkan statistik awal
print("=== Statistik Data Awal ===")
print(f"Jumlah total sampel: {len(df)}")
print(f"Jumlah fitur: {X.shape[1]}")
print("\nDistribusi kelas:")
print(y.value_counts(normalize=True))


# Pembagian data dengan stratified sampling
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.25, 
    random_state=42,  # Untuk reproducibility
    stratify=y       # Stratified sampling
)

# Menampilkan statistik hasil pembagian
print("\n=== Hasil Pembagian Data ===")
print(f"Jumlah data training: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
print(f"Jumlah data testing: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")

print("\nDistribusi kelas pada data training:")
print(y_train.value_counts(normalize=True))

print("\nDistribusi kelas pada data testing:")
print(y_test.value_counts(normalize=True))


# 10-fold stratified cross validation
skf = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\nFold {fold + 1}:")
    print(f"  Training: index {train_idx[0]}..{train_idx[-1]}")
    print(f"  Validation: index {val_idx[0]}..{val_idx[-1]}")
    print(f"  Class distribution in validation: {np.bincount(y_train.iloc[val_idx])}")


# Transformasi data (standardisasi)
scaler = StandardScaler()

# Hanya fit pada data training untuk menghindari data leakage
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Konversi kembali ke DataFrame untuk mempertahankan nama kolom
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("\nContoh data setelah standardisasi:")
print(X_train_scaled.head())


# Visualisasi pembagian data
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Distribusi kelas asli
plt.subplot(1, 2, 1)
y.value_counts().plot(kind='bar', color=['blue', 'red'])
plt.title('Distribusi Kelas Asli')

# Distribusi setelah split
plt.subplot(1, 2, 2)
pd.Series(y_train).value_counts().plot(kind='bar', color='blue', alpha=0.5, label='Training')
pd.Series(y_test).value_counts().plot(kind='bar', color='red', alpha=0.5, label='Testing')
plt.title('Distribusi Setelah Stratified Split')
plt.legend()

plt.tight_layout()
plt.show()

