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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report, accuracy_score

# Mengabaikan RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Konfigurasi Plot
plt.style.use('seaborn-v0_8-whitegrid')

# --- LOAD DATA ---
df = pd.read_csv('/kaggle/input/penguin-clustering-analysis/penguins.csv') 


print("Informasi Dataset:")
print(df.info())
display(df.head())


# Cek Missing Values
print("\nJumlah Missing Values per Kolom:")
print(df.isnull().sum())

# Visualisasi Distribusi & Korelasi (Hanya fitur numerik)
sns.pairplot(df, diag_kind='kde', corner=True)
plt.suptitle("Distribusi Fitur dan Hubungan Antar Variabel", y=1.02)
plt.show()



# Mengunakan data yang sudah bersih
if 'df_clean' not in locals():
    df_clean = df[(df['flipper_length_mm'] < 300) & (df['flipper_length_mm'] > 0)].copy()

X_raw = df_clean.copy()

# Sesuaikan Fitur dengan kolom yang ada
numeric_features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
categorical_features = ['sex']

print(f"Menggunakan fitur numerik: {numeric_features}")
print(f"Menggunakan fitur kategori: {categorical_features}")

# Setup Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Eksekusi
X_processed = preprocessor.fit_transform(X_raw)
print(f"Shape data final: {X_processed.shape}")
    
# Simpan nama fitur untuk interpretasi nanti
# Mengambil nama fitur baru hasil One-Hot Encoding untuk 'sex'
ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
feature_names = numeric_features + list(ohe_feature_names)
print("Nama fitur final:", feature_names)


inertia = []
silhouette_scores = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_processed)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_processed, kmeans.labels_))

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 5))

# Grafik Elbow (Inertia)
ax1.plot(K_range, inertia, 'bo-', label='Inertia (Elbow)')
ax1.set_xlabel('Jumlah Klaster (k)')
ax1.set_ylabel('Inertia (Sum of Squared Distances)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Grafik Silhouette
ax2 = ax1.twinx()
ax2.plot(K_range, silhouette_scores, 'rs--', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('Metode Elbow dan Silhouette Score untuk Menentukan k Optimal')
plt.show()


# Tentukan k optimal (Kita pilih 3 sesuai konteks Biologi)
k_final = 3

# Latih Model
kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_processed)

# Simpan hasil ke DataFrame yang bersih (df_clean)
# Gunakan .loc agar tidak muncul SettingWithCopyWarning
df_clean = df_clean.copy()
df_clean.loc[:, 'Cluster_Label'] = cluster_labels

# Reduksi Dimensi dengan PCA (untuk visualisasi 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

# Buat DataFrame khusus plotting
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Cluster'] = cluster_labels

# Visualisasi Scatter Plot
plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=pca_df, x='PC1', y='PC2', 
    hue='Cluster', palette='viridis', s=100, alpha=0.8
)
plt.title(f'Hasil Clustering K-Means (k={k_final})\nPC1 vs PC2', fontsize=15)
plt.xlabel('Principal Component 1 (Variansi Terbesar)')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster ID')
plt.show()

# Analisis Karakteristik Klaster (Interpretasi)
# Rata-rata fitur numerik asli untuk setiap klaster
numeric_cols_for_summary = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
cluster_summary = df_clean.groupby('Cluster_Label')[numeric_cols_for_summary].mean()

display(cluster_summary)

# Pairplot Komprehensif
sns.pairplot(df_clean, vars=numeric_cols_for_summary, hue='Cluster_Label', palette='viridis', corner=True)
plt.suptitle('Pairplot Fitur Berdasarkan Cluster', y=1.02, fontsize=16)
plt.show()

