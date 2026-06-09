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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings

# Konfigurasi agar output bersih
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

# Path file spesifik Kaggle
FILE_PATH = '/kaggle/input/penguin-clustering-analysis/penguins.csv'

# Membaca Data
try:
    df = pd.read_csv(FILE_PATH)
    print(" Dataset berhasil dimuat!")
    print(f"Dimensi Data: {df.shape}")
except FileNotFoundError:
    print(f" Error: File tidak ditemukan di {FILE_PATH}. Cek kembali path input.")

# Menampilkan 5 baris pertama
df.head()


df.head()
df.info()
df.describe(include='all')


print(df.columns.tolist())


df = df.copy()

# Buang baris yang ada missing pada kolom penting
df = df.dropna(subset=['culmen_length_mm', 'culmen_depth_mm', 
                       'flipper_length_mm', 'body_mass_g'])

# Hapus nilai flipper_length_mm yang tidak masuk akal (< 150 atau > 250)
df = df[(df['flipper_length_mm'] > 150) & (df['flipper_length_mm'] < 250)]

# Bersihkan kolom sex: isi missing → UNKNOWN
df['sex'] = df['sex'].fillna('UNKNOWN')

df.info()
df.describe(include='all')


# Ulangi dari DF sebelum di-encode
df['sex'] = df['sex'].fillna('UNKNOWN')
df['sex'] = df['sex'].replace({'': 'UNKNOWN', '.': 'UNKNOWN'})

# Encode ulang
df_encoded = pd.get_dummies(df, columns=['sex'], drop_first=False)

# Hapus kolom aneh jika masih ada
for col in df_encoded.columns:
    if col.startswith('sex_.'):
        df_encoded = df_encoded.drop(columns=[col])

df_encoded.head()
df_encoded.columns


df_encoded = pd.get_dummies(df, columns=['sex'], drop_first=False)
df_encoded.head()


from sklearn.preprocessing import StandardScaler

features = df_encoded.columns.tolist()
features.remove('sex_FEMALE')
features.remove('sex_MALE')
features.remove('sex_UNKNOWN')  # semua kategori dummy

X = df_encoded[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


num_cols = ['culmen_length_mm', 'culmen_depth_mm',
            'flipper_length_mm', 'body_mass_g']

X_num = df_encoded[num_cols]


scaler = StandardScaler()
X_scaled_num = scaler.fit_transform(X_num)


X = np.hstack([X_scaled_num, df_encoded[['sex_FEMALE','sex_MALE','sex_UNKNOWN']].values])


inertias = []
K_range = range(2, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, marker='o')
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()


sil_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    sil_scores.append(silhouette_score(X, labels))

plt.plot(K_range, sil_scores, marker='o')
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs k")
plt.show()


k = 3  
km_final = KMeans(n_clusters=k, random_state=42)
df_encoded['cluster'] = km_final.fit_predict(X)


pca = PCA(n_components=2)
pcs = pca.fit_transform(X)

df_encoded['PC1'] = pcs[:, 0]
df_encoded['PC2'] = pcs[:, 1]


plt.figure(figsize=(7,5))
sns.scatterplot(data=df_encoded, x='PC1', y='PC2', 
                hue='cluster', palette='Set2')
plt.title("Visualisasi PCA Cluster Penguins")
plt.show()


df_encoded.groupby('cluster')[num_cols].mean()


df_encoded.groupby('cluster')[['sex_FEMALE','sex_MALE','sex_UNKNOWN']].sum()

