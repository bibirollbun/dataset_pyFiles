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


# membaca data yang akan dicluster
df = pd.read_csv('/kaggle/input/penguin-clustering-analysis/penguins.csv')
# menampilkan 5 data pertama dari file csv nya besarta dengan atribut/kolom nya.
df.head()


# Melihat informasi tipe data dan nilai yang hilang (null)
df.info()


# mengecek jumlah nilai yang null per kolom
print(df.isnull().sum())


# definisikan kolom berdasarkan tipenya supaya lebih mudah diproses
kolom_numerik = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
kolom_kategori = 'sex'

# Mengisi kolom numerik dengan median dari datanya.
for col in kolom_numerik:
    nilai_tengah = df[col].median()
    df[col] = df[col].fillna(nilai_tengah)

# 3. Mengisi kolom kategiri yaitu sex dengan modus
nilai_terbanyak = df[kolom_kategori].mode()[0]
df[kolom_kategori] = df[kolom_kategori].fillna(nilai_terbanyak)

# melakukan verifikasi, yaitu bahwa jika pengisian sudah nilai null berhasil, maka jumlah nilai null dari code ini akan bernilai 0 
print(df.isnull().sum())


# menggunakan hanya kolom fitur numerik untuk clustering
X = df[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]

# menampilkan 5 baris pertama untuk memastikan isi dari variable X nya
X.head()


# Menyamakan derajat nilai semua fitur kolom dengan menggunakan StandardScaler supaya semua fitur dianggap setara
from sklearn.preprocessing import StandardScaler

# Inisialisasi
scaler = StandardScaler()

# malakukan scalling
X_scaled = scaler.fit_transform(X)

# menampilkan 5 data pertama untuk pengecekan
print(X_scaled[:5])


# import library yang akan kita gunakan
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# List untuk menampung Inertia
inertia = []

# mencoba jumlah kluster dari 1 sampai 10
for k in range (1,11):
    kmeans = KMeans(n_clusters=k, init='random', random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# visualisasi untuk inertia
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.title('Elbow Method (Mencari Nilai K Terbaik)')
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()


# menggunakan 3 cluster
K = 3

# Inisialisasi dan menjalankan modelnya
kmeans = KMeans(n_clusters=K, init='random', random_state=42)
labels = kmeans.fit_predict(X_scaled)

# memuat hasil prediksi ke DataFrame asli
df['result'] = labels
print("\nJumlah penguin di setiap cluster:")
print(df['result'].value_counts())

# Cek 5 data teratas
display(df.head())


import seaborn as sns

# supaya gaya visualisasi lebih bagus
sns.set_style("whitegrid")

# Hubungan Dimensi Paruh 
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='culmen_length_mm', y='culmen_depth_mm',
                hue='result', palette='viridis', s=100)

plt.title('Cluster Berdasarkan Dimensi Paruh (Panjang vs Tebal)')
plt.xlabel('Panjang Paruh (mm)')
plt.ylabel('Ketebalan Paruh (mm)')
plt.legend(title='Cluster')
plt.show()


# gaya untuk visualnya
sns.set_style("whitegrid")

# Hubungan Ukuran Badan 
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='flipper_length_mm', y='body_mass_g',
                hue='result', palette='magma', s=100)

plt.title('Cluster Berdasarkan Ukuran Badan (Sirip vs Berat)')
plt.xlabel('Panjang Sirip (mm)')
plt.ylabel('Berat Badan (gram)')
plt.legend(title='Cluster')
plt.show()


# Melihat rata-rata fisik setiap cluster
profil_cluster = df.groupby('result')[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']].mean()

print("Profil Rata-rata Fisik per Cluster:")
display(profil_cluster)


submission = pd.DataFrame()
submission['ID'] = df.index
submission['Cluster_Label'] = df['result']
submission.to_csv('submission_penguins.csv', index=False)
print("File 'submission_penguins.csv' berhasil dibuat!")

