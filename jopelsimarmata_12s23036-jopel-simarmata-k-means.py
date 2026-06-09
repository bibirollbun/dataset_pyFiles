# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
""
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Memuat file 'penguins.csv' ke dalam pandas DataFrame.
penguin_df = pd.read_csv('/kaggle/input/penguin-clustering-analysis/penguins.csv')


# Menampilkan 5 baris pertama dari DataFrame untuk memahami struktur dan nilai data.
penguin_df.head()


# Menghapus semua baris yang mengandung nilai-nilai yang hilang (NaN) dari DataFrame.
# Ini penting karena algoritma K-Means tidak dapat menangani nilai missing.
penguin_df = penguin_df.dropna()


# Memverifikasi 5 baris pertama setelah penghapusan baris NaN.
penguin_df.head()


# Memilih kolom-kolom numerik yang akan digunakan sebagai fitur (variabel) untuk proses clustering K-Means.
X = penguin_df[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]


# Membuat scatter plot untuk memvisualisasikan sebaran data (titik-titik) berdasarkan dua fitur:
# culmen_length_mm (panjang paruh) dan culmen_depth_mm (kedalaman paruh).
plt.figure(figsize=(12,6))
plt.scatter(X["culmen_length_mm"], X["culmen_depth_mm"])
plt.xlabel('culmen_length_mm')
plt.ylabel('culmen_depth_mm')
plt.show()


# Menentukan jumlah klaster (K) yang akan digunakan.
K=2

# Memilih K=2 observasi secara acak dari data X untuk dijadikan centroid awal.
Centroids = (X.sample(n=K))

# Visualisasi data dan centroid awal:
plt.figure(figsize=(12,6))
plt.scatter(X["culmen_length_mm"], X["culmen_depth_mm"])

# Menandai centroid awal dengan warna merah ('red')
plt.scatter (Centroids ["culmen_length_mm"], Centroids ["culmen_depth_mm"], c='red')
plt.xlabel('culmen_length_mm')
plt.ylabel('culmen_depth_mm')
plt.show()


# Algoritma iteratif K-Means berjalan hingga centroid tidak lagi berpindah (konvergen).
diff = 1
j=0

while(diff!=0):
    XD = X
    i=1
    # Langkah 1: Assignment (Menentukan Klaster Terdekat)
    for index1, row_c in Centroids.iterrows():
        ED=[]
        for index2, row_d in XD.iterrows():
            d1=(row_c["culmen_length_mm"]-row_d["culmen_length_mm"])**2
            d2=(row_c["culmen_depth_mm"]-row_d["culmen_depth_mm"])**2
            d=np.sqrt(d1+d2)
            ED.append(d)
        X[i]=ED
        i=i+1
    # Menentukan klaster berdasarkan jarak minimum
    C=[]
    for index, row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    X["Cluster"]=C
    
    # Langkah 2: Update (Memperbarui Centroid)
    # Menghitung centroid baru (rata-rata dari semua titik dalam klaster)
    Centroids_new = X.groupby(["Cluster"]).mean()[["culmen_length_mm","culmen_depth_mm"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['culmen_length_mm'] - Centroids['culmen_length_mm']).sum() + \
               (Centroids_new['culmen_depth_mm'] - Centroids['culmen_depth_mm']).sum()
        print(diff.sum())
    # Mengganti centroid lama dengan yang baru untuk iterasi berikutnya
    Centroids = X.groupby(["Cluster"]).mean()[["culmen_length_mm","culmen_depth_mm"]]


color=['#003f5c', '#7a5195']
plt.figure(figsize=(12,6))

# Memplot setiap klaster dengan warna yang berbeda
for k in range(K):
    data=X[X["Cluster"]==k+1]
    plt.scatter(data["culmen_length_mm"], data["culmen_depth_mm"], c=color[k])
    
# Memplot centroid akhir (ditandai dengan warna merah)
plt.scatter(Centroids["culmen_length_mm"], Centroids["culmen_depth_mm"], c='red')
plt.xlabel('culmen_length_mm')
plt.ylabel('culmen_depth_mm')
plt.show()


submission_data = pd.DataFrame({
    'Id': penguin_df.index.values,
    'Cluster_Label': 'Cluster'
})

submission_data.to_csv('submission.csv', index=False) 

print("File 'submission.csv' berhasil dibuat.")

