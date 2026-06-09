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

# Load data
df = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv')

# Struktur dan dimensi
print("Dimensi data (baris, kolom):", df.shape)

# Tipe data tiap kolom
print("\nTipe data per kolom:")
print(df.dtypes)

# Cek 5 baris pertama
print("\nContoh 5 baris pertama:")
print(df.head())


print("Jumlah missing values per kolom:")
print(df.isnull().sum())


print("Jumlah baris duplikat:", df.duplicated().sum())


print(df['price'].describe())


Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
print("Jumlah outlier pada kolom price:", outliers.shape[0])


# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['sales_channel_id'], prefix='channel')
print(df_encoded.head())


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df['price_normalized'] = scaler.fit_transform(df[['price']])


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['price_standardized'] = scaler.fit_transform(df[['price']])


print(df[['price', 'price_normalized', 'price_standardized']].head())


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.histplot(df['price'], bins=100, kde=True)
plt.title('Distribusi Harga Produk')
plt.xlabel('Harga')
plt.ylabel('Jumlah')
plt.show()


plt.figure(figsize=(6,4))
sns.countplot(x='sales_channel_id', data=df)
plt.title('Distribusi Kanal Penjualan')
plt.xlabel('Kanal (1 = Online, 2 = Offline)')
plt.ylabel('Jumlah Transaksi')
plt.show()


# Pastikan kolom t_dat bertipe datetime
df['t_dat'] = pd.to_datetime(df['t_dat'])

# Agregasi per minggu
transaksi_mingguan = df.groupby(pd.Grouper(key='t_dat', freq='W'))['price'].count()

# Plot
plt.figure(figsize=(12,5))
transaksi_mingguan.plot()
plt.title('Tren Jumlah Transaksi per Minggu')
plt.xlabel('Tanggal')
plt.ylabel('Jumlah Transaksi')
plt.grid(True)
plt.show()



import pandas as pd

# Load data transaksi
df = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Pastikan datetime sudah diparse
df['t_dat'] = pd.to_datetime(df['t_dat'])

# Buat fitur dari tanggal
df['day_of_week'] = df['t_dat'].dt.dayofweek
df['month'] = df['t_dat'].dt.month

# Ambil fitur dan target
X = df[['price', 'day_of_week', 'month']]
y = df['sales_channel_id']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model klasifikasi
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediksi dan evaluasi
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))




