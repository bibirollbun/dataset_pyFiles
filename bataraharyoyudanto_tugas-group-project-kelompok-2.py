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

# Load dataset utama
train_path = '/kaggle/input/home-credit-default-risk/application_train.csv'
df = pd.read_csv(train_path)



# 1. Eksplorasi Data Awal
print("\n--- Informasi Dataset ---")
print(df.info())

print("\n--- 5 Baris Pertama ---")
print(df.head())


# Mengatasi nilai NaN sebelum eksplorasi statistik
df_filled = df.fillna(0)  # Mengisi NaN dengan 0 agar tidak error saat statistik


# 2. Mengecek nilai yang hilang
missing_values = df.isnull().sum().sort_values(ascending=False)
missing_percent = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({'Total Missing': missing_values, 'Percent': missing_percent})
print("\n--- Nilai Hilang ---")
print(missing_data.head(10))


# 3. Statistik Deskriptif
desc = df_filled.describe()
print("\n--- Statistik Deskriptif ---")
print(desc)


# 4. Visualisasi Distribusi Variabel Target
plt.figure(figsize=(6,4))
sns.countplot(x='TARGET', data=df, palette='coolwarm')
plt.title('Distribusi Variabel Target')
plt.show()


# 5. Menangani Nilai Hilang (Hanya Kolom Numerik)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
print("\n--- Nilai Hilang Setelah Imputasi ---")
print(df[numeric_cols].isnull().sum().sum(), "nilai masih hilang")



# 6. Transformasi Variabel Kategorikal ke Numerik (One-Hot Encoding)
df = pd.get_dummies(df, drop_first=True)


# 7. Normalisasi Fitur Numerik dari 0-1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numeric_features = df.select_dtypes(include=[np.number]).columns
df[numeric_features] = scaler.fit_transform(df[numeric_features])



# 8. Analisis Ketidakseimbangan Kelas
target_counts = df['TARGET'].value_counts()
print("\n--- Analisis Ketidakseimbangan Kelas ---")
print(target_counts)

plt.figure(figsize=(6,4))
sns.barplot(x=target_counts.index, y=target_counts.values, palette='coolwarm')
plt.title('Distribusi Kelas dalam Variabel Target')
plt.show()

print("\n--- Prapemrosesan Data Selesai! ---")


