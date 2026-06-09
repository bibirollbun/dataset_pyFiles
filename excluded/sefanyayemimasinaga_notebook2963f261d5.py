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

# Path dasar ke folder input
base_path = "/kaggle/input/stanford-rna-3d-folding/"

# Load dataset utama dengan path lengkap
train_seq = pd.read_csv(base_path + "train_sequences.csv")
train_lbl = pd.read_csv(base_path + "train_labels.csv")
val_seq = pd.read_csv(base_path + "validation_sequences.csv")
val_lbl = pd.read_csv(base_path + "validation_labels.csv")
test_seq = pd.read_csv(base_path + "test_sequences.csv")
sample_sub = pd.read_csv(base_path + "sample_submission.csv")


# Cek bentuk data
print("Train sequences:", train_seq.shape)
print("Train labels:", train_lbl.shape)
print("Validation sequences:", val_seq.shape)
print("Validation labels:", val_lbl.shape)
print("Test sequences:", test_seq.shape)


# Lihat beberapa baris awal
print(train_seq.head())


print(train_lbl.head())


print(test_seq.head())


print(val_lbl.head())


print(val_seq.head())


# Memeriksa Tipe Data
print("\nTipe Data dalam Train Sequences:\n", train_seq.dtypes)
print("\nTipe Data dalam Train Labels:\n", train_lbl.dtypes)
print("\nTipe Data dalam Validation Sequences:\n", val_seq.dtypes)
print("\nTipe Data dalam Validation Labels:\n", val_lbl.dtypes)
print("\nTipe Data dalam Test Sequences:\n", test_seq.dtypes)


# Memeriksa Ukuran Dataset (Jumlah Baris dan Kolom)
print("\nJumlah baris dan kolom dalam Train Sequences:", train_seq.shape)
print("\nJumlah baris dan kolom dalam Train Labels:", train_lbl.shape)
print("\nJumlah baris dan kolom dalam Validation Sequences:", val_seq.shape)
print("\nJumlah baris dan kolom dalam Validation Labels:", val_lbl.shape)

# Memeriksa Ketidaksesuaian Kategori dalam Kolom (Jika ada kolom kategori)
if 'category_column' in train_seq.columns:  # Ganti 'category_column' dengan nama kolom kategori
    print("\nKategori dalam Kolom Category:\n", train_seq['category_column'].unique())



# Memeriksa Missing Values (Data yang Hilang)
missing_train_seq = train_seq.isnull().sum()
missing_train_lbl = train_lbl.isnull().sum()
missing_val_seq = val_seq.isnull().sum()
missing_val_lbl = val_lbl.isnull().sum()

# Menampilkan informasi missing values
print("\nMissing Values in Train Sequences:\n", missing_train_seq)
print("\nMissing Values in Train Labels:\n", missing_train_lbl)
print("\nMissing Values in Validation Sequences:\n", missing_val_seq)
print("\nMissing Values in Validation Labels:\n", missing_val_lbl)



# Memeriksa Duplikasi (Duplicate Data)
duplicate_train_seq = train_seq.duplicated().sum()
duplicate_train_lbl = train_lbl.duplicated().sum()
duplicate_val_seq = val_seq.duplicated().sum()
duplicate_val_lbl = val_lbl.duplicated().sum()

# Menampilkan jumlah duplikasi
print("\nJumlah baris duplikat dalam Train Sequences:", duplicate_train_seq)
print("Jumlah baris duplikat dalam Train Labels:", duplicate_train_lbl)
print("Jumlah baris duplikat dalam Validation Sequences:", duplicate_val_seq)
print("Jumlah baris duplikat dalam Validation Labels:", duplicate_val_lbl)


# Statistik Deskriptif (Descriptive Statistics) untuk Data Numerik
print("\nStatistik Deskriptif Train Labels:\n", train_lbl.describe())
print("\nStatistik Deskriptif Train Sequences:\n", train_seq.describe())



import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Gabungkan semua sekuens dalam train_sequences
all_nucleotides = ''.join(train_seq['sequence'].values)

# Hitung frekuensi setiap nukleotida
counter = Counter(all_nucleotides)

# Visualisasi frekuensi nukleotida
plt.figure(figsize=(6, 4))
sns.barplot(x=list(counter.keys()), y=list(counter.values()), palette='viridis')
plt.title('Frekuensi Nukleotida dalam Train Sequences')
plt.xlabel('Simbol Nukleotida')
plt.ylabel('Frekuensi')
plt.show()

# Menampilkan frekuensi simbol nukleotida
print("Frekuensi simbol nukleotida:", dict(counter))



import matplotlib.pyplot as plt
import seaborn as sns

# 1. Prosentase Nilai Hilang
missing_train_seq = train_seq.isnull().mean() * 100
missing_train_lbl = train_lbl.isnull().mean() * 100
missing_val_seq = val_seq.isnull().mean() * 100
missing_val_lbl = val_lbl.isnull().mean() * 100

# Plot missing values
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Train Sequences Missing
sns.barplot(x=missing_train_seq.index, y=missing_train_seq.values, ax=axes[0, 0])
axes[0, 0].set_title('Missing Data in Train Sequences')
axes[0, 0].set_ylabel('Percentage')

# Train Labels Missing
sns.barplot(x=missing_train_lbl.index, y=missing_train_lbl.values, ax=axes[0, 1])
axes[0, 1].set_title('Missing Data in Train Labels')
axes[0, 1].set_ylabel('Percentage')

# Validation Sequences Missing
sns.barplot(x=missing_val_seq.index, y=missing_val_seq.values, ax=axes[1, 0])
axes[1, 0].set_title('Missing Data in Validation Sequences')
axes[1, 0].set_ylabel('Percentage')

# Validation Labels Missing
sns.barplot(x=missing_val_lbl.index, y=missing_val_lbl.values, ax=axes[1, 1])
axes[1, 1].set_title('Missing Data in Validation Labels')
axes[1, 1].set_ylabel('Percentage')

plt.tight_layout()
plt.show()






