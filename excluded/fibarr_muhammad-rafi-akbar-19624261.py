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


#Muhammad Rafi Akbar
#13524125


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# baca data
train = pd.read_csv('/kaggle/input/airbnb/train.csv')
test = pd.read_csv('/kaggle/input/airbnb/test.csv')

# Statistik deskriptif harga
print("Statistik Harga:")
print(train['price'].describe())

# grafik distribusi harga
# plt.figure(figsize=(10, 5))
# sns.histplot(train['price'], bins=100, kde=True)
# plt.xlim(0, 1000)  
# plt.title("Distribusi Harga Sewa Airbnb (price <= 1000)")
# plt.xlabel("Harga per malam")
# plt.ylabel("Jumlah listing")
# plt.show()

# cek missing value
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("Kolom dengan missing value:")
print(missing)

# Grafik missing value
#plt.figure(figsize=(10, 6))
# sns.barplot(x=missing.values, y=missing.index)
# plt.title("Jumlah Missing Value per Kolom")
# plt.xlabel("Jumlah Missing")
# plt.ylabel("Kolom")
# plt.show()

# cari hubungan antar angka
numerics = train.select_dtypes(include=['float64', 'int64'])
corr = numerics.corr()['price'].sort_values(ascending=False)

print("Fitur paling berhubungan dengan harga:")
print(corr.head(10))
print("\nFitur paling negatif:")
print(corr.tail(5))


from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Pilih fitur dengan hubungan kuat
selected_features = [
    'accommodates', 'bedrooms', 'bathrooms', 'beds',
    'room_type', 'property_type', 'city',
    'review_scores_location', 'review_scores_rating',
]

# Isi missing value dengan nilai rata-rata
for col in ['bedrooms', 'bathrooms', 'beds', 'review_scores_location', 'review_scores_rating']:
    median_train = train[col].median()
    train[col] = train[col].fillna(median_train)
    test[col] = test[col].fillna(median_train)  

# Ubah text jadi angka
for col in ['room_type', 'property_type', 'city']:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Pisah menjadi fitur yang dipilih dan target model
X = train[selected_features]
y = train['price']


# Pisah data untuk train dan test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Uji model dengan sisa data yang dipisah untuk train dan test
y_pred = model.predict(X_val)

# Bandingkan hasil model dengan asli
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.2f}")

# Prediksi harga
X_test = test[selected_features]
test_preds = model.predict(X_test)

submission = pd.DataFrame({
    "id": test["id"],
    "price": test_preds
})

submission.to_csv("submission.csv", index=False)



