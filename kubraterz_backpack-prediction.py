import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# İki CSV dosyası yükleme
data_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
data_training_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')

# ilk birkaç satırı yazdır
print("Test Data:")
print(data_test.head())

print("\nTraining Extra Data:")
print(data_training_extra.head())


print("\nMissing Values in Train Data:")
print(data_training_extra.isnull().sum())

print("\nMissing Values in Test Data:")
print(data_test.isnull().sum())


# Sayısal özelliklerin istatistiksel özetleri
print("Train Data Statistical Summary:")
print(data_training_extra.describe())



print("\nTest Data Statistical Summary:")
print(data_test.describe())


# TRAİN_DATA İÇİN sütun sayısı
print("\nTrain Data Categorical Features Distribution:")
print(data_training_extra.select_dtypes(include=['object']).nunique())  



# TEST_DATA İÇİN sütun sayısı
print("\nTest Data Statistical Summary:")
print(data_test.select_dtypes(include=['object']).nunique())


# TRAIN DATA
for col in data_training_extra.select_dtypes(include=['object']).columns:
    print(f"\n{col} Distribution in Train Data:")
    print(data_training_extra[col].value_counts())


# TEST DATA
for col in data_test.select_dtypes(include=['object']).columns:
    print(f"\n{col} Distribution in Train Data:")
    print(data_test[col].value_counts())


import matplotlib.pyplot as plt
import seaborn as sns


df = pd.DataFrame({'A': [1, 2, float('inf'), 4]})

# Inf değerlerini NaN'ye dönüştürme
df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

print(df)


# PRICE DISTRIBUTION IN TRAIN DATA
plt.figure(figsize=(10, 6))
sns.histplot(data_training_extra['Price'], bins=30, kde=True)
plt.title('Price Distribution in Train Data')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


data_test['Size'] = data_test['Size'].dropna()


# SIZE DISTRIBUTION IN TEST DATA
plt.figure(figsize=(10, 6))
sns.histplot(data_test['Size'], bins=30, kde=True)
plt.title('Size Distribution in Test Data')
plt.xlabel('Size')
plt.ylabel('Frequency')
plt.show()


# Fiyat değişkeni için outlier analizi
plt.figure(figsize=(10, 6))
sns.boxplot(x=data_training_extra['Price'])
plt.title('Outliers in Price Column')
plt.show()


# Label Encoding 
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()


# 'Brand' sütununu sayısallşatırma
data_training_extra['Brand_encoded'] = label_encoder.fit_transform(data_training_extra['Brand'])


print("\nEncoded 'Brand' column in data_training_extra:")
print(data_training_extra[['Brand', 'Brand_encoded']].head())

