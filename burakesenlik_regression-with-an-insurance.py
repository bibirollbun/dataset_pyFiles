import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train_data = pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e12/sample_submission.csv')


import matplotlib.pyplot as plt
import seaborn as sns


train_data.head()


print(train_data.shape)


test_data.head()


print(test_data.shape)


sample_submission.head()


# Veri setleri hakkında genel bilgiler
print("Train Data Info:")
print(train_data.info(), "\n")
print("Train Data Null Values:")
print(train_data.isnull().sum(), "\n")


print("Test Data Info:")
print(test_data.info(), "\n")
print("Test Data Null Values:")
print(test_data.isnull().sum(), "\n")


# Kategorik ve sayısal sütunların analizi
print("Categorical Columns in Train Data:")
print(train_data.select_dtypes(include=['object']).columns, "\n")

print("Numerical Columns in Train Data:")
print(train_data.select_dtypes(include=['int64', 'float64']).columns, "\n")

print("Categorical Columns in Test Data:")
print(test_data.select_dtypes(include=['object']).columns, "\n")

print("Numerical Columns in Test Data:")
print(test_data.select_dtypes(include=['int64', 'float64']).columns, "\n")


# Her sütundaki benzersiz (unique) değerlerin sayısını gösterme
unique_counts = train_data.nunique()
print("Unique value counts for each column:\n")
print(unique_counts)


# Her sütundaki benzersiz (unique) değerlerin sayısını gösterme
unique_counts = test_data.nunique()
print("Unique value counts for each column:\n")
print(unique_counts)


# Her sütundaki benzersiz (unique) değerleri ve frekanslarını sıralama
for column in train_data.columns:
    print(f"Unique values for '{column}' (Top 10 most frequent):")
    print(train_data[column].value_counts().head(10), "\n")


# Veri tiplerini dönüştürme
train_data['Age'] = train_data['Age'].astype('Int64')  
train_data['Gender'] = train_data['Gender'].astype('category')
train_data['Marital Status'] = train_data['Marital Status'].astype('category')
train_data['Number of Dependents'] = train_data['Number of Dependents'].astype('Int64')  
train_data['Education Level'] = train_data['Education Level'].astype('category')
train_data['Occupation'] = train_data['Occupation'].astype('category')
train_data['Policy Type'] = train_data['Policy Type'].astype('category')
train_data['Policy Start Date'] = pd.to_datetime(train_data['Policy Start Date'], errors='coerce')
train_data['Customer Feedback'] = train_data['Customer Feedback'].astype('category')
train_data['Exercise Frequency'] = train_data['Exercise Frequency'].astype('category')
train_data['Property Type'] = train_data['Property Type'].astype('category')


# Veri tipi dönüşümlerini kontrol etme
print(train_data.dtypes)


# Veri tiplerini dönüştürme
test_data['Age'] = test_data['Age'].astype('Int64')  
test_data['Gender'] = test_data['Gender'].astype('category')
test_data['Marital Status'] = test_data['Marital Status'].astype('category')
test_data['Number of Dependents'] = test_data['Number of Dependents'].astype('Int64')  
test_data['Education Level'] = test_data['Education Level'].astype('category')
test_data['Occupation'] = test_data['Occupation'].astype('category')
test_data['Policy Type'] = test_data['Policy Type'].astype('category')
test_data['Policy Start Date'] = pd.to_datetime(test_data['Policy Start Date'], errors='coerce')
test_data['Customer Feedback'] = test_data['Customer Feedback'].astype('category')
test_data['Exercise Frequency'] = test_data['Exercise Frequency'].astype('category')
test_data['Property Type'] = test_data['Property Type'].astype('category')


# Veri tipi dönüşümlerini kontrol etme
print(test_data.dtypes)


# Temel istatistiksel bilgiler
print("Train Data Description:")
print(train_data.describe(), "\n")
print("Test Data Description:")
print(test_data.describe(), "\n")


plt.figure(figsize=(15,9))
plt.title("Visualizing Missing Values")
sns.heatmap(train_data.isnull(), cbar=False, cmap=sns.color_palette('magma'), yticklabels=False);
plt.show()


# Age sütunundaki eksik değerleri medyan ile doldur
age_median = train_data['Age'].median()
train_data['Age'].fillna(age_median, inplace=True)

# Vehicle Age ve Insurance Duration sütunlarındaki eksik değerleri mod ile doldur
vehicle_age_mode = train_data['Vehicle Age'].mode()[0]
insurance_duration_mode = train_data['Insurance Duration'].mode()[0]

train_data['Vehicle Age'].fillna(vehicle_age_mode, inplace=True)
train_data['Insurance Duration'].fillna(insurance_duration_mode, inplace=True)


def fill_missing_with_proportions(df, column_name):
    # Mevcut frekans dağılımı
    value_counts = df[column_name].value_counts(normalize=True)  # Oranları alıyoruz
    missing_count = df[column_name].isna().sum()  # Eksik değer sayısı

    # Eksik değerleri dolduracak unique değerlerin sayısını hesapla
    fill_values = np.random.choice(
        value_counts.index,  # Mevcut unique değerler
        size=missing_count,  # Eksik değer sayısı kadar rastgele seçim
        p=value_counts.values  # Orijinal oranlar
    )
    
    # Eksik değerleri doldur
    df.loc[df[column_name].isna(), column_name] = fill_values


# Previous Claims ve Occupation için uygulama
fill_missing_with_proportions(train_data, 'Previous Claims')
fill_missing_with_proportions(train_data, 'Occupation')
fill_missing_with_proportions(train_data, 'Marital Status')
fill_missing_with_proportions(train_data, 'Number of Dependents')
fill_missing_with_proportions(train_data, 'Customer Feedback')



print("Train Data Null Values:")
print(train_data.isnull().sum(), "\n")


# Her sütundaki benzersiz (unique) değerleri ve frekanslarını sıralama
for column in train_data.columns:
    print(f"Unique values for '{column}' (Top 10 most frequent):")
    print(train_data[column].value_counts().head(10), "\n")





# Kredi Skoru Kolonu İstatistiksel Özeti
credit_score_summary = train_data['Credit Score'].describe()
print("Kredi Skoru Kolonu İstatistiksel Özeti:")
print(credit_score_summary)

# Eksik değerlerin doldurulup doldurulmadığını kontrol et
missing_count = train_data['Credit Score'].isnull().sum()
print("\nEksik Değer Sayısı (Doldurma Sonrası):", missing_count)

# Kredi Skoru Histogramı
plt.figure(figsize=(10, 6))
sns.histplot(train_data['Credit Score'], bins=50, kde=True, color='royalblue')
plt.title('Kredi Skoru Dağılımı')
plt.xlabel('Kredi Skoru')
plt.ylabel('Frekans')
plt.show()

# Kredi Skoru Boxplot ile Uç Değer Analizi
plt.figure(figsize=(10, 4))
sns.boxplot(x=train_data['Credit Score'], color='orange')
plt.title('Kredi Skoru Uç Değer Analizi')
plt.show()

# Kredi Skoru'nun En Sık Kullanılan İlk 10 Değeri
most_frequent_scores = train_data['Credit Score'].value_counts().head(10)
print("\nKredi Skoru Kolonundaki En Sık Kullanılan İlk 10 Değer:")
print(most_frequent_scores)

# Verinin yoğun olduğu dilimlerin analizi için çeyrekler
q1 = train_data['Credit Score'].quantile(0.25)
q3 = train_data['Credit Score'].quantile(0.75)
iqr = q3 - q1

print("\nÇeyrek Değerler (Q1, Q3) ve IQR Bilgileri:")
print(f"Q1: {q1}, Q3: {q3}, IQR: {iqr}")
print(f"Alt sınır: {q1 - 1.5 * iqr}, Üst sınır: {q3 + 1.5 * iqr}")


