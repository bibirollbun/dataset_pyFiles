import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


df_Q4 = pd.read_excel("/kaggle/input/creditscoresample/credit_scoring_sample.xlsx")


df_Q4.head()


#Menghitung korelasi data numerik
selected_cols = ['DebtRatio', 'MonthlyIncome', 'NumberOfDependents', 'SeriousDlqin2yrs']
correlation_matrix_selected = df_Q4[selected_cols].corr()

# Menampilkan korelasi numerik
print("\nKorelasi antara DebtRatio, MonthlyIncome, NumberOfDependents dengan default:")
print(correlation_matrix_selected)

# Visualisasi heatmap korelasi dengan rentang -1 sampai 1
plt.figure(figsize=(6, 5))
sns.heatmap(
    correlation_matrix_selected, 
    annot=True, 
    cmap='coolwarm', 
    center=0, 
    linewidths=0.5,
    vmin=-1, vmax=1  # Menentukan batas minimum dan maksimum warna
)
plt.title('Heatmap Korelasi (DebtRatio, MonthlyIncome, NumberOfDependents, SeriousDlqin2yrs)')
plt.show()


# Memilih semua kolom numerik kecuali target (SeriousDlqin2yrs)
numerical_cols = df_Q4.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove('SeriousDlqin2yrs')  # Target dihapus dari analisis VIF

# Menghapus baris dengan nilai NaN (jika ada)
X = df_Q4[numerical_cols].dropna()

# Menambahkan konstanta untuk regresi (diperlukan untuk VIF)
X = sm.add_constant(X)

# Menghitung VIF untuk setiap variabel
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_data


plt.figure(figsize=(12, 6))
df_Q4['SeriousDlqin2yrs'].value_counts().plot(kind='bar')

plt.title('90 Days Past Due Delunquency or Worse', fontsize=14)
plt.xlabel('No or Yes', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.show()


print(df_Q4['SeriousDlqin2yrs'].value_counts())


import numpy as np

# Daftar variabel independen
independent_vars = ['age', 'DebtRatio', 'MonthlyIncome', 'NumberOfDependents']

# Menentukan batas IQR
Q1 = df_Q4[independent_vars].quantile(0.25)  # Kuartil 1 (Q1)
Q3 = df_Q4[independent_vars].quantile(0.75)  # Kuartil 3 (Q3)
IQR = Q3 - Q1  # Rentang antar kuartil

# Menentukan batas bawah dan atas untuk outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Menemukan outlier
outliers = ((df_Q4[independent_vars] < lower_bound) | (df_Q4[independent_vars] > upper_bound))

# Menampilkan jumlah outlier per kolom
print("Jumlah outlier per kolom:")
print(outliers.sum())

# Menampilkan baris yang mengandung outlier di setidaknya satu kolom
df_Q4_outliers = df_Q4[outliers.any(axis=1)]
print("\nData dengan outlier:")
print(df_Q4_outliers)


import seaborn as sns
import matplotlib.pyplot as plt

# Variabel yang ingin dicek
columns_to_check = ['age', 'DebtRatio', 'MonthlyIncome', 'NumberOfDependents', 'SeriousDlqin2yrs']

# Plot histogram dan KDE
plt.figure(figsize=(12, 8))
for i, col in enumerate(columns_to_check, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df_Q4[col], kde=True, bins=30)
    plt.title(f'Distribusi {col}')
plt.tight_layout()
plt.show()



missing_values = df_Q4.isnull().sum()
print(missing_values)


# Hapus baris yang memiliki missing value pada "NumberOfDependents"
df_Q4_cleaned = df_Q4.dropna(subset=['NumberOfDependents'])

# Isi missing value pada "MonthlyIncome" dengan median
median_income = df_Q4_cleaned['MonthlyIncome'].median()
df_Q4_cleaned['MonthlyIncome'].fillna(median_income, inplace=True)

# Hapus kolom yang tidak diperlukan
df_Q4_cleaned.drop(columns=['NumberOfTimes90DaysLate','NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse'], inplace=True)

# Cek hasil perubahan
print(df_Q4_cleaned.isnull().sum())  # Pastikan tidak ada missing value lagi di kedua kolom tersebut


duplicated_counts = df_Q4_cleaned.duplicated().sum()
print(duplicated_counts)


# Menghapus duplikat
df_Q4_NoDup = df_Q4_cleaned.drop_duplicates().reset_index(drop=True)


# Mengecek jumlah duplikat setelah dibersihkan
df_Q4_NoDup_sum = df_Q4_NoDup.duplicated().sum()
print(df_Q4_NoDup_sum)


df_Q4_NoDup.head()


# Pisahkan kelas mayoritas dan minoritas
majority_class = df_Q4_NoDup[df_Q4_NoDup['SeriousDlqin2yrs'] == False]
minority_class = df_Q4_NoDup[df_Q4_NoDup['SeriousDlqin2yrs'] == True]

# Oversampling: Duplikasi kelas minoritas hingga jumlahnya sama dengan kelas mayoritas
minority_oversampled = minority_class.sample(n=len(majority_class), replace=True, random_state=42)

# Gabungkan kembali
df_Q4_oversampled = pd.concat([majority_class, minority_oversampled])

# Acak ulang data
df_Q4_oversampled = df_Q4_oversampled.sample(frac=1, random_state=42).reset_index(drop=True)

# Cek hasilnya
print(df_Q4_oversampled['SeriousDlqin2yrs'].value_counts())


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
from sklearn.feature_selection import mutual_info_classif

# Pilih kolom yang akan dianalisis
selected_cols = ['DebtRatio', 'MonthlyIncome', 'NumberOfDependents', 'SeriousDlqin2yrs']

# Menghitung korelasi dengan Spearman (lebih cocok untuk target biner)
correlation_matrix_selected = df_Q4_oversampled[selected_cols].corr(method='spearman')

# Menampilkan korelasi numerik
print("\nKorelasi Spearman antara DebtRatio, MonthlyIncome, NumberOfDependents dengan SeriousDlqin2yrs:")
print(correlation_matrix_selected)

# Menambahkan Point Biserial Correlation untuk target biner
for col in ['DebtRatio', 'MonthlyIncome', 'NumberOfDependents']:
    correlation_pb, p_value = pointbiserialr(df_Q4_oversampled[col], df_Q4_oversampled['SeriousDlqin2yrs'])
    print(f"Point Biserial Correlation ({col} vs SeriousDlqin2yrs): {correlation_pb:.4f} (p={p_value:.4f})")

# Menambahkan Mutual Information Score
mi_scores = mutual_info_classif(df_Q4_oversampled[['DebtRatio', 'MonthlyIncome', 'NumberOfDependents']], df_Q4_oversampled['SeriousDlqin2yrs'])
mi_scores_dict = dict(zip(['DebtRatio', 'MonthlyIncome', 'NumberOfDependents'], mi_scores))
print("\nMutual Information Scores:")
for feature, score in mi_scores_dict.items():
    print(f"{feature}: {score:.4f}")

# Visualisasi heatmap korelasi
plt.figure(figsize=(6, 5))
sns.heatmap(
    correlation_matrix_selected, 
    annot=True, 
    cmap='coolwarm', 
    center=0, 
    linewidths=0.5,
    vmin=-1, vmax=1
)
plt.title('Heatmap Korelasi (Spearman) DebtRatio, MonthlyIncome, NumberOfDependents, SeriousDlqin2yrs')
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Pisahkan fitur (X) dan target (y)
X = df_Q4_oversampled.drop(columns=['SeriousDlqin2yrs'])
y = df_Q4_oversampled['SeriousDlqin2yrs']

# 2. Split data menjadi training dan testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Normalisasi dengan StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Buat model Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

# 5. Prediksi hasil
y_pred = log_reg.predict(X_test_scaled)

# 6. Evaluasi model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Visualisasi Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Logistic Regression')
plt.show()

# 8. Lihat Koefisien Fitur
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': log_reg.coef_[0]})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)

print("\nKoefisien Fitur dalam Logistic Regression:")
print(coefficients)





