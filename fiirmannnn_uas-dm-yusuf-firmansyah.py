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
df = pd.read_csv('/kaggle/input/allstate-claims-severity/train.csv')

# Struktur dan dimensi
print("Dimensi data:", df.shape)
print("\nTipe data tiap kolom:")
print(df.dtypes.value_counts())
print("\nContoh 5 baris pertama:")
print(df.head())



missing = df.isnull().sum()
print("Kolom dengan missing values:")
print(missing[missing > 0])



duplicates = df.duplicated().sum()
print("Jumlah baris duplikat:", duplicates)



import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot dari target loss
sns.boxplot(df['loss'])
plt.title("Outlier pada target 'loss'")
plt.show()

# Bisa juga menggunakan IQR method untuk mendeteksi secara numerik
Q1 = df['loss'].quantile(0.25)
Q3 = df['loss'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['loss'] < Q1 - 1.5 * IQR) | (df['loss'] > Q3 + 1.5 * IQR)]
print("Jumlah outlier pada 'loss':", outliers.shape[0])



from sklearn.preprocessing import OneHotEncoder

# Pisahkan data kategorikal dan numerik
cat_cols = [col for col in df.columns if 'cat' in col]
cont_cols = [col for col in df.columns if 'cont' in col]

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print("Dimensi data setelah encoding:", df_encoded.shape)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_encoded[cont_cols] = scaler.fit_transform(df_encoded[cont_cols])



import seaborn as sns
import matplotlib.pyplot as plt

# Distribusi loss (target)
plt.figure(figsize=(10, 6))
sns.histplot(df['loss'], bins=50, kde=True, color='skyblue')
plt.title('Distribusi Nilai Loss')
plt.xlabel('Loss')
plt.ylabel('Frekuensi')
plt.show()



# Korelasi antar fitur numerik
plt.figure(figsize=(12, 8))
corr = df[cont_cols + ['loss']].corr()
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title('Heatmap Korelasi: Fitur Kontinu dan Loss')
plt.show()



# Boxplot loss berdasarkan kategori
plt.figure(figsize=(8, 5))
sns.boxplot(x='cat1', y='loss', data=df)
plt.title('Distribusi Loss Berdasarkan Kategori cat1')
plt.xlabel('Kategori cat1')
plt.ylabel('Loss')
plt.show()



# ===============================================
# 1. Import Library
# ===============================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor



# ===============================================
# 3. Identifikasi Kolom Kategorikal & Kontinu
# ===============================================
cat_cols = [col for col in df.columns if 'cat' in col]
cont_cols = [col for col in df.columns if 'cont' in col]

# ===============================================
# 4. Encoding & Normalisasi
# ===============================================
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

scaler = StandardScaler()
df_encoded[cont_cols] = scaler.fit_transform(df_encoded[cont_cols])

# ===============================================
# 5. Split Fitur & Target
# ===============================================
X = df_encoded.drop(columns=['id', 'loss'])
y = df_encoded['loss']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================================
# 6. Modeling dengan LightGBM
# ===============================================
model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# ===============================================
# 7. Evaluasi Model
# ===============================================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# ===============================================
# 8. Output Evaluasi
# ===============================================
print("ðŸ“Š Evaluasi Model Regresi:")
print(f"âœ… MAE  : {mae:.2f}")
print(f"âœ… RMSE : {rmse:.2f}")
print(f"âœ… RÂ²   : {r2:.4f}")



# ===============================================
# 9. Visualisasi Hasil Prediksi
# ===============================================
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 6))

# Plot 1: y_test vs y_pred
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual Loss")
plt.ylabel("Predicted Loss")
plt.title("ðŸ“ˆ Prediksi vs Aktual (LightGBM)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')  # Garis y=x

# Plot 2: Distribusi Error
plt.subplot(1, 2, 2)
errors = y_test - y_pred
sns.histplot(errors, bins=50, kde=True, color='orange')
plt.xlabel("Prediction Error")
plt.title("ðŸ“‰ Distribusi Error (Residuals)")

plt.tight_layout()
plt.show()



# ========================
# 1. Import Libraries
# ========================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# ========================
# 3. Encoding & Scaling
# ========================
cat_cols = [col for col in df.columns if 'cat' in col]
cont_cols = [col for col in df.columns if 'cont' in col]

# One-hot encode kategori
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Normalisasi fitur kontinyu
scaler = StandardScaler()
df_encoded[cont_cols] = scaler.fit_transform(df_encoded[cont_cols])

# ========================
# 4. Split Data
# ========================
X = df_encoded.drop(columns=['id', 'loss'])
y = df_encoded['loss']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========================
# 5. Linear Regression Model
# ========================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# ========================
# 6. Evaluation
# ========================
y_pred = lr_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("ðŸ“Š Evaluasi Model Linear Regression")
print(f"âœ… MAE  : {mae:.2f}")
print(f"âœ… RMSE : {rmse:.2f}")
print(f"âœ… RÂ²   : {r2:.4f}")



# ========================
# 7. Visualisasi
# ========================

# --- Plot 1: Predicted vs Actual ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.4, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Ideal Fit')
plt.xlabel("Actual Loss")
plt.ylabel("Predicted Loss")
plt.title("Predicted vs Actual Loss (Linear Regression)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Residual Distribution ---
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=50, color='slateblue')
plt.title("Distribusi Residual (Actual - Predicted)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()





# ========================
# 1. Import Libraries
# ========================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt
import seaborn as sns

# ========================

# 3. Persiapan Fitur
# ========================
cat_cols = [col for col in df.columns if 'cat' in col]
cont_cols = [col for col in df.columns if 'cont' in col]

X = df.drop(columns=['id', 'loss'])
y = df['loss']

# ========================
# 4. Split Data
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================
# 5. CatBoost Regressor
# ========================
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    cat_features=cat_cols,
    verbose=0,
    random_seed=42
)

model.fit(X_train, y_train)

# ========================
# 6. Evaluation
# ========================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("ðŸ“Š Evaluasi Model CatBoost Regressor")
print(f"âœ… MAE  : {mae:.2f}")
print(f"âœ… RMSE : {rmse:.2f}")
print(f"âœ… RÂ²   : {r2:.4f}")



# --- Plot 1: Predicted vs Actual ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.4, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Ideal Fit')
plt.xlabel("Actual Loss")
plt.ylabel("Predicted Loss")
plt.title("Predicted vs Actual Loss (CatBoost)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Residual Distribution ---
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=50, color='slateblue')
plt.title("Distribusi Residual (Actual - Predicted)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

