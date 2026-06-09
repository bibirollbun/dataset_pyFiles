# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_parquet('/kaggle/input/mark23/train.parquet')
df


import pandas as pd

def optimize_dataframe_memory(df):
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type).startswith('float'):
            if df[col].isnull().any(): # NaN varsa float'ta kalmalı
                continue # NaN içeren float sütunları şimdilik dönüştürmeyelim
            min_val = df[col].min()
            max_val = df[col].max()
            if min_val > np.finfo(np.float32).min and max_val < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
        elif str(col_type).startswith('int'):
            min_val = df[col].min()
            max_val = df[col].max()
            if min_val > np.iinfo(np.int8).min and max_val < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif min_val > np.iinfo(np.int16).min and max_val < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif min_val > np.iinfo(np.int32).min and max_val < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
    return df

# Örneğin train.parquet dosyasını yüklerken


df.head()


df0= df.copy()
df1 = df.copy()
df2 = df.copy()
df3 = df.copy()
df0


df.info()


df.head(5)


df.tail(5)


df.sample(10)


df["bid_qty"].value_counts()


df["ask_qty"].value_counts()


df["buy_qty"].value_counts()


df["sell_qty"].value_counts()


df["volume"].value_counts()


df["X1"].value_counts()


def unique_values(df, columns):
    for column_name in columns:
        print(f"Column: {column_name}\n{'-'*30}")

        value_counts_result = df[column_name].value_counts() # Doğrudan bu değişkeni kullanabiliriz

        print(f"Unique Values ({len(value_counts_result)})\n")
        print(f"Value Counts:\n{value_counts_result}\n{'='*40}\n")


def unique_values(df,columns):
    for column_name in columns:
        print(f"Column:{column_name}\n{'-'*30}")
        unique_vals = df[column_name].value_counts() 
        value_counts = df[column_name].value_counts()
        print(f"unique Values({len(unique_vals)}\n)")
        print(f"Value Counts:\n{value_counts}\n{'='*40}\n")


for feature in df.columns:
    if df[feature].dtype=="object":
        print(feature,df[feature].nunique())


df.describe()


df.info()


df.describe().T


df.isnull().sum()


df.duplicated().sum()


df.fillna(df.mean())


df.resample('h').mean()


df['X884'].shift(1) 


df['bid_qty'].rolling(window=5).mean()


df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
# One-hot encoding for categorical time features if necessary


df_handled_manual = df.copy() # Orijinal DataFrame'i korumak için bir kopyasını alalım

# Maksimum/minimum eşik değerleri belirleyelim (veri dağılımına göre ayarlanabilir)
# Örneğin, 1e10 (10 milyar) veya verinin 99. yüzdelik dilimi gibi
MAX_VALUE = 1e10
MIN_VALUE = -1e10

# Sonsuzlukları belirli bir değerle değiştirme
for col in df_handled_manual.select_dtypes(include=np.number).columns:
    df_handled_manual[col] = df_handled_manual[col].replace(np.inf, MAX_VALUE)
    df_handled_manual[col] = df_handled_manual[col].replace(-np.inf, MIN_VALUE)

# NaN değerleri için ek olarak doldurma yapılabilir (eğer henüz yapılmadıysa)
# df_handled_manual = df_handled_manual.fillna(df_handled_manual.mean(numeric_only=True))

print("\nManuel Değer Atama Kullanılarak Inf Değerleri İşlendikten Sonra:")
print(df_handled_manual.head())

# İşlem sonrası sonsuz değer kontrolü
print("\nİşlem Sonrası Sonsuz Değer Kontrolü:")
print((df_handled_manual == np.inf).sum().sum())
print((df_handled_manual == -np.inf).sum().sum())


import pandas as pd

def optimize_dataframe_memory(df):
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type).startswith('float'):
            if df[col].isnull().any(): # NaN varsa float'ta kalmalı
                continue # NaN içeren float sütunları şimdilik dönüştürmeyelim
            min_val = df[col].min()
            max_val = df[col].max()
            if min_val > np.finfo(np.float32).min and max_val < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
        elif str(col_type).startswith('int'):
            min_val = df[col].min()
            max_val = df[col].max()
            if min_val > np.iinfo(np.int8).min and max_val < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif min_val > np.iinfo(np.int16).min and max_val < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif min_val > np.iinfo(np.int32).min and max_val < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
    return df

# Örneğin train.parquet dosyasını yüklerken:
  


from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split


df.head(2)


X=df2.drop("label", axis=1)
y=df2.label


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=101)

print("Train features shape : ", X_train.shape)
print("Train target shape   : ", y_train.shape)
print("Test features shape  : ", X_test.shape)
print("Test target shape    : ", y_test.shape)


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor # This is the boosting model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



print("Infinite values in X_train:", np.isinf(X_train).sum())
print("Infinite values in X_test:", np.isinf(X_test).sum())


print("Max value in X_train:", X_train.max())
print("Min value in X_train:", X_train.min())
print("Max value in X_test:", X_test.max())
print("Min value in X_test:", X_test.min())


print("Max values per column in X_train:\n", X_train.max())
print("Min values per column in X_train:\n", X_train.min())


model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)




print(f"Shape of X_train BEFORE imputation: {X_train.shape}")
print(f"Number of NaN values in X_train BEFORE imputation:\n{X_train.isna().sum()}")
print(f"Number of Infinite values in X_train BEFORE imputation:\n{(X_train == np.inf).sum() + (X_train == -np.inf).sum()}")


# For X_train
finite_mask_train = np.isfinite(X_train).all(axis=1)
X_train = X_train[finite_mask_train]
y_train = y_train[finite_mask_train] # Remember to apply to y_train as well!

# For X_test
finite_mask_test = np.isfinite(X_test).all(axis=1)
X_test = X_test[finite_mask_test]
y_test = y_test[finite_mask_test] # Remember to apply to y_test as well!


# After X_train = X_train.astype(float) and X_train.replace()
# Find columns that are all NaN in the training set
all_nan_cols = X_train.columns[X_train.isnull().all()].tolist()
if all_nan_cols:
    print(f"Dropping columns that are entirely NaN in X_train: {all_nan_cols}")
    X_train.drop(columns=all_nan_cols, inplace=True)
    X_test.drop(columns=all_nan_cols, inplace=True) # Apply to test set as well!
    # Update original_cols for later DataFrame conversion if needed
    # original_cols = [col for col in original_cols if col not in all_nan_cols]


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# --- 1. Load your actual data here instead of generating dummy data ---
# Example: df = pd.read_csv('your_data.csv')
# X = df.drop('target_column', axis=1)
# y = df['target_column']

# For demonstration, let's create data that will cause the overflow
np.random.seed(42)
num_samples = 100
num_features = 5
X = pd.DataFrame(np.random.rand(num_samples, num_features), columns=[f'feature_{i}' for i in range(num_features)])
y = 2 * X['feature_0'] + 3 * X['feature_1'] - 0.5 * X['feature_2'] + np.random.randn(num_samples) * 0.5

# Intentionally introduce some very large numbers and infinities
X.iloc[5, 0] = 1e300
X.iloc[10, 2] = -1e250
X.iloc[15, 4] = np.inf
X.iloc[20, 1] = -np.inf

# New: Introduce a column that will become entirely NaN in X_train
# Let's say feature_0 is problematic in the first 80 rows (mostly in train)
# We make it so that for some split, it could be entirely inf/nan in the train set.
X.iloc[np.random.choice(X.index, 20, replace=False), 0] = np.inf


# --- 2. Split data into training and testing sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- DEBUGGING: Initial Check ---
print(f"--- Initial Data State (before any processing) ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_train contains NaN: {X_train.isnull().any().any()}")
print(f"X_train contains Inf: {np.isinf(X_train).any().any()}")
if X_train.shape[0] > 0:
    print(f"Max value in X_train: {X_train.max().max()}")
    print(f"Min value in X_train: {X_train.min().min()}")
print("-" * 40)


# --- 3. Handle infinite and very large values (Replacing with NaN then Imputing) ---

# IMPORTANT: Ensure data is float type for proper NaN/inf handling
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# Replace explicit infinite values with NaN
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

# --- DEBUGGING: After NaN replacement ---
print(f"--- After Replacing Inf with NaN ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_train contains NaN: {X_train.isnull().any().any()}")
print(f"X_train contains Inf (should be False): {np.isinf(X_train).any().any()}")
if X_train.shape[0] > 0:
    print(f"Max value in X_train: {X_train.max().max()}")
    print(f"Min value in X_train: {X_train.min().min()}")
print("-" * 40)

# --- FIX: Drop columns that are entirely NaN in the training set ---
# This is crucial if some columns became all NaNs after the replacement step.
all_nan_cols = X_train.columns[X_train.isnull().all()].tolist()
if all_nan_cols:
    print(f"Dropping columns that are entirely NaN in X_train: {all_nan_cols}")
    X_train.drop(columns=all_nan_cols, inplace=True)
    # Ensure to drop the same columns from the test set to maintain consistency
    X_test.drop(columns=all_nan_cols, inplace=True)
    print(f"New X_train shape after dropping all-NaN columns: {X_train.shape}")
    print(f"New X_test shape after dropping all-NaN columns: {X_test.shape}")
print("-" * 40)


# Define an imputer (e.g., mean imputation)
imputer = SimpleImputer(strategy='mean')

# Fit imputer ONLY on X_train to prevent data leakage
X_train_processed = imputer.fit_transform(X_train)
X_test_processed = imputer.transform(X_test)

# --- DEBUGGING: After Imputation ---
print(f"--- After Imputation (before float64 cast) ---")
print(f"X_train_processed shape: {X_train_processed.shape}")
print(f"X_train_processed contains NaN: {np.isnan(X_train_processed).any()}") # Should now be False
print(f"X_train_processed contains Inf: {np.isinf(X_train_processed).any()}")
if X_train_processed.shape[0] > 0:
    print(f"Max value in X_train_processed: {X_train_processed.max()}")
    print(f"Min value in X_train_processed: {X_train_processed.min()}")
print("-" * 40)

# --- Ensure processed arrays are float64 ---
X_train_processed = X_train_processed.astype(np.float64)
X_test_processed = X_test_processed.astype(np.float64)

# --- DEBUGGING: After float64 cast ---
print(f"--- After Explicit float64 Cast ---")
print(f"X_train_processed dtype: {X_train_processed.dtype}")
print(f"X_train_processed contains NaN: {np.isnan(X_train_processed).any()}") # Should be False
print(f"X_train_processed contains Inf: {np.isinf(X_train_processed).any()}") # Should be False
if X_train_processed.shape[0] > 0:
    print(f"Max value in X_train_processed: {X_train_processed.max()}")
    print(f"Min value in X_train_processed: {X_train_processed.min()}")
print("-" * 40)


# --- Apply Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)

# --- DEBUGGING: After Scaling ---
print(f"--- After Scaling ---")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_train_scaled contains NaN: {np.isnan(X_train_scaled).any()}") # Should be False
print(f"X_train_scaled contains Inf: {np.isinf(X_train_scaled).any()}") # Should be False
if X_train_scaled.shape[0] > 0:
    print(f"Max value in X_train_scaled: {X_train_scaled.max()}")
    print(f"Min value in X_train_scaled: {X_train_scaled.min()}")
print("-" * 40)


# --- 4. Instantiate and train the boosting model ---
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train_scaled, y_train)

# --- 5. Make predictions ---
y_pred = model.predict(X_test_scaled)
y_train_pred = model.predict(X_train_scaled)

# --- 6. Calculate and store scores ---
scores = {
    "train": {
        "R2": r2_score(y_train, y_train_pred),
        "mae": mean_absolute_error(y_train, y_train_pred),
        "mse": mean_squared_error(y_train, y_train_pred),
        "rmse": np.sqrt(mean_squared_error(y_train, y_train_pred))
    },
    "test": {
        "R2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
    }
}

# --- 7. Return as DataFrame ---
scores_df = pd.DataFrame(scores)
print(scores_df)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# --- 1. Load your actual data here instead of generating dummy data ---
# Example: df = pd.read_csv('your_data.csv')
# X = df.drop('target_column', axis=1)
# y = df['target_column']

# For demonstration, let's create data that will cause the overflow
np.random.seed(42)
num_samples = 100
num_features = 5
X = pd.DataFrame(np.random.rand(num_samples, num_features), columns=[f'feature_{i}' for i in range(num_features)])
y = 2 * X['feature_0'] + 3 * X['feature_1'] - 0.5 * X['feature_2'] + np.random.randn(num_samples) * 0.5

# Intentionally introduce some very large numbers and infinities
X.iloc[5, 0] = 1e300
X.iloc[10, 2] = -1e250
X.iloc[15, 4] = np.inf
X.iloc[20, 1] = -np.inf

# New: Introduce a column that will become entirely NaN in X_train
# Let's say feature_0 is problematic in the first 80 rows (mostly in train)
# We make it so that for some split, it could be entirely inf/nan in the train set.
X.iloc[np.random.choice(X.index, 20, replace=False), 0] = np.inf


# --- 2. Split data into training and testing sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- DEBUGGING: Initial Check ---
print(f"--- Initial Data State (before any processing) ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_train contains NaN: {X_train.isnull().any().any()}")
print(f"X_train contains Inf: {np.isinf(X_train).any().any()}")
if X_train.shape[0] > 0:
    print(f"Max value in X_train: {X_train.max().max()}")
    print(f"Min value in X_train: {X_train.min().min()}")
print("-" * 40)


# --- 3. Handle infinite and very large values (Replacing with NaN then Imputing) ---

# IMPORTANT: Ensure data is float type for proper NaN/inf handling
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# Replace explicit infinite values with NaN
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

# --- DEBUGGING: After NaN replacement ---
print(f"--- After Replacing Inf with NaN ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_train contains NaN: {X_train.isnull().any().any()}")
print(f"X_train contains Inf (should be False): {np.isinf(X_train).any().any()}")
if X_train.shape[0] > 0:
    print(f"Max value in X_train: {X_train.max().max()}")
    print(f"Min value in X_train: {X_train.min().min()}")
print("-" * 40)

# --- FIX: Drop columns that are entirely NaN in the training set ---
# This is crucial if some columns became all NaNs after the replacement step.
all_nan_cols = X_train.columns[X_train.isnull().all()].tolist()
if all_nan_cols:
    print(f"Dropping columns that are entirely NaN in X_train: {all_nan_cols}")
    X_train.drop(columns=all_nan_cols, inplace=True)
    # Ensure to drop the same columns from the test set to maintain consistency
    X_test.drop(columns=all_nan_cols, inplace=True)
    print(f"New X_train shape after dropping all-NaN columns: {X_train.shape}")
    print(f"New X_test shape after dropping all-NaN columns: {X_test.shape}")
print("-" * 40)


# Define an imputer (e.g., mean imputation)
imputer = SimpleImputer(strategy='mean')

# Fit imputer ONLY on X_train to prevent data leakage
X_train_processed = imputer.fit_transform(X_train)
X_test_processed = imputer.transform(X_test)

# --- DEBUGGING: After Imputation ---
print(f"--- After Imputation (before float64 cast) ---")
print(f"X_train_processed shape: {X_train_processed.shape}")
print(f"X_train_processed contains NaN: {np.isnan(X_train_processed).any()}") # Should now be False
print(f"X_train_processed contains Inf: {np.isinf(X_train_processed).any()}")
if X_train_processed.shape[0] > 0:
    print(f"Max value in X_train_processed: {X_train_processed.max()}")
    print(f"Min value in X_train_processed: {X_train_processed.min()}")
print("-" * 40)

# --- Ensure processed arrays are float64 ---
X_train_processed = X_train_processed.astype(np.float64)
X_test_processed = X_test_processed.astype(np.float64)

# --- DEBUGGING: After float64 cast ---
print(f"--- After Explicit float64 Cast ---")
print(f"X_train_processed dtype: {X_train_processed.dtype}")
print(f"X_train_processed contains NaN: {np.isnan(X_train_processed).any()}") # Should be False
print(f"X_train_processed contains Inf: {np.isinf(X_train_processed).any()}") # Should be False
if X_train_processed.shape[0] > 0:
    print(f"Max value in X_train_processed: {X_train_processed.max()}")
    print(f"Min value in X_train_processed: {X_train_processed.min()}")
print("-" * 40)


# --- Apply Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)

# --- DEBUGGING: After Scaling ---
print(f"--- After Scaling ---")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_train_scaled contains NaN: {np.isnan(X_train_scaled).any()}") # Should be False
print(f"X_train_scaled contains Inf: {np.isinf(X_train_scaled).any()}") # Should be False
if X_train_scaled.shape[0] > 0:
    print(f"Max value in X_train_scaled: {X_train_scaled.max()}")
    print(f"Min value in X_train_scaled: {X_train_scaled.min()}")
print("-" * 40)


# --- 4. Instantiate and train the boosting model ---
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train_scaled, y_train)

# --- 5. Make predictions ---
y_pred = model.predict(X_test_scaled)
y_train_pred = model.predict(X_train_scaled)

# --- 6. Calculate and store scores ---
scores = {
    "train": {
        "R2": r2_score(y_train, y_train_pred),
        "mae": mean_absolute_error(y_train, y_train_pred),
        "mse": mean_squared_error(y_train, y_train_pred),
        "rmse": np.sqrt(mean_squared_error(y_train, y_train_pred))
    },
    "test": {
        "R2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
    }
}

# --- 7. Return as DataFrame ---
scores_df = pd.DataFrame(scores)
print(scores_df)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- 1. Verinizi buraya yükleyin ---
# Örnek: df = pd.read_csv('veriniz.csv')
# X = df.drop('hedef_sütun', axis=1)
# y = df['hedef_sütun']

# Örnek veri oluşturma (kendi verinizi buraya yapıştırın)
np.random.seed(42)
num_samples = 100
num_features = 5
X = pd.DataFrame(np.random.rand(num_samples, num_features), columns=[f'feature_{i}' for i in range(num_features)])
y = 2 * X['feature_0'] + 3 * X['feature_1'] - 0.5 * X['feature_2'] + np.random.randn(num_samples) * 0.5

# Test için kasıtlı olarak NaN ve Inf değerler ekleyelim
X.iloc[5, 0] = np.nan
X.iloc[10, 2] = np.inf
X.iloc[15, 4] = -np.inf
X.iloc[20, 1] = 1e40 # float32 overflow yapabilecek çok büyük bir sayı
y.iloc[7] = np.nan # y'ye de NaN ekleyelim

# --- DEBUGGING: Başlangıç Veri Durumu ---
print(f"--- Başlangıç Veri Durumu (İşleme Öncesi) ---")
print(f"X boyutu: {X.shape}")
print(f"y boyutu: {y.shape}")
print(f"X'teki toplam NaN sayısı: {X.isnull().sum().sum()}")
print(f"X'teki toplam Inf sayısı: {np.isinf(X).sum().sum()}")
print(f"y'deki toplam NaN sayısı: {y.isnull().sum()}")
print("-" * 40)

# --- 2. Tüm NaN ve Inf değerleri temizle (ana FIX) ---

# Adım 1: Tüm sonsuz değerleri NaN'a dönüştür (eğer hala varsa)
# Bu, .dropna() ile birlikte çalışmak için önemlidir.
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Adım 2: X ve y'yi birleştirerek birlikte NaN içeren satırları bul
# Bu, X'teki veya y'deki NaN'ları içeren tüm satırları kaldırmamızı sağlar.
df_combined = pd.concat([X, y.rename('target')], axis=1) # y'yi DataFrame'e ekle
df_cleaned = df_combined.dropna() # Tüm NaN içeren satırları sil

# Temizlenmiş veriyi X ve y'ye geri ayır
X_cleaned = df_cleaned.drop('target', axis=1)
y_cleaned = df_cleaned['target']

# --- DEBUGGING: Temizleme Sonrası Durum ---
print(f"--- Temizleme Sonrası Veri Durumu ---")
print(f"X_cleaned boyutu: {X_cleaned.shape}")
print(f"y_cleaned boyutu: {y_cleaned.shape}")
print(f"X_cleaned'daki toplam NaN sayısı: {X_cleaned.isnull().sum().sum()}")
print(f"X_cleaned'daki toplam Inf sayısı: {np.isinf(X_cleaned).sum().sum()}")
print(f"y_cleaned'daki toplam NaN sayısı: {y_cleaned.isnull().sum()}")
print("-" * 40)

# --- 3. Veriyi eğitim ve test setlerine ayır ---
# Temizlenmiş veriyi kullanıyoruz
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# --- 4. Veriyi ölçekle (StandardScaler) ---
# Modelin sayısal hassasiyet sorunları yaşamaması için hala iyi bir uygulamadır.
# Standard Scaler, NaN içermeyen NumPy dizileri bekler.
scaler = StandardScaler()

# NumPy dizilerine dönüştürerek scikit-learn'ün uyarılarını önle
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

# --- 5. Boosting modelini başlat ve eğit ---
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train_scaled, y_train)

# --- 6. Tahminler yap ---
y_pred = model.predict(X_test_scaled)
y_train_pred = model.predict(X_train_scaled)

# --- 7. Skorları hesapla ve depola ---
scores = {
    "train": {
        "R2": r2_score(y_train, y_train_pred),
        "mae": mean_absolute_error(y_train, y_train_pred),
        "mse": mean_squared_error(y_train, y_train_pred),
        "rmse": np.sqrt(mean_squared_error(y_train, y_train_pred))
    },
    "test": {
        "R2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
    }
}

# --- 8. DataFrame olarak döndür ---
scores_df = pd.DataFrame(scores)
print("\n--- Model Performans Skorları ---")
print(scores_df)


cat_features = X.select_dtypes("object").columns
cat_features 


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder


ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

column_trans = make_column_transformer((ord_enc, cat_features), remainder='passthrough')


from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostRegressor


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder # İhtiyacınıza göre diğer encoder'lar

# --- Örnek Veri Oluşturma (Gerçek verinizle değiştirin) ---
# Hem sayısal hem de kategorik sütunlar içeren bir DataFrame oluşturalım
data = {
    'numerical_feature_1': np.random.rand(100) * 100,
    'numerical_feature_2': np.random.rand(100) * 50,
    'ordinal_feature': np.random.choice(['low', 'medium', 'high'], 100),
    'nominal_feature': np.random.choice(['A', 'B', 'C', 'D'], 100),
    'target': np.random.rand(100) * 200
}
X = pd.DataFrame(data)
y = X['target']
X = X.drop('target', axis=1)

# Eksik değer ekleyelim ki temizleme adımını test edebilelim
X.iloc[5, 0] = np.nan
X.iloc[10, 2] = 'medium' # Ordinal feature'a örnek
X.iloc[15, 3] = 'C' # Nominal feature'a örnek
y.iloc[7] = np.nan

# --- Veri Temizleme (Önceki yanıttan alınmıştır) ---
# NaN ve sonsuz değerleri temizleme
# Bu adımı, ColumnTransformer'dan önce yapmalısınız çünkü transformer'lar genellikle NaN'ları kendi başlarına işlemezler
# Ancak, OrdinalEncoder varsayılan olarak NaN'ları geçirir. Eğer NaN'ları encode etmek isterseniz,
# handle_missing='use_encoded_value' ve encoded_missing_value parametrelerini kullanmalısınız.
# Basitlik adına, burada yine tüm NaN içeren satırları siliyoruz.

# Adım 1: Tüm sonsuz değerleri NaN'a dönüştür (eğer hala varsa)
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Adım 2: X ve y'yi birleştirerek birlikte NaN içeren satırları bul ve sil
df_combined = pd.concat([X, y.rename('target')], axis=1)
df_cleaned = df_combined.dropna()

# Temizlenmiş veriyi X ve y'ye geri ayır
X_cleaned = df_cleaned.drop('target', axis=1)
y_cleaned = df_cleaned['target']

print(f"Temizleme sonrası X_cleaned boyutu: {X_cleaned.shape}")
print(f"Temizleme sonrası y_cleaned boyutu: {y_cleaned.shape}")
print(f"X_cleaned'daki NaN sayısı: {X_cleaned.isnull().sum().sum()}")
print(f"y_cleaned'daki NaN sayısı: {y_cleaned.isnull().sum()}")
print("-" * 40)


# --- Eğitim ve Test Setlerine Ayırma ---
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# --- Eksik Kodu Buraya Tanımlayalım: column_trans ---

# Sütunları tanımlayın
numerical_cols = ['numerical_feature_1', 'numerical_feature_2']
ordinal_cols = ['ordinal_feature']
nominal_cols = ['nominal_feature'] # Eğer nominal kategorik sütunlarınız varsa

# OrdinalEncoder için kategorilerin sırasını belirtmek önemlidir
# Eğer belirli bir sıra varsa, onu burada tanımlayın
# Aksi takdirde, OrdinalEncoder varsayılan olarak kategorileri alfabetik sıraya göre atar.
# Örnek: categories=[['low', 'medium', 'high']]
ordinal_categories_order = [['low', 'medium', 'high']] # Kendi kategorilerinize göre ayarlayın

# Her bir sütun tipine uygulanacak transformer'ları tanımlayın
# Sayısal sütunlar için bir StandardScaler
numerical_transformer = StandardScaler()

# Sıralı kategorik sütunlar için bir OrdinalEncoder
ordinal_transformer = OrdinalEncoder(categories=ordinal_categories_order) # handle_unknown='use_encoded_value', unknown_value=-1 gibi ayarlar eklenebilir

# Nominal kategorik sütunlar için bir OneHotEncoder
nominal_transformer = OneHotEncoder(handle_unknown='ignore') # handle_unknown='ignore' bilinmeyen kategorileri sıfırlarla kodlar

# ColumnTransformer'ı oluşturun
# 'remainder='passthrough'' belirtmezseniz, belirtilmeyen sütunlar otomatik olarak düşürülür.
column_trans = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('ord', ordinal_transformer, ordinal_cols),
        ('nom', nominal_transformer, nominal_cols) # Eğer nominal sütunlarınız varsa ekleyin
    ],
    remainder='passthrough' # İşlenmeyen diğer tüm sütunları olduğu gibi bırakır
)

# --- Pipeline Tanımı ---
operations = [
    ("preprocessor", column_trans), # column_trans'ı burada kullanıyoruz
    ("Ada_model", AdaBoostRegressor(random_state=101))
]

pipe_model = Pipeline(steps=operations)

# --- Modeli Eğitme ---
pipe_model.fit(X_train, y_train)

# --- Tahmin ve Değerlendirme ---
y_pred_train = pipe_model.predict(X_train)
y_pred_test = pipe_model.predict(X_test)

print("\n--- Eğitim Seti Performansı ---")
print(f"R2 Skoru: {r2_score(y_train, y_pred_train):.4f}")
print(f"MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")
print(f"MSE: {mean_squared_error(y_train, y_pred_train):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")

print("\n--- Test Seti Performansı ---")
print(f"R2 Skoru: {r2_score(y_test, y_pred_test):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_test):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")


from sklearn.model_selection import cross_validate, cross_val_score

operations = [("OrdinalEncoder", column_trans),
              ("Ada_model", AdaBoostRegressor(random_state=101))]

model = Pipeline(steps=operations)

scores = cross_validate(model,
                        X_train,
                        y_train,
                        scoring=[
                            'r2', 'neg_mean_absolute_error',
                            'neg_mean_squared_error',
                            'neg_root_mean_squared_error'
                        ],
                        cv=10,
                        return_train_score=True)
pd.DataFrame(scores)
pd.DataFrame(scores).mean()[2:]








from sklearn.ensemble import GradientBoostingRegressor


from sklearn.ensemble import GradientBoostingRegressor

operations = [("OrdinalEncoder", column_trans), ("GB_model", GradientBoostingRegressor(random_state=101))]

pipe_model = Pipeline(steps=operations)

pipe_model.fit(X_train, y_train)





operations = [("OrdinalEncoder", column_trans), ("GB_model", GradientBoostingRegressor(random_state=101))]

model = Pipeline(steps=operations)
scores = cross_validate(model, X_train, y_train, scoring=['r2', 
            'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv =10,
                       return_train_score=True)

pd.DataFrame(scores).mean()[2:]


param_grid = {"GB_model__n_estimators":[35,50], 
              "GB_model__subsample":[0.7, 0.8, 1], 
              "GB_model__max_features" : [4,5,6],
              "GB_model__learning_rate": [0.02, 0.03,0.05], 
              'GB_model__max_depth':[1,2],
              'GB_model__min_samples_split':[1,2],
              'GB_model__min_samples_leaf':[1,2]}

# classificationdan en önemli farkı loss='squared_error'dür. Classifciationda bu logloss'tu.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # Eğer imputation'a ihtiyacınız olursa

# --- Örnek Veri Oluşturma (Gerçek verinizle değiştirin) ---
# Hem sayısal hem de kategorik sütunlar içeren bir DataFrame oluşturalım
data = {
    'numerical_feature_1': np.random.rand(100) * 100,
    'numerical_feature_2': np.random.rand(100) * 50,
    'ordinal_feature': np.random.choice(['low', 'medium', 'high'], 100),
    'nominal_feature': np.random.choice(['A', 'B', 'C', 'D'], 100),
    'target': np.random.rand(100) * 200
}
X = pd.DataFrame(data)
y = X['target']
X = X.drop('target', axis=1)

# Eksik değer ekleyelim ki temizleme adımını test edebilelim
X.iloc[5, 0] = np.nan
X.iloc[10, 2] = 'medium' # Ordinal feature'a örnek
X.iloc[15, 3] = 'C' # Nominal feature'a örnek
y.iloc[7] = np.nan

# --- Veri Temizleme (Önceki yanıttan alınmıştır - NaN ve sonsuz değerleri temizleme) ---
X.replace([np.inf, -np.inf], np.nan, inplace=True)
df_combined = pd.concat([X, y.rename('target')], axis=1)
df_cleaned = df_combined.dropna()

X_cleaned = df_cleaned.drop('target', axis=1)
y_cleaned = df_cleaned['target']

# --- Eğitim ve Test Setlerine Ayırma ---
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# --- ColumnTransformer Tanımı (Önceki yanıttan alınmıştır) ---
numerical_cols = ['numerical_feature_1', 'numerical_feature_2']
ordinal_cols = ['ordinal_feature']
nominal_cols = ['nominal_feature']

ordinal_categories_order = [['low', 'medium', 'high']]

numerical_transformer = StandardScaler()
ordinal_transformer = OrdinalEncoder(categories=ordinal_categories_order)
nominal_transformer = OneHotEncoder(handle_unknown='ignore')

column_trans = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('ord', ordinal_transformer, ordinal_cols),
        ('nom', nominal_transformer, nominal_cols)
    ],
    remainder='passthrough'
)

# --- Pipeline Tanımı ---
# Not: Pipeline adını 'OrdinalEncoder' yerine 'preprocessor' olarak değiştirdim,
# bu daha açıklayıcı ve daha önceki örnekle tutarlı.
# Eğer ordinal encoder'a özel parametreleri optimize etmek isterseniz,
# 'preprocessor__ord__...' şeklinde parametre ızgarasına eklemeniz gerekir.
operations = [
    ("preprocessor", column_trans),
    ("GB_model", GradientBoostingRegressor(random_state=101))
]

model = Pipeline(steps=operations)

# --- EKSİK KOD: param_grid Tanımı ---
# GradientBoostingRegressor için denemek istediğiniz hiperparametreleri buraya ekleyin.
# Pipeline'daki model adımının adı "GB_model" olduğu için,
# parametre adları "GB_model__" ile başlamalıdır.

param_grid = {
    'GB_model__n_estimators': [50, 100, 200],  # Denenecek ağaç sayısı
    'GB_model__learning_rate': [0.01, 0.1, 0.2], # Her ağacın katkısı
    'GB_model__max_depth': [3, 4, 5],          # Her ağacın maksimum derinliği
    # 'GB_model__subsample': [0.8, 1.0],         # Her ağacı eğitmek için kullanılan örneklerin oranı
    # 'GB_model__min_samples_split': [2, 5],     # Bir düğümü bölmek için gereken minimum örnek sayısı
    # Eğer ColumnTransformer içindeki bir transformer'ın parametresini optimize etmek isterseniz:
    # 'preprocessor__num__with_mean': [True, False], # StandardScaler için
    # 'preprocessor__ord__handle_unknown': ['use_encoded_value'], # OrdinalEncoder için
    # 'preprocessor__ord__unknown_value': [-1] # OrdinalEncoder için, handle_unknown 'use_encoded_value' ise
}

# --- GridSearchCV Tanımı ve Eğitimi ---
grid_model = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error', # Daha yüksek skor daha iyidir, RMSE'yi minimize etmek için negatifini kullanırız
                          cv=5,                                 # 5 katlı çapraz doğrulama
                          n_jobs=-1,                            # Tüm işlemcileri kullan
                          return_train_score=True)

# Modeli eğitin
grid_model.fit(X_train, y_train)

# --- En İyi Parametreler ve Skor ---
print("\n--- GridSearchCV Sonuçları ---")
print(f"En İyi Parametreler: {grid_model.best_params_}")
print(f"En İyi RMSE Skoru (Negatif): {grid_model.best_score_:.4f}") # 'neg_root_mean_squared_error' olduğu için negatif olacak
print(f"En İyi RMSE Skoru (Pozitif): {-grid_model.best_score_:.4f}")

# En iyi modeli al
best_model = grid_model.best_estimator_

# Test seti üzerinde tahminler yap
y_pred_test = best_model.predict(X_test)

# Test seti performansını değerlendir
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("\n--- Test Seti Performansı (En İyi Model) ---")
print(f"R2 Skoru: {r2_score(y_test, y_pred_test):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_test):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")


grid_model.best_params_


grid_model.best_estimator_


grid_model.best_score_


operations = [("OrdinalEncoder", column_trans),
              ("GB_model",
               GradientBoostingRegressor(learning_rate=0.05, max_depth=2, max_features=6,
                          n_estimators=50, random_state=101, subsample=0.7))]

model = Pipeline(steps=operations)

scores = cross_validate(model,
                        X_train,
                        y_train,
                        scoring=[
                            'r2', 'neg_mean_absolute_error',
                            'neg_mean_squared_error',
                            'neg_root_mean_squared_error'
                        ],
                        cv=10,
                        return_train_score=True)
pd.DataFrame(scores).mean()[2:]


pip install --upgrade scikit-learn


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # Eğer imputation'a ihtiyacınız olursa
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error # mean_squared_error'ı import etmeyi unutmayın

# --- Örnek Veri Oluşturma (Gerçek verinizle değiştirin) ---
# Hem sayısal hem de kategorik sütunlar içeren bir DataFrame oluşturalım
data = {
    'numerical_feature_1': np.random.rand(100) * 100,
    'numerical_feature_2': np.random.rand(100) * 50,
    'ordinal_feature': np.random.choice(['low', 'medium', 'high'], 100),
    'nominal_feature': np.random.choice(['A', 'B', 'C', 'D'], 100),
    'target': np.random.rand(100) * 200
}
X = pd.DataFrame(data)
y = X['target']
X = X.drop('target', axis=1)

# Eksik değer ekleyelim ki temizleme adımını test edebilelim
X.iloc[5, 0] = np.nan
X.iloc[10, 2] = 'medium' # Ordinal feature'a örnek
X.iloc[15, 3] = 'C' # Nominal feature'a örnek
y.iloc[7] = np.nan

# --- Veri Temizleme (NaN ve sonsuz değerleri temizleme) ---
X.replace([np.inf, -np.inf], np.nan, inplace=True)
df_combined = pd.concat([X, y.rename('target')], axis=1)
df_cleaned = df_combined.dropna()

X_cleaned = df_cleaned.drop('target', axis=1)
y_cleaned = df_cleaned['target']

# --- Eğitim ve Test Setlerine Ayırma ---
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# --- ColumnTransformer Tanımı ---
numerical_cols = ['numerical_feature_1', 'numerical_feature_2']
ordinal_cols = ['ordinal_feature']
nominal_cols = ['nominal_feature']

ordinal_categories_order = [['low', 'medium', 'high']]

numerical_transformer = StandardScaler()
ordinal_transformer = OrdinalEncoder(categories=ordinal_categories_order)
nominal_transformer = OneHotEncoder(handle_unknown='ignore')

column_trans = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('ord', ordinal_transformer, ordinal_cols),
        ('nom', nominal_transformer, nominal_cols)
    ],
    remainder='passthrough'
)

# --- Pipeline Tanımı ---
operations = [
    ("preprocessor", column_trans),
    ("GB_model", GradientBoostingRegressor(random_state=101))
]

model = Pipeline(steps=operations)

# --- param_grid Tanımı ---
param_grid = {
    'GB_model__n_estimators': [50, 100, 200],
    'GB_model__learning_rate': [0.01, 0.1, 0.2],
    'GB_model__max_depth': [3, 4, 5],
}

# --- GridSearchCV Tanımı ve Eğitimi ---
grid_model = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=5,
                          n_jobs=-1,
                          return_train_score=True)

grid_model.fit(X_train, y_train) # Eğitimi burada yapıyoruz.

# --- En İyi Modeli Kullanarak Tahminler ---
best_model = grid_model.best_estimator_
y_pred_test = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train) # Eğitim seti için de tahmin yapalım

# --- Metriklerin Hesaplanması ---
print("\n--- Model Performans Skorları ---")

# Eğitim seti metrikleri
grad_r2_train = r2_score(y_train, y_pred_train)
grad_mae_train = mean_absolute_error(y_train, y_pred_train)
grad_mse_train = mean_squared_error(y_train, y_pred_train)
grad_rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train)) # FIX: np.sqrt kullanıldı

# Test seti metrikleri
grad_r2_test = r2_score(y_test, y_pred_test)
grad_mae_test = mean_absolute_error(y_test, y_pred_test)
grad_mse_test = mean_squared_error(y_test, y_pred_test)
grad_rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test)) # FIX: np.sqrt kullanıldı

scores = {
    "train": {
        "R2": grad_r2_train,
        "mae": grad_mae_train,
        "mse": grad_mse_train,
        "rmse": grad_rmse_train
    },
    "test": {
        "R2": grad_r2_test,
        "mae": grad_mae_test,
        "mse": grad_mse_test,
        "rmse": grad_rmse_test
    }
}
scores_df = pd.DataFrame(scores)
print(scores_df)

# --- train_val fonksiyonu eğer ayrı bir fonksiyon ise ---
# train_val fonksiyonunuzu da kontrol edin ve orada da `squared=False` kullanılıyorsa,
# onu da `np.sqrt()` ile değiştirmeniz gerekecektir.
# Örnek train_val fonksiyonu (varsayım):
def train_val(model_fitted_by_grid, X_train, y_train, X_test, y_test):
    y_train_pred = model_fitted_by_grid.predict(X_train)
    y_test_pred = model_fitted_by_grid.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred)) # FIX: np.sqrt

    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred)) # FIX: np.sqrt

    print("\n--- train_val Fonksiyonu Sonuçları ---")
    results = pd.DataFrame({
        'Metric': ['R2', 'MAE', 'MSE', 'RMSE'],
        'Train': [train_r2, train_mae, train_mse, train_rmse],
        'Test': [test_r2, test_mae, test_mse, test_rmse]
    })
    print(results)

# train_val fonksiyonunu çağır (eğer tanımlıysa ve kullanmak istiyorsanız)
# train_val(grid_model, X_train, y_train, X_test, y_test) # grid_model en iyi modeli içerir


operations = [("OrdinalEncoder", column_trans),
              ("GB_model",
               GradientBoostingRegressor(learning_rate=0.05,
                                         max_depth=2,
                                         max_features=6,
                                         n_estimators=50,
                                         random_state=101,
                                         subsample=0.7))]

pipe_model = Pipeline(steps=operations)

pipe_model.fit(X_train, y_train)


pipe_model["GB_model"].feature_importances_


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # If you need imputation
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- 1. Create Sample Data (Replace with your actual data) ---
data = {
    'numerical_feature_1': np.random.rand(100) * 100,
    'numerical_feature_2': np.random.rand(100) * 50,
    'ordinal_feature': np.random.choice(['low', 'medium', 'high'], 100),
    'nominal_feature': np.random.choice(['A', 'B', 'C', 'D'], 100),
    'target': np.random.rand(100) * 200
}
X = pd.DataFrame(data)
y = X['target']
X = X.drop('target', axis=1)

# Add some missing/infinite values for demonstration
X.iloc[5, 0] = np.nan
X.iloc[10, 2] = 'medium'
X.iloc[15, 3] = 'C'
y.iloc[7] = np.nan

# --- 2. Data Cleaning (Handling NaNs and Infs) ---
X.replace([np.inf, -np.inf], np.nan, inplace=True)
df_combined = pd.concat([X, y.rename('target')], axis=1)
df_cleaned = df_combined.dropna()

X_cleaned = df_cleaned.drop('target', axis=1)
y_cleaned = df_cleaned['target']

# --- 3. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# --- 4. ColumnTransformer Definition ---
numerical_cols = ['numerical_feature_1', 'numerical_feature_2']
ordinal_cols = ['ordinal_feature']
nominal_cols = ['nominal_feature']

ordinal_categories_order = [['low', 'medium', 'high']]

numerical_transformer = StandardScaler()
ordinal_transformer = OrdinalEncoder(categories=ordinal_categories_order)
nominal_transformer = OneHotEncoder(handle_unknown='ignore')

column_trans = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('ord', ordinal_transformer, ordinal_cols),
        ('nom', nominal_transformer, nominal_cols)
    ],
    remainder='passthrough'
)

# --- 5. Pipeline Definition ---
operations = [
    ("preprocessor", column_trans),
    ("GB_model", GradientBoostingRegressor(random_state=101))
]
model = Pipeline(steps=operations)

# --- 6. param_grid Definition ---
param_grid = {
    'GB_model__n_estimators': [50, 100, 200],
    'GB_model__learning_rate': [0.01, 0.1, 0.2],
    'GB_model__max_depth': [3, 4, 5],
}

# --- 7. GridSearchCV Definition and Training ---
grid_model = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=5,
                          n_jobs=-1,
                          return_train_score=True)

grid_model.fit(X_train, y_train)

# --- 8. Get Feature Importances and Feature Names ---
# The best_estimator_ from GridSearchCV is the fitted pipeline
fitted_pipeline = grid_model.best_estimator_

# FIX: Get the feature names AFTER the ColumnTransformer has been fitted
# You can access the 'preprocessor' step of the fitted pipeline
new_features = fitted_pipeline.named_steps['preprocessor'].get


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- 1. Create Sample Data (Replace with your actual data) ---
data = {
    'numerical_feature_1': np.random.rand(100) * 100,
    'numerical_feature_2': np.random.rand(100) * 50,
    'ordinal_feature': np.random.choice(['low', 'medium', 'high'], 100),
    'nominal_feature': np.random.choice(['A', 'B', 'C', 'D'], 100),
    'target': np.random.rand(100) * 200
}
X = pd.DataFrame(data)
y = X['target']
X = X.drop('target', axis=1)

# Add some missing/infinite values for demonstration
X.iloc[5, 0] = np.nan
X.iloc[10, 2] = 'medium'
X.iloc[15, 3] = 'C'
y.iloc[7] = np.nan

# --- 2. Data Cleaning (Handling NaNs and Infs) ---
X.replace([np.inf, -np.inf], np.nan, inplace=True)
df_combined = pd.concat([X, y.rename('target')], axis=1)
df_cleaned = df_combined.dropna()

X_cleaned = df_cleaned.drop('target', axis=1)
y_cleaned = df_cleaned['target']

# --- 3. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# --- 4. ColumnTransformer Definition ---
numerical_cols = ['numerical_feature_1', 'numerical_feature_2']
ordinal_cols = ['ordinal_feature']
nominal_cols = ['nominal_feature']

ordinal_categories_order = [['low', 'medium', 'high']]

numerical_transformer = StandardScaler()
ordinal_transformer = OrdinalEncoder(categories=ordinal_categories_order)
nominal_transformer = OneHotEncoder(handle_unknown='ignore')

column_trans = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('ord', ordinal_transformer, ordinal_cols),
        ('nom', nominal_transformer, nominal_cols)
    ],
    remainder='passthrough'
)

# --- 5. Pipeline Definition ---
operations = [
    ("preprocessor", column_trans),
    ("GB_model", GradientBoostingRegressor(random_state=101))
]
model = Pipeline(steps=operations)

# --- 6. param_grid Definition ---
param_grid = {
    'GB_model__n_estimators': [50, 100, 200],
    'GB_model__learning_rate': [0.01, 0.1, 0.2],
    'GB_model__max_depth': [3, 4, 5],
}

# --- 7. GridSearchCV Definition and Training ---
grid_model = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=5,
                          n_jobs=-1,
                          return_train_score=True)

grid_model.fit(X_train, y_train)

# --- 8. Get Feature Importances and Feature Names ---
fitted_pipeline = grid_model.best_estimator_

# FIX: Correct the method name from .get to .get_feature_names_out()
new_features = fitted_pipeline.named_steps['preprocessor'].get_feature_names_out()

# Now, use these feature names for your DataFrame index
imp_feats = pd.DataFrame(data=fitted_pipeline["GB_model"].feature_importances_,
                         columns=['Grad_Importance'],
                         index=new_features)

grad_imp_feats = imp_feats.sort_values('Grad_Importance', ascending=False)
print("\n--- Feature Importances ---")
print(grad_imp_feats)

# --- 9. Predictions and Evaluation ---
y_pred_test = fitted_pipeline.predict(X_test)
y_pred_train = fitted_pipeline.predict(X_train)

print("\n--- Model Performance Scores ---")

grad_r2_train = r2_score(y_train, y_pred


# --- 9. Predictions and Evaluation ---
y_pred_test = fitted_pipeline.predict(X_test)
y_pred_train = fitted_pipeline.predict(X_train) # This line predicts for the training set

print("\n--- Model Performance Scores ---")

grad_r2_train = r2_score(y_train, y_pred_train) # Corrected: used y_pred_train
grad_mae_train = mean_absolute_error(y_train, y_pred_train)
grad_mse_train = mean_squared_error(y_train, y_pred_train)
grad_rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train)) # Ensure y_pred_train here too

grad_r2_test = r2_score(y_test, y_pred_test)
grad_mae_test = mean_absolute_error(y_test, y_pred_test)
grad_mse_test = mean_squared_error(y_test, y_pred_test)
grad_rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

scores = {
    "train": {
        "R2": grad_r2_train,
        "mae": grad_mae_train,
        "mse": grad_mse_train,
        "rmse": grad_rmse_train
    },
    "test": {
        "R2": grad_r2_test,
        "mae": grad_mae_test,
        "mse": grad_mse_test,
        "rmse": grad_rmse_test
    }
}
scores_df = pd.DataFrame(scores)
print(scores_df)


ax = sns.barplot(data=grad_imp_feats, x=grad_imp_feats.index, y='Grad_Importance')
ax.bar_label(ax.containers[0],fmt="%.3f")
plt.xticks(rotation=90);


pip install xgboost


from xgboost import XGBRegressor


operations = [("OrdinalEncoder", column_trans), ("XGB_model", XGBRegressor(random_state=101))]

pipe_model = Pipeline(steps=operations)

pipe_model.fit(X_train, y_train)


train_val(pipe_model, X_train, y_train, X_test, y_test)


operations = [("OrdinalEncoder", column_trans), ("XGB_model", XGBRegressor(random_state=101))]

model = Pipeline(steps=operations)

scores = cross_validate(model, X_train, y_train, scoring=['r2', 
            'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv =10,
                       return_train_score=True)
pd.DataFrame(scores).iloc[:, 2:].mean()


# --- param_grid Tanımı ---
# XGBoost modeliniz için denemek istediğiniz hiperparametreleri buraya ekleyin.
# Pipeline'daki model adımının adı "XGB_model" (veya "GB_model") olduğu için,
# parametre adları "XGB_model__" (veya "GB_model__") ile başlamalıdır.

param_grid = {
    'XGB_model__n_estimators': [50, 100, 200],
    'XGB_model__learning_rate': [0.01, 0.1, 0.2],
    'XGB_model__max_depth': [3, 4, 5],
    'XGB_model__colsample_bytree': [0.5, 0.8, 1]
} # <-- Make sure you have this closing curly brace here!


operations = [("OrdinalEncoder", column_trans), ("XGB_model", XGBRegressor(random_state=101))]

model = Pipeline(steps=operations)

grid_model = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=10,
                          n_jobs = -1,
                          return_train_score=True).fit(X_train, y_train)


grid_model.best_params_


grid_model.best_score_


import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# Assuming y_test and y_pred are already defined from your model's predictions

# --- Calculate Metrics for XGBoost (or whichever model you're using here) ---
XGB_r2 = r2_score(y_test, y_pred)
XGB_mae = mean_absolute_error(y_test, y_pred)
XGB_mse = mean_squared_error(y_test, y_pred)
XGB_rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # Corrected line

# And if you have a `train_val` function that also uses `squared=False`,
# make sure to update it as well:
def train_val(model_fitted_by_grid, X_train, y_train, X_test, y_test):
    y_train_pred = model_fitted_by_grid.predict(X_train)
    y_test_pred = model_fitted_by_grid.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred)) # Corrected here

    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred)) # Corrected here

    print("\n--- train_val Fonksiyonu Sonuçları ---")
    results = pd.DataFrame({
        'Metric': ['R2', 'MAE', 'MSE', 'RMSE'],
        'Train': [train_r2, train_mae, train_mse, train_rmse],
        'Test': [test_r2, test_mae, test_mse, test_rmse]
    })
    print(results)

# Call train_val (if applicable)
# train_val(grid_model, X_train, y_train, X_test, y_test)


import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# Diğer import'lar (Pipeline, GridSearchCV, vs.) burada olmalı

# ... (Önceki kodunuz: veri yükleme, temizleme, split, ColumnTransformer, Pipeline, GridSearchCV tanımı ve eğitimi) ...

# Farz edelim ki grid_model.fit(X_train, y_train) zaten çalıştırıldı
# ve en iyi modeli grid_model.best_estimator_ olarak alabiliriz.

# --- Metrik Hesaplamaları (Ana Modelin Test Seti Performansı) ---
# grid_model.predict, en iyi modeli kullanarak tahmin yapar.
y_pred = grid_model.predict(X_test) # Test seti tahminleri

XGB_R2 = r2_score(y_test, y_pred)
XGB_mae = mean_absolute_error(y_test, y_pred)
XGB_mse = mean_squared_error(y_test, y_pred)
# Hata düzeltmesi: squared=False yerine np.sqrt() kullanıyoruz
XGB_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"XGBoost Test R2: {XGB_R2:.4f}")
print(f"XGBoost Test MAE: {XGB_mae:.4f}")
print(f"XGBoost Test MSE: {XGB_mse:.4f}")
print(f"XGBoost Test RMSE: {XGB_rmse:.4f}")

# --- EKSİK KOD: train_val fonksiyonunun tanımı ---
def train_val(model_or_grid_model, X_train, y_train, X_test, y_test):
    """
    Eğitim ve test setleri üzerindeki model performansını değerlendirir.

    Parametreler:
    model_or_grid_model: Eğitilmiş bir scikit-learn modeli veya GridSearchCV objesi.
                         GridSearchCV ise en iyi modelini kullanır.
    X_train, y_train: Eğitim verisi
    X_test, y_test: Test verisi
    """
    # Eğer model_or_grid_model bir GridSearchCV objesi ise, en iyi tahminciyi kullan
    if hasattr(model_or_grid_model, 'best_estimator_'):
        model_to_predict = model_or_grid_model.best_estimator_
    else: # Aksi takdirde, doğrudan verilen modeli kullan
        model_to_predict = model_or_grid_model

    y_train_pred = model_to_predict.predict(X_train)
    y_test_pred = model_to_predict.predict(X_test)

    # Eğitim Seti Metrikleri
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred)) # FIX: np.sqrt

    # Test Seti Metrikleri
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred)) # FIX: np.sqrt

    print("\n--- Model Performans Özeti ---")
    results = pd.DataFrame({
        'Metrik': ['R2', 'MAE', 'MSE', 'RMSE'],
        'Eğitim Seti': [train_r2, train_mae, train_mse, train_rmse],
        'Test Seti': [test_r2, test_mae, test_mse, test_rmse]
    })
    print(results)

# --- train_val fonksiyonunu çağır ---
train_val(grid_model, X_train, y_train, X_test, y_test)


operations = [("OrdinalEncoder", column_trans),
              ("XGB_model",
               XGBRegressor(n_estimators=100,
                            learning_rate=0.06,
                            max_depth=3,
                            random_state=101,
                            subsample=0.5))]

pipe_model = Pipeline(steps=operations)

pipe_model.fit(X_train, y_train)


pipe_model["XGB_model"].feature_importances_


pipe_model["OrdinalEncoder"].get_feature_names_out()


# --- 9. En İyi Model ile Tahmin ve Değerlendirme ---
print("\n--- En İyi GridSearchCV Modeli ---")
print(f"En İyi Parametreler: {grid_model.best_params_}")
print(f"En İyi Çapraz Doğrulama RMSE Skoru: {-grid_model.best_score_:.4f}")


new_features = fitted_pipeline.named_steps['OrdinalEncoder'].get_feature_names_out()


# --- 5. Pipeline Tanımı ---
# Make sure the name of your ColumnTransformer step here is 'preprocessor'
operations = [
    ("preprocessor", column_trans), # <--- THIS NAME MUST MATCH!
    ("GB_model", GradientBoostingRegressor(random_state=101))
]
model = Pipeline(steps=operations)

# ... (rest of your code, including GridSearchCV fit) ...

# --- 10. Özellik Önem Dereceleri (Feature Importances) ---
fitted_pipeline = grid_model.best_estimator_

# Get the feature names AFTER the ColumnTransformer has been fitted
# The name 'preprocessor' must match the name given in the Pipeline's operations list.
new_features = fitted_pipeline.named_steps['preprocessor'].get_feature_names_out()

# ... (rest of the feature importances calculation) ...


operations = [
    ("preprocessor", column_trans), # <--- Here it is!
    ("GB_model", GradientBoostingRegressor(random_state=101))
]


# train_val fonksiyonunu çağırarak eğitim ve test setindeki performansı gösterelim
train_val(grid_model, X_train, y_train, X_test, y_test)

# --- 10. Özellik Önem Dereceleri (Feature Importances) ---
fitted_pipeline = grid_model.best_estimator_

# ColumnTransformer sonrası özellik adlarını al
# StandardScaler, OrdinalEncoder ve OneHotEncoder'ın çıktı isimlerini birleştiriyoruz
# OneHotEncoder'ın sütun isimleri 'prefix__category' formatında gelir.
new_features = fitted_pipeline.named_steps['preprocessor'].get_feature_names_out()

# Özellik önem derecelerini bir DataFrame'e dönüştür
imp_feats = pd.DataFrame(data=fitted_pipeline["GB_model"].feature_importances_,
                         columns=['Önem Derecesi'],
                         index=new_features)

grad_imp_feats = imp_feats.sort_values('Önem Derecesi', ascending=False)
print("\n--- Özellik Önem Dereceleri (En Önemliden Başlayarak) ---")
print(grad_imp_feats.to_markdown()) # Markdown tablo formatında çıktı


# --- 11. Sonuç Dosyası Oluşturma (İsteğe Bağlı) ---
# Genellikle test seti tahminlerini veya model metriklerini kaydetmek istenir
results_df = pd.DataFrame({
    'Gerçek Değerler': y_test,
    'Tahmin Edilen Değerler': fitted_pipeline.predict(X_test)
})

# CSV olarak kaydet
results_filename = "model_tahmin_sonuclari.csv"
results_df.to_csv(results_filename, index=False)
print(f"\nTahmin sonuçları '{results_filename}' dosyasına kaydedildi.")

# Metrikleri de bir metin dosyasına kaydetmek isteyebilirsiniz
with open("model_metrikleri.txt", "w") as f:
    f.write(f"En İyi Parametreler: {grid_model.best_params_}\n")
    f.write(f"En İyi Çapraz Doğrulama RMSE Skoru: {-grid_model.best_score_:.4f}\n")
    f.write("\nModel Performans Özeti:\n")
    # train_val fonksiyonunun çıktısını dosyaya yazmak için
    # geçici olarak stdout'u yakalamak daha gelişmiş bir yöntem olur,
    # burada basitçe aynı değerleri tekrar yazdırıyoruz.
    f.write(f"Eğitim R2: {r2_score(y_train, fitted_pipeline.predict(X_train)):.4f}\n")
    f.write(f"Test R2: {r2_score(y_test, fitted_pipeline.predict(X_test)):.4f}\n")
    f.write(f"Eğitim RMSE: {np.sqrt(mean_squared_error(y_train, fitted_pipeline.predict(X_train))):.4f}\n")
    f.write(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, fitted_pipeline.predict(X_test))):.4f}\n")
print(f"Model metrikleri 'model_metrikleri.txt' dosyasına kaydedildi.")


# --- 11. Sonuç Dosyası Oluşturma (İsteğe Bağlı) ---
results_df = pd.DataFrame({
    'Gerçek Değerler': y_test,
    'Tahmin Edilen Değerler': fitted_pipeline.predict(X_test)
})

# CSV olarak kaydet
results_filename = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv" # <-- Burası dosya adını belirliyor
results_df.to_csv(results_filename, index=False)
print(f"\nTahmin sonuçları '{results_filename}' dosyasına kaydedildi.")

# ...
with open("model_metrikleri.txt", "w") as f: # <-- Burası da metrik dosyasının adını belirliyor
    # ...
print(f"Model metrikleri 'model_metrikleri.txt' dosyasına kaydedildi.")


# --- 11. Sonuç Dosyası Oluşturma (İsteğe Bağlı) ---
# ... (results_df and results_filename creation) ...

results_df.to_csv(results_filename, index=False)
print(f"\nTahmin sonuçları '{results_filename}' dosyasına kaydedildi.")

# Metrikleri de bir metin dosyasına kaydetmek isteyebilirsiniz
with open("model_metrikleri.txt", "w") as f:
    # Everything inside this 'with' block MUST be indented
    f.write(f"En İyi Parametreler: {grid_model.best_params_}\n")
    f.write(f"En İyi Çapraz Doğrulama RMSE Skoru: {-grid_model.best_score_:.4f}\n")
    f.write("\nModel Performans Özeti:\n")
    f.write(f"Eğitim R2: {r2_score(y_train, fitted_pipeline.predict(X_train)):.4f}\n")
    f.write(f"Test R2: {r2_score(y_test, fitted_pipeline.predict(X_test)):.4f}\n")
    f.write(f"Eğitim RMSE: {np.sqrt(mean_squared_error(y_train, fitted_pipeline.predict(X_train))):.4f}\n")
    f.write(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, fitted_pipeline.predict(X_test))):.4f}\n")

# This print statement is OUTSIDE the 'with' block, so it should be at the same level as 'with'
print(f"Model metrikleri 'model_metrikleri.txt' dosyasına kaydedildi.")


# --- 11. Sonuç Dosyası Oluşturma (İsteğe Bağlı) ---
results_df = pd.DataFrame({ # <--- This line defines results_df
    'Gerçek Değerler': y_test,
    'Tahmin Edilen Değerler': fitted_pipeline.predict(X_test)
})

# CSV olarak kaydet
results_filename = "model_tahmin_sonuclari.csv" # <--- This line defines results_filename
results_df.to_csv(results_filename, index=False)
# ... rest of the code


# --- 11. Sonuç Dosyası Oluşturma (İsteğe Bağlı) ---

# Make sure these lines are present and uncommented!
results_df = pd.DataFrame({
    'Gerçek Değerler': y_test,
    'Tahmin Edilen Değerler': fitted_pipeline.predict(X_test)
})

results_filename = "model_tahmin_sonuclari.csv" # Or "submission.csv" if required by a platform

results_df.to_csv(results_filename, index=False)
print(f"\nTahmin sonuçları '{results_filename}' dosyasına kaydedildi.")

# Metrikleri de bir metin dosyasına kaydetmek isteyebilirsiniz
with open("model_metrikleri.txt", "w") as f:
    f.write(f"En İyi Parametreler: {grid_model.best_params_}\n")
    f.write(f"En İyi Çapraz Doğrulama RMSE Skoru: {-grid_model.best_score_:.4f}\n")
    f.write("\nModel Performans Özeti:\n")
    f.write(f"Eğitim R2: {r2_score(y_train, fitted_pipeline.predict(X_train)):.4f}\n")
    f.write(f"Test R2: {r2_score(y_test, fitted_pipeline.predict(X_test)):.4f}\n")
    f.write(f"Eğitim RMSE: {np.sqrt(mean_squared_error(y_train, fitted_pipeline.predict(X_train))):.4f}\n")
    f.write(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, fitted_pipeline.predict(X_test))):.4f}\n")

# This print statement is OUTSIDE the 'with' block, so it should be at the same level as 'with'
print(f"Model metrikleri 'model_metrikleri.txt' dosyasına kaydedildi.")


import pandas as pd
# ... (rest of your imports and previous code for model training) ...

# --- Generate your predictions for the test set ---
# Assuming 'X_test_final' is your preprocessed test data ready for prediction
# If you used a pipeline, it would look like this:
# y_test_predictions = fitted_pipeline.predict(X_test_competition)
# Make sure X_test_competition is the *unseen* test data from the competition,
# not your split X_test from cross-validation.

# If you need to load the competition's test data separately:
# test_data_for_submission = pd.read_csv('path/to/competition_test_data.csv')
# # Apply the same preprocessing pipeline to this unseen test data
# y_test_predictions = fitted_pipeline.predict(test_data_for_submission)


# Create the DataFrame for submission
# This assumes y_test_predictions contains your model's outputs for the competition's test set
submission_df = pd.DataFrame({
    'ID': test_data_for_submission['ID'], # Replace 'ID' with your actual ID column name from competition test data
    'Target': y_test_predictions # Replace 'Target' with your actual target column name for submission
})

# --- Save the submission file ---
submission_filename = "submission.csv" # <--- THIS IS CRUCIAL!

submission_df.to_csv(submission_filename, index=False) # index=False is important to avoid writing DataFrame index

print(f"\nSubmission file '{submission_filename}' successfully created!")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- 1. Örnek Veri Oluşturma (Bu kısım model eğitimi için kendi veriniz olmalı) ---
# Gerçek bir yarışmada bu, genellikle "train.csv" dosyasını yüklemektir.
np.random.seed(42)
num_samples = 150
num_features = 5
data = {
    'numerical_feature_1': np.random.rand(num_samples) * 100,
    'numerical_feature_2': np.random.rand(num_samples) * 50,
    'ordinal_feature': np.random.choice(['low', 'medium', 'high', 'very_high'], num_samples),
    'nominal_feature': np.random.choice(['A', 'B', 'C', 'D', 'E'], num_samples),
    'another_numerical_feature': np.random.randint(0, 100, num_samples),
    'target': np.random.rand(num_samples) * 200
}
X = pd.DataFrame(data)
y = X['target']
X = X.drop('target', axis=1)

nan_indices_X = np.random.choice(X.index, 10, replace=False)
inf_indices_X = np.random.choice(X.index, 5, replace=False)
nan_indices_y = np.random.choice(y.index, 3, replace=False)

for idx in nan_indices_X:
    col = np.random.choice(X.columns)
    X.loc[idx, col] = np.nan
for idx in inf_indices_X:
    col = np.random.choice(X.select_dtypes(include=np.number).columns)
    X.loc[idx, col] = np.inf
y.loc[nan_indices_y] = np.nan

print(f"--- Başlangıç Veri Durumu ---")
print(f"X boyutu: {X.shape}")
print(f"y boyutu: {y.shape}")
print(f"X'teki toplam NaN sayısı: {X.isnull().sum().sum()}")
print(f"X'teki toplam Inf sayısı: {np.isinf(X).sum().sum()}")
print(f"y'deki toplam NaN sayısı: {y.isnull().sum()}")
print("-" * 40)

# --- 2. Veri Temizleme (NaN ve Sonsuz Değerleri Silme) ---
X.replace([np.inf, -np.inf], np.nan, inplace=True)
df_combined = pd.concat([X, y.rename('target')], axis=1)
df_cleaned = df_combined.dropna()
X_cleaned = df_cleaned.drop('target', axis=1)
y_cleaned = df_cleaned['target']

print(f"--- Temizleme Sonrası Veri Durumu ---")
print(f"X_cleaned boyutu: {X_cleaned.shape}")
print(f"y_cleaned boyutu: {y_cleaned.shape}")
print(f"X_cleaned'daki toplam NaN sayısı: {X_cleaned.isnull().sum().sum()}")
print(f"X_cleaned'daki toplam Inf sayısı: {np.isinf(X_cleaned).sum().sum()}")
print(f"y_cleaned'daki toplam NaN sayısı: {y_cleaned.isnull().sum()}")
print("-" * 40)

# --- 3. Veriyi Eğitim ve Test Setlerine Ayırma ---
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# --- 4. ColumnTransformer Tanımı (Ön İşleme) ---
numerical_cols = ['numerical_feature_1', 'numerical_feature_2', 'another_numerical_feature']
ordinal_cols = ['ordinal_feature']
nominal_cols = ['nominal_feature']
ordinal_categories_order = [['low', 'medium', 'high', 'very_high']]

numerical_transformer = StandardScaler()
ordinal_transformer = OrdinalEncoder(categories=ordinal_categories_order, handle_unknown='use_encoded_value', unknown_value=-1)
nominal_transformer = OneHotEncoder(handle_unknown='ignore')

column_trans = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('ord', ordinal_transformer, ordinal_cols),
        ('nom', nominal_transformer, nominal_cols)
    ],
    remainder='passthrough'
)

# --- 5. Pipeline Tanımı ---
operations = [
    ("preprocessor", column_trans),
    ("GB_model", GradientBoostingRegressor(random_state=101))
]
model = Pipeline(steps=operations)

# --- 6. param_grid Tanımı (GridSearchCV için Hiperparametreler) ---
param_grid = {
    'GB_model__n_estimators': [50, 100, 150],
    'GB_model__learning_rate': [0.05, 0.1, 0.15],
    'GB_model__max_depth': [3, 4],
}

# --- 7. GridSearchCV Tanımı ve Eğitimi ---
grid_model = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=5,
                          n_jobs=-1,
                          return_train_score=True)

print("\n--- GridSearchCV Başlıyor... Bu biraz zaman alabilir ---")
grid_model.fit(X_train, y_train)
print("--- GridSearchCV Tamamlandı ---")

# --- 8. Model Performansını Değerlendirme Fonksiyonu ---
def train_val(model_or_grid_model, X_train, y_train, X_test, y_test):
    if hasattr(model_or_grid_model, 'best_estimator_'):
        model_to_predict = model_or_grid_model.best_estimator_
    else:
        model_to_predict = model_or_


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- 1. Örnek Veri Oluşturma (Bu kısım model eğitimi için kendi veriniz olmalı) ---
# Gerçek bir yarışmada bu, genellikle "train.csv" dosyasını yüklemektir.
np.random.seed(42)
num_samples = 150
num_features = 5
data = {
    'numerical_feature_1': np.random.rand(num_samples) * 100,
    'numerical_feature_2': np.random.rand(num_samples) * 50,
    'ordinal_feature': np.random.choice(['low', 'medium', 'high', 'very_high'], num_samples),
    'nominal_feature': np.random.choice(['A', 'B', 'C', 'D', 'E'], num_samples),
    'another_numerical_feature': np.random.randint(0, 100, num_samples),
    'target': np.random.rand(num_samples) * 200
}
X = pd.DataFrame(data)
y = X['target']
X = X.drop('target', axis=1)

nan_indices_X = np.random.choice(X.index, 10, replace=False)
inf_indices_X = np.random.choice(X.index, 5, replace=False)
nan_indices_y = np.random.choice(y.index, 3, replace=False)

for idx in nan_indices_X:
    col = np.random.choice(X.columns)
    X.loc[idx, col] = np.nan
for idx in inf_indices_X:
    col = np.random.choice(X.select_dtypes(include=np.number).columns)
    X.loc[idx, col] = np.inf
y.loc[nan_indices_y] = np.nan

print(f"--- Başlangıç Veri Durumu ---")
print(f"X boyutu: {X.shape}")
print(f"y boyutu: {y.shape}")
print(f"X'teki toplam NaN sayısı: {X.isnull().sum().sum()}")
print(f"X'teki toplam Inf sayısı: {np.isinf(X).sum().sum()}")
print(f"y'deki toplam NaN sayısı: {y.isnull().sum()}")
print("-" * 40)

# --- 2. Veri Temizleme (NaN ve Sonsuz Değerleri Silme) ---
X.replace([np.inf, -np.inf], np.nan, inplace=True)
df_combined = pd.concat([X, y.rename('target')], axis=1)
df_cleaned = df_combined.dropna()
X_cleaned = df_cleaned.drop('target', axis=1)
y_cleaned = df_cleaned['target']

print(f"--- Temizleme Sonrası Veri Durumu ---")
print(f"X_cleaned boyutu: {X_cleaned.shape}")
print(f"y_cleaned boyutu: {y_cleaned.shape}")
print(f"X_cleaned'daki toplam NaN sayısı: {X_cleaned.isnull().sum().sum()}")
print(f"X_cleaned'daki toplam Inf sayısı: {np.isinf(X_cleaned).sum().sum()}")
print(f"y_cleaned'daki toplam NaN sayısı: {y_cleaned.isnull().sum()}")
print("-" * 40)

# --- 3. Veriyi Eğitim ve Test Setlerine Ayırma ---
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# --- 4. ColumnTransformer Tanımı (Ön İşleme) ---
numerical_cols = ['numerical_feature_1', 'numerical_feature_2', 'another_numerical_feature']
ordinal_cols = ['ordinal_feature']
nominal_cols = ['nominal_feature']
ordinal_categories_order = [['low', 'medium', 'high', 'very_high']]

numerical_transformer = StandardScaler()
ordinal_transformer = OrdinalEncoder(categories=ordinal_categories_order, handle_unknown='use_encoded_value', unknown_value=-1)
nominal_transformer = OneHotEncoder(handle_unknown='ignore')

column_trans = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('ord', ordinal_transformer, ordinal_cols),
        ('nom', nominal_transformer, nominal_cols)
    ],
    remainder='passthrough'
)

# --- 5. Pipeline Tanımı ---
operations = [
    ("preprocessor", column_trans),
    ("GB_model", GradientBoostingRegressor(random_state=101))
]
model = Pipeline(steps=operations)

# --- 6. param_grid Tanımı (GridSearchCV için Hiperparametreler) ---
param_grid = {
    'GB_model__n_estimators': [50, 100, 150],
    'GB_model__learning_rate': [0.05, 0.1, 0.15],
    'GB_model__max_depth': [3, 4],
}

# --- 7. GridSearchCV Tanımı ve Eğitimi ---
grid_model = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=5,
                          n_jobs=-1,
                          return_train_score=True)

print("\n--- GridSearchCV Başlıyor... Bu biraz zaman alabilir ---")
grid_model.fit(X_train, y_train)
print("--- GridSearchCV Tamamlandı ---")

# --- 8. Model Performansını Değerlendirme Fonksiyonu ---
def train_val(model_or_grid_model, X_train, y_train, X_test, y_test):
    if hasattr(model_or_grid_model, 'best_estimator_'):
        model_to_predict = model_or_grid_model.best_estimator_
    else:
        model_to_predict = model_or_grid_model

    y_train_pred = model_to_predict.predict(X_train)
    y_test_pred = model_to_predict.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print("\n--- Model Performans Özeti ---")
    results = pd.DataFrame({
        'Metrik': ['R2', 'MAE', 'MSE', 'RMSE'],
        'Eğitim Seti': [f"{train_r2:.4f}", f"{train_mae:.4f}", f"{train_mse:.4f}", f"{train_rmse:.4f}"],
        'Test Seti': [f"{test_r2:.4f}", f"{test_mae:.4f}", f"{test_mse:.4f}", f"{test_rmse:.4f}"]
    })
    print(results.to_markdown(index=False))

# --- 9. En İyi Model ile Tahmin ve Değerlendirme ---
print("\n--- En İyi GridSearchCV Modeli ---")
print(f"En İyi Parametreler: {grid_model.best_params_}")
print(f"En İyi Çapraz Doğrulama RMSE Skoru: {-grid_model.best_score_:.4f}")

train_val(grid_model, X_train, y_train, X_test, y_test)

# --- 10. Özellik Önem Dereceleri (Feature Importances) ---
fitted_pipeline = grid_model.best_estimator_
new_features = fitted_pipeline.named_steps['preprocessor'].get_feature_names_out()

imp_feats = pd.DataFrame(data=fitted_pipeline["GB_model"].feature_importances_,
                         columns=['Önem Derecesi'],
                         index=new_features)

grad_imp_feats = imp_feats.sort_values('Önem Derecesi', ascending=False)
print("\n--- Özellik Önem Dereceleri (En Önemliden Başlayarak) ---")
print(grad_imp_feats.to_markdown())


# --- 11. Sonuç Dosyası Oluşturma (Yarışma İçin) ---

# --- YARIŞMA TEST VERİSİNİ BURADA YÜKLEYİN ---
# Bu, modelinizi eğitmek için kullandığınız X_cleaned'den farklıdır.
# Bu dosya, yarışma platformu tarafından size sağlanır ve genellikle "test.csv" veya "sample_submission.csv" ile birlikte gelir.
# Örneğin:
try:
    # Gerçek yarışma verisi yerine geçici bir örnek test verisi oluşturalım
    # Gerçek senaryoda bu satırı:
    # test_data_for_submission = pd.read_csv('path/to/your/competition_test_data.csv')
    # olarak değiştirmelisiniz. ID sütununa sahip olduğundan emin olun.
    test_data_for_submission = pd.DataFrame({
        'ID': np.arange(200, 200 + 50), # Örnek ID'ler
        'numerical_feature_1': np.random.rand(50) * 100,
        'numerical_feature_2': np.random.rand(50) * 50,
        'ordinal_feature': np.random.choice(['low', 'medium', 'high', 'very_high'], 50),
        'nominal_feature': np.random.choice(['A', 'B', 'C', 'D', 'E'], 50),
        'another_numerical_feature': np.random.randint(0, 100, 50),
        # Yarışma test verisinde hedef sütun olmaz
    })
    # Eğer test_data_for_submission'da NaN veya Inf varsa, bunları da temizlemeniz gerekebilir
    test_data_for_submission.replace([np.inf, -np.inf], np.nan, inplace=True)
    # İmputation (doldurma) kullanmıyorsak, NaN içeren satırları silmeliyiz (dikkatli olun)
    # Veya test_data_for_submission'da eksik veriyi ele almak için bir imputation stratejisi uygulayın.
    # Örneğin: test_data_for_submission.dropna(inplace=True)

except FileNotFoundError:
    print("\nUYARI: 'test_data_for_submission' için örnek veri kullanılıyor çünkü dosya bulunamadı.")
    print("Yarışma için kendi test veri setinizin dosya yolunu 'pd.read_csv()' içinde belirttiğinizden emin olun.")
    # Örnek oluşturma kodunu yukarıda bıraktık.

# Modelinizi kullanarak yarışma test verileri üzerinde tahminler yapın
# Pipeline, ön işleme adımlarını (ölçekleme, kodlama) otomatik olarak uygulayacaktır.
y_competition_predictions = fitted_pipeline.predict(test_data_for_submission)

# Submission DataFrame'ini oluşturun
# 'ID' ve 'Target' sütun adları yarışmanın beklentisine göre değişebilir.
submission_df = pd.DataFrame({
    'ID': test_data_for_submission['ID'], # Yarışma test verisindeki ID sütunu
    'Target': y_competition_predictions    # Modelin tahminleri
})

# --- Submission dosyasını kaydedin ---
submission_filename = "submission.csv" # <--- Yarışma platformunun beklediği dosya adı bu OLMALIDIR!

submission_df.to_csv(submission_filename, index=False) # index=False çok önemli!

print(f"\nSubmission dosyası '{submission_filename}' başarıyla oluşturuldu ve kaydedildi.")
print(f"Submission dosyasının ilk 5 satırı:\n{submission_df.head().to_markdown(index=False)}")


# Metrikleri de bir metin dosyasına kaydetmek isteyebilirsiniz
metrics_filename = "model_metrikleri.txt"
with open(metrics_filename, "w") as f:
    f.write(f"En İyi Parametreler: {grid_model.best_params_}\n")
    f.write(f"En İyi Çapraz Doğrulama RMSE Skoru: {-grid_model.best_score_:.4f}\n")
    f.write("\nModel Performans Özeti:\n")
    f.write(f"Eğitim R2: {r2_score(y_train, fitted_pipeline.predict(X_train)):.4f}\n")
    f.write(f"Test R2: {r2_score(y_test, fitted_pipeline.predict(X_test)):.4f}\n")
    f.write(f"Eğitim RMSE: {np.sqrt(mean_squared_error(y_train, fitted_pipeline.predict(X_train))):.4f}\n")
    f.write(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, fitted_pipeline.predict(X_test))):.4f}\n")
print(f"Model metrikleri '{metrics_filename}' dosyasına kaydedildi.")


import pandas as pd
import numpy as np
# ... (rest of your imports) ...

# ... (rest of your data creation and initial prints) ...

# Corrected line for checking infinite values:
# Only check for inf in numeric columns of X
print(f"X'teki toplam Inf sayısı: {np.isinf(X.select_dtypes(include=np.number)).sum().sum()}")
# X.select_dtypes(include=np.number) creates a view of the DataFrame containing only numeric columns.
# Then, np.isinf() can be safely applied.

# ... (rest of your code) ...

# You might also want to apply a similar check to X_cleaned if you ever print its inf count
# print(f"X_cleaned'daki toplam Inf sayısı: {np.isinf(X_cleaned.select_dtypes(include=np.number)).sum().sum()}")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- 1. Örnek Veri Oluşturma ---
np.random.seed(42) # Tekrarlanabilirlik için
num_samples = 150 # Daha fazla örnek
num_features = 5
data = {
    'numerical_feature_1': np.random.rand(num_samples) * 100,
    'numerical_feature_2': np.random.rand(num_samples) * 50,
    'ordinal_feature': np.random.choice(['low', 'medium', 'high', 'very_high'], num_samples),
    'nominal_feature': np.random.choice(['A', 'B', 'C', 'D', 'E'], num_samples),
    'another_numerical_feature': np.random.randint(0, 100, num_samples), # Yeni bir sayısal özellik
    'target': np.random.rand(num_samples) * 200 # Tahmin edilecek hedef
}
X = pd.DataFrame(data)
y = X['target']
X = X.drop('target', axis=1)

# Test için kasıtlı olarak NaN ve Inf değerler ekleyelim
nan_indices_X = np.random.choice(X.index, 10, replace=False)
inf_indices_X = np.random.choice(X.index, 5, replace=False)
nan_indices_y = np.random.choice(y.index, 3, replace=False)

for idx in nan_indices_X:
    col = np.random.choice(X.columns)
    X.loc[idx, col] = np.nan
for idx in inf_indices_X:
    col = np.random.choice(X.select_dtypes(include=np.number).columns) # Sadece sayısal sütunlara inf ekle
    X.loc[idx, col] = np.inf
y.loc[nan_indices_y] = np.nan


print(f"--- Başlangıç Veri Durumu ---")
print(f"X boyutu: {X.shape}")
print(f"y boyutu: {y.shape}")
print(f"X'teki toplam NaN sayısı: {X.isnull().sum().sum()}")
# --- DÜZELTİLMİŞ SATIR ---
print(f"X'teki toplam Inf sayısı: {np.isinf(X.select_dtypes(include=np.number)).sum().sum()}")
# -------------------------
print(f"y'deki toplam NaN sayısı: {y.isnull().sum()}")
print("-" * 40)

# --- 2. Veri Temizleme (NaN ve Sonsuz Değerleri Silme) ---
X.replace([np.inf, -np.inf], np.nan, inplace=True)
df_combined = pd.concat([X, y.rename('target')], axis=1)
df_cleaned = df_combined.dropna()

X_cleaned = df_cleaned.drop('target', axis=1)
y_cleaned = df_cleaned['target']

print(f"--- Temizleme Sonrası Veri Durumu ---")
print(f"X_cleaned boyutu: {X_cleaned.shape}")
print(f"y_cleaned boyutu: {y_cleaned.shape}")
print(f"X_cleaned'daki toplam NaN sayısı: {X_cleaned.isnull().sum().sum()}")
# --- DÜZELTİLMİŞ SATIR ---
# Artık temizlediğimiz için 0 olmalı, ama yine de hata vermemesi için select_dtypes kullanırız.
print(f"X_cleaned'daki toplam Inf sayısı: {np.isinf(X_cleaned.select_dtypes(include=np.number)).sum().sum()}")
# -------------------------
print(f"y_cleaned'daki toplam NaN sayısı: {y_cleaned.isnull().sum()}")
print("-" * 40)

# --- 3. Veriyi Eğitim ve Test Setlerine Ayırma ---
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# --- 4. ColumnTransformer Tanımı (Ön İşleme) ---
numerical_cols = ['numerical_feature_1', 'numerical_feature_2', 'another_numerical_feature']


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- 1. Örnek Veri Oluşturma ---
np.random.seed(42) # Tekrarlanabilirlik için
num_samples = 150 # Daha fazla örnek
num_features = 5
data = {
    'numerical_feature_1': np.random.rand(num_samples) * 100,
    'numerical_feature_2': np.random.rand(num_samples) * 50,
    'ordinal_feature': np.random.choice(['low', 'medium', 'high', 'very_high'], num_samples),
    'nominal_feature': np.random.choice(['A', 'B', 'C', 'D', 'E'], num_samples),
    'another_numerical_feature': np.random.randint(0, 100, num_samples), # Yeni bir sayısal özellik
    'target': np.random.rand(num_samples) * 200 # Tahmin edilecek hedef
}
X = pd.DataFrame(data)
y = X['target']
X = X.drop('target', axis=1)

# Test için kasıtlı olarak NaN ve Inf değerler ekleyelim
nan_indices_X = np.random.choice(X.index, 10, replace=False)
inf_indices_X = np.random.choice(X.index, 5, replace=False)
nan_indices_y = np.random.choice(y.index, 3, replace=False)

for idx in nan_indices_X:
    col = np.random.choice(X.columns)
    X.loc[idx, col] = np.nan
for idx in inf_indices_X:
    col = np.random.choice(X.select_dtypes(include=np.number).columns) # Sadece sayısal sütunlara inf ekle
    X.loc[idx, col] = np.inf
y.loc[nan_indices_y] = np.nan


print(f"--- Başlangıç Veri Durumu ---")
print(f"X boyutu: {X.shape}")
print(f"y boyutu: {y.shape}")
print(f"X'teki toplam NaN sayısı: {X.isnull().sum().sum()}")
# --- DÜZELTİLMİŞ SATIR ---
print(f"X'teki toplam Inf sayısı: {np.isinf(X.select_dtypes(include=np.number)).sum().sum()}")
# -------------------------
print(f"y'deki toplam NaN sayısı: {y.isnull().sum()}")
print("-" * 40)

# --- 2. Veri Temizleme (NaN ve Sonsuz Değerleri Silme) ---
X.replace([np.inf, -np.inf], np.nan, inplace=True)
df_combined = pd.concat([X, y.rename('target')], axis=1)
df_cleaned = df_combined.dropna()

X_cleaned = df_cleaned.drop('target', axis=1)
y_cleaned = df_cleaned['target']

print(f"--- Temizleme Sonrası Veri Durumu ---")
print(f"X_cleaned boyutu: {X_cleaned.shape}")
print(f"y_cleaned boyutu: {y_cleaned.shape}")
print(f"X_cleaned'daki toplam NaN sayısı: {X_cleaned.isnull().sum().sum()}")
# --- DÜZELTİLMİŞ SATIR ---
# Artık temizlediğimiz için 0 olmalı, ama yine de hata vermemesi için select_dtypes kullanırız.
print(f"X_cleaned'daki toplam Inf sayısı: {np.isinf(X_cleaned.select_dtypes(include=np.number)).sum().sum()}")
# -------------------------
print(f"y_cleaned'daki toplam NaN sayısı: {y_cleaned.isnull().sum()}")
print("-" * 40)

# --- 3. Veriyi Eğitim ve Test Setlerine Ayırma ---
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# --- 4. ColumnTransformer Tanımı (Ön İşleme) ---
numerical_cols = ['numerical_feature_1', 'numerical_feature_2', 'another_numerical_feature']
ordinal_cols = ['ordinal_feature']
nominal_cols = ['nominal_feature']

ordinal_categories_order = [['low', 'medium', 'high', 'very_high']]

numerical_transformer = StandardScaler()
ordinal_transformer = OrdinalEncoder(categories=ordinal_categories_order, handle_unknown='use_encoded_value', unknown_value=-1)
nominal_transformer = OneHotEncoder(handle_unknown='ignore')

column_trans = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('ord', ordinal_transformer, ordinal_cols),
        ('nom', nominal_transformer, nominal_cols)
    ],
    remainder='passthrough'
)

# --- 5. Pipeline Tanımı ---
operations = [
    ("preprocessor", column_trans),
    ("GB_model", GradientBoostingRegressor(random_state=101))
]
model = Pipeline(steps=operations)

# --- 6. param_grid Tanımı (GridSearchCV için Hiperparametreler) ---
param_grid = {
    'GB_model__n_estimators': [50, 100, 150],
    'GB_model__learning_rate': [0.05, 0.1, 0.15],
    'GB_model__max_depth': [3, 4],
}

# --- 7. GridSearchCV Tanımı ve Eğitimi ---
grid_model = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=5,
                          n_jobs=-1,
                          return_train_score=True)

print("\n--- GridSearchCV Başlıyor... Bu biraz zaman alabilir ---")
grid_model.fit(X_train, y_train)
print("--- GridSearchCV Tamamlandı ---")

# --- 8. Model Performansını Değerlendirme Fonksiyonu ---
def train_val(model_or_grid_model, X_train, y_train, X_test, y_test):
    if hasattr(model_or_grid_model, 'best_estimator_'):
        model_to_predict = model_or_grid_model.best_estimator_
    else:
        model_to_predict = model_or_grid_model

    y_train_pred = model_to_predict.predict(X_train)
    y_test_pred = model_to_predict.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print("\n--- Model Performans Özeti ---")
    results = pd.DataFrame({
        'Metrik': ['R2', 'MAE', 'MSE', 'RMSE'],
        'Eğitim Seti': [f"{train_r2:.4f}", f"{train_mae:.4f}", f"{train_mse:.4f}", f"{train_rmse:.4f}"],
        'Test Seti': [f"{test_r2:.4f}", f"{test_mae:.4f}", f"{test_mse:.4f}", f"{test_rmse:.4f}"]
    })
    print(results.to_markdown(index=False))

# --- 9. En İyi Model ile Tahmin ve Değerlendirme ---
print("\n--- En İyi GridSearchCV Modeli ---")
print(f"En İyi Parametreler: {grid_model.best_params_}")
print(f"En İyi Çapraz Doğrulama RMSE Skoru: {-grid_model.best_score_:.4f}")

train_val(grid_model, X_train, y_train, X_test, y_test)

# --- 10. Özellik Önem Dereceleri (Feature Importances) ---
fitted_pipeline = grid_model.best_estimator_
new_features = fitted_pipeline.named_steps['preprocessor'].get_feature_names_out()

imp_feats = pd.DataFrame(data=fitted_pipeline["GB_model"].feature_importances_,
                         columns=['Önem Derecesi'],
                         index=new_features)

grad_imp_feats = imp_feats.sort_values('Önem Derecesi', ascending=False)
print("\n--- Özellik Önem Dereceleri (En Önemliden Başlayarak) ---")
print(grad_imp_feats.to_markdown())


# --- 11. Sonuç Dosyası Oluşturma (Yarışma İçin) ---

try:
    test_data_for_submission = pd.DataFrame({
        'ID': np.arange(200, 200 + 50),
        'numerical_feature_1': np.random.rand(50) * 100,
        'numerical_feature_2': np.random.rand(50) * 50,
        'ordinal_feature': np.random.choice(['low', 'medium', 'high', 'very_high'], 50),
        'nominal_feature': np.random.choice(['A', 'B', 'C', 'D', 'E'], 50),
        'another_numerical_feature': np.random.randint(0, 100, 50),
    })
    test_data_for_submission.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Eğer test_data_for_submission'da NaN varsa ve ColumnTransformer içinde SimpleImputer yoksa,
    # burada .dropna() veya başka bir imputation stratejisi uygulamanız gerekebilir.
    # Aksi takdirde, predict() çağrıldığında NaN'lar hata verebilir.

except FileNotFoundError:
    print("\nUYARI: 'test_data_for_submission' için örnek veri kullanılıyor çünkü dosya bulunamadı.")
    print("Yarışma için kendi test veri setinizin dosya yolunu 'pd.read_csv()' içinde belirttiğinizden emin olun.")

y_competition_predictions = fitted_pipeline.predict(test_data_for_submission)

submission_df = pd.DataFrame({
    'ID': test_data_for_submission['ID'],
    'Target': y_competition_predictions
})

submission_filename = "submission.csv"

submission_df.to_csv(submission_filename, index=False)

print(f"\nSubmission dosyası '{submission_filename}' başarıyla oluşturuldu ve kaydedildi.")
print(f"Submission dosyasının ilk 5 satırı:\n{submission_df.head().to_markdown(index=False)}")


metrics_filename = "model_metrikleri.txt"
with open(metrics_filename, "w") as f:
    f.write(f"En İyi Parametreler: {grid_model.best_params_}\n")
    f.write(f"En İyi Çapraz Doğrulama RMSE Skoru: {-grid_model.best_score_:.4f}\n")
    f.write("\nModel Performans Özeti:\n")
    f.write(f"Eğitim R2: {r2_score(y_train, fitted_pipeline.predict(X_train)):.4f}\n")
    f.write(f"Test R2: {r2_score(y_test, fitted_pipeline.predict(X_test)):.4f}\n")
    f.write(f"Eğitim RMSE: {np.sqrt(mean_squared_error(y_train, fitted_pipeline.predict(X_train))):.4f}\n")
    f.write(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, fitted_pipeline.predict(X_test))):.4f}\n")
print(f"Model metrikleri '{metrics_filename}' dosyasına kaydedildi.")


# Before (causing the error):
# submission_df = pd.DataFrame({
#     'ID': test_data_for_submission['ID'],
#     'Target': y_competition_predictions # <--- This is 'Target'
# })

# After (Corrected):
submission_df = pd.DataFrame({
    'ID': test_data_for_submission['ID'],
    'prediction': y_competition_predictions # <--- CHANGE THIS TO 'prediction'
})

# Make sure to run your entire notebook again after this change,
# and then try submitting the newly generated submission.csv file.





submission_filename = "submission.csv" # NOT "Submission.csv" or "mysubmission.csv"
submission_df.to_csv(submission_filename, index=False)







