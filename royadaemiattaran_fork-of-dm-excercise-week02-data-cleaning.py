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
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')


# تعداد ردیف و ستون
print(df.shape)
# نمایش چند ردیف اول
df.head()
# اطلاعات کلی
df.info()


# لیست ستون‌هایی که باید حذف شوند
cols_to_drop = [
    'PurchDate', 'VehYear', 'Model', 'Trim', 'SubModel',
    'WheelTypeID', 'BYRNO', 'VNZIP1', 'VNST'
]

# حذف ستون‌ها
df = df.drop(columns=cols_to_drop, errors='ignore')

# نمایش شکل جدید دیتافریم
df.shape



print("Shape:", df.shape)
print("Columns:")
for col in df.columns:
    print(col)



for col in df.columns:
    print(repr(col))



df.index.name



# Target
y = df["IsBadBuy"]

# Features (تمام ستون‌ها به جز تارگت)
X = df.drop("IsBadBuy", axis=1)



X.head()



from sklearn.model_selection import train_test_split

# تقسیم داده‌ها به training و test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,       # 20% تست
    random_state=42,     # برای reproducibility
    stratify=y           # نگه داشتن توزیع تارگت مشابه در train و test
)

# اندازه داده‌ها
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)



import numpy as np

column_ranges = {
    'VehicleAge': (0, 30),
    'VehOdo': (0, 120000),
    'MMRAcquisitionAuctionAveragePrice': (800, 46000),
    'MMRAcquisitionAuctionCleanPrice': (1000, 46000),
    'MMRAcquisitionRetailAveragePrice': (1000, 46000),
    'MMRAcquisitonRetailCleanPrice': (1000, 46000),
    'MMRCurrentAuctionAveragePrice': (300, 46000),
    'MMRCurrentAuctionCleanPrice': (400, 46000),
    'MMRCurrentRetailAveragePrice': (800, 46000),
    'MMRCurrentRetailCleanPrice': (1000, 46000),
    'VehBCost': (1000, 46000),
    'WarrantyCost': (400, 8000)
}

for column, (min_val, max_val) in column_ranges.items():
    df[column] = df[column].apply(lambda x: x if min_val <= x <= max_val else np.nan)
df.head()
df.isnull().sum()



# تبدیل "NOT AVAIL" در Color به NaN
df['Color'] = df['Color'].replace('NOT AVAIL', np.nan)

# برای Make و Color، مقادیر با فراوانی کمتر از 1% را به 'OTHER' تبدیل کن
for col in ['Make', 'Color']:
    freq = df[col].value_counts(normalize=True)
    rare_categories = freq[freq < 0.01].index
    df[col] = df[col].replace(rare_categories, 'OTHER')



# جدا کردن ویژگی‌ها (X) و هدف (y)
y = df['IsBadBuy']
X = df.drop(columns=['IsBadBuy'])

# تعیین ستون‌های عددی و دسته‌ای
continuous_fields = X.select_dtypes(include=['float64','int64']).columns
categorical_fields = X.select_dtypes(include=['object','category']).columns



import numpy as np

low_cv_features = []
for col in continuous_fields:
    mean = X[col].mean()
    std = X[col].std()
    cv = std / mean if mean != 0 else 0
    if cv < 0.1:
        low_cv_features.append(col)

print("Low CV features:", low_cv_features)



mode_dominant_features = []
for col in categorical_fields:
    mode_ratio = X[col].value_counts(normalize=True, dropna=False).max()
    if mode_ratio > 0.99:
        mode_dominant_features.append(col)

print("Mode dominant categorical features:", mode_dominant_features)



high_cardinality_features = []
for col in categorical_fields:
    unique_ratio = X[col].nunique() / len(X)
    if unique_ratio > 0.9:
        high_cardinality_features.append(col)

print("High cardinality features:", high_cardinality_features)



df.head(10)  # ۱۰ ردیف اول
df.isna().sum()
print(df.columns)



'PRIMEUNIT' in df.columns, 'AUCGUART' in df.columns



import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np

# تابع بررسی رابطه با تارگت و تبدیل null به 'unknown' در صورت نیاز
def check_and_handle_missing(df, col, target='IsBadBuy', alpha=0.05):
    # فقط ردیف‌هایی که مقدار غیر null دارند
    contingency_table = pd.crosstab(df[col].dropna(), df[target])
    
    # آزمون کای-دو
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    
    print(f"{col} -> p-value: {p}")
    
    if p < alpha:
        print(f"Significant relationship found. Filling missing values in {col} with 'unknown'.")
        df[col] = df[col].fillna('unknown')
    else:
        print(f"No significant relationship. Dropping column {col}.")
        df.drop(columns=[col], inplace=True)
    
    return df

# اعمال روی PRIMEUNIT و AUCGUART
df = check_and_handle_missing(df, 'PRIMEUNIT')
df = check_and_handle_missing(df, 'AUCGUART')

# چک کردن وضعیت null‌ها
print(df[['PRIMEUNIT', 'AUCGUART']].isna().sum())



import pandas as pd
import numpy as np

# شناسایی ستون‌های عددی و دسته‌ای
numeric_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(include='object').columns

# Continuous features with low CV
low_cv_cols = []
for col in numeric_cols:
    if df[col].mean() != 0:  # جلوگیری از تقسیم بر صفر
        cv = df[col].std() / df[col].mean()
        if cv < 0.1:
            low_cv_cols.append(col)

# Categorical features with mode > 99%
high_mode_cols = []
high_unique_cols = []
for col in categorical_cols:
    mode_freq = df[col].value_counts(normalize=True).iloc[0]
    unique_pct = df[col].nunique() / len(df)
    if mode_freq > 0.99:
        high_mode_cols.append(col)
    if unique_pct > 0.9:
        high_unique_cols.append(col)

print("Continuous features with CV < 0.1:", low_cv_cols)
print("Categorical features with mode > 99%:", high_mode_cols)
print("Categorical features with unique > 90%:", high_unique_cols)



# فرض کنید df دیتافریم شماست و RefId اندیس شده است
y = df['IsBadBuy']            # تارگت
X = df.drop(columns=['IsBadBuy'])  # ویژگی‌ها



inputs_iso = X.copy()



continuous_fields = [
    'VehicleAge', 'VehOdo', 'MMRAcquisitionAuctionAveragePrice',
    'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice',
    'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice',
    'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice',
    'MMRCurrentRetailCleanPrice', 'VehBCost', 'WarrantyCost'
]

categorical_fields = [col for col in X.columns if col not in continuous_fields]



# پر کردن NaN ها قبل از IsolationForest
import numpy as np

# continuous fields
continuous_fields = [
    'VehicleAge', 'VehOdo', 'MMRAcquisitionAuctionAveragePrice',
    'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice',
    'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice',
    'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice',
    'MMRCurrentRetailCleanPrice', 'VehBCost', 'WarrantyCost'
]
inputs_iso[continuous_fields] = inputs_iso[continuous_fields].fillna(inputs_iso[continuous_fields].median())

# categorical fields
categorical_fields = [col for col in inputs_iso.columns if col not in continuous_fields]
for col in categorical_fields:
    inputs_iso[col] = inputs_iso[col].fillna(inputs_iso[col].mode()[0])



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest

# فرض کنید df دیتافریم شماست و RefId اندیس شده است
y = df['IsBadBuy']
X = df.drop(columns=['IsBadBuy'])

# تعریف ستون‌ها
continuous_fields = ['VehicleAge', 'VehOdo', 'MMRAcquisitionAuctionAveragePrice',
                     'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice',
                     'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice',
                     'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice',
                     'MMRCurrentRetailCleanPrice', 'VehBCost', 'WarrantyCost']

categorical_fields = ['Auction', 'Make', 'Color', 'Transmission', 'WheelType',
                      'Nationality', 'Size', 'TopThreeAmericanName', 'IsOnlineSale',
                      'PRIMEUNIT', 'AUCGUART']

# ایجاد کپی برای IsolationForest
iso_df = X.copy()

# 1. پر کردن NaNها
iso_df[continuous_fields] = iso_df[continuous_fields].fillna(iso_df[continuous_fields].median())
for col in categorical_fields:
    iso_df[col] = iso_df[col].fillna(iso_df[col].mode()[0])

# 2. scale کردن continuous ها
scaler = StandardScaler()
iso_df[continuous_fields] = scaler.fit_transform(iso_df[continuous_fields])

# 3. label encode کردن categorical ها
label_encoders = {}
for col in categorical_fields:
    le = LabelEncoder()
    iso_df[col] = le.fit_transform(iso_df[col])
    label_encoders[col] = le  # در صورت نیاز برای decode بعدی

# 4. اعمال IsolationForest
iso = IsolationForest(contamination=0.01, random_state=42)
outliers = iso.fit_predict(iso_df.to_numpy())  # تبدیل به numpy array برای جلوگیری از هشدار

# 5. اضافه کردن ستون outlier
iso_df['outlier'] = outliers
iso_df['outlier'] = iso_df['outlier'].map({1: 0, -1: 1})  # 1 = outlier

# 6. نمایش درصد ردیف‌های outlier
percentage_outliers = iso_df['outlier'].mean() * 100
print(f"Percentage of outliers: {percentage_outliers:.2f}%")

# 7. حذف ردیف‌های outlier از دیتافریم اصلی
df_cleaned = df.loc[iso_df[iso_df['outlier'] == 0].index].copy()
print(f"Shape after removing outliers: {df_cleaned.shape}")




import numpy as np

# ستون‌های قیمت
price_columns = [
    'MMRAcquisitionAuctionAveragePrice',
    'MMRAcquisitionAuctionCleanPrice',
    'MMRAcquisitionRetailAveragePrice',
    'MMRAcquisitonRetailCleanPrice',
    'MMRCurrentAuctionAveragePrice',
    'MMRCurrentAuctionCleanPrice',
    'MMRCurrentRetailAveragePrice',
    'MMRCurrentRetailCleanPrice'
]

# 1. حذف ردیف‌هایی که 4 یا بیشتر NaN در ستون‌های قیمت دارند
df_cleaned = df_cleaned[df_cleaned[price_columns].isna().sum(axis=1) < 4]

# 2. حذف ردیف‌هایی که 50٪ یا بیشتر مقادیرشون NaN هست
df_cleaned = df_cleaned[df_cleaned.isna().mean(axis=1) < 0.5]

# 3. پر کردن باقی‌مانده NaNها
# Continuous
continuous_cols = [
    'VehicleAge', 'VehOdo', 'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
    'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice',
    'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice',
    'VehBCost', 'WarrantyCost'
]
df_cleaned[continuous_cols] = df_cleaned[continuous_cols].fillna(df_cleaned[continuous_cols].median())

# Categorical
categorical_cols = ['Auction', 'Make', 'Color', 'Transmission', 'WheelType', 'Nationality', 'Size', 'TopThreeAmericanName', 'PRIMEUNIT', 'AUCGUART', 'IsOnlineSale']
for col in categorical_cols:
    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

# بررسی وضعیت نهایی
print(df_cleaned.isna().sum())
print(df_cleaned.shape)


