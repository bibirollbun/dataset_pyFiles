import pandas as pd

# بارگذاری داده (اسم فایل رو خودت جایگزین کن)
df = pd.read_csv("/kaggle/input/DontGetKicked/training.csv")

# حذف ستون‌های نامناسب
cols_to_drop = [
    "PurchDate", "VehYear", "Model", "Trim", "SubModel",
    "WheelTypeID", "BYRNO", "VNZIP1", "VNST"
]
df = df.drop(columns=cols_to_drop, errors="ignore")

# تنظیم RefId به عنوان index
if "RefId" in df.columns:
    df = df.set_index("RefId")

# نمایش اطلاعات اولیه
print(df.shape)
print(df.head())



from sklearn.model_selection import train_test_split

# تعریف y (ستون هدف) و X (ویژگی‌ها)
y = df["IsBadBuy"]
X = df.drop(columns=["IsBadBuy"])

# تقسیم داده به Train (80%) و Test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# نمایش شکل داده‌ها
print("Train X:", X_train.shape, "Train y:", y_train.shape)
print("Test X:", X_test.shape, "Test y:", y_test.shape)



from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. جدا کردن ستون‌های عددی و دسته‌ای
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X_train.select_dtypes(include=["object"]).columns

# 2. ساخت pipeline برای ستون‌های عددی
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),   # جایگزینی مقادیر گمشده با میانه
    ("scaler", StandardScaler())                     # نرمال‌سازی داده‌های عددی
])

# 3. ساخت pipeline برای ستون‌های دسته‌ای
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # جایگزینی NaN با پرتکرارترین مقدار
    ("encoder", OneHotEncoder(handle_unknown="ignore"))    # تبدیل به کد باینری
])

# 4. ترکیب دو بخش در ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 5. اعمال روی داده‌ها
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Train shape after preprocessing:", X_train_processed.shape)
print("Test shape after preprocessing:", X_test_processed.shape)



# تعریف محدوده‌های منطقی برای هر ستون
valid_ranges = {
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



import numpy as np

valid_ranges = {
    'VehicleAge': (0, 30),
    'VehOdo': (0, 120000)
}

for col, (low, high) in valid_ranges.items():
    if col in X_train.columns:  # فقط اگر ستون وجود داشت
        X_train.loc[~X_train[col].between(low, high), col] = np.nan
    if col in X_test.columns:
        X_test.loc[~X_test[col].between(low, high), col] = np.nan



# شمارش مقادیر Null بعد از این مرحله
print("Nulls in Train after applying ranges:\n", X_train.isnull().sum())
print("\nNulls in Test after applying ranges:\n", X_test.isnull().sum())



import pandas as pd
import numpy as np

# --- ۱. ستون‌های Price ---
price_cols = [col for col in X_train.columns if 'Price' in col]

# حذف ردیف‌هایی با ۴ یا بیشتر Null در ستون‌های قیمتی
X_train = X_train[X_train[price_cols].isnull().sum(axis=1) < 4]
X_test = X_test[X_test[price_cols].isnull().sum(axis=1) < 4]

# --- ۲. حذف ردیف‌هایی با بیش از 50% Null در کل ستون‌ها ---
X_train = X_train[X_train.isnull().mean(axis=1) < 0.5]
X_test = X_test[X_test.isnull().mean(axis=1) < 0.5]

# --- ۳. جای‌گذاری Nullها ---
for col in X_train.columns:
    if X_train[col].dtype in [np.float64, np.int64]:
        # عددی → Median
        median_train = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_train)
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(median_train)
    else:
        # دسته‌ای → Mode
        mode_train = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode_train)
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(mode_train)

# --- ۴. ستون‌های خیلی Null دار مثل PRIMEUNIT و AUCGUART ---
for col in ['PRIMEUNIT', 'AUCGUART']:
    if col in X_train.columns:
        # اگر همبستگی معنی‌دار با y نیست → حذف
        X_train.drop(columns=[col], inplace=True)
        if col in X_test.columns:
            X_test.drop(columns=[col], inplace=True)

# --- ۵. بررسی نهایی ---
print("Nulls in Train after filling:", X_train.isnull().sum())
print("Nulls in Test after filling:", X_test.isnull().sum())



import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- ۱. ستون‌های دسته‌ای ---
categorical_cols = ['Make', 'Color', 'Transmission', 'WheelType', 
                    'Nationality', 'Size', 'TopThreeAmericanName']

# ساخت یک دیکشنری برای LabelEncoder هر ستون
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # آموزش روی داده Train
    X_train[col] = le.fit_transform(X_train[col])
    # اعمال روی داده Test
    if col in X_test.columns:
        X_test[col] = le.transform(X_test[col])
    encoders[col] = le  # نگه داشتن Encoder برای استفاده بعدی

# --- ۲. ستون‌های عددی ---
numeric_cols = ['VehicleAge', 'VehOdo', 
                'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
                'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
                'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice',
                'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice',
                'VehBCost', 'WarrantyCost']

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# --- ۳. بررسی نهایی ---
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("نمونه داده‌ها بعد از Encode و Scale:\n", X_train.head())





