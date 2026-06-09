!pip install ydata_profiling


import pandas as pd
df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
df.info


#  حذف فیچرهای نامناسب در صورت مسئله
Columns_drop = [
    'PurchDate', 'VehYear', 'Model', 'Trim', 'SubModel',
    'WheelTypeID', 'BYRNO', 'VNZIP1', 'VNST'
]

df.drop(columns=Columns_drop, inplace=True, errors='ignore')
df.info()
df.describe()


df.set_index('RefId', inplace=True)


y = df['IsBadBuy']
X = df.drop(columns=['IsBadBuy'])


from sklearn.model_selection import train_test_split

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

inputs = X_train


# دیکشنری محدوده منطقی برای هر ستون
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

# اعمال محدودیت‌ها و تبدیل مقادیر خارج از محدوده به NaN
for col, (min_val, max_val) in valid_ranges.items():
    X_train[col] = X_train[col].apply(lambda x: x if min_val <= x <= max_val else None)
# Display the updated DataFrame
print(X_train[col])
X_train.describe()
X_train.info()



import pandas as pd

#  تبدیل 'NOT AVAIL' به Null
X_train['Color'] = X_train['Color'].replace('NOT AVAIL', pd.NA)

############################################
def merge_rare_categories(series, threshold=0.01):
    freq = series.value_counts(normalize=True)
    rare_labels = freq[freq < threshold].index
    return series.apply(lambda x: 'OTHER' if x in rare_labels else x)

X_train['Color'] = merge_rare_categories(X_train['Color'])
X_train['Make']  = merge_rare_categories(X_train['Make'])



# انتخاب ستون‌های عددی
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

# شناسایی ستون‌های با ضریب تغییر کم (<0.1)
low_variance_cols = [col for col in numeric_cols if X_train[col].std() / X_train[col].mean() < 0.1]

# حذف ستون‌ها
X_train.drop(columns=low_variance_cols, inplace=True)

print("Continuous features dropped due to low variance:", low_variance_cols)


# انتخاب ستون‌های دسته‌ای
categorical_cols = X_train.select_dtypes(include=['object']).columns

# شناسایی ستون‌های با بیش از 99% مقدار یکسان
highly_skewed_cols = [col for col in categorical_cols if (X_train[col].value_counts(normalize=True).max() > 0.99)]

# حذف ستون‌ها
X_train.drop(columns=highly_skewed_cols, inplace=True)

print("Categorical features dropped (99% same):", highly_skewed_cols)


# شناسایی ستون‌های دسته‌ای با بیش از 90% مقادیر یکتا
high_unique_cols = [col for col in categorical_cols if (X_train[col].nunique() / len(X_train)) > 0.9]

# حذف ستون‌ها
X_train.drop(columns=high_unique_cols, inplace=True)

print("Categorical features dropped (90% unique):", high_unique_cols)



from scipy.stats import fisher_exact
import pandas as pd

for col in ['PRIMEUNIT','AUCGUART']:
    if col in X_train.columns:
        # فقط ردیف‌های غیر Null
        temp_df = pd.concat([X_train[col], y_train], axis=1).dropna()
        
        if temp_df.empty:
            print(f"{col}: no non-null data, skipping")
            continue
        
        # جدول 2x2
        contingency_table = pd.crosstab(temp_df[col], temp_df['IsBadBuy'])
        
        if contingency_table.shape == (2,2):
            # آزمون Fisher
            _, p = fisher_exact(contingency_table)
            print(f"{col} -> p-value (Fisher): {p:.4f}")
            
            if p < 0.05:
                X_train[col] = X_train[col].fillna('UNKNOWN')
                print(f"{col}: significant association, missing replaced with 'UNKNOWN'")
            else:
                X_train.drop(columns=[col], inplace=True)
                print(f"{col}: no significant association, column dropped")
        else:
            print(f"{col}: not a 2x2 table, skipping Fisher")



import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder

# دیتافریم ورودی
inputs_iso = X_train.copy()

# حذف ردیف‌هایی که مقادیر NaN دارند
inputs_iso = inputs_iso.dropna()

# شناسایی خودکار ستون‌های عددی و دسته‌ای
continuous_fields = inputs_iso.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_fields = inputs_iso.select_dtypes(include=['object']).columns.tolist()

# مقیاس‌بندی ستون‌های عددی
scaler = StandardScaler()
inputs_iso[continuous_fields] = scaler.fit_transform(inputs_iso[continuous_fields])

# کدگذاری ستون‌های دسته‌ای
for col in categorical_fields:
    inputs_iso[col] = LabelEncoder().fit_transform(inputs_iso[col])

# اجرای IsolationForest
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(inputs_iso)

# پیش‌بینی Outlier
outliers = clf.predict(inputs_iso)

# اضافه کردن ستون Outlier
inputs_iso['outlier'] = outliers

# نمایش دیتا با اطلاعات Outlier
print(inputs_iso.head())

# محاسبه درصد داده‌های پرت
percentage_outliers = (outliers[outliers == -1].shape[0] / len(outliers)) * 100
print(f"Percentage of outliers: {percentage_outliers:.2f}%")



# پیدا کردن ایندکس داده‌های پرت
outlier_index = inputs_iso[inputs_iso['outlier'] == -1].index

# حذف داده‌های پرت از دیتافریم ورودی اصلی
inputs_outprep = X_train.drop(outlier_index)

# حذف همان ایندکس‌ها از y_train
y_train_outprep = y_train.drop(outlier_index)

# ترکیب دوباره ورودی‌ها و هدف برای دیتا آماده
train_outprep = pd.concat([inputs_outprep, y_train_outprep], axis=1)

# نمایش چند ردیف اول
print(train_outprep.head())


# ستون‌های مرتبط با قیمت
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

# شناسایی ردیف‌هایی با ۴ یا بیشتر مقدار null در ستون‌های قیمتی
rows_to_drop_price = train_outprep[price_columns].isnull().sum(axis=1) >= 4

# حذف این ردیف‌ها
train_outprep = train_outprep[~rows_to_drop_price]

print(f"Remaining rows after dropping based on price nulls: {train_outprep.shape[0]}")


# درصد مقادیر null در هر ردیف
row_null_percent = train_outprep.isnull().mean(axis=1)

# حذف ردیف‌هایی که بیش از 50% مقادیرشان null است
train_outprep = train_outprep[row_null_percent <= 0.5]

print(f"Remaining rows after dropping rows with >50% nulls: {train_outprep.shape[0]}")


# شناسایی ستون‌های عددی و دسته‌ای
numeric_cols = train_outprep.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = train_outprep.select_dtypes(include=['object']).columns

# جایگزینی مقادیر null در ستون‌های عددی با میانه
train_outprep[numeric_cols] = train_outprep[numeric_cols].fillna(train_outprep[numeric_cols].median())

# جایگزینی مقادیر null در ستون‌های دسته‌ای با مد
for col in categorical_cols:
    train_outprep[col] = train_outprep[col].fillna(train_outprep[col].mode()[0])

# بررسی نهایی
print(train_outprep.isnull().sum().sum(), "null values remaining")


