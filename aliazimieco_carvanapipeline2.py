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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.impute import SimpleImputer


def load_and_preprocess_data(file_path):
    df = pd.read_csv('/kaggle/input/DontGetKicked/training.csv')
    df = df.set_index("RefId")
    return df


def initial_preprocessing(data):
    columns_drop = [
        "PurchDate", "VehYear", "Model", "Trim", 
        "SubModel", "WheelTypeID", "BYRNO", 
        "VNZIP1", "VNST", "PRIMEUNIT", "AUCGUART"
    ]
    processed_data = data.drop(columns_drop, axis=1)
    return processed_data


def split_data(X, y, test_size=0.2, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def correct_inconsistencies(data):
    data["Transmission"] = data["Transmission"].replace("Manual", "MANUAL")
    data["Color"] = data["Color"].replace("NOT AVAIL", np.nan)
    return data


def separate_features(data):
    categorical_fields = [
        "Auction", "Make", "Color", "Transmission",
        "WheelType", "Nationality", "Size", 
        "TopThreeAmericanName", "IsOnlineSale"
    ]
    continuous_fields = [col for col in data.columns if col not in categorical_fields]
    return categorical_fields, continuous_fields


def replace_rare_classes(data, column, threshold=10):
    unique_elements, counts = np.unique(data[column].dropna(), return_counts=True)
    percentage = (counts / len(data[column])) * 100
    rare_classes = [elem for elem, pct in zip(unique_elements, percentage) if pct < threshold]
    return data[column].replace(rare_classes, 'OTHER')


def feature_screening(data, continuous_fields, categorical_fields, 
                      min_cv=0.1, mode_threshold=99, distinct_threshold=90):
    
    # CV-based screening
    cv_values = data[continuous_fields].std() / data[continuous_fields].mean()
    screen_cv = cv_values[cv_values < min_cv].index.tolist()
    
    # Mode-based screening
    mode_percentage = data[categorical_fields].apply(
        lambda x: x.value_counts().max() / len(x) * 100
    )
    screen_mode = mode_percentage[mode_percentage > mode_threshold].index.tolist()
    
    # Distinct categories screening
    distinct_percentage = data[categorical_fields].apply(
        lambda x: x.dropna().nunique() / x.count() * 100
    )
    screen_distinct = distinct_percentage[distinct_percentage > distinct_threshold].index.tolist()
    
    screened_features = list(set(screen_cv + screen_mode + screen_distinct))
    return screened_features


def handle_out_of_range(data, column_ranges):
    for column, (min_val, max_val) in column_ranges.items():
        data[column] = data[column].apply(lambda x: x if min_val <= x <= max_val else np.nan)
    return data


def detect_outliers(data, contamination=0.01):
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    
    processed_data = data.copy().dropna()
    processed_data[continuous_fields] = scaler.fit_transform(processed_data[continuous_fields])
    processed_data[categorical_fields] = processed_data[categorical_fields].apply(label_encoder.fit_transform)
    
    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(processed_data)
    outliers = clf.predict(processed_data)
    return processed_data[outliers == -1].index


def handle_missing_values(data, price_columns, max_missing_price=4, max_missing_row=5, max_missing_col=50):
    # Step 1: Remove rows with too many missing values in price columns
    data["num_missing_price"] = data[price_columns].isnull().sum(axis=1)
    data = data[data["num_missing_price"] < max_missing_price].drop(columns=["num_missing_price"])
    
    # Step 2: Remove rows with excessive missing values
    data["num_missing_total"] = data.isnull().sum(axis=1)
    data = data[data["num_missing_total"] <= max_missing_row].drop(columns=["num_missing_total"])
    
    # Step 3: Remove columns with too many missing values
    missing_percentage = data.isnull().mean() * 100
    cols_to_drop = missing_percentage[missing_percentage > max_missing_col].index.tolist()
    data = data.drop(cols_to_drop, axis=1)
    
    return data


def impute_missing_values(train, test, categorical_fields, continuous_fields):
    cat_imputer = SimpleImputer(strategy='most_frequent')
    con_imputer = SimpleImputer(strategy='median')
    
    train[categorical_fields] = cat_imputer.fit_transform(train[categorical_fields])
    train[continuous_fields] = con_imputer.fit_transform(train[continuous_fields])
    
    test[categorical_fields] = cat_imputer.transform(test[categorical_fields])
    test[continuous_fields] = con_imputer.transform(test[continuous_fields])
    
    return train, test


def discretize_features(train, test, y, columns):
    trans_cm = cm.ChiMerge(max_intervals=5, min_intervals=1, decimal=3, output_dataframe=True)
    trans_cm.fit(train[columns], y.astype('int').squeeze())
    
    for col in columns:
        boundaries = np.insert(trans_cm.boundaries_[col], 0, -np.inf)
        train[f"{col}_cat_cm"] = pd.cut(train[col], bins=boundaries, labels=False, right=False)
        test[f"{col}_cat_cm"] = pd.cut(test[col], bins=boundaries, labels=False, right=False)
    
    return train, test


!pip install scorecardbundle



from scorecardbundle.feature_discretization import ChiMerge as cm


def transform_features(train, test, columns):
    for col in columns:
        has_negative = (train[col] <= 0).any()
        transformer = PowerTransformer(
            method='yeo-johnson' if has_negative else 'box-cox', 
            standardize=False
        )
        train[f"{col}_transformed"] = transformer.fit_transform(train[[col]])
        test[f"{col}_transformed"] = transformer.transform(test[[col]])
    return train, test

# اجرای توابع
if __name__ == "__main__":
    # بارگذاری داده
    df = load_and_preprocess_data("/kaggle/input/DontGetKicked/training.csv")
    
    # پیشپردازش اولیه
    df = initial_preprocessing(df)
    
    # تقسیم داده
    y = df.iloc[:, 0:1]
    X = df.iloc[:, 1:]
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # اصلاح ناسازگاری‌ها
    X_train = correct_inconsistencies(X_train)
    X_test = correct_inconsistencies(X_test)
    
    # جداسازی ویژگی‌ها
    categorical_fields, continuous_fields = separate_features(X_train)
    
    # جایگزینی کلاس‌های نادر
    X_train['Make'] = replace_rare_classes(X_train, 'Make', threshold=1)
    X_train['Color'] = replace_rare_classes(X_train, 'Color', threshold=1)
    
    # غربالگری ویژگی‌ها
    drop_list = feature_screening(X_train, continuous_fields, categorical_fields)
    X_train = X_train.drop(drop_list, axis=1)
    X_test = X_test.drop(drop_list, axis=1)
    
    # محدوده‌بندی مقادیر
    column_ranges = {
        'VehicleAge': (0, 30),
        'VehOdo': (0, 120000),
        # ... (بقیه محدوده‌ها)
    }
    X_train = handle_out_of_range(X_train, column_ranges)
    X_test = handle_out_of_range(X_test, column_ranges)
    
    # شناسایی و حذف نقاط پرت
    outlier_index = detect_outliers(X_train)
    X_train = X_train.drop(outlier_index)
    y_train = y_train.drop(outlier_index)
    
    # مدیریت مقادیر گمشده
    price_columns = [
        "MMRAcquisitionAuctionAveragePrice",
        "MMRAcquisitionAuctionCleanPrice",
        # ... (بقیه ستون‌های قیمتی)
    ]
    X_train = handle_missing_values(X_train, price_columns)
    X_test = handle_missing_values(X_test, price_columns)
    
    # جایگزینی مقادیر گمشده
    X_train, X_test = impute_missing_values(X_train, X_test, categorical_fields, continuous_fields)
    
    # دیسکریت‌سازی
    chi_merge_cols = ['VehBCost', 'WarrantyCost']
    X_train, X_test = discretize_features(X_train, X_test, y_train, chi_merge_cols)
    
    # تبدیل ویژگی‌ها
    transform_cols = ['VehBCost', 'WarrantyCost']
    X_train, X_test = transform_features(X_train, X_test, transform_cols)
    
    # تبدیل نوع داده نهایی
    X_train['IsOnlineSale'] = X_train['IsOnlineSale'].astype(int)
    X_test['IsOnlineSale'] = X_test['IsOnlineSale'].astype(int)




