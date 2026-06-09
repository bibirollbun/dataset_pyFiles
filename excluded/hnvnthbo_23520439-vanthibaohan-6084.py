import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


data1 = pd.read_csv ('/kaggle/input/ds-108-p-21-assigment-06/delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv')
data2 = pd.read_csv ('/kaggle/input/ds-108-p-21-assigment-06/not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv')
data3 = pd.read_csv ('/kaggle/input/ds-108-p-21-assigment-06/delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv')
data4 = pd.read_csv ('/kaggle/input/ds-108-p-21-assigment-06/not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv')


test = pd.read_csv ('/kaggle/input/ds-108-p-21-assigment-06/PILOT_10.csv')


data_4_6 = pd.concat([data3, data4], ignore_index=True)
data_7_9 = pd.concat([data1, data2], ignore_index=True)


data_4_6.info()


data_7_9.info()


cols_to_drop = data_4_6.columns.difference(data_7_9.columns)
# Drop các cột này
data_4_6.drop(columns=cols_to_drop, inplace=True)


def convert_datetime(data):
    data['Order date'] = pd.to_datetime(data['Order date'])
    data['VSD'] = pd.to_datetime(data['VSD'])
    data['num_date'] = (data['VSD'] - data['Order date']).dt.days
    data = data.drop(columns=['Order date', 'VSD'])
    return data

data_4_6 = convert_datetime(data_4_6)
data_7_9 = convert_datetime(data_7_9)
test = convert_datetime(test)


columns_to_convert = ['OTHER AREA SHIP DIV', 'REASON_CD', 'Consider count hodiday Saturday']
for col in columns_to_convert:
    if col in data_4_6.columns:
        data_4_6[col] = data_4_6[col].astype(str)
        data_4_6[col] = data_4_6[col].str.replace(' ', '')
        data_4_6[col] = data_4_6[col].replace('nan', np.nan)
        data_4_6[col] = pd.to_numeric(data_4_6[col])


train = pd.concat([data_4_6, data_7_9])


train.info()


train['GLOBAL_NO'] = train['GLOBAL_NO'].astype(str)


for col in train.columns:
    unique_value_count = train[col].nunique(dropna=False)
    if unique_value_count == 1:
        unique_value = train[col].unique()[0]
        print(f"Column '{col}' has only one unique value: '{unique_value}'")


columns_to_drop = ['SUBSIDIARY_CD']
train = train.drop(columns=columns_to_drop, axis=1, errors='ignore')


# Tỉ lệ missing value trong mỗi thuộc tính
total_rows = len(train)
missing_percentages = {}
for col in train.columns:
    missing_count = train[col].isnull().sum()
    if missing_count > 0:
        missing_percentage = (missing_count / total_rows) * 100
        missing_percentages[col] = missing_percentage
        print(f"Column '{col}' has {missing_count} missing values, which is {missing_percentage:.2f}%")


# Xử lí 'OTHER AREA SHIP DIV'
print (train['OTHER AREA SHIP DIV'].unique())
train['OTHER AREA SHIP DIV'] = train['OTHER AREA SHIP DIV'].fillna(0)# Điền 0 - Vận chuyển trong khu vực riêng


# Xử lí các thuộc tính với phần trăm nan nhỏ hơn 10%
for col, percentage in missing_percentages.items():
    if percentage < 10:
        train = train.dropna(subset=[col])


# Xử lí thuộc tính có missing value lớn:
columns_to_drop = ['REASON_CD', 'SOUF_RCV_NO', 'QTUF_RCV_NO']
train = train.drop(columns=columns_to_drop, axis=1, errors='ignore')


# Tính ma trận tương quan cho các thuộc tính số
correlation_matrix = train.corr(numeric_only=True)
# Vẽ heatmap
plt.figure(figsize=(30, 24))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Mức độ tương quan giữa các thuộc tính')
plt.show()


# Kiểm tra những thuộc tính không tương quan với label
non_correlated_features = []
correlation_threshold = 0.01

for col in train.columns:
    if col != 'label' and pd.api.types.is_numeric_dtype(train[col]):
        correlation = train[col].corr(train['label'])
        if abs(correlation.round(decimals=2)) < correlation_threshold:
            non_correlated_features.append(col)

print(non_correlated_features)


columns_to_drop = non_correlated_features
train = train.drop(columns=columns_to_drop, axis=1, errors='ignore')


# Tìm các cặp thuộc tính có độ tương quan cao
threshold = 0.8

high_corr_pairs = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        col1 = correlation_matrix.columns[i]
        col2 = correlation_matrix.columns[j]
        corr_value = correlation_matrix.loc[col1, col2]

        if abs(corr_value) >= threshold:
            high_corr_pairs.append((col1, col2, corr_value))

print(f"\nCác cặp thuộc tính có độ tương quan cao (ngưỡng >= {threshold}):")
if high_corr_pairs:
    for pair in high_corr_pairs:
        print(f"  {pair[0]} - {pair[1]}: {pair[2]:.2f}")
else:
    print("Không tìm thấy cặp thuộc tính nào có độ tương quan cao theo ngưỡng đã đặt.")



columns_to_drop = ['Stock class', 'SO QTY', 'SUPPLIER INV AMOUNT', 'SPECIAL DIV']
train = train.drop(columns=columns_to_drop, axis=1, errors='ignore')


# Kiểm tra và tìm outliers
numeric_columns = train.select_dtypes(include=np.number).columns
outliers_col = []

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_indices = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
    return outlier_indices

def plot_boxplot(df, column):
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot của cột {column}')
    plt.show()
    
print("Kiểm tra outliers cho các thuộc tính không missing:")
for col in numeric_columns:
    if train[col].nunique() > 2 and col not in missing_percentages:
        print(f"\n--- Kiểm tra outliers cho cột '{col}' ---")
        outlier_indices_iqr = detect_outliers_iqr(train, col)
        if not outlier_indices_iqr.empty:
            outliers_col.append(col)
            outlier_percentage = len(outlier_indices_iqr) / len(train) * 100
            print(f"  Percentage of outliers in {col} (IQR): {outlier_percentage:.2f}%")
            # Vẽ Box Plot
            plot_boxplot(train, col)
        else:
            print("  Không có outliers đáng kể theo phương pháp IQR.")


# Xứ lí
for col in outliers_col:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Đưa outliers về giá trị biên dựa trên IQR
    train[col] = np.where(train[col] < lower_bound, lower_bound,
                                                np.where(train[col] > upper_bound, upper_bound, train[col]))
    print(f"Outliers (IQR) ở cột '{col}' đã được đưa về giá trị biên.")


# Loại bỏ các hàng trùng lặp dựa trên tất cả các cột
train.drop_duplicates()


train.info()


cols_to_drop = test.columns.difference(train.columns)
# Drop các cột này
test.drop(columns=cols_to_drop, inplace=True)


test.info()


# Tỉ lệ missing value trong mỗi thuộc tính
total_rows = len(test)
missing_percentages = {}
for col in test.columns:
    missing_count = test[col].isnull().sum()
    if missing_count > 0:
        missing_percentage = (missing_count / total_rows) * 100
        missing_percentages[col] = missing_percentage
        print(f"Column '{col}' has {missing_count} missing values, which is {missing_percentage:.2f}%")


# Xử lí 'OTHER AREA SHIP DIV'
test['OTHER AREA SHIP DIV'] = test['OTHER AREA SHIP DIV'].fillna(0) # Điền 0 - Vận chuyển trong khu vực riêng


# Kiểm tra và tìm outliers
numeric_columns = test.select_dtypes(include=np.number).columns
outliers_col = []

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_indices = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
    return outlier_indices

def plot_boxplot(df, column):
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot của cột {column}')
    plt.show()
    
print("Kiểm tra outliers cho các thuộc tính không missing:")
for col in numeric_columns:
    if test[col].nunique() > 2 and col not in missing_percentages:
        print(f"\n--- Kiểm tra outliers cho cột '{col}' ---")
        outlier_indices_iqr = detect_outliers_iqr(test, col)
        if not outlier_indices_iqr.empty:
            outliers_col.append(col)
            outlier_percentage = len(outlier_indices_iqr) / len(test) * 100
            print(f"  Percentage of outliers in {col} (IQR): {outlier_percentage:.2f}%")
            # Vẽ Box Plot
            plot_boxplot(test, col)
        else:
            print("  Không có outliers đáng kể theo phương pháp IQR.")


# Xứ lí
for col in outliers_col:
    Q1 = test[col].quantile(0.25)
    Q3 = test[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Đưa outliers về giá trị biên dựa trên IQR
    test[col] = np.where(test[col] < lower_bound, lower_bound,
                                                np.where(test[col] > upper_bound, upper_bound, test[col]))
    print(f"Outliers (IQR) ở cột '{col}' đã được đưa về giá trị biên.")


y_train = train['label']
X_train = train.drop(columns='label')


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
from sklearn.pipeline import Pipeline

numeric_features = X_train.select_dtypes(include=['int64','float64']).columns
categorical_features = X_train.select_dtypes(include='object').columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Các tham số cụ thể
lgbm_classifier = lgb.LGBMClassifier(
    objective='binary',
    metric='logloss',
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    reg_alpha=0.1,
    reg_lambda=0.1,
)

# Xây dựng Pipeline
model_lgbm = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lgbm_classifier)
])

# Huấn luyện mô hình
model_lgbm.fit(X_train, y_train)
print("Huấn luyện mô hình hoàn tất.")


# Thực hiện dự đoán nhị phân trên tập kiểm tra 
y_pred = model_lgbm.predict(test)

print(f"Kích thước của mảng dự đoán: {y_pred.shape}")


# Tạo DataFrame nộp bài
id = range(1, len(test) + 1)

# Tạo DataFrame nộp bài
submission = pd.DataFrame({
    'ID': id,
    'label': y_pred
})

# Lưu DataFrame vào file CSV
submission.to_csv('submission.csv', index=False)

