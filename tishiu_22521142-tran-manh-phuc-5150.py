!pip install -q --upgrade scikit-learn imbalanced-learn


import numpy as np
import pandas as pd
import os
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA


base_path = '/kaggle/input/ds-108-p-21-assigment-06/'
file_paths = {
    'delay_4_6': os.path.join(base_path, 'delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv'),
    'not_delay_4_6': os.path.join(base_path, 'not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv'),
    'delay_7_9': os.path.join(base_path, 'delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv'),
    'not_delay_7_9': os.path.join(base_path, 'not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv'),
    'pilot_10': os.path.join(base_path, 'PILOT_10.csv')
}


print("--- Loading Training Data ---")
df_delay_4_6 = pd.read_csv(file_paths['delay_4_6'])
print(f"Loaded delay_4_6: {df_delay_4_6.shape}")

df_not_delay_4_6 = pd.read_csv(file_paths['not_delay_4_6'])
print(f"Loaded not_delay_4_6: {df_not_delay_4_6.shape}")

df_delay_7_9 = pd.read_csv(file_paths['delay_7_9'])
print(f"Loaded delay_7_9: {df_delay_7_9.shape}")

df_not_delay_7_9 = pd.read_csv(file_paths['not_delay_7_9'])
print(f"Loaded not_delay_7_9: {df_not_delay_7_9.shape}")
print("\n")


def preprocess_dataframe(df):
    """
    Hàm này thực hiện các bước tiền xử lý cho DataFrame:
    - Xử lý cột 'Consider count hodiday Saturday' và 'SOUF_RCV_NO' (kiểu dữ liệu hỗn hợp).
    - Tạo các feature thời gian từ 'Order date' và 'VSD'.
    """
    df = df.copy() # Tránh SettingWithCopyWarning

    # Xử lý cột 'Consider count hodiday Saturday'
    col_holiday = 'Consider count hodiday Saturday'
    if col_holiday in df.columns:
        temp_col = df[col_holiday].copy()
        # Chuyển đổi các giá trị chuỗi '0', '1', '2', '3', '4' sang số nguyên
        # và các giá trị khoảng trắng ' ' sang NaN
        df.loc[(temp_col.apply(type) == str) & (temp_col != ' '), col_holiday] = \
            temp_col[(temp_col.apply(type) == str) & (temp_col != ' ')].map({'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}).astype('Int64')
        df.loc[temp_col == ' ', col_holiday] = np.nan
        # Chuyển đổi các giá trị float (nếu có) sang int và sau đó sang Int64 (nullable int)
        df.loc[(temp_col.apply(type) == float) & (temp_col.notnull()), col_holiday] = \
            temp_col[(temp_col.apply(type) == float) & (temp_col.notnull())].apply(lambda x: int(x)).astype('Int64')
    
    # Xử lý cột 'SOUF_RCV_NO'
    col_souf_rcv_no = 'SOUF_RCV_NO'
    if col_souf_rcv_no in df.columns:
        temp_col = df[col_souf_rcv_no].copy()
        # Chuyển đổi các giá trị float (nếu có) sang chuỗi với 6 chữ số
        df.loc[(temp_col.apply(type) == float) & (temp_col.notnull()), col_souf_rcv_no] = \
            temp_col[(temp_col.apply(type) == float) & (temp_col.notnull())].apply(lambda x: str(int(x)).zfill(6))
        # Các giá trị NaN sẽ được giữ nguyên hoặc xử lý bởi Imputer trong pipeline

    # Feature Engineering thời gian
    if 'Order date' in df.columns:
        df['Order date'] = pd.to_datetime(df['Order date'], format='mixed', errors='coerce')
        df['order_dayofweek'] = df['Order date'].dt.dayofweek
        df['order_hour'] = df['Order date'].dt.hour
    
    if 'VSD' in df.columns:
        df['VSD'] = pd.to_datetime(df['VSD'], format='mixed', errors='coerce')
        df['VSD-Order date'] = (df['VSD'] - df['Order date']).dt.days
        df['VSD_dayofweek'] = df['VSD'].dt.dayofweek
        df['VSD_hour'] = df['VSD'].dt.hour

    return df


print("--- Preprocessing DataFrames ---")
df_delay_4_6 = preprocess_dataframe(df_delay_4_6)
df_not_delay_4_6 = preprocess_dataframe(df_not_delay_4_6)
df_delay_7_9 = preprocess_dataframe(df_delay_7_9)
df_not_delay_7_9 = preprocess_dataframe(df_not_delay_7_9)

# Tạo cột 'label' như trong notebook gốc
df_delay_4_6['label'] = 1
df_not_delay_4_6['label'] = 0
df_delay_7_9['label'] = 1
df_not_delay_7_9['label'] = 0
print("Added 'label' column to training dataframes.\n")


print("--- Merge Training Data ---")
# Đảm bảo các cột được chọn là nhất quán giữa các df_delay và df_not_delay
# Lấy danh sách các cột chung giữa df_delay_4_6 và df_delay_7_9
common_cols_delay = df_delay_4_6.columns.intersection(df_delay_7_9.columns).tolist()
df_delay = pd.concat([df_delay_4_6[common_cols_delay].copy(), df_delay_7_9[common_cols_delay].copy()], axis=0, ignore_index=True)

# Lấy danh sách các cột chung giữa df_not_delay_4_6 và df_not_delay_7_9
common_cols_not_delay = df_not_delay_4_6.columns.intersection(df_not_delay_7_9.columns).tolist()
df_not_delay = pd.concat([df_not_delay_4_6[common_cols_not_delay].copy(), df_not_delay_7_9[common_cols_not_delay].copy()], axis=0, ignore_index=True)


# df_all sẽ chứa các cột chung của df_delay và df_not_delay
df_all_common_cols = df_delay.columns.intersection(df_not_delay.columns).tolist()
df_all = pd.concat([df_delay[df_all_common_cols].copy(), df_not_delay[df_all_common_cols].copy()], axis=0, ignore_index=True)

df_extra = pd.concat([df_delay_4_6.copy(), df_not_delay_4_6.copy()], axis=0, ignore_index=True)
print(f"df_all shape: {df_all.shape}")
print(f"df_extra shape: {df_extra.shape}\n")


# Xác định các cột để loại bỏ dựa trên tỷ lệ NaN
nan_ratio_all = df_all.isnull().mean()
df_all_drop_nan = nan_ratio_all[nan_ratio_all > 0.5].index.tolist()
print(f"Columns to drop from df_all (null ratio > 0.5): {df_all_drop_nan}")

nan_ratio_extra = df_extra.isnull().mean()
df_extra_drop_nan = nan_ratio_extra[nan_ratio_extra > 0.5].index.tolist()
print(f"Columns to drop from df_extra (null ratio > 0.5): {df_extra_drop_nan}\n")

# Cập nhật df_all_drop và df_extra_drop với các cột object có unique values > 10 hoặc < 2
# Lưu ý: Các cột có số lượng unique values quá cao (như ID) hoặc quá thấp (chỉ 1 giá trị) thường không hữu ích
cols_to_evaluate_all = df_all.drop(columns=df_all_drop_nan, errors='ignore').select_dtypes(include='object').columns
df_all_drop_cardinality = [col for col in cols_to_evaluate_all if df_all[col].nunique() > 1000 or df_all[col].nunique() < 2] # Giả định ngưỡng 1000 cho cardinality cao
df_all_drop = list(set(df_all_drop_nan + df_all_drop_cardinality)) # Kết hợp và loại bỏ trùng lặp
# Thêm các cột ID rõ ràng nếu chưa có
if 'GLOBAL_NO' in df_all.columns:
    df_all_drop.append('GLOBAL_NO')
if 'PRODUCT_CD' in df_all.columns:
    df_all_drop.append('PRODUCT_CD')
if 'INNER_CD' in df_all.columns:
    df_all_drop.append('INNER_CD')
if 'BRAND_CD' in df_all.columns:
    df_all_drop.append('BRAND_CD')
if 'VSD' in df_all.columns: # Nếu VSD vẫn là datetime sau khi feature engineering, có thể loại bỏ nó
    df_all_drop.append('VSD')
if 'Order date' in df_all.columns: # Tương tự với Order date
    df_all_drop.append('Order date')

df_all_drop = list(set(df_all_drop)) # Loại bỏ trùng lặp lần cuối

print(f"Total columns to drop from df_all: {len(df_all_drop)} columns - {df_all_drop}\n")


cols_to_evaluate_extra = df_extra.drop(columns=df_extra_drop_nan, errors='ignore').select_dtypes(include='object').columns
df_extra_drop_cardinality = [col for col in cols_to_evaluate_extra if df_extra[col].nunique() > 1000 or df_extra[col].nunique() < 2] # Giả định ngưỡng 1000 cho cardinality cao
df_extra_drop = list(set(df_extra_drop_nan + df_extra_drop_cardinality)) # Kết hợp và loại bỏ trùng lặp
# Thêm các cột ID rõ ràng nếu chưa có
if 'GLOBAL_NO' in df_extra.columns:
    df_extra_drop.append('GLOBAL_NO')
if 'PRODUCT_CD' in df_extra.columns:
    df_extra_drop.append('PRODUCT_CD')
if 'INNER_CD' in df_extra.columns:
    df_extra_drop.append('INNER_CD')
if 'BRAND_CD' in df_extra.columns:
    df_extra_drop.append('BRAND_CD')
if 'VSD' in df_extra.columns:
    df_extra_drop.append('VSD')
if 'Order date' in df_extra.columns:
    df_extra_drop.append('Order date')

df_extra_drop = list(set(df_extra_drop)) # Loại bỏ trùng lặp lần cuối

print(f"Total columns to drop from df_extra: {len(df_extra_drop)} columns - {df_extra_drop}\n")


# Chuẩn bị dữ liệu cho mô hình
# Loại bỏ các cột đã xác định và cột 'label' khỏi X
X = df_all.drop(columns=df_all_drop + ['label'], errors='ignore').copy()
y = df_all['label'].copy()

# Chuẩn bị X_extra, y_extra (nếu bạn vẫn muốn giữ phần này)
X_extra = df_extra.drop(columns=df_extra_drop + ['label'], errors='ignore').copy()
y_extra = df_extra['label'].copy()

# Xác định features số và categorical sau khi loại bỏ các cột
numeric_features = X.select_dtypes(exclude='object').columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Define transformers for the preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()) # Giữ nguyên StandardScaler ở đây
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # THAY ĐỔI TẠI ĐÂY: sparse_output=False
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Tách tập train/validation với StratifiedSplit để giữ tỷ lệ lớp
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# X_train_extra, X_val_extra, y_train_extra, y_val_extra = train_test_split(X_extra, y_extra, test_size=0.2, random_state=42) # Nếu bạn dùng df_extra

# Tải dữ liệu test và tiền xử lý
print("--- Loading Test Data ---")
df_pilot_10 = pd.read_csv(file_paths['pilot_10'])
print(f"Loaded pilot_10: {df_pilot_10.shape}\n")

# Áp dụng hàm tiền xử lý lên tập test
df_test = preprocess_dataframe(df_pilot_10)

# Loại bỏ các cột không cần thiết khỏi df_test để khớp với X
# Sử dụng df_all_drop đã được cập nhật ở Cell 6
X_test = df_test.drop(columns=df_all_drop, errors='ignore').copy()

# Đảm bảo tập test có cùng các cột với tập huấn luyện (sau tiền xử lý của preprocessor)
# Tìm các cột bị thiếu trong X_test so với X
missing_cols_in_test = set(X.columns) - set(X_test.columns)
for col in missing_cols_in_test:
    X_test[col] = 0 # Hoặc giá trị mặc định phù hợp, 0 nếu là cột OneHotEncoded

# Tìm các cột thừa trong X_test so với X
extra_cols_in_test = set(X_test.columns) - set(X.columns)
X_test = X_test.drop(columns=list(extra_cols_in_test), errors='ignore')

# Sắp xếp lại thứ tự cột của X_test để khớp với X
X_test = X_test[X.columns]

print(f"X_train shape after processing: {X_train.shape}")
print(f"X_test shape after processing: {X_test.shape}")
print(f"X_train columns: {X_train.columns.tolist()}")
print(f"X_test columns: {X_test.columns.tolist()}")
print(f"Number of columns in X_train: {X_train.shape[1]}")
print(f"Number of columns in X_test: {X_test.shape[1]}")


# Định nghĩa pipeline mô hình
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('oversample', SMOTE(random_state=42)),
    ('scaler', StandardScaler()), # StandardScaler sau SMOTE là hợp lý nếu bạn muốn scale trên dữ liệu đã oversampled
    ('pca', PCA(n_components='mle', svd_solver='full')), # Có thể thử fixed number of components
    ('classifier', LGBMClassifier(random_state=42)) # Thêm random_state để có kết quả reproducible
])

# Huấn luyện mô hình trên tập huấn luyện đã chia
print("--- Training Model on X_train, y_train ---")
model.fit(X_train, y_train)
y_pred_val = model.predict(X_val)

# Huấn luyện lại mô hình trên toàn bộ dữ liệu (để dự đoán trên tập test)
print("--- Training Model on Full Data (X, y) ---")
model.fit(X, y)
y_pred_big = model.predict(X_test)


# Đánh giá hiệu suất trên tập validation
f1 = f1_score(y_val, y_pred_val, average='binary') 
print(f"F1-score on validation set: {f1}")

# In thêm các metric khác để hiểu rõ hơn
from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report on Validation Set:")
print(classification_report(y_val, y_pred_val))
print("\nConfusion Matrix on Validation Set:")
print(confusion_matrix(y_val, y_pred_val))


# Tải file mẫu submission
df_submission = pd.read_csv(os.path.join(base_path, "sample_Solution.csv"))

# Gán kết quả dự đoán vào cột 'label'
df_submission['label'] = y_pred_big

# Lưu file submission
submission_filename = "submission.csv"
df_submission.to_csv(submission_filename, index=False)
print(f"\nSubmission file saved as: {submission_filename}")

# Hiển thị thông tin file submission
print("\nSubmission DataFrame Info:")
df_submission.info()
print("\nSubmission DataFrame Head:")
print(df_submission.head())

