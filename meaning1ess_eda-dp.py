#Tải các thư viện

import numpy as np 
import pandas as pd 

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

#Kiểm tra thư mục

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




train = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
test = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')

print(f"Kích thước dataset: {train.shape}")
print(f"Kích thước test: {test.shape}")


#In thông tin cơ bản của dataset
print(train.info())


#Kiểm tra số lượng unique trong các cột object
print("Số lượng giá trị unique cho các cột có kiểu 'object':")
for col in train.columns:
    if train[col].dtype == 'object':
        num_unique = train[col].nunique()
        print(f"- Cột '{col}': {num_unique} giá trị unique")


#Chuyển cột thời gian về đúng định dạng
train['sale_date'] = pd.to_datetime(train['sale_date'])
train['year_sale'] = train['sale_date'].dt.year


#Các cột năm cần quan sát
year_columns = ['year_sale', 'join_year', 'year_built', 'year_reno']

#Vẽ matplotlb
plt.figure(figsize=(15, 10))

for i, col in enumerate(year_columns):
    plt.subplot(2, 2, i + 1)
    sns.histplot(train[col].dropna(), kde=True, bins=30)
    plt.title(f'Phân bố của {col}')
    plt.xlabel(col)
    plt.ylabel('Tần suất')

plt.tight_layout()
plt.savefig('histograms_years.png')
plt.show()


binary_view_columns = [
    'wfnt', 'golf', 'greenbelt', 'noise_traffic',
    'view_rainier', 'view_olympics', 'view_cascades',
    'view_territorial', 'view_skyline', 'view_sound',
    'view_lakewash', 'view_lakesamm', 'view_otherwater', 'view_other'
]

plt.figure(figsize=(20, 25))  # Tổng thể figure

for i, col in enumerate(binary_view_columns):
    # Đếm số lượng "Không" (0) và "Có" (khác 0)
    count_0 = (train[col] == 0).sum()
    count_nonzero = (train[col] != 0).sum()

    if count_0 == 0 or count_nonzero == 0:
        print(f"Cột '{col}' chỉ có 1 loại giá trị, không vẽ.")
        continue

    # Chuẩn bị pie chart
    labels = ['Không (0)', 'Có (khác 0)']
    sizes = [count_0, count_nonzero]
    colors = sns.color_palette('pastel')

    # Tạo subplot và pie chart
    ax = plt.subplot(5, 3, i + 1)
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,  # Không in label trực tiếp lên pie
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )

    ax.set_title(f'Phân bố của {col}')
    ax.axis('equal')  # Hình tròn

    # Thêm legend thay vì label trực tiếp
    ax.legend(wedges, labels, title="Giá trị", loc="center left", bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig('binary_features_zero_vs_nonzero_pie_with_legend.png', bbox_inches='tight')
plt.show()


def extract_date_features(df, date_column='sale_date'):
    """
    Trích xuất các đặc trưng từ cột ngày tháng:
    - 'age_built': tuổi của ngôi nhà tại thời điểm bán
    - 'age_reno': số năm kể từ lần cải tạo (nếu có)
    Sau khi trích xuất, loại bỏ các cột gốc: 'sale_date', 'year_sale', 'year_built', 'year_reno'
    """
    df_copy = df.copy()

    if date_column in df_copy.columns:
        # Chuyển đổi sang định dạng datetime
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        
        # Trích xuất năm bán
        df_copy['year_sale'] = df_copy[date_column].dt.year

        # Tính tuổi nhà
        df_copy['age_built'] = df_copy['year_sale'] - df_copy['year_built']
        
        # Tính số năm kể từ lần cải tạo (nếu có), nếu không thì NaN
        df_copy['age_reno'] = np.where(df_copy['year_reno'] > 0,
                                       df_copy['year_sale'] - df_copy['year_reno'],
                                       0)

        # Loại bỏ các cột gốc không còn cần thiết
        df_copy = df_copy.drop(columns=[date_column, 'year_sale', 'year_built', 'year_reno'])

        print(f"Đã trích xuất các đặc trưng từ {date_column}")

    return df_copy

# Chạy trên 2 tập dữ liệu
train_with_date_features = extract_date_features(train, 'sale_date')
test_with_date_features = extract_date_features(test, 'sale_date')


categorical_columns = ['sale_warning', 'join_status', 'city', 'zoning', 'subdivision', 'submarket', 'join_year']

def preprocess_categorical(df, categorical_cols, label_encoders=None, is_training=True):
    """
    Tiền xử lý các cột phân loại bằng Label Encoding.
    Nếu là dữ liệu huấn luyện, sẽ khởi tạo và huấn luyện encoder.
    Nếu là dữ liệu kiểm tra, sử dụng encoder đã huấn luyện để mã hóa.

    Tham số:
    - df: DataFrame gốc
    - categorical_cols: danh sách các cột phân loại cần xử lý
    - label_encoders: dict chứa các LabelEncoder (chỉ dùng khi is_training=False)
    - is_training: True nếu đang xử lý tập huấn luyện, False nếu là tập kiểm tra/test

    Trả về:
    - df_processed: DataFrame đã được mã hóa
    - encoders: dict các LabelEncoder (chỉ trả về nếu là is_training=True)
    """
    df_processed = df.copy()

    if is_training:
        encoders = {}
        for col in categorical_cols:
            if col in df_processed.columns:
                le = LabelEncoder()
                
                # Thay thế giá trị thiếu bằng 'unknown'
                df_processed[col] = df_processed[col].fillna('unknown')
                df_processed[col] = df_processed[col].astype(str)

                # Đảm bảo 'unknown' nằm trong danh sách các giá trị duy nhất
                unique_values = df_processed[col].unique().tolist()
                if 'unknown' not in unique_values:
                    unique_values.append('unknown')

                # Huấn luyện encoder và biến đổi dữ liệu
                le.fit(unique_values)
                df_processed[col] = le.transform(df_processed[col])
                
                # Lưu lại encoder cho cột đó
                encoders[col] = le

        return df_processed, encoders

    else:
        for col in categorical_cols:
            if col in df_processed.columns and col in label_encoders:
                df_processed[col] = df_processed[col].fillna('unknown')
                df_processed[col] = df_processed[col].astype(str)

                le = label_encoders[col]

                # Gán 'unknown' cho các giá trị mới chưa từng gặp
                unknown_mask = ~df_processed[col].isin(le.classes_)
                df_processed.loc[unknown_mask, col] = 'unknown'

                # Mã hóa dữ liệu theo encoder đã huấn luyện
                df_processed[col] = le.transform(df_processed[col])

        return df_processed

# Gọi hàm cho tập huấn luyện
train_processed, label_encoders = preprocess_categorical(train_with_date_features, categorical_columns, is_training=True)


# Lấy danh sách các cột kiểu số (numeric) trong DataFrame
numeric_columns = train_processed.select_dtypes(include=[np.number]).columns.tolist()

# Loại bỏ cột đích (target) 'sale_price' ra khỏi danh sách vì không nên xử lý thiếu ở cột này tại bước này
if 'sale_price' in numeric_columns:
    numeric_columns.remove('sale_price')

# Duyệt qua từng cột số còn lại
for col in numeric_columns:
    # Kiểm tra nếu cột có giá trị thiếu
    if train_processed[col].isnull().sum() > 0:
        # Tính giá trị trung vị (median) của cột đó
        median_val = train_processed[col].median()
        # Điền các giá trị thiếu bằng median
        train_processed[col] = train_processed[col].fillna(median_val)
        # In ra thông báo để theo dõi
        print(f"Điền {col} với giá trị median: {median_val}")


#Tách dữ liệu ra tập dữ liệu feature (X) và label (y)
feature_columns = [col for col in train_processed.columns if col not in ['sale_price']]
X = train_processed[feature_columns]
y = train_processed['sale_price']
#Kiểm tra lại
print(f"\nSố lượng features: {X.shape[1]}")


#Chia dữ liệu
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

print(f"\nKích thước tập train: {X_train.shape}")
print(f"Kích thước tập validation: {X_val.shape}")
#Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("\nĐã hoàn thành chuẩn hóa dữ liệu")


print(f"\nThống kê về sale_price:")
print(f"Mean: {y.mean():,.0f}")
print(f"Median: {y.median():,.0f}")
print(f"Std: {y.std():,.0f}")
print(f"Min: {y.min():,.0f}")
print(f"Max: {y.max():,.0f}")


# Áp dụng hàm xử lý categorical giống như tập huấn luyện, sử dụng các LabelEncoder đã được fit từ train
test_processed = preprocess_categorical(
    test_with_date_features, 
    categorical_columns, 
    label_encoders, 
    is_training=False
)

# Duyệt qua các cột feature, nếu cột tồn tại và là kiểu số có giá trị thiếu → điền bằng median từ tập huấn luyện
for col in feature_columns:
    if (
        col in test_processed.columns 
        and test_processed[col].dtype in ['int64', 'float64'] 
        and test_processed[col].isnull().sum() > 0
    ):
        # Sử dụng median từ train để tránh data leakage
        median_val = train_processed[col].median()
        test_processed[col] = test_processed[col].fillna(median_val)

# Trích xuất các cột đặc trưng theo danh sách feature_columns
X_test = test_processed[feature_columns]

# Chuẩn hóa dữ liệu test theo scaler đã fit từ tập huấn luyện
X_test_scaled = scaler.transform(X_test)

# In ra kích thước tập test và số lượng đặc trưng sau xử lý
print(f"Kích thước tập test sau xử lý: {X_test_scaled.shape}")
print(f"Đã xử lý tập test với {len(feature_columns)} features")


os.makedirs('processed', exist_ok=True)

joblib.dump(scaler, 'processed/scaler.pkl')
joblib.dump(label_encoders, 'processed/label_encoders.pkl')
joblib.dump(feature_columns, 'processed/feature_columns.pkl')

np.save('processed/X_train_scaled.npy', X_train_scaled)
np.save('processed/X_val_scaled.npy', X_val_scaled)
np.save('processed/X_test_scaled.npy', X_test_scaled)
np.save('processed/y_train.npy', y_train.values)
np.save('processed/y_val.npy', y_val.values)

print("\nĐã lưu tất cả dữ liệu đã xử lý:")
print("- processed/scaler.pkl")
print("- processed/label_encoders.pkl")
print("- processed/feature_columns.pkl")
print("- processed/X_train_scaled.npy")
print("- processed/X_val_scaled.npy")
print("- processed/X_test_scaled.npy")
print("- processed/y_train.npy")
print("- processed/y_val.npy")

print("\nHoàn thành xử lý dữ liệu!")

