# Bước 1: Tải và Khám phá Dữ liệu (Load & Explore Data)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Tải dữ liệu từ các file csv
# ---
# Giả sử bạn đã tải file train.csv và test.csv từ Kaggle
# và đặt chúng trong cùng một thư mục với file code này.
try:
    train_df = pd.read_csv('/kaggle/input/neolen-house-price-prediction/train.csv')
    test_df = pd.read_csv('/kaggle/input/neolen-house-price-prediction/test.csv')
except FileNotFoundError:
    print("Vui lòng tải file train.csv và test.csv từ trang cuộc thi Kaggle.")
    # Bạn có thể tải tại: https://www.kaggle.com/competitions/neolen-house-price-prediction/data
    exit()


# Giữ lại ID của tập test để tạo file submission sau này
test_ids = test_df['Id']

# Để xử lý đồng bộ, chúng ta sẽ tạm thời bỏ cột 'Id' và 'SalePrice'
# và gộp 2 dataframe lại với nhau.
train_df = train_df.drop('Id', axis=1)
test_df = test_df.drop('Id', axis=1)

# Lấy ra biến mục tiêu (target) và bỏ nó khỏi tập huấn luyện
y = train_df['SalePrice']
train_df = train_df.drop('SalePrice', axis=1)

# Gộp train và test để thực hiện feature engineering đồng bộ
# Điều này đảm bảo các phép biến đổi được áp dụng nhất quán
all_data = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

print("Kích thước dữ liệu gộp:", all_data.shape)
print("Số lượng đặc trưng ban đầu:", all_data.shape[1])
all_data.head()


# Bước 2: Tiền xử lý và Kỹ thuật Đặc trưng (Preprocessing & Feature Engineering)
# 2.1. Phân tích và Xử lý Biến Mục tiêu (SalePrice)
# Vẽ biểu đồ phân phối của SalePrice
sns.displot(y, kde=True)
plt.title('Phân phối của SalePrice (trước khi biến đổi)')
plt.show()

# Nhận xét: Phân phối lệch phải (positive skew)
# Áp dụng biến đổi logarit để đưa về phân phối chuẩn hơn
y_log = np.log1p(y)

sns.displot(y_log, kde=True)
plt.title('Phân phối của SalePrice (sau khi biến đổi log)')
plt.show()

# Từ giờ, chúng ta sẽ huấn luyện mô hình để dự đoán y_log
# và chuyển đổi ngược lại (np.expm1) khi tạo kết quả cuối cùng.


# 2.2. Xử lý Giá trị bị thiếu (Handling Missing Values)
# Tính toán tỷ lệ giá trị thiếu cho mỗi cột
missing_vals = all_data.isnull().sum()
missing_vals_percent = (missing_vals / len(all_data)) * 100
missing_df = pd.DataFrame({'Count': missing_vals, 'Percent': missing_vals_percent})
missing_df = missing_df[missing_df['Count'] > 0].sort_values(by='Percent', ascending=False)

print("Các cột có giá trị thiếu:")
print(missing_df)

# Xử lý các giá trị thiếu dựa trên mô tả dữ liệu (data_description.txt)
# ---
# Một số giá trị 'NA' không thực sự là "thiếu", mà nó mang ý nghĩa "Không có"
# Ví dụ: PoolQC = NA nghĩa là "Không có hồ bơi".
# Chúng ta sẽ điền 'None' cho các cột categorical và 0 cho các cột numerical tương ứng.

cols_fill_none = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType',
    'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'
]
for col in cols_fill_none:
    all_data[col] = all_data[col].fillna('None')

cols_fill_zero = [
    'GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
]
for col in cols_fill_zero:
    all_data[col] = all_data[col].fillna(0)

# Đối với các cột còn lại, ta có thể điền bằng giá trị mode (phổ biến nhất)
# vì chúng là biến categorical.
cols_fill_mode = ['MSZoning', 'Utilities', 'Functional', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType', 'Electrical']
for col in cols_fill_mode:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# LotFrontage: Diện tích tiếp xúc với đường. Giá trị này có thể liên quan đến
# khu vực lân cận (Neighborhood). Ta sẽ điền bằng median của mỗi khu.
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)

# Kiểm tra lại
print("\nSố giá trị thiếu sau khi xử lý:", all_data.isnull().sum().sum())


# 2.3. Tạo Đặc trưng mới (Creating New Features)
# Tạo các đặc trưng mới dựa trên sự kết hợp hoặc biến đổi các cột hiện có
print("\nBắt đầu tạo đặc trưng mới...")

# 1. Tổng diện tích sử dụng
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

# 2. Tuổi của ngôi nhà khi được bán
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']

# 3. Tổng số phòng tắm
all_data['TotalBath'] = all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath'] + \
                        all_data['FullBath'] + 0.5 * all_data['HalfBath']

# 4. Đặc trưng cho biết nhà có được sửa sang lại hay không
all_data['IsRemodeled'] = (all_data['YearRemodAdd'] != all_data['YearBuilt']).astype(int)

# 5. Các đặc trưng về hiên nhà
all_data['TotalPorchSF'] = all_data['OpenPorchSF'] + all_data['EnclosedPorch'] + \
                           all_data['3SsnPorch'] + all_data['ScreenPorch']

# 6. Các đặc trưng cho biết sự tồn tại của các tiện ích
all_data['HasPool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['Has2ndFlr'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasGarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasBsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasFireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

print("Số lượng đặc trưng sau khi tạo mới:", all_data.shape[1])


# 2.4. Mã hóa Biến Phân loại (Encoding Categorical Variables)
# Lấy danh sách các cột dạng 'object' (categorical)
categorical_cols = all_data.select_dtypes(include=['object']).columns

print(f"\nSố cột categorical cần mã hóa: {len(categorical_cols)}")

# Áp dụng One-Hot Encoding
all_data_encoded = pd.get_dummies(all_data, columns=categorical_cols, drop_first=True)

print("Kích thước dữ liệu sau khi mã hóa:", all_data_encoded.shape)


# Bước 3: Xây dựng và Đánh giá Mô hình (Modeling & Evaluation)
# Tách lại dữ liệu thành tập train và test
X = all_data_encoded.iloc[:len(y), :]
X_test = all_data_encoded.iloc[len(y):, :]

print("\nKích thước tập huấn luyện (X):", X.shape)
print("Kích thước tập kiểm tra (X_test):", X_test.shape)

# Khởi tạo mô hình XGBoost Regressor
# Đây là một mô hình Gradient Boosting mạnh mẽ, thường cho kết quả tốt trên Kaggle
model = XGBRegressor(n_estimators=1000, max_depth=5, learning_rate=0.05,
                     colsample_bytree=0.5, subsample=0.7,
                     random_state=42, n_jobs=-1)

# Huấn luyện mô hình trên toàn bộ tập dữ liệu huấn luyện với y_log
model.fit(X, y_log)

# Dự đoán trên tập test
predictions_log = model.predict(X_test)

# Chuyển đổi kết quả dự đoán về thang đo ban đầu
predictions = np.expm1(predictions_log)


# Bước 4: Tạo tệp Submission
# Tạo DataFrame cho file submission
submission_df = pd.DataFrame({'Id': test_ids, 'SalePrice': predictions})

# Lưu ra file csv
submission_df.to_csv('submission.csv', index=False)

print("\nĐã tạo file submission.csv thành công!")
print("Một vài dòng đầu của file submission:")
print(submission_df.head())

