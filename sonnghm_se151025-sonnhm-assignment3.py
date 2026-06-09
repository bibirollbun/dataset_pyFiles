# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data = pd.read_csv('/kaggle/input/au-1131-house-prices-prediction/train1121.csv')
test_data = pd.read_csv('/kaggle/input/au-1131-house-prices-prediction/test1121.csv')

# Xem 5 dòng đầu của dữ liệu huấn luyện
print(train_data.head())

# Kiểm tra kích thước dữ liệu
print("Kích thước dữ liệu huấn luyện:", train_data.shape)
print("Kích thước dữ liệu kiểm tra:", test_data.shape)


# Xem thông tin tổng quan
print(train_data.info())

# Xem thống kê mô tả
print(train_data.describe())

# Kiểm tra giá trị thiếu
missing_values = train_data.isnull().sum()
print("Các cột có giá trị thiếu:\n", missing_values[missing_values > 0])


# Tiền xử lý dữ liệu
# Tách đặc trưng và mục tiêu
X_train = train_data.drop(['Id', 'SalePrice'], axis=1)
y_train = train_data['SalePrice']
X_test = test_data.drop('Id', axis=1)


# Xử lý giá trị thiếu và inf bên trên
categorical_cols = X_train.select_dtypes(include=['object']).columns
numerical_cols = X_train.select_dtypes(exclude=['object']).columns


# Xử lý cột phân loại
for col in categorical_cols:
    X_train[col] = X_train[col].fillna('None')
    X_test[col] = X_test[col].fillna('None')


# Xử lý cột số (NaN và inf)
for col in numerical_cols:
    X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan)
    X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)
    X_train[col] = X_train[col].fillna(X_train[col].median())
    X_test[col] = X_test[col].fillna(X_train[col].median())


# Kiểm tra lại giá trị thiếu và inf
print("Sau xử lý, số NaN trong X_train:", X_train.isnull().sum().sum())
print("Sau xử lý, số inf trong X_train:", X_train[numerical_cols].apply(lambda x: np.isinf(x)).sum().sum())


# Mã hóa biến phân loại bằng cách sử dụng one-hot encoding
X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Cần đảm bảo tập train và test có cùng số cột, điền 0 cho các cột không xuất hiện trong tập test (nếu có) để tránh lỗi khi dự đoán.
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)


# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


sns.histplot(train_data['SalePrice'], kde=True)
plt.title('Phân phối của SalePrice')
plt.show()


# Chuyển đổi SalePrice sang logarit scale
y_train_log = np.log1p(y_train)


# Xây dựng và huấn luyện vài mô hình
# Chia dữ liệu để kiểm tra
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train_log, test_size=0.2, random_state=42)


# Dictionary lưu RMSE
models = {}
rmse_scores = {}


# Linear Regression
models['Linear Regression'] = LinearRegression()
models['Linear Regression'].fit(X_train_split, y_train_split)
y_pred_log_lr = models['Linear Regression'].predict(X_val_split)
rmse_scores['Linear Regression'] = np.sqrt(mean_squared_error(y_val_split, y_pred_log_lr))

# Ridge Regression
models['Ridge Regression'] = Ridge(alpha=1.0)
models['Ridge Regression'].fit(X_train_split, y_train_split)
y_pred_log_ridge = models['Ridge Regression'].predict(X_val_split)
rmse_scores['Ridge Regression'] = np.sqrt(mean_squared_error(y_val_split, y_pred_log_ridge))

# Random Forest Regression
models['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
models['Random Forest'].fit(X_train_split, y_train_split)
y_pred_log_rf = models['Random Forest'].predict(X_val_split)
rmse_scores['Random Forest'] = np.sqrt(mean_squared_error(y_val_split, y_pred_log_rf))

# Support Vector Regression (SVR)
models['SVR'] = SVR(kernel='rbf')
models['SVR'].fit(X_train_split, y_train_split)
y_pred_log_svr = models['SVR'].predict(X_val_split)
rmse_scores['SVR'] = np.sqrt(mean_squared_error(y_val_split, y_pred_log_svr))

# XGBoost
models['XGBoost'] = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
models['XGBoost'].fit(X_train_split, y_train_split)
y_pred_log_xgb = models['XGBoost'].predict(X_val_split)
rmse_scores['XGBoost'] = np.sqrt(mean_squared_error(y_val_split, y_pred_log_xgb))

# In kết quả RMSE của các mô hình
print("\nRMSE (log scale) của các mô hình:")
for model_name, rmse in rmse_scores.items():
    print(f"{model_name}: {rmse}")


# Kiểm tra dữ liệu sau tiền xử lý
print("Có NaN trong X_train_scaled không?", np.isnan(X_train_scaled).sum())
print("Có inf trong X_train_scaled không?", np.isinf(X_train_scaled).sum())
print("Có NaN trong y_train_log không?", np.isnan(y_train_log).sum())
print("Có inf trong y_train_log không?", np.isinf(y_train_log).sum())

# Kiểm tra dự đoán của Linear Regression
y_pred_log_lr = models['Linear Regression'].predict(X_val_split)
print("Có NaN trong y_pred_log_lr không?", np.isnan(y_pred_log_lr).sum())
print("Có inf trong y_pred_log_lr không?", np.isinf(y_pred_log_lr).sum())


# Chọn mô hình tốt nhất và dự đoán trên tập test
best_model_name = min(rmse_scores, key=rmse_scores.get)
best_model = models[best_model_name]
print(f"\nMô hình tốt nhất: {best_model_name} với RMSE: {rmse_scores[best_model_name]}")

# Huấn luyện lại mô hình tốt nhất trên toàn bộ dữ liệu
best_model.fit(X_train_scaled, y_train_log)
y_test_log_pred = best_model.predict(X_test_scaled)
y_test_pred = np.expm1(y_test_log_pred)  # Chuyển từ log scale về giá trị thực


# Tạo file submission
submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': y_test_pred
})
submission.to_csv('sample_submission1121.csv', index=False)
print("File submission đã được tạo: sample_submission1121.csv")

