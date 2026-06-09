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


# Cài đặt giải nén file csv.7z
!pip -q install py7zr


# Import thư viện cần thiết
import pandas as pd
import warnings
import py7zr
import os, gc
import re
from subprocess import check_output
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
warnings.filterwarnings('ignore', category=RuntimeWarning)


# Danh sách các file CSV muốn giải nén
files_to_extract = [
    'train.csv',
    'transactions.csv',
    'stores.csv',
    'oil.csv',
    'items.csv',
    'holidays_events.csv'
]

# Thư mục đích để giải nén
extract_path = "/kaggle/working"
os.makedirs(extract_path, exist_ok=True) # Đảm bảo thư mục đích tồn tại

print("Bắt đầu giải nén các file được chọn...")

for dirname, _, filenames in os.walk('/kaggle/input/favorita-grocery-sales-forecasting'):
    for filename in filenames:
        archive_path = os.path.join(dirname, filename)
        try:
            # Mở file 7z
            with py7zr.SevenZipFile(archive_path, mode='r') as archive:
                # Lặp qua các file muốn giải nén
                for file_to_extract in files_to_extract:
                    print(f"  Đang giải nén: {file_to_extract}")
                    # Sử dụng extract() để chỉ giải nén file mong muốn
                    archive.extract(path=extract_path, targets=[file_to_extract])

            print("Giải nén hoàn tất các file được chọn.")

        except py7zr.Bad7zFile:
            print(f"Lỗi: File {filename} không phải là file 7z hợp lệ hoặc bị hỏng.")
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file {archive_path}.")
        except Exception as e:
            print(f"Đã xảy ra lỗi: {e}")


# Kiểm tra lại các file đã được giải nén trong thư mục working
print("\n--- Danh sách file trong thư mục 'working' ---")
# Sử dụng 'ls -l' để hiển thị chi tiết hơn
print(check_output(["ls", "-l", "../working"]).decode("utf8"))


# Hàm hỗ trợ để đọc, chuyển đổi ngày và lọc dữ liệu
def prune_data(filename):
    """Đọc file CSV, lọc các bản ghi có năm 2016-2017, và trả về DataFrame."""
    
    # Đọc file CSV
    df = pd.read_csv(filename, low_memory=False)
    
    # Chuyển đổi cột 'date' thành kiểu datetime để dễ dàng lọc
    df['date'] = pd.to_datetime(df['date'])
    
    # 3. Lọc dữ liệu cho năm 2016-2017
    # Lấy năm từ cột 'date' và so sánh
    df_pruned = df[df['date'].dt.year.isin([2016, 2017])].copy()
    
    # Trả về DataFrame đã được cắt tỉa
    return df_pruned


# 1. train.csv
train_pruned = prune_data("train.csv")
print(f"train_pruned: {len(train_pruned)} bản ghi đã được tải (2016-2017).")

# 2. transactions.csv
transactions_pruned = prune_data("transactions.csv")
print(f"transactions_pruned: {len(transactions_pruned)} bản ghi đã được tải (2016-2017).")

# 3. oil.csv
oil_pruned = prune_data("oil.csv")
print(f"oil_pruned: {len(oil_pruned)} bản ghi đã được tải (2016-2017).")

# 4. holidays_events.csv
holidays_pruned = prune_data("holidays_events.csv")
print(f"holidays_pruned: {len(holidays_pruned)} bản ghi đã được tải (2016-2017).")


# --- Kết quả ---
print("\nĐã lọc và lưu toàn bộ dữ liệu năm 2016-2017 thành công!")


# Kiểm tra
print("Kiểm tra 5 dòng đầu tâp train")
print(train_pruned.head())

print("\nKiểm tra 5 dòng đầu của transactions")
print(transactions_pruned.head())

print("\nKiểm tra 5 dòng đầu của oil")
print(oil_pruned.head())

print("\nKiểm tra 5 dòng đầu của holidays_events")
print(holidays_pruned.head())


def check_null(df: pd.DataFrame) -> dict:
    """Trả về số lượng giá trị null theo từng cột (Pandas) dưới dạng dict."""
    return df.isnull().sum().to_dict()

def check_duplicate(df: pd.DataFrame) -> int:
    """
    Trả về tổng số dòng trùng lặp trong DataFrame (Pandas).
    """
    dup_count = df.duplicated().sum()
    print(f"Số dòng trùng lặp: {dup_count}")
    return dup_count
    
def check_negative(df: pd.DataFrame) -> dict:
    """
    Trả về tổng số giá trị âm trong từng cột số (dưới dạng dict).
    """
    # Lọc danh sách cột kiểu số (integer/float)
    numeric_df = df.select_dtypes(include=np.number)
    return (numeric_df < 0).sum().to_dict()

def check_strange_char(df: pd.DataFrame, pattern=r"[^a-zA-Z0-9\s.,:/\-_]") -> dict:
    """
    Kiểm tra số lượng giá trị chứa ký tự lạ (ngoài chữ,số,khoảng trắng và vài dấu cơ bản) trong các cột chuỗi.
    (Phiên bản đã sửa lỗi)
    """
    # Lọc danh sách cột kiểu chuỗi (object)
    str_cols = df.select_dtypes(include='object').columns.tolist()
    results = {}

    for c in str_cols:
        try:
            count = int(df[c].astype(str).str.contains(pattern).sum())
            results[c] = count
        except Exception as e:
            print(f"Không thể xử lý cột '{c}': {e}")
            results[c] = 0

    return results

# Tạo feature về thời gian (season)- mùa trong năm
def season_from_month(m):
    if m in [12, 1, 2]: return 0  # Winter
    elif m in [3, 4, 5]: return 1  # Spring
    elif m in [6, 7, 8]: return 2  # Summer
    else: return 3  # Fall


train_pruned.shape


# Kiểm tra null
print("Tổng số giá trị null: ", check_null(train_pruned))

# Kiểm tra trùng lặp
print("Tổng số giá trị trùng lặp: ", check_duplicate(train_pruned))

# Kiểm tra giá trị âm
print("Tổng số giá trị âm: ", check_negative(train_pruned))

# Kiểm tra giá trị có ký tự đặc biệt
print("Tổng số giá trị có ký tự đăc biệt: ", check_strange_char(train_pruned))


# Xử lý giá trị < 0 trong unit_sales
negative_before = (train_pruned['unit_sales'] < 0).sum()
print(f"Số giá trị 'unit_sales' âm (trước khi sửa): {negative_before}")

# Một số dòng có unit_sales < 0 → thay bằng 0 đại diện cho không bán được
train_pruned['unit_sales'] = train_pruned['unit_sales'].clip(lower=0)

# --- Kiểm tra lại kết quả ---
negative_after = (train_pruned['unit_sales'] < 0).sum()
print(f"Số giá trị 'unit_sales' < 0 (sau khi sửa): {negative_after}")

if negative_after == 0:
    print("Đã cập nhật thành công: Tất cả giá trị < 0 trong 'unit_sales' đã được đổi thành 0")
else:
    print("Lỗi: Vẫn còn giá trị <= 0")


print("Tổng số giá trị âm: ", check_negative(train_pruned))


# Trực quan dữ liệu để kiểm tra outlier
sns.set_theme(style="whitegrid")

# Danh sách các cột kiểm tra outlier
cols_to_plot = ['id', 'store_nbr', 'item_nbr', 'unit_sales']

n_features = len(cols_to_plot)

# Vẽ Boxplot
fig, axes = plt.subplots(nrows=1, ncols=n_features, figsize=(7 * n_features, 6))

# Đặt tiêu đề chung cho tất cả biểu đồ
fig.suptitle('Phân tích Ngoại lai (Outliers) bằng Boxplot', fontsize=16, y=1.02)

if n_features == 1:
    axes = [axes]

for i, col in enumerate(cols_to_plot):
    print(f"Đang tạo boxplot cho cột: {col}...")
    
    sns.boxplot(data=train_pruned, y=col, ax=axes[i])
    
    axes[i].set_title(f"Boxplot của {col}", fontsize=12)
    axes[i].set_ylabel('')

plt.tight_layout()
plt.show()


# Danh sách các cột cần kiểm tra outlier
cols_to_check = ['unit_sales']
total_rows = len(train_pruned)
print(f"Tổng số dòng trong DataFrame: {total_rows}\n")

print("--- Thống kê Outlier bằng phương pháp IQR ---")

for col in cols_to_check:
    # 1. Tính Q1 (Quartile 1) và Q3 (Quartile 3)
    Q1 = train_pruned[col].quantile(0.25)
    Q3 = train_pruned[col].quantile(0.75)
    
    # 2. Tính IQR (Interquartile Range)
    IQR = Q3 - Q1
    
    # 3. Xác định Ngưỡng Outlier
    # Ngưỡng dưới: Q1 - 1.5 * IQR
    # Ngưỡng trên: Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 4. Đếm số lượng Outlier
    outliers = train_pruned[
        (train_pruned[col] < lower_bound) | (train_pruned[col] > upper_bound)
    ]
    
    num_outliers = len(outliers)
    
    # 5. Tính Tỷ lệ
    outlier_ratio = (num_outliers / total_rows) * 100
    
    print(f"Cột: {col}")
    print(f"Số lượng Outlier: {num_outliers}")
    print(f"Tỷ lệ Outlier: {outlier_ratio:.4f}%")
    
    if col == 'unit_sales':
        print(f"  --> Q1: {Q1:.2f}, Q3: {Q3:.2f}, Ngưỡng Trên: {upper_bound:.2f}")



# Chuyển cột onpromotion từ bool sang binary (0 và 1)
train_pruned['onpromotion'] = train_pruned['onpromotion'].astype('int8')
train_pruned.head()


# Thêm các cột thời gian

train_pruned['year'] = train_pruned['date'].dt.year
train_pruned['month'] = train_pruned['date'].dt.month
train_pruned['day'] = train_pruned['date'].dt.day
train_pruned['dayofweek'] = train_pruned['date'].dt.dayofweek

train_pruned["weekofyear"] = train_pruned["date"].dt.isocalendar().week.astype(int)
train_pruned["is_weekend"] = train_pruned["dayofweek"].isin([5, 6]).astype(int)
train_pruned["quarter"] = train_pruned["date"].dt.quarter

# Đánh dấu đầu tháng/cuối tháng
train_pruned["is_month_start"] = train_pruned["date"].dt.is_month_start.astype(int)
train_pruned["is_month_end"] = train_pruned["date"].dt.is_month_end.astype(int)

train_pruned["season"] = train_pruned["month"].apply(season_from_month)


# --- Kiểm tra kết quả ---
print("\nĐã thêm các cột thành công!")
print("Kiểm tra 5 dòng đầu của train_pruned (với cột mới):")
print(train_pruned.head())


# Bỏ cột id
train_pruned = train_pruned.drop('id', axis=1)
print("Các cột hiện tại của train_pruned:")
print(train_pruned.columns)


train_pruned.shape


# Ghi file csv
train_pruned.to_csv("train_final.csv", index=False)
print("Đã xuất file thành công!")


transactions_pruned.shape


transactions_pruned.head()


# Kiểm tra null
print("Tổng số giá trị null: ", check_null(transactions_pruned))

# Kiểm tra trùng lặp
print("Tổng số giá trị trùng lặp: ", check_duplicate(transactions_pruned))

# Kiểm tra giá trị âm
print("Tổng số giá trị âm: ", check_negative(transactions_pruned))

# Kiểm tra giá trị có ký tự đặc biệt
print("Tổng số giá trị có ký tự đặc biệt: ", check_strange_char(transactions_pruned))


# Trực quan dữ liệu để kiểm tra outlier
sns.set_theme(style="whitegrid")

# Danh sách các cột kiểm tra outlier
cols_to_plot = ['store_nbr', 'transactions']

n_features = len(cols_to_plot)

# Vẽ Boxplot
fig, axes = plt.subplots(nrows=1, ncols=n_features, figsize=(7 * n_features, 6))

# Đặt tiêu đề chung cho tất cả biểu đồ
fig.suptitle('Phân tích Ngoại lai (Outliers) bằng Boxplot', fontsize=16, y=1.02)

if n_features == 1:
    axes = [axes]

for i, col in enumerate(cols_to_plot):
    print(f"Đang tạo boxplot cho cột: {col}...")
    
    sns.boxplot(data=transactions_pruned, y=col, ax=axes[i])
    
    axes[i].set_title(f"Boxplot của {col}", fontsize=12)
    axes[i].set_ylabel('')

plt.tight_layout()
plt.show()


# Xác định ngưỡng Outlier cho 'transactions' (dựa trên phương pháp IQR)
Q1 = transactions_pruned['transactions'].quantile(0.25)
Q3 = transactions_pruned['transactions'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

print(f"Ngưỡng Q1: {Q1}")
print(f"Ngưỡng Q3: {Q3}")
print(f"Ngưỡng Outlier (Giới hạn trên = Q3 + 1.5*IQR): {upper_bound:.2f}")

# Lọc tất cả các ngày có transactions được coi là ngoại lai
outlier_trans_df = transactions_pruned[
    transactions_pruned['transactions'] > upper_bound
].copy()

print(f"\nTìm thấy {len(outlier_trans_df)} ngày có transactions là ngoại lai (cao hơn {upper_bound:.2f}).")


outlier_trans_df


# Thêm các cột thời gian

transactions_pruned['year'] = transactions_pruned['date'].dt.year
transactions_pruned['month'] = transactions_pruned['date'].dt.month
transactions_pruned['day'] = transactions_pruned['date'].dt.day
transactions_pruned['dayofweek'] = transactions_pruned['date'].dt.dayofweek

transactions_pruned["weekofyear"] = transactions_pruned["date"].dt.isocalendar().week.astype(int)
transactions_pruned["is_weekend"] = transactions_pruned["dayofweek"].isin([5, 6]).astype(int)
transactions_pruned["quarter"] = transactions_pruned["date"].dt.quarter

# Đánh dấu đầu tháng/cuối tháng
transactions_pruned["is_month_start"] = transactions_pruned["date"].dt.is_month_start.astype(int)
transactions_pruned["is_month_end"] = transactions_pruned["date"].dt.is_month_end.astype(int)

transactions_pruned["season"] = transactions_pruned["month"].apply(season_from_month)

# --- Kiểm tra kết quả ---
print("\nĐã thêm cột thành công!")
print("Kiểm tra 5 dòng đầu của transactions_pruned (với cột mới):")
print(transactions_pruned.head())


# Ghi file csv
transactions_pruned.to_csv("transactions_final.csv", index=False)
print("Đã xuất file thành công!")


oil_pruned.shape


oil_pruned.head()


# Kiểm tra null
print("Tổng số giá trị null: ", check_null(oil_pruned))

# Kiểm tra trùng lặp
print("Tổng số giá trị trùng lặp: ", check_duplicate(oil_pruned))

# Kiểm tra giá trị âm
print("Tổng số giá trị âm: ", check_negative(oil_pruned))

# Kiểm tra giá trị có ký tự đặc biệt
print("Tổng số giá trị có ký tự đặc biệt: ", check_strange_char(oil_pruned))


# Kiểm tra thông tin dữ liệu có dcoilwtico bị null
oil_pruned[oil_pruned['dcoilwtico'].isnull()]


# Xử lý null
print(f"Số lượng giá trị null trong 'dcoilwtico' ban đầu: {oil_pruned['dcoilwtico'].isnull().sum()}")

# Sử dụng 'ffill' (forward fill) để fill giá trị null bằng giá trị của ngày trước đó
oil_pruned['dcoilwtico'] = oil_pruned['dcoilwtico'].fillna(method='ffill')

# Xử lý trường hợp đặc biệt nếu phía trước dữ liệu null không có dữ liệu nào trước đó -> Dùng 'bfill' (backward fill)
oil_pruned['dcoilwtico'] = oil_pruned['dcoilwtico'].fillna(method='bfill')

# Kiểm tra lại
final_null_count = oil_pruned['dcoilwtico'].isnull().sum()
print(f"Số lượng giá trị null trong 'dcoilwtico' sau khi fill: {final_null_count}")

if final_null_count == 0:
    print("Đã fill thành công!")
else:
    print("Vẫn còn giá trị null, vui lòng kiểm tra lại.")


# Thêm các cột thời gian

oil_pruned['year'] = oil_pruned['date'].dt.year
oil_pruned['month'] = oil_pruned['date'].dt.month
oil_pruned['day'] = oil_pruned['date'].dt.day
oil_pruned['dayofweek'] = oil_pruned['date'].dt.dayofweek

oil_pruned["weekofyear"] = oil_pruned["date"].dt.isocalendar().week.astype(int)
oil_pruned["is_weekend"] = oil_pruned["dayofweek"].isin([5, 6]).astype(int)
oil_pruned["quarter"] = oil_pruned["date"].dt.quarter

# Đánh dấu đầu tháng/cuối tháng
oil_pruned["is_month_start"] = oil_pruned["date"].dt.is_month_start.astype(int)
oil_pruned["is_month_end"] = oil_pruned["date"].dt.is_month_end.astype(int)

oil_pruned["season"] = oil_pruned["month"].apply(season_from_month)

# --- Kiểm tra kết quả ---
print("\nĐã thêm cột thành công!")
print("Kiểm tra 5 dòng đầu của oil_pruned (với cột mới):")
print(oil_pruned.head())


# Trực quan dữ liệu để kiểm tra outlier
sns.set_theme(style="whitegrid")

# Danh sách các cột kiểm tra outlier
cols_to_plot = ['dcoilwtico']

n_features = len(cols_to_plot)

# Vẽ Boxplot
fig, axes = plt.subplots(nrows=1, ncols=n_features, figsize=(7 * n_features, 6))

# Đặt tiêu đề chung cho tất cả biểu đồ
fig.suptitle('Phân tích Ngoại lai (Outliers) bằng Boxplot', fontsize=16, y=1.02)

if n_features == 1:
    axes = [axes]

for i, col in enumerate(cols_to_plot):
    print(f"Đang tạo boxplot cho cột: {col}...")
    
    sns.boxplot(data=oil_pruned, y=col, ax=axes[i])
    
    axes[i].set_title(f"Boxplot của {col}", fontsize=12)
    axes[i].set_ylabel('')

plt.tight_layout()
plt.show()


# Xác định ngưỡng Outlier cho 'dcoilwtico' (dựa trên phương pháp IQR)
Q1 = oil_pruned['dcoilwtico'].quantile(0.25)
Q3 = oil_pruned['dcoilwtico'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR

print(f"Ngưỡng Q1: {Q1}")
print(f"Ngưỡng Q3: {Q3}")
print(f"Ngưỡng Outlier (Giới hạn dưới = Q1 - 1.5*IQR): {lower_bound:.2f}")

# Lọc tất cả các ngày có dcoilwtico được coi là ngoại lai
outlier_oil_df = oil_pruned[
    oil_pruned['dcoilwtico'] < lower_bound
].copy()

print(f"\nTìm thấy {len(outlier_oil_df)} ngày có dcoilwtico là ngoại lai (thấp hơn {lower_bound:.2f}).")


outlier_oil_df


# Ghi file csv
oil_pruned.to_csv("oil_final.csv", index=False)
print("Đã xuất file thành công!")


holidays_pruned.shape


holidays_pruned.head()


# Kiểm tra null
print("Tổng số giá trị null: ", check_null(holidays_pruned))

# Kiểm tra trùng lặp
print("Tổng số giá trị trùng lặp: ", check_duplicate(holidays_pruned))

# Kiểm tra giá trị âm
print("Tổng số giá trị âm: ", check_negative(holidays_pruned))

# Kiểm tra giá trị có ký tự đặc biệt
print("Tổng số giá trị có ký tự đặc biệt: ", check_strange_char(holidays_pruned))


# Định nghĩa pattern ký tự đặc biệt (có thể tinh chỉnh)
pattern = r"[^a-zA-Z0-9\s.,:/\-_]"

condition = holidays_pruned["description"].str.contains(pattern, na=False)

rows_with_special = holidays_pruned[condition]

# In kết quả
print(f"Số dòng chứa ký tự đặc biệt: {len(rows_with_special)}")

print("\n--- Hiển thị các dòng chứa ký tự đặc biệt ---")

print(rows_with_special)


# Chuyển cột transferred từ bool sang binary (0 và 1)
holidays_pruned['transferred'] = holidays_pruned['transferred'].astype('int8')
holidays_pruned.head()


# Thêm các cột thời gian

holidays_pruned['year'] = holidays_pruned['date'].dt.year
holidays_pruned['month'] = holidays_pruned['date'].dt.month
holidays_pruned['day'] = holidays_pruned['date'].dt.day
holidays_pruned['dayofweek'] = holidays_pruned['date'].dt.dayofweek

holidays_pruned["weekofyear"] = holidays_pruned["date"].dt.isocalendar().week.astype(int)
holidays_pruned["is_weekend"] = holidays_pruned["dayofweek"].isin([5, 6]).astype(int)
holidays_pruned["quarter"] = holidays_pruned["date"].dt.quarter

# Đánh dấu đầu tháng/cuối tháng
holidays_pruned["is_month_start"] = holidays_pruned["date"].dt.is_month_start.astype(int)
holidays_pruned["is_month_end"] = holidays_pruned["date"].dt.is_month_end.astype(int)

holidays_pruned["season"] = holidays_pruned["month"].apply(season_from_month)

# --- Kiểm tra kết quả ---
print("\nĐã thêm cột thành công!")
print("Kiểm tra 5 dòng đầu của holidays_pruned (với cột mới):")
print(holidays_pruned.head())


# Ghi file csv
holidays_pruned.to_csv("holidays_final.csv", index=False)
print("Đã xuất file thành công!")


items = pd.read_csv("/kaggle/working/items.csv")


items.shape


items.head()


# Kiểm tra giá trị null
print("Tổng số giá trị null:\n", items.isnull().sum(), "\n")

# Kiểm tra trùng lặp
print("Số dòng trùng lặp hoàn toàn:", items.duplicated().sum())
print("Số item_nbr trùng:", items.duplicated(subset=['item_nbr']).sum(), "\n")

# Kiểm tra giá trị âm (chỉ các cột numeric)
num_cols = items.select_dtypes(include='number').columns
print("Số giá trị âm theo cột:\n", (items[num_cols] < 0).sum(), "\n")

# Kiểm tra ký tự đặc biệt (chỉ cột string)
pattern = r"[^a-zA-Z0-9\s.,:/\-_+]"
str_cols = items.select_dtypes(include='object').columns
print("Số giá trị có ký tự đặc biệt theo cột:\n",
      items[str_cols].apply(lambda s: s.astype(str).str.contains(pattern, regex=True, na=False).sum()))


stores = pd.read_csv("/kaggle/working/stores.csv")


stores.shape


stores.head()


# Kiểm tra giá trị null
print("Tổng số giá trị null:\n", stores.isnull().sum(), "\n")

# Kiểm tra trùng lặp
print("Số dòng trùng lặp hoàn toàn:", stores.duplicated().sum())
print("Số store_nbr trùng:", stores.duplicated(subset=['store_nbr']).sum(), "\n")

# Kiểm tra giá trị âm (chỉ các cột numeric)
num_cols = stores.select_dtypes(include='number').columns
print("Số giá trị âm theo cột:\n", (stores[num_cols] < 0).sum(), "\n")

# Kiểm tra ký tự đặc biệt (chỉ cột string)
pattern = r"[^a-zA-Z0-9\s.,:/\-_+]"
str_cols = stores.select_dtypes(include='object').columns
print("Số giá trị có ký tự đặc biệt theo cột:\n",
      stores[str_cols].apply(lambda s: s.astype(str).str.contains(pattern, regex=True, na=False).sum()))


# Train
print("Thống kê kiểu dữ liệu: ")
print(train_pruned.dtypes)

print("\n=====================\n")
print("Thống kê mô tả unit_sales:")
print("Min: ", train_pruned['unit_sales'].min())
print("Max: ", train_pruned['unit_sales'].max())

print("\n=====================\n")
print("Thống kê mô tả onpromotion:")
# Tính số lượng và tỷ lệ phần trăm cho mỗi giá trị unique
promotion_counts = train_pruned['onpromotion'].value_counts()
promotion_ratios = train_pruned['onpromotion'].value_counts(normalize=True) * 100

# In kết quả cho 0 (Không Khuyến mãi)
count_0 = promotion_counts.get(0, 0)
ratio_0 = promotion_ratios.get(0, 0)
print(f"0 (Không Khuyến mãi)         | {count_0:<17} | {ratio_0:.4f}%")

# In kết quả cho 1 (Có Khuyến mãi)
count_1 = promotion_counts.get(1, 0)
ratio_1 = promotion_ratios.get(1, 0)
print(f"1 (Có Khuyến mãi)            | {count_1:<17} | {ratio_1:.4f}%")


# Transactions
print("Thống kê kiểu dữ liệu: ")
print(transactions_pruned.dtypes)

print("\n=====================\n")
print("Thống kê mô tả transactions:")
transactions_pruned['transactions'].describe().round()


# Oil
print("Thống kê kiểu dữ liệu: ")
print(oil_pruned.dtypes)

print("\n=====================\n")
print("Thống kê mô tả oil:")
oil_pruned['dcoilwtico'].describe().round(2)


# Holidays
print("Thống kê kiểu dữ liệu: ")
print(holidays_pruned.dtypes)

print("\n=====================\n")
print("Thống kê mô tả transferred:")
# Tính số lượng và tỷ lệ phần trăm cho mỗi giá trị unique
transferred_counts = holidays_pruned['transferred'].value_counts()
transferred_ratios = holidays_pruned['transferred'].value_counts(normalize=True) * 100

# In kết quả cho 0 (Ngày tổ chức lễ gốc)
count_0 = transferred_counts.get(0, 0)
ratio_0 = transferred_ratios.get(0, 0)
print(f"0 (Ngày tổ chức lễ gốc)                   | {count_0:<17} | {ratio_0:.4f}%")

# In kết quả cho 1 (Ngày tổ chức lễ bị dời sang ngày khác)
count_1 = transferred_counts.get(1, 0)
ratio_1 = transferred_ratios.get(1, 0)
print(f"1 (Ngày tổ chức lễ bị dời sang ngày khác) | {count_1:<17} | {ratio_1:.4f}%")


# Items
print("Thống kê kiểu dữ liệu: ")
print(items.dtypes)

print("\n=====================\n")
print("Thống kê mô tả perishable:")
# Tính số lượng và tỷ lệ phần trăm cho mỗi giá trị unique
perishable_counts = items['perishable'].value_counts()
perishable_ratios = items['perishable'].value_counts(normalize=True) * 100

# In kết quả cho 0 (Sản phẩm bền)
count_0 = perishable_counts.get(0, 0)
ratio_0 = perishable_ratios.get(0, 0)
print(f"0 (Sản phẩm bền)         | {count_0:<17} | {ratio_0:.4f}%")

# In kết quả cho 1 (Sản phẩm dễ hư hỏng)
count_1 = perishable_counts.get(1, 0)
ratio_1 = perishable_ratios.get(1, 0)
print(f"1 (Sản phẩm dễ hư hỏng)  | {count_1:<17} | {ratio_1:.4f}%")


# Stores
print("Thống kê kiểu dữ liệu: ")
print(stores.dtypes)

print("\n=====================\n")
print("Thống kê giá trị unique của city/state/type/cluster:")
# Danh sách các cột cần thống kê
cols_to_check = ['city', 'state', 'type', 'cluster']

for col in cols_to_check:
    print(f"\nCột: {col} ({len(stores[col].unique())} giá trị duy nhất)")
    # In ra tất cả các giá trị unique
    print(stores[col].unique())

