import numpy as np
import pandas as pd

import os

# Duyệt toàn bộ thư mục con bên trong thư mục gốc (file zip)
for dirname, _, filenames in os.walk('/kaggle/input/favorita-grocery-sales-forecasting'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


!pip install py7zr


import py7zr
from subprocess import check_output

for dirname, _, filenames in os.walk('/kaggle/input/favorita-grocery-sales-forecasting'):
    for filename in filenames:
        # Mở lần lượt từng file zip
        archive = py7zr.SevenZipFile(os.path.join(dirname, filename), mode='r')
        # Giải nén lần lượt từng file zip rồi xuất ra output
        archive.extractall(path="/kaggle/working")
        archive.close()

# Giải nén thành công thì in ra màn hình để thông báo
print(check_output(["ls", "../working"]).decode("utf8"))


# Đọc từng file csv
train = pd.read_csv('../working/train.csv', parse_dates=['date'], low_memory = False)
test = pd.read_csv('../working/test.csv', parse_dates=['date'])
oil = pd.read_csv('../working/oil.csv', parse_dates=['date'])
stores = pd.read_csv('../working/stores.csv')
items = pd.read_csv('../working/items.csv')
transactions = pd.read_csv('../working/transactions.csv', parse_dates=['date'])
holidays = pd.read_csv('../working/holidays_events.csv', parse_dates=['date'])


oil['dcoilwtico'] = oil['dcoilwtico'].interpolate()


oil['dcoilwtico'] = oil['dcoilwtico'].bfill()


oil.isnull().sum()


train.isnull().sum()


train['onpromotion'] = train['onpromotion'].fillna('False')


train.isnull().sum()


print(train)


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from tqdm import tqdm


# Kiểm tra số lượng sản phẩm
num_items = train['item_nbr'].nunique()
print("Số lượng sản phẩm khác nhau:", num_items)


# Tính tổng doanh thu theo item_nbr
item_sales = train.groupby('item_nbr')['unit_sales'].sum().sort_values(ascending=False)

# Lấy top 100 sản phẩm có doanh thu cao nhất
top_100_items = item_sales.head(100).index.tolist()

# Tạo tập dữ liệu train_top chỉ chứa các sản phẩm này
train_top = train[train['item_nbr'].isin(top_100_items)].copy()


# Chuyển kiểu dữ liệu ngày
train_top['date'] = pd.to_datetime(train_top['date'])

# Loại bỏ giá trị âm hoặc bất thường trong unit_sales
train_top = train_top[train_top['unit_sales'] >= 0]

# Chuyển onpromotion về kiểu bool
train_top['onpromotion'] = train_top['onpromotion'].map({'False': False, 'True': True}).fillna(False).astype('bool')


print(train_top)


# Tạo danh sách ngày đầy đủ từ min đến max
full_dates = pd.date_range(start=train_top['date'].min(), end=train_top['date'].max())

actual_dates = train_top['date'].drop_duplicates().sort_values()

missing_dates = full_dates.difference(actual_dates)

# In ra các ngày bị thiếu (nếu có)
print("Số ngày bị thiếu:", len(missing_dates))
print("Các ngày bị thiếu:", missing_dates)


# Danh sách ngày thiếu
missing_dates = pd.to_datetime(['2013-12-25', '2014-12-25', '2015-12-25', '2016-12-25'])

# Lấy danh sách store và item đã có
stores = train_top['store_nbr'].unique()
items = train_top['item_nbr'].unique()

# Tạo tổ hợp đầy đủ cho các ngày thiếu
from itertools import product
missing_index = pd.DataFrame(product(missing_dates, stores, items), columns=['date', 'store_nbr', 'item_nbr'])

# Gán giá trị mặc định
missing_index['unit_sales'] = 0.0
missing_index['onpromotion'] = False

# Gộp dữ liệu bị thiếu vào train_top
train_top = pd.concat([train_top, missing_index], ignore_index=True)

# Sắp xếp lại theo thời gian để đảm bảo đúng thứ tự chuỗi
train_top = train_top.sort_values(['date', 'store_nbr', 'item_nbr']).reset_index(drop=True)


# Tạo đặc trưng thời gian cho toàn bộ dữ liệu
train_top['day_of_week'] = train_top['date'].dt.dayofweek.astype('int8')
train_top['week_of_year'] = train_top['date'].dt.isocalendar().week.astype('int8')
train_top['month'] = train_top['date'].dt.month.astype('int8')
train_top['year'] = train_top['date'].dt.year.astype('int16')
train_top['is_weekend'] = train_top['day_of_week'].isin([5, 6])


print(train_top)


train_top.isnull().sum()


train_top.isna().sum()


# Tạo danh sách ngày đầy đủ từ min đến max
full_dates = pd.date_range(start=train_top['date'].min(), end=train_top['date'].max())

actual_dates = train_top['date'].drop_duplicates().sort_values()

missing_dates = full_dates.difference(actual_dates)

# In ra các ngày bị thiếu (nếu có)
print("Số ngày bị thiếu:", len(missing_dates))
print("Các ngày bị thiếu:", missing_dates)


# Ép kiểu để giảm bộ nhớ trước khi merge
oil['dcoilwtico'] = oil['dcoilwtico'].astype('float32')

# Merge giá dầu vào train
train_top = train_top.merge(oil[['date', 'dcoilwtico']], on='date', how='left')

# Xử lý thiếu giá dầu sau khi merge (do train có ngày không khớp với oil)
train_top['dcoilwtico'] = train_top['dcoilwtico'].interpolate(method='linear')
train_top['dcoilwtico'] = train_top['dcoilwtico'].fillna(method='ffill').fillna(method='bfill')


# Merge ngày lễ
holidays['date'] = pd.to_datetime(holidays['date'])
holidays['transferred'] = holidays['transferred'].astype('bool')
holidays['type'] = holidays['type'].astype('category')
holidays['locale'] = holidays['locale'].astype('category')
holidays['locale_name'] = holidays['locale_name'].astype('category')

regular = holidays[holidays['transferred'] == False]
transfer_map = holidays[holidays['type'] == 'Transfer'][['description', 'date']]
transferred = holidays[holidays['transferred'] == True][['description', 'locale', 'locale_name']]
transferred = transferred.merge(transfer_map, on='description', how='left')
holiday_actual = pd.concat([
    regular[['date', 'locale', 'locale_name']],
    transferred[['date', 'locale', 'locale_name']]
], ignore_index=True)
holiday_actual['is_holiday'] = True

# Đọc lại stores.csv để đảm bảo là DataFrame
stores = pd.read_csv('../working/stores.csv')

# Merge city từ stores
stores['store_nbr'] = stores['store_nbr'].astype('int8')
stores['city'] = stores['city'].astype('category')
train_top = train_top.merge(stores[['store_nbr', 'city']], on='store_nbr', how='left')

# Merge ngày lễ theo city
train_top = train_top.merge(
    holiday_actual,
    left_on=['date', 'city'],
    right_on=['date', 'locale_name'],
    how='left'
)
train_top['is_holiday'] = train_top['is_holiday'].fillna(False).astype('bool')
train_top.drop(columns=['locale', 'locale_name'], inplace=True)


print(train_top)


train_top.isna().sum()


train_top.isnull().sum()


# Sắp xếp theo thời gian
train_top = train_top.sort_values('date')

# Chia theo thời gian (không shuffle)
split_date = train_top['date'].quantile(0.8)
train_set = train_top[train_top['date'] <= split_date]
test_set = train_top[train_top['date'] > split_date]


print(train_set)


print(test_set)


# Chuẩn hóa giá trị đầu vào
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
cols_to_scale = ['unit_sales', 'dcoilwtico']

# Fit trên tập train để học min/max
scaler.fit(train_set[cols_to_scale])

# Tạo biến mới chứa dữ liệu đã chuẩn hóa
train_scaled = train_set.copy()
test_scaled = test_set.copy()

train_scaled[cols_to_scale] = scaler.transform(train_set[cols_to_scale])
test_scaled[cols_to_scale] = scaler.transform(test_set[cols_to_scale])


print(train_scaled)


print(test_scaled)


from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import numpy as np
import pandas as pd

# Tổng hợp theo tuần
train_set['week'] = train_set['date'] - pd.to_timedelta(train_set['date'].dt.dayofweek, unit='d')
test_set['week'] = test_set['date'] - pd.to_timedelta(test_set['date'].dt.dayofweek, unit='d')

# Khởi tạo danh sách lưu kết quả
mae_list, rmse_list, mape_list = [], [], []
item_ids = []

# Lặp qua từng sản phẩm
for item in tqdm(train_set['item_nbr'].unique()):
    try:
        # Lọc dữ liệu theo item
        train_grp = train_set[train_set['item_nbr'] == item]
        test_grp = test_set[test_set['item_nbr'] == item]

        # Tổng hợp theo tuần
        train_weekly = train_grp.groupby('week').agg({
            'unit_sales': 'sum',
            'onpromotion': 'mean',
            'dcoilwtico': 'mean',
            'is_holiday': 'mean'
        }).reset_index()

        test_weekly = test_grp.groupby('week').agg({
            'unit_sales': 'sum',
            'onpromotion': 'mean',
            'dcoilwtico': 'mean',
            'is_holiday': 'mean'
        }).reset_index()

        # Kiểm tra đủ dữ liệu
        if len(train_weekly) < 10 or len(test_weekly) < 5:
            continue

        # Biến đầu ra và biến ngoại sinh
        train_y = train_weekly['unit_sales']
        train_exog = train_weekly[['onpromotion', 'dcoilwtico', 'is_holiday']]
        test_exog = test_weekly[['onpromotion', 'dcoilwtico', 'is_holiday']]

        # Huấn luyện mô hình
        model = SARIMAX(train_y, exog=train_exog, order=(1,1,1), seasonal_order=(1,1,1,52),
                        enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)

        # Dự báo
        forecast = results.predict(start=len(train_y), end=len(train_y)+len(test_exog)-1, exog=test_exog)

        # Đánh giá
        y_true = test_weekly['unit_sales'].values
        y_pred = forecast.values

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100

        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)
        item_ids.append(item)

    except Exception as e:
        print("Lỗi")
        continue  # Bỏ qua nếu lỗi

# Tổng hợp kết quả
results_df = pd.DataFrame({
    'item_nbr': item_ids,
    'MAE': mae_list,
    'RMSE': rmse_list,
    'MAPE': mape_list
})

# In chỉ số trung bình
print("MAE trung bình:", results_df['MAE'].mean())
print("RMSE trung bình:", results_df['RMSE'].mean())
print("MAPE trung bình:", results_df['MAPE'].mean())


print(results_df)

