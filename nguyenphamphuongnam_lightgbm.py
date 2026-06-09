!pip install -U lightautoml
!pip install flaml[automl] matplotlib openml
!pip install -U ipywidgets


import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from flaml import AutoML
import json

from flaml.automl.model import LGBMEstimator


calendar_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv')
inventory_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
train_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv')
test_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv')
df5 = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')
weights_df = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')




df5


train_df.info()


test_columns = list(test_df.columns)
keep_columns =  list(train_df.columns)
print(test_columns)
keep_columns 


# --- Tạo dictionary scalers cho train_df ---
scalers = {}
for unique_id in tqdm(train_df["unique_id"].unique()):
    scaler = StandardScaler()
    sales = train_df.loc[train_df["unique_id"] == unique_id, "sales"].values.reshape(-1, 1)
    scaler.fit(sales)
    scalers[unique_id] = scaler
    train_df.loc[train_df["unique_id"] == unique_id, "sales"] = scaler.transform(sales).flatten()


# --- Định nghĩa hàm inverse_norm ---
def inverse_norm(df_, indexes, y_pred, scalers):
    df_ = df_.copy()  # Tạo bản sao để tránh thay đổi df_ gốc
    df_.loc[indexes, "prediction_norm"] = y_pred
    df_.loc[indexes, "y_pred"] = df_.groupby("unique_id")["prediction_norm"].transform(
        lambda x: scalers[x.name].inverse_transform(x.values.reshape(-1, 1)).flatten()
        if x.name in scalers else x.values  # Nếu unique_id không có scaler, giữ nguyên
    )
    return df_.loc[indexes, "y_pred"].values


 #Định dạng lại format date
calendar_df['date'] = pd.to_datetime(calendar_df['date'])
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])



# Lọc dữ liệu theo từng kho và chỉ lấy từ ngày 01/08/2020 trở đi
Frankfurt_1 = calendar_df.query('date >= "2016-01-01 00:00:00" and warehouse =="Frankfurt_1"')
Prague_2 = calendar_df.query('date >= "2016-01-01 00:00:00" and warehouse =="Prague_2"')
Brno_1 = calendar_df.query('date >= "2016-01-01 00:00:00" and warehouse =="Brno_1"')
Munich_1 = calendar_df.query('date >= "2016-01-01 00:00:00" and warehouse =="Munich_1"')
Prague_3 = calendar_df.query('date >= "2016-01-01 00:00:00" and warehouse =="Prague_3"')
Prague_1 = calendar_df.query('date >= "2016-01-01 00:00:00" and warehouse =="Prague_1"')
Budapest_1 = calendar_df.query('date >= "2016-01-01 00:00:00" and warehouse =="Budapest_1"')
def process_calendar(df):
    """
    - days_to_holiday
    - days_to_shops_closed
    - day_after_closing
    - long_weekend
    - weekday
    - ...
    """
    df = df.sort_values('date').reset_index(drop=True)


    # 4. long_weekend 
    # Xác định các ngày thuộc kỳ nghỉ cuối tuần dài (khi shops_closed = 1 và ngày trước đó cũng đóng cửa
    df['long_weekend'] = (
        (df['shops_closed'] == 1) & (df['shops_closed'].shift(1) == 1)
    ).astype(int)

    # 5. weekday Ngày trong tuần (0 = Thứ Hai, 6 = Chủ Nhật).
    df['weekday'] = df['date'].dt.weekday 

    # 6. week of month 
    df['week_of_month'] = df['date'].apply(lambda x: (x.day - 1) // 7 + 1)

    # 9. quarter
    df['quarter'] = df['date'].dt.quarter

    # 10. is weekend
    df['is_weekend'] = df['date'].dt.weekday.isin([5, 6]).astype(int)

    return df


dfs = ['Frankfurt_1', 'Prague_2', 'Brno_1', 'Munich_1', 'Prague_3', 'Prague_1', 'Budapest_1']
# Áp dụng hàm xử lý cho từng DataFrame và gom lại thành danh sách
processed_dfs = [process_calendar(globals()[df]) for df in dfs]

# Gộp tất cả các DataFrame lại thành một bảng duy nhất
calendar_extended = pd.concat(processed_dfs).sort_values('date').reset_index(drop=True)
print(calendar_extended.isna().sum())


train_calendar = train_df.merge(calendar_extended, on=['date', 'warehouse'], how='left')
train_inventory = train_calendar.merge(inventory_df, on=['unique_id', 'warehouse'], how='left')
train_df = train_inventory.merge(weights_df, on=['unique_id'], how='left')

test_calendar = test_df.merge(calendar_extended, on=['date', 'warehouse'], how='left')
test_df = test_calendar.merge(inventory_df, on=['unique_id', 'warehouse'], how='left')


#Tìm giá trị null
train_df.isnull().sum()



train_df['sales'] = train_df['sales'].fillna(0)
train_df['total_orders'] = train_df['total_orders'].fillna(0)
train_df['sell_price_main'] = train_df['sell_price_main'].interpolate()


train_df.isnull().sum()


print("Số lượng NaN trong holiday_name theo holiday:")
print(train_df.groupby('holiday')['holiday_name'].apply(lambda x: x.isna().sum()))


try:
    if 'train_df' not in globals():
        raise NameError("train_df không được định nghĩa. Vui lòng đảm bảo train_df đã được tạo từ mã trước đó.")
    
    # Kiểm tra các cột cần thiết
    required_columns = ['date', 'warehouse', 'holiday', 'holiday_name']
    missing_columns = [col for col in required_columns if col not in train_df.columns]
    if missing_columns:
        raise ValueError(f"train_df thiếu các cột: {missing_columns}")

    # Lọc các ngày lễ thiếu tên (holiday == 1 và holiday_name là NaN)
    missing_holidays = train_df[(train_df['holiday'] == 1) & (train_df['holiday_name'].isna())][['date', 'warehouse']]

    # Kiểm tra nếu không có dữ liệu nào
    if missing_holidays.empty:
        print("Không tìm thấy ngày lễ nào thiếu tên (holiday == 1 và holiday_name là NaN).")
    else:
        # Nhóm theo warehouse và lấy danh sách các ngày duy nhất, sắp xếp theo ngày
        missing_by_warehouse = missing_holidays.groupby('warehouse').agg({
            'date': lambda x: sorted(x.dt.strftime('%Y-%m-%d').unique().tolist())
        }).reset_index()

        # Đổi tên cột cho rõ ràng
        missing_by_warehouse.columns = ['warehouse', 'missing_holiday_dates']

        # In kết quả
        print("Các ngày lễ thiếu tên theo từng kho (sắp xếp theo ngày):")
        for _, row in missing_by_warehouse.iterrows():
            warehouse = row['warehouse']
            dates = row['missing_holiday_dates']
            print(f"\nKho: {warehouse}")
            print(f"Số ngày lễ thiếu tên: {len(dates)}")
            print("Các ngày thiếu tên lễ:", ", ".join(dates) if dates else "Không có ngày lễ thiếu tên")



except NameError as e:
    print(f"Lỗi: {e}")
except ValueError as e:
    print(f"Lỗi: {e}")
except Exception as e:
    print(f"Lỗi không xác định: {e}")


# Danh sách ngày lễ cập nhật từ artifact trước
brno_holiday = [
    (['04/04/2021', '04/17/2022', '04/09/2023', '03/31/2024'], 'Easter Day'),
    (['04/03/2021', '04/16/2022', '04/08/2023', '03/30/2024'], 'Holy Saturday'),
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother's Day"),
]
prague_1_holidays = [
    (['04/04/2021', '04/17/2022', '04/09/2023', '03/31/2024'], 'Easter Day'),
    (['04/03/2021', '04/16/2022', '04/08/2023', '03/30/2024'], 'Holy Saturday'),
]
prague_2_holidays = [
    (['04/04/2021', '04/17/2022', '04/09/2023', '03/31/2024'], 'Easter Day'),
    (['04/03/2021', '04/16/2022', '04/08/2023', '03/30/2024'], 'Holy Saturday'),
]
prague_3_holidays = [
    (['04/04/2021', '04/17/2022', '04/09/2023', '03/31/2024'], 'Easter Day'),
    (['04/03/2021', '04/16/2022', '04/08/2023', '03/30/2024'], 'Holy Saturday'),
]
budapest_holidays = [
    (['04/04/2021', '04/17/2022', '04/09/2023', '03/31/2024'], 'Easter Day'),
    (['04/03/2021', '04/08/2023', '03/30/2024'], 'Holy Saturday'),
]
frank_holidays = [
    (['04/17/2022', '04/09/2023', '03/31/2024'], 'Easter Day'),
    (['04/16/2022', '04/08/2023', '03/30/2024'], 'Holy Saturday'),
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], "Mother's Day"),
]
munich_holidays = [
    (['04/17/2022', '04/09/2023', '03/31/2024'], 'Easter Day'),
    (['04/16/2022', '04/08/2023', '03/30/2024'], 'Holy Saturday'),
]

# Hàm fill_loss_holidays (giữ nguyên)
def fill_loss_holidays(df_fill, warehouses, holidays):
    df = df_fill.copy()
    for item in holidays:
        dates, holiday_name = item
        generated_dates = [datetime.strptime(date, '%m/%d/%Y').strftime('%Y-%m-%d') for date in dates]
        for generated_date in generated_dates:
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday'] = 1
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday_name'] = holiday_name
    return df

# Kiểm tra xem train_df và test_df có tồn tại không
try:
    if 'train_df' not in globals() or 'test_df' not in globals():
        raise NameError("train_df hoặc test_df không được định nghĩa. Vui lòng đảm bảo cả hai đã được tạo từ mã trước đó.")

    # Kiểm tra các cột cần thiết
    required_columns = ['date', 'warehouse', 'holiday', 'holiday_name']
    for df_name, df in [('train_df', train_df), ('test_df', test_df)]:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"{df_name} thiếu các cột: {missing_columns}")

    # Kiểm tra NaN trong holiday_name trước khi xử lý
    print("Số lượng NaN trong holiday_name (train_df) trước khi xử lý:", train_df['holiday_name'].isna().sum())
    print("\nSố lượng NaN trong holiday_name theo holiday (train_df):")
    print(train_df.groupby('holiday')['holiday_name'].apply(lambda x: x.isna().sum()))

    # Điền ngày lễ vào train_df
    train_df = fill_loss_holidays(df_fill=train_df, warehouses=['Prague_1', 'Prague_2', 'Prague_3'], holidays=prague_1_holidays)
    train_df = fill_loss_holidays(df_fill=train_df, warehouses=['Brno_1'], holidays=brno_holiday)
    train_df = fill_loss_holidays(df_fill=train_df, warehouses=['Munich_1'], holidays=munich_holidays)
    train_df = fill_loss_holidays(df_fill=train_df, warehouses=['Frankfurt_1'], holidays=frank_holidays)
    train_df = fill_loss_holidays(df_fill=train_df, warehouses=['Budapest_1'], holidays=budapest_holidays)

    # Điền các NaN còn lại trong holiday_name bằng "No Holiday"
    train_df['holiday_name'] = train_df['holiday_name'].fillna("No Holiday")

    # Kiểm tra NaN sau khi xử lý
    print("\nSố lượng NaN trong holiday_name (train_df) sau khi xử lý:", train_df['holiday_name'].isna().sum())
    print("\nPhân bố giá trị trong holiday_name (train_df):")
    print(train_df['holiday_name'].value_counts())
    print("\nGiá trị holiday_name khi holiday == 1 (train_df):")
    print(train_df[train_df['holiday'] == 1]['holiday_name'].value_counts())

    # Xử lý test_df (tương tự)
    print("\nSố lượng NaN trong holiday_name (test_df) trước khi xử lý:", test_df['holiday_name'].isna().sum())
    test_df = fill_loss_holidays(df_fill=test_df, warehouses=['Prague_1', 'Prague_2', 'Prague_3'], holidays=prague_1_holidays)
    test_df = fill_loss_holidays(df_fill=test_df, warehouses=['Brno_1'], holidays=brno_holiday)
    test_df = fill_loss_holidays(df_fill=test_df, warehouses=['Munich_1'], holidays=munich_holidays)
    test_df = fill_loss_holidays(df_fill=test_df, warehouses=['Frankfurt_1'], holidays=frank_holidays)
    test_df = fill_loss_holidays(df_fill=test_df, warehouses=['Budapest_1'], holidays=budapest_holidays)

    # Điền NaN còn lại trong test_df
    test_df['holiday_name'] = test_df['holiday_name'].fillna("No Holiday")

    # Kiểm tra NaN sau khi xử lý
    print("\nSố lượng NaN trong holiday_name (test_df) sau khi xử lý:", test_df['holiday_name'].isna().sum())
    print("\nPhân bố giá trị trong holiday_name (test_df):")
    print(test_df['holiday_name'].value_counts())


except NameError as e:
    print(f"Lỗi: {e}")
except ValueError as e:
    print(f"Lỗi: {e}")
except Exception as e:
    print(f"Lỗi không xác định: {e}")





Q1 = np.log1p(train_df["sales"]).quantile(0.25)
Q3 = np.log1p(train_df["sales"]).quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

train_df = train_df[(np.log1p(train_df["sales"]) >= lower) & (np.log1p(train_df["sales"]) <= upper)]


plt.figure(figsize=(6, 5))

sns.histplot(x=np.log1p(train_df['sales']), bins=100, kde=True)

plt.title(f'Log histplot of sales')
plt.xlabel('sales')
plt.ylabel('Frequency')

plt.show()


train_df.loc[train_df['type_0_discount'] < 0, 'type_0_discount'] = 0
train_df.loc[train_df['type_4_discount'] < 0, 'type_4_discount'] = 0
train_df.loc[train_df['type_6_discount'] < 0, 'type_6_discount'] = 0


train_df.head()


train_df.info()


test_df.info()


test_df['sales'] = 0.0


test_df.info()


# 1. Tính category_sales cho train_df và map vào train_df
category_sales_train = train_df.groupby('L1_category_name_en')['sales'].agg(['mean']).reset_index()
category_sales_train.rename(columns={'mean': 'category_sales_avg'}, inplace=True)
train_df['category_sales_avg'] = train_df['L1_category_name_en'].map(
    category_sales_train.set_index('L1_category_name_en')['category_sales_avg']
)

# 2. Tính category_sales cho test_df và merge vào test_df
category_sales_test = test_df.groupby('L1_category_name_en')['sales'].agg(['mean']).reset_index()
category_sales_test.rename(columns={'mean': 'category_sales_avg'}, inplace=True)
test_df = test_df.merge(category_sales_test, on='L1_category_name_en', how='left')

# 3. Tính cat_order_stats cho train_df và map vào train_df
cat_order_stats_train = train_df.groupby('L1_category_name_en')['total_orders'].agg(['mean']).reset_index()
cat_order_stats_train.rename(columns={'mean': 'category_orders_avg'}, inplace=True)
train_df['category_orders_avg'] = train_df['L1_category_name_en'].map(
    cat_order_stats_train.set_index('L1_category_name_en')['category_orders_avg']
)

# 4. Tính cat_order_stats cho test_df và merge vào test_df
cat_order_stats_test = test_df.groupby('L1_category_name_en')['total_orders'].agg(['mean']).reset_index()
cat_order_stats_test.rename(columns={'mean': 'category_orders_avg'}, inplace=True)
test_df = test_df.merge(cat_order_stats_test, on='L1_category_name_en', how='left')

# Convert date to datetime and extract components
for df in [train_df, test_df]:
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['day_of_year'] = df['date'].dt.dayofyear

# Xử lý NaN trong test_df (nếu cần)
test_df['category_sales_avg'].fillna(0.0, inplace=True)  # Vì sales trong test_df là 0
test_df['category_orders_avg'].fillna(test_df['total_orders'].mean(), inplace=True)


# Calculate the maximum discount applied in Sales_Train
# tạo đặc trưng max_discount trong các loại discount
discount_columns = [f'type_{i}_discount' for i in range(7)]
train_df['max_discount'] = train_df[discount_columns].max(axis=1)

# Calculate the maximum discount applied in Sales_Test
discount_columns_test = [f'type_{i}_discount' for i in range(7)]
test_df['max_discount'] = test_df[discount_columns_test].max(axis=1)


train_df.info()
test_df.info()


from sklearn.preprocessing import LabelEncoder
# Kiểm tra và chuyển cột date thành datetime
if 'date' not in train_df.columns:
    raise ValueError("Cột 'date' không tồn tại trong train_df")
if 'date' not in test_df.columns:
    raise ValueError("Cột 'date' không tồn tại trong test_df")

train_df['date'] = pd.to_datetime(train_df['date'], errors='coerce')
test_df['date'] = pd.to_datetime(test_df['date'], errors='coerce')

# Kiểm tra giá trị datetime hợp lệ
if train_df['date'].isna().all():
    raise ValueError("Cột 'date' trong train_df không chứa giá trị datetime hợp lệ")
if test_df['date'].isna().all():
    raise ValueError("Cột 'date' trong test_df không chứa giá trị datetime hợp lệ")

# Thêm các đặc trưng sin và cos cho train_df
# Trích xuất các thành phần thời gian (dayofyear, day, month, year) từ cột date cho cả train_df và test_df
train_df['dayofyear'] = train_df['date'].dt.dayofyear
train_df["day"] = train_df["date"].dt.day
train_df["month"] = train_df["date"].dt.month
train_df["year"] = train_df["date"].dt.year

# Tạo các đặc trưng year_sin, year_cos, month_sin, month_cos, day_sin, day_cos bằng 
# cách áp dụng biến đổi sin và cos để biểu diễn tính chu kỳ của năm, tháng, và ngày.
# Các đặc trưng này giúp mô hình học được các mẫu lặp lại theo thời gian (ví dụ:
# doanh số tăng vào cuối năm hoặc vào các ngày cụ thể trong tháng).
train_df['year_sin'] = np.sin(2 * np.pi * train_df['year'] / train_df['year'].max())
train_df['year_cos'] = np.cos(2 * np.pi * train_df['year'] / train_df['year'].max())
train_df['month_sin'] = np.sin(2 * np.pi * train_df['month'] / 12)
train_df['month_cos'] = np.cos(2 * np.pi * train_df['month'] / 12)
train_df['day_sin'] = np.sin(2 * np.pi * train_df['day'] / 31)
train_df['day_cos'] = np.cos(2 * np.pi * train_df['day'] / 31)



 # 1. days_to_holiday (Tính số ngày đến kỳ nghỉ tiếp theo)
train_df['next_holiday_date'] = train_df.loc[train_df['holiday'] == 1, 'date'].shift(-1)
train_df['next_holiday_date'] = train_df['next_holiday_date'].bfill()
train_df['days_to_holiday'] = (train_df['next_holiday_date'] - train_df['date']).dt.days
train_df.drop(columns=['next_holiday_date'], inplace=True)

    # 2. days_to_shops_closed: Số ngày đến ngày cửa hàng đóng cửa tiếp theo.
train_df['next_shops_closed_date'] = train_df.loc[train_df['shops_closed'] == 1, 'date'].shift(-1)
train_df['next_shops_closed_date'] =train_df['next_shops_closed_date'].bfill()
train_df['days_to_shops_closed'] = (train_df['next_shops_closed_date'] - train_df['date']).dt.days
train_df.drop(columns=['next_shops_closed_date'], inplace=True)

    # 3. day_after_closing: Đánh dấu các ngày ngay sau khi cửa hàng đóng cửa.
train_df['day_after_closing'] = (
    (train_df['shops_closed'] == 0) & (train_df['shops_closed'].shift(1) == 1)
    ).astype(int)
train_df['long_weekend'] = (
        (train_df['shops_closed'] == 1) & (train_df['shops_closed'].shift(1) == 1)
    ).astype(int)





# Thêm các đặc trưng sin và cos cho test_df
test_df['dayofyear'] = test_df['date'].dt.dayofyear

test_df["day"] = test_df["date"].dt.day
test_df["month"] = test_df["date"].dt.month
test_df["year"] = test_df["date"].dt.year
test_df['year_sin'] = np.sin(2 * np.pi * test_df['year'] / test_df['year'].max())  # Sử dụng max từ train_df
test_df['year_cos'] = np.cos(2 * np.pi * test_df['year'] / test_df['year'].max())  # Sử dụng max từ train_df
test_df['month_sin'] = np.sin(2 * np.pi * test_df['month'] / 12)
test_df['month_cos'] = np.cos(2 * np.pi * test_df['month'] / 12)
test_df['day_sin'] = np.sin(2 * np.pi * test_df['day'] / 31)
test_df['day_cos'] = np.cos(2 * np.pi * test_df['day'] / 31)



                            
 # 1. days_to_holiday (Tính số ngày đến kỳ nghỉ tiếp theo)
test_df['next_holiday_date'] = test_df.loc[test_df['holiday'] == 1, 'date'].shift(-1)
test_df['next_holiday_date'] = test_df['next_holiday_date'].bfill()
test_df['days_to_holiday'] = (test_df['next_holiday_date'] - test_df['date']).dt.days
test_df.drop(columns=['next_holiday_date'], inplace=True)

    # 2. days_to_shops_closed
test_df['next_shops_closed_date'] = test_df.loc[test_df['shops_closed'] == 1, 'date'].shift(-1)
test_df['next_shops_closed_date'] =test_df['next_shops_closed_date'].bfill()
test_df['days_to_shops_closed'] = (test_df['next_shops_closed_date'] - test_df['date']).dt.days
test_df.drop(columns=['next_shops_closed_date'], inplace=True)

    # 3. day_after_closing
test_df['day_after_closing'] = (
    (test_df['shops_closed'] == 0) & (test_df['shops_closed'].shift(1) == 1)
    ).astype(int)

test_df['long_weekend'] = (
        (test_df['shops_closed'] == 1) & (test_df['shops_closed'].shift(1) == 1)
    ).astype(int)


# Kiểm tra kết quả
print("Cột trong train_df sau khi thêm đặc trưng:", train_df.columns.tolist())
print("Cột trong test_df sau khi thêm đặc trưng:", test_df.columns.tolist())


# keep_columns.append('category_sales_sum')
# keep_columns.append('category_sales_avg')
# keep_columns.append('category_orders_sum')
# keep_columns.append('category_orders_avg')
# keep_columns.append('total_discount')
# keep_columns.append('L1_category_name_en')
# train_df = train_df[keep_columns]
train_df.info()
test_df.info()


# Định nghĩa lag_features
lag_features = [1, 7, 14, 28]  # Lag 1, 7, 14, 28 ngày

# Đảm bảo cột date là datetime
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

# --- Xử lý train_df ---

# Nhóm dữ liệu theo unique_id và warehouse
train_grouped = train_df.groupby(["unique_id", "warehouse"])

# Tính lag features cho train_df
for i in lag_features:
    # Tạo cột lag
    train_df[f"sales_item_warehouse_lag_{i}"] = train_grouped["sales"].shift(i)
    # Điền NaN bằng giá trị sales cuối cùng của nhóm, nếu không có thì điền 0
    train_df[f"sales_item_warehouse_lag_{i}"] = train_df[f"sales_item_warehouse_lag_{i}"].fillna(
        train_grouped["sales"].transform("last")
    ).fillna(0)

# Kiểm tra NaN trong các cột lag của train_df
print("\nKiểm tra NaN trong các cột lag của train_df:")
print(train_df[[f"sales_item_warehouse_lag_{i}" for i in lag_features]].isna().sum())

# Kiểm tra phân bố của các cột lag trong train_df
print("\nThống kê mô tả cho các cột lag trong train_df:")
print(train_df[[f"sales_item_warehouse_lag_{i}" for i in lag_features]].describe())

# --- Xử lý test_df ---

# Nhóm dữ liệu theo unique_id và warehouse
test_grouped = test_df.groupby(["unique_id", "warehouse"])

# Tính lag features cho test_df
for i in lag_features:
    # Tạo cột lag
    test_df[f"sales_item_warehouse_lag_{i}"] = test_grouped["sales"].shift(i)
    # Điền NaN bằng giá trị sales cuối cùng của nhóm, nếu không có thì điền 0
    test_df[f"sales_item_warehouse_lag_{i}"] = test_df[f"sales_item_warehouse_lag_{i}"].fillna(
        test_grouped["sales"].transform("last")
    ).fillna(0)

# Kiểm tra NaN trong các cột lag của test_df
print("\nKiểm tra NaN trong các cột lag của test_df:")
print(test_df[[f"sales_item_warehouse_lag_{i}" for i in lag_features]].isna().sum())

# Kiểm tra phân bố của các cột lag trong test_df
print("\nThống kê mô tả cho các cột lag trong test_df:")
print(test_df[[f"sales_item_warehouse_lag_{i}" for i in lag_features]].describe())


train_df.info()
test_df.info()


plt.figure(figsize=(16, 12))
corr_matrix = train_df.corr(numeric_only=True).abs()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


    # Loại bỏ cột weight và availability
    columns_to_drop = ['weight', 'availability']
    train_df = train_df.drop(columns=[col for col in columns_to_drop if col in train_df.columns], errors='ignore')


from sklearn.preprocessing import LabelEncoder
# Tối ưu hóa kiểu dữ liệu: Chuyển đổi các cột số thực (float64) thành float32, số nguyên (int64) thành
# int32, và cột boolean (bool) thành uint8 trong cả train_df và test_df để giảm bộ nhớ sử dụng
for col in train_df.select_dtypes('float64'):
    train_df[col] = train_df[col].astype('float32')
for col in train_df.select_dtypes('int64'):
    train_df[col] = train_df[col].astype('int32')
for col in train_df.select_dtypes('bool'):
    train_df[col] = train_df[col].astype('uint8')

for col in test_df.select_dtypes('float64'):
    test_df[col] = test_df[col].astype('float32')
for col in test_df.select_dtypes('int64'):
    test_df[col] = test_df[col].astype('int32')
for col in test_df.select_dtypes('bool'):
    test_df[col] = test_df[col].astype('uint8')
# Label encoding các cột object thành số nguyên, giúp mô hình học máy xử lý được các giá trị chuỗi.
categorical_cols = ['warehouse', 'holiday_name', 'name', 'L1_category_name_en', 
                    'L2_category_name_en', 'L3_category_name_en', 'L4_category_name_en']

label_encoders = {}
for col in categorical_cols:
    if col in train_df.columns and col in test_df.columns:
        le = LabelEncoder()
        # Xử lý từng cột để giảm RAM
        train_vals = train_df[col].astype(str).values
        test_vals = test_df[col].astype(str).values
        combined = np.concatenate([train_vals, test_vals])
        le.fit(combined)
        train_df[col] = le.transform(train_vals)
        test_df[col] = le.transform(test_vals)
        label_encoders[col] = le
        del combined  # Xóa biến tạm

# # Hàm inverse_norm
# def inverse_norm(df, period, values):
#     sales_min = train_df["sales"].min()
#     sales_max = train_df["sales"].max()
#     return values * (sales_max - sales_min) + sales_min


# Loại bỏ cột datetime64 và chỉ giữ cột số
# Xác định các đặc trưng (features) bằng cách chọn các cột số từ train_df và test_df, loại bỏ các cột 
# không liên quan (unique_id, date, sales, availability).
# Lấy tập hợp chung (set(train_features) & set(test_features)) để đảm bảo các đặc trưng giống nhau 
# trong cả hai tập.
train_features = [c for c in train_df.columns if c not in ["unique_id", "date", "sales", "availability"] 
                 and train_df[c].dtype in [np.float32, np.int32, np.uint8, np.int64]]
test_features = [c for c in test_df.columns if c not in ["unique_id", "date", "sales", "availability"] 
                 and test_df[c].dtype in [np.float32, np.int32, np.uint8, np.int64]]

features = list(set(train_features) & set(test_features))

# Kiểm tra kiểu dữ liệu của features
print("Feature dtypes in train_df:")
for col in features:
    print(f"{col}: {train_df[col].dtype}")
print("\nFeature dtypes in test_df:")
for col in features:
    print(f"{col}: {test_df[col].dtype}")

# Xử lý NaN
train_df[features] = train_df[features].fillna(train_df[features].mean())
test_df[features] = test_df[features].fillna(train_df[features].mean())


# Giới hạn tập train_df lại từ năm 2022 đến ngày cuối cùng trừ 28 ngày bởi vì giai đoạn từ 2022 trở đi COVID 19
# không còn tác động quá lớn tới kinh tế, do đó bỏ đi thời gian phía trước để tính tổng quát hơn. 
target = "sales"
training_dates = (pd.to_datetime('2022-01-01'), train_df["date"].max() - pd.Timedelta(days=14))
validation_dates = (training_dates[1] + pd.Timedelta(days=1), train_df["date"].max())
test_dates = (test_df["date"].min(), test_df["date"].max())
weight_map = weights_df.set_index('unique_id')['weight'].to_dict()


# Chuẩn bị dữ liệu, chia tập train và tập validate nhằm huấn luyện và đánh giá mô hình
X_train = train_df[train_df["date"].between(*training_dates)][features]
y_train = train_df[train_df["date"].between(*training_dates)][target]
X_val = train_df[train_df["date"].between(*validation_dates)][features]
y_val = train_df[train_df["date"].between(*validation_dates)][target]
# Tạo weight_map từ weights_df để ánh xạ trọng số cho mỗi unique_id
unique_id_train = train_df[train_df["date"].between(*training_dates)]["unique_id"]
unique_id_val = train_df[train_df["date"].between(*validation_dates)]["unique_id"]
# Kiểm tra NaN
if X_train.isna().any().any() or X_val.isna().any().any():
    print("NaN detected in input data. Filling remaining NaNs with mean.")
    X_train = X_train.fillna(X_train.mean())
    X_val = X_val.fillna(X_val.mean())# Kiểm tra NaN
if X_train.isna().any().any() or X_val.isna().any().any():
    print("NaN detected in input data. Filling remaining NaNs with mean.")
    X_train = X_train.fillna(X_train.mean())
    X_val = X_val.fillna(X_val.mean())
# Kiểm tra weight_map
missing_ids = set(unique_id_val) - set(weight_map.keys())
if missing_ids:
    print(f"Missing unique_id in weight_map: {missing_ids}")
    # Gán trọng số mặc định = 1 cho các unique_id thiếu
    for missing_id in missing_ids:
        weight_map[missing_id] = 1.0


train_df.info()
test_df.info()


import pickle
import json
from flaml import AutoML

# Khởi tạo AutoML
automl = AutoML()

# Thiết lập cấu hình với hỗ trợ GPU
settings = {
    "time_budget": 15000,  # Giảm xuống 2 giờ để tránh vượt giới hạn Kaggle
    "metric": "mae",
    "estimator_list": ["xgboost"],
    "task": "regression",
    "log_file_name": "experiment_xgboost_gpu_flaml.log",
    "seed": 43,
    "n_jobs": 4,  # Giới hạn số luồng CPU
    "max_iter": 50,  # Giới hạn số vòng lặp để tiết kiệm thời gian
    "gpu_per_trial": 1  # Sử dụng 1 GPU cho mỗi trial
}

# Huấn luyện mô hình với sample_weight
automl.fit(
    X_train=X_train,
    y_train=y_train,
    sample_weight=unique_id_train.map(weight_map).values,
    **settings
)

# Lấy mô hình tốt nhất
best_model = automl.model.estimator
print("\nTham số tối ưu:", automl.best_config)

# Lưu mô hình tốt nhất (pickle format)
with open('best_model_xgboost_gpu.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# In kết quả tốt nhất từ best_result
print("\nKết quả tốt nhất từ automl.best_result:")
print(json.dumps(automl.best_result, indent=4))

# Tải lại mô hình tốt nhất (pickle format)
with open('best_model_xgboost_gpu.pkl', 'rb') as f:
    loaded_best_model = pickle.load(f)

# --- Dự đoán và tính WMAE cho validation trong train_df ---
val_period = train_df["date"].between(*validation_dates)
y_pred_val = inverse_norm(train_df, val_period, loaded_best_model.predict(X_val), scalers)
y_val_true = inverse_norm(train_df, val_period, y_val, scalers)
wmae = mean_absolute_error(y_val_true, y_pred_val, sample_weight=unique_id_val.map(weight_map).values)
print(f"WMAE: {wmae}")


# # Chuyển dữ liệu sang GPU
# X_train_gpu = cp.array(X_train.values, dtype=cp.float32)
# y_train_gpu = cp.array(y_train.values, dtype=cp.float32)
# X_val_gpu = cp.array(X_val.values, dtype=cp.float32)

# # Huấn luyện mô hình XGBoost với GridSearchCV và GPU
# xgb = XGBRegressor(
#     objective='reg:absoluteerror',
#     random_state=2025,
#     use_label_encoder=False,
#     tree_method='hist',
#     device='cuda'  # Chạy trên GPU
# )
# xgb_param_grid = {
#     'n_estimators': [200, 500],
#     'max_depth': [7],
#     'learning_rate': [0.1]
# }
# xgb_gs = GridSearchCV(
#     xgb,
#     xgb_param_grid,
#     cv=3,
#     scoring='neg_mean_absolute_error',
#     n_jobs=1,  # Tắt song song để giảm RAM
#     verbose=1
# )

# # Huấn luyện và kiểm tra lỗi
# try:
#     xgb_gs.fit(X_train_gpu.get(), y_train_gpu.get())  # Chuyển về numpy cho GridSearchCV
#     print("GridSearchCV completed successfully.")
# except Exception as e:
#     print(f"Error during GridSearchCV fit: {e}")
#     raise



# y_pred_xgb = cp.asnumpy(xgb_gs.predict(X_val_gpu)) 

# # Đánh giá mô hình
# wmae = mean_absolute_error(y_val, y_pred_xgb, 
#                           sample_weight=unique_id_val.map(weight_map).values)

# # In kết quả
# print(f"Best Parameters: {xgb_gs.best_params_}")
# print(f"Validation WMAE: {wmae}")
# print(f"Test Predictions: {y_pred_xgb}")


# # === 5. Ridge Regression ===
# ridge_pipe = Pipeline([
#     ('scaler', StandardScaler()),
#     ('ridge', Ridge())
# ])
# ridge_param_grid = {
#     'ridge__alpha': [0.1, 1, 10, 100],
#     'ridge__solver': ['auto', 'lsqr'],
#     'ridge__fit_intercept': [True]
# }
# ridge_gs = GridSearchCV(
#     ridge_pipe,
#     ridge_param_grid,
#     cv=5,
#     n_jobs=1,
#     scoring='neg_mean_absolute_error', 
#     verbose=1
# )
# ridge_gs.fit(X_train, y_train)
# # Dự đoán trên tập xác thực và kiểm tra
# y_pred_ridge = ridge_gs.predict(X_val)

# # Đánh giá mô hình
# wmae = mean_absolute_error(y_val, y_pred_ridge, 
#                           sample_weight=unique_id_val.map(weight_map).values)

# # In kết quả
# print(f"Best Parameters: {ridge_gs.best_params_}")
# print(f"Validation WMAE: {wmae}")
# print(f"Test Predictions: {y_pred_ridge}")


# # Chuyển dữ liệu sang GPU
# X_train_np = X_train.values.astype(np.float32)
# y_train_np = y_train.values.astype(np.float32)
# X_val_np = X_val.values.astype(np.float32)



# # === 6. LightGBM (GridSearch) ===
# lgb = LGBMRegressor(objective= 'regression', random_state=43, force_row_wise=True, verbose=-1, device='gpu')
# lgb_param_dist = {
#     'n_estimators': [5000],
#     'max_depth': [10],
#     'learning_rate': [0.054],
#     'num_leaves': [273],
#     'min_child_samples': [40],
#     'colsample_bytree' : [0.852], 
#     'colsample_bynode' : [0.72], 
#     'min_data_in_leaf' : [6],
#     'reg_alpha': [5],
#     'reg_lambda': [0.005]
# }
# lgb_gs = GridSearchCV(
#     lgb, lgb_param_dist,
#     cv=5,
#     scoring='neg_mean_absolute_error',
#     n_jobs=1, verbose=1)
# # Huấn luyện
# try:
#     lgb_gs.fit(X_train_np, y_train_np)  # Chuyển về numpy cho RandomizedSearchCV
#     print("GridSearchCV completed successfully.")
# except Exception as e:
#     print(f"Error during RandomizedSearchCV fit: {e}")
#     raise


# y_pred_lgb = cp.asnumpy(lgb_gs.predict(X_val_np)) 
# # Đánh giá mô hình
# wmae = mean_absolute_error(y_val, y_pred_lgb, 
#                           sample_weight=unique_id_val.map(weight_map).values)

# # In kết quả
# print(f"Best Parameters: {lgb_gs.best_params_}")
# print(f"Validation WMAE: {wmae}")
# print(f"Test Predictions: {y_pred_lgb}")


# # === 7. ElasticNet ===
# en_pipe = Pipeline([
#     ('scaler', StandardScaler()),
#     ('enet', ElasticNet(random_state=43))
# ])
# en_param_grid = {
#     'enet__alpha': [0.01, 0.1, 1],
#     'enet__l1_ratio': [0.1, 0.5, 0.9],
#     'enet__fit_intercept': [True, False],
#     'enet__max_iter': [1000]
# }
# en_gs = GridSearchCV(
#     en_pipe,
#     en_param_grid,
#     cv=3,
#     scoring='neg_mean_absolute_error',
#     n_jobs=1,
#     verbose=1
# )
# en_gs.fit(X_train, y_train)
# y_pred_en = en_gs.predict(X_val)
# # Đánh giá mô hình
# wmae = mean_absolute_error(y_val, y_pred_en, 
#                           sample_weight=unique_id_val.map(weight_map).values)

# # In kết quả
# print(f"Best Parameters: {en_gs.best_params_}")
# print(f"Validation WMAE: {wmae}")
# print(f"Test Predictions: {y_pred_en}")



# #8. catboost
# # Huấn luyện mô hình CatBoost với GridSearchCV và GPU
# X_train_np = X_train.values.astype(np.float32)
# y_train_np = y_train.values.astype(np.float32)
# X_val_np = X_val.values.astype(np.float32)


# # Huấn luyện mô hình CatBoost với GridSearchCV và GPU
# cat = CatBoostRegressor(random_seed=43, verbose=0, task_type='GPU', grow_policy='Lossguide')
# cat_param_grid = {
#     'iterations': [500, 1000],
#     'depth': [8, 10],
#     'bagging_temperature': [0.5],
#     'learning_rate': [0.05, 0.1],
#     'max_leaves': [128],
#     'l2_leaf_reg': [2],
#     'min_data_in_leaf': [24]
# }
# cat_gs = GridSearchCV(
#     cat,
#     cat_param_grid,
#     cv=5,
#     scoring='neg_mean_absolute_error',
#     n_jobs=1,  
#     verbose=1
# )


# try:
#     cat_gs.fit(X_train_np, y_train_np) 
#     print("GridSearchCV completed successfully.")
# except Exception as e:
#     print(f"Error during RandomizedSearchCV fit: {e}")
#     raise

# y_pred_cat = cat_gs.predict(X_val_np)
# # Đánh giá mô hình
# wmae = mean_absolute_error(y_val, y_pred_cat, 
#                           sample_weight=unique_id_val.map(weight_map).values)

# # In kết quả
# print(f"Best Parameters: {cat_gs.best_params_}")
# print(f"Validation WMAE: {wmae}")
# print(f"Test Predictions: {y_pred_cat}")


# keep_columns_1=train_df.columns
# keep_columns_1


# train_df.info()


# test_df.info()


# # 1. Identify missing columns
# missing_cols = list(set(train_df.columns) - set(test_df.columns))
# missing_cols


# keep_columns=train_df.columns
# keep_columns


# # 2. Mã hóa và chuyển đổi date
# le = LabelEncoder()
# # test_df['L1_category_name_en'] = le.fit_transform(test_df['L1_category_name_en'])
# # test_df['holiday_name'] = le.fit_transform(test_df['holiday_name'])  # Apply Label Encoding
# test_df['name'] = le.fit_transform(test_df['name'])
# test_df['L2_category_name_en'] = le.fit_transform(test_df['L2_category_name_en'].astype(str))
# test_df['L3_category_name_en'] = le.fit_transform(test_df['L3_category_name_en'].astype(str))
# test_df['L4_category_name_en'] = le.fit_transform(test_df['L4_category_name_en'].astype(str))


# # test_df['date'] = pd.to_datetime(test_df['date'])
# # test_df['date'] = (test_df['date'] - test_df['date'].min()).dt.days
# for col in test_df.select_dtypes('float64'): test_df[col] = test_df[col].astype('float32')
# for col in test_df.select_dtypes('int64'):   test_df[col] = test_df[col].astype('int32')
# for col in test_df.select_dtypes('bool'):    test_df[col] = test_df[col].astype('uint8')


# # Dự đoán trên test_df
# try:
#     X_test_new = test_df[test_df["date"].between(*test_dates)][features]  # Lọc theo test_dates
#     X_test_new_np = X_test_new.values.astype(np.float32)
#     y_pred_test = lgb_gs.predict(X_test_new_np)
#     # Nếu cần inverse_norm, bỏ comment dòng dưới
#     # y_pred_test = inverse_norm(test_df, y_pred_test)
#     test_df.loc[test_df["date"].between(*test_dates), 'sales'] = y_pred_test
# except Exception as e:
#     print(f"Error during test prediction: {e}")
#     raise


try:
    test_period = test_df["date"].between(*test_dates)
    if test_period.sum() == 0:
        raise ValueError("Không có dữ liệu trong khoảng test_dates.")
    X_test_new = test_df[test_period][features]
    y_pred_test = inverse_norm(test_df, test_period, loaded_best_model.predict(X_test_new), scalers)
    test_df.loc[test_period, 'sales'] = y_pred_test
except Exception as e:
    print(f"Error during test prediction: {e}")
    raise



# save submission file with predictions
test_df["id"] = test_df["unique_id"].astype(str) + "_" + pd.to_datetime(test_df["date"]).dt.strftime("%Y-%m-%d")
test_df[["id", "sales"]].to_csv(f"submission.csv", index=False)


