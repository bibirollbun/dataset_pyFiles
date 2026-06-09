import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập hiển thị cho Pandas (tùy chọn, để dễ nhìn hơn)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Đường dẫn tới file dữ liệu (thay đổi nếu cần)
data_path = "./" # Giả sử file nằm cùng thư mục với notebook/script

# Đọc dữ liệu
try:
    df_train = pd.read_csv("/kaggle/input/rossmann-store-sales/train.csv", low_memory=False)
    df_store = pd.read_csv("/kaggle/input/rossmann-store-sales/store.csv", low_memory=False)
    df_test = pd.read_csv("/kaggle/input/rossmann-store-sales/test.csv", low_memory=False)
    df_submission = pd.read_csv("/kaggle/input/rossmann-store-sales/sample_submission.csv", low_memory=False)
except FileNotFoundError:
    print("Lỗi: Một hoặc nhiều file dữ liệu không tìm thấy. Hãy kiểm tra đường dẫn!")
    # Thoát hoặc xử lý lỗi ở đây nếu cần
    exit()


# print("--- Thông tin df_train ---")
# print(df_train.head())
# print(f"\nKích thước df_train: {df_train.shape}")
# df_train.info()
# print("\nThống kê mô tả df_train:")
# print(df_train.describe(include='all'))
# print("\nSố lượng giá trị duy nhất df_train:")
# print(df_train.nunique())

# print("\n\n--- Thông tin df_store ---")
# print(df_store.head())
# print(f"\nKích thước df_store: {df_store.shape}")
# df_store.info()
# print("\nThống kê mô tả df_store:")
# print(df_store.describe(include='all'))
# print("\nSố lượng giá trị duy nhất df_store:")
# print(df_store.nunique())

# print("\n\n--- Thông tin df_test ---")
# print(df_test.head())
# print(f"\nKích thước df_test: {df_test.shape}")
# df_test.info()
# print("\nThống kê mô tả df_test:")
# print(df_test.describe(include='all'))
# print("\nSố lượng giá trị duy nhất df_test:")
# print(df_test.nunique())

# print("\n\n--- Thông tin df_submission ---")
# print(df_submission.head())
# print(f"\nKích thước df_submission: {df_submission.shape}")
# df_submission.info()


# Phân tích biến Sales trong df_train
plt.figure(figsize=(12, 6))
sns.histplot(df_train['Sales'], bins=100, kde=True)
plt.title('Phân phối của Doanh số (Sales)')
plt.xlabel('Sales')
plt.ylabel('Tần suất')
plt.show()

print(f"\nThống kê Sales:\n{df_train['Sales'].describe()}")
print(f"Số lượng Sales = 0: {len(df_train[df_train['Sales'] == 0])}")
print(f"Tỷ lệ Sales = 0: {len(df_train[df_train['Sales'] == 0]) / len(df_train):.2%}")

# # Phân phối Sales khi cửa hàng mở cửa (Open=1)
# plt.figure(figsize=(12, 6))
# sns.histplot(df_train[df_train['Open'] == 1]['Sales'], bins=100, kde=True, color='green')
# plt.title('Phân phối của Doanh số (Sales) khi cửa hàng Mở cửa (Open=1)')
# plt.xlabel('Sales')
# plt.ylabel('Tần suất')
# plt.show()

# # Xem xét log(Sales)
# plt.figure(figsize=(12, 6))
# sales_open_log = np.log1p(df_train[(df_train['Open'] == 1) & (df_train['Sales'] > 0)]['Sales']) # Chỉ log các sales > 0
# sns.histplot(sales_open_log, bins=100, kde=True, color='purple')
# plt.title('Phân phối của log(1+Sales) khi cửa hàng Mở cửa và Sales > 0')
# plt.xlabel('log(1+Sales)')
# plt.ylabel('Tần suất')
# plt.show()


# Chuyển đổi cột Date sang datetime
df_train['Date'] = pd.to_datetime(df_train['Date'])
df_test['Date'] = pd.to_datetime(df_test['Date'])

print("\n--- Kiểu dữ liệu sau khi chuyển đổi Date ---")
print("df_train Date type:", df_train['Date'].dtype)
print("df_test Date type:", df_test['Date'].dtype)

# Kiểm tra các cửa hàng đóng cửa (Open=0) và Sales
closed_sales_non_zero = df_train[(df_train['Open'] == 0) & (df_train['Sales'] != 0)]
print(f"\nSố lượng trường hợp cửa hàng đóng cửa (Open=0) nhưng Sales != 0: {len(closed_sales_non_zero)}")
# Thông thường, kết quả này nên là 0. Nếu có, đó là dữ liệu nhiễu cần xem xét.

# Gộp df_store vào df_train và df_test
# Lưu ý: df_test có thể có các cửa hàng không có trong df_train (ít khả năng với cuộc thi này nhưng nên cẩn thận)
# Tuy nhiên, tất cả các cửa hàng trong df_train và df_test đều có trong df_store.

df_train_merged = pd.merge(df_train, df_store, on='Store', how='left')
df_test_merged = pd.merge(df_test, df_store, on='Store', how='left')

print("\n--- Kích thước sau khi merge ---")
print(f"Kích thước df_train_merged: {df_train_merged.shape}")
print(f"Kích thước df_test_merged: {df_test_merged.shape}")

# Kiểm tra xem có store nào trong train/test không có trong store_df không (không nên xảy ra)
# print(df_train_merged[df_train_merged['StoreType'].isnull()]['Store'].unique())
# print(df_test_merged[df_test_merged['StoreType'].isnull()]['Store'].unique())

print("\n--- df_train_merged head ---")
print(df_train_merged.head())


print("\n--- Missing values trong df_train_merged ---")
print(df_train_merged.isnull().sum()[df_train_merged.isnull().sum() > 0])

print("\n--- Missing values trong df_test_merged ---")
print(df_test_merged.isnull().sum()[df_test_merged.isnull().sum() > 0])


# Kiểm tra các hàng có Open = NaN trong test_merged
# print(df_test_merged[df_test_merged['Open'].isnull()])
# Giả sử chúng ta điền 1 cho các giá trị NaN này
df_test_merged.loc[df_test_merged['Open'].isnull(), 'Open'] = 1
print(f"\nSố Open NaN sau khi điền trong test: {df_test_merged['Open'].isnull().sum()}")


median_competition_distance = df_store['CompetitionDistance'].median()

# Điền giá trị median vào các ô NaN trong train và test
df_train_merged.loc[df_train_merged['CompetitionDistance'].isna(), 'CompetitionDistance'] = median_competition_distance
df_test_merged.loc[df_test_merged['CompetitionDistance'].isna(), 'CompetitionDistance'] = median_competition_distance

print(f"Số CompetitionDistance NaN sau khi điền trong train: {df_train_merged['CompetitionDistance'].isnull().sum()}")
print(f"Số CompetitionDistance NaN sau khi điền trong test: {df_test_merged['CompetitionDistance'].isnull().sum()}")


# Điền các giá trị liên quan đến Competition bằng mode
cols_to_fill_competition = ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']

for col in cols_to_fill_competition:
    # Tính mode cho từng dataset
    mode_val_train = df_train_merged[col].mode()[0] if not df_train_merged[col].mode().empty else 0
    mode_val_test = df_test_merged[col].mode()[0] if not df_test_merged[col].mode().empty else 0
    
    # Điền giá trị bằng cách gán lại cột (thay vì dùng inplace)
    df_train_merged[col] = df_train_merged[col].fillna(mode_val_train)
    df_test_merged[col] = df_test_merged[col].fillna(mode_val_test)

print(f"NaNs in CompetitionOpenSinceMonth (train): {df_train_merged['CompetitionOpenSinceMonth'].isnull().sum()}")
print(f"NaNs in CompetitionOpenSinceYear (train): {df_train_merged['CompetitionOpenSinceYear'].isnull().sum()}")
print(f"NaNs in CompetitionOpenSinceMonth (test): {df_test_merged['CompetitionOpenSinceMonth'].isnull().sum()}")
print(f"NaNs in CompetitionOpenSinceYear (test): {df_test_merged['CompetitionOpenSinceYear'].isnull().sum()}")


print("\nKiểm tra NaNs cho Promo2=1:")
print("Train, Promo2=1, Promo2SinceWeek is NaN:",
      df_train_merged[(df_train_merged['Promo2'] == 1) & (df_train_merged['Promo2SinceWeek'].isnull())].shape[0])
print("Test, Promo2=1, Promo2SinceWeek is NaN:",
      df_test_merged[(df_test_merged['Promo2'] == 1) & (df_test_merged['Promo2SinceWeek'].isnull())].shape[0])

# Xử lý các trường hợp Promo2=0
for df in [df_train_merged, df_test_merged]:
    df.loc[df['Promo2'] == 0, ['Promo2SinceWeek', 'Promo2SinceYear']] = 0
    df.loc[df['Promo2'] == 0, 'PromoInterval'] = 'None'

# Xử lý các trường hợp Promo2=1 nhưng vẫn còn NaN
for col in ['Promo2SinceWeek', 'Promo2SinceYear']:
    # Tính mode chung từ cả train và test để nhất quán
    combined_mode = pd.concat([
        df_train_merged[df_train_merged['Promo2'] == 1][col],
        df_test_merged[df_test_merged['Promo2'] == 1][col]
    ]).mode()
    
    mode_val = combined_mode[0] if not combined_mode.empty else 0
    
    # Điền giá trị cho cả train và test
    df_train_merged.loc[(df_train_merged['Promo2'] == 1) & (df_train_merged[col].isnull()), col] = mode_val
    df_test_merged.loc[(df_test_merged['Promo2'] == 1) & (df_test_merged[col].isnull()), col] = mode_val

# Xử lý PromoInterval
combined_interval_mode = pd.concat([
    df_train_merged[df_train_merged['Promo2'] == 1]['PromoInterval'],
    df_test_merged[df_test_merged['Promo2'] == 1]['PromoInterval']
]).mode()

interval_mode = combined_interval_mode[0] if not combined_interval_mode.empty else 'None'

df_train_merged.loc[(df_train_merged['Promo2'] == 1) & (df_train_merged['PromoInterval'].isnull()), 'PromoInterval'] = interval_mode
df_test_merged.loc[(df_test_merged['Promo2'] == 1) & (df_test_merged['PromoInterval'].isnull()), 'PromoInterval'] = interval_mode

# Kiểm tra kết quả
print("\n--- Missing values sau khi xử lý ---")
print("Train:")
print(df_train_merged[['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']].isnull().sum())
print("\nTest:")
print(df_test_merged[['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']].isnull().sum())


def create_date_features(df):
    df_copy = df.copy()
    df_copy['Year'] = df_copy['Date'].dt.year
    df_copy['Month'] = df_copy['Date'].dt.month
    df_copy['Day'] = df_copy['Date'].dt.day
    df_copy['WeekOfYear'] = df_copy['Date'].dt.isocalendar().week.astype(int) # Sử dụng isocalendar() để có số tuần ISO
    df_copy['DayOfYear'] = df_copy['Date'].dt.dayofyear
    # df_copy['DayOfWeek'] đã có sẵn, nhưng chúng ta có thể muốn nó bắt đầu từ 0 (0=Thứ Hai, 6=Chủ Nhật) nếu cần
    # df_copy['DayOfWeek_0_6'] = df_copy['Date'].dt.dayofweek

    # Đặc trưng Cyclical (giúp mô hình hiểu tính chu kỳ)
    df_copy['MonthSin'] = np.sin(2 * np.pi * df_copy['Month']/12)
    df_copy['MonthCos'] = np.cos(2 * np.pi * df_copy['Month']/12)
    df_copy['DayOfWeekSin'] = np.sin(2 * np.pi * df_copy['DayOfWeek']/7) # Giả sử DayOfWeek từ 1-7
    df_copy['DayOfWeekCos'] = np.cos(2 * np.pi * df_copy['DayOfWeek']/7)
    df_copy['WeekOfYearSin'] = np.sin(2 * np.pi * df_copy['WeekOfYear']/52) # 52 hoặc 53 tuần
    df_copy['WeekOfYearCos'] = np.cos(2 * np.pi * df_copy['WeekOfYear']/52)

    # Cờ (flag)
    df_copy['IsWeekend'] = df_copy['DayOfWeek'].apply(lambda x: 1 if x >= 6 else 0) # Thứ 7, Chủ Nhật
    df_copy['StartOfMonth'] = df_copy['Date'].dt.is_month_start.astype(int)
    df_copy['EndOfMonth'] = df_copy['Date'].dt.is_month_end.astype(int)
    # df_copy['StartOfYear'] = df_copy['Date'].dt.is_year_start.astype(int) # Ít quan trọng hơn
    # df_copy['EndOfYear'] = df_copy['Date'].dt.is_year_end.astype(int)   # Ít quan trọng hơn

    # Thời gian kể từ đầu năm (hoặc một mốc cố định)
    # df_copy['DaysSinceYearStart'] = (df_copy['Date'] - pd.to_datetime(df_copy['Year'].astype(str) + '-01-01')).dt.days
    # Số ngày trong tháng hiện tại (có thể hữu ích)
    df_copy['DaysInMonth'] = df_copy['Date'].dt.days_in_month


    # Thời gian đến/từ ngày lễ đặc biệt (ví dụ: Giáng Sinh) - có thể thêm sau nếu cần độ phức tạp cao hơn
    # christmas_date = pd.to_datetime(str(df_copy['Year'].iloc[0]) + '-12-25') # Cần xử lý cho từng năm
    # df_copy['DaysToChristmas'] = (christmas_date - df_copy['Date']).dt.days

    return df_copy

df_train_fe = create_date_features(df_train_merged)
df_test_fe = create_date_features(df_test_merged)

print("\n--- df_train_fe sau khi tạo đặc trưng ngày (một vài cột mới) ---")
print(df_train_fe[['Date', 'Year', 'Month', 'Day', 'WeekOfYear', 'IsWeekend', 'MonthSin', 'DaysInMonth']].head())
print("\n--- df_test_fe sau khi tạo đặc trưng ngày (một vài cột mới) ---")
print(df_test_fe[['Date', 'Year', 'Month', 'Day', 'WeekOfYear', 'IsWeekend', 'MonthSin', 'DaysInMonth']].head())


# Kiểm tra các giá trị duy nhất của StateHoliday
print("\nGiá trị duy nhất của StateHoliday trong train:", df_train_fe['StateHoliday'].unique())
print("Giá trị duy nhất của StateHoliday trong test:", df_test_fe['StateHoliday'].unique())

# '0' là string, 0 là float/int. Chúng ta cần đồng nhất.
# Trong dữ liệu train.csv, '0' là string. Trong store.csv, có thể có số 0.
# Khi đọc, Pandas có thể tự động nhận diện. Hãy kiểm tra.
# print(df_train_fe['StateHoliday'].apply(type).value_counts())

# Ánh xạ StateHoliday
# '0' (string) -> 0 (None)
# 'a' -> 1 (Public holiday)
# 'b' -> 2 (Easter holiday)
# 'c' -> 3 (Christmas)
# Bất kỳ giá trị NaN hoặc khác (nếu có) cũng có thể được map về 0.

def map_state_holiday(holiday_val):
    if holiday_val == '0' or holiday_val == 0: # Xử lý cả string '0' và số 0
        return 0
    elif holiday_val == 'a':
        return 1
    elif holiday_val == 'b':
        return 2
    elif holiday_val == 'c':
        return 3
    else: # Bao gồm cả NaN nếu có
        return 0

df_train_fe['StateHolidayNumeric'] = df_train_fe['StateHoliday'].apply(map_state_holiday)
df_test_fe['StateHolidayNumeric'] = df_test_fe['StateHoliday'].apply(map_state_holiday)

# Kiểm tra lại
print("\nStateHolidayNumeric trong train:", df_train_fe['StateHolidayNumeric'].value_counts())
print("StateHolidayNumeric trong test:", df_test_fe['StateHolidayNumeric'].value_counts())

# Chúng ta có thể bỏ cột StateHoliday gốc nếu muốn
# df_train_fe.drop('StateHoliday', axis=1, inplace=True)
# df_test_fe.drop('StateHoliday', axis=1, inplace=True)


def competition_features(df):
    # Tạo bản sao để tránh thay đổi DataFrame gốc
    df_copy = df.copy()
    
    # Chuyển đổi kiểu dữ liệu an toàn
    df_copy['CompetitionOpenSinceMonth'] = df_copy['CompetitionOpenSinceMonth'].astype(int)
    df_copy['CompetitionOpenSinceYear'] = df_copy['CompetitionOpenSinceYear'].astype(int)
    
    # Tạo cột datetime với xử lý lỗi
    competition_date = pd.to_datetime(
        df_copy['CompetitionOpenSinceYear'].astype(str) + '-' +
        df_copy['CompetitionOpenSinceMonth'].astype(str) + '-01',
        format='%Y-%m-%d',
        errors='coerce'
    )
    
    # Tính toán các đặc trưng
    months_since_open = ((df_copy['Year'] - competition_date.dt.year) * 12 + 
                        (df_copy['Month'] - competition_date.dt.month))
    
    # Xử lý giá trị thiếu và không hợp lệ
    months_since_open = months_since_open.fillna(0)  # Thay thế NaN bằng 0
    months_since_open = np.clip(months_since_open, 0, 24*5)  # Giới hạn trong 5 năm
    
    # Tạo cờ competition
    is_open = ((df_copy['Date'] >= competition_date) & 
              (competition_date.notna())).astype(int)
    
    # Gán các cột mới vào DataFrame
    df_copy = df_copy.assign(
        CompetitionOpenSinceDate=competition_date,
        MonthsSinceCompetitionOpen=months_since_open,
        IsCompetitionOpen=is_open
    )
    
    return df_copy

# Áp dụng hàm
df_train_fe = competition_features(df_train_fe)
df_test_fe = competition_features(df_test_fe)

print("\n--- df_train_fe sau khi tạo đặc trưng Competition ---")
print(df_train_fe[['Store', 'Date', 'CompetitionDistance', 
                  'CompetitionOpenSinceDate', 'MonthsSinceCompetitionOpen', 
                  'IsCompetitionOpen']].head())


def promo2_features(df):
    df_copy = df.copy()
    
    # Convert to int without inplace
    df_copy['Promo2SinceWeek'] = df_copy['Promo2SinceWeek'].astype(int)
    df_copy['Promo2SinceYear'] = df_copy['Promo2SinceYear'].astype(int)

    # Create Promo2SinceDate safely
    promo2_mask = (df_copy['Promo2'] == 1)
    promo2_dates = pd.to_datetime(
        df_copy.loc[promo2_mask, 'Promo2SinceYear'].astype(str) + 
        df_copy.loc[promo2_mask, 'Promo2SinceWeek'].astype(str) + '1',
        format='%Y%W%w',
        errors='coerce'
    )
    df_copy['Promo2SinceDate'] = pd.NaT
    df_copy.loc[promo2_mask, 'Promo2SinceDate'] = promo2_dates

    # Calculate months since Promo2 start
    months_since = ((df_copy['Year'] - df_copy['Promo2SinceDate'].dt.year) * 12 + 
                   (df_copy['Month'] - df_copy['Promo2SinceDate'].dt.month))
    months_since = months_since.fillna(0)
    months_since = np.clip(months_since, 0, 24*5)
    df_copy['MonthsSincePromo2Start'] = months_since
    df_copy.loc[df_copy['Promo2'] == 0, 'MonthsSincePromo2Start'] = 0

    # Determine if Promo2 is active
    df_copy['IsPromo2ActiveOnDate'] = 0
    active_mask = (df_copy['Promo2'] == 1) & \
                 (df_copy['Date'] >= df_copy['Promo2SinceDate']) & \
                 (df_copy['Promo2SinceDate'].notna())
    
    for index in df_copy[active_mask].index:
        row = df_copy.loc[index]
        current_month = row['Date'].strftime('%b')
        promo_interval = row['PromoInterval']
        if pd.notna(promo_interval) and isinstance(promo_interval, str) and promo_interval != 'None':
            if current_month in promo_interval.split(','):
                df_copy.at[index, 'IsPromo2ActiveOnDate'] = 1

    return df_copy

df_train_fe = promo2_features(df_train_fe)
df_test_fe = promo2_features(df_test_fe)

print("\n--- df_train_fe sau khi tạo đặc trưng Promo2 ---")
print(df_train_fe[df_train_fe['Promo2']==1][['Store', 'Date', 'Promo2', 'Promo2SinceWeek', 
                                           'Promo2SinceYear', 'PromoInterval', 'Promo2SinceDate',
                                           'MonthsSincePromo2Start', 'IsPromo2ActiveOnDate']].head(10))


print("\nGiá trị duy nhất StoreType:", df_train_fe['StoreType'].unique())
print("Giá trị duy nhất Assortment:", df_train_fe['Assortment'].unique())

# Define mappings
assortment_map = {'a': 1, 'b': 2, 'c': 3}
storetype_map = {'a': 1, 'b': 2, 'c': 3, 'd': 4}

# Create new columns without inplace operations
for df in [df_train_fe, df_test_fe]:
    df['AssortmentNumeric'] = df['Assortment'].map(assortment_map)
    df['StoreTypeNumeric'] = df['StoreType'].map(storetype_map)

# Check results
print("\nAssortmentNumeric counts:\n", df_train_fe['AssortmentNumeric'].value_counts())
print("StoreTypeNumeric counts:\n", df_train_fe['StoreTypeNumeric'].value_counts())

# Alternative safer way to drop columns if needed (commented out as per original)
# df_train_fe = df_train_fe.drop(['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval'], axis=1)
# df_test_fe = df_test_fe.drop(['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval'], axis=1)


# Liệt kê các cột hiện tại
print("\nCột trong df_train_fe trước khi drop:", df_train_fe.columns.tolist())

# Các cột có thể drop (tùy vào chiến lược)
cols_to_drop = [
    'StateHoliday', # Đã có StateHolidayNumeric
    'StoreType',    # Đã có StoreTypeNumeric
    'Assortment',   # Đã có AssortmentNumeric
    'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceDate',
    'Promo2SinceWeek', 'Promo2SinceYear', 'Promo2SinceDate',
    'PromoInterval'
]

# Tạo bản sao trước khi thay đổi
df_train_final = df_train_fe.copy()
df_test_final = df_test_fe.copy()

# Lọc chỉ các cột tồn tại để drop
cols_to_drop_train = [col for col in cols_to_drop if col in df_train_final.columns]
cols_to_drop_test = [col for col in cols_to_drop if col in df_test_final.columns]

# Drop columns an toàn
if cols_to_drop_train:
    df_train_final = df_train_final.drop(columns=cols_to_drop_train)
if cols_to_drop_test:
    df_test_final = df_test_final.drop(columns=cols_to_drop_test)

# Kiểm tra kết quả
print("\n--- df_train_final sau khi dọn dẹp ---")
print(df_train_final.head())
print("\nThông tin df_train_final:")
print(df_train_final.info())

print("\n--- df_test_final sau khi dọn dẹp ---")
print(df_test_final.head())
print("\nThông tin df_test_final:")
print(df_test_final.info())


if 'Customers' in df_train_final.columns:
    df_train_final = df_train_final.drop(columns=['Customers'])
    print("\nĐã loại bỏ cột 'Customers' khỏi df_train_final.")


print("\nNaNs cuối cùng trong train:", df_train_final.isnull().sum().sum())
print("NaNs cuối cùng trong test:", df_test_final.isnull().sum().sum())


# Tách những ngày cửa hàng đóng cửa ra khỏi tập huấn luyện
# Chúng ta vẫn giữ những ngày Sales = 0 nhưng Open = 1 (nếu có, rất hiếm và có thể là lỗi dữ liệu)
# nhưng chủ yếu là Open = 0 -> Sales = 0

# Trước khi lọc, hãy đảm bảo không có trường hợp Open=0 mà Sales != 0
print(f"Số dòng Open=0, Sales!=0 trong train_final: {len(df_train_final[(df_train_final['Open'] == 0) & (df_train_final['Sales'] != 0)])}")
# Nếu có, đây là điểm bất thường. Giả sử là không có hoặc rất ít.

# Lọc những ngày cửa hàng thực sự mở cửa để huấn luyện mô hình
df_train_opened = df_train_final[df_train_final['Open'] == 1].copy()

# Kiểm tra xem còn Sales = 0 khi Open = 1 không
print(f"Số dòng Open=1, Sales=0 trong df_train_opened: {len(df_train_opened[df_train_opened['Sales'] == 0])}")
# Nếu vẫn còn Sales = 0 khi Open = 1, RMSPE sẽ bị lỗi (chia cho 0).
# Chúng ta cần loại bỏ những điểm này khỏi tập huấn luyện VÀ validation khi tính RMSPE.
# Hoặc, khi tính RMSPE, chỉ lấy y_true > 0.
# Để an toàn, khi huấn luyện mô hình dự đoán log(Sales+1), chúng ta chỉ nên dùng các hàng Sales > 0.
df_train_opened_sales_positive = df_train_opened[df_train_opened['Sales'] > 0].copy()

print(f"Kích thước df_train_final ban đầu: {df_train_final.shape}")
print(f"Kích thước df_train_opened (Open=1): {df_train_opened.shape}")
print(f"Kích thước df_train_opened_sales_positive (Open=1, Sales>0): {df_train_opened_sales_positive.shape}")

# Áp dụng log transformation cho Sales
df_train_opened_sales_positive['SalesLog'] = np.log1p(df_train_opened_sales_positive['Sales'])

# Biến mục tiêu của chúng ta bây giờ là 'SalesLog'
# Các đặc trưng là các cột còn lại (trừ 'Sales', 'Date', và có thể 'Id' nếu có)


# Chọn các đặc trưng cho mô hình
# Loại bỏ các cột không phải là đặc trưng hoặc là biến mục tiêu gốc
# Cột 'Date' có thể hữu ích cho việc sắp xếp hoặc time-series split, nhưng không phải là feature dạng datetime cho LightGBM.
# Các thành phần của Date (Year, Month, Day, WeekOfYear, etc.) đã được tạo.

# Loại bỏ 'Id' nếu có trong tập train (thường không có)
# Loại bỏ 'Store' nếu không muốn dùng Store ID làm categorical feature trực tiếp (LGBM có thể xử lý)
# Tuy nhiên, thường thì Store ID được dùng.

features = [col for col in df_train_opened_sales_positive.columns if col not in ['Sales', 'SalesLog', 'Date', 'Id']]
# 'Id' không có trong train, 'Store' nên được giữ lại làm feature.

# Kiểm tra lại danh sách features
print("\nDanh sách các đặc trưng được sử dụng:")
# print(features)
if 'Store' not in features: # Đảm bảo 'Store' là một feature nếu chưa có dạng số
     print("Cảnh báo: 'Store' ID không có trong features. Cân nhắc đưa vào nếu nó quan trọng.")

# Các cột cần đảm bảo là kiểu phù hợp cho LightGBM (ví dụ, int, float, hoặc category)
# LightGBM có thể tự xử lý 'Store' ID như một biến phân loại nếu khai báo.
# df_train_opened_sales_positive['Store'] = df_train_opened_sales_positive['Store'].astype('category')
# df_test_final['Store'] = df_test_final['Store'].astype('category') # Cần làm điều này cho cả test set

X_train = df_train_opened_sales_positive[features]
y_train_log = df_train_opened_sales_positive['SalesLog']

# Đối với tập test, chúng ta cũng cần chuẩn bị các features tương tự
X_test = df_test_final[features] # Đảm bảo df_test_final có cùng các cột features

# Kiểm tra sự nhất quán về cột giữa X_train và X_test
train_cols = set(X_train.columns)
test_cols = set(X_test.columns)

if train_cols != test_cols:
    print("\nCẢNH BÁO: Tập hợp cột trong X_train và X_test không giống nhau!")
    print(f"Các cột chỉ có trong X_train: {train_cols - test_cols}")
    print(f"Các cột chỉ có trong X_test: {test_cols - train_cols}")
    # Đây là vấn đề nghiêm trọng cần sửa. Thường do lỗi ở bước Feature Engineering hoặc lúc chọn 'features'.
    # Ví dụ, nếu 'Open' vẫn còn trong features của X_train nhưng đã bị loại hoặc không có trong X_test theo logic trên.
    # Trong trường hợp này, 'Open' trong X_train (sau khi lọc df_train_opened_sales_positive) sẽ luôn là 1.
    # Cột 'Open' trong X_test thì có thể là 0 hoặc 1.
    # Chúng ta nên loại bỏ 'Open' khỏi danh sách features vì nó đã được dùng để lọc dữ liệu.
    if 'Open' in features:
        features.remove('Open')
        X_train = df_train_opened_sales_positive[features]
        X_test = df_test_final[features] # Cập nhật lại X_test
        print("Đã loại bỏ 'Open' khỏi features và cập nhật X_train, X_test.")


print(f"\nKích thước X_train: {X_train.shape}")
print(f"Kích thước y_train_log: {y_train_log.shape}")
print(f"Kích thước X_test: {X_test.shape}")


# Convert Store column to category type without warnings
X_train = X_train.assign(Store=X_train['Store'].astype('category'))
X_test = X_test.assign(Store=X_test['Store'].astype('category'))

# Check the first few rows
print("\nX_train head:")
print(X_train.head())
print("\ny_train_log head:")
print(y_train_log.head())


import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

# Định nghĩa RMSPE
def rmspe(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    valid_indices = y_true > 0
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]
    if len(y_true_valid) == 0:
        return 0.0
    percentage_error = (y_true_valid - y_pred_valid) / y_true_valid
    return np.sqrt(np.mean(np.square(percentage_error)))

# Custom metric cho LightGBM sklearn API
def lgbm_rmspe_metric_sklearn(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return ('rmspe', rmspe(y_true, y_pred), False)

# Sắp xếp dữ liệu theo Date
temp_train_df_for_sorting = df_train_opened_sales_positive[['Date']].copy().reset_index(drop=True)
X_train_cv = X_train.reset_index(drop=True)
y_train_log_cv = y_train_log.reset_index(drop=True)
sorted_indices = temp_train_df_for_sorting.sort_values(by=['Date']).index

X_train_sorted = X_train_cv.iloc[sorted_indices]
y_train_log_sorted = y_train_log_cv.iloc[sorted_indices]

# Categorical features cho LightGBM
categorical_features_lgbm = [
    'Store', 'DayOfWeek', 'Month', 'Year', 'StateHolidayNumeric',
    'StoreTypeNumeric', 'AssortmentNumeric', 'IsWeekend',
    'Promo', 'SchoolHoliday'
]
categorical_features_lgbm = [col for col in categorical_features_lgbm if col in X_train_sorted.columns]

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)

# Khởi tạo biến lưu kết quả
oof_preds_log = np.zeros(X_train_sorted.shape[0])
models = []
fold_rmspe_scores = []

# # Huấn luyện theo fold
# print("\n--- Bắt đầu Cross-Validation với LightGBM ---")
# for fold_idx, (train_index, val_index) in enumerate(tscv.split(X_train_sorted)):
#     print(f"\n--- Fold {fold_idx+1} ---")
#     X_train_fold, X_val_fold = X_train_sorted.iloc[train_index], X_train_sorted.iloc[val_index]
#     y_train_fold_log, y_val_fold_log = y_train_log_sorted.iloc[train_index], y_train_log_sorted.iloc[val_index]

#     print(f"Train shape: {X_train_fold.shape}, Validation shape: {X_val_fold.shape}")

#     lgb_params = {
#         'objective': 'regression_l1',
#         'metric': 'None',
#         'n_estimators': 2000,
#         'learning_rate': 0.05,
#         'feature_fraction': 0.8,
#         'bagging_fraction': 0.8,
#         'bagging_freq': 1,
#         'num_leaves': 31,
#         'verbose': -1,
#         'n_jobs': -1,
#         'seed': 42 + fold_idx,
#         'boosting_type': 'gbdt'
#     }

#     model = lgb.LGBMRegressor(**lgb_params)
#     model.fit(
#         X_train_fold, y_train_fold_log,
#         eval_set=[(X_val_fold, y_val_fold_log)],
#         eval_metric=lgbm_rmspe_metric_sklearn,
#         callbacks=[lgb.early_stopping(100, verbose=True)],
#         categorical_feature=categorical_features_lgbm
#     )

#     val_preds_log = model.predict(X_val_fold)
#     oof_preds_log[val_index] = val_preds_log

#     y_val_true_original = np.expm1(y_val_fold_log)
#     val_preds_original = np.expm1(val_preds_log)
#     fold_score = rmspe(y_val_true_original, val_preds_original)
#     fold_rmspe_scores.append(fold_score)
#     print(f"Fold {fold_idx+1} RMSPE: {fold_score:.4f}")

#     models.append(model)

# # Đánh giá tổng thể
# print(f"\nRMSPE trung bình qua các fold: {np.mean(fold_rmspe_scores):.4f} (+/- {np.std(fold_rmspe_scores):.4f})")
# oof_rmspe_score = rmspe(np.expm1(y_train_log_sorted), np.expm1(oof_preds_log[oof_preds_log != 0]))
# print(f"RMSPE trên toàn bộ OOF predictions: {oof_rmspe_score:.4f}")



# # Chỉ tính RMSPE trên các mẫu mà OOF prediction đã được thực hiện (tức là oof_preds_log != 0)
# oof_made_prediction_indices = oof_preds_log != 0

# y_true_for_oof_eval_log = y_train_log_sorted[oof_made_prediction_indices]
# y_pred_for_oof_eval_log = oof_preds_log[oof_made_prediction_indices]

# # Chuyển đổi về thang đo gốc
# y_true_for_oof_eval = np.expm1(y_true_for_oof_eval_log)
# y_pred_for_oof_eval = np.expm1(y_pred_for_oof_eval_log)

# # Tính RMSPE trên các giá trị OOF đã được lọc
# if len(y_true_for_oof_eval) > 0: # Đảm bảo có dữ liệu để đánh giá
#     oof_rmspe_score = rmspe(y_true_for_oof_eval, y_pred_for_oof_eval)
#     print(f"RMSPE trên toàn bộ OOF predictions (đã sửa): {oof_rmspe_score:.4f}")
# else:
#     print("Không có dự đoán OOF nào được thực hiện để tính điểm tổng thể.")
print("XOng")


# import optuna
# # Optuna sẽ bị lỗi nếu chạy trong notebook Kaggle khi submit, nhưng dùng được khi interactive.
# # Nếu không dùng Optuna, có thể dùng GridSearchCV hoặc RandomizedSearchCV, hoặc tinh chỉnh thủ công.

# # Hàm objective cho Optuna
# def objective(trial):
#     params = {
#         'objective': 'regression_l1',
#         'metric': 'None', # Sẽ dùng feval thông qua eval_metric của sklearn wrapper
#         'n_estimators': trial.suggest_int('n_estimators', 500, 3000, step=100),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 150),
#         'max_depth': trial.suggest_int('max_depth', 3, 12),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#         'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
#         'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
#         'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
#         'verbose': -1,
#         'n_jobs': -1,
#         'seed': 42, # Giữ seed cố định để có thể tái tạo kết quả cho cùng bộ params
#         'boosting_type': 'gbdt',
#     }

#     current_fold_scores = []
#     # temp_oof_preds_log = np.zeros(X_train_sorted.shape[0]) # Không cần thiết cho việc trả về score của trial

#     # Sử dụng TimeSeriesSplit đã định nghĩa (ví dụ: tscv)
#     for fold_idx, (train_index, val_index) in enumerate(tscv.split(X_train_sorted)):
#         X_train_fold, X_val_fold = X_train_sorted.iloc[train_index], X_train_sorted.iloc[val_index]
#         y_train_fold_log, y_val_fold_log = y_train_log_sorted.iloc[train_index], y_train_log_sorted.iloc[val_index]

#         model = lgb.LGBMRegressor(**params)
#         model.fit(X_train_fold, y_train_fold_log,
#                   eval_set=[(X_val_fold, y_val_fold_log)],
#                   eval_metric=lgbm_rmspe_metric_sklearn, # <<< SỬA LỖI Ở ĐÂY
#                   callbacks=[lgb.early_stopping(50, verbose=False)],
#                   categorical_feature=categorical_features_lgbm)

#         val_preds_log = model.predict(X_val_fold)
#         # temp_oof_preds_log[val_index] = val_preds_log # Không cần thiết cho việc trả về score của trial

#         y_val_true_original = np.expm1(y_val_fold_log)
#         val_preds_original = np.expm1(val_preds_log)
#         score = rmspe(y_val_true_original, val_preds_original)
#         current_fold_scores.append(score)
#         # Dừng sớm nếu điểm số quá tệ ở fold đầu tiên để tiết kiệm thời gian (tùy chọn)
#         # if fold_idx == 0 and score > 0.3: # Ví dụ ngưỡng
#         #     return score # Trả về điểm tệ để Optuna không khám phá nhánh này nữa

#     avg_rmspe = np.mean(current_fold_scores)
#     # print(f"Trial {trial.number} - Avg RMSPE: {avg_rmspe:.5f} with params: {trial.params}") # Bỏ print để Optuna chạy nhanh hơn
#     return avg_rmspe

# # # --- Phần này giả định các biến sau đã được định nghĩa ---
# # # df_train_opened_sales_positive = ... (DataFrame chứa cột 'Date', 'Sales', etc.)
# # # X_train = ... (DataFrame các features trước khi sắp xếp)
# # # y_train_log = ... (Series log(Sales) trước khi sắp xếp)

# # # # Sắp xếp dữ liệu (nếu chưa làm ở global scope)
# # # temp_train_df_for_sorting = df_train_opened_sales_positive[['Date']].copy().reset_index(drop=True)
# # # X_train_cv = X_train.reset_index(drop=True)
# # # y_train_log_cv = y_train_log.reset_index(drop=True)
# # # sorted_indices = temp_train_df_for_sorting.sort_values(by=['Date']).index
# # # X_train_sorted = X_train_cv.iloc[sorted_indices]
# # # y_train_log_sorted = y_train_log_cv.iloc[sorted_indices]

# # # # Categorical features (nếu chưa làm ở global scope)
# # # categorical_features_lgbm = [
# # #     'Store', 'DayOfWeek', 'Month', 'Year', 'StateHolidayNumeric',
# # #     'StoreTypeNumeric', 'AssortmentNumeric', 'IsWeekend',
# # #     'Promo', 'SchoolHoliday'
# # # ]
# # # categorical_features_lgbm = [col for col in categorical_features_lgbm if col in X_train_sorted.columns]

# # # # TimeSeriesSplit (nếu chưa làm ở global scope)
# # # tscv = TimeSeriesSplit(n_splits=3)
# # # --- Hết phần giả định ---


# # Chạy Optuna (có thể tốn thời gian)
# # Đảm bảo X_train_sorted, y_train_log_sorted, tscv, categorical_features_lgbm đã được định nghĩa
# # ở global scope hoặc được truyền vào/truy cập đúng cách.

# print("Bắt đầu tối ưu hóa Hyperparameters với Optuna...")
# study_name = 'rossmann-lgbm-optimization-v2' # Đổi tên để không ghi đè study cũ nếu cần
# storage_name = f"sqlite:///{study_name}.db"

# study = optuna.create_study(
#     direction='minimize',
#     study_name=study_name,
#     storage=storage_name,
#     load_if_exists=True
# )

# # Giới hạn thời gian hoặc số trials
# # Để chạy thử nhanh, giảm n_trials
# study.optimize(objective, n_trials=10, timeout=60*30) # Ví dụ: 10 trials hoặc 30 phút

# print("\n--- Kết quả tối ưu hóa Hyperparameters ---")
# if study.best_trial:
#     print(f"Best trial number: {study.best_trial.number}")
#     print(f"Best RMSPE: {study.best_value:.5f}")
#     print("Best hyperparameters:")
#     for key, value in study.best_params.items():
#         print(f"  {key}: {value}")

#     best_lgbm_params = study.best_params.copy() # Tạo bản sao
#     best_lgbm_params['objective'] = 'regression_l1'
#     best_lgbm_params['metric'] = 'None'
#     best_lgbm_params['verbose'] = -1
#     best_lgbm_params['n_jobs'] = -1
#     best_lgbm_params['seed'] = 42
#     best_lgbm_params['boosting_type'] = 'gbdt'
#     # best_lgbm_params['n_estimators'] đã có từ study.best_params, không cần gán lại trừ khi muốn cố định
# else:
#     print("Optuna không tìm thấy trial nào tốt nhất (có thể do timeout hoặc lỗi sớm).")
#     best_lgbm_params = None # Hoặc một bộ params mặc định

# # Sử dụng best_lgbm_params để huấn luyện mô hình cuối cùng hoặc tiếp tục CV
# # Ví dụ:
# # if best_lgbm_params:
# #     final_model = lgb.LGBMRegressor(**best_lgbm_params)
# #     # Huấn luyện trên toàn bộ X_train_sorted, y_train_log_sorted
# #     # Hoặc huấn luyện lại theo từng fold với params tốt nhất để có các model cho ensemble


import xgboost as xgb
import optuna # Đã import trước đó
import numpy as np # Đã import trước đó
import pandas as pd # Đã import trước đó

# Giả sử các biến dữ liệu X_train_cv, y_train_cv, X_val_cv, y_val_cv
# và hàm rmspe vẫn được định nghĩa từ các bước trước.
# Kiểm tra lại sự tồn tại của chúng:
try:
    X_train_cv.shape
    y_train_cv.shape
    X_val_cv.shape
    y_val_cv.shape
    callable(rmspe)
    print("Các tập dữ liệu CV và hàm rmspe đã sẵn sàng cho XGBoost.")
except NameError:
    print("Lỗi: Một hoặc nhiều biến/hàm cần thiết chưa được định nghĩa.")
    print("Hãy chắc chắn bạn đã chạy các bước chuẩn bị dữ liệu và định nghĩa hàm rmspe.")
    exit()



# # Hàm rmspe gốc của bạn:
# # def rmspe(y_true_orig, y_pred_orig): ... (đã định nghĩa)

# # Hàm đánh giá cho XGBoost (tương tự lgbm_rmspe_eval nhưng không có is_higher_better)
# def xgb_rmspe_eval(y_true_log, y_pred_log):
#     y_true_orig = np.expm1(y_true_log)
#     y_pred_orig = np.expm1(y_pred_log)
#     y_pred_orig[y_pred_orig < 0] = 0 # Xử lý giá trị âm
#     score = rmspe(y_true_orig, y_pred_orig)
#     return 'rmspe', score # XGBoost chỉ cần (name, value)


# def objective_xgb(trial):
#     params_optuna_xgb = {
#         'objective': 'reg:squarederror', # Mục tiêu cho hồi quy, lỗi bình phương
#                                          # XGBoost sẽ tối ưu MSE trên dữ liệu log-transformed
#         'eval_metric': 'rmse',           # Theo dõi RMSE trên dữ liệu log-transformed trong quá trình train
#                                          # Chúng ta sẽ tính RMSPE cuối cùng cho Optuna
#         'booster': 'gbtree',
#         'verbosity': 0, # Tắt log của XGBoost
#         'nthread': -1,  # Sử dụng tất cả các core
#         'seed': 42,
#         'tree_method': 'hist', # Sử dụng 'hist' để tăng tốc độ, tương tự LightGBM

#         # Siêu tham số để Optuna tối ưu
#         'eta': trial.suggest_float('eta', 0.005, 0.05, log=True), # Tương tự learning_rate
#         'max_depth': trial.suggest_int('max_depth', 3, 12),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 100), # Khác với min_child_samples của LGBM
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0), # Tỷ lệ mẫu cho mỗi cây
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), # Tỷ lệ features cho mỗi cây
#         'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True), # Min split loss
#         'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True), # L1 regularization (reg_alpha)
#         'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True), # L2 regularization (reg_lambda)

#         'n_estimators': 4000, # Đặt giá trị lớn, early stopping sẽ quyết định
#     }

#     model_xgb_opt = xgb.XGBRegressor(**params_optuna_xgb)

#     model_xgb_opt.fit(X_train_cv, y_train_cv,
#                       eval_set=[(X_val_cv, y_val_cv)],
#                       early_stopping_rounds=100, # Dừng nếu không cải thiện sau 100 vòng (theo dõi eval_metric)
#                       verbose=False) # Tắt log trong quá trình fit của từng trial

#     preds_val_log = model_xgb_opt.predict(X_val_cv)
#     preds_val_orig = np.expm1(preds_val_log)
#     y_val_orig = np.expm1(y_val_cv)
#     preds_val_orig[preds_val_orig < 0] = 0

#     # Tính RMSPE thực tế để Optuna tối ưu
#     # Mặc dù eval_metric trong params là 'rmse', chúng ta trả về RMSPE cho Optuna
#     rmspe_score = rmspe(y_val_orig, preds_val_orig)

#     return rmspe_score


# print("\n--- Bắt đầu Tối ưu hóa Siêu tham số cho XGBoost với Optuna ---")
# study_xgb = optuna.create_study(direction='minimize', study_name='Rossmann_XGB_Optimization')

# try:
#     study_xgb.optimize(objective_xgb, n_trials=30, timeout=3600) # Ví dụ: 30 trials hoặc tối đa 1 giờ
# except KeyboardInterrupt:
#     print("Tối ưu hóa XGBoost bị ngắt bởi người dùng.")
# except Exception as e:
#     print(f"Có lỗi xảy ra trong quá trình tối ưu hóa XGBoost: {e}")

# print("\n--- Tối ưu hóa XGBoost Hoàn tất (hoặc bị ngắt) ---")


# if study_xgb.trials:
#     print(f"Số lượng trials XGBoost đã hoàn thành: {len(study_xgb.trials)}")
#     best_trial_xgb = study_xgb.best_trial
#     print("Best XGBoost trial:")
#     print(f"  Value (RMSPE): {best_trial_xgb.value:.5f}")
#     print("  Params: ")
#     for key, value in best_trial_xgb.params.items():
#         print(f"    {key}: {value}")
#     best_params_xgb_optuna = best_trial_xgb.params
#     # Thêm lại các params cố định nếu chúng không được suggest trong trial
#     best_params_xgb_optuna['objective'] = 'reg:squarederror'
#     best_params_xgb_optuna['booster'] = 'gbtree'
#     best_params_xgb_optuna['verbosity'] = 0
#     best_params_xgb_optuna['nthread'] = -1
#     best_params_xgb_optuna['seed'] = 42
#     best_params_xgb_optuna['tree_method'] = 'hist'
# else:
#     print("Không có trial XGBoost nào được hoàn thành.")
#     best_params_xgb_optuna = None

