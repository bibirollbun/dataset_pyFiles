
import pandas as pd
import numpy as np
import dask.dataframe as dd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn import preprocessing, metrics
import gc
import joblib
import warnings
warnings.filterwarnings('ignore')


INPUT_DIR_PATH = '../input/m5-forecasting-accuracy/'


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics: 
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def read_data():
    sell_prices_df = pd.read_csv(INPUT_DIR_PATH + 'sell_prices.csv')
    sell_prices_df = reduce_mem_usage(sell_prices_df)
    print('Sell prices has {} rows and {} columns'.format(sell_prices_df.shape[0], sell_prices_df.shape[1]))

    calendar_df = pd.read_csv(INPUT_DIR_PATH + 'calendar.csv')
    calendar_df = reduce_mem_usage(calendar_df)
    print('Calendar has {} rows and {} columns'.format(calendar_df.shape[0], calendar_df.shape[1]))

    sales_train_validation_df = pd.read_csv(INPUT_DIR_PATH + 'sales_train_validation.csv')
    print('Sales train validation has {} rows and {} columns'.format(sales_train_validation_df.shape[0], sales_train_validation_df.shape[1]))

    submission_df = pd.read_csv(INPUT_DIR_PATH + 'sample_submission.csv')
    return sell_prices_df, calendar_df, sales_train_validation_df, submission_df
    


sell_prices_df, calendar_df, sales_train_validation_df, submission_df = read_data()


NUM_ITEMS = sales_train_validation_df.shape[0]  # 30490
DAYS_PRED = 28
nrows = 365 * 2 * NUM_ITEMS


def encode_categorical(df, cols):
    for col in cols:
        # Leave NaN as it is.
        le = preprocessing.LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)

    return df


calendar_df = encode_categorical(calendar_df, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]).pipe(reduce_mem_usage)
sales_train_validation_df = encode_categorical(sales_train_validation_df, ["item_id", "dept_id", "cat_id", "store_id", "state_id"]).pipe(reduce_mem_usage)
sell_prices_df = encode_categorical(sell_prices_df, ["item_id", "store_id"]).pipe(reduce_mem_usage)


# # function to read the data and merge it

# def melt_and_merge(calendar, sell_prices, sales_train_validation, submission, nrows = 55000000, merge = False):
    
#     # melt sales data, get it ready for training
#     sales_train_validation = pd.melt(sales_train_validation, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
#     print('Melted sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
#     sales_train_validation = reduce_mem_usage(sales_train_validation)
    
#     sales_train_validation = sales_train_validation.iloc[-nrows:,:]
    
    
#     # seperate test dataframes
#     test1_rows = [row for row in submission['id'] if 'validation' in row]
#     test2_rows = [row for row in submission['id'] if 'evaluation' in row]
#     test1 = submission[submission['id'].isin(test1_rows)]
#     test2 = submission[submission['id'].isin(test2_rows)]
    
#     # change column names
#     test1.columns = ['id', 'd_1914', 'd_1915', 'd_1916', 'd_1917', 'd_1918', 'd_1919', 'd_1920', 'd_1921', 'd_1922', 'd_1923', 'd_1924', 'd_1925', 'd_1926', 'd_1927', 'd_1928', 'd_1929', 'd_1930', 'd_1931', 
#                       'd_1932', 'd_1933', 'd_1934', 'd_1935', 'd_1936', 'd_1937', 'd_1938', 'd_1939', 'd_1940', 'd_1941']
#     test2.columns = ['id', 'd_1942', 'd_1943', 'd_1944', 'd_1945', 'd_1946', 'd_1947', 'd_1948', 'd_1949', 'd_1950', 'd_1951', 'd_1952', 'd_1953', 'd_1954', 'd_1955', 'd_1956', 'd_1957', 'd_1958', 'd_1959', 
#                       'd_1960', 'd_1961', 'd_1962', 'd_1963', 'd_1964', 'd_1965', 'd_1966', 'd_1967', 'd_1968', 'd_1969']
    
#     # get product table
#     product = sales_train_validation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()
    
#     # merge with product table
#     test2['id'] = test2['id'].str.replace('_evaluation','_validation')
#     test1 = test1.merge(product, how = 'left', on = 'id')
#     test2 = test2.merge(product, how = 'left', on = 'id')
#     test2['id'] = test2['id'].str.replace('_validation','_evaluation')
    
#     # 
#     test1 = pd.melt(test1, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
#     test2 = pd.melt(test2, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
    
#     sales_train_validation['part'] = 'train'
#     test1['part'] = 'test1'
#     test2['part'] = 'test2'
    
#     data = pd.concat([sales_train_validation, test1, test2], axis = 0)
    
#     del sales_train_validation, test1, test2
    
#     print(data.shape)
    
#     # get only a sample for fst training
# #     data = data.loc[nrows:]
    
#     # drop some calendar features
#     calendar.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1)
    
#     # delete test2 for now
#     data = data[data['part'] != 'test2']
    
#     if merge:
#         # notebook crash with the entire dataset (maybee use tensorflow, dask, pyspark xD)
#         data = pd.merge(data, calendar, how = 'left', left_on = ['day'], right_on = ['d'])
#         data.drop(['d', 'day'], inplace = True, axis = 1)
#         # get the sell price data (this feature should be very important)
#         data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
#         print('Our final dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1]))
#     else: 
#         pass
    
#     gc.collect()
    
#     return data


def melt_and_merge(calendar, sell_prices, sales_train_validation, submission, nrows = 55000000, merge = False):
    
    # melt sales data, get it ready for training
    sales_train_validation = pd.melt(sales_train_validation, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
    print('Melted sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
    sales_train_validation = reduce_mem_usage(sales_train_validation)
    
    sales_train_validation = sales_train_validation.iloc[-nrows:,:]
    
    
    # seperate test dataframes
    test1_rows = [row for row in submission['id'] if 'validation' in row]
    test2_rows = [row for row in submission['id'] if 'evaluation' in row]
    test1 = submission[submission['id'].isin(test1_rows)]
    test2 = submission[submission['id'].isin(test2_rows)]
    
    # change column names
    test1.columns = ['id', 'd_1914', 'd_1915', 'd_1916', 'd_1917', 'd_1918', 'd_1919', 'd_1920', 'd_1921', 'd_1922', 'd_1923', 'd_1924', 'd_1925', 'd_1926', 'd_1927', 'd_1928', 'd_1929', 'd_1930', 'd_1931', 
                     'd_1932', 'd_1933', 'd_1934', 'd_1935', 'd_1936', 'd_1937', 'd_1938', 'd_1939', 'd_1940', 'd_1941']
    test2.columns = ['id', 'd_1942', 'd_1943', 'd_1944', 'd_1945', 'd_1946', 'd_1947', 'd_1948', 'd_1949', 'd_1950', 'd_1951', 'd_1952', 'd_1953', 'd_1954', 'd_1955', 'd_1956', 'd_1957', 'd_1958', 'd_1959', 
                     'd_1960', 'd_1961', 'd_1962', 'd_1963', 'd_1964', 'd_1965', 'd_1966', 'd_1967', 'd_1968', 'd_1969']
    
    # get product table
    product = sales_train_validation[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()
    
    # merge with product table
    test2['id'] = test2['id'].str.replace('_evaluation','_validation')
    test1 = test1.merge(product, how = 'left', on = 'id')
    test2 = test2.merge(product, how = 'left', on = 'id')
    test2['id'] = test2['id'].str.replace('_validation','_evaluation')
    
    # 
    test1 = pd.melt(test1, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
    test2 = pd.melt(test2, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
    
    sales_train_validation['part'] = 'train'
    test1['part'] = 'test1'
    test2['part'] = 'test2'
    
    data = pd.concat([sales_train_validation, test1, test2], axis = 0)
    
    del sales_train_validation, test1, test2
    
    print(data.shape)
    
    # get only a sample for fst training
    # data = data.loc[nrows:]
    
    # === SỬA ĐỔI QUAN TRỌNG TẠI ĐÂY ===
    # Giữ lại cột 'date' từ calendar và đổi tên 'd' thành 'day' (nếu cần)
    # calendar.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1) # BỎ COMMENT DÒNG NÀY NẾU BẠN MUỐN LOẠI BỎ CÁC CỘT ĐÓ SAU KHI MERGE
    
    # delete test2 for now
    data = data[data['part'] != 'test2']
    
    if merge:
        # Merge với calendar_df để có các đặc trưng thời gian và 'date'
        # Đảm bảo cột 'date' từ calendar_df được giữ lại
        # calendar_df có các cột: 'date', 'wm_yr_wk', 'weekday', 'wday', 'event_name_1', ...
        # Bạn cần merge trên 'day' (cột trong data) và 'd' (cột trong calendar)
        data = pd.merge(data, calendar, how = 'left', left_on = ['day'], right_on = ['d'])
        
        # BÂY GIỜ CHỈ DROP CỘT 'd' THAY VÌ CẢ 'd' VÀ 'day'
        # Bởi vì 'day' là cột để merge, bạn cần giữ lại 'date' từ calendar
        data.drop(['d'], inplace = True, axis = 1) # Chỉ drop cột 'd'
        
        # Đổi tên cột 'date' (từ calendar_df) thành 'date' nếu cần.
        # Nếu calendar_df đã có cột 'date', nó sẽ tự động được merge.
        # Nếu không, hãy đảm bảo rằng bạn đã xử lý cột 'd' của calendar thành 'date'
        # trước khi merge hoặc sau khi merge.

        # get the sell price data (this feature should be very important)
        data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
        print('Our final dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1]))
    else:  
        pass
    
    gc.collect()
    
    return data


# nrows = 365 * 2 * NUM_ITEMS

# nrows = 27500000
nrows = 15000000
data = melt_and_merge(calendar_df, sell_prices_df, sales_train_validation_df, submission_df, nrows = nrows, merge = True)
# nrows = 27500000


# def transform(data):
#     nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
#     for feature in nan_features:
#         data[feature].fillna('unknown', inplace = True)
        
#     cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
#     for feature in cat:
#         encoder = preprocessing.LabelEncoder()
#         data[feature] = encoder.fit_transform(data[feature])
#     return data


# def simple_fe(data):
    
#     # rolling demand features
    
#     for val in [28, 29, 30]:
#         data[f"shift_t{val}"] = data.groupby(["id"])["demand"].transform(lambda x: x.shift(val))
#     for val in [7, 30, 60, 90, 180]:
#         data[f"rolling_std_t{val}"] = data.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(val).std())
#     for val in [7, 30, 60, 90, 180]:
#         data[f"rolling_mean_t{val}"] = data.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(val).mean())

#     data["rolling_skew_t30"] = data.groupby(["id"])["demand"].transform( lambda x: x.shift(28).rolling(30).skew())
#     data["rolling_kurt_t30"] = data.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(30).kurt())
    
#     # price features
#     data['lag_price_t1'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))
#     data['price_change_t1'] = (data['lag_price_t1'] - data['sell_price']) / (data['lag_price_t1'])
#     data['rolling_price_max_t365'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1).rolling(365).max())
#     data['price_change_t365'] = (data['rolling_price_max_t365'] - data['sell_price']) / (data['rolling_price_max_t365'])
#     data['rolling_price_std_t7'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(7).std())
#     data['rolling_price_std_t30'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(30).std())
#     data.drop(['rolling_price_max_t365', 'lag_price_t1'], inplace = True, axis = 1)
    
# #     # time features
#     data['date'] = pd.to_datetime(data['date'])
#     attrs = ["year", "quarter", "month", "week", "day", "dayofweek", "is_year_end", "is_year_start", "is_quarter_end", \
#         "is_quarter_start", "is_month_end","is_month_start",
#     ]

#     for attr in attrs:
#         dtype = np.int16 if attr == "year" else np.int8
#         data[attr] = getattr(data['date'].dt, attr).astype(dtype)
#     data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(np.int8)
    
#     return data




# def transform(data):
#     nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
#     for feature in nan_features:
#         data[feature].fillna('unknown', inplace = True)
        
#     cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
#     for feature in cat:
#         encoder = preprocessing.LabelEncoder()
#         data[feature] = encoder.fit_transform(data[feature])
#     return data
def transform(data):
    # nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    # for feature in nan_features:
    #     # Nếu các cột này đã được LabelEncode và có thể chứa NaN (kiểu float)
    #     # LightGBM có thể xử lý NaN.
    #     # Nếu bạn muốn điền, hãy điền bằng một số (ví dụ -1)
    #     # data[feature].fillna(-1, inplace = True) # Hoặc giá trị khác phù hợp
    #     pass # Không làm gì ở đây, để LightGBM xử lý NaN hoặc đã xử lý ở encode_categorical

    cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for feature in cat:
        # Các cột này đã được encode_categorical ở cấp độ DataFrame gốc.
        # Khi merge vào 'data', chúng đã là số nguyên.
        # Do đó, việc gọi LabelEncoder ở đây lần nữa là không cần thiết
        # và có thể gây ra vấn đề nếu có giá trị mới (ví dụ 'unknown' từ fillna trước đó).
        # LightGBM có thể xử lý các số nguyên đã được LabelEncode.
        pass # Không làm gì ở đây
    return data

# --- Hàm simple_fe (thêm nhiều đặc trưng hơn) ---
def simple_fe(data):
    
    # Đảm bảo cột 'date' là datetime trước khi tạo time features
    data['date'] = pd.to_datetime(data['date'])

    # === Các đặc trưng demand (Lagging và Rolling) ===
    # Thêm nhiều giá trị shift
    for val in [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]: # Các lag gần hơn
        data[f"shift_t{val}"] = data.groupby(["id"])["demand"].transform(lambda x: x.shift(val))
    
    # Thêm các lag theo năm
    # M5 data có 365 ngày mỗi năm, nên 365 +/- 1, 2 để bắt các hiệu ứng cuối tuần/lễ lân cận
    for val in [365, 366, 367]:
        data[f"shift_t{val}"] = data.groupby(["id"])["demand"].transform(lambda x: x.shift(val))

    # Thêm nhiều giá trị rolling std/mean với các cửa sổ khác nhau
    for val in [7, 14, 28, 30, 60, 90, 180, 365]: # Thêm 14, 28, 365
        data[f"rolling_std_t{val}"] = data.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(val).std())
        data[f"rolling_mean_t{val}"] = data.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(val).mean())

    # Thêm các tính năng khác cho rolling stats
    for val in [7, 14, 28, 30, 60, 90, 180, 365]:
        data[f"rolling_min_t{val}"] = data.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(val).min())
        data[f"rolling_max_t{val}"] = data.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(val).max())
        # Tạo thêm tính năng range (max-min)
        data[f"rolling_range_t{val}"] = data[f"rolling_max_t{val}"] - data[f"rolling_min_t{val}"]

    data["rolling_skew_t30"] = data.groupby(["id"])["demand"].transform( lambda x: x.shift(28).rolling(30).skew())
    data["rolling_kurt_t30"] = data.groupby(["id"])["demand"].transform(lambda x: x.shift(28).rolling(30).kurt())
    
    # Bạn có thể thêm các tính năng theo tỷ lệ (ratio)
    # data[f"rolling_mean_t30_ratio"] = data[f"rolling_mean_t30"] / data[f"shift_t28"]

    # === Các đặc trưng giá (Price Features) ===
    data['lag_price_t1'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))
    data['price_change_t1'] = (data['lag_price_t1'] - data['sell_price']) / (data['lag_price_t1'])
    
    # Thêm các lag giá khác (nếu có ý nghĩa, ví dụ: lag giá theo tuần/tháng)
    for val in [7, 28, 30]:
        data[f'lag_price_t{val}'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(val))

    data['rolling_price_max_t365'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1).rolling(365).max())
    data['price_change_t365'] = (data['rolling_price_max_t365'] - data['sell_price']) / (data['rolling_price_max_t365'])
    
    # Thêm các cửa sổ rolling cho price std/mean
    for val in [7, 14, 30, 60]:
        data[f"rolling_price_std_t{val}"] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(val).std())
        data[f"rolling_price_mean_t{val}"] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(val).mean())

    data.drop(['rolling_price_max_t365', 'lag_price_t1'], inplace = True, axis = 1) # Xóa các cột tạm


    # === Các đặc trưng thời gian (Time Features) ===
    # data['date'] đã được chuyển đổi thành datetime ở đầu hàm
    attrs = ["year", "quarter", "month", "week", "day", "dayofweek", "is_year_end", "is_year_start", "is_quarter_end", \
        "is_quarter_start", "is_month_end","is_month_start",
    ]

    for attr in attrs:
        dtype = np.int16 if attr == "year" else np.int8
        data[attr] = getattr(data['date'].dt, attr).astype(dtype)
    data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(np.int8)

    # Thêm các đặc trưng thời gian khác
    # data['day_of_year'] = data['date'].dt.dayofyear # Ngày trong năm
    # data['week_of_year'] = data['date'].dt.weekofyear # Tuần trong năm
    # data['weekday'] = data['date'].dt.weekday # Ngày trong tuần (0=Monday, 6=Sunday)

    return data


# features = [
#     "item_id", "dept_id", "cat_id", "store_id", "state_id", "event_name_1", "event_type_1", "snap_CA", "snap_TX", \
#     "snap_WI", "sell_price", \
#     # demand features.
#     "shift_t28", "rolling_std_t7", "rolling_std_t30", "rolling_std_t90", "rolling_std_t180", \
#     "rolling_mean_t7", "rolling_mean_t30", "rolling_mean_t60", \
#     # price features
#     "price_change_t1", "price_change_t365", "rolling_price_std_t7",
#     # time features.
#     "year", "month", "dayofweek",
# ]


# ("wday", "month", "year", 
#        "event_name_1", "event_type_1", #"event_name_2", "event_type_2", 
#        "snap_CA", "snap_TX", "snap_WI",
#        "sell_price", "sell_price_rel_diff", "sell_price_cumrel", "sell_price_roll_sd7",
#        "lag_t28", "rolling_mean_t7", "rolling_mean_t30", "rolling_mean_t60", 
#        "rolling_mean_t90", "rolling_mean_t180", "rolling_sd_t7", "rolling_sd_t30",
#        "item_id", "dept_id", "cat_id", "store_id", "state_id")


# def run_lgb(data):
    
#     # going to evaluate with the last 28 days
#     x_train = data[data['date'] <= '2016-03-27']
#     y_train = x_train['demand']
#     x_val = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]
#     y_val = x_val['demand']
#     test = data[(data['date'] > '2016-04-24')]
#     del data
#     gc.collect()
    
#     params = {
# #         'boosting_type': 'gbdt',
#         'metric': 'rmse',
#         'objective': 'poisson',
#         'n_jobs': -1,
#         'seed': 20,
#         'learning_rate': 0.1,
#         'alpha': 0.1,
#         'lambda': 0.1,
#         'bagging_fraction': 0.66,
#         'bagging_freq': 2, 
#         'colsample_bytree': 0.77}

#     train_set = lgb.Dataset(x_train[features], y_train)
#     val_set = lgb.Dataset(x_val[features], y_val)
    
#     del x_train, y_train
    
    
#     model = lgb.train(params, train_set, num_boost_round = 2000, early_stopping_rounds = 200, valid_sets = [train_set, val_set], verbose_eval = 100)
#     joblib.dump(model, 'lgbm_0.sav')
    
#     val_pred = model.predict(x_val[features], num_iteration=model.best_iteration)
#     val_score = np.sqrt(metrics.mean_squared_error(val_pred, y_val))
#     r2 = metrics.r2_score(val_pred, y_val)
#     mae = metrics.mean_absolute_error(val_pred, y_val)
#     print(f'Our val rmse score is {val_score}')
#     print(f'Our val mae score is {mae}')
#     print(f'Our val r2 score is {r2}')
#     y_pred = model.predict(test[features], num_iteration=model.best_iteration)
#     test['demand'] = y_pred
#     return test


# def predict(test, submission):
#     predictions = test[['id', 'date', 'demand']]
#     predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'demand').reset_index()
#     predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

#     evaluation_rows = [row for row in submission['id'] if 'evaluation' in row] 
#     evaluation = submission[submission['id'].isin(evaluation_rows)]

#     validation = submission[['id']].merge(predictions, on = 'id')
#     final = pd.concat([validation, evaluation])
#     final.to_csv('submission.csv', index = False)
    


# def transform_train_and_eval(data):
# #     data = transform(data)
#     data = simple_fe(data)
#     # reduce memory for new features so we can train
#     data = reduce_mem_usage(data)
#     test = run_lgb(data)
#     predict(test, submission_df)
    



# def run_lgb(data):
#     # Dữ liệu sẽ được chia tương tự
#     x_train = data[data['date'] <= '2016-03-27'].copy() # Sử dụng .copy() để tránh SettingWithCopyWarning
#     y_train = x_train['demand']
#     x_val = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')].copy()
#     y_val = x_val['demand']
#     test = data[(data['date'] > '2016-04-24')].copy() # Sử dụng .copy()
    
#     del data
#     gc.collect()
    
#     # === GIAI ĐOẠN 1: HUẤN LUYỆN MÔ HÌNH ĐẦU TIÊN ĐỂ LẤY FEATURE IMPORTANCE ===
#     print("--- Training initial model for feature importance ---")

#     # Thu thập tất cả các đặc trưng tiềm năng
#     # Lấy tất cả các cột trong x_train ngoại trừ 'id', 'date', 'demand' (và 'item_id', 'dept_id', v.v. nếu chúng không phải features chính)
#     # Trong trường hợp của bạn, 'id' và 'demand' là những cột cần loại bỏ.
#     # Còn các cột ID và Event/Snap đã được LabelEncode là features.
    
#     # Xác định các cột không phải features (cols_to_drop)
#     cols_to_drop = ['id', 'demand', 'date'] # Cột 'date' đã được sử dụng để tạo time features, có thể bỏ

#     # Tạo danh sách tất cả các cột đặc trưng tiềm năng
#     initial_features = [col for col in x_train.columns if col not in cols_to_drop]
    
#     # Loại bỏ các cột NaN hoàn toàn sau khi tạo features (nếu có)
#     # Một số rolling/lag feature sẽ có NaN ở đầu chuỗi.
#     # Cẩn thận với việc loại bỏ hàng vì bạn cần khớp các ID.
#     # Thay vào đó, LightGBM có thể xử lý NaN.
#     # Chỉ loại bỏ các cột nếu chúng có quá nhiều NaN hoặc không bao giờ có giá trị.
    
#     # Có thể loại bỏ các cột mà không có variance nếu muốn
#     # initial_features = [col for col in initial_features if x_train[col].nunique() > 1]
    
#     # Tạo LGBM Dataset với tất cả các features tiềm năng
#     train_set_initial = lgb.Dataset(x_train[initial_features], y_train)
#     val_set_initial = lgb.Dataset(x_val[initial_features], y_val)

#     # Các tham số cho huấn luyện ban đầu (có thể ít vòng hơn hoặc không cần early stopping quá chặt)
#     params_initial = {
#         'metric': 'rmse',
#         'objective': 'poisson', # Phù hợp cho đếm
#         'n_jobs': -1,
#         'seed': 20,
#         'learning_rate': 0.05, # Tốc độ học nhỏ hơn để đảm bảo tính toán importance ổn định hơn
#         'num_leaves': 128, # Tăng số lá để mô hình có thể học phức tạp hơn
#         'min_data_in_leaf': 100, # Giảm để tránh overfitting quá mức trong giai đoạn đầu
#         'lambda_l1': 0.1,
#         'lambda_l2': 0.1,
#         'bagging_fraction': 0.7,
#         'bagging_freq': 1,
#         'feature_fraction': 0.7,
#         'verbose': -1, # Tắt verbose trong giai đoạn này
#     }

#     model_initial = lgb.train(
#         params_initial,
#         train_set_initial,
#         num_boost_round = 1000, # Ví dụ 1000 vòng để có đủ importance
#         early_stopping_rounds = 100, # Dừng sớm
#         valid_sets = [val_set_initial],
#         verbose_eval = False # Không in chi tiết mỗi 100 vòng
#     )

#     # Thu thập Feature Importance
#     feature_importances = pd.DataFrame({
#         'feature': initial_features,
#         'importance': model_initial.feature_importance(importance_type='gain') # Sử dụng 'gain' thường tốt hơn 'split'
#     })
#     feature_importances = feature_importances.sort_values(by='importance', ascending=False).reset_index(drop=True)
#     print("Top 20 Feature Importances:")
#     print(feature_importances.head(20))

#     # === GIAI ĐOẠN 2: LỌC ĐẶC TRƯNG VÀ HUẤN LUYỆN LẠI MÔ HÌNH CUỐI CÙNG ===
#     print("\n--- Training final model with selected features ---")

#     # Lọc ra các đặc trưng quan trọng (ví dụ: importance > 0 hoặc top N)
#     # imp_threshold = feature_importances['importance'].quantile(0.25) # Giữ lại 75% các feature quan trọng nhất
#     # Hoặc chỉ đơn giản là giữ lại những feature có importance > 0
#     selected_features = feature_importances[feature_importances['importance'] > 0]['feature'].tolist()
    
#     # Nếu danh sách quá dài, bạn có thể giới hạn top N
#     # selected_features = feature_importances.head(200)['feature'].tolist() # Ví dụ top 200

#     print(f"Selected {len(selected_features)} features out of {len(initial_features)}.")

#     # Kiểm tra xem có cột nào bị NaN hoàn toàn sau khi chọn features không
#     # (Điều này có thể xảy ra nếu một rolling feature quá xa lịch sử và toàn NaN)
#     # Ví dụ: x_train[selected_features].dropna(axis=1, how='all', inplace=True)
#     # LightGBM có thể xử lý NaN, nhưng nếu toàn bộ cột là NaN thì nó không hữu ích.
    
#     # Cập nhật train_set và val_set với các đặc trưng đã chọn
#     train_set_final = lgb.Dataset(x_train[selected_features], y_train)
#     val_set_final = lgb.Dataset(x_val[selected_features], y_val)

#     # Các tham số cho huấn luyện cuối cùng (có thể tinh chỉnh lại)
#     params_final = {
#         'metric': 'rmse',
#         'objective': 'poisson',
#         'n_jobs': -1,
#         'seed': 20,
#         'learning_rate': 0.05, # Tốc độ học có thể thấp hơn để đạt hiệu suất tốt hơn
#         'num_leaves': 64, # Số lá có thể tinh chỉnh (64-128 là phổ biến)
#         'min_data_in_leaf': 20, # Có thể nhỏ hơn so với giai đoạn đầu
#         'lambda_l1': 0.1,
#         'lambda_l2': 0.1,
#         'bagging_fraction': 0.7,
#         'bagging_freq': 1,
#         'feature_fraction': 0.7,
#         # 'boosting_type': 'gbdt', # Mặc định là gbdt
#     }

#     model_final = lgb.train(
#         params_final,
#         train_set_final,
#         num_boost_round = 3000, # Tăng số vòng tối đa
#         early_stopping_rounds = 200, # Giữ early stopping
#         valid_sets = [train_set_final, val_set_final],
#         verbose_eval = 100
#     )
#     joblib.dump(model_final, 'lgbm_final_model.sav') # Lưu mô hình cuối cùng

#     # === Đánh giá và Dự đoán ===
#     val_pred = model_final.predict(x_val[selected_features], num_iteration=model_final.best_iteration)
#     val_score = np.sqrt(metrics.mean_squared_error(val_pred, y_val))
#     r2 = metrics.r2_score(val_pred, y_val)
#     mae = metrics.mean_absolute_error(val_pred, y_val)
#     print(f'Our final val rmse score is {val_score}')
#     print(f'Our final val mae score is {mae}')
#     print(f'Our final val r2 score is {r2}')
    
#     y_pred = model_final.predict(test[selected_features], num_iteration=model_final.best_iteration)
#     test['demand'] = y_pred
#     return test
def run_lgb(data):
    # Dữ liệu sẽ được chia tương tự
    x_train = data[data['date'] <= '2016-03-27'].copy()
    y_train = x_train['demand']
    x_val = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')].copy()
    y_val = x_val['demand']
    test = data[(data['date'] > '2016-04-24')].copy()
    
    del data
    gc.collect()
    
    print("--- Training initial model for feature importance ---")

    # Xác định các cột không phải features
    cols_to_drop = ['id', 'demand', 'date', 'part'] # Thêm 'part' vì nó là cột tạm thời
    
    # Tạo danh sách tất cả các cột đặc trưng tiềm năng
    initial_features = [col for col in x_train.columns if col not in cols_to_drop]
    
    # LỌC BỎ CÁC CỘT KHÔNG PHẢI SỐ HOẶC CÁC CỘT CÓ QUÁ NHIỀU NaN
    # Đây là bước quan trọng để tránh lỗi "argument must be a string or number"
    numeric_features = x_train[initial_features].select_dtypes(include=np.number).columns.tolist()
    initial_features = numeric_features
    
    # Optional: remove features with too many NaNs if LightGBM struggles or you want cleaner data
    # initial_features = [f for f in initial_features if x_train[f].isnull().sum() / len(x_train) < 0.9]

    print(f"Number of initial features before training: {len(initial_features)}")
    
    # Kiểm tra xem có cột nào còn NaN trong các feature được chọn không (LightGBM có thể xử lý NaN)
    # Tuy nhiên, nếu một cột toàn NaN, nó sẽ không có ý nghĩa.
    
    # Tạo LGBM Dataset với tất cả các features tiềm năng
    train_set_initial = lgb.Dataset(x_train[initial_features], y_train)
    val_set_initial = lgb.Dataset(x_val[initial_features], y_val)

    params_initial = {
        'metric': 'rmse',
        'objective': 'poisson',
        'n_jobs': -1,
        'seed': 20,
        'learning_rate': 0.05,
        'num_leaves': 128,
        'min_data_in_leaf': 100,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'feature_fraction': 0.7,
        'verbose': -1,
        'device': 'gpu', 
        'gpu_platform_id': 0,
        'gpu_device_id': 0    
    }

    model_initial = lgb.train(
        params_initial,
        train_set_initial,
        num_boost_round = 1000,
        early_stopping_rounds = 100,
        valid_sets = [val_set_initial],
        verbose_eval = False
    )

    feature_importances = pd.DataFrame({
        'feature': initial_features,
        'importance': model_initial.feature_importance(importance_type='gain')
    })
    feature_importances = feature_importances.sort_values(by='importance', ascending=False).reset_index(drop=True)
    print("Top 20 Feature Importances:")
    print(feature_importances.head(20))

    print("\n--- Training final model with selected features ---")

    selected_features = feature_importances[feature_importances['importance'] > 0]['feature'].tolist()
    
    # Lặp lại bước lọc số một lần nữa để chắc chắn sau khi feature importance
    selected_features = [f for f in selected_features if f in x_train.columns and np.issubdtype(x_train[f].dtype, np.number)]
    
    print(f"Selected {len(selected_features)} features out of {len(initial_features)}.")

    train_set_final = lgb.Dataset(x_train[selected_features], y_train)
    val_set_final = lgb.Dataset(x_val[selected_features], y_val)

    params_final = {
        'metric': 'rmse',
        'objective': 'poisson',
        'n_jobs': -1,
        'seed': 20,
        'learning_rate': 0.05,
        'num_leaves': 64,
        'min_data_in_leaf': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'feature_fraction': 0.7,
    }

    model_final = lgb.train(
        params_final,
        train_set_final,
        num_boost_round = 2000,
        early_stopping_rounds = 200,
        valid_sets = [train_set_final, val_set_final],
        verbose_eval = 100
    )
    joblib.dump(model_final, 'lgbm_final_model.sav')

    val_pred = model_final.predict(x_val[selected_features], num_iteration=model_final.best_iteration)
    val_score = np.sqrt(metrics.mean_squared_error(val_pred, y_val))
    r2 = metrics.r2_score(val_pred, y_val)
    mae = metrics.mean_absolute_error(val_pred, y_val)
    print(f'Our final val rmse score is {val_score}')
    print(f'Our final val mae score is {mae}')
    print(f'Our final val r2 score is {r2}')
    
    y_pred = model_final.predict(test[selected_features], num_iteration=model_final.best_iteration)
    test['demand'] = y_pred
    return test
def predict(test, submission):
    predictions = test[['id', 'date', 'demand']]
    predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'demand').reset_index()
    predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

    evaluation_rows = [row for row in submission['id'] if 'evaluation' in row]
    evaluation = submission[submission['id'].isin(evaluation_rows)]

    validation = submission[['id']].merge(predictions, on = 'id')
    final = pd.concat([validation, evaluation])
    final.to_csv('submission.csv', index = False)
    

# def transform_train_and_eval(data):
#     # ĐẢM BẢO GỌI HÀM TRANSFORM ĐẦU TIÊN ĐỂ XỬ LÝ ID VÀ EVENT
#     data = transform(data) # <--- BỎ COMMENT DÒNG NÀY HOẶC ĐẢM BẢO transform ĐƯỢC GỌI TRƯỚC
#     data = simple_fe(data)
#     # reduce memory for new features so we can train
#     data = reduce_mem_usage(data) # Đảm bảo hàm reduce_mem_usage được định nghĩa
#     test = run_lgb(data)
#     predict(test, submission_df) # Đảm bảo submission_df đã được load
def transform_train_and_eval(data):
    # data = transform(data) # <--- BỎ COMMENT DÒNG NÀY HOẶC XÓA HÀM TRANSFORM HOÀN TOÀN
    # Lý do: Các cột phân loại (item_id, event_name_1, v.v.) đã được
    # LabelEncode bởi hàm encode_categorical() khi đọc dữ liệu ban đầu.
    # Việc gọi transform() ở đây lần nữa là không cần thiết và có thể gây lỗi
    # nếu nó cố gắng xử lý các cột đã là số nguyên.

    data = simple_fe(data)
    print(data.shape)
    # reduce memory for new features so we can train
    data = reduce_mem_usage(data) # Đảm bảo hàm reduce_mem_usage được định nghĩa
    test = run_lgb(data)
    predict(test, submission_df)


transform_train_and_eval(data)


# import pandas as pd
# import numpy as np
# import joblib
# # Giả sử các hàm reduce_mem_usage, encode_categorical, simple_fe, read_data đã được định nghĩa
# # và bạn đã chạy phần đọc dữ liệu ban đầu
# # sell_prices_df, calendar_df, sales_train_validation_df, submission_df = read_data()
# # ... (thực hiện encode_categorical và melt_and_merge) ...
# # data = melt_and_merge(...)
# # data = simple_fe(data)
# # data = reduce_mem_usage(data) # Đây là data đã có đầy đủ đặc trưng và giảm bộ nhớ

# # Huấn luyện và lưu model (đã có trong code của bạn)
# # model = lgb.train(...)
# # joblib.dump(model, 'lgbm_0.sav')

# # --- Phần để dự đoán cho một dòng bất kỳ ---

# # 1. Tải lại model
# loaded_model = joblib.load('lgbm_0.sav')
# print("Mô hình đã được tải thành công!")

# # 2. Chuẩn bị dữ liệu lịch sử cần thiết cho một ID cụ thể
# # Để dự đoán cho một ngày cụ thể (ví dụ d_1942) của một ID cụ thể
# # bạn cần lịch sử của ID đó để tính các features như rolling mean/std, shift.
# # Giả sử bạn muốn dự đoán cho 'FOODS_1_001_CA_1_validation' cho ngày 'd_1942'

# target_id = 'FOODS_1_001_CA_1_validation'
# target_day = 'd_1942' # Ngày đầu tiên của giai đoạn evaluation

# # Lấy dữ liệu lịch sử của ID này từ DataFrame `data` gốc trước khi chia train/val/test
# # Bạn cần đảm bảo `data` gốc (đã qua `simple_fe` và `melt_and_merge`) vẫn có sẵn
# # Hoặc tái tạo nó bằng cách đọc lại và xử lý như đã làm trong `melt_and_merge`

# # Cách đơn giản nhất để làm điều này là tái tạo một phần nhỏ của DataFrame `data`
# # chứa lịch sử cần thiết và hàng bạn muốn dự đoán.
# # Giả sử bạn muốn look-back 180 ngày (tối đa cho rolling features của bạn)
# # Lấy 180 ngày cuối cùng của dữ liệu train (d_1734 đến d_1913) + ngày target (d_1914 hoặc d_1942)
# # Đây là phần phức tạp nhất, bạn cần tái tạo chính xác logic của `melt_and_merge` và `simple_fe`
# # để tạo ra DataFrame cho `target_id` và `target_day` với đầy đủ các đặc trưng

# # Ví dụ (cách đơn giản hóa, cần điều chỉnh để khớp chính xác với logic của bạn):
# # Lấy dữ liệu cho 1 item_id/store_id cụ thể từ DataFrame `data` đã có đặc trưng
# # Trong trường hợp thực tế, bạn sẽ cần tạo một DataFrame mới với các ngày lịch sử
# # của ID đó và sau đó thêm hàng tương lai (d_1942) vào để tính features.

# # Giả sử `data` là DataFrame đầy đủ đã qua `simple_fe`
# # Lấy các hàng liên quan đến một ID cụ thể và một khoảng thời gian nhất định để tính features
# # Ví dụ: Để dự đoán d_1942, bạn cần dữ liệu đến d_1941 (có thể là d_1913 từ lịch sử training)
# # cộng thêm ngày d_1942 (với demand = NaN)
# # Sau đó chạy `simple_fe` trên phần đó và lấy hàng d_1942

# # Đây là một cách tiếp cận mẫu:
# # 1. Tạo một DataFrame giả định cho ngày bạn muốn dự đoán (d_1942)
# #    với các thông tin cơ bản của item_id, store_id.
# #    Cột `demand` sẽ là NaN.
# # 2. Lấy `N` ngày lịch sử cuối cùng (ví dụ: 365 ngày) của item/store đó từ `sales_train_validation_df`
# #    (sau khi melt và reduce_mem_usage).
# # 3. Nối (concat) dữ liệu lịch sử và ngày dự đoán.
# # 4. Hợp nhất với `calendar_df` và `sell_prices_df` (đã được encode và reduce_mem_usage).
# # 5. Chạy hàm `simple_fe` của bạn trên DataFrame kết hợp này.
# # 6. Trích xuất hàng của ngày `target_day` và các cột `features` cần thiết.

# # Để đơn giản hóa cho ví dụ này, giả sử bạn đã có một DataFrame `single_row_df`
# # đại diện cho một hàng dữ liệu đã được chuẩn bị đầy đủ các đặc trưng (shift, rolling, price, time)
# # và có cùng các cột trong `features` list.

# # Tạo một hàng mẫu giả định (trong thực tế, bạn phải tạo ra nó từ pipeline FE)
# # Đây là phần bạn cần triển khai lại logic chuẩn bị dữ liệu cho 1 dòng
# # Giả sử `x_val` từ hàm `run_lgb` là một tập dữ liệu đã có đặc trưng.
# # Lấy một hàng bất kỳ từ đó làm ví dụ.

# # Giả sử bạn muốn dự đoán cho hàng đầu tiên của `x_val`
# # Lấy một mẫu từ `x_val` đã có các đặc trưng
# # (Trong thực tế, bạn sẽ tạo ra một dòng mới với dữ liệu tương lai và chạy FE)
# # For this example, let's just pick one row from the already processed `data`
# # that would typically be in `x_val` or `test`
# # For a true single prediction, you'd feed in the last N days of real data,
# # then append the new day with NaN demand, then run simple_fe on this small combined DF,
# # then extract the last row.

# # This part is crucial:
# # The `transform_train_and_eval` function already processed `data`
# # and split it into `x_train`, `x_val`, `test`.
# # Let's use `x_val` as an example of a pre-processed DataFrame.
# # If you want to predict a truly arbitrary single row that wasn't in `x_val` or `test`,
# # you would need to perform the `melt_and_merge` and `simple_fe` steps on a smaller
# # DataFrame containing the required historical context for that specific item.

# # Let's take the first row of `x_val` to demonstrate:
# # (You'd need to re-run parts of the notebook to get `x_val` after `run_lgb` has `del` `data`)

# # Let's simulate:
# # 1. Read data
# # 2. Encode categorical
# # 3. Melt and merge
# # 4. Simple FE
# # 5. Select a single row
# sell_prices_df, calendar_df, sales_train_validation_df, submission_df = read_data()
# calendar_df = encode_categorical(calendar_df, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]).pipe(reduce_mem_usage)
# sales_train_validation_df = encode_categorical(sales_train_validation_df, ["item_id", "dept_id", "cat_id", "store_id", "state_id"]).pipe(reduce_mem_usage)
# sell_prices_df = encode_categorical(sell_prices_df, ["item_id", "store_id"]).pipe(reduce_mem_usage)

# nrows_val = 27500000 # Use the same nrows as in the main script
# data_for_single_pred = melt_and_merge(calendar_df, sell_prices_df, sales_train_validation_df, submission_df, nrows = nrows_val, merge = True)
# data_for_single_pred = simple_fe(data_for_single_pred)
# data_for_single_pred = reduce_mem_usage(data_for_single_pred) # Reduce memory after FE

# # Select a specific row to predict (e.g., first row of the validation set equivalent)
# # This will be '2016-03-28' for item 'FOODS_1_001_CA_1_validation' assuming the data ordering
# single_row_to_predict = data_for_single_pred[(data_for_single_pred['id'] == 'FOODS_1_001_CA_1_validation') & (data_for_single_pred['date'] == '2016-03-28')].copy()

# if not single_row_to_predict.empty:
#     # Ensure it only has the required features
#     single_row_input = single_row_to_predict[features]

#     # Drop rows with NaN in features (especially important for first few days after shift)
#     # The model expects non-NaN features.
#     single_row_input.dropna(inplace=True)

#     if not single_row_input.empty:
#         # Predict
#         predicted_demand = loaded_model.predict(single_row_input, num_iteration=loaded_model.best_iteration)
#         print(f"\nDự đoán demand cho {single_row_to_predict['id'].iloc[0]} vào ngày {single_row_to_predict['date'].iloc[0]} là: {predicted_demand[0]:.2f}")
#     else:
#         print("\nKhông đủ dữ liệu lịch sử để tính các đặc trưng cho dòng đã chọn.")
# else:
#     print("\nKhông tìm thấy dòng dữ liệu cần dự đoán trong tập đã xử lý.")


# # Giả sử `single_row_prepared_for_prediction` là DataFrame 1 hàng đã được chuẩn bị đầy đủ đặc trưng
# predicted_demand = loaded_model.predict(single_row_prepared_for_prediction[features], num_iteration=loaded_model.best_iteration)
# print(f"Dự đoán demand cho dòng đó là: {predicted_demand[0]}")




