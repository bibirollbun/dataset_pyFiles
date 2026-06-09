%%capture
!pip install -U lightautoml
!pip install flaml[automl] matplotlib openml
!pip install -U ipywidgets


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import requests
import joblib

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import torch

from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task

from flaml import AutoML

from flaml.automl.model import LGBMEstimator


 
RANDOM_STATE = 42 

np.random.seed(RANDOM_STATE) 


train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv', parse_dates=['date'])
ss = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')

inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
weights = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')
calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv', parse_dates=['date'])


from datetime import datetime

# Danh sách ngày lễ bổ sung (các ngày bị thiếu trong dữ liệu gốc) cho từng kho
czech_holiday = [ 
    (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),#loss
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"), #loss
]
brno_holiday = [
    (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),#loss
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"), #loss
]

budapest_holidays = []
munich_holidays = [
    (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),#loss
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),#loss
]

frank_holidays = [
    (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),#loss
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),#loss
]

def fill_loss_holidays(df_fill, warehouses, holidays):
    df = df_fill.copy()
    for item in holidays:
        dates, holiday_name = item # Lấy danh sách ngày và tên ngày lễ
        # Chuyển đổi định dạng chuỗi ngày sang yyyy-mm-dd (giống định dạng trong dataframe)
        generated_dates = [datetime.strptime(date, '%m/%d/%Y').strftime('%Y-%m-%d') for date in dates]
        for generated_date in generated_dates:
            # Gán giá trị 1 cho cột 'holiday' nếu khớp ngày và tên kho
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday'] = 1
            # Ghi lại tên của ngày lễ tương ứng
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday_name'] = holiday_name
    return df

calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Prague_1', 'Prague_2', 'Prague_3'], holidays=czech_holiday)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Brno_1'], holidays=brno_holiday)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Munich_1'], holidays=munich_holidays)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Frankfurt_1'], holidays=frank_holidays)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Budapest_1'], holidays=budapest_holidays)


# Lọc dữ liệu theo từng kho và chỉ lấy từ ngày 01/08/2020 trở đi
Frankfurt_1 = calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Frankfurt_1"')
Prague_2 = calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Prague_2"')
Brno_1 = calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Brno_1"')
Munich_1 = calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Munich_1"')
Prague_3 = calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Prague_3"')
Prague_1 = calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Prague_1"')
Budapest_1 = calendar.query('date >= "2020-08-01 00:00:00" and warehouse =="Budapest_1"')

def process_calendar(df):
    """
    Xử lý DataFrame lịch (calendar) bằng cách thêm các cột đặc trưng:
    - days_to_holiday: số ngày đến kỳ nghỉ tiếp theo
    - days_to_shops_closed: số ngày đến lần đóng cửa tiếp theo
    - day_after_closing: ngày liền sau khi cửa hàng đóng
    - long_weekend: có phải cuối tuần dài không (cửa hàng đóng liên tiếp)
    - weekday: thứ trong tuần (0 = Thứ 2, 6 = Chủ Nhật)
    """
    # Đảm bảo dữ liệu được sắp xếp theo ngày
    df = df.sort_values('date').reset_index(drop=True)

    # 1. days_to_holiday (Tính số ngày đến kỳ nghỉ tiếp theo)
    df['next_holiday_date'] = df.loc[df['holiday'] == 1, 'date'].shift(-1)
    df['next_holiday_date'] = df['next_holiday_date'].bfill()
    df['days_to_holiday'] = (df['next_holiday_date'] - df['date']).dt.days
    df.drop(columns=['next_holiday_date'], inplace=True)

    # 2. days_to_shops_closed
    df['next_shops_closed_date'] = df.loc[df['shops_closed'] == 1, 'date'].shift(-1)
    df['next_shops_closed_date'] = df['next_shops_closed_date'].bfill()
    df['days_to_shops_closed'] = (df['next_shops_closed_date'] - df['date']).dt.days
    df.drop(columns=['next_shops_closed_date'], inplace=True)

    # 3. day_after_closing
    df['day_after_closing'] = (
        (df['shops_closed'] == 0) & (df['shops_closed'].shift(1) == 1)
    ).astype(int)

    # 4. long_weekend
    df['long_weekend'] = (
        (df['shops_closed'] == 1) & (df['shops_closed'].shift(1) == 1)
    ).astype(int)

    # 5. weekday
    df['weekday'] = df['date'].dt.weekday

    return df


# Danh sách DataFrames
dfs = ['Frankfurt_1', 'Prague_2', 'Brno_1', 'Munich_1', 'Prague_3', 'Prague_1', 'Budapest_1']

# Áp dụng hàm xử lý cho từng DataFrame và gom lại thành danh sách
processed_dfs = [process_calendar(globals()[df]) for df in dfs]

# Gộp tất cả các DataFrame lại thành một bảng duy nhất
calendar_extended = pd.concat(processed_dfs).sort_values('date').reset_index(drop=True)


train_calendar = train.merge(calendar_extended, on=['date', 'warehouse'], how='left')
train_inventory = train_calendar.merge(inventory, on=['unique_id', 'warehouse'], how='left')
train_data = train_inventory.merge(weights, on=['unique_id'], how='left')

test_calendar = test.merge(calendar_extended, on=['date', 'warehouse'], how='left')
test_data = test_calendar.merge(inventory, on=['unique_id', 'warehouse'], how='left')


#train_data = train_data.drop(columns=['availability']) 
test_data['availability'] = 1

test_data.head()


train_data.dropna(subset=['sales'], inplace=True)


train_data.dtypes


train_data = train_data.sort_values(['unique_id', 'date'])
#train_data = train_data.set_index('date')
train_data.head()


%%time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gc
 

# Ngày bắt đầu và ngày kết thúc (trong tập huấn luyện)
start_date = train_data['date'].min()  # Ngày nhỏ nhất
end_date = train_data['date'].max()    # Ngày lớn nhất

# Khởi tạo biến ngày hiện tại
current_date = start_date
weekends = []

# Lấy danh sách tất cả các ngày cuối tuần
while current_date <= end_date:
    if current_date.weekday() == 5 or current_date.weekday() == 6:
        weekends.append(current_date.strftime('%Y-%m-%d'))
    current_date += timedelta(days=1)

print("-" * 30)

# Phân loại loại thực phẩm dựa vào tên
def get_food_type(food):
    food_types = {
        "fruit": [
            "Apple", "Avocado", "Banana", "Cucumber", "Lemon", "Mango", "Melon", 
            "Orange", "Pear", "Pineapple", "Pomegranate", "Grape", "Watermelon", 
            "Blueberry", "Lime", "Zucchini", "Grapefruit", "Physalis", "Berry", 
            "Tangerine", "Apricot", "Pomelo", "Blackberry", "Cherry", "Raspberry", 
            "Passion fruit", "Date", "Plum", "Fig", "Cactus Fruit", "Peach", 
            "Nectarine", "Strawberry", "Mandarin", "Persimmon", "Canteloupe", 
            "Lamb's lettuce"
        ],
        "vegetable": [
            "Tomato", "Potato", "Mushroom", "Onion", "Lettuce", "Cabbage", "Carrot", 
            "Pepper", "Bell Pepper", "Radish", "Pumpkin", "Broccoli", "Basil", 
            "Cauliflower", "Leek", "Chive", "Eggplant", "Kohlrabi", "Asparagus", 
            "Rosemary", "Mint", "Chicory", "Fennel", "Strawberry", "Raspberry", 
            "Ginger", "Pak choi", "Green Bean", "Cress", "Pea", "Pomelo", "Chili", 
            "Squash", "Paprika", "Nut", "Plantain", "Soybean sprout", "Cantaloupe"
        ],
        "meat": [
            "Chicken", "Pork", "Beef", "Turkey", "Mix meat", "Duck", "Plant meat", "Burger"
        ],
        "fish": [
            "Salmon", "Shrimp", "Surimi"
        ],
        "other": [
            "Herb", "Salad", "Parsley", "Garlic", "Beet", "Spinach", "Sweet Potato", 
            "Thyme", "Snack", "Arugula", "Grapefruit", "Physalis", "Berry", 
            "Shallot", "Corn", "Sprout", "Bean", "Cauliflower", "Leek", "Chive", 
            "Eggplant", "Kohlrabi", "Asparagus", "Rosemary", "Mint", "Chicory", 
            "Peach", "Nectarine", "Thyme", "Fennel", "Strawberry", "Raspberry", 
            "Ginger", "Passion fruit", "Date", "Plum", "Fig", "Bell pepper", 
            "Soup", "Cactus Fruit", "Pak choi", "Drink", "Pappudia", "Tangerine", 
            "Apricot", "Pea", "Pomelo", "Bag", "Chili", "Blackberry", "Granadilla", 
            "Cherry", "Squash", "Paprika", "Nut", "Plantain", "Mandarin", 
            "Soybean sprout", "Soil", "Cantaloupe", "Green Bean", "Persimmon", 
            "Cress", "Pepperoni", "Gooseberry", "Currant", "Flower"
        ],
        'Bakery': [
            'Bread', 'Pastry', 'Roll', 'Baguette', 'Toust', 'Croissant', 'Tortilla',
            'Donut', 'Snack', 'Cake', 'Pretzel', 'Cracker', 'Muffin', 'Bagel',
            'Breadcrumb', 'Pita', 'Rice Cake', 'Bun', 'Waffle', 'Biscuit',
            'Sandwich', 'Cheese', 'Wrap', 'Breadcrumbs', 'Focaccia', 'Cookie',
            'Cream', 'Cornmeal', 'Dessert', 'Grain', 'Hot Dog', 'Pasta', 'Pizza',
            'Flatbread', 'Yogurt', 'Bakery', 'Lucki', 'Brioche'
        ]
    }

    for food_type, food_list in food_types.items():
        if food in food_list:
            return food_type
    return 'other'

print("-" * 30)
import datetime as dt

# Hàm xử lý đặc trưng (Feature Engineering)
def FE(df):
    print("< Date Feature Processing >")

    # Chuyển cột 'date' sang kiểu dữ liệu datetime
    df['date_copy'] = pd.to_datetime(df['date'])

    # Trích xuất các đặc trưng từ ngày: ngày, tuần, tháng, quý, năm,...

    # Chuyển ngày thành số thứ tự kể từ 01/08/2020 (ngày đầu tiên trong dữ liệu)
    df['time_no'] = (df['date_copy'] - dt.datetime(2020, 8, 1)) // dt.timedelta(days=1)

    # Biến đổi tuần hoàn dạng sin/cos để mô hình học được tính chu kỳ theo nửa năm (~182.5 ngày)
    # Lưu ý: có thể là đặc trưng làm giảm điểm nếu dùng không phù hợp
    # Đặc trưng tuần hoàn với chu kỳ 182.5 ngày (nửa năm)
    df['year_sin_1'] = np.sin(np.pi * df['time_no'] / 182.5) # Đặc trưng tuần hoàn - nửa năm (sin)
    df['year_cos_1'] = np.cos(np.pi * df['time_no'] / 182.5) # Đặc trưng tuần hoàn - nửa năm (cos)

    # Đặc trưng tuần hoàn với chu kỳ 365 ngày (1 năm)
    df['year_sin_0.5'] = np.sin(np.pi * df['time_no'] / 365.0) # Đặc trưng tuần hoàn - 1 năm (cos)
    df['year_cos_0.5'] = np.cos(np.pi * df['time_no'] / 365.0) # Đặc trưng tuần hoàn - 1 năm (cos)
    
    df['year'] = df['date_copy'].dt.year
    df['quarter'] = df['date_copy'].dt.quarter 
    df['month'] = df['date_copy'].dt.month
    df['day'] = df['date_copy'].dt.day
    df['weekday'] = df['date_copy'].dt.weekday
    df['day_of_week'] = df['date_copy'].dt.dayofweek
    df['weekend'] = (df['day_of_week'] > 4).astype(np.int8)
    df['week_of_year'] = df['date_copy'].dt.isocalendar().week

    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # 1 = Weekend, 0 = Weekday

    
    df['is_month_start'] = df['date_copy'].dt.is_month_start
    df['is_month_end'] = df['date_copy'].dt.is_month_end

    df['dayofyear'] = df['date_copy'].dt.dayofyear
    df['sin_dayofyear'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['cos_dayofyear'] = np.cos(2 * np.pi * df['dayofyear'] / 365) 

    df['sin_dayofweek'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['cos_dayofweek'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
     
    df['sin_weekofyear'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['cos_weekofyear'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    df['year_sin'] = np.sin(2 * np.pi * df['year'] / 7.0)
    df['year_cos'] = np.cos(2 * np.pi * df['year'] / 7.0)
    df['group']=(df['year']-2020)*48+df['month']*4+df['day']//4 
 
    df['max_type_discount'] = 0
    
    # Duyệt qua tất cả các loại giảm giá (từ type_0_discount đến type_6_discount) 
    # Lấy giá trị giảm giá lớn nhất trên mỗi hàng (sản phẩm/ngày)
    for i in range(7):
        # Take the larger value between type_i_discount and total_type_discount.
        df['max_type_discount'] = df[['max_type_discount', f'type_{i}_discount']].max(axis=1) 

    discount_columns = [col for col in df.columns if 'discount' in col]
    df['max_discount'] = df[discount_columns].max(axis=1)


    print("Transform the data from 2020 to 2023")
    tmp = df.groupby(['year', 'warehouse', 'product_unique_id'])['sales'].mean().reset_index()
    tmp.columns = ['year', 'warehouse', 'product_unique_id', 'mean_sales']
    df = df.merge(tmp, on=['year', 'warehouse', 'product_unique_id'], how='left')

    print("< top1 solution >")
    # Dựa theo giải pháp đạt top 1 trên Kaggle:
    # https://www.kaggle.com/code/yunsuxiaozi/rohlik-top1-solution/notebook
    
    # Tạo một từ điển ánh xạ để rút gọn hoặc chuẩn hóa tên các ngày nghỉ lễ
    rename_dict = {
        "Memorial Day for the Victims of the Holocaust": "Victims of the Holocaust",
        "Memorial Day for the Victims of the Communist Dictatorships": "Victims of the Communist",
        "Den vzniku samostatneho ceskoslovenskeho statu": "Den vzniku"
    }

    # Thay thế tên kỳ nghỉ trong cột 'holiday_name' bằng các giá trị đã rút gọn
    # Giảm độ phức tạp và số lượng nhãn rời rạc để mô hình học tốt hơn
    df['holiday_name'] = df['holiday_name'].replace(rename_dict)
    
    print("< Data cleaning >")
    df['name_0'] = df['name'].apply(lambda x: x.split("_")[0])
    df['name_1'] = df['name'].apply(lambda x: x.split("_")[1])

    df.drop(['name', 'product_unique_id'], axis=1, inplace=True)
    df['L5_category_name_en'] = df['name_1'].apply(lambda x: get_food_type(x))
    for i in range(2, 5):
        df[f'L{i}_category_name_en'] = df[f'L{i}_category_name_en'].apply(lambda x: x.split('_')[2]) 

    # Điều chỉnh ngày nghỉ lễ đặc biệt cho một số kho
    datesx = ['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020']
    holidaysx = [datetime.strptime(date, '%m/%d/%Y') - timedelta(days=1) for date in datesx]
    warehouses = ['Prague_1', 'Prague_2', 'Prague_3']
    df.loc[(df['date'].isin(holidaysx)) & (df['warehouse'].isin(warehouses)), 'holiday'] = 1
    discount_columns = ['type_0_discount', 'type_1_discount', 'type_2_discount', 
                        'type_3_discount', 'type_4_discount', 'type_5_discount', 'type_6_discount']

    # Tạo các đặc trưng kết hợp giảm giá bằng phép cộng và nhân
    for i in range(len(discount_columns)):
        # Chỉ lấy từng cặp (i, j) duy nhất để tránh tạo đặc trưng lặp lại (i + j và j + i)
        for j in range(i+1, len(discount_columns)):  

            col1, col2 = discount_columns[i], discount_columns[j]
            
            # Đặc trưng tương tác cộng giữa hai loại giảm giá
            df[f'{col1}_plus_{col2}'] = df[col1] + df[col2] 
            
            # Đặc trưng tương tác nhân giữa hai loại giảm giá
            df[f'{col1}_times_{col2}'] = df[col1] * df[col2]

    # Đếm số lần xuất hiện của một đặc trưng tổ hợp giảm giá cụ thể
    temp = df['type_0_discount_plus_type_3_discount'].value_counts().to_dict()
    df['type_0_discount_plus_type_3_discount_counts'] = df['type_0_discount_plus_type_3_discount'].map(temp)

    # Xoá các cột không cần thiết
    df.drop(['date','date_copy','L1_category_name_en',
             'type_0_discount', 'type_1_discount', 'type_2_discount', 
                        'type_3_discount', 'type_4_discount', 'type_5_discount', 'type_6_discount',
             'type_0_discount_plus_type_3_discount'], axis=1, inplace=True)
 
    return df

# Gộp tập train và test để xử lý chung
total = pd.concat((train_data, test_data))

# Gọi feature engineering
total = FE(total)

# Xoá những cột có quá nhiều giá trị thiếu (NaN > 95%)
# drop_cols = ['availability']
total.drop([col for col in total.columns if total[col].isna().mean() > 0.95] , axis=1, inplace=True)


# Loại bỏ các đặc trưng không hợp lệ
drop_features =  [ 
    "type_0_discount_times_type_6_discount", 
    "type_5_discount_plus_type_6_discount", 
    "type_4_discount_plus_type_5_discount",
    # "type_5_discount_relative_to_price", 
    # "type_5_discount_binned",
    "type_4_discount_times_type_6_discount",
    # "type_6_discount_binned",
    "type_0_discount_times_type_4_discount" 
]
 
 
total.drop(columns=drop_features, inplace=True)


# Tách lại thành tập train và test ban đầu
train_data = total[:len(train_data)]
test_data = total[len(train_data):].drop(['sales', 'weight'], axis=1)

# Giải phóng bộ nhớ
del total
gc.collect()

print(f"train.shape: {train_data.shape}, test.shape: {test_data.shape}")
train_data.head() 


train_data.info()


%%time
# Tối ưu bộ nhớ cho tập test
def optimize_dataframe(df): 
    """
    Tối ưu hóa bộ nhớ của DataFrame bằng cách:
    - Giảm kiểu dữ liệu của các cột float/int nếu có thể
    - Chuyển object thành category nếu số lượng giá trị duy nhất ít
    - Tối ưu cột dạng sparse
    """
    
    # Xử lý các cột có kiểu float64 và float32
    for col in df.select_dtypes(include=["float64", "float32"]).columns:
        min_val, max_val = df[col].min(), df[col].max()
        # Nếu giá trị nằm trong khoảng cho phép của float16 → chuyển về float16        
        if min_val > np.finfo(np.float16).min and max_val < np.finfo(np.float16).max:
            df[col] = df[col].astype("float16")
        # Nếu không chuyển được về float16 thì vẫn giữ kiểu float32
        elif min_val > np.finfo(np.float32).min and max_val < np.finfo(np.float32).max:
            df[col] = df[col].astype("float32")  
    
    # Xử lý các cột có kiểu int64 và int32
    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        min_val, max_val = df[col].min(), df[col].max()
        # Nếu giá trị nằm trong khoảng int16 → chuyển về int16
        if min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
            df[col] = df[col].astype("int16")
        # Nếu không chuyển được về int16 thì giữ kiểu int32
        elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
            df[col] = df[col].astype("int32")
    
    # Xử lý các cột có kiểu dữ liệu object 
    for col in df.select_dtypes(include=["object"]).columns:
        # Nếu số lượng giá trị duy nhất (unique) trong cột chiếm ít hơn 50% tổng số dòng
        # → nghĩa là cột đó có ít giá trị phân biệt → chuyển thành kiểu category để tiết kiệm bộ nhớ
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype("category")  # Ép kiểu sang category

    
    #Xử lý các cột dạng Sparse (dữ liệu thưa, Ví dụ: Sparse[float64, 0])
    for col in df.select_dtypes(include=[pd.SparseDtype("float64", 0), pd.SparseDtype("float32", 0)]).columns:
        # Nếu là Sparse[float64, 0] → chuyển thành Sparse[float32, 0]
        if df[col].dtype == pd.SparseDtype("float64", 0):
            df[col] = df[col].astype(pd.SparseDtype("float32", 0))
        # Nếu là Sparse[float32, 0] → chuyển thành Sparse[float16, 0]
        if df[col].dtype == pd.SparseDtype("float32", 0):
            df[col] = df[col].astype(pd.SparseDtype("float16", 0))
    
    # Trả về DataFrame đã được tối ưu hóa bộ nhớ
    return df

# Ví dụ sử dụng: Tối ưu tập huấn luyện train_data
train_data = optimize_dataframe(train_data)
train_data.info()



task = Task('reg')


from sklearn.model_selection import train_test_split

# Tách lại dữ liệu (nếu cần)
X = train_data.drop(columns=["sales", "weight"])
y = train_data["sales"]
w = train_data["weight"]

X_train, X_valid, y_train, y_valid, w_train, valid_weights = train_test_split(
    X, y, w, test_size=0.2, random_state=42
)

# Huấn luyện lại mô hình
automl = AutoML()
automl.fit(
    X_train=X_train,
    y_train=y_train,
    sample_weight=w_train,
    task="regression",
    time_budget=15000,
    metric="mae",
    estimator_list=["lgbm"],
    log_file_name="retrain.log",
    seed=42
)

# Dự đoán và tính WMAE
y_pred = automl.predict(X_valid)

def wmae(y_true, y_pred, weights):
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

print(f"WMAE: {wmae(y_valid, y_pred, valid_weights):.5f}")



# HÀM TÍNH WMAE THỦ CÔNG (Weighted Mean Absolute Error)
def wmae(y_true, y_pred, weights):
    """
    Tính sai số tuyệt đối trung bình có trọng số (WMAE)
    """
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

y_pred = automl.predict(X_valid)
score = wmae(y_valid, y_pred, valid_weights)
print(f"WMAE: {score:.5f}")


#from lightautoml.report.report_deco import ReportDeco, ReportDecoUtilized
#from lightautoml.addons.tabular_interpretation import SSWARM


test_predictions = automl.predict(test_data)


test_predictions


ss = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')


submission = pd.DataFrame({
    'id': ss.id.values,
    'sales_hat': test_predictions,
})


submission


# Xem giá trị dự đoán lớn nhất ban đầu
max_value = submission['sales_hat'].max()

# Xem giá trị dự đoán nhỏ nhất ban đầu
min_value = submission['sales_hat'].min()

print(f"Maximum value: {max_value}, Minimum value: {min_value}")

# Đảm bảo tất cả giá trị dự đoán đều ≥ 0 (không âm)
submission['sales_hat'] = np.maximum(submission['sales_hat'], 0)

# Kiểm tra lại sau khi xử lý giá trị âm
# Giá trị lớn nhất
max_value = submission['sales_hat'].max()

# Giá trị nhỏ nhất
min_value = submission['sales_hat'].min()

print(f"Maximum value: {max_value}, Minimum value: {min_value}")


submission.to_csv('submission.csv', index = False)

