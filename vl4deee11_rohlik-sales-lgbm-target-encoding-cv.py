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


import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.linear_model import LinearRegression, GammaRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

warnings.filterwarnings('ignore')
R = 42


def calculate_wmae(X, y_true, y_pred, weight_map):
    """
    Calculates the Weighted Mean Absolute Error (WMAE).

    Parameters:
    - y_true: Actual (true) values.
    - y_pred: Predicted values from the model.
    - weights: Weights associated with each sample.

    Returns:
    - The calculated WMAE.
    """
    weights = np.array([weight_map[uid] for uid in X['unique_id']])

    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)


def smoothing(x):
    return np.log1p(x)


def r_smoothing(x):
    return np.expm1(x)


def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    return mis_val_table_ren_columns


PATH = '/kaggle/input/rohlik-sales-forecasting-challenge-v2/'


czech_holiday = [ 
    (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"),
]
brno_holiday = [
    (['03/31/2024', '04/09/2023', '04/17/2022', '04/04/2021', '04/12/2020'], 'Easter Day'),
    (['05/12/2024', '05/10/2020', '05/09/2021', '05/08/2022', '05/14/2023'], "Mother Day"),
]
munich_holidays = [
    (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),
]
frankfurt_holidays = [
    (['03/30/2024', '04/08/2023', '04/16/2022', '04/03/2021'], 'Holy Saturday'),
    (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day'),
]



def fill_loss_holidays(df_fill, warehouses, holidays):
    df = df_fill.copy()
    for item in holidays:
        dates, holiday_name = item
        generated_dates = [datetime.strptime(date, '%m/%d/%Y').strftime('%Y-%m-%d') for date in dates]
        for generated_date in generated_dates:
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday'] = 1
            df.loc[(df['warehouse'].isin(warehouses)) & (df['date'] == generated_date), 'holiday_name'] = holiday_name
    return df

def enrich_calendar(df):
    df = df.sort_values('date').reset_index(drop=True)

    # Number of days until next holiday
    df['next_holiday_date'] = df.loc[df['holiday'] == 1, 'date'].shift(-1)
    # Fill NaT values by using the next valid observation to fill the gap
    df['next_holiday_date'] = df['next_holiday_date'].bfill() 
    df['date_days_to_next_holiday'] = (df['next_holiday_date'] - df['date']).dt.days
    df.drop(columns=['next_holiday_date'], inplace=True)

    # Number of days until shops are closed
    df['next_shops_closed_date'] = df.loc[df['shops_closed'] == 1, 'date'].shift(-1)
    df['next_shops_closed_date'] = df['next_shops_closed_date'].bfill()
    df['date_days_to_shops_closed'] = (df['next_shops_closed_date'] - df['date']).dt.days
    df.drop(columns=['next_shops_closed_date'], inplace=True)

    # Was the shop closed yesterday?
    df['day_after_closed_day'] = ((df['shops_closed'] == 0) & (df['shops_closed'].shift(1) == 1)).astype(int)

    # Are shops closed today and were they also closed yesterday (e.g., December 26 in Germany)?
    df['second_closed_day'] = ((df['shops_closed'] == 1) & (df['shops_closed'].shift(1) == 1)).astype(int)

    # Was the shop closed the last two days?
    df['day_after_two_closed_days'] = ((df['shops_closed'] == 0) & (df['second_closed_day'].shift(1) == 1)).astype(int)
    
    df['weekday'] = df['date'].dt.weekday 

    return df


calendar = pd.read_csv(f'{PATH}calendar.csv', parse_dates=['date'])

calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Prague_1', 'Prague_2', 'Prague_3'], holidays=czech_holiday)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Brno_1'], holidays=brno_holiday)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Munich_1'], holidays=munich_holidays)
calendar = fill_loss_holidays(df_fill=calendar, warehouses=['Frankfurt_1'], holidays=frankfurt_holidays)


calendar_enriched = pd.DataFrame()

for location in ['Frankfurt_1', 'Prague_2', 'Brno_1', 'Munich_1', 'Prague_3', 'Prague_1', 'Budapest_1']:
    calendar_enriched = pd.concat([
        calendar_enriched,enrich_calendar(calendar.query('date >= "2020-08-01 00:00:00" and warehouse ==@location'))])
calendar_enriched.loc[:,'year'] = calendar_enriched['date'].dt.year


calendar_enriched.to_csv('calendar_enriched.csv', index=False)


dtype_train = {'unique_id': 'int32',
               'warehouse': 'category',
               'total_orders': 'float32',
               'sales': 'float32',
               'sell_price_main': 'float32',
               'availability': 'float32',
               'type_0_discount': 'float32',
               'type_1_discount': 'float32',
               'type_2_discount': 'float32',
               'type_3_discount': 'float32',
               'type_4_discount': 'float32',
               'type_5_discount': 'float32'}

dtype_test = {'unique_id': 'int32',
              'warehouse': 'category',
              'total_orders': 'float32',
              'sell_price_main': 'float32',
              'type_0_discount': 'float32',
              'type_1_discount': 'float32',
              'type_2_discount': 'float32',
              'type_3_discount': 'float32',
              'type_4_discount': 'float32',
              'type_5_discount': 'float32',
              'type_6_discount': 'float32'}

sales_train = pd.read_csv(f'{PATH}sales_train.csv',
                          dtype=dtype_train, parse_dates=['date'])
sales_test = pd.read_csv(f'{PATH}sales_test.csv',
                         dtype=dtype_test, parse_dates=['date'])
calendar = pd.read_csv(f'./calendar_enriched.csv',
                       parse_dates=['date'])

test_weights = pd.read_csv(f'{PATH}test_weights.csv',
                           dtype={'unique_id': 'int32', 'weight': 'float32'})

inventory = pd.read_csv(f'{PATH}inventory.csv',
                        dtype={'unique_id': 'int32',
                               'product_unique_id': 'int32',
                               'name': 'category',
                               'L1_category_name': 'category',
                               'L2_category_name': 'category',
                               'warehouse': 'category'}
                        )


weight_map = dict(zip(test_weights['unique_id'], test_weights['weight']))
train_unique_ids = sales_train['unique_id'].unique()
missing_weights = set(train_unique_ids) - set(weight_map.keys())
if missing_weights:
    print(f"Warning: {len(missing_weights)} unique_ids in train data don't have weights")
    default_weight = 1.0
    for uid in missing_weights:
        weight_map[uid] = default_weight


class FE:
    def __init__(self,):
        self.labenc_wh = LabelEncoder()
        
    def proc(self, sales_train, sales_test, calendar, inventory):
        def calculate_original_price(row):
            max_discount = max(row['type_0_discount'], row['type_1_discount'], row['type_2_discount'], row['type_3_discount'], row['type_4_discount'], row['type_5_discount'], row['type_6_discount'])
            if max_discount > 0:
                original_price = row['sell_price_main'] / (1 - max_discount)
            else:
                original_price = row['sell_price_main']
            return original_price
        
        def add_holiday_features(sales, calendar):
            calendar = calendar.sort_values(by='date')
            calendar['next_holiday_date'] = calendar['date'].where(calendar['holiday'] == 1)
            calendar['prev_holiday_date'] = calendar['date'].where(calendar['holiday'] == 1)

            calendar['next_holiday_date'] = calendar['next_holiday_date'].fillna(method='bfill')
            calendar['prev_holiday_date'] = calendar['prev_holiday_date'].fillna(method='ffill')

            calendar['days_to_next_holiday'] = (calendar['next_holiday_date'] - calendar['date']).dt.days
            calendar['days_from_prev_holiday'] = (calendar['date'] - calendar['prev_holiday_date']).dt.days

            holiday_features = calendar

            sales = sales.merge(holiday_features, on=['warehouse', 'date'], how='left')
            
            sales['days_to_next_holiday'] = sales['days_to_next_holiday'].fillna(9999)
            sales['days_from_prev_holiday'] = sales['days_from_prev_holiday'].fillna(-9999)
            return sales
        
        sales_train = add_holiday_features(sales_train, calendar)
        sales_test = add_holiday_features(sales_test, calendar)
        
        sales_train = sales_train.merge(inventory,on=['warehouse','unique_id'],how='left')
        sales_test = sales_test.merge(inventory,on=['warehouse','unique_id'],how='left')
        
        sales_train['date'] = pd.to_datetime(sales_train['date'])
        min_date = sales_train["date"].min()
        
        print("Performing main features...")
        for df in [sales_train, sales_test]:
            df['original_price'] = df.apply(calculate_original_price, axis=1)
            
            df['price_diff'] = df['sell_price_main'] - df['original_price']
            
            df['holiday_name'] = df['holiday_name'].fillna('None')
            
            df['max_discount'] = df[['type_0_discount', 'type_1_discount', 'type_2_discount', 'type_3_discount', 'type_4_discount', 'type_5_discount', 'type_6_discount']].max(axis=1)
            df['has_discount'] = (df['max_discount'] > 0).astype(int)
            
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['day'] = df['date'].dt.day
            df['month'] = df['date'].dt.month
            df['month_name'] = df['date'].dt.month_name()
            df['day_of_week'] = df['date'].apply(lambda x: x.dayofweek)
            df["day_of_year"] = df["date"].apply(lambda x: x.timetuple().tm_yday)
            df['week'] = df['date'].dt.isocalendar().week
            
            df['year_sin'] = np.sin(2 * np.pi * df['year'])
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12) 
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
            df['week_sin'] = np.sin(2*np.pi*df['week']/53)
            df['day_year_sin'] = np.sin(2*np.pi*df['day_of_year']/365)
            df['day_of_week_sin']=np.sin(2*np.pi*df['day_of_week']/7)
            
            df['group'] = (df['year'] - 2020) * 48 + df['month'] * 4 + df['day'] // 7
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype('int8')
            df['is_month_start'] = df['date'].dt.is_month_start
            df['is_month_end'] = df['date'].dt.is_month_end
            df['quarter'] = df['date'].dt.quarter
            
            df["days_since_start"] = df["date"].apply(lambda x: (x-min_date).days)



        def target_enc(sales_train, sales_test, col, stat, fill_na):
            te_col = f'{col}_target_encoded_{stat}'
            print(f"Performing Target Encoding for {col}...")
            sales_train[te_col] = sales_train.groupby(col)['sales'].transform(stat)

            unique_id_target = sales_train[[col, te_col]].drop_duplicates()
            print(f"Unique IDs after drop_duplicates: {unique_id_target.shape[0]}")
            if te_col in sales_test.columns:
                print(f"The column {te_col} already exists in sales_test. Removing to avoid duplication.")
                sales_test = sales_test.drop(te_col, axis=1)
            
            sales_test = sales_test.merge(unique_id_target, on=col, how='left')
            if te_col in sales_test.columns:
                print(f"The column {te_col} was successfully added to sales_test.")
            else:
                print(f"Error: The column {te_col} was NOT added to sales_test.")


            sales_test[te_col] = sales_test[te_col].fillna(fill_na)

            missing_values = sales_test[te_col].isnull().sum()
            if missing_values == 0:
                print("All NaN values were filled successfully.")
            else:
                print(f"There are still {missing_values} NaN values in the {te_col} column.")
            
            return sales_train, sales_test
        
        def target_kurtosis_enc(sales_train, sales_test, col, stat):
            te_col = f'{col}_target_encoded_{stat}'
            print(f"Performing Target Encoding for {col}...")
            sales_train[te_col] = sales_train.groupby(col)['sales'].transform(lambda x: kurtosis(x, fisher=True, bias=False))

            unique_id_target = sales_train[[col, te_col]].drop_duplicates()
            print(f"Unique IDs after drop_duplicates: {unique_id_target.shape[0]}")


            if te_col in sales_test.columns:
                print(f"The column {te_col} already exists in sales_test. Removing to avoid duplication.")
                sales_test = sales_test.drop(te_col, axis=1)

            sales_test = sales_test.merge(unique_id_target, on=col, how='left')

            if te_col in sales_test.columns:
                print(f"The column {te_col} was successfully added to sales_test.")
            else:
                print(f"Error: The column {te_col} was NOT added to sales_test.")


            fill_na = kurtosis(sales_train['sales'], fisher=True, bias=False)
            sales_test[te_col] = sales_test[te_col].fillna(fill_na)

            missing_values = sales_test[te_col].isnull().sum()
            if missing_values == 0:
                print("All NaN values were filled successfully.")
            else:
                print(f"There are still {missing_values} NaN values in the {te_col} column.")
            
            return sales_train, sales_test
        
        def target_quantile_enc(sales_train, sales_test, col, quantile):
            te_col = f'{col}_target_encoded_q{int(quantile * 100)}'
            print(f"Performing Target Encoding for {col} with quantile {quantile}...")
            sales_train[te_col] = sales_train.groupby(col)['sales'].transform(lambda x: x.quantile(quantile))

            unique_id_target = sales_train[[col, te_col]].drop_duplicates()
            print(f"Unique IDs after drop_duplicates: {unique_id_target.shape[0]}")

            if te_col in sales_test.columns:
                print(f"The column {te_col} already exists in sales_test. Removing to avoid duplication.")
                sales_test = sales_test.drop(te_col, axis=1)

            sales_test = sales_test.merge(unique_id_target, on=col, how='left')

            if te_col in sales_test.columns:
                print(f"The column {te_col} was successfully added to sales_test.")
            else:
                print(f"Error: The column {te_col} was NOT added to sales_test.")

            fill_na = sales_train['sales'].quantile(quantile)
            sales_test[te_col] = sales_test[te_col].fillna(fill_na)

            missing_values = sales_test[te_col].isnull().sum()
            if missing_values == 0:
                print("All NaN values were filled successfully.")
            else:
                print(f"There are still {missing_values} NaN values in the {te_col} column.")

            return sales_train, sales_test
        
        roll_coll = [
            'unique_id',
            'warehouse',
            'holiday_name',
            'month_name',
            'L1_category_name_en',
            'L2_category_name_en',
            'L3_category_name_en',
            'L4_category_name_en'
            
        ]
        
        roll_coll2 = []
        for i in range(len(roll_coll)):
            for j in range(i+1, len(roll_coll)):
                coln = f"{roll_coll[i]}_{roll_coll[j]}"
                roll_coll2.append(coln)
                sales_train[coln] = (
                    sales_train[roll_coll[i]].fillna('').astype(str) + 
                    "_" + 
                    sales_train[roll_coll[j]].fillna('').astype(str)
                ).str.strip('_')
                sales_test[coln] = (
                    sales_test[roll_coll[i]].fillna('').astype(str) + 
                    "_" + 
                    sales_test[roll_coll[j]].fillna('').astype(str)
                ).str.strip('_')
        
        mean_stat_fill_na = sales_train['sales'].mean()
        std_stat_fill_na = sales_train['sales'].std()
        for col in roll_coll+roll_coll2:
            sales_train, sales_test = target_enc(sales_train, sales_test, col, 'mean', mean_stat_fill_na)
            sales_train, sales_test = target_enc(sales_train, sales_test, col, 'std', mean_stat_fill_na)
            sales_train, sales_test = target_kurtosis_enc(sales_train, sales_test, col, 'kurtosis')
            sales_train, sales_test = target_quantile_enc(sales_train, sales_test, col, 0.25)
            sales_train, sales_test = target_quantile_enc(sales_train, sales_test, col, 0.75)


        print("Performing Label Encoding for 'warehouse'...")
        sales_train['warehouse_encoded'] = self.labenc_wh.fit_transform(sales_train['warehouse'])
        sales_test['warehouse_encoded'] = self.labenc_wh.transform(sales_test['warehouse'])
        
        for i in range(7):
            for df in [sales_train, sales_test]:
                df[f'is_warehouse_encoded_{i}'] = df['warehouse_encoded'] == i
        
        return sales_train, sales_test
 


fe = FE()
sales_train, sales_test = fe.proc(sales_train, sales_test, calendar, inventory)


missing_values_table(sales_train)


missing_values_table(sales_test)


def prepare_to_train(sales_train, days=1):
    features = ['unique_id'] + model_feaures
    cutoff_date = sales_train['date'].max() - pd.Timedelta(days=days)
    train_data = sales_train[sales_train['date'] <= cutoff_date]
    val_data = sales_train[sales_train['date'] > cutoff_date]

    train_data = train_data.dropna(subset=['sales'])
    val_data = val_data.dropna(subset=['sales'])
    X_train = train_data[features].astype('float32')  
    y_train = train_data['sales'].astype('float32') 
    X_val = val_data[features].astype('float32')  
    y_val = val_data['sales'].astype('float32')
    
    return X_train, X_val, y_train, y_val


model_feaures = [
    'unique_id_target_encoded_mean',
    'warehouse_target_encoded_mean',
    'holiday_name_target_encoded_mean',
    'month_name_target_encoded_mean',
    'L1_category_name_en_target_encoded_mean',
    'L2_category_name_en_target_encoded_mean',
    'L3_category_name_en_target_encoded_mean',
    'L4_category_name_en_target_encoded_mean',
    'unique_id_warehouse_target_encoded_mean',
    'unique_id_holiday_name_target_encoded_mean',
    'unique_id_month_name_target_encoded_mean',
    'unique_id_L1_category_name_en_target_encoded_mean',
    'unique_id_L2_category_name_en_target_encoded_mean',
    'unique_id_L3_category_name_en_target_encoded_mean',
    'unique_id_L4_category_name_en_target_encoded_mean',
    'warehouse_holiday_name_target_encoded_mean',
    'warehouse_month_name_target_encoded_mean',
    'warehouse_L1_category_name_en_target_encoded_mean',
    'warehouse_L2_category_name_en_target_encoded_mean',
    'warehouse_L3_category_name_en_target_encoded_mean',
    'warehouse_L4_category_name_en_target_encoded_mean',
    'holiday_name_month_name_target_encoded_mean',
    'holiday_name_L1_category_name_en_target_encoded_mean',
    'holiday_name_L2_category_name_en_target_encoded_mean',
    'holiday_name_L3_category_name_en_target_encoded_mean',
    'holiday_name_L4_category_name_en_target_encoded_mean',
    'month_name_L1_category_name_en_target_encoded_mean',
    'month_name_L2_category_name_en_target_encoded_mean',
    'month_name_L3_category_name_en_target_encoded_mean',
    'month_name_L4_category_name_en_target_encoded_mean',
    'L1_category_name_en_L2_category_name_en_target_encoded_mean',
    'L1_category_name_en_L3_category_name_en_target_encoded_mean',
    'L1_category_name_en_L4_category_name_en_target_encoded_mean',
    'L2_category_name_en_L3_category_name_en_target_encoded_mean',
    'L2_category_name_en_L4_category_name_en_target_encoded_mean',
    'L3_category_name_en_L4_category_name_en_target_encoded_mean',
    
    
    'unique_id_target_encoded_std',
    'warehouse_target_encoded_std',
    'holiday_name_target_encoded_std',
    'month_name_target_encoded_std',
    'L1_category_name_en_target_encoded_std',
    'L2_category_name_en_target_encoded_std',
    'L3_category_name_en_target_encoded_std',
    'L4_category_name_en_target_encoded_std',
    'unique_id_warehouse_target_encoded_std',
    'unique_id_holiday_name_target_encoded_std',
    'unique_id_month_name_target_encoded_std',
    'unique_id_L1_category_name_en_target_encoded_std',
    'unique_id_L2_category_name_en_target_encoded_std',
    'unique_id_L3_category_name_en_target_encoded_std',
    'unique_id_L4_category_name_en_target_encoded_std',
    'warehouse_holiday_name_target_encoded_std',
    'warehouse_month_name_target_encoded_std',
    'warehouse_L1_category_name_en_target_encoded_std',
    'warehouse_L2_category_name_en_target_encoded_std',
    'warehouse_L3_category_name_en_target_encoded_std',
    'warehouse_L4_category_name_en_target_encoded_std',
    'holiday_name_month_name_target_encoded_std',
    'holiday_name_L1_category_name_en_target_encoded_std',
    'holiday_name_L2_category_name_en_target_encoded_std',
    'holiday_name_L3_category_name_en_target_encoded_std',
    'holiday_name_L4_category_name_en_target_encoded_std',
    'month_name_L1_category_name_en_target_encoded_std',
    'month_name_L2_category_name_en_target_encoded_std',
    'month_name_L3_category_name_en_target_encoded_std',
    'month_name_L4_category_name_en_target_encoded_std',
    'L1_category_name_en_L2_category_name_en_target_encoded_std',
    'L1_category_name_en_L3_category_name_en_target_encoded_std',
    'L1_category_name_en_L4_category_name_en_target_encoded_std',
    'L2_category_name_en_L3_category_name_en_target_encoded_std',
    'L2_category_name_en_L4_category_name_en_target_encoded_std',
    'L3_category_name_en_L4_category_name_en_target_encoded_std',
    
    
    'unique_id_target_encoded_kurtosis',
    'warehouse_target_encoded_kurtosis',
    'L3_category_name_en_target_encoded_kurtosis',
    'unique_id_warehouse_target_encoded_kurtosis',
    'unique_id_holiday_name_target_encoded_kurtosis',
    'unique_id_month_name_target_encoded_kurtosis',
    'unique_id_L1_category_name_en_target_encoded_kurtosis',
    'unique_id_L2_category_name_en_target_encoded_kurtosis',
    'unique_id_L3_category_name_en_target_encoded_kurtosis',
    'unique_id_L4_category_name_en_target_encoded_kurtosis',
    'warehouse_holiday_name_target_encoded_kurtosis',
    'warehouse_month_name_target_encoded_kurtosis',
    'warehouse_L1_category_name_en_target_encoded_kurtosis',
    'warehouse_L2_category_name_en_target_encoded_kurtosis',
    'warehouse_L3_category_name_en_target_encoded_kurtosis',
    'warehouse_L4_category_name_en_target_encoded_kurtosis',
    'holiday_name_L3_category_name_en_target_encoded_kurtosis',
    'month_name_L3_category_name_en_target_encoded_kurtosis',
    'L1_category_name_en_L3_category_name_en_target_encoded_kurtosis',
    'L2_category_name_en_L3_category_name_en_target_encoded_kurtosis',
    'L3_category_name_en_L4_category_name_en_target_encoded_kurtosis',
    
    
    'unique_id_target_encoded_q25',
    'warehouse_target_encoded_q25',
    'holiday_name_target_encoded_q25',
    'month_name_target_encoded_q25',
    'L1_category_name_en_target_encoded_q25',
    'L2_category_name_en_target_encoded_q25',
    'L3_category_name_en_target_encoded_q25',
    'L4_category_name_en_target_encoded_q25',
    'unique_id_warehouse_target_encoded_q25',
    'unique_id_holiday_name_target_encoded_q25',
    'unique_id_month_name_target_encoded_q25',
    'unique_id_L1_category_name_en_target_encoded_q25',
    'unique_id_L2_category_name_en_target_encoded_q25',
    'unique_id_L3_category_name_en_target_encoded_q25',
    'unique_id_L4_category_name_en_target_encoded_q25',
    'warehouse_holiday_name_target_encoded_q25',
    'warehouse_month_name_target_encoded_q25',
    'warehouse_L1_category_name_en_target_encoded_q25',
    'warehouse_L2_category_name_en_target_encoded_q25',
    'warehouse_L3_category_name_en_target_encoded_q25',
    'warehouse_L4_category_name_en_target_encoded_q25',
    'holiday_name_month_name_target_encoded_q25',
    'holiday_name_L1_category_name_en_target_encoded_q25',
    'holiday_name_L2_category_name_en_target_encoded_q25',
    'holiday_name_L3_category_name_en_target_encoded_q25',
    'holiday_name_L4_category_name_en_target_encoded_q25',
    'month_name_L1_category_name_en_target_encoded_q25',
    'month_name_L2_category_name_en_target_encoded_q25',
    'month_name_L3_category_name_en_target_encoded_q25',
    'month_name_L4_category_name_en_target_encoded_q25',
    'L1_category_name_en_L2_category_name_en_target_encoded_q25',
    'L1_category_name_en_L3_category_name_en_target_encoded_q25',
    'L1_category_name_en_L4_category_name_en_target_encoded_q25',
    'L2_category_name_en_L3_category_name_en_target_encoded_q25',
    'L2_category_name_en_L4_category_name_en_target_encoded_q25',
    'L3_category_name_en_L4_category_name_en_target_encoded_q25',
    
    
    'unique_id_target_encoded_q75',
    'warehouse_target_encoded_q75',
    'holiday_name_target_encoded_q75',
    'month_name_target_encoded_q75',
    'L1_category_name_en_target_encoded_q75',
    'L2_category_name_en_target_encoded_q75',
    'L3_category_name_en_target_encoded_q75',
    'L4_category_name_en_target_encoded_q75',
    'unique_id_warehouse_target_encoded_q75',
    'unique_id_holiday_name_target_encoded_q75',
    'unique_id_month_name_target_encoded_q75',
    'unique_id_L1_category_name_en_target_encoded_q75',
    'unique_id_L2_category_name_en_target_encoded_q75',
    'unique_id_L3_category_name_en_target_encoded_q75',
    'unique_id_L4_category_name_en_target_encoded_q75',
    'warehouse_holiday_name_target_encoded_q75',
    'warehouse_month_name_target_encoded_q75',
    'warehouse_L1_category_name_en_target_encoded_q75',
    'warehouse_L2_category_name_en_target_encoded_q75',
    'warehouse_L3_category_name_en_target_encoded_q75',
    'warehouse_L4_category_name_en_target_encoded_q75',
    'holiday_name_month_name_target_encoded_q75',
    'holiday_name_L1_category_name_en_target_encoded_q75',
    'holiday_name_L2_category_name_en_target_encoded_q75',
    'holiday_name_L3_category_name_en_target_encoded_q75',
    'holiday_name_L4_category_name_en_target_encoded_q75',
    'month_name_L1_category_name_en_target_encoded_q75',
    'month_name_L2_category_name_en_target_encoded_q75',
    'month_name_L3_category_name_en_target_encoded_q75',
    'month_name_L4_category_name_en_target_encoded_q75',
    'L1_category_name_en_L2_category_name_en_target_encoded_q75',
    'L1_category_name_en_L3_category_name_en_target_encoded_q75',
    'L1_category_name_en_L4_category_name_en_target_encoded_q75',
    'L2_category_name_en_L3_category_name_en_target_encoded_q75',
    'L2_category_name_en_L4_category_name_en_target_encoded_q75',
    'L3_category_name_en_L4_category_name_en_target_encoded_q75',
    
    
    'warehouse_encoded',
    'is_warehouse_encoded_0',
    'is_warehouse_encoded_1',
    'is_warehouse_encoded_2',
    'is_warehouse_encoded_3',
    'is_warehouse_encoded_4',
    'is_warehouse_encoded_5',
    'is_warehouse_encoded_6',
    
    
    'total_orders',
    'sell_price_main',
    'original_price',
    'price_diff',
    
    
    'type_0_discount',
    'type_1_discount',
    'type_2_discount',
    'type_3_discount',
    'type_4_discount',
    'type_5_discount',
    'type_6_discount',
    'max_discount',
    'has_discount',
    
    
    'year_sin',
    'month_sin',
    'day_sin',
    'week_sin',
    'day_year_sin',
    'day_of_week_sin',
    'group',
    'is_weekend',
    'is_month_start',
    'is_month_end',
    'quarter',
    
    
    'days_to_next_holiday',
    'days_from_prev_holiday',
    'shops_closed',
    'winter_school_holidays',
    'school_holidays',
    'day_after_closed_day',
    'second_closed_day',
    'date_days_to_shops_closed',
]


class WMAELGBM:
    def __init__(self,
                 model_params=None,
                 features=[],
                 weight_map={},
                 distf='norm',
                 ):
        self.model_params = model_params
        self.model = None
        self.features = features
        self.weight_map = weight_map
        self.distf = distf
        self.r_smoothing = lambda x:x
    
    def fit(self, X, y):
        weights = np.array([self.weight_map[uid] for uid in X['unique_id']])
        X = X[self.features].to_numpy()
        y = y.to_numpy()
        
        over_m = None
        if self.distf=='log-norm':
            over_m = LinearRegression()
            y = smoothing(y)
            self.r_smoothing = r_smoothing
        elif self.distf=='norm':
            over_m = LinearRegression()
            y = y
        elif self.distf=='gamma':
            over_m = GammaRegressor()
            y = y
        
        def wmae_feval(y_pred, data):
            y_true = data.get_label()
            weights = data.get_weight()
            error = np.abs(y_true - y_pred)
            weighted_error = np.sum(weights * error)
            normalized_error = weighted_error / np.sum(weights)
            return 'wmae', normalized_error, False
        
        
        
        lgb_train = lgb.Dataset(X, y,
                                weight=weights
                               )

        self.model = lgb.train(
            self.model_params,
            lgb_train,
            feval=wmae_feval
        )
        
    def predict(self, X):
        X = X[self.features].to_numpy()
        return self.r_smoothing(self.model.predict(X))

class WMAELGBM_WH:
    def __init__(self,
                 model_params=None,
                 features=[],
                 weight_map={},
                 distf='norm',
                 ):
        self.model_params = model_params
        self.model_dict = {}
        self.features = features
        self.weight_map = weight_map
        self.distf = distf
        self.r_smoothing = lambda x: x
    
    def fit(self, X, y):
        for warehouse_id in X['warehouse_encoded'].unique():
            print(f"Training model for warehouse {warehouse_id}...")
            X_warehouse = X[X['warehouse_encoded'] == warehouse_id]
            y_warehouse = y[X['warehouse_encoded'] == warehouse_id]
            weights = np.array([self.weight_map[uid] for uid in X_warehouse['unique_id']])
            X_warehouse = X_warehouse[self.features].to_numpy()
            y_warehouse = y_warehouse.to_numpy()
            
            over_m = None
            if self.distf == 'log-norm':
                over_m = LinearRegression()
                y_warehouse = self.smoothing(y_warehouse)
                self.r_smoothing = self.r_smoothing
            elif self.distf == 'norm':
                over_m = LinearRegression()
                y_warehouse = y_warehouse
            elif self.distf == 'gamma':
                over_m = GammaRegressor()
                y_warehouse = y_warehouse
            
            def wmae_feval(y_pred, data):
                y_true = data.get_label()
                weights = data.get_weight()
                error = np.abs(y_true - y_pred)
                weighted_error = np.sum(weights * error)
                normalized_error = weighted_error / np.sum(weights)
                return 'wmae', normalized_error, False

            lgb_train = lgb.Dataset(X_warehouse, y_warehouse, weight=weights)

            model = lgb.train(
                self.model_params,
                lgb_train,
                feval=wmae_feval
            )
            self.model_dict[warehouse_id] = model
    
    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            warehouse_id = row['warehouse_encoded']
            model = self.model_dict.get(warehouse_id)
            
            if model is not None:
                X_warehouse = row[self.features].to_numpy().reshape(1, -1)
                predictions.append(self.r_smoothing(model.predict(X_warehouse))[0])
            else:
                predictions.append(0)
        
        return np.array(predictions)


n_splits = 10
tscv = TimeSeriesSplit(n_splits=n_splits)

validation_errors = []

model_params = {
    'verbosity': -1,
    'objective': 'tweedie',
    'tweedie_variance_power': 1.1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'n_estimators': 750,
    'max_depth': -1,
    'min_child_samples': 64,
    'min_split_gain': 0.0,
    'reg_alpha': 0.05,
    'reg_lambda': 0.05,
    'feature_fraction': 0.7,
    'n_jobs': -1,
    'seed': R,
}

sales_train = sales_train.sort_values(by='date')
X = sales_train[['unique_id'] + model_feaures + ['date']]
y = sales_train['sales']

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X, y)):
    print(f"Processing fold {fold_idx + 1}/{n_splits}...")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    print(f"Train MIN:{X_train['date'].min()} -> MAX:{X_train['date'].max()}")
    print(f"Validation MIN:{X_val['date'].min()} -> MAX:{X_val['date'].max()}")
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = WMAELGBM_WH(
        model_params=model_params,
        weight_map=weight_map,
        distf='norm',
        features=model_feaures,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    wmae = calculate_wmae(X_val, y_val, y_pred, weight_map)

    validation_errors.append(wmae)
    print(f"Fold {fold_idx + 1} WMAE: {wmae:.4f}")


mean_wmae = np.mean(validation_errors)
print(f"\nAverage WMAE across folds: {mean_wmae:.4f}")


plt.figure(figsize=(10, 6))
plt.plot(range(1, n_splits + 1), validation_errors, marker='o')
plt.title('Validation Error Across Folds', fontsize=16)
plt.xlabel('Fold Index', fontsize=14)
plt.ylabel('WMAE', fontsize=14)
plt.grid()
plt.show()


def evaluate_by_warehouse(model, X_val, y_val, weight_map, warehouses):
    warehouse_errors = {}

    for warehouse in warehouses:
        mask = X_val['warehouse_encoded'] == warehouse
        X_warehouse = X_val[mask]
        y_warehouse = y_val[mask]

        if len(X_warehouse) == 0:
            print(f"No validation data for warehouse: {warehouse}")
            continue

        y_pred = model.predict(X_warehouse)

        wmae = calculate_wmae(X_warehouse, y_warehouse, y_pred, weight_map)
        warehouse_errors[warehouse] = wmae

        print(f"Warehouse: {warehouse}, WMAE: {wmae:.4f}, COUNT: {X_warehouse.shape[0]}")

    return warehouse_errors

warehouses = sales_train['warehouse_encoded'].unique()

warehouse_errors = evaluate_by_warehouse(
    model=model, 
    X_val=X_val, 
    y_val=y_val, 
    weight_map=weight_map, 
    warehouses=warehouses
)

plt.figure(figsize=(12, 6))
plt.bar(warehouse_errors.keys(), warehouse_errors.values(), color='skyblue')
plt.title("WMAE by Warehouse", fontsize=16)
plt.xlabel("Warehouse", fontsize=14)
plt.ylabel("WMAE", fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



X_train, X_val, y_train, y_val = prepare_to_train(sales_train, days=14)


model = WMAELGBM_WH(
        model_params={          
            'verbosity': -1,
            'objective': 'tweedie',
            'tweedie_variance_power': 1.1,
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'n_estimators': 750,
            'max_depth': -1,
            'min_child_samples': 64,
            'min_split_gain': 0.0,
            'reg_alpha': 0.05,
            'reg_lambda': 0.05,
            'feature_fraction': 0.7,
            'n_jobs': -1,
            'seed': R,
        },
        weight_map=weight_map,
        distf='norm',
        features=model_feaures,
)
model.fit(X_train, y_train)


wmae = calculate_wmae(X_val, y_val, model.predict(X_val), weight_map)

print(f"WMAE: {wmae:.4f}")


class PermutationFeatureImportance:
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def calculate_importance(self, X, y, weight_map, n_repeats=10, random_state=42):
        np.random.seed(random_state)
        base_metric = calculate_wmae(X, y, model.predict(X), weight_map)
        importances = {}

        for col in self.features:
            permuted_metrics = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                X_permuted[col] = np.random.permutation(X[col])
                permuted_metric = calculate_wmae(val_data, y, model.predict(X_permuted), weight_map)
                permuted_metrics.append(permuted_metric)
            importances[col] = np.mean(permuted_metrics) - base_metric
            
        importances_df = pd.DataFrame(
            sorted(importances.items(), key=lambda x: x[1], reverse=True),
            columns=["feature", "importance"]
        )
        return importances_df


# TODO: ENABLE LATER
#pfi = PermutationFeatureImportance(model=model, features=model_feaures)
# importances = pfi.calculate_importance(X_val, y_val, weight_map)

# importances.head(100)


X_train, X_val, y_train, y_val = prepare_to_train(sales_train, days=0)

model = WMAELGBM_WH(
        model_params={          
            'verbosity': -1,
            'objective': 'tweedie',
            'tweedie_variance_power': 1.1,
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'n_estimators': 750,
            'max_depth': -1,
            'min_child_samples': 64,
            'min_split_gain': 0.0,
            'reg_alpha': 0.05,
            'reg_lambda': 0.05,
            'feature_fraction': 0.7,
            'n_jobs': -1,
            'seed': R,
        },
        weight_map=weight_map,
        distf='norm',
        features=model_feaures,
)
model.fit(X_train, y_train)


predictions_test = model.predict(sales_test)
sales_test['sales_hat'] = predictions_test
sales_test['id'] = sales_test['unique_id'].astype(str) + '_' + sales_test['date'].dt.strftime('%Y-%m-%d')
submission = sales_test[['id', 'sales_hat']]
submission.head(n=11)


submission.to_csv('submission.csv', index=False)

