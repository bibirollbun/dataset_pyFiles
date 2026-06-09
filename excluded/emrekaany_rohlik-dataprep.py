# Data preprocessing and feature engineering is applied for the following notebooks.

import numpy as np  # Linear algebra operations
import pandas as pd  # Data processing, CSV file I/O (e.g., pd.read_csv)
import gc  # Garbage collection for memory management


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




# Load datasets
calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv')
inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv')
sales_train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv')
sales_test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv')
solution = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/solution.csv')
test_weights= pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv')


# Add an empty (NaN) target column to test set since it is missing
sales_test['sales'] = np.nan

# Combine train and test datasets to ensure consistent processing
combined_df = pd.concat([sales_train, sales_test], ignore_index=True)

# Convert date columns to datetime format for better handling
sales_train['date'] = pd.to_datetime(sales_train['date'])
sales_test['date'] = pd.to_datetime(sales_test['date'])

# Compute the minimum and maximum dates for train and test sets
train_min_date = sales_train['date'].min()
train_max_date = sales_train['date'].max()
test_min_date = sales_test['date'].min()
test_max_date = sales_test['date'].max()

# SonuÃ§larÄ± yazdÄ±rma
print("Sales Train - En KÃ¼Ã§Ã¼k Tarih:", train_min_date)
print("Sales Train - En BÃ¼yÃ¼k Tarih:", train_max_date)
print("Sales Test - En KÃ¼Ã§Ã¼k Tarih:", test_min_date)
print("Sales Test - En BÃ¼yÃ¼k Tarih:", test_max_date)
   


#Take all holidays
fixed_holidays = {
    'Germany': [
        ('01-01', 'New Year'),
        ('05-01', 'Labour Day'),
        ('10-03', 'German Unity Day'),
        ('12-24', 'Christmas Eve'),
        ('12-25', 'Christmas Day'),
        ('12-26', 'Boxing Day')
    ],
    'Czechia': [
        ('01-01', 'New Year'),
        ('05-01', 'Labour Day'),
        ('09-28', 'St. Wenceslas Day'),
        ('10-28', 'Czech Independence Day'),
        ('12-24', 'Christmas Eve'),
        ('12-25', 'Christmas Day'),
        ('12-26', 'Boxing Day')
    ],
    'Hungary': [
        ('01-01', 'New Year'),
        ('03-15', 'Revolution Day'),
        ('05-01', 'Labour Day'),
        ('08-20', 'St. Stephenâ€™s Day'),
        ('10-23', 'Republic Day'),
        ('12-24', 'Christmas Eve'),
        ('12-25', 'Christmas Day'),
        ('12-26', 'Boxing Day')
    ]
}

from datetime import date, timedelta
import dateutil.easter

years = list(range(2016, 2025))

# Initialize dictionary for dynamically calculated holidays
dynamic_holidays = {year: [] for year in years}

# Compute dynamic holidays such as Easter and Mother's Day
for year in years:
    # ğŸ“Œ Paskalya (Easter) tatilleri
    easter_sunday = dateutil.easter.easter(year)  # Easter Sunday
    good_friday = easter_sunday - timedelta(days=2)  # Good Friday
    easter_monday = easter_sunday + timedelta(days=1)  # Easter Monday
    holy_saturday = easter_sunday - timedelta(days=1)  # Holy Saturday

    # ğŸ“Œ Anneler GÃ¼nÃ¼ (MayÄ±s'Ä±n 2. Pazar gÃ¼nÃ¼)
    may_first = date(year, 5, 1)
    first_sunday_may = may_first + timedelta(days=(6 - may_first.weekday() + 7) % 7)
    mother_day = first_sunday_may + timedelta(days=7)  # 2. Pazar

    # ğŸ“Œ Tatilleri ekleyelim
    dynamic_holidays[year].extend([
        (good_friday.strftime('%Y-%m-%d'), 'Good Friday'),
        (easter_sunday.strftime('%Y-%m-%d'), 'Easter Day'),
        (easter_monday.strftime('%Y-%m-%d'), 'Easter Monday'),
        (holy_saturday.strftime('%Y-%m-%d'), 'Holy Saturday'),
        (mother_day.strftime('%Y-%m-%d'), 'Mother Day')
    ])


# Map warehouses to corresponding countries
warehouse_country_map = {
    'Frankfurt_1': 'Germany',
    'Munich_1': 'Germany',
    'Prague_1': 'Czechia',
    'Prague_2': 'Czechia',
    'Prague_3': 'Czechia',
    'Brno_1': 'Czechia',
    'Budapest_1': 'Hungary'
}


all_holidays = []

# ğŸ“Œ Her yÄ±l iÃ§in sabit tatilleri ekle
for year in years:
    for country, holidays in fixed_holidays.items():
        for date_suffix, holiday_name in holidays:
            date_str = f"{year}-{date_suffix}"  # Ã–rn: 2024-01-01
            for warehouse in [w for w, c in warehouse_country_map.items() if c == country]:
                all_holidays.append({
                    'warehouse': warehouse,
                    'date': date_str,
                    'holiday_name': holiday_name,
                    'holiday': 1,
                    'shops_closed': 1,
                    'winter_school_holidays': 1 if 'Christmas' in holiday_name else 0,
                    'school_holidays': 1 if holiday_name in ['Labour Day', 'St. Wenceslas Day'] else 0
                })

# ğŸ“Œ Her yÄ±l iÃ§in hareketli tatilleri ekle
for year, holidays in dynamic_holidays.items():
    for date_str, holiday_name in holidays:
        for country in fixed_holidays.keys():
            for warehouse in [w for w, c in warehouse_country_map.items() if c == country]:
                all_holidays.append({
                    'warehouse': warehouse,
                    'date': date_str,
                    'holiday_name': holiday_name,
                    'holiday': 1,
                    'shops_closed': 1,
                    'winter_school_holidays': 1 if 'Christmas' in holiday_name else 0,
                    'school_holidays': 1 if holiday_name in ['Labour Day', 'St. Wenceslas Day'] else 0
                })

# Create a DataFrame from the holiday list
holidays_df = pd.DataFrame(all_holidays)
print(holidays_df['holiday_name'].unique())


# Function to update calendar data with missing holidays
def update_calendar_holidays(calendar_df, holidays_df):
    """
    Update existing calendar with new holiday information, only when there's actually a holiday
    
    Parameters:
    calendar_df: Existing calendar DataFrame
    holidays_df: New holidays DataFrame with updated information
    
    Returns:
    Updated calendar DataFrame
    """
    # Tarihleri datetime formatÄ±na Ã§evir
    calendar_df['date'] = pd.to_datetime(calendar_df['date'])
    holidays_df['date'] = pd.to_datetime(holidays_df['date'])
    
    # Sadece gerÃ§ekten holiday olan gÃ¼nleri al
    valid_holidays = holidays_df[holidays_df['holiday'] == 1].copy()
    
    # GÃ¼ncellenmiÅŸ calendar'Ä± oluÅŸtur
    calendar_updated = calendar_df.copy()
    
    # GÃ¼ncellenecek kolonlar
    update_columns = ['holiday_name', 'holiday', 'shops_closed', 
                     'winter_school_holidays', 'school_holidays']
    
    # Her bir holiday iÃ§in gÃ¼ncelleme yap
    for _, holiday_row in valid_holidays.iterrows():
        mask = ((calendar_updated['warehouse'] == holiday_row['warehouse']) & 
                (calendar_updated['date'] == holiday_row['date']))
        
        if mask.any():
            for col in update_columns:
                calendar_updated.loc[mask, col] = holiday_row[col]
    
    # Tarihleri string formatÄ±na geri Ã§evir
    calendar_updated['date'] = calendar_updated['date'].dt.strftime('%Y-%m-%d')
    
    return calendar_updated

# GÃ¼ncellenmiÅŸ calendar'Ä± oluÅŸtur
calendar_updated = update_calendar_holidays(calendar, holidays_df)

# YapÄ±lan deÄŸiÅŸiklikleri kontrol et
changes = calendar_updated[
    (calendar_updated['holiday_name'] != calendar['holiday_name']) & 
    (calendar_updated['holiday_name'].notna())  # Sadece geÃ§erli holiday name'leri gÃ¶ster
]

calendar = update_calendar_holidays(calendar, holidays_df)

print(f"Toplam {len(changes)} satÄ±r gÃ¼ncellendi.")
print("\nÃ–rnek gÃ¼ncellemeler:")
print(changes[['warehouse', 'date', 'holiday_name', 'holiday']].head())

# DetaylÄ± kontrol iÃ§in
print("\nGÃ¼ncellenen tatil gÃ¼nleri daÄŸÄ±lÄ±mÄ±:")
print(changes['holiday_name'].value_counts().head())


print(calendar['holiday_name'].unique())



# --------------------------------------------------------
# ğŸ“Œ Holiday Mapping for Standardized Categorization
# --------------------------------------------------------

# Define a mapping for different holiday names into broader categories
holiday_mapping = {
    # ğŸ�„ Christmas & Related Holidays
    "Christmas Holiday": "Christmas",
    "Christmas Eve": "Christmas Eve",
    "1st Christmas Day": "Christmas",
    "2nd Christmas Day": "Christmas",
    "Christmas Day": "Christmas",
    "Boxing Day": "Boxing Day",

    # âœ�ï¸� Easter & Related Holidays
    "Easter Day": "Easter",
    "Easter Monday": "Easter",
    "Good Friday": "Easter",
    "Holy Saturday": "Easter",

    # ğŸ”¥ Whitsun (Pentecost)
    "Whit sunday": "Whitsun",
    "Whit monday": "Whitsun",

    # ğŸ•Šï¸� All Saints' Day
    "All Saints Day": "All Saints' Day",
    "All Saints' Day Holiday": "All Saints' Day",

    # ğŸ�† New Year
    "New Years Day": "New Year",
    "New Year": "New Year",

    # ğŸ› ï¸� Labour Day
    "Labour Day": "Labor Day",

    # ğŸ‡¨ğŸ‡¿ Czech Republic Holidays
    "Cyrila a Metodej": "Saints Cyril and Methodius Day",
    "Jan Hus": "Jan Hus Day",
    "St. Wenceslas Day": "Czech Statehood Day",
    "Den vzniku samostatneho ceskoslovenskeho statu": "Czechoslovakia Independence Day",
    "Czech Independence Day": "Czechoslovakia Independence Day",
    "Den boje za svobodu a demokracii": "Freedom and Democracy Day",
    "Den osvobozeni": "Victory Day",
    "Den ceske statnosti": "Czech Statehood Day",  # Eklenen mapping

    # ğŸ‡­ğŸ‡º Hungary Holidays
    "1848 Revolution Memorial Day (Extra holiday)": "Hungary Revolution Memorial Day",
    "Hungary National Day Holiday": "Hungary National Day",
    "State Foundation Day": "Hungary National Day",
    "Memorial day of the 1956 Revolution": "1956 Revolution Memorial Day",
    "Independent Hungary Day": "Hungarian Independence Day",
    "Memorial Day for the Victims of the Communist Dictatorships": "Communist Victims Memorial Day",
    
    # ğŸ‡©ğŸ‡ª Germany Holidays
    "German Unity Day": "German Unity Day",

    # ğŸŒ� Other Holidays
    "International womens day": "International Women's Day",
    "Ascension day": "Ascension Day",
    "Corpus Christi": "Corpus Christi",
    "Reformation Day": "Reformation Day",
    "Assumption of the Virgin Mary": "Assumption of Mary",
    "Epiphany": "Epiphany",
    "Mother Day": "Mother's Day",
    "Memorial Day of the Republic": "Republic Memorial Day",
    "Memorial Day for the Victims of the Holocaust": "Holocaust Memorial Day",
    "Memorial Day for the Martyrs of Arad": "Martyrs of Arad Memorial Day",
    "Day of National Unity": "National Unity Day",
    "National Defense Day": "National Defense Day",
    "Peace Festival in Augsburg": "Peace Festival",
    
    # Ek olarak; 
    "weekend": "weekend"  # "weekend" deÄŸeri de tek bir kategori altÄ±nda toplanacak.
}


# ğŸ“Œ Tatilleri haritalama iÅŸlemi
calendar["holiday_name_mapped"] = calendar["holiday_name"].map(holiday_mapping).fillna(calendar["holiday_name"])

# ğŸ“Œ Mapping sonrasÄ± benzersiz tatilleri kontrol et
print(calendar["holiday_name_mapped"].unique())




# --------------------------------------------------------
# ğŸ“Œ Handling Weekends as Holidays
# --------------------------------------------------------
calendar["date"] = pd.to_datetime(calendar["date"])
mask = (calendar["date"].dt.weekday >= 5) & calendar["holiday_name_mapped"].isna()
calendar.loc[mask, "holiday_name_mapped"] = "weekend"
print(calendar["holiday_name_mapped"].unique())


calendar['any_school_holiday'] = np.where(
    (calendar['winter_school_holidays'] + calendar['school_holidays']) > 0,
    1,
    0
)



# --------------------------------------------------------
# ğŸ“Œ Adding Country Information for Warehouses
# --------------------------------------------------------
calendar['country'] = calendar['warehouse'].map(warehouse_country_map)


calendar=calendar.drop(columns=['holiday_name', 'winter_school_holidays', 'school_holidays'])


# --------------------------------------------------------
# ğŸ“Œ Creating Shift Features for Holidays and Closures
# --------------------------------------------------------
calendar=calendar.sort_values(['date']).reset_index(drop=True)
for gap in [-1,1,2,3,4,5,6,7]:
        for col in ['any_school_holiday', 'holiday' , 'shops_closed']:
            calendar[col+f"_shift{gap}"]=calendar.groupby(['warehouse'])[col].shift(gap)


# Ã–rnek: calendar DataFrame'inde, aÅŸaÄŸÄ±daki Ã¶nekler iÃ§in iÅŸlemi yapalÄ±m:
prefixes = ['any_school_holiday', 'holiday', 'shops_closed']

for prefix in prefixes:
    # shift3'ten shift7'ye kadar olan sÃ¼tun isimlerini oluÅŸturuyoruz
    cols = [f"{prefix}_shift{i}" for i in range(3, 8)]
    # SÃ¼tunlarÄ±n hepsinin mevcut olduÄŸunu kontrol edelim
    existing_cols = [col for col in cols if col in calendar.columns]
    if existing_cols:
        # shift3 sÃ¼tunu varsa; yoksa oluÅŸturabiliriz
        shift3_col = f"{prefix}_shift3"
        if shift3_col in calendar.columns:
            # Mevcut shift3 deÄŸerine diÄŸer shift sÃ¼tunlarÄ±nÄ±n toplamÄ±nÄ± ekleyelim
            calendar[shift3_col] = calendar[existing_cols].sum(axis=1)
        else:
            # EÄŸer shift3 sÃ¼tunu yoksa, direkt toplamÄ± atayalÄ±m
            calendar[shift3_col] = calendar[existing_cols].sum(axis=1)
        # shift4'ten shift7'ye kadar olan sÃ¼tunlarÄ± sil
        cols_to_drop = [f"{prefix}_shift{i}" for i in range(4, 8) if f"{prefix}_shift{i}" in calendar.columns]
        calendar.drop(columns=cols_to_drop, inplace=True)



# --------------------------------------------------------
# ğŸ“Œ Merge Calendar, Inventory, and Weights into the Main Dataset
# --------------------------------------------------------

def data_frame_merge(df):
    dt=df.merge(inventory, on=['unique_id'], how='left')
    dt=dt.rename(columns={'warehouse_x': 'warehouse'})
    dt["date"] = pd.to_datetime(dt["date"])
    dt=dt.merge(calendar, on=['warehouse', 'date'], how='left')
    dt=dt.merge(test_weights, on=['unique_id'], how='left')
    

    return dt



train_merged  = data_frame_merge(combined_df)



# Free memory by deleting unused variables
del calendar
del inventory
del sales_train
del sales_test
del solution
del test_weights
del combined_df
del holidays_df
del calendar_updated
del changes
gc.collect()


null_count = train_merged.isnull().sum()
print(null_count)


import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def fill_complex_nulls_by_group_iterative(df, 
                                          group_cols=['warehouse', 'unique_id'], 
                                          date_col='date', 
                                          fill_cols=None,
                                          dist_cols=None,
                                          n_neighbors=5, 
                                          weights="uniform"):
    """
    Bu fonksiyon, df iÃ§indeki fill_cols listesinde belirtilen sayÄ±sal kolonlardaki (NaN) eksikleri,
    Ã¶nce belirtilen grup sÃ¼tunlarÄ±na gÃ¶re (Ã¶rn. warehouse, unique_id) gruplandÄ±rÄ±p, 
    her grup iÃ§inde date kolonuna gÃ¶re sÄ±raladÄ±ktan sonra, kolon kolon (iteratif) doldurur.
    
    Her doldurma iterasyonunda, imputation iÃ§in kullanÄ±lacak distance sÃ¼tunlarÄ±,
    dist_cols listesinde belirtilir. EÄŸer doldurulmasÄ± istenen kolon (Ã¶rneÄŸin 'sales'),
    aynÄ± anda distance sÃ¼tunlarÄ± arasÄ±nda yer alÄ±yorsa ve bu sÃ¼tunlardan herhangi birinde eksik deÄŸer varsa,
    o sÃ¼tun bu iterasyonda predictor setine dahil edilmez.
    
    Parametreler:
      - df: Ä°ÅŸlenecek DataFrame.
      - group_cols: Gruplama yapÄ±lacak sÃ¼tunlar (varsayÄ±lan ['warehouse', 'unique_id']).
      - date_col: Tarih bilgisini iÃ§eren sÃ¼tun adÄ± (varsayÄ±lan 'date').
      - fill_cols: DoldurulmasÄ± gereken kolonlarÄ±n listesi. EÄŸer None verilirse,
                   Ã¶rneÄŸin ['total_orders', 'sales', 'sell_price_main', 'availability'] kullanÄ±lÄ±r.
      - dist_cols: Mesafe hesaplamasÄ±nda kullanÄ±lacak sÃ¼tunlarÄ±n listesi. Ã–rneÄŸin: 
                   ['date', 'sell_price_main', 'total_orders', 'holiday'].
                   Burada 'date' yerine, imputation sÄ±rasÄ±nda 'date_numeric' kullanÄ±lacaktÄ±r.
      - n_neighbors: KNN'de kullanÄ±lacak komÅŸu sayÄ±sÄ±.
      - weights: KomÅŸu aÄŸÄ±rlÄ±klandÄ±rma yÃ¶ntemi ("uniform" veya "distance").
    """
    if fill_cols is None:
        fill_cols = ['availability', 'total_orders','sales' ]
    if dist_cols is None:
        dist_cols = ['date', 'sell_price_main', 'total_orders',  'availability','sales']
    
    # Ã–ncelikle DataFrame'i grup sÃ¼tunlarÄ± ve tarih sÃ¼tununa gÃ¶re sÄ±ralÄ±yoruz.
    df = df.sort_values(group_cols + [date_col]).copy()
    
    def impute_group(group):
        group = group.copy()
        # Grup iÃ§indeki 'date' sÃ¼tununu, en erken tarihten itibaren geÃ§en gÃ¼n sayÄ±sÄ±na Ã§evirip 'date_numeric' olarak ekleyelim.
        group['date_numeric'] = (group[date_col] - group[date_col].min()).dt.days
        
        # Her fill kolonu iÃ§in iterasyon (sÄ±rasÄ±yla doldurma)
        for col in fill_cols:
            missing_idx = group[group[col].isnull()].index
            if len(missing_idx) == 0:
                continue  # Bu kolonda eksik yoksa geÃ§
            
            # Ä°lk olarak, doldurulmasÄ± istenen kolonu distance sÃ¼tunlarÄ± arasÄ±ndan Ã§Ä±kartÄ±yoruz.
            predictors = [p for p in dist_cols if p != col]
            # 'date' sÃ¼tunu varsa, onun yerine 'date_numeric' kullanÄ±lacak.
            predictors = ['date_numeric' if p == 'date' else p for p in predictors]
            
            # Å�imdi, eÄŸer ilgili grup iÃ§inde herhangi bir predictor sÃ¼tununda (predictors listesinde)
            # eksik deÄŸer varsa, o sÃ¼tunu predictor setinden Ã§Ä±karÄ±yoruz.
            predictors = [p for p in predictors if not group[p].isnull().any()]
            
            # Ä°mputation iÃ§in kullanÄ±lacak sÃ¼tun setini belirleyelim: doldurulmasÄ± istenen kolon + predictors.
            impute_features = [col] + predictors
            # EÄŸer bazÄ± impute_features grupta yoksa, onlarÄ± Ã§Ä±karalÄ±m.
            impute_features = [feat for feat in impute_features if feat in group.columns]
            
            # Alt DataFrame oluÅŸturup, sadece bu sÃ¼tunlarÄ± seÃ§elim.
            data = group[impute_features]
            
            # Ã–lÃ§ekleme: TÃ¼m sÃ¼tunlarÄ± aynÄ± Ã¶lÃ§eÄŸe getiriyoruz.
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # KNN imputer ile eksik deÄŸerleri dolduralÄ±m.
            imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
            data_imputed_scaled = imputer.fit_transform(data_scaled)
            data_imputed = scaler.inverse_transform(data_imputed_scaled)
            
            # Ä°mputed veriyi DataFrame olarak elde edelim.
            data_imputed_df = pd.DataFrame(data_imputed, columns=impute_features, index=group.index)
            
            # Sadece doldurulmasÄ± istenen kolon iÃ§in imputed deÄŸerleri, eksik olan yerlerde gÃ¼ncelleyelim.
            group.loc[missing_idx, col] = data_imputed_df.loc[missing_idx, col]
        
        # Ä°ÅŸlem sonunda geÃ§ici eklenen 'date_numeric' sÃ¼tununu kaldÄ±rÄ±yoruz.
        group.drop(columns=['date_numeric'], inplace=True)
        return group
    
    # Gruplama iÅŸlemi: Her grup iÃ§in imputation uyguluyoruz.
    df_imputed = df.groupby(group_cols, group_keys=False).apply(impute_group)
    return df_imputed

# Ã–rnek kullanÄ±m:
# VarsayalÄ±m ki train_merged DataFrame'inizde 'warehouse', 'unique_id', 'date' gibi sÃ¼tunlar ve doldurulmasÄ± istenen
# sayÄ±sal kolonlar (Ã¶rneÄŸin 'total_orders', 'sales', 'sell_price_main', 'availability') mevcut.
fill_columns = ['availability', 'total_orders','sales' ]
group_columns = ['warehouse', 'unique_id']
date_column = 'date'
# Distance hesaplamasÄ±nda kullanÄ±lacak sÃ¼tunlar:
distance_columns = ['date', 'sell_price_main', 'total_orders', 'availability','sales']

# Fonksiyonu Ã§aÄŸÄ±rÄ±yoruz:
train_merged = fill_complex_nulls_by_group_iterative(train_merged, 
                                                     group_cols=group_columns, 
                                                     date_col=date_column, 
                                                     fill_cols=fill_columns, 
                                                     dist_cols=distance_columns, 
                                                     n_neighbors=5, 
                                                     weights="uniform")

# SonuÃ§: Her grup iÃ§inde, sÄ±ralamaya gÃ¶re (tarih) ve belirlenen distance sÃ¼tunlarÄ±na gÃ¶re
# eksik deÄŸerler, iteratif olarak (kolon bazÄ±nda) doldurulmuÅŸ olacaktÄ±r.



# I want to see outliers here
"""
from sklearn.ensemble import IsolationForest

# Sadece sayÄ±sal sÃ¼tunlarla Ã§alÄ±ÅŸ
numeric_cols = ['sales', 'total_orders', 'sell_price_main', 'availability']  # Ã–rnek
X = train_merged[numeric_cols].dropna()

# Isolation Forest modeli
iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
preds = iso_forest.fit_predict(X)

# AykÄ±rÄ± deÄŸerler (-1)
aykiri_veriler = train_merged[preds == -1]
"""


import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler

def detect_and_correct_anomalies(dataframe, columns_to_check):
    """
    Advanced anomaly detection and correction pipeline
    
    Parameters:
    - dataframe: Input DataFrame
    - columns_to_check: Numeric columns for anomaly detection
    
    Returns:
    - Processed DataFrame with anomaly corrections
    """
    
    # Create a copy of the DataFrame to prevent modifications
    df_processed = dataframe.copy()
    
    # 1. Prepare data for anomaly detection
    data_for_detection = df_processed[columns_to_check].copy()
    
    # Handle any remaining missing values
    data_for_detection.fillna(data_for_detection.median(), inplace=True)
    
    # 2. Standardize data for better anomaly detection
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_detection)
    
    # 3. Anomaly Detection using Isolation Forest
    iso_forest = IsolationForest(
        contamination=0.0001,  # 2% of data considered anomalous
        random_state=42,
        max_samples='auto',
        bootstrap=False
    )
    
    # Detect anomalies
    anomaly_labels = iso_forest.fit_predict(scaled_data)
    
    # 4. Add anomaly flag to DataFrame
    df_processed['is_anomaly'] = anomaly_labels == -1
    
    # 5. Detailed Anomaly Analysis
    def calculate_anomaly_statistics(df, columns):
        anomaly_stats = {}
        for col in columns:
            anomaly_stats[col] = {
                'total_anomalies': df[df['is_anomaly']][col].count(),
                'anomaly_percentage': (df['is_anomaly'].sum() / len(df)) * 100,
                'original_mean': df[col].mean(),
                'anomaly_mean': df[df['is_anomaly']][col].mean(),
                'original_std': df[col].std(),
                'anomaly_std': df[df['is_anomaly']][col].std()
            }
        return anomaly_stats
    
    anomaly_statistics = calculate_anomaly_statistics(df_processed, columns_to_check)
    
    # 6. Winsorization for Anomaly Correction
    def winsorize_columns(df, columns, limits=(0.05, 0.05)):
        df_winsorized = df.copy()
        for col in columns:
            df_winsorized[col] = winsorize(df_winsorized[col], limits=limits)
        return df_winsorized
    
    # Apply Winsorization
    df_winsorized = winsorize_columns(df_processed, columns_to_check)
    
    # 7. Correction Strategy
    # Option 1: Replace anomalies with winsorized values
    for col in columns_to_check:
        df_processed.loc[df_processed['is_anomaly'], col] = df_winsorized.loc[df_processed['is_anomaly'], col]
    
    # 8. Additional Validation
    def print_anomaly_report(stats):
        print("Anomaly Detection Report:")
        for col, stat in stats.items():
            print(f"\nColumn: {col}")
            print(f"Total Anomalies: {stat['total_anomalies']}")
            print(f"Anomaly Percentage: {stat['anomaly_percentage']:.2f}%")
            print(f"Original Mean: {stat['original_mean']:.2f}")
            print(f"Anomaly Mean: {stat['anomaly_mean']:.2f}")
    
    print_anomaly_report(anomaly_statistics)
    
    return df_processed

# Example Usage
columns_to_analyze = [
    'sales'
]

#outlier handling does not applied since it didn't improve results
# Apply anomaly detection and correction
train_merged_corrected = train_merged

"""
# Validation
print("\nOriginal Train DataFrame Shape:", train_merged.shape)
print("Corrected Train DataFrame Shape:", train_merged_corrected.shape)
print("\nAnomaly Detection Summary (Train):")
print("Total Anomalies Detected:", train_merged_corrected['is_anomaly'].sum())
"""

del train_merged
gc.collect()






# ğŸ“Œ Tarih formatÄ±nÄ± dÃ¼zelt ve sÄ±rala
train_merged_corrected['date'] = pd.to_datetime(train_merged_corrected['date'])
train_merged_corrected = train_merged_corrected.sort_values('date')


# Tarihi bileÅŸenlerine ayÄ±rma
train_merged_corrected['date'] = pd.to_datetime(train_merged_corrected['date'])
train_merged_corrected['year'] = train_merged_corrected['date'].dt.year
train_merged_corrected['month'] = train_merged_corrected['date'].dt.month
train_merged_corrected['day'] = train_merged_corrected['date'].dt.day
train_merged_corrected['day_of_week'] = train_merged_corrected['date'].dt.dayofweek
train_merged_corrected['is_weekend'] = train_merged_corrected['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
train_merged_corrected['week_of_year'] = train_merged_corrected['date'].dt.isocalendar().week
train_merged_corrected['day_of_year']=train_merged_corrected['date'].dt.dayofyear
train_merged_corrected['quarter'] = train_merged_corrected['date'].dt.quarter


# --------------------------------------------------------
# ğŸ“Œ Creating Cyclical Features Using Sine and Cosine Transformations
# --------------------------------------------------------
train_merged_corrected['month_sin'] = np.sin(2 * np.pi * train_merged_corrected['month'] / 12) 
train_merged_corrected['month_cos'] = np.cos(2 * np.pi * train_merged_corrected['month'] / 12)
train_merged_corrected['day_sin'] = np.sin(2 * np.pi * train_merged_corrected['day'] / 31)  
train_merged_corrected['day_cos'] = np.cos(2 * np.pi * train_merged_corrected['day'] / 31)
train_merged_corrected['sin_week']=np.sin(2*np.pi*train_merged_corrected['week_of_year']/52)
train_merged_corrected['cos_week']=np.cos(2*np.pi*train_merged_corrected['week_of_year']/52)
train_merged_corrected['sin_day_of_year']=np.sin(2*np.pi*train_merged_corrected['day_of_year']/365)
train_merged_corrected['cos_day_of_year']=np.cos(2*np.pi*train_merged_corrected['day_of_year']/365)
train_merged_corrected['day_of_week_sin'] = np.sin(
    2 * np.pi * train_merged_corrected['day_of_week'] / 7
)
train_merged_corrected['day_of_week_cos'] = np.cos(
    2 * np.pi * train_merged_corrected['day_of_week'] / 7
)
train_merged_corrected['quarter_sin'] = np.sin(
    2 * np.pi * (train_merged_corrected['quarter']) / 4
)
train_merged_corrected['quarter_cos'] = np.cos(
    2 * np.pi * (train_merged_corrected['quarter']) / 4
)






train_merged_corrected=train_merged_corrected.drop(columns='warehouse_y')
train_merged_corrected.info()





# --------------------------------------------------------
# ğŸ“Œ Time Series Trend Features (Lag & Rolling Mean Features)
# --------------------------------------------------------
train_merged_corrected = train_merged_corrected.sort_values(by=['unique_id', 'warehouse', 'date'])
def create_time_features(df, windows=[7,14,20,28,35,84,356], group_cols=['unique_id', 'warehouse'], target_col='sales'):
    """
    Create lag and rolling mean features for specified windows
    Parameters:
    df: pandas DataFrame
    windows: list of integers representing the window sizes
    group_cols: list of columns to group by
    target_col: column to create features from
    
    Returns:
    pandas DataFrame with new features added
    """
    result_df = df.copy()
    result_df = result_df.sort_values(by=['unique_id', 'warehouse', 'date'])
    # Create lag features
    for window in windows:
        result_df[f'{target_col}_lag_{window}'] = (
            result_df.groupby(group_cols)[target_col]
            .shift(window)
        )
    # Create rolling mean features
    for window in windows:
        result_df[f'{target_col}_rolling_mean_{window}'] = (
            result_df.groupby(group_cols)[target_col]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=list(range(len(group_cols))), drop=True)
        )
    return result_df
# Uygula
windows = [7,14,20,28,35,84,356]
train_merged_corrected = create_time_features(
    df=train_merged_corrected,
    windows=windows,
    group_cols=['unique_id', 'warehouse'],
    target_col='sales'
)





# ğŸ“Œ max sales
train_merged_corrected['max_sales'] = (
    train_merged_corrected.groupby(['unique_id', 'warehouse'])['sales']
    .transform('max')
)




discount_columns = ['type_0_discount', 'type_1_discount', 'type_2_discount', 'type_3_discount', 
                    'type_4_discount', 'type_5_discount', 'type_6_discount']

train_merged_corrected['max_discount'] = train_merged_corrected[discount_columns].max(axis=1)
train_merged_corrected['total_discount'] = train_merged_corrected[discount_columns].apply(
    lambda x: x[x >= 0].sum(), axis=1
)

# **Price Change (Fiyat DeÄŸiÅŸimi)**
train_merged_corrected['price_change'] = train_merged_corrected.groupby(['unique_id', 'warehouse'])['sell_price_main'].pct_change()



# ğŸ“Œ 'unique_id' kolonundaki alt Ã§izgiden Ã¶nceki kÄ±smÄ± Ã§Ä±kar
train_merged_corrected['base_unique_id'] = train_merged_corrected['product_unique_id'].astype(str).str.split('_').str[0]

# ğŸ“Œ Yeni 'base_unique_id' bazÄ±nda ortalama fiyat hesapla
average_price_df = (
    train_merged_corrected.groupby('base_unique_id')['sell_price_main']
    .mean()
    .reset_index()
    .rename(columns={'sell_price_main': 'avg_sell_price_main'})
)

# ğŸ“Œ Ana veri setine ortalama fiyatÄ± ekleme (Join iÅŸlemi)
train_merged_corrected = train_merged_corrected.merge(average_price_df, on='base_unique_id', how='left')



train_merged_corrected['product_name'] = train_merged_corrected['name'].str.split('_').str[0]



#deactivated because external data is prohibited
"""
import wbdata
import pandas as pd
from datetime import datetime

# 1. Warehouse sÃ¼tunundaki ÅŸehirleri Ã¼lkelere eÅŸleÅŸtirme
warehouse_to_country = {
    "Budapest_1": "Hungary",
    "Prague_2": "Czechia",
    "Brno_1": "Czechia",
    "Prague_1": "Czechia",
    "Prague_3": "Czechia",
    "Munich_1": "Germany",
    "Frankfurt_1": "Germany"
}
# Ãœlke isimlerini ekleyelim
train_merged_corrected["country"] = train_merged_corrected["warehouse"].map(warehouse_to_country)

# 2. DÃ¼nya BankasÄ± gÃ¶stergesi: GSYÄ°H (GDP)
indicator = {"NY.GDP.MKTP.CD": "GDP"}

# 3. GSYÄ°H verilerini Ã§ekme (convert_date kaldÄ±rÄ±ldÄ±)
gdp_data = wbdata.get_dataframe(indicator)

# 4. Ä°ndeksi sÄ±fÄ±rlayÄ±p 'country' ve 'date' sÃ¼tunlarÄ±nÄ± elde ediyoruz.
gdp_data.reset_index(inplace=True)  
# gdp_data sÃ¼tunlarÄ± Ã¶rneÄŸin: 'country', 'date', 'GDP' ÅŸeklinde olacak.

# 5. 'date' sÃ¼tununu datetime formatÄ±na Ã§evirip 'year' sÃ¼tununu oluÅŸturuyoruz.
gdp_data['date'] = pd.to_datetime(gdp_data['date'])
gdp_data['year'] = gdp_data['date'].dt.year

# 6. Gdp verilerinde kullanÄ±lan 'year' sÃ¼tununun tipini int64'e dÃ¶nÃ¼ÅŸtÃ¼relim.
gdp_data['year'] = gdp_data['year'].astype('int64')

# 7. Train veri setinde, her satÄ±r iÃ§in bir Ã¶nceki yÄ±lÄ± belirten "prev_year" sÃ¼tununu oluÅŸturuyoruz.
train_merged_corrected["prev_year"] = train_merged_corrected["year"] - 1

# 8. Train veri setindeki 'year' ve 'prev_year' sÃ¼tunlarÄ±nÄ±n tipini int64 yapalÄ±m.
train_merged_corrected["year"] = train_merged_corrected["year"].astype('int64')
train_merged_corrected["prev_year"] = train_merged_corrected["prev_year"].astype('int64')

# 9. GSYÄ°H verilerini, Ã¼lke ve "prev_year" bilgisi Ã¼zerinden birleÅŸtiriyoruz.
#    Merge iÅŸlemi sol tarafta "prev_year" ile, saÄŸ tarafta "year" sÃ¼tunu Ã¼zerinden gerÃ§ekleÅŸecek.
train_merged_corrected = train_merged_corrected.merge(
    gdp_data,
    how="left",
    left_on=["country", "prev_year"],
    right_on=["country", "year"]
)

# 10. Merge sonrasÄ± gelen GDP sÃ¼tununu "GDP_prev" olarak yeniden adlandÄ±rÄ±yoruz.
train_merged_corrected.rename(columns={"year_x": "year"}, inplace=True)
train_merged_corrected.rename(columns={"date_x": "date"}, inplace=True)

train_merged_corrected.drop(columns=[  "year_y"], errors="ignore", inplace=True)

# SonuÃ§ kontrolÃ¼:
print(train_merged_corrected[['warehouse', 'year', 'prev_year', 'country', 'GDP']].head())
"""




def create_statistical_features_optimized(df, target_col, group_cols=['unique_id', 'warehouse']):
    """
    Verilen veri seti Ã¼zerinde, group_cols bazÄ±nda target_col iÃ§in Ã§eÅŸitli istatistiksel Ã¶znitelikler oluÅŸturur.
    Tek bir groupby-aggregation ile hesaplanan Ã¶znitelikler, orijinal df'e merge edilir.

    Parameters:
      df: pandas DataFrame
      group_cols: gruplama iÃ§in kullanÄ±lacak sÃ¼tunlar
      target_col: istatistiklerin hesaplanacaÄŸÄ± hedef sÃ¼tun

    Returns:
      Yeni Ã¶zniteliklerle geniÅŸletilmiÅŸ pandas DataFrame
    """
    # Orijinal veri setini kopyalÄ±yoruz.
    result_df = df.copy()

    # 1) Tek seferde groupby-agg sÃ¶zlÃ¼ÄŸÃ¼
    agg_dict = {
        f'{target_col}_max': 'max',
        f'{target_col}_mean': 'mean',
        f'{target_col}_median': 'median',
        f'{target_col}_std': 'std',
        f'{target_col}_skew': 'skew',
        f'{target_col}_zero_ratio': lambda x: (x == 0).mean()
    }

    # 2) groupby ile Ã¶zet tablomuza Ã§eviriyoruz
    agg_df = (
        result_df
        .groupby(group_cols)[target_col]
        .agg(**agg_dict)
        .reset_index()
    )


    # 4) Orijinal df ile hesaplanan Ã¶zellikleri merge
    result_df = result_df.merge(agg_df, on=group_cols, how='left')

    return result_df

# Ã–rnek kullanÄ±m
for col in ['sales', 'total_orders', 'sell_price_main', 'total_discount']:
    train_merged_corrected = create_statistical_features_optimized(
        df=train_merged_corrected,
        target_col=col,
        group_cols=['unique_id', 'warehouse']
    )




def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type not in [object, 'category', 'datetime64[ns]']:
            c_min = df[col].min()
            c_max = df[col].max()

            # tamsayÄ± sÃ¼tunlar
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                # float sÃ¼tunlar
                df[col] = df[col].astype(np.float32)
                
        elif col_type == object:
            # EÄŸer gerÃ§ekte kategorik veya sayÄ±sal deÄŸilse, kategorik dÃ¶nÃ¼ÅŸtÃ¼rebilirsiniz
            # df[col] = df[col].astype('category')
            pass
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Bellek kullanÄ±mÄ±: {start_mem:.2f} MB -> {end_mem:.2f} MB")
    return df
train_merged_corrected=  reduce_mem_usage  (train_merged_corrected)




def create_statistical_features_optimized2(df, target_col, group_cols=['unique_id', 'warehouse','year']):
    """
    Verilen veri seti Ã¼zerinde, group_cols bazÄ±nda target_col iÃ§in Ã§eÅŸitli istatistiksel Ã¶znitelikler oluÅŸturur.
    Tek bir groupby-aggregation ile hesaplanan Ã¶znitelikler, orijinal df'e merge edilir.

    Parameters:
      df: pandas DataFrame
      group_cols: gruplama iÃ§in kullanÄ±lacak sÃ¼tunlar
      target_col: istatistiklerin hesaplanacaÄŸÄ± hedef sÃ¼tun

    Returns:
      Yeni Ã¶zniteliklerle geniÅŸletilmiÅŸ pandas DataFrame
    """
    # Orijinal veri setini kopyalÄ±yoruz.
    result_df = df.copy()

    # 1) Tek seferde groupby-agg sÃ¶zlÃ¼ÄŸÃ¼
    agg_dict = {
        f'{target_col}_yearly_max': 'max',
        f'{target_col}_yearly_min': 'min',
        f'{target_col}_yearly_mean': 'mean',
        f'{target_col}_yearly_median': 'median',
        f'{target_col}_yearly_std': 'std',
        f'{target_col}_yearly_skew': 'skew',
        f'{target_col}_yearly_q25': lambda x: x.quantile(0.25),
        f'{target_col}_yearly_q75': lambda x: x.quantile(0.75),
        f'{target_col}_yearly_zero_ratio': lambda x: (x == 0).mean()
    }

    # 2) groupby ile Ã¶zet tablomuza Ã§eviriyoruz
    agg_df = (
        result_df
        .groupby(group_cols)[target_col]
        .agg(**agg_dict)
        .reset_index()
    )

    # 3) CV (Coefficient of Variation) = std / mean
    #    mean=0 durumunda bÃ¶lme hatasÄ± almamak iÃ§in 0'larÄ± NaN'a Ã§eviriyoruz
    agg_df[f'{target_col}_yearly_cv'] = (
        agg_df[f'{target_col}_yearly_std'] / agg_df[f'{target_col}_yearly_mean'].replace(0, np.nan)
    )

    # 4) Orijinal df ile hesaplanan Ã¶zellikleri merge
    result_df = result_df.merge(agg_df, on=group_cols, how='left')

    return result_df

# Ã–rnek kullanÄ±m
for col in ['sales', 'total_orders', 'sell_price_main', 'availability', 'total_discount']:
    train_merged_corrected = create_statistical_features_optimized2(
        df=train_merged_corrected,
        target_col=col,
        group_cols=['unique_id', 'warehouse','year']
    )





# Tarih sÃ¼tununu datetime formatÄ±na Ã§eviriyoruz.
train_merged_corrected['date'] = pd.to_datetime(train_merged_corrected['date'])
train_merged_corrected['holiday_name'] = train_merged_corrected['holiday_name_mapped'].fillna("nan")
# GeÃ§ersiz tatil deÄŸerleri
invalid_values = ['weekend', 'nan']

def assign_holiday_columns_vectorized(group):
    # Her warehouse iÃ§in tarih sÄ±ralamasÄ±
    group = group.sort_values('date').reset_index(drop=True)
    # Sadece geÃ§erli tatilleri filtreleyelim
    valid = group.loc[(~group['holiday_name'].isin(invalid_values)) & (group['holiday_name'].notna())].copy()
    
    # EÄŸer geÃ§erli tatil yoksa, ilgili sÃ¼tunlarÄ± NaN ile dolduruyoruz.
    if valid.empty:
        group['next_5_days_holidays'] = np.nan
        group['next_15_days_holidays'] = np.nan
        group['prev_5_days_holidays'] = np.nan
        return group
    
    # --- SONRAKÄ° TATÄ°LLER Ä°Ã‡Ä°N ---
    # Arama iÃ§in: mevcut tarihe 1 gÃ¼n ekleyelim.
    group['search_date'] = group['date'] + pd.Timedelta(days=1)
    # merge_asof ile, search_date'e en yakÄ±n (ilerideki) geÃ§erli tatili buluyoruz.
    merged_next = pd.merge_asof(
        group.sort_values('search_date'),
        valid.sort_values('date'),
        left_on='search_date',
        right_on='date',
        direction='forward',
        suffixes=('', '_hol')
    )
    merged_next = merged_next.sort_index()
    
    # VektÃ¶rize koÅŸullu atamalar:

    group['next_5_days_holidays'] = np.where(
        merged_next['date_hol'] <= group['date'] + pd.Timedelta(days=5),
        merged_next['holiday_name_hol'], np.nan
    )
    group['next_15_days_holidays'] = np.where(
        merged_next['date_hol'] <= group['date'] + pd.Timedelta(days=15),
        merged_next['holiday_name_hol'], np.nan
    )
    
    # --- Ã–NCEKÄ° TATÄ°LLER Ä°Ã‡Ä°N ---
    # Arama iÃ§in: mevcut tarihten 1 gÃ¼n Ã§Ä±karalÄ±m.
    group['search_date_prev'] = group['date'] - pd.Timedelta(days=1)
    # merge_asof ile, search_date_prev'e en yakÄ±n (gerideki) geÃ§erli tatili buluyoruz.
    merged_prev = pd.merge_asof(
        group.sort_values('search_date_prev'),
        valid.sort_values('date'),
        left_on='search_date_prev',
        right_on='date',
        direction='backward',
        suffixes=('', '_hol')
    )
    merged_prev = merged_prev.sort_index()
    

    group['prev_5_days_holidays'] = np.where(
        merged_prev['date_hol'] >= group['date'] - pd.Timedelta(days=5),
        merged_prev['holiday_name_hol'], np.nan
    )

    
    # GeÃ§ici sÃ¼tunlarÄ± temizliyoruz.
    group.drop(columns=['search_date', 'search_date_prev'], inplace=True)
    
    return group

# Her warehouse iÃ§in gruplandÄ±rarak vektÃ¶rize iÅŸlemi uyguluyoruz.
train_merged_corrected = train_merged_corrected.groupby('warehouse', group_keys=False).apply(assign_holiday_columns_vectorized)

# SonuÃ§larÄ± kontrol edelim.
print(train_merged_corrected.head())



# Ã–rnek pandemi dÃ¶nemlerini tanÄ±mlayan fonksiyon:
def get_pandemic_phase(date):
    """
    Bu fonksiyon, verilen date deÄŸeri iÃ§in pandemi dÃ¶nemini belirler.
    
    Ã–rnek dÃ¶nemler:
      - "pre-pandemic": 11 Mart 2020 Ã¶ncesi
      - "pandemic": 11 Mart 2020 - 31 AralÄ±k 2021
      - "post-pandemic": 1 Ocak 2022 - 31 MayÄ±s 2022
      - "healthy": 1 Haziran 2022 sonrasÄ±
      
    Bu dÃ¶nemler, analiz ihtiyaÃ§larÄ±nÄ±za gÃ¶re yeniden dÃ¼zenlenebilir.
    """
    if date < pd.Timestamp("2020-03-11"):
        return "pre-pandemic"
    elif date <= pd.Timestamp("2021-12-31"):
        return "pandemic"
    elif date <= pd.Timestamp("2022-05-31"):
        return "post-pandemic"
    else:
        return "healthy"

# Tarih sÃ¼tununu zaten datetime formatÄ±na Ã§evirdiÄŸinizi varsayarsak:
train_merged_corrected["pandemic_phase"] = train_merged_corrected["date"].apply(get_pandemic_phase)

# Ã–rnek olarak ilk 5 satÄ±rÄ± gÃ¶rÃ¼ntÃ¼leyelim:
print(train_merged_corrected[["date", "pandemic_phase"]].head())


# Reset index to ensure uniqueness
train_merged_corrected = train_merged_corrected.reset_index(drop=True)
train_merged_corrected = train_merged_corrected.sort_values(['warehouse','date'])

# Now, create the 'next_shops_closed_date' column:
train_merged_corrected['next_shops_closed_date'] = train_merged_corrected.loc[train_merged_corrected['shops_closed'] == 1, 'date'].shift(-1)
train_merged_corrected['next_shops_closed_date'] = train_merged_corrected['next_shops_closed_date'].bfill()
train_merged_corrected['days_to_next_closed'] = (train_merged_corrected['next_shops_closed_date'] - train_merged_corrected['date']).dt.days

# Create a binary column indicating if the day immediately after a closed day (shops_closed==1) is open:
train_merged_corrected['day_after_closing'] = ((train_merged_corrected['shops_closed'] == 0) & (train_merged_corrected['shops_closed'].shift(1) == 1)).astype(int)


# Her grup iÃ§in flag hesaplama: 1500, 5200 ve 10000 eÅŸiÄŸine gÃ¶re holiday koÅŸullu ve koÅŸulsuz flag'lar
flag_df = train_merged_corrected.groupby(['warehouse', 'unique_id']).apply(
    lambda df: pd.Series({
        'high_sales_1500_holidaysiz': int((df['sales'] > 1500).any()),
        'high_sales_5200_holidaysiz': int((df['sales'] > 5200).any()),
        'high_sales_10000_holidaysiz': int((df['sales'] > 10000).any())
    })
).reset_index()

# Orijinal veri setine flag'larÄ± merge edelim
train_merged_corrected = train_merged_corrected.merge(flag_df, on=['warehouse', 'unique_id'], how='left')




columns_to_drop = [
    'unique_id', 'date'
]



del average_price_df
del flag_df
train_merged_corrected=train_merged_corrected.drop(columns=['holiday_name',
 'next_shops_closed_date', 'base_unique_id','day_after_closing', 'max_sales'])
gc.collect()


import pandas as pd
import numpy as np

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type not in [object, 'category', 'datetime64[ns]']:
            c_min = df[col].min()
            c_max = df[col].max()

            # tamsayÄ± sÃ¼tunlar
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                # float sÃ¼tunlar
                df[col] = df[col].astype(np.float32)
                
        elif col_type == object:
            # EÄŸer gerÃ§ekte kategorik veya sayÄ±sal deÄŸilse, kategorik dÃ¶nÃ¼ÅŸtÃ¼rebilirsiniz
            # df[col] = df[col].astype('category')
            pass
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Bellek kullanÄ±mÄ±: {start_mem:.2f} MB -> {end_mem:.2f} MB")
    return df
train_merged_corrected=  reduce_mem_usage  (train_merged_corrected)



# Export without index for easier reading later
train_merged_corrected.to_csv("train_merged_corrected.csv", index=False)


