!pip install lightgbm scikit-learn pandas numpy matplotlib seaborn -q


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import gc
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("Libraries imported successfully!")
print(f"LightGBM version: {lgb.__version__}")


# Memory management
def clear_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    print("âœ“ Memory cleared")

def get_memory_usage():
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"RAM: {mem_info.rss / 1024**3:.2f} GB")
    try:
        result = !nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
        if result:
            used, total = map(int, result[0].split(','))
            print(f"GPU: {used/1024:.2f} GB / {total/1024:.2f} GB")
    except:
        pass

get_memory_usage()


train_df = pd.read_csv('/kaggle/input/ctai-ctd-hackathon/train.csv')
test_df = pd.read_csv('/kaggle/input/ctai-ctd-hackathon/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"\nColumns: {train_df.columns.tolist()}")
train_df.head()


# Convert numeric columns to float32
print("Converting data types...")
all_numeric_cols = ['id', 'MW', 'SIZE_BUILDINGSIZE', 'NUMFLOORS', 'NUMROOMS', 'NUMBEDS',
                    'invoiceTotal', 'ExtendedQuantity', 'UnitPrice', 'ExtendedPrice', 
                    'REVISED_ESTIMATE', 'QtyShipped']

for col in all_numeric_cols:
    if col in train_df.columns:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').astype('float32')
    if col in test_df.columns:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').astype('float32')

print("âœ“ Data types converted")
get_memory_usage()


def parse_date(date_str):
    """Parse date strings with multiple formats"""
    if pd.isna(date_str) or date_str == '':
        return None
    formats = ['%m/%d/%Y %H:%M', '%m/%d/%Y %H:%S', '%d/%m/%Y %H:%M', 
               '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    try:
        return pd.to_datetime(date_str)
    except:
        return None

print("Date parser defined!")


def advanced_feature_engineering(df, is_train=True, train_stats=None):
    """
    Advanced feature engineering for supply chain time series prediction.
    Optimized for LightGBM with categorical feature support.
    """
    df = df.copy()
    stats = {} if train_stats is None else train_stats.copy()
    
    # 1. BASIC NUMERIC CONVERSION
    numeric_columns = ['MW', 'SIZE_BUILDINGSIZE', 'NUMFLOORS', 'NUMROOMS', 'NUMBEDS',
                      'invoiceTotal', 'ExtendedQuantity', 'UnitPrice', 
                      'ExtendedPrice', 'REVISED_ESTIMATE']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
    
    # 2. DATE PARSING & TEMPORAL FEATURES
    date_columns = ['invoiceDate', 'CONSTRUCTION_START_DATE', 'SUBSTANTIAL_COMPLETION_DATE']
    for col in date_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_date)
    
    if 'invoiceDate' in df.columns:
        df['invoice_year'] = df['invoiceDate'].dt.year.astype('float32')
        df['invoice_month'] = df['invoiceDate'].dt.month.astype('float32')
        df['invoice_quarter'] = df['invoiceDate'].dt.quarter.astype('float32')
        df['invoice_day'] = df['invoiceDate'].dt.day.astype('float32')
        df['invoice_day_of_week'] = df['invoiceDate'].dt.dayofweek.astype('float32')
        df['invoice_day_of_year'] = df['invoiceDate'].dt.dayofyear.astype('float32')
        df['invoice_week_of_year'] = df['invoiceDate'].dt.isocalendar().week.astype('float32')
        
        # Calendar flags
        df['is_weekend'] = (df['invoice_day_of_week'] >= 5).astype('int8')
        df['is_month_start'] = df['invoiceDate'].dt.is_month_start.astype('int8')
        df['is_month_end'] = df['invoiceDate'].dt.is_month_end.astype('int8')
        df['is_quarter_start'] = df['invoiceDate'].dt.is_quarter_start.astype('int8')
        df['is_quarter_end'] = df['invoiceDate'].dt.is_quarter_end.astype('int8')
        df['is_year_start'] = df['invoiceDate'].dt.is_year_start.astype('int8')
        df['is_year_end'] = df['invoiceDate'].dt.is_year_end.astype('int8')
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['invoice_month'] / 12).astype('float32')
        df['month_cos'] = np.cos(2 * np.pi * df['invoice_month'] / 12).astype('float32')
        df['dow_sin'] = np.sin(2 * np.pi * df['invoice_day_of_week'] / 7).astype('float32')
        df['dow_cos'] = np.cos(2 * np.pi * df['invoice_day_of_week'] / 7).astype('float32')
        df['doy_sin'] = np.sin(2 * np.pi * df['invoice_day_of_year'] / 365).astype('float32')
        df['doy_cos'] = np.cos(2 * np.pi * df['invoice_day_of_year'] / 365).astype('float32')
        df['quarter_sin'] = np.sin(2 * np.pi * df['invoice_quarter'] / 4).astype('float32')
        df['quarter_cos'] = np.cos(2 * np.pi * df['invoice_quarter'] / 4).astype('float32')
        
        # Season
        df['season'] = df['invoice_month'].map({
            12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
            6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3
        }).fillna(0).astype('int8')
    
    # 3. CONSTRUCTION PROJECT TIMELINE
    if 'CONSTRUCTION_START_DATE' in df.columns and 'SUBSTANTIAL_COMPLETION_DATE' in df.columns:
        df['construction_duration'] = (df['SUBSTANTIAL_COMPLETION_DATE'] - 
                                       df['CONSTRUCTION_START_DATE']).dt.days.astype('float32')
        if is_train:
            stats['duration_median'] = df['construction_duration'].median()
        df['construction_duration'] = df['construction_duration'].fillna(stats.get('duration_median', 365))
        df['is_long_project'] = (df['construction_duration'] > 730).astype('int8')
        df['is_short_project'] = (df['construction_duration'] < 365).astype('int8')
    
    if 'invoiceDate' in df.columns and 'CONSTRUCTION_START_DATE' in df.columns:
        df['days_since_start'] = (df['invoiceDate'] - df['CONSTRUCTION_START_DATE']).dt.days.astype('float32')
        df['days_since_start'] = df['days_since_start'].fillna(0)
        
        if 'construction_duration' in df.columns:
            df['project_phase_pct'] = (df['days_since_start'] / (df['construction_duration'] + 1)).astype('float32')
            df['project_phase_pct'] = df['project_phase_pct'].clip(0, 2)
            df['phase_early'] = (df['project_phase_pct'] < 0.25).astype('int8')
            df['phase_mid'] = ((df['project_phase_pct'] >= 0.25) & (df['project_phase_pct'] < 0.75)).astype('int8')
            df['phase_late'] = ((df['project_phase_pct'] >= 0.75) & (df['project_phase_pct'] <= 1.0)).astype('int8')
            df['phase_overrun'] = (df['project_phase_pct'] > 1.0).astype('int8')
    
    if 'invoiceDate' in df.columns and 'SUBSTANTIAL_COMPLETION_DATE' in df.columns:
        df['days_to_completion'] = (df['SUBSTANTIAL_COMPLETION_DATE'] - df['invoiceDate']).dt.days.astype('float32')
        if is_train:
            stats['completion_median'] = df['days_to_completion'].median()
        df['days_to_completion'] = df['days_to_completion'].fillna(stats.get('completion_median', 180))
        df['is_urgent'] = (df['days_to_completion'] < 30).astype('int8')
        df['is_near_completion'] = (df['days_to_completion'] < 90).astype('int8')
    
    # 4. FINANCIAL FEATURES
    if 'invoiceTotal' in df.columns:
        df['log_invoice_total'] = np.log1p(df['invoiceTotal'].fillna(0)).astype('float32')
    if 'UnitPrice' in df.columns:
        df['log_unit_price'] = np.log1p(df['UnitPrice'].fillna(0)).astype('float32')
        df['unit_price_squared'] = (df['UnitPrice'].fillna(0) ** 2).astype('float32')
    if 'ExtendedPrice' in df.columns:
        df['log_extended_price'] = np.log1p(df['ExtendedPrice'].fillna(0)).astype('float32')
    if 'ExtendedQuantity' in df.columns:
        df['log_extended_quantity'] = np.log1p(df['ExtendedQuantity'].fillna(0)).astype('float32')
        df['qty_squared'] = (df['ExtendedQuantity'].fillna(0) ** 2).astype('float32')
        df['qty_sqrt'] = np.sqrt(df['ExtendedQuantity'].fillna(0)).astype('float32')
    
    if 'invoiceTotal' in df.columns and 'ExtendedPrice' in df.columns:
        df['price_to_invoice_ratio'] = (df['ExtendedPrice'].fillna(0) / (df['invoiceTotal'].fillna(0) + 1)).astype('float32')
        df['price_pct_of_invoice'] = (df['ExtendedPrice'].fillna(0) / (df['invoiceTotal'].fillna(1)) * 100).clip(0, 100).astype('float32')
    
    if 'UnitPrice' in df.columns and 'ExtendedQuantity' in df.columns:
        df['calculated_extended'] = (df['UnitPrice'].fillna(0) * df['ExtendedQuantity'].fillna(0)).astype('float32')
        if 'ExtendedPrice' in df.columns:
            df['price_calc_error'] = np.abs(df['ExtendedPrice'].fillna(0) - df['calculated_extended']).astype('float32')
    
    if 'REVISED_ESTIMATE' in df.columns:
        df['log_revised_estimate'] = np.log1p(df['REVISED_ESTIMATE'].fillna(0)).astype('float32')
        if 'SIZE_BUILDINGSIZE' in df.columns:
            df['cost_per_sqft'] = (df['REVISED_ESTIMATE'].fillna(0) / (df['SIZE_BUILDINGSIZE'].fillna(1) + 1)).astype('float32')
    
    # 5. BUILDING CHARACTERISTICS
    if 'SIZE_BUILDINGSIZE' in df.columns:
        df['log_building_size'] = np.log1p(df['SIZE_BUILDINGSIZE'].fillna(0)).astype('float32')
        df['building_size_squared'] = (df['SIZE_BUILDINGSIZE'].fillna(0) ** 2).astype('float32')
        if is_train:
            stats['size_25'] = df['SIZE_BUILDINGSIZE'].quantile(0.25)
            stats['size_75'] = df['SIZE_BUILDINGSIZE'].quantile(0.75)
        df['is_small_building'] = (df['SIZE_BUILDINGSIZE'] < stats.get('size_25', 50000)).astype('int8')
        df['is_large_building'] = (df['SIZE_BUILDINGSIZE'] > stats.get('size_75', 300000)).astype('int8')
    
    if 'NUMFLOORS' in df.columns:
        df['NUMFLOORS'] = df['NUMFLOORS'].fillna(0)
        df['has_floors'] = (df['NUMFLOORS'] > 0).astype('int8')
        df['is_highrise'] = (df['NUMFLOORS'] > 10).astype('int8')
        df['is_lowrise'] = ((df['NUMFLOORS'] > 0) & (df['NUMFLOORS'] <= 3)).astype('int8')
        df['log_floors'] = np.log1p(df['NUMFLOORS']).astype('float32')
    
    if 'NUMROOMS' in df.columns:
        df['NUMROOMS'] = df['NUMROOMS'].fillna(0)
        df['has_rooms'] = (df['NUMROOMS'] > 0).astype('int8')
        df['log_rooms'] = np.log1p(df['NUMROOMS']).astype('float32')
    
    if 'NUMBEDS' in df.columns:
        df['NUMBEDS'] = df['NUMBEDS'].fillna(0)
        df['has_beds'] = (df['NUMBEDS'] > 0).astype('int8')
        df['is_healthcare'] = (df['NUMBEDS'] > 0).astype('int8')
        df['log_beds'] = np.log1p(df['NUMBEDS']).astype('float32')
    
    if all(col in df.columns for col in ['NUMFLOORS', 'NUMROOMS', 'NUMBEDS', 'SIZE_BUILDINGSIZE']):
        df['building_complexity'] = (
            df['NUMFLOORS'].fillna(0) * 10 + 
            df['NUMROOMS'].fillna(0) * 0.1 + 
            df['NUMBEDS'].fillna(0) * 0.5 +
            np.log1p(df['SIZE_BUILDINGSIZE'].fillna(0))
        ).astype('float32')
    
    # 6. TEXT FEATURES
    if 'ItemDescription' in df.columns:
        desc = df['ItemDescription'].fillna('').astype(str)
        df['item_desc_length'] = desc.str.len().astype('int16')
        df['item_word_count'] = desc.str.split().str.len().fillna(0).astype('int16')
        df['item_char_per_word'] = (df['item_desc_length'] / (df['item_word_count'] + 1)).astype('float32')
        df['item_has_number'] = desc.str.contains(r'\d').astype('int8')
        df['item_has_inch'] = desc.str.contains('"').astype('int8')
        df['item_has_dimension'] = desc.str.contains(r'\d+["x]').astype('int8')
        df['item_has_fraction'] = desc.str.contains(r'\d/\d').astype('int8')
        df['item_has_gauge'] = desc.str.contains(r'\d+ga', case=False).astype('int8')
        df['item_has_mil'] = desc.str.contains(r'\d+mil', case=False).astype('int8')
        df['is_steel'] = desc.str.contains('steel|stud|flange', case=False).astype('int8')
        df['is_wood'] = desc.str.contains('wood|lumber|plywood', case=False).astype('int8')
        df['is_concrete'] = desc.str.contains('concrete|cement|masonry', case=False).astype('int8')
        df['is_electrical'] = desc.str.contains('wire|electrical|cable|volt', case=False).astype('int8')
        df['is_plumbing'] = desc.str.contains('pipe|plumb|drain|water', case=False).astype('int8')
        df['is_hardware'] = desc.str.contains('screw|bolt|nail|fastener|clip', case=False).astype('int8')
        df['is_insulation'] = desc.str.contains('insul|foam|gasket', case=False).astype('int8')
        df['has_large_size'] = desc.str.contains(r'\b(1[0-9]|20)[\'"\s]', case=False).astype('int8')
        df['has_small_size'] = desc.str.contains(r'\b[1-4][\'"\s/]', case=False).astype('int8')
    
    # 7. FREQUENCY ENCODING
    categorical_cols = ['PROJECT_CITY', 'STATE', 'PROJECT_COUNTRY', 'CORE_MARKET', 
                       'PROJECT_TYPE', 'UOM', 'PriceUOM']
    for col in categorical_cols:
        if col in df.columns:
            if is_train:
                stats[f'{col}_freq'] = df[col].value_counts(normalize=True).to_dict()
            df[f'{col}_freq'] = df[col].map(stats.get(f'{col}_freq', {})).fillna(0).astype('float32')
    
    # 8. PROJECT AGGREGATIONS
    if 'PROJECTNUMBER' in df.columns:
        if 'invoiceDate' in df.columns:
            df = df.sort_values(['PROJECTNUMBER', 'invoiceDate']).reset_index(drop=True)
        
        project_counts = df.groupby('PROJECTNUMBER').size().reset_index(name='project_invoice_count')
        df = df.merge(project_counts, on='PROJECTNUMBER', how='left')
        df['project_invoice_count'] = df['project_invoice_count'].astype('int32')
        df['invoice_seq_in_project'] = df.groupby('PROJECTNUMBER').cumcount().astype('int32')
        df['invoice_seq_pct'] = (df['invoice_seq_in_project'] / (df['project_invoice_count'] + 1)).astype('float32')
        df['is_first_invoice'] = (df['invoice_seq_in_project'] == 0).astype('int8')
        df['is_last_invoice'] = (df['invoice_seq_in_project'] == df['project_invoice_count'] - 1).astype('int8')
        
        if 'invoiceTotal' in df.columns:
            project_stats = df.groupby('PROJECTNUMBER')['invoiceTotal'].agg(['mean', 'sum', 'std', 'min', 'max']).reset_index()
            project_stats.columns = ['PROJECTNUMBER', 'project_avg_invoice', 'project_total_invoice', 
                                    'project_std_invoice', 'project_min_invoice', 'project_max_invoice']
            for col in project_stats.columns[1:]:
                project_stats[col] = project_stats[col].astype('float32')
            df = df.merge(project_stats, on='PROJECTNUMBER', how='left')
            df['invoice_vs_project_avg'] = (df['invoiceTotal'] / (df['project_avg_invoice'] + 1)).astype('float32')
        
        if 'ExtendedPrice' in df.columns:
            price_stats = df.groupby('PROJECTNUMBER')['ExtendedPrice'].agg(['mean', 'sum', 'std']).reset_index()
            price_stats.columns = ['PROJECTNUMBER', 'project_avg_price', 'project_total_price', 'project_std_price']
            for col in price_stats.columns[1:]:
                price_stats[col] = price_stats[col].astype('float32')
            df = df.merge(price_stats, on='PROJECTNUMBER', how='left')
        
        if 'ExtendedQuantity' in df.columns:
            qty_stats = df.groupby('PROJECTNUMBER')['ExtendedQuantity'].agg(['mean', 'sum', 'std']).reset_index()
            qty_stats.columns = ['PROJECTNUMBER', 'project_avg_qty', 'project_total_qty', 'project_std_qty']
            for col in qty_stats.columns[1:]:
                qty_stats[col] = qty_stats[col].astype('float32')
            df = df.merge(qty_stats, on='PROJECTNUMBER', how='left')
    
    # 9. TIME SERIES LAG FEATURES
    if 'PROJECTNUMBER' in df.columns and 'invoiceDate' in df.columns:
        for lag in [1, 2, 3]:
            if 'ExtendedQuantity' in df.columns:
                df[f'qty_lag_{lag}'] = df.groupby('PROJECTNUMBER')['ExtendedQuantity'].shift(lag).astype('float32')
            if 'ExtendedPrice' in df.columns:
                df[f'price_lag_{lag}'] = df.groupby('PROJECTNUMBER')['ExtendedPrice'].shift(lag).astype('float32')
            if 'invoiceTotal' in df.columns:
                df[f'invoice_lag_{lag}'] = df.groupby('PROJECTNUMBER')['invoiceTotal'].shift(lag).astype('float32')
        
        lag_cols = [col for col in df.columns if '_lag_' in col]
        df[lag_cols] = df[lag_cols].fillna(0)
        
        if 'ExtendedQuantity' in df.columns:
            df['qty_rolling_mean_3'] = df.groupby('PROJECTNUMBER')['ExtendedQuantity'].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            ).astype('float32')
            df['qty_rolling_std_3'] = df.groupby('PROJECTNUMBER')['ExtendedQuantity'].transform(
                lambda x: x.rolling(3, min_periods=1).std()
            ).fillna(0).astype('float32')
        
        if 'ExtendedPrice' in df.columns:
            df['price_rolling_mean_3'] = df.groupby('PROJECTNUMBER')['ExtendedPrice'].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            ).astype('float32')
        
        if 'ExtendedQuantity' in df.columns:
            df['qty_cumsum'] = df.groupby('PROJECTNUMBER')['ExtendedQuantity'].cumsum().astype('float32')
        if 'ExtendedPrice' in df.columns:
            df['price_cumsum'] = df.groupby('PROJECTNUMBER')['ExtendedPrice'].cumsum().astype('float32')
    
    # 10. GLOBAL TIME TRENDS
    if 'invoiceDate' in df.columns:
        if is_train:
            stats['min_date'] = df['invoiceDate'].min()
        df['days_since_dataset_start'] = (df['invoiceDate'] - stats.get('min_date', df['invoiceDate'].min())).dt.days.astype('float32')
        df['days_since_dataset_start'] = df['days_since_dataset_start'].fillna(0)
        if is_train:
            stats['max_days'] = df['days_since_dataset_start'].max()
        df['time_normalized'] = (df['days_since_dataset_start'] / (stats.get('max_days', 1) + 1)).astype('float32')
    
    # 11. CROSS-FEATURE INTERACTIONS
    if 'CORE_MARKET' in df.columns and 'PROJECT_TYPE' in df.columns:
        df['market_type_combo'] = (df['CORE_MARKET'].astype(str) + '_' + df['PROJECT_TYPE'].astype(str))
        if is_train:
            stats['market_type_freq'] = df['market_type_combo'].value_counts(normalize=True).to_dict()
        df['market_type_freq'] = df['market_type_combo'].map(stats.get('market_type_freq', {})).fillna(0).astype('float32')
        df = df.drop('market_type_combo', axis=1)
    
    if 'SIZE_BUILDINGSIZE' in df.columns and 'project_phase_pct' in df.columns:
        df['size_phase_interaction'] = (df['SIZE_BUILDINGSIZE'].fillna(0) * df['project_phase_pct']).astype('float32')
    
    if 'UnitPrice' in df.columns and 'days_to_completion' in df.columns:
        df['price_urgency'] = (df['UnitPrice'].fillna(0) / (df['days_to_completion'].fillna(1) + 1)).astype('float32')
    
    return df, stats

print("Advanced feature engineering function defined!")


# Apply feature engineering
print("Applying advanced feature engineering...")
train_df_fe, train_stats = advanced_feature_engineering(train_df, is_train=True)
print(f"Train FE complete: {train_df_fe.shape}")

test_df_fe, _ = advanced_feature_engineering(test_df, is_train=False, train_stats=train_stats)
print(f"Test FE complete: {test_df_fe.shape}")

print(f"\nNew features created: {len(train_df_fe.columns) - len(train_df.columns)}")

del train_df, test_df
clear_memory()
get_memory_usage()


# Define feature columns
exclude_cols = ['id', 'MasterItemNo', 'QtyShipped', 'invoiceDate', 
                'CONSTRUCTION_START_DATE', 'SUBSTANTIAL_COMPLETION_DATE',
                'PROJECTNUMBER', 'invoiceId', 'ItemDescription', 'MW']

numeric_cols = train_df_fe.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in exclude_cols]

# Label encode categorical columns
label_encoders = {}
categorical_cols = ['PROJECT_CITY', 'STATE', 'PROJECT_COUNTRY', 'CORE_MARKET', 
                   'PROJECT_TYPE', 'UOM', 'PriceUOM']

for col in categorical_cols:
    if col in train_df_fe.columns:
        le = LabelEncoder()
        train_df_fe[f'{col}_encoded'] = le.fit_transform(train_df_fe[col].fillna('Unknown').astype(str)).astype('int16')
        test_vals = test_df_fe[col].fillna('Unknown').astype(str)
        test_df_fe[f'{col}_encoded'] = test_vals.apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1).astype('int16')
        label_encoders[col] = le
        feature_cols.append(f'{col}_encoded')

print(f"Total features: {len(feature_cols)}")


# Prepare data
X = train_df_fe[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).astype('float32')
y_class = train_df_fe['MasterItemNo'].astype(str)
y_reg = pd.to_numeric(train_df_fe['QtyShipped'], errors='coerce').fillna(0).astype('float32')

X_test = test_df_fe[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).astype('float32')
test_ids = test_df_fe['id'].copy()

del train_df_fe, test_df_fe
clear_memory()

print(f"X: {X.shape}, X_test: {X_test.shape}")
print(f"Classes: {len(np.unique(y_class))}")
get_memory_usage()


# Handle rare classes
class_counts = y_class.value_counts()
valid_classes = class_counts[class_counts >= 2].index.tolist()
print(f"Valid classes: {len(valid_classes)} / {len(class_counts)}")

mask = y_class.isin(valid_classes)
X_filtered = X[mask].reset_index(drop=True)
y_class_filtered = y_class[mask].reset_index(drop=True)
y_reg_filtered = y_reg[mask].reset_index(drop=True)

# Encode
le_target = LabelEncoder()
y_class_encoded = le_target.fit_transform(y_class_filtered).astype('int32')

# Split
try:
    X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val = train_test_split(
        X_filtered, y_class_encoded, y_reg_filtered, 
        test_size=0.2, random_state=RANDOM_STATE, stratify=y_class_encoded
    )
    print("âœ“ Stratified split")
except:
    X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val = train_test_split(
        X_filtered, y_class_encoded, y_reg_filtered, 
        test_size=0.2, random_state=RANDOM_STATE
    )
    print("Random split")

del X_filtered, y_class_filtered, y_reg_filtered
clear_memory()

print(f"Train: {X_train.shape}, Val: {X_val.shape}")
get_memory_usage()


# Random Forest Classifier (baseline)
print("Training Random Forest Classifier...")
rf_clf = RandomForestClassifier(
    n_estimators=100, max_depth=15, min_samples_split=15, min_samples_leaf=5,
    max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1, verbose=1
)
rf_clf.fit(X_train, y_class_train)

rf_clf_acc_val = accuracy_score(y_class_val, rf_clf.predict(X_val))
print(f"\nRF Classifier Val Accuracy: {rf_clf_acc_val:.4f}")

rf_clf_results = {'val_acc': rf_clf_acc_val}
clear_memory()


# LightGBM Classifier
print("Training LightGBM Classifier...")

# Handle validation set classes
train_classes = set(np.unique(y_class_train))
val_classes = set(np.unique(y_class_val))
missing = val_classes - train_classes

if missing:
    val_mask = np.isin(y_class_val, list(train_classes))
    X_val_lgb, y_class_val_lgb = X_val[val_mask], y_class_val[val_mask]
    print(f"Filtered validation: {len(missing)} missing classes")
else:
    X_val_lgb, y_class_val_lgb = X_val, y_class_val

# LightGBM parameters optimized for multi-class classification
lgb_clf_params = {
    'objective': 'multiclass',
    'num_class': len(np.unique(y_class_train)),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'min_child_samples': 20,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1,
    'device': 'gpu',  # GPU acceleration
    'gpu_platform_id': 0,
    'gpu_device_id': 0
}

# Create datasets
train_data = lgb.Dataset(X_train, label=y_class_train)
val_data = lgb.Dataset(X_val_lgb, label=y_class_val_lgb, reference=train_data)

# Train with early stopping
lgb_clf = lgb.train(
    lgb_clf_params,
    train_data,
    num_boost_round=200,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'val'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=30),
        lgb.log_evaluation(period=50)
    ]
)

# Predictions
lgb_clf_pred_val = lgb_clf.predict(X_val, num_iteration=lgb_clf.best_iteration)
lgb_clf_pred_val_class = np.argmax(lgb_clf_pred_val, axis=1)
lgb_clf_acc_val = accuracy_score(y_class_val, lgb_clf_pred_val_class)

print(f"\nLightGBM Classifier Val Accuracy: {lgb_clf_acc_val:.4f}")
print(f"Best iteration: {lgb_clf.best_iteration}")

lgb_clf_results = {'val_acc': lgb_clf_acc_val, 'best_iter': lgb_clf.best_iteration}

del train_data, val_data, lgb_clf_pred_val, lgb_clf_pred_val_class, X_val_lgb, y_class_val_lgb
clear_memory()


# Random Forest Regressor (baseline)
print("Training Random Forest Regressor...")
rf_reg = RandomForestRegressor(
    n_estimators=100, max_depth=20, min_samples_split=15, min_samples_leaf=5,
    max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1, verbose=1
)
rf_reg.fit(X_train, y_reg_train)

rf_reg_pred = np.maximum(rf_reg.predict(X_val), 0)
rf_reg_mae = mean_absolute_error(y_reg_val, rf_reg_pred)
rf_reg_rmse = np.sqrt(mean_squared_error(y_reg_val, rf_reg_pred))
rf_reg_r2 = r2_score(y_reg_val, rf_reg_pred)

print(f"\nRF Regressor: MAE={rf_reg_mae:.4f}, RMSE={rf_reg_rmse:.4f}, RÂ²={rf_reg_r2:.4f}")

rf_reg_results = {'mae': rf_reg_mae, 'rmse': rf_reg_rmse, 'r2': rf_reg_r2}
del rf_reg_pred
clear_memory()


# LightGBM Regressor
print("Training LightGBM Regressor...")

lgb_reg_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'min_child_samples': 20,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0
}

train_data_reg = lgb.Dataset(X_train, label=y_reg_train)
val_data_reg = lgb.Dataset(X_val, label=y_reg_val, reference=train_data_reg)

lgb_reg = lgb.train(
    lgb_reg_params,
    train_data_reg,
    num_boost_round=200,
    valid_sets=[train_data_reg, val_data_reg],
    valid_names=['train', 'val'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=30),
        lgb.log_evaluation(period=50)
    ]
)

lgb_reg_pred = np.maximum(lgb_reg.predict(X_val, num_iteration=lgb_reg.best_iteration), 0)
lgb_reg_mae = mean_absolute_error(y_reg_val, lgb_reg_pred)
lgb_reg_rmse = np.sqrt(mean_squared_error(y_reg_val, lgb_reg_pred))
lgb_reg_r2 = r2_score(y_reg_val, lgb_reg_pred)

print(f"\nLightGBM Regressor: MAE={lgb_reg_mae:.4f}, RMSE={lgb_reg_rmse:.4f}, RÂ²={lgb_reg_r2:.4f}")
print(f"Best iteration: {lgb_reg.best_iteration}")

lgb_reg_results = {'mae': lgb_reg_mae, 'rmse': lgb_reg_rmse, 'r2': lgb_reg_r2, 'best_iter': lgb_reg.best_iteration}

del train_data_reg, val_data_reg, lgb_reg_pred
del lgb_clf, lgb_reg, rf_clf, rf_reg
del X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val
clear_memory()
get_memory_usage()


print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print("\nðŸ“Š CLASSIFICATION")
print(f"  Random Forest: {rf_clf_results['val_acc']:.4f}")
print(f"  LightGBM:      {lgb_clf_results['val_acc']:.4f}")

if lgb_clf_results['val_acc'] > rf_clf_results['val_acc']:
    best_clf_type = 'lgb'
    print("  âœ“ LightGBM selected")
else:
    best_clf_type = 'rf'
    print("  âœ“ Random Forest selected")

print("\nðŸ“ˆ REGRESSION")
print(f"  Random Forest: MAE={rf_reg_results['mae']:.4f}, RÂ²={rf_reg_results['r2']:.4f}")
print(f"  LightGBM:      MAE={lgb_reg_results['mae']:.4f}, RÂ²={lgb_reg_results['r2']:.4f}")

if lgb_reg_results['mae'] < rf_reg_results['mae']:
    best_reg_type = 'lgb'
    print("  âœ“ LightGBM selected")
else:
    best_reg_type = 'rf'
    print("  âœ“ Random Forest selected")


# Re-encode ALL classes
le_target_final = LabelEncoder()
y_class_all_encoded = le_target_final.fit_transform(y_class.astype(str)).astype('int32')
print(f"Final model classes: {len(le_target_final.classes_)}")


# Train final classifier
print("\nTraining final classifier...")

if best_clf_type == 'lgb':
    print("Using LightGBM")
    
    final_clf_params = {
        'objective': 'multiclass',
        'num_class': len(le_target_final.classes_),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'max_depth': 10,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_child_samples': 10,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }
    
    train_data_final = lgb.Dataset(X, label=y_class_all_encoded)
    final_clf = lgb.train(
        final_clf_params,
        train_data_final,
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(period=100)]
    )
    
    test_class_pred_proba = final_clf.predict(X_test)
    test_class_pred_encoded = np.argmax(test_class_pred_proba, axis=1)
    del train_data_final, test_class_pred_proba
else:
    print("Using Random Forest")
    final_clf = RandomForestClassifier(
        n_estimators=150, max_depth=20, min_samples_split=10, min_samples_leaf=3,
        max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1, verbose=1
    )
    final_clf.fit(X, y_class_all_encoded)
    test_class_pred_encoded = final_clf.predict(X_test)

test_class_pred = le_target_final.inverse_transform(test_class_pred_encoded)
print(f"Classification predictions: {test_class_pred.shape}")

del final_clf, test_class_pred_encoded, y_class_all_encoded
clear_memory()


# Train final regressor
print("\nTraining final regressor...")

if best_reg_type == 'lgb':
    print("Using LightGBM")
    
    final_reg_params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'max_depth': 10,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_child_samples': 10,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }
    
    train_data_reg_final = lgb.Dataset(X, label=y_reg)
    final_reg = lgb.train(
        final_reg_params,
        train_data_reg_final,
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(period=100)]
    )
    
    test_reg_pred = np.maximum(final_reg.predict(X_test), 0)
    del train_data_reg_final
else:
    print("Using Random Forest")
    final_reg = RandomForestRegressor(
        n_estimators=150, max_depth=20, min_samples_split=10, min_samples_leaf=3,
        max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1, verbose=1
    )
    final_reg.fit(X, y_reg)
    test_reg_pred = np.maximum(final_reg.predict(X_test), 0)

print(f"Regression predictions: {test_reg_pred.shape}")

del final_reg, X, y_class, y_reg, X_test
clear_memory()


submission = pd.DataFrame({
    'id': test_ids,
    'MasterItemNo': test_class_pred,
    'QtyShipped': np.round(test_reg_pred, 2)
})

del test_class_pred, test_reg_pred, test_ids
clear_memory()

# Validate
assert submission['id'].notna().all()
assert submission['MasterItemNo'].notna().all()
assert submission['QtyShipped'].notna().all()
assert (submission['QtyShipped'] >= 0).all()

print(f"âœ“ Submission validated")
print(f"Shape: {submission.shape}")
print(submission.head(10))

submission.to_csv('submission.csv', index=False)
print("\nâœ“ Saved to submission.csv")


print("="*70)
print("FINAL RESULTS - LIGHTGBM VERSION")
print("="*70)

print("\nðŸ“Š CLASSIFICATION (MasterItemNo)")
print(f"  RF Val Accuracy:  {rf_clf_results['val_acc']:.4f}")
print(f"  LGB Val Accuracy: {lgb_clf_results['val_acc']:.4f}")
print(f"  Best Model: {'LightGBM' if best_clf_type == 'lgb' else 'Random Forest'}")
print(f"  Classes: {len(le_target_final.classes_)}")

print("\nðŸ“ˆ REGRESSION (QtyShipped)")
print(f"  RF:  MAE={rf_reg_results['mae']:.4f}, RMSE={rf_reg_results['rmse']:.4f}, RÂ²={rf_reg_results['r2']:.4f}")
print(f"  LGB: MAE={lgb_reg_results['mae']:.4f}, RMSE={lgb_reg_results['rmse']:.4f}, RÂ²={lgb_reg_results['r2']:.4f}")
print(f"  Best Model: {'LightGBM' if best_reg_type == 'lgb' else 'Random Forest'}")

print("\nðŸ”§ LIGHTGBM ADVANTAGES")
print(f"  âœ“ Faster training than XGBoost")
print(f"  âœ“ Lower memory usage")
print(f"  âœ“ GPU acceleration support")
print(f"  âœ“ Early stopping for optimal iterations")
print(f"  âœ“ Leaf-wise tree growth (better accuracy)")
print(f"  âœ“ Handles large datasets efficiently")

print("\nðŸŽ¯ FEATURE ENGINEERING")
print(f"  Total Features: {len(feature_cols)}")
print(f"  âœ“ Cyclical time encoding")
print(f"  âœ“ Time series lags & rolling features")
print(f"  âœ“ Project phase tracking")
print(f"  âœ“ Cumulative features")
print(f"  âœ“ Material type detection")
print(f"  âœ“ Cross-feature interactions")

print("\nðŸ“¤ SUBMISSION")
print(f"  File: submission.csv")
print(f"  Rows: {len(submission):,}")

print("\n" + "="*70)
print("âœ“ Pipeline Complete!")
print("="*70)
get_memory_usage()

