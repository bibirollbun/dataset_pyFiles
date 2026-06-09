import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time
import re
import os
from datetime import datetime
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Added MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Import display explicitly if not in a standard Jupyter environment
try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

VERBOSE_PRINT = True


def reduce_memory_usage(df, name='', verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose: print(f'Memory usage of dataframe {name} is {start_mem:.2f} MB')
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max: df[col] = df[col].astype(np.int64)
            else:
                if df[col].isnull().all():
                    if pd.isna(c_min) or (c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max): df[col] = df[col].astype(np.float16)
                    elif pd.isna(c_min) or (c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max): df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df_temp_f16 = df[col].dropna().astype(np.float16)
                    if df_temp_f16.empty or np.allclose(df_temp_f16, df[col].dropna(), rtol=1e-3, atol=1e-3, equal_nan=True): df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df_temp_f32 = df[col].dropna().astype(np.float32)
                        if df_temp_f32.empty or np.allclose(df_temp_f32, df[col].dropna(), rtol=1e-5, atol=1e-5, equal_nan=True): df[col] = df[col].astype(np.float32)
                        else: df[col] = df[col].astype(np.float64)
                    else: df[col] = df[col].astype(np.float64)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df_temp_f32 = df[col].dropna().astype(np.float32)
                    if df_temp_f32.empty or np.allclose(df_temp_f32, df[col].dropna(), rtol=1e-5, atol=1e-5, equal_nan=True): df[col] = df[col].astype(np.float32)
                    else: df[col] = df[col].astype(np.float64)
                else: df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage of {name} after optimization is: {end_mem:.2f} MB')
        if start_mem > 0 : print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
        else: print("Initial memory usage was zero.")
    return df

def one_hot_encoder(df_input, nan_as_category=True, limit_categories=50, prefix_sep='_'): # Reduced limit_categories
    df = df_input.copy()
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    if VERBOSE_PRINT and len(categorical_columns) > 0 : print(f"Found {len(categorical_columns)} categorical columns for OHE.")
    for col in categorical_columns:
        if df[col].isnull().any() and not nan_as_category: df[col] = df[col].fillna('Missing_OHE_Placeholder')
        if limit_categories and df[col].nunique(dropna=False) > limit_categories:
            if VERBOSE_PRINT: print(f"Limiting categories for column: {col} (had {df[col].nunique(dropna=False)} unique values)")
            top_cats = df[col].value_counts(dropna=False).nlargest(limit_categories - 1).index.tolist()
            df[col] = df[col].apply(lambda x: x if x in top_cats else 'Other_Aggregated')
    if len(categorical_columns) > 0:
        df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category, prefix_sep=prefix_sep)
        new_columns = [c for c in df.columns if c not in original_columns]
        if VERBOSE_PRINT and len(new_columns) > 0 : print(f"Created {len(new_columns)} new columns from OHE.")
    else:
        if VERBOSE_PRINT: print("No categorical columns found for OHE.")
        new_columns = []
    return df, new_columns

def display_df_info(df, name, head_n=3):
    if df is not None and not df.empty:
        print(f"\n--- Info for: {name} ---")
        print(f"Shape: {df.shape}")
        display(df.head(head_n))
        gc.collect()
    else: print(f"{name} is not loaded or is empty.")

def clean_col_names(df): # Logistic Regression is less sensitive, but good practice
    df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df


# For Kaggle, the input data is usually in /kaggle/input/
base_path = '/kaggle/input/home-credit-default-risk/'

print(f"Data will be loaded from: {base_path}")
if not os.path.exists(base_path):
    print(f"WARNING: The Kaggle path {base_path} does not exist or is not accessible.")
    print("Please ensure you have added the 'Home Credit Default Risk' dataset to your Kaggle Notebook.")
else:
    print(f"Kaggle path {base_path} confirmed to exist.")
    print("Files in base_path:", os.listdir(base_path)) 


file_paths = {
    'app_train': os.path.join(base_path, 'application_train.csv'),
    'app_test': os.path.join(base_path, 'application_test.csv'),
    'bureau': os.path.join(base_path, 'bureau.csv'),
    'bureau_balance': os.path.join(base_path, 'bureau_balance.csv'),
    'previous_application': os.path.join(base_path, 'previous_application.csv'),
    'pos_cash': os.path.join(base_path, 'POS_CASH_balance.csv'), # Corrected filename assuming no leading/trailing spaces
    'installments_payments': os.path.join(base_path, 'installments_payments.csv'),
    'credit_card_balance': os.path.join(base_path, 'credit_card_balance.csv')
}
# Check for sample submission, not used for training but good to know it's there
if os.path.exists(os.path.join(base_path, 'sample_submission.csv')):
    print("sample_submission.csv found.")

dfs = {}
for name, path in file_paths.items():
    try:
        dfs[name] = pd.read_csv(path)
        dfs[name] = reduce_memory_usage(dfs[name], name=name, verbose=VERBOSE_PRINT)
        print(f"{name} loaded successfully. Shape: {dfs[name].shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {path}. Creating an empty DataFrame for {name}.")
        # Check if the filename for pos_cash might have a space
        if name == 'pos_cash' and os.path.exists(os.path.join(base_path, 'pos_cash_balance .csv')): # Note the space
            print("Attempting to load 'pos_cash_balance .csv' with a leading space in the filename...")
            try:
                dfs[name] = pd.read_csv(os.path.join(base_path, 'pos_cash_balance .csv'))
                dfs[name] = reduce_memory_usage(dfs[name], name=name, verbose=VERBOSE_PRINT)
                print(f"{name} (with space in filename) loaded successfully. Shape: {dfs[name].shape}")
            except Exception as e_space:
                 print(f"Failed to load 'pos_cash_balance .csv' as well: {e_space}")
                 dfs[name] = pd.DataFrame()

        else:
            dfs[name] = pd.DataFrame()
    print("-" * 50); gc.collect()

display_df_info(dfs.get('app_train'), "Initial app_train")
display_df_info(dfs.get('app_test'), "Initial app_test")


epsilon = 1e-6

def preprocess_and_fe_application_logreg(df_input, median_dict_storage, is_train_set=True):
    df = df_input.copy()
    if VERBOSE_PRINT: print(f"LogReg FE: Processing application data. Shape: {df.shape}")

    # 1. Handle and Convert Days Columns to Positive/Years
    # DAYS_EMPLOYED: Anomaly first, then NaN, then imputation, then to positive years
    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
        df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
        # Imputation for DAYS_EMPLOYED will happen in the numerical imputation step below.
        # The _YEARS version will be created after imputation.

    if 'DAYS_BIRTH' in df.columns:
        df['DAYS_BIRTH_YEARS'] = df['DAYS_BIRTH'] / -365.25 # Age in positive years

    # For other days columns, convert to positive days first, then to years after imputation if needed.
    for col in ['DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE']:
        if col in df.columns:
            df[col + '_POSITIVE_DAYS'] = df[col] * -1 # Makes days positive
            df[col + '_POSITIVE_DAYS'].fillna(0, inplace=True) # Fill NaNs if any after * -1

    # 2. Categorical Imputation (Mode or 'Unknown')
    cat_cols_to_impute_mode = {
        'OCCUPATION_TYPE': 'Unknown_Occupation', 'FONDKAPREMONT_MODE': 'not_specified',
        'WALLSMATERIAL_MODE': 'Unknown_Material', 'HOUSETYPE_MODE': 'block_of_flats',
        'NAME_TYPE_SUITE': 'Unaccompanied', 'NAME_FAMILY_STATUS': 'Married'
    }
    for col, fill_val in cat_cols_to_impute_mode.items():
        if col in df.columns and df[col].isnull().any():
            if is_train_set:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else fill_val
                median_dict_storage[f'app_cat_mode_{col}'] = mode_val
            impute_with = median_dict_storage.get(f'app_cat_mode_{col}', fill_val)
            df[col].fillna(impute_with, inplace=True)

    if 'CODE_GENDER' in df.columns:
        df['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
        if is_train_set: median_dict_storage['app_cat_mode_CODE_GENDER'] = df['CODE_GENDER'].mode()[0] if not df['CODE_GENDER'].mode().empty else 'F'
        df['CODE_GENDER'].fillna(median_dict_storage.get('app_cat_mode_CODE_GENDER', 'F'), inplace=True)

    if 'EMERGENCYSTATE_MODE' in df.columns:
        df['EMERGENCYSTATE_MODE'].fillna('No', inplace=True) # OHE will handle 'Yes'/'No'

    # 3. Numerical Imputation (Median) - Crucial for LogReg
    numerical_cols_to_impute_median = [
        'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'CNT_FAM_MEMBERS', 'OWN_CAR_AGE',
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_EMPLOYED', # Impute DAYS_EMPLOYED now
        # Housing features
        'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG',
        'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG',
        'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
        'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE',
        'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE',
        'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
        'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
        'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI',
        'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
        'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'TOTALAREA_MODE',
    ]
    # Add bureau request columns
    
    bureau_req_cols_app = [col for col in df.columns if 'AMT_REQ_CREDIT_BUREAU' in col]
    numerical_cols_to_impute_median.extend(bureau_req_cols_app)

    for col in numerical_cols_to_impute_median:
        if col in df.columns:
            # NAN flags can be useful
            df[f'{col}_NAN_FLAG'] = df[col].isnull().astype(int)
            if is_train_set:
                median_val = df[col].median()
                median_dict_storage[f'app_num_median_{col}'] = median_val if pd.notna(median_val) else 0
            impute_val = median_dict_storage.get(f'app_num_median_{col}', 0)
            df[col].fillna(impute_val, inplace=True)

    # 4. Create Final _YEARS versions for Days columns
    
    if 'DAYS_EMPLOYED' in df.columns: # DAYS_EMPLOYED is now imputed
        df['DAYS_EMPLOYED_YEARS'] = df['DAYS_EMPLOYED'] / -365.25 # negative because the date of application
        df['DAYS_EMPLOYED_YEARS'] = df['DAYS_EMPLOYED_YEARS'].abs() 

    for col_base in ['REGISTRATION', 'ID_PUBLISH', 'LAST_PHONE_CHANGE']:
        days_col_positive = f'DAYS_{col_base}_POSITIVE_DAYS'
        years_col = f'DAYS_{col_base}_YEARS'
        if days_col_positive in df.columns:
            df[years_col] = df[days_col_positive] / 365.25
        elif f'DAYS_{col_base}' in df.columns: # Fallback if _POSITIVE_DAYS wasn't created (e.g. if original was not negative)
            df[years_col] = (df[f'DAYS_{col_base}'] * -1).fillna(0) / 365.25 # Ensure positive and fillna before division
            df[years_col] = df[years_col].abs()


    # 5. Feature Engineering using Positive/Years durations
    
    # Financial Ratios
    
    if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'].replace(0, epsilon) + epsilon)
    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'].replace(0, epsilon) + epsilon)
    if 'AMT_INCOME_TOTAL' in df.columns and 'CNT_FAM_MEMBERS' in df.columns:
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'].replace(0,1).fillna(1) + epsilon)
    if 'AMT_INCOME_TOTAL' in df.columns and 'CNT_CHILDREN' in df.columns:
        df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (df['CNT_CHILDREN'].replace(0,1).fillna(1) + epsilon) # Assuming CNT_CHILDREN not NaN after imputation/default
    if 'AMT_ANNUITY' in df.columns and 'AMT_CREDIT' in df.columns:
        df['ANNUITY_TO_CREDIT_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'].replace(0, epsilon) + epsilon)
    if 'AMT_CREDIT' in df.columns and 'AMT_GOODS_PRICE' in df.columns:
        
        # If AMT_GOODS_PRICE is NaN (after imputation attempt), using AMT_CREDIT as a fallback can make sense
        
        amt_goods_price_filled = df['AMT_GOODS_PRICE'].fillna(df['AMT_CREDIT'])
        df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / (amt_goods_price_filled.replace(0, epsilon) + epsilon)

    # Time Ratios (using _YEARS columns which are positive)
    
    if 'DAYS_EMPLOYED_YEARS' in df.columns and 'DAYS_BIRTH_YEARS' in df.columns:
        df['EMPLOYED_TO_AGE_RATIO'] = df['DAYS_EMPLOYED_YEARS'] / (df['DAYS_BIRTH_YEARS'].replace(0,epsilon) + epsilon)
    if 'DAYS_REGISTRATION_YEARS' in df.columns and 'DAYS_BIRTH_YEARS' in df.columns:
        df['REGISTRATION_TO_AGE_RATIO'] = df['DAYS_REGISTRATION_YEARS'] / (df['DAYS_BIRTH_YEARS'].replace(0,epsilon) + epsilon)

    
    # External Sources (already imputed)
    
    ext_sources = [col for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if col in df.columns]
    if ext_sources:
        df['EXT_SOURCES_MEAN'] = df[ext_sources].mean(axis=1)
        df['EXT_SOURCES_SUM'] = df[ext_sources].sum(axis=1)
        df['EXT_SOURCES_PROD'] = df[ext_sources].prod(axis=1, skipna=False) # skipna=False for LogReg better than True
        df['EXT_SOURCES_MIN'] = df[ext_sources].min(axis=1)
        df['EXT_SOURCES_MAX'] = df[ext_sources].max(axis=1)
        # Polynomials and interactions
        for source in ext_sources:
            df[f'{source}_SQ'] = df[source]**2
            if 'DAYS_BIRTH_YEARS' in df.columns: # Interact with positive age
                df[f'{source}_X_AGE_YEARS'] = df[source] * df['DAYS_BIRTH_YEARS']
            if 'DAYS_EMPLOYED_YEARS' in df.columns: # Interact with positive employment duration
                 df[f'{source}_X_EMPLOYED_YEARS'] = df[source] * df['DAYS_EMPLOYED_YEARS']


    # Document count and other flags sum
    
    flag_doc_cols = [col for col in df.columns if 'FLAG_DOCUMENT_' in col]
    if flag_doc_cols: df['DOC_COUNT'] = df[flag_doc_cols].sum(axis=1)
    phone_email_flags = [col for col in ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL'] if col in df.columns]
    if phone_email_flags: df['PHONE_EMAIL_FLAGS_SUM'] = df[phone_email_flags].sum(axis=1)


    # 6. One-Hot Encode (NaNs should have been handled for categoricals)
    
    df, _ = one_hot_encoder(df, nan_as_category=False, limit_categories=50) # nan_as_category=False

    if VERBOSE_PRINT: print(f"LogReg FE: Application FE finished. Shape: {df.shape}")
    return df, median_dict_storage


def fe_bureau_and_balance_logreg(bureau_input_df, bb_input_df, median_dict_storage, is_train_set=True, verbose=False):
    if bureau_input_df.empty: return pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage
    bureau = bureau_input_df.copy()

    # Convert relevant days columns to positive values or handle their sign appropriately
    # For aggregations, original signs might be okay if interpreted correctly, but for direct features, positive is better.
    if 'DAYS_CREDIT' in bureau.columns:
        bureau['DAYS_CREDIT_POSITIVE'] = bureau['DAYS_CREDIT'] * -1
        bureau['DAYS_CREDIT_POSITIVE'].fillna(0, inplace=True)
    if 'DAYS_CREDIT_UPDATE' in bureau.columns:
        bureau['DAYS_CREDIT_UPDATE_POSITIVE'] = bureau['DAYS_CREDIT_UPDATE'] * -1
        bureau['DAYS_CREDIT_UPDATE_POSITIVE'].fillna(0, inplace=True)
    # DAYS_CREDIT_ENDDATE can be positive (future) or negative (past). Keep its sign for now.
    # DAYS_ENDDATE_FACT is usually negative.

    # Impute key bureau columns
    cols_to_impute_bureau_logreg = {
        'DAYS_CREDIT_ENDDATE': bureau['DAYS_CREDIT'].fillna(0) -1, # Impute relative to DAYS_CREDIT or a large negative if DAYS_CREDIT is NaN
        'DAYS_ENDDATE_FACT': bureau['DAYS_CREDIT_ENDDATE'], # Impute with planned or a distinct past date
        'AMT_CREDIT_MAX_OVERDUE': 0, 'AMT_ANNUITY': 0, 'AMT_CREDIT_SUM_DEBT': 0,
        'AMT_CREDIT_SUM_LIMIT': 0, 'AMT_CREDIT_SUM_OVERDUE': 0
    }
    for col, default_val_or_series in cols_to_impute_bureau_logreg.items():
        if col in bureau.columns:
            if bureau[col].isnull().any():
                if is_train_set: # Learn medians/means if applicable, or store fixed values
                    if isinstance(default_val_or_series, (int, float)):
                        median_dict_storage[f'bureau_fill_{col}'] = default_val_or_series
                    else: # It's a series, calculate median of that series for imputation
                        median_dict_storage[f'bureau_fill_{col}'] = default_val_or_series[bureau[col].isnull()].median() if not default_val_or_series[bureau[col].isnull()].empty else 0

                fill_value = median_dict_storage.get(f'bureau_fill_{col}')
                if fill_value is None: # If not learned, use the provided default
                    if isinstance(default_val_or_series, pd.Series):
                         bureau[col].fillna(default_val_or_series, inplace=True) # Series fill
                    else:
                         bureau[col].fillna(default_val_or_series, inplace=True) # Value fill
                else:
                    bureau[col].fillna(fill_value, inplace=True)


    # Feature Engineering (ensure results are well-defined for LogReg)
    if 'DAYS_CREDIT_ENDDATE' in bureau.columns and 'DAYS_CREDIT_POSITIVE' in bureau.columns:
        # Duration can be tricky if enddate is before startdate due to data issues.
        # We use DAYS_CREDIT_POSITIVE for start, DAYS_CREDIT_ENDDATE for end.
        # If ENDDATE is negative (past), duration from start (positive) to end (negative) needs care.
        # Let's calculate duration as: End_Day_Number - Start_Day_Number (relative to application)
        # If DAYS_CREDIT = -100 (100 days ago), DAYS_CREDIT_ENDDATE = -50 (50 days ago), duration = -50 - (-100) = 50 days
        # If DAYS_CREDIT = -100, DAYS_CREDIT_ENDDATE = +50 (in future), duration = 50 - (-100) = 150 days
        bureau['BUREAU_LOAN_DURATION_DAYS'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_CREDIT']


    bureau_agg_spec = {
        'DAYS_CREDIT_POSITIVE': ['count', 'mean', 'max', 'min', 'sum'],
        'DAYS_CREDIT_UPDATE_POSITIVE': ['mean', 'max', 'min'],
        'CREDIT_DAY_OVERDUE': ['sum', 'mean', 'max'], # Already positive or 0
        'AMT_CREDIT_SUM': ['sum', 'mean', 'max', 'std'],
        'AMT_CREDIT_SUM_DEBT': ['sum', 'mean', 'max', 'std'],
        'AMT_CREDIT_SUM_OVERDUE': ['sum', 'mean'],
        'AMT_ANNUITY': ['sum', 'mean', 'std'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'BUREAU_LOAN_DURATION_DAYS': ['mean', 'max', 'min', 'std']
    }
    bureau_agg_dict = {k:v for k,v in bureau_agg_spec.items() if k in bureau.columns}

    bureau, bureau_cat_cols = one_hot_encoder(bureau, nan_as_category=False, limit_categories=10)
    for cat_col in bureau_cat_cols:
        if cat_col in bureau.columns and cat_col not in ['SK_ID_CURR', 'SK_ID_BUREAU']:
            bureau_agg_dict[cat_col] = ['sum', 'mean']

    if 'SK_ID_CURR' not in bureau.columns: return pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage
    bureau_agg = bureau.drop(columns=['SK_ID_BUREAU'], errors='ignore').groupby('SK_ID_CURR').agg(bureau_agg_dict)
    bureau_agg.columns = pd.Index(['BUREAU_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    if bb_input_df is not None and not bb_input_df.empty:
        bb = bb_input_df.copy()
        status_map = {'C':0, 'X':-1, '0':0, '1':1, '2':2, '3':3, '4':4, '5':5} # Map X to a different value
        bb['STATUS_NUM'] = bb['STATUS'].map(status_map).fillna(-2) # Fill NaN with another distinct value

        bb_agg_simple = bb.groupby('SK_ID_BUREAU').agg(
            BB_MONTHS_COUNT=('MONTHS_BALANCE', 'count'),
            BB_MEAN_STATUS=('STATUS_NUM', 'mean'),
            BB_MAX_STATUS=('STATUS_NUM', 'max'),
            BB_MIN_MONTHS_AGO=('MONTHS_BALANCE', 'min'), # Most recent month (largest negative)
            BB_MAX_MONTHS_AGO=('MONTHS_BALANCE', 'max')  # Oldest month (smallest negative or 0)
        )
        # Need to link SK_ID_CURR to SK_ID_BUREAU from bureau_input_df if not already merged
        # For simplicity, let's assume bureau_input_df has SK_ID_CURR
        bb_merged_with_curr = bureau_input_df[['SK_ID_CURR', 'SK_ID_BUREAU']].drop_duplicates().merge(
            bb_agg_simple.reset_index(), on='SK_ID_BUREAU', how='inner' # inner to only keep bureaus with balance info
        )
        if not bb_merged_with_curr.empty:
            bb_final_agg_by_curr = bb_merged_with_curr.drop(columns=['SK_ID_BUREAU']).groupby('SK_ID_CURR').agg({
                'BB_MONTHS_COUNT': ['sum', 'mean'], # Sum of months over all bureau accounts, mean months per bureau account
                'BB_MEAN_STATUS': ['mean'],        # Mean of mean statuses
                'BB_MAX_STATUS': ['max'],          # Max of max statuses
                'BB_MIN_MONTHS_AGO': ['min'],      # Overall most recent month
                'BB_MAX_MONTHS_AGO': ['max']       # Overall oldest month
            })
            bb_final_agg_by_curr.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_final_agg_by_curr.columns.tolist()])
            if not bureau_agg.empty:
                bureau_agg = bureau_agg.reset_index().merge(bb_final_agg_by_curr.reset_index(), on='SK_ID_CURR', how='left')
            else: # If bureau_agg was empty (e.g. no bureau data for any SK_ID_CURR in app)
                bureau_agg = bb_final_agg_by_curr.reset_index()


    if verbose: print(f"LogReg FE: Bureau FE finished. Shape: {bureau_agg.shape}")
    return bureau_agg.reset_index() if not bureau_agg.empty else pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage


def fe_previous_applications_logreg(prev_app_df_input, median_dict_storage, is_train_set=True, verbose=False):
    if prev_app_df_input.empty: return pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage
    prev_df = prev_app_df_input.copy()

    # Days columns: convert anomaly to NaN, then impute, then convert to positive days/years
    days_cols_prev_anomaly = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
    for col in days_cols_prev_anomaly:
        if col in prev_df.columns:
            prev_df[col].replace(365243, np.nan, inplace=True)
            # Imputation will happen below

    # Impute NaNs for key numeric and days columns
    cols_to_impute_prev_logreg = [
        'AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'CNT_PAYMENT', 'AMT_APPLICATION', 'AMT_DOWN_PAYMENT',
        'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED'
    ]
    cols_to_impute_prev_logreg.extend(days_cols_prev_anomaly) # Add days columns for imputation
    if 'DAYS_DECISION' in prev_df.columns: cols_to_impute_prev_logreg.append('DAYS_DECISION')


    for col in cols_to_impute_prev_logreg:
        if col in prev_df.columns:
            df_col_nan_flag = f'{col}_NAN_FLAG' # Create NaN flag
            prev_df[df_col_nan_flag] = prev_df[col].isnull().astype(int)

            if is_train_set:
                median_val = prev_df[col].median()
                median_dict_storage[f'prev_app_median_{col}'] = median_val if pd.notna(median_val) else 0 # Default to 0 if all NaN
            impute_val = median_dict_storage.get(f'prev_app_median_{col}', 0)
            prev_df[col].fillna(impute_val, inplace=True)

    # Convert days to positive years/days after imputation
    if 'DAYS_DECISION' in prev_df.columns:
        prev_df['DAYS_DECISION_POSITIVE'] = prev_df['DAYS_DECISION'] * -1
    for col_day in days_cols_prev_anomaly:
        if col_day in prev_df.columns:
            prev_df[col_day + '_POSITIVE'] = prev_df[col_day] * -1 # Assuming they were negative after imputation if original was negative
            prev_df[col_day + '_POSITIVE'] = prev_df[col_day + '_POSITIVE'].abs() # Ensure positive


    # Basic aggregations
    prev_agg_spec = {
        'AMT_ANNUITY': ['sum', 'mean', 'max'],
        'AMT_CREDIT': ['sum', 'mean', 'max'],
        'AMT_APPLICATION': ['sum', 'mean'],
        'DAYS_DECISION_POSITIVE': ['count', 'mean', 'max', 'min', 'sum'], # Count of previous apps
        'CNT_PAYMENT': ['sum', 'mean', 'max'],
        # Add flags
    }
    # Add flags for imputed columns
    for col in cols_to_impute_prev_logreg:
        if f'{col}_NAN_FLAG' in prev_df.columns:
            prev_agg_spec[f'{col}_NAN_FLAG'] = ['sum', 'mean']


    prev_agg_dict = {k:v for k,v in prev_agg_spec.items() if k in prev_df.columns}

    prev_df, prev_cat_cols = one_hot_encoder(prev_df, nan_as_category=False, limit_categories=15)
    for cat_col in prev_cat_cols:
        if cat_col in prev_df.columns and cat_col not in ['SK_ID_CURR', 'SK_ID_PREV']:
            prev_agg_dict[cat_col] = ['sum', 'mean'] # Sum of OHE flags, mean proportion

    if 'SK_ID_CURR' not in prev_df.columns: return pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage
    prev_agg = prev_df.drop(columns=['SK_ID_PREV'], errors='ignore').groupby('SK_ID_CURR').agg(prev_agg_dict)
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    if verbose: print(f"LogReg FE: Prev App FE finished. Shape: {prev_agg.shape}")
    return prev_agg.reset_index() if not prev_agg.empty else pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage


def fe_pos_cash_logreg(pos_cash_df_input, verbose=False):
    if pos_cash_df_input.empty: return pd.DataFrame(columns=['SK_ID_CURR'])
    pos_df = pos_cash_df_input.copy()

    # MONTHS_BALANCE is negative, convert to positive "months ago"
    if 'MONTHS_BALANCE' in pos_df.columns:
        pos_df['MONTHS_AGO'] = pos_df['MONTHS_BALANCE'] * -1
        pos_df['MONTHS_AGO'].fillna(0, inplace=True)

    # Impute DPD with 0
    for col_dpd in ['SK_DPD', 'SK_DPD_DEF', 'CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE']:
        if col_dpd in pos_df.columns:
            pos_df[col_dpd].fillna(0, inplace=True) # Assuming 0 is a safe imputation for these counts/DPD

    pos_agg_spec = {
        'MONTHS_AGO': ['count', 'mean', 'max', 'min'], # Min MONTHS_AGO is most recent
        'SK_DPD': ['sum', 'mean', 'max'],
        'SK_DPD_DEF': ['sum', 'mean', 'max'],
        'CNT_INSTALMENT': ['mean', 'sum', 'max'],
        'CNT_INSTALMENT_FUTURE': ['mean', 'sum', 'min']
    }
    pos_agg_dict = {k:v for k,v in pos_agg_spec.items() if k in pos_df.columns}

    pos_df, pos_cat_cols = one_hot_encoder(pos_df, nan_as_category=False, limit_categories=5) # NAME_CONTRACT_STATUS
    for cat_col in pos_cat_cols:
         if cat_col in pos_df.columns and cat_col not in ['SK_ID_CURR', 'SK_ID_PREV']:
            pos_agg_dict[cat_col] = ['sum', 'mean']

    if 'SK_ID_CURR' not in pos_df.columns: return pd.DataFrame(columns=['SK_ID_CURR'])
    pos_agg = pos_df.drop(columns=['SK_ID_PREV'], errors='ignore').groupby('SK_ID_CURR').agg(pos_agg_dict)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

    if verbose: print(f"LogReg FE: POS CASH FE finished. Shape: {pos_agg.shape}")
    return pos_agg.reset_index() if not pos_agg.empty else pd.DataFrame(columns=['SK_ID_CURR'])


def fe_installments_logreg(installments_df_input, median_dict_storage, is_train_set=True, verbose=False):
    if installments_df_input.empty: return pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage
    pay_df = installments_df_input.copy()

    # Convert days to positive "days ago"
    for day_col_inst in ['DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT']:
        if day_col_inst in pay_df.columns:
            pay_df[day_col_inst + '_AGO'] = pay_df[day_col_inst] * -1
            # Impute NaNs for these _AGO columns
            if is_train_set:
                median_val = pay_df[day_col_inst + '_AGO'].median()
                median_dict_storage[f'install_median_{day_col_inst}_AGO'] = median_val if pd.notna(median_val) else 0
            impute_val = median_dict_storage.get(f'install_median_{day_col_inst}_AGO', 0)
            pay_df[day_col_inst + '_AGO'].fillna(impute_val, inplace=True)


    # Impute amounts
    for amt_col_inst in ['AMT_PAYMENT', 'AMT_INSTALMENT']:
        if amt_col_inst in pay_df.columns:
            pay_df[f'{amt_col_inst}_NAN_FLAG'] = pay_df[amt_col_inst].isnull().astype(int)
            if is_train_set:
                median_val = pay_df[amt_col_inst].median()
                median_dict_storage[f'install_median_{amt_col_inst}'] = median_val if pd.notna(median_val) else 0
            impute_val = median_dict_storage.get(f'install_median_{amt_col_inst}', 0)
            pay_df[amt_col_inst].fillna(impute_val, inplace=True)

    # Feature Engineering (using positive days ago)
    if 'DAYS_ENTRY_PAYMENT_AGO' in pay_df.columns and 'DAYS_INSTALMENT_AGO' in pay_df.columns:
        # DPD: payment_day_ago - instalment_day_ago. If payment is later, payment_day_ago is smaller (more recent).
        # So, DPD = instalment_day_ago - payment_day_ago. Positive if paid late.
        pay_df['INS_DPD_CALC'] = pay_df['DAYS_INSTALMENT_AGO'] - pay_df['DAYS_ENTRY_PAYMENT_AGO']
        pay_df['INS_DPD_CALC'] = pay_df['INS_DPD_CALC'].apply(lambda x: x if x > 0 else 0) # Only positive DPD

    if 'AMT_INSTALMENT' in pay_df.columns and 'AMT_PAYMENT' in pay_df.columns:
        pay_df['INS_PAYMENT_PERC'] = pay_df['AMT_PAYMENT'] / (pay_df['AMT_INSTALMENT'] + epsilon)
        pay_df['INS_PAYMENT_DIFF_AMT'] = pay_df['AMT_INSTALMENT'] - pay_df['AMT_PAYMENT']


    ins_agg_spec = {
        'DAYS_INSTALMENT_AGO': ['count', 'mean', 'max', 'min', 'std'], # Min is most recent
        'AMT_PAYMENT': ['sum', 'mean', 'max', 'std'],
        'AMT_INSTALMENT': ['sum', 'mean', 'max', 'std'],
        'INS_DPD_CALC': ['sum', 'mean', 'max', 'std'],
        'INS_PAYMENT_PERC': ['mean', 'std', 'min', 'max'],
        'INS_PAYMENT_DIFF_AMT':['sum','mean','max','min','std'],
        'AMT_PAYMENT_NAN_FLAG':['sum','mean'], # Aggregating NaN flags
        'AMT_INSTALMENT_NAN_FLAG':['sum','mean']
    }
    ins_agg_dict = {k:v for k,v in ins_agg_spec.items() if k in pay_df.columns}

    if 'SK_ID_CURR' not in pay_df.columns: return pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage
    ins_agg = pay_df.drop(columns=['SK_ID_PREV'], errors='ignore').groupby('SK_ID_CURR').agg(ins_agg_dict)
    ins_agg.columns = pd.Index(['INS_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])

    if verbose: print(f"LogReg FE: Installments FE finished. Shape: {ins_agg.shape}")
    return ins_agg.reset_index() if not ins_agg.empty else pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage


def fe_credit_card_logreg(cc_balance_df_input, median_dict_storage, is_train_set=True, verbose=False):
    if cc_balance_df_input.empty: return pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage
    cc_df = cc_balance_df_input.copy()

    # MONTHS_BALANCE to positive "months ago"
    if 'MONTHS_BALANCE' in cc_df.columns:
        cc_df['CC_MONTHS_AGO'] = cc_df['MONTHS_BALANCE'] * -1
        cc_df['CC_MONTHS_AGO'].fillna(0, inplace=True)

    # Impute NaNs for key amounts and DPD
    cols_to_impute_cc_logreg = [
        'AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_ATM_CURRENT',
        'AMT_DRAWINGS_POS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT',
        'AMT_RECEIVABLE_PRINCIPAL', 'AMT_TOTAL_RECEIVABLE', 'AMT_INST_MIN_REGULARITY',
        'CNT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_POS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT',
        'CNT_INSTALMENT_MATURE_CUM', 'SK_DPD', 'SK_DPD_DEF'
    ]
    if 'AMT_RECIVABLE' in cc_df.columns: # Typo in original data sometimes
        cc_df.rename(columns={'AMT_RECIVABLE':'AMT_RECEIVABLE'}, inplace=True)
        if 'AMT_RECEIVABLE' not in cols_to_impute_cc_logreg : cols_to_impute_cc_logreg.append('AMT_RECEIVABLE')


    for col in cols_to_impute_cc_logreg:
        if col in cc_df.columns:
            cc_df[f'{col}_NAN_FLAG'] = cc_df[col].isnull().astype(int) # NaN flag
            if is_train_set:
                median_val = cc_df[col].median()
                median_dict_storage[f'cc_median_{col}'] = median_val if pd.notna(median_val) else 0
            impute_val = median_dict_storage.get(f'cc_median_{col}', 0)
            cc_df[col].fillna(impute_val, inplace=True)

    # Feature Engineering
    if 'AMT_BALANCE' in cc_df.columns and 'AMT_CREDIT_LIMIT_ACTUAL' in cc_df.columns:
         cc_df['CC_LIMIT_UTILIZATION'] = cc_df['AMT_BALANCE'] / (cc_df['AMT_CREDIT_LIMIT_ACTUAL'] + epsilon)
    if 'AMT_PAYMENT_CURRENT' in cc_df.columns and 'AMT_INST_MIN_REGULARITY' in cc_df.columns:
         cc_df['CC_PAYMENT_TO_MIN_RATIO'] = cc_df['AMT_PAYMENT_CURRENT'] / (cc_df['AMT_INST_MIN_REGULARITY'] + epsilon)


    cc_agg_spec = {
        'CC_MONTHS_AGO': ['count', 'mean', 'max', 'min', 'std'], # Min is most recent
        'AMT_BALANCE': ['mean', 'max', 'sum', 'std'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['mean', 'max'],
        'AMT_DRAWINGS_CURRENT': ['sum', 'mean', 'max'],
        'AMT_PAYMENT_CURRENT': ['sum', 'mean', 'max'],
        'SK_DPD': ['sum', 'mean', 'max'],
        'SK_DPD_DEF': ['sum', 'mean', 'max'],
        'CC_LIMIT_UTILIZATION': ['mean', 'max', 'std'],
        'CC_PAYMENT_TO_MIN_RATIO':['mean','max','std']
        # Add NaN flags for all imputed cols
    }
    for col in cols_to_impute_cc_logreg: # Add NaN flag aggregations
        if f'{col}_NAN_FLAG' in cc_df.columns:
            cc_agg_spec[f'{col}_NAN_FLAG'] = ['sum', 'mean']

    cc_agg_dict = {k:v for k,v in cc_agg_spec.items() if k in cc_df.columns}

    cc_df, cc_cat_cols = one_hot_encoder(cc_df, nan_as_category=False, limit_categories=5) # For NAME_CONTRACT_STATUS
    for cat_col in cc_cat_cols:
        if cat_col in cc_df.columns and cat_col not in ['SK_ID_CURR', 'SK_ID_PREV']:
            cc_agg_dict[cat_col] = ['sum', 'mean']

    if 'SK_ID_CURR' not in cc_df.columns: return pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage
    cc_agg = cc_df.drop(columns=['SK_ID_PREV'], errors='ignore').groupby('SK_ID_CURR').agg(cc_agg_dict)
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

    if verbose: print(f"LogReg FE: Credit Card FE finished. Shape: {cc_agg.shape}")
    return cc_agg.reset_index() if not cc_agg.empty else pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage


print("--- Applying Feature Engineering for Logistic Regression Model ---")

df_train_logreg = dfs.get('app_train', pd.DataFrame()).copy()
df_test_logreg = dfs.get('app_test', pd.DataFrame()).copy()

# Initialize median storage dictionaries for this specific run
median_storage_app_logreg = {}
median_storage_bureau_logreg = {}
median_storage_prev_app_logreg = {}
median_storage_install_logreg = {}
median_storage_cc_logreg = {}

if not df_train_logreg.empty and not df_test_logreg.empty:
    df_train_logreg, median_storage_app_logreg = preprocess_and_fe_application_logreg(df_train_logreg, median_storage_app_logreg, is_train_set=True)
    df_test_logreg, _ = preprocess_and_fe_application_logreg(df_test_logreg, median_storage_app_logreg, is_train_set=False)
    display_df_info(df_train_logreg, "df_train_logreg (after app FE)")

    # Bureau
    bureau_data = dfs.get('bureau', pd.DataFrame()).copy()
    bb_data = dfs.get('bureau_balance', pd.DataFrame()).copy()
    if not bureau_data.empty:
        bureau_feats, median_storage_bureau_logreg = fe_bureau_and_balance_logreg(bureau_data, bb_data, median_storage_bureau_logreg, is_train_set=True, verbose=VERBOSE_PRINT)
        if not bureau_feats.empty and 'SK_ID_CURR' in bureau_feats.columns:
            df_train_logreg = df_train_logreg.merge(bureau_feats, on='SK_ID_CURR', how='left')
        bureau_feats_test, _ = fe_bureau_and_balance_logreg(bureau_data, bb_data, median_storage_bureau_logreg, is_train_set=False)
        if not bureau_feats_test.empty and 'SK_ID_CURR' in bureau_feats_test.columns:
            df_test_logreg = df_test_logreg.merge(bureau_feats_test, on='SK_ID_CURR', how='left')
        display_df_info(df_train_logreg, "df_train_logreg (after Bureau FE)")
    del bureau_data, bb_data; gc.collect()

    # Previous Application
    prev_app_data = dfs.get('previous_application', pd.DataFrame()).copy()
    if not prev_app_data.empty:
        prev_app_feats, median_storage_prev_app_logreg = fe_previous_applications_logreg(prev_app_data, median_storage_prev_app_logreg, is_train_set=True, verbose=VERBOSE_PRINT)
        if not prev_app_feats.empty and 'SK_ID_CURR' in prev_app_feats.columns:
            df_train_logreg = df_train_logreg.merge(prev_app_feats, on='SK_ID_CURR', how='left')
        prev_app_feats_test, _ = fe_previous_applications_logreg(prev_app_data, median_storage_prev_app_logreg, is_train_set=False)
        if not prev_app_feats_test.empty and 'SK_ID_CURR' in prev_app_feats_test.columns:
            df_test_logreg = df_test_logreg.merge(prev_app_feats_test, on='SK_ID_CURR', how='left')
        display_df_info(df_train_logreg, "df_train_logreg (after Prev App FE)")
    del prev_app_data; gc.collect()

    # POS CASH
    pos_cash_data = dfs.get('pos_cash', pd.DataFrame()).copy()
    if not pos_cash_data.empty:
        pos_cash_feats = fe_pos_cash_logreg(pos_cash_data, verbose=VERBOSE_PRINT) # No state for this simple version
        if not pos_cash_feats.empty and 'SK_ID_CURR' in pos_cash_feats.columns:
            df_train_logreg = df_train_logreg.merge(pos_cash_feats, on='SK_ID_CURR', how='left')
            df_test_logreg = df_test_logreg.merge(pos_cash_feats, on='SK_ID_CURR', how='left')
        display_df_info(df_train_logreg, "df_train_logreg (after POS CASH FE)")
    del pos_cash_data; gc.collect()

    # Installments
    install_data = dfs.get('installments_payments', pd.DataFrame()).copy()
    if not install_data.empty:
        install_feats, median_storage_install_logreg = fe_installments_logreg(install_data, median_storage_install_logreg, is_train_set=True, verbose=VERBOSE_PRINT)
        if not install_feats.empty and 'SK_ID_CURR' in install_feats.columns:
            df_train_logreg = df_train_logreg.merge(install_feats, on='SK_ID_CURR', how='left')
        install_feats_test, _ = fe_installments_logreg(install_data, median_storage_install_logreg, is_train_set=False)
        if not install_feats_test.empty and 'SK_ID_CURR' in install_feats_test.columns:
            df_test_logreg = df_test_logreg.merge(install_feats_test, on='SK_ID_CURR', how='left')
        display_df_info(df_train_logreg, "df_train_logreg (after Installments FE)")
    del install_data; gc.collect()

    # Credit Card
    cc_data = dfs.get('credit_card_balance', pd.DataFrame()).copy()
    if not cc_data.empty:
        cc_feats, median_storage_cc_logreg = fe_credit_card_logreg(cc_data, median_storage_cc_logreg, is_train_set=True, verbose=VERBOSE_PRINT)
        if not cc_feats.empty and 'SK_ID_CURR' in cc_feats.columns:
            df_train_logreg = df_train_logreg.merge(cc_feats, on='SK_ID_CURR', how='left')
        cc_feats_test, _ = fe_credit_card_logreg(cc_data, median_storage_cc_logreg, is_train_set=False)
        if not cc_feats_test.empty and 'SK_ID_CURR' in cc_feats_test.columns:
            df_test_logreg = df_test_logreg.merge(cc_feats_test, on='SK_ID_CURR', how='left')
        display_df_info(df_train_logreg, "df_train_logreg (after Credit Card FE)")
    del cc_data; gc.collect()
else:
    print("Initial application_train or application_test data is empty. Stopping FE process.")
    df_train_logreg = pd.DataFrame() # Ensure defined
    df_test_logreg = pd.DataFrame()


X_logreg_train, y_logreg_train, X_logreg_test, test_ids_logreg = pd.DataFrame(), pd.Series(dtype='float64'), pd.DataFrame(), pd.Series(dtype='int64')

if 'df_train_logreg' in locals() and not df_train_logreg.empty:
    print(f"Shape of df_train_logreg before final NaN fill: {df_train_logreg.shape}")
    df_train_logreg.fillna(0, inplace=True) # Final catch-all for NaNs
    print(f"NaNs in df_train_logreg after fill: {df_train_logreg.isnull().sum().sum()}")

    if 'df_test_logreg' in locals() and not df_test_logreg.empty:
        print(f"Shape of df_test_logreg before final NaN fill: {df_test_logreg.shape}")
        df_test_logreg.fillna(0, inplace=True)
        print(f"NaNs in df_test_logreg after fill: {df_test_logreg.isnull().sum().sum()}")
    else:
        print("df_test_logreg is empty or not defined for final NaN fill.")
        df_test_logreg = pd.DataFrame() # Ensure it's a DF

    gc.collect()

    if 'TARGET' in df_train_logreg.columns: y_logreg_train = df_train_logreg['TARGET']
    if 'SK_ID_CURR' in df_test_logreg.columns: test_ids_logreg = df_test_logreg['SK_ID_CURR']

    X_logreg_train_unaligned = df_train_logreg.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
    X_logreg_test_unaligned = df_test_logreg.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore') # TARGET might not exist in test

    if not X_logreg_train_unaligned.empty and not X_logreg_test_unaligned.empty:
        common_cols_logreg = list(set(X_logreg_train_unaligned.columns) & set(X_logreg_test_unaligned.columns))
        if common_cols_logreg: # Ensure there are common columns
            X_logreg_train = X_logreg_train_unaligned[common_cols_logreg].copy()
            X_logreg_test = X_logreg_test_unaligned[common_cols_logreg].copy()
            print(f"LogReg Train X shape after alignment: {X_logreg_train.shape}")
            print(f"LogReg Test X shape after alignment: {X_logreg_test.shape}")
        else:
            print("No common columns found between train and test after FE for LogReg.")
            X_logreg_train = pd.DataFrame() # Empty DF
            X_logreg_test = pd.DataFrame()  # Empty DF
    elif X_logreg_train_unaligned.empty:
        print("X_logreg_train_unaligned is empty after drops.")
    elif not X_logreg_train_unaligned.empty and X_logreg_test_unaligned.empty:
        print("X_logreg_test_unaligned is empty. Test set might be missing or problematic.")
        X_logreg_train = X_logreg_train_unaligned.copy()
        X_logreg_test = pd.DataFrame(columns=X_logreg_train.columns) # Align with empty test
        print(f"LogReg Train X shape: {X_logreg_train.shape}")
        print(f"LogReg Test X shape: {X_logreg_test.shape} (aligned but empty)")
else:
    print("df_train_logreg is empty. Cannot proceed with final preparation.")


cols_to_scale_logreg = []
if 'X_logreg_train' in locals() and not X_logreg_train.empty:
    for col in X_logreg_train.columns:
        # Check if column is numeric and not binary-like (more than 2 unique values)
        if X_logreg_train[col].dtype in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
            # Check for columns that are not just flags or OHE (which are often 0/1)
            # A simple check: if a column has many unique values or its range is large, it might need scaling.
            # For simplicity, scale if more than 2 unique values, assuming OHE/flags are mostly 0/1.
            if X_logreg_train[col].nunique(dropna=False) > 2:
                 cols_to_scale_logreg.append(col)
    print(f"Identified {len(cols_to_scale_logreg)} columns to scale for Logistic Regression.")
    if VERBOSE_PRINT and cols_to_scale_logreg: print("First 10 cols to scale:", cols_to_scale_logreg[:10])
else:
    print("X_logreg_train is not available for identifying columns to scale.")


if 'X_logreg_train' in locals() and not X_logreg_train.empty:
    X_logreg_train = clean_col_names(X_logreg_train.copy())
if 'X_logreg_test' in locals() and not X_logreg_test.empty:
    X_logreg_test = clean_col_names(X_logreg_test.copy())

if 'X_logreg_train' in locals(): display_df_info(X_logreg_train, "X_logreg_train (final, to be scaled in CV)")
if 'X_logreg_test' in locals(): display_df_info(X_logreg_test, "X_logreg_test (final, to be scaled in CV)")



print("--- Feature Matrix for Training (X_logreg_train) ---")
if 'X_logreg_train' in locals() and not X_logreg_train.empty:
    print(f"Shape of X_logreg_train: {X_logreg_train.shape}")
    print("\nFirst 5 rows of X_logreg_train:")
    display(X_logreg_train.head())
    print("\nLast 5 rows of X_logreg_train:")
    display(X_logreg_train.tail())
    print("\nBasic statistics for a few columns in X_logreg_train:")
    
    if X_logreg_train.shape[1] > 5:
        display(X_logreg_train.iloc[:, :5].describe()) # Stats for first 5 columns
    else:
        display(X_logreg_train.describe())
    print("\nData types in X_logreg_train:")
    display(X_logreg_train.dtypes.value_counts())
    print("\nNumber of NaN values per column in X_logreg_train (should be 0 here):")
    print(X_logreg_train.isnull().sum()[X_logreg_train.isnull().sum() > 0]) # Only show columns with NaNs
    if X_logreg_train.isnull().sum().sum() == 0:
        print("No NaN values found in X_logreg_train. Great!")

else:
    print("X_logreg_train is not defined or is empty.")

print("\n\n--- Feature Matrix for Testing (X_logreg_test) ---")
if 'X_logreg_test' in locals() and not X_logreg_test.empty:
    print(f"Shape of X_logreg_test: {X_logreg_test.shape}")
    print("\nFirst 5 rows of X_logreg_test:")
    display(X_logreg_test.head())
    print("\nLast 5 rows of X_logreg_test:")
    display(X_logreg_test.tail())
    print("\nBasic statistics for a few columns in X_logreg_test:")
    if X_logreg_test.shape[1] > 5:
        display(X_logreg_test.iloc[:, :5].describe())
    else:
        display(X_logreg_test.describe())
    print("\nData types in X_logreg_test:")
    display(X_logreg_test.dtypes.value_counts())
    print("\nNumber of NaN values per column in X_logreg_test (should be 0 here):")
    print(X_logreg_test.isnull().sum()[X_logreg_test.isnull().sum() > 0])
    if X_logreg_test.isnull().sum().sum() == 0:
        print("No NaN values found in X_logreg_test. Great!")
else:
    print("X_logreg_test is not defined or is empty.")

print("\n\n--- Target Variable for Training (y_logreg_train) ---")
if 'y_logreg_train' in locals() and not y_logreg_train.empty:
    print(f"Shape of y_logreg_train: {y_logreg_train.shape}")
    print("\nFirst 5 values of y_logreg_train:")
    display(y_logreg_train.head())
    print("\nValue counts for y_logreg_train (Target Distribution):")
    display(y_logreg_train.value_counts(normalize=True) * 100) # Percentage
else:
    print("y_logreg_train is not defined or is empty.")


def run_logistic_regression_cv_final(X_train, y_train, X_test, cols_to_scale, n_folds=5, random_seed=42, C_param=0.1):
    if X_train.empty or y_train.empty:
        print("LogReg: Training data is empty. Skipping.")
        return None, (np.zeros(X_test.shape[0]) if X_test is not None and not X_test.empty else None), 0.0, []

    print(f"\nStarting Logistic Regression training with {n_folds} folds... X_train shape: {X_train.shape}")
    if X_test is not None and not X_test.empty: print(f"X_test shape: {X_test.shape}")
    else: print("X_test is empty or not provided.")

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    oof_predictions = np.zeros(X_train.shape[0])

    if X_test is not None and not X_test.empty: test_predictions = np.zeros(X_test.shape[0])
    else: test_predictions = None

    models = []

    for fold_num, (train_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print(f"--- Fold {fold_num + 1}/{n_folds} ---")
        # Ensure correct indexing for y_fold_train
        X_fold_train, y_fold_train_series = X_train.iloc[train_idx].copy(), y_train.iloc[train_idx].copy()
        X_fold_val, y_fold_val_series = X_train.iloc[val_idx].copy(), y_train.iloc[val_idx].copy()

        y_fold_train = y_fold_train_series.values # Convert to numpy array for fitting
        y_fold_val = y_fold_val_series.values


        X_test_fold_scaled = pd.DataFrame() # Initialize for this fold

        if cols_to_scale:
            # print(f"Scaling {len(cols_to_scale)} features for fold {fold_num + 1}...")
            scaler = StandardScaler()
            # Ensure cols_to_scale exist in the current fold's data
            train_cols_to_scale_present = [col for col in cols_to_scale if col in X_fold_train.columns]
            val_cols_to_scale_present = [col for col in cols_to_scale if col in X_fold_val.columns]

            if train_cols_to_scale_present:
                X_fold_train[train_cols_to_scale_present] = scaler.fit_transform(X_fold_train[train_cols_to_scale_present])
            if val_cols_to_scale_present: # Should be same as train_cols_to_scale_present
                X_fold_val[val_cols_to_scale_present] = scaler.transform(X_fold_val[val_cols_to_scale_present])

            if X_test is not None and not X_test.empty:
                X_test_fold_scaled = X_test.copy()
                test_cols_to_scale_present = [col for col in cols_to_scale if col in X_test_fold_scaled.columns]
                if test_cols_to_scale_present:
                    X_test_fold_scaled[test_cols_to_scale_present] = scaler.transform(X_test_fold_scaled[test_cols_to_scale_present])
        else: 
             if X_test is not None and not X_test.empty:
                X_test_fold_scaled = X_test.copy() # Use unscaled if no scaling specified


        model = LogisticRegression(solver='liblinear', C=C_param, class_weight='balanced',
                                   random_state=random_seed + fold_num, max_iter=1000)
        try:
            model.fit(X_fold_train, y_fold_train)
            models.append(model)
            oof_predictions[val_idx] = model.predict_proba(X_fold_val)[:, 1]

            if test_predictions is not None and not X_test_fold_scaled.empty : # Check if X_test_fold_scaled was prepared
                test_predictions += model.predict_proba(X_test_fold_scaled)[:, 1] / folds.n_splits

            # Ensure y_fold_val is also a numpy array for roc_auc_score
            print(f'Fold {fold_num+1} OOF AUC: {roc_auc_score(y_fold_val, oof_predictions[val_idx]):.6f}')

        except Exception as e:
            print(f"Error in fold {fold_num+1}: {e}")
            oof_predictions[val_idx] = 0.5

        del X_fold_train, y_fold_train, X_fold_val, y_fold_val, y_fold_train_series, y_fold_val_series
        if 'X_test_fold_scaled' in locals() : del X_test_fold_scaled
        gc.collect()

    overall_oof_auc = 0.0
    if len(y_train) > 0 and len(oof_predictions) == len(y_train) and not np.all(np.isnan(oof_predictions)):
        try:
            overall_oof_auc = roc_auc_score(y_train, oof_predictions)
            print(f'Logistic Regression Full OOF AUC: {overall_oof_auc:.6f}')
        except ValueError as ve:
            print(f"Could not calculate overall OOF AUC: {ve}. Check OOF predictions and y_train.")
    else:
        print("Could not calculate overall OOF AUC due to NaN or length mismatch in predictions.")


    return oof_predictions, test_predictions, overall_oof_auc, models

# Initialize results
lr_oof_final, lr_sub_preds_final, lr_overall_auc_final, lr_models_final = (
    np.array([]), None, 0.0, []
)

if 'X_logreg_train' in locals() and not X_logreg_train.empty and \
   'y_logreg_train' in locals() and not y_logreg_train.empty and \
   'cols_to_scale_logreg' in locals():

    print("\n--- Training Final Logistic Regression Model ---")
    lr_oof_final, lr_sub_preds_final, lr_overall_auc_final, lr_models_final = run_logistic_regression_cv_final(
        X_logreg_train,
        y_logreg_train,
        X_logreg_test if 'X_logreg_test' in locals() and not X_logreg_test.empty else pd.DataFrame(),
        cols_to_scale_logreg,
        n_folds=5,
        random_seed=12345,
        C_param=0.05 # Example C value, adjust as needed
    )
else:
    print("Training data (X_logreg_train, y_logreg_train, or cols_to_scale_logreg) is not ready. Skipping LogReg training.")


print("--- Logistic Regression Model Evaluation ---")
if 'lr_overall_auc_final' in locals() and lr_overall_auc_final > 0:
    print(f"Final Logistic Regression Full OOF AUC: {lr_overall_auc_final:.6f}")
else:
    print("Logistic Regression model was not run successfully or OOF AUC not available.")


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

if 'y_logreg_train' in locals() and 'lr_oof_final' in locals() and lr_oof_final.size > 0:
    fpr, tpr, thresholds = roc_curve(y_logreg_train, lr_oof_final)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression (OOF predictions)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
else:
    print("OOF predictions or training target variable not available for plotting ROC curve.")



if 'y_logreg_train' in locals() and 'lr_oof_final' in locals() and lr_oof_final.size > 0:
    fpr, tpr, thresholds = roc_curve(y_logreg_train, lr_oof_final)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression (OOF predictions)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    roc_data = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        'Thresholds': thresholds
    })

    display(roc_data.sample(20, random_state=42).reset_index(drop=True))

else:
    print("OOF predictions or training target variable not available for plotting ROC curve.")




if 'y_logreg_train' in locals() and 'lr_oof_final' in locals() and lr_oof_final.size > 0:
    fpr, tpr, thresholds = roc_curve(y_logreg_train, lr_oof_final)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12,8))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')

    num_points = 15
    step = max(1, len(thresholds) // num_points)
    
    for i in range(0, len(thresholds), step):
        plt.scatter(fpr[i], tpr[i], color='red', s=100, alpha=0.7, edgecolor='black', zorder=5)
        plt.annotate(f'Thresh={thresholds[i]:.2f}',
                     (fpr[i], tpr[i]),
                     textcoords="offset points",
                     xytext=(10,-10),
                     ha='left',
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", color='black'))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.title('ROC Curve with Detailed Thresholds - Logistic Regression', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.text(0.6, 0.2, f'AUC = {roc_auc:.4f}', fontsize=20, bbox=dict(facecolor='white', alpha=0.8))

    plt.show()

else:
    print("OOF predictions or training target variable not available for plotting ROC curve.")



submission_filename_lr_final = "submission_logistic_regression_final.csv"
model_source_lr_final = "N/A"

kaggle_output_path = "/kaggle/working/" 

if 'lr_sub_preds_final' in locals() and lr_sub_preds_final is not None and \
   'test_ids_logreg' in locals() and test_ids_logreg is not None and not test_ids_logreg.empty and \
   'lr_overall_auc_final' in locals() and lr_overall_auc_final > 0:

    model_source_lr_final = f"Logistic Regression (OOF AUC: {lr_overall_auc_final:.4f})"
    print(f"\nUsing predictions from: {model_source_lr_final} for submission.")

    if len(lr_sub_preds_final) == len(test_ids_logreg):
        submission_df_lr_final = pd.DataFrame({'SK_ID_CURR': test_ids_logreg, 'TARGET': lr_sub_preds_final})
        
        # Ensure the output directory exists (it usually does in Kaggle)
        if not os.path.exists(kaggle_output_path):
            os.makedirs(kaggle_output_path)
            
        submission_path_lr_final = os.path.join(kaggle_output_path, submission_filename_lr_final) # Use kaggle_output_path
        
        try:
            submission_df_lr_final.to_csv(submission_path_lr_final, index=False, float_format='%.8f')
            print(f"Submission file '{submission_filename_lr_final}' created at: {submission_path_lr_final}")
            # It's good practice to display head if display function is available
            try:
                from IPython.display import display
                display(submission_df_lr_final.head())
            except ImportError:
                print(submission_df_lr_final.head())

        except Exception as e:
            print(f"Error creating submission file: {e}")
    else:
        print(f"Error: Length mismatch for submission. SK_IDs: {len(test_ids_logreg)}, Preds: {len(lr_sub_preds_final)}.")
else:
    print("\nNo valid Logistic Regression predictions or SK_IDs available to create submission file.")








def predict_and_save_logreg_test(lr_models_final, X_logreg_test, cols_to_scale_logreg, output_path='/kaggle/working/logreg_test_predictions.csv', id_col='SK_ID_CURR', threshold=0.5):
    if X_logreg_test is None or X_logreg_test.empty:
        print("X_logreg_test is empty or not provided.")
        return
    
    X_test_scaled = X_logreg_test.copy()
    
    if cols_to_scale_logreg:
        scaler = StandardScaler()
        cols_to_scale_present = [col for col in cols_to_scale_logreg if col in X_test_scaled.columns]
        if cols_to_scale_present:
            X_test_scaled[cols_to_scale_present] = scaler.fit_transform(X_test_scaled[cols_to_scale_present])
    
    n_models = len(lr_models_final)
    if n_models == 0:
        print("No trained logistic regression models found.")
        return
    
    preds_proba = None
    for model in lr_models_final:
        pred = model.predict_proba(X_test_scaled)[:, 1]
        if preds_proba is None:
            preds_proba = pred
        else:
            preds_proba += pred
    preds_proba /= n_models
    
    preds_class = (preds_proba >= threshold).astype(int)
    
    results_df = pd.DataFrame()
    if id_col in X_logreg_test.columns:
        results_df[id_col] = X_logreg_test[id_col]
    results_df['predicted_proba'] = preds_proba
    results_df['predicted_class'] = preds_class
    
    results_df.to_csv(output_path, index=False)
    print(f"Logistic Regression test predictions saved to: {output_path}")

predict_and_save_logreg_test(
    lr_models_final=lr_models_final,
    X_logreg_test=X_logreg_test if 'X_logreg_test' in locals() else pd.DataFrame(),
    cols_to_scale_logreg=cols_to_scale_logreg if 'cols_to_scale_logreg' in locals() else [],
    output_path='/kaggle/working/logreg_test_predictions.csv',
    id_col='SK_ID_CURR',
    threshold=0.5
)








