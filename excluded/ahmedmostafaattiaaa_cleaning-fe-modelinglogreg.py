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

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, fbeta_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer 

import joblib 

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

VERBOSE_PRINT = True
# Global small constant to prevent division by zero errors
epsilon = 1e-6


# --- Function to reduce DataFrame memory usage ---

def reduce_memory_usage(df, name='', verbose=True):
    """
    Iterates through all numeric columns of a DataFrame and downcasts their data type
    to the smallest possible type that can hold their values, reducing memory usage.
    Args:
        df (pd.DataFrame): The DataFrame to process.
        name (str): A name for the DataFrame, used in print statements.
        verbose (bool): If True, prints memory usage before and after optimization.
    Returns:
        pd.DataFrame: The DataFrame with optimized memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose: print(f'Memory usage of dataframe {name} is {start_mem:.2f} MB')
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category': # Exclude object and category types
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int': 
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max: df[col] = df[col].astype(np.int64)
            else: # Float types
                if df[col].isnull().all(): # Handle columns with all NaNs
                    if pd.isna(c_min) or (c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max): df[col] = df[col].astype(np.float16)
                    elif pd.isna(c_min) or (c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max): df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max: # Try float16
                    df_temp_f16 = df[col].dropna().astype(np.float16)
                    if df_temp_f16.empty or np.allclose(df_temp_f16, df[col].dropna(), rtol=1e-3, atol=1e-3, equal_nan=True): df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: # Fallback to float32
                        df_temp_f32 = df[col].dropna().astype(np.float32)
                        if df_temp_f32.empty or np.allclose(df_temp_f32, df[col].dropna(), rtol=1e-5, atol=1e-5, equal_nan=True): df[col] = df[col].astype(np.float32)
                        else: df[col] = df[col].astype(np.float64) # Fallback to float64
                    else: df[col] = df[col].astype(np.float64) # Fallback to float64
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: # Try float32
                    df_temp_f32 = df[col].dropna().astype(np.float32)
                    if df_temp_f32.empty or np.allclose(df_temp_f32, df[col].dropna(), rtol=1e-5, atol=1e-5, equal_nan=True): df[col] = df[col].astype(np.float32)
                    else: df[col] = df[col].astype(np.float64) # Fallback to float64
                else: df[col] = df[col].astype(np.float64) # Default to float64
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage of {name} after optimization is: {end_mem:.2f} MB')
        if start_mem > 0 : print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
        else: print("Initial memory usage was zero.")
    return df


# --- Function for One-Hot Encoding categorical features ---
def one_hot_encoder(df_input, nan_as_category=True, limit_categories=50, prefix_sep='_'):
    """
    Performs one-hot encoding on categorical features of a DataFrame.
    Optionally limits the number of new features by grouping rare categories into 'Other_Aggregated'.
    Handles NaNs in categorical columns based on `nan_as_category`.
    Args:
        df_input (pd.DataFrame): DataFrame to encode.
        nan_as_category (bool): If True, NaNs are treated as a separate category.
                                If False, NaNs in categoricals are imputed with 'Missing_OHE_Placeholder_Value'.
        limit_categories (int): Maximum number of unique categories to keep per column (excluding 'Other').
                                 Rare categories are grouped.
        prefix_sep (str): Separator for new column names generated by get_dummies.
    Returns:
        tuple: (pd.DataFrame: DataFrame with OHE columns, list: names of new OHE columns)
    """
    df = df_input.copy()
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']

    if VERBOSE_PRINT and len(categorical_columns) > 0 : print(f"Found {len(categorical_columns)} categorical columns for OHE.")

    for col in categorical_columns:
        if df[col].isnull().any(): # Check if column has NaNs
            if not nan_as_category: # If not treating NaNs as a category, impute them
                df[col] = df[col].fillna('Missing_OHE_Placeholder_Value')
            # If nan_as_category is True, pd.get_dummies will handle NaNs if dummy_na=True

        if limit_categories and df[col].nunique(dropna=False) > limit_categories:
            if VERBOSE_PRINT: print(f"Limiting categories for column: {col} (had {df[col].nunique(dropna=False)} unique values)")
            # Keep top N-1 categories (where N = limit_categories), group rest into 'Other_Aggregated'
            # dropna=False in value_counts to include NaNs if they are being treated as a category
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


# --- Function to display DataFrame info ---
def display_df_info(df, name, head_n=3):
    """
    Prints the shape and head of a DataFrame for quick inspection.
    Args:
        df (pd.DataFrame): The DataFrame to display.
        name (str): A name for the DataFrame.
        head_n (int): Number of rows to display from the head.
    """
    if df is not None and not df.empty:
        print(f"\n--- Info for: {name} ---")
        print(f"Shape: {df.shape}")
        display(df.head(head_n))
        gc.collect() # Perform garbage collection
    else: print(f"{name} is not loaded or is empty.")


# --- Function to clean column names for model compatibility ---
def clean_col_names(df):
    """
    Cleans column names by replacing non-alphanumeric characters with underscores
    and handling duplicate column names that might arise after cleaning.
    Args:
        df (pd.DataFrame): DataFrame whose column names need cleaning.
    Returns:
        pd.DataFrame: DataFrame with cleaned column names.
    """
    df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    # Handle duplicate column names (e.g., if "Feature A" and "Feature_A" both become "Feature_A")
    cols = pd.Series(df.columns)
    for dup_col_name in cols[cols.duplicated()].unique():
        # Get indices of all occurrences of the duplicated column name
        dup_indices = cols[cols == dup_col_name].index.tolist()
        # Append suffix _0, _1, _2, etc. to make them unique, keeping the first as is
        for i, idx in enumerate(dup_indices):
            if i > 0: # Only change subsequent duplicates
                cols[idx] = f"{dup_col_name}_{i}"
    df.columns = cols
    return df


base_path = '/kaggle/input/home-credit-default-risk/'
kaggle_working_path = '/kaggle/working/'

print(f"Data input path set to: {base_path}")
print(f"Data output path set to: {kaggle_working_path}")

if not os.path.exists(base_path):
    print(f"CRITICAL WARNING: Kaggle input path {base_path} does not exist or is not accessible.")
    print("Please ensure the 'Home Credit Default Risk' dataset is correctly added to this Kaggle Notebook's input sources.")
else:
    print(f"Kaggle input path {base_path} confirmed.")
    try:
        print(f"Sample files in input path: {os.listdir(base_path)[:5]}...") # List first 5 files/dirs
    except Exception as e:
        print(f"Could not list files in input path: {e}")


# Ensure the working directory exists (it should by default in Kaggle)
if not os.path.exists(kaggle_working_path):
    try:
        os.makedirs(kaggle_working_path)
        print(f"Kaggle working path {kaggle_working_path} created.")
    except Exception as e:
        print(f"Could not create Kaggle working path {kaggle_working_path}: {e}")

else:
    print(f"Kaggle working path {kaggle_working_path} confirmed.")


# Define file names within the base_path
file_names = {
    'app_train': 'application_train.csv',
    'app_test': 'application_test.csv',
    'bureau': 'bureau.csv',
    'bureau_balance': 'bureau_balance.csv',
    'previous_application': 'previous_application.csv',
    'pos_cash': 'POS_CASH_balance.csv',
    'installments_payments': 'installments_payments.csv',
    'credit_card_balance': 'credit_card_balance.csv'
}

dfs = {}  # Dictionary to store all loaded DataFrames

print("--- Starting Data Loading ---")

for name_key, file_name_val in file_names.items():
    full_path = os.path.join(base_path, file_name_val)
    print(f"\nAttempting to load: {name_key} from {full_path}")
    
    try:
        dfs[name_key] = pd.read_csv(full_path)
        dfs[name_key] = reduce_memory_usage(
            dfs[name_key],
            name=name_key,
            verbose=VERBOSE_PRINT
        )
        print(f"SUCCESS: {name_key} loaded. Shape: {dfs[name_key].shape}")

    except FileNotFoundError:
        print(f"ERROR: File not found at {full_path}.")

        # Specific fallback for POS_CASH_balance with a potential space in filename
        if name_key == 'pos_cash' and file_name_val == 'POS_CASH_balance.csv':
            alt_file_name = 'pos_cash_balance .csv'  # With a space
            alt_full_path = os.path.join(base_path, alt_file_name)

            if os.path.exists(alt_full_path):
                print(f"Attempting alternative load for {name_key} from {alt_full_path}...")
                try:
                    dfs[name_key] = pd.read_csv(alt_full_path)
                    dfs[name_key] = reduce_memory_usage(
                        dfs[name_key],
                        name=name_key + " (alt name)",
                        verbose=VERBOSE_PRINT
                    )
                    print(f"SUCCESS (alt name): {name_key} loaded. Shape: {dfs[name_key].shape}")
                except Exception as e_alt:
                    print(f"ERROR (alt name): Failed to load {alt_full_path}: {e_alt}")
                    dfs[name_key] = pd.DataFrame()
            else:
                dfs[name_key] = pd.DataFrame()

        else:
            dfs[name_key] = pd.DataFrame()

    except Exception as e_load:
        print(f"ERROR: An unexpected error occurred while loading {full_path}: {e_load}")
        dfs[name_key] = pd.DataFrame()

    finally:
        gc.collect()  # Garbage collect after each file load attempt

print("\n--- Data Loading Complete ---")

# Display info for the main application DataFrames
display_df_info(dfs.get('app_train'), "Initial application_train DataFrame")
display_df_info(dfs.get('app_test'), "Initial application_test DataFrame")



# ------------------------------------------------------------------------------
# Preprocessing and Feature Engineering for Logistic Regression on Application Data
# ------------------------------------------------------------------------------

def preprocess_and_fe_application_logreg(df_input, median_dict_storage, is_train_set=True):
    df = df_input.copy()
    if VERBOSE_PRINT: print(f"LogReg FE: Processing application data. Shape: {df.shape}")
    global epsilon # Make sure epsilon is accessible

    # 1. Handle and Convert Days Columns
    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
        df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
    if 'DAYS_BIRTH' in df.columns:
        df['DAYS_BIRTH_YEARS'] = df['DAYS_BIRTH'] / -365.25 # Will be positive
        df['DAYS_BIRTH_YEARS'].fillna(df['DAYS_BIRTH_YEARS'].median() if is_train_set else median_dict_storage.get('app_median_DAYS_BIRTH_YEARS',0), inplace=True) # Impute after creation
        if is_train_set: median_dict_storage['app_median_DAYS_BIRTH_YEARS'] = df['DAYS_BIRTH_YEARS'].median()


    for col in ['DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE']:
        if col in df.columns:
            df[col + '_POSITIVE_DAYS'] = (df[col] * -1) # Convert, keep NaNs for now
            df[f'{col}_POSITIVE_DAYS_NAN_FLAG'] = df[col + '_POSITIVE_DAYS'].isnull().astype(int)
            if is_train_set:
                median_val = df[col + '_POSITIVE_DAYS'].median()
                median_dict_storage[f'app_median_{col}_POSITIVE_DAYS'] = median_val if pd.notna(median_val) else 0
            impute_val = median_dict_storage.get(f'app_median_{col}_POSITIVE_DAYS', 0)
            df[col + '_POSITIVE_DAYS'].fillna(impute_val, inplace=True)

    # 2. Categorical Imputation
    cat_cols_to_impute_mode = {'OCCUPATION_TYPE': 'Unknown_Occupation', 'FONDKAPREMONT_MODE': 'not_specified','WALLSMATERIAL_MODE': 'Unknown_Material', 'HOUSETYPE_MODE': 'block_of_flats','NAME_TYPE_SUITE': 'Unaccompanied', 'NAME_FAMILY_STATUS': 'Married'}
    for col, fill_val in cat_cols_to_impute_mode.items():
        if col in df.columns: # Impute NaNs directly
            df[col].fillna(fill_val, inplace=True)

    if 'CODE_GENDER' in df.columns:
        df['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
        mode_gender = 'F'; temp_series = df['CODE_GENDER'].dropna();
        if not temp_series.mode().empty: mode_gender = temp_series.mode()[0]
        if is_train_set: median_dict_storage['app_cat_mode_CODE_GENDER'] = mode_gender
        df['CODE_GENDER'].fillna(median_dict_storage.get('app_cat_mode_CODE_GENDER', 'F'), inplace=True)
    if 'EMERGENCYSTATE_MODE' in df.columns: df['EMERGENCYSTATE_MODE'].fillna('No', inplace=True)

    # 3. Numerical Imputation
    numerical_cols_to_impute_median = ['AMT_ANNUITY', 'AMT_GOODS_PRICE', 'CNT_FAM_MEMBERS', 'OWN_CAR_AGE','EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_EMPLOYED','APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG','YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG','FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG','LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG','APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE','YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE','FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE','LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE','APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI','YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI','FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI','LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI','TOTALAREA_MODE']
    bureau_req_cols_app = [col for col in df.columns if 'AMT_REQ_CREDIT_BUREAU' in col]
    numerical_cols_to_impute_median.extend(bureau_req_cols_app)
    for col in numerical_cols_to_impute_median:
        if col in df.columns:
            df[f'{col}_NAN_FLAG'] = df[col].isnull().astype(int)
            if is_train_set:
                median_val = df[col].median()
                median_dict_storage[f'app_num_median_{col}'] = median_val if pd.notna(median_val) else 0
            impute_val = median_dict_storage.get(f'app_num_median_{col}', 0)
            df[col].fillna(impute_val, inplace=True)

    # 4. Create Final _YEARS versions
    if 'DAYS_EMPLOYED' in df.columns: df['DAYS_EMPLOYED_YEARS'] = (df['DAYS_EMPLOYED'] / -365.25).abs().fillna(0)
    for col_base in ['REGISTRATION', 'ID_PUBLISH', 'LAST_PHONE_CHANGE']:
        days_col_positive = f'DAYS_{col_base}_POSITIVE_DAYS'
        years_col = f'DAYS_{col_base}_YEARS'
        if days_col_positive in df.columns: df[years_col] = (df[days_col_positive] / 365.25).fillna(0)
        elif f'DAYS_{col_base}' in df.columns: df[years_col] = ((df[f'DAYS_{col_base}'] * -1).abs().fillna(0) / 365.25).fillna(0)


    # 5. Feature Engineering
    ratio_cols_to_handle = []
    if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'].replace(0, epsilon) + epsilon); ratio_cols_to_handle.append('CREDIT_TO_INCOME_RATIO')
    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'].replace(0, epsilon) + epsilon); ratio_cols_to_handle.append('ANNUITY_TO_INCOME_RATIO')
    # ... (apply to other ratio creations)
    for ratio_col in ratio_cols_to_handle: # And all other ratio columns created
        if ratio_col in df.columns:
            df[ratio_col].fillna(0, inplace=True)
            df[ratio_col].replace([np.inf, -np.inf], 0, inplace=True)
    # (Ensure all other new features are also handled for NaNs/Infs)
    # Example for EXT_SOURCES_PROD (if it can be 0 leading to inf in ratios later)
    ext_sources = [col for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if col in df.columns]
    if ext_sources:
        df['EXT_SOURCES_MEAN'] = df[ext_sources].mean(axis=1).fillna(0)
        df['EXT_SOURCES_SUM'] = df[ext_sources].sum(axis=1).fillna(0)
        # For prod, if any source is 0, prod is 0. If any is NaN (after imputation, shouldn't be), prod is NaN.
        # skipna=True means if one is NaN, it's ignored. skipna=False means if one is NaN, result is NaN.
        # Since EXT_SOURCEs are imputed to 0 if NaN, prod will be 0 if any original was NaN.
        df['EXT_SOURCES_PROD'] = df[ext_sources].prod(axis=1, skipna=False).fillna(0) # skipna=False and then fillna
        df['EXT_SOURCES_MIN'] = df[ext_sources].min(axis=1).fillna(0)
        df['EXT_SOURCES_MAX'] = df[ext_sources].max(axis=1).fillna(0)
        for source in ext_sources:
            df[f'{source}_SQ'] = (df[source]**2).fillna(0)
            if 'DAYS_BIRTH_YEARS' in df.columns: df[f'{source}_X_AGE_YEARS'] = (df[source] * df['DAYS_BIRTH_YEARS']).fillna(0)
            if 'DAYS_EMPLOYED_YEARS' in df.columns: df[f'{source}_X_EMPLOYED_YEARS'] = (df[source] * df['DAYS_EMPLOYED_YEARS']).fillna(0)


    flag_doc_cols = [col for col in df.columns if 'FLAG_DOCUMENT_' in col]
    if flag_doc_cols: df['DOC_COUNT'] = df[flag_doc_cols].sum(axis=1)
    phone_email_flags = [col for col in ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE','FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL'] if col in df.columns]
    if phone_email_flags: df['PHONE_EMAIL_FLAGS_SUM'] = df[phone_email_flags].sum(axis=1)

    # 6. One-Hot Encode
    df, _ = one_hot_encoder(df, nan_as_category=False, limit_categories=50)

    if VERBOSE_PRINT: print(f"LogReg FE: Application FE finished. Shape: {df.shape}")
    return df, median_dict_storage



def fe_bureau_and_balance_logreg(bureau_input_df, bb_input_df, median_dict_storage, is_train_set=True, verbose=False):
    if bureau_input_df.empty:
        if verbose: print("Bureau data is empty. Skipping Bureau FE.")
        return pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage

    bureau = bureau_input_df.copy()
    if verbose: print(f"LogReg FE: Processing Bureau data. Initial shape: {bureau.shape}")
    global epsilon # Ensure epsilon is accessible

    if 'DAYS_CREDIT' in bureau.columns:
        bureau['DAYS_CREDIT_POSITIVE'] = (bureau['DAYS_CREDIT'] * -1).fillna(0)
    if 'DAYS_CREDIT_UPDATE' in bureau.columns:
        bureau['DAYS_CREDIT_UPDATE_POSITIVE'] = (bureau['DAYS_CREDIT_UPDATE'] * -1).fillna(0)

    bureau_cols_to_impute_spec = {
        'DAYS_CREDIT_ENDDATE': 'median_or_relative',
        'DAYS_ENDDATE_FACT': 'median_or_relative',
        'AMT_CREDIT_MAX_OVERDUE': 0.0,
        'AMT_ANNUITY': 0.0,
        'AMT_CREDIT_SUM_DEBT': 0.0,
        'AMT_CREDIT_SUM_LIMIT': 0.0,
        'AMT_CREDIT_SUM_OVERDUE': 0.0,
        'CNT_CREDIT_PROLONG': 0
    }

    for col, impute_strategy_or_value in bureau_cols_to_impute_spec.items():
        if col in bureau.columns:
            bureau[f'{col}_NAN_FLAG'] = bureau[col].isnull().astype(int)
            if is_train_set:
                if impute_strategy_or_value == 'median_or_relative':
                    median_val = bureau[col].median()
                    fallback_val = 0
                    if col == 'DAYS_CREDIT_ENDDATE' and 'DAYS_CREDIT' in bureau.columns and pd.notna(bureau['DAYS_CREDIT'].median()):
                        fallback_val = bureau['DAYS_CREDIT'].median() - 1
                    elif col == 'DAYS_ENDDATE_FACT' and 'DAYS_CREDIT_ENDDATE' in bureau.columns and pd.notna(bureau['DAYS_CREDIT_ENDDATE'].median()): # Use already imputed DAYS_CREDIT_ENDDATE
                        fallback_val = bureau['DAYS_CREDIT_ENDDATE'].median()

                    median_dict_storage[f'bureau_num_median_{col}'] = median_val if pd.notna(median_val) else fallback_val
                elif isinstance(impute_strategy_or_value, (int, float)):
                    median_dict_storage[f'bureau_num_median_{col}'] = impute_strategy_or_value
                else:
                    median_val = bureau[col].median()
                    median_dict_storage[f'bureau_num_median_{col}'] = median_val if pd.notna(median_val) else 0
            
            impute_val_final = median_dict_storage.get(f'bureau_num_median_{col}', 0)
            bureau[col].fillna(impute_val_final, inplace=True)
        elif col in bureau.columns: 
             bureau[f'{col}_NAN_FLAG'] = 0


    if 'DAYS_CREDIT_ENDDATE' in bureau.columns and 'DAYS_CREDIT' in bureau.columns:
        bureau['BUREAU_LOAN_DURATION_DAYS'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_CREDIT']
        bureau['BUREAU_LOAN_DURATION_DAYS_NAN_FLAG'] = bureau['BUREAU_LOAN_DURATION_DAYS'].isnull().astype(int)
        bureau['BUREAU_LOAN_DURATION_DAYS'].fillna(0, inplace=True)

    if 'AMT_CREDIT_SUM_DEBT' in bureau.columns and 'AMT_CREDIT_SUM' in bureau.columns:
        bureau['BUREAU_DEBT_TO_CREDIT_RATIO'] = bureau['AMT_CREDIT_SUM_DEBT'] / (bureau['AMT_CREDIT_SUM'].replace(0, epsilon) + epsilon)
        bureau['BUREAU_DEBT_TO_CREDIT_RATIO'].fillna(0, inplace=True) # Separated
        bureau['BUREAU_DEBT_TO_CREDIT_RATIO'].replace([np.inf, -np.inf], 0, inplace=True) # Separated

    # --- Bureau Balance Aggregations ---
    bb_final_agg_by_curr_df = pd.DataFrame() # Initialize
    if bb_input_df is not None and not bb_input_df.empty:
        bb = bb_input_df.copy()
        if verbose: print(f"LogReg FE: Processing Bureau Balance data. Shape: {bb.shape}")
        status_map_bb = {'C': 0, 'X': -1, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
        bb['BB_STATUS_NUMERIC'] = bb['STATUS'].map(status_map_bb).fillna(-2)
        bb, bb_cat_cols = one_hot_encoder(bb, nan_as_category=False, limit_categories=7, prefix_sep='_BB_STATUS_')

        bb_aggregations = {'MONTHS_BALANCE': ['count', 'min', 'max', 'mean'], 'BB_STATUS_NUMERIC': ['mean', 'max', 'min', 'std']}
        for bb_ohe_col in bb_cat_cols:
            if bb_ohe_col in bb.columns and bb_ohe_col != 'SK_ID_BUREAU': bb_aggregations[bb_ohe_col] = ['sum', 'mean']
        
        bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
        bb_agg.columns = pd.Index(['BB_' + e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
        
        if 'SK_ID_BUREAU' in bureau_input_df.columns and 'SK_ID_CURR' in bureau_input_df.columns:
            bb_merged_with_curr = bureau_input_df[['SK_ID_CURR', 'SK_ID_BUREAU']].drop_duplicates().merge(
                bb_agg.reset_index(), on='SK_ID_BUREAU', how='left'
            )
            bb_cols_to_agg_by_curr = [col for col in bb_agg.columns if col != 'SK_ID_BUREAU'] # Use columns from bb_agg
            if not bb_merged_with_curr.empty and bb_cols_to_agg_by_curr:
                for bb_col_fill in bb_cols_to_agg_by_curr: bb_merged_with_curr[bb_col_fill].fillna(0, inplace=True)
                
                agg_functions_for_bb_final = ['sum', 'mean', 'max', 'min', 'std']
                bb_final_agg_by_curr_df = bb_merged_with_curr.drop(columns=['SK_ID_BUREAU'], errors='ignore').groupby('SK_ID_CURR').agg({
                    col: agg_functions_for_bb_final for col in bb_cols_to_agg_by_curr if col in bb_merged_with_curr.columns # Ensure col exists
                })
                bb_final_agg_by_curr_df.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_final_agg_by_curr_df.columns.tolist()])
        del bb, bb_agg; gc.collect()
    else:
        if verbose: print("Bureau Balance data not provided or empty.")

    # --- Aggregating Bureau features by SK_ID_CURR ---
    bureau_agg_spec_final = {
        'DAYS_CREDIT_POSITIVE': ['count', 'mean', 'max', 'min', 'sum', 'std'],
        'DAYS_CREDIT_UPDATE_POSITIVE': ['mean', 'max', 'min', 'std'],
        'CREDIT_DAY_OVERDUE': ['sum', 'mean', 'max'],
        'AMT_CREDIT_SUM': ['sum', 'mean', 'max', 'std'],
        'AMT_CREDIT_SUM_DEBT': ['sum', 'mean', 'max', 'std'],
        'AMT_CREDIT_SUM_OVERDUE': ['sum', 'mean', 'max'],
        'AMT_ANNUITY': ['sum', 'mean', 'max', 'std'],
        'CNT_CREDIT_PROLONG': ['sum', 'mean'],
        'BUREAU_LOAN_DURATION_DAYS': ['mean', 'max', 'min', 'std'],
        'BUREAU_DEBT_TO_CREDIT_RATIO':['mean','max','min','std']
    }
    for col_key, _ in bureau_cols_to_impute_spec.items():
        if f'{col_key}_NAN_FLAG' in bureau.columns: bureau_agg_spec_final[f'{col_key}_NAN_FLAG'] = ['sum', 'mean']
    if 'BUREAU_LOAN_DURATION_DAYS_NAN_FLAG' in bureau.columns:
        bureau_agg_spec_final['BUREAU_LOAN_DURATION_DAYS_NAN_FLAG'] = ['sum','mean']

    bureau_agg_dict_final = {k:v for k,v in bureau_agg_spec_final.items() if k in bureau.columns}
    bureau_ohe, bureau_ohe_cat_cols = one_hot_encoder(bureau.drop(columns=['SK_ID_BUREAU'], errors='ignore'), nan_as_category=False, limit_categories=10)

    for cat_col in bureau_ohe_cat_cols:
        if cat_col in bureau_ohe.columns and cat_col != 'SK_ID_CURR':
            bureau_agg_dict_final[cat_col] = ['sum', 'mean']

    if 'SK_ID_CURR' not in bureau_ohe.columns:
        print("CRITICAL WARNING: SK_ID_CURR is missing from bureau_ohe before final aggregation.")
        return pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage
        
    bureau_final_agg = bureau_ohe.groupby('SK_ID_CURR').agg(bureau_agg_dict_final)
    bureau_final_agg.columns = pd.Index(['BUREAU_' + e[0] + "_" + e[1].upper() for e in bureau_final_agg.columns.tolist()])
    bureau_final_agg_res = bureau_final_agg.reset_index()

    # Merge with aggregated bureau_balance features
    if not bb_final_agg_by_curr_df.empty:
        bureau_final_agg_res = bureau_final_agg_res.merge(bb_final_agg_by_curr_df.reset_index(), on='SK_ID_CURR', how='left')
        # Fill NaNs that might result from the left merge if some SK_ID_CURR in bureau_final_agg_res don't have BB info
        bb_final_cols = [col for col in bureau_final_agg_res.columns if col.startswith("BB_")]
        for col in bb_final_cols: bureau_final_agg_res[col].fillna(0, inplace=True)


    if verbose: print(f"LogReg FE: Bureau & BB FE finished. Aggregated shape: {bureau_final_agg_res.shape}")
    return bureau_final_agg_res if not bureau_final_agg_res.empty else pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage


def fe_previous_applications_logreg(prev_app_df_input, median_dict_storage, is_train_set=True, verbose=False):
    if prev_app_df_input.empty: return pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage
    prev_df = prev_app_df_input.copy()
    if verbose: print(f"LogReg FE: Processing Previous Application data. Initial shape: {prev_df.shape}")
    global epsilon

    days_cols_prev_anomaly = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
    for col in days_cols_prev_anomaly:
        if col in prev_df.columns: prev_df[col].replace(365243, np.nan, inplace=True)

    cols_to_impute_prev = ['AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'CNT_PAYMENT','AMT_APPLICATION', 'AMT_DOWN_PAYMENT','RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED']
    cols_to_impute_prev.extend(days_cols_prev_anomaly)
    if 'DAYS_DECISION' in prev_df.columns: cols_to_impute_prev.append('DAYS_DECISION')

    for cat_col_prev in prev_df.select_dtypes(include='object').columns: # Impute categoricals
        if prev_df[cat_col_prev].isnull().any(): prev_df[cat_col_prev].fillna('Unknown_PrevApp_Cat', inplace=True)

    for col in cols_to_impute_prev:
        if col in prev_df.columns:
            prev_df[f'{col}_NAN_FLAG'] = prev_df[col].isnull().astype(int)
            if is_train_set:
                median_val = prev_df[col].median()
                median_dict_storage[f'prev_app_median_{col}'] = median_val if pd.notna(median_val) else 0
            impute_val = median_dict_storage.get(f'prev_app_median_{col}', 0)
            prev_df[col].fillna(impute_val, inplace=True)
        elif col in prev_df.columns: prev_df[f'{col}_NAN_FLAG'] = 0

    if 'DAYS_DECISION' in prev_df.columns: prev_df['DAYS_DECISION_POSITIVE_DAYS'] = (prev_df['DAYS_DECISION'] * -1).fillna(0)
    for col_day in days_cols_prev_anomaly:
        if col_day in prev_df.columns: prev_df[col_day + '_POSITIVE_DAYS'] = (prev_df[col_day] * -1).abs().fillna(0)

    if 'AMT_APPLICATION' in prev_df.columns and 'AMT_CREDIT' in prev_df.columns:
        prev_df['PREV_APP_CREDIT_GRANTED_RATIO'] = prev_df['AMT_CREDIT'] / (prev_df['AMT_APPLICATION'].replace(0, epsilon) + epsilon)
        prev_df['PREV_APP_CREDIT_GRANTED_RATIO'].fillna(0, inplace=True)
        prev_df['PREV_APP_CREDIT_GRANTED_RATIO'].replace([np.inf, -np.inf], 0, inplace=True)
    if 'AMT_ANNUITY' in prev_df.columns and 'AMT_CREDIT' in prev_df.columns:
        prev_df['PREV_ANNUITY_TO_CREDIT_RATIO'] = prev_df['AMT_ANNUITY'] / (prev_df['AMT_CREDIT'].replace(0,epsilon) + epsilon)
        prev_df['PREV_ANNUITY_TO_CREDIT_RATIO'].fillna(0, inplace=True)
        prev_df['PREV_ANNUITY_TO_CREDIT_RATIO'].replace([np.inf, -np.inf], 0, inplace=True)

    prev_agg_spec = {'AMT_ANNUITY': ['sum', 'mean', 'max', 'min', 'std'], 'AMT_CREDIT': ['sum', 'mean', 'max', 'min', 'std'],'AMT_APPLICATION': ['sum', 'mean', 'max'],'AMT_GOODS_PRICE': ['sum', 'mean', 'max'], 'DAYS_DECISION_POSITIVE_DAYS': ['count', 'mean', 'max', 'min', 'sum', 'std'], 'CNT_PAYMENT': ['sum', 'mean', 'max', 'min', 'std'], 'PREV_APP_CREDIT_GRANTED_RATIO': ['mean', 'min', 'max', 'std'], 'PREV_ANNUITY_TO_CREDIT_RATIO': ['mean', 'min', 'max', 'std']}
    for col_key in cols_to_impute_prev:
        if f'{col_key}_NAN_FLAG' in prev_df.columns: prev_agg_spec[f'{col_key}_NAN_FLAG'] = ['sum', 'mean']
    prev_agg_dict_final = {k:v for k,v in prev_agg_spec.items() if k in prev_df.columns}

    prev_df_ohe, prev_ohe_cat_cols = one_hot_encoder(prev_df, nan_as_category=False, limit_categories=15)
    for cat_col in prev_ohe_cat_cols:
        if cat_col in prev_df_ohe.columns and cat_col not in ['SK_ID_CURR', 'SK_ID_PREV']:
            prev_agg_dict_final[cat_col] = ['sum', 'mean']

    if 'SK_ID_CURR' not in prev_df_ohe.columns: return pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage
    prev_final_agg = prev_df_ohe.drop(columns=['SK_ID_PREV'], errors='ignore').groupby('SK_ID_CURR').agg(prev_agg_dict_final)
    prev_final_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_final_agg.columns.tolist()])
    prev_final_agg_res = prev_final_agg.reset_index()

    if verbose: print(f"LogReg FE: Previous Application FE finished. Aggregated shape: {prev_final_agg_res.shape}")
    return prev_final_agg_res if not prev_final_agg_res.empty else pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage



def fe_pos_cash_logreg(pos_cash_df_input, verbose=False):
    if pos_cash_df_input.empty:
        if verbose: print("POS CASH Balance data is empty. Skipping POS CASH FE.")
        return pd.DataFrame(columns=['SK_ID_CURR'])

    pos_df = pos_cash_df_input.copy()
    if verbose: print(f"LogReg FE: Processing POS CASH Balance data. Initial shape: {pos_df.shape}")

    # 1. Convert MONTHS_BALANCE to positive "months ago" and impute
    if 'MONTHS_BALANCE' in pos_df.columns:
        pos_df['POS_MONTHS_AGO'] = (pos_df['MONTHS_BALANCE'] * -1).fillna(0)

    # 2. Impute DPD and installment count columns with 0
    cols_to_impute_pos = ['SK_DPD', 'SK_DPD_DEF', 'CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE']
    for col in cols_to_impute_pos:
        if col in pos_df.columns:
            pos_df[f'{col}_NAN_FLAG'] = pos_df[col].isnull().astype(int) # Create NaN flag
            pos_df[col].fillna(0, inplace=True)
        elif col in pos_df.columns: # Ensure flag exists even if no NaNs
            pos_df[f'{col}_NAN_FLAG'] = 0


    # 3. Aggregation Specification
    pos_agg_spec = {
        'POS_MONTHS_AGO': ['count', 'mean', 'max', 'min', 'std'], # Min MONTHS_AGO is most recent
        'SK_DPD': ['sum', 'mean', 'max', 'std'],
        'SK_DPD_DEF': ['sum', 'mean', 'max', 'std'],
        'CNT_INSTALMENT': ['mean', 'sum', 'max', 'min', 'std'],
        'CNT_INSTALMENT_FUTURE': ['mean', 'sum', 'max', 'min', 'std']
    }
    # Add NaN flag aggregations
    for col_key in cols_to_impute_pos:
        if f'{col_key}_NAN_FLAG' in pos_df.columns:
            pos_agg_spec[f'{col_key}_NAN_FLAG'] = ['sum', 'mean']

    pos_agg_dict_final = {k:v for k,v in pos_agg_spec.items() if k in pos_df.columns}

    # 4. One-Hot Encode `NAME_CONTRACT_STATUS`
    # Impute NaNs in NAME_CONTRACT_STATUS before OHE
    if 'NAME_CONTRACT_STATUS' in pos_df.columns and pos_df['NAME_CONTRACT_STATUS'].isnull().any():
        pos_df['NAME_CONTRACT_STATUS'].fillna('Unknown_POS_Status', inplace=True)

    pos_df_ohe, pos_ohe_cat_cols = one_hot_encoder(pos_df, nan_as_category=False, limit_categories=7, prefix_sep='_POS_STATUS_') # Limit categories for status

    for cat_col in pos_ohe_cat_cols:
         if cat_col in pos_df_ohe.columns and cat_col not in ['SK_ID_CURR', 'SK_ID_PREV']:
            pos_agg_dict_final[cat_col] = ['sum', 'mean']

    # 5. Perform final aggregation
    if 'SK_ID_CURR' not in pos_df_ohe.columns:
        print("CRITICAL WARNING: SK_ID_CURR missing from pos_df_ohe. Cannot aggregate.")
        return pd.DataFrame(columns=['SK_ID_CURR'])
        
    pos_final_agg = pos_df_ohe.drop(columns=['SK_ID_PREV'], errors='ignore').groupby('SK_ID_CURR').agg(pos_agg_dict_final)
    pos_final_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_final_agg.columns.tolist()])
    
    pos_final_agg_res = pos_final_agg.reset_index()

    if verbose: print(f"LogReg FE: POS CASH FE finished. Aggregated shape: {pos_final_agg_res.shape}")
    return pos_final_agg_res if not pos_final_agg_res.empty else pd.DataFrame(columns=['SK_ID_CURR'])


def fe_installments_logreg(installments_df_input, median_dict_storage, is_train_set=True, verbose=False):
    if installments_df_input.empty: return pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage
    pay_df = installments_df_input.copy()
    if verbose: print(f"LogReg FE: Processing Installments Payments data. Initial shape: {pay_df.shape}")
    global epsilon

    for day_col_inst in ['DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT']:
        if day_col_inst in pay_df.columns:
            pay_df[day_col_inst + '_AGO'] = (pay_df[day_col_inst] * -1)
            pay_df[f'{day_col_inst}_AGO_NAN_FLAG'] = pay_df[day_col_inst + '_AGO'].isnull().astype(int)
            if is_train_set:
                median_val = pay_df[day_col_inst + '_AGO'].median()
                median_dict_storage[f'install_median_{day_col_inst}_AGO'] = median_val if pd.notna(median_val) else 0
            impute_val = median_dict_storage.get(f'install_median_{day_col_inst}_AGO', 0)
            pay_df[day_col_inst + '_AGO'].fillna(impute_val, inplace=True)
        elif day_col_inst in pay_df.columns: pay_df[f'{day_col_inst}_AGO_NAN_FLAG'] = 0

    for amt_col_inst in ['AMT_PAYMENT', 'AMT_INSTALMENT']:
        if amt_col_inst in pay_df.columns:
            pay_df[f'{amt_col_inst}_NAN_FLAG'] = pay_df[amt_col_inst].isnull().astype(int)
            if is_train_set:
                median_val = pay_df[amt_col_inst].median()
                median_dict_storage[f'install_median_{amt_col_inst}'] = median_val if pd.notna(median_val) else 0
            impute_val = median_dict_storage.get(f'install_median_{amt_col_inst}', 0)
            pay_df[amt_col_inst].fillna(impute_val, inplace=True)
        elif amt_col_inst in pay_df.columns: pay_df[f'{amt_col_inst}_NAN_FLAG'] = 0

    if 'DAYS_ENTRY_PAYMENT_AGO' in pay_df.columns and 'DAYS_INSTALMENT_AGO' in pay_df.columns:
        pay_df['INS_DPD_CALC'] = pay_df['DAYS_INSTALMENT_AGO'] - pay_df['DAYS_ENTRY_PAYMENT_AGO']
        pay_df['INS_DPD_CALC'] = pay_df['INS_DPD_CALC'].apply(lambda x: x if x > 0 else 0).fillna(0)
    if 'AMT_INSTALMENT' in pay_df.columns and 'AMT_PAYMENT' in pay_df.columns:
        pay_df['INS_PAYMENT_PERC'] = pay_df['AMT_PAYMENT'] / (pay_df['AMT_INSTALMENT'].replace(0,epsilon) + epsilon)
        pay_df['INS_PAYMENT_DIFF_AMT'] = pay_df['AMT_INSTALMENT'] - pay_df['AMT_PAYMENT']
        pay_df['INS_PAYMENT_PERC'].fillna(0, inplace=True); pay_df['INS_PAYMENT_PERC'].replace([np.inf, -np.inf], 0, inplace=True)
        pay_df['INS_PAYMENT_DIFF_AMT'].fillna(0, inplace=True) # Diff can be negative, no inf replace needed unless AMT_INSTALMENT was inf

    ins_agg_spec = {'DAYS_INSTALMENT_AGO': ['count', 'mean', 'max', 'min', 'std'],'DAYS_ENTRY_PAYMENT_AGO': ['mean', 'max', 'min', 'std'],'AMT_PAYMENT': ['sum', 'mean', 'max', 'min', 'std'],'AMT_INSTALMENT': ['sum', 'mean', 'max', 'min', 'std'],'INS_DPD_CALC': ['sum', 'mean', 'max', 'std', 'count'],'INS_PAYMENT_PERC': ['mean', 'std', 'min', 'max'],'INS_PAYMENT_DIFF_AMT':['sum','mean','max','min','std']}
    for day_col_flag in ['DAYS_INSTALMENT_AGO_NAN_FLAG', 'DAYS_ENTRY_PAYMENT_AGO_NAN_FLAG']:
        if day_col_flag in pay_df.columns: ins_agg_spec[day_col_flag] = ['sum', 'mean']
    for amt_col_flag in ['AMT_PAYMENT_NAN_FLAG', 'AMT_INSTALMENT_NAN_FLAG']:
        if amt_col_flag in pay_df.columns: ins_agg_spec[amt_col_flag] = ['sum', 'mean']
    ins_agg_dict_final = {k:v for k,v in ins_agg_spec.items() if k in pay_df.columns}

    if 'SK_ID_CURR' not in pay_df.columns: return pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage
    ins_final_agg = pay_df.drop(columns=['SK_ID_PREV'], errors='ignore').groupby('SK_ID_CURR').agg(ins_agg_dict_final)
    ins_final_agg.columns = pd.Index(['INS_' + e[0] + "_" + e[1].upper() for e in ins_final_agg.columns.tolist()])
    ins_final_agg_res = ins_final_agg.reset_index()

    if verbose: print(f"LogReg FE: Installments FE finished. Aggregated shape: {ins_final_agg_res.shape}")
    return ins_final_agg_res if not ins_final_agg_res.empty else pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage


def fe_credit_card_logreg(cc_balance_df_input, median_dict_storage, is_train_set=True, verbose=False):
    if cc_balance_df_input.empty: return pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage
    cc_df = cc_balance_df_input.copy()
    if verbose: print(f"LogReg FE: Processing Credit Card Balance data. Initial shape: {cc_df.shape}")
    global epsilon

    if 'MONTHS_BALANCE' in cc_df.columns: cc_df['CC_MONTHS_AGO'] = (cc_df['MONTHS_BALANCE'] * -1).fillna(0)
    
    cols_to_impute_cc = ['AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT','AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT','AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT','AMT_RECEIVABLE_PRINCIPAL', 'AMT_TOTAL_RECEIVABLE','CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT','CNT_DRAWINGS_POS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM', 'SK_DPD', 'SK_DPD_DEF']
    if 'AMT_RECIVABLE' in cc_df.columns: cc_df.rename(columns={'AMT_RECIVABLE':'AMT_RECEIVABLE'}, inplace=True);
    if 'AMT_RECEIVABLE' not in cols_to_impute_cc and 'AMT_RECEIVABLE' in cc_df.columns : cols_to_impute_cc.append('AMT_RECEIVABLE')

    for col in cols_to_impute_cc:
        if col in cc_df.columns:
            cc_df[f'{col}_NAN_FLAG'] = cc_df[col].isnull().astype(int)
            if is_train_set:
                median_val = cc_df[col].median()
                median_dict_storage[f'cc_median_{col}'] = median_val if pd.notna(median_val) else 0
            impute_val = median_dict_storage.get(f'cc_median_{col}', 0)
            cc_df[col].fillna(impute_val, inplace=True)
        elif col in cc_df.columns: cc_df[f'{col}_NAN_FLAG'] = 0

    if 'AMT_BALANCE' in cc_df.columns and 'AMT_CREDIT_LIMIT_ACTUAL' in cc_df.columns:
         cc_df['CC_LIMIT_UTILIZATION'] = cc_df['AMT_BALANCE'] / (cc_df['AMT_CREDIT_LIMIT_ACTUAL'].replace(0,epsilon) + epsilon)
    if 'AMT_PAYMENT_CURRENT' in cc_df.columns and 'AMT_INST_MIN_REGULARITY' in cc_df.columns:
         cc_df['CC_PAYMENT_TO_MIN_RATIO'] = cc_df['AMT_PAYMENT_CURRENT'] / (cc_df['AMT_INST_MIN_REGULARITY'].replace(0,epsilon) + epsilon)
    for new_ratio_col in ['CC_LIMIT_UTILIZATION', 'CC_PAYMENT_TO_MIN_RATIO']:
        if new_ratio_col in cc_df.columns:
            cc_df[new_ratio_col].fillna(0, inplace=True)
            cc_df[new_ratio_col].replace([np.inf, -np.inf], 0, inplace=True)

    cc_agg_spec = {'CC_MONTHS_AGO': ['count', 'mean', 'max', 'min', 'std'],'AMT_BALANCE': ['mean', 'max', 'sum', 'std', 'min'],'AMT_CREDIT_LIMIT_ACTUAL': ['mean', 'max', 'min'],'AMT_DRAWINGS_CURRENT': ['sum', 'mean', 'max', 'std'],'AMT_PAYMENT_CURRENT': ['sum', 'mean', 'max', 'std'],'SK_DPD': ['sum', 'mean', 'max', 'std', 'count'],'SK_DPD_DEF': ['sum', 'mean', 'max', 'std', 'count'],'CNT_DRAWINGS_CURRENT':['sum', 'mean', 'max', 'std'],'CNT_INSTALMENT_MATURE_CUM':['max', 'mean', 'min', 'std'],'CC_LIMIT_UTILIZATION': ['mean', 'max', 'std', 'min'],'CC_PAYMENT_TO_MIN_RATIO':['mean','max','std', 'min']}
    for col_key in cols_to_impute_cc:
        if f'{col_key}_NAN_FLAG' in cc_df.columns: cc_agg_spec[f'{col_key}_NAN_FLAG'] = ['sum', 'mean']
    cc_agg_dict_final = {k:v for k,v in cc_agg_spec.items() if k in cc_df.columns}

    if 'NAME_CONTRACT_STATUS' in cc_df.columns and cc_df['NAME_CONTRACT_STATUS'].isnull().any():
        cc_df['NAME_CONTRACT_STATUS'].fillna('Unknown_CC_Status', inplace=True)
    cc_df_ohe, cc_ohe_cat_cols = one_hot_encoder(cc_df, nan_as_category=False, limit_categories=7, prefix_sep='_CC_STATUS_')
    for cat_col in cc_ohe_cat_cols:
        if cat_col in cc_df_ohe.columns and cat_col not in ['SK_ID_CURR', 'SK_ID_PREV']:
            cc_agg_dict_final[cat_col] = ['sum', 'mean']

    if 'SK_ID_CURR' not in cc_df_ohe.columns: return pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage
    cc_final_agg = cc_df_ohe.drop(columns=['SK_ID_PREV'], errors='ignore').groupby('SK_ID_CURR').agg(cc_agg_dict_final)
    cc_final_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_final_agg.columns.tolist()])
    cc_final_agg_res = cc_final_agg.reset_index()

    if verbose: print(f"LogReg FE: Credit Card FE finished. Aggregated shape: {cc_final_agg_res.shape}")
    return cc_final_agg_res if not cc_final_agg_res.empty else pd.DataFrame(columns=['SK_ID_CURR']), median_dict_storage



print("--- Applying All Feature Engineering Steps for Logistic Regression Model ---")

# Make fresh copies for this processing run to avoid modifying original dfs loaded
df_train_logreg = dfs.get('app_train', pd.DataFrame()).copy()
df_test_logreg = dfs.get('app_test', pd.DataFrame()).copy()

# Check if primary dataframes are loaded
if df_train_logreg.empty or df_test_logreg.empty:
    print("CRITICAL ERROR: application_train or application_test DataFrame is empty. Cannot proceed with FE.")
    # Initialize to empty DataFrames to prevent errors in subsequent cells if this block is skipped
    df_train_logreg = pd.DataFrame()
    df_test_logreg = pd.DataFrame()
    # Also initialize X, y, and other critical variables for later cells
    X_logreg_train, y_logreg_train, X_logreg_test, test_ids_logreg = pd.DataFrame(), pd.Series(dtype='float64'), pd.DataFrame(), pd.Series(dtype='int64')
    cols_to_scale_logreg = []
else:
    # Initialize median storage dictionaries for this specific run
    median_storage_app_logreg = {}
    median_storage_bureau_logreg = {}
    median_storage_prev_app_logreg = {}
    median_storage_install_logreg = {}
    median_storage_cc_logreg = {}

    # 1. Process Application Data
    print("\n--- Processing Application Data ---")
    df_train_logreg, median_storage_app_logreg = preprocess_and_fe_application_logreg(df_train_logreg, median_storage_app_logreg, is_train_set=True)
    df_test_logreg, _ = preprocess_and_fe_application_logreg(df_test_logreg, median_storage_app_logreg, is_train_set=False)
    display_df_info(df_train_logreg, "df_train_logreg (after app FE)")
    display_df_info(df_test_logreg, "df_test_logreg (after app FE)")
    gc.collect()

    # 2. Process Bureau and Bureau Balance Data
    print("\n--- Processing Bureau & Bureau Balance Data ---")
    bureau_data_orig = dfs.get('bureau', pd.DataFrame()).copy()
    bb_data_orig = dfs.get('bureau_balance', pd.DataFrame()).copy()
    if not bureau_data_orig.empty:
        bureau_feats, median_storage_bureau_logreg = fe_bureau_and_balance_logreg(bureau_data_orig, bb_data_orig, median_storage_bureau_logreg, is_train_set=True, verbose=VERBOSE_PRINT)
        if not bureau_feats.empty and 'SK_ID_CURR' in bureau_feats.columns:
            df_train_logreg = df_train_logreg.merge(bureau_feats, on='SK_ID_CURR', how='left') # Suffixes handled by unique feature names
        bureau_feats_test, _ = fe_bureau_and_balance_logreg(bureau_data_orig, bb_data_orig, median_storage_bureau_logreg, is_train_set=False, verbose=False)
        if not bureau_feats_test.empty and 'SK_ID_CURR' in bureau_feats_test.columns and not df_test_logreg.empty:
            df_test_logreg = df_test_logreg.merge(bureau_feats_test, on='SK_ID_CURR', how='left')
        display_df_info(df_train_logreg, "df_train_logreg (after Bureau FE)")
    else: print("Bureau data is empty or not loaded. Skipping Bureau FE.")
    del bureau_data_orig, bb_data_orig, bureau_feats, bureau_feats_test; gc.collect()


    # 3. Process Previous Application Data
    print("\n--- Processing Previous Application Data ---")
    prev_app_data_orig = dfs.get('previous_application', pd.DataFrame()).copy()
    if not prev_app_data_orig.empty:
        prev_app_feats, median_storage_prev_app_logreg = fe_previous_applications_logreg(prev_app_data_orig, median_storage_prev_app_logreg, is_train_set=True, verbose=VERBOSE_PRINT)
        if not prev_app_feats.empty and 'SK_ID_CURR' in prev_app_feats.columns:
            df_train_logreg = df_train_logreg.merge(prev_app_feats, on='SK_ID_CURR', how='left')
        prev_app_feats_test, _ = fe_previous_applications_logreg(prev_app_data_orig, median_storage_prev_app_logreg, is_train_set=False, verbose=False)
        if not prev_app_feats_test.empty and 'SK_ID_CURR' in prev_app_feats_test.columns and not df_test_logreg.empty:
            df_test_logreg = df_test_logreg.merge(prev_app_feats_test, on='SK_ID_CURR', how='left')
        display_df_info(df_train_logreg, "df_train_logreg (after Prev App FE)")
    else: print("Previous application data is empty or not loaded. Skipping Previous App FE.")
    del prev_app_data_orig, prev_app_feats, prev_app_feats_test; gc.collect()

    # 4. Process POS CASH Balance Data
    print("\n--- Processing POS CASH Balance Data ---")
    pos_cash_data_orig = dfs.get('pos_cash', pd.DataFrame()).copy()
    if not pos_cash_data_orig.empty:
        # This simplified version of fe_pos_cash_logreg does not maintain state (median_dict_storage)
        pos_cash_feats_agg = fe_pos_cash_logreg(pos_cash_data_orig, verbose=VERBOSE_PRINT) # Run once for all data
        if not pos_cash_feats_agg.empty and 'SK_ID_CURR' in pos_cash_feats_agg.columns:
            df_train_logreg = df_train_logreg.merge(pos_cash_feats_agg, on='SK_ID_CURR', how='left')
            if not df_test_logreg.empty:
                df_test_logreg = df_test_logreg.merge(pos_cash_feats_agg, on='SK_ID_CURR', how='left')
        display_df_info(df_train_logreg, "df_train_logreg (after POS CASH FE)")
    else: print("POS CASH data is empty or not loaded. Skipping POS CASH FE.")
    del pos_cash_data_orig, pos_cash_feats_agg; gc.collect()

    # 5. Process Installments Payments Data
    print("\n--- Processing Installments Payments Data ---")
    install_data_orig = dfs.get('installments_payments', pd.DataFrame()).copy()
    if not install_data_orig.empty:
        install_feats, median_storage_install_logreg = fe_installments_logreg(install_data_orig, median_storage_install_logreg, is_train_set=True, verbose=VERBOSE_PRINT)
        if not install_feats.empty and 'SK_ID_CURR' in install_feats.columns:
            df_train_logreg = df_train_logreg.merge(install_feats, on='SK_ID_CURR', how='left')
        install_feats_test, _ = fe_installments_logreg(install_data_orig, median_storage_install_logreg, is_train_set=False, verbose=False)
        if not install_feats_test.empty and 'SK_ID_CURR' in install_feats_test.columns and not df_test_logreg.empty:
            df_test_logreg = df_test_logreg.merge(install_feats_test, on='SK_ID_CURR', how='left')
        display_df_info(df_train_logreg, "df_train_logreg (after Installments FE)")
    else: print("Installments Payments data is empty or not loaded. Skipping Installments FE.")
    del install_data_orig, install_feats, install_feats_test; gc.collect()

    # 6. Process Credit Card Balance Data
    print("\n--- Processing Credit Card Balance Data ---")
    cc_data_orig = dfs.get('credit_card_balance', pd.DataFrame()).copy()
    if not cc_data_orig.empty:
        cc_feats, median_storage_cc_logreg = fe_credit_card_logreg(cc_data_orig, median_storage_cc_logreg, is_train_set=True, verbose=VERBOSE_PRINT)
        if not cc_feats.empty and 'SK_ID_CURR' in cc_feats.columns:
            df_train_logreg = df_train_logreg.merge(cc_feats, on='SK_ID_CURR', how='left')
        cc_feats_test, _ = fe_credit_card_logreg(cc_data_orig, median_storage_cc_logreg, is_train_set=False, verbose=False)
        if not cc_feats_test.empty and 'SK_ID_CURR' in cc_feats_test.columns and not df_test_logreg.empty:
            df_test_logreg = df_test_logreg.merge(cc_feats_test, on='SK_ID_CURR', how='left')
        display_df_info(df_train_logreg, "df_train_logreg (after Credit Card FE)")
    else: print("Credit Card Balance data is empty or not loaded. Skipping Credit Card FE.")
    del cc_data_orig, cc_feats, cc_feats_test; gc.collect()

    print("\n--- All Feature Engineering and Merging Steps Completed ---")

# Ensure dataframes are defined for subsequent cells even if initial app_train/test were empty or FE failed
if 'df_train_logreg' not in locals(): df_train_logreg = pd.DataFrame()
if 'df_test_logreg' not in locals(): df_test_logreg = pd.DataFrame()


# Initialize final data variables that will be used for modeling
X_logreg_train, y_logreg_train, X_logreg_test, test_ids_logreg = pd.DataFrame(), pd.Series(dtype='float64'), pd.DataFrame(), pd.Series(dtype='int64')
train_ids_logreg = pd.Series(dtype='int64') # SK_ID_CURR for train set

if 'df_train_logreg' in locals() and not df_train_logreg.empty:
    print(f"\n--- Final NaN Handling and Alignment for df_train_logreg ---")
    print(f"Shape before final NaN/inf fill: {df_train_logreg.shape}")
    # Replace infinite values that might have resulted from divisions by very small numbers or edge cases
    df_train_logreg.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Final catch-all for NaNs. Imputing with 0 is a common strategy here.
    # Consider if another strategy (e.g., mean/median of the column if significant NaNs remain) is better.
    df_train_logreg.fillna(0, inplace=True)
    print(f"Total NaNs after final fill: {df_train_logreg.isnull().sum().sum()}")

    # Handle df_test_logreg similarly
    if 'df_test_logreg' in locals() and not df_test_logreg.empty:
        print(f"\n--- Final NaN Handling and Alignment for df_test_logreg ---")
        print(f"Shape before final NaN/inf fill: {df_test_logreg.shape}")
        df_test_logreg.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_test_logreg.fillna(0, inplace=True)
        print(f"Total NaNs after final fill: {df_test_logreg.isnull().sum().sum()}")
    else:
        print("WARNING: df_test_logreg is empty or not defined. Test set operations might fail.")
        df_test_logreg = pd.DataFrame() # Ensure it's a DF to prevent errors later

    gc.collect()

    # Extract target and IDs
    if 'TARGET' in df_train_logreg.columns:
        y_logreg_train = df_train_logreg['TARGET'].copy()
    else:
        print("CRITICAL WARNING: 'TARGET' column not found in df_train_logreg. Model training will fail.")
        y_logreg_train = pd.Series(dtype='float64') # Empty series

    if 'SK_ID_CURR' in df_train_logreg.columns:
        train_ids_logreg = df_train_logreg['SK_ID_CURR'].copy()
    if 'SK_ID_CURR' in df_test_logreg.columns:
        test_ids_logreg = df_test_logreg['SK_ID_CURR'].copy()
    else:
        # If SK_ID_CURR is missing from test, submission will be impossible.
        # This might happen if df_test_logreg became empty.
        print("WARNING: 'SK_ID_CURR' column not found in df_test_logreg. Submission IDs will be missing.")


    # Prepare X by dropping Target and SK_ID_CURR
    X_logreg_train_unaligned = df_train_logreg.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
    X_logreg_test_unaligned = df_test_logreg.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore') # TARGET might not exist in test

    # Align columns - crucial step
    if not X_logreg_train_unaligned.empty:
        if not X_logreg_test_unaligned.empty:
            print(f"\nAligning columns. Train unaligned shape: {X_logreg_train_unaligned.shape}, Test unaligned shape: {X_logreg_test_unaligned.shape}")
            # Get common columns
            common_cols_logreg = list(set(X_logreg_train_unaligned.columns) & set(X_logreg_test_unaligned.columns))
            
            if common_cols_logreg:
                X_logreg_train = X_logreg_train_unaligned[common_cols_logreg].copy()
                X_logreg_test = X_logreg_test_unaligned[common_cols_logreg].copy()
                # Ensure order of columns is the same in train and test
                X_logreg_test = X_logreg_test[X_logreg_train.columns] # Match train's column order
                print(f"SUCCESS: Columns aligned. X_logreg_train shape: {X_logreg_train.shape}, X_logreg_test shape: {X_logreg_test.shape}")
                print(f"Number of common features: {len(common_cols_logreg)}")
            else:
                print("ERROR: No common columns found between train and test after FE. Modeling cannot proceed correctly.")
                X_logreg_train, X_logreg_test = pd.DataFrame(), pd.DataFrame() # Empty them
        else: # Test set is empty, but train is not
            print("WARNING: Test set (X_logreg_test_unaligned) is empty. Creating X_logreg_test with columns from X_logreg_train for compatibility, but test predictions will be based on an empty frame.")
            X_logreg_train = X_logreg_train_unaligned.copy()
            X_logreg_test = pd.DataFrame(columns=X_logreg_train.columns) # Empty DF with train's columns
            print(f"X_logreg_train shape: {X_logreg_train.shape}, X_logreg_test shape: {X_logreg_test.shape}")
    else:
        print("CRITICAL ERROR: Training features (X_logreg_train_unaligned) are empty after dropping columns. Cannot proceed.")
else:
    print("CRITICAL ERROR: df_train_logreg is empty. Cannot proceed with final data preparation.")


cols_to_scale_logreg = []
if 'X_logreg_train' in locals() and not X_logreg_train.empty:
    print("\n--- Identifying Numeric Columns for Scaling ---")
    for col in X_logreg_train.columns:
        # Check if column is numeric
        if X_logreg_train[col].dtype in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
            # Scale if more than 2 unique values (not a simple binary/flag)
            # AND it's not a NaN indicator flag (which are already 0/1)
            if X_logreg_train[col].nunique(dropna=False) > 2 and not col.endswith('_NAN_FLAG'):
                 cols_to_scale_logreg.append(col)
    print(f"Identified {len(cols_to_scale_logreg)} columns to be scaled for Logistic Regression.")
    if VERBOSE_PRINT and cols_to_scale_logreg: print(f"Sample columns to scale (first 10): {cols_to_scale_logreg[:10]}...")
else:
    print("X_logreg_train is not available or empty. Cannot identify columns to scale.")


if 'X_logreg_train' in locals() and not X_logreg_train.empty:
    print("\n--- Cleaning Feature Names ---")
    X_logreg_train = clean_col_names(X_logreg_train.copy()) # Use .copy() to avoid SettingWithCopyWarning
    print("Cleaned column names for X_logreg_train.")
if 'X_logreg_test' in locals() and not X_logreg_test.empty:
    X_logreg_test = clean_col_names(X_logreg_test.copy())
    print("Cleaned column names for X_logreg_test.")

# Display final prepared data overview
print("\n--- Overview of Final Prepared Data for Modeling ---")
if 'X_logreg_train' in locals(): display_df_info(X_logreg_train, "X_logreg_train (Final, before scaling in CV)")
else: print("X_logreg_train is not available for display.")

if 'X_logreg_test' in locals(): display_df_info(X_logreg_test, "X_logreg_test (Final, before scaling in CV)")
else: print("X_logreg_test is not available for display.")

if 'y_logreg_train' in locals() and not y_logreg_train.empty:
    print("\nTarget variable (y_logreg_train) info:")
    print(f"Shape: {y_logreg_train.shape}, Type: {type(y_logreg_train)}, Dtype: {y_logreg_train.dtype}")
    print(f"Value counts (normalized %):\n{y_logreg_train.value_counts(normalize=True) * 100}")
else: print("y_logreg_train is not available for display.")

