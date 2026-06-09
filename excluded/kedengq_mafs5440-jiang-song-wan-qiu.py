!pip install scorecardpy
!pip install autogluon.tabular
!pip install scikit-learn==1.2
!pip install "ray>=2.10.0,<2.45.0"
!pip install optuna-integration[lightgbm]


import numpy as np
import pandas as pd
import gc
import os
import time
from contextlib import contextmanager
import gc
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py  # è‹¥ä¸�éœ€è¦�äº¤äº’å›¾ï¼Œå�¯æ³¨é‡Šæ�‰
import scorecardpy as sc     # é£�æ�§åˆ†æ��å¸¸ç”¨åˆ†ç®±ã€�WOEã€�IV
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold,train_test_split,KFold,cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import optuna
from optuna import Trial
import lightgbm as lgb
# ================== Pandasæ˜¾ç¤ºè®¾ç½® ==================
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.6f}'.format)
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.simplefilter(action='ignore', category=FutureWarning)


# Timer function
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# One-hot encoder with missing value handling
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']

    for col in categorical_columns:
        df[col] = df[col].fillna('MISSING')

    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Function to clean numerical columns
def clean_numerical_data(df, numerical_columns):
    df_clean = df.copy()

    for col in numerical_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)

    return df_clean


def _safe_div(numer, denom):
    """Safe element-wise division returning a Pandas Series."""
    numer = pd.Series(numer)
    denom = pd.Series(denom)
    result = np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numer / denom
        result[(denom == 0) | denom.isna()] = np.nan
    return result


# Label encoding helper
def label_encode_categoricals(df):
    df_encoded = df.copy()
    cat_cols = [col for col in df_encoded.columns if df_encoded[col].dtype == 'object']
    label_info = {}  # ä¿�å­˜æ¯�åˆ—çš„å”¯ä¸€å€¼ä¸ªæ•°

    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = df_encoded[col].astype(str).fillna('MISSING')
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_info[col] = len(le.classes_)  # è®°å½•è¯¥åˆ—ç±»åˆ«æ•°

    return df_encoded, label_info

def safe_mode(series):
    """Return mode if available, else NaN."""
    try:
        mode_vals = series.mode()
        return mode_vals.iloc[0] if not mode_vals.empty else np.nan
    except Exception:
        return np.nan


# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows=None, nan_as_category=False):
    # Read train, test, and external feature data
    df = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv', nrows=num_rows)
    testdata = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv', nrows=num_rows)
    previousdata = pd.read_csv('/kaggle/input/feature/krakowlublinzhabinka_feats.csv')
    print(f"Train samples: {len(df)}, test samples: {len(testdata)}")

    # Concatenate train and test vertically, then add extra feature set horizontally
    df = pd.concat([df, testdata], ignore_index=True)
    df = pd.concat([df, previousdata], axis=1)

    # Remove the 4 rows with invalid gender code "XNA"
    df = df[df['CODE_GENDER'] != 'XNA']

    # Binary categorical features (two unique values) - factorized to 0/1
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    # Replace One-Hot encoding with Label Encoding for all remaining categorical features
    df, label_info = label_encode_categoricals(df)

    # Replace 365243 with NaN in DAYS_EMPLOYED (invalid placeholder)
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # === Feature engineering ===
    # Employment days to age ratio â€” proportion of life spent working
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    # Total income to credit ratio â€” lower ratio = higher debt burden
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']

    # Income per family member â€” indicates economic pressure
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

    # Annuity to income ratio â€” repayment burden from income
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

    # Annuity to credit ratio â€” repayment speed
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    # Credit amount minus total income â€” over-borrowing indicator
    df['CREDIT_INCOME_DIFF'] = df['AMT_CREDIT'] - df['AMT_INCOME_TOTAL']

    # Approximate loan term â€” how long to repay the credit
    df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']

    # Income per child (+1 avoids division by zero)
    df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (df['CNT_CHILDREN'] + 1)

    # Credit to goods price ratio â€” checks if loan exceeds product cost
    df['CREDIT_GOODS_PERC'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']

    # Employment length to age ratio â€” career stability
    df['EMPLOY_TO_AGE_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    # Children to family members ratio â€” dependency load
    df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']

    # Car age relative to personâ€™s age (years)
    df['CAR_AGE_RATIO'] = df['OWN_CAR_AGE'] / (-df['DAYS_BIRTH'] / 365)

    # Binary flag â€” owns both car and house
    df['REALTY_AND_CAR_FLAG'] = df['FLAG_OWN_CAR'] * df['FLAG_OWN_REALTY']

    # Log-transformed income â€” reduces skewness
    df['LOG_INCOME'] = np.log1p(df['AMT_INCOME_TOTAL'])

    # Log-transformed credit amount â€” reduces outlier impact
    df['LOG_CREDIT'] = np.log1p(df['AMT_CREDIT'])

    # Total lifetime earnings approximation (income Ã— employment days)
    df['INCOME_X_EMPLOY'] = df['AMT_INCOME_TOTAL'] * (-df['DAYS_EMPLOYED'])

    # Credit amount multiplied by family size â€” household debt load
    df['CREDIT_X_FAMILY'] = df['AMT_CREDIT'] * df['CNT_FAM_MEMBERS']

    # Credit per child â€” child-related debt pressure
    df['CREDIT_PER_CHILD'] = df['AMT_CREDIT'] / (df['CNT_CHILDREN'] + 1)

    # Mark train vs test samples
    df['is_train'] = 0
    df.loc[:len(pd.read_csv('../input/home-credit-default-risk/application_train.csv')) - 1, 'is_train'] = 1

    # Clean up temporary data
    del testdata
    del previousdata
    gc.collect()

    print(f"Label-encoded categorical features: {len(label_info)} columns processed.")
    return df


# Preprocess bureau.csv and bureau_balance.csv
def safe_mode(series):
    """Return mode if available, else NaN."""
    try:
        mode_vals = series.mode()
        return mode_vals.iloc[0] if not mode_vals.empty else np.nan
    except Exception:
        return np.nan


def bureau_and_balance(num_rows=None, nan_as_category=True):
    """
    Safe and stable version of bureau_and_balance():
    - Keeps original string values of CREDIT_ACTIVE (for Active/Closed split)
    - Performs label encoding on other categorical columns
    - Aggregates numeric and categorical features
    - Adds derived ratio and recency features
    """
    import pandas as pd
    import numpy as np
    import gc

    # === Load raw data ===
    bureaudata = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv', nrows=num_rows)
    balancedata = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv', nrows=num_rows)
    print(f"ğŸ“„ Bureau raw shape: {bureaudata.shape}, Bureau_balance raw shape: {balancedata.shape}")

    # === Backup original CREDIT_ACTIVE column before encoding ===
    bureau_cat_backup = {}
    if 'CREDIT_ACTIVE' in bureaudata.columns:
        bureau_cat_backup['CREDIT_ACTIVE'] = bureaudata['CREDIT_ACTIVE'].copy()

    # === Encode and aggregate bureau_balance ===
    balancedata, bb_label_info = label_encode_categoricals(balancedata)

    bb_num_agg = {'MONTHS_BALANCE': ['min', 'max', 'mean', 'size']}
    bb_cat_agg = {}
    for col, n_unique in bb_label_info.items():
        if n_unique == 2:
            bb_cat_agg[col] = ['mean']
        elif n_unique > 2:
            bb_cat_agg[col] = [safe_mode]

    balancedata_aggregated = balancedata.groupby('SK_ID_BUREAU').agg({**bb_num_agg, **bb_cat_agg})
    balancedata_aggregated.columns = pd.Index([
        f"BB_{col[0]}_BINARY_MEAN" if col[1] == 'mean' and bb_label_info.get(col[0], 0) == 2
        else (f"BB_{col[0]}_MODE" if col[1] == 'safe_mode' or 'lambda' in col[1] else f"BB_{col[0]}_{col[1].upper()}")
        for col in balancedata_aggregated.columns
    ])
    print(f"âœ… Bureau balance aggregated shape: {balancedata_aggregated.shape}")

    # === Merge aggregated bureau_balance with bureau ===
    bureaudata = bureaudata.join(balancedata_aggregated, how='left', on='SK_ID_BUREAU')
    del balancedata, balancedata_aggregated
    gc.collect()

    # === Label encode bureau (excluding CREDIT_ACTIVE) ===
    bureau_encoded, bureau_label_info = label_encode_categoricals(
        bureaudata.drop(columns=['CREDIT_ACTIVE'], errors='ignore')
    )

    # Restore original CREDIT_ACTIVE strings for Active/Closed filtering
    if 'CREDIT_ACTIVE' in bureaudata.columns:
        bureau_encoded['CREDIT_ACTIVE'] = bureau_cat_backup['CREDIT_ACTIVE']

    bureaudata = bureau_encoded

    # === Define numeric aggregations ===
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var', 'median'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean', 'var', 'median'],
        'DAYS_CREDIT_UPDATE': ['mean', 'min', 'max', 'var'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean', 'sum', 'var', 'median'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max', 'sum', 'var'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum', 'var', 'median'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum', 'var', 'median'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean', 'max', 'sum', 'var'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum', 'max', 'var', 'median'],
        'AMT_ANNUITY': ['max', 'mean', 'sum', 'var', 'median'],
        'CNT_CREDIT_PROLONG': ['sum', 'mean', 'max'],
    }

    # === Define categorical aggregations ===
    cat_aggregations = {}
    for col, n_unique in bureau_label_info.items():
        if n_unique == 2:
            cat_aggregations[col] = ['mean']
        elif n_unique > 2:
            cat_aggregations[col] = [safe_mode]

    # === Main aggregation at SK_ID_CURR level ===
    bureaudata_aggregated = bureaudata.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureaudata_aggregated.columns = pd.Index([
        f"BURO_{col[0]}_BINARY_MEAN" if col[1] == 'mean' and bureau_label_info.get(col[0], 0) == 2
        else (f"BURO_{col[0]}_MODE" if col[1] == 'safe_mode' or 'lambda' in col[1] else f"BURO_{col[0]}_{col[1].upper()}")
        for col in bureaudata_aggregated.columns
    ])
    print(f"âœ… Bureau aggregated shape: {bureaudata_aggregated.shape}")

    # === Separate Active and Closed credit records ===
    if 'CREDIT_ACTIVE' in bureaudata.columns:
        active = bureaudata[bureaudata['CREDIT_ACTIVE'] == 'Active']
        closed = bureaudata[bureaudata['CREDIT_ACTIVE'] == 'Closed']

        # Fallback: use mode if 'Active' or 'Closed' not found
        if active.empty:
            active = bureaudata[bureaudata['CREDIT_ACTIVE'] == bureaudata['CREDIT_ACTIVE'].mode().iloc[0]]
        if closed.empty:
            closed = bureaudata[bureaudata['CREDIT_ACTIVE'] != bureaudata['CREDIT_ACTIVE'].mode().iloc[0]]

        # Aggregate active credits
        active_aggeragated = active.groupby('SK_ID_CURR').agg(num_aggregations)
        active_aggeragated.columns = pd.Index([f"ACTIVE_{e[0]}_{e[1].upper()}" for e in active_aggeragated.columns])
        bureaudata_aggregated = bureaudata_aggregated.join(active_aggeragated, how='left', on='SK_ID_CURR')
        del active, active_aggeragated
        gc.collect()

        # Aggregate closed credits
        closed_aggeragated = closed.groupby('SK_ID_CURR').agg(num_aggregations)
        closed_aggeragated.columns = pd.Index([f"CLOSED_{e[0]}_{e[1].upper()}" for e in closed_aggeragated.columns])
        bureaudata_aggregated = bureaudata_aggregated.join(closed_aggeragated, how='left', on='SK_ID_CURR')
        del closed, closed_aggeragated
        gc.collect()

    # === Derived ratio & recency features ===
    bureaudata_aggregated['BURO_DEBT_CREDIT_RATIO'] = _safe_div(
        bureaudata_aggregated.get('BURO_AMT_CREDIT_SUM_DEBT_SUM'),
        bureaudata_aggregated.get('BURO_AMT_CREDIT_SUM_SUM')
    )

    bureaudata_aggregated['BURO_LIMIT_USAGE_RATIO'] = _safe_div(
        bureaudata_aggregated.get('BURO_AMT_CREDIT_SUM_DEBT_SUM'),
        bureaudata_aggregated.get('BURO_AMT_CREDIT_SUM_LIMIT_SUM')
    )

    bureaudata_aggregated['BURO_OVERDUE_DEBT_RATIO'] = _safe_div(
        bureaudata_aggregated.get('BURO_AMT_CREDIT_SUM_OVERDUE_SUM'),
        bureaudata_aggregated.get('BURO_AMT_CREDIT_SUM_DEBT_SUM')
    )

    bureaudata_aggregated['BURO_MAX_OVERDUE_TO_CREDIT'] = _safe_div(
        bureaudata_aggregated.get('BURO_AMT_CREDIT_MAX_OVERDUE_MAX'),
        bureaudata_aggregated.get('BURO_AMT_CREDIT_SUM_SUM')
    )

    bureaudata_aggregated['BURO_RECENT_CREDIT_DAYS'] = -bureaudata_aggregated.get('BURO_DAYS_CREDIT_MAX')

    if 'BURO_CREDIT_ACTIVE_BINARY_MEAN' in bureaudata_aggregated.columns:
        bureaudata_aggregated['BURO_ACTIVE_SHARE'] = bureaudata_aggregated['BURO_CREDIT_ACTIVE_BINARY_MEAN']

    bureaudata_aggregated['ACTIVE_DEBT_CREDIT_RATIO'] = _safe_div(
        bureaudata_aggregated.get('ACTIVE_AMT_CREDIT_SUM_DEBT_SUM'),
        bureaudata_aggregated.get('ACTIVE_AMT_CREDIT_SUM_SUM')
    )

    bureaudata_aggregated['ACTIVE_LIMIT_USAGE_RATIO'] = _safe_div(
        bureaudata_aggregated.get('ACTIVE_AMT_CREDIT_SUM_DEBT_SUM'),
        bureaudata_aggregated.get('ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM')
    )

    bureaudata_aggregated['ACTIVE_OVERDUE_DEBT_RATIO'] = _safe_div(
        bureaudata_aggregated.get('ACTIVE_AMT_CREDIT_SUM_OVERDUE_SUM'),
        bureaudata_aggregated.get('ACTIVE_AMT_CREDIT_SUM_DEBT_SUM')
    )

    # === Cleanup ===
    del bureaudata
    gc.collect()

    print(f"ğŸ�¯ Final Bureau data shape: {bureaudata_aggregated.shape}")
    return bureaudata_aggregated


def credit_card_balance(num_rows=None, nan_as_category=True):
    """
    Process and aggregate credit_card_balance.csv

    Steps:
      - Label encode categorical columns
      - Aggregate only the original numeric columns (min, max, mean, sum, var)
      - Aggregate categorical columns (binary â†’ mean, multi-category â†’ mode)
      - Create derived ratio features such as utilization and payment ratios
      - Add count-based and recent-activity indicators
    """
    creditcarddata = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv', nrows=num_rows)
    print(f"ğŸ“„ Credit card balance raw shape: {creditcarddata.shape}")

    # === Record original numeric columns BEFORE encoding ===
    original_numeric = creditcarddata.select_dtypes(include=[np.number]).columns.tolist()

    # Label encode categorical columns
    creditcarddata, cc_label_info = label_encode_categoricals(creditcarddata)

    # Drop SK_ID_PREV because it is just a grouping key
    if 'SK_ID_PREV' in creditcarddata.columns:
        creditcarddata.drop(['SK_ID_PREV'], axis=1, inplace=True)

    # === Numeric aggregations (only on original numeric columns) ===
    valid_num_cols = [col for col in original_numeric if col in creditcarddata.columns and col not in ['SK_ID_CURR']]
    num_agg = creditcarddata[valid_num_cols + ['SK_ID_CURR']].groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    num_agg.columns = ['{}_{}'.format(col[0], col[1].upper()) for col in num_agg.columns]

    # === Categorical aggregations: binary â†’ mean, multi â†’ mode ===
    cat_agg_dict = {}
    for col, n_unique in cc_label_info.items():
        if col in valid_num_cols or col in ['SK_ID_CURR']:
            continue
        if n_unique == 2:
            cat_agg_dict[col] = ['mean']
        elif n_unique > 2:
            # Mode aggregation for multi-category columns
            cat_agg_dict[col] = [lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan]

    # Combine numeric and categorical aggregations
    if cat_agg_dict:
        cat_agg = creditcarddata.groupby('SK_ID_CURR').agg(cat_agg_dict)
        cat_agg.columns = pd.Index([
            f"{col[0]}_BINARY_MEAN" if col[1] == 'mean' and cc_label_info.get(col[0], 0) == 2
            else (f"{col[0]}_MODE" if 'lambda' in col[1] else f"{col[0]}_{col[1].upper()}")
            for col in cat_agg.columns
        ])
        creditcarddata_aggregated = num_agg.join(cat_agg, how='left')
    else:
        creditcarddata_aggregated = num_agg

    # Add CC_ prefix to all feature names
    creditcarddata_aggregated.columns = pd.Index([
        f"CC_{col}" if not col.startswith('CC_') else col
        for col in creditcarddata_aggregated.columns
    ])

    # === Basic count features ===
    creditcarddata_aggregated['CC_COUNT'] = creditcarddata.groupby('SK_ID_CURR').size()

    # Helper function to safely get a column (return NaN Series if missing)
    def _col(name):
        return creditcarddata_aggregated[name] if name in creditcarddata_aggregated.columns else pd.Series(np.nan, index=creditcarddata_aggregated.index)

    # === Derived ratio features ===
    # Average utilization ratio (balance / credit limit)
    creditcarddata_aggregated['CC_UTILIZATION_MEAN'] = _safe_div(
        _col('CC_AMT_BALANCE_MEAN'),
        _col('CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN')
    )
    # Total utilization ratio
    creditcarddata_aggregated['CC_UTILIZATION_SUM'] = _safe_div(
        _col('CC_AMT_BALANCE_SUM'),
        _col('CC_AMT_CREDIT_LIMIT_ACTUAL_SUM')
    )

    # Payment ratio = total paid / total receivable
    total_receivable = _col('CC_AMT_TOTAL_RECEIVABLE_SUM').fillna(_col('CC_AMT_RECIVABLE_SUM'))
    creditcarddata_aggregated['CC_PAYMENT_RATIO'] = _safe_div(
        _col('CC_AMT_PAYMENT_TOTAL_CURRENT_SUM').fillna(_col('CC_AMT_PAYMENT_CURRENT_SUM')),
        total_receivable
    )

    # Minimum regular payment ratio
    creditcarddata_aggregated['CC_MIN_PAY_REG_RATIO'] = _safe_div(
        _col('CC_AMT_INST_MIN_REGULARITY_MEAN'),
        _col('CC_AMT_TOTAL_RECEIVABLE_MEAN').fillna(_col('CC_AMT_RECIVABLE_MEAN'))
    )

    # Drawings (cash or POS) relative to credit limit
    creditcarddata_aggregated['CC_DRAWINGS_TO_LIMIT'] = _safe_div(
        _col('CC_AMT_DRAWINGS_CURRENT_SUM'),
        _col('CC_AMT_CREDIT_LIMIT_ACTUAL_SUM')
    )

    # ATM drawings share
    creditcarddata_aggregated['CC_ATM_DRAW_SHARE'] = _safe_div(
        _col('CC_AMT_DRAWINGS_ATM_CURRENT_SUM'),
        _col('CC_AMT_DRAWINGS_CURRENT_SUM')
    )

    # POS drawings share
    creditcarddata_aggregated['CC_POS_DRAW_SHARE'] = _safe_div(
        _col('CC_AMT_DRAWINGS_POS_CURRENT_SUM'),
        _col('CC_AMT_DRAWINGS_CURRENT_SUM')
    )

    # Flag: has any delinquency (DPD or DPD_DEF > 0)
    any_dpd = _col('CC_SK_DPD_MAX').fillna(0).gt(0) | _col('CC_SK_DPD_DEF_MAX').fillna(0).gt(0)
    creditcarddata_aggregated['CC_ANY_DPD_FLAG'] = any_dpd.astype(np.int8)

    # Flag: recent activity within the last 3 months
    creditcarddata_aggregated['CC_RECENT_ACTIVITY_FLAG'] = _col('CC_MONTHS_BALANCE_MAX').gt(-3).astype(float)

    del creditcarddata
    gc.collect()

    print(f"âœ… Credit card balance aggregation completed: {creditcarddata_aggregated.shape}")
    return creditcarddata_aggregated


def previous_applications(num_rows=None, nan_as_category=True):
    """
    Safe and enhanced version of previous_applications():
    - Splits 'Approved' and 'Refused' applications BEFORE label encoding
    - Cleans numeric columns and replaces placeholder values
    - Performs label encoding for categorical columns
    - Creates engineered features (ratios, timing, interest rate estimate)
    - Aggregates numeric and categorical features
    - Adds approval/refusal ratios and counts
    """
    import pandas as pd
    import numpy as np
    import gc

    # === Load data ===
    previousdata = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv', nrows=num_rows)
    print(f"ğŸ“„ Previous applications raw data shape: {previousdata.shape}")

    # === Split 'Approved' and 'Refused' subsets BEFORE label encoding ===
    approved, refused = None, None
    if 'NAME_CONTRACT_STATUS' in previousdata.columns:
        approved = previousdata[previousdata['NAME_CONTRACT_STATUS'] == 'Approved'].copy()
        refused = previousdata[previousdata['NAME_CONTRACT_STATUS'] == 'Refused'].copy()

    # === Clean numeric and date columns ===
    numerical_columns = previousdata.select_dtypes(include=[np.number]).columns.tolist()
    date_columns = [
        'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION',
        'DAYS_LAST_DUE', 'DAYS_TERMINATION'
    ]

    # Replace invalid placeholder 365243 with NaN
    for col in date_columns:
        if col in previousdata.columns:
            previousdata[col] = previousdata[col].replace(365243, np.nan)

    # Clean numeric columns (replace inf/-inf, fill missing with median)
    previousdata = clean_numerical_data(previousdata, numerical_columns)

    # === Label encode categorical columns ===
    previousdata, prev_label_info = label_encode_categoricals(previousdata)

    # === Feature engineering ===
    previousdata['APP_CREDIT_PERC'] = _safe_div(previousdata['AMT_APPLICATION'], previousdata['AMT_CREDIT'])
    previousdata['APP_GOODS_PERC'] = _safe_div(previousdata['AMT_APPLICATION'], previousdata['AMT_GOODS_PRICE'])
    previousdata['CREDIT_GOODS_RATIO'] = _safe_div(previousdata['AMT_CREDIT'], previousdata['AMT_GOODS_PRICE'])
    previousdata['ANNUITY_CREDIT_RATIO'] = _safe_div(previousdata['AMT_ANNUITY'], previousdata['AMT_CREDIT'])
    previousdata['DOWN_PAYMENT_RATIO'] = _safe_div(previousdata['AMT_DOWN_PAYMENT'], previousdata['AMT_CREDIT'])
    previousdata['DOWN_PAYMENT_GOODS_RATIO'] = _safe_div(previousdata['AMT_DOWN_PAYMENT'], previousdata['AMT_GOODS_PRICE'])
    previousdata['APP_DECISION_TIMING'] = previousdata['DAYS_DECISION'] - previousdata['DAYS_FIRST_DRAWING']

    # Estimated interest rate based on annuity, credit amount, and payment count
    previousdata['INTEREST_RATE_EST'] = _safe_div(
        (previousdata['AMT_ANNUITY'] * previousdata['CNT_PAYMENT'] / previousdata['AMT_CREDIT'].replace(0, np.nan) - 1),
        previousdata['CNT_PAYMENT']
    )
    # === Debug check: ensure feature engineering columns exist ===
    print("Feature engineered columns:", [c for c in previousdata.columns if 'APP_' in c or 'RATIO' in c or 'INTEREST' in c])

    # Clean the engineered features
    feature_columns = [
        'APP_CREDIT_PERC', 'APP_GOODS_PERC', 'CREDIT_GOODS_RATIO',
        'ANNUITY_CREDIT_RATIO', 'DOWN_PAYMENT_RATIO', 'DOWN_PAYMENT_GOODS_RATIO',
        'APP_DECISION_TIMING', 'INTEREST_RATE_EST'
    ]
    previousdata = clean_numerical_data(previousdata, feature_columns)

    # === Numeric feature aggregations ===
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean', 'sum'],
        'AMT_APPLICATION': ['min', 'max', 'mean', 'sum'],
        'AMT_CREDIT': ['min', 'max', 'mean', 'sum'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean', 'sum'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum', 'max'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'APP_GOODS_PERC': ['min', 'max', 'mean', 'var'],
        'CREDIT_GOODS_RATIO': ['min', 'max', 'mean'],
        'ANNUITY_CREDIT_RATIO': ['min', 'max', 'mean'],
        'DOWN_PAYMENT_RATIO': ['min', 'max', 'mean'],
        'INTEREST_RATE_EST': ['min', 'max', 'mean'],
        'APP_DECISION_TIMING': ['min', 'max', 'mean']
    }

    # === Categorical feature aggregations ===
    cat_aggregations = {}
    for col, n_unique in prev_label_info.items():
        if n_unique == 2:
            cat_aggregations[col] = ['mean']  # Binary features â†’ mean
        elif n_unique > 2:
            cat_aggregations[col] = [lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan]  # Multi-cat â†’ mode

    # === Aggregate by SK_ID_CURR ===
    previousdata_aggregated = previousdata.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    previousdata_aggregated.columns = pd.Index([
        f"PREV_{col[0]}_BINARY_MEAN" if col[1] == 'mean' and prev_label_info.get(col[0], 0) == 2
        else (f"PREV_{col[0]}_MODE" if 'lambda' in col[1] else f"PREV_{col[0]}_{col[1].upper()}")
        for col in previousdata_aggregated.columns
    ])

    # === Approved / Refused aggregations (use the pre-encoded subsets) ===
    if approved is not None and not approved.empty:
        approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
        approved_agg.columns = pd.Index([f"APPROVED_{e[0]}_{e[1].upper()}" for e in approved_agg.columns])
        previousdata_aggregated = previousdata_aggregated.join(approved_agg, how='left', on='SK_ID_CURR')

    if refused is not None and not refused.empty:
        refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
        refused_agg.columns = pd.Index([f"REFUSED_{e[0]}_{e[1].upper()}" for e in refused_agg.columns])
        previousdata_aggregated = previousdata_aggregated.join(refused_agg, how='left', on='SK_ID_CURR')

    # === Count features ===
    previousdata_aggregated['PREV_APP_COUNT'] = previousdata.groupby('SK_ID_CURR').size()
    if approved is not None and not approved.empty:
        previousdata_aggregated['PREV_APPROVED_COUNT'] = approved.groupby('SK_ID_CURR').size()
    if refused is not None and not refused.empty:
        previousdata_aggregated['PREV_REFUSED_COUNT'] = refused.groupby('SK_ID_CURR').size()

    # === Ratio features ===
    if 'PREV_APPROVED_COUNT' in previousdata_aggregated.columns and 'PREV_APP_COUNT' in previousdata_aggregated.columns:
        previousdata_aggregated['PREV_APPROVAL_RATE'] = _safe_div(
            previousdata_aggregated['PREV_APPROVED_COUNT'], previousdata_aggregated['PREV_APP_COUNT']
        ).fillna(0)

    # === Cleanup ===
    del previousdata, approved, refused
    gc.collect()

    print(f"âœ… Previous applications aggregated shape: {previousdata_aggregated.shape}")
    return previousdata_aggregated


def pos_cash(num_rows=None, nan_as_category=True):
    posdata = pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv', nrows=num_rows)
    print(f"POS_CASH balance raw data shape: {posdata.shape}")

    numerical_columns = posdata.select_dtypes(include=[np.number]).columns.tolist()
    posdata = clean_numerical_data(posdata, numerical_columns)

    # replaced one-hot with label encoding
    posdata, pos_label_info = label_encode_categoricals(posdata)

    # Feature Engineering
    posdata['INSTALMENT_PROGRESS'] = posdata['CNT_INSTALMENT'] - posdata['CNT_INSTALMENT_FUTURE']
    posdata['INSTALMENT_COMPLETION_RATIO'] = _safe_div(posdata['INSTALMENT_PROGRESS'], posdata['CNT_INSTALMENT'])

    posdata['POS_IS_DPD'] = posdata['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    posdata['POS_IS_DPD_UNDER_30'] = posdata['SK_DPD'].apply(lambda x: 1 if (x > 0) & (x <= 30) else 0)
    posdata['POS_IS_DPD_OVER_30'] = posdata['SK_DPD'].apply(lambda x: 1 if x > 30 else 0)
    posdata['POS_IS_DPD_OVER_120'] = posdata['SK_DPD'].apply(lambda x: 1 if x >= 120 else 0)
    posdata['POS_IS_SEVERE_DPD'] = posdata['SK_DPD_DEF'].apply(lambda x: 1 if x > 0 else 0)

    posdata = clean_numerical_data(posdata, ['INSTALMENT_COMPLETION_RATIO'])

    # Numeric aggregations
    aggregations = {
        'MONTHS_BALANCE': ['min', 'max', 'mean', 'size'],
        'SK_DPD': ['min', 'max', 'mean', 'sum'],
        'SK_DPD_DEF': ['max', 'mean', 'sum'],
        'CNT_INSTALMENT': ['min', 'max', 'mean'],
        'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean'],
        'POS_IS_DPD': ['mean', 'sum'],
        'POS_IS_DPD_UNDER_30': ['mean', 'sum'],
        'POS_IS_DPD_OVER_30': ['mean', 'sum'],
        'POS_IS_DPD_OVER_120': ['mean', 'sum'],
        'POS_IS_SEVERE_DPD': ['mean', 'sum'],
        'INSTALMENT_PROGRESS': ['min', 'max', 'mean', 'sum'],
        'INSTALMENT_COMPLETION_RATIO': ['min', 'max', 'mean']
    }

    # categorical aggregation logic (binary â†’ mean, multi â†’ mode)
    for col, n_unique in pos_label_info.items():
        if n_unique == 2:
            aggregations[col] = ['mean']
        elif n_unique > 2:
            aggregations[col] = [lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan]

    posdata_aggrgated = posdata.groupby('SK_ID_CURR').agg(aggregations)

    # rename columns reflecting binary/multi
    posdata_aggrgated.columns = pd.Index([
        f"POS_{col[0]}_BINARY_MEAN" if col[1] == 'mean' and pos_label_info.get(col[0], 0) == 2
        else (f"POS_{col[0]}_MODE" if 'lambda' in col[1] else f"POS_{col[0]}_{col[1].upper()}")
        for col in posdata_aggrgated.columns
    ])

    # Counts
    posdata_aggrgated['POS_COUNT'] = posdata.groupby('SK_ID_CURR').size()
    posdata_aggrgated['POS_ACTIVE_COUNT'] = posdata.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique()

    # Recent behavior
    recent_pos = posdata[posdata['MONTHS_BALANCE'] >= -12]
    if len(recent_pos) > 0:
        recent_agg = recent_pos.groupby('SK_ID_CURR').agg({
            'SK_DPD': ['max', 'mean'],
            'POS_IS_DPD': ['mean', 'sum'],
            'POS_IS_DPD_OVER_30': ['mean', 'sum']
        })
        recent_agg.columns = pd.Index(['POS_RECENT_' + e[0] + "_" + e[1].upper() for e in recent_agg.columns.tolist()])
        posdata_aggrgated = posdata_aggrgated.join(recent_agg, how='left', on='SK_ID_CURR')

    del posdata
    gc.collect()

    print(f"POS_CASH balance aggregated shape: {posdata_aggrgated.shape}")
    return posdata_aggrgated


def installments_payments(num_rows=None, nan_as_category=True):
    installmentdata = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv', nrows=num_rows)
    print(f"Installments payments raw data shape: {installmentdata.shape}")

    numerical_columns = installmentdata.select_dtypes(include=[np.number]).columns.tolist()
    installmentdata = clean_numerical_data(installmentdata, numerical_columns)

    # replaced one-hot with label encoding
    installmentdata, ins_label_info = label_encode_categoricals(installmentdata)

    # Feature Engineering
    installmentdata['PAYMENT_PERC'] = _safe_div(installmentdata['AMT_PAYMENT'], installmentdata['AMT_INSTALMENT'])
    installmentdata['PAYMENT_DIFF'] = installmentdata['AMT_INSTALMENT'] - installmentdata['AMT_PAYMENT']

    installmentdata['DPD'] = installmentdata['DAYS_ENTRY_PAYMENT'] - installmentdata['DAYS_INSTALMENT']
    installmentdata['DBD'] = installmentdata['DAYS_INSTALMENT'] - installmentdata['DAYS_ENTRY_PAYMENT']
    installmentdata['DPD'] = installmentdata['DPD'].apply(lambda x: x if x > 0 else 0)
    installmentdata['DBD'] = installmentdata['DBD'].apply(lambda x: x if x > 0 else 0)

    installmentdata['INS_IS_DPD'] = installmentdata['DPD'].apply(lambda x: 1 if x > 0 else 0)
    installmentdata['INS_IS_DPD_UNDER_30'] = installmentdata['DPD'].apply(lambda x: 1 if (x > 0) & (x <= 30) else 0)
    installmentdata['INS_IS_DPD_OVER_30'] = installmentdata['DPD'].apply(lambda x: 1 if x > 30 else 0)
    installmentdata['INS_IS_EARLY_PAYMENT'] = installmentdata['DBD'].apply(lambda x: 1 if x > 7 else 0)

    # Payment consistency
    payment_variance = installmentdata.groupby('SK_ID_PREV')['PAYMENT_PERC'].var().reset_index()
    payment_variance.columns = ['SK_ID_PREV', 'PAYMENT_VARIANCE']
    installmentdata = installmentdata.merge(payment_variance, on='SK_ID_PREV', how='left')
    installmentdata['PAYMENT_CONSISTENCY'] = 1 / (1 + installmentdata['PAYMENT_VARIANCE'].fillna(0))

    installmentdata = clean_numerical_data(installmentdata, ['PAYMENT_PERC', 'PAYMENT_DIFF', 'DPD', 'DBD', 'PAYMENT_CONSISTENCY'])

    # Numeric aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['min', 'max', 'mean', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum'],
        'DAYS_INSTALMENT': ['max', 'mean', 'sum'],
        'INS_IS_DPD': ['mean', 'sum'],
        'INS_IS_DPD_UNDER_30': ['mean', 'sum'],
        'INS_IS_DPD_OVER_30': ['mean', 'sum'],
        'INS_IS_EARLY_PAYMENT': ['mean', 'sum'],
        'PAYMENT_CONSISTENCY': ['min', 'max', 'mean']
    }

    # categorical aggregation logic (binary â†’ mean, multi â†’ mode)
    for col, n_unique in ins_label_info.items():
        if n_unique == 2:
            aggregations[col] = ['mean']
        elif n_unique > 2:
            aggregations[col] = [lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan]

    installmentdata_aggregated = installmentdata.groupby('SK_ID_CURR').agg(aggregations)

    # rename columns reflecting binary/multi
    installmentdata_aggregated.columns = pd.Index([
        f"INSTAL_{col[0]}_BINARY_MEAN" if col[1] == 'mean' and ins_label_info.get(col[0], 0) == 2
        else (f"INSTAL_{col[0]}_MODE" if 'lambda' in col[1] else f"INSTAL_{col[0]}_{col[1].upper()}")
        for col in installmentdata_aggregated.columns
    ])

    installmentdata_aggregated['INSTAL_COUNT'] = installmentdata.groupby('SK_ID_CURR').size()
    installmentdata_aggregated['INSTAL_ACCOUNT_COUNT'] = installmentdata.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique()

    # Recent behavior
    recent_ins = installmentdata[installmentdata['DAYS_ENTRY_PAYMENT'] >= -365]
    if len(recent_ins) > 0:
        recent_agg = recent_ins.groupby('SK_ID_CURR').agg({
            'PAYMENT_PERC': ['mean', 'min'],
            'DPD': ['max', 'mean'],
            'INS_IS_DPD': ['mean', 'sum'],
            'INS_IS_EARLY_PAYMENT': ['mean']
        })
        recent_agg.columns = pd.Index(
            ['INSTAL_RECENT_' + e[0] + "_" + e[1].upper() for e in recent_agg.columns.tolist()])
        installmentdata_aggregated = installmentdata_aggregated.join(recent_agg, how='left', on='SK_ID_CURR')

    del installmentdata, payment_variance
    gc.collect()

    print(f"Installments payments aggregated shape: {installmentdata_aggregated.shape}")
    return installmentdata_aggregated


# ========== MAIN TABLE PROCESSING AND MERGING ==========

def merge_all_processed_tables(df, num_rows=None):
    """
    Merge bureau_agg, cc_agg, prev_agg, pos_agg, ins_agg
    into your preprocessed main dataframe (df).
    Each step is timed, logged, and safely handled.
    """
    print(f"Starting with main dataframe: {df.shape}")

    with timer("Merging bureau and balance data"):
        try:
            bureaudata_aggregated = bureau_and_balance(num_rows)
            df = df.merge(bureaudata_aggregated, on='SK_ID_CURR', how='left')
            print(f"After merging bureau data: {df.shape}")
            del bureaudata_aggregated; gc.collect()
        except Exception as e:
            print(f"âš ï¸� Skipping bureau data due to error: {e}")

    with timer("Merging credit card balance data"):
        try:
            creditcarddata_aggregated = credit_card_balance(num_rows)
            df = df.merge(creditcarddata_aggregated, on='SK_ID_CURR', how='left')
            print(f"After merging credit card data: {df.shape}")
            del creditcarddata_aggregated; gc.collect()
        except Exception as e:
            print(f"âš ï¸� Skipping credit card data due to error: {e}")

    with timer("Merging previous applications"):
        try:
            previousdata_aggregated = previous_applications(num_rows)
            df = df.merge(previousdata_aggregated, on='SK_ID_CURR', how='left')
            print(f"After merging previous applications: {df.shape}")
            del previousdata_aggregated; gc.collect()
        except Exception as e:
            print(f"âš ï¸� Skipping previous applications due to error: {e}")

    with timer("Merging POS-CASH balance"):
        try:
            posdata_aggregated = pos_cash(num_rows)
            df = df.merge(posdata_aggregated, on='SK_ID_CURR', how='left')
            print(f"After merging POS-CASH balance: {df.shape}")
            del posdata_aggregated; gc.collect()
        except Exception as e:
            print(f"âš ï¸� Skipping POS-CASH balance due to error: {e}")

    with timer("Merging installments payments"):
        try:
            installmentdata_aggregated = installments_payments(num_rows)
            df = df.merge(installmentdata_aggregated, on='SK_ID_CURR', how='left')
            print(f"After merging installments payments: {df.shape}")
            del installmentdata_aggregated; gc.collect()
        except Exception as e:
            print(f"âš ï¸� Skipping installments payments due to error: {e}")

    print(f"\nâœ… Final merged dataframe shape: {df.shape}")

    # Split back into train and test
    train_df = df[df['is_train'] == 1].drop('is_train', axis=1)
    test_df = df[df['is_train'] == 0].drop('is_train', axis=1)

    print(f"Final train shape: {train_df.shape}")
    print(f"Final test shape: {test_df.shape}")

    return train_df, test_df, df

if __name__ == "__main__":
    with timer("Full data processing pipeline"):
        main_df = application_train_test(num_rows=None, nan_as_category=False)
        train_df, test_df, full_df = merge_all_processed_tables(main_df, num_rows=None)

    print("\n=== ğŸ�‰ PROCESSING COMPLETE ===")
    print(f"Final train data: {train_df.shape[0]:,} rows Ã— {train_df.shape[1]:,} columns")
    print(f"Final test data:  {test_df.shape[0]:,} rows Ã— {test_df.shape[1]:,} columns")
    print(f"Full merged data: {full_df.shape[0]:,} rows Ã— {full_df.shape[1]:,} columns")


train_df.info()


train_df.head()


def simple_statics(df):
    # è¯»å…¥æ•°æ�®
    stats = []
    for col in df.columns:
        stats.append((col, df[col].nunique(), 
                      (df[col].isnull()).sum() * 100 / df.shape[0],
                      df[col].value_counts(normalize=True, dropna=False).values[0] * 100, 
                      df[col].dtype))

    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage_of_null',
                                            'Percentage_of_mode', 'Type'])
    stats_df.sort_values('Unique_values', ascending=False, inplace=True)
    return stats_df
sts_df = simple_statics(train_df)
sts_df.sort_values(by=['Percentage_of_null'],ascending=False)


sts_df[sts_df['Percentage_of_null']<10.0].sort_values(by=['Percentage_of_null'],ascending=False)


yflag = 'TARGET'
plt.figure(figsize=(10,6))
train_df['TARGET'].value_counts(dropna=False).plot.bar()


train_df['TARGET'] = train_df['TARGET'].map({0.0:0,1.0:1})
train_df['TARGET'].value_counts(dropna=False)


col = train_df.columns.difference([yflag,'SK_ID_CURR'])
# ç­›é€‰floatçš„æ•°å€¼ç±»å�‹å�˜é‡�
num_list = train_df[col].select_dtypes(include=['float','int']).columns.tolist()
# ç­›é€‰intå­—ç¬¦å�‹çš„æ•°å€¼ç±»å�‹å�˜é‡�
int_list = train_df[col].select_dtypes(include=['int']).columns.tolist()
print('float&intå�‹å�˜é‡�å…±',len(num_list))
print('intç±»å�‹å�˜é‡�å…±',len(int_list))


def cal_iv(df0, var_iv, y_flag, breaks_list, stop_limit0):
    """
    åˆ†ç®±å¹¶è®¡ç®—IVå€¼
    :param df0: DFæ ¼å¼�çš„æ•°æ�®
    :param var_iv: éœ€è¦�è®¡ç®—ivçš„åˆ—çš„åˆ—è¡¨
    :param y_flag: yæ ‡ç­¾
    :return iv_df: variableå’Œivå€¼
    :return bins_base: å­—å…¸ï¼Œåˆ†ç®±å€¼
    """
    iv_list = []
    bins_base = sc.woebin(df0[var_iv + [y_flag]], y=y_flag, breaks_list=breaks_list, method='tree', stop_limit=stop_limit0)
    # åˆ†ç®±å�¯è§†åŒ–å›¾
    # bins_show = sc.woebin_plot(bins_base)
    for col, iv_df_i in bins_base.items():
        iv_df_i['bad_distr'] = iv_df_i['bad']/iv_df_i['bad'].sum() #è¾¹é™…å��å� æ¯”
        iv_df_i['good_distr'] = iv_df_i['good']/iv_df_i['good'].sum() #è¾¹é™…å¥½å� æ¯”
        iv_df_i = iv_df_i.rename(columns={'variable':'å�˜é‡�å��','bin':'åˆ†ç®±','count':'åˆ†ç®±å®¢æˆ·æ•°','count_distr':'åˆ†ç®±å®¢æˆ·æ•°å� æ¯”',
                                         'good':'å¥½å®¢æˆ·æ•°','bad':'å��å®¢æˆ·æ•°','badprob':'åŒºé—´å��è´¦ç�‡','bad_distr':'è¾¹é™…å��å®¢æˆ·å� æ¯”',
                                         'good_distr':'è¾¹é™…å¥½å®¢æˆ·å� æ¯”'})
        iv_df_i = iv_df_i[['å�˜é‡�å��','åˆ†ç®±','åˆ†ç®±å®¢æˆ·æ•°','å¥½å®¢æˆ·æ•°','å��å®¢æˆ·æ•°','åˆ†ç®±å®¢æˆ·æ•°å� æ¯”','è¾¹é™…å¥½å®¢æˆ·å� æ¯”','è¾¹é™…å��å®¢æˆ·å� æ¯”',
                           'åŒºé—´å��è´¦ç�‡','woe','bin_iv','total_iv']]
        bins_base[col] = iv_df_i
        iv_list.append((col,iv_df_i['total_iv'][0])) 
    iv_df = pd.DataFrame.from_records(iv_list,columns=['variable','iv_train'])
    iv_df = iv_df.sort_values(by=['iv_train'], ascending=False)
    return iv_df, bins_base


iv_table, bins_df = cal_iv(train_df, num_list, yflag ,breaks_list={}, stop_limit0=0)


iv_table[iv_table['iv_train']>0.05]


bins_df['INCOME_PER_CHILD']


iv_select = iv_table[iv_table['iv_train']>0.02]
cols =iv_select["variable"].tolist()
cols = cols+["TARGET"]
new_df = train_df[cols]
# new_df.to_csv("selected_features.csv", index=False)
new_df.shape


test = test_df.reset_index()
test_id = test['SK_ID_CURR']
train_data = new_df.copy()
features = new_df.columns.difference(["TARGET"]).tolist()
X = train_data[features].values
y = train_data["TARGET"].astype('int32')
test_data = test[cols]
X_test = test_data[features].values
print(train_data.shape)
print(test_data.shape)


features = new_df.columns.difference(["TARGET"]).tolist()
X = train_data[features].values
y = train_data["TARGET"].astype('int32')
X_test = test_data[features].values
train = train_df[train_df['TARGET'].isnull()==False]
test = train_df[train_df['TARGET'].isnull()==True]


def fit_lgbm_with_pruning(trial, train, val, devices=(-1,), seed=42, cat_features=None, num_rounds=200):
    """
    è®­ç»ƒLightGBMæ¨¡å�‹ï¼ˆé€‚é…� LightGBM v4+: ç”¨ callbacks å®�ç�°æ—©å�œä¸�æ—¥å¿—ï¼›æ”¯æŒ� Optuna å‰ªæ��ï¼‰
    """

    X_train, y_train = train
    X_valid, y_valid = val
    metric = 'auc'

    # ---- Optuna é‡‡æ ·ï¼ˆæ�¨è��çš„ APIï¼Œé�¿å…�å¼ƒç”¨è­¦å‘Šï¼‰----
    params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.1, 10.0, log=True),
        'bagging_freq': trial.suggest_int('bagging_freq', 2, 10),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'max_depth': trial.suggest_int('max_depth', 2, 4),
        'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 2.0, 5.0),
        'min_split_gain': trial.suggest_float('min_split_gain', 2.0, 10.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 500, 5000),
        'max_bin': trial.suggest_int('max_bin', 2, 30),
        'metric': metric,
        'verbosity': -1,
        'seed': seed,
    }

    # ---- è®¾å¤‡é€‰æ‹© ----
    device = devices[0]
    if device != -1:
        print(f'using gpu device_id {device}...')
        params.update({'device': 'gpu', 'gpu_device_id': device})

    # ---- æ•°æ�®é›† ----
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    dvalid = lgb.Dataset(X_valid,  label=y_valid,  categorical_feature=cat_features)

    # èµ·å›ºå®šå��å­—ï¼Œå��ç»­ best_score/å‰ªæ��éƒ½ç”¨ 'valid'
    valid_sets = [dtrain, dvalid]
    valid_names = ['training', 'valid']

    # ---- callbacksï¼šæ—©å�œ + æ—¥å¿— + å‰ªæ�� ----
    early_stop_rounds = 100
    log_period = 100
    pruning_cb = optuna.integration.LightGBMPruningCallback(trial, metric=metric, valid_name='valid')

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stop_rounds),
        lgb.log_evaluation(period=log_period),
        pruning_cb,
    ]

    print('training LGB:')
    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=num_rounds,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks
    )

    # é¢„æµ‹
    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)

    # è®°å½•åˆ†æ•°ï¼ˆæŒ‰æˆ‘ä»¬ç»™çš„ valid å��ç§°å�–ï¼‰
    # model.best_score ç»“æ�„ï¼š{'training': {'auc': ...}, 'valid': {'auc': ...}}
    print('best_score', model.best_score)
    log = {
        'train/auc': model.best_score['training'][metric],
        'valid/auc': model.best_score['valid'][metric],
    }
    return model, y_pred_valid, log



def objective_with_prune(trial: Trial, fast_check=False):
    """
    ç›®æ ‡å‡½æ•°
    """
    folds = 5
    seed = 42
    shuffle = True
    kf = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=seed) # 5æŠ˜Kfold

    X_train, y_train = X, y
    y_valid_pred_total = np.zeros(X_train.shape[0])
    gc.collect()

    models0 = [] # æ¨¡å�‹å¯¹è±¡åˆ—è¡¨
    valid_score = 0 
    
    # 5æŠ˜äº¤å�‰éªŒè¯�
    for train_idx, valid_idx in kf.split(X_train, y_train):
        train_data = X_train[train_idx, :], y_train[train_idx]
        valid_data = X_train[valid_idx, :], y_train[valid_idx]

        print('train', len(train_idx), 'valid', len(valid_idx))
        model, y_pred_valid, log = fit_lgbm_with_pruning(trial, train_data,
                                                         valid_data,
                                                         num_rounds=200)
        y_valid_pred_total[valid_idx] = y_pred_valid
        models0.append(model)
        gc.collect()
        valid_score += log["valid/auc"] 
        if fast_check:
            break
    valid_score /= len(models0)
    return valid_score

study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize")
study.optimize(objective_with_prune, n_trials=50)


params = study.best_params
params


params['seed'] = 42
params['objective'] = 'binary'
params['boosting'] = "gbdt"
params['metric'] = 'auc'
params['verbosity'] = -1
early_stop = 100
log_period = 20
num_rounds = 200

# åˆ‡åˆ†
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(X_train.shape)
print(X_valid.shape)

# Datasetï¼ˆå¦‚æ�œæœ‰ç±»åˆ«åˆ—ï¼Œä¼ å…¥åˆ—å��æˆ–ç´¢å¼•åˆ—è¡¨ï¼‰
d_train = lgb.Dataset(X_train, label=y_train, categorical_feature=None)
d_valid = lgb.Dataset(X_valid,  label=y_valid,  categorical_feature=None)

valid_sets  = [d_train, d_valid]
valid_names = ['training', 'valid']

print('training LGB:')
model = lgb.train(
    params=params,
    train_set=d_train,
    num_boost_round=num_rounds,
    valid_sets=valid_sets,
    valid_names=valid_names,
    callbacks=[
        lgb.early_stopping(stopping_rounds=early_stop),
        lgb.log_evaluation(period=log_period)
    ]
)

print('best iteration', model.best_iteration)

# é¢„æµ‹
y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)

# åˆ†æ•°ï¼ˆç”¨æˆ‘ä»¬èµ·çš„å��å­— 'training' å’Œ 'valid'ï¼‰
train_auc = model.best_score['training']['auc']
valid_auc = model.best_score['valid']['auc']
print('best_score', {'training/auc': train_auc, 'valid/auc': valid_auc})



# é€šè¿‡æ¨¡å�‹å¯¹è±¡é¢„æµ‹ç»“æ�œ
def pred(X_test, models):
    y_test_pred_total = np.zeros(X_test.shape[0])
    for i, model in enumerate(models):
        print(f'predicting {i}-th model')
        y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)
        y_test_pred_total += y_pred_test
    y_test_pred_total /= len(models)
    return y_test_pred_total

# å�¯è§†åŒ–ç‰¹å¾�é‡�è¦�æ€§
def plot_feature_importance(model,features):
    importance_df = pd.DataFrame(model.feature_importance(),
                                 index=features,
                                 columns=['importance']).sort_values('importance').sort_values(by='importance',ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(15, 10))
    importance_df.plot.barh(ax=ax)
    fig.show()


# å±•ç¤ºæ�’å��å‰�20çš„ç‰¹å¾�é‡�è¦�æ€§
plot_feature_importance(model, features)


y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
# å¦‚æ�œæœ‰æµ‹è¯•é›† IDï¼š
submission = pd.DataFrame({'SK_ID_CURR': test_id, 'TARGET': y_test_pred})
submission.to_csv('submission1.csv', index=False)


from autogluon.tabular import TabularPredictor,TabularDataset
train_Dataset=TabularDataset(train_data)
test_Dataset=TabularDataset(test_data)
predictor = TabularPredictor(label='TARGET',problem_type='binary',eval_metric='roc_auc').fit(train_Dataset,presets='best_quality')


y_pred = predictor.predict_proba(test_Dataset)
y_pred = y_pred.drop(y_pred.columns[0], axis=1)
y_pred.index=test_id
y_pred = y_pred.rename(columns={y_pred.columns[0]: 'TARGET'})
y_pred.to_csv('submission2.csv',index=True)


predictor.feature_importance(train_Dataset)

