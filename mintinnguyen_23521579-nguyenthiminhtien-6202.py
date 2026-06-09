import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, log_loss
from sklearn.preprocessing import MinMaxScaler
import gc
import os
from typing import List, Tuple, Dict
import logging
from datetime import datetime
import subprocess
import sys
!pip install unidecode
from unidecode import unidecode

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_PATH = '/kaggle/input/ds-108-p-21-assigment-06/'
DELAY_4_6_PATH = os.path.join(DATA_PATH, 'delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv')
DELAY_7_9_PATH = os.path.join(DATA_PATH, 'delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv')
NOT_DELAY_4_6_PATH = os.path.join(DATA_PATH, 'not_delay_4_6_CONDITION_PRODUCT_SUPPLIER.csv')
NOT_DELAY_7_9_PATH = os.path.join(DATA_PATH, 'not_delay_7_9_CONDITION_PRODUCT_SUPPLIER.csv')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'PILOT_10.csv')
OUTPUT_PATH = '/kaggle/working/'

TARGET_COL = 'label'
ID_SUBMISSION_COL = 'ID'
ID_RULE_COL = 'GLOBAL_NO'

FILTER_COLS_CONFIG = {
    'SUBSIDIARY_CD': 'MJP',
    'MC_PLANT_DIV_CHAR_3': '2',
    'VOUCHER_DIV_VALUES': ["110", "160", "170", "180"],
    'CUSTSUB_SUBSIDIARY_CD_IS_NULL': True
}
FILTER_COLUMN_NAMES = ['SUBSIDIARY_CD', 'MC_PLANT_DIV', 'VOUCHER_DIV', 'CUSTSUB_SUBSIDIARY_CD']

DATETIME_FEATURES = ['Order date', 'VSD']
NUMERICAL_FEATURES_RAW = [
    'SUPPLIER_INV_AMOUNT', 'WEIGHT PER PIECE', 'SO QTY', 'Consider count hodiday Saturday'
]
CATEGORICAL_FEATURES_ORIGINAL = [
    'Ship Mode', 'CUST_CD', 'CLASSIFY_CD', 'INNER_CD', 'PACKING_RANK', 'PRODUCT_CD',
    'OTHER_AREA_SHIP_DIV', 'PRODUCT_ATTRIBUTION', 'SUPPLIER_CD', 'DELI_DIV', 'SUPPLIER_DIV',
    ID_RULE_COL
]

USEFUL_FEATURES_BASE = list(set(DATETIME_FEATURES + NUMERICAL_FEATURES_RAW + CATEGORICAL_FEATURES_ORIGINAL))
REMOVE_FEATURES_BASE = list(set([
    'SO_TIME', 'QTUF_RCV_NO', 'SOUF_RCV_NO', 'HEAVY_FLG', 'EXPENSIVE_FLG', 'SPECIAL_DIV',
    'WEIGHT_UNIT', 'PACK_QTY', 'IO_UNFIT_FLG', 'ACTUAL_SHIP_DAYS', 'SPECIFY_PRODUCTION_DAYS',
    'SPECIFY_SHIP_DAYS', 'HAZARD_FLG', 'SUPPLIER_CATEGORY_CD', 'PRODUCT_ASSORT', 'REASON_CD',
    'BRAND_CD', 'Sales order line number', 'Stock class', 'ALLOCATION QTY', 'LOGICAL PLANT',
    'PURCHASE AMOUNT', 'DIRECT SHIP FLG', 'SHIP DECISION NO'
] + [fc for fc in FILTER_COLUMN_NAMES if fc not in USEFUL_FEATURES_BASE]))

# Hyperparameters
RANDOM_STATE = 42
N_FOLDS = 5
CATBOOST_BASE_PARAMS = {
    'iterations': 1200,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_seed': RANDOM_STATE,
    'verbose': 200,
    'early_stopping_rounds': 150,
    'eval_metric': 'AUC',
    'task_type': 'CPU'
}
FEATURE_SELECTION_TOP_N = 25
UNDERSAMPLING_RATIO_DELAY_TO_NOT_DELAY = 8
HIGH_DELAY_SUPPLIER_THRESHOLD = 0.80
LOW_DELAY_SUPPLIER_THRESHOLD = 0.20
RULE_PROBABILITY_SMOOTHING = 0.9
SUBMISSION_THRESHOLD = 0.5
ENSEMBLE_SEEDS = [42, 43, 44]

def install_unidecode():
    try:
        from unidecode import unidecode
        return unidecode
    except ImportError:
        logger.warning("unidecode not found, attempting to install...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "unidecode"])
            from unidecode import unidecode
            logger.info("unidecode installed successfully.")
            return unidecode
        except Exception as e:
            logger.error(f"Failed to install unidecode: {e}")
            raise

unidecode = install_unidecode()

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type).startswith(('int', 'float')):
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type).startswith('int'):
                for dtype in [np.int8, np.int16, np.int32, np.int64]:
                    if c_min > np.iinfo(dtype).min and c_max < np.iinfo(dtype).max:
                        df[col] = df[col].astype(dtype)
                        break
            else:
                df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logger.info(f'Mem usage decreased to {end_mem:.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> bool:
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    return True

def load_and_filter_data(
    path: str, required_cols: List[str], filter_cols: List[str], filter_config: Dict, is_test: bool = False
) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    try:
        df_cols = pd.read_csv(path, nrows=0).columns.tolist()
        load_cols = list(set(required_cols + filter_cols) & set(df_cols))
        if is_test:
            load_cols = list(set(load_cols + [ID_SUBMISSION_COL, ID_RULE_COL]) & set(df_cols))
        elif TARGET_COL not in load_cols:
            load_cols.append(TARGET_COL) if TARGET_COL in df_cols else None
        df = pd.read_csv(path, usecols=load_cols, low_memory=False, on_bad_lines='skip')
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns from {path}")
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

    if df.empty or not validate_columns(df, required_cols):
        return df

    # Apply filters
    for col, config in filter_config.items():
        if col == 'SUBSIDIARY_CD' and col in df:
            df = df[df[col] == config]
        elif col == 'MC_PLANT_DIV_CHAR_3' and 'MC_PLANT_DIV' in df:
            df['MC_PLANT_DIV'] = df['MC_PLANT_DIV'].astype(str)
            df = df[df['MC_PLANT_DIV'].str[2:3] == config]
        elif col == 'VOUCHER_DIV_VALUES' and 'VOUCHER_DIV' in df:
            df['VOUCHER_DIV'] = df['VOUCHER_DIV'].astype(str)
            df = df[df['VOUCHER_DIV'].isin(config)]
        elif col == 'CUSTSUB_SUBSIDIARY_CD_IS_NULL' and 'CUSTSUB_SUBSIDIARY_CD' in df:
            if config:
                df = df[df['CUSTSUB_SUBSIDIARY_CD'].isnull()]
        else:
            logger.warning(f"Filter column {col} not applicable for {path}")

    logger.info(f"Shape after filtering: {df.shape}")
    final_cols = list(set(required_cols + ([ID_SUBMISSION_COL, ID_RULE_COL] if is_test else [TARGET_COL, ID_RULE_COL])))
    final_cols = [col for col in final_cols if col in df.columns]
    return reduce_mem_usage(df[final_cols])

def handle_nulls(
    df: pd.DataFrame, num_feats: List[str], cat_feats: List[str], date_feats: List[str]
) -> pd.DataFrame:
    df_copy = df.copy()
    num_feats = [f for f in num_feats if f in df_copy.columns and f not in [ID_SUBMISSION_COL, ID_RULE_COL]]
    cat_feats = [f for f in cat_feats if f in df_copy.columns and f != ID_SUBMISSION_COL]
    date_feats = [f for f in date_feats if f in df_copy.columns]

    for col in num_feats:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(-1).astype(np.float32)
    for col in cat_feats:
        df_copy[col] = df_copy[col].astype(str).fillna('missing_value')
    for col in date_feats:
        df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
        if df_copy[col].isna().any():
            df_copy[col] = df_copy[col].ffill().bfill()
    return df_copy

def frequency_encoding(df: pd.DataFrame, cat_cols: List[str]) -> Tuple[List[str], pd.DataFrame]:
    new_cols = []
    for col in cat_cols:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[f'{col}_freq'] = df[col].map(freq).astype(np.float32)
            new_cols.append(f'{col}_freq')
    return new_cols, df

def feature_engineering(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    logger.info(f"Starting feature engineering. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    train_len = len(train_df)
    labels = train_df[TARGET_COL].copy() if TARGET_COL in train_df else pd.Series()
    train_df = train_df.drop(columns=[TARGET_COL], errors='ignore')

    # Align IDs
    for col in [ID_SUBMISSION_COL, ID_RULE_COL]:
        if col in test_df and col not in train_df:
            train_df[col] = f'train_placeholder_{col}'

    combined = pd.concat([train_df, test_df], ignore_index=True)
    logger.info(f"Combined shape: {combined.shape}")

    # Text Normalization
    for col in CATEGORICAL_FEATURES_ORIGINAL:
        if col in combined and col != ID_SUBMISSION_COL:
            combined[col] = combined[col].astype(str).str.lower().apply(
                lambda x: unidecode(x) if pd.notnull(x) else 'missing_value'
            ).fillna('missing_value')

    # Specific Categorical Mappings
    if 'OTHER_AREA_SHIP_DIV' in combined:
        empty_vals = ['missing_value', 'nan', 'none', '', 'khoang trang']
        combined['OTHER_AREA_SHIP_DIV'] = combined['OTHER_AREA_SHIP_DIV'].astype(str).str.strip()
        combined.loc[combined['OTHER_AREA_SHIP_DIV'].isin(empty_vals), 'OTHER_AREA_SHIP_DIV'] = 'own_area'
        combined.loc[combined['OTHER_AREA_SHIP_DIV'] == '1', 'OTHER_AREA_SHIP_DIV'] = 'other_area'
        combined.loc[~combined['OTHER_AREA_SHIP_DIV'].isin(['own_area', 'other_area']), 'OTHER_AREA_SHIP_DIV'] = 'unknown_area'

    # Datetime Features
    derived_date_cols = []
    for col in DATETIME_FEATURES:
        if col in combined and pd.api.types.is_datetime64_any_dtype(combined[col]):
            prefix = col.replace(' ', '_')
            combined[f'{prefix}_month'] = combined[col].dt.month.fillna(-1).astype(np.int8)
            combined[f'{prefix}_day'] = combined[col].dt.day.fillna(-1).astype(np.int8)
            combined[f'{prefix}_dayofweek'] = combined[col].dt.dayofweek.fillna(-1).astype(np.int8)
            combined[f'{prefix}_week'] = combined[col].dt.isocalendar().week.fillna(-1).astype(np.int8)
            combined[f'{prefix}_quarter'] = combined[col].dt.quarter.fillna(-1).astype(np.int8)
            combined[f'{prefix}_is_weekend'] = (combined[col].dt.dayofweek >= 5).astype(np.int8)
            derived_date_cols.extend([
                f'{prefix}_month', f'{prefix}_day', f'{prefix}_dayofweek',
                f'{prefix}_week', f'{prefix}_quarter', f'{prefix}_is_weekend'
            ])

    if 'VSD' in combined and 'Order date' in combined:
        combined['days_vsd_order'] = (
            (combined['VSD'] - combined['Order date']).dt.days.fillna(-1).astype(np.int16)
        )
        derived_date_cols.append('days_vsd_order')

    # Numerical Calculations
    derived_num_cols = []
    if 'WEIGHT PER PIECE' in combined and 'SO QTY' in combined:
        combined['WEIGHT_calc'] = (
            pd.to_numeric(combined['WEIGHT PER PIECE'], errors='coerce').fillna(-1) *
            pd.to_numeric(combined['SO QTY'], errors='coerce').fillna(-1)
        ).astype(np.float32)
        derived_num_cols.append('WEIGHT_calc')

    # Interaction Features
    if 'SUPPLIER_INV_AMOUNT' in combined and 'WEIGHT_calc' in combined:
        combined['price_per_weight'] = (
            combined['SUPPLIER_INV_AMOUNT'].astype(np.float32) /
            (combined['WEIGHT_calc'] + 1e-6)  # Avoid division by zero
        ).astype(np.float32)
        derived_num_cols.append('price_per_weight')

    # Frequency Encoding
    freq_cols, combined = frequency_encoding(combined, CATEGORICAL_FEATURES_ORIGINAL)

    # Scaling Numerical Features
    num_cols = list(set(
        [f for f in NUMERICAL_FEATURES_RAW if f in combined] +
        derived_date_cols + derived_num_cols + freq_cols
    ))
    if num_cols:
        scaler = MinMaxScaler()
        combined[num_cols] = pd.DataFrame(
            scaler.fit_transform(combined[num_cols]),
            columns=num_cols,
            index=combined.index
        ).astype(np.float32)

    # Drop Unused Features
    cols_to_drop = list(set(
        DATETIME_FEATURES + ['WEIGHT PER PIECE', 'SO QTY'] +
        [f for f in REMOVE_FEATURES_BASE if f in combined]
    ))
    cols_to_drop = [col for col in cols_to_drop if col not in [ID_SUBMISSION_COL, ID_RULE_COL]]
    combined = combined.drop(columns=cols_to_drop, errors='ignore')

    # Split back
    train_fe = combined.iloc[:train_len].copy()
    test_fe = combined.iloc[train_len:].copy()
    train_fe[TARGET_COL] = labels.astype(np.int8)
    if ID_SUBMISSION_COL in train_fe:
        train_fe = train_fe.drop(columns=[ID_SUBMISSION_COL], errors='ignore')

    logger.info(f"FE completed. Train shape: {train_fe.shape}, Test shape: {test_fe.shape}")
    return train_fe, test_fe, num_cols + freq_cols

def kfold_catboost(
    train_df: pd.DataFrame, test_df: pd.DataFrame, num_folds: int, seed: int, run_for_importance: bool = False
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    if not run_for_importance and ID_SUBMISSION_COL not in test_df:
        logger.error(f"{ID_SUBMISSION_COL} missing in test_df")
        raise ValueError(f"{ID_SUBMISSION_COL} missing")

    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(train_df), dtype=np.float32)
    sub_preds = np.zeros(len(test_df), dtype=np.float32)
    fi_df = pd.DataFrame()

    feats = [f for f in train_df.columns if f not in [TARGET_COL, ID_SUBMISSION_COL]]
    cat_feats = [f for f in CATEGORICAL_FEATURES_ORIGINAL if f in feats]
    logger.info(f"Training on {len(feats)} features, {len(cat_feats)} categorical")

    scale_pos_weight = train_df[TARGET_COL].value_counts()[0] / train_df[TARGET_COL].value_counts()[1] if 1 in train_df[TARGET_COL].value_counts() else 1.0
    logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")

    for fold, (train_idx, val_idx) in enumerate(folds.split(train_df[feats], train_df[TARGET_COL])):
        logger.info(f"Fold {fold+1}/{num_folds}")
        train_pool = Pool(train_df.iloc[train_idx][feats], train_df.iloc[train_idx][TARGET_COL], cat_features=cat_feats)
        val_pool = Pool(train_df.iloc[val_idx][feats], train_df.iloc[val_idx][TARGET_COL], cat_features=cat_feats)
        test_pool = Pool(test_df[feats], cat_features=cat_feats)

        model = CatBoostClassifier(**{**CATBOOST_BASE_PARAMS, 'random_seed': seed + fold, 'scale_pos_weight': scale_pos_weight})
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        oof_preds[val_idx] = model.predict_proba(val_pool)[:, 1]
        sub_preds += model.predict_proba(test_pool)[:, 1] / num_folds

        fi_df = pd.concat([fi_df, pd.DataFrame({
            'feature': feats,
            'importance': model.get_feature_importance(),
            'fold': fold + 1
        })], axis=0)

        if not run_for_importance:
            auc = roc_auc_score(train_df.iloc[val_idx][TARGET_COL], oof_preds[val_idx])
            binary_preds = (oof_preds[val_idx] > SUBMISSION_THRESHOLD).astype(np.int8)
            logger.info(
                f"Fold {fold+1} AUC: {auc:.4f}, F1: {f1_score(train_df.iloc[val_idx][TARGET_COL], binary_preds):.4f}, "
                f"Prec: {precision_score(train_df.iloc[val_idx][TARGET_COL], binary_preds, zero_division=0):.4f}, "
                f"Rec: {recall_score(train_df.iloc[val_idx][TARGET_COL], binary_preds, zero_division=0):.4f}"
            )

        del train_pool, val_pool, test_pool, model
        gc.collect()

    if not run_for_importance:
        auc = roc_auc_score(train_df[TARGET_COL], oof_preds)
        binary_preds = (oof_preds > SUBMISSION_THRESHOLD).astype(np.int8)
        logger.info(f"Full OOF AUC: {auc:.4f}, F1: {f1_score(train_df[TARGET_COL], binary_preds):.4f}")

    mean_fi = fi_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    selected_feats = mean_fi.head(FEATURE_SELECTION_TOP_N).index.tolist()
    logger.info(f"Top {FEATURE_SELECTION_TOP_N} features:\n{mean_fi.head(FEATURE_SELECTION_TOP_N)}")

    return sub_preds, fi_df, selected_feats

def ensemble_catboost(
    train_df: pd.DataFrame, test_df: pd.DataFrame, num_folds: int
) -> np.ndarray:
    all_sub_preds = np.zeros(len(test_df))
    for seed in ENSEMBLE_SEEDS:
        logger.info(f"Running CatBoost ensemble with seed {seed}")
        sub_preds, _, _ = kfold_catboost(train_df, test_df, num_folds, seed, run_for_importance=False)
        all_sub_preds += sub_preds / len(ENSEMBLE_SEEDS)
    return all_sub_preds

# --- Main Execution ---
os.makedirs(OUTPUT_PATH, exist_ok=True)
logger.info("Listing input files:")
for root, _, files in os.walk(DATA_PATH):
    for f in files:
        logger.info(os.path.join(root, f))

cols_needed_train = list(set(USEFUL_FEATURES_BASE + [TARGET_COL, ID_RULE_COL, 'SUPPLIER_CD'] + FILTER_COLUMN_NAMES))
cols_needed_test = list(set(USEFUL_FEATURES_BASE + [ID_SUBMISSION_COL, ID_RULE_COL, 'SUPPLIER_CD'] + FILTER_COLUMN_NAMES))

logger.info("Loading training data...")
data_delay_46 = load_and_filter_data(DELAY_4_6_PATH, cols_needed_train, FILTER_COLUMN_NAMES, FILTER_COLS_CONFIG)
data_delay_79 = load_and_filter_data(DELAY_7_9_PATH, cols_needed_train, FILTER_COLUMN_NAMES, FILTER_COLS_CONFIG)
data_not_delay_46 = load_and_filter_data(NOT_DELAY_4_6_PATH, cols_needed_train, FILTER_COLUMN_NAMES, FILTER_COLS_CONFIG)
data_not_delay_79 = load_and_filter_data(NOT_DELAY_7_9_PATH, cols_needed_train, FILTER_COLUMN_NAMES, FILTER_COLS_CONFIG)

data_delay = pd.concat([data_delay_46, data_delay_79], ignore_index=True).drop_duplicates()
data_not_delay = pd.concat([data_not_delay_46, data_not_delay_79], ignore_index=True).drop_duplicates()
del data_delay_46, data_delay_79, data_not_delay_46, data_not_delay_79
gc.collect()

if data_delay.empty or data_not_delay.empty:
    logger.error("Delay or not_delay data is empty. Exiting.")
    raise ValueError("Empty training data")

n_not_delay = min(len(data_delay) * UNDERSAMPLING_RATIO_DELAY_TO_NOT_DELAY, len(data_not_delay))
data_not_delay = data_not_delay.sample(n=n_not_delay, random_state=RANDOM_STATE)
data_delay[TARGET_COL] = 1
data_not_delay[TARGET_COL] = 0

train_data = pd.concat([data_delay, data_not_delay], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)
train_data = reduce_mem_usage(train_data)
logger.info(f"Training data shape: {train_data.shape}, Label dist:\n{train_data[TARGET_COL].value_counts(normalize=True)}")

logger.info("Loading test data...")
test_data = load_and_filter_data(TEST_DATA_PATH, cols_needed_test, FILTER_COLUMN_NAMES, FILTER_COLS_CONFIG, is_test=True)
test_data = reduce_mem_usage(test_data)
if test_data.empty or ID_SUBMISSION_COL not in test_data or ID_RULE_COL not in test_data:
    logger.error("Test data invalid or missing ID columns")
    raise ValueError("Invalid test data")

train_data = handle_nulls(train_data, NUMERICAL_FEATURES_RAW, CATEGORICAL_FEATURES_ORIGINAL, DATETIME_FEATURES)
test_data = handle_nulls(test_data, NUMERICAL_FEATURES_RAW, CATEGORICAL_FEATURES_ORIGINAL, DATETIME_FEATURES)

train_fe, test_fe, derived_num_cols = feature_engineering(train_data.copy(), test_data.copy())
del train_data, test_data
gc.collect()

if train_fe.empty or TARGET_COL not in train_fe:
    logger.error("Training data after FE is empty or missing target")
    raise ValueError("Invalid training data after FE")

logger.info("First pass: Feature importance...")
_, _, selected_feats = kfold_catboost(train_fe, test_fe, N_FOLDS, RANDOM_STATE, run_for_importance=True)

if not selected_feats:
    logger.warning("No features selected. Using all features for final model.")
    selected_feats = [f for f in train_fe.columns if f not in [TARGET_COL, ID_SUBMISSION_COL]]

train_final = train_fe[list(set(selected_feats + [TARGET_COL]))]
test_final = test_fe[list(set(selected_feats + [ID_SUBMISSION_COL, ID_RULE_COL]))]
logger.info(f"Final train shape: {train_final.shape}, Final test shape: {test_final.shape}")

logger.info("Second pass: Ensemble training...")
sub_preds = ensemble_catboost(train_final, test_final, N_FOLDS)
del train_final, test_final
gc.collect()

# --- Rules ---
logger.info("Generating rules...")
rule_cols = [ID_RULE_COL, 'SUPPLIER_CD']
rule_delay_46 = load_and_filter_data(DELAY_4_6_PATH, rule_cols, FILTER_COLUMN_NAMES, FILTER_COLS_CONFIG)
rule_delay_79 = load_and_filter_data(DELAY_7_9_PATH, rule_cols, FILTER_COLUMN_NAMES, FILTER_COLS_CONFIG)
rule_not_delay_46 = load_and_filter_data(NOT_DELAY_4_6_PATH, rule_cols, FILTER_COLUMN_NAMES, FILTER_COLS_CONFIG)
rule_not_delay_79 = load_and_filter_data(NOT_DELAY_7_9_PATH, rule_cols, FILTER_COLUMN_NAMES, FILTER_COLS_CONFIG)

data_delay_rules = pd.concat([rule_delay_46, rule_delay_79], ignore_index=True).drop_duplicates(subset=[ID_RULE_COL, 'SUPPLIER_CD'])
data_not_delay_rules = pd.concat([rule_not_delay_46, rule_not_delay_79], ignore_index=True).drop_duplicates(subset=[ID_RULE_COL, 'SUPPLIER_CD'])
del rule_delay_46, rule_delay_79, rule_not_delay_46, rule_not_delay_79
gc.collect()

rules_df = pd.DataFrame(columns=[ID_RULE_COL, 'DELAY_RULE'])
if not data_delay_rules.empty and not data_not_delay_rules.empty:
    data_delay_rules[TARGET_COL] = 1
    data_not_delay_rules[TARGET_COL] = 0
    rule_data = pd.concat([data_delay_rules, data_not_delay_rules], ignore_index=True)
    rule_data = rule_data.drop_duplicates(subset=[ID_RULE_COL], keep='first')

    sdr = rule_data.groupby('SUPPLIER_CD')[TARGET_COL].agg(['sum', 'count']).reset_index()
    sdr['Delay_Ratio'] = sdr['sum'] / sdr['count']
    high_delay = sdr[sdr['Delay_Ratio'] >= HIGH_DELAY_SUPPLIER_THRESHOLD]['SUPPLIER_CD'].unique()
    low_delay = sdr[sdr['Delay_Ratio'] <= LOW_DELAY_SUPPLIER_THRESHOLD]['SUPPLIER_CD'].unique()

    supplier_rules = {sc: 'Y' for sc in high_delay}
    supplier_rules.update({sc: 'N' for sc in low_delay})
    rules = [
        {ID_RULE_COL: row[ID_RULE_COL], 'DELAY_RULE': supplier_rules.get(row['SUPPLIER_CD'])}
        for _, row in rule_data[[ID_RULE_COL, 'SUPPLIER_CD']].drop_duplicates().iterrows()
        if row['SUPPLIER_CD'] in supplier_rules
    ]
    if rules:
        rules_df = pd.DataFrame(rules).drop_duplicates(subset=[ID_RULE_COL])
    rules_df.to_csv(os.path.join(OUTPUT_PATH, 'rules.csv'), index=False)
    logger.info(f"Generated {len(rules_df)} rules")
else:
    logger.warning("Rule data empty or missing columns. Skipping rule generation.")

# --- Submission ---
logger.info("Preparing submission...")
submission_df = pd.DataFrame({
    ID_SUBMISSION_COL: test_fe[ID_SUBMISSION_COL].astype(str),
    ID_RULE_COL: test_fe[ID_RULE_COL].astype(str),
    'Predicted_Probability': sub_preds
})
submission_df['Predicted_Probability'] *= RULE_PROBABILITY_SMOOTHING

if not rules_df.empty:
    submission_df = submission_df.merge(rules_df, on=ID_RULE_COL, how='left')
    submission_df.loc[submission_df['DELAY_RULE'] == 'Y', 'Predicted_Probability'] = (
        submission_df['Predicted_Probability'] * 0.2 + 0.8
    )
    submission_df.loc[submission_df['DELAY_RULE'] == 'N', 'Predicted_Probability'] = (
        submission_df['Predicted_Probability'] * 0.2 + 0.2
    )
    submission_df = submission_df.drop(columns=['DELAY_RULE'], errors='ignore')

submission_df[TARGET_COL] = (submission_df['Predicted_Probability'] > SUBMISSION_THRESHOLD).astype(np.int8)
final_submission = submission_df[[ID_SUBMISSION_COL, TARGET_COL]].rename(columns={ID_SUBMISSION_COL: 'ID'})
final_submission = final_submission.drop_duplicates(subset=['ID'], keep='first')
final_submission.to_csv(os.path.join(OUTPUT_PATH, 'submission.csv'), index=False)
logger.info(f"Submission saved with {len(final_submission)} rows\n{final_submission.head()}")
logger.info(f"Label distribution:\n{final_submission[TARGET_COL].value_counts(normalize=True)}")

logger.info("Script completed successfully.")

