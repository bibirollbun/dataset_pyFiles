# ================== IMPORTS ==================
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
from catboost import CatBoostRanker
from sklearn.model_selection import GroupKFold
import gc
import re
import warnings
warnings.filterwarnings("ignore")


# ================== PARAMETERS ==================
SAMPLE_GROUP_FRAC = 0.30  # Use 0.30 for RAM safety, increase if possible!
TRAIN_PATH = '/kaggle/input/aeroclub-recsys-2025/train.parquet'
TEST_PATH = '/kaggle/input/aeroclub-recsys-2025/test.parquet'


# ============= UTILITY: CONVERT DURATION =============
def duration_to_minutes(s):
    """Convert 'hh:mm:ss' or 'mm:ss' to minutes as float, else NaN."""
    if pd.isnull(s):
        return np.nan
    if isinstance(s, str):
        parts = list(map(float, re.findall(r'\d+', s)))
        if len(parts) == 3:
            return parts[0]*60 + parts[1] + parts[2]/60
        elif len(parts) == 2:
            return parts[0] + parts[1]/60
        elif len(parts) == 1:
            return parts[0]
    return np.nan

def convert_duration_cols(df):
    duration_cols = [col for col in df.columns if 'duration' in col]
    for col in duration_cols:
        df[col] = df[col].apply(duration_to_minutes)
    return df


# ============= FEATURE ENGINEERING =============
def add_features(df):
    df['reqDate'] = pd.to_datetime(df['requestDate'], errors='coerce')
    df['depDate'] = pd.to_datetime(df['legs0_departureAt'], errors='coerce')
    df['hours_to_departure'] = (df['depDate'] - df['reqDate']).dt.total_seconds() / 3600
    df['dep_hour'] = df['depDate'].dt.hour
    df['dep_dow'] = df['depDate'].dt.dayofweek
    df['is_cheapest'] = (df['totalPrice'] == df.groupby('ranker_id')['totalPrice'].transform('min')).astype(int)
    df['price_rank'] = df.groupby('ranker_id')['totalPrice'].rank(method='min')
    df['price_rel'] = df['totalPrice'] / df.groupby('ranker_id')['totalPrice'].transform('mean')
    seg_cols = [f'legs0_segments{i}_flightNumber' for i in range(4)]
    df['n_stops'] = df[seg_cols].notna().sum(axis=1)
    df['is_direct'] = (df['n_stops'] == 1).astype(int)
    df['loyalty_match'] = (df['frequentFlyer'] == df['legs0_segments0_marketingCarrier_code']).astype(int)
    for col in ['totalPrice', 'taxes']:
        if col in df.columns:
            df[col + '_min'] = df.groupby('ranker_id')[col].transform('min')
            df[col + '_max'] = df.groupby('ranker_id')[col].transform('max')
            df[col + '_gap'] = df[col] - df[col + '_min']
            df[col + '_pct'] = df[col] / (df[col + '_max'] + 1e-6)
    df['group_size'] = df.groupby('ranker_id')['Id'].transform('count')
    if 'legs0_departureAt' in df.columns:
        df['is_earliest_dep'] = (df['legs0_departureAt'] == df.groupby('ranker_id')['legs0_departureAt'].transform('min')).astype(int)
    if 'legs0_arrivalAt' in df.columns:
        df['is_latest_arr'] = (df['legs0_arrivalAt'] == df.groupby('ranker_id')['legs0_arrivalAt'].transform('max')).astype(int)
    return df


# ============= PREPROCESSING =============
cat_cols_cb = [
    'frequentFlyer', 'nationality', 'legs0_segments0_aircraft_code',
    'legs0_segments0_arrivalTo_airport_city_iata', 'legs0_segments0_arrivalTo_airport_iata',
    'legs0_segments0_cabinClass', 'legs0_segments0_departureFrom_airport_iata',
    'legs0_segments0_flightNumber', 'legs0_segments0_marketingCarrier_code', 'legs0_segments0_operatingCarrier_code',
    'legs0_segments1_aircraft_code', 'legs0_segments1_arrivalTo_airport_city_iata', 'legs0_segments1_arrivalTo_airport_iata',
    'legs0_segments1_departureFrom_airport_iata', 'legs0_segments1_flightNumber', 'legs0_segments1_marketingCarrier_code',
    'legs0_segments1_operatingCarrier_code', 'legs0_segments2_aircraft_code', 'legs0_segments2_arrivalTo_airport_city_iata',
    'legs0_segments2_arrivalTo_airport_iata', 'legs0_segments2_departureFrom_airport_iata', 'legs0_segments2_flightNumber',
    'legs0_segments2_marketingCarrier_code', 'legs0_segments2_operatingCarrier_code', 'legs0_segments3_aircraft_code',
    'legs0_segments3_arrivalTo_airport_city_iata', 'legs0_segments3_arrivalTo_airport_iata', 'legs0_segments3_departureFrom_airport_iata',
    'legs0_segments3_flightNumber', 'legs0_segments3_marketingCarrier_code', 'legs0_segments3_operatingCarrier_code',
    'legs1_arrivalAt', 'legs1_departureAt', 'legs1_segments0_aircraft_code', 'legs1_segments0_arrivalTo_airport_city_iata',
    'legs1_segments0_arrivalTo_airport_iata', 'legs1_segments0_departureFrom_airport_iata', 'legs1_segments0_flightNumber',
    'legs1_segments0_marketingCarrier_code', 'legs1_segments0_operatingCarrier_code', 'legs1_segments1_aircraft_code',
    'legs1_segments1_arrivalTo_airport_city_iata', 'legs1_segments1_arrivalTo_airport_iata', 'legs1_segments1_departureFrom_airport_iata',
    'legs1_segments1_flightNumber', 'legs1_segments1_marketingCarrier_code', 'legs1_segments1_operatingCarrier_code',
    'legs1_segments2_aircraft_code', 'legs1_segments2_arrivalTo_airport_city_iata', 'legs1_segments2_arrivalTo_airport_iata',
    'legs1_segments2_departureFrom_airport_iata', 'legs1_segments2_flightNumber', 'legs1_segments2_marketingCarrier_code',
    'legs1_segments2_operatingCarrier_code', 'legs1_segments3_aircraft_code', 'legs1_segments3_arrivalTo_airport_city_iata',
    'legs1_segments3_arrivalTo_airport_iata', 'legs1_segments3_departureFrom_airport_iata', 'legs1_segments3_flightNumber',
    'legs1_segments3_marketingCarrier_code', 'legs1_segments3_operatingCarrier_code', 'searchRoute', 'sex'
]

def preprocess(df, is_train=True):
    # Convert durations FIRST!
    df = convert_duration_cols(df)
    # Datetime breakdown
    for col in ['legs0_departureAt', 'legs0_arrivalAt', 'requestDate']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col + "_hour"] = df[col].dt.hour
            df[col + "_dow"] = df[col].dt.dayofweek

    # Categorical encoding for LightGBM (label encoding)
    lgb_cat_cols = [
        'frequentFlyer', 'legs0_segments0_marketingCarrier_code', 'legs0_segments0_cabinClass',
        'legs0_segments0_arrivalTo_airport_city_iata', 'legs0_segments0_departureFrom_airport_iata',
        'sex', 'nationality'
    ]
    if not hasattr(preprocess, "label_maps"):
        preprocess.label_maps = {}
    for col in lgb_cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("missing").astype(str)
            if is_train:
                le_map = {v: i for i, v in enumerate(df[col].unique())}
                preprocess.label_maps[col] = le_map
                df[col + "_le"] = df[col].map(le_map).astype(int)
            else:
                le_map = preprocess.label_maps.get(col, {})
                df[col + "_le"] = df[col].map(le_map).fillna(-1).astype(int)
    # Fillna for numerics
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].median())
    df = add_features(df)
    # Ensure all CatBoost categoricals are correct string/cat dtype and no NaN!
    for col in cat_cols_cb:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('missing').astype('category')
    return df


# ============= LOAD TRAIN DATA (BATCHED) =============
train_file = pq.ParquetFile(TRAIN_PATH)
sampled_chunks = []
for i in range(train_file.num_row_groups):
    print(f"Reading row group {i+1}/{train_file.num_row_groups}...")
    chunk = train_file.read_row_groups([i]).to_pandas()
    group_sample = chunk['ranker_id'].drop_duplicates().sample(frac=SAMPLE_GROUP_FRAC, random_state=42)
    sampled_chunk = chunk[chunk['ranker_id'].isin(group_sample)]
    sampled_chunks.append(sampled_chunk)
train = pd.concat(sampled_chunks, ignore_index=True)
del sampled_chunks, chunk, sampled_chunk; gc.collect()
print("Sampled train shape:", train.shape)
print("Unique ranker_id:", train['ranker_id'].nunique())

train = preprocess(train, is_train=True)


# ============= SPLIT DATA =============
drop_cols = [
    'selected', 'Id', 'ranker_id', 'profileId', 'requestDate', 'legs0_departureAt', 'legs0_arrivalAt', 'depDate', 'reqDate'
]


# LightGBM
X_lgb = train.drop(columns=[c for c in drop_cols if c in train.columns], errors='ignore').select_dtypes(include=["int", "float", "bool"])
y = train['selected'].astype(int)
groups = train.groupby('ranker_id').size().values


# CatBoost: use all but drop only drop_cols
X_cb = train.drop(columns=[c for c in drop_cols if c in train.columns], errors='ignore')
y_cb = y.copy()
groups_cb = groups.copy()


# ============= LIGHTGBM MODEL =============
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'boosting_type': 'gbdt',
    'learning_rate': 0.07,
    'num_leaves': 51,
    'random_state': 42,
    'verbosity': -1,
}
final_train_data = lgb.Dataset(X_lgb, y, group=groups)
final_model = lgb.train(
    params,
    final_train_data,
    num_boost_round=120
)


# ============= CATBOOST MODEL =============
cb_model = CatBoostRanker(
    iterations=120,
    learning_rate=0.07,
    depth=8,
    loss_function='YetiRank',
    cat_features=cat_cols_cb,
    random_seed=42,
    verbose=20
)
cb_model.fit(X_cb, y_cb, group_id=train['ranker_id'])


# ============= PREDICT ON TEST =============
test_file = pq.ParquetFile(TEST_PATH)
test_chunks_lgb = []
test_chunks_cb = []

for i in range(test_file.num_row_groups):
    print(f"Predicting on test row group {i+1}/{test_file.num_row_groups}...")
    test_chunk = test_file.read_row_groups([i]).to_pandas()
    test_chunk = preprocess(test_chunk, is_train=False)
    # LightGBM
    X_test_lgb = test_chunk.drop(columns=[c for c in drop_cols if c in test_chunk.columns], errors='ignore')
    X_test_lgb = X_test_lgb.select_dtypes(include=["int", "float", "bool"])
    test_chunk['lgb_score'] = final_model.predict(X_test_lgb)
    # CatBoost: All categorical string/cat, no NaN
    X_test_cb = test_chunk.drop(columns=[c for c in drop_cols if c in test_chunk.columns], errors='ignore')
    for col in cat_cols_cb:
        if col in X_test_cb.columns:
            X_test_cb[col] = X_test_cb[col].astype(str).fillna('missing').astype('category')
    test_chunk['cb_score'] = cb_model.predict(X_test_cb)
    test_chunks_lgb.append(test_chunk[['Id', 'ranker_id', 'lgb_score']])
    test_chunks_cb.append(test_chunk[['Id', 'ranker_id', 'cb_score']])


# ============= ENSEMBLE AND RANK =============
test_full_lgb = pd.concat(test_chunks_lgb, ignore_index=True)
test_full_cb = pd.concat(test_chunks_cb, ignore_index=True)
test_full = test_full_lgb.merge(test_full_cb, on=['Id', 'ranker_id'])
test_full['ensemble_score'] = 0.5 * test_full['lgb_score'] + 0.5 * test_full['cb_score']

test_full = test_full.sort_values(['ranker_id', 'ensemble_score', 'Id'], ascending=[True, False, True])
test_full['selected'] = test_full.groupby('ranker_id')['ensemble_score'].rank(method='first', ascending=False).astype(int)
submission = test_full[['Id', 'ranker_id', 'selected']].sort_values('Id')


# ============= DUPLICATE CHECK =============
print("Checking for duplicate or missing ranks in groups...")
error_count = 0
for rid, group in submission.groupby('ranker_id'):
    expected = set(range(1, len(group) + 1))
    actual = set(group['selected'])
    if expected != actual:
        print(f"❌ Error in group {rid}: expected ranks {expected}, got {actual}")
        error_count += 1
if error_count == 0:
    print("✅ All groups have a valid permutation of ranks (no duplicates, no gaps).")
else:
    print(f"❌ {error_count} groups have an invalid ranking.")


# ============= SAVE SUBMISSION =============
submission.to_parquet('/kaggle/working/submission_5_1.parquet', index=False)
submission.to_csv('/kaggle/working/submission_5_1.csv', index=False)
print(submission.head())
print("Submission shape:", submission.shape)




