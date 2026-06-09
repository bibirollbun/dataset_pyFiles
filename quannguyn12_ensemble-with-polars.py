%%capture
!pip install -U xgboost
!pip install -U polars
!pip install -U optuna
!pip install -U catboost
!pip install -U lightgbm
!pip install -U gensim



import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import time
import xgboost as xgb
import catboost
import lightgbm as lgb
import optuna

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# Load data
train = pl.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet').drop('__index_level_0__')
test = pl.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet').drop('__index_level_0__').with_columns(pl.lit(0, dtype=pl.Int64).alias("selected"))

data_raw = pl.concat((train, test))


# Embedding Feature Creation

from gensim.models import Word2Vec
import polars as pl
import numpy as np
import gc

print("--- Step 4: Creating Embedding Features ---")

if 'data_raw' not in globals() or data_raw.is_empty():
    raise NameError("`data_raw` DataFrame not found. Please run the data loading cell first.")

# 1. Prepare "sentences" for training Word2Vec models
print("  1. Preparing corpus from flight data...")

# Build sentences, ensuring no None values are passed to Word2Vec
# We collect the lists, then process them in Python to handle potential None values inside lists
sentences_df = (
    data_raw
    .group_by("ranker_id")
    .agg([
        pl.col('legs0_segments0_marketingCarrier_code'),
        pl.col('legs0_segments0_departureFrom_airport_iata'),
        pl.col('legs0_segments0_arrivalTo_airport_iata')
    ])
)

corpus = []
for row in sentences_df.iter_rows():
    # row is a tuple, e.g., (ranker_id, [carriers], [deps], [arrs])
    # Flatten the lists and remove any None values
    sentence = [item for sublist in row[1:] for item in sublist if item is not None]
    if sentence:
        corpus.append(sentence)

print(f"  Created {len(corpus)} non-empty sentences for Word2Vec training.")

# 2. Train Word2Vec models
# Ensure there is a corpus to train on
if not corpus:
    raise ValueError("Corpus is empty. Cannot train Word2Vec model.")
    
W2V_PARAMS = {"vector_size": 16, "window": 10, "min_count": 3, "workers": -1, "seed": RANDOM_STATE}

print(f"  2. Training Word2Vec model with params: {W2V_PARAMS}")
w2v_model = Word2Vec(corpus, **W2V_PARAMS)

# Create a dictionary for fast lookups
if not w2v_model.wv.index_to_key:
    print("Warning: Word2Vec model vocabulary is empty. No embeddings will be generated.")
    w2v_dict = {}
else:
    w2v_dict = {word: w2v_model.wv[word] for word in w2v_model.wv.index_to_key}

vector_size = w2v_model.vector_size
default_vector = np.zeros(vector_size, dtype=np.float32)

# 3. Create embedding columns in the main DataFrame
embedding_cols_to_create = {
    'carrier_emb': 'legs0_segments0_marketingCarrier_code',
    'dep_airport_emb': 'legs0_segments0_departureFrom_airport_iata',
    'arr_airport_emb': 'legs0_segments0_arrivalTo_airport_iata'
}

# Use a temporary variable to avoid modifying `data` in a loop
data_with_embeddings = data_raw.clone()

print("  3. Applying embeddings to DataFrame...")
for new_col_prefix, source_col in embedding_cols_to_create.items():
    if source_col in data_with_embeddings.columns:
        # Create a list of vectors first
        embedding_vectors = [w2v_dict.get(key, default_vector) for key in data_with_embeddings[source_col].to_list()]
        
        # Convert list of arrays to a 2D numpy array
        embedding_array = np.array(embedding_vectors)
        
        # Create new columns from the numpy array
        for i in range(vector_size):
            data_with_embeddings = data_with_embeddings.with_columns(
                pl.Series(f"{new_col_prefix}_{i}", embedding_array[:, i])
            )

# Replace the original data variable with the new one
data = data_with_embeddings
print(f"  Embedding features created. New DataFrame shape: {data.shape}")

del data_raw, data_with_embeddings, corpus, w2v_model, w2v_dict
gc.collect()


def hitrate_at_3(y_true, y_pred, groups):
    df = pl.DataFrame({
        'group': groups,
        'pred': y_pred,
        'true': y_true
    })
    
    return (
        df.filter(pl.col("group").count().over("group") > 10)
        .sort(["group", "pred"], descending=[False, True])
        .group_by("group", maintain_order=True)
        .head(3)
        .group_by("group")
        .agg(pl.col("true").max())
        .select(pl.col("true").mean())
        .item()
    )


# Cell 7: Feature Engineering

# Make a clone to work on
df = data.clone()

# More efficient duration to minutes converter
def dur_to_min(col):
    days = col.str.extract(r"^(\d+)\.", 1).cast(pl.Int64).fill_null(0) * 1440
    time_str = pl.when(col.str.contains(r"^\d+\.")).then(col.str.replace(r"^\d+\.", "")).otherwise(col)
    hours = time_str.str.extract(r"^(\d+):", 1).cast(pl.Int64).fill_null(0) * 60
    minutes = time_str.str.extract(r":(\d+):", 1).cast(pl.Int64).fill_null(0)
    return (days + hours + minutes).fill_null(0)

# Process duration columns
dur_cols = ["legs0_duration", "legs1_duration"] + [f"legs{l}_segments{s}_duration" for l in (0, 1) for s in (0, 1)]
dur_exprs = [dur_to_min(pl.col(c)).alias(c) for c in dur_cols if c in df.columns]
if dur_exprs:
    df = df.with_columns(dur_exprs)

# Precompute marketing carrier columns check
mc_cols = [f'legs{l}_segments{s}_marketingCarrier_code' for l in (0, 1) for s in range(4)]
mc_exists = [col for col in mc_cols if col in df.columns]

# Combine all initial transformations
df = df.with_columns([
    (pl.col("totalPrice") / (pl.col("taxes") + 1)).alias("price_per_tax"),
    (pl.col("taxes") / (pl.col("totalPrice") + 1)).alias("tax_rate"),
    pl.col("totalPrice").log1p().alias("log_price"),
    (pl.col("legs0_duration").fill_null(0) + pl.col("legs1_duration").fill_null(0)).alias("total_duration"),
    pl.when(pl.col("legs1_duration").fill_null(0) > 0).then(pl.col("legs0_duration") / (pl.col("legs1_duration") + 1)).otherwise(1.0).alias("duration_ratio"),
    (pl.col("legs1_duration").is_null() | (pl.col("legs1_duration") == 0) | pl.col("legs1_segments0_departureFrom_airport_iata").is_null()).cast(pl.Int32).alias("is_one_way"),
    (pl.sum_horizontal(pl.col(col).is_not_null().cast(pl.UInt8) for col in mc_exists) if mc_exists else pl.lit(0)).alias("l0_seg"),
    (pl.col("frequentFlyer").fill_null("").str.count_matches("/") + (pl.col("frequentFlyer").fill_null("") != "").cast(pl.Int32)).alias("n_ff_programs"),
    pl.col("corporateTariffCode").is_not_null().cast(pl.Int32).alias("has_corporate_tariff"),
    (pl.col("pricingInfo_isAccessTP") == 1).cast(pl.Int32).alias("has_access_tp"),
    (pl.col("legs0_segments0_baggageAllowance_quantity").fill_null(0) + pl.col("legs1_segments0_baggageAllowance_quantity").fill_null(0)).alias("baggage_total"),
    (pl.col("miniRules0_monetaryAmount").fill_null(0) + pl.col("miniRules1_monetaryAmount").fill_null(0)).alias("total_fees"),
    pl.col("searchRoute").is_in(["MOWLED/LEDMOW", "LEDMOW/MOWLED", "MOWLED", "LEDMOW", "MOWAER/AERMOW"]).cast(pl.Int32).alias("is_popular_route"),
    pl.mean_horizontal(["legs0_segments0_cabinClass", "legs1_segments0_cabinClass"]).alias("avg_cabin_class"),
    (pl.col("legs0_segments0_cabinClass").fill_null(0) - pl.col("legs1_segments0_cabinClass").fill_null(0)).alias("cabin_class_diff"),
])

# Segment counts
seg_exprs = []
for leg in (0, 1):
    seg_cols = [f"legs{leg}_segments{s}_duration" for s in range(4) if f"legs{leg}_segments{s}_duration" in df.columns]
    if seg_cols:
        seg_exprs.append(pl.sum_horizontal(pl.col(c).is_not_null() for c in seg_cols).cast(pl.Int32).alias(f"n_segments_leg{leg}"))
    else:
        seg_exprs.append(pl.lit(0).cast(pl.Int32).alias(f"n_segments_leg{leg}"))
df = df.with_columns(seg_exprs)

# Derived features
df = df.with_columns([
    (pl.col("n_segments_leg0") + pl.col("n_segments_leg1")).alias("total_segments"),
    (pl.col("n_segments_leg0") == 1).cast(pl.Int32).alias("is_direct_leg0"),
    pl.when(pl.col("is_one_way") == 1).then(0).otherwise((pl.col("n_segments_leg1") == 1).cast(pl.Int32)).alias("is_direct_leg1"),
]).with_columns([
    (pl.col("is_direct_leg0") & pl.col("is_direct_leg1")).cast(pl.Int32).alias("both_direct"),
    ((pl.col("isVip") == 1) | (pl.col("n_ff_programs") > 0)).cast(pl.Int32).alias("is_vip_freq"),
    (pl.col("baggage_total") > 0).cast(pl.Int32).alias("has_baggage"),
    (pl.col("total_fees") > 0).cast(pl.Int32).alias("has_fees"),
    (pl.col("total_fees") / (pl.col("totalPrice") + 1)).alias("fee_rate"),
    pl.col("Id").count().over("ranker_id").alias("group_size"),
])

if "legs0_segments0_marketingCarrier_code" in df.columns:
    df = df.with_columns(pl.col("legs0_segments0_marketingCarrier_code").is_in(["SU", "S7", "U6"]).cast(pl.Int32).alias("is_major_carrier"))
else:
    df = df.with_columns(pl.lit(0).alias("is_major_carrier"))

df = df.with_columns(pl.col("group_size").log1p().alias("group_size_log"))

# Time features
time_exprs = []
for col in ("legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt"):
    if col in df.columns:
        dt = pl.col(col).str.to_datetime(strict=False)
        h = dt.dt.hour().fill_null(12)
        time_exprs.extend([
            h.alias(f"{col}_hour"),
            dt.dt.weekday().fill_null(0).alias(f"{col}_weekday"),
            (((h >= 6) & (h <= 9)) | ((h >= 17) & (h <= 20))).cast(pl.Int32).alias(f"{col}_business_time")
        ])
if time_exprs:
    df = df.with_columns(time_exprs)

# Group-wise features
df = df.with_columns(
    pl.col("totalPrice").rank().over("ranker_id").alias("price_rank"),
    pl.col("total_duration").rank().over("ranker_id").alias("duration_rank"),
    (pl.col("totalPrice").rank("average").over("ranker_id") / pl.col("totalPrice").count().over("ranker_id")).alias("price_pct_rank"),
    (pl.col("totalPrice") == pl.col("totalPrice").min().over("ranker_id")).cast(pl.Int32).alias("is_cheapest"),
    ((pl.col("totalPrice") - pl.col("totalPrice").median().over("ranker_id")) / (pl.col("totalPrice").std().over("ranker_id") + 1)).alias("price_from_median"),
    (pl.col("l0_seg") == pl.col("l0_seg").min().over("ranker_id")).cast(pl.Int32).alias("is_min_segments"),
)

# Initialize layover columns with a default value (0)
df = df.with_columns(
    pl.lit(0, dtype=pl.Int64).alias('leg0_layover_minutes'),
    pl.lit(0, dtype=pl.Int64).alias('leg1_layover_minutes')
)

# Convert necessary time columns
time_cols_to_convert = ['legs0_segments0_arrivalAt', 'legs0_segments1_departureAt', 'legs1_segments0_arrivalAt', 'legs1_segments1_departureAt']
for col in time_cols_to_convert:
    if col in df.columns:
        df = df.with_columns(pl.col(col).str.to_datetime(strict=False, cache=False))

# Calculate Layover Leg 0 if possible
if 'legs0_segments0_arrivalAt' in df.columns and 'legs0_segments1_departureAt' in df.columns:
    df = df.with_columns(
        ((pl.col('legs0_segments1_departureAt') - pl.col('legs0_segments0_arrivalAt'))
         .dt.total_minutes()).alias('leg0_layover_minutes')
    )
# Calculate Layover Leg 1 if possible
if 'legs1_segments0_arrivalAt' in df.columns and 'legs1_segments1_departureAt' in df.columns:
    df = df.with_columns(
        ((pl.col('legs1_segments1_departureAt') - pl.col('legs1_segments0_arrivalAt'))
         .dt.total_minutes()).alias('leg1_layover_minutes')
    )
# Now, safely calculate total_layover_minutes
df = df.with_columns(
    (pl.col('leg0_layover_minutes').fill_null(0) + pl.col('leg1_layover_minutes').fill_null(0)).alias('total_layover_minutes')
)

# 2. Các feature so sánh với lựa chọn tốt nhất trong nhóm
df = df.with_columns([
    (pl.col('totalPrice') / (pl.col('totalPrice').min().over('ranker_id') + 1e-6)).alias('price_ratio_vs_min'),
    (pl.col('total_duration') / (pl.col('total_duration').min().over('ranker_id') + 1e-6)).alias('duration_ratio_vs_min'),
    (pl.col('total_segments') - pl.col('total_segments').min().over('ranker_id')).alias('segments_diff_vs_min'),
])

# Cheapest direct - more efficient
direct_cheapest = (
    df.filter(pl.col("is_direct_leg0") == 1)
    .group_by("ranker_id")
    .agg(pl.col("totalPrice").min().alias("min_direct"))
)
df = df.join(direct_cheapest, on="ranker_id", how="left").with_columns(
    ((pl.col("is_direct_leg0") == 1) & 
     (pl.col("totalPrice") <= pl.col("min_direct"))).cast(pl.Int32).fill_null(0).alias("is_direct_cheapest")
).drop("min_direct")


# Fill nulls
data = df.with_columns(
    [pl.col(c).fill_null(0) for c in df.select(pl.selectors.numeric()).columns] +
    [pl.col(c).fill_null("missing") for c in df.select(pl.selectors.string()).columns]
)


# Categorical features
cat_features = [
    'nationality', 'searchRoute', 'corporateTariffCode',
    'bySelf', 'sex', 'companyID',
    # Leg 0 segments 0-1
    'legs0_segments0_aircraft_code', 'legs0_segments0_arrivalTo_airport_city_iata',
    'legs0_segments0_arrivalTo_airport_iata', 'legs0_segments0_departureFrom_airport_iata',
    'legs0_segments0_marketingCarrier_code', 'legs0_segments0_operatingCarrier_code',
    'legs0_segments0_flightNumber',
    'legs0_segments1_aircraft_code', 'legs0_segments1_arrivalTo_airport_city_iata',
    'legs0_segments1_arrivalTo_airport_iata', 'legs0_segments1_departureFrom_airport_iata',
    'legs0_segments1_marketingCarrier_code', 'legs0_segments1_operatingCarrier_code',
    'legs0_segments1_flightNumber',
    # Leg 1 segments 0-1
    'legs1_segments0_aircraft_code', 'legs1_segments0_arrivalTo_airport_city_iata',
    'legs1_segments0_arrivalTo_airport_iata', 'legs1_segments0_departureFrom_airport_iata',
    'legs1_segments0_marketingCarrier_code', 'legs1_segments0_operatingCarrier_code',
    'legs1_segments0_flightNumber',
    'legs1_segments1_aircraft_code', 'legs1_segments1_arrivalTo_airport_city_iata',
    'legs1_segments1_arrivalTo_airport_iata', 'legs1_segments1_departureFrom_airport_iata',
    'legs1_segments1_marketingCarrier_code', 'legs1_segments1_operatingCarrier_code',
    'legs1_segments1_flightNumber',
]

# Columns to exclude (uninformative or problematic)
exclude_cols = [
    'Id', 'ranker_id', 'selected', 'profileId', 'requestDate',
    'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt',
    'miniRules0_percentage', 'miniRules1_percentage',  # >90% missing
    'frequentFlyer',  # Already processed
    # Exclude constant columns
    'pricingInfo_passengerCount'
]


# Exclude segment 2-3 columns (>98% missing)
for leg in [0, 1]:
    for seg in [2, 3]:
        for suffix in ['aircraft_code', 'arrivalTo_airport_city_iata', 'arrivalTo_airport_iata',
                      'baggageAllowance_quantity', 'baggageAllowance_weightMeasurementType',
                      'cabinClass', 'departureFrom_airport_iata', 'duration', 'flightNumber',
                      'marketingCarrier_code', 'operatingCarrier_code', 'seatsAvailable']:
            exclude_cols.append(f'legs{leg}_segments{seg}_{suffix}')

feature_cols = [col for col in data.columns if col not in exclude_cols]
cat_features_final = [col for col in cat_features if col in feature_cols]

print(f"Using {len(feature_cols)} features ({len(cat_features_final)} categorical)")

X = data.select(feature_cols)
y = data.select('selected')
groups = data.select('ranker_id')


data_xgb = X.with_columns([(pl.col(c).rank("dense") - 1).fill_null(-1).cast(pl.Int32) for c in cat_features_final])

n1 = 16487352 # split train to train and val (10%) in time
n2 = train.height
data_xgb_tr, data_xgb_va, data_xgb_te = data_xgb[:n1], data_xgb[n1:n2], data_xgb[n2:]
y_tr, y_va, y_te = y[:n1], y[n1:n2], y[n2:]
groups_tr, groups_va, groups_te = groups[:n1], groups[n1:n2], groups[n2:]

group_sizes_tr = groups_tr.group_by('ranker_id').agg(pl.len()).sort('ranker_id')['len'].to_numpy()
group_sizes_va = groups_va.group_by('ranker_id').agg(pl.len()).sort('ranker_id')['len'].to_numpy()
group_sizes_te = groups_te.group_by('ranker_id').agg(pl.len()).sort('ranker_id')['len'].to_numpy()
dtrain = xgb.DMatrix(data_xgb_tr, label=y_tr, group=group_sizes_tr, feature_names=data_xgb.columns)
dval   = xgb.DMatrix(data_xgb_va, label=y_va, group=group_sizes_va, feature_names=data_xgb.columns)
dtest  = xgb.DMatrix(data_xgb_te, label=y_te, group=group_sizes_te, feature_names=data_xgb.columns)


# CODE CELL
final_xgb_params = {'objective': 'rank:pairwise', 'eval_metric': 'ndcg@3', 
                    'max_depth': 8, 'min_child_weight': 14, 'subsample': 0.9, 
                    'colsample_bytree': 1.0, 'lambda': 3.5330891736457763 , 
                    'learning_rate': 0.0521879929228514 ,
                    'seed': RANDOM_STATE, 'n_jobs': -1}
final_xgb_params.update(final_xgb_params)

print("\nTraining final XGBoost model with optimized parameters...")
xgb_model = xgb.train(
    final_xgb_params, dtrain,
    num_boost_round=1500,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=100,
    verbose_eval=50
)


# CELL 11: LightGBM Model (FIXED ComputeError)

print("Creating LightGBM Datasets with NATIVE categorical support...")

# 1. Dữ liệu đã được chuẩn bị sẵn (X, y, groups)
# X đã chứa các cột categorical dưới dạng số (đã được LabelEncoded/cat.codes)

# 2. Split dữ liệu
# n1, n2, y_tr, y_va, group_sizes_tr, group_sizes_va đã có từ cell XGBoost
# Chúng ta không cần tạo lại X_lgb, mà sẽ dùng trực tiếp X đã có
X_tr, X_va, X_te = X[:n1], X[n1:n2], X[n2:]

# 3. Tạo LightGBM Datasets
# LightGBM có thể nhận trực tiếp Pandas DataFrame.
# Chúng ta chỉ cần chỉ định cột nào là categorical bằng tên.
lgb_train = lgb.Dataset(
    data=X_tr.to_pandas(), 
    label=y_tr.to_pandas(), 
    group=group_sizes_tr,
    feature_name=X.columns,
    categorical_feature=cat_features_final, # <-- Báo cho LGBM biết đây là cột categorical
    free_raw_data=False
)

lgb_val = lgb.Dataset(
    data=X_va.to_pandas(), 
    label=y_va.to_pandas(), 
    group=group_sizes_va,
    feature_name=X.columns,
    categorical_feature=cat_features_final, 
    reference=lgb_train,
    free_raw_data=False
)

print("LightGBM Datasets created successfully.")



# --- Train the model ---
final_lgb_params = {
    'objective': 'lambdarank', 'metric': 'ndcg', 'boosting_type': 'gbdt','eval_at': [3],
    'num_leaves': 137, 'learning_rate': 0.19236092380700556, 'min_child_samples': 69, 
    'lambda_l1': 0.001786334561662628, 'lambda_l2': 7.881799636447006, 
    'feature_fraction': 0.6015465928218717, 'bagging_fraction': 0.8535794374747682, 
    'bagging_freq': 7, 'n_jobs': -1, 'random_state': RANDOM_STATE, 'label_gain': [0, 1]
}

print("\nTraining final LightGBM model with optimized parameters...")
lgb_model = lgb.train(
    final_lgb_params,
    lgb_train,
    num_boost_round=1500,
    valid_sets=[lgb_train, lgb_val],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)]
)


# CELL 12: Training LightGBM DART Model

print("\n--- Training LightGBM DART Model ---")

dart_params = {
    'objective': 'lambdarank', 'metric': 'ndcg', 'eval_at': [3],
    'boosting_type': 'dart', 
    'n_estimators': 1500,     
    'learning_rate': 0.04,
    'num_leaves': 50,
    'drop_rate': 0.1,        
    'skip_drop': 0.5,        
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
    'label_gain': [0, 1]
}

# Reuse the lgb.Dataset objects created in the previous cell
lgb_model_dart = lgb.train(
    dart_params,
    lgb_train, 
    num_boost_round=dart_params['n_estimators'], 
    valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)],
    # Pass categorical_feature again as it's not stored in the Dataset object itself
    categorical_feature=cat_features_final 
)


# CELL 13 : Blending and Final Evaluation

print("Evaluating all models on the validation set...")

# 1. Get predictions from all three models on the validation set
X_va_pd = X_va.to_pandas()
X_va_num_pd = X_va_num.to_pandas()

xgb_va_preds = xgb_model.predict(xgb.DMatrix(X_va_num_pd))
lgb_va_preds = lgb_model.predict(X_va_pd)
lgb_dart_va_preds = lgb_model_dart.predict(X_va_pd)

# 2. Create a Polars DataFrame for efficient rank calculation
val_scores_df = pl.DataFrame({
    "group": groups_va['ranker_id'],
    "true": y_va['selected'],
    "xgb_score": xgb_va_preds,
    "lgb_gbdt_score": lgb_va_preds,
    "lgb_dart_score": lgb_dart_va_preds
})

# 3. Calculate ranks for each model
val_scores_df = val_scores_df.with_columns(
    pl.col("xgb_score").rank(method="average", descending=True).over("group").alias("xgb_rank"),
    pl.col("lgb_gbdt_score").rank(method="average", descending=True).over("group").alias("lgb_gbdt_rank"),
    pl.col("lgb_dart_score").rank(method="average", descending=True).over("group").alias("lgb_dart_rank")
)

# 4. Calculate smart blending weights based on individual validation performance
xgb_hr3 = hitrate_at_3(y_va['selected'], xgb_va_preds, groups_va['ranker_id'])
lgb_hr3 = hitrate_at_3(y_va['selected'], lgb_va_preds, groups_va['ranker_id'])
lgb_dart_hr3 = hitrate_at_3(y_va['selected'], lgb_dart_va_preds, groups_va['ranker_id'])

total_hr3 = xgb_hr3 + lgb_hr3 + lgb_dart_hr3 + 1e-9 
w_xgb = xgb_hr3 / total_hr3
w_lgb = lgb_hr3 / total_hr3
w_dart = lgb_dart_hr3 / total_hr3

# 5. Calculate the blended score using the new weights
val_scores_df = val_scores_df.with_columns(
    (w_xgb * pl.col("xgb_rank") + w_lgb * pl.col("lgb_gbdt_rank") + w_dart * pl.col("lgb_dart_rank")).alias("blend_score")
)

# 6. Calculate the HitRate@3 of the smart blend
blend_hr3 = hitrate_at_3(val_scores_df['true'], -val_scores_df['blend_score'], val_scores_df['group'])

# 7. Print all results
print("-" * 40)
print(f"XGBoost HitRate@3:     {xgb_hr3:.5f}")
print(f"LGBM GBDT HitRate@3:   {lgb_hr3:.5f}")
print(f"LGBM DART HitRate@3:   {lgb_dart_hr3:.5f}")
print("-" * 40)
print(f"Blending Weights (XGB/LGB/DART): {w_xgb:.3f} / {w_lgb:.3f} / {w_dart:.3f}")
print(f"SMART BLEND HitRate@3: {blend_hr3:.5f}  <-- Final Validation Score")
print("-" * 40)


# CELL 14 Submission Generation

print("Generating predictions for the test set with all three models...")

# 1. Predict with all models on the test set
dtest_xgb = xgb.DMatrix(X_te_num)
X_te_pd = X_te.to_pandas()

xgb_test_preds = xgb_model.predict(dtest_xgb)
lgb_gbdt_test_preds = lgb_model.predict(X_te_pd) 
lgb_dart_test_preds = lgb_model_dart.predict(X_te_pd)

# 2. Use the smart blending weights calculated from the validation set
# The weights w_xgb, w_lgb, w_dart must exist from the previous cell
try:
    print(f"\nUsing calculated blending weights:")
    print(f"XGB: {w_xgb:.3f}, LGBM GBDT: {w_lgb:.3f}, LGBM DART: {w_dart:.3f}")
except NameError:
    print("Warning: Validation weights not found. Falling back to default weights.")
    w_xgb, w_lgb, w_dart = 0.5, 0.3, 0.2 

# 3. Create submission file using the calculated weights
submission_df = test.select(['Id', 'ranker_id'])[n2:].with_columns(
    pl.Series('xgb_score', xgb_test_preds),
    pl.Series('lgb_gbdt_score', lgb_gbdt_test_preds),
    pl.Series('lgb_dart_score', lgb_dart_test_preds)
).with_columns(
    xgb_rank=pl.col("xgb_score").rank(method="average", descending=True).over("ranker_id"),
    lgb_gbdt_rank=pl.col("lgb_gbdt_score").rank(method="average", descending=True).over("ranker_id"),
    lgb_dart_rank=pl.col("lgb_dart_score").rank(method="average", descending=True).over("ranker_id")
).with_columns(
    blend_score=(w_xgb * pl.col("xgb_rank") + w_lgb * pl.col("lgb_gbdt_rank") + w_dart * pl.col("lgb_dart_rank"))
).with_columns(
    selected=pl.col('blend_score').rank(method='ordinal', descending=False).over('ranker_id').cast(pl.Int32)
).select(['Id', 'ranker_id', 'selected'])

submission_df.write_csv('submission.csv')

print("\nSubmission file 'submission.csv' created successfully.")
print(submission_df.head())

