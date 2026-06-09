%%capture
!pip install -U xgboost
!pip install -U polars
!pip install -U optuna
!pip install -U catboost
!pip install -U lightgbm


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


df = data_raw.clone()

# More efficient duration to minutes converter
def dur_to_min(col):
    # Extract days and time parts in one pass
    days = col.str.extract(r"^(\d+)\.", 1).cast(pl.Int64).fill_null(0) * 1440
    time_str = pl.when(col.str.contains(r"^\d+\.")).then(col.str.replace(r"^\d+\.", "")).otherwise(col)
    hours = time_str.str.extract(r"^(\d+):", 1).cast(pl.Int64).fill_null(0) * 60
    minutes = time_str.str.extract(r":(\d+):", 1).cast(pl.Int64).fill_null(0)
    return (days + hours + minutes).fill_null(0)

# Process duration columns
dur_cols = ["legs0_duration", "legs1_duration"] + [f"legs{l}_segments{s}_duration" for l in (0, 1) for s in (0, 1)]
dur_exprs = [dur_to_min(pl.col(c)).alias(c) for c in dur_cols if c in df.columns]

# Apply duration transformations first
if dur_exprs:
    df = df.with_columns(dur_exprs)

# Precompute marketing carrier columns check
mc_cols = [f'legs{l}_segments{s}_marketingCarrier_code' for l in (0, 1) for s in range(4)]
mc_exists = [col for col in mc_cols if col in df.columns]

# Combine all initial transformations
df = df.with_columns([
        # Price features
        (pl.col("totalPrice") / (pl.col("taxes") + 1)).alias("price_per_tax"),
        (pl.col("taxes") / (pl.col("totalPrice") + 1)).alias("tax_rate"),
        pl.col("totalPrice").log1p().alias("log_price"),
        
        # Duration features
        (pl.col("legs0_duration").fill_null(0) + pl.col("legs1_duration").fill_null(0)).alias("total_duration"),
        pl.when(pl.col("legs1_duration").fill_null(0) > 0)
            .then(pl.col("legs0_duration") / (pl.col("legs1_duration") + 1))
            .otherwise(1.0).alias("duration_ratio"),
        
        # Trip type
        (pl.col("legs1_duration").is_null() | 
         (pl.col("legs1_duration") == 0) | 
         pl.col("legs1_segments0_departureFrom_airport_iata").is_null()).cast(pl.Int32).alias("is_one_way"),
        
        # Total segments count
        (pl.sum_horizontal(pl.col(col).is_not_null().cast(pl.UInt8) for col in mc_exists) 
         if mc_exists else pl.lit(0)).alias("l0_seg"),
        
        # FF features
        (pl.col("frequentFlyer").fill_null("").str.count_matches("/") + 
         (pl.col("frequentFlyer").fill_null("") != "").cast(pl.Int32)).alias("n_ff_programs"),
        
        # Binary features
        pl.col("corporateTariffCode").is_not_null().cast(pl.Int32).alias("has_corporate_tariff"),
        (pl.col("pricingInfo_isAccessTP") == 1).cast(pl.Int32).alias("has_access_tp"),
        
        # Baggage & fees
        (pl.col("legs0_segments0_baggageAllowance_quantity").fill_null(0) + 
         pl.col("legs1_segments0_baggageAllowance_quantity").fill_null(0)).alias("baggage_total"),
        (pl.col("miniRules0_monetaryAmount").fill_null(0) + 
         pl.col("miniRules1_monetaryAmount").fill_null(0)).alias("total_fees"),
        
        # Routes & carriers
        pl.col("searchRoute").is_in(["MOWLED/LEDMOW", "LEDMOW/MOWLED", "MOWLED", "LEDMOW", "MOWAER/AERMOW"])
            .cast(pl.Int32).alias("is_popular_route"),
        
        # Cabin
        pl.mean_horizontal(["legs0_segments0_cabinClass", "legs1_segments0_cabinClass"]).alias("avg_cabin_class"),
        (pl.col("legs0_segments0_cabinClass").fill_null(0) - 
         pl.col("legs1_segments0_cabinClass").fill_null(0)).alias("cabin_class_diff"),
])

# Segment counts - more efficient
seg_exprs = []
for leg in (0, 1):
    seg_cols = [f"legs{leg}_segments{s}_duration" for s in range(4) if f"legs{leg}_segments{s}_duration" in df.columns]
    if seg_cols:
        seg_exprs.append(
            pl.sum_horizontal(pl.col(c).is_not_null() for c in seg_cols)
                .cast(pl.Int32).alias(f"n_segments_leg{leg}")
        )
    else:
        seg_exprs.append(pl.lit(0).cast(pl.Int32).alias(f"n_segments_leg{leg}"))

# Add segment-based features
# First create segment counts
df = df.with_columns(seg_exprs)

# Then use them for derived features
df = df.with_columns([
    (pl.col("n_segments_leg0") + pl.col("n_segments_leg1")).alias("total_segments"),
    (pl.col("n_segments_leg0") == 1).cast(pl.Int32).alias("is_direct_leg0"),
    pl.when(pl.col("is_one_way") == 1).then(0)
        .otherwise((pl.col("n_segments_leg1") == 1).cast(pl.Int32)).alias("is_direct_leg1"),
])

# More derived features
df = df.with_columns([
    (pl.col("is_direct_leg0") & pl.col("is_direct_leg1")).cast(pl.Int32).alias("both_direct"),
    ((pl.col("isVip") == 1) | (pl.col("n_ff_programs") > 0)).cast(pl.Int32).alias("is_vip_freq"),
    (pl.col("baggage_total") > 0).cast(pl.Int32).alias("has_baggage"),
    (pl.col("total_fees") > 0).cast(pl.Int32).alias("has_fees"),
    (pl.col("total_fees") / (pl.col("totalPrice") + 1)).alias("fee_rate"),
    pl.col("Id").count().over("ranker_id").alias("group_size"),
])

# Add major carrier flag if column exists
if "legs0_segments0_marketingCarrier_code" in df.columns:
    df = df.with_columns(
        pl.col("legs0_segments0_marketingCarrier_code").is_in(["SU", "S7", "U6"])
            .cast(pl.Int32).alias("is_major_carrier")
    )
else:
    df = df.with_columns(pl.lit(0).alias("is_major_carrier"))

df = df.with_columns(pl.col("group_size").log1p().alias("group_size_log"))

# Time features - batch process
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

# Batch rank computations - more efficient with single pass
# First apply the columns that will be used for ranking
df = df.with_columns([
    pl.col("group_size").log1p().alias("group_size_log"),
])

# Price and duration basic ranks
rank_exprs = []
for col, alias in [("totalPrice", "price"), ("total_duration", "duration")]:
    rank_exprs.append(pl.col(col).rank().over("ranker_id").alias(f"{alias}_rank"))

# Price-specific features
price_exprs = [
    (pl.col("totalPrice").rank("average").over("ranker_id") / 
     pl.col("totalPrice").count().over("ranker_id")).alias("price_pct_rank"),
    (pl.col("totalPrice") == pl.col("totalPrice").min().over("ranker_id")).cast(pl.Int32).alias("is_cheapest"),
    ((pl.col("totalPrice") - pl.col("totalPrice").median().over("ranker_id")) / 
     (pl.col("totalPrice").std().over("ranker_id") + 1)).alias("price_from_median"),
    (pl.col("l0_seg") == pl.col("l0_seg").min().over("ranker_id")).cast(pl.Int32).alias("is_min_segments"),
]

# Apply initial ranks
df = df.with_columns(rank_exprs + price_exprs)

# Cheapest direct - more efficient
direct_cheapest = (
    df.filter(pl.col("is_direct_leg0") == 1)
    .group_by("ranker_id")
    .agg(pl.col("totalPrice").min().alias("min_direct"))
)

df = df.join(direct_cheapest, on="ranker_id", how="left").with_columns(
    ((pl.col("is_direct_leg0") == 1) & 
     (pl.col("totalPrice") == pl.col("min_direct"))).cast(pl.Int32).fill_null(0).alias("is_direct_cheapest")
).drop("min_direct")


## NEW: Advanced Feature Engineering: Group-wise Interaction Features
print("Starting advanced group-wise feature engineering...")

# Define the main carrier column, handling potential missing columns
main_carrier_col = "legs0_segments0_marketingCarrier_code"
if main_carrier_col not in df.columns:
    df = df.with_columns(pl.lit(None, dtype=pl.String).alias(main_carrier_col))

# Calculate group-wise statistics in a separate dataframe
group_stats = df.group_by("ranker_id").agg(
    pl.col("totalPrice").min().alias("min_price_group"),
    pl.col("totalPrice").max().alias("max_price_group"),
    pl.col("totalPrice").mean().alias("mean_price_group"),
    pl.col("total_duration").min().alias("min_duration_group"),
    pl.col(main_carrier_col).mode().first().alias("mode_carrier_group")
)

# Join the group stats back to the main dataframe
df = df.join(group_stats, on="ranker_id", how="left")

# --- FIX IS HERE ---

# Step 1: Create the initial set of interaction features
df = df.with_columns(
    # Price interaction features
    (pl.col("totalPrice") - pl.col("min_price_group")).alias("price_vs_min"),
    (pl.col("max_price_group") - pl.col("totalPrice")).alias("price_vs_max"),
    (pl.col("totalPrice") / pl.col("mean_price_group")).alias("price_vs_mean"),
    
    # Duration interaction features
    (pl.col("total_duration") - pl.col("min_duration_group")).alias("duration_vs_min"),
    
    # Carrier interaction features
    (pl.col(main_carrier_col) == pl.col("mode_carrier_group")).cast(pl.Int32).alias("is_mode_carrier")
)

# Step 2: Now that 'price_vs_min' exists, create the combined feature
df = df.with_columns(
    (pl.col("price_vs_min") / (pl.col("total_duration") + 1)).alias("price_vs_min_per_duration")
)

# --- END OF FIX ---

# Drop the intermediate stats columns
df = df.drop(["min_price_group", "max_price_group", "mean_price_group", "min_duration_group", "mode_carrier_group"])

print("Advanced features created successfully.")


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


## NEW: Prepare data for modeling
# Encode categorical features for compatibility with both LGBM and XGB
X_encoded = X.with_columns([(pl.col(c).rank("dense") - 1).fill_null(-1).cast(pl.Int32) for c in cat_features_final])

# Define train/val/test splits
n1 = 16487352 # split train to train and val (10%) in time
n2 = train.height

# Full datasets
X_tr, X_va, X_te = X_encoded[:n1], X_encoded[n1:n2], X_encoded[n2:]
y_tr, y_va, y_te = y[:n1], y[n1:n2], y[n2:]
groups_tr, groups_va, groups_te = groups[:n1], groups[n1:n2], groups[n2:]

# Get group sizes for the full datasets
group_sizes_tr = groups_tr.group_by('ranker_id').agg(pl.len()).sort('ranker_id')['len'].to_numpy()
group_sizes_va = groups_va.group_by('ranker_id').agg(pl.len()).sort('ranker_id')['len'].to_numpy()


## NEW: Train Stage 1 Model
print("--- Training Stage 1: LightGBM First-Pass Ranker ---")

# Create LightGBM datasets
lgb_train_s1 = lgb.Dataset(
    data=X_tr, 
    label=y_tr.to_numpy().flatten(), 
    group=group_sizes_tr,
    feature_name=feature_cols,
    free_raw_data=False
)

lgb_val_s1 = lgb.Dataset(
    data=X_va, 
    label=y_va.to_numpy().flatten(), 
    group=group_sizes_va,
    feature_name=feature_cols,
    reference=lgb_train_s1,
    free_raw_data=False
)

# Use a relatively simple and fast LightGBM configuration
s1_lgb_params = {
    'objective': 'lambdarank', 'metric': 'ndcg', 'boosting_type': 'gbdt','eval_at': [3],
    'num_leaves': 40, 'learning_rate': 0.1, 'n_estimators': 500,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1,
    'n_jobs': -1, 'random_state': RANDOM_STATE, 'label_gain': [0, 1]
}

lgb_model_s1 = lgb.train(
    s1_lgb_params,
    lgb_train_s1,
    valid_sets=[lgb_train_s1, lgb_val_s1],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)


## NEW: Filter data for Stage 2
TOP_K = 15 # Number of candidates to select for the re-ranking stage
print(f"\n--- Filtering for Stage 2: Selecting Top {TOP_K} Candidates ---")

def filter_top_k(data_X, data_y, data_groups, model, k):
    """Uses a trained model to predict and filter for the top k candidates per group."""
    # Get predictions
    preds = model.predict(data_X)
    
    # Create a temporary dataframe with predictions
    df_preds = data_groups.with_columns(
        pl.Series("preds", preds),
        pl.Series("true", data_y.to_series()),
        pl.Series("row_nr", np.arange(len(data_X))) # Original index
    )
    
    # Get the original row numbers of the top k candidates for each group
    top_k_indices = (
        df_preds
        .sort("preds", descending=True)
        .group_by("ranker_id")
        .head(k)
        .get_column("row_nr")
        .to_numpy()
    )
    
    # Filter the original dataframes using these indices
    X_rerank = data_X[top_k_indices]
    y_rerank = data_y[top_k_indices]
    groups_rerank = data_groups[top_k_indices]
    
    return X_rerank, y_rerank, groups_rerank

# Create the re-ranking training set from the original training data
X_tr_rerank, y_tr_rerank, groups_tr_rerank = filter_top_k(X_tr, y_tr, groups_tr, lgb_model_s1, TOP_K)

# Create the re-ranking validation set from the original validation data
X_va_rerank, y_va_rerank, groups_va_rerank = filter_top_k(X_va, y_va, groups_va, lgb_model_s1, TOP_K)

print(f"Original training size: {len(X_tr)}, Re-ranking training size: {len(X_tr_rerank)}")
print(f"Original validation size: {len(X_va)}, Re-ranking validation size: {len(X_va_rerank)}")


## NEW: Train Stage 2 Model
print("\n--- Training Stage 2: XGBoost Re-Ranker ---")

# Get group sizes for the new re-ranking datasets
group_sizes_tr_rerank = groups_tr_rerank.group_by('ranker_id').agg(pl.len()).sort('ranker_id')['len'].to_numpy()
group_sizes_va_rerank = groups_va_rerank.group_by('ranker_id').agg(pl.len()).sort('ranker_id')['len'].to_numpy()

# Create XGBoost DMatrix objects for the re-ranking data
dtrain_rerank = xgb.DMatrix(X_tr_rerank, label=y_tr_rerank, group=group_sizes_tr_rerank, feature_names=X_tr_rerank.columns)
dval_rerank   = xgb.DMatrix(X_va_rerank, label=y_va_rerank, group=group_sizes_va_rerank, feature_names=X_va_rerank.columns)

# Use your optimized XGBoost parameters
s2_xgb_params = {
    'objective': 'rank:pairwise', 'eval_metric': 'ndcg@3', 
    'max_depth': 8, 'min_child_weight': 14, 'subsample': 0.9, 
    'colsample_bytree': 1.0, 'lambda': 3.5330891736457763 , 
    'learning_rate': 0.0521879929228514 ,
    'seed': RANDOM_STATE, 'n_jobs': -1
}

print("\nTraining final XGBoost re-ranking model...")
xgb_model_s2 = xgb.train(
    s2_xgb_params,
    dtrain_rerank,
    num_boost_round=2000, # Can increase boosting rounds as the dataset is smaller
    evals=[(dtrain_rerank, 'train'), (dval_rerank, 'val')],
    early_stopping_rounds=150, # More patience for the final model
    verbose_eval=50
)


## NEW: Evaluate the Stage 2 model
print("\n--- Evaluating Re-ranking Model on Validation Set ---")

# Predict with the Stage 2 model on the re-ranking validation set
rerank_va_preds = xgb_model_s2.predict(dval_rerank)

# Calculate HitRate@3
rerank_hr3 = hitrate_at_3(
    y_va_rerank['selected'], 
    rerank_va_preds, 
    groups_va_rerank['ranker_id']
)

print("-" * 30)
print(f"Stage 2 XGBoost Re-ranker HitRate@3: {rerank_hr3:.4f}")
print("-" * 30)


## FINAL: Generate submission using the two-stage pipeline (Rank Fix)
print("\n--- Generating Submission File using Two-Stage Pipeline ---")

# Stage 1: Predict on the full test set to find top K candidates
print(f"Stage 1: Filtering test set to Top {TOP_K} candidates...")
s1_test_preds = lgb_model_s1.predict(X_te)

# Create dataframe with predictions and original row numbers
df_test_preds = groups_te.with_columns(
    pl.Series("preds", s1_test_preds),
    pl.Series("row_nr", np.arange(len(X_te)))
)

# Get top K indices per ranker_id
top_k_test_indices = (
    df_test_preds
    .sort("preds", descending=True)
    .group_by("ranker_id")
    .head(TOP_K)
    .get_column("row_nr")
    .to_numpy()
)

# Filter test set to top K only
X_te_rerank = X_te[top_k_test_indices]
test_rerank = test[top_k_test_indices]

# Stage 2: Re-rank candidates using XGBoost
print("Stage 2: Re-ranking candidates with XGBoost model...")
dtest_rerank = xgb.DMatrix(X_te_rerank, feature_names=X_te_rerank.columns)
s2_test_preds = xgb_model_s2.predict(dtest_rerank)

# Create ranked dataframe
submission_df = test_rerank.select(['Id', 'ranker_id']).with_columns(
    pl.Series('final_score', s2_test_preds)
).with_columns(
    pl.col('final_score')
      .rank(method='ordinal', descending=True)
      .over('ranker_id')
      .cast(pl.Int32)
      .alias('selected')
).select(['Id', 'ranker_id', 'selected'])

# Merge with full test set and fill non-top-K with rank 99
full_submission = (
    test.select(['Id', 'ranker_id'])
    .join(submission_df, on=['Id', 'ranker_id'], how='left')
    .with_columns(
        pl.col('selected').fill_null(99)
    )
)

# Save CSV
full_submission.write_csv('submission.csv')

print("\n✅ Submission file 'submission.csv' created successfully.")
print(full_submission.sort("ranker_id").head(20))


