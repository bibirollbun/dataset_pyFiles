%%capture
!pip install -U xgboost
!pip install -U polars
!pip install -U lightgbm


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import time
import xgboost as xgb


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


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

# Popularity features - efficient join
df = (
    df.join(
        train.group_by('legs0_segments0_marketingCarrier_code').agg(pl.mean('selected').alias('carrier0_pop')),
        on='legs0_segments0_marketingCarrier_code', 
        how='left'
    )
    .join(
        train.group_by('legs1_segments0_marketingCarrier_code').agg(pl.mean('selected').alias('carrier1_pop')),
        on='legs1_segments0_marketingCarrier_code', 
        how='left'
    )
    .with_columns([
        pl.col('carrier0_pop').fill_null(0.0),
        pl.col('carrier1_pop').fill_null(0.0),
    ])
)

# Final features including popularity
df = df.with_columns([
    (pl.col('carrier0_pop') * pl.col('carrier1_pop')).alias('carrier_pop_product'),
])


# Fill nulls
data = df.with_columns(
    [pl.col(c).fill_null(0) for c in df.select(pl.selectors.numeric()).columns] +
    [pl.col(c).fill_null("missing") for c in df.select(pl.selectors.string()).columns]
)


# Categorical features
cat_features = [
    'nationality', 'searchRoute', 'corporateTariffCode','route',
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
    #'miniRules0_percentage', 'miniRules1_percentage',  # >90% missing
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


from sklearn.model_selection import GroupKFold
import lightgbm as lgb  # --- NEW CODE ---

data_xgb = X.with_columns([(pl.col(c).rank("dense") - 1).fill_null(-1).cast(pl.Int32) for c in cat_features_final])

# Initialize GroupKFold (unchanged)
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(data_xgb[:train.height], y[:train.height], groups=groups["ranker_id"][:train.height]))

# Split data (unchanged)
data_xgb_tr, data_xgb_va = data_xgb[train_idx], data_xgb[val_idx]
y_tr, y_va = y[train_idx], y[val_idx]
groups_tr, groups_va = groups[train_idx], groups[val_idx]
data_xgb_te = data_xgb[train.height:]
y_te = y[train.height:]
groups_te = groups[train.height:]

# --- NEW CODE: Prepare LightGBM data ---
lgb_train = lgb.Dataset(
    data_xgb_tr.to_pandas(), 
    label=y_tr.to_numpy().ravel(),
    group=groups_tr.group_by('ranker_id').agg(pl.len())['len'].to_numpy(),
    categorical_feature=cat_features_final
)

# XGBoost DMatrices (unchanged)
group_sizes_tr = groups_tr.group_by('ranker_id').agg(pl.len())['len'].to_numpy()
group_sizes_va = groups_va.group_by('ranker_id').agg(pl.len())['len'].to_numpy()
group_sizes_te = groups_te.group_by('ranker_id').agg(pl.len())['len'].to_numpy()

dtrain = xgb.DMatrix(data_xgb_tr, label=y_tr, group=group_sizes_tr, feature_names=data_xgb.columns)
dval = xgb.DMatrix(data_xgb_va, label=y_va, group=group_sizes_va, feature_names=data_xgb.columns)
dtest = xgb.DMatrix(data_xgb_te, label=y_te, group=group_sizes_te, feature_names=data_xgb.columns)


xgb_params = {
    'objective': 'rank:pairwise',
    'eval_metric': 'ndcg@3',
    'learning_rate': 0.05,  # Increased from 0.022
    'max_depth': 8,         # Reduced from 14 to prevent overfitting
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'gamma': 1.0,
    'lambda': 5.0,
    'alpha': 0.5,
    'seed': RANDOM_STATE,
    'n_jobs': -1,
    'early_stopping_rounds': 50,
    'max_bin': 256,         # Added for better numerical stability
    'tree_method': 'hist'   # Faster training
}

print("Training XGBoost...")
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=1250,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=100,
    verbose_eval=50
)


lgb_params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'eval_at': [3],
    'seed': RANDOM_STATE,
    'verbose': -1,
    'learning_rate': 0.05,  # Increased from 0.03
    'num_leaves': 31,       # Reduced from 60
    'max_depth': 7,         # Reduced from 10
    'min_data_in_leaf': 20,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 1.0,
    'lambda_l2': 1.0
}

print("\nTraining LightGBM...")
lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)]
)


# First generate validation predictions
xgb_va_preds = xgb_model.predict(dval)
lgb_va_preds = lgb_model.predict(data_xgb_va.to_pandas())

# Now we can test different weight combinations
best_score = 0
best_weights = (0, 0)

# Test different weight combinations
for xgb_weight in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    lgb_weight = 1 - xgb_weight
    blend_va_preds = xgb_weight * xgb_va_preds + lgb_weight * lgb_va_preds
    current_score = hitrate_at_3(y_va, blend_va_preds, groups_va)
    
    if current_score > best_score:
        best_score = current_score
        best_weights = (xgb_weight, lgb_weight)

print(f"Best weights: XGBoost {best_weights[0]}, LightGBM {best_weights[1]}")
print(f"Best validation score: {best_score:.5f}")


import matplotlib.pyplot as plt

# Get feature importance
importance = xgb_model.get_score(importance_type='gain')
importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

# Print top 10 features
print("Top 10 features by gain:")
for feat, score in importance[:10]:
    print(f"{feat}: {score}")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar([x[0] for x in importance[:20]], [x[1] for x in importance[:20]])
plt.xticks(rotation=45, ha='right')
plt.title("Top 20 Feature Importance (Gain)")
plt.tight_layout()
plt.show()



# Use best weights for final predictions
xgb_test_preds = xgb_model.predict(dtest)
lgb_test_preds = lgb_model.predict(data_xgb_te.to_pandas())
blend_test_preds = best_weights[0] * xgb_test_preds + best_weights[1] * lgb_test_preds

def re_rank(test: pl.DataFrame, submission_xgb: pl.DataFrame, penalty_factor=0.1):
    COLS_TO_COMPARE = [
        "legs0_departureAt",
        "legs0_arrivalAt",
        "legs1_departureAt",
        "legs1_arrivalAt",
        "legs0_segments0_flightNumber",
        "legs1_segments0_flightNumber",
        "legs0_segments0_aircraft_code",
        "legs1_segments0_aircraft_code",
        "legs0_segments0_departureFrom_airport_iata",
        "legs1_segments0_departureFrom_airport_iata",
    ]

    test = test.with_columns(
        [pl.col(c).cast(str).fill_null("NULL") for c in COLS_TO_COMPARE]
    )

    df = submission_xgb.join(test, on=["Id", "ranker_id"], how="left")

    df = df.with_columns(
        (
            pl.col("legs0_departureAt")
            + "_"
            + pl.col("legs0_arrivalAt")
            + "_"
            + pl.col("legs1_departureAt")
            + "_"
            + pl.col("legs1_arrivalAt")
            + "_"
            + pl.col("legs0_segments0_flightNumber")
            + "_"
            + pl.col("legs1_segments0_flightNumber")
        ).alias("flight_hash")
    )

    df = df.with_columns(
        pl.max("pred_score")
        .over(["ranker_id", "flight_hash"])
        .alias("max_score_same_flight")
    )

    df = df.with_columns(
        (
            pl.col("pred_score")
            - penalty_factor * (pl.col("max_score_same_flight") - pl.col("pred_score"))
        ).alias("reorder_score")
    )

    df = df.with_columns(
        pl.col("reorder_score")
        .rank(method="ordinal", descending=True)
        .over("ranker_id")
        .cast(pl.Int32)
        .alias("new_selected")
    )

    return df.select(["Id", "ranker_id", "new_selected", "pred_score", "reorder_score"])
# --- Prepare Submission (unchanged except for pred_score) ---
submission_xgb = (
    test.select(['Id', 'ranker_id'])
    .with_columns(pl.Series('pred_score', blend_test_preds))  # <-- Use blended scores
    .with_columns(
        pl.col('pred_score')
        .rank(method='ordinal', descending=True)
        .over('ranker_id')
        .cast(pl.Int32)
        .alias('selected')
    )
    .select(['Id', 'ranker_id', 'selected', 'pred_score'])
)

# --- Your Existing Re-Ranking Logic (unchanged) ---
top = re_rank(test, submission_xgb)

final_submission = (
    submission_xgb.join(top, on=["Id", "ranker_id"], how="left")
    .with_columns(
        pl.when(pl.col("new_selected").is_not_null())
        .then(pl.col("new_selected"))
        .otherwise(pl.col("selected"))
        .alias("selected")
    )
    .select(["Id", "ranker_id", "selected"])
)

final_submission.write_csv('submission.csv')




# Add this to verify the file exists
import os
if os.path.exists('/kaggle/working/submission.csv'):
    print("Submission file created successfully at /kaggle/working/submission.csv")
else:
    print("Error: Submission file not found!")

