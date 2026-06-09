%%capture
!pip install -U xgboost
!pip install -U polars


import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import time
import xgboost as xgb

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
        # (pl.col("totalPrice") / (pl.col("taxes") + 1)).alias("price_per_tax"),
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
        # (pl.col("legs0_segments0_baggageAllowance_quantity").fill_null(0) + 
        #  pl.col("legs1_segments0_baggageAllowance_quantity").fill_null(0)).alias("baggage_total"),
        # (pl.col("miniRules0_monetaryAmount").fill_null(0) + 
        #  pl.col("miniRules1_monetaryAmount").fill_null(0)).alias("total_fees"),

        (
            (pl.col("miniRules0_monetaryAmount") == 0)
            & (pl.col("miniRules0_statusInfos") == 1)
        )
        .cast(pl.Int8)
        .alias("free_cancel"),
        (
            (pl.col("miniRules1_monetaryAmount") == 0)
            & (pl.col("miniRules1_statusInfos") == 1)
        )
        .cast(pl.Int8)
        .alias("free_exchange"),
    
        # Routes & carriers
        pl.col("searchRoute").is_in(["MOWLED/LEDMOW", "LEDMOW/MOWLED", "MOWLED", "LEDMOW"])
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
    # (pl.col("baggage_total") > 0).cast(pl.Int32).alias("has_baggage"),
    # (pl.col("total_fees") > 0).cast(pl.Int32).alias("has_fees"),
    # (pl.col("total_fees") / (pl.col("totalPrice") + 1)).alias("fee_rate"),
    pl.col("Id").count().over("ranker_id").alias("group_size"),
])

# Add major carrier flag if column exists
if "legs0_segments0_marketingCarrier_code" in df.columns:
    df = df.with_columns(
        pl.col("legs0_segments0_marketingCarrier_code").is_in(["SU", "S7"])
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

for leg in [0, 1]:
    for seg in [0, 1]:
        if seg == 0:
            suffixes = [
                "seatsAvailable",
            ]
        else:
            suffixes = [
                "cabinClass",
                "seatsAvailable",
                "baggageAllowance_quantity",
                "baggageAllowance_weightMeasurementType",
                "aircraft_code",
                "arrivalTo_airport_city_iata",
                "arrivalTo_airport_iata",
                "departureFrom_airport_iata",
                "flightNumber",
                "marketingCarrier_code",
                "operatingCarrier_code",
            ]
        for suffix in suffixes:
            exclude_cols.append(f"legs{leg}_segments{seg}_{suffix}")


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


data_xgb = X.with_columns([(pl.col(c).rank("dense") - 1).fill_null(-1).cast(pl.Int16) for c in cat_features_final])

n1 = 16487352 # split train to train and val (10%) in time
n2 = train.height
data_xgb_tr, data_xgb_va, data_xgb_te = data_xgb[:n2], data_xgb[n1:n2], data_xgb[n2:]
y_tr, y_va, y_te = y[:n2], y[n1:n2], y[n2:]
groups_tr, groups_va, groups_te = groups[:n2], groups[n1:n2], groups[n2:]

group_sizes_tr = groups_tr.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
group_sizes_va = groups_va.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
group_sizes_te = groups_te.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
dtrain = xgb.DMatrix(data_xgb_tr, label=y_tr, group=group_sizes_tr, feature_names=data_xgb.columns)
dval   = xgb.DMatrix(data_xgb_va, label=y_va, group=group_sizes_va, feature_names=data_xgb.columns)
dtest  = xgb.DMatrix(data_xgb_te, label=y_te, group=group_sizes_te, feature_names=data_xgb.columns)


# XGBoost parameters
xgb_params = {
    'objective': 'rank:pairwise',
    'eval_metric': 'ndcg@3',
    "learning_rate": 0.022641389657079056,
    "max_depth": 14,
    "min_child_weight": 2,
    "subsample": 0.8842234913702768,
    "colsample_bytree": 0.45840689146263086,
    "gamma": 3.3084297630544888,
    "lambda": 6.952586917313028,
    "alpha": 0.6395254133055179,
    'seed': RANDOM_STATE,
    'n_jobs': -1,
    # 'device': 'cuda'
}

# Train XGBoost model
print("Training XGBoost model...")
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=800,
    evals=[(dtrain, 'train'), (dval, 'val')],
#     early_stopping_rounds=100,
    verbose_eval=50
)


# Evaluate XGBoost
xgb_va_preds = xgb_model.predict(dval)
xgb_hr3 = hitrate_at_3(y_va, xgb_va_preds, groups_va)
print(f"HitRate@3: {xgb_hr3:.3f}")


xgb_importance = xgb_model.get_score(importance_type='gain')
xgb_importance_df = pl.DataFrame(
    [{'feature': k, 'importance': v} for k, v in xgb_importance.items()]
).sort('importance', descending=bool(1))
print(xgb_importance_df.head(20).to_pandas().to_string())


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

submission_xgb = (
    test.select(['Id', 'ranker_id'])
    .with_columns(pl.Series('pred_score', xgb_model.predict(dtest)))
    .with_columns(
        pl.col('pred_score')
        .rank(method='ordinal', descending=True)
        .over('ranker_id')
        .cast(pl.Int32)
        .alias('selected')
    )
    .select(['Id', 'ranker_id', 'selected', 'pred_score'])
)

top = re_rank(test, submission_xgb)

submission_xgb = (
    submission_xgb.join(top, on=["Id", "ranker_id"], how="left")
    .with_columns(
        [
            pl.when(pl.col("new_selected").is_not_null())
            .then(pl.col("new_selected"))
            .otherwise(pl.col("selected"))
            .alias("selected")
        ]
    )
    .select(["Id", "ranker_id", "selected"])
)


submission_xgb.write_csv('submission.csv')




