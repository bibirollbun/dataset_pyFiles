!pip install -U xgboost
!pip install -U polars
!pip install -U optuna
!pip install -U catboost
!pip install -U lightgbm


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





import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import time
import xgboost as xgb
import catboost
import lightgbm as lgb
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# Load data
train = pl.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet').drop('__index_level_0__')
test = pl.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet').drop('__index_level_0__').with_columns(pl.lit(0, dtype=pl.Int64).alias("selected"))

data_raw = pl.concat((train, test))


top_hubs = (
    train.select("legs0_segments1_departureFrom_airport_iata")
         .drop_nulls()
         .group_by("legs0_segments1_departureFrom_airport_iata")
         .agg(pl.count().alias("count"))
         .sort("count", descending=True)
         .head(10)
)

hub_airports = top_hubs["legs0_segments1_departureFrom_airport_iata"].to_list()
hub_airports


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

# Step 1: Add independent features
df = df.with_columns([
    # Carrier popularity
    (pl.col('carrier0_pop') * pl.col('carrier1_pop')).alias('carrier_pop_product'),
])

df = df.with_columns(
    (
        # Policy compliance (25% weight)
        (pl.col("pricingInfo_isAccessTP") * 0.25) +
        # Direct flights (25% weight)
        (pl.col("is_direct_leg0") * 0.25) +
        # Business-hour departures/arrivals (25% weight)
        ((pl.col("legs0_departureAt_business_time") + pl.col("legs1_departureAt_business_time")) * 0.125) +
        # VIP preference for business class (25% weight)
        ((pl.col("isVip") == 1) & (pl.col("avg_cabin_class") >= 1.5)).cast(pl.Int8) * 0.25
    ).alias("business_traveler_perfect_match"),

    # Timezone diff only
    (pl.col("legs0_arrivalAt_hour") - pl.col("legs0_departureAt_hour") -
     (pl.col("legs0_duration") / 60)).alias("timezone_diff_leg0"),
)

df = df.with_columns(
    (
        (pl.col("is_one_way") == 0) &  # Round-trip
        (pl.col("legs0_arrivalAt_hour") >= 8) &  # Arrive by morning
        (pl.col("legs1_departureAt_hour") <= 18) &  # Return by evening
        (pl.col("timezone_diff_leg0").abs() < 3)   # Minimal jetlag
    ).cast(pl.Int8).alias("meeting_friendly_itinerary")
)




data = df.with_columns(
    [pl.col(c).fill_null(0) for c in df.select(pl.selectors.numeric()).columns] +
    [pl.col(c).fill_null("missing") for c in df.select(pl.selectors.string()).columns]
)


#df = df.with_columns([
    #(df["legs0_segments0_departureFrom_airport_iata"] == df["legs1_segments0_arrivalTo_airport_iata"]).cast(pl.Int8).alias("roundtrip_symmetric"),
    #(df["carrier0_pop"] - df["carrier1_pop"]).fill_null(0).alias("carrier_popularity_diff"),
    #(1 / (df["group_size"] + 1)).alias("inv_group_size"),
    # (pl.col("is_min_segments") == True) & (pl.col("group_size") <= 15).cast(pl.Int32).fill_null(0).alias("is_min_segmentand_group_size")
#])


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
    'legs1_segments1_flightNumber'
]

# Columns to exclude (uninformative or problematic)
exclude_cols = [
    'Id', 'ranker_id', 'selected', 'profileId', 'requestDate',
    'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt',
    'miniRules0_percentage', 'miniRules1_percentage',  # >90% missing
    'frequentFlyer',  # Already processed
    # Exclude constant columns
    'pricingInfo_passengerCount','bySelf','n_segments_leg1','timezone_diff_leg0','meeting_friendly_itinerary','business_traveler_perfect_match'
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

X = data.select(feature_cols + ['ranker_id'])
y = data.select(['selected', 'ranker_id'])
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

group_sizes_tr_lgb = groups_tr.group_by('ranker_id').agg(pl.len()).sort('ranker_id')['len'].to_numpy()
group_sizes_va_lgb = groups_va.group_by('ranker_id').agg(pl.len()).sort('ranker_id')['len'].to_numpy()
group_sizes_te_lgb = groups_te.group_by('ranker_id').agg(pl.len()).sort('ranker_id')['len'].to_numpy()



# --- Step 1: group size & median ---
ranker_group_sizes = groups_tr.group_by('ranker_id').agg(pl.len().alias('group_size'))
median_group_size = ranker_group_sizes['group_size'].median()
print(f"Median group size: {median_group_size}")

# --- Step 2: get ranker_id lists ---
small_rankers = ranker_group_sizes.filter(pl.col('group_size') < median_group_size)['ranker_id']
big_rankers   = ranker_group_sizes.filter(pl.col('group_size') >= median_group_size)['ranker_id']

# --- Step 3: rankers where is_min_segments == 0 ---
min_segment_0 = (
    data_xgb_tr
    .filter(pl.col('is_min_segments') == 0)
    .select('ranker_id')
    .unique()
)['ranker_id']

min_segment_1 = (
    data_xgb_tr
    .filter(pl.col('is_min_segments') == 1)
    .select('ranker_id')
    .unique()
)['ranker_id']

is_one_way_ranker = (
    data_xgb_tr
    .filter(pl.col('is_one_way') == 1)
    .select('ranker_id')
    .unique()
)['ranker_id']

is_popular_route_ranker = (
    data_xgb_tr
    .filter(pl.col('is_popular_route') == 1)
    .select('ranker_id')
    .unique()
)['ranker_id']

# --- Step 4: filtering helper ---
def split_data(ranker_ids):
    ranker_list = ranker_ids.to_list() if hasattr(ranker_ids, "to_list") else list(ranker_ids)
    data = data_xgb_tr.filter(pl.col('ranker_id').is_in(ranker_list))
    y    = y_tr.filter(pl.col('ranker_id').is_in(ranker_list))
    grp  = groups_tr.filter(pl.col('ranker_id').is_in(ranker_list))
    return (
        data.drop('ranker_id').to_pandas(),
        y.drop('ranker_id')['selected'].to_numpy(),
        grp
    )


# last_minute_rankers = (
#     data_xgb_tr
#     .with_columns(
#         (pl.col("legs0_departureAt").str.to_datetime() - pl.col("search_date").str.to_datetime())
#         .dt.days().alias("days_until_departure")
#     )
#     .filter(pl.col("days_until_departure") <= 3)
#     .select("ranker_id")
#     .unique()["ranker_id"]
# )

# 2. Cheapest flights in each search
cheapest_rankers = (
    data_xgb_tr
    .filter(pl.col("is_cheapest") == 1)
    .select("ranker_id")
    .unique()["ranker_id"]
)

# 3. Direct flights only
direct_flight_rankers = (
    data_xgb_tr
    .filter((pl.col("is_direct_leg0") == 1) & (pl.col("is_one_way") == 1))
    .select("ranker_id")
    .unique()["ranker_id"]
)

# 4. High loyalty passengers
loyalty_rankers = (
    data_xgb_tr
    .filter(pl.col("n_ff_programs") >= 2)
    .select("ranker_id")
    .unique()["ranker_id"]
)

# 5. Business-friendly itineraries
# biz_friendly_rankers = (
#     data_xgb_tr
#     .filter(pl.col("business_traveler_perfect_match") >= 0.75)
#     .select("ranker_id")
#     .unique()["ranker_id"]
# )

# --- Step Z: create new scenario datasets ---
data_xgb_tr_small_pd, y_tr_small_pd, groups_tr_small   = split_data(small_rankers)
data_xgb_tr_big_pd,   y_tr_big_pd,   groups_tr_big     = split_data(big_rankers)
data_xgb_tr_min0_pd,  y_tr_min0_pd,  groups_tr_min0    = split_data(min_segment_0)
data_xgb_tr_one_way_pd,  y_tr_one_way_pd,  groups_tr_one_way    = split_data(is_one_way_ranker)
data_xgb_tr_popular_route_pd,  y_tr_popular_route_pd,  groups_tr_popular_route_way    = split_data(is_popular_route_ranker)
#data_xgb_tr_last_minute_pd, y_tr_last_minute_pd, groups_tr_last_minute = split_data(last_minute_rankers)
#data_xgb_tr_cheapest_pd,   y_tr_cheapest_pd,   groups_tr_cheapest     = split_data(cheapest_rankers)
#data_xgb_tr_direct_pd,     y_tr_direct_pd,     groups_tr_direct       = split_data(direct_flight_rankers)
data_xgb_tr_loyalty_pd,    y_tr_loyalty_pd,    groups_tr_loyalty      = split_data(loyalty_rankers)
#data_xgb_tr_biz_pd,        y_tr_biz_pd,        groups_tr_biz          = split_data(biz_friendly_rankers)



data_xgb_tr.shape


print("helloooo")


len(min_segment_1) 


import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

# ----------------- Constants and Globals -----------------
RANDOM_STATE = 42
shap_lgbm_seed = {}
xgb_seeds = [50]
lgb_seeds = [42]
all_models = []
feature_importances = []

# ----------------- XGBoost Params -----------------
xgb_rank_params = {
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
}

xgb_rank_params_small = {
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
    'seed': 22,
    'n_jobs': -1,
}

xgb_rank_params_big = {
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
    'seed': 19,
    'n_jobs': -1,
}

xgb_rank_params_min0 = {
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
    'seed': 19,
    'n_jobs': -1,
}

xgb_params = {
    'full':    xgb_rank_params,
    'small':   xgb_rank_params,
    'big':     xgb_rank_params,
    'min0':    xgb_rank_params,
    #'min1':    xgb_rank_params,
    'one_way': xgb_rank_params,
    #'last_minute': xgb_rank_params,
    #'cheapest':    xgb_rank_params,
    'loyalty':     xgb_rank_params,
    'popular_route': xgb_rank_params,
    #'biz':         xgb_rank_params
}

# ----------------- LightGBM Params -----------------
lgb_rank_params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'boosting_type': 'gbdt',
    'eval_at': [3],
    'num_leaves': 137,
    'learning_rate': 0.1923609,
    'min_child_samples': 69,
    'lambda_l1': 0.0017863,
    'lambda_l2': 7.8818,
    'feature_fraction': 0.6015,
    'bagging_fraction': 0.8536,
    'bagging_freq': 7,
    'verbosity': -1,
    'label_gain': [0, 1]
}

# ----------------- Drop ranker_id from pandas frames -----------------
for df in [
    data_xgb_tr_small_pd, data_xgb_tr_big_pd, data_xgb_tr_min0_pd,
    data_xgb_tr_one_way_pd, data_xgb_tr_popular_route_pd,
    data_xgb_tr_loyalty_pd
]:
    if 'ranker_id' in df.columns:
        df.drop(columns=['ranker_id'], inplace=True)

# ----------------- Group sizes for small/big -----------------
group_sizes_tr_small        = groups_tr_small.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
group_sizes_tr_big          = groups_tr_big.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
group_size_tr_min_0         = groups_tr_min0.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
#group_size_tr_min_1        = groups_tr_min1.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
group_size_tr_one_way        = groups_tr_one_way.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
group_size_tr_popular_route        = groups_tr_popular_route_way.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
#group_sizes_tr_last_minute  = groups_tr_last_minute.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
#group_sizes_tr_cheapest     = groups_tr_cheapest.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
#group_sizes_tr_direct       = groups_tr_direct.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
group_sizes_tr_loyalty      = groups_tr_loyalty.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
#group_sizes_tr_biz          = groups_tr_biz.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()

# ----------------- DMatrix Setup -----------------
dtrain_full = xgb.DMatrix(
    data_xgb_tr.drop('ranker_id').to_pandas(),
    label=y_tr['selected'].to_numpy(),
    group=group_sizes_tr,
    feature_names=data_xgb.drop('ranker_id').columns
)

dtrain_small = xgb.DMatrix(data_xgb_tr_small_pd, label=y_tr_small_pd, group=group_sizes_tr_small, feature_names=list(data_xgb_tr_small_pd.columns))
dtrain_big   = xgb.DMatrix(data_xgb_tr_big_pd,   label=y_tr_big_pd,   group=group_sizes_tr_big,   feature_names=list(data_xgb_tr_big_pd.columns))
dtrain_min_0 = xgb.DMatrix(data_xgb_tr_min0_pd,  label=y_tr_min0_pd,  group=group_size_tr_min_0,  feature_names=list(data_xgb_tr_min0_pd.columns))
#dtrain_min_1 = xgb.DMatrix(data_xgb_tr_min1_pd,  label=y_tr_min1_pd,  group=group_size_tr_min_1,  feature_names=list(data_xgb_tr_min1_pd.columns))
dtrain_one_way = xgb.DMatrix(data_xgb_tr_one_way_pd,  label=y_tr_one_way_pd,  group=group_size_tr_one_way,  feature_names=list(data_xgb_tr_one_way_pd.columns))
dtrain_popular_route = xgb.DMatrix(data_xgb_tr_popular_route_pd,  label=y_tr_popular_route_pd,  group=group_size_tr_popular_route,  feature_names=list(data_xgb_tr_popular_route_pd.columns))

# New scenarios
#dtrain_last_minute = xgb.DMatrix(data_xgb_tr_last_minute_pd, label=y_tr_last_minute_pd, group=group_sizes_tr_last_minute, feature_names=list(data_xgb_tr_last_minute_pd.columns))
#dtrain_cheapest    = xgb.DMatrix(data_xgb_tr_cheapest_pd,    label=y_tr_cheapest_pd,    group=group_sizes_tr_cheapest,    feature_names=list(data_xgb_tr_cheapest_pd.columns))
dtrain_loyalty     = xgb.DMatrix(data_xgb_tr_loyalty_pd,     label=y_tr_loyalty_pd,     group=group_sizes_tr_loyalty,     feature_names=list(data_xgb_tr_loyalty_pd.columns))
#dtrain_biz         = xgb.DMatrix(data_xgb_tr_biz_pd,         label=y_tr_biz_pd,         group=group_sizes_tr_biz,         feature_names=list(data_xgb_tr_biz_pd.columns))

# Validation and Test remain unchanged
dval = xgb.DMatrix(data_xgb_va.drop('ranker_id').to_pandas(), label=y_va['selected'].to_numpy(), group=group_sizes_va, feature_names=data_xgb.drop('ranker_id').columns)
dtest = xgb.DMatrix(data_xgb_te.drop('ranker_id').to_pandas(), label=y_te['selected'].to_numpy(), group=group_sizes_te, feature_names=data_xgb.drop('ranker_id').columns)

# ----------------- Train XGBoost Models -----------------
for portion, params in xgb_params.items():
    print(f"\nTraining XGBoost model with portion: {portion}...")

    train_map = {
        'full':        dtrain_full,
        'small':       dtrain_small,
        'big':         dtrain_big,
        'min0':        dtrain_min_0,
        #'min1':        dtrain_min_1,
        'one_way':     dtrain_one_way,
        'popular_route': dtrain_popular_route,
        #'last_minute': dtrain_last_minute,
        #'cheapest':    dtrain_cheapest,
        #'direct':      dtrain_direct,
        'loyalty':     dtrain_loyalty,
        #'biz':         dtrain_biz
    }

    train = train_map[portion]
    val   = dval

    params = params.copy()
    params['seed'] = RANDOM_STATE

    model = xgb.train(
        params,
        train,
        num_boost_round=860,
        evals=[(train, 'train'), (val, 'val')],
        # early_stopping_rounds=100,
        verbose_eval=50
    )

    all_models.append(('xgb', RANDOM_STATE, portion, model))

    # Feature importance
    xgb_fi = pd.DataFrame.from_dict(
        model.get_score(importance_type='gain'),
        orient='index',
        columns=['importance_gain']
    ).reset_index()
    xgb_fi.columns = ['feature', 'importance_gain']
    xgb_fi['importance_split'] = list(model.get_score(importance_type='weight').values())
    xgb_fi['seed'] = RANDOM_STATE
    xgb_fi['model_type'] = 'xgb'
    xgb_fi['portion'] = portion
    feature_importances.append(xgb_fi)

# ----------------- Optional: Train LightGBM -----------------
# Uncomment if needed

# for seed in lgb_seeds:
#     print(f"\nTraining LightGBM model with seed {seed}...")
#     lgb_train = lgb.Dataset(
#         data=data_xgb_tr.drop('ranker_id').to_pandas(),
#         label=y_tr['selected'].to_numpy(),
#         group=group_sizes_tr_lgb,
#         feature_name=data_xgb.drop('ranker_id').columns,
#         free_raw_data=False
#     )
#     lgb_val = lgb.Dataset(
#         data=data_xgb_va.drop('ranker_id').to_pandas(),
#         label=y_va['selected'].to_numpy(),
#         group=group_sizes_va_lgb,
#         feature_name=data_xgb.drop('ranker_id').columns,
#         reference=lgb_train,
#         free_raw_data=False
#     )
#     params = lgb_rank_params.copy()
#     params['seed'] = seed

#     model = lgb.train(
#         params,
#         lgb_train,
#         num_boost_round=1700,
#         valid_sets=[lgb_train, lgb_val],
#         callbacks=[lgb.early_stopping(300), lgb.log_evaluation(50)]
#     )

#     all_models.append(('lgb', seed, model))

#     fi_df = pd.DataFrame({
#         'feature': data_xgb.drop('ranker_id').columns,
#         'importance_split': model.feature_importance(importance_type='split'),
#         'importance_gain': model.feature_importance(importance_type='gain'),
#         'seed': seed,
#         'model_type': 'lgb'
#     })
#     feature_importances.append(fi_df)

# ----------------- Combine Feature Importances -----------------
all_feature_importance = pd.concat(feature_importances, ignore_index=True)

print("\nâœ… Training completed. Total models:", len(all_models))



# Group by feature and model_type
agg_importance = (
    all_feature_importance
    .groupby(["model_type", "feature"])[["importance_split", "importance_gain"]]
    .mean()
    .reset_index()
)

# Separate and sort LightGBM and XGBoost
# lgb_fi_sorted = (
#     agg_importance[agg_importance["model_type"] == "lgb"]
#     .sort_values("importance_gain", ascending=False)
# )

xgb_fi_sorted = (
    agg_importance[agg_importance["model_type"] == "xgb"]
    .sort_values("importance_gain", ascending=False)
)

# # Display top features
# print("ğŸ”� Top LightGBM Features (by Gain):")
# print(lgb_fi_sorted.head(30))

pd.set_option('display.max_rows', None)     # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 0)           # Auto-detect width
pd.set_option('display.max_colwidth', None) # Show full content in cells

print(xgb_fi_sorted.head(100))


from sklearn.metrics import ndcg_score
import numpy as np

# --- Separate LGB and XGB models from all_models ---
#lgb_models = [m for (typ, _, m) in all_models if typ == 'lgb']
xgb_models = [model for (typ, _, _, model) in all_models if typ == 'xgb']

# --- Predict on validation set and average ---
print("\nğŸ“Š Averaging predictions on validation set...")

# LightGBM predictions
#preds_val_lgb = np.mean([model.predict(data_xgb_va) for model in lgb_models], axis=0)

# XGBoost predictions
data_xgb_va_pd = data_xgb_va.drop('ranker_id').to_pandas()
dval = xgb.DMatrix(data_xgb_va_pd)
preds_val_xgb = np.mean([model.predict(dval) for model in xgb_models], axis=0)

# Combine XGBoost predictions
preds_val = preds_val_xgb

# Convert labels to numeric if needed
y_true = y_va['selected'].to_numpy().flatten()

# Evaluate NDCG@3
ndcg_val = ndcg_score([y_true], [preds_val], k=3)
print(f"âœ… Ensemble NDCG@3: {ndcg_val:.4f}")

# Evaluate HitRate@3 (if function is defined)
ensemble_hr3 = hitrate_at_3(y_true, preds_val, groups_va['ranker_id'])
print(f"ğŸ�¯ Ensemble HitRate@3: {ensemble_hr3:.4f}")


def re_rank(test: pl.DataFrame, submission_xgb: pl.DataFrame, penalty_factor=0.12):
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


# --- Predict on test set ---
print("\nğŸ“¦ Generating predictions on test set...")

# LightGBM test predictions
#preds_test_lgb = np.mean([model.predict(data_xgb_te) for model in lgb_models], axis=0)

# # XGBoost test predictions

data_xgb_te_pd = data_xgb_te.drop('ranker_id').to_pandas()
dtest = xgb.DMatrix(data_xgb_te_pd)
preds_test_xgb = np.mean([model.predict(dtest) for model in xgb_models], axis=0)

# Combined test ensemble
ensemble_test_preds =  preds_test_xgb

submission_df = (
    test.select(['Id', 'ranker_id'])
    .with_columns(pl.Series('pred_score', ensemble_test_preds))
    .with_columns(
        pl.col('pred_score')
        .rank(method='ordinal', descending=True)
        .over('ranker_id')
        .cast(pl.Int32)
        .alias('selected')
    )
    .select(['Id', 'ranker_id', 'selected', 'pred_score'])
)

top = re_rank(test, submission_df)

submission_df = (
    submission_df.join(top, on=["Id", "ranker_id"], how="left")
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


# --- Save to CSV ---
submission_df.write_csv('submission.csv')
print("\nâœ… Submission file 'submission.csv' created successfully.")
print(submission_df.head())


# import xgboost as xgb

# xgb_rank_params = {
#     'objective': 'rank:ndcg',       # or use 'rank:pairwise' as alternative
#     'learning_rate': 0.1,
#     'gamma': 1.0,
#     'min_child_weight': 30,
#     'max_depth': 6,
#     'subsample': 0.85,
#     'colsample_bytree': 0.6,
#     'lambda': 1.0,
#     'alpha': 0.1,
#     'eval_metric': 'ndcg@3',
#     'verbosity': 1,
#     'seed': 42,
#     'tree_method': 'hist',          # optional for speed
# }

# print("Training XGBoost ranker...")

# # Create DMatrix for XGBoost
# xgb_train = xgb.DMatrix(data_xgb_tr, label=y_tr.to_numpy().flatten())
# xgb_val = xgb.DMatrix(data_xgb_va, label=y_va.to_numpy().flatten())

# xgb_train.set_group(group_sizes_tr)
# xgb_val.set_group(group_sizes_va)

# # Train XGBoost ranking model
# xgb_model = xgb.train(
#     xgb_rank_params,
#     dtrain=xgb_train,
#     num_boost_round=500,
#     evals=[(xgb_train, "train"), (xgb_val, "valid")],
#     early_stopping_rounds=50,
#     verbose_eval=50
# )

# xgb_fi = pd.DataFrame.from_dict(xgb_model.get_score(importance_type='gain'), orient='index', columns=['importance_gain']).reset_index()
# xgb_fi.columns = ['feature', 'importance_gain']
# xgb_fi = xgb_fi.sort_values('importance_gain', ascending=False)



# xgb_fi.head(30)


# import lightgbm as lgb
# import numpy as np
# from sklearn.metrics import ndcg_score

# lgb_rank_params = {
#     'objective': 'lambdarank',
#     'metric': 'ndcg',
#     'boosting_type': 'gbdt',
#     'eval_at': [3],
#     'num_leaves': 137,
#     'learning_rate': 0.1923609,
#     'min_child_samples': 69,
#     'lambda_l1': 0.0017863,
#     'lambda_l2': 7.8818,
#     'feature_fraction': 0.6015,
#     'bagging_fraction': 0.8536,
#     'bagging_freq': 7,
#     'verbosity': -1,
#     'label_gain': [0, 1]
# }


# models = []
# lgb_seeds = [12, 32, 42]

# feature_importances = []

# for seed in lgb_seeds:
#     params = lgb_rank_params.copy()
#     params['seed'] = seed

#     print(f"Training LightGBM model with seed {seed}...")
    
#     lgb_train = lgb.Dataset(
#         data=data_xgb_tr,
#         label=y_tr.to_numpy().flatten(),
#         group=group_sizes_tr,
#         feature_name=feature_cols,
#         free_raw_data=False
#     )

#     lgb_val = lgb.Dataset(
#         data=data_xgb_va,
#         label=y_va.to_numpy().flatten(),
#         group=group_sizes_va,
#         feature_name=feature_cols,
#         reference=lgb_train,
#         free_raw_data=False
#     )

#     model = lgb.train(
#         params,
#         lgb_train,
#         num_boost_round=1500,
#         valid_sets=[lgb_train, lgb_val],
#         callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)]
#     )

#     models.append(model)

#     # Store feature importance
#     fi_df = pd.DataFrame({
#         'feature': feature_cols,
#         'importance_split': model.feature_importance(importance_type='split'),
#         'importance_gain': model.feature_importance(importance_type='gain'),
#         'seed': seed
#     })
#     feature_importances.append(fi_df)


# # Combine all importance dataframes
# all_fi_df = pd.concat(feature_importances)

# # Average importance across seeds
# avg_fi = all_fi_df.groupby("feature")[["importance_split", "importance_gain"]].mean().sort_values("importance_gain", ascending=False)

# # Show top 20 features
# print(avg_fi.head(30))


# avg_fi['norm_gain'] = 100 * avg_fi['importance_gain'] / avg_fi['importance_gain'].sum()
# avg_fi['norm_gain'].head(20) 


# import matplotlib.pyplot as plt
# avg_fi.head(30).sort_values("importance_gain").plot(kind='barh', figsize=(15, 12))
# plt.title("Top 30 Features by Gain")
# plt.xlabel("Average Gain Importance")
# plt.tight_layout()
# plt.show()





# # Predict on validation set and average
# preds_val = np.mean([model.predict(data_xgb_va) for model in models], axis=0)

# from sklearn.metrics import ndcg_score
# # Evaluate using NDCG@3
# ndcg_val = ndcg_score([y_va.to_numpy().flatten()], [preds_val], k=3)
# print(f"âœ… Ensemble NDCG@3: {ndcg_val:.4f}")

# lgb_ensemble_hr3 = hitrate_at_3(y_va['selected'], preds_val, groups_va['ranker_id'])
# print(f"LGBM ensemble HitRate@3:   {lgb_ensemble_hr3:.4f}")

# lgb_ensemble_test_preds  = np.mean([model.predict(data_xgb_te) for model in models], axis=0)

# submission_df = test.select(['Id', 'ranker_id']).with_columns(
#     pl.Series(name="lgb_score", values=lgb_ensemble_test_preds)
# ).with_columns(
#     # Rank predictions descending (best scores first) within each group
#     pl.col("lgb_score").rank(method="ordinal", descending=True).over("ranker_id").cast(pl.Int32).alias("selected")
# ).select(["Id", "ranker_id", "selected"])

# # Save to CSV
# submission_df.write_csv('submission.csv')

# print("\nâœ… Submission file 'submission.csv' created successfully.")
# print(submission_df.head())




