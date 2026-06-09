!pip install xgboost
!pip install polars
!pip install optuna


pip install tqdm


import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import time
import xgboost as xgb
from sklearn.preprocessing import QuantileTransformer
import optuna
from sklearn.metrics import ndcg_score
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# Load data
train = pl.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet').drop('__index_level_0__')
test = pl.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet').drop('__index_level_0__').with_columns(pl.lit(0, dtype=pl.Int64).alias("selected"))

data_raw = pl.concat((train, test))
print(f"Total data shape: {data_raw.shape}")
print(f"Train shape: {train.shape}, Test shape: {test.shape}")


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

def hitrate_at_k(y_true, y_pred, groups, k):
    """Calculate hitrate@k for any k"""
    df = pl.DataFrame({
        'group': groups,
        'pred': y_pred,
        'true': y_true
    })
    
    return (
        df.filter(pl.col("group").count().over("group") > 10)
        .sort(["group", "pred"], descending=[False, True])
        .group_by("group", maintain_order=True)
        .head(k)
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

print("Duration columns processed...")


# Validate key columns and add fallbacks
required_cols = ['totalPrice', 'taxes', 'ranker_id', 'selected']
missing_required = [col for col in required_cols if col not in df.columns]
if missing_required:
    print(f"Missing required columns: {missing_required}")
    raise ValueError(f"Required columns missing: {missing_required}")

# Add fallback columns if they don't exist
fallback_columns = []
if 'legs0_duration' not in df.columns:
    fallback_columns.append(pl.lit(0).alias('legs0_duration'))
if 'legs1_duration' not in df.columns:
    fallback_columns.append(pl.lit(0).alias('legs1_duration'))
if 'pricingInfo_passengerCount' not in df.columns:
    fallback_columns.append(pl.lit(1).alias('pricingInfo_passengerCount'))

if fallback_columns:
    df = df.with_columns(fallback_columns)
    print(f"Added {len(fallback_columns)} fallback columns")

# Precompute marketing carrier columns check
mc_cols = [f'legs{l}_segments{s}_marketingCarrier_code' for l in (0, 1) for s in range(4)]
mc_exists = [col for col in mc_cols if col in df.columns]
print(f"Found {len(mc_exists)} marketing carrier columns")

# STEP 1: ORIGINAL BASIC FEATURES - Enhanced with additional features (no self-references)
df = df.with_columns([
        # === ORIGINAL PRICE FEATURES ===
        (pl.col("totalPrice") / (pl.col("taxes") + 1)).alias("price_per_tax"),
        (pl.col("taxes") / (pl.col("totalPrice") + 1)).alias("tax_rate"),
        pl.col("totalPrice").log1p().alias("log_price"),
        
        # === NEW ENHANCED PRICE FEATURES ===
        pl.col("totalPrice").sqrt().alias("sqrt_price"),
        (pl.col("totalPrice") ** 0.25).alias("fourth_root_price"),
        (pl.col("totalPrice") / pl.col("pricingInfo_passengerCount").clip(lower_bound=1)).alias("price_per_passenger"),
        (pl.col("taxes").log1p()).alias("log_taxes"),
        (pl.col("totalPrice") - pl.col("taxes")).clip(lower_bound=0).alias("base_fare"),
        
        # === ORIGINAL DURATION FEATURES ===
        (pl.col("legs0_duration").fill_null(0) + pl.col("legs1_duration").fill_null(0)).alias("total_duration"),
        pl.when(pl.col("legs1_duration").fill_null(0) > 0)
            .then(pl.col("legs0_duration") / (pl.col("legs1_duration") + 1))
            .otherwise(1.0).alias("duration_ratio"),
            
        # === NEW ENHANCED DURATION FEATURES ===
        (pl.col("legs0_duration").fill_null(0)).log1p().alias("log_leg0_duration"),
        (pl.col("legs1_duration").fill_null(0)).log1p().alias("log_leg1_duration"),
        pl.when(pl.col("legs1_duration").fill_null(0) > 0)
            .then((pl.col("legs0_duration") - pl.col("legs1_duration")).abs())
            .otherwise(0).alias("duration_difference"),
        
        # === ORIGINAL TRIP TYPE ===
        (pl.col("legs1_duration").is_null() | 
         (pl.col("legs1_duration") == 0) | 
         pl.col("legs1_segments0_departureFrom_airport_iata").is_null()).cast(pl.Int32).alias("is_one_way"),
        
        # === ORIGINAL SEGMENT COUNT ===
        (pl.sum_horizontal(pl.col(col).is_not_null().cast(pl.UInt8) for col in mc_exists) 
         if mc_exists else pl.lit(0)).alias("l0_seg"),
        
        # === ORIGINAL FF FEATURES ===
        (pl.col("frequentFlyer").fill_null("").str.count_matches("/") + 
         (pl.col("frequentFlyer").fill_null("") != "").cast(pl.Int32)).alias("n_ff_programs"),
        
        # === ORIGINAL BINARY FEATURES ===
        pl.col("corporateTariffCode").is_not_null().cast(pl.Int32).alias("has_corporate_tariff"),
        (pl.col("pricingInfo_isAccessTP") == 1).cast(pl.Int32).alias("has_access_tp"),
        
        # === ORIGINAL BAGGAGE & FEES (BASE) ===
        (pl.col("legs0_segments0_baggageAllowance_quantity").fill_null(0) + 
         pl.col("legs1_segments0_baggageAllowance_quantity").fill_null(0)).alias("baggage_total"),
        (pl.col("miniRules0_monetaryAmount").fill_null(0) + 
         pl.col("miniRules1_monetaryAmount").fill_null(0)).alias("total_fees"),
        (pl.col("miniRules0_monetaryAmount").fill_null(0)).log1p().alias("log_fees0"),
        (pl.col("miniRules1_monetaryAmount").fill_null(0)).log1p().alias("log_fees1"),
        
        # === ORIGINAL ROUTES & CARRIERS ===
        pl.col("searchRoute").is_in(["MOWLED/LEDMOW", "LEDMOW/MOWLED", "MOWLED", "LEDMOW", "MOWAER/AERMOW"])
            .cast(pl.Int32).alias("is_popular_route"),
        
        # === NEW ENHANCED ROUTE FEATURES ===
        pl.col("searchRoute").str.contains("MOW").fill_null(False).cast(pl.Int32).alias("has_moscow"),
        pl.col("searchRoute").str.contains("LED").fill_null(False).cast(pl.Int32).alias("has_stpetersburg"),
        pl.col("searchRoute").str.contains("AER").fill_null(False).cast(pl.Int32).alias("has_sochi"),
        pl.col("searchRoute").str.contains("SVO").fill_null(False).cast(pl.Int32).alias("has_sheremetyevo"),
        pl.col("searchRoute").str.contains("VKO").fill_null(False).cast(pl.Int32).alias("has_vnukovo"),
        (pl.col("searchRoute").str.count_matches("/") + 1).alias("route_complexity"),
        pl.col("searchRoute").str.len_chars().alias("route_length"),
        
        # === ORIGINAL CABIN ===
        pl.mean_horizontal(["legs0_segments0_cabinClass", "legs1_segments0_cabinClass"]).alias("avg_cabin_class"),
        (pl.col("legs0_segments0_cabinClass").fill_null(0) - 
         pl.col("legs1_segments0_cabinClass").fill_null(0)).alias("cabin_class_diff"),
         
        # === NEW ENHANCED CABIN FEATURES ===
        pl.max_horizontal(["legs0_segments0_cabinClass", "legs1_segments0_cabinClass"]).alias("max_cabin_class"),
        pl.min_horizontal(["legs0_segments0_cabinClass", "legs1_segments0_cabinClass"]).alias("min_cabin_class"),
        (pl.col("legs0_segments0_cabinClass").fill_null(0) >= 3).cast(pl.Int32).alias("has_business_leg0"),
        (pl.col("legs1_segments0_cabinClass").fill_null(0) >= 3).cast(pl.Int32).alias("has_business_leg1"),
])

# STEP 2: Add features that depend on columns created in step 1
df = df.with_columns([
        # === ENHANCED BAGGAGE & FEES (using baggage_total and total_fees) ===
        pl.when(pl.col("baggage_total") > 0)
            .then(pl.col("total_fees") / pl.col("baggage_total"))
            .otherwise(0).alias("fee_per_baggage"),
])

print("Basic features created...")


# ORIGINAL SEGMENT COUNTS - Enhanced with additional analysis
print("Processing segment features...")

# First, ensure all duration columns are numeric
duration_cols_to_check = []
for leg in (0, 1):
    for seg in range(4):
        col_name = f"legs{leg}_segments{seg}_duration"
        if col_name in df.columns:
            duration_cols_to_check.append(col_name)

# Convert any string duration columns to numeric
conversion_exprs = []
for col in duration_cols_to_check:
    # Check if column exists and convert to numeric if it's string type
    if col in df.columns:
        col_dtype = df.select(pl.col(col)).dtypes[0]
        if col_dtype in [pl.Utf8, pl.String]:
            print(f"Converting {col} from {col_dtype} to numeric")
            # Try to convert string to numeric, fallback to 0
            conversion_exprs.append(
                pl.col(col).cast(pl.Float64, strict=False).fill_null(0).alias(col)
            )
        elif col_dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            print(f"Converting {col} from {col_dtype} to numeric")
            conversion_exprs.append(
                pl.col(col).cast(pl.Float64, strict=False).fill_null(0).alias(col)
            )

if conversion_exprs:
    df = df.with_columns(conversion_exprs)
    print(f"Converted {len(conversion_exprs)} duration columns to numeric")

# Now process segment features safely
seg_exprs = []
for leg in (0, 1):
    seg_cols = [f"legs{leg}_segments{s}_duration" for s in range(4) if f"legs{leg}_segments{s}_duration" in df.columns]
    if seg_cols:
        # Count non-null segments
        seg_exprs.append(
            pl.sum_horizontal(pl.col(c).is_not_null() for c in seg_cols)
                .cast(pl.Int32).alias(f"n_segments_leg{leg}")
        )
        
        # Average segment duration per leg - ensure numeric operations
        try:
            seg_exprs.append(
                (pl.sum_horizontal(pl.col(c).fill_null(0).cast(pl.Float64) for c in seg_cols) / 
                 pl.sum_horizontal(pl.col(c).is_not_null().cast(pl.Int32) for c in seg_cols).clip(lower_bound=1).cast(pl.Float64)
                ).alias(f"avg_segment_duration_leg{leg}")
            )
        except:
            print(f"Warning: Failed to create avg_segment_duration_leg{leg}, using fallback")
            seg_exprs.append(pl.lit(0.0).alias(f"avg_segment_duration_leg{leg}"))
        
        # Max segment duration per leg - ensure numeric operations
        try:
            seg_exprs.append(
                pl.max_horizontal([pl.col(c).fill_null(0).cast(pl.Float64) for c in seg_cols]).alias(f"max_segment_duration_leg{leg}")
            )
        except:
            print(f"Warning: Failed to create max_segment_duration_leg{leg}, using fallback")
            seg_exprs.append(pl.lit(0.0).alias(f"max_segment_duration_leg{leg}"))
    else:
        seg_exprs.extend([
            pl.lit(0).cast(pl.Int32).alias(f"n_segments_leg{leg}"),
            pl.lit(0.0).alias(f"avg_segment_duration_leg{leg}"),
            pl.lit(0.0).alias(f"max_segment_duration_leg{leg}")
        ])

# Add segment-based features with error handling
try:
    df = df.with_columns(seg_exprs)
    print(f"Added {len(seg_exprs)} segment features successfully")
except Exception as e:
    print(f"Error adding segment features: {e}")
    # Add fallback features
    fallback_exprs = [
        pl.lit(0).cast(pl.Int32).alias("n_segments_leg0"),
        pl.lit(0).cast(pl.Int32).alias("n_segments_leg1"),
        pl.lit(0.0).alias("avg_segment_duration_leg0"),
        pl.lit(0.0).alias("avg_segment_duration_leg1"),
        pl.lit(0.0).alias("max_segment_duration_leg0"),
        pl.lit(0.0).alias("max_segment_duration_leg1")
    ]
    df = df.with_columns(fallback_exprs)
    print("Added fallback segment features")

# ORIGINAL DERIVED FEATURES - Enhanced
df = df.with_columns([
    # === ORIGINAL ===
    (pl.col("n_segments_leg0") + pl.col("n_segments_leg1")).alias("total_segments"),
    (pl.col("n_segments_leg0") == 1).cast(pl.Int32).alias("is_direct_leg0"),
    pl.when(pl.col("is_one_way") == 1).then(0)
        .otherwise((pl.col("n_segments_leg1") == 1).cast(pl.Int32)).alias("is_direct_leg1"),
        
    # === NEW ENHANCED SEGMENT FEATURES ===
    (pl.col("n_segments_leg0") * pl.col("n_segments_leg1")).alias("segment_interaction"),
    pl.when(pl.col("n_segments_leg1") > 0)
        .then(pl.col("n_segments_leg0") / pl.col("n_segments_leg1"))
        .otherwise(1.0).alias("segment_ratio"),
    (pl.col("avg_segment_duration_leg0") + pl.col("avg_segment_duration_leg1") / 2).alias("overall_avg_segment_duration"),
])

# ORIGINAL MORE DERIVED FEATURES - Enhanced
df = df.with_columns([
    # === ORIGINAL ===
    (pl.col("is_direct_leg0") & pl.col("is_direct_leg1")).cast(pl.Int32).alias("both_direct"),
    ((pl.col("isVip") == 1) | (pl.col("n_ff_programs") > 0)).cast(pl.Int32).alias("is_vip_freq"),
    (pl.col("baggage_total") > 0).cast(pl.Int32).alias("has_baggage"),
    (pl.col("total_fees") > 0).cast(pl.Int32).alias("has_fees"),
    (pl.col("total_fees") / (pl.col("totalPrice") + 1)).alias("fee_rate"),
    pl.col("Id").count().over("ranker_id").alias("group_size"),
    
    # === NEW ENHANCED FEATURES ===
    (pl.col("baggage_total") >= 20).cast(pl.Int32).alias("has_checked_baggage"),
    (pl.col("n_ff_programs") >= 2).cast(pl.Int32).alias("multi_ff_programs"),
    ((pl.col("isVip") == 1) & (pl.col("n_ff_programs") > 0)).cast(pl.Int32).alias("is_vip_and_freq"),
    (pl.col("total_duration") / (pl.col("total_segments").clip(lower_bound=1))).alias("duration_per_segment"),
    (pl.col("totalPrice") / (pl.col("total_duration") + 1)).alias("price_per_minute"),
])

print("Enhanced segment and derived features created...")


# ORIGINAL MAJOR CARRIER - Enhanced with more carrier analysis
if "legs0_segments0_marketingCarrier_code" in df.columns:
    df = df.with_columns([
        # Original
        pl.col("legs0_segments0_marketingCarrier_code").is_in(["SU", "S7", "U6"])
            .cast(pl.Int32).alias("is_major_carrier"),
        # Enhanced
        pl.col("legs0_segments0_marketingCarrier_code").is_in(["SU"]).cast(pl.Int32).alias("is_aeroflot"),
        pl.col("legs0_segments0_marketingCarrier_code").is_in(["S7"]).cast(pl.Int32).alias("is_s7"),
        pl.col("legs0_segments0_marketingCarrier_code").is_in(["U6"]).cast(pl.Int32).alias("is_ural"),
        # Low cost carriers
        pl.col("legs0_segments0_marketingCarrier_code").is_in(["U6", "DP", "5N"]).cast(pl.Int32).alias("is_low_cost"),
    ])
    
    # Cross-leg carrier consistency
    if "legs1_segments0_marketingCarrier_code" in df.columns:
        df = df.with_columns([
            (pl.col("legs0_segments0_marketingCarrier_code") == pl.col("legs1_segments0_marketingCarrier_code"))
                .fill_null(True).cast(pl.Int32).alias("same_carrier_both_legs"),
        ])
    else:
        df = df.with_columns(pl.lit(1).alias("same_carrier_both_legs"))
else:
    df = df.with_columns([
        pl.lit(0).alias("is_major_carrier"),
        pl.lit(0).alias("is_aeroflot"),
        pl.lit(0).alias("is_s7"),
        pl.lit(0).alias("is_ural"),
        pl.lit(0).alias("is_low_cost"),
        pl.lit(1).alias("same_carrier_both_legs"),
    ])

# Enhanced group size features
df = df.with_columns([
    pl.col("group_size").log1p().alias("group_size_log"),
    pl.col("group_size").sqrt().alias("group_size_sqrt"),
    # Group size categories
    pl.when(pl.col("group_size") <= 5).then(0)
        .when(pl.col("group_size") <= 15).then(1)
        .when(pl.col("group_size") <= 30).then(2)
        .otherwise(3).alias("group_size_category")
])

print("Carrier and group features enhanced...")


time_exprs = []
for col in ("legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt"):
    if col in df.columns:
        dt = pl.col(col).str.to_datetime(strict=False)
        h = dt.dt.hour().fill_null(12)
        
        # Original features
        time_exprs.extend([
            h.alias(f"{col}_hour"),
            dt.dt.weekday().fill_null(0).alias(f"{col}_weekday"),
            (((h >= 6) & (h <= 9)) | ((h >= 17) & (h <= 20))).cast(pl.Int32).alias(f"{col}_business_time")
        ])
        
        # NEW: Enhanced time features
        time_exprs.extend([
            dt.dt.month().fill_null(6).alias(f"{col}_month"),
            dt.dt.day().fill_null(15).alias(f"{col}_day"),
            dt.dt.quarter().fill_null(2).alias(f"{col}_quarter"),
            (dt.dt.weekday() >= 5).fill_null(False).cast(pl.Int32).alias(f"{col}_is_weekend"),
            
            # Time of day categories
            pl.when(h < 6).then(0)  # Night
                .when(h < 12).then(1)  # Morning  
                .when(h < 18).then(2)  # Afternoon
                .otherwise(3).alias(f"{col}_time_period"),
            
            # Peak travel times
            (((h >= 6) & (h <= 9)) | ((h >= 17) & (h <= 20))).cast(pl.Int32).alias(f"{col}_peak_time"),
            ((h >= 22) | (h <= 5)).cast(pl.Int32).alias(f"{col}_red_eye"),
            ((h >= 5) & (h <= 7)).cast(pl.Int32).alias(f"{col}_early_morning"),
            
            # Holiday periods (approximate Russian holidays)
            ((dt.dt.month() == 12) & (dt.dt.day() >= 25)).fill_null(False).cast(pl.Int32).alias(f"{col}_new_year_period"),
            ((dt.dt.month() == 1) & (dt.dt.day() <= 8)).fill_null(False).cast(pl.Int32).alias(f"{col}_january_holidays"),
            ((dt.dt.month() == 5) & (dt.dt.day() <= 9)).fill_null(False).cast(pl.Int32).alias(f"{col}_may_holidays"),
            
            # Season indicators
            pl.when(dt.dt.month().is_in([12, 1, 2])).then(0)  # Winter
                .when(dt.dt.month().is_in([3, 4, 5])).then(1)  # Spring
                .when(dt.dt.month().is_in([6, 7, 8])).then(2)  # Summer
                .otherwise(3).alias(f"{col}_season"),
        ])

if time_exprs:
    df = df.with_columns(time_exprs)

# NEW: Cross-time features
if all(col in df.columns for col in ["legs0_departureAt", "legs0_arrivalAt"]):
    df = df.with_columns([
        # Flight timing patterns
        (pl.col("legs0_departureAt_hour") == pl.col("legs0_arrivalAt_hour")).cast(pl.Int32).alias("same_hour_leg0"),
        ((pl.col("legs0_departureAt_hour") < 12) & (pl.col("legs0_arrivalAt_hour") >= 12)).cast(pl.Int32).alias("morning_to_afternoon_leg0"),
        
        # Travel day patterns
        (pl.col("legs0_departureAt_weekday") == pl.col("legs0_arrivalAt_weekday")).cast(pl.Int32).alias("same_day_leg0"),
    ])
    
    # Connection time if both legs exist
    if "legs1_departureAt" in df.columns:
        df = df.with_columns([
            ((pl.col("legs1_departureAt").str.to_datetime(strict=False) - 
              pl.col("legs0_arrivalAt").str.to_datetime(strict=False)).dt.total_minutes() / 60
            ).fill_null(0).clip(lower_bound=0, upper_bound=48).alias("connection_time_hours"),
        ])
        
        # Connection time categories
        df = df.with_columns([
            (pl.col("connection_time_hours") <= 2).cast(pl.Int32).alias("tight_connection"),
            ((pl.col("connection_time_hours") > 2) & (pl.col("connection_time_hours") <= 6)).cast(pl.Int32).alias("normal_connection"),
            (pl.col("connection_time_hours") > 12).cast(pl.Int32).alias("overnight_layover"),
        ])
    else:
        df = df.with_columns([
            pl.lit(0.0).alias("connection_time_hours"),
            pl.lit(0).alias("tight_connection"),
            pl.lit(0).alias("normal_connection"),
            pl.lit(0).alias("overnight_layover"),
        ])

print("Enhanced time features created...")


# NEW: Advanced Aircraft Features
if "legs0_segments0_aircraft_code" in df.columns:
    # Wide body aircraft
    widebody_codes = ['77W', '777', '330', '340', '350', '380', '787', 'A33', 'A34', 'A35', '763', '764', '767']
    # Modern efficient aircraft
    modern_codes = ['787', '350', 'A35', '38M', '32N', '321', '32Q']
    
    df = df.with_columns([
        pl.col("legs0_segments0_aircraft_code").is_in(widebody_codes).cast(pl.Int32).alias("is_widebody_leg0"),
        pl.col("legs0_segments0_aircraft_code").is_in(modern_codes).cast(pl.Int32).alias("is_modern_aircraft_leg0"),
    ])
    
    if "legs1_segments0_aircraft_code" in df.columns:
        df = df.with_columns([
            pl.col("legs1_segments0_aircraft_code").is_in(widebody_codes).cast(pl.Int32).alias("is_widebody_leg1"),
            pl.col("legs1_segments0_aircraft_code").is_in(modern_codes).cast(pl.Int32).alias("is_modern_aircraft_leg1"),
            (pl.col("legs0_segments0_aircraft_code") == pl.col("legs1_segments0_aircraft_code")).fill_null(False).cast(pl.Int32).alias("same_aircraft_type")
        ])
    else:
        df = df.with_columns([
            pl.lit(0).alias("is_widebody_leg1"),
            pl.lit(0).alias("is_modern_aircraft_leg1"),
            pl.lit(0).alias("same_aircraft_type")
        ])
else:
    df = df.with_columns([
        pl.lit(0).alias("is_widebody_leg0"),
        pl.lit(0).alias("is_modern_aircraft_leg0"),
        pl.lit(0).alias("is_widebody_leg1"),
        pl.lit(0).alias("is_modern_aircraft_leg1"),
        pl.lit(0).alias("same_aircraft_type")
    ])

print("Aircraft features enhanced...")


# STEP 1: ORIGINAL BATCH RANK COMPUTATIONS - Enhanced
print("Creating ranking features...")

rank_exprs = []
for col, alias in [("totalPrice", "price"), ("total_duration", "duration")]:
    # Original ranks
    rank_exprs.append(pl.col(col).rank().over("ranker_id").alias(f"{alias}_rank"))
    
    # NEW: Enhanced ranking features
    rank_exprs.extend([
        pl.col(col).rank(method="average").over("ranker_id").alias(f"{alias}_rank_avg"),
        pl.col(col).rank(method="dense").over("ranker_id").alias(f"{alias}_rank_dense"),
        (pl.col(col).rank().over("ranker_id") / pl.col(col).count().over("ranker_id")).alias(f"{alias}_rank_pct"),
    ])

# STEP 1: Apply basic ranks first
try:
    df = df.with_columns(rank_exprs)
    print(f" Added {len(rank_exprs)} basic ranking features")
except Exception as e:
    print(f" Error creating basic ranks: {e}")
    # Add fallback ranks
    fallback_ranks = [
        pl.lit(1).alias("price_rank"),
        pl.lit(1).alias("duration_rank"),
        pl.lit(1).alias("price_rank_avg"),
        pl.lit(1).alias("duration_rank_avg"),
        pl.lit(1).alias("price_rank_dense"),
        pl.lit(1).alias("duration_rank_dense"),
        pl.lit(0.5).alias("price_rank_pct"),
        pl.lit(0.5).alias("duration_rank_pct"),
    ]
    df = df.with_columns(fallback_ranks)
    print("Added fallback ranking features")

# STEP 2: PRICE-SPECIFIC FEATURES that depend on ranks - Enhanced
price_exprs = [
    # Original
    (pl.col("totalPrice").rank("average").over("ranker_id") / 
     pl.col("totalPrice").count().over("ranker_id")).alias("price_pct_rank"),
    (pl.col("totalPrice") == pl.col("totalPrice").min().over("ranker_id")).cast(pl.Int32).alias("is_cheapest"),
    ((pl.col("totalPrice") - pl.col("totalPrice").median().over("ranker_id")) / 
     (pl.col("totalPrice").std().over("ranker_id") + 1)).alias("price_from_median"),
    (pl.col("l0_seg") == pl.col("l0_seg").min().over("ranker_id")).cast(pl.Int32).alias("is_min_segments"),
    
    # NEW: Enhanced price analysis
    (pl.col("totalPrice") == pl.col("totalPrice").max().over("ranker_id")).cast(pl.Int32).alias("is_most_expensive"),
    (pl.col("totalPrice") / pl.col("totalPrice").min().over("ranker_id")).alias("price_ratio_to_min"),
    (pl.col("totalPrice") / pl.col("totalPrice").max().over("ranker_id")).alias("price_ratio_to_max"),
    (pl.col("totalPrice") / pl.col("totalPrice").mean().over("ranker_id")).alias("price_ratio_to_mean"),
    
    # Price z-score (standardized)
    ((pl.col("totalPrice") - pl.col("totalPrice").mean().over("ranker_id")) / 
     (pl.col("totalPrice").std().over("ranker_id") + 1)).alias("price_z_score"),
     
    # Market spread analysis
    (pl.col("totalPrice").max().over("ranker_id") - pl.col("totalPrice").min().over("ranker_id")).alias("price_range"),
    pl.col("totalPrice").std().over("ranker_id").alias("price_std"),
    
    # Duration rankings
    (pl.col("total_duration") == pl.col("total_duration").min().over("ranker_id")).cast(pl.Int32).alias("is_fastest"),
    (pl.col("total_duration") / pl.col("total_duration").min().over("ranker_id")).alias("duration_ratio_to_min"),
]

# STEP 2: Apply price-specific features
try:
    df = df.with_columns(price_exprs)
    print(f"Added {len(price_exprs)} price analysis features")
except Exception as e:
    print(f" Error creating price features: {e}")
    print("Continuing without advanced price features...")

# STEP 3: Combined value scores that depend on both price_rank and duration_rank
combined_exprs = [
    # Combined value scores (now that we have price_rank and duration_rank)
    ((1.0 / pl.col("price_rank").clip(lower_bound=1)) + (1.0 / pl.col("duration_rank").clip(lower_bound=1))).alias("combined_value_score"),
    (pl.col("price_rank") * pl.col("duration_rank")).alias("rank_product"),
]

# STEP 3: Apply combined features
try:
    df = df.with_columns(combined_exprs)
    print(f" Added {len(combined_exprs)} combined ranking features")
except Exception as e:
    print(f" Error creating combined features: {e}")
    # Add fallback combined features
    fallback_combined = [
        pl.lit(2.0).alias("combined_value_score"),
        pl.lit(1.0).alias("rank_product"),
    ]
    df = df.with_columns(fallback_combined)
    print("Added fallback combined features")

print("Enhanced ranking features completed.")


# ENHANCED CHEAPEST DIRECT - Enhanced with more complex logic and error handling
print("Creating direct flight features...")

try:
    # Check if is_direct_leg0 exists
    if "is_direct_leg0" not in df.columns:
        print("Warning: is_direct_leg0 column not found, creating fallback")
        df = df.with_columns(pl.lit(0).cast(pl.Int32).alias("is_direct_leg0"))
    
    # Calculate direct flight statistics
    direct_cheapest = (
        df.filter(pl.col("is_direct_leg0") == 1)
        .group_by("ranker_id")
        .agg([
            pl.col("totalPrice").min().alias("min_direct_price"),
            pl.col("total_duration").min().alias("min_direct_duration"),
            pl.len().alias("n_direct_options")
        ])
    )
    
    # STEP 1: Join the direct flight statistics
    df = df.join(direct_cheapest, on="ranker_id", how="left")
    print(" Joined direct flight statistics")
    
    # STEP 2: Create basic direct flight features (no self-references)
    direct_features_1 = [
        # Original
        ((pl.col("is_direct_leg0") == 1) & 
         (pl.col("totalPrice") == pl.col("min_direct_price"))).cast(pl.Int32).fill_null(0).alias("is_direct_cheapest"),
        
        # NEW: Enhanced direct flight analysis
        ((pl.col("is_direct_leg0") == 1) & 
         (pl.col("total_duration") == pl.col("min_direct_duration"))).cast(pl.Int32).fill_null(0).alias("is_direct_fastest"),
        pl.col("n_direct_options").fill_null(0).alias("direct_options_available"),
        pl.when(pl.col("min_direct_price").is_not_null())
            .then(pl.col("totalPrice") / pl.col("min_direct_price"))
            .otherwise(1.0).alias("price_ratio_to_cheapest_direct"),
    ]
    
    df = df.with_columns(direct_features_1)
    print(f" Added {len(direct_features_1)} basic direct flight features")
    
    # STEP 3: Create features that depend on columns from step 2
    direct_features_2 = [
        (pl.col("direct_options_available") > 0).cast(pl.Int32).alias("has_direct_options"),
    ]
    
    df = df.with_columns(direct_features_2)
    print(f" Added {len(direct_features_2)} derived direct flight features")
    
    # STEP 4: Clean up intermediate columns
    df = df.drop(["min_direct_price", "min_direct_duration"])
    print(" Cleaned up intermediate columns")

except Exception as e:
    print(f" Error creating direct flight features: {e}")
    print("Adding fallback direct flight features...")
    
    # Add fallback features
    fallback_direct = [
        pl.lit(0).cast(pl.Int32).alias("is_direct_cheapest"),
        pl.lit(0).cast(pl.Int32).alias("is_direct_fastest"),
        pl.lit(0).alias("direct_options_available"),
        pl.lit(0).cast(pl.Int32).alias("has_direct_options"),
        pl.lit(1.0).alias("price_ratio_to_cheapest_direct"),
    ]
    df = df.with_columns(fallback_direct)
    print("Added fallback direct flight features")

print("Enhanced direct flight features completed.")


# ORIGINAL POPULARITY FEATURES - Enhanced with more sophisticated analysis
df = (
    df.join(
        train.group_by('legs0_segments0_marketingCarrier_code').agg([
            pl.mean('selected').alias('carrier0_pop'),
            pl.count().alias('carrier0_frequency'),
            pl.std('selected').alias('carrier0_selection_variance')
        ]),
        on='legs0_segments0_marketingCarrier_code', 
        how='left'
    )
    .join(
        train.group_by('legs1_segments0_marketingCarrier_code').agg([
            pl.mean('selected').alias('carrier1_pop'),
            pl.count().alias('carrier1_frequency'),
            pl.std('selected').alias('carrier1_selection_variance')
        ]),
        on='legs1_segments0_marketingCarrier_code', 
        how='left'
    )
    .with_columns([
        # Original
        pl.col('carrier0_pop').fill_null(0.0),
        pl.col('carrier1_pop').fill_null(0.0),
        
        # NEW: Enhanced carrier popularity
        pl.col('carrier0_frequency').fill_null(0),
        pl.col('carrier1_frequency').fill_null(0),
        pl.col('carrier0_selection_variance').fill_null(0.0),
        pl.col('carrier1_selection_variance').fill_null(0.0),
    ])
)

# ORIGINAL FINAL FEATURES - Enhanced
df = df.with_columns([
    # Original
    (pl.col('carrier0_pop') * pl.col('carrier1_pop')).alias('carrier_pop_product'),
    
    # NEW: Enhanced carrier analysis
    (pl.col('carrier0_pop') + pl.col('carrier1_pop')).alias('carrier_pop_sum'),
    ((pl.col('carrier0_pop') - pl.col('carrier1_pop')).abs()).alias('carrier_pop_diff'),
    pl.max_horizontal(['carrier0_pop', 'carrier1_pop']).alias('max_carrier_pop'),
    pl.min_horizontal(['carrier0_pop', 'carrier1_pop']).alias('min_carrier_pop'),
    
    # Market presence
    (pl.col('carrier0_frequency') + pl.col('carrier1_frequency')).alias('total_carrier_frequency'),
    (pl.col('carrier0_frequency') >= 1000).cast(pl.Int32).alias('carrier0_high_frequency'),
    (pl.col('carrier1_frequency') >= 1000).cast(pl.Int32).alias('carrier1_high_frequency'),
])

print("Enhanced popularity features created...")


# ORIGINAL POPULARITY FEATURE BASED ON ROUND TRIP - Enhanced with route analysis
required_cols = [
    "legs0_segments0_departureFrom_airport_iata",
    "legs0_segments0_arrivalTo_airport_iata",
    "legs1_segments0_departureFrom_airport_iata",
    "legs1_segments0_arrivalTo_airport_iata"
]

if all(col in df.columns for col in required_cols):
    # Original round trip route
    df = df.with_columns([
        (pl.col("legs0_segments0_departureFrom_airport_iata") + "_" + 
         pl.col("legs0_segments0_arrivalTo_airport_iata") + "__" +
         pl.col("legs1_segments0_departureFrom_airport_iata") + "_" + 
         pl.col("legs1_segments0_arrivalTo_airport_iata")).alias("round_trip_route"),
         
        # NEW: Enhanced route analysis
        (pl.col("legs0_segments0_departureFrom_airport_iata") + "_" + 
         pl.col("legs0_segments0_arrivalTo_airport_iata")).alias("leg0_route"),
        (pl.col("legs1_segments0_departureFrom_airport_iata") + "_" + 
         pl.col("legs1_segments0_arrivalTo_airport_iata")).alias("leg1_route"),
         
        # Route symmetry (true round trip vs open jaw)
        ((pl.col("legs0_segments0_departureFrom_airport_iata") == pl.col("legs1_segments0_arrivalTo_airport_iata")) &
         (pl.col("legs0_segments0_arrivalTo_airport_iata") == pl.col("legs1_segments0_departureFrom_airport_iata"))
        ).cast(pl.Int32).alias("is_true_round_trip"),
    ])

    # Calculate original round trip frequency
    round_trip_freq = (
        train.with_columns([
            (pl.col("legs0_segments0_departureFrom_airport_iata") + "_" + 
             pl.col("legs0_segments0_arrivalTo_airport_iata") + "__" +
             pl.col("legs1_segments0_departureFrom_airport_iata") + "_" + 
             pl.col("legs1_segments0_arrivalTo_airport_iata")).alias("round_trip_route")
        ])
        .group_by("round_trip_route")
        .agg([
            pl.count().alias("rt_route_count"),
            pl.mean('selected').alias('rt_route_popularity')
        ])
    )
    
    # NEW: Individual leg route frequencies
    leg0_freq = (
        train.with_columns([
            (pl.col("legs0_segments0_departureFrom_airport_iata") + "_" + 
             pl.col("legs0_segments0_arrivalTo_airport_iata")).alias("leg0_route")
        ])
        .group_by("leg0_route")
        .agg([
            pl.count().alias("leg0_route_count"),
            pl.mean('selected').alias('leg0_route_popularity')
        ])
    )
    
    leg1_freq = (
        train.with_columns([
            (pl.col("legs1_segments0_departureFrom_airport_iata") + "_" + 
             pl.col("legs1_segments0_arrivalTo_airport_iata")).alias("leg1_route")
        ])
        .group_by("leg1_route")
        .agg([
            pl.count().alias("leg1_route_count"),
            pl.mean('selected').alias('leg1_route_popularity')
        ])
    )

    # Join all route features
    df = (df
        .join(round_trip_freq, on="round_trip_route", how="left")
        .join(leg0_freq, on="leg0_route", how="left")
        .join(leg1_freq, on="leg1_route", how="left")
        .with_columns([
            # Original
            pl.col("rt_route_count").fill_null(0).alias("round_trip_freq"),
            
            # NEW: Enhanced route features
            pl.col("rt_route_popularity").fill_null(0.0).alias("round_trip_popularity"),
            pl.col("leg0_route_count").fill_null(0).alias("leg0_route_freq"),
            pl.col("leg1_route_count").fill_null(0).alias("leg1_route_freq"),
            pl.col("leg0_route_popularity").fill_null(0.0).alias("leg0_route_popularity"),
            pl.col("leg1_route_popularity").fill_null(0.0).alias("leg1_route_popularity"),
            
            # Route frequency categories
            (pl.col("rt_route_count").fill_null(0) >= 100).cast(pl.Int32).alias("is_popular_round_trip"),
            (pl.col("leg0_route_count").fill_null(0) >= 500).cast(pl.Int32).alias("is_popular_leg0_route"),
            (pl.col("leg1_route_count").fill_null(0) >= 500).cast(pl.Int32).alias("is_popular_leg1_route"),
        ])
        .drop(["round_trip_route", "leg0_route", "leg1_route", "rt_route_count", "rt_route_popularity",
               "leg0_route_count", "leg1_route_count", "leg0_route_popularity", "leg1_route_popularity"])
    )
else:
    df = df.with_columns([
        pl.lit(0).alias("round_trip_freq"),
        pl.lit(0.0).alias("round_trip_popularity"),
        pl.lit(0).alias("leg0_route_freq"),
        pl.lit(0).alias("leg1_route_freq"),
        pl.lit(0.0).alias("leg0_route_popularity"),
        pl.lit(0.0).alias("leg1_route_popularity"),
        pl.lit(0).alias("is_true_round_trip"),
        pl.lit(0).alias("is_popular_round_trip"),
        pl.lit(0).alias("is_popular_leg0_route"),
        pl.lit(0).alias("is_popular_leg1_route"),
    ])

print("Enhanced route analysis completed...")


# NEW: Advanced Interaction Features
interaction_features = [
    # Price-time interactions
    (pl.col("totalPrice") * pl.col("legs0_departureAt_peak_time")).alias("price_peak_time_interaction"),
    (pl.col("totalPrice") * pl.col("legs0_departureAt_is_weekend")).alias("price_weekend_interaction"),
    (pl.col("totalPrice") * pl.col("is_direct_leg0")).alias("price_direct_interaction"),
    
    # Duration-segment interactions  
    (pl.col("total_duration") * pl.col("total_segments")).alias("duration_segments_interaction"),
    (pl.col("total_duration") * pl.col("is_major_carrier")).alias("duration_major_carrier_interaction"),
    
    # Service level interactions
    (pl.col("avg_cabin_class") * pl.col("totalPrice")).alias("cabin_price_interaction"),
    (pl.col("baggage_total") * pl.col("total_fees")).alias("baggage_fees_interaction"),
    (pl.col("is_vip_freq") * pl.col("avg_cabin_class")).alias("vip_cabin_interaction"),
    
    # Market competition interactions
    (pl.col("group_size") * pl.col("price_std")).alias("competition_variance_interaction"),
    (pl.col("carrier_pop_product") * pl.col("totalPrice")).alias("popularity_price_interaction"),
    
    # Route-carrier interactions
    (pl.col("is_popular_route") * pl.col("is_major_carrier")).alias("popular_route_major_carrier"),
    (pl.col("round_trip_freq") * pl.col("carrier0_pop")).alias("route_freq_popularity_interaction"),
]

df = df.with_columns(interaction_features)

print("Advanced interaction features created...")


# NEW: Booking Pattern Features with type checking
print("Creating booking pattern features...")

if 'requestDate' in df.columns:
    try:
        # Check the data type of requestDate column
        request_date_dtype = df.select(pl.col('requestDate')).dtypes[0]
        print(f"requestDate column type: {request_date_dtype}")
        
        # Convert to datetime based on current type
        if request_date_dtype in [pl.Utf8, pl.String]:
            # String type - convert from string
            df = df.with_columns([
                pl.col('requestDate').str.to_datetime(strict=False).alias('request_dt')
            ])
        elif request_date_dtype in [pl.Datetime, pl.Date]:
            # Already datetime type - just alias it
            df = df.with_columns([
                pl.col('requestDate').alias('request_dt')
            ])
        else:
            # Unknown type - try to cast
            print(f"Warning: Unexpected requestDate type {request_date_dtype}, attempting conversion")
            df = df.with_columns([
                pl.col('requestDate').cast(pl.Datetime, strict=False).alias('request_dt')
            ])
        
        print(" Successfully converted requestDate to datetime")
        
        # Request timing features
        df = df.with_columns([
            pl.col('request_dt').dt.hour().alias('request_hour'),
            pl.col('request_dt').dt.weekday().alias('request_weekday'),
            pl.col('request_dt').dt.month().alias('request_month'),
            (pl.col('request_dt').dt.weekday() >= 5).cast(pl.Int32).alias('request_weekend'),
            ((pl.col('request_dt').dt.hour() >= 9) & (pl.col('request_dt').dt.hour() <= 17)).cast(pl.Int32).alias('request_business_hours'),
        ])
        
        print(" Added request timing features")
        
        # Lead time calculation
        if 'legs0_departureAt' in df.columns:
            try:
                # Check legs0_departureAt type
                departure_dtype = df.select(pl.col('legs0_departureAt')).dtypes[0]
                print(f"legs0_departureAt column type: {departure_dtype}")
                
                if departure_dtype in [pl.Utf8, pl.String]:
                    # String type - convert from string
                    lead_time_expr = (
                        pl.col('legs0_departureAt').str.to_datetime(strict=False) - pl.col('request_dt')
                    ).dt.total_days().clip(lower_bound=0, upper_bound=365).alias('booking_lead_days')
                else:
                    # Already datetime type
                    lead_time_expr = (
                        pl.col('legs0_departureAt') - pl.col('request_dt')
                    ).dt.total_days().clip(lower_bound=0, upper_bound=365).alias('booking_lead_days')
                
                df = df.with_columns([lead_time_expr])
                print(" Calculated booking lead days")
                
                # Lead time categories
                df = df.with_columns([
                    (pl.col('booking_lead_days') <= 1).cast(pl.Int32).alias('same_day_booking'),
                    (pl.col('booking_lead_days') <= 7).cast(pl.Int32).alias('week_ahead_booking'),
                    (pl.col('booking_lead_days') >= 30).cast(pl.Int32).alias('advance_booking'),
                    (pl.col('booking_lead_days') >= 60).cast(pl.Int32).alias('far_advance_booking'),
                ])
                print(" Added lead time categories")
                
            except Exception as e:
                print(f" Error calculating lead time: {e}")
                # Fallback lead time features
                df = df.with_columns([
                    pl.lit(7.0).alias('booking_lead_days'),
                    pl.lit(0).alias('same_day_booking'),
                    pl.lit(1).alias('week_ahead_booking'),
                    pl.lit(0).alias('advance_booking'),
                    pl.lit(0).alias('far_advance_booking'),
                ])
                print("Added fallback lead time features")
        else:
            df = df.with_columns([
                pl.lit(7.0).alias('booking_lead_days'),
                pl.lit(0).alias('same_day_booking'),
                pl.lit(1).alias('week_ahead_booking'),
                pl.lit(0).alias('advance_booking'),
                pl.lit(0).alias('far_advance_booking'),
            ])
            print("Added default lead time features (legs0_departureAt not found)")
        
        # Clean up intermediate column
        df = df.drop('request_dt')
        print(" Cleaned up intermediate datetime column")

    except Exception as e:
        print(f" Error processing requestDate: {e}")
        print("Adding fallback booking features...")
        # Add fallback features
        df = df.with_columns([
            pl.lit(12).alias('request_hour'),
            pl.lit(2).alias('request_weekday'),
            pl.lit(6).alias('request_month'),
            pl.lit(0).alias('request_weekend'),
            pl.lit(1).alias('request_business_hours'),
            pl.lit(7.0).alias('booking_lead_days'),
            pl.lit(0).alias('same_day_booking'),
            pl.lit(1).alias('week_ahead_booking'),
            pl.lit(0).alias('advance_booking'),
            pl.lit(0).alias('far_advance_booking'),
        ])
        print("Added fallback booking features")
        
else:
    # Default values if requestDate not available
    print("requestDate column not found, using default values")
    df = df.with_columns([
        pl.lit(12).alias('request_hour'),
        pl.lit(2).alias('request_weekday'),
        pl.lit(6).alias('request_month'),
        pl.lit(0).alias('request_weekend'),
        pl.lit(1).alias('request_business_hours'),
        pl.lit(7.0).alias('booking_lead_days'),
        pl.lit(0).alias('same_day_booking'),
        pl.lit(1).alias('week_ahead_booking'),
        pl.lit(0).alias('advance_booking'),
        pl.lit(0).alias('far_advance_booking'),
    ])

print("Booking pattern features completed.")


# ORIGINAL FILL NULLS - Enhanced with smarter filling strategies
# Get all numeric columns
numeric_cols = df.select(pl.selectors.numeric()).columns
string_cols = df.select(pl.selectors.string()).columns

# Smart null filling for numeric columns
numeric_fill_exprs = []
for col in numeric_cols:
    null_count = df[col].is_null().sum()
    if null_count > 0:
        if col.endswith('_pop') or col.endswith('_popularity'):
            # Popularity metrics: fill with median
            fill_value = df[col].median()
        elif 'price' in col.lower() or 'fee' in col.lower() or 'cost' in col.lower():
            # Price-related: fill with 0
            fill_value = 0
        elif 'duration' in col.lower() or 'time' in col.lower():
            # Duration-related: fill with 0
            fill_value = 0
        elif col.endswith('_std') or col.endswith('_variance'):
            # Variance metrics: fill with 0
            fill_value = 0
        elif col.endswith('_count') or col.endswith('_frequency') or col.endswith('_freq'):
            # Count metrics: fill with 0
            fill_value = 0
        else:
            # Default: fill with 0
            fill_value = 0
        
        numeric_fill_exprs.append(pl.col(col).fill_null(fill_value))

# Fill string columns
string_fill_exprs = [pl.col(c).fill_null("missing") for c in string_cols]

# Apply all fills
data = df.with_columns(numeric_fill_exprs + string_fill_exprs)

print(f"Filled nulls in {len(numeric_fill_exprs)} numeric and {len(string_fill_exprs)} string columns")
print(f"Final data shape: {data.shape}")


# ORIGINAL CATEGORICAL FEATURES - Enhanced with new categoricals
cat_features = [
    # Original core categoricals
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
    
    # NEW: Enhanced categoricals
    'frequentFlyer',  # Re-include for encoding
    'group_size_category',  # New categorical
    
    # Time-based categoricals
    'legs0_departureAt_time_period', 'legs0_arrivalAt_time_period',
    'legs1_departureAt_time_period', 'legs1_arrivalAt_time_period',
    'legs0_departureAt_season', 'legs1_departureAt_season',
    'request_month', 'request_weekday',
]

# ORIGINAL COLUMNS TO EXCLUDE - Enhanced exclusion list
exclude_cols = [
    # Original exclusions
    'Id', 'ranker_id', 'selected', 'profileId', 'requestDate',
    'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt',
    'pricingInfo_passengerCount',  # Constant
    
    # NEW: Additional exclusions
    # Remove intermediate computation columns that shouldn't be features
    'carrier0_frequency', 'carrier1_frequency', 
    'carrier0_selection_variance', 'carrier1_selection_variance',
    'n_direct_options',  # Leakage risk
    
    # Remove redundant ranking columns (keep only the most important ones)
    'price_rank_dense', 'duration_rank_dense',
    'price_rank_avg', 'duration_rank_avg',
]

# ORIGINAL SEGMENT EXCLUSIONS - Enhanced
for leg in [0, 1]:
    for seg in [2, 3]:
        for suffix in ['aircraft_code', 'arrivalTo_airport_city_iata', 'arrivalTo_airport_iata',
                      'baggageAllowance_quantity', 'baggageAllowance_weightMeasurementType',
                      'cabinClass', 'departureFrom_airport_iata', 'duration', 'flightNumber',
                      'marketingCarrier_code', 'operatingCarrier_code', 'seatsAvailable']:
            exclude_cols.append(f'legs{leg}_segments{seg}_{suffix}')

# Final feature selection
feature_cols = [col for col in data.columns if col not in exclude_cols]
cat_features_final = [col for col in cat_features if col in feature_cols]

# Remove features with too many nulls or too little variance
low_variance_features = []
for col in feature_cols:
    if col in data.columns:
        if data[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            # Check variance
            var = data[col].var()
            if var is not None and var < 1e-10:
                low_variance_features.append(col)
        else:
            # Check unique values for categorical
            unique_count = data[col].n_unique()
            if unique_count <= 1:
                low_variance_features.append(col)

# Remove low variance features
feature_cols = [col for col in feature_cols if col not in low_variance_features]
cat_features_final = [col for col in cat_features_final if col not in low_variance_features]

print(f"Removed {len(low_variance_features)} low variance features: {low_variance_features[:10]}...")
print(f"Using {len(feature_cols)} features ({len(cat_features_final)} categorical)")

X = data.select(feature_cols)
y = data.select('selected')
groups = data.select('ranker_id')

print(f"Final feature matrix shape: {X.shape}")
print(f"Number of groups: {groups.n_unique()}")


# ORIGINAL CATEGORICAL ENCODING - Enhanced
print("Applying enhanced categorical encoding...")
data_xgb = X.with_columns([
    (pl.col(c).rank("dense") - 1).fill_null(-1).cast(pl.Int32) 
    for c in cat_features_final
])

# ORIGINAL DATA SPLITS - Enhanced with validation
n1 = 16487352  # split train to train and val (10%) in time
n2 = train.height
data_xgb_tr, data_xgb_va, data_xgb_te = data_xgb[:n2], data_xgb[n1:n2], data_xgb[n2:]
y_tr, y_va, y_te = y[:n2], y[n1:n2], y[n2:]
groups_tr, groups_va, groups_te = groups[:n2], groups[n1:n2], groups[n2:]

# Compute group sizes
group_sizes_tr = groups_tr.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
group_sizes_va = groups_va.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()
group_sizes_te = groups_te.group_by('ranker_id', maintain_order=True).agg(pl.len())['len'].to_numpy()

print(f"Train split: {data_xgb_tr.shape}, Groups: {len(group_sizes_tr)}")
print(f"Validation split: {data_xgb_va.shape}, Groups: {len(group_sizes_va)}")
print(f"Test split: {data_xgb_te.shape}, Groups: {len(group_sizes_te)}")

# Create DMatrix objects
dtrain = xgb.DMatrix(data_xgb_tr.to_numpy(), label=y_tr.to_numpy().flatten(), group=group_sizes_tr)  ###unquote these two lines if you need training
dval = xgb.DMatrix(data_xgb_va.to_numpy(), label=y_va.to_numpy().flatten(), group=group_sizes_va)
dtest = xgb.DMatrix(data_xgb_te.to_numpy(), label=y_te.to_numpy().flatten(), group=group_sizes_te)

print("DMatrix objects created successfully")


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
        
        # NEW: Enhanced parameters
        "colsample_bynode": 0.8,
        "colsample_bylevel": 0.9,
        "max_delta_step": 1,  # Helps with imbalanced ranking
        
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
    }


xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=75,
        verbose_eval= 50
    )




import pickle


model_path = f'xgb_full_final_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(xgb_model, f)


# model_path = '/kaggle/input/single-model-final-submission/xgb_full_model (1).pkl'
# with open(model_path, 'rb') as f:
#     xgb_model = pickle.load(f)


# ENHANCED EVALUATION with multiple metrics
# print("=== Enhanced Model Evaluation ===")

# Predictions
# xgb_tr_preds = xgb_model.predict(dtrain)
# xgb_va_preds = xgb_model.predict(dval)
xgb_te_preds = xgb_model.predict(dtest)

# Multiple hitrate metrics
# y_tr_flat = y_tr.to_numpy().flatten()
# y_va_flat = y_va.to_numpy().flatten()
y_te_flat = y_te.to_numpy().flatten()
# groups_tr_flat = groups_tr.to_numpy().flatten()
# groups_va_flat = groups_va.to_numpy().flatten()
groups_te_flat = groups_te.to_numpy().flatten()

# Calculate multiple k values
# k_values = [1, 3, 5, 10]
# print("\nTrain Set Performance:")
# for k in k_values:
#     hr = hitrate_at_k(y_tr_flat, xgb_tr_preds, groups_tr_flat, k)
#     print(f"HitRate@{k}: {hr:.6f}")

# print("\nValidation Set Performance:")
# for k in k_values:
#     hr = hitrate_at_k(y_va_flat, xgb_va_preds, groups_va_flat, k)
#     print(f"HitRate@{k}: {hr:.6f}")

# print("\nTest Set Performance:")
# for k in k_values:
#     hr = hitrate_at_k(y_te_flat, xgb_te_preds, groups_te_flat, k)
#     print(f"HitRate@{k}: {hr:.6f}")

# # Store primary metric
# xgb_hr3 = hitrate_at_3(y_va_flat, xgb_va_preds, groups_va_flat)
# print(f"\n*** Primary Validation HitRate@3: {xgb_hr3:.6f} ***")


# # ENHANCED FEATURE IMPORTANCE ANALYSIS
# print("\n=== Enhanced Feature Importance Analysis ===")

# # Get multiple importance types
# gain_importance = xgb_model.get_score(importance_type='gain')
# # freq_importance = xgb_model.get_score(importance_type='frequency')
# cover_importance = xgb_model.get_score(importance_type='cover')

# # Create comprehensive importance dataframe
# all_features = set(gain_importance.keys()) | set(cover_importance.keys())
# importance_data = []

# for feature in all_features:
#     importance_data.append({
#         'feature': feature,
#         'gain': gain_importance.get(feature, 0),
#         # 'frequency': freq_importance.get(feature, 0),
#         'cover': cover_importance.get(feature, 0)
#     })

# importance_df = pl.DataFrame(importance_data).sort('gain', descending=True)

# print("\nTop 30 Features by Gain:")
# print(importance_df.head(30).to_pandas().to_string(index=False))

# # Feature categories analysis
# feature_categories = {
#     'price': ['price', 'cost', 'fee', 'fare'],
#     'time': ['time', 'hour', 'day', 'duration', 'date'],
#     'route': ['route', 'airport', 'carrier', 'flight'],
#     'ranking': ['rank', 'pct', 'ratio'],
#     'group': ['group', 'size'],
#     'service': ['cabin', 'baggage', 'vip', 'class']
# }

# category_importance = {cat: 0 for cat in feature_categories.keys()}
# category_importance['other'] = 0

# for row in importance_df.to_dicts():
#     feature_name = row['feature'].lower()
#     categorized = False
    
#     for category, keywords in feature_categories.items():
#         if any(keyword in feature_name for keyword in keywords):
#             category_importance[category] += row['gain']
#             categorized = True
#             break
    
#     if not categorized:
#         category_importance['other'] += row['gain']

# print("\nFeature Importance by Category:")
# for category, importance in sorted(category_importance.items(), key=lambda x: x[1], reverse=True):
#     print(f"{category.capitalize()}: {importance:.2f}")


def re_rank(test: pl.DataFrame, submission_xgb: pl.DataFrame, penalty_factor=0.1):
    """Enhanced re-ranking function with improved flight deduplication"""
    
    # Enhanced flight comparison columns
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
        "legs0_segments0_arrivalTo_airport_iata",
        "legs1_segments0_arrivalTo_airport_iata",
        "legs0_segments0_marketingCarrier_code",
        "legs1_segments0_marketingCarrier_code",
    ]

    # Ensure columns exist and convert to string
    available_cols = [c for c in COLS_TO_COMPARE if c in test.columns]
    test_processed = test.with_columns(
        [pl.col(c).cast(str).fill_null("NULL") for c in available_cols]
    )

    df = submission_xgb.join(test_processed, on=["Id", "ranker_id"], how="left")

    # Create comprehensive flight hash
    if len(available_cols) >= 6:  # Minimum required columns
        hash_expr = pl.concat_str([pl.col(c) for c in available_cols[:10]], separator="_")
    else:
        # Fallback to basic hash
        hash_expr = (
            pl.col("legs0_departureAt").cast(str).fill_null("NULL") + "_" +
            pl.col("legs0_arrivalAt").cast(str).fill_null("NULL") + "_" +
            pl.col("legs1_departureAt").cast(str).fill_null("NULL") + "_" +
            pl.col("legs1_arrivalAt").cast(str).fill_null("NULL")
        )
    
    df = df.with_columns(hash_expr.alias("flight_hash"))

    # Enhanced scoring logic
    df = df.with_columns([
        pl.max("pred_score").over(["ranker_id", "flight_hash"]).alias("max_score_same_flight"),
        pl.count().over(["ranker_id", "flight_hash"]).alias("duplicate_count")
    ])

    # Apply penalty with consideration for duplicate count
    df = df.with_columns(
        (
            pl.col("pred_score") - 
            penalty_factor * (pl.col("max_score_same_flight") - pl.col("pred_score")) *
            pl.col("duplicate_count").clip(upper_bound=5) / 5.0  # Scale penalty by duplicates
        ).alias("reorder_score")
    )

    # Re-rank with enhanced score
    df = df.with_columns(
        pl.col("reorder_score")
        .rank(method="ordinal", descending=True)
        .over("ranker_id")
        .cast(pl.Int32)
        .alias("new_selected")
    )

    return df.select(["Id", "ranker_id", "new_selected", "pred_score", "reorder_score", "duplicate_count"])

# ENHANCED SUBMISSION GENERATION WITH ENSEMBLE

# Use ensemble predictions for submission
submission_xgb = (
    test.select(['Id', 'ranker_id'])
    .with_columns(pl.Series('pred_score', xgb_te_preds))  # Using ensemble predictions
    .with_columns(
        pl.col('pred_score')
        .rank(method='ordinal', descending=True)
        .over('ranker_id')
        .cast(pl.Int32)
        .alias('selected')
    )
    .select(['Id', 'ranker_id', 'selected', 'pred_score'])
)

print("Applying enhanced re-ranking...")
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





submission_xgb


submission_xgb.write_csv('submission.csv')

