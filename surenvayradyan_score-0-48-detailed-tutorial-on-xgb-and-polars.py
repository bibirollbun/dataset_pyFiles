# %%capture
# !pip install -U xgboost
# !pip install -U lightgbm
# !pip install -U polars


# Base
import os
import subprocess
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold
from itertools import product
from sklearn.model_selection import GroupShuffleSplit
from itertools import chain

# Plots
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pl.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet').drop('__index_level_0__')
test = pl.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet').drop('__index_level_0__').with_columns(pl.lit(0, dtype=pl.Int64).alias("selected"))

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Merge data
train = train.with_columns(pl.lit(1).alias("is_train"))
test = test.with_columns([
    pl.lit(0).alias("is_train"),
    pl.lit(None).cast(pl.Int64).alias("selected")
])

combined = pl.concat([train, test], how="diagonal")
print("Combined shape:", combined.shape)


selected_analysis = combined.filter(pl.col("is_train") == 1).group_by("ranker_id").agg([
    pl.sum("selected").alias("selected_count"),
    pl.len().alias("total_options")
]).select([
    pl.mean("total_options").alias("avg_options_per_session"),
    pl.min("total_options").alias("min_options"),
    pl.max("total_options").alias("max_options"),
    pl.len().alias("total_sessions")
])

print("Selected analysis:")
print(selected_analysis)


def create_advanced_features(df):
    """Creating Features"""
    
    # Duration conversion
    def dur_to_min(col_name):
        col = pl.col(col_name)
        # days, hours, min
        days = col.str.extract(r"^(\d+)\.", 1).cast(pl.Int64).fill_null(0) * 1440
        time_str = pl.when(col.str.contains(r"^\d+\.")).then(
            col.str.replace(r"^\d+\.", "")
        ).otherwise(col)
        hours = time_str.str.extract(r"^(\d+):", 1).cast(pl.Int64).fill_null(0) * 60
        minutes = time_str.str.extract(r":(\d+):", 1).cast(pl.Int64).fill_null(0)
        return (days + hours + minutes).fill_null(0)

    # Processing duration columns
    dur_cols = ["legs0_duration", "legs1_duration"] + [
        f"legs{l}_segments{s}_duration" for l in (0, 1) for s in (0, 1, 2, 3)
    ]
    dur_exprs = [dur_to_min(c).alias(c) for c in dur_cols if c in df.columns]

    # Apply duration transformations
    if dur_exprs:
        df = df.with_columns(dur_exprs)

    # Pre-processing of marketing carriers
    mc_cols = [f'legs{l}_segments{s}_marketingCarrier_code' for l in (0, 1) for s in range(4)]
    mc_exists = [col for col in mc_cols if col in df.columns]

    # Basic transformations
    df = df.with_columns([
        # Price indicators
        (pl.col("totalPrice") / (pl.col("taxes") + 1)).alias("price_per_tax"),
        (pl.col("taxes") / (pl.col("totalPrice") + 1)).alias("tax_rate"),
        pl.col("totalPrice").log1p().alias("log_price"),
        
        # Signs of duration
        (pl.col("legs0_duration").fill_null(0) + pl.col("legs1_duration").fill_null(0)).alias("total_duration"),
        pl.when(pl.col("legs1_duration").fill_null(0) > 0)
          .then(pl.col("legs0_duration") / (pl.col("legs1_duration") + 1))
          .otherwise(1.0).alias("duration_ratio"),
        
        # Trip type
        (pl.col("legs1_duration").is_null() | 
         (pl.col("legs1_duration") == 0) | 
         pl.col("legs1_segments0_departureFrom_airport_iata").is_null()).cast(pl.Int32).alias("is_one_way"),
        
        # Total number of segments
        (pl.sum_horizontal([pl.col(col).is_not_null().cast(pl.UInt8) for col in mc_exists]) 
         if mc_exists else pl.lit(0)).alias("total_segments_count"),
        
        # Signs of a frequency program
        (pl.col("frequentFlyer").fill_null("").str.count_matches("/") + 
         (pl.col("frequentFlyer").fill_null("") != "").cast(pl.Int32)).alias("n_ff_programs"),
        
        # Binary features
        pl.col("corporateTariffCode").is_not_null().cast(pl.Int32).alias("has_corporate_tariff"),
        (pl.col("pricingInfo_isAccessTP") == 1).cast(pl.Int32).alias("has_access_tp"),
        
        # Baggage and fees
        (pl.col("legs0_segments0_baggageAllowance_quantity").fill_null(0) + 
         pl.col("legs1_segments0_baggageAllowance_quantity").fill_null(0)).alias("baggage_total"),
        (pl.col("miniRules0_monetaryAmount").fill_null(0) + 
         pl.col("miniRules1_monetaryAmount").fill_null(0)).alias("total_fees"),
        
        # Popular routes
        pl.col("searchRoute").is_in(["MOWLED/LEDMOW", "LEDMOW/MOWLED", "MOWLED", "LEDMOW", "MOWAER/AERMOW"])
          .cast(pl.Int32).alias("is_popular_route"),
        
        # Service class
        pl.mean_horizontal(["legs0_segments0_cabinClass", "legs1_segments0_cabinClass"]).alias("avg_cabin_class"),
        (pl.col("legs0_segments0_cabinClass").fill_null(0) - 
         pl.col("legs1_segments0_cabinClass").fill_null(0)).alias("cabin_class_diff"),
    ])

    # Count segments for each leg
    seg_exprs = []
    for leg in (0, 1):
        seg_cols = [f"legs{leg}_segments{s}_duration" for s in range(4) if f"legs{leg}_segments{s}_duration" in df.columns]
        if seg_cols:
            seg_exprs.append(
                pl.sum_horizontal([pl.col(c).is_not_null() for c in seg_cols])
                  .cast(pl.Int32).alias(f"n_segments_leg{leg}")
            )
        else:
            seg_exprs.append(pl.lit(0).cast(pl.Int32).alias(f"n_segments_leg{leg}"))

    # Add segment counting
    df = df.with_columns(seg_exprs)

    # Derived features
    df = df.with_columns([
        (pl.col("n_segments_leg0") + pl.col("n_segments_leg1")).alias("total_segments"),
        (pl.col("n_segments_leg0") == 1).cast(pl.Int32).alias("is_direct_leg0"),
        pl.when(pl.col("is_one_way") == 1).then(0)
          .otherwise((pl.col("n_segments_leg1") == 1).cast(pl.Int32)).alias("is_direct_leg1"),
    ])

    # Additional signs
    df = df.with_columns([
        (pl.col("is_direct_leg0") & pl.col("is_direct_leg1")).cast(pl.Int32).alias("both_direct"),
        ((pl.col("isVip") == 1) | (pl.col("n_ff_programs") > 0)).cast(pl.Int32).alias("is_vip_freq"),
        (pl.col("baggage_total") > 0).cast(pl.Int32).alias("has_baggage"),
        (pl.col("total_fees") > 0).cast(pl.Int32).alias("has_fees"),
        (pl.col("total_fees") / (pl.col("totalPrice") + 1)).alias("fee_rate"),
        pl.col("Id").count().over("ranker_id").alias("group_size"),
    ])

    # Flags of major airlines
    if "legs0_segments0_marketingCarrier_code" in df.columns:
        df = df.with_columns(
            pl.col("legs0_segments0_marketingCarrier_code").is_in(["SU", "S7", "U6", "Aeroflot"])
              .cast(pl.Int32).alias("is_major_carrier")
        )
    else:
        df = df.with_columns(pl.lit(0).alias("is_major_carrier"))

    df = df.with_columns(pl.col("group_size").log1p().alias("group_size_log"))

    # Temporary signs
    time_exprs = []
    time_cols = ["legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt"]
    
    for col_name in time_cols:
        if col_name in df.columns:
            # Convert the string to datetime
            dt_col = pl.col(col_name).str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f", strict=False)
            h = dt_col.dt.hour().fill_null(12)
            time_exprs.extend([
                h.alias(f"{col_name}_hour"),
                dt_col.dt.weekday().fill_null(0).alias(f"{col_name}_weekday"),
                (((h >= 6) & (h <= 9)) | ((h >= 17) & (h <= 20))).cast(pl.Int32).alias(f"{col_name}_business_time"),
                # Business hours (9-17)
                ((h >= 9) & (h <= 17)).cast(pl.Int32).alias(f"{col_name}_business_hours"),
                # Night time
                ((h >= 22) | (h <= 6)).cast(pl.Int32).alias(f"{col_name}_night_time")
            ])
    
    if time_exprs:
        df = df.with_columns(time_exprs)

    # Ranking features
    df = df.with_columns([
        pl.col("group_size").log1p().alias("group_size_log"),
    ])

    # Base Ranks
    rank_exprs = []
    for col_name, alias in [("totalPrice", "price"), ("total_duration", "duration")]:
        if col_name in df.columns:
            rank_exprs.append(
                pl.col(col_name).rank().over("ranker_id").alias(f"{alias}_rank")
            )

    
    price_exprs = [
        (pl.col("totalPrice").rank("average").over("ranker_id") / 
         pl.col("totalPrice").count().over("ranker_id")).alias("price_pct_rank"),
        (pl.col("totalPrice") == pl.col("totalPrice").min().over("ranker_id")).cast(pl.Int32).alias("is_cheapest"),
        ((pl.col("totalPrice") - pl.col("totalPrice").median().over("ranker_id")) / 
         (pl.col("totalPrice").std().over("ranker_id") + 1)).alias("price_from_median"),
        (pl.col("total_segments") == pl.col("total_segments").min().over("ranker_id")).cast(pl.Int32).alias("is_min_segments"),
    ]

    if rank_exprs:
        df = df.with_columns(rank_exprs + price_exprs)

    # Cheapest direct flight
    direct_cheapest = (
        df.filter(pl.col("is_direct_leg0") == 1)
        .group_by("ranker_id")
        .agg(pl.col("totalPrice").min().alias("min_direct_price"))
    )

    df = df.join(direct_cheapest, on="ranker_id", how="left").with_columns([
        ((pl.col("is_direct_leg0") == 1) & 
         (pl.col("totalPrice") == pl.col("min_direct_price"))).cast(pl.Int32).fill_null(0).alias("is_direct_cheapest")
    ]).drop("min_direct_price")
    

    return df

combined_features = create_advanced_features(combined)


def add_business_logic_features(df):
    """Adding business logic to features"""
    
    # First we create a temporary column for the class categories
    df = df.with_columns([
        pl.when(pl.col("legs0_segments0_cabinClass") == 1.0).then(pl.lit("economy"))
         .when(pl.col("legs0_segments0_cabinClass") == 2.0).then(pl.lit("business"))
         .when(pl.col("legs0_segments0_cabinClass") == 4.0).then(pl.lit("premium"))
         .otherwise(pl.lit("other")).alias("cabin_class_temp")
    ])
    
    # Check which columns are datetime
    datetime_cols = []
    string_cols = []
    
    for col in ["legs0_departureAt", "requestDate"]:
        if col in df.columns:
            if df[col].dtype == pl.Datetime:
                datetime_cols.append(col)
            else:
                string_cols.append(col)
    
    print(f"Datetime columns: {datetime_cols}")
    print(f"String columns: {string_cols}")
    
    datetime_exprs = []
    
    # For string columns - parse
    for col in string_cols:
        datetime_exprs.append(
            pl.col(col).str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f", strict=False)
            .alias(f"{col}_dt")
        )
    
    if datetime_exprs:
        df = df.with_columns(datetime_exprs)
    
    # Determine which columns to use for calculations
    departure_col = "legs0_departureAt" if "legs0_departureAt" in datetime_cols else "legs0_departureAt_dt"
    request_col = "requestDate" if "requestDate" in datetime_cols else "requestDate_dt"
    
    df = df.with_columns([
        # Time before departure (in days) - only if both columns exist
        pl.when(pl.col(departure_col).is_not_null() & pl.col(request_col).is_not_null())
          .then((pl.col(departure_col) - pl.col(request_col)).dt.total_days())
          .otherwise(None)
          .alias("days_to_departure"),
        
        ((pl.col("legs0_departureAt_hour") >= 6) & (pl.col("legs0_departureAt_hour") <= 9)).cast(pl.Int32)
         .alias("departure_early_morning"),
        
        ((pl.col("legs0_departureAt_hour") >= 17) & (pl.col("legs0_departureAt_hour") <= 21)).cast(pl.Int32)
         .alias("departure_evening"),
        
        pl.col(departure_col).dt.month().alias("departure_month"),
        
        (pl.col("totalPrice") / (pl.col("total_duration") / 60 + 1)).alias("price_per_flying_hour"),
        
        (pl.col("legs0_segments0_seatsAvailable").fill_null(0) + 
         pl.col("legs1_segments0_seatsAvailable").fill_null(0)).alias("total_seats_available"),
        
        (pl.col("legs0_segments0_baggageAllowance_quantity").fill_null(0) >= 20).cast(pl.Int32)
         .alias("has_good_baggage"),
        
        (pl.col("legs0_segments0_baggageAllowance_quantity").fill_null(0) > 0).cast(pl.Int32)
         .alias("has_any_baggage"),
        
        pl.col("cabin_class_temp").alias("cabin_class_category"),
        
    ]).drop("cabin_class_temp") 
    
    temp_dt_cols = [col for col in df.columns if col.endswith("_dt")]
    if temp_dt_cols:
        df = df.drop(temp_dt_cols)
    
    return df

combined_features = add_business_logic_features(combined_features)


# Split back into train/test
train_data = combined_features.filter(pl.col("is_train") == 1)
test_data = combined_features.filter(pl.col("is_train") == 0)

print(f"Train shape: {train_data.shape}")
print(f"Test shape: {test_data.shape}")

# Let's look at the balance of classes
class_balance = train_data.select([
    pl.sum("selected").alias("positive_samples"),
    pl.len().alias("total_samples")
]).with_columns([
    (pl.col("positive_samples") / pl.col("total_samples") * 100).alias("positive_rate_percent")
])

print("Class balance:")
print(class_balance)


# Let's define numerical features (excluding identifiers and categorical ones)
exclude_cols = [
    "Id", "ranker_id", "profileId", "companyID", "searchRoute", 
    "legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt",
    "requestDate", "corporateTariffCode", "frequentFlyer", "nationality",
    "legs0_segments0_marketingCarrier_code", "legs1_segments0_marketingCarrier_code",
    "sex", "selected", "is_train", "cabin_class_category"
]

# Add columns that can contain rows
string_cols = [col for col in train_data.columns if train_data[col].dtype == pl.Utf8]
exclude_cols.extend([col for col in string_cols if col not in exclude_cols])

feature_cols = [col for col in train_data.columns if col not in exclude_cols and col != "selected"]
print(f"Count features: {len(feature_cols)}")
print("Features:", feature_cols[:20])


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


print("Preparing data for XGBoost...")


X_train = train_data.select(feature_cols)
y_train_array = train_data.select("selected").to_numpy().ravel()
groups_train = train_data.select("ranker_id")

# Calculate the sizes of groups
group_sizes_train = (groups_train
                    .group_by('ranker_id', maintain_order=True)
                    .agg(pl.len())['len']
                    .to_numpy())

print(f"Train shape: {X_train.shape}")
print(f"Group sizes length: {len(group_sizes_train)}")


dtrain = xgb.DMatrix(
    X_train.to_numpy(), 
    label=y_train_array, 
    group=group_sizes_train
)

xgb_params = {
    'objective': 'rank:pairwise',  
    'eval_metric': 'ndcg@3',       
    'max_depth': 8,                
    'min_child_weight': 20,       
    'subsample': 0.8,             
    'colsample_bytree': 0.8,       
    'reg_lambda': 10.0,            
    'learning_rate': 0.05,         
    'random_state': 42,
    'n_jobs': -1                  
}


print("Training XGBoost model...")
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=1000,
    verbose_eval=100
)

print("Model complite!")


# Predictions on train
print("Predictions on train...")
train_predictions_xgb = xgb_model.predict(dtrain)

# Metrics
hitrate_xgb = hitrate_at_3(y_train_array, train_predictions_xgb, groups_train.to_numpy().ravel())
print(f"XGBoost HitRate@3 on train: {hitrate_xgb:.4f}")

# Predictions on test
print("Predictions on test...")
X_test = test_data.select(feature_cols)
dtest = xgb.DMatrix(X_test.to_numpy())
test_predictions_xgb = xgb_model.predict(dtest)


def create_submission_xgb(test_df, predictions):
    """Create sub"""
    
    test_with_pred = test_df.with_columns([
        pl.Series("prediction_score", predictions)
    ])
    
    result_frames = []
    
    for ranker_data in test_with_pred.partition_by("ranker_id"):
        ranked_group = (ranker_data
                       .sort("prediction_score", descending=True)
                       .with_row_index("rank", offset=1)
                       .select([
                           "Id", 
                           "ranker_id", 
                           pl.col("rank").alias("selected")
                       ]))
        result_frames.append(ranked_group)
    
    submission = pl.concat(result_frames).sort("Id")
    
    return submission

final_submission = create_submission_xgb(test_data, test_predictions_xgb)
print(f"Submission done: {final_submission.shape}")

final_submission.write_csv("submission_xgb.csv")
print("Submission save in submission_xgb.csv!")


importance_dict = xgb_model_final.get_score(importance_type='gain')
importance_df = pl.DataFrame([
    {'feature': k, 'importance': v} 
    for k, v in importance_dict.items()
]).sort('importance', descending=True)

print("Топ-20 importance features:")
print(importance_df.head(20))







