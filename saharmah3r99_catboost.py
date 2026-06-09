import numpy as np 
import pandas as pd 
import os


!pip install polars


import polars as pl


import re


dft = pl.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet')


pl.read_parquet_schema('/kaggle/input/aeroclub-recsys-2025/train.parquet')


dft


dft= dft.drop('bySelf','sex','legs0_segments0_cabinClass','nationality','legs0_segments0_marketingCarrier_code','legs0_segments2_marketingCarrier_code','legs0_segments3_marketingCarrier_code','legs1_segments0_marketingCarrier_code','legs1_segments2_marketingCarrier_code','legs1_segments3_marketingCarrier_code')


dft = dft.drop('legs1_segments0_duration')


dft = dft.with_columns(
    TotalPrice = pl.col("totalPrice") + pl.col("taxes"))


dft['TotalPrice']


dft = dft.drop('legs0_segments0_duration','legs0_segments1_duration','legs0_segments2_duration','legs0_segments3_duration')


dft = dft.drop('legs1_segments1_duration','legs1_segments2_duration','legs1_segments3_duration')


dft['legs0_departureAt',
'legs0_arrivalAt',
'legs0_duration',
'legs1_departureAt',
'legs1_arrivalAt',
'legs1_duration']


dft= dft.with_columns(
    pl.col('legs0_departureAt',
'legs0_arrivalAt',
'legs0_duration',
'legs1_departureAt',
'legs1_arrivalAt',
'legs1_duration').str.replace(r"T", "  "))


dft['legs0_departureAt',
'legs0_arrivalAt',
'legs0_duration',
'legs1_departureAt',
'legs1_arrivalAt',
'legs1_duration']


dft= dft.with_columns(
    pl.col('legs0_departureAt',
'legs0_arrivalAt',
'legs1_departureAt',
'legs1_arrivalAt').str.to_datetime("%Y-%m-%d %H:%M:%S"))


dft['legs0_departureAt',
'legs0_arrivalAt',
'legs1_departureAt',
'legs1_arrivalAt']


dft['legs0_duration',
'legs1_duration']


dft = dft.with_columns(
    pl.col('legs0_duration',
'legs1_duration').str.replace_all(r"\.\d{2}", "")
)


dft=dft.with_columns(
    pl.col('legs0_duration',
'legs1_duration').str.to_time("%H:%M:%S"))


dft['legs0_departureAt',
'legs0_arrivalAt',
'legs0_duration',
'legs1_departureAt',
'legs1_arrivalAt',
'legs1_duration']


dftt = pl.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet')


dftt= dftt.drop('bySelf','sex','legs0_segments0_cabinClass','nationality','legs0_segments0_marketingCarrier_code','legs0_segments2_marketingCarrier_code','legs0_segments3_marketingCarrier_code','legs1_segments0_marketingCarrier_code','legs1_segments2_marketingCarrier_code','legs1_segments3_marketingCarrier_code',)


dftt = dftt.with_columns(
    TotalPrice = pl.col("totalPrice") + pl.col("taxes"))


dftt = dftt.drop('legs1_segments0_duration')


dftt = dftt.drop('legs0_segments0_duration','legs0_segments1_duration','legs0_segments2_duration','legs0_segments3_duration')


dftt = dftt.drop('legs1_segments1_duration','legs1_segments2_duration','legs1_segments3_duration')


dftt['legs0_departureAt',
'legs0_arrivalAt',
'legs0_duration',
'legs1_departureAt',
'legs1_arrivalAt',
'legs1_duration']


dftt= dftt.with_columns(
    pl.col('legs0_departureAt',
'legs0_arrivalAt',
'legs0_duration',
'legs1_departureAt',
'legs1_arrivalAt',
'legs1_duration').str.replace(r"T", "  "))


dftt= dftt.with_columns(
    pl.col('legs0_departureAt',
'legs0_arrivalAt',
'legs1_departureAt',
'legs1_arrivalAt').str.to_datetime("%Y-%m-%d %H:%M:%S"))


dftt = dftt.with_columns(
    pl.col('legs0_duration',
'legs1_duration').str.replace_all(r"\.\d{2}", "")
)


dftt=dftt.with_columns(
    pl.col('legs0_duration',
'legs1_duration').str.to_time("%H:%M:%S"))


dftt['legs0_departureAt',
'legs0_arrivalAt',
'legs0_duration',
'legs1_departureAt',
'legs1_arrivalAt',
'legs1_duration']


def preprocess(df: pl.DataFrame) -> pl.DataFrame:
    datetime_cols = ["legs0_departureAt", "legs0_arrivalAt",
                     "legs1_departureAt", "legs1_arrivalAt"]

    for col in datetime_cols:
        # Ensure datetime type
        df = df.with_columns(
            pl.col(col).cast(pl.Datetime(time_unit="us"))
        )

        # Basic datetime features
        df = df.with_columns([
            pl.col(col).dt.hour().alias(f"{col}_hour"),
            pl.col(col).dt.weekday().alias(f"{col}_dow"),
            pl.col(col).dt.month().alias(f"{col}_month"),
            (pl.col(col).dt.weekday() >= 5).cast(pl.Int8).alias(f"{col}_is_weekend"),
        ])

        # Cyclical hour encoding
        df = df.with_columns([
            (pl.col(f"{col}_hour") * (2 * np.pi) / 24).sin().alias(f"{col}_hour_sin"),
            (pl.col(f"{col}_hour") * (2 * np.pi) / 24).cos().alias(f"{col}_hour_cos"),
        ])

    # Duration to minutes
    duration_cols = ["legs0_duration", "legs1_duration"]
    for col in duration_cols:
    # If already a Duration, just convert to minutes
        if df[col].dtype == pl.Duration(time_unit="us"):
            df = df.with_columns(
                (pl.col(col).dt.seconds() / 60).alias(col)
            )
        # If it's a Time (HH:MM:SS)
        elif df[col].dtype == pl.Time:
            df = df.with_columns(
                (pl.col(col).dt.hour() * 60 +
                 pl.col(col).dt.minute() +
                 pl.col(col).dt.second() / 60).alias(col)
            )
        # If it's a string like "02:40:00"
        else:
            df = df.with_columns(
                (pl.col(col).str.strptime(pl.Time, "%H:%M:%S", strict=False).dt.hour() * 60 +
                 pl.col(col).str.strptime(pl.Time, "%H:%M:%S", strict=False).dt.minute() +
                 pl.col(col).str.strptime(pl.Time, "%H:%M:%S", strict=False).dt.second() / 60
                ).alias(col)
            )
    # Layover
    if all(c in df.columns for c in ["legs1_departureAt", "legs0_arrivalAt"]):
    
        df = df.with_columns(
            (
                pl.col("legs1_departureAt").cast(pl.Datetime("us")) -
                pl.col("legs0_arrivalAt").cast(pl.Datetime("us"))
            ).dt.total_seconds() / 60
            
        )
    return df


print("Preprocessing...")
dft = preprocess(dft)
dftt = preprocess(dftt)


x = dft['Id','TotalPrice','searchRoute','corporateTariffCode','profileId','companyID',
'legs0_duration' ,
'legs1_duration','legs0_segments0_seatsAvailable','legs0_segments1_seatsAvailable','legs0_segments2_seatsAvailable','legs0_segments3_seatsAvailable','legs1_segments0_seatsAvailable','legs1_segments1_seatsAvailable','legs1_segments2_seatsAvailable','legs1_segments3_seatsAvailable',
'miniRules0_monetaryAmount' ,
'miniRules0_percentage',
'miniRules0_statusInfos' ,
'miniRules1_monetaryAmount' ,
'miniRules1_percentage' ,
'miniRules1_statusInfos' ,
'pricingInfo_isAccessTP' ,
'pricingInfo_passengerCount','legs0_segments0_baggageAllowance_quantity','legs0_segments1_baggageAllowance_quantity','legs0_segments2_baggageAllowance_quantity','legs0_segments3_baggageAllowance_quantity','legs1_segments0_baggageAllowance_quantity','legs1_segments1_baggageAllowance_quantity','legs1_segments2_baggageAllowance_quantity','legs1_segments3_baggageAllowance_quantity'
,'legs0_segments0_arrivalTo_airport_iata','legs0_segments1_arrivalTo_airport_iata','legs0_segments2_arrivalTo_airport_iata','legs0_segments3_arrivalTo_airport_iata',
'legs1_segments0_arrivalTo_airport_iata','legs1_segments1_arrivalTo_airport_iata','legs1_segments2_arrivalTo_airport_iata','legs1_segments3_arrivalTo_airport_iata',
'legs0_segments0_arrivalTo_airport_city_iata','legs0_segments1_arrivalTo_airport_city_iata','legs0_segments2_arrivalTo_airport_city_iata','legs0_segments3_arrivalTo_airport_city_iata',
'legs1_segments0_arrivalTo_airport_city_iata','legs1_segments1_arrivalTo_airport_city_iata','legs1_segments2_arrivalTo_airport_city_iata','legs1_segments3_arrivalTo_airport_city_iata']


string_columns = [col_name for col_name, dtype in zip(x.columns, x.dtypes) if dtype == pl.Utf8]
print(string_columns)


CATEGORICAL_COLS = ['searchRoute', 'legs0_segments0_arrivalTo_airport_iata', 'legs0_segments1_arrivalTo_airport_iata', 'legs0_segments2_arrivalTo_airport_iata', 'legs0_segments3_arrivalTo_airport_iata', 'legs1_segments0_arrivalTo_airport_iata', 'legs1_segments1_arrivalTo_airport_iata', 'legs1_segments2_arrivalTo_airport_iata', 'legs1_segments3_arrivalTo_airport_iata', 'legs0_segments0_arrivalTo_airport_city_iata', 'legs0_segments1_arrivalTo_airport_city_iata', 'legs0_segments2_arrivalTo_airport_city_iata', 'legs0_segments3_arrivalTo_airport_city_iata', 'legs1_segments0_arrivalTo_airport_city_iata', 'legs1_segments1_arrivalTo_airport_city_iata', 'legs1_segments2_arrivalTo_airport_city_iata', 'legs1_segments3_arrivalTo_airport_city_iata']



xt = dftt['Id','TotalPrice','searchRoute','corporateTariffCode','profileId'
,'companyID','legs0_duration',
'legs1_duration','legs0_segments0_seatsAvailable','legs0_segments1_seatsAvailable','legs0_segments2_seatsAvailable','legs0_segments3_seatsAvailable','legs1_segments0_seatsAvailable','legs1_segments1_seatsAvailable','legs1_segments2_seatsAvailable','legs1_segments3_seatsAvailable',
'miniRules0_monetaryAmount','miniRules0_percentage','miniRules0_statusInfos',
'miniRules1_monetaryAmount','miniRules1_percentage','miniRules1_statusInfos',
'pricingInfo_isAccessTP',
'pricingInfo_passengerCount','legs0_segments0_baggageAllowance_quantity','legs0_segments1_baggageAllowance_quantity','legs0_segments2_baggageAllowance_quantity','legs0_segments3_baggageAllowance_quantity','legs1_segments0_baggageAllowance_quantity','legs1_segments1_baggageAllowance_quantity','legs1_segments2_baggageAllowance_quantity','legs1_segments3_baggageAllowance_quantity'
,'legs0_segments0_arrivalTo_airport_iata','legs0_segments1_arrivalTo_airport_iata','legs0_segments2_arrivalTo_airport_iata','legs0_segments3_arrivalTo_airport_iata',
'legs1_segments0_arrivalTo_airport_iata','legs1_segments1_arrivalTo_airport_iata','legs1_segments2_arrivalTo_airport_iata','legs1_segments3_arrivalTo_airport_iata',
'legs0_segments0_arrivalTo_airport_city_iata','legs0_segments1_arrivalTo_airport_city_iata','legs0_segments2_arrivalTo_airport_city_iata','legs0_segments3_arrivalTo_airport_city_iata',
'legs1_segments0_arrivalTo_airport_city_iata','legs1_segments1_arrivalTo_airport_city_iata','legs1_segments2_arrivalTo_airport_city_iata','legs1_segments3_arrivalTo_airport_city_iata']


TARGET_COL = "selected"
GROUP_COL = "ranker_id"
MODEL_OUT = "flight_ranker.cbm"
PREDICTIONS_OUT = "ranked_predictions.csv"


exclude_cols = [TARGET_COL, GROUP_COL]
feature_cols = [c for c in x.columns if c not in exclude_cols]

# CatBoost categorical column indices
cat_feature_indices = [feature_cols.index(c) for c in CATEGORICAL_COLS if c in feature_cols]


feature_cols


cat_feature_indices


dft['legs0_departureAt',
 'legs0_arrivalAt',
 'legs0_duration',
 'legs1_departureAt',
 'legs1_arrivalAt',
 'legs1_duration']





! pip install catboost


import numpy as np
from catboost import CatBoostRanker, Pool


# 1. Check for nulls in critical columns
print("NaNs in target:", dft[TARGET_COL].is_null().sum())
print("NaNs in group_id:", dft[GROUP_COL].is_null().sum())

# 2. Fill missing group_id with a placeholder
dft = dft.with_columns(
    pl.col(GROUP_COL).fill_null(-1)  # or some valid integer ID
)

# 3. Fill or drop missing target
dft = dft.filter(pl.col(TARGET_COL).is_not_null())  # safest for classification

# 4. CatBoost can handle NaNs in numeric features, but ensure proper type casting for cat_features
for cat_col in cat_feature_indices:
    dft = dft.with_columns(pl.col(feature_cols[cat_col]).cast(pl.Utf8))



for col in feature_cols:
    if dft[col].dtype in [pl.Int64, pl.Int32, pl.UInt32, pl.UInt64, pl.Float64, pl.Float32]:
        dft = dft.with_columns(pl.col(col).cast(pl.Float64))
    else:
        # keep categorical columns as string
        dft = dft.with_columns(pl.col(col).cast(pl.Utf8))

# Fill nulls in numeric cols with np.nan explicitly
dft = dft.fill_null(np.nan)

# Make sure group_id and label have no NaNs
dft = dft.filter(
    (pl.col(GROUP_COL).is_not_null()) &
    (pl.col(TARGET_COL).is_not_null())
)


for col in feature_cols:
    if dft[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                          pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                          pl.Float32, pl.Float64]:
        dft = dft.with_columns(pl.col(col).cast(pl.Float64))

# Replace nulls in numeric with np.nan
dft = dft.fill_null(np.nan)

# Drop rows with missing label or group_id
dft = dft.filter(
    pl.col(TARGET_COL).is_not_null() & pl.col(GROUP_COL).is_not_null()
)


import numpy as np
import polars as pl
from catboost import Pool

# 1. Ensure numeric columns are floats → np.nan
for col in feature_cols:
    if dft[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                          pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                          pl.Float32, pl.Float64]:
        dft = dft.with_columns(pl.col(col).cast(pl.Float64))

# 2. Handle categorical columns
for idx in cat_feature_indices:
    col = feature_cols[idx]
    dft = dft.with_columns(pl.col(col).cast(pl.Utf8))
    dft = dft.with_columns(pl.col(col).fill_null("__MISSING__"))

# 3. Fill remaining nulls in numeric with np.nan
dft = dft.fill_null(np.nan)

# 4. Drop rows where label or group_id is null
dft = dft.filter(
    pl.col(TARGET_COL).is_not_null() & pl.col(GROUP_COL).is_not_null()
)

# 5. Create Pool
train_pool = Pool(
    data=dft.select(feature_cols).to_numpy(),
    label=dft[TARGET_COL].to_numpy(),
    group_id=dft[GROUP_COL].to_numpy(),
    cat_features=cat_feature_indices
)



# 1. Check for nulls in critical columns
print("NaNs in group_id:", dftt[GROUP_COL].is_null().sum())

# 2. Fill missing group_id with a placeholder
dftt = dftt.with_columns(
    pl.col(GROUP_COL).fill_null(-1)  # or some valid integer ID
)


# 4. CatBoost can handle NaNs in numeric features, but ensure proper type casting for cat_features
for cat_col in cat_feature_indices:
    dftt = dftt.with_columns(pl.col(feature_cols[cat_col]).cast(pl.Utf8))



for col in feature_cols:
    if dftt[col].dtype in [pl.Int64, pl.Int32, pl.UInt32, pl.UInt64, pl.Float64, pl.Float32]:
        dftt = dftt.with_columns(pl.col(col).cast(pl.Float64))
    else:
        # keep categorical columns as string
        dftt = dftt.with_columns(pl.col(col).cast(pl.Utf8))

# Fill nulls in numeric cols with np.nan explicitly
dftt = dftt.fill_null(np.nan)

# Make sure group_id and label have no NaNs
dftt = dftt.filter(
    (pl.col(GROUP_COL).is_not_null()) )
   



for col in feature_cols:
    if dftt[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                          pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                          pl.Float32, pl.Float64]:
        dftt = dftt.with_columns(pl.col(col).cast(pl.Float64))

# Replace nulls in numeric with np.nan
dftt = dftt.fill_null(np.nan)

# Drop rows with missing label or group_id
dftt = dftt.filter(
    pl.col(GROUP_COL).is_not_null()
)


# 1. Ensure numeric columns are floats → np.nan
for col in feature_cols:
    if dftt[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                          pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                          pl.Float32, pl.Float64]:
        dftt = dftt.with_columns(pl.col(col).cast(pl.Float64))

# 2. Handle categorical columns
for idx in cat_feature_indices:
    col = feature_cols[idx]
    dftt = dftt.with_columns(pl.col(col).cast(pl.Utf8))
    dftt = dftt.with_columns(pl.col(col).fill_null("__MISSING__"))

# 3. Fill remaining nulls in numeric with np.nan
dftt = dftt.fill_null(np.nan)

# 4. Drop rows where label or group_id is null
dftt = dftt.filter(
     pl.col(GROUP_COL).is_not_null()
)
test_pool = Pool(
    data=dftt.select(feature_cols).to_numpy(),
    group_id=dftt[GROUP_COL].to_numpy(),
    cat_features=cat_feature_indices
)



print("Training CatBoostRanker...")
model = CatBoostRanker(
    iterations=300,
    learning_rate=0.05,
    depth=8,
    loss_function="YetiRank",
    eval_metric="NDCG",
    random_seed=42,
    verbose=100
)
model.fit(train_pool)
model.save_model('/kaggle/working/MODEL_OUT')


# Predict & rank
print("Predicting...")
scores = model.predict(test_pool)
test_df = dftt.with_columns(pl.Series("score", scores))




print("Ranking per group...")
test_df = test_df.with_columns(
    pl.col("score").rank("dense", descending=True).over(GROUP_COL).alias("rank")
)

# Save
test_df.sort(['Id',GROUP_COL, "rank"]).write_csv('/kaggle/working/PREDICTIONS_OUT')
print(f"Predictions saved to {PREDICTIONS_OUT}")


import  pandas as pd
z = pd.read_csv('ranked_predictions.csv')
z.shape
z.head(20)

