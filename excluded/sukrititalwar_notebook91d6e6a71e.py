!pip install lightgbm pyarrow fastparquet polars


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import gc
import polars as pl # Import Polars

# Define the paths to your data
TRAIN_PATH = '/kaggle/input/aeroclub-recsys-2025/train.parquet'
TEST_PATH = '/kaggle/input/aeroclub-recsys-2025/test.parquet'
SUBMISSION_PATH = '/kaggle/input/aeroclub-recsys-2025/sample_submission.parquet'

# Define a strict list of columns to load with Polars to save memory upfront
polars_load_cols = [
    'Id', 'ranker_id', 'selected',
    'totalPrice', 'taxes',
    'requestDate',
    'legs0_departureAt', 'legs0_arrivalAt', 'legs0_duration',
    'legs1_departureAt', 'legs1_arrivalAt', 'legs1_duration',
    'searchRoute',
    # Add other simple, directly usable columns you need from the original dataset here.
]

# Define the schema for problematic non-datetime columns explicitly.
# CRITICAL: DO NOT include requestDate or other datetime columns here. Let Polars infer them.
custom_schema = {
    "Id": pl.Int64, # Assuming Id is a large integer (e.g., 100, 101)
    "ranker_id": pl.Utf8, # ranker_id is typically a string identifier (e.g., 'abc123')
    "selected": pl.Int64, # Corrected based on previous error: must be Int64 (0 or 1)
    
    "totalPrice": pl.Float64, # Ensure these are numeric
    "taxes": pl.Float64,      # Ensure these are numeric
    
    "legs0_duration": pl.Float64, # Durations are numerical, ensure float
    "legs1_duration": pl.Float64,
    
    "searchRoute": pl.Utf8, # Routes are strings
}


try:
    print("Loading train.parquet with Polars (selected columns and explicit schema for non-datetimes)...")
    train_pl = pl.read_parquet(TRAIN_PATH, columns=polars_load_cols, schema=custom_schema, rechunk=False)
    print(f"Train data loaded into Polars DataFrame. Shape: {train_pl.shape}")

    print("\nLoading test.parquet with Polars (selected columns and explicit schema for non-datetimes)...")
    test_load_cols = [col for col in polars_load_cols if col != 'selected']
    test_custom_schema = {k: v for k, v in custom_schema.items() if k != 'selected'}
    
    test_pl = pl.read_parquet(TEST_PATH, columns=test_load_cols, schema=test_custom_schema, rechunk=False)
    print(f"Test data loaded into Polars DataFrame. Shape: {test_pl.shape}")

    sample_submission_df = pd.read_parquet(SUBMISSION_PATH)

except Exception as e:
    print(f"Error loading data with Polars: {e}")
    print("Please ensure the dataset path is correct and accessible.")
    print("Also verify that the `custom_schema` matches the actual data types in the parquet files.")

gc.collect()


def feature_engineer_polars(df_pl: pl.DataFrame) -> pl.DataFrame:
    """
    Creates new features using Polars expressions, assuming core numeric columns are already numeric.
    """
    
    expressions = []

    # 1. Datetime Conversions (make robust to missing columns and simplified format)
    def create_dt_expr(col_name):
        return (
            pl.col(col_name)
            .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
            .alias(f"{col_name}_dt")
        ) if col_name in df_pl.columns else pl.lit(None, dtype=pl.Datetime).alias(f"{col_name}_dt")

    expressions.append(create_dt_expr("requestDate"))
    expressions.append(create_dt_expr("legs0_departureAt"))
    expressions.append(create_dt_expr("legs0_arrivalAt"))
    expressions.append(create_dt_expr("legs1_departureAt"))
    expressions.append(create_dt_expr("legs1_arrivalAt"))

    # 2. Time-based features
    if "legs0_departureAt_dt" in df_pl.columns and "requestDate_dt" in df_pl.columns:
        expressions.append(
            (pl.col("legs0_departureAt_dt") - pl.col("requestDate_dt"))
            .dt.total_seconds()
            .cast(pl.Float32)
            .fill_null(0)
            .truediv(24 * 3600)
            .alias("booking_lead_time_days")
        )
    else:
        expressions.append(pl.lit(np.nan, dtype=pl.Float32).alias("booking_lead_time_days"))
    
    if "legs0_departureAt_dt" in df_pl.columns:
        expressions.append(
            pl.col("legs0_departureAt_dt").dt.hour().cast(pl.UInt8).alias("departure_hour")
        )
        expressions.append(
            pl.col("legs0_departureAt_dt").dt.weekday().cast(pl.UInt8).alias("departure_day_of_week")
        )
    else:
        expressions.append(pl.lit(np.nan, dtype=pl.UInt8).alias("departure_hour"))
        expressions.append(pl.lit(np.nan, dtype=pl.UInt8).alias("departure_day_of_week"))


    # 3. Price-based features (totalPrice and taxes are now guaranteed numeric by schema)
    if "totalPrice" in df_pl.columns and "taxes" in df_pl.columns:
        expressions.append(
            (pl.col("totalPrice") - pl.col("taxes")).alias("price_before_tax")
        )
    else:
        expressions.append(pl.lit(np.nan, dtype=pl.Float32).alias("price_before_tax"))
    
    # Total duration (legs0_duration and legs1_duration are now guaranteed numeric by schema)
    if "legs0_duration" in df_pl.columns and "legs1_duration" in df_pl.columns:
        total_duration_expr = pl.col("legs0_duration").fill_null(0) + pl.col("legs1_duration").fill_null(0)
        expressions.append(
            pl.col("totalPrice").truediv(total_duration_expr.replace(0, pl.lit(np.nan))).alias("price_per_minute")
        )
    else:
        expressions.append(pl.lit(np.nan, dtype=pl.Float32).alias("price_per_minute"))


    # 4. Route-based features
    if "searchRoute" in df_pl.columns:
        expressions.append(
            pl.col("searchRoute").str.contains("/").cast(pl.UInt8).alias("is_round_trip")
        )
        expressions.append(
            pl.col("searchRoute").str.split("/").arr.get(0).alias("origin_airport")
        )
        expressions.append(
            pl.when(pl.col("searchRoute").str.contains("/")).then(pl.col("searchRoute").str.split("/").arr.get(1))
            .otherwise(pl.col("searchRoute").str.split("/").arr.get(0)).alias("destination_airport")
        )
    else:
        expressions.append(pl.lit(np.nan, dtype=pl.UInt8).alias("is_round_trip"))
        expressions.append(pl.lit("UNKNOWN", dtype=pl.Utf8).alias("origin_airport"))
        expressions.append(pl.lit("UNKNOWN", dtype=pl.Utf8).alias("destination_airport"))


    # Apply all transformations
    df_pl = df_pl.with_columns(expressions)
    
    # Fill remaining NaNs for numeric columns (after all computations)
    # Note: totalPrice, taxes, legsX_duration are already filled with 0 from schema or initial FE.
    numeric_cols_to_fill = [
        "booking_lead_time_days", "departure_hour", "departure_day_of_week",
        "price_before_tax", "price_per_minute"
    ]
    
    for c in numeric_cols_to_fill:
        if c in df_pl.columns:
            df_pl = df_pl.with_columns(pl.col(c).fill_null(pl.col(c).median()).alias(c))

    # Convert specific string/object columns to Categorical in Polars for memory efficiency
    categorical_cols_pl_final = ["origin_airport", "destination_airport", "companyID", "profileId",
                               "sex", "nationality", "corporateTariffCode"]
    # Add other categorical columns if they exist from the initial load
    for c in ["legs0_segments0_marketingCarrier_code", "legs0_segments0_operatingCarrier_code",
              "legs0_segments0_departureFrom_airport_iata", "legs0_segments0_arrivalTo_airport_iata",
              "legs0_segments0_cabinClass", "legs0_segments0_aircraft_code"]:
        if c in df_pl.columns:
            categorical_cols_pl_final.append(c)

    for c in categorical_cols_pl_final:
        if c in df_pl.columns:
            df_pl = df_pl.with_columns(pl.col(c).fill_null("NULL_CATEGORY").cast(pl.Categorical).alias(c))
    
    return df_pl

print("Performing feature engineering on Polars training data...")
train_pl = feature_engineer_polars(train_pl)
print(f"Train data shape after feature engineering: {train_pl.shape}")

print("\nPerforming feature engineering on Polars test data...")
test_pl = feature_engineer_polars(test_pl)
print(f"Test data shape after feature engineering: {test_pl.shape}")

gc.collect()


print("Converting Polars DataFrames to Pandas DataFrames...")

# Convert to Pandas
train_df_pd = train_pl.to_pandas()
test_df_pd = test_pl.to_pandas()

# Drop original datetime columns as they are now replaced by engineered features
# Note: if any of these columns were not loaded (e.g., in minimal_cols setting), they won't exist
# We will explicitly drop the original datetime columns that feature_engineer_polars created new versions for.
# And drop other potentially problematic original columns if not numeric/categorical.
cols_to_drop_after_fe = [
    'requestDate', 'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt',
    'searchRoute' # Original searchRoute column
]

train_df_pd = train_df_pd.drop(columns=[col for col in cols_to_drop_after_fe if col in train_df_pd.columns])
test_df_pd = test_df_pd.drop(columns=[col for col in cols_to_drop_after_fe if col in test_df_pd.columns])


# Identify categorical features for encoding in Pandas
# Ensure we don't include Id, ranker_id, selected, or original datetimes
categorical_cols_pd = train_df_pd.select_dtypes(include=['object', 'category']).columns.tolist()
# Filter out identifiers that shouldn't be encoded
categorical_cols_pd = [col for col in categorical_cols_pd if col not in ['Id', 'ranker_id']]


print(f"Categorical columns for Label Encoding: {categorical_cols_pd}")

# Combine train and test for consistent encoding
combined_df_pd = pd.concat([train_df_pd.drop('selected', axis=1), test_df_pd], ignore_index=True)

for col in categorical_cols_pd:
    if col in combined_df_pd.columns:
        le = LabelEncoder()
        # Convert to string first to handle any mixed types that slipped through or NaNs
        combined_df_pd.loc[:, col] = combined_df_pd[col].astype(str)
        combined_df_pd.loc[:, col] = le.fit_transform(combined_df_pd[col])

# Separate back into training and testing sets
train_df_encoded = combined_df_pd.iloc[:len(train_df_pd)].copy()
test_df_encoded = combined_df_pd.iloc[len(train_df_pd):].copy()

# Add the target variable back to the training set
train_df_encoded.loc[:, 'selected'] = train_df_pd['selected']

del train_pl, test_pl, train_df_pd, test_df_pd, combined_df_pd
gc.collect()

print("Data prepared for LightGBM.")

# Define the feature set
# Exclude original identifiers and target
features = [col for col in train_df_encoded.columns if col not in [
    'Id', 'ranker_id', 'selected'
]]

# Remove any features that might have been generated but are all NaNs (e.g., if their source column wasn't loaded)
features = [f for f in features if not train_df_encoded[f].isnull().all()]
# Fill any remaining NaNs in numeric features with 0 or median (should be minimal after Polars FE)
for col in features:
    if pd.api.types.is_numeric_dtype(train_df_encoded[col]):
        train_df_encoded.loc[:, col].fillna(train_df_encoded[col].median(), inplace=True)
        test_df_encoded.loc[:, col].fillna(test_df_encoded[col].median(), inplace=True)

print(f"Number of features selected for training: {len(features)}")
print(f"Selected features: {features}")

# Prepare data for the ranking model
X_train = train_df_encoded[features]
y_train = train_df_encoded['selected']
X_test = test_df_encoded[features]

# Group by ranker_id for the ranking task
train_groups = train_df_encoded.groupby('ranker_id').size().to_numpy()

print(f"Training data shape (X_train): {X_train.shape}, (y_train): {y_train.shape}")
print(f"Test data shape (X_test): {X_test.shape}")

gc.collect()


# Identify categorical features for encoding
# Exclude original date/time columns as they've been used for features or will be dropped
# Exclude already processed identifiers
categorical_cols = [col for col in train_df.select_dtypes(include=['object', 'category']).columns.tolist() if col not in [
    'Id', 'ranker_id', 'searchRoute', # These are identifiers or transformed
    # Original datetime columns are now properly typed and handled, no need to exclude from this list
    # 'requestDate', 'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt'
]]

print(f"Categorical columns to encode: {categorical_cols}")

# Combine train and test for consistent encoding. Drop 'selected' from train_df temporarily.
# Use .copy() to prevent SettingWithCopyWarning
combined_df = pd.concat([train_df.drop('selected', axis=1), test_df], ignore_index=True)

for col in categorical_cols:
    if col in combined_df.columns:
        le = LabelEncoder()
        # Ensure column is string type before encoding to avoid errors with mixed types/NaN
        combined_df.loc[:, col] = combined_df[col].astype(str)
        combined_df.loc[:, col] = le.fit_transform(combined_df[col])

# Separate back into training and testing sets
train_df_encoded = combined_df.iloc[:len(train_df)].copy() # Use .copy() to prevent SettingWithCopyWarning
test_df_encoded = combined_df.iloc[len(train_df):].copy() # Use .copy()

# Add the target variable back to the training set
train_df_encoded.loc[:, 'selected'] = train_df['selected']

del combined_df
gc.collect()

print("Categorical features have been encoded.")

