import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import gc
from tqdm.notebook import tqdm

# --- Configuration ---
# Define the base directory where the competition data is located.
# This path typically points to the input directory in a Kaggle environment.
DATA_DIR = Path("/kaggle/input/aeroclub-recsys-2025/")

# Define the full paths to the training, testing, and sample submission files.
TRAIN_PATH = DATA_DIR / "train.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.parquet"

# Define the output path for the final submission file.
# In Kaggle, "/kaggle/working/" is the designated directory for output files.
OUTPUT_PATH = Path("/kaggle/working/submission.parquet")

# --- Memory Optimization Function ---
# This function is designed to reduce the memory footprint of a Pandas DataFrame.
# Large datasets can quickly consume available RAM, leading to crashes or slow processing.
# By downcasting numerical columns to the smallest possible data types (e.g., int64 to int8),
# we can significantly optimize memory usage without losing data integrity.

def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2 # Calculate initial memory usage in MB
    for col in tqdm(df.columns, desc="Reducing Memory"):
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                # Check if integer column can be downcasted to a smaller integer type
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Check if float column can be downcasted to a smaller float type
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2 # Calculate final memory usage in MB
    if verbose: 
        print(f"Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    return df


def create_base_features(df):
    """Engineers a rich set of features from the raw data for individual flight options."""
    
    # Convert datetime columns to proper datetime objects. Errors are coerced to NaT (Not a Time).
    for col in ["legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt", "requestDate"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ensure numerical columns are of a numeric type. This is a crucial fix from the original script
    # to prevent errors in arithmetic operations if these columns were loaded as objects.
    numeric_cols = ["legs0_duration", "legs1_duration", "totalPrice", "taxes", "legs0_segments0_seatsAvailable"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Time-based Features ---
    # Extract hour and day of the week from departure and arrival times.
    # These can capture daily and weekly travel patterns.
    df["dep_hour_0"] = df["legs0_departureAt"].dt.hour
    df["dep_dayofweek_0"] = df["legs0_departureAt"].dt.dayofweek
    df["arr_hour_0"] = df["legs0_arrivalAt"].dt.hour
    
    # For return flights, fill NaN values with -1 and cast to int8 to save memory.
    # -1 indicates no return flight (single trip).
    df["dep_hour_1"] = df["legs1_departureAt"].dt.hour.fillna(-1).astype("int8")
    df["arr_hour_1"] = df["legs1_arrivalAt"].dt.hour.fillna(-1).astype("int8")
    
    # Extract the hour of the search request.
    df["request_hour"] = df["requestDate"].dt.hour
    
    # Calculate the time difference between when the search was made and the flight's departure.
    # This can indicate how far in advance a flight is booked, which might correlate with price sensitivity or urgency.
    df["time_to_departure_hrs"] = (df["legs0_departureAt"] - df["requestDate"]).dt.total_seconds() / 3600
    
    # --- Flight Characteristics ---
    # Determine if the flight is a round trip based on the presence of a return leg departure time.
    df["is_round_trip"] = df["legs1_departureAt"].notna().astype("int8")
    
    # Calculate the total duration of the flight, handling potential missing values by filling with 0.
    df["total_duration"] = df["legs0_duration"].fillna(0) + df["legs1_duration"].fillna(0)
    
    # Count the number of segments (connections) for each leg of the flight.
    # More segments usually mean longer travel times and potentially less convenience.
    df["num_segments_0"] = df.filter(like="legs0_segments").notna().sum(axis=1)
    df["num_segments_1"] = df.filter(like="legs1_segments").notna().sum(axis=1)
    df["total_segments"] = df["num_segments_0"] + df["num_segments_1"]
    
    # --- Pricing Features ---
    # Calculate the ratio of taxes to total price. A higher tax ratio might indicate different fare structures.
    df["tax_ratio"] = df["taxes"] / (df["totalPrice"] + 1e-6) # Add a small epsilon to prevent division by zero
    
    # Calculate price per unit of duration and per segment. These normalize price by travel effort.
    df["price_per_duration"] = df["totalPrice"] / (df["total_duration"] + 1e-6)
    df["price_per_segment"] = df["totalPrice"] / (df["total_segments"] + 1e-6)

    # --- Interaction Features ---
    # Create interaction terms by multiplying existing features. These can capture complex relationships.
    df["price_x_duration"] = df["totalPrice"] * df["total_duration"]
    df["seats_x_price"] = df["legs0_segments0_seatsAvailable"].fillna(0) * df["totalPrice"]
    
    # --- Policy and User Features ---
    # Convert boolean-like columns to integer type for consistency and memory efficiency.
    df["is_vip"] = df["isVip"].astype("int8")
    df["books_by_self"] = df["bySelf"].astype("int8")

    return df


def create_group_features(df):
    """Engineers features based on group-wise statistics within each ranker_id (search session)."""
    
    # Define a list of numerical columns for which we want to create group-wise features.
    group_agg_features = [
        "totalPrice", "total_duration", "time_to_departure_hrs", 
        "total_segments", "tax_ratio"
    ]
    
    for col in tqdm(group_agg_features, desc="Creating Group Features"):
        # --- Rank Features ---
        # Rank flights within each 'ranker_id' group based on the current feature.
        # 'dense' method assigns consecutive ranks without gaps.
        # 'ascending=True' means lower values get lower ranks (e.g., cheaper flights get rank 1 for price).
        df[f"{col}_rank_asc"] = df.groupby("ranker_id")[col].rank(method="dense", ascending=True)
        # 'ascending=False' means higher values get lower ranks (e.g., longer duration gets rank 1 for duration).
        df[f"{col}_rank_desc"] = df.groupby("ranker_id")[col].rank(method="dense", ascending=False)
        
        # --- Normalized Features (Value / Group Mean) ---
        # Calculate the mean of the current feature for each 'ranker_id' group.
        group_mean = df.groupby("ranker_id")[col].transform("mean")
        # Normalize the feature by its group mean. This shows if a flight is above or below average for its session.
        df[f"{col}_norm_by_mean"] = df[col] / (group_mean + 1e-6) # Add epsilon to avoid division by zero
        
        # --- Difference from Group Mean ---
        # Calculate the difference between the feature value and the group mean.
        # This provides an absolute measure of how a flight deviates from the average in its session.
        df[f"{col}_diff_from_mean"] = df[col] - group_mean
        
        # --- Ratio to Group Max ---
        # Calculate the maximum value of the current feature for each 'ranker_id' group.
        group_max = df.groupby("ranker_id")[col].transform("max")
        # Calculate the ratio of the feature value to the group maximum. 
        # This indicates how a flight compares to the most extreme option in its session.
        df[f"{col}_ratio_to_max"] = df[col] / (group_max + 1e-6) # Add epsilon to avoid division by zero

    return df


print("Loading data...")
# Define all columns that might be used to ensure they are loaded.
# This list is carefully curated to include all columns that will be used for feature engineering
# or as identifiers, minimizing memory usage by not loading unnecessary columns.
all_cols = [
    "Id", "ranker_id", "profileId", "companyID", "isVip", "bySelf",
    "requestDate", "searchRoute", "totalPrice", "taxes",
    "legs0_departureAt", "legs0_arrivalAt", "legs0_duration",
    "legs1_departureAt", "legs1_arrivalAt", "legs1_duration",
    "legs0_segments0_seatsAvailable", "legs0_segments0_cabinClass",
    "legs0_segments1_flightNumber", "legs1_segments0_flightNumber"
]

# Load train and test data using the defined columns.
# For the training data, we also include the 'selected' target variable.
train_cols = [c for c in all_cols if c != "selected"] + ["selected"]
test_cols = [c for c in all_cols if c != "selected"]

train_df = pd.read_parquet(TRAIN_PATH, columns=train_cols)
test_df = pd.read_parquet(TEST_PATH, columns=test_cols)

# Explicitly call garbage collector to free up memory after loading data.
# This is a good practice when dealing with large datasets.
gc.collect()

print("Feature engineering...")

# Apply the base feature engineering function to both training and testing datasets.
# This creates features that are independent of other flights within the same search session.
train_df = create_base_features(train_df)
test_df = create_base_features(test_df)
gc.collect()

# Apply the group-wise feature engineering function to both training and testing datasets.
# These features capture the relative position of each flight within its search session,
# which is crucial for ranking problems.
train_df = create_group_features(train_df)
test_df = create_group_features(test_df)
gc.collect()

# --- Categorical Feature Encoding ---
print("Encoding categorical features...")
# Define a list of categorical columns that need to be converted to numerical format.
# Label Encoding assigns a unique integer to each unique category.
categorical_cols = ["profileId", "companyID", "searchRoute", "legs0_segments0_cabinClass"]
for col in tqdm(categorical_cols, desc="Label Encoding"):
    le = LabelEncoder()
    # To ensure consistent encoding across train and test sets, we fit the encoder
    # on a combined series of both datasets. This prevents issues if a category
    # exists in test but not in train, or vice-versa.
    combined_series = pd.concat([train_df[col].astype(str), test_df[col].astype(str)])
    le.fit(combined_series)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

# --- Memory Optimization ---
# After feature engineering and encoding, apply memory reduction again.
# New columns might have been created with default data types that can be optimized.
train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)
gc.collect()


print("Preparing data for training...")

# Define the list of features to be used for training the model.
# We exclude identifiers, raw datetime columns (as we've engineered features from them),
# and columns that might have high cardinality or low signal for this baseline.
features = [col for col in train_df.columns if col not in [
    "Id", "ranker_id", "selected", 
    "legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt", "requestDate",
    "legs0_segments1_flightNumber", "legs1_segments0_flightNumber", # High cardinality, often low direct signal for baseline
    "isVip", "bySelf" # These were already converted to int8 and are now part of the feature set
]]

# --- Time-based Split for Validation ---
# A time-based split is crucial for time-series or recommendation problems
# to prevent data leakage and ensure the model generalizes to future data.
# We use the 85th percentile of `requestDate` to define our training cutoff.
train_cutoff_date = train_df["requestDate"].quantile(0.85, interpolation="nearest")

# Create boolean masks for training and validation sets based on the cutoff date.
train_idx = train_df[train_df["requestDate"] <= train_cutoff_date].index
val_idx = train_df[train_df["requestDate"] > train_cutoff_date].index

# Separate features (X) and target (y) for training and validation sets.
X_train, y_train = train_df.loc[train_idx, features], train_df.loc[train_idx, "selected"]
X_val, y_val = train_df.loc[val_idx, features], train_df.loc[val_idx, "selected"]

# --- Group Information for LightGBM LambdaRank ---
# For LambdaRank, LightGBM needs to know the number of items in each group (search session).
# We calculate the size of each `ranker_id` group for both training and validation sets.
train_groups = train_df.loc[train_idx].groupby("ranker_id").size().to_numpy()
val_groups = train_df.loc[val_idx].groupby("ranker_id").size().to_numpy()

# Create LightGBM Dataset objects.
# `group` parameter is essential for ranking objectives.
# `free_raw_data=False` prevents LightGBM from freeing the underlying data,
# which is useful if you need to access X_train/X_val later (though we delete them here).
lgb_train = lgb.Dataset(X_train, y_train, group=train_groups, free_raw_data=False)
lgb_val = lgb.Dataset(X_val, y_val, group=val_groups, reference=lgb_train, free_raw_data=False)

# Free up memory by deleting the original DataFrames and calling garbage collector.
del X_train, y_train, X_val, y_val, train_df
gc.collect()


print("Starting model training...")

# --- LightGBM Model Parameters ---
# These parameters are crucial for the model's performance and training behavior.
# - `objective`: `lambdarank` is specified for learning-to-rank tasks.
# - `metric`: `ndcg` (Normalized Discounted Cumulative Gain) is a common ranking metric,
#             closely related to HitRate@3.
# - `boosting_type`: `gbdt` (Gradient Boosting Decision Tree) is the standard boosting type.
# - `n_estimators`: Maximum number of boosting rounds.
# - `learning_rate`: Controls the step size shrinkage to prevent overfitting.
# - `num_leaves`: Maximum number of leaves in one tree. Higher values increase model complexity.
# - `max_depth`: Maximum depth of the tree. -1 means no limit.
# - `seed`: Random seed for reproducibility.
# - `n_jobs`: Number of parallel threads. -1 uses all available cores.
# - `verbose`: Controls the verbosity of training output. -1 means silent.
# - `colsample_bytree`: Fraction of features to consider at each iteration (column subsampling).
# - `subsample`: Fraction of data to sample for each tree (row subsampling).
# - `reg_alpha`, `reg_lambda`: L1 and L2 regularization terms to prevent overfitting.
# - `device_type`: Set to `gpu` to leverage GPU acceleration if available.
# - `gpu_platform_id`, `gpu_device_id`: Specific GPU device to use.
# - `max_bin`: Maximum number of bins for feature discretization. Higher values can improve accuracy but increase memory/time.
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "boosting_type": "gbdt",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 80,
    "max_depth": -1,
    "seed": 42,
    "n_jobs": -1,
    "verbose": -1,
    "colsample_bytree": 0.7,
    "subsample": 0.7,
    "reg_alpha": 0.15,
    "reg_lambda": 0.15,
    # GPU settings for P100 (adjust based on your GPU)
    "device_type": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "max_bin": 127, # Can be higher on GPU for better performance
}

# Train the LightGBM model.
# `valid_sets`: Specifies validation datasets to monitor performance during training.
# `callbacks`: Used for early stopping and logging evaluation metrics.
# - `lgb.early_stopping(500)`: Stops training if validation metric doesn't improve for 500 rounds.
# - `lgb.log_evaluation(100)`: Prints evaluation metrics every 100 boosting rounds.
model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_val],
    callbacks=[lgb.early_stopping(500, verbose=True), lgb.log_evaluation(100)]
)

# Free up memory after model training.
del lgb_train, lgb_val
gc.collect()


print("Generating predictions on the test set...")
# Select the same features used for training from the test DataFrame.
X_test = test_df[features]

# Use the trained model to predict scores for each flight option in the test set.
# `num_iteration=model.best_iteration` ensures we use the model at its optimal point (determined by early stopping).
predictions = model.predict(X_test, num_iteration=model.best_iteration)

# Add the predicted scores as a new column to the test_df.
test_df["score"] = predictions

# --- Rank Predictions within Each Group ---
# This is the most critical step for generating the submission file.
# We group the test_df by `ranker_id` and then rank the `score` column within each group.
# `ascending=False`: Higher scores get lower ranks (e.g., highest score gets rank 1).
# `method=\'first\'`: Assigns ranks based on the order of appearance for ties (ensures unique ranks).
# `.astype(int)`: Converts the ranks to integer type as required for submission.
test_df["selected"] = test_df.groupby("ranker_id")["score"].rank(ascending=False, method="first").astype(int)

# --- Create Submission File ---
print("Creating submission file...")
# Select only the required columns for the submission: Id, ranker_id, and the newly assigned ranks.
submission = test_df[["Id", "ranker_id", "selected"]]

# Save the submission DataFrame to a Parquet file.
# `index=False` prevents writing the DataFrame index to the file.
submission.to_parquet(OUTPUT_PATH, index=False)

print(f"Submission file created successfully at: {OUTPUT_PATH}")
print("Submission head:")
print(submission.head())
print("\nSubmission tail:")
print(submission.tail())




