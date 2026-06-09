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


import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import gc

from tqdm.notebook import tqdm
tqdm.pandas()  # Enables progress bars on pandas operations

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


import pandas as pd
import gc

# Define selected relevant columns for training
selected_cols_train = [
    "Id", "ranker_id", "profileId", "companyID", "isVip", "bySelf",
    "requestDate", "searchRoute", "totalPrice", "taxes",
    "legs0_departureAt", "legs0_arrivalAt", "legs0_duration",
    "legs1_departureAt", "legs1_arrivalAt", "legs1_duration",
    "legs0_segments0_seatsAvailable", "legs0_segments0_cabinClass",
    "legs0_segments0_flightNumber", "legs1_segments0_flightNumber",
    "legs0_segments0_baggageAllowance_quantity",
    "legs0_segments0_baggageAllowance_weightMeasurementType",
    "legs0_segments0_marketingCarrier_code",
    "legs0_segments0_operatingCarrier_code",
    "legs0_segments0_duration",
    "legs0_segments0_departureFrom_airport_iata",
    "legs0_segments0_arrivalTo_airport_city_iata",
    "miniRules0_statusInfos", "miniRules0_monetaryAmount", "miniRules0_percentage",
    "miniRules1_statusInfos", "miniRules1_monetaryAmount", "miniRules1_percentage",
    "selected"  # Target column for training only
]

# For test data (no target column)
selected_cols_test = [col for col in selected_cols_train if col != "selected"]

# Load using PyArrow
train = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet', columns=selected_cols_train, engine='pyarrow')
test = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet', columns=selected_cols_test, engine='pyarrow')

gc.collect()


from tqdm import tqdm
import gc

def add_datetime_features(df):
    print("Extracting datetime features...")
    
    # Convert datetime columns
    datetime_cols = [
        'requestDate', 
        'legs0_departureAt', 'legs0_arrivalAt',
        'legs1_departureAt', 'legs1_arrivalAt'
    ]
    
    for col in tqdm(datetime_cols):
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Request datetime features
    df['request_hour'] = df['requestDate'].dt.hour
    df['request_dayofweek'] = df['requestDate'].dt.dayofweek
    df['request_weekend'] = (df['request_dayofweek'] >= 5).astype(int)

    # Flight leg 0 (outbound) datetime features
    df['leg0_depart_hour'] = df['legs0_departureAt'].dt.hour
    df['leg0_arrive_hour'] = df['legs0_arrivalAt'].dt.hour
    df['leg0_depart_day'] = df['legs0_departureAt'].dt.dayofweek
    df['leg0_arrive_day'] = df['legs0_arrivalAt'].dt.dayofweek

    # Flight leg 1 (inbound) datetime features
    df['leg1_depart_hour'] = df['legs1_departureAt'].dt.hour
    df['leg1_arrive_hour'] = df['legs1_arrivalAt'].dt.hour
    df['leg1_depart_day'] = df['legs1_departureAt'].dt.dayofweek
    df['leg1_arrive_day'] = df['legs1_arrivalAt'].dt.dayofweek

    # Flight duration (round trip / one way indicator)
    df['round_trip'] = df['legs1_departureAt'].notna().astype(int)

    # Drop original datetime columns (optional, saves memory)
    # df.drop(datetime_cols, axis=1, inplace=True)

    gc.collect()
    return df



def add_duration_price_features(df):
    print("Extracting duration and price-based features...")

    # Convert timestamps to datetime
    df["legs0_departureAt"] = pd.to_datetime(df["legs0_departureAt"], errors="coerce")
    df["legs0_arrivalAt"] = pd.to_datetime(df["legs0_arrivalAt"], errors="coerce")
    df["legs1_departureAt"] = pd.to_datetime(df["legs1_departureAt"], errors="coerce")
    df["legs1_arrivalAt"] = pd.to_datetime(df["legs1_arrivalAt"], errors="coerce")

    # Convert durations to numeric (in minutes)
    df["legs0_duration"] = pd.to_numeric(df["legs0_duration"], errors="coerce")
    df["legs1_duration"] = pd.to_numeric(df["legs1_duration"], errors="coerce")

    # Create delay features
    df['leg0_arrival_delay'] = (
        (df['legs0_arrivalAt'] - df['legs0_departureAt']).dt.total_seconds() / 60
    ) - df['legs0_duration']

    df['leg1_arrival_delay'] = (
        (df['legs1_arrivalAt'] - df['legs1_departureAt']).dt.total_seconds() / 60
    ) - df['legs1_duration']

    # Total trip duration
    df['total_trip_duration'] = df['legs0_duration'] + df['legs1_duration']

    # Absolute duration difference
    df['duration_diff'] = np.abs(df['legs0_duration'] - df['legs1_duration'])

    return df



def add_route_segment_features(df):
    # Route string length
    df['route_length'] = df['searchRoute'].apply(lambda x: len(str(x).split('-')))

    # Is round trip
    df['is_round_trip'] = df['searchRoute'].apply(
        lambda x: str(x).split('-')[0] == str(x).split('-')[-1] if pd.notna(x) else False
    ).astype(int)

    # Airline consistency
    df['is_same_airline'] = (
        df['legs0_segments0_marketingCarrier_code'] == df['legs0_segments0_operatingCarrier_code']
    ).astype(int)

    # Domestic leg0: crude check using first and last letter of IATA
    df['is_domestic_leg0'] = (
        df['legs0_segments0_departureFrom_airport_iata'].str[:1] ==
        df['legs0_segments0_arrivalTo_airport_city_iata'].str[:1]
    ).astype(int)

    # Same airport for departure and arrival
    df['same_depart_arrive_city'] = (
        df['legs0_segments0_departureFrom_airport_iata'] ==
        df['legs0_segments0_arrivalTo_airport_city_iata']
    ).astype(int)

    return df



def add_passenger_profile_features(df, full_df):
    # Frequency of each profileId
    profile_freq = full_df['profileId'].value_counts().to_dict()
    df['profile_booking_freq'] = df['profileId'].map(profile_freq).fillna(0).astype(int)

    # Frequency of each companyID
    company_freq = full_df['companyID'].value_counts().to_dict()
    df['company_booking_freq'] = df['companyID'].map(company_freq).fillna(0).astype(int)

    # Frequency of (profileId, companyID) pairs
    combo_freq = full_df.groupby(['profileId', 'companyID']).size().to_dict()
    df['profile_company_combo_freq'] = df.set_index(['profileId', 'companyID']).index.map(combo_freq).fillna(0).astype(int)

    # Convert boolean columns to int
    df['isVip'] = df['isVip'].astype(int)
    df['bySelf'] = df['bySelf'].astype(int)

    return df



def add_price_tax_features(df):
    df['price_per_leg'] = df['totalPrice'] / (
        df[['legs0_duration', 'legs1_duration']].notnull().sum(axis=1).clip(lower=1)
    )

    df['tax_ratio'] = (df['taxes'] / df['totalPrice']).fillna(0)
    df['price_minus_tax'] = (df['totalPrice'] - df['taxes']).fillna(0)

    df['is_high_tax_ratio'] = (df['tax_ratio'] > 0.5).astype(int)

    # Log transformations to reduce skew
    df['log_price'] = np.log1p(df['totalPrice'])
    df['log_tax'] = np.log1p(df['taxes'])

    return df



train = reduce_mem_usage(train)
test = reduce_mem_usage(test)



train = add_datetime_features(train)
gc.collect()
train = reduce_mem_usage(train)

train = add_duration_price_features(train)
gc.collect()
train = reduce_mem_usage(train)

train = add_route_segment_features(train)
gc.collect()
train = reduce_mem_usage(train)

train = add_passenger_profile_features(train, full_df=train)
gc.collect()
train = reduce_mem_usage(train)

train = add_price_tax_features(train)
gc.collect()
train = reduce_mem_usage(train)



# Recreate mini profile-company dataframe used in passenger profile features
full_profile_company_df = pd.concat([
    train[["profileId", "companyID"]],
    test[["profileId", "companyID"]]
], axis=0)
full_profile_company_df = reduce_mem_usage(full_profile_company_df)

# Apply feature engineering functions sequentially
test = add_datetime_features(test)
gc.collect()
test = reduce_mem_usage(test)

test = add_duration_price_features(test)
gc.collect()
test = reduce_mem_usage(test)

test = add_route_segment_features(test)
gc.collect()
test = reduce_mem_usage(test)

test = add_passenger_profile_features(test, full_df=full_profile_company_df)
gc.collect()
test = reduce_mem_usage(test)

test = add_price_tax_features(test)
gc.collect()
test = reduce_mem_usage(test)



print(train.columns.tolist())


# Fill NaNs in object-type (categorical) columns with "missing"
cat_cols = train.select_dtypes(include=["object", "category"]).columns

for col in cat_cols:
    train[col] = train[col].fillna("missing")
    test[col] = test[col].fillna("missing")



from sklearn.preprocessing import LabelEncoder

# Identify categorical columns
cat_cols = train.select_dtypes(include='object').columns.tolist()

# Encode only if column exists in both train and test
for col in cat_cols:
    if col in test.columns:
        le = LabelEncoder()
        all_vals = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(all_vals)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))



import gc
import lightgbm as lgb
from lightgbm.callback import CallbackEnv
from tqdm import tqdm
import numpy as np

# --------------------------
# 1. Select Features
# --------------------------
print("Preparing data for training...")

features = [col for col in train.columns if col not in [
    "Id", "ranker_id", "selected", 
    "legs0_departureAt", "legs0_arrivalAt", 
    "legs1_departureAt", "legs1_arrivalAt", 
    "requestDate", 
    "legs0_segments1_flightNumber", "legs1_segments0_flightNumber"
]]

# --------------------------
# 2. Time-Based Split
# --------------------------
train_cutoff_date = train["requestDate"].quantile(0.85, interpolation="nearest")
train_idx = train[train["requestDate"] <= train_cutoff_date].index
val_idx = train[train["requestDate"] > train_cutoff_date].index

X_train = train.loc[train_idx, features]
y_train = train.loc[train_idx, "selected"]
X_val = train.loc[val_idx, features]
y_val = train.loc[val_idx, "selected"]

# Group sizes for LambdaRank
train_groups = train.loc[train_idx].groupby("ranker_id").size().to_numpy()
val_groups = train.loc[val_idx].groupby("ranker_id").size().to_numpy()

# --------------------------
# 3. LightGBM Dataset
# --------------------------
lgb_train = lgb.Dataset(X_train, y_train, group=train_groups, free_raw_data=False)
lgb_val = lgb.Dataset(X_val, y_val, group=val_groups, reference=lgb_train, free_raw_data=False)

# Clean memory
del X_train, y_train, X_val, y_val, train
gc.collect()




# --------------------------
# 4. TQDM Callback
# --------------------------
class TQDMProgressBar:
    def __init__(self, total):
        self.pbar = tqdm(total=total)

    def __call__(self, env: CallbackEnv):
        self.pbar.update(1)
        if env.iteration + 1 == self.pbar.total:
            self.pbar.close()

# --------------------------
# 5. Custom Eval: HitRate@3
# --------------------------
def hit_rate_at_3(preds, train_data):
    labels = train_data.get_label()
    group = train_data.get_group()
    
    hits = 0
    current = 0
    for size in group:
        group_preds = preds[current:current + size]
        group_labels = labels[current:current + size]
        top3_idx = np.argsort(-group_preds)[:3]
        if any(group_labels[top3_idx]):
            hits += 1
        current += size
    return 'HitRate@3', hits / len(group), True



params = {
    "objective": "lambdarank",
    "boosting_type": "gbdt",
    "metric": "None",  # We will use our own
    "ndcg_eval_at": [3],
    "verbosity": -1,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42,
    # GPU settings for P100 (adjust based on your GPU)
    "device_type": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0 # Can be higher on GPU for better performance
}

print("Training LightGBM model...")
num_boost_round = 1000

model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_val],
    valid_names=["val"],
    feval=hit_rate_at_3,
    num_boost_round=num_boost_round,
    callbacks=[
        TQDMProgressBar(total=num_boost_round),
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)



# print("Predicting on test set...")
# test_preds = model.predict(test[features], num_iteration=model.best_iteration)

# # Prepare dataframe for ranking
# submission_df = test[["Id", "ranker_id"]].copy()
# submission_df["score"] = test_preds

# # Sort by ranker_id and prediction score (descending)
# submission_df = submission_df.sort_values(["ranker_id", "score"], ascending=[True, False])

# # Rank items within each group
# submission_df["rank"] = submission_df.groupby("ranker_id")["score"].rank("first", ascending=False)

# # Pick top-ranked Id per ranker_id (i.e., rank 1)
# top_submission = submission_df.loc[submission_df["rank"] == 1, ["Id"]]

# # Save to CSV
# top_submission.to_csv("submission.csv", index=False)

# print("submission.csv")
print("Predicting on test set...")

# Predict scores
test_preds = model.predict(test[features], num_iteration=model.best_iteration)

# Prepare dataframe for ranking
submission_df = test[["Id", "ranker_id"]].copy()
submission_df["score"] = test_preds

# Sort by ranker_id and prediction score (descending)
submission_df = submission_df.sort_values(["ranker_id", "score"], ascending=[True, False])

# Assign rank within each group (1 is highest score)
submission_df["selected"] = submission_df.groupby("ranker_id")["score"].rank("first", ascending=False).astype(int)

# Save final submission
final_submission = submission_df[["Id", "ranker_id", "selected"]]
final_submission.to_csv("submission.csv", index=False)
print("submission.csv created with shape:", final_submission.shape)




print("Submission head:")
print(final_submission.head())
print("\nSubmission tail:")
print(final_submission.tail())




