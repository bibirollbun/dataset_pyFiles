!pip install lightgbm


import pandas as pd
import os
import gc
import subprocess
import zipfile
import matplotlib.pyplot as plt

import lightgbm as lgb
import numpy as np

from itertools import product
from sklearn.model_selection import GroupShuffleSplit
from itertools import chain
from sklearn.model_selection import GroupKFold


# resetting the settings
pd.reset_option('display.max_columns')

# setting the maximum column display option
pd.set_option('display.max_columns', None)




def prepare_kaggle_dataset():
    # Set the competition input path (automatically provided by Kaggle)
    extract_path = "/kaggle/input/aeroclub-recsys-2025"
    
    # List files in the competition folder
    print("ğŸ“‚ Files available in the dataset folder:")
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            print(f"- {file}")
    
    return extract_path  # So you can load files from here
gc.collect()


data_path = prepare_kaggle_dataset()
# download_files()
gc.collect()


gc.collect()
# train = pd.read_parquet("data/aeroclub/train.parquet")
data_path = "/kaggle/input/aeroclub-recsys-2025"
train = pd.read_parquet(f"{data_path}/train.parquet")
gc.collect()


def reduce_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtypes
        
        if col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif col_type == 'object':
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')
    
    return df
train = reduce_memory_usage(train)


train.head()


df_train_raw = train.copy()
gc.collect()



# Define the columns you want to keep

columns_to_keep = [
    # Identifiers
    'Id',  # num
    'ranker_id', 
    'profileId', 
    'companyID',
    
    # User info
    'sex', 'nationality', 'frequentFlyer', 'isVip', 'bySelf', 'isAccess3D',

    # Company info
    'corporateTariffCode',

    # Search & route
    'searchRoute', 'requestDate',

    # Pricing
    'totalPrice', 'taxes',

    # Flight timing
    'legs0_departureAt', 'legs0_arrivalAt', 'legs0_duration',
    'legs1_departureAt', 'legs1_arrivalAt', 'legs1_duration',

    # Segment-level info (sÃ³ do segmento 0 da ida para simplificar no baseline)
    'legs0_segments0_departureFrom_airport_iata',
    'legs0_segments0_arrivalTo_airport_iata',
    'legs0_segments0_arrivalTo_airport_city_iata',
    'legs0_segments0_marketingCarrier_code',
    'legs0_segments0_operatingCarrier_code',
    'legs0_segments0_aircraft_code',
    'legs0_segments0_flightNumber',
    'legs0_segments0_duration',
    'legs0_segments0_baggageAllowance_quantity',
    'legs0_segments0_baggageAllowance_weightMeasurementType',
    'legs0_segments0_cabinClass',
    'legs0_segments0_seatsAvailable', 
    'legs0_segments1_departureFrom_airport_iata',
    'legs0_segments2_departureFrom_airport_iata',
    'legs0_segments3_departureFrom_airport_iata',

    # Cancellation & exchange rules
    'miniRules0_monetaryAmount', 'miniRules0_percentage', 'miniRules0_statusInfos',
    'miniRules1_monetaryAmount', 'miniRules1_percentage', 'miniRules1_statusInfos',

    # Pricing policy
    'pricingInfo_isAccessTP', 'pricingInfo_passengerCount',

    # Target
    'selected'
]

# Filter the data for the baseline
def load_subset(df, columns,  max_rows=None):
    if max_rows:
        return df[columns].iloc[:max_rows].copy()
    else:
        return df[columns].copy()

# Example of usage
df_train = load_subset(df_train_raw, columns_to_keep, max_rows=1_000_000) # ONLY 1M

#############################          IMPORTANT      ########################################
#############################          IMPORTANT      ########################################
#############################          IMPORTANT      ########################################
#############################          IMPORTANT      ########################################
#df_train = load_subset(df_train_raw, columns_to_keep) # ALL REGISTERS


def fix_column_types(df):
    df_fixed = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            # Try converting to numeric type
            try:
                df_fixed[col] = pd.to_numeric(df[col])
            except:
                # If not numeric, try boolean
                unique_vals = df[col].dropna().unique()
                if set(unique_vals) <= {True, False}:
                    df_fixed[col] = df[col].astype(bool)
                else:
                    df_fixed[col] = df[col].astype(str)
    return df_fixed

df_train = fix_column_types(df_train)

# Adjust nationality (currently in int format)
df_train["nationality"] = df_train["nationality"].astype("str")

# Convert companyID to category type
df_train['companyID'] = df_train['companyID'].astype('category')

# Check result
df_train.dtypes


def feature_engineer(df):
    """
    Engineers a comprehensive set of features for the flight ranking model.
    """
    df = df.copy()

    # 1. Time-Based Features
    for col in ['requestDate', 'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df['request_dayofweek'] = df['requestDate'].dt.dayofweek
    df['request_hour'] = df['requestDate'].dt.hour
    df['departure_hour'] = df['legs0_departureAt'].dt.hour
    df['is_weekend'] = df['legs0_departureAt'].dt.dayofweek.isin([5, 6]).astype(int)
    df['is_business_hours'] = ((df['departure_hour'] >= 8) & (df['departure_hour'] <= 18)).astype(int)
    df['days_to_departure'] = (df['legs0_departureAt'] - df['requestDate']).dt.days

    # 2. Duration Conversion (String to Hours)
    def to_hours(duration_str):
        if pd.isna(duration_str):
            return np.nan
        try:
            parts = str(duration_str).split(':')
            hours = int(parts[0]) + int(parts[1]) / 60
            return hours
        except:
            return np.nan

    df['legs0_duration_hr'] = df['legs0_duration'].apply(to_hours)
    df['legs1_duration_hr'] = df['legs1_duration'].apply(to_hours)
    df['total_duration'] = df['legs0_duration_hr'].fillna(0) + df['legs1_duration_hr'].fillna(0)

    # 3. Segment & Connection Features
    stop_cols = [c for c in df.columns if 'segments' in c and 'departureFrom' in c]
    df['num_stops'] = df[stop_cols].notna().sum(axis=1) - 1  # Subtract 1 for the origin
    df['is_direct_flight'] = (df['num_stops'] == 0).astype(int)

    # 4. Interaction and Ratio Features
    df['tax_ratio'] = df['taxes'] / (df['totalPrice'] + 1e-6)
    df['price_per_duration'] = df['totalPrice'] / (df['total_duration'] + 1e-6)
    df['price_to_stops_ratio'] = df['totalPrice'] / (df['num_stops'] + 1)

    # 5. Group-wise (Ranker ID) Features
    group_features = ['totalPrice', 'total_duration', 'num_stops']
    for feature in group_features:
        df[f'{feature}_avg_ranker'] = df.groupby('ranker_id')[feature].transform('mean')
        df[f'{feature}_max_ranker'] = df.groupby('ranker_id')[feature].transform('max')
        df[f'{feature}_min_ranker'] = df.groupby('ranker_id')[feature].transform('min')
        df[f'{feature}_std_ranker'] = df.groupby('ranker_id')[feature].transform('std')

        df[f'{feature}_diff_from_avg'] = df[feature] - df[f'{feature}_avg_ranker']
        df[f'{feature}_ratio_to_avg'] = df[feature] / (df[f'{feature}_avg_ranker'] + 1e-6)

    # 6. Categorical and Boolean Handling
    bool_cols = ['isVip', 'bySelf', 'isAccess3D', 'pricingInfo_isAccessTP']
    for col in bool_cols:
        df[col] = df[col].astype(bool)

    cat_cols = [
        'companyID', 'sex', 'nationality', 'corporateTariffCode',
        'legs0_segments0_departureFrom_airport_iata', 'legs0_segments0_arrivalTo_airport_iata',
        'legs0_segments0_marketingCarrier_code', 'legs0_segments0_operatingCarrier_code',
        'legs0_segments0_cabinClass'
    ]
    for col in cat_cols:
        df[col] = pd.factorize(df[col])[0]

    # 7. Clean up and Final Touches
    df = df.drop(columns=[
        'requestDate', 'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt',
        'legs0_duration', 'legs1_duration', 'frequentFlyer', 'searchRoute',
        'legs0_segments0_duration', 'legs0_segments1_departureFrom_airport_iata',
        'legs0_segments2_departureFrom_airport_iata', 'legs0_segments3_departureFrom_airport_iata'
    ], errors='ignore')

    # Fill NaNs created during feature engineering
    df.fillna(-1, inplace=True)

    return df



print("Starting feature engineering for the training set...")
df_train_fe = feature_engineer(df_train)
print("Feature engineering complete.")
gc.collect()


test = pd.read_parquet(f"{data_path}/test.parquet")
test = reduce_memory_usage(test)


df_test_raw = test.copy()
print(f"Full test data loaded with {len(df_test_raw)} rows.")
print("\nStarting feature engineering for the full test set...")

gc.collect()



df_test = fix_column_types(df_test_raw)

# Adjust nationality (currently in int format)
df_test["nationality"] = df_test["nationality"].astype("str")

# Convert companyID to category type
df_test['companyID'] = df_test['companyID'].astype('category')

# Check result
df_test.dtypes

df_test_fe = feature_engineer(df_test)
print("Feature engineering on the full test set complete.")
print(f"Processed test data has {len(df_test_fe)} rows.")
gc.collect()


# --- Target and group column
target_col = "selected"
group_col = "ranker_id"


features = [col for col in df_train_fe.columns if col not in ['Id', 'ranker_id', 'profileId', 'selected']]
categorical_features = [
'companyID', 'sex', 'nationality', 'corporateTariffCode',
'legs0_segments0_departureFrom_airport_iata', 'legs0_segments0_arrivalTo_airport_iata',
'legs0_segments0_marketingCarrier_code', 'legs0_segments0_operatingCarrier_code',
'legs0_segments0_cabinClass','legs0_segments0_arrivalTo_airport_city_iata',  
'legs0_segments0_aircraft_code' 
]


categorical_features = [f for f in categorical_features if f in features]
# for col in categorical_features:
#     df_train_fe[col] = pd_train_fe.factorize(df[col])[0]


for col in categorical_features:
    df_train_fe[col] = df_train_fe[col].astype("category")
    df_test_fe[col] = df_test_fe[col].astype("category")


gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(df_train_fe, groups=df_train_fe["ranker_id"]))
df_train_split = df_train_fe.iloc[train_idx].copy()
df_val = df_train_fe.iloc[val_idx].copy()


X_train = df_train_split[features]
y_train = df_train_split[target_col]
groups_train = df_train_split.groupby(group_col).size().to_numpy()
X_val = df_val[features]
y_val = df_val[target_col]
groups_val = df_val.groupby(group_col).size().to_numpy()
dataset_params = {
"max_bin": 63
}


train_dataset = lgb.Dataset(
X_train,
label=y_train,
group=groups_train,
categorical_feature=categorical_features,
params=dataset_params
)


val_dataset = lgb.Dataset(
X_val,
label=y_val,
group=groups_val,
categorical_feature=categorical_features,
reference=train_dataset,
params=dataset_params
)
print("Train and validation datasets created successfully.")


param_grid = {
'learning_rate': [0.05, 0.1],
'num_leaves': [63, 127],

'min_child_samples':[50, 70, 100]

}


param_combinations = list(product(*param_grid.values()))
param_keys = list(param_grid.keys())
best_score = -1
best_model = None
best_params = None
for combo in param_combinations:
    param_set = dict(zip(param_keys, combo))
print(f"Training with: {param_set}")


params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at":[3],
    "boosting_type": "gbdt",
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "verbosity": -1,
    "num_threads": -1, # Use all available threads
    **param_set
}


model = lgb.train(
    params,
    train_dataset,
    valid_sets=[val_dataset],
    valid_names=["valid"],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)

score = model.best_score["valid"]["ndcg@3"]
print(f"  -> Achieved NDCG@3: {score:.5f}")


if score > best_score:
    best_score = score
    best_model = model
    best_params = param_set

print("\nâœ… Best combination found:")
print(f"   Parameters: {best_params}")
print(f"   Best NDCG@3 on validation set: {best_score:.5f}")


X_full = df_train_fe[features]
y_full = df_train_fe[target_col]
groups_full = df_train_fe.groupby(group_col).size().to_numpy()
full_dataset = lgb.Dataset(
X_full,
y_full,
group=groups_full,
categorical_feature=categorical_features,
params=dataset_params
)


final_params = {
"objective": "lambdarank",
"metric": "ndcg",
"ndcg_eval_at": [3],
"boosting_type": "gbdt",
"feature_fraction": 0.8,
"bagging_fraction": 0.8,
"bagging_freq": 1,
"seed": 42,
"verbosity": -1,
"num_threads": -1,
**best_params  # Use the best parameters found in the grid search
}


print(f"Training final model with best parameters for {best_model.best_iteration} rounds...")
final_model = lgb.train(
final_params,
full_dataset,
num_boost_round=best_model.best_iteration
)
print("âœ… Final model training complete.")



print("Predicting on the test set...")
X_test = df_test_fe[features]
df_test_fe['y_pred'] = final_model.predict(X_test)


df_test_sorted = df_test_fe.sort_values(['ranker_id', 'y_pred'], ascending=[True, False])


df_test_sorted['selected'] = df_test_sorted.groupby('ranker_id').cumcount() + 1


submission = df_test_sorted[['Id', 'ranker_id', 'selected']]


submission.to_csv("submission.csv", index=False)
print("âœ… Submission file 'submission.csv' created successfully!")
print(submission.head())
print(submission.tail())

