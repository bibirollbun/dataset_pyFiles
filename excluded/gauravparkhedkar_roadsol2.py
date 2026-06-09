import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans # <-- Re-importing
from sklearn.isotonic import IsotonicRegression
import warnings
import re
import gc # Garbage Collector

# Suppress warnings
warnings.filterwarnings('ignore')

print("Starting RoadSense Prediction Script...")

# --- 1. Configuration & Loading ---
DATA_DIR = "/kaggle/input/etiq-roadsense/"
files = [
    'accidents_train.csv', 'places_train.csv', 'users_train.csv', 'vehicles_train.csv',
    'accidents_test.csv', 'places_test.csv', 'users_test.csv', 'vehicles_test.csv'
]
na_values = ['NA', 'NULL', '-', '', ' ', '#N/A']
try:
    data = {}
    for f in files:
        df_temp = pd.read_csv(f"{DATA_DIR}{f}", na_values=na_values, low_memory=False)
        data[f.replace('.csv', '')] = df_temp
    print("✅ 1/7: All 8 data files loaded successfully.")
except FileNotFoundError:
    print(f"❌ ERROR: Could not find data files in {DATA_DIR}")
    exit()

# --- 2. Data Cleaning Function (Robust Version) ---
def clean_datetime(df):
    if 'Date' not in df.columns or 'Hour' not in df.columns:
        return df
    parsed_date = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    hour_str = df['Hour'].astype(str).str.split('.').str[0].str.split(':').str[:2].str.join(':')
    parsed_time = pd.to_datetime(hour_str, format='mixed', errors='coerce').dt.time
    df['datetime'] = parsed_date.dt.normalize() + pd.to_timedelta(parsed_time.astype(str))
    failed_count = df['datetime'].isnull().sum()
    if failed_count > 0:
        print(f"Warning: {failed_count} datetime rows still failed to parse.")
    return df

def clean_data(data_dict):
    for name, df in data_dict.items():
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
             if col not in ['AccidentId', 'VehicleId', 'Date', 'Hour']:
                 if col in df.columns:
                     df[col] = df[col].astype(str).str.lower().str.strip()
        if 'accidents' in name: df = clean_datetime(df) 
        if 'AccidentId' in df.columns: df['AccidentId'] = df['AccidentId'].astype(str)
        if 'VehicleId' in df.columns: df['VehicleId'] = df['VehicleId'].astype(str)
        if 'users' in name:
            df['BirthYear_missing'] = df['BirthYear'].isnull().astype(int)
            df['SafetyDevice_missing'] = df['SafetyDevice'].isnull().astype(int)
        data_dict[name] = df
    print("✅ 2/7: Data cleaning complete.")
    return data_dict
data = clean_data(data)

# --- 2.5: Consolidate Messy Categories ---
def consolidate_categories(df, col_name='Category'):
    if col_name not in df.columns: return df
    df[col_name] = df[col_name].astype(str).str.lower()
    mapping = {
        r'auto|car|voiture|car<=3.5t|light vehicle|light vehide': 'car',
        r'motorbike|scooter|moped': 'motorcycle',
        r'bicycle|bicyde': 'bicycle',
        r'truck|largecar|bus|coach|tractor|semitrailer': 'heavy_vehicle',
        r'quad|tramway|specialengine|train': 'other_special',
        r'pedestrian': 'pedestrian'
    }
    df[col_name + '_clean'] = df[col_name]
    for pattern, replacement in mapping.items():
        df[col_name + '_clean'] = df[col_name + '_clean'].str.replace(pattern, replacement, regex=True)
    known_cats = mapping.values()
    df[col_name + '_clean'] = df[col_name + '_clean'].apply(lambda x: x if x in known_cats else 'other')
    df[col_name] = df[col_name + '_clean']
    df = df.drop(columns=[col_name + '_clean'])
    return df
print("Consolidating main 'Category' column...")
for name in ['users_train', 'users_test', 'vehicles_train', 'vehicles_test']:
    data[name] = consolidate_categories(data[name])
print("✅ 2.5/7: Main Category consolidation complete.")

# --- Define Target and Handle Missing Values BEFORE Merging ---
target = 'Gravity'
print("Mapping target variable 'Gravity'...")
data['accidents_train'][target] = data['accidents_train'][target].astype(str).str.lower().str.strip()
data['accidents_train'][target] = data['accidents_train'][target].map({'nonlethal': 0, 'lethal': 1})
print(f"Original accidents_train shape before dropna: {data['accidents_train'].shape}")
data['accidents_train'] = data['accidents_train'].dropna(subset=[target, 'datetime'])
data['accidents_train'][target] = data['accidents_train'][target].astype(int)
print(f"New accidents_train shape after dropping missing mapped Gravity/Datetime: {data['accidents_train'].shape}")


# --- 3. Merging & Aggregation (Best Feature Set) ---
def aggregate_and_merge(accidents, places, users, vehicles):
    mode_or_nan = lambda x: x.mode()[0] if not x.mode().empty else np.nan
    # Aggregate users
    if 'BirthYear' in users.columns:
        users['BirthYear'] = pd.to_numeric(users['BirthYear'], errors='coerce')
        users['age'] = 2023 - users['BirthYear']
    if 'SafetyDeviceUsed' in users.columns:
        users['SafetyDeviceUsed'] = pd.to_numeric(users['SafetyDeviceUsed'], errors='coerce')
    users_agg = users.groupby('AccidentId').agg(
        user_count=('VehicleId', 'size'), avg_user_age=('age', 'mean'), min_user_age=('age', 'min'), max_user_age=('age', 'max'),
        was_pedestrian_involved=('Category', lambda x: (x == 'pedestrian').any().astype(int)),
        safety_device_used_count=('SafetyDeviceUsed', lambda x: (x > 0).sum()),
        BirthYear_missing_sum=('BirthYear_missing', 'sum'), SafetyDevice_missing_sum=('SafetyDevice_missing', 'sum'),
        vulnerable_user_count=('Category', lambda x: (x.isin(['pedestrian', 'bicycle'])).sum())
    ).reset_index()
    users_agg['safety_device_usage_rate'] = users_agg['safety_device_used_count'] / (users_agg['user_count'] + 1e-6)
    
    # Aggregate vehicles (Keep physics)
    if 'PassengerNumber' in vehicles.columns: vehicles['PassengerNumber'] = pd.to_numeric(vehicles['PassengerNumber'], errors='coerce')
    vehicles_agg = vehicles.groupby('AccidentId').agg(
        vehicle_count=('VehicleId', 'size'), passenger_sum=('PassengerNumber', 'sum'),
        num_cars=('Category', lambda x: (x == 'car').sum()), num_motorcycles=('Category', lambda x: (x == 'motorcycle').sum()),
        num_heavy_vehicles=('Category', lambda x: (x == 'heavy_vehicle').sum()),
        mode_fixed_obstacle=('FixedObstacle', mode_or_nan),
        mode_mobile_obstacle=('MobileObstacle', mode_or_nan),
        mode_impact_point=('ImpactPoint', mode_or_nan),
        mode_maneuver=('Maneuver', mode_or_nan)
    ).reset_index()

    # Aggregate places (Focused features)
    print("Aggregating 'places' data...")
    places_cols_to_agg = ['RoadType', 'Circulation', 'LaneNumber', 'Slope', 'SurfaceCondition', 'Infrastructure']
    places_cols_to_agg = [col for col in places_cols_to_agg if col in places.columns]
    agg_dict = {col: mode_or_nan for col in places_cols_to_agg}
    places_agg = places.groupby('AccidentId').agg(agg_dict).reset_index()
    print("'places' data aggregated.")

    # Merge
    df = accidents.copy()
    df = pd.merge(df, places_agg, on='AccidentId', how='left')
    df = pd.merge(df, users_agg, on='AccidentId', how='left')
    df = pd.merge(df, vehicles_agg, on='AccidentId', how='left')
    
    count_cols = [col for col in df.columns if 'count' in col or 'num_' in col or 'sum' in col or 'was_' in col]
    for col in count_cols: df[col] = df[col].fillna(0)
    return df
print("Merging training data...")
train_df = aggregate_and_merge(data['accidents_train'], data['places_train'], data['users_train'], data['vehicles_train'])
print("Merging test data...")
test_df = aggregate_and_merge(data['accidents_test'], data['places_test'], data['users_test'], data['vehicles_test'])
print("✅ 3/7: Train and test data merged.")


# --- 4. Feature Engineering (*** ADDING KMEANS BACK ***) ---
def engineer_features(df_train, df_test):
    """Creates new features from existing data."""
    print("Engineering datetime, interaction, and KMEANS features...")
    
    for df in [df_train, df_test]:
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            
    # --- Impute Lat/Lon BEFORE binning ---
    lat_median = df_train['Latitude'].median()
    lon_median = df_train['Longitude'].median()
    for df in [df_train, df_test]:
        df['Latitude'] = df['Latitude'].fillna(lat_median)
        df['Longitude'] = df['Longitude'].fillna(lon_median)
    
    # --- Combine Lat/Lon from both sets to create global bins ---
    all_coords = pd.concat([
        df_train[['Latitude', 'Longitude']],
        df_test[['Latitude', 'Longitude']]
    ])
    
    # --- Fit on ALL data ---
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    all_clusters = kmeans.fit_predict(all_coords)
    
    df_train['location_cluster'] = all_clusters[:len(df_train)]
    df_test['location_cluster'] = all_clusters[len(df_train):]
    
    for df in [df_train, df_test]:
        if 'datetime' in df.columns and not df['datetime'].isnull().all():
            df['hour'] = df['datetime'].dt.hour
            df['dayofweek'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month
            df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        else: 
            df['hour'] = np.nan
            df['dayofweek'] = np.nan
            df['month'] = np.nan
            df['is_weekend'] = np.nan
        df['users_per_vehicle'] = df['user_count'] / (df['vehicle_count'] + 1e-6)

    print("KMeans location features created.")
    return df_train, df_test

train_df, test_df = engineer_features(train_df, test_df)
print("✅ 4/7: Feature engineering complete.")


# --- 5. Preprocessing for Modeling (Adding KMeans) ---
y_train = train_df[target]
categorical_features = [
    'Light', 'InAgglomeration', 'IntersectionType', 'Weather', 'CollisionType',
    'RoadType', 'Circulation', 'SurfaceCondition', 'Infrastructure', 'Slope',
    'dayofweek', 'month', 'is_weekend',
    'mode_fixed_obstacle', 'mode_mobile_obstacle', 'mode_impact_point', 'mode_maneuver',
    'location_cluster' # <-- THE "GAMBLE" FEATURE
]
numerical_features = [
    'LaneNumber', 'user_count', 'avg_user_age', 'min_user_age', 'max_user_age',
    'safety_device_used_count', 'was_pedestrian_involved', 'vehicle_count',
    'passenger_sum', 'hour', 'users_per_vehicle', 'num_cars',
    'num_motorcycles', 'num_heavy_vehicles',
    'BirthYear_missing_sum', 'SafetyDevice_missing_sum', 'safety_device_usage_rate',
    'vulnerable_user_count'
]
final_cat_features = [col for col in categorical_features if col in train_df.columns]
final_num_features = [col for col in numerical_features if col in train_df.columns]
features = [f for f in final_num_features + final_cat_features if f != target]
missing_features = set(features) - set(train_df.columns)
if missing_features: print(f"Warning: Features not found in DataFrame: {missing_features}")

X_train = train_df[features].copy()
X_test = test_df[features].reindex(columns=X_train.columns).copy()

print("Handling final missing values and data types...")
for col in final_num_features:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)
for col in final_cat_features:
    X_train[col] = X_train[col].fillna('Missing').astype(str).astype('category')
    X_test[col] = X_test[col].fillna('Missing').astype(str).astype('category')
print("✅ 5/7: Data preprocessed for modeling.")


# --- 6. Model Training (*** AGGRESSIVE MODEL ***) ---
print("Starting model training (LGBM with 5-fold CV to collect OOF)...")
# --- Parameters to overfit ---
lgb_params = {
    'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
    'n_estimators': 4000, 'learning_rate': 0.01,
    'num_leaves': 31, # <-- More complex
    'max_depth': -1, 'seed': 42, 'n_jobs': -1, 'verbose': -1,
    'colsample_bytree': 0.7, 'subsample': 0.7,
    'reg_alpha': 0.1, 'reg_lambda': 0.1, # <-- Less regularization
    'min_child_samples': 20,
    'is_unbalance': True # <-- The "aggressive" setting
}
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))
oof_preds_proba = np.zeros(len(X_train))
models = []
if 'y_train' not in locals() or y_train is None:
     y_train = train_df[target]
     if y_train.isnull().any(): print("ERROR: NaNs found in y_train!"); exit()
lgbm_categorical_features = [col for col in features if X_train[col].dtype == 'category']

for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    if y_train_fold.isnull().any() or y_val_fold.isnull().any():
         print(f"ERROR: NaNs found in y_train_fold or y_val_fold for fold {fold+1}!"); continue
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_val_fold, y_val_fold)], eval_metric='f1',
              callbacks=[lgb.early_stopping(100, verbose=False)],
              categorical_feature=lgbm_categorical_features)
    val_preds_proba = model.predict_proba(X_val_fold)[:, 1]
    oof_preds_proba[val_index] = val_preds_proba
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
    models.append(model)
    gc.collect()

# --- Find the best CV threshold ---
thresholds = np.linspace(0, 1, 100)
oof_f1_scores = [f1_score(y_train, (oof_preds_proba > t).astype(int), average='macro') for t in thresholds]
best_oof_threshold = thresholds[np.argmax(oof_f1_scores)]
best_oof_f1 = np.max(oof_f1_scores)
print(f"\n✅ 6/7: Model training complete. Raw OOF F1 (at {best_oof_threshold:.2f}): {best_oof_f1:.4f}")


# --- 7. Create Submission File (*** AGGRESSIVE CALIBRATION ***) ---
# --- NO Isotonic, NO Manual Percentile. Just use the OOF threshold. ---
print(f"Applying OOF-derived threshold: {best_oof_threshold:.4f}")
submission_threshold = best_oof_threshold
final_test_preds = (test_preds >= submission_threshold).astype(int)

print("Creating submission file...")
label_map = {0: 'NonLethal', 1: 'Lethal'}
submission_labels = [label_map[pred] for pred in final_test_preds]
submission_df = pd.DataFrame({
    'AccidentId': data['accidents_test']['AccidentId'],
    'Gravity': submission_labels
})

print("\n--- Submission File Analysis ---")
print("Predicted class distribution:")
print(submission_df['Gravity'].value_counts(normalize=True))
print("--------------------------------\n")
submission_df.to_csv('submission.csv', index=False)
print(f"✅ 7/7: Submission file 'submission.csv' created successfully!")
print("\nScript finished.")

