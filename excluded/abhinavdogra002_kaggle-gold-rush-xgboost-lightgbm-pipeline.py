! pip install -U xgboost lightgbm tensorflow scikit-learn


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import gc # Garbage Collector
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def calculate_hit_rate_at_3(df_preds_with_true_and_rank):
    """
    Calculates HitRate@3.
    df_preds_with_true_and_rank must have:
        - 'ranker_id'
        - 'selected' (true binary target, 1 for chosen)
        - 'predicted_rank' (rank assigned by the model, 1 is best)
    """
    hits = 0
    valid_queries_count = 0
    
    for ranker_id, group in df_preds_with_true_and_rank.groupby('ranker_id'):
        if len(group) <= 10:
            continue  # Skip groups with 10 or fewer options as per competition rules
        
        valid_queries_count += 1
        
        true_selected_item = group[group['selected'] == 1]
        
        if not true_selected_item.empty:
            # Get the rank of the true selected item
            rank_of_true_item = true_selected_item.iloc[0]['predicted_rank']
            if rank_of_true_item <= 3:
                hits += 1
        # else:
            # This shouldn't happen in validation if data is prepared correctly from train
            # print(f"Warning: No selected item found for ranker_id {ranker_id} in HitRate calculation.")
            
    if valid_queries_count == 0:
        return 0.0
    return hits / valid_queries_count


# Cell 3: Load Data
import pandas as pd
import numpy as np
import gc

initial_core_columns = [
    'Id', 'ranker_id', 'selected', 'profileId', 'companyID',
    'requestDate', 'totalPrice', 'taxes',
    'legs0_departureAt', 'legs0_arrivalAt', 'legs0_duration',
    'legs1_departureAt', 'legs1_arrivalAt', 'legs1_duration',
    'legs0_segments0_departureFrom_airport_iata', 'legs0_segments0_arrivalTo_airport_iata',
    'legs0_segments0_marketingCarrier_code', 'legs0_segments0_cabinClass',
    'legs0_segments0_baggageAllowance_quantity',
    'searchRoute',
    'pricingInfo_isAccessTP', 'pricingInfo_passengerCount',
    'sex', 'nationality', 'isVip',
    'miniRules0_monetaryAmount', 'miniRules0_percentage', 
    'miniRules1_monetaryAmount', 'miniRules1_percentage'
]
initial_core_columns_test = [col for col in initial_core_columns if col != 'selected']

print("Loading a subset of columns for train_df...")
train_df = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet', columns=initial_core_columns)
print("Loading a subset of columns for test_df...")
test_df = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet', columns=initial_core_columns_test)
sample_submission_df = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/sample_submission.parquet')

print("\nTrain DataFrame (after loading subset - BEFORE reduce_mem_usage and any FE):")
train_df.info(memory_usage='deep')
print(f"\nShape: {train_df.shape}")
print("\nTest DataFrame (after loading subset - BEFORE reduce_mem_usage and any FE):")
test_df.info(memory_usage='deep')
print(f"\nShape: {test_df.shape}")

if 'Id' in test_df.columns and 'ranker_id' in test_df.columns:
    test_ids_df = test_df[['Id', 'ranker_id']].copy()
else:
    print("Warning: 'Id' or 'ranker_id' not found in loaded test_df columns. Submission might fail.")
    try:
        temp_ids = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet', columns=['Id', 'ranker_id'])
        test_ids_df = temp_ids.copy()
        del temp_ids
    except Exception as e:
        print(f"Fallback to load test Ids failed: {e}")
        test_ids_df = pd.DataFrame()
gc.collect()


test_df['ranker_id'].unique()


# Cell 4: Feature Engineering

def create_initial_datetime_features(df):
    loaded_cols = df.columns
    potential_dt_cols = ['requestDate', 'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt']
    for col in potential_dt_cols:
        if col in loaded_cols:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                current_dtype = df[col].dtype
                print(f"Converting column {col} (current dtype: {current_dtype}) to datetime.")
                df[col] = pd.to_datetime(df[col].astype(str), errors='coerce')
    return df

def create_remaining_features(df, is_train=True):
    # --- Date/Time Component Extraction ---
    potential_dt_cols_for_components = ['legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt']
    for col in potential_dt_cols_for_components:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
             df[col + '_hour'] = df[col].dt.hour.astype(np.int8, errors='ignore')
             df[col + '_dow'] = df[col].dt.dayofweek.astype(np.int8, errors='ignore')

    # --- Booking Lead Time ---
    if 'legs0_departureAt' in df.columns and 'requestDate' in df.columns and \
       pd.api.types.is_datetime64_any_dtype(df['legs0_departureAt']) and \
       pd.api.types.is_datetime64_any_dtype(df['requestDate']):
        df['booking_lead_days'] = (df['legs0_departureAt'] - df['requestDate']).dt.total_seconds() / (24 * 60 * 60)
        df['booking_lead_days'] = df['booking_lead_days'].fillna(-1).astype(np.float32)
    else:
        missing_cols = [c for c in ['legs0_departureAt', 'requestDate'] if c not in df.columns]
        if missing_cols: print(f"Warning: Columns {missing_cols} not found for booking_lead_days.")
        else: print(f"Warning: Dtype issue for booking_lead_days. legs0_dep: {df.get('legs0_departureAt', pd.Series(dtype='object')).dtype}, reqDate: {df.get('requestDate', pd.Series(dtype='object')).dtype}")
        df['booking_lead_days'] = -1.0

    # --- Route Features ---
    if 'searchRoute' in df.columns: df['is_round_trip'] = df['searchRoute'].astype(str).str.contains('/').astype(np.int8)
    else: df['is_round_trip'] = -1 
    
    if 'legs1_departureAt' in df.columns and pd.api.types.is_datetime64_any_dtype(df['legs1_departureAt']):
        df['num_legs'] = 1 + df['legs1_departureAt'].notna().astype(np.int8)
    elif 'legs1_departureAt' in df.columns :
         df['num_legs'] = 1 + pd.to_datetime(df['legs1_departureAt'].astype(str),errors='coerce').notna().astype(np.int8)
    else: df['num_legs'] = 1

    # --- Segment Count ---
    df['num_segments_leg0'] = 0; df['num_segments_leg1'] = 0
    if 'legs0_segments0_departureFrom_airport_iata' in df.columns: df['num_segments_leg0'] += df['legs0_segments0_departureFrom_airport_iata'].notna().astype(np.int8)
    if 'legs1_segments0_departureFrom_airport_iata' in df.columns: df['num_segments_leg1'] += df['legs1_segments0_departureFrom_airport_iata'].notna().astype(np.int8)
   
    df['total_segments'] = (df['num_segments_leg0'] + df['num_segments_leg1']).astype(np.int8)
    
    # --- Flight Duration ---
    for dur_col in ['legs0_duration', 'legs1_duration']:
        if dur_col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[dur_col]):
                df[dur_col] = pd.to_numeric(df[dur_col].astype(str), errors='coerce').fillna(0)
            else: df[dur_col] = df[dur_col].fillna(0)
        else: df[dur_col] = 0 
    df['total_flight_duration'] = (df['legs0_duration'] + df['legs1_duration']).astype(np.float32)

    # --- Price Features ---
    if 'totalPrice' in df.columns and 'taxes' in df.columns:
        df['price_per_duration'] = (df['totalPrice'] / (df['total_flight_duration'] + 1e-6)).fillna(0).astype(np.float32)
        df['tax_percentage'] = (df['taxes'] / (df['totalPrice'] + 1e-6)).fillna(0) * 100
        df['tax_percentage'] = df['tax_percentage'].astype(np.float32)
    else: df['price_per_duration'] = 0.0; df['tax_percentage'] = 0.0

    # --- Policy/Convenience ---
    if 'pricingInfo_isAccessTP' in df.columns: df['is_compliant'] = df['pricingInfo_isAccessTP'].fillna(0).astype(np.int8)
    else: df['is_compliant'] = -1
    
    if 'legs0_segments0_baggageAllowance_quantity' in df.columns: df['baggage_leg0_included'] = (df['legs0_segments0_baggageAllowance_quantity'].fillna(0) > 0).astype(np.int8)
    else: df['baggage_leg0_included'] = -1
        
    if 'legs1_segments0_baggageAllowance_quantity' in df.columns: # Giáº£ sá»­ chá»‰ cÃ³ segment 0 Ä‘Æ°á»£c táº£i cho leg 1
        df['baggage_leg1_included'] = (df['legs1_segments0_baggageAllowance_quantity'].fillna(0) > 0).astype(np.int8)
        if 'baggage_leg0_included' in df.columns and df['baggage_leg0_included'].iloc[0] != -1:
             df['baggage_both_legs_included'] = (df['baggage_leg0_included'] & df['baggage_leg1_included']).astype(np.int8)
        else: df['baggage_both_legs_included'] = -1
    else: 
        df['baggage_leg1_included'] = 0 
        if 'baggage_leg0_included' in df.columns and df['baggage_leg0_included'].iloc[0] != -1:
            df['baggage_both_legs_included'] = df['baggage_leg0_included'].astype(np.int8)
        else: df['baggage_both_legs_included'] = -1
    
    # --- Cancellation/Exchange ---
    df['free_cancel'] = -1; df['free_exchange'] = -1
    if 'miniRules0_monetaryAmount' in df.columns and 'miniRules0_percentage' in df.columns:
        df['free_cancel'] = ((pd.to_numeric(df['miniRules0_monetaryAmount'], errors='coerce').fillna(1) == 0) & \
                             (pd.to_numeric(df['miniRules0_percentage'], errors='coerce').fillna(1) == 0)).astype(np.int8)
    if 'miniRules1_monetaryAmount' in df.columns and 'miniRules1_percentage' in df.columns:
        df['free_exchange'] = ((pd.to_numeric(df['miniRules1_monetaryAmount'], errors='coerce').fillna(1) == 0) & \
                              (pd.to_numeric(df['miniRules1_percentage'], errors='coerce').fillna(1) == 0)).astype(np.int8)

    # --- Group-wise Features ---
    group_key = 'ranker_id'
    if group_key not in df.columns: return df

    cols_for_group_features = []
    if 'totalPrice' in df.columns and pd.api.types.is_numeric_dtype(df['totalPrice']):
        cols_for_group_features.append('totalPrice')
        
    print(f"Processing group-wise features for {'train' if is_train else 'test'} on columns: {cols_for_group_features}")
    for col in cols_for_group_features:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            print(f"  Calculating rank for {col}...") # Chá»‰ giá»¯ láº¡i rank
            df[f'{col}_rank_in_group'] = df.groupby(group_key)[col].rank(method='dense', ascending=True).astype(np.float32)
            gc.collect() 
        elif col in df.columns:
             print(f"Warning: Column '{col}' for group feature is not numeric (dtype: {df[col].dtype}). Skipping.")

    # --- User/Company Categorical ---
    user_company_cats_loaded = [c for c in ['sex', 'nationality', 'isVip'] if c in df.columns]
    for col in user_company_cats_loaded:
        if df[col].dtype == 'bool': df[col] = df[col].astype(str)
        df[col] = df[col].fillna('MISSING').astype('category')
    
    binary_cols_loaded = [c for c in ['bySelf', 'isAccess3D'] if c in df.columns] # ThÃªm cÃ¡c cá»™t nÃ y vÃ o initial_core_columns náº¿u muá»‘n sá»­ dá»¥ng
    for col in binary_cols_loaded: df[col] = df[col].fillna(0).astype(np.int8)
    return df

# --- Execution part of Cell 4 ---
print("--- Processing TRAIN_DF ---")
print("Initial datetime conversion for train_df...")
train_df_processed = create_initial_datetime_features(train_df.copy())
del train_df; gc.collect()

print("Applying reduce_mem_usage to train_df_processed...")
train_df_processed = reduce_mem_usage(train_df_processed) # reduce_mem_usage from Cell 2
gc.collect()

print("Creating remaining features for train_df_processed...")
train_df_processed = create_remaining_features(train_df_processed, is_train=True)
gc.collect()

train_labels = train_df_processed['selected']
train_ids = train_df_processed['Id']
train_ranker_ids = train_df_processed['ranker_id']

raw_datetime_col_names = ['requestDate', 'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt']
id_cols_and_target = ['Id', 'ranker_id', 'selected', 'profileId', 'companyID', 'searchRoute']
excluded_for_X_train = id_cols_and_target + raw_datetime_col_names
train_feature_cols = [col for col in train_df_processed.columns if col not in excluded_for_X_train]

X = train_df_processed[train_feature_cols].copy()
y = train_labels.copy()
print(f"Shape of X_train: {X.shape}")
print(f"X_train memory usage: {X.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
del train_df_processed; gc.collect()

print("\n--- Processing TEST_DF ---")
print("Initial datetime conversion for test_df...")
test_df_processed = create_initial_datetime_features(test_df.copy())
del test_df; gc.collect()

print("Applying reduce_mem_usage to test_df_processed...")
test_df_processed = reduce_mem_usage(test_df_processed)
gc.collect()

print("Creating remaining features for test_df_processed...")
test_df_processed = create_remaining_features(test_df_processed, is_train=False)
gc.collect()

# Create X_test
X_test = pd.DataFrame(columns=train_feature_cols, index=test_df_processed.index)
for col in train_feature_cols:
    if col in test_df_processed.columns:
        X_test[col] = test_df_processed[col]
    else:
        print(f"Warning: Feature '{col}' from train not found in processed test_df. Filling with 0.")
        X_test[col] = 0 

# âœ… Save group IDs before cleaning up
groups_test = test_df_processed['ranker_id'].values

del test_df_processed; gc.collect()


print(f"Shape of X_test: {X_test.shape}")
print(f"X_test memory usage: {X_test.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\nFinal shapes before LabelEncoding: X_train: {X.shape}, X_test: {X_test.shape}")


# Cell 5: Label Encoding
categorical_features_for_encoding = []
print("\nIdentifying categorical features for Label Encoding from X.columns...")
for col in X.columns:
    if X[col].dtype.name == 'object' or X[col].dtype.name == 'category':
        print(f"Column '{col}' (dtype: {X[col].dtype}) identified as categorical for encoding.")
        categorical_features_for_encoding.append(col)
        
        le = LabelEncoder()
        if col in X_test.columns:
            combined_col_data = pd.concat([X[col].astype(str), X_test[col].astype(str)], axis=0).unique()
            le.fit(combined_col_data)
            X[col] = le.transform(X[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
        else:
            X[col] = le.fit_transform(X[col].astype(str))

print(f"\nCategorical features processed with LabelEncoder: {categorical_features_for_encoding}")

print("\nChecking for non-numeric columns after LabelEncoding...")
for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        print(f"Warning: Non-numeric column post-LE: {col}, dtype: {X[col].dtype}. Forcing numeric.")
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-1)
        if col in X_test.columns: X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(-1)

final_features_list = list(X.columns)
print(f"\nFinal features for model ({len(final_features_list)}): {final_features_list}")
print("\nX dtypes after all processing:")
print(X.dtypes.value_counts())
gc.collect()


X


y.unique()


# Install required packages
!pip install -U xgboost lightgbm tensorflow scikit-learn

# Imports
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
RANDOM_STATE = 42

# Utility functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x / 10, -500, 500)))

def calculate_hitrate_at_k(df, k=3):
    hits = []
    for ranker_id, group in df.groupby('ranker_id'):
        if len(group) > 10:
            top_k = group.nlargest(k, 'pred')
            hit = (top_k['selected'] == 1).any()
            hits.append(hit)
    return np.mean(hits) if hits else 0.0

def evaluate_model(y_true, y_pred, groups, model_name="Model"):
    df = pd.DataFrame({
        'ranker_id': groups,
        'pred': y_pred,
        'selected': y_true
    })
    top_preds = df.loc[df.groupby('ranker_id')['pred'].idxmax()]
    top_preds['prob'] = sigmoid(top_preds['pred'])
    logloss = log_loss(top_preds['selected'], top_preds['prob'])
    hitrate_at_3 = calculate_hitrate_at_k(df, k=3)
    accuracy = (top_preds['selected'] == 1).mean()
    print(f"{model_name} Validation Metrics:")
    print(f"HitRate@3 (groups >10): {hitrate_at_3:.4f}")
    print(f"LogLoss:                {logloss:.4f}")
    print(f"Top-1 Accuracy:         {accuracy:.4f}")
    print("-" * 40)
    return df, hitrate_at_3, logloss, accuracy

def create_proportional_splits(X, y, train_frac=0.4, val_frac=0.2, test_frac=0.4, random_state=42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-5, "Fractions must sum to 1."
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=random_state, stratify=y)
    val_frac_adj = val_frac / (train_frac + val_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_frac_adj, random_state=random_state, stratify=y_train_val)
    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_tree_data(X_tr, X_val, X_test, cat_features):
    if len(cat_features) == 0:
        return X_tr.values, X_val.values, X_test.values
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    ohe.fit(X_tr[cat_features])
    X_tr_cat = ohe.transform(X_tr[cat_features])
    X_val_cat = ohe.transform(X_val[cat_features])
    X_test_cat = ohe.transform(X_test[cat_features])
    X_tr_final = np.hstack([X_tr.drop(columns=cat_features).values, X_tr_cat])
    X_val_final = np.hstack([X_val.drop(columns=cat_features).values, X_val_cat])
    X_test_final = np.hstack([X_test.drop(columns=cat_features).values, X_test_cat])
    return X_tr_final, X_val_final, X_test_final

def prepare_nn_data(X_tr, X_val, X_test, cat_features):
    num_features = [col for col in X_tr.columns if col not in cat_features]
    scaler = StandardScaler()
    X_tr_num = scaler.fit_transform(X_tr[num_features])
    X_val_num = scaler.transform(X_val[num_features])
    X_test_num = scaler.transform(X_test[num_features])
    label_encoders = {}
    X_tr_cat = np.zeros((len(X_tr), len(cat_features)))
    X_val_cat = np.zeros((len(X_val), len(cat_features)))
    X_test_cat = np.zeros((len(X_test), len(cat_features)))
    for i, col in enumerate(cat_features):
        le = LabelEncoder()
        all_vals = pd.concat([X_tr[col], X_val[col], X_test[col]]).astype(str)
        le.fit(all_vals)
        label_encoders[col] = le
        X_tr_cat[:, i] = le.transform(X_tr[col].astype(str))
        X_val_cat[:, i] = le.transform(X_val[col].astype(str))
        X_test_cat[:, i] = le.transform(X_test[col].astype(str))
    X_tr_nn = np.concatenate([X_tr_num, X_tr_cat], axis=1)
    X_val_nn = np.concatenate([X_val_num, X_val_cat], axis=1)
    X_test_nn = np.concatenate([X_test_num, X_test_cat], axis=1)
    return X_tr_nn, X_val_nn, X_test_nn, scaler, label_encoders

def create_nn_model(input_dim):
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


from sklearn.utils import resample

def stratified_downsample(X, y, fraction=0.6, random_state=42):
    df = X.copy()
    df['target'] = y
    downsampled_df = []

    for label in df['target'].unique():
        class_subset = df[df['target'] == label]
        n_samples = int(len(class_subset) * fraction)
        class_downsampled = resample(
            class_subset, replace=False, n_samples=n_samples, random_state=random_state
        )
        downsampled_df.append(class_downsampled)

    final_df = pd.concat(downsampled_df)
    X_down = final_df.drop(columns='target')
    y_down = final_df['target']
    return X_down.reset_index(drop=True), y_down.reset_index(drop=True)



# Reduce dataset size with class balance
X_reduced, y_reduced = stratified_downsample(X, y, fraction=0.4, random_state=RANDOM_STATE)

# Now split the reduced set
X_tr, X_val, X_test, y_tr, y_val, y_test = create_proportional_splits(X_reduced, y_reduced)

cat_features_final = X_tr.select_dtypes(include=['object', 'category']).columns.tolist()
groups_val = np.arange(len(X_val))

# Train XGBoost
print("Training XGBoost")
X_tr_xgb, X_val_xgb, X_test_xgb = prepare_tree_data(X_tr, X_val, X_test, cat_features_final)
dtrain = xgb.DMatrix(X_tr_xgb, label=y_tr)
dval = xgb.DMatrix(X_val_xgb, label=y_val)
xgb_model = xgb.train({
    'objective': 'binary:logistic', 'eval_metric': 'logloss', 'max_depth': 10, 'min_child_weight': 5,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'lambda': 15.0, 'alpha': 5.0, 'learning_rate': 0.05,
    'seed': RANDOM_STATE, 'n_jobs': -1, 'tree_method': 'hist'
}, dtrain, num_boost_round=300, evals=[(dtrain, 'train'), (dval, 'val')],
   early_stopping_rounds=150, verbose_eval=100)
xgb_val_preds = np.log(xgb_model.predict(dval) / (1 - xgb_model.predict(dval) + 1e-8))
xgb_val_df, xgb_hr3, xgb_logloss, xgb_acc = evaluate_model(y_val, xgb_val_preds, groups_val, "XGBoost")

# Train LightGBM
print("Training LightGBM")
X_tr_lgb, X_val_lgb, X_test_lgb = prepare_tree_data(X_tr, X_val, X_test, cat_features_final)
lgb_train = lgb.Dataset(X_tr_lgb, label=y_tr)
lgb_val = lgb.Dataset(X_val_lgb, label=y_val, reference=lgb_train)
lgb_model = lgb.train({
    'objective': 'binary', 'metric': 'binary_logloss', 'num_leaves': 255, 'max_depth': 12,
    'min_data_in_leaf': 20, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'lambda_l1': 5.0, 'lambda_l2': 15.0, 'learning_rate': 0.05, 'seed': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1
}, lgb_train, valid_sets=[lgb_train, lgb_val], valid_names=['train', 'val'], num_boost_round=300,
   callbacks=[lgb.early_stopping(150), lgb.log_evaluation(100)])
lgb_val_preds = np.log(lgb_model.predict(X_val_lgb) / (1 - lgb_model.predict(X_val_lgb) + 1e-8))
lgb_val_df, lgb_hr3, lgb_logloss, lgb_acc = evaluate_model(y_val, lgb_val_preds, groups_val, "LightGBM")

# # Train Neural Network
# print("Training Neural Network")
# X_tr_nn, X_val_nn, X_test_nn, scaler, label_encoders = prepare_nn_data(X_tr, X_val, X_test, cat_features_final)
# nn_model = create_nn_model(X_tr_nn.shape[1])
# nn_model.fit(X_tr_nn, y_tr, validation_data=(X_val_nn, y_val), epochs=10, batch_size=1024,
#              callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
#                         keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)], verbose=1)
# nn_val_preds = np.log(nn_model.predict(X_val_nn).flatten() / (1 - nn_model.predict(X_val_nn).flatten() + 1e-8))
# nn_val_df, nn_hr3, nn_logloss, nn_acc = evaluate_model(y_val, nn_val_preds, groups_val, "Neural Network")

# # Ensemble
# print("Generating Ensemble Test Predictions")
# dtest_xgb = xgb.DMatrix(X_test_xgb)
# xgb_test_preds = xgb_model.predict(dtest_xgb)
# lgb_test_preds = lgb_model.predict(X_test_lgb)
# nn_test_preds = nn_model.predict(X_test_nn).flatten()
# ensemble_test_preds = (xgb_test_preds + lgb_test_preds + nn_test_preds) / 3
# print("All test predictions generated.")



# Generate final test predictions from XGBoost and LightGBM
dtest_xgb = xgb.DMatrix(X_test_xgb)
xgb_test_preds = xgb_model.predict(dtest_xgb)
lgb_test_preds = lgb_model.predict(X_test_lgb)

# Ensemble (average) or pick one
final_test_preds = (xgb_test_preds + lgb_test_preds) / 2


groups_test = test_df_processed.loc[X_test.index, 'ranker_id'].values



# print("=" * 50)
# print("GENERATING TEST PREDICTIONS WITH XGBOOST")
# print("=" * 50)

# dtest = xgb.DMatrix(X_test_xgb)
# final_test_preds = xgb_model.predict(dtest)
# final_test_preds = np.log(final_test_preds / (1 - final_test_preds + 1e-8))

# print(f"Test predictions shape: {final_test_preds.shape}")
# print(f"Test predictions stats: min={final_test_preds.min():.4f}, max={final_test_preds.max():.4f}, mean={final_test_preds.mean():.4f}")



print("Length of final_test_preds:", len(final_test_preds))
print("Length of groups_test:", len(groups_test))



def predictions_to_ranks(predictions, group_ids):
    df = pd.DataFrame({
        'pred': predictions,
        'ranker_id': group_ids,
        'idx': range(len(predictions))
    })
    df['rank'] = df.groupby('ranker_id')['pred'].rank(method='first', ascending=False)
    df = df.sort_values('idx')
    return df['rank'].values.astype(int)

test_ranks = predictions_to_ranks(final_test_preds, groups_test)



submission = pd.DataFrame({
    'Id': range(len(final_test_preds)),  # Replace with real ID if available
    'ranker_id': groups_test,
    'selected': test_ranks
})

submission.to_csv('xgb_ranking_submission.csv', index=False)
print("Submission saved as 'xgb_ranking_submission.csv'")

# Show sample and statistics
print("\nSample predictions:")
print(submission.head(10))

print("\nSubmission stats:")
print(f"Total: {len(submission)}")
print(f"Unique ranker_ids: {submission['ranker_id'].nunique()}")
print(f"Avg group size: {len(submission) / submission['ranker_id'].nunique():.2f}")


