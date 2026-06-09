# Import Libraries and Load Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Define column names
IDCOL = 'id'
TARGET = 'accident_risk'

# Load datasets from Kaggle competition path
print("="*60)
print("LOADING DATA FROM KAGGLE COMPETITION")
print("="*60)

df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

print(f"\n Data loaded successfully!")
print(f"   Train shape: {df.shape}")
print(f"   Test shape:  {df_test.shape}")
print(f"   Sample submission shape: {df_sub.shape}")

# Verify target column exists
assert TARGET in df.columns, f"Target column '{TARGET}' not found in train data!"
assert IDCOL in df.columns and IDCOL in df_test.columns, f"ID column '{IDCOL}' not found!"

print(f"\n Target Statistics:")
print(df[TARGET].describe())




print("="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# Basic info
print("\n Dataset Overview:")
print(f"   Train samples: {len(df):,}")
print(f"   Test samples:  {len(df_test):,}")
print(f"   Total features: {df.shape[1] - 2}")  # Exclude id and target
print(f"   Missing values in train: {df.isnull().sum().sum()}")
print(f"   Missing values in test:  {df_test.isnull().sum().sum()}")

# Display column types
print(f"\n Column Data Types:")
print(df.dtypes.value_counts())

# Target distribution analysis
print(f"\n Target Variable Analysis ('{TARGET}'):")
print(df[TARGET].describe())
print(f"   Range: [{df[TARGET].min():.6f}, {df[TARGET].max():.6f}]")
print(f"   Skewness: {df[TARGET].skew():.4f}")
print(f"   Kurtosis: {df[TARGET].kurtosis():.4f}")

# Plot target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Histogram
axes[0].hist(df[TARGET], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_title(f'Distribution of {TARGET}', fontsize=14, fontweight='bold')
axes[0].set_xlabel(TARGET)
axes[0].set_ylabel('Frequency')
axes[0].grid(alpha=0.3)

# Box plot
axes[1].boxplot(df[TARGET], vert=True, patch_artist=True)
axes[1].set_title(f'Box Plot of {TARGET}', fontsize=14, fontweight='bold')
axes[1].set_ylabel(TARGET)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Identify feature types
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [col for col in numeric_features if col not in [IDCOL, TARGET]]

categorical_features = df.select_dtypes(include=['object', 'bool']).columns.tolist()

print(f"\n Feature Types:")
print(f"   Numeric features: {len(numeric_features)}")
print(f"   Categorical features: {len(categorical_features)}")
print(f"\n   Numeric: {numeric_features[:10]}{'...' if len(numeric_features) > 10 else ''}")
print(f"   Categorical: {categorical_features}")

# Compute Spearman correlation with target for numeric features
print(f"\n Top 15 Features by Spearman Correlation with Target:")
correlations = df[numeric_features].corrwith(df[TARGET], method='spearman').abs().sort_values(ascending=False)
print(correlations.head(15).to_string())

# Visualize top correlations
plt.figure(figsize=(10, 6))
correlations.head(15).plot(kind='barh', color='steelblue', edgecolor='black')
plt.xlabel('|Spearman Correlation|')
plt.title('Top 15 Features by Correlation with Target', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("\n EDA completed!")




print("="*60)
print("DEFINING HELPER FUNCTIONS")
print("="*60)

def coerce_booleans(df):
    """Convert string boolean representations to actual boolean dtype."""
    for col in df.select_dtypes(include=['object']).columns:
        unique_vals = df[col].dropna().unique()
        if set(map(str.lower, map(str, unique_vals))).issubset({'true', 'false', 'yes', 'no', '1', '0'}):
            df[col] = df[col].map({
                'True': True, 'False': False,
                'true': True, 'false': False,
                'yes': True, 'no': False,
                'Yes': True, 'No': False,
                '1': True, '0': False,
                1: True, 0: False
            })
            print(f" converted '{col}' to boolean")
    return df


def map_time_to_angle(time_series):
    """
    Convert time_of_day (0-23) to cyclical features.
    Handles NaN values gracefully.
    """
    time_numeric = pd.to_numeric(time_series, errors='coerce')
    radians = 2 * np.pi * time_numeric / 24.0
    return np.sin(radians), np.cos(radians)


def engineer_features(df):
    """
    Create features based on ACTUAL columns in dataset:
    - road_type, num_lanes, curvature, speed_limit, lighting, weather
    - road_signs_present, public_road, time_of_day, holiday, school_season
    """
    df = df.copy()
    
    print(f"\n Engineering features based on available columns...")
    
    # ----- Interaction: speed_limit × curvature -----
    if 'speed_limit' in df.columns and 'curvature' in df.columns:
        df['speed_x_curv'] = df['speed_limit'] * df['curvature']
        print(f" Created 'speed_x_curv' (speed_limit × curvature)")
    
    # ----- Interaction: weather × road_type -----
    if 'weather' in df.columns and 'road_type' in df.columns:
        df['weather_x_road'] = df['weather'].astype(str) + '_' + df['road_type'].astype(str)
        print(f" Created 'weather_x_road' interaction")
    
    # ----- Risk Flag: High curvature -----
    if 'curvature' in df.columns:
        curv_threshold = df['curvature'].quantile(0.75)
        df['high_curv_flag'] = (df['curvature'] > curv_threshold).astype(int)
        print(f"  Created 'high_curv_flag' (> {curv_threshold:.3f})")
    
    # ----- Risk Flag: High speed limit -----
    if 'speed_limit' in df.columns:
        speed_threshold = df['speed_limit'].quantile(0.75)
        df['high_speed_flag'] = (df['speed_limit'] > speed_threshold).astype(int)
        print(f" Created 'high_speed_flag' (> {speed_threshold:.1f})")
    
    # ----- Risk Flag: Poor lighting conditions -----
    if 'lighting' in df.columns:
        poor_lighting = ['dark', 'dusk', 'dawn', 'Dark', 'Dusk', 'Dawn']
        df['poor_light_flag'] = df['lighting'].isin(poor_lighting).astype(int)
        print(f" Created 'poor_light_flag'")
    
    # ----- Risk Flag: Adverse weather -----
    if 'weather' in df.columns:
        bad_weather = ['rain', 'snow', 'fog', 'Rain', 'Snow', 'Fog', 'rainy', 'snowy', 'foggy']
        df['bad_weather_flag'] = df['weather'].isin(bad_weather).astype(int)
        print(f" Created 'bad_weather_flag'")
    
    # ----- Cyclical Time Encoding (only if not all NaN) -----
    if 'time_of_day' in df.columns:
        if df['time_of_day'].notna().sum() > 0:  # Check if we have any valid values
            sin_time, cos_time = map_time_to_angle(df['time_of_day'])
            df['time_sin'] = sin_time
            df['time_cos'] = cos_time
            print(f"  Created cyclical time features ('time_sin', 'time_cos')")
        else:
            print(f"  Skipped time features (all values are NaN)")
    
    # ----- Lane density (speed_limit / num_lanes) -----
    if 'speed_limit' in df.columns and 'num_lanes' in df.columns:
        df['speed_per_lane'] = df['speed_limit'] / (df['num_lanes'] + 1)  # +1 to avoid division by zero
        print(f"  Created 'speed_per_lane'")
    
    # ----- Composite risk score -----
    if all(col in df.columns for col in ['curvature', 'speed_limit', 'num_lanes']):
        df['composite_risk'] = (df['curvature'] * df['speed_limit']) / (df['num_lanes'] + 1)
        print(f"   Created 'composite_risk'")
    
    print(f"\n Feature engineering completed! New shape: {df.shape}")
    
    return df


print("\n Helper functions defined successfully!")




print("="*60)
print("PHASE 1: BOOLEAN COERCION & FEATURE ENGINEERING")
print("="*60)

# Step 1: Convert string booleans to actual booleans
print("\n converting string booleans to actual boolean dtype...")
df = coerce_booleans(df)
df_test = coerce_booleans(df_test)

print(f"\n  Boolean columns in train: {df.select_dtypes(include=['bool']).shape[1]}")
print(f"   {df.select_dtypes(include=['bool']).columns.tolist()}")
print(f" Boolean columns in test: {df_test.select_dtypes(include=['bool']).shape[1]}")
print(f"   {df_test.select_dtypes(include=['bool']).columns.tolist()}")

# Step 2: Engineer features
print("\n Applying feature engineering to train and test sets...")
original_train_cols = df.shape[1]
original_test_cols = df_test.shape[1]

print("\n Train set:")
df = engineer_features(df)

print("\n Test set:")
df_test = engineer_features(df_test)

# Verification
new_features = [col for col in df.columns if col not in df.columns[:original_train_cols]]

print("\n" + "="*60)
print("VERIFICATION")
print("="*60)
print(f"\n Train set shape: {df.shape}")
print(f" Test set shape:  {df_test.shape}")
print(f"\n New features created: {len(new_features)}")
print(f"   {new_features}")

# Show sample of new features (exclude time features if all NaN)
sample_cols = [col for col in new_features if df[col].notna().sum() > 0][:5]
if sample_cols:
    print(f"\n Sample of engineered features (first 5 rows):")
    print(df[sample_cols].head())

print("\n Boolean coercion and feature engineering completed successfully!")




print("="*60)
print("PHASE 2: TRAIN/VALIDATION SPLIT")
print("="*60)

from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop(['id', 'accident_risk'], axis=1)
y = df['accident_risk']

print(f"\nOriginal dataset:")
print(f"   Features shape: {X.shape}")
print(f"   Target shape: {y.shape}")
print(f"   Target range: [{y.min():.4f}, {y.max():.4f}]")
print(f"   Target mean: {y.mean():.4f}")

# Create stratified bins for regression target
y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')

# Split with stratification
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_binned
)

print(f"\nAfter split:")
print(f"   Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   Val set:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")

print(f"\nTarget distribution:")
print(f"   Train - mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")
print(f"   Val   - mean: {y_val.mean():.4f}, std: {y_val.std():.4f}")

print("\nTrain/validation split completed successfully!")




print("="*60)
print("PHASE 3: PREPROCESSING AND ENCODING")
print("="*60)

from sklearn.preprocessing import LabelEncoder

# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
boolean_cols = X_train.select_dtypes(include=['bool']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nColumn types detected:")
print(f"   Categorical: {len(categorical_cols)} columns")
print(f"   Boolean: {len(boolean_cols)} columns")
print(f"   Numerical: {len(numerical_cols)} columns")

print(f"\nCategorical columns: {categorical_cols}")
print(f"Boolean columns: {boolean_cols}")

# Convert booleans to integers (0/1) for models
for col in boolean_cols:
    X_train[col] = X_train[col].astype(int)
    X_val[col] = X_val[col].astype(int)
    df_test[col] = df_test[col].astype(int)

print(f"\nConverted {len(boolean_cols)} boolean columns to integers")

# Label encode categorical columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    
    # Fit on combined train+val+test to ensure all categories are seen
    all_categories = pd.concat([
        X_train[col].astype(str),
        X_val[col].astype(str),
        df_test[col].astype(str)
    ]).unique()
    
    le.fit(all_categories)
    
    # Transform each set
    X_train[col] = le.transform(X_train[col].astype(str))
    X_val[col] = le.transform(X_val[col].astype(str))
    df_test[col] = le.transform(df_test[col].astype(str))
    
    label_encoders[col] = le
    print(f"   Encoded '{col}': {len(le.classes_)} unique values")

print(f"\nEncoded {len(categorical_cols)} categorical columns")

# Verify all columns are numeric
print(f"\nFinal verification:")
print(f"   Train dtypes: {X_train.dtypes.value_counts().to_dict()}")
print(f"   Validation dtypes: {X_val.dtypes.value_counts().to_dict()}")
print(f"   Test dtypes: {df_test.dtypes.value_counts().to_dict()}")

print("\nPreprocessing and encoding completed successfully!")




print("="*60)
print("PHASE 4: FINAL DATA VALIDATION")
print("="*60)

# Get final feature names
feature_names = X_train.columns.tolist()

print(f"\nFinal feature set:")
print(f"   Total features: {len(feature_names)}")
print(f"   Features: {feature_names}")

# Check for any missing values
print(f"\nMissing values check:")
print(f"   Train: {X_train.isnull().sum().sum()} missing values")
print(f"   Validation: {X_val.isnull().sum().sum()} missing values")
print(f"   Test: {df_test.isnull().sum().sum()} missing values")

# Check for infinite values
print(f"\nInfinite values check:")
print(f"   Train: {np.isinf(X_train.select_dtypes(include=[np.number])).sum().sum()} infinite values")
print(f"   Validation: {np.isinf(X_val.select_dtypes(include=[np.number])).sum().sum()} infinite values")
print(f"   Test: {np.isinf(df_test.select_dtypes(include=[np.number])).sum().sum()} infinite values")

# Display basic statistics
print(f"\nTrain set statistics:")
print(X_train.describe())

print(f"\nTarget statistics:")
print(f"   Train - min: {y_train.min():.4f}, max: {y_train.max():.4f}")
print(f"   Val   - min: {y_val.min():.4f}, max: {y_val.max():.4f}")

print("\nData validation completed successfully!")
print("Ready for model training!")




print("="*60)
print("PHASE 5: FEATURE CLEANUP")
print("="*60)

# Identify columns with all NaN or zero variance
columns_to_drop = []

# Check for all-NaN columns
for col in X_train.columns:
    if X_train[col].isnull().all():
        columns_to_drop.append(col)
        print(f"Found all-NaN column: {col}")

# Check for zero-variance columns
for col in X_train.select_dtypes(include=[np.number]).columns:
    if X_train[col].std() == 0:
        columns_to_drop.append(col)
        print(f"Found zero-variance column: {col}")

columns_to_drop = list(set(columns_to_drop))

if columns_to_drop:
    print(f"\nDropping {len(columns_to_drop)} problematic columns: {columns_to_drop}")
    
    X_train = X_train.drop(columns=columns_to_drop)
    X_val = X_val.drop(columns=columns_to_drop)
    df_test = df_test.drop(columns=columns_to_drop)
    
    print(f"\nAfter cleanup:")
    print(f"   Remaining features: {X_train.shape[1]}")
    print(f"   Train missing values: {X_train.isnull().sum().sum()}")
    print(f"   Validation missing values: {X_val.isnull().sum().sum()}")
    print(f"   Test missing values: {df_test.isnull().sum().sum()}")
else:
    print("\nNo problematic columns found!")

# Final feature list
feature_names = X_train.columns.tolist()
print(f"\nFinal feature list ({len(feature_names)} features):")
for i, feat in enumerate(feature_names, 1):
    print(f"   {i:2d}. {feat}")

print("\nFeature cleanup completed successfully!")
print("Data is ready for model training!")




print("="*60)
print("PHASE 6: CROSS-VALIDATION SETUP")
print("="*60)

from sklearn.model_selection import KFold

# Setup 5-fold cross-validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

print(f"\nCross-validation configuration:")
print(f"   Number of folds: {n_folds}")
print(f"   Shuffle: True")
print(f"   Random state: 42")

# Prepare arrays for out-of-fold predictions
oof_catboost = np.zeros(len(X_train))
oof_lightgbm = np.zeros(len(X_train))
test_preds_catboost = np.zeros(len(df_test))
test_preds_lightgbm = np.zeros(len(df_test))

print(f"\nOut-of-fold arrays initialized:")
print(f"   OOF predictions shape: {oof_catboost.shape}")
print(f"   Test predictions shape: {test_preds_catboost.shape}")

# Verify fold distribution
print(f"\nFold distribution preview:")
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    y_fold_train = y_train.iloc[train_idx]
    y_fold_val = y_train.iloc[val_idx]
    print(f"   Fold {fold_idx}: Train={len(train_idx):6d} ({y_fold_train.mean():.4f}), "
          f"Val={len(val_idx):6d} ({y_fold_val.mean():.4f})")

print("\nCross-validation setup completed successfully!")
print("Ready to train models!")




print("="*60)
print("PHASE 7: MODEL CONFIGURATION")
print("="*60)

from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor

# CatBoost configuration
catboost_params = {
    'iterations': 5000,
    'learning_rate': 0.03,
    'depth': 8,
    'l2_leaf_reg': 3,
    'min_data_in_leaf': 20,
    'random_strength': 0.5,
    'bagging_temperature': 0.2,
    'od_type': 'Iter',
    'od_wait': 50,
    'random_seed': 42,
    'verbose': 500,
    'task_type': 'CPU',
    'thread_count': -1
}

# LightGBM configuration
lightgbm_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 64,
    'learning_rate': 0.03,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_estimators': 5000,
    'verbose': -1,
    'n_jobs': -1
}

print("\nCatBoost Configuration:")
for key, value in catboost_params.items():
    print(f"   {key:20s}: {value}")

print("\nLightGBM Configuration:")
for key, value in lightgbm_params.items():
    print(f"   {key:20s}: {value}")

# Identify categorical features for CatBoost
cat_features = [col for col in X_train.columns if X_train[col].dtype == 'int64' and 
                col in ['road_type', 'lighting', 'weather', 'weather_x_road']]

print(f"\nCategorical features for CatBoost: {cat_features}")
print(f"Total features: {len(feature_names)}")

print("\nModel configurations completed successfully!")
print("Ready to start training!")



# ============================================================
# PHASE 8: MODEL TRAINING (5-Fold CV) -- Robust & Error-free
# ============================================================

import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

print("="*60)
print("PHASE 8: MODEL TRAINING")
print("="*60)

# -------------------------------
# 0) Basic safety initializations
# -------------------------------
# If these exist already in your notebook, this won't hurt.
n_folds = 5 if 'n_folds' not in globals() else n_folds
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42) if 'kf' not in globals() else kf

# Make safe copies to avoid chained assignment issues
X_train_base = X_train.copy()
X_test_base  = df_test.copy()

# Keep only cat features that actually exist
valid_cat_features = [c for c in cat_features if c in X_train_base.columns]
if len(valid_cat_features) == 0:
    raise ValueError("No valid categorical features found in X_train. Check `cat_features` list.")

print("\nPreparing categorical features (two pipelines: CatBoost & LightGBM)...")

# ----------------------------------------------------------------
# 1) CATBOOST: keep categorical columns as STRINGS (no numeric enc)
# ----------------------------------------------------------------
def to_catboost_strings(df, cat_cols):
    """Convert categorical columns to strings; fill NaN as 'NA'."""
    df_out = df.copy()
    for c in cat_cols:
        # Cast to string and unify missing values
        df_out[c] = df_out[c].astype(str)
        # After astype(str), NaNs become 'nan' string; that's fine for CatBoost
    return df_out

X_train_cb = to_catboost_strings(X_train_base, valid_cat_features)
X_test_cb  = to_catboost_strings(X_test_base,  valid_cat_features)

# Ensure same column order between train and test
X_test_cb = X_test_cb[X_train_cb.columns]

# -------------------------------------------------------------------------
# 2) LIGHTGBM: Ordinal-encode ONLY the categorical columns (stable mapping)
# -------------------------------------------------------------------------
# Fit encoder on the union of train+test to stabilize codes (avoids unseen)
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_lgb = X_train_base.copy()
X_test_lgb  = X_test_base.copy()

if len(valid_cat_features) > 0:
    union_cats = pd.concat([X_train_lgb[valid_cat_features], X_test_lgb[valid_cat_features]], axis=0)
    oe.fit(union_cats)
    X_train_lgb[valid_cat_features] = oe.transform(X_train_lgb[valid_cat_features]).astype(np.int32)
    X_test_lgb[valid_cat_features]  = oe.transform(X_test_lgb[valid_cat_features]).astype(np.int32)

# Ensure same column order between train and test
X_test_lgb = X_test_lgb[X_train_lgb.columns]

# ----------------------------
# 3) CV storage & sanity prints
# ----------------------------
fold_scores_catboost = []
fold_scores_lightgbm = []

# OOF and test preds
oof_catboost = np.zeros(len(X_train_cb), dtype=float)
oof_lightgbm = np.zeros(len(X_train_cb), dtype=float)
test_preds_catboost = np.zeros(len(X_test_cb), dtype=float)
test_preds_lightgbm = np.zeros(len(X_test_lgb), dtype=float)

print(f"\nCategorical features for CatBoost (names): {valid_cat_features}")
print("Starting 5-Fold Cross-Validation Training...\n")

# --------------------
# 4) CV training loop
# --------------------
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_cb), 1):
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx}/{n_folds}")
    print(f"{'='*60}")

    # Split for this fold (CatBoost & LightGBM use their own matrices)
    X_cb_tr, X_cb_val   = X_train_cb.iloc[train_idx], X_train_cb.iloc[val_idx]
    X_lgb_tr, X_lgb_val = X_train_lgb.iloc[train_idx], X_train_lgb.iloc[val_idx]
    y_tr, y_val         = y_train.iloc[train_idx], y_train.iloc[val_idx]

    print(f"Train size: {len(X_cb_tr)}, Val size: {len(X_cb_val)}")

    # ------------------------
    # CatBoost Training
    # ------------------------
    print(f"\nTraining CatBoost (Fold {fold_idx})...")
    start_time = time.time()

    # If hyperparams are defined elsewhere, this uses them; otherwise fallback
    default_cb_params = dict(
        loss_function='RMSE',
        eval_metric='RMSE',
        learning_rate=0.05,
        depth=8,
        iterations=20000,   # large; rely on early stopping
        random_seed=42,
        od_type='IncToDec',
        od_wait=50,
        verbose=False
    )
    cb_params = catboost_params if 'catboost_params' in globals() else default_cb_params
    cb_model = CatBoostRegressor(**cb_params)

    cb_model.fit(
        X_cb_tr, y_tr,
        eval_set=(X_cb_val, y_val),
        cat_features=valid_cat_features,  # pass NAMES, not indices
        use_best_model=True,
        verbose=False
    )

    cb_time = round(time.time() - start_time, 1)

    # Predict on validation & test
    cb_val_preds = cb_model.predict(X_cb_val)
    cb_test_preds = cb_model.predict(X_test_cb)

    # RMSE
    cb_rmse = float(np.sqrt(mean_squared_error(y_val, cb_val_preds)))
    fold_scores_catboost.append(cb_rmse)

    # Store OOF & accumulate test preds
    oof_catboost[val_idx] = cb_val_preds
    test_preds_catboost += cb_test_preds / n_folds

    print(f"CatBoost - RMSE: {cb_rmse:.6f}, Time: {cb_time}s, Best Iteration: {cb_model.best_iteration_}")

    # ------------------------
    # LightGBM Training
    # ------------------------
    print(f"\nTraining LightGBM (Fold {fold_idx})...")
    start_time = time.time()

    default_lgb_params = dict(
        objective='regression',
        metric='rmse',
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=100000,  # large; rely on early stopping
        random_state=42
    )
    lgb_params = lightgbm_params if 'lightgbm_params' in globals() else default_lgb_params
    lgb_model = LGBMRegressor(**lgb_params)

    lgb_model.fit(
        X_lgb_tr, y_tr,
        eval_set=[(X_lgb_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

    lgb_time = round(time.time() - start_time, 1)

    # Predict on validation & test
    lgb_val_preds  = lgb_model.predict(X_lgb_val)
    lgb_test_preds = lgb_model.predict(X_test_lgb)

    # RMSE
    lgb_rmse = float(np.sqrt(mean_squared_error(y_val, lgb_val_preds)))
    fold_scores_lightgbm.append(lgb_rmse)

    # Store OOF & accumulate test preds
    oof_lightgbm[val_idx] = lgb_val_preds
    test_preds_lightgbm += lgb_test_preds / n_folds

    print(f"LightGBM - RMSE: {lgb_rmse:.6f}, Time: {lgb_time}s, Best Iteration: {getattr(lgb_model, 'best_iteration_', 'n/a')}")

    # Fold summary
    print(f"\nFold {fold_idx} Summary:")
    print(f"   CatBoost RMSE : {cb_rmse:.6f}")
    print(f"   LightGBM RMSE : {lgb_rmse:.6f}")
    print(f"   Difference    : {abs(cb_rmse - lgb_rmse):.6f}")

print(f"\n{'='*60}")
print("TRAINING COMPLETED")
print(f"{'='*60}")

# Keep the last trained models for later (e.g., feature importance)
final_cb_model = cb_model
final_lgb_model = lgb_model

print("\nAll folds trained successfully!")

# ---------------------------------------------
# (Optional) quick stability print at the end:
# ---------------------------------------------
print("\nCross-Validation Stability:")
print(f"   CatBoost CV Std : {np.std(fold_scores_catboost):.6f}")
print(f"   LightGBM CV Std : {np.std(fold_scores_lightgbm):.6f}")



print("="*60)
print("PHASE 9: CROSS-VALIDATION RESULTS")
print("="*60)

from sklearn.metrics import mean_squared_error
import numpy as np

# ------------------------------------------------------------
# 1) Compute OOF RMSEs for both models (lower is better)
#    oof_* must have length == len(y_train)
# ------------------------------------------------------------
oof_rmse_catboost  = float(np.sqrt(mean_squared_error(y_train, oof_catboost)))
oof_rmse_lightgbm  = float(np.sqrt(mean_squared_error(y_train, oof_lightgbm)))

# ------------------------------------------------------------
# 2) Pretty-print aggregate CV stats (mean/std across folds)
#    fold_scores_* are per-fold RMSEs, gathered during CV loop
# ------------------------------------------------------------
print("\nOut-of-Fold (OOF) Results:")
print(f"{'Model':<15} {'Mean CV RMSE':<15} {'Std CV RMSE':<15} {'OOF RMSE':<15}")
print("-"*60)
print(f"{'CatBoost':<15} {np.mean(fold_scores_catboost):<15.6f} {np.std(fold_scores_catboost):<15.6f} {oof_rmse_catboost:<15.6f}")
print(f"{'LightGBM':<15} {np.mean(fold_scores_lightgbm):<15.6f} {np.std(fold_scores_lightgbm):<15.6f} {oof_rmse_lightgbm:<15.6f}")

# ------------------------------------------------------------
# 3) Per-fold listing (already computed during CV)
# ------------------------------------------------------------
print("\nPer-Fold Scores:")
print(f"{'Fold':<10} {'CatBoost RMSE':<20} {'LightGBM RMSE':<20}")
print("-"*50)
for i, (cb_score, lgb_score) in enumerate(zip(fold_scores_catboost, fold_scores_lightgbm), 1):
    print(f"{'Fold ' + str(i):<10} {cb_score:<20.6f} {lgb_score:<20.6f}")

# ------------------------------------------------------------
# 4) "Validation" check on the last CV fold, aligned correctly.
#    Do NOT use X_val/y_val defined elsewhere; instead,
#    rebuild the last fold's validation indices and compare
#    y_true with the corresponding slice from OOF predictions.
# ------------------------------------------------------------
# Recreate the same splits and fetch the last fold indices
last_train_idx, last_val_idx = list(kf.split(X_train))[-1]

# Ground-truth on the last validation fold
y_val_last = y_train.iloc[last_val_idx]

# Predictions for that fold come from the OOF arrays
val_rmse_catboost_last = float(np.sqrt(mean_squared_error(y_val_last, oof_catboost[last_val_idx])))
val_rmse_lightgbm_last = float(np.sqrt(mean_squared_error(y_val_last, oof_lightgbm[last_val_idx])))

print(f"\nLast-Fold Validation Results (from OOF slices):")
print(f"   CatBoost RMSE : {val_rmse_catboost_last:.6f}")
print(f"   LightGBM RMSE : {val_rmse_lightgbm_last:.6f}")

print("\nCross-validation evaluation completed!")




print("="*60)
print("PHASE 10: ENSEMBLE OPTIMIZATION")
print("="*60)

print("\nSearching for optimal blend weights...")

best_weight = 0.0
best_rmse = float('inf')
weight_results = []

# Grid search for optimal weight
for weight in np.arange(0.0, 1.01, 0.05):
    blend_preds = weight * oof_catboost + (1 - weight) * oof_lightgbm
    blend_rmse = np.sqrt(mean_squared_error(y_train, blend_preds))
    weight_results.append((weight, blend_rmse))
    
    if blend_rmse < best_rmse:
        best_rmse = blend_rmse
        best_weight = weight

print(f"\nOptimal blend weight: {best_weight:.2f}")
print(f"   CatBoost weight : {best_weight:.2f}")
print(f"   LightGBM weight : {1-best_weight:.2f}")
print(f"   Blend OOF RMSE  : {best_rmse:.6f}")

print(f"\nComparison:")
print(f"   CatBoost alone  : {oof_rmse_catboost:.6f}")
print(f"   LightGBM alone  : {oof_rmse_lightgbm:.6f}")
print(f"   Optimal Blend   : {best_rmse:.6f}")

improvement_cb = ((oof_rmse_catboost - best_rmse) / oof_rmse_catboost) * 100
improvement_lgb = ((oof_rmse_lightgbm - best_rmse) / oof_rmse_lightgbm) * 100

print(f"\nImprovement:")
print(f"   vs CatBoost : {improvement_cb:+.3f}%")
print(f"   vs LightGBM : {improvement_lgb:+.3f}%")

# Create final blend predictions for test set
test_preds_blend = best_weight * test_preds_catboost + (1 - best_weight) * test_preds_lightgbm

print("\nEnsemble optimization completed!")



print("="*60)
print("PHASE 11: SUBMISSION PREPARATION")
print("="*60)

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 0) Ensure blend predictions exist (if not created earlier)
# ------------------------------------------------------------
if 'test_preds_blend' not in globals():
    assert 'best_weight' in globals(), "best_weight is missing."
    test_preds_blend = best_weight * test_preds_catboost + (1 - best_weight) * test_preds_lightgbm

# ------------------------------------------------------------
# 1) Clip predictions to valid range [0, 1]
# ------------------------------------------------------------
test_preds_catboost_clipped = np.clip(test_preds_catboost, 0, 1)
test_preds_lightgbm_clipped = np.clip(test_preds_lightgbm, 0, 1)
test_preds_blend_clipped = np.clip(test_preds_blend, 0, 1)

print("\nPrediction Statistics:")
print(f"{'Model':<15} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
print("-"*60)
print(f"{'CatBoost':<15} {test_preds_catboost_clipped.min():<12.6f} {test_preds_catboost_clipped.max():<12.6f} {test_preds_catboost_clipped.mean():<12.6f} {test_preds_catboost_clipped.std():<12.6f}")
print(f"{'LightGBM':<15} {test_preds_lightgbm_clipped.min():<12.6f} {test_preds_lightgbm_clipped.max():<12.6f} {test_preds_lightgbm_clipped.mean():<12.6f} {test_preds_lightgbm_clipped.std():<12.6f}")
print(f"{'Blend':<15} {test_preds_blend_clipped.min():<12.6f} {test_preds_blend_clipped.max():<12.6f} {test_preds_blend_clipped.mean():<12.6f} {test_preds_blend_clipped.std():<12.6f}")

# ------------------------------------------------------------
# 2) Build submissions directly from df_sub template
#    (df_sub should come from sample_submission and already
#     contain the correct ID column(s) in the right order.)
# ------------------------------------------------------------
def make_submission(template_df: pd.DataFrame, preds: np.ndarray, path: str) -> pd.DataFrame:
    """Fill the 'accident_risk' column of a template (df_sub) and save it."""
    sub = template_df.copy()
    assert len(sub) == len(preds), f"Length mismatch: template {len(sub)} vs preds {len(preds)}"
    sub['accident_risk'] = preds
    sub.to_csv(path, index=False)
    return sub

submission_catboost = make_submission(df_sub, test_preds_catboost_clipped, 'submission_catboost.csv')
submission_lightgbm = make_submission(df_sub, test_preds_lightgbm_clipped, 'submission_lightgbm.csv')
submission_blend    = make_submission(df_sub, test_preds_blend_clipped,    'submission_blend.csv')

print("\nSubmission files created:")
print(f"   1. submission_catboost.csv  (OOF RMSE: {oof_rmse_catboost:.6f})")
print(f"   2. submission_lightgbm.csv  (OOF RMSE: {oof_rmse_lightgbm:.6f})")
print(f"   3. submission_blend.csv     (OOF RMSE: {best_rmse:.6f}) <- RECOMMENDED")

# ------------------------------------------------------------
# 3) (Optional) choose the winner by lowest OOF RMSE and also
#    write a unified 'submission.csv' for convenience.
# ------------------------------------------------------------
scores = {
    "catboost": oof_rmse_catboost,
    "lightgbm": oof_rmse_lightgbm,
    "blend":    best_rmse
}
winner = min(scores, key=scores.get)
if winner == "catboost":
    final_preds = test_preds_catboost_clipped
elif winner == "lightgbm":
    final_preds = test_preds_lightgbm_clipped
else:
    final_preds = test_preds_blend_clipped

submission_final = make_submission(df_sub, final_preds, 'submission.csv')
print(f"\nFinal selection -> {winner.upper()} (OOF RMSE: {scores[winner]:.6f})")
print("\nSubmission format preview (top-5 rows of final file):")
print(submission_final.head())

print("\nSubmission preparation completed successfully!")




print("="*60)
print("PHASE 12: FEATURE IMPORTANCE ANALYSIS")
print("="*60)

import matplotlib.pyplot as plt

# Get feature importance from LightGBM
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': final_lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features (LightGBM):")
print(f"{'Rank':<6} {'Feature':<25} {'Importance':<15}")
print("-"*46)
for idx, row in feature_importance.head(15).iterrows():
    print(f"{feature_importance.index.get_loc(idx)+1:<6} {row['feature']:<25} {row['importance']:<15.1f}")

# Plot feature importance
plt.figure(figsize=(10, 8))
plt.barh(range(len(feature_importance)), feature_importance['importance'])
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Importance')
plt.title('Feature Importance (LightGBM)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nFeature importance plot saved as 'feature_importance.png'")
print("\nFeature importance analysis completed!")




print("\n" + "="*60)
print("PIPELINE EXECUTION SUMMARY")
print("="*60)

print("\nDataset Information:")
print(f"   Training samples    : {len(X_train):,}")
print(f"   Validation samples  : {len(X_val):,}")
print(f"   Test samples        : {len(df_test):,}")
print(f"   Number of features  : {len(feature_names)}")

print("\nModel Performance (Out-of-Fold):")
print(f"   CatBoost RMSE      : {oof_rmse_catboost:.6f}")
print(f"   LightGBM RMSE      : {oof_rmse_lightgbm:.6f}")
print(f"   Ensemble RMSE      : {best_rmse:.6f} (Weight: CB={best_weight:.2f}, LGB={1-best_weight:.2f})")

print("\nCross-Validation Stability:")
print(f"   CatBoost CV Std    : {np.std(fold_scores_catboost):.6f}")
print(f"   LightGBM CV Std    : {np.std(fold_scores_lightgbm):.6f}")

print("\nFiles Generated:")
print("   1. submission_catboost.csv")
print("   2. submission_lightgbm.csv")
print("   3. submission_blend.csv (RECOMMENDED)")
print("   4. feature_importance.png")

print("\n" + "="*60)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nReady for Kaggle submission!")



# FINAL CELL — choose winner by lowest OOF RMSE and save submission.csv
# (Assumes you already computed these: oof_rmse_catboost, oof_rmse_lightgbm, best_rmse,
#  and you have df_sub + test_preds_* from earlier cells.)

import numpy as np
import pandas as pd

# If blend preds not explicitly saved earlier, compute them now
if 'test_preds_blend' not in globals():
    assert 'best_weight' in globals(), "best_weight is missing"
    test_preds_blend = best_weight * test_preds_catboost + (1 - best_weight) * test_preds_lightgbm

# Clip to [0, 1] for safety
preds = {
    "catboost": np.clip(test_preds_catboost,  0, 1),
    "lightgbm": np.clip(test_preds_lightgbm,  0, 1),
    "blend":    np.clip(test_preds_blend,     0, 1),
}
scores = {
    "catboost": oof_rmse_catboost,
    "lightgbm": oof_rmse_lightgbm,
    "blend":    best_rmse
}
winner = min(scores, key=scores.get)
print(f"Winner by OOF RMSE -> {winner.upper()}  (RMSE={scores[winner]:.6f})")

# Build final submission from df_sub template
sub = df_sub.copy()
assert len(sub) == len(preds[winner]), "Length mismatch between df_sub and predictions!"
sub['accident_risk'] = preds[winner]
sub.to_csv('submission.csv', index=False)
print("Saved: submission.csv")
print(sub.head())


