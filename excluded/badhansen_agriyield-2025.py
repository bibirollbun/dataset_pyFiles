# AGRIYIELD 2025 - COMPLETE GPU-OPTIMIZED KAGGLE SUBMISSION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, RidgeCV
from itertools import combinations
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import time
import gc
import warnings
warnings.filterwarnings("ignore")

# GPU Detection and Configuration
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"âœ… GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"ğŸ”¥ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        DEVICE = 'cuda'
    else:
        print("âš ï¸� GPU not detected, using CPU")
        DEVICE = 'cpu'
        GPU_AVAILABLE = False
except ImportError:
    print("âš ï¸� PyTorch not available, using CPU")
    GPU_AVAILABLE = False
    DEVICE = 'cpu'

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# =====================================================
# 1. LOAD DATA
# =====================================================

train = pd.read_csv('/kaggle/input/agriyield-2025/train.csv')
test = pd.read_csv('/kaggle/input/agriyield-2025/test.csv')
submission = pd.read_csv('/kaggle/input/agriyield-2025/sample_submission.csv')

print(f"âœ… Train shape: {train.shape}")
print(f"âœ… Test shape: {test.shape}")
print(f"âœ… Submission shape: {submission.shape}")

# Basic data info
print(f"Train columns: {list(train.columns)}")


# =====================================================
# 2. FEATURE ENGINEERING
# =====================================================

def create_advanced_features(df):
    """
    Advanced feature engineering for AgriYield
    """
    df_new = df.copy()
    print(f"ğŸ“Š Starting with {df_new.shape[1]} features...")
    
    # Base features
    base_features = ['soil_ph', 'organic_matter', 'sand_pct', 'temperature', 'humidity', 'rainfall', 'ndvi']
    
    # =====================================================
    # DOMAIN-SPECIFIC AGRICULTURAL FEATURES
    # =====================================================
    
    print("ğŸŒ¾ Creating agricultural domain features...")
    
    # 1. Soil Fertility Index (pH optimized around 6.5)
    df_new['soil_fertility_index'] = (
        df_new['organic_matter'] * np.exp(-((df_new['soil_ph'] - 6.5) ** 2))
    )
    
    # 2. Soil Texture Classification
    def get_soil_texture(sand_pct):
        if sand_pct >= 70:
            return 2  # Sandy
        elif sand_pct <= 30:
            return 0  # Clayey
        else:
            return 1  # Loamy
    
    df_new['soil_texture'] = df_new['sand_pct'].apply(get_soil_texture)
    
    # 3. Climate Stress Index
    df_new['climate_stress_index'] = (
        (df_new['temperature'] - 28).clip(lower=0)**2 +
        (60 - df_new['humidity']).clip(lower=0)**2 +
        (100 - df_new['rainfall']).clip(lower=0)**2
    )
    
    # 4. Growing Degree Days (base temp 10Â°C)
    df_new['growing_degree_days'] = np.maximum(0, df_new['temperature'] - 10)
    
    # 5. Water Stress Index
    df_new['water_stress_index'] = (
        (df_new['temperature'] - 20) * 0.1 +
        (100 - df_new['humidity']) * 0.05 -
        df_new['rainfall'] * 0.02
    ).clip(lower=0)
    
    # 6. Vapor Pressure Deficit (affects transpiration)
    df_new['vpd_approx'] = df_new['temperature'] * (1 - df_new['humidity'] / 100)
    
    # 7. NDVI transformations
    df_new['ndvi_squared'] = df_new['ndvi'] ** 2
    df_new['ndvi_cubed'] = df_new['ndvi'] ** 3
    df_new['vegetation_health'] = df_new['ndvi'] * np.sqrt(df_new['humidity'] / 100)
    
    # 8. Soil Quality Score
    ph_score = 1 - np.abs(df_new['soil_ph'] - 6.5) / 3.5
    organic_score = np.minimum(1, df_new['organic_matter'] / 5.0)
    texture_score = 1 - np.abs(df_new['sand_pct'] - 50) / 50
    df_new['soil_quality_score'] = (ph_score + organic_score + texture_score) / 3
    
    # 9. Moisture Availability
    df_new['moisture_index'] = (
        df_new['rainfall'] * (df_new['humidity'] / 100) / 
        np.maximum(1, df_new['temperature'] - 15)
    )
    
    # =====================================================
    # MATHEMATICAL TRANSFORMATIONS
    # =====================================================
    
    print("ğŸ”¢ Creating mathematical transformations...")
    
    for col in base_features:
        if col in df_new.columns:
            # Non-linear transformations
            df_new[f'{col}_squared'] = df_new[col] ** 2
            df_new[f'{col}_sqrt'] = np.sqrt(np.maximum(0, df_new[col]))
            df_new[f'{col}_log'] = np.log1p(np.maximum(0, df_new[col]))
            df_new[f'{col}_inv'] = 1 / (df_new[col] + 1e-5)
    
    # =====================================================
    # PAIRWISE INTERACTIONS
    # =====================================================
    
    print("ğŸ”— Creating pairwise interactions...")
    
    for f1, f2 in combinations(base_features, 2):
        if f1 in df_new.columns and f2 in df_new.columns:
            df_new[f'{f1}_plus_{f2}'] = df_new[f1] + df_new[f2]
            df_new[f'{f1}_minus_{f2}'] = df_new[f1] - df_new[f2]
            df_new[f'{f1}_times_{f2}'] = df_new[f1] * df_new[f2]
            df_new[f'{f1}_div_{f2}'] = df_new[f1] / (df_new[f2] + 1e-5)
            df_new[f'{f1}_ratio_{f2}'] = df_new[f1] / (df_new[f2] + 1)
    
    # =====================================================
    # KEY RATIOS
    # =====================================================
    
    print("ğŸ“Š Creating key ratios...")
    
    df_new['organic_to_sand_ratio'] = df_new['organic_matter'] / (df_new['sand_pct'] + 1)
    df_new['temp_humidity_ratio'] = df_new['temperature'] / (df_new['humidity'] + 1)
    df_new['rainfall_temp_ratio'] = df_new['rainfall'] / (df_new['temperature'] + 1)
    df_new['ndvi_rainfall_ratio'] = df_new['ndvi'] / (df_new['rainfall'] + 1)
    
    # =====================================================
    # CLEANUP
    # =====================================================
    
    print("ğŸ§¹ Cleaning features...")
    
    # Handle inf/nan
    df_new = df_new.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with median
    for col in df_new.select_dtypes(include=[np.number]).columns:
        if df_new[col].isnull().any():
            df_new[col] = df_new[col].fillna(df_new[col].median())
    
    # Remove constant features
    constant_cols = [col for col in df_new.columns if df_new[col].nunique() == 1]
    if constant_cols:
        print(f"Removing {len(constant_cols)} constant features")
        df_new = df_new.drop(columns=constant_cols)
    
    print(f"âœ… Final feature count: {df_new.shape[1]} (+{df_new.shape[1] - df.shape[1]} new)")
    
    return df_new


train_features = create_advanced_features(train)
test_features = create_advanced_features(test)


# =====================================================
# 3. PREPARE DATA
# =====================================================

print("\\nğŸ“‹ Preparing modeling data...")

# Extract features and target
feature_cols = [col for col in train_features.columns if col not in ['field_id', 'yield']]
common_cols = [col for col in feature_cols if col in test_features.columns]

X = train_features[common_cols]
y = train_features['yield']
X_test = test_features[common_cols]

print(f"Feature matrix: {X.shape}")
print(f"Target vector: {y.shape}")
print(f"Test matrix: {X_test.shape}")

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
)

print(f"Train: {X_train.shape}")
print(f"Validation: {X_val.shape}")


# =====================================================
# 4. FEATURE SCALING
# =====================================================

print("ğŸ�¯ Scaling features... Start")

scaler = RobustScaler()  # More robust to outliers than StandardScaler
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
print("ğŸ�¯ Scaling features... End")


# =====================================================
# 5. MODEL TRAINING - GPU OPTIMIZED
# =====================================================

print("Training GPU-optimized models... Start")

models = {}
val_predictions = {}
test_predictions = {}

# 1. XGBoost with GPU
print("Training XGBoost...")
xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'early_stopping_rounds': 100,
    'eval_metric': 'rmse'
}

if GPU_AVAILABLE:
    xgb_params.update({
        'tree_method': 'gpu_hist',
        'gpu_id': 0
    })

start_time = time.time()
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=False
)
models['XGBoost'] = xgb_model
val_predictions['XGBoost'] = xgb_model.predict(X_val_scaled)
test_predictions['XGBoost'] = xgb_model.predict(X_test_scaled)
print(f"XGBoost training time: {time.time() - start_time:.2f}s")

# 2. LightGBM with GPU
print("Training LightGBM...")
lgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1
}

if GPU_AVAILABLE:
    lgb_params.update({
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    })

start_time = time.time()
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)
models['LightGBM'] = lgb_model
val_predictions['LightGBM'] = lgb_model.predict(X_val_scaled)
test_predictions['LightGBM'] = lgb_model.predict(X_test_scaled)
print(f"LightGBM training time: {time.time() - start_time:.2f}s")

# 3. CatBoost with GPU
print("Training CatBoost...")
cb_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 8,
    'random_seed': RANDOM_STATE,
    'verbose': False,
    'early_stopping_rounds': 100
}

if GPU_AVAILABLE:
    cb_params.update({
        'task_type': 'GPU',
        'gpu_ram_part': 0.8
    })

start_time = time.time()
cb_model = cb.CatBoostRegressor(**cb_params)
cb_model.fit(
    X_train_scaled, y_train,
    eval_set=(X_val_scaled, y_val),
    use_best_model=True
)
models['CatBoost'] = cb_model
val_predictions['CatBoost'] = cb_model.predict(X_val_scaled)
test_predictions['CatBoost'] = cb_model.predict(X_test_scaled)
print(f"CatBoost training time: {time.time() - start_time:.2f}s")

# 4. Random Forest (CPU optimized)
print("Training Random Forest...")
start_time = time.time()
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
models['RandomForest'] = rf_model
val_predictions['RandomForest'] = rf_model.predict(X_val_scaled)
test_predictions['RandomForest'] = rf_model.predict(X_test_scaled)
print(f"Random Forest training time: {time.time() - start_time:.2f}s")

# 5. Extra Trees (CPU optimized)
print("Training Extra Trees...")
start_time = time.time()
et_model = ExtraTreesRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
et_model.fit(X_train_scaled, y_train)
models['ExtraTrees'] = et_model
val_predictions['ExtraTrees'] = et_model.predict(X_val_scaled)
test_predictions['ExtraTrees'] = et_model.predict(X_test_scaled)
print(f"Extra Trees training time: {time.time() - start_time:.2f}s")

# Clean up GPU memory
if GPU_AVAILABLE:
    torch.cuda.empty_cache()
gc.collect()

print("Training GPU-optimized models... End")


# =====================================================
# 6. MODEL EVALUATION
# =====================================================

print("ğŸ“Š Evaluating models...")

model_scores = {}
for name, preds in val_predictions.items():
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    
    model_scores[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    print(f"{name:12} | RMSE: {rmse:6.3f} | MAE: {mae:6.3f} | RÂ²: {r2:6.3f}")

# Find best single model
best_model_name = min(model_scores.keys(), key=lambda x: model_scores[x]['RMSE'])
print(f"ğŸ�† Best single model: {best_model_name} (RMSE: {model_scores[best_model_name]['RMSE']:.3f})")


# =====================================================
# 7. ENSEMBLE METHODS
# =====================================================

print("Creating ensemble predictions...")

# Simple average ensemble
ensemble_val_preds = np.mean(list(val_predictions.values()), axis=0)
ensemble_test_preds = np.mean(list(test_predictions.values()), axis=0)

ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_preds))
print(f"Average Ensemble RMSE: {ensemble_rmse:.3f}")

# Weighted ensemble (based on validation performance)
weights = []
total_inv_rmse = sum(1/model_scores[name]['RMSE'] for name in model_scores.keys())

for name in val_predictions.keys():
    weight = (1/model_scores[name]['RMSE']) / total_inv_rmse
    weights.append(weight)

weighted_val_preds = np.average(list(val_predictions.values()), axis=0, weights=weights)
weighted_test_preds = np.average(list(test_predictions.values()), axis=0, weights=weights)

weighted_rmse = np.sqrt(mean_squared_error(y_val, weighted_val_preds))
print(f"Weighted Ensemble RMSE: {weighted_rmse:.3f}")

# Meta-model (Stacking)
print("ğŸ§  Training meta-model...")
meta_X_train = np.column_stack(list(val_predictions.values()))
meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
meta_model.fit(meta_X_train, y_val)

meta_X_test = np.column_stack(list(test_predictions.values()))
meta_test_preds = meta_model.predict(meta_X_test)
meta_val_preds = meta_model.predict(meta_X_train)

meta_rmse = np.sqrt(mean_squared_error(y_val, meta_val_preds))
print(f"Meta-model RMSE: {meta_rmse:.3f}")


# =====================================================
# 8. FINAL PREDICTIONS
# =====================================================

print("ğŸ�¯ Selecting final predictions...")

# Compare all ensemble methods
ensemble_methods = {
    'Simple Average': (ensemble_rmse, ensemble_test_preds),
    'Weighted Average': (weighted_rmse, weighted_test_preds),
    'Meta-model': (meta_rmse, meta_test_preds),
    f'Best Single ({best_model_name})': (model_scores[best_model_name]['RMSE'], test_predictions[best_model_name])
}

# Select best method
best_method, (best_score, final_predictions) = min(ensemble_methods.items(), key=lambda x: x[1][0])
print(f"ğŸ�† Best method: {best_method} (RMSE: {best_score:.3f})")


# =====================================================
# 9. CREATE SUBMISSION
# =====================================================

print("ğŸ’¾ Creating submission...")

submission['yield'] = final_predictions
submission.to_csv("submission.csv", index=False)

print(f"âœ… Submission saved!")


# =====================================================
# 10. SUMMARY REPORT
# =====================================================

print("FINAL SUMMARY REPORT")


print(f"\\nğŸ”§ Feature Engineering:")
print(f"   Original features: {train.shape[1] - 1}")  # -1 for target
print(f"   Engineered features: {X.shape[1]}")
print(f"   Feature expansion ratio: {X.shape[1] / (train.shape[1] - 1):.1f}x")

print(f"ğŸš€ GPU Acceleration: {'ENABLED' if GPU_AVAILABLE else 'DISABLED'}")

print(f"ğŸ�† Model Performance:")
for name, scores in model_scores.items():
    print(f"   {name:12}: RMSE={scores['RMSE']:.3f}, RÂ²={scores['R2']:.3f}")

print(f"ğŸ¤� Ensemble Performance:")
for method, (score, _) in ensemble_methods.items():
    print(f"   {method:15}: RMSE={score:.3f}")

print(f"ğŸ�¯ Final Selection: {best_method}")
print(f"ğŸ�¯ Final RMSE: {best_score:.3f}")

