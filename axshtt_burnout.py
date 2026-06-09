import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

print("ğŸš€ XGBOOST SOLO MODE - SPEED & PERFORMANCE ğŸ�¯")

DATASET_PATH = '/kaggle/input/burnout-datathon-ieeecsmuj/'
train = pd.read_csv(DATASET_PATH + 'train.csv')
val = pd.read_csv(DATASET_PATH + 'val.csv')
test = pd.read_csv(DATASET_PATH + 'test.csv')
sample_submission = pd.read_csv(DATASET_PATH + 'sample_submission.csv')

def power_features(df):
    """Focused feature engineering for XGBoost"""
    df = df.copy()
    
    # Speed transformations (most important)
    if 'Avg_Speed_kmh' in df.columns:
        df['Speed_Sq'] = df['Avg_Speed_kmh'] ** 2
        df['Speed_Cube'] = df['Avg_Speed_kmh'] ** 3
        df['Speed_Inv'] = 1 / (df['Avg_Speed_kmh'] + 0.01)
        df['Speed_Log'] = np.log1p(df['Avg_Speed_kmh'])
        df['Speed_Sqrt'] = np.sqrt(df['Avg_Speed_kmh'])
    
    # Grid transformations
    if 'Grid_Position' in df.columns:
        df['Grid_Inv'] = 1 / (df['Grid_Position'] + 1)
        df['Grid_Log'] = np.log1p(df['Grid_Position'])
        df['Grid_Sq'] = df['Grid_Position'] ** 2
    
    # Temperature
    if 'Track_Temperature_Celsius' in df.columns:
        df['Temp_Sq'] = df['Track_Temperature_Celsius'] ** 2
        df['Temp_Log'] = np.log1p(df['Track_Temperature_Celsius'] + 50)
    
    # Power interactions
    if 'Avg_Speed_kmh' in df.columns and 'Grid_Position' in df.columns:
        df['Speed_Grid_Ratio'] = df['Avg_Speed_kmh'] / (df['Grid_Position'] + 1)
        df['Speed_Grid_Prod'] = df['Avg_Speed_kmh'] * df['Grid_Position']
        df['Speed_Grid_Diff'] = df['Avg_Speed_kmh'] - df['Grid_Position']
        df['Speed_Sq_Grid'] = (df['Avg_Speed_kmh'] ** 2) / (df['Grid_Position'] + 1)
    
    if 'Track_Temperature_Celsius' in df.columns and 'Avg_Speed_kmh' in df.columns:
        df['Temp_Speed_Ratio'] = df['Track_Temperature_Celsius'] / (df['Avg_Speed_kmh'] + 0.01)
        df['Temp_Speed_Prod'] = df['Track_Temperature_Celsius'] * df['Avg_Speed_kmh']
    
    # Penalty interactions (ensure numeric)
    if 'Penalty' in df.columns and 'Avg_Speed_kmh' in df.columns:
        # Make sure Penalty is numeric
        if df['Penalty'].dtype == 'object':
            df['Penalty'] = pd.to_numeric(df['Penalty'], errors='coerce').fillna(0)
        df['Penalty_Speed'] = df['Penalty'] * df['Avg_Speed_kmh']
    
    return df

def turbo_prep():
    """Optimized preprocessing for XGBoost"""
    tr, v, te = train.copy(), val.copy(), test.copy()
    
    # Smart missing value handling
    for df in [tr, v, te]:
        # Penalty special treatment - convert to numeric first
        if 'Penalty' in df.columns:
            df['Has_Penalty'] = df['Penalty'].notna().astype(int)
            # Convert to numeric, coercing errors to NaN
            df['Penalty'] = pd.to_numeric(df['Penalty'], errors='coerce')
            df['Penalty_Zero'] = (df['Penalty'] == 0).astype(int)
            df['Penalty'] = df['Penalty'].fillna(0)
        
        # Numeric columns - use different strategies
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if col != 'Lap_Time_Seconds':
                # For highly skewed data, use median; otherwise mean
                if abs(df[col].skew()) > 1:
                    fill_val = df[col].median()
                else:
                    fill_val = df[col].mean()
                df[col] = df[col].fillna(fill_val)
    
    # Apply power features
    tr, v, te = power_features(tr), power_features(v), power_features(te)
    
    # Enhanced categorical encoding
    cat_cols = tr.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        # Frequency encoding first
        freq_map = tr[col].value_counts().to_dict()
        for df in [tr, v, te]:
            df[f'{col}_freq'] = df[col].map(freq_map).fillna(0)
        
        # Then label encoding
        le = LabelEncoder()
        all_vals = pd.concat([tr[col], v[col], te[col]]).astype(str)
        le.fit(all_vals)
        tr[col] = le.transform(tr[col].astype(str))
        v[col] = le.transform(v[col].astype(str))
        te[col] = le.transform(te[col].astype(str))
    
    # Drop irrelevant columns
    drop_cols = ['Unique ID', 'Rider_ID', 'Rider', 'Rider_name', 'Team_name', 'Bike_name', 'Shortname']
    for col in drop_cols:
        for df in [tr, v, te]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
    
    return tr, v, te

print("âš¡ Turbo preprocessing...")
train_df, val_df, test_df = turbo_prep()

X_train = train_df.drop('Lap_Time_Seconds', axis=1)
y_train = train_df['Lap_Time_Seconds']
X_val = val_df.drop('Lap_Time_Seconds', axis=1)
y_val = val_df['Lap_Time_Seconds']
X_test = test_df.copy()

print(f"ğŸ�¯ Features engineered: {X_train.shape[1]}")

# Outlier removal (keep more aggressive)
z_scores = np.abs((y_train - y_train.mean()) / y_train.std())
mask = z_scores < 2.5  # More aggressive outlier removal
X_train_clean, y_train_clean = X_train[mask], y_train[mask]
print(f"ğŸ§¹ Training samples: {len(X_train_clean)} (removed {len(X_train) - len(X_train_clean)} outliers)")

# OPTIMIZED XGBOOST PARAMETERS
base_params = {
    'n_estimators': 1500,
    'max_depth': 8,
    'learning_rate': 0.06,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'colsample_bylevel': 0.85,
    'reg_alpha': 8,
    'reg_lambda': 8,
    'min_child_weight': 4,
    'gamma': 1.5,
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': -1,
    'objective': 'reg:squarederror'
}

print("ğŸš€ Training optimized XGBoost...")

# Train base model
model = XGBRegressor(**base_params)
model.fit(X_train_clean, y_train_clean, 
          eval_set=[(X_val, y_val)], 
          early_stopping_rounds=75, 
          verbose=False)

base_pred = model.predict(X_val)
base_rmse = mean_squared_error(y_val, base_pred, squared=False)
print(f"ğŸ�¯ Base XGBoost RMSE: {base_rmse:.4f}")

# QUICK HYPERPARAMETER VARIANTS (parallel testing)
variants = [
    # Variant 1: Lower LR, more trees
    {'learning_rate': 0.04, 'n_estimators': 2000, 'max_depth': 7},
    # Variant 2: Higher regularization
    {'reg_alpha': 15, 'reg_lambda': 15, 'min_child_weight': 6},
    # Variant 3: Different tree structure
    {'max_depth': 9, 'min_child_weight': 2, 'gamma': 2.5}
]

best_rmse = base_rmse
best_model = model
best_name = "Base"

print("âš¡ Testing variants...")
for i, variant_params in enumerate(variants):
    variant = base_params.copy()
    variant.update(variant_params)
    
    var_model = XGBRegressor(**variant)
    var_model.fit(X_train_clean, y_train_clean, 
                  eval_set=[(X_val, y_val)], 
                  early_stopping_rounds=50, 
                  verbose=False)
    
    var_pred = var_model.predict(X_val)
    var_rmse = mean_squared_error(y_val, var_pred, squared=False)
    
    print(f"ğŸ”§ Variant {i+1} RMSE: {var_rmse:.4f}")
    
    if var_rmse < best_rmse:
        best_rmse = var_rmse
        best_model = var_model
        best_name = f"Variant {i+1}"

print(f"ğŸ�† BEST MODEL: {best_name} with RMSE: {best_rmse:.4f}")

# FINAL BOOST: Ensemble of top 2 models if improvement is significant
if best_rmse < base_rmse - 0.05:  # Significant improvement
    print("ğŸš€ Creating mini-ensemble...")
    
    # Predictions from both models
    pred1 = model.predict(X_val)
    pred2 = best_model.predict(X_val)
    
    # Optimal weight (simple search)
    best_w = 0.5
    best_ensemble_rmse = float('inf')
    
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        ensemble_pred = w * pred1 + (1-w) * pred2
        ensemble_rmse = mean_squared_error(y_val, ensemble_pred, squared=False)
        if ensemble_rmse < best_ensemble_rmse:
            best_ensemble_rmse = ensemble_rmse
            best_w = w
    
    print(f"ğŸ�¯ Ensemble RMSE: {best_ensemble_rmse:.4f} (weight: {best_w})")
    
    if best_ensemble_rmse < best_rmse:
        # Use ensemble for final prediction
        test_pred1 = model.predict(X_test)
        test_pred2 = best_model.predict(X_test)
        final_pred = best_w * test_pred1 + (1-best_w) * test_pred2
        final_rmse = best_ensemble_rmse
        print("ğŸ�† Using ENSEMBLE for final prediction!")
    else:
        final_pred = best_model.predict(X_test)
        final_rmse = best_rmse
        print("ğŸ�† Using BEST SINGLE MODEL for final prediction!")
else:
    final_pred = best_model.predict(X_test)
    final_rmse = best_rmse

# Create submission
submission = sample_submission.copy()
submission['Lap_Time_Seconds'] = final_pred
submission.to_csv('teamrocket_xgb_only.csv', index=False)

# Results
print(f"\nğŸ�� FINAL RESULTS:")
print(f"ğŸ�¯ Target: RMSE < 2.0 (ideally < 1.0)")
print(f"ğŸ“Š Achieved: {final_rmse:.4f}")
print(f"{'ğŸ�‰ EXCELLENT!' if final_rmse < 1.5 else 'âœ… GOOD PROGRESS!' if final_rmse < 2.0 else 'âš¡ KEEP PUSHING!'}")
print(f"â�±ï¸� Estimated runtime: 15-18 minutes")
print("ğŸšš teamrocket_xgb_only.csv ready for submission!")

if final_rmse > 1.5:
    print(f"\nğŸ’¡ Quick tips to improve further:")
    print(f"   - Try changing outlier threshold to 2.0 or 3.0")
    print(f"   - Increase n_estimators to 2500")
    print(f"   - Experiment with learning_rate: 0.03-0.05")

