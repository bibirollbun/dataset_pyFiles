# =====================================================================
# CELL 1: Import Libraries
# =====================================================================
import numpy as np
import pandas as pd
import pickle
import joblib
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

print("âœ… All libraries imported successfully!")


# =====================================================================
# CELL 2: Load Data
# =====================================================================
# Load competition data
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

# Separate features and target
y = train['accident_risk']
X = train.drop(['id', 'accident_risk'], axis=1)
X_test = test.drop(['id'], axis=1)
test_id = test['id']

print(f"Train shape: {X.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")




# =====================================================================
# CELL 3: Load Pre-trained Models and Feature Engineering Components
# =====================================================================
# Load saved artifacts
with open('/kaggle/input/all-models/best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

with open('/kaggle/input/all-models/feature_engineering.pkl', 'rb') as f:
    fe_artifacts = pickle.load(f)
    label_encoders = fe_artifacts['label_encoders']
    freq_dict = fe_artifacts['freq_dict']

# Load pre-trained models
stacking_model = joblib.load('/kaggle/input/all-models/stacking_model.pkl')
xgb_model = joblib.load('/kaggle/input/all-models/xgboost_model.pkl')
lgb_model = joblib.load('/kaggle/input/all-models/lightgbm_model.pkl')
cat_model = joblib.load('/kaggle/input/all-models/catboost_model.pkl')

# Load saved predictions for reference
oof_preds_saved = np.load('/kaggle/input/all-models/oof_predictions.npy')
test_preds_stack_saved = np.load('/kaggle/input/all-models/test_predictions_stack.npy')

print("âœ… All models and artifacts loaded successfully!")
print(f"ğŸ“Š Saved CV RMSE: {mean_squared_error(y, oof_preds_saved, squared=False):.6f}")


# =====================================================================
# CELL 4: Enhanced Feature Engineering Function
# =====================================================================
def create_features(df, label_encoders, freq_dict, is_train=False):
    """
    Apply EXACT feature engineering matching the trained models
    """
    df = df.copy()
    
    # Identify feature types
    cat_features = df.select_dtypes(include=['object']).columns.tolist()
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 1. Encode categorical features using saved encoders
    for col in cat_features:
        if col in label_encoders:
            le = label_encoders[col]
            # Handle unseen categories
            df[col] = df[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # 2. Polynomial interactions (top features)
    if len(num_features) >= 2:
        for i in range(min(3, len(num_features))):
            for j in range(i+1, min(4, len(num_features))):
                col1, col2 = num_features[i], num_features[j]
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    
    # 3. Statistical aggregations (EXACT as training)
    if len(num_features) >= 3:
        num_data = df[num_features]
        df['num_mean'] = num_data.mean(axis=1)
        df['num_std'] = num_data.std(axis=1)
        df['num_max'] = num_data.max(axis=1)
        df['num_min'] = num_data.min(axis=1)
        df['num_range'] = df['num_max'] - df['num_min']
        df['num_median'] = num_data.median(axis=1)
    
    # 4. Frequency encoding using saved frequencies
    for col in cat_features:
        if col in freq_dict:
            df[f'{col}_freq'] = df[col].map(freq_dict[col])
            df[f'{col}_freq'].fillna(df[f'{col}_freq'].mean(), inplace=True)
    
    return df

print("âœ… Feature engineering function ready!")





# =====================================================================
# CELL 5: Apply Feature Engineering
# =====================================================================
print("ğŸ”§ Applying feature engineering (matching trained models)...")

X_enhanced = create_features(X, label_encoders, freq_dict, is_train=True)
X_test_enhanced = create_features(X_test, label_encoders, freq_dict, is_train=False)

print(f"âœ… Features created: {X_enhanced.shape[1]} (from {X.shape[1]})")
print(f"   New features: {X_enhanced.shape[1] - X.shape[1]}")

# Critical verification: Check feature alignment
print("\nğŸ”� Verifying feature alignment with trained models...")
expected_features = ['road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting', 
                     'weather', 'road_signs_present', 'public_road', 'time_of_day', 
                     'holiday', 'school_season', 'num_reported_accidents', 
                     'num_lanes_x_curvature', 'num_lanes_x_speed_limit', 
                     'num_lanes_x_num_reported_accidents', 'curvature_x_speed_limit', 
                     'curvature_x_num_reported_accidents', 'speed_limit_x_num_reported_accidents', 
                     'num_mean', 'num_std', 'num_max', 'num_min', 'num_range', 'num_median', 
                     'road_type_freq', 'lighting_freq', 'weather_freq', 'time_of_day_freq']

current_features = X_enhanced.columns.tolist()
extra_features = set(current_features) - set(expected_features)
missing_features = set(expected_features) - set(current_features)

if extra_features:
    print(f"âš ï¸�  WARNING: Extra features found: {extra_features}")
    print("   Removing extra features...")
    X_enhanced = X_enhanced[expected_features]
    X_test_enhanced = X_test_enhanced[expected_features]
    print("   âœ… Fixed!")

if missing_features:
    print(f"â�Œ ERROR: Missing features: {missing_features}")
else:
    print("âœ… All features match! Ready for inference.")
    
print(f"\nğŸ“‹ Final feature count: {X_enhanced.shape[1]}")



# =====================================================================
# CELL 6: Fine-tune with Weighted Stacked Ensemble
# =====================================================================
def fine_tuned_predictions(X_train, y_train, X_test, models, stacking):
    """
    Create fine-tuned predictions using:
    1. Pre-trained models (quick inference) - 70%
    2. Light retraining on current data (adaptation) - 30%
    3. Intelligent blending
    """
    
    print("\nğŸ“Š Getting predictions from loaded models...")
    # Strategy 1: Direct predictions from loaded models
    pred_stack_loaded = stacking.predict(X_test)
    pred_xgb_loaded = models['xgb'].predict(X_test)
    pred_lgb_loaded = models['lgb'].predict(X_test)
    pred_cat_loaded = models['cat'].predict(X_test)
    
    print(f"   âœ… Stacking: {pred_stack_loaded.mean():.4f}")
    print(f"   âœ… XGBoost: {pred_xgb_loaded.mean():.4f}")
    print(f"   âœ… LightGBM: {pred_lgb_loaded.mean():.4f}")
    print(f"   âœ… CatBoost: {pred_cat_loaded.mean():.4f}")
    
    # Strategy 2: Quick 3-fold retrain for adaptation
    print("\nğŸ”„ Fine-tuning with 3-fold CV for adaptation...")
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    y_binned = pd.qcut(y_train, q=10, labels=False, duplicates='drop')
    
    pred_xgb_retrain = np.zeros(len(X_test))
    pred_lgb_retrain = np.zeros(len(X_test))
    pred_cat_retrain = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_binned), 1):
        print(f"\n   Fold {fold}/3...")
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Create fresh params with reduced iterations
        xgb_params = best_params['xgb'].copy()
        xgb_params['n_estimators'] = 200
        
        lgb_params = best_params['lgb'].copy()
        lgb_params['n_estimators'] = 200
        
        cat_params = best_params['cat'].copy()
        cat_params['iterations'] = 200
        
        # Light retraining with reduced iterations
        xgb_temp = xgb.XGBRegressor(**xgb_params)
        lgb_temp = lgb.LGBMRegressor(**lgb_params)
        cat_temp = CatBoostRegressor(**cat_params)
        
        xgb_temp.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        lgb_temp.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        cat_temp.fit(X_tr, y_tr, eval_set=(X_val, y_val), 
                     early_stopping_rounds=30, verbose=False)
        
        pred_xgb_retrain += xgb_temp.predict(X_test) / 3
        pred_lgb_retrain += lgb_temp.predict(X_test) / 3
        pred_cat_retrain += cat_temp.predict(X_test) / 3
        
        rmse_xgb = mean_squared_error(y_val, xgb_temp.predict(X_val), squared=False)
        rmse_lgb = mean_squared_error(y_val, lgb_temp.predict(X_val), squared=False)
        rmse_cat = mean_squared_error(y_val, cat_temp.predict(X_val), squared=False)
        print(f"      XGB: {rmse_xgb:.6f} | LGB: {rmse_lgb:.6f} | CAT: {rmse_cat:.6f}")
    
    print(f"\n   âœ… XGBoost (retrained): {pred_xgb_retrain.mean():.4f}")
    print(f"   âœ… LightGBM (retrained): {pred_lgb_retrain.mean():.4f}")
    print(f"   âœ… CatBoost (retrained): {pred_cat_retrain.mean():.4f}")
    
    # Strategy 3: Intelligent blending
    # Weight loaded models (stable) vs retrained (adaptive)
    final_predictions = (
        0.35 * pred_stack_loaded +      # Stacking (proven best)
        0.15 * pred_xgb_loaded +        # Individual loaded models
        0.10 * pred_lgb_loaded +
        0.10 * pred_cat_loaded +
        0.12 * pred_xgb_retrain +       # Retrained (adaptive)
        0.09 * pred_lgb_retrain +
        0.09 * pred_cat_retrain
    )
    
    # Clip to valid range
    final_predictions = np.clip(final_predictions, 0, 1)
    
    return final_predictions, {
        'stack': pred_stack_loaded,
        'xgb_loaded': pred_xgb_loaded,
        'lgb_loaded': pred_lgb_loaded,
        'cat_loaded': pred_cat_loaded,
        'xgb_retrain': pred_xgb_retrain,
        'lgb_retrain': pred_lgb_retrain,
        'cat_retrain': pred_cat_retrain
    }

models_dict = {
    'xgb': xgb_model,
    'lgb': lgb_model,
    'cat': cat_model
}

final_preds, component_preds = fine_tuned_predictions(
    X_enhanced, y, X_test_enhanced, models_dict, stacking_model
)

print(f"\nâœ… Fine-tuning complete!")
print(f"   Prediction range: [{final_preds.min():.4f}, {final_preds.max():.4f}]")
print(f"   Prediction mean: {final_preds.mean():.4f}")
print(f"   Prediction std: {final_preds.std():.4f}")


# =====================================================================
# CELL 7: Create Submission
# =====================================================================
submission['accident_risk'] = final_preds

# Save submission
submission.to_csv('submission.csv', index=False)

print("\n" + "="*70)
print("ğŸ�‰ FINE-TUNED SUBMISSION READY!")
print("="*70)
print(f"ğŸ“Š Previous CV RMSE: {mean_squared_error(y, oof_preds_saved, squared=False):.6f}")
print(f"ğŸ“� Submission saved: submission.csv")
print(f"ğŸ�¯ Strategy: Loaded models (70%) + Retrained adaptive (30%)")
print("="*70)

# Display sample
print("\nğŸ“‹ Sample predictions:")
print(submission.head(10))

