import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

print("âœ… Libraries loaded!")


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
train.head()


# Check for target variable
if 'accident_risk' in train.columns:
    print("Target variable found: accident_risk")
    print(f"\nTarget statistics:")
    print(train['accident_risk'].describe())
    
    # Plot target distribution
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(train['accident_risk'], bins=50, edgecolor='black')
    plt.title('Target Distribution')
    plt.xlabel('Accident Risk')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=train['accident_risk'])
    plt.title('Target Boxplot')
    plt.tight_layout()
    plt.show()

# Data info
print("\nData Info:")
print(train.info())
print("\nMissing values:")
print(train.isnull().sum())



def feature_engineering(df):
    df = df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Risk interaction features
    if 'curvature' in df.columns and 'speed_limit' in df.columns:
        df['curvature_speed_risk'] = df['curvature'] * (100 - df['speed_limit']) / 100
    
    if 'num_lanes' in df.columns and 'curvature' in df.columns:
        df['lane_curvature_ratio'] = df['curvature'] / (df['num_lanes'] + 1)
    
    # Weather + lighting risk
    if 'weather_encoded' in df.columns and 'lighting_encoded' in df.columns:
        df['weather_lighting_risk'] = df['weather_encoded'] * df['lighting_encoded']
    
    # Road complexity score
    if all(col in df.columns for col in ['curvature', 'num_lanes', 'speed_limit']):
        df['road_complexity'] = (df['curvature'] * df['num_lanes']) / (df['speed_limit'] + 1)
    
    # Boolean to numeric
    bool_cols = ['road_signs_present', 'public_road']
    for col in bool_cols:
        if col in df.columns:
            df[col + '_int'] = df[col].astype(int)
    
    # Time-based risk
    time_risk_map = {'morning': 0.6, 'afternoon': 0.4, 'evening': 0.8, 'night': 1.0}
    if 'time_of_day' in df.columns:
        df['time_risk'] = df['time_of_day'].map(time_risk_map).fillna(0.5)
    
    # Weather risk
    weather_risk_map = {'clear': 0.3, 'rainy': 0.8, 'foggy': 0.7, 'snowy': 0.9}
    if 'weather' in df.columns:
        df['weather_risk'] = df['weather'].map(weather_risk_map).fillna(0.5)
    
    # Lighting risk
    lighting_risk_map = {'daylight': 0.3, 'dim': 0.6, 'night': 0.9}
    if 'lighting' in df.columns:
        df['lighting_risk'] = df['lighting'].map(lighting_risk_map).fillna(0.5)
    
    # Combined environmental risk
    if all(col in df.columns for col in ['weather_risk', 'lighting_risk', 'time_risk']):
        df['environmental_risk'] = (df['weather_risk'] + df['lighting_risk'] + df['time_risk']) / 3
    
    return df

# Apply feature engineering
train_fe = feature_engineering(train)
test_fe = feature_engineering(test)

print(f"Features after engineering: {train_fe.shape[1]}")
print("\nNew features created:")
print([col for col in train_fe.columns if col not in train.columns])



# Prepare features
feature_cols = [col for col in train_fe.columns if col not in ['id', 'accident_risk', 'road_type', 'lighting', 'weather', 'time_of_day', 'road_signs_present', 'public_road']]

X = train_fe[feature_cols]
y = train_fe['accident_risk']
X_test = test_fe[feature_cols]

print(f"Training features shape: {X.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"\nFeatures used: {len(feature_cols)}")
print(feature_cols)



# Cross-validation setup
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Storage for predictions
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

# Model 1: LightGBM
print("Training LightGBM...")
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

lgb_test_preds = np.zeros(len(X_test))
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(lgb_params, train_data, num_boost_round=1000, valid_sets=[val_data], 
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    oof_preds[val_idx] = model.predict(X_val)
    lgb_test_preds += model.predict(X_test) / n_folds
    
    print(f"Fold {fold+1} RMSE: {np.sqrt(mean_squared_error(y_val, model.predict(X_val))):.6f}")

lgb_oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"\nâœ… LightGBM OOF RMSE: {lgb_oof_rmse:.6f}")


# Model 2: XGBoost
print("\nTraining XGBoost...")
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist'
}

oof_preds_xgb = np.zeros(len(X))
xgb_test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBRegressor(**xgb_params, n_estimators=1000)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    
    oof_preds_xgb[val_idx] = model.predict(X_val)
    xgb_test_preds += model.predict(X_test) / n_folds
    
    print(f"Fold {fold+1} RMSE: {np.sqrt(mean_squared_error(y_val, model.predict(X_val))):.6f}")

xgb_oof_rmse = np.sqrt(mean_squared_error(y, oof_preds_xgb))
print(f"\nâœ… XGBoost OOF RMSE: {xgb_oof_rmse:.6f}")


# Model 3: CatBoost
print("\nTraining CatBoost...")
oof_preds_cat = np.zeros(len(X))
cat_test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, 
                              random_state=42, verbose=False, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    oof_preds_cat[val_idx] = model.predict(X_val)
    cat_test_preds += model.predict(X_test) / n_folds
    
    print(f"Fold {fold+1} RMSE: {np.sqrt(mean_squared_error(y_val, model.predict(X_val))):.6f}")

cat_oof_rmse = np.sqrt(mean_squared_error(y, oof_preds_cat))
print(f"\nâœ… CatBoost OOF RMSE: {cat_oof_rmse:.6f}")


# Weighted ensemble
print("\n" + "="*50)
print("ENSEMBLE RESULTS")
print("="*50)

# Simple average ensemble
ensemble_test_preds = (lgb_test_preds + xgb_test_preds + cat_test_preds) / 3
ensemble_oof_preds = (oof_preds + oof_preds_xgb + oof_preds_cat) / 3
ensemble_rmse = np.sqrt(mean_squared_error(y, ensemble_oof_preds))

print(f"\nLightGBM OOF RMSE: {lgb_oof_rmse:.6f}")
print(f"XGBoost OOF RMSE: {xgb_oof_rmse:.6f}")
print(f"CatBoost OOF RMSE: {cat_oof_rmse:.6f}")
print(f"Ensemble OOF RMSE: {ensemble_rmse:.6f}")

# Use best performing model or ensemble
final_predictions = ensemble_test_preds

print(f"\nğŸ�¯ Final predictions using: Ensemble of 3 models")
print(f"ğŸ�† Best CV Score: {ensemble_rmse:.6f}")



# Train final model to get feature importance
final_model = lgb.LGBMRegressor(**{k: v for k, v in lgb_params.items() if k != 'verbose'}, 
                                n_estimators=1000, verbose=-1)
final_model.fit(X, y)

# Plot feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(15), y='feature', x='importance', palette='viridis')
plt.title('Top 15 Feature Importances', fontsize=16, fontweight='bold')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))


# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': final_predictions
})

# Ensure predictions are within reasonable bounds
submission['accident_risk'] = submission['accident_risk'].clip(0, 1)

print("Submission Preview:")
print(submission.head(20))
print(f"\nSubmission shape: {submission.shape}")
print(f"Prediction stats:")
print(submission['accident_risk'].describe())

# Save submission
submission.to_csv('submission.csv', index=False)
print("\nâœ… Submission file created successfully!")

