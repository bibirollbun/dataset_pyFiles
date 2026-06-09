# Kaggle Notebook: Predicting Road Accident Risk
# Playground Series - Season 5, Episode 10

# This notebook performs a comprehensive EDA, feature engineering, model training, and submission generation.

# =====================
# 1. Library Imports
# =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# =====================
# 2. Load Datasets
# =====================
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

# Optional: Use the larger synthetic datasets for augmentation
synthetic_100k = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')

# =====================
# 3. Exploratory Data Analysis (EDA)
# =====================
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print("\nTrain Info:")
print(train.info())
print("\nTrain Description:")
print(train.describe())

# Target distribution
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.histplot(train['accident_risk'], bins=50, kde=True)
plt.title('Accident Risk Distribution')
plt.subplot(1,2,2)
sns.boxplot(y=train['accident_risk'])
plt.title('Accident Risk Box Plot')
plt.tight_layout()
plt.show()

# Check missing values
missing_train = train.isnull().sum()
missing_test = test.isnull().sum()
print("\nMissing values in train:")
print(missing_train[missing_train>0] if missing_train.sum() > 0 else "No missing values")
print("\nMissing values in test:")
print(missing_test[missing_test>0] if missing_test.sum() > 0 else "No missing values")

# Feature correlations - ONLY numeric columns
numeric_cols_for_corr = train.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(12,10))
sns.heatmap(train[numeric_cols_for_corr].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix (Numeric Features)')
plt.tight_layout()
plt.show()

# Categorical feature analysis
cat_cols = train.select_dtypes(include=['object', 'bool']).columns.tolist()
print(f"\nCategorical columns: {cat_cols}")

for col in cat_cols:
    if col != 'id':
        print(f"\n{col} value counts:")
        print(train[col].value_counts())
        
        # Plot categorical features vs target
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        train.groupby(col)['accident_risk'].mean().plot(kind='bar')
        plt.title(f'Mean Accident Risk by {col}')
        plt.ylabel('Mean Accident Risk')
        plt.xticks(rotation=45)
        plt.subplot(1,2,2)
        train.groupby(col)['accident_risk'].count().plot(kind='bar')
        plt.title(f'Count by {col}')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# =====================
# 4. Feature Engineering
# =====================
# Store ids for submission
test_ids = test['id'].copy()

# Drop id from both datasets
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

# Identify categorical and numerical columns
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
bool_cols = train.select_dtypes(include=['bool']).columns.tolist()
num_cols = train.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'accident_risk' in num_cols:
    num_cols.remove('accident_risk')

print(f"\nCategorical columns: {cat_cols}")
print(f"Boolean columns: {bool_cols}")
print(f"Numerical columns: {num_cols}")

# Convert boolean to int
for col in bool_cols:
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)

# Create interaction features (before encoding categoricals)
print("\nCreating interaction features...")
interaction_features = []

# Numerical interactions
if len(num_cols) >= 2:
    # Core interactions - TOP performers
    train['speed_curvature_interaction'] = train['speed_limit'] * train['curvature']
    test['speed_curvature_interaction'] = test['speed_limit'] * test['curvature']
    interaction_features.append('speed_curvature_interaction')
    
    train['lanes_speed_ratio'] = train['num_lanes'] / (train['speed_limit'] + 1)
    test['lanes_speed_ratio'] = test['num_lanes'] / (test['speed_limit'] + 1)
    interaction_features.append('lanes_speed_ratio')
    
    if 'num_reported_accidents' in train.columns:
        train['accidents_per_lane'] = train['num_reported_accidents'] / (train['num_lanes'] + 1)
        test['accidents_per_lane'] = test['num_reported_accidents'] / (test['num_lanes'] + 1)
        interaction_features.append('accidents_per_lane')
        
        # Additional powerful interactions
        train['accidents_curvature'] = train['num_reported_accidents'] * train['curvature']
        test['accidents_curvature'] = test['num_reported_accidents'] * test['curvature']
        interaction_features.append('accidents_curvature')
        
        train['accidents_speed'] = train['num_reported_accidents'] * train['speed_limit']
        test['accidents_speed'] = test['num_reported_accidents'] * test['speed_limit']
        interaction_features.append('accidents_speed')
        
        # Complex multi-way interactions
        train['accidents_speed_curvature'] = train['num_reported_accidents'] * train['speed_limit'] * train['curvature']
        test['accidents_speed_curvature'] = test['num_reported_accidents'] * test['speed_limit'] * test['curvature']
        interaction_features.append('accidents_speed_curvature')
        
        train['accident_density'] = train['num_reported_accidents'] / (train['num_lanes'] * train['speed_limit'] + 1)
        test['accident_density'] = test['num_reported_accidents'] / (test['num_lanes'] * test['speed_limit'] + 1)
        interaction_features.append('accident_density')
    
    # Polynomial features
    train['curvature_squared'] = train['curvature'] ** 2
    test['curvature_squared'] = test['curvature'] ** 2
    interaction_features.append('curvature_squared')
    
    train['curvature_cubed'] = train['curvature'] ** 3
    test['curvature_cubed'] = test['curvature'] ** 3
    interaction_features.append('curvature_cubed')
    
    train['speed_squared'] = train['speed_limit'] ** 2
    test['speed_squared'] = test['speed_limit'] ** 2
    interaction_features.append('speed_squared')
    
    train['lanes_squared'] = train['num_lanes'] ** 2
    test['lanes_squared'] = test['num_lanes'] ** 2
    interaction_features.append('lanes_squared')
    
    # Risk indicators
    train['high_risk_indicator'] = ((train['curvature'] > 0.7) & 
                                     (train['speed_limit'] >= 60)).astype(int)
    test['high_risk_indicator'] = ((test['curvature'] > 0.7) & 
                                    (test['speed_limit'] >= 60)).astype(int)
    interaction_features.append('high_risk_indicator')
    
    train['extreme_curvature'] = (train['curvature'] > 0.8).astype(int)
    test['extreme_curvature'] = (test['curvature'] > 0.8).astype(int)
    interaction_features.append('extreme_curvature')
    
    train['low_speed_high_curve'] = ((train['speed_limit'] <= 35) & 
                                      (train['curvature'] > 0.6)).astype(int)
    test['low_speed_high_curve'] = ((test['speed_limit'] <= 35) & 
                                     (test['curvature'] > 0.6)).astype(int)
    interaction_features.append('low_speed_high_curve')
    
    # Binning features
    train['speed_bin'] = pd.cut(train['speed_limit'], bins=[0, 35, 50, 65, 100], labels=[0, 1, 2, 3])
    test['speed_bin'] = pd.cut(test['speed_limit'], bins=[0, 35, 50, 65, 100], labels=[0, 1, 2, 3])
    interaction_features.append('speed_bin')
    
    train['curvature_bin'] = pd.cut(train['curvature'], bins=[0, 0.3, 0.6, 0.8, 1.0], labels=[0, 1, 2, 3])
    test['curvature_bin'] = pd.cut(test['curvature'], bins=[0, 0.3, 0.6, 0.8, 1.0], labels=[0, 1, 2, 3])
    interaction_features.append('curvature_bin')

print(f"Created {len(interaction_features)} interaction features")

# Label encode categorical features for LightGBM
from sklearn.preprocessing import LabelEncoder
le_dict = {}

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    le_dict[col] = le
    print(f"Encoded {col}: {len(le.classes_)} unique values")

# Create categorical feature interactions
print("\nCreating categorical interaction features...")
cat_interactions = []

# Road type interactions
train['road_weather'] = train['road_type'].astype(str) + '_' + train['weather'].astype(str)
test['road_weather'] = test['road_type'].astype(str) + '_' + test['weather'].astype(str)
le_road_weather = LabelEncoder()
train['road_weather'] = le_road_weather.fit_transform(train['road_weather'])
test['road_weather'] = le_road_weather.transform(test['road_weather'])
cat_interactions.append('road_weather')

train['road_lighting'] = train['road_type'].astype(str) + '_' + train['lighting'].astype(str)
test['road_lighting'] = test['road_type'].astype(str) + '_' + test['lighting'].astype(str)
le_road_lighting = LabelEncoder()
train['road_lighting'] = le_road_lighting.fit_transform(train['road_lighting'])
test['road_lighting'] = le_road_lighting.transform(test['road_lighting'])
cat_interactions.append('road_lighting')

train['weather_time'] = train['weather'].astype(str) + '_' + train['time_of_day'].astype(str)
test['weather_time'] = test['weather'].astype(str) + '_' + test['time_of_day'].astype(str)
le_weather_time = LabelEncoder()
train['weather_time'] = le_weather_time.fit_transform(train['weather_time'])
test['weather_time'] = le_weather_time.transform(test['weather_time'])
cat_interactions.append('weather_time')

train['lighting_time'] = train['lighting'].astype(str) + '_' + train['time_of_day'].astype(str)
test['lighting_time'] = test['lighting'].astype(str) + '_' + test['time_of_day'].astype(str)
le_lighting_time = LabelEncoder()
train['lighting_time'] = le_lighting_time.fit_transform(train['lighting_time'])
test['lighting_time'] = le_lighting_time.transform(test['lighting_time'])
cat_interactions.append('lighting_time')

print(f"Created {len(cat_interactions)} categorical interaction features")

# =====================
# 5. Prepare Data for Modeling
# =====================
X = train.drop('accident_risk', axis=1)
y = train['accident_risk']
X_test = test.copy()

print(f"\nFinal feature set: {X.shape[1]} features")
print(f"Training samples: {X.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# =====================
# 6. Model Training with LightGBM
# =====================
# Highly optimized hyperparameters for best performance
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.003,  # Even lower for better convergence
    'num_leaves': 511,  # Maximum complexity
    'max_depth': 15,  # Deep trees for complex patterns
    'feature_fraction': 0.6,  # More randomness to reduce overfitting
    'bagging_fraction': 0.6,
    'bagging_freq': 3,
    'min_child_samples': 5,  # Allow very granular splits
    'min_child_weight': 0.0001,
    'min_split_gain': 0.0,
    'reg_alpha': 0.5,  # Strong L1 regularization
    'reg_lambda': 0.5,  # Strong L2 regularization
    'max_bin': 511,  # More bins for better splits
    'subsample_for_bin': 200000,
    'colsample_bytree': 0.7,
    'seed': 42,
    'n_estimators': 30000,  # Many iterations with very low LR
    'verbose': -1,
    'force_col_wise': True
}

# Cross-validation - 15 folds for maximum stability
folds = KFold(n_splits=15, shuffle=True, random_state=42)
oof_preds = np.zeros(X.shape[0])
test_preds = np.zeros(X_test.shape[0])
rmse_scores = []
feature_importance = pd.DataFrame()

print("\nStarting K-Fold Cross-Validation...")
for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    print(f'\n{"="*50}')
    print(f'Fold {fold+1}/{folds.n_splits}')
    print(f'{"="*50}')
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        callbacks=[lgb.early_stopping(stopping_rounds=1000, verbose=False)]  # Maximum patience
    )
    
    # Out-of-fold predictions
    oof_preds[val_idx] = model.predict(X_val)
    
    # Validation score
    val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f'Fold {fold+1} RMSE: {rmse:.6f}')
    rmse_scores.append(rmse)
    
    # Test predictions
    test_preds += model.predict(X_test) / folds.n_splits
    
    # Feature importance
    fold_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_,
        'fold': fold + 1
    })
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

print(f'\n{"="*50}')
print(f'Mean CV RMSE: {np.mean(rmse_scores):.6f} (+/- {np.std(rmse_scores):.6f})')
print(f'{"="*50}')

# OOF RMSE
oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f'Out-of-Fold RMSE: {oof_rmse:.6f}')

# Feature importance analysis
feature_importance_agg = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
print("\nTop 15 Most Important Features:")
print(feature_importance_agg.head(15))

# Plot feature importance
plt.figure(figsize=(10,8))
feature_importance_agg.head(20).plot(kind='barh')
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances (Average across folds)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# =====================
# 7. Submission
# =====================
submission = sample_submission.copy()
submission['accident_risk'] = np.clip(test_preds, 0, 1)  # Ensure values between 0-1

# Verify submission format
print(f"\nSubmission shape: {submission.shape}")
print(f"Submission columns: {submission.columns.tolist()}")
print(f"\nSubmission statistics:")
print(submission['accident_risk'].describe())

submission.to_csv('submission.csv', index=False)
print('\n✓ Submission file created successfully!')

# =====================
# End of Notebook
# =====================

