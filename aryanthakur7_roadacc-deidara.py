# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Cell 1: Imports and Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import randint, uniform
import os

# List input files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Cell 2: Load datasets
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

# Cell 3: Check missing values
print("Missing values in train:\n", df_train.isnull().sum())
print("\nMissing values in test:\n", df_test.isnull().sum())

# Cell 4: Define original feature categories
# We will redefine numerical_cols after feature engineering
target = "accident_risk"
categorical_cols = ["road_type", "weather", "lighting", "time_of_day"]
bool_cols = ["road_signs_present", "public_road", "holiday", "school_season"]
original_numerical_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

# Cell 5: Numerical summary
print("\nNumerical summary:\n", df_train[original_numerical_cols].describe())

# Cell 6: Plot numerical distributions
for col in original_numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df_train[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Cell 7: Check categorical value counts
for col in categorical_cols:
    print(f"\n{col} value counts:\n", df_train[col].value_counts())

# Cell 8: Preprocessing (Dropping ID, type conversion)
test_id = df_test['id']
df_test = df_test.drop(columns=['id'])
df_train = df_train.drop(columns=['id'])

# Convert booleans to int
for col in bool_cols:
    df_train[col] = df_train[col].astype(int)
    df_test[col] = df_test[col].astype(int)

# Ensure categorical columns are string
for col in categorical_cols:
    df_train[col] = df_train[col].astype(str)
    df_test[col] = df_test[col].astype(str)

# --- NEW CELL 9: Advanced Feature Engineering (Part 1) ---
# Create temporary one-hot encoded dataframes to build interactions
# We use drop_first=False here to ensure we capture all categories
print("Starting feature engineering...")
temp_train_dummies = pd.get_dummies(df_train, columns=categorical_cols, drop_first=False)
temp_test_dummies = pd.get_dummies(df_test, columns=categorical_cols, drop_first=False)

# Align columns to ensure test set has same features as train set
train_cols = set(temp_train_dummies.columns)
test_cols = set(temp_test_dummies.columns)

missing_in_test = list(train_cols - test_cols)
if 'accident_risk' in missing_in_test:
    missing_in_test.remove('accident_risk') # Remove target variable
for col in missing_in_test:
    temp_test_dummies[col] = 0

# Align test set columns to train set columns (excluding target)
temp_test_dummies = temp_test_dummies.reindex(columns=temp_train_dummies.drop(columns=['accident_risk']).columns, fill_value=0)

# --- NEW CELL 10: Advanced Feature Engineering (Part 2) ---
# 1. Fixed interaction: weather_rainy * lighting_night
df_train['weather_light_interaction'] = temp_train_dummies['weather_rainy'] * temp_train_dummies['lighting_night']
df_test['weather_light_interaction'] = temp_test_dummies['weather_rainy'] * temp_test_dummies['lighting_night']

# 2. New interaction: speed limit * num_lanes
df_train['speed_lanes'] = df_train['speed_limit'] * df_train['num_lanes']
df_test['speed_lanes'] = df_test['speed_limit'] * df_test['num_lanes']

# 3. New polynomial features
df_train['curvature_sq'] = df_train['curvature']**2
df_test['curvature_sq'] = df_test['curvature']**2

df_train['accidents_sq'] = df_train['num_reported_accidents']**2
df_test['accidents_sq'] = df_test['num_reported_accidents']**2
print("Feature engineering complete.")

# --- NEW CELL 11: Redefine Feature Lists ---
# We do this *after* creating the new features
new_features = ['weather_light_interaction', 'speed_lanes', 'curvature_sq', 'accidents_sq']
numerical_cols = original_numerical_cols + new_features

print(f"Updated numerical features: {numerical_cols}")

# Cell 12: Scale numerical features (Original Cell 11)
# This now scales the original AND new numerical features
scaler = StandardScaler()
df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])
df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])

# Cell 13: Define X and y (Original Cell 12)
X = df_train.drop(columns=[target])
y = df_train[target]

# Cell 14: Split data and encode categoricals (Original Cell 13)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode categorical variables
X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_val_encoded = pd.get_dummies(X_val, columns=categorical_cols, drop_first=True)

# This is the full training set encoded, for later
X_encoded_full = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# This is the full test set encoded, for the submission
X_test_encoded = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)

# Align all encoded dataframes
X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
X_encoded_full = X_encoded_full.reindex(columns=X_train_encoded.columns, fill_value=0)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)


# Cell 15: Train LR and Tuned RF (Original Cell 14, modified)
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_encoded, y_train)
y_pred_lr = lr_model.predict(X_val_encoded)

# Random Forest with better tuning
print("Tuning Random Forest...")
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
# Using n_iter=2 as in your original to keep it fast, but searching a better grid
rf_search = RandomizedSearchCV(
    estimator=rf_model, 
    param_distributions=param_grid_rf, 
    cv=3, 
    scoring='neg_mean_squared_error', 
    n_jobs=-1, 
    n_iter=2, # Keep this low for speed, increase for accuracy
    random_state=42,
    verbose=1
)
rf_search.fit(X_train_encoded, y_train)
best_rf = rf_search.best_estimator_
y_pred_rf = best_rf.predict(X_val_encoded)
print("Best Random Forest params:", rf_search.best_params_)

# --- NEW CELL 16: Tune XGBoost and LightGBM (Replaces Original Cell 15) ---
# XGBoost Tuning
print("\nTuning XGBoost...")
xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
param_dist_xgb = {
    'n_estimators': randint(100, 500),
    'learning_rate': uniform(0.01, 0.1),
    'max_depth': randint(3, 10),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
}
xgb_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist_xgb,
    n_iter=10,  # Increase for more accuracy
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
xgb_search.fit(X_train_encoded, y_train)
print("Best XGBoost params:", xgb_search.best_params_)
best_xgb = xgb_search.best_estimator_
y_pred_xgb = best_xgb.predict(X_val_encoded)

# LightGBM Tuning
print("\nTuning LightGBM...")
lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
param_dist_lgb = {
    'n_estimators': randint(100, 500),
    'learning_rate': uniform(0.01, 0.1),
    'num_leaves': randint(20, 50),
    'max_depth': [-1, 10, 20],
}
lgb_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_dist_lgb,
    n_iter=10, # Increase for more accuracy
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
lgb_search.fit(X_train_encoded, y_train)
print("Best LightGBM params:", lgb_search.best_params_)
best_lgb = lgb_search.best_estimator_
y_pred_lgb = best_lgb.predict(X_val_encoded)


# Cell 17: Model comparison and ensemble (Original Cell 16, modified)
# Create a simple weighted ensemble
# Giving more weight to the (usually) stronger XGB and LGBM models
y_pred_ensemble = (y_pred_rf * 0.2) + (y_pred_xgb * 0.4) + (y_pred_lgb * 0.4)

models = {
    'Linear Regression': y_pred_lr,
    'Tuned Random Forest': y_pred_rf,
    'Tuned XGBoost': y_pred_xgb,
    'Tuned LightGBM': y_pred_lgb,
    'Ensemble (RF+XGB+LGBM)': y_pred_ensemble
}

# Compute metrics
metrics = {}
for name, y_pred in models.items():
    metrics[name] = {
        'RMSE': np.sqrt(mean_squared_error(y_val, y_pred)),
        'MAE': mean_absolute_error(y_val, y_pred),
        'R2': r2_score(y_val, y_pred)
    }

# Display metrics
metrics_df = pd.DataFrame(metrics).T.sort_values(by='RMSE')
print("\nModel Comparison Metrics:\n", metrics_df)

# Visualizations (optional, can be skipped to save time)
print("\nGenerating visualizations...")
for name, y_pred in models.items():
    if name == 'Linear Regression': # Skip LR plot, it's not our focus
        continue
    plt.figure(figsize=(10, 4))
    # Predicted vs Actual
    plt.subplot(1, 2, 1)
    plt.scatter(y_val, y_pred, alpha=0.2)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.title(f"{name}: Predicted vs Actual")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    
    # Residual Plot
    residuals = y_val - y_pred
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.2)
    plt.axhline(0, color='r', linestyle='--')
    plt.title(f"{name}: Residuals vs Predicted")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    
    plt.tight_layout()
    plt.show()


# Cell 18: Retrain best models and create ENSEMBLE submission (Replaces 17 & 18)
print("Retraining all tuned models on the full dataset...")
# We already have X_encoded_full and X_test_encoded from Cell 14
best_rf.fit(X_encoded_full, y)
best_xgb.fit(X_encoded_full, y)
best_lgb.fit(X_encoded_full, y)

print("Generating predictions from all models...")
# Predict on the test set
preds_rf = best_rf.predict(X_test_encoded)
preds_xgb = best_xgb.predict(X_test_encoded)
preds_lgb = best_lgb.predict(X_test_encoded)

# Blend the final predictions using the same weights
final_predictions = (preds_rf * 0.2) + (preds_xgb * 0.4) + (preds_lgb * 0.4)

# Create submission
submission_df = pd.DataFrame({
    'id': test_id,
    'accident_risk': final_predictions
})
submission_df.to_csv('submission_csv', index=False)

print("\n✅ submission.csv created successfully using the ensemble model!")




