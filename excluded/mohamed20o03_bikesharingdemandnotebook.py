import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost.callback import EarlyStopping
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def rmsle(y_true, y_pred):
    """
    Root Mean Squared Logarithmic Error
    This is the competition metric!
    """
    # Ensure no negative predictions
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))


train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain columns:", train.columns.tolist())
print("\nFirst 5 rows:")
print(train.head())


print(train.info())


def create_features(df):
    """
    Create all features from the dataset
    This pipeline works for both train and test
    """
    df = df.copy()
    
    # DateTime features
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if x in [7,8,9,17,18,19] else 0)
    
    # Weather features
    if 'temp' in df.columns and 'atemp' in df.columns:
        df['temp_feel_diff'] = df['temp'] - df['atemp']
    
    if 'temp' in df.columns and 'humidity' in df.columns:
        df['comfort_index'] = df['temp'] * (1 - df['humidity']/100)
    
    if 'temp' in df.columns and 'windspeed' in df.columns:
        df['windchill'] = df['temp'] - (df['windspeed'] * 0.1)
    
    # Ideal cycling conditions
    if all(col in df.columns for col in ['temp', 'humidity', 'windspeed', 'weather']):
        df['ideal_conditions'] = (
            (df['temp'] > 15) & 
            (df['temp'] < 30) & 
            (df['humidity'] < 70) & 
            (df['windspeed'] < 20) & 
            (df['weather'] == 1)
        ).astype(int)
    
    return df

# Apply feature engineering
train = create_features(train)
test = create_features(test)

print("\nâœ“ Feature engineering completed")
print(f"Train shape after features: {train.shape}")
print(f"Test shape after features: {test.shape}")


# Drop columns not needed for modeling
drop_cols = ['datetime', 'casual', 'registered']
drop_cols = [col for col in drop_cols if col in train.columns]

# Separate target variable
y = train['count'].copy()

# Drop unnecessary columns
X_train = train.drop(columns=drop_cols + ['count'], errors='ignore')
X_test = test.drop(columns=drop_cols, errors='ignore')

# Keep only common columns between train and test
common_cols = list(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common_cols]
X_test = X_test[common_cols]

print("\nâœ“ Data prepared for modeling")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y shape: {y.shape}")
print(f"\nFeatures: {X_train.columns.tolist()}")


# Visualize all numeric columns
num_cols = X_train.select_dtypes(include=["int64", "float64", "int32"]).columns
X_train[num_cols].hist(figsize=(15, 10), bins=30)
plt.suptitle('Feature Distributions', fontsize=16)
plt.tight_layout()
plt.show()

# Visualize target variable
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
y.hist(bins=50, edgecolor='black')
plt.title('Target Distribution (count)')
plt.xlabel('Count')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
np.log1p(y).hist(bins=50, edgecolor='black', color='orange')
plt.title('Log-Transformed Target Distribution')
plt.xlabel('Log(count + 1)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

print(f"\nTarget statistics:")
print(f"Mean: {y.mean():.2f}")
print(f"Median: {y.median():.2f}")
print(f"Std: {y.std():.2f}")
print(f"Min: {y.min():.2f}")
print(f"Max: {y.max():.2f}")


# SPLIT DATA FOR VALIDATION
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y, test_size=0.2, random_state=42
)

print(f"\nâœ“ Data split completed")
print(f"Training set: {X_tr.shape}")
print(f"Validation set: {X_val.shape}")


# XGBoost doesn't require scaling or normalization
# It handles non-normal distributions well - tree-based models split on values

print("\n" + "="*60)
print("Training XGBoost Model...")
print("="*60)

# XGBoost model optimized for RMSLE
# We'll predict log(count) and then exponentiate
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)

# Train on log-transformed target (better for RMSLE)
y_tr_log = np.log1p(y_tr)
y_val_log = np.log1p(y_val)

model.fit(
    X_tr, y_tr_log,
    eval_set=[(X_val, y_val_log)],
    verbose=50
)

print("\nâœ“ Model training completed")


# Predictions (remember to exponentiate back from log space)
y_pred_train_log = model.predict(X_tr)
y_pred_val_log = model.predict(X_val)

y_pred_train = np.expm1(y_pred_train_log)  # Inverse of log1p
y_pred_val = np.expm1(y_pred_val_log)

# Ensure no negative predictions
y_pred_train = np.maximum(y_pred_train, 0)
y_pred_val = np.maximum(y_pred_val, 0)

# Calculate RMSLE (competition metric)
train_rmsle = rmsle(y_tr, y_pred_train)
val_rmsle = rmsle(y_val, y_pred_val)

# Calculate other metrics
train_rmse = np.sqrt(mean_squared_error(y_tr, y_pred_train))
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
train_mae = mean_absolute_error(y_tr, y_pred_train)
val_mae = mean_absolute_error(y_val, y_pred_val)
train_r2 = r2_score(y_tr, y_pred_train)
val_r2 = r2_score(y_val, y_pred_val)

# Calculate accuracy metrics (percentage-based)
train_mape = np.mean(np.abs((y_tr - y_pred_train) / (y_tr + 1))) * 100  # +1 to avoid division by zero
val_mape = np.mean(np.abs((y_val - y_pred_val) / (y_val + 1))) * 100

# Accuracy as percentage (inverse of error)
train_accuracy = 100 - train_mape
val_accuracy = 100 - val_mape

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)
print(f"ðŸŽ¯ Training RMSLE:   {train_rmsle:.4f}  (Competition Metric)")
print(f"ðŸŽ¯ Validation RMSLE: {val_rmsle:.4f}  (Competition Metric)")
print(f"\nTraining RMSE:   {train_rmse:.2f} bikes")
print(f"Validation RMSE: {val_rmse:.2f} bikes")
print(f"\nTraining MAE:    {train_mae:.2f} bikes (avg error)")
print(f"Validation MAE:  {val_mae:.2f} bikes (avg error)")
print(f"\nTraining RÂ²:     {train_r2:.4f} ({train_r2*100:.1f}% variance explained)")
print(f"Validation RÂ²:   {val_r2:.4f} ({val_r2*100:.1f}% variance explained)")


# Get feature importance
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*60)
print("TOP 15 MOST IMPORTANT FEATURES")
print("="*60)
print(importance_df.head(15).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 8))
top_features = importance_df.head(15)
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training predictions
axes[0].scatter(y_tr, y_pred_train, alpha=0.3, s=10)
axes[0].plot([y_tr.min(), y_tr.max()], [y_tr.min(), y_tr.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Count')
axes[0].set_ylabel('Predicted Count')
axes[0].set_title(f'Training Set (RÂ² = {train_r2:.4f})')
axes[0].grid(alpha=0.3)

# Validation predictions
axes[1].scatter(y_val, y_pred_val, alpha=0.3, s=10, color='orange')
axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Count')
axes[1].set_ylabel('Predicted Count')
axes[1].set_title(f'Validation Set (RÂ² = {val_r2:.4f})')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


print("\n" + "="*60)
print("Making predictions on test set...")
print("="*60)

test_predictions = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'datetime': test['datetime'],
    'count': test_predictions
})

submission.to_csv('submission.csv', index=False)

print("\nâœ“ Predictions completed!")
print(f"Submission shape: {submission.shape}")
print(f"\nPrediction statistics:")
print(f"Mean: {test_predictions.mean():.2f}")
print(f"Median: {np.median(test_predictions):.2f}")
print(f"Min: {test_predictions.min():.2f}")
print(f"Max: {test_predictions.max():.2f}")
print(f"\nSubmission file saved as 'submission.csv'")


print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"âœ“ Features used: {len(X_train.columns)}")
print(f"âœ“ Training samples: {len(X_tr)}")
print(f"âœ“ Validation RMSE: {val_rmse:.2f}")
print(f"âœ“ Validation RÂ²: {val_r2:.4f}")
print(f"âœ“ Test predictions: {len(test_predictions)}")
print("\nðŸŽ¯ Ready for Kaggle submission!")
print("="*60)

