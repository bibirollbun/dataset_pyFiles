import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

# Show the unmodified tables
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")


# Plot target variable distribution
plt.figure(figsize=(10,6))
sns.histplot(train['BeatsPerMinute'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Beats Per Minute (BPM)')
plt.xlabel('Beats Per Minute')
plt.ylabel('Count')
plt.show()

# Correlation heatmap to see which features correlate with BPM
plt.figure(figsize=(12,8))
corr = train.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')
plt.show()

# Scatter plots for a few interesting features
features_to_plot = ['RhythmScore', 'AudioLoudness', 'TrackDurationMs', 'Energy']
for feat in features_to_plot:
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=train[feat], y=train['BeatsPerMinute'], alpha=0.5)
    plt.title(f'{feat} vs BeatsPerMinute')
    plt.show()


# Prepare features and target
X = train.drop(['id','BeatsPerMinute'], axis=1)
y = train['BeatsPerMinute']
X_test = test.drop(['id'], axis=1)

# Feature engineering
X['Energy_Rhythm'] = X['Energy'] * X['RhythmScore']
X_test['Energy_Rhythm'] = X_test['Energy'] * X_test['RhythmScore']

# Fill any missing values
X = X.fillna(-1)
X_test = X_test.fillna(-1)

print(f"Processed training features shape: {X.shape}")
print(f"Processed test features shape: {X_test.shape}")


# 5-Fold Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize test prediction arrays
lgb_test_pred = np.zeros(len(X_test))
xgb_test_pred = np.zeros(len(X_test))
cat_test_pred = np.zeros(len(X_test))

# ------------------------------
# LightGBM
# ------------------------------
lgb_params = {
    'objective':'regression',
    'metric':'rmse',
    'boosting_type':'gbdt',
    'num_leaves':31,
    'learning_rate':0.05,
    'feature_fraction':0.8,
    'bagging_fraction':0.8,
    'bagging_freq':5,
    'random_state':42,
    'n_estimators':1000
}

print("Training LightGBM...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    
    # Use callbacks for early stopping to avoid TypeError
    lgb_model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
    )
    
    lgb_test_pred += lgb_model.predict(X_test) / 5

# ------------------------------
# XGBoost
# ------------------------------
xgb_params = {
    'objective':'reg:squarederror',
    'max_depth':6,
    'learning_rate':0.05,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'n_estimators':1000,
    'random_state':42
}

print("Training XGBoost...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        early_stopping_rounds=100, 
        verbose=100
    )
    
    xgb_test_pred += xgb_model.predict(X_test) / 5

# ------------------------------
# CatBoost
# ------------------------------
cat_params = {
    'iterations':1000,
    'learning_rate':0.05,
    'depth':6,
    'random_seed':42,
    'loss_function':'RMSE',
    'verbose':100
}

print("Training CatBoost...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(
        X_train, 
        y_train, 
        eval_set=(X_val, y_val)
    )
    
    cat_test_pred += cat_model.predict(X_test) / 5


# Weighted ensemble
final_pred = 0.4*lgb_test_pred + 0.35*xgb_test_pred + 0.25*cat_test_pred

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'BeatsPerMinute': final_pred
})
submission.to_csv('submission.csv', index=False)

print("âœ… Submission file created successfully!")
print(f"ðŸ“Š Prediction range: {final_pred.min():.2f} - {final_pred.max():.2f}")
print(f"ðŸ“ˆ Mean prediction: {final_pred.mean():.2f}")

