# Core data handling libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning tools
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
# Set default aesthetics for plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

# Inspect dimensions and structure
print(f"âœ… Training set shape: {train.shape}")
print(f"âœ… Test set shape: {test.shape}")
display(train.head())


# Select key market columns and sample anonymized columns
market_columns = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
anonymized_sample = [f'X{i}' for i in range(1, 6)]  # Sample only first 5 anonymized features

# Summary statistics
summary_stats = train[market_columns + anonymized_sample + ['label']].describe().T

# Check for missing values
missing_values = train.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

# Display outputs
display(summary_stats)
display(missing_values if not missing_values.empty else "No missing values found.")


# Plot the distribution of the target label
plt.figure(figsize=(10, 5))
sns.histplot(train['label'], bins=100, kde=True, color='orange')
plt.title("Distribution of Target Variable: label", fontsize=14)
plt.xlabel("Label Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Correlation with label (only a few select features to keep it readable)
selected_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume'] + [f'X{i}' for i in range(1, 6)]
correlation_with_label = train[selected_features + ['label']].corr()['label'].sort_values(ascending=False)

# Display top 10 most positively and negatively correlated
display(correlation_with_label.head(10))
display(correlation_with_label.tail(10))


# Copy the dataframe to preserve original
train_fe = train.copy()

# 1. Bid-Ask Spread
train_fe['bid_ask_spread'] = train_fe['ask_qty'] - train_fe['bid_qty']

# 2. Buy-Sell Volume Imbalance
train_fe['volume_imbalance'] = (train_fe['buy_qty'] - train_fe['sell_qty']) / (train_fe['buy_qty'] + train_fe['sell_qty'] + 1e-6)

# 3. Normalized volume
train_fe['log_volume'] = np.log1p(train_fe['volume'])

# 4. Ratio-based features
train_fe['buy_to_volume_ratio'] = train_fe['buy_qty'] / (train_fe['volume'] + 1e-6)
train_fe['sell_to_volume_ratio'] = train_fe['sell_qty'] / (train_fe['volume'] + 1e-6)

# Display new feature distributions
new_features = ['bid_ask_spread', 'volume_imbalance', 'log_volume', 'buy_to_volume_ratio', 'sell_to_volume_ratio']
display(train_fe[new_features].describe().T)


# Step 1: Feature selection (same)
feature_cols = [
    'bid_ask_spread', 'volume_imbalance', 'log_volume',
    'buy_to_volume_ratio', 'sell_to_volume_ratio',
    'X1', 'X2', 'X3', 'X4', 'X5'
]

# Step 2: Time-based split
train_cutoff = int(len(train_fe) * 0.8)
X_train, X_val = train_fe[feature_cols].iloc[:train_cutoff], train_fe[feature_cols].iloc[train_cutoff:]
y_train, y_val = train_fe['label'].iloc[:train_cutoff], train_fe['label'].iloc[train_cutoff:]

# Step 3: Train model with callbacks
model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[
        early_stopping(stopping_rounds=20),
        log_evaluation(period=50)  # Logs every 50 iterations
    ]
)

# Step 4: Evaluate
y_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f"ğŸ“‰ Validation RMSE: {rmse:.5f}")


# Step 1: Load sample submission
sample_submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')

# Create engineered features in the test set
test_fe = test.copy()

test_fe['bid_ask_spread'] = test_fe['ask_qty'] - test_fe['bid_qty']
test_fe['volume_imbalance'] = (test_fe['buy_qty'] - test_fe['sell_qty']) / (test_fe['buy_qty'] + test_fe['sell_qty'] + 1e-6)
test_fe['log_volume'] = np.log1p(test_fe['volume'])
test_fe['buy_to_volume_ratio'] = test_fe['buy_qty'] / (test_fe['volume'] + 1e-6)
test_fe['sell_to_volume_ratio'] = test_fe['sell_qty'] / (test_fe['volume'] + 1e-6)

# Step 2: Apply same features to test set
X_test = test_fe[feature_cols]

# Step 3: Predict
test_preds = model.predict(X_test)

# Step 4: Fill submission dataframe
sample_submission['prediction'] = test_preds

# Step 5: Save to CSV
sample_submission.to_csv('submission.csv', index=False)
print("submission.csv file created successfully.")





sample_submission.head()


# Define new feature set
engineered_features = [
    'bid_ask_spread', 'volume_imbalance', 'log_volume',
    'buy_to_volume_ratio', 'sell_to_volume_ratio'
]

anonymized_features = [f'X{i}' for i in range(1, 101)]

feature_cols = engineered_features + anonymized_features

# Split again
X_train, X_val = train_fe[feature_cols].iloc[:train_cutoff], train_fe[feature_cols].iloc[train_cutoff:]
y_train, y_val = train_fe['label'].iloc[:train_cutoff], train_fe['label'].iloc[train_cutoff:]

# Retrain model
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[
        early_stopping(stopping_rounds=20),
        log_evaluation(period=50)
    ]
)

# Evaluate
y_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f"ğŸ“‰ Expanded Feature Set - Validation RMSE: {rmse:.5f}")


import optuna
from lightgbm import LGBMRegressor

def objective(trial):
    params = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1
    }

    model = LGBMRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[
            early_stopping(stopping_rounds=20),
            log_evaluation(period=0)
        ]
    )

    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print(f"ğŸ”� Best RMSE: {study.best_value:.5f}")
print("âœ… Best Params:")
print(study.best_params)


# Use best parameters from Optuna
best_params = {
    'n_estimators': 1000,
    'learning_rate': 0.06762745224761675,
    'num_leaves': 97,
    'max_depth': 4,
    'min_child_samples': 98,
    'subsample': 0.5185830564077009,
    'colsample_bytree': 0.6456671540143484,
    'reg_alpha': 0.12148076904003635,
    'reg_lambda': 0.19786770408218737,
    'random_state': 42,
    'n_jobs': -1
}

# Train on full dataset
X_full = train_fe[feature_cols]
y_full = train_fe['label']

final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(X_full, y_full)

# Predict on test
X_test = test_fe[feature_cols]
final_preds = final_model.predict(X_test)

# Load submission template
sample_submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
sample_submission['prediction'] = final_preds

# Save submission file
sample_submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Final submission file saved to /kaggle/working/submission.csv")




