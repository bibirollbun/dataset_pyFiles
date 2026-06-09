


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s3e19/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e19/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s3e19/sample_submission.csv')

print("Dataset shapes:")
print(f"Train: {train.shape}, Test: {test.shape}")

# Convert dates
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

# =============================================================================
# SIMPLIFIED MULTI-TARGET APPROACH
# =============================================================================
print("\nCreating simplified multi-target structure...")

# Feature engineering
def create_features(df):
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['dayofyear'] = df['date'].dt.dayofyear
    return df

train_feat = create_features(train)
test_feat = create_features(test)

# Create combination identifier
train_feat['combination'] = (
    train_feat['country'] + '_' + 
    train_feat['store'] + '_' + 
    train_feat['product']
)

test_feat['combination'] = (
    test_feat['country'] + '_' + 
    test_feat['store'] + '_' + 
    test_feat['product']
)

print(f"Unique combinations in train: {train_feat['combination'].nunique()}")
print(f"Unique combinations in test: {test_feat['combination'].nunique()}")

# Get all unique combinations
all_combinations = sorted(set(train_feat['combination'].unique()) | set(test_feat['combination'].unique()))
print(f"Total unique combinations: {len(all_combinations)}")

# =============================================================================
# CREATE MULTI-TARGET TRAINING DATA
# =============================================================================
# Pivot training data to wide format
train_pivot = train_feat.pivot_table(
    index=['date', 'year', 'month', 'day', 'dayofweek', 'weekofyear', 'quarter', 'dayofyear'],
    columns='combination',
    values='num_sold',
    aggfunc='sum'
).fillna(0)

# Ensure all combinations are present (add missing columns with zeros)
for combo in all_combinations:
    if combo not in train_pivot.columns:
        train_pivot[combo] = 0

# Reorder columns to match all_combinations order
train_pivot = train_pivot[all_combinations]

print(f"Training pivot shape: {train_pivot.shape}")

# Prepare training data - keep date separately for validation
X_train_full = train_pivot.reset_index()
X_train = X_train_full[['year', 'month', 'day', 'dayofweek', 'weekofyear', 'quarter', 'dayofyear']]
y_train = train_pivot.values
train_dates = X_train_full['date']

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# =============================================================================
# CREATE TEST DATA STRUCTURE
# =============================================================================
test_dates = test_feat[['date', 'year', 'month', 'day', 'dayofweek', 'weekofyear', 'quarter', 'dayofyear']].drop_duplicates()
test_dates = test_dates.sort_values('date').reset_index(drop=True)

print(f"Unique test dates: {len(test_dates)}")

X_test = test_dates[['year', 'month', 'day', 'dayofweek', 'weekofyear', 'quarter', 'dayofyear']]
test_date_values = test_dates['date']

print(f"X_test shape: {X_test.shape}")

# =============================================================================
# MULTI-TARGET LIGHTGBM TRAINING - ONE MODEL PER TARGET
# =============================================================================
print(f"\nTraining {len(all_combinations)} LightGBM models (one per combination)...")

# LGBM parameters
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'random_state': 42
}

# Train one model for each target (combination)
models = []
feature_importances = []

for i, combo in enumerate(all_combinations):
    if i % 10 == 0:  # Progress update every 10 models
        print(f"Training model {i+1}/{len(all_combinations)} for {combo}")
    
    # Create dataset for this specific target
    train_data = lgb.Dataset(X_train, label=y_train[:, i])
    
    # Train model
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
        ]
    )
    
    models.append(model)
    feature_importances.append(model.feature_importance(importance_type='gain'))

print("✓ All LightGBM models training completed!")

# =============================================================================
# VALIDATION
# =============================================================================
print("\nPerforming validation...")

# Validation split
unique_dates = train_dates.unique()
split_idx = int(len(unique_dates) * 0.8)
split_date = unique_dates[split_idx]

print(f"Total training dates: {len(unique_dates)}")
print(f"Split date: {split_date}")

train_mask = train_dates < split_date
val_mask = train_dates >= split_date

X_tr = X_train[train_mask]
y_tr = y_train[train_mask]
X_val = X_train[val_mask]
y_val = y_train[val_mask]

print(f"Train dates: {X_tr.shape[0]}, Validation dates: {X_val.shape[0]}")

# Validation predictions
y_val_pred = np.zeros_like(y_val)
for i, model in enumerate(models):
    y_val_pred[:, i] = model.predict(X_val)

val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = np.sqrt(val_mse)
print(f"Validation MSE: {val_mse:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")

# =============================================================================
# TEST PREDICTIONS
# =============================================================================
print("\nMaking test predictions...")

# Predict for all test dates and all combinations
test_predictions = np.zeros((len(X_test), len(all_combinations)))
for i, model in enumerate(models):
    test_predictions[:, i] = model.predict(X_test)

print(f"Test predictions shape: {test_predictions.shape}")

# =============================================================================
# RECONSTRUCT SUBMISSION FORMAT
# =============================================================================
print("\nReconstructing submission format...")

# Create mapping from combination to column index
combo_to_idx = {combo: idx for idx, combo in enumerate(all_combinations)}

# Create a DataFrame to help with reconstruction
test_reference = test_feat[['id', 'date', 'combination']].copy()

# Merge with test_dates to get the row index for each test date
test_dates_with_idx = test_dates[['date']].reset_index().rename(columns={'index': 'date_row_idx'})
test_reference = test_reference.merge(test_dates_with_idx, on='date')

# Map predictions to original test format
final_predictions = []
for _, row in test_reference.iterrows():
    combo = row['combination']
    date_idx = row['date_row_idx']
    
    if combo in combo_to_idx:
        pred = test_predictions[date_idx, combo_to_idx[combo]]
        final_predictions.append(max(0, pred))
    else:
        final_predictions.append(np.median(y_train))

# Create submission
submission = sample_submission.copy()
submission['num_sold'] = final_predictions

print(f"Submission shape: {submission.shape}")
print(f"Prediction stats: min={submission['num_sold'].min():.2f}, "
      f"max={submission['num_sold'].max():.2f}, "
      f"mean={submission['num_sold'].mean():.2f}")

# =============================================================================
# ANALYSIS AND VISUALIZATION
# =============================================================================
plt.figure(figsize=(15, 10))

# Average feature importance across all models
plt.subplot(2, 3, 1)
avg_feature_importance = np.mean(feature_importances, axis=0)
feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': avg_feature_importance
}).sort_values('importance', ascending=True)
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
plt.title('Average LGBM Feature Importance')
plt.xlabel('Importance (gain)')

# Validation predictions vs actual (first target)
plt.subplot(2, 3, 2)
plt.scatter(y_val[:, 0], y_val_pred[:, 0], alpha=0.6)
max_val = max(y_val[:, 0].max(), y_val_pred[:, 0].max())
plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.8)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Validation: First Target')

# Prediction distribution
plt.subplot(2, 3, 3)
plt.hist(submission['num_sold'], bins=50, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Predicted num_sold')
plt.ylabel('Frequency')
plt.title('Test Predictions Distribution')

# Top combinations by average prediction
plt.subplot(2, 3, 4)
avg_predictions = pd.Series(test_predictions.mean(axis=0), index=all_combinations)
top_combinations = avg_predictions.nlargest(10)
plt.barh(range(len(top_combinations)), top_combinations.values)
plt.yticks(range(len(top_combinations)), [str(combo)[:20] + '...' for combo in top_combinations.index])
plt.title('Top 10 Combinations by Avg Prediction')

# Temporal pattern example (first combination)
plt.subplot(2, 3, 5)
example_combo_idx = 0
plt.plot(test_date_values.iloc[:100], test_predictions[:100, example_combo_idx])
plt.title(f'Sales Pattern for {all_combinations[example_combo_idx][:20]}...')
plt.xlabel('Date')
plt.ylabel('Predicted Sales')
plt.xticks(rotation=45)

# Model performance by month
plt.subplot(2, 3, 6)
val_results = pd.DataFrame({
    'actual_mean': y_val.mean(axis=1),
    'predicted_mean': y_val_pred.mean(axis=1),
    'month': train_dates[val_mask].dt.month
})
monthly_avg = val_results.groupby('month').mean()
plt.plot(monthly_avg.index, monthly_avg['actual_mean'], label='Actual', marker='o')
plt.plot(monthly_avg.index, monthly_avg['predicted_mean'], label='Predicted', marker='s')
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.title('Monthly Sales Pattern')
plt.legend()
plt.xticks(range(1, 13))

plt.tight_layout()
plt.show()

# =============================================================================
# SAVE SUBMISSION
# =============================================================================
submission_file = 'multi_target_lgbm_individual_submission.csv'
submission.to_csv(submission_file, index=False)
print(f"\n✓ Submission saved as: {submission_file}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("MULTI-TARGET LIGHTGBM (INDIVIDUAL MODELS) SUMMARY")
print("="*60)
print(f"✓ Unique combinations: {len(all_combinations)}")
print(f"✓ Models trained: {len(models)}")
print(f"✓ Training dates: {X_train.shape[0]}")
print(f"✓ Test dates: {X_test.shape[0]}")
print(f"✓ Validation RMSE: {val_rmse:.4f}")
print(f"✓ Most important feature: {feature_importance_df['feature'].iloc[-1]}")
print(f"✓ Predictions range: {submission['num_sold'].min():.2f} - {submission['num_sold'].max():.2f}")
print("✓ Individual LGBM models training completed")

