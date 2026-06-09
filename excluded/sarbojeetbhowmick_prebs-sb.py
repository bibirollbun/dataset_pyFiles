import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

target = 'BeatsPerMinute'

# Display basic information about the datasets
print("Train dataset shape:", train.shape)
print("Test dataset shape:", test.shape)

# Check for missing values
print("\nMissing values in train:")
print(train.isnull().sum())
print("\nMissing values in test:")
print(test.isnull().sum())

# Prepare features and target
X = train.drop(['id', target], axis=1)
y = train[target]
X_test = test.drop('id', axis=1)

# Initialize K-Fold
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Arrays to store OOF predictions and test predictions
oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))
models = []
best_iterations = []
fold_metrics = []

# K-Fold Cross Validation
print(f"\nStarting {n_splits}-Fold Cross Validation...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1}/{n_splits}")
    print(f"{'='*50}")

    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")

    # Initialize LightGBM model
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1
    )

    # Create callbacks
    early_stopping_callback = lgb.early_stopping(
        stopping_rounds=50,
        verbose=False
    )

    log_evaluation_callback = lgb.log_evaluation(period=50)

    # Train the model
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='l2',
        callbacks=[early_stopping_callback, log_evaluation_callback]
    )

    # Make predictions on validation set
    y_pred_val = lgb_model.predict(X_val)
    oof_predictions[val_idx] = y_pred_val

    # Make test predictions
    test_preds = lgb_model.predict(X_test)
    test_predictions += test_preds / n_splits

    # Store model and best iteration
    models.append(lgb_model)
    best_iterations.append(lgb_model.best_iteration_)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_val, y_pred_val)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)

    fold_metrics.append({
        'fold': fold + 1,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'best_iteration': lgb_model.best_iteration_
    })

    print(f"\nFold {fold + 1} Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Best iteration: {lgb_model.best_iteration_}")

# Calculate overall OOF metrics
oof_mse = mean_squared_error(y, oof_predictions)
oof_rmse = np.sqrt(oof_mse)
oof_mae = mean_absolute_error(y, oof_predictions)
oof_r2 = r2_score(y, oof_predictions)

print("\n" + "="*60)
print("OVERALL OUT-OF-FOLD (OOF) EVALUATION METRICS")
print("="*60)
print(f"OOF Mean Squared Error (MSE): {oof_mse:.4f}")
print(f"OOF Root Mean Squared Error (RMSE): {oof_rmse:.4f}")
print(f"OOF Mean Absolute Error (MAE): {oof_mae:.4f}")
print(f"OOF R² Score: {oof_r2:.4f}")
print("="*60)

# Display fold-wise metrics
print("\nFold-wise Metrics Summary:")
metrics_df = pd.DataFrame(fold_metrics)
print(metrics_df.round(4))

# Save OOF predictions
np.save('oof_predictions.npy', oof_predictions)
print("\nOOF predictions saved to 'oof_predictions.npy'")

# Feature importance (average across folds)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': np.mean([model.feature_importances_ for model in models], axis=0)
}).sort_values('importance', ascending=False)

print("\nAverage Feature Importance across folds:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Average LightGBM Feature Importance (Across Folds)')
plt.tight_layout()
plt.show()

# Actual vs Predicted values plot (OOF)
plt.figure(figsize=(10, 6))
plt.scatter(y, oof_predictions, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (OOF)')
plt.show()

# Residual plot (OOF)
residuals = y - oof_predictions
plt.figure(figsize=(10, 6))
plt.scatter(oof_predictions, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot (OOF)')
plt.show()

# =============================================================================
# PREDICT ON TEST DATA AND CREATE SUBMISSION
# =============================================================================

# Make predictions on test data (already averaged across folds)
print("\nMaking predictions on test data...")

# Create submission dataframe
submit = pd.DataFrame({
    'id': test['id'],
    target: test_predictions
})

# Display sample of predictions
print("\nSample of test predictions:")
print(submit.head(10))

# Save submission file
submit.to_csv('submission.csv', index=False)
print("\nSubmission file 'submission.csv' created successfully!")

# =============================================================================
# OPTIONAL: TRAIN FINAL MODEL ON ALL TRAINING DATA
# =============================================================================
print("\nTraining final model on all training data...")

# Get average best iteration from all folds
avg_best_iteration = int(np.mean(best_iterations))
print(f"Average best iteration across folds: {avg_best_iteration}")

# Train final model on all data
final_model = lgb.LGBMRegressor(
    n_estimators=avg_best_iteration,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X, y)

# Make final predictions on test data
final_test_predictions = final_model.predict(X_test)

# Create final submission
final_submit = pd.DataFrame({
    'id': test['id'],
    target: final_test_predictions
})

final_submit.to_csv('final_submission.csv', index=False)
print("Final submission file 'final_submission.csv' created successfully!")

# Final model feature importance
final_feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=final_feature_importance.head(15))
plt.title('Final Model Feature Importance')
plt.tight_layout()
plt.show()

# Learning curves for the last fold
eval_results = models[-1].evals_result_
plt.figure(figsize=(10, 6))
plt.plot(eval_results['valid_0']['l2'], label='Validation MSE')
plt.title('Learning Curve (Last Fold)')
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

print("\nK-Fold Cross Validation completed successfully!")

