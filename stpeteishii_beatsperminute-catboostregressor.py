


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
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
print("\nTrain columns:", train.columns.tolist())
print("\nTest columns:", test.columns.tolist())

# Check for missing values
print("\nMissing values in train:")
print(train.isnull().sum())
print("\nMissing values in test:")
print(test.isnull().sum())

# Prepare features and target
X = train.drop(['id', target], axis=1)
y = train[target]

# Initialize K-Fold cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize arrays for OOF predictions and scores
oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(test))
fold_scores = []
models = []

print(f"Starting {n_splits}-Fold Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nFold {fold + 1}/{n_splits}")
    
    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Initialize CatBoost model
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_state=42,
        verbose=100,
        loss_function='RMSE',
        eval_metric='RMSE',
        early_stopping_rounds=50
    )
    
    # Train the model
    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        plot=False
    )
    
    # Store the trained model
    models.append(cat_model)
    
    # Make predictions on validation set
    y_pred_val = cat_model.predict(X_val)
    
    # Store OOF predictions
    oof_predictions[val_idx] = y_pred_val
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_val, y_pred_val)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)
    
    fold_scores.append({
        'fold': fold + 1, 
        'mse': mse, 
        'rmse': rmse, 
        'mae': mae, 
        'r2': r2,
        'best_iteration': cat_model.get_best_iteration()
    })
    
    print(f"Fold {fold + 1} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    print(f"Best iteration: {cat_model.get_best_iteration()}")
    
    # Make predictions on test data for this fold
    X_test = test.drop('id', axis=1)
    test_predictions += cat_model.predict(X_test) / n_splits

# Convert fold scores to DataFrame
fold_results = pd.DataFrame(fold_scores)

# Calculate overall OOF metrics
oof_mse = mean_squared_error(y, oof_predictions)
oof_rmse = np.sqrt(oof_mse)
oof_mae = mean_absolute_error(y, oof_predictions)
oof_r2 = r2_score(y, oof_predictions)

print("\n" + "="*60)
print("CROSS-VALIDATION RESULTS (OOF PREDICTIONS)")
print("="*60)
print(f"Overall OOF RMSE: {oof_rmse:.4f}")
print(f"Overall OOF MAE: {oof_mae:.4f}")
print(f"Overall OOF R²: {oof_r2:.4f}")
print("\nFold-wise results:")
print(fold_results)
print("="*60)

# Create OOF predictions dataframe
oof_df = pd.DataFrame({
    'id': train['id'],
    'actual': y,
    'predicted': oof_predictions,
    'residual': y - oof_predictions
})

np.save('oof.npy',oof_predictions)

print("\nOOF Predictions sample:")
print(oof_df.head(10))

# Plot actual vs predicted values for OOF
plt.figure(figsize=(10, 6))
plt.scatter(y, oof_predictions, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('OOF Predicted Values')
plt.title('Actual vs OOF Predicted Values')
plt.show()

# Residual plot for OOF predictions
plt.figure(figsize=(10, 6))
plt.scatter(oof_predictions, oof_df['residual'], alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('OOF Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot (OOF Predictions)')
plt.show()

# =============================================================================
# CREATE FINAL SUBMISSION WITH AVERAGED PREDICTIONS
# =============================================================================
final_submit = pd.DataFrame({
    'id': test['id'],
    target: test_predictions
})

print("\nTest predictions sample:")
print(final_submit.head(10))

# Save submission file
final_submit.to_csv('submission.csv', index=False)
print("\nFinal submission file 'submission.csv' created successfully!")

# =============================================================================
# TRAIN FINAL MODEL ON FULL DATA (OPTIONAL)
# =============================================================================
print("\nTraining final model on full dataset...")

# Get average best iteration from cross-validation
avg_best_iteration = int(fold_results['best_iteration'].mean())

final_model = CatBoostRegressor(
    iterations=avg_best_iteration,
    learning_rate=0.05,
    depth=6,
    random_state=42,
    verbose=0,
    loss_function='RMSE'
)

final_model.fit(X, y)

# Make final predictions on test data
final_test_predictions = final_model.predict(X_test)

# Create alternative submission with full model
final_submit_full = pd.DataFrame({
    'id': test['id'],
    target: final_test_predictions
})

final_submit_full.to_csv('submission_full.csv', index=False)
print("Alternative submission file 'submission_full.csv' created successfully!")

# Feature importance from final model
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.get_feature_importance()
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('CatBoost Feature Importance')
plt.tight_layout()
plt.show()

# Plot learning curves from the last fold model
plt.figure(figsize=(10, 6))
plt.plot(models[-1].evals_result_['learn']['RMSE'], label='Training RMSE')
plt.plot(models[-1].evals_result_['validation']['RMSE'], label='Validation RMSE')
plt.title('Learning Curves (Last Fold)')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.show()

# Additional analysis: OOF performance by fold
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(fold_results['fold'], fold_results['rmse'])
plt.title('RMSE by Fold')
plt.xlabel('Fold')
plt.ylabel('RMSE')

plt.subplot(1, 2, 2)
plt.bar(fold_results['fold'], fold_results['r2'])
plt.title('R² Score by Fold')
plt.xlabel('Fold')
plt.ylabel('R²')

plt.tight_layout()
plt.show()





