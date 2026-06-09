import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler

import warnings

# Suppress the specific matplotlib warning
warnings.filterwarnings('ignore', category=RuntimeWarning, module='matplotlib.colors')


'''
# Optimal parameters for LightGBM from optuna
best_params = {
    'learning_rate': 0.02493303813836915,
    'num_leaves': 34,
    'max_depth': 13,
    'reg_alpha': 0.07319459385373907,
    'reg_lambda': 0.08870554420170793,
    'subsample': 0.5514983899131313,
    'random_state': 42,
    'verbose': -1  # Suppress warnings
}
'''


# Load csv
X = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')

# Print data
X



# Get summary
X.info()
X.describe()


# Plot the histograms to evaluate the distributions

# Select only numeric columns
numeric_cols = X.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col.lower() != 'id']

# Create a grid of subplots
n_cols = 3  # number of columns in the grid
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes

for i, col in enumerate(numeric_cols):
    X[col].hist(bins=30, ax=axes[i], edgecolor='black', alpha=0.7)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# Remove empty subplots if necessary
for j in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# Pearson Correlation

# Create subset with only numeric columns
X_numeric = X[numeric_cols]

# Calculate Pearson correlation matrix
correlation_matrix = X_numeric.corr(method='pearson')

# Create correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, 
            annot=True,           # Show correlation values
            cmap='coolwarm',      # Color scheme
            center=0,             # Center colormap at 0
            square=True,          # Square cells
            fmt='.2f',            # Format numbers to 2 decimals
            cbar_kws={'shrink': 0.8})

plt.title('Pearson Correlation Heatmap', fontsize=16, pad=20)
plt.tight_layout()
plt.show()


# Spearman Correlation

# Create subset with only numeric columns
X_numeric = X[numeric_cols]

# Calculate Spearman correlation matrix
correlation_matrix = X_numeric.corr(method='spearman')

# Create correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, 
            annot=True,           # Show correlation values
            cmap='coolwarm',      # Color scheme
            center=0,             # Center colormap at 0
            square=True,          # Square cells
            fmt='.2f',            # Format numbers to 2 decimals
            cbar_kws={'shrink': 0.8})

plt.title('Spearman Correlation Heatmap', fontsize=16, pad=20)
plt.tight_layout()
plt.show()


# Select only numeric columns (excluding 'id' and target 'BeatsPerMinute')
numeric_cols = X.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col.lower() != 'id' and col != 'BeatsPerMinute']

# Detect outliers
print(f"Original dataset shape: {X.shape}")
print(f"Analyzing {len(numeric_cols)} numeric variables for outliers (excluding target 'BeatsPerMinute')")
print("="*60)

# Create a mask to identify outliers using IQR method
outlier_mask = pd.Series([False] * len(X), index=X.index)

for col in numeric_cols:
    # Calculate Q1, Q3 and IQR
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers for this column
    column_outliers = (X[col] < lower_bound) | (X[col] > upper_bound)
    outlier_count = column_outliers.sum()
    
    print(f"{col}:")
    print(f"  Q1: {Q1:.3f}, Q3: {Q3:.3f}, IQR: {IQR:.3f}")
    print(f"  Bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")
    print(f"  Outliers found: {outlier_count} ({(outlier_count/len(X)*100):.1f}%)")
    
    # Update the global outlier mask
    outlier_mask = outlier_mask | column_outliers

# Count total rows with at least one outlier
total_outlier_rows = outlier_mask.sum()
print("="*60)
print(f"Total rows with at least one outlier: {total_outlier_rows} ({(total_outlier_rows/len(X)*100):.1f}%)")

# Remove outlier rows
X_clean = X[~outlier_mask].copy()

print(f"Dataset shape after outlier removal: {X_clean.shape}")
print(f"Rows removed: {len(X) - len(X_clean)}")
print(f"Rows retained: {len(X_clean)} ({(len(X_clean)/len(X)*100):.1f}%)")

# Update X with the clean dataset
X = X_clean.copy()
print(f"\nFinal dataset shape: {X.shape}")


# Create a new feature Power=Energy+AudioLoudness that are Positively correlated
X['Power'] = X['Energy'] + X['AudioLoudness']
X = X.drop(['Energy', 'AudioLoudness'], axis=1)


# Prepare data
X_features = X.drop(columns=['BeatsPerMinute', 'id'])
y = X['BeatsPerMinute']

# Split train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_features, y, test_size=0.2, random_state=42
)

# Objective function for Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
    }
    
    model = xgb.XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
              early_stopping_rounds=50, verbose=False)
    
    y_pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, y_pred))

# Optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=300)

# Final model with best parameters
best_model = xgb.XGBRegressor(**study.best_params, random_state=42)
best_model.fit(X_train, y_train)

# Evaluation
y_pred = best_model.predict(X_val)
print(f"Best RMSE: {np.sqrt(mean_squared_error(y_val, y_pred)):.4f}")
print(f"R²: {r2_score(y_val, y_pred):.4f}")
print("\nBest parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")


# Prepare dat
X_features = X.drop(columns=['BeatsPerMinute', 'id'])
y = X['BeatsPerMinute']

# Use the best parameters found from optimization
final_model = xgb.XGBRegressor(**study.best_params, random_state=42)

# Define RMSE scorer for cross-validation
rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), 
                         greater_is_better=False)

# Perform 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(final_model, X_features, y, cv=kfold, scoring=rmse_scorer)

# Convert negative scores to positive (sklearn returns negative for error metrics)
cv_scores = -cv_scores

# Print results
print("5-Fold Cross-Validation Results:")
print("-" * 40)
for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: RMSE = {score:.4f}")
print("-" * 40)
print(f"Mean RMSE: {cv_scores.mean():.4f}")
print(f"Std RMSE: {cv_scores.std():.4f}")
print(f"95% CI: [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, {cv_scores.mean() + 1.96*cv_scores.std():.4f}]")

# Train final model on full dataset for deployment
final_model.fit(X_features, y)
print(f"\n Final model trained on full dataset ({len(X_features)} samples)")


# Load csv
submission = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

# Feature Engineering
submission['Power'] = submission['Energy'] + submission['AudioLoudness']
submission = submission.drop(['Energy', 'AudioLoudness'], axis=1)

# Store id for final submission
submission_ids = submission['id'].copy()

# Prepare features (remove id column, same as training)
submission_features = submission.drop(columns=['id'])

# Generate predictions using the trained model
predictions = final_model.predict(submission_features)


# Create final submission dataframe
final_submission = pd.DataFrame({
    'id': submission_ids,
    'BeatsPerMinute': predictions
})

# Display first few predictions
print("First 10 predictions:")
print(final_submission.head(10))

# Save submission file
final_submission.to_csv('submission.csv', index=False)
print(f"\n Submission file saved as 'submission.csv' with {len(final_submission)} predictions")

