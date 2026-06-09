import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


root_path = '/kaggle/input/agriyield-2025/'
train_df = pd.read_csv(f'{root_path}train.csv')
test_df = pd.read_csv(f'{root_path}test.csv')
sample_submission_df = pd.read_csv(f'{root_path}sample_submission.csv')


print("Train Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)


print("\nTraining Data Info:")
train_df.info()

print("\nTest Data Info:")
test_df.info()


plt.figure(figsize=(12, 5))
sns.histplot(train_df['yield'], kde=True, bins=50)
plt.title('Distribution of Maize Yield (kg/ha)', fontsize=16)
plt.xlabel('Yield (kg/ha)')
plt.ylabel('Frequency')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()


features = [col for col in train_df.columns if col not in ['field_id', 'yield']]

train_df[features].hist(bins=30, figsize=(15, 10), layout=(3, 3))
plt.suptitle('Distribution of Features in Training Data', y=1.02, fontsize=18)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
correlation_matrix = train_df[features + ['yield']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix', fontsize=16)
plt.show()


# Define features (X) and target (y)
features = [col for col in train_df.columns if col not in ['field_id', 'yield']]
X = train_df[features]
y = train_df['yield']
X_test = test_df[features]

# --- Model Parameters ---
# These are good starting parameters. You can tune them later using techniques like Optuna or GridSearchCV.
LGBM_PARAMS = {
    'objective': 'regression_l1', # MAE is less sensitive to outliers than MSE (regression_l2)
    'metric': 'rmse',
    'n_estimators': 2000,         # High number, will be stopped by early stopping
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}

# --- Cross-Validation Setup ---
N_SPLITS = 10
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# --- Training Loop ---
oof_predictions = np.zeros(X.shape[0])
test_predictions = np.zeros(X_test.shape[0])
models = []
oof_rmse_scores = []

print("Starting training with K-Fold Cross-Validation...")

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"===== FOLD {fold+1} =====")
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_val, y_val = X.iloc[val_index], y.iloc[val_index]

    # Initialize and train the model
    model = lgb.LGBMRegressor(**LGBM_PARAMS)

    # Use early stopping
    callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=callbacks)

    # Store predictions
    val_preds = model.predict(X_val)
    oof_predictions[val_index] = val_preds
    test_predictions += model.predict(X_test) / N_SPLITS
    models.append(model)

    # Calculate and store RMSE for the fold
    fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    oof_rmse_scores.append(fold_rmse)
    print(f"Fold {fold+1} RMSE: {fold_rmse:.5f}")

# --- Overall CV Score ---
overall_oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
print("\n---------------------------------")
print(f"Overall OOF RMSE: {overall_oof_rmse:.5f}")
print("---------------------------------")


def plot_feature_importance(models, features):
    """Plots the average feature importance across all folds."""
    feature_importances = pd.DataFrame()
    for i, model in enumerate(models):
        fold_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_,
            'fold': i + 1
        })
        feature_importances = pd.concat([feature_importances, fold_importance], axis=0)

    # Calculate mean importance
    mean_importance = feature_importances.groupby('feature')['importance'].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=mean_importance, y=mean_importance.index)
    plt.title('Average Feature Importance Across Folds', fontsize=16)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

plot_feature_importance(models, features)


# Create the submission DataFrame
submission_df = pd.DataFrame({'field_id': test_df['field_id'], 'yield': test_predictions})

# Ensure no negative predictions (yield cannot be negative)
submission_df['yield'] = submission_df['yield'].clip(0)

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file created successfully!")
print("Submission file head:")
display(submission_df.head())

