import pandas as pd

train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

print(train.info())
print('='*70)
print(test.info())
print('='*70)
print(sample.info())
print('='*70)


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set_style('whitegrid')

# Plot the distribution of the target variable
plt.figure(figsize=(12, 6))
sns.histplot(train['BeatsPerMinute'], kde=True, bins=50, color='blue')
plt.title('Distribution of BeatsPerMinute', fontsize=16)
plt.xlabel('Beats Per Minute', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()


plt.figure(figsize=(12, 10))
# Calculate the correlation matrix
corr_matrix = train.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Features', fontsize=16)
plt.show()


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Define features (X) and target (y)
features = [col for col in train.columns if col not in ['id', 'BeatsPerMinute']]
X_train = train[features]
y_train = train['BeatsPerMinute']
X_test = test[features]

# Split the training data to create a validation set for Optuna
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print("Data preparation is complete. Ready for hyperparameter tuning.")


import optuna
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# We use the full training data (X_train, y_train) from Cell 4
# K-Fold will handle the splitting internally

def objective(trial):
    # We align the objective with the competition metric (RMSE)
    params = {
        'objective': 'regression_l2', # 'l2' is for RMSE, which is the competition metric
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 1000, 4000), # Increased range
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05), # More focused range
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 5, 25),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0), # Increased range for regularization
        'random_state': 42,
        'n_jobs': -1
    }

    # K-Fold Cross-Validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    # Loop through each fold
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        preds = model.predict(X_val_fold)
        rmse = mean_squared_error(y_val_fold, preds, squared=False)
        fold_scores.append(rmse)

    # Return the average RMSE across all folds
    return np.mean(fold_scores)

# Create and run the study. Let's increase trials for a better search.
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=70) # Increased to 70 trials

# Print the best results
print("Best trial found:")
best_trial = study.best_trial
print(f"  Average RMSE across 5 Folds: {best_trial.value}")
print("  Best Parameters: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")


# Get the best parameters from the Optuna study
best_params = study.best_params

# Initialize and train the final model on the ENTIRE traoning data
print("Training the final model using the best parameters...")
final_model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1)
final_model.fit(X_train, y_train)

# Make predictions on the test data
print("Making predictions on the test set...")
test_predictions = final_model.predict(X_test)

# Create and save the new submission file
submission = pd.DataFrame({'id': test['id'], 'BeatsPerMinute': test_predictions})
submission.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission_optuna.csv' created successfully!")
print(submission.head())

