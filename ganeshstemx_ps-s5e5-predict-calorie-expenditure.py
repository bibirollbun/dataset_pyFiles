print('Begin')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

import lightgbm as lgb
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

import gc
gc.collect()
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train_df.head()


test_df.head()


submission_df.head()


train_df.shape, test_df.shape, submission_df.shape


# Check for the duplicate instances
train_df.drop_duplicates().shape


train_df.isnull().sum()


test_df.isnull().sum()


print(train_df['Sex'].value_counts())
print("="*30)
print(test_df['Sex'].value_counts())


# Convert 'Sex' column to numeric: 0 for male, 1 for female
sex_map = {'male': 0, 'female': 1}
train_df['Sex'] = train_df['Sex'].map(sex_map)
test_df['Sex'] = test_df['Sex'].map(sex_map)


print(train_df['Sex'].value_counts())
print("="*30)
print(test_df['Sex'].value_counts())


# Get all the columns
numerical_columns = train_df.columns.tolist()

# Keep only the continous columns
numerical_columns.remove('id')
numerical_columns.remove('Sex')

# Determine the number of columns for the grid
n_cols = 3  # Number of columns in the grid
n_rows = (len(numerical_columns) + n_cols - 1) // n_cols  # Number of rows needed

# Create a grid of subplots
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 6 * n_rows), sharey=True)

# Flatten the axes array if it is multi-dimensional
axes = axes.flatten()

# Plot KDE for each numerical column
for ax, column in zip(axes, numerical_columns):
    sns.kdeplot(data=train_df, x=column, ax=ax, fill=True)
    ax.set_title(column)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    
# Turn off any unused subplots
for ax in axes[len(numerical_columns):]:
    ax.axis('off')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# Compute the correlation matrix
corr_matrix = train_df[numerical_columns].corr()

# Set the figure size
plt.figure(figsize=(12, 8))

# Create the heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Add a title
plt.title("Correlation Heatmap of Continuous Features")

# Show the plot
plt.show()


# Prepare features and target
X_train = train_df.drop('Calories', axis = 1)
y_train = train_df['Calories']


# Defining Configuration for k-fold cross-validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)


lgbm_params = {
    'verbosity': -1,
    'n_estimators': 6000,
    'random_state': 42,
    'reg_alpha': 0.06876635751774487, 
    'reg_lambda': 9.738899198284985,
    'metric': 'rmse',
    'learning_rate': 0.03,
    'max_depth': 4,
    # 'num_leaves': 64
}


# Initialize the lists to hold the cross-validation scores
rmse_scores = []
r2_scores = []

test_preds = np.zeros(len(test_df)) 

for fold, (train_index, val_index) in enumerate(kf.split(X_train), start=1):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    model = LGBMRegressor(**lgbm_params)
    
    # Train the model on the current fold
    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)],
              callbacks=[lgb.log_evaluation(100), lgb.early_stopping(500)])
    
    # Predict on the validation set
    val_preds = model.predict(X_val_fold)
    
    # Calculate RMSE and R-squared for the current fold
    rmse = np.sqrt(mean_squared_error(y_val_fold, val_preds))
    r2 = r2_score(y_val_fold, val_preds)
    
    rmse_scores.append(rmse)
    r2_scores.append(r2)
    
    # Predict on the test set and accumulate the predictions
    test_preds += model.predict(test_df) / kf.get_n_splits()
    
    # Print fold results
    print(f"Fold {fold}: RMSE = {rmse:.4f}, R2 = {r2:.4f}")
    
    print("\n" + "=" * 75 + "\n")

# Print overall results
print(f"\nAverage RMSE across folds: {np.mean(rmse_scores):.4f}")
print(f"Average R2 across folds: {np.mean(r2_scores):.4f}")


submission_df['Calories'] = test_preds
submission_df.to_csv('submission.csv', index = False)
submission_df.head()

