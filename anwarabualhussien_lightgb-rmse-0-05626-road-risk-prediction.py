# =============================================================================
# 1. SETUP AND IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# For reproducibility of results
SEED = 42

print("Libraries imported successfully.")





# 2. DATA PREPARATION

# Load the datasets
train_df  = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_df= pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)




 train_df.head()



 test_df.head()



# Store test IDs for final submission file
test_ids = test_df['id']


# This is safer and can be run multiple times
train_df = train_df.drop("id", axis=1, errors='ignore')
test_df = test_df.drop("id", axis=1, errors='ignore')


# Separate features (X) and target (y)
X = train_df.drop('accident_risk', axis=1)
y = train_df['accident_risk']
X_test = test_df


# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object', 'bool']).columns
numerical_features = X.select_dtypes(include=np.number).columns

print(f"Categorical Features: {list(categorical_features)}")
print(f"Numerical Features: {list(numerical_features)}")

# Apply One-Hot Encoding
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

# Align columns - crucial for when test set might miss a category
train_cols = X.columns
test_cols = X_test.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0
missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X[c] = 0

X_test = X_test[train_cols] # Ensure order is the same

# Apply Scaling
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

print("\nData preprocessed successfully.")
print("Shape of X_train after preprocessing:", X.shape)
print("Shape of X_test after preprocessing:", X_test.shape)


# =============================================================================
# 4. MODEL TRAINING (LIGHTGBM WITH 5-FOLD CV AND GPU)
# =============================================================================
print("\nStarting model training with 5-Fold Cross-Validation...")

NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(X.shape[0])
sub_preds = np.zeros(X_test.shape[0])
feature_importances = pd.DataFrame(index=X.columns)

# LightGBM parameters - configured for GPU
lgb_params = {
    'objective': 'regression_l1', # MAE is often more robust to outliers
    'metric': 'rmse',
    'n_estimators': 10000,        # High number, will be stopped by early stopping
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 64,
    'verbose': -1,
    'n_jobs': -1,
    'seed': SEED,
    'boosting_type': 'gbdt',
    'device': 'gpu'               # <-- KEY PARAMETER FOR GPU USAGE
}

for n_fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    
    model.fit(X_train, y_train, 
              eval_set=[(X_valid, y_valid)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(200, verbose=False)])

    # Store predictions
    oof_preds[valid_idx] = model.predict(X_valid)
    sub_preds += model.predict(X_test) / NFOLDS
    
    # Store feature importances
    feature_importances[f'fold_{n_fold+1}'] = model.feature_importances_
    
    fold_rmse = np.sqrt(mean_squared_error(y_valid, oof_preds[valid_idx]))
    print(f"Fold {n_fold+1} RMSE: {fold_rmse}")

# Calculate overall Out-of-Fold RMSE
overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"\nOverall Out-of-Fold CV RMSE: {overall_rmse}")








# =============================================================================
# 5. FEATURE IMPORTANCE VISUALIZATION
# =============================================================================
print("\nVisualizing feature importances...")

# Calculate mean importance across all folds
feature_importances['mean'] = feature_importances.mean(axis=1)
feature_importances.sort_values('mean', ascending=False, inplace=True)

# Plot
plt.figure(figsize=(12, 16))
sns.barplot(x='mean', y=feature_importances.index, data=feature_importances)
plt.title('LightGBM Feature Importance (Mean over 5 Folds)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.grid(True)
plt.show()


# =============================================================================
# 6. GENERATE SUBMISSION FILE
# =============================================================================
print("\nGenerating submission file...")

submission_df = pd.DataFrame({'id': test_ids, 'accident_risk': sub_preds})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print("Top 5 rows of the submission file:")
print(submission_df.head())

