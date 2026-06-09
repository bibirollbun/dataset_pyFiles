# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')

# Data path and load
path = '/kaggle/input/playground-series-s5e8/'
train = pd.read_csv(path + 'train.csv', index_col='id')
test = pd.read_csv(path + 'test.csv', index_col='id')
submission = pd.read_csv(path + 'sample_submission.csv', index_col='id')

# Column Definitions
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Feature Engineering: Adding new features (log transforms or interactions)
def feature_engineering(df):
    # Avoiding log transformation errors by adding a small constant to the balance and duration columns
    df['log_balance'] = np.log1p(df['balance'])  # Log-transformed balance
    df['log_duration'] = np.log1p(df['duration'])  # Log-transformed duration
    df['age_campaign_interaction'] = df['age'] * df['campaign']  # Interaction term
    
    return df

# Preprocessing function
def preprocess(df):
    # Clipping extreme values for numerical columns
    for col in numerical_cols:
        lower_bound = df[col].quantile(0.005)
        upper_bound = df[col].quantile(0.995)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Apply feature engineering
    df = feature_engineering(df)
    
    # Check and handle NaN and Infinity
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)  # Replace NaN with column means
    
    # Convert categorical columns
    for feature in cat_cols:
        df[feature] = df[feature].astype("category")
    
    # Scaling numerical features
    scaler = StandardScaler()
    df[numerical_cols + ['log_balance', 'log_duration', 'age_campaign_interaction']] = scaler.fit_transform(df[numerical_cols + ['log_balance', 'log_duration', 'age_campaign_interaction']])
    
    return df

# Preprocess both train and test data
train = preprocess(train)
test = preprocess(test)

# Features and target
X = train.drop(columns='y')
y = train['y']

# LightGBM Model Parameters
model_lgb = lgb.LGBMClassifier(
    max_depth=5,        # Not too deep to prevent overfitting
    num_leaves=31,      # Typical value to avoid overfitting
    n_estimators=10000, # Larger n_estimators for deeper learning
    learning_rate=0.05, # Decrease learning rate for a smoother fit
    reg_alpha=0.5,      # L1 regularization
    reg_lambda=0.5,     # L2 regularization
    colsample_bytree=0.7,  # Slightly more columns sampled
    subsample=0.8,      # Ensuring some randomness in training data
    min_child_samples=50,  # Minimum number of data points in a leaf
    categorical_feature=cat_cols,
    random_state=42,
    verbosity=-1,
    objective='binary',
    metric='auc',
    importance_type='split'  # Change to 'gain' if you want to track feature importance
)

# Cross-validation setup
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(test.shape[0])
val_losses = []
roc_auc_scores = []

# Training Loop with Improved Logging and Balanced Folds
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nTraining fold {fold + 1}/{n_splits}...\n")
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model_lgb.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        eval_metric='auc',
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(period=200)]
    )
    
    # Validation Predictions
    val_preds = model_lgb.predict_proba(X_val)[:, 1]
    
    # Log Loss Calculation
    val_loss = log_loss(y_val, val_preds)
    val_losses.append(val_loss)
    print(f"Fold {fold + 1} Validation Log Loss: {val_loss:.4f}")
    
    # ROC-AUC Calculation
    roc_auc = roc_auc_score(y_val, val_preds)
    roc_auc_scores.append(roc_auc)
    print(f"Fold {fold + 1} ROC-AUC: {roc_auc:.4f}")
    
    # Test Set Predictions (Averaging over folds)
    lgb_preds += model_lgb.predict_proba(test)[:, 1] / n_splits

# Final Output Metrics
avg_val_loss = np.mean(val_losses)
avg_roc_auc = np.mean(roc_auc_scores)
print(f"\nAverage Log Loss across folds: {avg_val_loss:.4f}")
print(f"Average ROC-AUC across folds: {avg_roc_auc:.4f}")

# Prepare Submission
submission['y'] = lgb_preds  # Final predicted probabilities
submission['id'] = test.index  # Ensure 'id' is added
submission = submission[['id', 'y']]  # Re-order columns

# Save submission to CSV
submission.to_csv('submission.csv', index=False)

print("\n✅ Submission saved as 'submission.csv'")
print(submission.head())





