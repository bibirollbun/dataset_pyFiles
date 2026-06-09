"""
Title: LightGBM Baseline for Playground Series S5E8
Author: Jay Prajapati
Description:
    - Binary classification using LightGBM
    - 5-Fold Stratified Cross-Validation
    - Generates predictions for submission
"""


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


# Standard libraries
import os  # for file system operations

# Data handling
import pandas as pd  # CSV and dataframe operations
import numpy as np  # numerical computations

# Machine learning
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb  # LightGBM classifier

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# Read train, test and sample submission files

train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_subission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


# Define Categorical Columns

cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']


# Convert Columns to Category dtype

for col in cat_cols:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')


# Select feature columns (exclude 'id' and target 'y')

features = [col for col in train.columns if col not in ['id', 'y']]


# Assign features and target

X = train[features]
y = train['y']
X_test = test[features]


params = {
    'objective': 'binary',   # Binary classification
    'metric': 'auc',         # Evaluate using AUC
    'verbosity': -1,         # Silent mode
    'seed': 42,
    'learning_rate': 0.05,
    'num_leaves': 64,
    'min_data_in_leaf': 100,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    # Handle class imbalance
    'scale_pos_weight': (y == 0).sum() / (y == 1).sum()
}


n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oops = np.zeros(len(train))  # Out-of-fold predictions
test_preds = np.zeros(len(test))  # Test predictions

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Training fold {fold + 1}/{n_folds}")

    # Split train/validation data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Create LightGBM datasets
    train_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    val_ds = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=train_ds)
    
    # Train model with early stopping
    model = lgb.train(
        params,
        train_ds,
        num_boost_round=5000,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)],
    )
    
    # Save OOF predictions
    oof_preds = model.predict(X_val)
    oops[val_idx] = oof_preds

    # Average test predictions over folds
    test_preds += model.predict(X_test) / n_folds

    # Fold AUC
    fold_auc = roc_auc_score(y_val, oof_preds)
    print(f"Fold {fold + 1} AUC: {fold_auc:.4f}")

# Overall OOF AUC
oof_auc = roc_auc_score(y, oops)
print(f"\nOverall OOF AUC: {oof_auc:.4f}")


lgb.plot_importance(model, max_num_features=20, figsize=(10, 10))
plt.show()


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

# Create the submission DataFrame
submission = pd.DataFrame({
    "id": sample_submission["id"],
    "y": test_preds              
})

submission.to_csv('lgbm_baseline_v1.csv', index=False)
print("✅ Submission file created!")

