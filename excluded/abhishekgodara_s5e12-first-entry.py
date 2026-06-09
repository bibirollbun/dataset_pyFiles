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


import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- 1. Load Data ---
# Load the training, testing, and sample submission files
train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


# Prepare features and target
train_features = train_df.drop(['id', 'diagnosed_diabetes'], axis=1)
test_features = test_df.drop('id', axis=1)
# The target variable 'diagnosed_diabetes' is converted to integer (0 or 1)
y = train_df['diagnosed_diabetes'].astype(int)

# --- 2. Data Preprocessing (One-Hot Encoding) ---
# Combine train and test features for consistent encoding
combined_df = pd.concat([train_features, test_features], ignore_index=True)

# Identify categorical columns (type 'object')
categorical_cols = combined_df.select_dtypes(include='object').columns

# Apply One-Hot Encoding
combined_df = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True)

# Separate back into training and test sets
X_train_processed = combined_df.iloc[:len(train_features)]
X_test_processed = combined_df.iloc[len(train_features):]


# model = RandomForestClassifier(
#     n_estimators=100,           # Number of trees in the forest
#     random_state=42,            # Seed for reproducibility
#     n_jobs=-1,                  # Use all available CPU cores
#     max_depth=10,               # Maximum depth of the trees
#     min_samples_leaf=5          # Minimum number of samples required to be at a leaf node
# )

# # Train the model
# model.fit(X_train_processed, y)


import xgboost as xgb
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=500,        # Number of boosting rounds (trees)
    learning_rate=0.05,      # Step size shrinkage used to prevent overfitting
    max_depth=7,             # Maximum depth of a tree
    min_child_weight=1,      # Minimum sum of instance weight (hessian) needed in a child
    gamma=0.1,               # Minimum loss reduction required to make a further partition
    subsample=0.8,           # Subsample ratio of the training instance
    colsample_bytree=0.8,    # Subsample ratio of columns when constructing each tree
    scale_pos_weight=(len(y[y==0]) / len(y[y==1])), # Handle class imbalance (if applicable)
    random_state=42,
    use_label_encoder=False, # Suppress warning
    eval_metric='logloss',   # Evaluation metric for monitoring training
    n_jobs=-1                # Use all available CPU cores
)
model.fit(X_train_processed,y)


predictions = model.predict(X_test_processed)

submission_df['diagnosed_diabetes'] = predictions.astype(int)

# Save the predictions to a CSV file
submission_df.to_csv('submission.csv', index=False)

print("Model training complete and 'submission.csv' saved successfully.")







