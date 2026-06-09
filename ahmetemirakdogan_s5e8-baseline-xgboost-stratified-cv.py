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


# Basic imports
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Encoding, evaluation, model selection
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score

# Model
import xgboost as xgb

# Misc
import warnings
warnings.filterwarnings("ignore")



# Load competition data
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

# Load original Bank Marketing dataset
original = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=';')


print("Train Set Shape:", train.shape)
train


print("Test Set Shape:", test.shape)
test


print("Original Set Shape:", original.shape)
original


sample_submission


print("Unique values in original['y']:", original["y"].unique())
print("Unique values in train['y']:", train["y"].unique())

# Drop 'id' column from train
train = train.drop(columns=["id"])

# Map 'yes'/'no' to 1/0 in original dataset
original["y"] = original["y"].map({"no": 0, "yes": 1})

# Reorder original columns to match train
original = original[train.columns]

# Concatenate both datasets
merged = pd.concat([train, original], ignore_index=True)

# Confirm everything is numeric and merged
print("Merged shape:", merged.shape)
print("Merged target distribution:\n", merged["y"].value_counts())


# Show data types of each column
print(merged.dtypes)

# List of numeric columns
numeric_cols = merged.select_dtypes(include=["int64", "float64"]).columns.tolist()

# List of categorical (object/string) columns
categorical_cols = merged.select_dtypes(include=["object"]).columns.tolist()

print("Numeric features:", numeric_cols)
print("Categorical features:", categorical_cols)


# Check for missing values in each column
missing_values = merged.isnull().sum()

# Show only columns with at least one missing value
missing_values = missing_values[missing_values > 0]

# Display
print("Missing values:\n", missing_values)



# Identify categorical columns
categorical_cols = merged.select_dtypes(include=["object"]).columns

# Apply Label Encoding to each categorical column
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    merged[col] = le.fit_transform(merged[col])
    label_encoders[col] = le



# Separate features and target
X = merged.drop(columns=["y"])
y = merged["y"]



## Optimized XGBoost parameters (from Optuna)
xgb_params = {
    'learning_rate': 0.02,
    'max_depth': 7,
    'min_child_weight': 14,
    'gamma': 4.586254878038162,
    'subsample': 0.8829082972307842,
    'colsample_bytree': 0.7,
    'lambda': 9.976935506747019,
    'alpha': 0.0018063487584367348,
    'scale_pos_weight': 3.6963430205760193,
    'objective': 'binary:logistic',
    'use_label_encoder': False,
    'n_jobs': -1,
    'random_state': 42,
    'tree_method': 'gpu_hist',
    'n_estimators': 10000
}



# Stratified 5-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # XGBoost Classifier
    model = xgb.XGBClassifier(**xgb_params)

    # Fit model
    model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=150,
    eval_metric='auc',
    verbose=False
)

    # Predict class and probabilities
    preds_class = model.predict(X_val)
    preds_proba = model.predict_proba(X_val)[:, 1]  # for ROC AUC

    # Metrics
    acc = accuracy_score(y_val, preds_class)
    auc = roc_auc_score(y_val, preds_proba)

    print(f"Fold {fold} - Accuracy: {acc:.4f} - ROC AUC: {auc:.4f}")
    fold += 1




# 1. Drop test id
test_id = test["id"]  # Save for submission
test = test.drop(columns=["id"])

# 2. Apply same LabelEncoders to test set
for col in categorical_cols:
    le = label_encoders[col]
    test[col] = le.transform(test[col])  # same categories as train

# 3. Final X and y for training
X_final = merged.drop(columns=["y"])
y_final = merged["y"]

# 4. Train XGBoost model on full data
final_model = xgb.XGBClassifier(**xgb_params)

final_model.fit(
    X_final, y_final,
    eval_set=[(X_final, y_final)],
    early_stopping_rounds=150,
    eval_metric="auc",
    verbose=False
)


# 5. Predict probabilities on test set (for class=1)
test_preds_proba = final_model.predict_proba(test)[:, 1]

# 6. Create submission file using raw probabilities
submission = pd.DataFrame({
    "id": test_id,
    "y": test_preds_proba
})

# 7. Save to CSV
submission.to_csv("submission.csv", index=False)
print(" submission.csv file created")



print(submission.head())

