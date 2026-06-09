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
import lightgbm as lgb

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


from lightgbm import early_stopping, log_evaluation

# Parameters from PS-S5E8 | LightGB Model by HaohuanChen

lgb_params = {
    "n_estimators": 20000,
    "learning_rate": 0.06,
    "num_leaves": 100,
    "max_depth": 15,
    "min_child_samples": 9,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "reg_alpha": 0.78,
    "reg_lambda": 3.0,
    "max_bin": 4523,
    "random_state": 42,
    "verbosity": -1
}




# Stratified 5-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
auc_scores = []
acc_scores = []

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(period=0)
        ]
    )

    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, val_preds)
    auc = roc_auc_score(y_val, val_probs)

    print(f"Fold {fold} - Accuracy: {acc:.4f} - ROC AUC: {auc:.4f}")
    auc_scores.append(auc)
    acc_scores.append(acc)
    fold += 1

print("\nMean Accuracy:", sum(acc_scores) / len(acc_scores))
print("Mean ROC AUC:", sum(auc_scores) / len(auc_scores))


# 1. Save test IDs for submission
test_id = test["id"]

# 2. Drop 'id' column before prediction
X_test = test.drop(columns=["id"])

# 3. Apply same LabelEncoders to test set
for col in categorical_cols:
    le = label_encoders[col]
    X_test[col] = le.transform(X_test[col])  # ensure same encoding as training


# 4. Train final LightGBM model on full training data
final_model_lgb = lgb.LGBMClassifier(**lgb_params)
final_model_lgb.fit(X, y)

# 5. Predict probabilities for class 1
test_probs = final_model_lgb.predict_proba(X_test)[:, 1]

# 6. Create submission DataFrame
submission = pd.DataFrame({
    "id": test_id,
    "y": test_probs
})

# 7. Save submission file
submission.to_csv("submission_lgb.csv", index=False)
print("submission_lgb.csv file created.")



print(submission.head())

