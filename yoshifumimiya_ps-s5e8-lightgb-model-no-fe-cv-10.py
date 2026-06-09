# Print all file paths in the input directory
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd

train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


# Check and print the count of missing values in each column of the dataframe

print(train_df.isnull().sum())
print(test_df.isnull().sum())


# Get all non-numeric columns from the training dataframe

object_cols = train_df.select_dtypes(include="object").columns
print(object_cols)


# Iterate through each non-numeric column to compare unique values between train and test sets

for col_name in object_cols:
    print(f"{col_name} \n >>> {sorted(train_df[col_name].unique())} \n >>> {sorted(test_df[col_name].unique())} \n")

# This comparison helps verify:
# 1. If test set contains values not seen in training (out-of-domain categories)
# 2. If categorical encodings can be safely applied to both sets


# Add new features with predictive power (More attempts can be made)
import numpy as np

def data_process(df):
    df = df.copy()
    # df['log_duration'] = np.log1p(df['duration'])
    # df['log_age'] = np.log1p(df['age'])
    return df

train_process = data_process(train_df)
test_process = data_process(test_df)


# Prepare feature matrix (X) and target vector (y) for training, and feature matrix (X_test) for testing

X = train_process.drop(["y", "id"], axis=1)
y = train_process["y"]
X_test = test_process.drop(["id"], axis=1)


# Map non-numeric columns to numeric values using LabelEncoder for both training and test sets
from sklearn.preprocessing import LabelEncoder

for col_name in object_cols:
    le = LabelEncoder()
    X[col_name] = le.fit_transform(X[col_name])
    X_test[col_name] = le.transform(X_test[col_name])


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

# Use 5-fold stratified cross-validation
n_splits = 10
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
y_probs = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Training fold {fold + 1}/{n_splits} >>>")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(
        n_estimators=20000,
        learning_rate=0.06,
        num_leaves=100,
        max_depth=10,
        min_child_samples=9,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.79,
        reg_lambda=3.0,
        max_bin=4523,
        random_state=42,
        verbosity=-1
    )
    
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Average predictions across all folds
    y_probs += model.predict_proba(X_test)[:, 1] / n_splits


# Plot horizontal bar chart of feature importances if model supports it
# This helps us add new features or remove useless ones

if hasattr(model, "feature_importances_"):
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.sort_values().plot(kind='barh')


submission = pd.DataFrame({"id": test_df["id"], "y": y_probs})
submission.to_csv("submission.csv", index=False)
submission.head()

