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


# Add new features (More attempts can be made)
import numpy as np

def add_features(df):
    df = df.copy()
    
    def credit(x):
        credit_level = (x['default'] == 'no') + (x['housing'] == 'no') + (x['loan'] == 'no')
        return {3: 27, 2: 9, 1: 3}.get(credit_level, 0)
    df['credit'] = df.apply(lambda x: credit(x), axis=1)
    
    def risk(x):
        risk_level = (x['default'] == 'yes') + (x['housing'] == 'yes') + (x['loan'] == 'yes')
        return {3: 27, 2: 9, 1: 3}.get(risk_level, 0)
    df['risk'] = df.apply(lambda x: risk(x), axis=1)
    
    df['balance_log'] = df['balance'].where(df['balance'] > -1, 0).apply(np.log1p)
    df['duration_log'] = df['duration'].where(df['duration'] > -1, 0).apply(np.log1p)
    
    month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6, 'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
    df['month'] = df['month'].map(month_map)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df['age_bin'] = pd.cut(df['age'], bins=[0, 40, 70, 100], labels=[1, 2, 0], right=False)
    
    return df

train_df = add_features(train_df)
test_df = add_features(test_df)


# Get all non-numeric columns from the training dataframe

cat_cols = train_df.select_dtypes(include="object").columns.tolist()
cat_cols


# Iterate through each non-numeric column to compare unique values between train and test sets

for col in cat_cols:
    print(f"\"{col}\" colum:\n>>> {sorted(train_df[col].unique())}\n>>> {sorted(test_df[col].unique())}\n")

# This comparison helps verify:
# 1. If test set contains values not seen in training (out-of-domain categories)
# 2. If categorical encodings can be safely applied to both sets


# from itertools import combinations

# for cols in combinations(cat_cols, 2):
#     col = "_".join(cols)
#     cat_cols.append(col)
#     train_df[col] = train_df[cols[0]].astype(str) + "_" + train_df[cols[1]].astype(str)
#     test_df[col] = test_df[cols[0]].astype(str) + "_" + test_df[cols[1]].astype(str)


# Add count encoding
from collections import Counter

for col in cat_cols:
    counts = train_df[col].value_counts()
    train_df[col + "_count"] = train_df[col].map(counts)
    test_df[col + "_count"] = test_df[col].map(counts).fillna(0)


# Prepare feature matrix (X) and target vector (y) for training, and feature matrix (X_test) for testing

X = train_df.drop(["y", "id"], axis=1)
y = train_df["y"]
X_test = test_df.drop(["id"], axis=1)


# Map non-numeric columns to numeric values using LabelEncoder for both training and test sets
from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])


# Target Encoding needs to be done within each fold to avoid target leakage

def add_target_encoding(X_train, y_train, X_val, X_test):
    X_train_te = X_train.copy()
    X_val_te = X_val.copy()
    X_test_te = X_test.copy()
    
    for col in cat_cols:
        mapping = (
            pd.DataFrame({col: X_train[col], "y": y_train})
            .groupby(col)["y"]
            .mean()
        )
        new_col = f"{col}_te"
        X_train_te[new_col] = X[col].map(mapping)
        X_val_te[new_col] = X_val[col].map(mapping)
        X_test_te[new_col] = X_test[col].map(mapping)
    
    return X_train_te, X_val_te, X_test_te


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder

# Use 5-fold stratified cross-validation
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
y_probs = np.zeros(len(X_test))
models = []
val_aucs = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Training fold {fold + 1}/{n_splits} >>>")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # target encoder
    X_train, X_val, X_test = add_target_encoding(X_train, y_train, X_val, X_test)
    
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=20000,
        learning_rate=0.03714878183568318,
        num_leaves=123,
        max_depth=18,
        min_child_samples=6,
        subsample=0.7952314244127197,
        colsample_bytree=0.31074715025280003,
        reg_alpha=1.976306329666338,
        reg_lambda=1.6478507936286915,
        max_bin=8500,
        random_state=42,
        verbosity=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(500),
            lgb.log_evaluation(period=500)
        ]
    )
    
    models.append(model)
    
    # Average predictions across all folds
    y_probs += model.predict_proba(X_test)[:, 1] / n_splits
    
    # Val AUC
    val_pred = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    val_aucs.append(val_auc)

print(f"Mean AUC: {np.mean(val_aucs):.6f}")


# Plot horizontal bar chart of feature importances if model supports it
# This helps us add new features or remove useless ones

all_importances = []

for model in models:
    if hasattr(model, "feature_importances_"):
        all_importances.append(model.feature_importances_)

if all_importances:
    index = X.columns.tolist()
    for col in cat_cols:
        index.append(f"{col}_ce")
    avg_importances = pd.Series(np.mean(all_importances, axis=0), index=index)
    avg_importances.sort_values().tail(25).plot(kind='barh')


submission = pd.DataFrame({"id": test_df["id"], "y": y_probs})
submission.to_csv("submission.csv", index=False)
submission.head()

