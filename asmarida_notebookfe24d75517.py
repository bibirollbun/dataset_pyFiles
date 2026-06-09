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


import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
test = pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")
bureau = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau.csv")
bureau_balance = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau_balance.csv")
pos_cash = pd.read_csv("/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv")
credit_card = pd.read_csv("/kaggle/input/home-credit-default-risk/credit_card_balance.csv")
installments = pd.read_csv("/kaggle/input/home-credit-default-risk/installments_payments.csv")

# Function for numeric feature aggregation
def aggregate_numeric(df, group_var, df_name):
    numeric_df = df.select_dtypes(include=[np.number])
    agg = numeric_df.groupby(group_var).agg(["mean", "sum", "min", "max", "std"])
    agg.columns = [f"{df_name}_{col[0]}_{col[1]}" for col in agg.columns]
    return agg.reset_index()

def encode_categorical(df, group_var, df_name):
    categorical = df.select_dtypes(exclude=[np.number])
    if categorical.empty:
        return pd.DataFrame({group_var: df[group_var]})
    encoded = pd.get_dummies(categorical)
    encoded[group_var] = df[group_var]
    agg = encoded.groupby(group_var).agg(["sum", "mean"])
    agg.columns = [f"{df_name}_{col[0]}_{col[1]}" for col in agg.columns]
    return agg.reset_index()

# Process bureau & bureau_balance
bureau_balance_agg = aggregate_numeric(bureau_balance, "SK_ID_BUREAU", "bureau_balance")
bureau = bureau.merge(bureau_balance_agg, on="SK_ID_BUREAU", how="left")

bureau_numeric_agg = aggregate_numeric(bureau, "SK_ID_CURR", "bureau")
bureau_categorical_agg = encode_categorical(bureau, "SK_ID_CURR", "bureau")
bureau_agg = bureau_numeric_agg.merge(bureau_categorical_agg, on="SK_ID_CURR", how="left")

# Process other datasets
pos_cash_agg = aggregate_numeric(pos_cash, "SK_ID_CURR", "pos_cash")
credit_card_agg = aggregate_numeric(credit_card, "SK_ID_CURR", "credit_card")
installments_agg = aggregate_numeric(installments, "SK_ID_CURR", "installments")

# Merge all aggregated datasets into train & test
train = train.merge(bureau_agg, on="SK_ID_CURR", how="left")
test = test.merge(bureau_agg, on="SK_ID_CURR", how="left")
train = train.merge(pos_cash_agg, on="SK_ID_CURR", how="left")
test = test.merge(pos_cash_agg, on="SK_ID_CURR", how="left")
train = train.merge(credit_card_agg, on="SK_ID_CURR", how="left")
test = test.merge(credit_card_agg, on="SK_ID_CURR", how="left")
train = train.merge(installments_agg, on="SK_ID_CURR", how="left")
test = test.merge(installments_agg, on="SK_ID_CURR", how="left")

# Interaction Feature: Add INCOME_BY_FAMILY
train['INCOME_BY_FAMILY'] = train['AMT_INCOME_TOTAL'] / (train['CNT_FAM_MEMBERS'] + 1)
test['INCOME_BY_FAMILY'] = test['AMT_INCOME_TOTAL'] / (test['CNT_FAM_MEMBERS'] + 1)

# One-Hot Encoding
if "TARGET" in train.columns:
    target = train.pop("TARGET")
    train = pd.get_dummies(train)
    test = pd.get_dummies(test)
    train, test = train.align(test, join="left", axis=1)
    train["TARGET"] = target  # Reattach TARGET
else:
    print("⚠️ 'TARGET' column not found in train dataset!")

# Clean column names by replacing special characters with underscores
train.columns = train.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
test.columns = test.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)

# Ensure train and test have the same columns
train, test = train.align(test, join="left", axis=1)

# Splitting Data
X = train.drop(columns=["TARGET"])
y = train["TARGET"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning: RandomizedSearchCV
param_dist = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'num_leaves': [31, 60, 100],
    'max_depth': [-1, 5, 7, 10],
    'min_data_in_leaf': [10, 20, 30],
    'n_estimators': [100, 200, 500, 700]
}

model = lgb.LGBMClassifier(objective='binary', metric='auc', boosting_type='gbdt')
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=3, verbose=1, random_state=42)
random_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = random_search.best_params_
best_model = random_search.best_estimator_

print(f"Best Hyperparameters: {best_params}")

# Model Training
best_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="auc")

# Model Evaluation: AUC Score
preds = best_model.predict_proba(X_valid)[:, 1]
auc_score = roc_auc_score(y_valid, preds)
print(f'Validation AUC: {auc_score:.4f}')

# Feature Importance
lgb.plot_importance(best_model, max_num_features=20, importance_type='split', figsize=(10, 6))
plt.title('Feature Importance')
plt.show()

# Or manually extract and plot feature importance
importance = best_model.feature_importances_
features = X.columns
feature_importance = pd.DataFrame({'feature': features, 'importance': importance})
feature_importance = feature_importance.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
plt.title('Top 20 Features by Importance')
plt.show()

# Cross-validation using StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

roc_auc_scores = []
for train_idx, val_idx in kf.split(X, y):
    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
    
    best_model.fit(X_train_cv, y_train_cv)
    preds = best_model.predict_proba(X_val_cv)[:, 1]
    auc_score = roc_auc_score(y_val_cv, preds)
    roc_auc_scores.append(auc_score)

print(f'Cross-validated AUC: {np.mean(roc_auc_scores):.4f}')

# Final Model Training with Best Hyperparameters
best_model.fit(X_train, y_train)

# Final predictions on the test set
test_preds = best_model.predict_proba(test.drop(columns=['SK_ID_CURR']))[:, 1]
submission = pd.DataFrame({"SK_ID_CURR": test['SK_ID_CURR'], "TARGET": test_preds})
submission.to_csv("final_submission.csv", index=False)
print("✅ Final submission file saved!")


import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Load datasets
train = pd.read_csv("/kaggle/input/home-credit-default-risk/application_train.csv")
test = pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")
bureau = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau.csv")
bureau_balance = pd.read_csv("/kaggle/input/home-credit-default-risk/bureau_balance.csv")
pos_cash = pd.read_csv("/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv")
credit_card = pd.read_csv("/kaggle/input/home-credit-default-risk/credit_card_balance.csv")
installments = pd.read_csv("/kaggle/input/home-credit-default-risk/installments_payments.csv")

# Function for numeric feature aggregation
def aggregate_numeric(df, group_var, df_name):
    numeric_df = df.select_dtypes(include=[np.number])
    agg = numeric_df.groupby(group_var).agg(["mean", "sum", "min", "max", "std"])
    agg.columns = [f"{df_name}_{col[0]}_{col[1]}" for col in agg.columns]
    return agg.reset_index()

def encode_categorical(df, group_var, df_name):
    categorical = df.select_dtypes(exclude=[np.number])
    if categorical.empty:
        return pd.DataFrame({group_var: df[group_var]})
    encoded = pd.get_dummies(categorical)
    encoded[group_var] = df[group_var]
    agg = encoded.groupby(group_var).agg(["sum", "mean"])
    agg.columns = [f"{df_name}_{col[0]}_{col[1]}" for col in agg.columns]
    return agg.reset_index()

# Process bureau & bureau_balance
bureau_balance_agg = aggregate_numeric(bureau_balance, "SK_ID_BUREAU", "bureau_balance")
bureau = bureau.merge(bureau_balance_agg, on="SK_ID_BUREAU", how="left")

bureau_numeric_agg = aggregate_numeric(bureau, "SK_ID_CURR", "bureau")
bureau_categorical_agg = encode_categorical(bureau, "SK_ID_CURR", "bureau")
bureau_agg = bureau_numeric_agg.merge(bureau_categorical_agg, on="SK_ID_CURR", how="left")

# Process other datasets
pos_cash_agg = aggregate_numeric(pos_cash, "SK_ID_CURR", "pos_cash")
credit_card_agg = aggregate_numeric(credit_card, "SK_ID_CURR", "credit_card")
installments_agg = aggregate_numeric(installments, "SK_ID_CURR", "installments")

# Merge all aggregated datasets into train & test
train = train.merge(bureau_agg, on="SK_ID_CURR", how="left")
test = test.merge(bureau_agg, on="SK_ID_CURR", how="left")
train = train.merge(pos_cash_agg, on="SK_ID_CURR", how="left")
test = test.merge(pos_cash_agg, on="SK_ID_CURR", how="left")
train = train.merge(credit_card_agg, on="SK_ID_CURR", how="left")
test = test.merge(credit_card_agg, on="SK_ID_CURR", how="left")
train = train.merge(installments_agg, on="SK_ID_CURR", how="left")
test = test.merge(installments_agg, on="SK_ID_CURR", how="left")

# One-Hot Encoding
if "TARGET" in train.columns:
    target = train.pop("TARGET")
    train = pd.get_dummies(train)
    test = pd.get_dummies(test)
    train, test = train.align(test, join="left", axis=1)
    train["TARGET"] = target  # Reattach TARGET
else:
    print("⚠️ 'TARGET' column not found in train dataset!")

# Clean column names by replacing special characters with underscores
train.columns = train.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
test.columns = test.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)

# Ensure train and test have the same columns
train, test = train.align(test, join="left", axis=1)

# Splitting Data
X = train.drop(columns=["TARGET"])
y = train["TARGET"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM Model Training with Fixed Parameters
def train_model(train, test):
    features = [col for col in train.columns if col not in ['SK_ID_CURR', 'TARGET']]
    X = train[features]
    y = train['TARGET']
    
    best_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',  # Change to 'dart' if overfitting persists
    'n_estimators': 2000,  # Increased for better learning
    'learning_rate': 0.02,  # Slower but steadier learning
    'num_leaves': 100,  # Increased to allow more leaf splits
    'max_depth': 11,  # Increased depth for more feature interactions
    'min_data_in_leaf': 30,  # Helps prevent overfitting
    'feature_fraction': 0.75,  # Slightly reduced to introduce feature randomness
    'bagging_fraction': 0.85,  # More samples per iteration
    'bagging_freq': 3,  # More frequent bagging
    'lambda_l1': 0.2,  # More L1 regularization
    'lambda_l2': 0.4,  # More L2 regularization
    'extra_trees': True,  # Randomize tree splits for better generalization
    'min_gain_to_split': 0.02,  # Minimum gain for a split to reduce overfitting
    }
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = lgb.LGBMClassifier(**best_params)
    #model.fit(
     #   X_train, y_train, eval_set=[(X_valid, y_valid)], 
      #  eval_metric="auc", 
    #)
   
    model.fit(
    X_train, y_train, 
    eval_set=[(X_valid, y_valid)], 
    eval_metric="auc", 
)

    preds = model.predict_proba(X_valid)[:, 1]
    auc_score = roc_auc_score(y_valid, preds)
    print(f'Validation AUC: {auc_score:.4f}')
    
    # Ensure test features match train features
    test_preds = model.predict_proba(test[features])[:, 1]
    test['TARGET'] = test_preds
    test[['SK_ID_CURR', 'TARGET']].to_csv('submission.csv', index=False)
    
    return model, auc_score

# Train the model with fixed parameters
model, auc_score = train_model(train, test)

# Ensure the test set has the same columns as the training set
test = test[train.columns.drop('TARGET')]  # Align test to have only the same columns as train

# Check if any columns are missing in the test set
missing_cols = set(train.columns) - set(test.columns)
if missing_cols:
    print(f"Missing columns in test set: {missing_cols}")
    for col in missing_cols:
        test[col] = 0  # Add missing columns with default value 0

# Ensure that the 'TARGET' column is not in the test set
test = test.drop(columns=['TARGET'], errors='ignore')  # Remove 'TARGET' if it exists

# Align test columns with the train columns, ensuring only features used for prediction
test = test[train.columns.drop('TARGET')]  # Ensure test has the same features as train

# Check if any columns are missing or extra in the test set
missing_cols = set(train.columns) - set(test.columns)
extra_cols = set(test.columns) - set(train.columns)

if missing_cols:
    print(f"Missing columns in test set: {missing_cols}")
    for col in missing_cols:
        test[col] = 0  # Add missing columns with default value 0

if extra_cols:
    print(f"Extra columns in test set: {extra_cols}")
    test = test.drop(columns=extra_cols)  # Drop extra columns

# Ensure the test set columns are in the same order as the train set
test = test[train.columns.drop('TARGET')]

# Now, make predictions
test_preds = model.predict_proba(test.drop(columns=['SK_ID_CURR']))[:, 1]


# Save the submission
submission = pd.DataFrame({"SK_ID_CURR": test['SK_ID_CURR'], "TARGET": test_preds})
submission.to_csv("submission.csv", index=False)
print("✅ Submission file saved!")


