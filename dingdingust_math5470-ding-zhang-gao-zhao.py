# Step 1 Python code: Basic imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For modeling
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Aesthetic settings
sns.set_style('whitegrid')
pd.set_option('display.max_columns', 100)

print("Step 1: Libraries imported successfully.")


# Step 2 Python code: Load application_train and application_test

train_df = pd.read_csv('../input/home-credit-default-risk/application_train.csv')
test_df = pd.read_csv('../input/home-credit-default-risk/application_test.csv')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Preview the first few rows of the training data
display(train_df.head())


# Step 3 Python code: Basic EDA

# 1) Distribution of TARGET
print("Target distribution (train_df):")
print(train_df['TARGET'].value_counts(normalize=True))

# 2) Missing values in train_df
missing_train = train_df.isnull().sum()
missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
print("\nNumber of missing values per column in train_df (only columns with NAs):")
print(missing_train.head(10))

# 3) Quick numeric summary
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumeric columns sample (first 5):")
print(train_df[numeric_cols].head(5))


# Step 4 Python code: Basic Data Cleaning

# 1) Fill numeric missing values with 0
num_cols = train_df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    train_df[col] = train_df[col].fillna(0)

num_cols_test = test_df.select_dtypes(include=[np.number]).columns
for col in num_cols_test:
    test_df[col] = test_df[col].fillna(0)

# 2) Handle DAYS_EMPLOYED outliers (example: if > 36500, set to NaN or 0)
# In some solutions, 365243 is treated as an "indicator" for no official employment.
train_df.loc[train_df['DAYS_EMPLOYED'] > 36500, 'DAYS_EMPLOYED'] = 0
test_df.loc[test_df['DAYS_EMPLOYED'] > 36500, 'DAYS_EMPLOYED'] = 0

print("After basic cleaning:")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Quick check to ensure no NaNs remain in numeric columns
print("\nAny NaNs left in numeric cols (train)?", train_df[num_cols].isnull().sum().sum())
print("Any NaNs left in numeric cols (test)?", test_df[num_cols_test].isnull().sum().sum())


# Step 5 Python code: Simple Feature Engineering with bureau.csv

# Load bureau data
bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv')
print("bureau shape:", bureau.shape)
display(bureau.head())

# Example: Aggregated feature - total number of previous bureau loans per client
bureau_agg = bureau.groupby('SK_ID_CURR')['SK_ID_BUREAU'].count().reset_index()
bureau_agg.columns = ['SK_ID_CURR', 'PREV_CREDIT_COUNT']

# Merge into train/test
train_df = train_df.merge(bureau_agg, on='SK_ID_CURR', how='left')
test_df  = test_df.merge(bureau_agg, on='SK_ID_CURR', how='left')

# Fill any new NaNs introduced by the merge with 0
train_df['PREV_CREDIT_COUNT'] = train_df['PREV_CREDIT_COUNT'].fillna(0)
test_df['PREV_CREDIT_COUNT']  = test_df['PREV_CREDIT_COUNT'].fillna(0)

print("\nAfter merging 'PREV_CREDIT_COUNT':")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Quick check of the new feature distribution in training data
print("\nDistribution of PREV_CREDIT_COUNT in train:")
print(train_df['PREV_CREDIT_COUNT'].describe())


# Step 6 Python code: Creating ratio features and age

# 1) ANNUITY_INCOME_RATIO
train_df['ANNUITY_INCOME_RATIO'] = (train_df['AMT_ANNUITY'] / (train_df['AMT_INCOME_TOTAL'] + 1e-6))
test_df['ANNUITY_INCOME_RATIO']  = (test_df['AMT_ANNUITY']  / (test_df['AMT_INCOME_TOTAL'] + 1e-6))

# 2) CREDIT_INCOME_RATIO
train_df['CREDIT_INCOME_RATIO'] = (train_df['AMT_CREDIT'] / (train_df['AMT_INCOME_TOTAL'] + 1e-6))
test_df['CREDIT_INCOME_RATIO']  = (test_df['AMT_CREDIT']  / (test_df['AMT_INCOME_TOTAL'] + 1e-6))

# 3) AGE_YEARS (approx)
train_df['AGE_YEARS'] = (-1 * train_df['DAYS_BIRTH']) / 365
test_df['AGE_YEARS']  = (-1 * test_df['DAYS_BIRTH']) / 365

# Quick check of new columns
new_cols = ['ANNUITY_INCOME_RATIO','CREDIT_INCOME_RATIO','AGE_YEARS']
print("\nSample of new ratio/age features (first 5 rows):")
display(train_df[new_cols].head())


# Step 7 Python code:

from sklearn.preprocessing import LabelEncoder

# 1) Separate target
y = train_df['TARGET']
train_df_features = train_df.drop(['TARGET','SK_ID_CURR'], axis=1, errors='ignore')
test_df_features  = test_df.drop(['SK_ID_CURR'], axis=1, errors='ignore')

# 2) Align columns
common_cols = list(set(train_df_features.columns) & set(test_df_features.columns))
X = train_df_features[common_cols]
X_test = test_df_features[common_cols]

# Identify object columns and apply label encoding 
for col in X.select_dtypes('object').columns:
    combined_data = pd.concat([X[col], X_test[col]], axis=0)
    lb = LabelEncoder()
    lb.fit(list(combined_data.astype(str).values))
    X[col] = lb.transform(list(X[col].astype(str).values))
    X_test[col] = lb.transform(list(X_test[col].astype(str).values))

print("Final X shape:", X.shape)
print("Final X_test shape:", X_test.shape)

# 3) Cross-validation setup
n_folds = 5
folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
auc_scores = []

# 4) LightGBM training
for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
        'seed': 42
    }

    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=1000,
        valid_sets=[dvalid],  # Evaluate on validation fold
        callbacks=[
            lgb.early_stopping(50), 
            lgb.log_evaluation(0)    
        ]
    )
    
    valid_preds = model.predict(X_valid, num_iteration=model.best_iteration)
    fold_auc = roc_auc_score(y_valid, valid_preds)
    auc_scores.append(fold_auc)
    print(f"Fold {fold_idx+1} AUC: {fold_auc:.4f}")

print(f"\nMean AUC across {n_folds} folds: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")


# Step 8 Python code: Final Model & Submission

# We can use the same hyperparameters from our CV step
final_model = lgb.LGBMClassifier(
    objective='binary',
    learning_rate=0.05,
    num_leaves=31,
    n_estimators=500,  # Chosen somewhat arbitrarily; can be refined
    random_state=42
)

# Train on the full dataset
final_model.fit(X, y)

# Predict probabilities for the test set
test_preds = final_model.predict_proba(X_test)[:, 1]

# Build submission DataFrame
submission = pd.DataFrame({
    'SK_ID_CURR': test_df['SK_ID_CURR'],
    'TARGET': test_preds
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")
print(submission.head(10))


# Step 9 Python Code: Advanced Feature Engineering with previous_application.csv

# 1. Load previous_application data
prev_app = pd.read_csv('../input/home-credit-default-risk/previous_application.csv')
print("previous_application shape:", prev_app.shape)
display(prev_app.head())

# 2. Example Aggregations
#    We'll group by SK_ID_CURR and compute stats like:
#      - COUNT of prev loans
#      - MEAN of AMT_CREDIT
#      - MEAN of AMT_ANNUITY
#      - MAX of CNT_PAYMENT (if not null)
# Feel free to add more if you want.

agg_dict = {
    'AMT_CREDIT': ['mean'],
    'AMT_ANNUITY': ['mean'],
    'CNT_PAYMENT': ['max']
}
prev_agg = prev_app.groupby('SK_ID_CURR').agg(agg_dict)
prev_agg.columns = ['PREVAPP_CREDIT_MEAN', 'PREVAPP_ANNUITY_MEAN', 'PREVAPP_CNT_PAYMENT_MAX']
prev_agg.reset_index(inplace=True)

# Also, let's add a simple count of how many previous applications per SK_ID_CURR
prev_count = prev_app.groupby('SK_ID_CURR')['SK_ID_PREV'].count().reset_index()
prev_count.columns = ['SK_ID_CURR', 'PREVAPP_COUNT']

# Merge the two aggregated dataframes
prev_agg = prev_agg.merge(prev_count, on='SK_ID_CURR', how='left')

print("\nAggregated previous_application data (head):")
display(prev_agg.head())

# 3. Merge into train_df and test_df
train_df = train_df.merge(prev_agg, on='SK_ID_CURR', how='left')
test_df  = test_df.merge(prev_agg, on='SK_ID_CURR', how='left')

# 4. Fill newly introduced NaNs with 0 (simple approach)
for col in ['PREVAPP_CREDIT_MEAN', 'PREVAPP_ANNUITY_MEAN','PREVAPP_CNT_PAYMENT_MAX','PREVAPP_COUNT']:
    train_df[col] = train_df[col].fillna(0)
    test_df[col]  = test_df[col].fillna(0)

print("\nAfter merging aggregated features from previous_application:")
print("train_df shape:", train_df.shape)
print("test_df shape:", test_df.shape)


# Step 10 Python Code: Feature Engineering with installments_payments.csv

# 1. Load installments_payments data
inst_pay = pd.read_csv('../input/home-credit-default-risk/installments_payments.csv')
print("installments_payments shape:", inst_pay.shape)
display(inst_pay.head(5))

# 2. Example Aggregations:
#    - TOTAL_PAYMENT_SUM: sum of AMT_PAYMENT
#    - PAYMENT_RATIO (AMT_PAYMENT / AMT_INSTALMENT) statistics
#    - COUNT of late payments

# Create a 'payment_ratio' column
inst_pay['PAYMENT_RATIO'] = inst_pay['AMT_PAYMENT'] / (inst_pay['AMT_INSTALMENT'] + 1e-6)

# Flag late payments
inst_pay['LATE_PAYMENT'] = (inst_pay['DAYS_ENTRY_PAYMENT'] - inst_pay['DAYS_INSTALMENT']) > 0

# Group by SK_ID_CURR
agg_dict_inst = {
    'AMT_PAYMENT': ['sum'],
    'PAYMENT_RATIO': ['mean'],
    'LATE_PAYMENT': ['sum']  # total number of late payments
}
inst_agg = inst_pay.groupby('SK_ID_CURR').agg(agg_dict_inst)
inst_agg.columns = ['INSTALL_PAYMENT_SUM','INSTALL_RATIO_MEAN','INSTALL_LATE_COUNT']
inst_agg.reset_index(inplace=True)

print("\nAggregated installments_payments data (head):")
display(inst_agg.head())

# 3. Merge into train_df and test_df
train_df = train_df.merge(inst_agg, on='SK_ID_CURR', how='left')
test_df  = test_df.merge(inst_agg, on='SK_ID_CURR', how='left')

# 4. Fill newly introduced NaN with 0
for col in ['INSTALL_PAYMENT_SUM','INSTALL_RATIO_MEAN','INSTALL_LATE_COUNT']:
    train_df[col] = train_df[col].fillna(0)
    test_df[col] = test_df[col].fillna(0)

print("\nAfter merging from installments_payments:")
print("train_df shape:", train_df.shape)
print("test_df shape:", test_df.shape)


# Step 11 Python Code: Credit Card Balance features

# 1) Load credit card balance
cc_balance = pd.read_csv('../input/home-credit-default-risk/credit_card_balance.csv')
print("credit_card_balance shape:", cc_balance.shape)
display(cc_balance.head(5))

# 2) Quick visualization of AMT_BALANCE distribution
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.histplot(cc_balance['AMT_BALANCE'], bins=50, kde=True)
plt.title("Distribution of AMT_BALANCE (credit_card_balance)")
plt.xlim(0, cc_balance['AMT_BALANCE'].quantile(0.95)) # focus on 95th percentile
plt.show()

# 3) Groupby aggregation
# Let's create mean and max for AMT_BALANCE, and sum for CNT_DRAWINGS_CURRENT
cc_balance['CNT_DRAWINGS_CURRENT'] = cc_balance['CNT_DRAWINGS_CURRENT'].fillna(0)
agg_dict_cc = {
    'AMT_BALANCE': ['mean', 'max'],
    'CNT_DRAWINGS_CURRENT': ['sum']
}
cc_agg = cc_balance.groupby('SK_ID_CURR').agg(agg_dict_cc)
cc_agg.columns = ['CC_BALANCE_MEAN', 'CC_BALANCE_MAX', 'CC_DRAWINGS_SUM']
cc_agg.reset_index(inplace=True)

print("\nAggregated credit_card_balance data (head):")
display(cc_agg.head(5))

# 4) Merge into train_df, test_df
train_df = train_df.merge(cc_agg, on='SK_ID_CURR', how='left')
test_df  = test_df.merge(cc_agg, on='SK_ID_CURR', how='left')

# 5) Fill new NaNs with 0
for col in ['CC_BALANCE_MEAN','CC_BALANCE_MAX','CC_DRAWINGS_SUM']:
    train_df[col] = train_df[col].fillna(0)
    test_df[col]  = test_df[col].fillna(0)

print("\nAfter merging from credit_card_balance:")
print("train_df shape:", train_df.shape)
print("test_df shape:", test_df.shape)


# Step 12 Python Code: POS_CASH_balance features

pos_cash = pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv')
print("POS_CASH_balance shape:", pos_cash.shape)
display(pos_cash.head(5))

# 1) Aggregations
agg_dict_pos = {
    'MONTHS_BALANCE': ['min'],   # earliest record
    'SK_DPD': ['max'],           # worst overdue
    'SK_DPD_DEF': ['max']        # worst overdue in terms of official DPD
}

pos_agg = pos_cash.groupby('SK_ID_CURR').agg(agg_dict_pos)
pos_agg.columns = ['POS_MONTHS_BALANCE_MIN','POS_SK_DPD_MAX','POS_SK_DPD_DEF_MAX']
pos_agg.reset_index(inplace=True)

# Also, add total records count
pos_count = pos_cash.groupby('SK_ID_CURR')['MONTHS_BALANCE'].count().reset_index()
pos_count.columns = ['SK_ID_CURR', 'POS_COUNT']

pos_agg = pos_agg.merge(pos_count, on='SK_ID_CURR', how='left')

print("\nAggregated POS_CASH data (head):")
display(pos_agg.head(5))

# 2) Merge into train_df and test_df
train_df = train_df.merge(pos_agg, on='SK_ID_CURR', how='left')
test_df  = test_df.merge(pos_agg, on='SK_ID_CURR', how='left')

# 3) Fill newly introduced NaNs with 0
for col in ['POS_MONTHS_BALANCE_MIN','POS_SK_DPD_MAX','POS_SK_DPD_DEF_MAX','POS_COUNT']:
    train_df[col] = train_df[col].fillna(0)
    test_df[col]  = test_df[col].fillna(0)

print("\nAfter merging from POS_CASH_balance:")
print("train_df shape:", train_df.shape)
print("test_df shape:", test_df.shape)


# Step 13 Python Code: Re-run cross-validation with new, expanded features

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 1) Separate the target again (in case train_df changed shape)
y = train_df['TARGET']

# Drop TARGET and SK_ID_CURR to form our feature set
train_features_updated = train_df.drop(['TARGET', 'SK_ID_CURR'], axis=1, errors='ignore')
test_features_updated  = test_df.drop(['SK_ID_CURR'], axis=1, errors='ignore')

# Identify common columns
common_cols_updated = list(set(train_features_updated.columns) & set(test_features_updated.columns))

X_updated = train_features_updated[common_cols_updated]
X_test_updated = test_features_updated[common_cols_updated]

# 2) Label encode any new object columns
for col in X_updated.select_dtypes('object').columns:
    combined_data = pd.concat([X_updated[col], X_test_updated[col]], axis=0)
    lb = LabelEncoder()
    lb.fit(list(combined_data.astype(str).values))
    X_updated[col] = lb.transform(list(X_updated[col].astype(str).values))
    X_test_updated[col] = lb.transform(list(X_test_updated[col].astype(str).values))

print("New X shape:", X_updated.shape)
print("New X_test shape:", X_test_updated.shape)

# 3) Cross-validation setup
n_folds = 5
folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
auc_scores_updated = []

# 4) LightGBM model with cross-validation
import lightgbm as lgb

for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(X_updated, y)):
    X_train, X_valid = X_updated.iloc[train_idx], X_updated.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
        'seed': 42
    }

    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(0)
        ]
    )
    
    valid_preds = model.predict(X_valid, num_iteration=model.best_iteration)
    fold_auc = roc_auc_score(y_valid, valid_preds)
    auc_scores_updated.append(fold_auc)
    print(f"Fold {fold_idx+1} AUC: {fold_auc:.4f}")

print(f"\nUpdated Mean AUC across {n_folds} folds: {np.mean(auc_scores_updated):.4f} ± {np.std(auc_scores_updated):.4f}")


from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import lightgbm as lgb

# Create a LightGBM classifier (sklearn API)
lgb_estimator = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=500,  # we can tune this as well
    random_state=42
)

# Define a parameter grid for random search
param_dist = {
    'learning_rate': [0.01, 0.02, 0.05, 0.1],
    'num_leaves': [15, 31, 63, 127],
    'max_depth': [-1, 5, 7, 10],
    'min_child_samples': [20, 50, 100],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
}

# We'll use 3-fold stratified CV for the search
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(
    estimator=lgb_estimator,
    param_distributions=param_dist,
    n_iter=20,        # number of parameter sets to try
    scoring='roc_auc',
    cv=kfold,
    verbose=2,        # real-time progress logs
    random_state=42,
    n_jobs=-1         # use all available CPU cores
)

# Fit on the expanded feature set (X_updated, y) to tune parameters
random_search.fit(X_updated, y)

print("\nBest Hyperparameters:", random_search.best_params_)
print("Best CV AUC: {:.4f}".format(random_search.best_score_))

# The best model is automatically refit on the entire data of the best fold
best_lgb_model = random_search.best_estimator_


# Step 15 Python Code: Retrain with best hyperparams & submit

best_params = {
    'subsample': 0.8,
    'num_leaves': 63,
    'min_child_samples': 100,
    'max_depth': 5,
    'learning_rate': 0.05,
    'colsample_bytree': 1.0
}

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

n_folds = 5
folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
auc_scores_final_cv = []

for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(X_updated, y)):
    X_train, X_valid = X_updated.iloc[train_idx], X_updated.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbose': -1,
        'seed': 42,
        **best_params
    }
    
    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=2000,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(0)
        ]
    )
    
    valid_preds = model.predict(X_valid, num_iteration=model.best_iteration)
    fold_auc = roc_auc_score(y_valid, valid_preds)
    auc_scores_final_cv.append(fold_auc)
    print(f"Fold {fold_idx+1} AUC: {fold_auc:.4f}")

print(f"\nMean AUC with best hyperparams (5-fold): {np.mean(auc_scores_final_cv):.4f} ± {np.std(auc_scores_final_cv):.4f}")

# Train final model on entire data
final_model = lgb.LGBMClassifier(
    objective='binary',
    random_state=42,
    n_estimators=2000,  # large enough, early_stopping might not apply here
    **best_params
)
final_model.fit(X_updated, y)

# Predict on test set
test_preds_best = final_model.predict_proba(X_test_updated)[:, 1]

# Build submission
submission_best = pd.DataFrame({
    'SK_ID_CURR': test_df['SK_ID_CURR'],
    'TARGET': test_preds_best
})

submission_best.to_csv('submission_best.csv', index=False)
print("\nFinal submission file 'submission_best.csv' created.")
print(submission_best.head(10))


# Step 16 Python code: Feature Importance & SHAP Analysis

!pip install shap --quiet
import shap

# 1) Train a final model (best hyperparams) on the entire training set
import lightgbm as lgb

best_params_shap = {
    'subsample': 0.8,
    'num_leaves': 63,
    'min_child_samples': 100,
    'max_depth': 5,
    'learning_rate': 0.05,
    'colsample_bytree': 1.0,
    'n_estimators': 500,
    'objective': 'binary',
    'random_state': 42
}

final_model_shap = lgb.LGBMClassifier(**best_params_shap)
final_model_shap.fit(X_updated, y)

# 2) Basic LightGBM feature importance
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
lgb.plot_importance(final_model_shap, max_num_features=20, importance_type='gain')
plt.title("LightGBM Feature Importance (Top 20) - By Gain")
plt.show()

# 3) SHAP summary plot
#    We'll compute SHAP values on a sample of data to reduce runtime
sample_size = 3000  # Adjust as you like, but large samples can be slow
X_sample = X_updated.sample(sample_size, random_state=42)
explainer = shap.TreeExplainer(final_model_shap)
shap_values = explainer.shap_values(X_sample)

# SHAP summary plot
shap.summary_plot(shap_values, X_sample, plot_type='bar', max_display=20)

# Alternatively, for a detailed summary dot plot:
shap.summary_plot(shap_values, X_sample, max_display=20)


# Step 17 Python code: Simple Feature Pruning Approach

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 1) Let's collect feature importances from the final_model_shap we trained in Step 16
#    Or we can train a fresh model with the best hyperparams again, then get importances.

feature_importances = pd.DataFrame({
    'feature': X_updated.columns,
    'importance': final_model_shap.feature_importances_
})

# Sort features by importance descending
feature_importances.sort_values(by='importance', ascending=False, inplace=True)
feature_importances.reset_index(drop=True, inplace=True)

# Let's choose top 100 as an example
top_k = 100
top_features = feature_importances['feature'][:top_k].tolist()

print(f"Selecting top {top_k} features out of {X_updated.shape[1]} total.")

# 2) Filter X_updated and X_test_updated to only include those top features
X_pruned = X_updated[top_features]
X_test_pruned = X_test_updated[top_features]

print("X_pruned shape:", X_pruned.shape, "X_test_pruned shape:", X_test_pruned.shape)

# 3) Retrain with 5-fold CV to see if performance changes
best_params_pruning = {
    'objective': 'binary',
    'metric': 'auc',
    'verbose': -1,
    'seed': 42,
    'subsample': 0.8,
    'num_leaves': 63,
    'min_child_samples': 100,
    'max_depth': 5,
    'learning_rate': 0.05,
    'colsample_bytree': 1.0
}
n_folds = 5
folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
auc_scores_pruned = []

for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(X_pruned, y)):
    X_train, X_valid = X_pruned.iloc[train_idx], X_pruned.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)
    
    model_pruned = lgb.train(
        params=best_params_pruning,
        train_set=dtrain,
        num_boost_round=2000,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(0)
        ]
    )
    valid_preds = model_pruned.predict(X_valid, num_iteration=model_pruned.best_iteration)
    fold_auc = roc_auc_score(y_valid, valid_preds)
    auc_scores_pruned.append(fold_auc)
    print(f"Fold {fold_idx+1} AUC (pruned): {fold_auc:.4f}")

print(f"\nMean AUC with pruned features (5-fold): {np.mean(auc_scores_pruned):.4f} ± {np.std(auc_scores_pruned):.4f}")

# If AUC is similar or better, we can proceed to produce a new submission with these pruned features.
# That might reduce model complexity and possibly improve inference speed.


import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# We'll reuse the pruned feature sets and labels from Step 17 as an example
# X_pruned, X_test_pruned, y

n_folds = 5
folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Arrays to hold out-of-fold predictions
oof_lgb = np.zeros(len(X_pruned))
oof_xgb = np.zeros(len(X_pruned))

# Arrays to hold test predictions (averaged by folds)
preds_lgb = np.zeros(len(X_test_pruned))
preds_xgb = np.zeros(len(X_test_pruned))

# --- First-Level Model 1: LightGBM ---
params_lgb = {
    'objective': 'binary',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': 5,
    'min_child_samples': 100,
    'subsample': 0.8,
    'colsample_bytree': 1.0,
    'verbose': -1,
    'seed': 42
}

for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(X_pruned, y)):
    X_train, X_valid = X_pruned.iloc[train_idx], X_pruned.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    dtrain_lgb = lgb.Dataset(X_train, label=y_train)
    dvalid_lgb = lgb.Dataset(X_valid, label=y_valid)
    
    model_lgb = lgb.train(
        params=params_lgb,
        train_set=dtrain_lgb,
        num_boost_round=2000,
        valid_sets=[dvalid_lgb],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(0)
        ]
    )
    
    # OOF predictions for the validation fold
    oof_lgb[valid_idx] = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
    # Predictions on test data
    preds_lgb += model_lgb.predict(X_test_pruned, num_iteration=model_lgb.best_iteration) / n_folds

# --- First-Level Model 2: XGBoost ---
params_xgb = {
    'objective': 'binary:logistic',
    'learning_rate': 0.05,
    'max_depth': 5,
    'min_child_weight': 10,
    'subsample': 0.8,
    'colsample_bytree': 1.0,
    'eval_metric': 'auc',
    'seed': 42
}

for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(X_pruned, y)):
    X_train, X_valid = X_pruned.iloc[train_idx], X_pruned.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    dtrain_x = xgb.DMatrix(X_train, label=y_train)
    dvalid_x = xgb.DMatrix(X_valid, label=y_valid)
    
    watchlist = [(dtrain_x, 'train'), (dvalid_x, 'eval')]
    
    model_xgb = xgb.train(
        params=params_xgb,
        dtrain=dtrain_x,
        num_boost_round=2000,
        evals=watchlist,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # The attribute best_ntree_limit may not exist, so we use iteration_range or best_iteration
    best_iter = model_xgb.best_iteration  # best iteration index
    oof_xgb[valid_idx] = model_xgb.predict(xgb.DMatrix(X_valid), iteration_range=(0, best_iter+1))
    preds_xgb += model_xgb.predict(xgb.DMatrix(X_test_pruned), iteration_range=(0, best_iter+1)) / n_folds

# --- Build the stack (second-level input) ---
stack_train = np.vstack([oof_lgb, oof_xgb]).T  # shape: (num_samples, 2)
stack_test = np.vstack([preds_lgb, preds_xgb]).T  # shape: (num_test_samples, 2)

# --- Train a meta-model (Logistic Regression) ---
lr_meta = LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000)
lr_meta.fit(stack_train, y)

# Evaluate on entire training set (stack features)
oof_meta = lr_meta.predict_proba(stack_train)[:, 1]
auc_meta = roc_auc_score(y, oof_meta)
print(f"Meta-model AUC on stacked features: {auc_meta:.4f}")

# --- Generate final predictions for submission ---
test_preds_ensemble = lr_meta.predict_proba(stack_test)[:, 1]

submission_ensemble = pd.DataFrame({
    'SK_ID_CURR': test_df['SK_ID_CURR'],
    'TARGET': test_preds_ensemble
})
submission_ensemble.to_csv('submission_ensemble.csv', index=False)
print("\nEnsemble submission file 'submission_ensemble.csv' created!")
print(submission_ensemble.head(10))


# Step 19 Python Code: Optuna-based Bayesian Optimization for LightGBM

!pip install optuna --quiet
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# We'll reuse X_pruned, y from prior steps for demonstration,
# but you can also try X_updated for more features.

def objective(trial):
    # Define the hyperparameter search space
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbose': -1,
        'seed': 42,
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int("num_leaves", 15, 127, step=1),
        'max_depth': trial.suggest_int("max_depth", -1, 10),
        'min_child_samples': trial.suggest_int("min_child_samples", 20, 200, step=10),
        'subsample': trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
        'min_split_gain': trial.suggest_float("min_split_gain", 0.0, 0.5, step=0.05),
        'reg_alpha': trial.suggest_float("reg_alpha", 0.0, 5.0, step=0.1),
        'reg_lambda': trial.suggest_float("reg_lambda", 0.0, 5.0, step=0.1),
    }
    
    # 5-fold Stratified CV for AUC
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    for train_idx, valid_idx in folds.split(X_pruned, y):
        X_train, X_valid = X_pruned.iloc[train_idx], X_pruned.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid)

        # Train
        model = lgb.train(
            params=param,
            train_set=dtrain,
            num_boost_round=3000,
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(0)
            ]
        )

        preds = model.predict(X_valid, num_iteration=model.best_iteration)
        fold_auc = roc_auc_score(y_valid, preds)
        auc_scores.append(fold_auc)
    
    return np.mean(auc_scores)

# Create an Optuna study
study = optuna.create_study(direction="maximize")
# Run optimization for a specified number of trials
study.optimize(objective, n_trials=20, show_progress_bar=True)

# Best parameters found
print("\nBest trial:")
best_trial = study.best_trial
print(f"AUC: {best_trial.value:.4f}")
print("Best hyperparameters:", best_trial.params)

# If you want, you can retrain on full data with the best params:
best_params_optuna = best_trial.params
best_params_optuna.update({
    'objective': 'binary',
    'metric': 'auc',
    'verbose': -1,
    'seed': 42
})
final_model_optuna = lgb.LGBMClassifier(**best_params_optuna, n_estimators=1000)
final_model_optuna.fit(X_pruned, y)

# At this point, you can do another CV or create a new submission with final_model_optuna.

