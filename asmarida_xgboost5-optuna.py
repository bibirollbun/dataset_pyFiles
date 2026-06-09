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


# ==============================
# 1. Libraries
# ==============================
import pandas as pd, numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score
import gc
import pickle

# ==============================
# 2. Load Data
# ==============================
print("Loading data...")
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
train_identity    = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')
test_transaction  = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
test_identity     = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')

train = train_transaction.merge(train_identity, how='left', on='TransactionID')
test  = test_transaction.merge(test_identity, how='left', on='TransactionID')

print("Train shape:", train.shape)
print("Test shape:", test.shape)

del train_transaction, train_identity, test_transaction, test_identity
gc.collect()

# ==============================
# 3. Feature Engineering
# ==============================
print("Creating magic features...")
train['uid'] = train['card1'].astype(str) + '_' + train['card2'].astype(str)
test['uid']  = test['card1'].astype(str) + '_' + test['card2'].astype(str)

train['uid2'] = train['uid'] + '_' + train['card3'].astype(str) + '_' + train['card5'].astype(str)
test['uid2']  = test['uid'] + '_' + test['card3'].astype(str) + '_' + test['card5'].astype(str)

for df in [train, test]:
    df['TransactionAmt_to_mean_card1'] = df['TransactionAmt'] / df.groupby('card1')['TransactionAmt'].transform('mean')
    df['TransactionAmt_to_std_card1']  = df['TransactionAmt'] / df.groupby('card1')['TransactionAmt'].transform('std')
    df['TransactionAmt_to_mean_uid']   = df['TransactionAmt'] / df.groupby('uid')['TransactionAmt'].transform('mean')
    df['TransactionAmt_to_std_uid']    = df['TransactionAmt'] / df.groupby('uid')['TransactionAmt'].transform('std')
    df['card1_card2_ratio'] = df['card1'] / df['card2']

gc.collect()

train['DT_M'] = ((train['TransactionDT'] / (3600*24*30)) % 12).astype(int)
test['DT_M']  = ((test['TransactionDT'] / (3600*24*30)) % 12).astype(int)

# ==============================
# 4. Label Encoding
# ==============================
print("Label encoding...")
common_cols = [col for col in train.columns if col in test.columns]

for col in common_cols:
    if train[col].dtype == 'object' or test[col].dtype == 'object':
        le = LabelEncoder()
        combined_data = pd.concat([train[col].astype(str), test[col].astype(str)], axis=0)
        le.fit(combined_data)
        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))

gc.collect()

# ==============================
# 5. Prepare Data
# ==============================
print("Preparing final datasets...")
TARGET = 'isFraud'
all_features = [col for col in train.columns if col not in ['TransactionID', 'TransactionDT', 'DT_M', TARGET]]
features = [col for col in all_features if col in test.columns]

X_train = train[features].replace([np.inf, -np.inf], np.nan).fillna(-999)
y_train = train[TARGET]
X_test  = test[features].replace([np.inf, -np.inf], np.nan).fillna(-999)

print(f"âœ… Features being used: {len(features)}")

# ==============================
# 6. Optuna Hyperparameter Search
# ==============================
print("\nðŸ”Ž Starting Optuna hyperparameter search...")

def objective(trial):
    params = {
        'n_estimators': 3000,
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.02),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'missing': -1,
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 2025,
        'n_jobs': 2,
        'early_stopping_rounds': 100
    }
    
    clf = xgb.XGBClassifier(**params)
    
    skf = GroupKFold(n_splits=3)
    oof = np.zeros(len(X_train))
    
    for idxT, idxV in skf.split(X_train, y_train, groups=train['DT_M']):
        clf.fit(
            X_train.iloc[idxT], y_train.iloc[idxT],
            eval_set=[(X_train.iloc[idxV], y_train.iloc[idxV])],
            verbose=False
        )
        oof[idxV] = clf.predict_proba(X_train.iloc[idxV])[:,1]
    
    return roc_auc_score(y_train, oof)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("\nâœ… Optuna best params:")
print(study.best_params)

# ==============================
# 7. XGBoost with GroupKFold using Best Params
# ==============================
print("\nTraining XGBoost with GroupKFold using Optuna best params...")

oof = np.zeros(len(X_train))
preds = np.zeros(len(X_test))

skf = GroupKFold(n_splits=6)

for i, (idxT, idxV) in enumerate(skf.split(X_train, y_train, groups=train['DT_M'])):
    month = train.iloc[idxV]['DT_M'].iloc[0]
    print(f"Fold {i} withholding month {month}")

    clf = xgb.XGBClassifier(
        **study.best_params,
        n_estimators=8000,
        missing=-1,
        eval_metric='auc',
        tree_method='hist',
        device='cuda',
        random_state=2025,
        n_jobs=2,
        early_stopping_rounds=200
    )

    clf.fit(
        X_train.iloc[idxT], y_train.iloc[idxT],
        eval_set=[(X_train.iloc[idxV], y_train.iloc[idxV])],
        verbose=100
    )

    oof[idxV] = clf.predict_proba(X_train.iloc[idxV])[:,1]
    preds += clf.predict_proba(X_test)[:,1]/skf.n_splits

    del clf
    gc.collect()

print('#'*20)
print('âœ… XGB96 OOF CV Score =', roc_auc_score(y_train, oof))

# ==============================
# 8. Final Model Refit
# ==============================
print("\nRefitting final model on full training data...")

final_model = xgb.XGBClassifier(
    **study.best_params,
    n_estimators=8000,
    missing=-1,
    eval_metric='auc',
    tree_method='hist',
    device='cuda',
    random_state=2025,
    n_jobs=2
)

final_model.fit(X_train, y_train)

with open('xgb_magic_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print("âœ… Model saved as 'xgb_magic_model.pkl'")

# ==============================
# 9. Save Test Set
# ==============================
X_test.to_csv('X_test_engineered.csv', index=False)
print("âœ… Feature-engineered Test Set saved.")

# ==============================
# 10. Save Submission
# ==============================
submission_raw = pd.DataFrame({
    'TransactionID': test['TransactionID'],
    'isFraud': preds
})
submission_raw.to_csv('submission_raw.csv', index=False)
print("âœ… Raw probability submission saved.")

# ==============================
# 11. Post-processing threshold optimization
# ==============================
print("\nOptimizing threshold on OOF predictions...")

best_thresh = 0.5
best_f1 = 0
for thresh in np.arange(0.1, 0.9, 0.01):
    f1 = f1_score(y_train, (oof > thresh).astype(int))
    if f1 > best_f1:
        best_thresh = thresh
        best_f1 = f1

print(f"âœ… Best threshold = {best_thresh:.2f} with F1 = {best_f1:.5f}")

final_preds = (preds > best_thresh).astype(int)

submission_postprocessed = pd.DataFrame({
    'TransactionID': test['TransactionID'],
    'isFraud': final_preds
})
submission_postprocessed.to_csv('submission_postprocessed.csv', index=False)
print("âœ… Post-processed thresholded submission saved.")

# ==============================
# 12. Save OOF
# ==============================
oof_preds_df = pd.DataFrame({'isFraud_OOF': oof})
oof_preds_df.to_csv('oof_preds.csv', index=False)
print("âœ… OOF predictions saved as 'oof_preds.csv'")

