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
import gc, pickle, re
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, matthews_corrcoef

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

del train_transaction, train_identity, test_transaction, test_identity
gc.collect()

print("Train shape:", train.shape)
print("Test shape:", test.shape)

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

# Clean Device Info and Emails
def clean_device(x):
    if isinstance(x, str):
        x = x.lower()
        x = re.sub('[^a-z0-9]', '', x)
    return x

for col in ['DeviceType', 'DeviceInfo']:
    if col in train.columns:
        train[col] = train[col].fillna('unknown').apply(clean_device)
        test[col]  = test[col].fillna('unknown').apply(clean_device)

for col in ['P_emaildomain', 'R_emaildomain']:
    if col in train.columns:
        train[col] = train[col].fillna('unknown').str.lower()
        test[col]  = test[col].fillna('unknown').str.lower()

train['DT_M'] = ((train['TransactionDT'] / (3600*24*30)) % 12).astype(int)
test['DT_M']  = ((test['TransactionDT'] / (3600*24*30)) % 12).astype(int)
train['hour'] = (train['TransactionDT'] / 3600) % 24
test['hour']  = (test['TransactionDT'] / 3600) % 24
train['weekday'] = (train['TransactionDT'] / (3600*24)) % 7
test['weekday']  = (test['TransactionDT'] / (3600*24)) % 7
train['email_match'] = (train['P_emaildomain'] == train['R_emaildomain']).astype(int)
test['email_match']  = (test['P_emaildomain'] == test['R_emaildomain']).astype(int)

# ==============================
# 4. Prepare Data
# ==============================
print("Preparing datasets...")
TARGET = 'isFraud'
all_features = [col for col in train.columns if col not in ['TransactionID', 'TransactionDT', TARGET]]
features = [col for col in all_features if col in test.columns]

X_train = train[features].replace([np.inf, -np.inf], np.nan).fillna(-999)
y_train = train[TARGET]
X_test  = test[features].replace([np.inf, -np.inf], np.nan).fillna(-999)

print(f"✅ Features being used: {len(features)}")

for col in X_train.columns:
    if X_train[col].dtype == 'object':
        le = LabelEncoder()
        full_col = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
        le.fit(full_col)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col]  = le.transform(X_test[col].astype(str))

# ==============================
# 5. Set Fixed Best Parameters
# ==============================
best_params = {
    'max_depth': 12,
    'learning_rate': 0.005781337422346563,
    'subsample': 0.9068477789362549,
    'colsample_bytree': 0.7119270238791713,
    'reg_alpha': 1.2339011119623482,
    'reg_lambda': 7.463546263502217,
    'gamma': 1.4439322514815511,
    'min_child_weight': 2,
    'missing': -1,
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 2025,
    'n_jobs': 2
}

# ==============================
# 6. Train XGBoost with GroupKFold
# ==============================
print("\nTraining 6-Fold model with fixed params...")

oof = np.zeros(len(X_train))
preds = np.zeros(len(X_test))

skf = GroupKFold(n_splits=6)

for i, (idxT, idxV) in enumerate(skf.split(X_train, y_train, groups=train['DT_M'])):
    print(f"Fold {i}")
    clf = xgb.XGBClassifier(
        **best_params,
        n_estimators=8000,
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

print('#'*30)
print('✅ Full OOF CV AUC:', roc_auc_score(y_train, oof))

# ==============================
# 7. Threshold Optimization (MCC)
# ==============================
print("\nOptimizing threshold...")

best_thresh = 0.5
best_mcc = 0
for thresh in np.arange(0.1, 0.9, 0.01):
    mcc = matthews_corrcoef(y_train, (oof > thresh).astype(int))
    if mcc > best_mcc:
        best_thresh = thresh
        best_mcc = mcc

print(f"✅ Best threshold: {best_thresh:.2f} | MCC: {best_mcc:.5f}")

final_preds = (preds > best_thresh).astype(int)

# ==============================
# 8. Save Outputs
# ==============================
print("\nSaving outputs...")

submission = pd.DataFrame({'TransactionID': test['TransactionID'], 'isFraud': preds})
submission.to_csv('submission_raw.csv', index=False)

submission_post = pd.DataFrame({'TransactionID': test['TransactionID'], 'isFraud': final_preds})
submission_post.to_csv('submission_postprocessed.csv', index=False)

oof_preds_df = pd.DataFrame({'isFraud_OOF': oof})
oof_preds_df.to_csv('oof_preds.csv', index=False)

# ==============================
# 9. Final Full Refit
# ==============================
print("\nRefitting final model on full training data...")

final_model = xgb.XGBClassifier(
    **best_params,
    n_estimators=8000
)

final_model.fit(X_train, y_train)

final_preds_full = final_model.predict_proba(X_test)[:,1]
final_preds_full_post = (final_preds_full > best_thresh).astype(int)

submission_full = pd.DataFrame({'TransactionID': test['TransactionID'], 'isFraud': final_preds_full})
submission_full.to_csv('submission_raw_full_refit.csv', index=False)

submission_post_full = pd.DataFrame({'TransactionID': test['TransactionID'], 'isFraud': final_preds_full_post})
submission_post_full.to_csv('submission_postprocessed_full_refit.csv', index=False)

with open('xgb_magic_model_full.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print("✅ All full-data outputs saved successfully.")

