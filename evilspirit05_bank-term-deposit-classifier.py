import pandas as pd
import numpy as np
import warnings, gc, os
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from IPython.display import display
pd.set_option('display.max_columns', None)
!rm -rf /kaggle/working/*


train = pd.read_csv("/kaggle/input/binary-classification-with-a-bank-database/train.csv")
test  = pd.read_csv("/kaggle/input/binary-classification-with-a-bank-database/test.csv")
train.drop(columns=["id"],axis=1,inplace=True)

print("Check Out Train DaTA Null Values: ",train.isnull().sum())
print(f"Train Data Shape: {train.shape}")
print(f"Train Data INFO: {train.info()}")

def bank_feature_engineering(df):
    df = df.copy()
  
    df['no_previous_contact'] = (df['pdays'].isna()).astype(int)
    df['ever_contacted']      = 1 - df['no_previous_contact']
    df['duration_hour'] = df['duration'] // 60
    df['duration_min']  = df['duration'] % 60
    df['is_long_call']  = (df['duration'] > 500).astype(int)
    df['is_very_long_call'] = (df['duration'] > 900).astype(int)
    df['is_young'] = (df['age'] <= 30).astype(int)
    df['is_senior'] = (df['age'] >= 60).astype(int)
    df['age_x_balance'] = df['age'] * df['balance'].clip(lower=0)
    df['has_balance'] = (df['balance'] > 0).astype(int)
    df['high_balance'] = (df['balance'] > 3000).astype(int)
    df['negative_balance'] = (df['balance'] < 0).astype(int)
    df['was_contacted_before'] = (df['previous'] > 0).astype(int)
    df['many_campaigns'] = (df['campaign'] > 4).astype(int)
    df['previous_per_campaign'] = df['previous'] / (df['campaign'] + 1)
    month_order = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                   'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    df['month_num'] = df['month'].map(month_order)
    df['is_quarter_end'] = df['month'].isin(['mar', 'jun', 'sep', 'dec']).astype(int)
    df['job_education'] = df['job'] + "_" + df['education']
    df['job_marital']   = df['job'] + "_" + df['marital']
    df['balance_bin']   = pd.qcut(df['balance'], q=10, duplicates='drop').astype(str)
    df['poutcome_success_before'] = (df['poutcome'] == 'success').astype(int)
    df['poutcome_failure_before'] = (df['poutcome'] == 'failure').astype(int)
    df['contact_month'] = df['contact'] + "_" + df['month']
    df['debt_burden'] = df['housing'].apply(lambda x: 1 if x=='yes' else 0) + df['loan'].apply(lambda x: 1 if x=='yes' else 0)
    df['call_in_best_month'] = df['month'].isin(['mar', 'sep', 'oct', 'dec']).astype(int)
    df['student_or_retired'] = df['job'].isin(['student', 'retired']).astype(int)
    return df


train = bank_feature_engineering(train)
test=bank_feature_engineering(test)

balance_bins = [-10000, 0, 500, 1000, 3000, 10000, 100000]  # adjust based on your dataset
balance_labels = ['neg', 'very_low', 'low', 'medium', 'high', 'very_high']
train['balance_bin'] = pd.cut(train['balance'], bins=balance_bins, labels=balance_labels, include_lowest=True)
train['balance_bin'] = train['balance_bin'].astype(str)

test['balance_bin'] = pd.cut(test['balance'], bins=balance_bins, labels=balance_labels, include_lowest=True)
# test['balance_bin'] = test['balance_bin'].astype(str)


categorical_cols = train.select_dtypes(include='object').columns.tolist()

if 'y' in categorical_cols:
    categorical_cols.remove('y')  


label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
  
    test[col] = le.transform(test[col])  
    
    label_encoders[col] = le


numeric_cols = ['age','balance','duration','campaign','previous','age_x_balance']
scaler = StandardScaler()
train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
test[numeric_cols] = scaler.transform(test[numeric_cols])
Id=test.id
test.drop(columns=["id"],axis=1,inplace=True)

print("Display Train Data:")
display(train.head())
print("#"*130)
print("\n")
print("Display Test Data:")
display(test.head())


features = [c for c in train.columns if c not in ['id', 'y']]

params_lgb = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 1e-3,
    'num_leaves': 128,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42,
    'max_depth': -1,
    'min_data_in_leaf': 100,
    'lambda_l1': 0.1,
    'lambda_l2': 0.5,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds_lgb = np.zeros(len(test))

for fold, (trn_idx, val_idx) in enumerate(skf.split(train, train['y'])):
    print(f"\nFold {fold+1}/10")
    X_trn, y_trn = train.iloc[trn_idx][features], train.iloc[trn_idx]['y']
    X_val, y_val = train.iloc[val_idx][features], train.iloc[val_idx]['y']
    
    dtrain = lgb.Dataset(X_trn, y_trn)
    dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)
    
    model = lgb.train(params_lgb,dtrain,num_boost_round=5000,valid_sets=[dtrain, dvalid],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    
    oof_preds[val_idx] = model.predict(X_val)
    test_preds_lgb += model.predict(test[features]) / skf.n_splits
    
    print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, oof_preds[val_idx]):.6f}")

print(f"\nOverall OOF AUC: {roc_auc_score(train['y'], oof_preds):.6f}")



params_cb = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'learning_rate': 1e-3,
    'iterations': 5000,
    'depth': 8,
    'l2_leaf_reg': 3,
    'random_strength': 0.8,
    'bagging_temperature': 0.7,
    'random_seed': 42,
    'task_type': 'GPU',
    'devices': '0',
    'early_stopping_rounds': 400,
    'verbose': 500
}

test_preds_cb = np.zeros(len(test))
oof_preds_cb = np.zeros(len(train))

for fold, (trn_idx, val_idx) in enumerate(skf.split(train, train['y'])):
    print(f"\nCAT Fold {fold+1}/10")
    
    X_trn = train.iloc[trn_idx][features].values
    X_val = train.iloc[val_idx][features].values
    y_trn = train.iloc[trn_idx]['y'].values
    y_val = train.iloc[val_idx]['y'].values
    
    model = cb.CatBoost(params_cb)
    model.fit(X_trn, y_trn, eval_set=(X_val, y_val), use_best_model=True, verbose=500)
    
    oof_preds_cb[val_idx] = model.predict(X_val, prediction_type='Probability')[:, 1]
    test_preds_cb += model.predict(test[features].values, prediction_type='Probability')[:, 1] / skf.n_splits

print(f"\nCatBoost OOF AUC: {roc_auc_score(train['y'], oof_preds_cb):.6f}")



best_auc = 0
best_w = None

for w in np.linspace(0, 1, 101):
    blend = w * oof_preds + (1-w) * oof_preds_cb
    auc = roc_auc_score(train['y'], blend)
    if auc > best_auc:
        best_auc = auc
        best_w = w

print("Best weight:", best_w, "AUC:", best_auc)



final_pred = 0.86 * test_preds_lgb + 0.14 * test_preds_cb
submission=pd.DataFrame({"id":Id,"y":final_pred})
submission.to_csv('submission.csv', index=False)
print("Submission saved!")
submission.head(5)




