# Standard imports and settings
import os, gc, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb


# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
print('Train shape:', train.shape)
print('Test shape:', test.shape)
# show first rows
train.head()


# Inspect columns to identify target (handle case where it might be named differently)
print('Train columns:', train.columns.tolist())
print('Test columns:', test.columns.tolist())
print('\nTrain dtypes:')
print(train.dtypes)
# Find target column (typically last column or named 'target', 'Target', 'diabetes', etc)
possible_targets = [c for c in train.columns if c not in test.columns]
target_col_name = possible_targets[0] if possible_targets else 'target'
print(f'\nInferred target column: {target_col_name}')
if target_col_name in train.columns:
    print(f'Target distribution:\n{train[target_col_name].value_counts(normalize=True)}')
print('\nMissing values per column:')
display(train.isna().sum().sort_values(ascending=False).head(30))
print('\nNumeric describe:')
display(train.describe().T)


# Prepare combined dataframe - use detected target column name
train = train.copy()
test = test.copy()
train['is_train'] = 1
test['is_train'] = 0
test[target_col_name] = -1
df = pd.concat([train, test], ignore_index=True)
print('Combined shape:', df.shape)


# Identify numeric and categorical columns - use detected target column
target_col = target_col_name  # Use the detected target column name
id_col = 'patient_id' if 'patient_id' in df.columns else df.columns[0]
cols = [c for c in df.columns if c not in ['is_train', target_col]]
nunique = df[cols].nunique()
categorical = list(nunique[nunique <= 50].index)
numeric = [c for c in cols if c not in categorical]
print('ID col:', id_col)
print('Target col:', target_col)
print('Num numeric:', len(numeric), 'Num categorical:', len(categorical))


# Simple imputations and encodings
for c in numeric:
    if df[c].isna().any():
        df[c + '_n_missing'] = df[c].isna().astype(int)
        df[c] = df[c].fillna(df[c].median())
for c in categorical:
    df[c] = df[c].fillna('NA').astype(str)
    if df[c].nunique() <= 200:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])
# Row aggregates
if len(numeric) > 0:
    df['row_mean'] = df[numeric].mean(axis=1)
    df['row_std'] = df[numeric].std(axis=1).fillna(0)
    df['row_sum'] = df[numeric].sum(axis=1)
# Pairwise ratios for top numeric features by variance
num_top = sorted(numeric, key=lambda x: df[x].var() if x in numeric else 0, reverse=True)[:6]
for i in range(len(num_top)):
    for j in range(i+1, len(num_top)):
        a=num_top[i]; b=num_top[j]
        df[f'{a}_div_{b}'] = df[a] / (df[b] + 1e-6)
print('Feature engineering complete. New shape:', df.shape)


# Split back - use target_col variable
train_df = df[df['is_train']==1].drop(['is_train'], axis=1).reset_index(drop=True)
test_df = df[df['is_train']==0].drop(['is_train', target_col], axis=1).reset_index(drop=True)
X = train_df.drop([target_col], axis=1)
y = train_df[target_col].astype(int)
features = [c for c in X.columns if c != id_col]
X[features] = X[features].astype(float)
test_df[features] = test_df[features].astype(float)
print('Train X shape:', X.shape, 'y shape:', y.shape)
gc.collect()


def train_lgbm_oof(X, y, X_test, features, params=None, n_splits=5, random_state=42):
    if params is None:
        params = {'objective':'binary','metric':'auc','verbosity':-1,'boosting_type':'gbdt','learning_rate':0.05,'num_leaves':31,'seed':random_state}
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold,(tr_idx,val_idx) in enumerate(folds.split(X,y)):
        X_tr, X_val = X.iloc[tr_idx][features], X.iloc[val_idx][features]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val)
        model = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dval], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
        oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        preds += model.predict(X_test[features], num_iteration=model.best_iteration)/n_splits
    print('LGB OOF AUC:', roc_auc_score(y,oof))
    return oof, preds


# Train LightGBM and CatBoost baselines
lgb_params = {'objective':'binary','metric':'auc','verbosity':-1,'boosting_type':'gbdt','learning_rate':0.03,'num_leaves':128,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'seed':42}
oof_lgb, pred_lgb = train_lgbm_oof(X,y,test_df,features,params=lgb_params,n_splits=5)
train_df['oof_lgb'] = oof_lgb
test_df['pred_lgb'] = pred_lgb

def train_cat_oof(X,y,X_test,features,n_splits=5,random_state=42):
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold,(tr_idx,val_idx) in enumerate(folds.split(X,y)):
        X_tr, X_val = X.iloc[tr_idx][features], X.iloc[val_idx][features]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model = CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=6, eval_metric='AUC', random_seed=random_state, verbose=200, early_stopping_rounds=100)
        model.fit(X_tr,y_tr,eval_set=(X_val,y_val))
        oof[val_idx] = model.predict_proba(X_val)[:,1]
        preds += model.predict_proba(X_test[features])[:,1]/n_splits
    print('CatBoost OOF AUC:', roc_auc_score(y,oof))
    return oof,preds

oof_cat, pred_cat = train_cat_oof(X,y,test_df,features,n_splits=5)
train_df['oof_cat'] = oof_cat
test_df['pred_cat'] = pred_cat


meta_features = ['oof_lgb','oof_cat']
X_meta = train_df[meta_features].copy()
X_meta_test = test_df[['pred_lgb','pred_cat']].rename(columns={'pred_lgb':'oof_lgb','pred_cat':'oof_cat'}).copy()
meta = LogisticRegression(max_iter=1000)
meta.fit(X_meta,y)
meta_oof = meta.predict_proba(X_meta)[:,1]
meta_test = meta.predict_proba(X_meta_test)[:,1]
print('Meta OOF AUC:', roc_auc_score(y, meta_oof))
# Calibrate 
cal = CalibratedClassifierCV(estimator=LogisticRegression(max_iter=1000), cv=5, method='isotonic')
cal.fit(X_meta,y)
meta_cal_oof = cal.predict_proba(X_meta)[:,1]
meta_cal_test = cal.predict_proba(X_meta_test)[:,1]
print('Calibrated Meta OOF AUC:', roc_auc_score(y, meta_cal_oof))


# Final blended prediction and save submission
w_lgb, w_cat, w_meta = 0.4, 0.4, 0.2

test_pred_final = (
    w_lgb * test_df['pred_lgb']
    + w_cat * test_df['pred_cat']
    + w_meta * meta_cal_test
)

test_pred_final = np.clip(test_pred_final, 0, 1)

submission = pd.DataFrame({
    id_col: test_df[id_col].values,
    'target': test_pred_final
})

submission.to_csv('submission_blend.csv', index=False)
print('Saved submission_blend.csv')

submission.head()


