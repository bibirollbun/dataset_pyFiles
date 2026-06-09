import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
print('train shape:', train.shape)
print('test shape:', test.shape)
train.head()


print(train.info())
print('\nTarget distribution:')
print(train['y'].value_counts(normalize=True))


train.describe()


# Plot target balance
plt.figure(figsize=(6,4))
sns.countplot(x='y', data=train)
plt.title('Target distribution')
plt.show()


# Helper functions
def summarize(df):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print('Numeric columns:', numeric)
    print('Categorical columns:', cat)
    return numeric, cat

num_cols, cat_cols = summarize(train.drop(columns=['id', 'y'], errors='ignore'))



# Correlation heatmap for numeric features
plt.figure(figsize=(10,8))
sns.heatmap(train[num_cols].corr(), annot=False, cmap='coolwarm', center=0)
plt.title('Numeric features correlation')
plt.show()


#  Distribution plots for top numeric features
for c in (num_cols[:5]):
    plt.figure(figsize=(6,3))
    sns.kdeplot(train.loc[train['y']==0, c], label='y=0', fill=True)
    sns.kdeplot(train.loc[train['y']==1, c], label='y=1', fill=True)
    plt.title(f'Distribution of {c} by target')
    plt.legend()
    plt.tight_layout()
    plt.show()


# For categorical variables show top categories vs target
for c in cat_cols:
    vc = train[c].value_counts().index[:8]
    plt.figure(figsize=(8,3))
    sns.barplot(x=vc, y=train[train[c].isin(vc)].groupby(c)['y'].mean().loc[vc].values)
    plt.title(f'Average target by top categories of {c}')
    plt.xticks(rotation=45)
    plt.show()


def prepare_features(train_df, test_df, target_col='y'):
    df_all = pd.concat([train_df.drop(columns=[target_col]) if target_col in train_df else train_df,
                        test_df], axis=0, ignore_index=True)
    # Keep ids
    ids = df_all['id'] if 'id' in df_all else pd.Series(np.arange(len(df_all)))

    # Identify types
    num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in num_cols:
        num_cols.remove('id')
    cat_cols = df_all.select_dtypes(include=['object', 'category']).columns.tolist()

    # Impute numeric with median
    num_imputer = SimpleImputer(strategy='median')
    df_all[num_cols] = num_imputer.fit_transform(df_all[num_cols])

    # Frequency encode categorical
    for c in cat_cols:
        freq = df_all[c].value_counts(dropna=False)
        df_all[c + '_freq_enc'] = df_all[c].map(freq)

    # Example interaction: count of positive numeric features > median
    for c in num_cols[:3]:
        df_all[c + '_above_median'] = (df_all[c] > df_all[c].median()).astype(int)

    # Drop original categorical to keep feature count reasonable (but keep freq encoding)
    df_all = df_all.drop(columns=cat_cols)

    X_all = df_all.drop(columns=['id']) if 'id' in df_all else df_all
    return X_all, ids

X_all, all_ids = prepare_features(train, test)
print('Prepared feature shape:', X_all.shape)



# Split back
X_train = X_all.iloc[:len(train), :].reset_index(drop=True)
y_train = train['y'].reset_index(drop=True)
X_test = X_all.iloc[len(train):, :].reset_index(drop=True)


NFOLDS = 5
kf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_STATE)

# Containers
oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))
models_oof = []
feature_importances = pd.DataFrame()
feature_importances['feature'] = X_train.columns


params_lgb = {
    'objective': 'binary',            # change to 'regression' for regression tasks
    'metric': 'auc',                   # or 'binary_logloss', etc.
    'boosting_type': 'gbdt',           # Gradient Boosted Decision Trees
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,                   # unlimited depth
    'feature_fraction': 0.8,           # subsample features
    'bagging_fraction': 0.8,           # subsample data
    'bagging_freq': 5,                 # frequency of bagging
    'verbosity': -1,
    'random_state': 42
}



from lightgbm import early_stopping, log_evaluation

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f'Fold {fold+1}')
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    train_set = lgb.Dataset(X_tr, label=y_tr)
    valid_set = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        params_lgb,
        train_set,
        num_boost_round=2000,
        valid_sets=[train_set, valid_set],
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(100)  # print every 10 iterations
        ]
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / NFOLDS
    feature_importances[f'fold_{fold+1}'] = model.feature_importance(importance_type='gain')
    models_oof.append(model)

print('OOF ROC AUC:', roc_auc_score(y_train, oof_preds))



feature_importances['importance_mean'] = feature_importances.filter(like='fold_').mean(axis=1)
fi = feature_importances.sort_values(by='importance_mean', ascending=False).head(30)
plt.figure(figsize=(8,6))
sns.barplot(x='importance_mean', y='feature', data=fi)
plt.title('Top 30 feature importances (LightGBM)')
plt.tight_layout()
plt.show()


# XGBoost baseline
params_xgb = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.03,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': RANDOM_STATE,
    'nthread': -1
}

xgb_oof = np.zeros(len(X_train))
xgb_test = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    bst = xgb.train(
        params_xgb,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, 'train'), (dval, 'valid')],
        early_stopping_rounds=100,
        verbose_eval=200
    )
    
    pred_val = bst.predict(dval, iteration_range=(0, bst.best_iteration))
    pred_test = bst.predict(dtest, iteration_range=(0, bst.best_iteration))
    
    xgb_oof[val_idx] = pred_val
    xgb_test += pred_test / NFOLDS

print('XGB OOF ROC AUC:', roc_auc_score(y_train, xgb_oof))



# CatBoost baseline (categorical cols were removed; if you keep original cats, pass cat_features)
cat_oof = np.zeros(len(X_train))
cat_test = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    model = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6,
                               eval_metric='AUC', random_seed=RANDOM_STATE, verbose=200,
                               early_stopping_rounds=100)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    cat_oof[val_idx] = model.predict_proba(X_val)[:,1]
    cat_test += model.predict_proba(X_test)[:,1] / NFOLDS

print('CatBoost OOF ROC AUC:', roc_auc_score(y_train, cat_oof))



# Use out-of-fold predictions from LightGBM, XGBoost, CatBoost as features for a Logistic Regression meta-model.


stack_X = np.vstack([oof_preds, xgb_oof, cat_oof]).T
stack_test = np.vstack([test_preds, xgb_test, cat_test]).T

meta_oof = np.zeros(len(stack_X))
meta_test = np.zeros(stack_test.shape[0])

for fold, (tr_idx, val_idx) in enumerate(kf.split(stack_X, y_train)):
    X_tr, X_val = stack_X[tr_idx], stack_X[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    meta = LogisticRegression()
    meta.fit(X_tr, y_tr)
    meta_oof[val_idx] = meta.predict_proba(X_val)[:,1]
    meta_test += meta.predict_proba(stack_test)[:,1] / NFOLDS

print('Stacked OOF ROC AUC:', roc_auc_score(y_train, meta_oof))



# Quick calibration demonstration using Platt scaling on the stacked predictions
calibrator = CalibratedClassifierCV(base_estimator=LogisticRegression(), method='sigmoid', cv=5)
calibrator.fit(stack_X, y_train)
calib_test = calibrator.predict_proba(stack_test)[:,1]
print('Calibrated OOF (approx):', roc_auc_score(y_train, calibrator.predict_proba(stack_X)[:,1]))



import shutil

# Remove the catboost_info folder if it exists
shutil.rmtree('catboost_info', ignore_errors=True)



# Save submission
submission_path = '/kaggle/working/submission.csv'
submission = pd.DataFrame({'id': test['id'], 'y': test_preds})
submission.to_csv(submission_path, index=False)
print(f'Submission saved to {submission_path}')
print(submission.head())

