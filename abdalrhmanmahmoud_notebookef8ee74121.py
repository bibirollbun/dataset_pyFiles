import os
import gc
import random
from pathlib import Path


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


# plotting defaults
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)


DATA_DIR = '/kaggle/input/playground-series-s5e8/'
# For local/Colab you may need to change DATA_DIR to the unzipped folder path


train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))


print('train shape:', train.shape)
print('test shape: ', test.shape)
train.head()


# Basic info
print('\n--- INFO ---')
print(train.info())
print('\n--- Describe numeric ---')
print(train.describe().T)


print('\nTarget value counts:')
print(train['y'].value_counts(normalize=True))


# Plot target distribution
plt.figure()
sns.countplot(x='y', data=train)
plt.title('Target distribution (train)')
plt.show()


# Identify numeric and categorical features
id_col = 'id'
target_col = 'y'


all_features = [c for c in train.columns if c not in [id_col, target_col]]


numeric_feats = train[all_features].select_dtypes(include=[np.number]).columns.tolist()
cat_feats = [c for c in all_features if c not in numeric_feats]


print('\nNumeric features:', numeric_feats)
print('Categorical features:', cat_feats)


print('\n--- Missing counts (train) ---')
missing_train = train[all_features].isnull().sum()
print(missing_train[missing_train>0])


for c in cat_feats:
   top = train[c].value_counts(dropna=False).nlargest(6)
   print(f"\n{c} top values:\n", top)


train[numeric_feats].hist(bins=40, layout=(len(numeric_feats)//3+1, 3), figsize=(14, 12))
plt.tight_layout()
plt.show()


# Correlation heatmap on numeric features with target
plt.figure(figsize=(8,6))
num_plus_target = numeric_feats + [target_col]
corr = train[num_plus_target].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation (numeric features + target)')
plt.show()


for c in ['job', 'marital', 'education']:
   if c in cat_feats:
      plt.figure(figsize=(8,4))
      sns.countplot(x=c, hue=target_col, data=train, order=train[c].value_counts().index[:15])
      plt.title(f'{c} vs target')
      plt.xticks(rotation=45)
      plt.show()


# Copy datasets to avoid accidental modification
train_df = train.copy()
test_df = test.copy()


# Map y to 0/1 if not already
if train_df[target_col].dtype != 'int64' and train_df[target_col].dtype != 'int32':
   train_df[target_col] = train_df[target_col].astype(int)


# Clean boolean-like columns that may be strings like 'yes'/'no' or 'true'/'false'
bool_like = ['default', 'housing', 'loan']
for col in bool_like:
   if col in train_df.columns:
      train_df[col] = train_df[col].map({"yes":1, "no":0, "true":1, "false":0})
      test_df[col] = test_df[col].map({"yes":1, "no":0, "true":1, "false":0})


# month mapping to numeric order
month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
if 'month' in train_df.columns:
   train_df['month_num'] = train_df['month'].map(month_map)
   test_df['month_num'] = test_df['month'].map(month_map)
# create season feature
train_df['season'] = (train_df['month_num']%12 // 3).astype(object)
test_df['season'] = (test_df['month_num']%12 // 3).astype(object)
# add to categorical
if 'month' in cat_feats:
   cat_feats.append('month_num')
   cat_feats.append('season')


# Feature: interaction duration * campaign (example)
train_df['dur_x_campaign'] = train_df['duration'] * train_df['campaign']
test_df['dur_x_campaign'] = test_df['duration'] * test_df['campaign']


# Update numeric/categorical lists after new features
numeric_feats = [c for c in train_df[all_features + ['month_num','dur_x_campaign']] .select_dtypes(include=[np.number]).columns.tolist() if c in train_df.columns]
cat_feats = [c for c in all_features if c not in numeric_feats]


print('\nAfter engineering - numeric:', numeric_feats)
print('After engineering - cat:', cat_feats)


# Keep consistent columns order
FEATURES = [c for c in train_df.columns if c not in [id_col, target_col]]


from sklearn.base import clone

def run_cv_predict(clf, X, y, X_test, folds=5, stratify=True, model_name='model'):
    """Run StratifiedKFold CV (returns oof preds, test preds (mean), and cv score list)"""
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/{folds} - {model_name}")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # fit
        clf_ = clone(clf)
        clf_.fit(X_tr, y_tr)

        # predict
        val_pred = clf_.predict_proba(X_val)[:, 1]
        test_fold_pred = clf_.predict_proba(X_test)[:, 1]

        oof[val_idx] = val_pred
        test_preds += test_fold_pred / folds

        score = roc_auc_score(y_val, val_pred)
        scores.append(score)
        print(f"Fold {fold+1} AUC: {score:.5f}\n")

    print(f"CV mean AUC for {model_name}: {np.mean(scores):.5f} ± {np.std(scores):.5f}")
    return oof, test_preds, scores




X = train_df.drop(columns=[id_col, target_col]).copy()
y = train_df[target_col].copy()
X_test = test_df.drop(columns=[id_col]).copy()



X_cat = [c for c in X.columns if X[c].dtype == 'object' or c in ['season']]
X_num = [c for c in X.columns if c not in X_cat]

print('\nColumns split for preprocessing:')
print('Numeric:', X_num)
print('Categorical:', X_cat)


# ---- Strategy A: OneHot + scaling (for LogisticRegression) ----
# Use ColumnTransformer to build pipeline
ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
scaler = StandardScaler()

preproc_ohe = ColumnTransformer([
    ('num', scaler, X_num),
    ('cat', ohe_encoder, X_cat)
], remainder='drop')


# fit_transform on train and transform on test
X_ohe = pd.DataFrame(preproc_ohe.fit_transform(X), index=X.index)
X_test_ohe = pd.DataFrame(preproc_ohe.transform(X_test), index=X_test.index)

print('\nX_ohe shape:', X_ohe.shape)



# ---- Strategy B: Ordinal encoding for tree models (fast) ----
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_tree = X.copy()
X_test_tree = X_test.copy()
for col in X_cat:
    X_tree[col] = X_tree[col].astype(str).fillna('NA')
    X_test_tree[col] = X_test_tree[col].astype(str).fillna('NA')

X_tree[X_cat] = ord_enc.fit_transform(X_tree[X_cat])
X_test_tree[X_cat] = ord_enc.transform(X_test_tree[X_cat])

print('X_tree shape:', X_tree.shape)



results = {}


# 6.1 Logistic Regression (baseline) - use X_ohe
print('\n--- Logistic Regression (baseline) ---')
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
_oof_lr, _test_lr, _scores_lr = run_cv_predict(lr, X_ohe, y, X_test_ohe, folds=5, model_name='LogisticRegression')
results['LogisticRegression'] = {'oof': _oof_lr, 'test': _test_lr, 'scores': _scores_lr}



# 6.2 Random Forest (tree-based) - use X_tree
print('\n--- Random Forest ---')
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
_oof_rf, _test_rf, _scores_rf = run_cv_predict(rf, X_tree, y, X_test_tree, folds=5, model_name='RandomForest')
results['RandomForest'] = {'oof': _oof_rf, 'test': _test_rf, 'scores': _scores_rf}


# 6.3 LightGBM
if lgb is not None:
    print('\n--- LightGBM ---')
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'seed': RANDOM_STATE,
        'n_jobs': -1
    }
    # We'll use sklearn wrapper for convenience
    lgb_model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, **lgb_params)
    _oof_lgb, _test_lgb, _scores_lgb = run_cv_predict(lgb_model, X_tree, y, X_test_tree, folds=5, model_name='LightGBM')
    results['LightGBM'] = {'oof': _oof_lgb, 'test': _test_lgb, 'scores': _scores_lgb}
else:
    print('LightGBM not available in this environment.')



# 6.4 XGBoost
if xgb is not None:
    print('\n--- XGBoost ---')
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='auc', n_estimators=1000, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=-1)
    _oof_xgb, _test_xgb, _scores_xgb = run_cv_predict(xgb_model, X_tree, y, X_test_tree, folds=5, model_name='XGBoost')
    results['XGBoost'] = {'oof': _oof_xgb, 'test': _test_xgb, 'scores': _scores_xgb}
else:
    print('XGBoost not available in this environment.')



summary = {}
for k, v in results.items():
    summary[k] = np.mean(v['scores'])

print('\nCV AUC summary:')
for k, v in sorted(summary.items(), key=lambda x: -x[1]):
    print(f"{k}: {v:.5f}")


# Simple average ensemble of available model test preds
available_tests = [v['test'] for v in results.values()]
ensemble_test = np.mean(np.vstack(available_tests), axis=0)


best_model_name = max(summary.items(), key=lambda x: x[1])[0]
print('\nBest model by CV:', best_model_name)

if best_model_name == 'LightGBM' and lgb is not None:
    final_clf = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, **lgb_params)
    final_X = X_tree
    final_X_test = X_test_tree
elif best_model_name == 'XGBoost' and xgb is not None:
    final_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='auc', n_estimators=1000, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=-1)
    final_X = X_tree
    final_X_test = X_test_tree

elif best_model_name == 'RandomForest':
    final_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=RANDOM_STATE)
    final_X = X_tree
    final_X_test = X_test_tree
else:
    # fallback to logistic regression baseline
    final_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    final_X = X_ohe
    final_X_test = X_test_ohe

print('Training final model on full train...')
if isinstance(final_clf, CatBoostClassifier):
    # CatBoost accepts DataFrame and list of categorical names
    final_clf.fit(train_df.drop(columns=[id_col, target_col]), train_df[target_col], cat_features=X_cat)
    final_preds = final_clf.predict_proba(test_df.drop(columns=[id_col]))[:,1]
else:
    final_clf.fit(final_X, y)
    final_preds = final_clf.predict_proba(final_X_test)[:,1]



submission = pd.DataFrame({
    'id': test_df[id_col],
    'y': final_preds
})
submission_path = 'submission.csv'
submission.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path}")

# Show top rows
submission.head()


