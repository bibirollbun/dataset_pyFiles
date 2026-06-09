# Kaggle Playground Series — Loan Paid Back prediction
# Run this notebook on Kaggle (or Colab with dataset mounted).
# Features:
# - Load train/test
# - Simple EDA and visualizations (matplotlib only)
# - Preprocessing (impute, encode, scale)
# - Train Logistic Regression, RandomForest, LightGBM (if available)\# - Out-of-fold CV, stacking, simple weighted blend
# - Produce submission to /kaggle/working/submission.csv

import os
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# LightGBM if available
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

INPUT_DIR = '/kaggle/input/playground-series-s5e11'
WORK_DIR = '/kaggle/working'

# 1) Load data
train = pd.read_csv(Path(INPUT_DIR)/'train.csv')
test  = pd.read_csv(Path(INPUT_DIR)/'test.csv')
print('train shape:', train.shape)
print('test shape :', test.shape)

# Quick peek
print(train.head())

TARGET = 'loan_paid_back'
IDCOL = 'id'

# 2) Basic EDA
# Target distribution
plt.figure(figsize=(6,4))
vals = train[TARGET].value_counts(normalize=True).sort_index()
plt.bar(vals.index.astype(str), vals.values)
plt.title('Target distribution (proportion)')
plt.xlabel('loan_paid_back')
plt.ylabel('Proportion')
plt.grid(alpha=0.2)
plt.show()

# Missing value summary
def missing_summary(df):
    miss = df.isnull().sum()
    miss = miss[miss>0].sort_values(ascending=False)
    return miss

print('\nMissing values in train:')
print(missing_summary(train))

# Quick numeric summary (first 15 numeric features)
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in [IDCOL, TARGET]]
print('\nNumeric features (sample):', num_cols[:15])

# Correlation with target for numeric features
corrs = train[num_cols + [TARGET]].corr()[TARGET].sort_values(ascending=False)
print('\nTop correlations with target:')
print(corrs.head(15))

# Plot top 6 numeric feature distributions vs target
top_feats = corrs.abs().sort_values(ascending=False).index[1:7].tolist()
plt.figure(figsize=(12,8))
for i, f in enumerate(top_feats,1):
    plt.subplot(3,2,i)
    plt.hist(train.loc[train[TARGET]==0, f].dropna(), bins=40, alpha=0.6, label='0')
    plt.hist(train.loc[train[TARGET]==1, f].dropna(), bins=40, alpha=0.6, label='1')
    plt.title(f)
    plt.legend()
plt.tight_layout()
plt.show()

# 3) Preprocessing
# Combine train/test for consistent encoding
train['is_train'] = 1
test['is_train']  = 0
combined = pd.concat([train.drop(columns=[TARGET]), test], axis=0)

# Identify categorical columns
cat_cols = combined.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = [c for c in combined.columns if c not in cat_cols + [IDCOL, 'is_train']]
print('\nCategorical cols sample:', cat_cols[:20])
print('\nNumeric cols sample:', num_cols[:20])

# Fill missing numeric with median
for c in num_cols:
    med = combined[c].median()
    combined[c] = combined[c].fillna(med)

# Fill categorical missing with 'MISSING'
for c in cat_cols:
    combined[c] = combined[c].fillna('MISSING')

# One-hot encode categorical columns but limit to reasonable number of dummies
# Keep top 30 frequent levels for each categorical, others grouped as 'OTHER'
def top_n_levels(series, n=30):
    vc = series.value_counts()
    return set(vc.index[:n])

for c in cat_cols:
    top = top_n_levels(combined[c], n=30)
    combined[c] = combined[c].where(combined[c].isin(top), 'OTHER')

combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

# Split back
train_proc = combined[combined['is_train']==1].drop(columns=['is_train']).copy()
test_proc  = combined[combined['is_train']==0].drop(columns=['is_train']).copy()

# Align columns (just in case)
train_proc, test_proc = train_proc.align(test_proc, join='left', axis=1, fill_value=0)

X = train_proc.drop(columns=[IDCOL])
y = train[TARGET].values
X_test = test_proc.drop(columns=[IDCOL])
ids = test_proc[IDCOL].values

print('Processed X shape:', X.shape)
print('Processed X_test shape:', X_test.shape)

# Scale numeric features (StandardScaler)
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# 4) Modeling utilities - OOF predictions
NFOLDS = 5
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)

def get_oof_preds(clf, X, y, X_test, clf_name='model'):
    oof = np.zeros(X.shape[0])
    preds_test = np.zeros(X_test.shape[0])
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y),1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        clf.fit(X_tr, y_tr)
        oof[val_idx] = clf.predict_proba(X_val)[:,1]
        preds_test += clf.predict_proba(X_test)[:,1] / NFOLDS
        auc = roc_auc_score(y_val, oof[val_idx])
        print(f'{clf_name} fold {fold} AUC: {auc:.5f}')
    full_auc = roc_auc_score(y, oof)
    print(f'{clf_name} OOF AUC: {full_auc:.5f}')
    return oof, preds_test

# 5) Train three models
# Logistic Regression
lr = LogisticRegression(max_iter=2000)
print('\nTraining Logistic Regression')
lr_oof, lr_test = get_oof_preds(lr, X, y, X_test, 'LogisticRegression')

# Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=8, n_jobs=-1, random_state=42)
print('\nTraining RandomForest')
rf_oof, rf_test = get_oof_preds(rf, X, y, X_test, 'RandomForest')

# LightGBM (if available)
if HAS_LGB:
    print('\nTraining LightGBM')
    lgb_params = {
        'objective':'binary', 'metric':'auc', 'verbosity':-1, 'boosting_type':'gbdt',
        'learning_rate':0.05, 'num_leaves':31, 'seed':42
    }
    lgb_oof = np.zeros(X.shape[0])
    lgb_test = np.zeros(X_test.shape[0])
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y),1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val)
        bst = lgb.train(lgb_params, train_data, num_boost_round=1000,
                        valid_sets=[train_data, val_data], early_stopping_rounds=50, verbose_eval=100)
        lgb_oof[val_idx] = bst.predict(X_val, num_iteration=bst.best_iteration)
        lgb_test += bst.predict(X_test, num_iteration=bst.best_iteration) / NFOLDS
        print(f'LightGBM fold {fold} AUC: {roc_auc_score(y_val, lgb_oof[val_idx]):.5f}')
    print(f'LightGBM OOF AUC: {roc_auc_score(y, lgb_oof):.5f}')
else:
    lgb_oof = None
    lgb_test = None
    print('\nLightGBM not installed in this environment; skipping.')

# 6) Simple stacking: Logistic regression on OOF preds
stack_train = np.vstack([lr_oof, rf_oof] + ([lgb_oof] if HAS_LGB else [])).T
stack_test  = np.vstack([lr_test, rf_test] + ([lgb_test] if HAS_LGB else [])).T

meta = LogisticRegression(max_iter=2000)
meta.fit(stack_train, y)
meta_preds = meta.predict_proba(stack_test)[:,1]
print('\nStacking meta OOF AUC (cv on stack) - approximate (retrain on full stack):')
# We can estimate stack OOF by cross-val on stack_train
from sklearn.model_selection import cross_val_score
print('Stacking CV AUC:', cross_val_score(meta, stack_train, y, cv=skf, scoring='roc_auc').mean())

# 7) Weighted blend
# weights can be tuned; here's an example
weights = {'lr':0.2, 'rf':0.3, 'lgb':0.5 if HAS_LGB else 0.8}
if HAS_LGB:
    final_test_pred = weights['lr']*lr_test + weights['rf']*rf_test + weights['lgb']*lgb_test
else:
    final_test_pred = 0.4*lr_test + 0.6*rf_test

# Also include stack
final_test_pred = 0.7*final_test_pred + 0.3*meta_preds

# 8) Visualizations: ROC curves of OOF predictions
plt.figure(figsize=(8,6))
for name, preds in [('LogReg', lr_oof), ('RandomForest', rf_oof), ('Stack', meta.predict_proba(stack_train)[:,1])]:
    fpr, tpr, _ = roc_curve(y, preds)
    auc = roc_auc_score(y, preds)
    plt.plot(fpr, tpr, label=f'{name} AUC={auc:.4f}')
plt.plot([0,1],[0,1],'k--', alpha=0.4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('OOF ROC curves')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Feature importances from RandomForest
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)[:30]
plt.figure(figsize=(8,10))
plt.barh(range(len(feat_imp)), feat_imp[::-1])
plt.yticks(range(len(feat_imp)), feat_imp.index[::-1])
plt.title('Top 30 feature importances (RandomForest)')
plt.tight_layout()
plt.show()

# 9) Prepare submission
sub = pd.DataFrame({IDCOL: ids, TARGET: final_test_pred})
sub.to_csv(Path(WORK_DIR)/'submission.csv', index=False)
print('\nSaved submission to /kaggle/working/submission.csv')

# 10) Cleanup
gc.collect()

# End of notebook
print('\nDone.\n')


