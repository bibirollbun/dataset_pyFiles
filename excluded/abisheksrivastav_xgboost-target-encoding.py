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




import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# XGBoost
try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError("xgboost not found. Install/enable it. Error: " + str(e))

# -------------------------
# Config
# -------------------------
SEED = 42
N_SPLITS = 5                # keep 5 as you used
NFOLD_TE = 5
N_ESTIMATORS = 10000        # early stopping will cut
EARLY_STOP = 200
LEARNING_RATE = 0.01
USE_GPU = True              # set False if GPU unavailable
NUM_TOP_NUMERIC = 5         # how many top numeric to interact with cats
OUT_SUB = 'submission_xgb_fixed.csv'
OUT_OOF = 'oof_xgb_fixed.csv'

# -------------------------
# Paths (adjust if needed)
# -------------------------
TRAIN_PATH = '/kaggle/input/playground-series-s5e11/train.csv'
TEST_PATH  = '/kaggle/input/playground-series-s5e11/test.csv'
ORIG_PATH  = '/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv'

# -------------------------
# Load data
# -------------------------
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
orig  = pd.read_csv(ORIG_PATH) if os.path.exists(ORIG_PATH) else None

print('Train:', train.shape, 'Test:', test.shape, 'Orig:', None if orig is None else orig.shape)

# -------------------------
# Basic vars
# -------------------------
TARGET = 'loan_paid_back'
ID_COL = 'id' if 'id' in train.columns else train.columns[0]
BASE = [c for c in train.columns if c not in [ID_COL, TARGET]]

# Keep the categorical columns you listed only if present
CATS = [c for c in ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade'] if c in train.columns]
print('CATS used:', CATS)

# -------------------------
# Create controlled interactions (CATS x top numeric quantiles)
# -------------------------
numeric_cols = [c for c in train.select_dtypes(include=[np.number]).columns if c not in [ID_COL, TARGET]]
if len(numeric_cols) > 0:
    corr_vals = train[numeric_cols + [TARGET]].corr()[TARGET].abs().sort_values(ascending=False)
    top_nums = [c for c in corr_vals.index if c != TARGET][:NUM_TOP_NUMERIC]
else:
    top_nums = []

INTER = []
# Precompute quantile bins for each numeric based on train so we can apply same bins to test
quantile_bins = {}
for num in top_nums:
    try:
        # qcut to get distinct bin edges; handle duplicates
        _, bins = pd.qcut(train[num].rank(method='first'), q=8, retbins=True, duplicates='drop')
        quantile_bins[num] = bins
    except Exception:
        # fallback to simple equal-width bins
        quantile_bins[num] = np.linspace(train[num].min(), train[num].max(), num=9)

for cat in CATS:
    for num in top_nums:
        new = f'{cat}__{num}_q'
        INTER.append(new)
        # produce train column
        try:
            train[new] = train[cat].astype(str) + '_' + pd.cut(train[num].rank(method='first'), bins=quantile_bins[num], include_lowest=True).astype(str)
            test[new]  = test[cat].astype(str)  + '_' + pd.cut(test[num].rank(method='first'), bins=quantile_bins[num], include_lowest=True).astype(str)
        except Exception:
            # fallback: coarse bucketing by std multiples
            train[new] = train[cat].astype(str) + '_' + (train[num] / (train[num].std() + 1e-9)).round(1).astype(str)
            test[new]  = test[cat].astype(str)  + '_' + (test[num]  / (test[num].std()  + 1e-9)).round(1).astype(str)

print('Created INTER count:', len(INTER))

# -------------------------
# ROUND features
# -------------------------
ROUND = []
rounding_levels = {'1s': 0, '10s': -1}
for col in ['annual_income', 'loan_amount']:
    if col in train.columns:
        for suffix, level in rounding_levels.items():
            new_col = f'{col}_ROUND_{suffix}'
            ROUND.append(new_col)
            for df in (train, test):
                if col in df.columns:
                    df[new_col] = df[col].round(level).astype('Int64')
                else:
                    df[new_col] = pd.Series([pd.NA]*len(df), index=df.index)
print('Created ROUND count:', len(ROUND))

# -------------------------
# Orig-derived features (limited and safe)
# -------------------------
ORIG_FEATS = []
if orig is not None:
    if 'loan_paid_back' not in orig.columns:
        for cand in ['Loan_Status', 'target', 'loan_status']:
            if cand in orig.columns:
                orig.rename(columns={cand: 'loan_paid_back'}, inplace=True)
                break
    # limit columns to avoid explosion: take columns present in orig and train with moderate cardinality
    cand_cols = [c for c in BASE if c in orig.columns]
    cand_cols = [c for c in cand_cols if train[c].nunique() < 500][:40]  # cap 40
    for c in cand_cols:
        try:
            mean_map = orig.groupby(c)['loan_paid_back'].mean().rename(f'orig_mean_{c}')
            train = train.merge(mean_map, how='left', left_on=c, right_index=True)
            test  = test.merge(mean_map, how='left', left_on=c, right_index=True)
            ORIG_FEATS.append(f'orig_mean_{c}')
            count_map = orig.groupby(c).size().reset_index(name=f'orig_count_{c}')
            train = train.merge(count_map, how='left', on=c)
            test  = test.merge(count_map, how='left', on=c)
            ORIG_FEATS.append(f'orig_count_{c}')
        except Exception:
            pass
    print('Created ORIG features:', len(ORIG_FEATS))
else:
    print('No orig dataset found — skipping orig features.')

# -------------------------
# Build FEATURES list (only common columns)
# -------------------------
FEATURES = BASE + ORIG_FEATS + INTER + ROUND
FEATURES = [c for c in FEATURES if c in train.columns and c in test.columns]
print('Total FEATURES used:', len(FEATURES))

# -------------------------
# Robust TargetEncoder (OOF) for interactions
# -------------------------
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_encode, agg='mean', n_splits=5, smoothing=20, drop_original=False, seed=SEED):
        self.cols = [c for c in cols_to_encode if c is not None]
        self.agg = agg
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.drop_original = drop_original
        self.global_mean_ = None
        self.full_map_ = {}

    def fit(self, X, y):
        self.global_mean_ = y.mean()
        temp = X.copy()
        temp['_target_'] = y
        for c in self.cols:
            try:
                self.full_map_[c] = temp.groupby(c)['_target_'].agg(['mean','count'])
            except Exception:
                self.full_map_[c] = None
        return self

    def transform(self, X):
        X_out = X.copy()
        for c in self.cols:
            new = f'TE_{c}'
            if self.full_map_.get(c) is not None:
                m = self.full_map_[c]['count']
                mean = self.full_map_[c]['mean']
                smooth = (m * mean + self.smoothing * self.global_mean_) / (m + self.smoothing)
                X_out[new] = X_out[c].map(smooth).fillna(self.global_mean_)
            else:
                X_out[new] = self.global_mean_
        if self.drop_original:
            X_out = X_out.drop(columns=[c for c in self.cols if c in X_out.columns])
        return X_out

    def fit_transform(self, X, y):
        X_out = X.copy()
        self.global_mean_ = y.mean()
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=SEED)
        enc_df = pd.DataFrame(index=X.index)
        for tr_idx, val_idx in skf.split(X, y):
            X_tr = X.iloc[tr_idx].copy(); y_tr = y.iloc[tr_idx]
            temp = X_tr.copy(); temp['_target_'] = y_tr
            for c in self.cols:
                try:
                    stats = temp.groupby(c)['_target_'].agg(['mean','count'])
                    m = stats['count']; mean = stats['mean']
                    smooth = (m * mean + self.smoothing * y_tr.mean()) / (m + self.smoothing)
                    new = f'TE_{c}'
                    vals = X.iloc[val_idx][c].map(smooth).fillna(y_tr.mean())
                    enc_df.loc[vals.index, new] = vals.values
                except Exception:
                    enc_df.loc[X.iloc[val_idx].index, f'TE_{c}'] = y_tr.mean()
        # merge into X_out
        for col in enc_df.columns:
            X_out[col] = enc_df[col].fillna(self.global_mean_)
        if self.drop_original:
            X_out = X_out.drop(columns=[c for c in self.cols if c in X_out.columns])
        # store full-train mapping for test transform
        self.fit(X, y)
        return X_out

# -------------------------
# Prepare X, y, X_test once
# -------------------------
X = train[FEATURES].copy()
y = train[TARGET].astype(int).copy()
X_test = test[FEATURES].copy()

# fill obvious NAs for numeric vs object prior to encoding (we will label-encode objects later)
for c in X.columns:
    if X[c].dtype.kind in 'fiu':
        X[c].fillna(-999, inplace=True)
        X_test[c].fillna(-999, inplace=True)
    else:
        X[c].fillna('__MISSING__', inplace=True)
        X_test[c].fillna('__MISSING__', inplace=True)

# -------------------------
# Prepare TE columns selection and fit global mapping for test usage
# -------------------------
# Limit number of TE columns to avoid heavy computation; encode interactions first
te_cols = [c for c in INTER if c in X.columns]
MAX_TE = 200
te_cols = te_cols[:MAX_TE]
print('TE columns count (capped):', len(te_cols))

global_te = TargetEncoder(cols_to_encode=te_cols, n_splits=NFOLD_TE, smoothing=20)
if len(te_cols) > 0:
    global_te.fit(X, y)  # used to map test later

# -------------------------
# XGBoost training with robust per-fold transforms and label-encoding union
# -------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

# compute class imbalance weight
n_pos = (y == 1).sum()
n_neg = (y == 0).sum()
scale_pos_weight = n_neg / (n_pos + 1e-9)
print('scale_pos_weight:', scale_pos_weight)

# shared XGB params (do not include early_stopping_rounds here)
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'colsample_bytree': 0.3,
    'subsample': 0.55,
    'n_estimators': N_ESTIMATORS,
    'learning_rate': LEARNING_RATE,
    'use_label_encoder': False,
    'random_state': SEED,
    'n_jobs': -1,
    'verbosity': 0,
    'scale_pos_weight': scale_pos_weight
}
if USE_GPU:
    xgb_params['tree_method'] = 'gpu_hist'

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'\n=== Fold {fold}/{N_SPLITS} ===')
    X_tr = X.iloc[tr_idx].copy(); X_val = X.iloc[val_idx].copy()
    y_tr = y.iloc[tr_idx]; y_val = y.iloc[val_idx]

    # OOF TE for te_cols inside fold
    if len(te_cols) > 0:
        TE_fold = TargetEncoder(cols_to_encode=te_cols, n_splits=NFOLD_TE, smoothing=20, drop_original=True)
        X_tr = TE_fold.fit_transform(X_tr, y_tr)
        X_val = TE_fold.transform(X_val)
        X_test_te = global_te.transform(X_test.copy())
    else:
        X_test_te = X_test.copy()

    # Build union of object/category columns across X_tr, X_val, X_test_te
    obj_cols_fold = sorted(list(
        set(X_tr.select_dtypes(include=['object','category']).columns.tolist()) |
        set(X_val.select_dtypes(include=['object','category']).columns.tolist()) |
        set(X_test_te.select_dtypes(include=['object','category']).columns.tolist())
    ))
    # Label-encode each such column on the union of values (train+val+test for fold)
    for col in obj_cols_fold:
        combined = pd.concat([
            X_tr[col].astype(str),
            X_val[col].astype(str),
            X_test_te[col].astype(str)
        ], axis=0).fillna('__MISSING__').astype(str)
        le = LabelEncoder()
        le.fit(combined.values)
        if col in X_tr.columns:
            X_tr[col] = le.transform(X_tr[col].astype(str).fillna('__MISSING__').values)
        if col in X_val.columns:
            X_val[col] = le.transform(X_val[col].astype(str).fillna('__MISSING__').values)
        if col in X_test_te.columns:
            X_test_te[col] = le.transform(X_test_te[col].astype(str).fillna('__MISSING__').values)

    # final numeric NA fill
    X_tr.fillna(-999, inplace=True)
    X_val.fillna(-999, inplace=True)
    X_test_te.fillna(-999, inplace=True)

    # Train XGBoost with early stopping
    model = XGBClassifier(**xgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=EARLY_STOP, verbose=200)

    # Predict & accumulate
    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(X_test_te)[:, 1]

    oof_preds[val_idx] = val_pred
    test_preds += test_pred / N_SPLITS

    fold_auc = roc_auc_score(y_val, val_pred)
    print(f'Fold {fold} AUC: {fold_auc:.6f}')

# overall OOF
overall_auc = roc_auc_score(y, oof_preds)
print('\n====================')
print('Overall OOF AUC:', overall_auc)
print('====================')

# Save OOF and submission
oof_df = train[[ID_COL, TARGET]].copy()
oof_df['oof_pred'] = oof_preds
oof_df.to_csv(OUT_OOF, index=False)
submission = pd.DataFrame({ID_COL: test[ID_COL] if ID_COL in test.columns else np.arange(len(test)), TARGET: test_preds})
submission.to_csv(OUT_SUB, index=False)
print('Saved', OUT_SUB, OUT_OOF)


