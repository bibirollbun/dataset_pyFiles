# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold # Best for classification/imbalance
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore', category=FutureWarning) 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train=pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
df_train.drop('id',axis=1,inplace=True)
df_test.drop('id',axis=1,inplace=True)


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from category_encoders import TargetEncoder

rand_seed = 111
target = 'loan_paid_back'

cat_cols = df_train.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in df_train.select_dtypes(include="number").columns if c != target]

print(f"\nğŸ”µ Initial categorical columns: {cat_cols}")
print(f"ğŸ”µ Initial numeric columns: {num_cols}")
print(f"ğŸ”µ Train shape: {df_train.shape} | Test shape: {df_test.shape}")

# --- Step 1: Consistent dtype for categoricals ---
def convert_dtype(df, cat_cols):
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df

print("\nğŸ”„ Converting dtypes for categoricals...")
df_train_fe = convert_dtype(df_train, cat_cols)
df_test_fe  = convert_dtype(df_test, cat_cols)

# --- Step 2: Label Encoding ---
def encode_categorical(tr_df, ts_df, cols):
    tr_df, ts_df = tr_df.copy(), ts_df.copy()
    for col in cols:
        le = LabelEncoder()
        tr_df[col] = le.fit_transform(tr_df[col].astype(str))
        mapped = ts_df[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
        ts_df[col] = le.transform(mapped)
        print(f"   âœ“ Label encoded {col} | Classes: {len(le.classes_)}")
    return tr_df, ts_df

print("\nğŸ”„ Label encoding categoricals...")
df_train_fe, df_test_fe = encode_categorical(df_train_fe, df_test_fe, cat_cols)

# --- Step 3: Numeric Interactions ---
def num_inter(df, num_cols):
    df = df.copy()
    pairs = [(i, j) for i in range(len(num_cols)) for j in range(i+1, len(num_cols))]
    for i, j in pairs:
        c1, c2 = num_cols[i], num_cols[j]
        df[f"{c1}_plus_{c2}"]  = df[c1] + df[c2]
        df[f"{c1}_times_{c2}"] = df[c1] * df[c2]
        df[f"{c1}_div_{c2}"]   = df[c1] / (df[c2].replace(0, 1e-5) + 1e-5)
    print(f"   âœ“ Added {len(pairs)*3} numeric interaction features.")
    return df

print("\nğŸ”„ Adding numeric interactions...")
df_train_fe = num_inter(df_train_fe, num_cols)
df_test_fe  = num_inter(df_test_fe, num_cols)

# --- Step 4: Numeric Transformations & Binning ---
def num_trans(df, num_cols, ref_df=None):
    df = df.copy()
    binned, logged, rooted = 0, 0, 0
    for col in num_cols:
        if ref_df is not None:
            try:
                _, bin_edges = pd.qcut(ref_df[col], 5, retbins=True, duplicates='drop')
                df[f'{col}_bin'] = pd.cut(df[col], bins=bin_edges, labels=False, include_lowest=True)
                binned += 1
            except Exception:
                continue
        else:
            try:
                df[f'{col}_bin'] = pd.qcut(df[col], 5, labels=False, duplicates='drop')
                binned += 1
            except Exception:
                continue
        df[f'{col}_log']  = np.log1p(np.abs(df[col])); logged += 1
        df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col])); rooted += 1
    print(f"   âœ“ Binned: {binned}, Log-transformed: {logged}, Sqrt: {rooted}")
    return df

print("\nğŸ”„ Numeric transforms and binning...")
df_train_fe = num_trans(df_train_fe, num_cols, ref_df=df_train_fe)
df_test_fe  = num_trans(df_test_fe, num_cols, ref_df=df_train_fe)

# --- Step 5: Grade/Sub-Grade split ---
print("\nğŸ”„ Splitting grade_subgrade into grade and sub_grade...")
df_train_fe['grade'] = df_train_fe['grade_subgrade'].astype(str).str[0]
df_train_fe['sub_grade'] = df_train_fe['grade_subgrade'].astype(str).str[1]
df_test_fe['grade'] = df_test_fe['grade_subgrade'].astype(str).str[0]
df_test_fe['sub_grade'] = df_test_fe['grade_subgrade'].astype(str).str[1]
print(f"   âœ“ Added 'grade' and 'sub_grade' columns.")

# --- Step 6: Scaling numerics ---
exclude_for_scaling = set(cat_cols + [target])
scale_cols = [c for c in df_train_fe.columns if c not in exclude_for_scaling and df_train_fe[c].dtype in [np.float64, np.int64, float, int]]

print(f"\nğŸ”„ Scaling {len(scale_cols)} numeric columns...")
scaler = StandardScaler()
df_train_fe[scale_cols] = scaler.fit_transform(df_train_fe[scale_cols])
df_test_fe[scale_cols]  = scaler.transform(df_test_fe[scale_cols])

print("   âœ“ Finished scaling.")

# --- Step 7: Target encoding ---
bin_features = ['annual_income', 'loan_amount', 'credit_score']
te_cols = [col+'_bin' for col in bin_features] + ['grade', 'sub_grade']
print(f"\nğŸ”„ Target encoding columns: {te_cols}")
te = TargetEncoder(cols=te_cols)

te.fit(df_train_fe.drop(columns=[target]), df_train_fe[target])
df_train_fe = te.transform(df_train_fe.drop(columns=[target]))
df_test_fe  = te.transform(df_test_fe)
print("   âœ“ Target encoding complete.")

# --- Output for modeling ---
X = df_train_fe
y = df_train[target]
X_test = df_test_fe

print(f"\nâœ… Feature engineering complete. X shape: {X.shape}, X_test shape: {X_test.shape}")



import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
import warnings
warnings.filterwarnings('ignore')

def blended_feature_selection(df_train, df_test, target_col='loan_paid_back', n_features=50,
                             mi_thresh=0.03, lgb_thresh=0.03, et_thresh=0.03, min_sources=2):
    print("\n=*= FEATURE SELECTION PIPELINE (Multiple Importances, Robust) =*=")
    y = df_train[target_col]
    X_train = df_train.drop(columns=[target_col])
    X_test = df_test.copy()

    # Remove constant features
    constant_features = [c for c in X_train.columns if X_train[c].nunique(dropna=False) <= 1]
    if constant_features:
        print(f"âœ“ Dropped {len(constant_features)} constant features: {constant_features}")
    X_train = X_train.drop(columns=constant_features)
    X_test  = X_test.drop(columns=constant_features, errors='ignore')

    # Remove highly correlated features
    corr_matrix = X_train.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [col for col in upper_tri.columns if (upper_tri[col] > 0.99).any()]
    if high_corr:
        print(f"âœ“ Dropped {len(high_corr)} highly correlated features: {high_corr}")
    X_train = X_train.drop(columns=high_corr)
    X_test  = X_test.drop(columns=high_corr, errors='ignore')

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    Xf = X_train[num_cols].fillna(0)  # For selectors

    print("ğŸ”� Calculating importances and selecting features...")
    # Mutual Info
    mi_scores = mutual_info_classif(Xf, y, random_state=42)
    mi_scores_norm = mi_scores / (np.max(mi_scores) + 1e-10)
    mi_features = [f for f, s in zip(num_cols, mi_scores_norm) if s > mi_thresh]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, importance_type='gain')
    lgb_model.fit(Xf, y)
    lgb_imp = lgb_model.feature_importances_
    lgb_imp_norm = lgb_imp / (np.max(lgb_imp) + 1e-10)
    lgb_features = [f for f, s in zip(num_cols, lgb_imp_norm) if s > lgb_thresh]

    # Extra Trees
    et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    et_model.fit(Xf, y)
    et_imp = et_model.feature_importances_
    et_imp_norm = et_imp / (np.max(et_imp) + 1e-10)
    et_features = [f for f, s in zip(num_cols, et_imp_norm) if s > et_thresh]

    # Create robust selector: keep only features identified by at least two selectors
    selectors = {'mi': mi_features, 'lgb': lgb_features, 'et': et_features}
    def count_sources(feat):
        return sum([feat in selectors[s] for s in selectors])

    print("  Counting agreement of sources for each feature...")
    agreement = {f: count_sources(f) for f in num_cols}
    robust_features = [f for f, cnt in agreement.items() if cnt >= min_sources]
    print(f"â˜… Features with agreement from at least {min_sources} selectors: {len(robust_features)}")

    # Average rank filter (only rank robust features)
    imp_df = pd.DataFrame({'feature': num_cols,
                           'mi': mi_scores_norm, 'lgb': lgb_imp_norm, 'et': et_imp_norm})
    for c in ['mi', 'lgb', 'et']:
        imp_df[f'{c}_rank'] = imp_df[c].rank(ascending=False)
    imp_df['avg_rank'] = imp_df[[f'{c}_rank' for c in ['mi','lgb','et']]].mean(axis=1)

    imp_df = imp_df[imp_df['feature'].isin(robust_features)]
    imp_df = imp_df.sort_values('avg_rank')
    selected = imp_df['feature'].tolist()[:n_features]

    print(f"âœ… Robust selected features ({len(selected)}):\n - {selected[:10]}{' ...' if len(selected) > 10 else ''}")
    X_final = X_train[selected].copy()
    X_test_final = X_test[selected].copy()
    print(f"âœ�ï¸� X_final: {X_final.shape}, X_test_final: {X_test_final.shape}")

    return X_final, X_test_final, y

# Usage
df_train_blend = X.copy()
df_train_blend['loan_paid_back'] = y

X_final, X_test_final, _ = blended_feature_selection(
    df_train_blend, X_test, target_col='loan_paid_back',
    n_features=50, mi_thresh=0.03, lgb_thresh=0.03, et_thresh=0.03, min_sources=2)



# ============================================================
# L0 STACK (NO OPTUNA): Multi-seed K-Fold for LGBM, XGB, CAT
# Uses your best params and builds OOF/test preds + meta-features
# Seeds = (42, 77, 123)
# ============================================================
import gc, sys, os, warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from contextlib import contextmanager

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

# -----------------------------
# Silence noisy logs
# -----------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

@contextmanager
def suppress_output():
    devnull = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = devnull, devnull
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        devnull.close()

# -----------------------------
# Helpers for categorical safety
# -----------------------------
def detect_cats(df: pd.DataFrame):
    return list(df.select_dtypes(include=['object', 'category']).columns)

def align_categories(X: pd.DataFrame, X_test: pd.DataFrame, cat_cols):
    Xc, Xt = X.copy(), X_test.copy()
    for c in cat_cols:
        base = pd.Index(Xc[c].astype('category').cat.categories)
        if c in Xt.columns:
            base = base.union(pd.Index(Xt[c].astype('category').cat.categories))
        Xc[c]  = Xc[c].astype('category').cat.set_categories(base)
        if c in Xt.columns:
            Xt[c] = Xt[c].astype('category').cat.set_categories(base)
    return Xc, Xt

def fix_dtypes_for_lgb(df: pd.DataFrame, cat_cols):
    df = df.copy()
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype('category')
    for c in df.columns:  # ensure no stray 'object'
        if df[c].dtype == 'object':
            df[c] = df[c].astype('category')
    return df

# -----------------------------
# Your BEST parameters (as provided)
# -----------------------------
BEST_LGBM = {
    'learning_rate': 0.06189585591734019,
    'num_leaves': 71,
    'max_depth': 3,
    'min_child_samples': 47,
    'subsample': 0.7882534840295796,
    'colsample_bytree': 0.50516850988847,
    'reg_alpha': 4.108406108850896,
    'reg_lambda': 0.7082653396924283,
    'n_estimators': 4040,
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'class_weight': 'balanced',
    'verbosity': -1
}

BEST_XGB = {
    'learning_rate': 0.06688084442321802,
    'max_depth': 3,
    'min_child_weight': 57,
    'subsample': 0.853551584789076,
    'colsample_bytree': 0.9782757144921325,
    'colsample_bylevel': 0.5465267973481097,
    'gamma': 0.942315585496616,
    'reg_alpha': 4.943158155300519,
    'reg_lambda': 1.4827433844229039,
    'n_estimators': 3261,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'enable_categorical': True,
    'scale_pos_weight': 0.25184723094496453,  # as provided
    'n_jobs': -1,
    'verbosity': 0,
    'random_state': 42
}

# Earlier CatBoost best (from your messages)
BEST_CAT = {
    'iterations': 658,
    'depth': 4,
    'learning_rate': 0.14892724411121927,
    'subsample': 0.6720305549704694,
    'colsample_bylevel': 0.6201948153318813,
    'l2_leaf_reg': 4.593721851846689,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': False
}

# -----------------------------
# Single-seed L0 fit
# -----------------------------
def fit_l0_one_seed(
    X, y, X_test,
    models=('lgbm','xgb','cat'),
    n_folds: int = 7,
    seed: int = 123,
    es_rounds: int = 200
):
    # Align categories once
    cat_cols = detect_cats(X)
    X_aligned, Xtest_aligned = align_categories(X, X_test, cat_cols)
    # LGBM needs category dtype
    X_lgb  = fix_dtypes_for_lgb(X_aligned, cat_cols)
    Xtest_lgb = fix_dtypes_for_lgb(Xtest_aligned, cat_cols)
    # CatBoost cat indices
    cat_idx = [X_aligned.columns.get_loc(c) for c in cat_cols if c in X_aligned.columns]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds  = {m: np.zeros(len(X), dtype=float) for m in models}
    test_preds = {m: np.zeros(len(X_test), dtype=float) for m in models}

    print("="*62)
    print(f"L0 training (seed={seed}) | models={list(models)} | folds={n_folds}")
    print("="*62)

    for m in models:
        print(f"\nğŸš€ Fitting {m.upper()} ...")
        fold_test = np.zeros((n_folds, len(X_test)), dtype=float)

        for i, (tr, va) in enumerate(skf.split(X_aligned, y), 1):
            y_tr, y_va = y.iloc[tr], y.iloc[va]

            with suppress_output():
                if m == 'lgbm':
                    model = LGBMClassifier(**deepcopy(BEST_LGBM), n_jobs=-1)
                    model.fit(
                        X_lgb.iloc[tr], y_tr,
                        eval_set=[(X_lgb.iloc[va], y_va)],
                        categorical_feature=[c for c in X_lgb.columns if X_lgb[c].dtype.name == 'category'],
                        callbacks=[lgb.early_stopping(es_rounds), lgb.log_evaluation(0)]
                    )
                    va_pred = model.predict_proba(X_lgb.iloc[va])[:, 1]
                    te_pred = model.predict_proba(Xtest_lgb)[:, 1]

                elif m == 'xgb':
                    params = deepcopy(BEST_XGB)
                    # ensure seed consistency for each run
                    params['random_state'] = seed
                    model = XGBClassifier(**params)
                    model.fit(
                        X_aligned.iloc[tr], y_tr,
                        eval_set=[(X_aligned.iloc[va], y_va)],
                        early_stopping_rounds=es_rounds,
                        verbose=False
                    )
                    va_pred = model.predict_proba(X_aligned.iloc[va])[:, 1]
                    te_pred = model.predict_proba(Xtest_aligned)[:, 1]

                else:  # cat
                    params = deepcopy(BEST_CAT)
                    params['random_seed'] = seed
                    train_pool = Pool(X_aligned.iloc[tr], y_tr, cat_features=cat_idx)
                    val_pool   = Pool(X_aligned.iloc[va], y_va, cat_features=cat_idx)
                    test_pool  = Pool(Xtest_aligned, cat_features=cat_idx)
                    model = CatBoostClassifier(**params)
                    model.fit(
                        train_pool,
                        eval_set=val_pool,
                        use_best_model=True,
                        early_stopping_rounds=es_rounds,
                        verbose=False
                    )
                    va_pred = model.predict_proba(val_pool)[:, 1]
                    te_pred = model.predict_proba(test_pool)[:, 1]

            oof_preds[m][va] = va_pred
            fold_test[i-1]   = te_pred
            print(f"   [{m.upper()}|seed={seed}] Fold {i}/{n_folds} AUC: {roc_auc_score(y_va, va_pred):.5f}")

            del model; gc.collect()

        test_preds[m] = fold_test.mean(axis=0)
        print(f"[{m.upper()}] OOF AUC (seed={seed}): {roc_auc_score(y, oof_preds[m]):.5f}")

    X_meta_train = pd.DataFrame({m: oof_preds[m]  for m in models})
    X_meta_test  = pd.DataFrame({m: test_preds[m] for m in models})
    return oof_preds, test_preds, X_meta_train, X_meta_test

# -----------------------------
# Multi-seed wrapper: average predictions for seeds (42,77,123)
# -----------------------------
def fit_l0_multi_seed(
    X, y, X_test,
    models=('lgbm','xgb','cat'),
    n_folds: int = 7,
    seeds=(42, 77, 123),
    es_rounds: int = 200
):
    oof_acc  = {m: np.zeros(len(X), dtype=float)     for m in models}
    test_acc = {m: np.zeros(len(X_test), dtype=float) for m in models}

    for sd in seeds:
        oof_s, test_s, _, _ = fit_l0_one_seed(
            X, y, X_test,
            models=models, n_folds=n_folds, seed=sd, es_rounds=es_rounds
        )
        for m in models:
            oof_acc[m]  += oof_s[m]
            test_acc[m] += test_s[m]

    for m in models:
        oof_acc[m]  /= float(len(seeds))
        test_acc[m] /= float(len(seeds))

    X_meta_train = pd.DataFrame({m: oof_acc[m]  for m in models})
    X_meta_test  = pd.DataFrame({m: test_acc[m] for m in models})

    print("\n" + "="*62)
    print("Averaged across seeds:")
    for m in models:
        print(f"  {m.upper()} OOF AUC (avg): {roc_auc_score(y, oof_acc[m]):.5f}")
    print("="*62)

    return oof_acc, test_acc, X_meta_train, X_meta_test

# -----------------------------
# RUN (define X, y, X_test before executing)
# -----------------------------
if __name__ == "__main__":
    # Example:
    # X = train.drop(columns=[TARGET, 'id'])
    # y = train[TARGET].astype(int)
    # X_test = test.drop(columns=['id'])

    oof_preds, test_preds, X_meta_train, X_meta_test = fit_l0_multi_seed(
        X, y, X_test,
        models=('lgbm','xgb','cat'),
        n_folds=7,
        seeds=(42, 77, 123),
        es_rounds=200
    )

    # Quick access vars if you stack next:
    lgb_oof, xgb_oof, cat_oof   = oof_preds['lgbm'], oof_preds['xgb'], oof_preds['cat']
    lgb_test, xgb_test, cat_test = test_preds['lgbm'], test_preds['xgb'], test_preds['cat']

    print("\nâœ… L0 multi-seed finished.")
    print("Meta-train shape:", X_meta_train.shape, "| Meta-test shape:", X_meta_test.shape)



# --- Stacker on top of existing L0 predictions (no L0 retrain) ---
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

# Assumes you already have: y, lgb_oof, xgb_oof, cat_oof, lgb_test, xgb_test, cat_test

# -----------------------------
# 1) Build base-model blends from L0
# -----------------------------
lgb_score = roc_auc_score(y, lgb_oof)
xgb_score = roc_auc_score(y, xgb_oof)
cat_score = roc_auc_score(y, cat_oof)
print(f"[L0] OOF AUCs -> LGBM: {lgb_score:.5f} | XGB: {xgb_score:.5f} | CAT: {cat_score:.5f}")

# AUC-weighted
total = lgb_score + xgb_score + cat_score
w_lgb = lgb_score / total
w_xgb = xgb_score / total
w_cat = cat_score / total
weighted_test = w_lgb * lgb_test + w_xgb * xgb_test + w_cat * cat_test
weighted_oof  = w_lgb * lgb_oof  + w_xgb * xgb_oof  + w_cat * cat_oof

# AUC^2-weighted
sum_sq = lgb_score**2 + xgb_score**2 + cat_score**2
w_lgb2 = lgb_score**2 / sum_sq
w_xgb2 = xgb_score**2 / sum_sq
w_cat2 = cat_score**2 / sum_sq
opt_test = w_lgb2 * lgb_test + w_xgb2 * xgb_test + w_cat2 * cat_test
opt_oof  = w_lgb2 * lgb_oof  + w_xgb2 * xgb_oof  + w_cat2 * cat_oof

# Rank blend (strict [0,1] scaling)
def rank01(a):
    r = rankdata(a) - 1
    return r / (len(a) - 1)
rank_test = (rank01(lgb_test) + rank01(xgb_test) + rank01(cat_test)) / 3.0
rank_oof  = (rank01(lgb_oof)  + rank01(xgb_oof)  + rank01(cat_oof))  / 3.0

# Simple tri-blend of the three styles (still only L0)
ensemble_test = 0.4 * weighted_test + 0.3 * rank_test + 0.3 * opt_test
ensemble_oof  = 0.4 * weighted_oof  + 0.3 * rank_oof  + 0.3 * opt_oof

# -----------------------------
# 2) OOF-optimized non-negative weights for L0 (safer than Nelder-Mead)
# -----------------------------
oofs  = np.vstack([lgb_oof, xgb_oof, cat_oof]).T  # (n,3)
tests = np.vstack([lgb_test, xgb_test, cat_test]).T

def neg_auc(w):
    w = np.clip(w, 0, 1)
    s = w.sum()
    if s == 0:
        w = np.array([1/3,1/3,1/3])
    else:
        w = w / s
    blend = oofs @ w
    return -roc_auc_score(y, blend)

cons   = ({'type':'eq', 'fun': lambda w: w.sum() - 1})
bounds = [(0,1)]*3
w0     = np.array([1/3,1/3,1/3])
res    = minimize(neg_auc, w0, method='SLSQP', bounds=bounds, constraints=cons)
w_opt  = res.x / res.x.sum()
opt_oof_auc = roc_auc_score(y, (oofs @ w_opt))
print(f"[L0 Blend Opt] Weights LGB/XGB/CAT = {w_opt[0]:.3f}, {w_opt[1]:.3f}, {w_opt[2]:.3f} | OOF AUC: {opt_oof_auc:.5f}")

optimized_L0_test = tests @ w_opt
optimized_L0_oof  = oofs  @ w_opt

# -----------------------------
# 3) Meta-features for stacking
# -----------------------------
X_meta_train = pd.DataFrame({'lgbm': lgb_oof, 'xgb': xgb_oof, 'cat': cat_oof})
X_meta_test  = pd.DataFrame({'lgbm': lgb_test, 'xgb': xgb_test, 'cat': cat_test})

# -----------------------------
# 4) Logistic Regression stacker with OOF and test via CV
#     - robust for tiny meta feature space
#     - class_weight balanced for 80/20
# -----------------------------
n_folds = 7
meta_skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

lr_oof  = np.zeros(len(X_meta_train))
lr_test_f = np.zeros((n_folds, len(X_meta_test)))

for f, (tr_idx, va_idx) in enumerate(meta_skf.split(X_meta_train, y), 1):
    Xm_tr, Xm_va = X_meta_train.iloc[tr_idx], X_meta_train.iloc[va_idx]
    y_tr,  y_va  = y.iloc[tr_idx], y.iloc[va_idx]

    lr = LogisticRegression(
        C=1.0, penalty='l2', solver='lbfgs', max_iter=2000,
        class_weight='balanced', random_state=42
    )
    lr.fit(Xm_tr, y_tr)
    lr_oof[va_idx] = lr.predict_proba(Xm_va)[:, 1]
    lr_test_f[f-1] = lr.predict_proba(X_meta_test)[:, 1]

lr_pred = lr_test_f.mean(axis=0)
print(f"[Stacker LR] OOF ROC AUC: {roc_auc_score(y, lr_oof):.5f}")

# -----------------------------
# 5) Final prediction (balanced to avoid double-counting)
#     Combine meta (LR) with OOF-optimized L0 blend
# -----------------------------
final_prediction = 0.5 * lr_pred + 0.5 * optimized_L0_test

# (If you prefer your previous style, swap to:
# final_prediction = (lr_pred + ensemble_test) / 2.0
# or even keep your exact spec:
# final_prediction = (lr_pred + ensemble_test + optimized_L0_test) / 3.0)

# -----------------------------
# 6) Save submission (no accidental slicing)
# -----------------------------
submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
submission['loan_paid_back'] = final_prediction
submission.to_csv('submission_with_all_models.csv', index=False)
print("âœ… Saved: submission_stacked_xgb_cat_plus_ensemble.csv")




