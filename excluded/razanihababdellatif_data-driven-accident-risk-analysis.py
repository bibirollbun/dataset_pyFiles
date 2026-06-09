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


# ============================================================
# ğŸ�� PREDICT ROAD ACCIDENT RISK â€” TOP KAGGLER PIPELINE
# ============================================================

# # ğŸ”¹ Imports
# import os
# import gc
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from dataclasses import dataclass, field
# from typing import List, Dict, Tuple
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# import lightgbm as lgb
# import logging

# # ğŸ”¹ Logging
# log = logging.getLogger('accident_risk')
# logging.basicConfig(level=logging.INFO, format='%(message)s')

# # ğŸ”¹ Reproducible Seed
# RANDOM_SEED = 42
# np.random.seed(RANDOM_SEED)

# ============================================================
# ğŸ”¹ Load Data
# ============================================================
# DATA_DIR = "/kaggle/input/playground-series-s5e10"
# train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
# test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# ============================================================
# ğŸ”¹ Sanity Checks
# ============================================================
# assert 'id' in train.columns and 'accident_risk' in train.columns, "Train missing required columns"
# assert 'id' in test.columns, "Test missing 'id' column"
# train.drop_duplicates(subset=['id'], inplace=True)
# test.drop_duplicates(subset=['id'], inplace=True)
# assert not train['accident_risk'].isna().any(), "Target contains NaNs"

# ============================================================
# ğŸ”¹ Feature Identification
# ============================================================
# NUMERIC_DTYPES = ['int16','int32','int64','float16','float32','float64']

# numeric_features = [c for c in train.columns if train[c].dtype.name in NUMERIC_DTYPES and c not in ['id','accident_risk']]
# categorical_features = [c for c in train.columns if c not in numeric_features + ['id','accident_risk']]

# # Convert low-cardinality numeric to categorical
# for c in numeric_features.copy():
#     if train[c].nunique() < 15:
#         numeric_features.remove(c)
#         categorical_features.append(c)
#         train[c] = train[c].astype('category')
#         test[c] = test[c].astype('category')

# log.info(f"Numeric: {numeric_features}")
# log.info(f"Categorical: {categorical_features}")

# ============================================================
# ğŸ”¹ Preprocessing Class
# ============================================================
# @dataclass
# class Preprocessor:
#     numeric_features: List[str]
#     categorical_features: List[str]
#     fill_numeric_with: float = 0.0
#     le_map: Dict[str, LabelEncoder] = field(default_factory=dict)

#     def fit_transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         train = train_df.copy()
#         test = test_df.copy()
#         # Numeric
#         for c in self.numeric_features:
#             fill_val = train[c].median() if not np.isnan(train[c].median()) else self.fill_numeric_with
#             train[c] = train[c].fillna(fill_val)
#             test[c] = test[c].fillna(fill_val)
#         # Categorical
#         for c in self.categorical_features:
#             le = LabelEncoder()
#             combined = pd.concat([train[c].astype(str), test[c].astype(str)], axis=0).fillna('missing')
#             le.fit(combined)
#             train[c] = le.transform(train[c].astype(str).fillna('missing'))
#             test[c] = le.transform(test[c].astype(str).fillna('missing'))
#             self.le_map[c] = le
#         return train, test

# pp = Preprocessor(numeric_features, categorical_features)
# train_p, test_p = pp.fit_transform(train.drop(columns=['accident_risk']), test)
# train_p['accident_risk'] = train['accident_risk'].values

# ============================================================
# ğŸ”¹ Smoothed Target Encoding (High-Cardinality)
# ============================================================
# HIGH_CARD_THRESHOLD = 30

# def target_encode_kfold_smooth(train_df, test_df, feature, target, n_splits=5, noise=0.01, smoothing_param=10, seed=42):
#     oof = np.zeros(len(train_df))
#     test_encoded = np.zeros(len(test_df))
#     global_mean = train_df[target].mean()
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
#     for tr_idx, val_idx in kf.split(train_df):
#         X_tr, X_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
#         means = X_tr.groupby(feature)[target].mean()
#         counts = X_tr.groupby(feature)[target].count()
#         smoothing = 1 / (1 + np.exp(-(counts - smoothing_param)/smoothing_param))
#         smooth_means = global_mean*(1-smoothing) + means*smoothing
#         X_val_encoded = X_val[feature].map(smooth_means)
#         oof[val_idx] = X_val_encoded.fillna(global_mean).values
#     means = train_df.groupby(feature)[target].mean()
#     counts = train_df.groupby(feature)[target].count()
#     smoothing = 1 / (1 + np.exp(-(counts - smoothing_param)/smoothing_param))
#     smooth_means = global_mean*(1-smoothing) + means*smoothing
#     test_encoded = test_df[feature].map(smooth_means).fillna(global_mean).values
#     rng = np.random.RandomState(seed)
#     oof += rng.normal(0, noise, len(oof))
#     test_encoded += rng.normal(0, noise, len(test_encoded))
#     return oof, test_encoded

# encoded_features = []
# for c in categorical_features:
#     if train[c].nunique() > HIGH_CARD_THRESHOLD:
#         te_train, te_test = target_encode_kfold_smooth(train_p, test_p, c, 'accident_risk', seed=RANDOM_SEED)
#         train_p[c+'_te'] = te_train
#         test_p[c+'_te'] = te_test
#         encoded_features.append(c+'_te')

# ============================================================
# ğŸ”¹ Interaction Features
# ============================================================
# MAX_INTERACTIONS = 3
# cats_sorted = sorted(categorical_features, key=lambda c: train[c].nunique())
# cats_to_use = cats_sorted[:MAX_INTERACTIONS]

# interaction_pairs = []
# for i in range(len(cats_to_use)):
#     for j in range(i+1,len(cats_to_use)):
#         a,b = cats_to_use[i], cats_to_use[j]
#         new_col = f"{a}_x_{b}"
#         train_p[new_col] = train_p[a].astype(str) + "_" + train_p[b].astype(str)
#         test_p[new_col] = test_p[a].astype(str) + "_" + test_p[b].astype(str)
#         le = LabelEncoder()
#         combined = pd.concat([train_p[new_col], test_p[new_col]], axis=0)
#         le.fit(combined)
#         train_p[new_col] = le.transform(train_p[new_col])
#         test_p[new_col] = le.transform(test_p[new_col])
#         interaction_pairs.append(new_col)

# ============================================================
# ğŸ”¹ Final Feature List
# ============================================================
# exclude_cols = ['id','accident_risk']
# all_features = [c for c in train_p.columns if c not in exclude_cols and c in test_p.columns]

# ============================================================
# ğŸ”¹ Multi-Seed LightGBM + Ridge Ensemble
# ============================================================
# SEEDS = [42, 7, 2025]
# oof_preds = np.zeros(len(train_p))
# test_preds = np.zeros((len(test_p), len(SEEDS)))

# for s_idx, seed in enumerate(SEEDS):
#     kf = KFold(n_splits=5, shuffle=True, random_state=seed)
#     fold_test_preds = np.zeros((len(test_p), 5))
#     for fold, (tr_idx, val_idx) in enumerate(kf.split(train_p)):
#         tr_data = lgb.Dataset(train_p.iloc[tr_idx][all_features], label=train_p.iloc[tr_idx]['accident_risk'])
#         val_data = lgb.Dataset(train_p.iloc[val_idx][all_features], label=train_p.iloc[val_idx]['accident_risk'])
#         params = {
#             'objective':'regression','metric':'rmse','boosting_type':'gbdt',
#             'learning_rate':0.05,'num_leaves':64,'feature_fraction':0.9,
#             'bagging_fraction':0.9,'bagging_freq':5,'min_child_samples':20,
#             'verbosity':-1,'n_jobs':-1,'seed':seed
#         }
#         model = lgb.train(params, tr_data, valid_sets=[val_data], num_boost_round=5000,
#                           callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)])
#         oof_preds[val_idx] = model.predict(train_p.iloc[val_idx][all_features], num_iteration=model.best_iteration)
#         fold_test_preds[:, fold] = model.predict(test_p[all_features], num_iteration=model.best_iteration)
#         del model, tr_data, val_data; gc.collect()
#     test_preds[:, s_idx] = fold_test_preds.mean(axis=1)

# # Average LightGBM predictions
# test_pred_lgb = test_preds.mean(axis=1)

# # Ridge meta-ensemble
# ridge = Ridge(alpha=1.0)
# ridge.fit(oof_preds.reshape(-1,1), train_p['accident_risk'])
# meta_pred = ridge.predict(test_pred_lgb.reshape(-1,1))

# final_pred = 0.7*test_pred_lgb + 0.3*meta_pred

# ============================================================
# ğŸ”¹ Submission
# ============================================================
# submission = pd.DataFrame({
#     'id': test['id'],
#     'accident_risk': np.clip(final_pred,0,1)
# })
# submission.to_csv('/kaggle/working/submission.csv', index=False)

# print("\nâœ… Final submission preview:")
# display(submission.head(10))



"""
# ============================================================
# ğŸ�� PREDICT ROAD ACCIDENT RISK â€” TOP KAGGLER PIPELINE (IMPROVED)
# Same approach: Multi-seed LightGBM -> Ridge meta. Cleaner, safer, stronger.
# ============================================================

# ğŸ”¹ Imports
import os
import gc
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import logging

# Optional nice progress
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = lambda x: x

# ğŸ”¹ Logging (clean, consistent)
log = logging.getLogger("accident_risk")
if not log.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    log.addHandler(ch)
log.setLevel(logging.INFO)

# ğŸ”¹ Reproducible Seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================
# ğŸ”¹ Load Data
# ============================================================
DATA_DIR = "/kaggle/input/playground-series-s5e10"  # change if needed
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

log.info(f"Train shape: {train.shape} | Test shape: {test.shape}")

# ============================================================
# ğŸ”¹ Sanity Checks & basic cleanup
# ============================================================
assert 'id' in train.columns and 'accident_risk' in train.columns, "Train missing required columns"
assert 'id' in test.columns, "Test missing 'id' column"

train.drop_duplicates(subset=['id'], inplace=True)
test.drop_duplicates(subset=['id'], inplace=True)
assert not train['accident_risk'].isna().any(), "Target contains NaNs"

# ============================================================
# ğŸ”¹ Feature Identification
# ============================================================
NUMERIC_DTYPES = ['int16','int32','int64','float16','float32','float64']
numeric_features = [c for c in train.columns if train[c].dtype.name in NUMERIC_DTYPES and c not in ['id','accident_risk']]
categorical_features = [c for c in train.columns if c not in numeric_features + ['id','accident_risk']]

# Convert low-cardinality numeric -> category (safe)
for c in numeric_features.copy():
    if train[c].nunique() < 15:
        numeric_features.remove(c)
        categorical_features.append(c)
        train[c] = train[c].astype('category')
        test[c] = test[c].astype('category')

log.info(f"Numeric features ({len(numeric_features)}): {numeric_features}")
log.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# ============================================================
# ğŸ”¹ Preprocessing Class (safe, repeatable label encoding)
# ============================================================
@dataclass
class Preprocessor:
    numeric_features: List[str]
    categorical_features: List[str]
    fill_numeric_with: float = 0.0
    le_map: Dict[str, LabelEncoder] = field(default_factory=dict)

    def fit_transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train = train_df.copy().reset_index(drop=True)
        test = test_df.copy().reset_index(drop=True)

        # Numeric fills with median (train median)
        for c in self.numeric_features:
            if c not in train.columns:
                continue
            median_val = train[c].median() if not train[c].dropna().empty else self.fill_numeric_with
            train[c] = train[c].fillna(median_val)
            test[c] = test[c].fillna(median_val)

        # Categorical: safe LabelEncoder fitted on combined (stringified) with 'missing' for NaNs
        for c in self.categorical_features:
            if c not in train.columns:
                continue
            le = LabelEncoder()
            combined = pd.concat([train[c].astype(str), test[c].astype(str)], axis=0).fillna('missing')
            combined = combined.fillna('missing').astype(str)
            le.fit(combined)
            train[c] = le.transform(train[c].astype(str).fillna('missing'))
            test[c] = le.transform(test[c].astype(str).fillna('missing'))
            self.le_map[c] = le

        return train, test

# Fit preprocessor
pp = Preprocessor(numeric_features=numeric_features, categorical_features=categorical_features)
train_p, test_p = pp.fit_transform(train.drop(columns=['accident_risk']), test)
train_p['accident_risk'] = train['accident_risk'].values

# ============================================================
# ğŸ”¹ Smoothed Target Encoding (KFold)
# ============================================================
HIGH_CARD_THRESHOLD = 30

def target_encode_kfold_smooth(train_df, test_df, feature, target, n_splits=5, noise=0.01, smoothing_k=20, min_samples_leaf=1, seed=42):
    rng = np.random.RandomState(seed)
    oof = np.zeros(len(train_df))
    test_encoded = np.zeros(len(test_df))
    global_mean = train_df[target].mean()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(train_df):
        X_tr = train_df.iloc[tr_idx]
        X_val = train_df.iloc[val_idx]
        stats = X_tr.groupby(feature)[target].agg(['mean','count']).rename(columns={'mean':'cat_mean','count':'cat_count'})
        stats['cat_count'] = stats['cat_count'].clip(lower=min_samples_leaf)
        stats['smoothing'] = stats['cat_count'] / (stats['cat_count'] + smoothing_k)
        stats['smooth_mean'] = global_mean * (1 - stats['smoothing']) + stats['cat_mean'] * stats['smoothing']
        X_val_enc = X_val[feature].map(stats['smooth_mean'])
        oof[val_idx] = X_val_enc.fillna(global_mean).values

    stats_full = train_df.groupby(feature)[target].agg(['mean','count']).rename(columns={'mean':'cat_mean','count':'cat_count'})
    stats_full['cat_count'] = stats_full['cat_count'].clip(lower=min_samples_leaf)
    stats_full['smoothing'] = stats_full['cat_count'] / (stats_full['cat_count'] + smoothing_k)
    stats_full['smooth_mean'] = global_mean * (1 - stats_full['smoothing']) + stats_full['cat_mean'] * stats_full['smoothing']
    test_encoded = test_df[feature].map(stats_full['smooth_mean']).fillna(global_mean).values

    scale = train_df[target].std() if not math.isnan(train_df[target].std()) else 1.0
    oof += rng.normal(0, noise * scale, len(oof))
    test_encoded += rng.normal(0, noise * scale, len(test_encoded))
    return oof, test_encoded

encoded_features = []
for c in categorical_features:
    if c in train_p.columns and train_p[c].nunique() > HIGH_CARD_THRESHOLD:
        log.info(f"Target-encoding high-card categorical: {c}")
        te_train, te_test = target_encode_kfold_smooth(
            train_df=pd.concat([train_p, train['accident_risk']], axis=1),
            test_df=test_p,
            feature=c,
            target='accident_risk'
        )
        train_p[c + '_te'] = te_train
        test_p[c + '_te'] = te_test
        encoded_features.append(c + '_te')

# ============================================================
# ğŸ”¹ Interaction Features (limited to few pairs)
# ============================================================
MAX_INTERACTIONS = 3
cats_sorted = sorted([c for c in categorical_features if c in train_p.columns], key=lambda c: train[c].nunique())
cats_to_use = cats_sorted[:MAX_INTERACTIONS]

interaction_pairs = []
for i in range(len(cats_to_use)):
    for j in range(i+1, len(cats_to_use)):
        a, b = cats_to_use[i], cats_to_use[j]
        new_col = f"{a}_x_{b}"
        train_p[new_col] = train_p[a].astype(str) + "_" + train_p[b].astype(str)
        test_p[new_col] = test_p[a].astype(str) + "_" + test_p[b].astype(str)
        le = LabelEncoder()
        combined = pd.concat([train_p[new_col], test_p[new_col]], axis=0).astype(str)
        le.fit(combined)
        train_p[new_col] = le.transform(train_p[new_col].astype(str))
        test_p[new_col] = le.transform(test_p[new_col].astype(str))
        interaction_pairs.append(new_col)

log.info(f"Created interactions: {interaction_pairs}")

# ============================================================
# ğŸ”¹ Feature Pruning
# ============================================================
exclude_cols = ['id','accident_risk']
all_candidate_features = [c for c in train_p.columns if c not in exclude_cols and c in test_p.columns]
non_constant = [c for c in all_candidate_features if train_p[c].nunique() > 1]
all_candidate_features = non_constant

low_var_thresh = 1e-6
vars_kept = [c for c in all_candidate_features if train_p[c].std() > low_var_thresh]
all_candidate_features = vars_kept

corr_thresh = 0.98
num_like = [c for c in all_candidate_features if train_p[c].dtype.name in NUMERIC_DTYPES or c.endswith('_te')]
if len(num_like) > 1:
    corr = train_p[num_like].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_thresh)]
    all_candidate_features = [c for c in all_candidate_features if c not in to_drop]

all_features = all_candidate_features
log.info(f"Final feature count used for modeling: {len(all_features)}")

# ============================================================
# ğŸ”¹ Multi-Seed LightGBM + Ridge Ensemble
# ============================================================
SEEDS = [42, 7, 2025]
n_splits = 5
oof_matrix = np.zeros((len(train_p), len(SEEDS)))
test_matrix = np.zeros((len(test_p), len(SEEDS)))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'verbosity': -1,
    'n_jobs': -1
}

seed_fold_scores = defaultdict(list)

for s_idx, seed in enumerate(SEEDS):
    log.info(f"=== SEED {seed} / {len(SEEDS)} ===")
    oof_preds_seed = np.zeros(len(train_p))
    fold_test_preds = np.zeros((len(test_p), n_splits))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_p)):
        t0 = time.time()
        tr_x = train_p.iloc[tr_idx][all_features]
        tr_y = train_p.iloc[tr_idx]['accident_risk']
        val_x = train_p.iloc[val_idx][all_features]
        val_y = train_p.iloc[val_idx]['accident_risk']

        dtrain = lgb.Dataset(tr_x, label=tr_y)
        dvalid = lgb.Dataset(val_x, label=val_y)

        params = lgb_params.copy()
        params['seed'] = seed + fold

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid],
            num_boost_round=5000,
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
        )

        best_iter = model.best_iteration
        oof_preds_seed[val_idx] = model.predict(val_x, num_iteration=best_iter)
        fold_test_preds[:, fold] = model.predict(test_p[all_features], num_iteration=best_iter)

        fold_rmse = mean_squared_error(val_y, oof_preds_seed[val_idx], squared=False)
        seed_fold_scores[seed].append(fold_rmse)
        log.info(f"Seed {seed} Fold {fold+1}/{n_splits} RMSE: {fold_rmse:.6f} | iters: {best_iter} | time: {time.time()-t0:.1f}s")

        del model, dtrain, dvalid, tr_x, tr_y, val_x, val_y
        gc.collect()

    test_matrix[:, s_idx] = fold_test_preds.mean(axis=1)
    oof_matrix[:, s_idx] = oof_preds_seed
    seed_avg = np.mean(seed_fold_scores[seed])
    log.info(f"Seed {seed} average CV RMSE: {seed_avg:.6f}")

test_pred_lgb = test_matrix.mean(axis=1)
oof_pred_lgb_mean = oof_matrix.mean(axis=1)
overall_cv_rmse = mean_squared_error(train_p['accident_risk'], oof_pred_lgb_mean, squared=False)
log.info(f"LightGBM multi-seed CV RMSE (mean of seeds): {overall_cv_rmse:.6f}")

# ============================================================
# ğŸ”¹ Ridge meta-ensemble
# ============================================================
scaler = StandardScaler()
oof_meta = scaler.fit_transform(oof_matrix)
test_meta = scaler.transform(test_matrix)

ridge = Ridge(alpha=1.0, random_state=RANDOM_SEED)
ridge.fit(oof_meta, train_p['accident_risk'])
meta_pred = ridge.predict(test_meta)

meta_oof_pred = ridge.predict(oof_meta)
meta_cv_rmse = mean_squared_error(train_p['accident_risk'], meta_oof_pred, squared=False)
log.info(f"Ridge meta CV RMSE (on oof_meta): {meta_cv_rmse:.6f}")

final_pred = 0.7 * test_pred_lgb + 0.3 * meta_pred

# ============================================================
# ğŸ”¹ Submission
# ============================================================
submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': np.clip(final_pred, 0, 1)
})

out_path = '/kaggle/working/submission.csv'
submission.to_csv(out_path, index=False)
log.info(f"Saved submission to {out_path}")

print("\\nâœ… Final submission preview (top 10):")
display(submission.head(10))

# ============================================================
# ğŸ”¹ Feature Importance Snapshot
# ============================================================
log.info("Top 10 features by importance (from last fold of last seed):")
try:
    tmp_idx = next(iter(KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(train_p)))[0]
    tmp_model = lgb.train({**lgb_params, 'seed': RANDOM_SEED}, lgb.Dataset(train_p.iloc[tmp_idx][all_features], label=train_p.iloc[tmp_idx]['accident_risk']), num_boost_round=200)
    imp = pd.DataFrame({'feature': all_features, 'importance': tmp_model.feature_importance(importance_type='gain')})
    imp = imp.sort_values('importance', ascending=False).reset_index(drop=True)
    display(imp.head(20))
    del tmp_model
    gc.collect()
except Exception as e:
    log.warning(f"Could not compute importances: {e}")

del oof_matrix, test_matrix, oof_meta, test_meta
gc.collect()
log.info("Pipeline finished successfully.")
"""




# ============================================================
# ğŸš¦ ROAD ACCIDENT RISK PREDICTION â€” STACKED ENSEMBLE SOLUTION
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import itertools

# ============================================================
# ğŸ“‚ Load Data
# ============================================================
train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

TARGET = "accident_risk"
categorical_features = ["road_type", "lighting", "weather", "time_of_day", "holiday", "school_season"]

# ============================================================
# ğŸ”  Encode Categorical Columns
# ============================================================
for col in categorical_features:
    encoder = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    encoder.fit(combined)
    train_df[col] = encoder.transform(train_df[col].astype(str))
    test_df[col] = encoder.transform(test_df[col].astype(str))

# ============================================================
# âš™ï¸� Feature Engineering
# ============================================================
for dataset in [train_df, test_df]:
    dataset["speed_curvature"] = dataset["speed_limit"] * dataset["curvature"]
    dataset["lane_density"] = dataset["num_lanes"] / (dataset["speed_limit"] + 1)
    dataset["accidents_per_lane"] = dataset["num_reported_accidents"] / (dataset["num_lanes"] + 1)

features = [f for f in train_df.columns if f not in ["id", TARGET]]

# ============================================================
# ğŸ§  Model Configuration
# ============================================================
params_lgb = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.02,
    "num_leaves": 150,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "random_state": 42,
    "n_estimators": 3000
}

params_cb = {
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 8,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": False
}

params_xgb = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 3000,
    "random_state": 42
}

# ============================================================
# ğŸ”� K-Fold Training
# ============================================================
folds = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = {"lgb": np.zeros(len(train_df)), "cb": np.zeros(len(train_df)), "xgb": np.zeros(len(train_df))}
test_preds = {"lgb": np.zeros(len(test_df)), "cb": np.zeros(len(test_df)), "xgb": np.zeros(len(test_df))}

for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(train_df)):
    print(f"\nğŸŒ¿ Fold {fold_idx + 1}")
    
    X_tr, X_val = train_df.iloc[train_idx][features], train_df.iloc[valid_idx][features]
    y_tr, y_val = train_df.iloc[train_idx][TARGET], train_df.iloc[valid_idx][TARGET]
    
    # LightGBM
    model_lgb = lgb.LGBMRegressor(**params_lgb)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    oof_preds["lgb"][valid_idx] = model_lgb.predict(X_val)
    test_preds["lgb"] += model_lgb.predict(test_df[features]) / folds.n_splits

    # CatBoost
    model_cb = CatBoostRegressor(**params_cb)
    model_cb.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=100
    )
    oof_preds["cb"][valid_idx] = model_cb.predict(X_val)
    test_preds["cb"] += model_cb.predict(test_df[features]) / folds.n_splits

    # XGBoost
    model_xgb = XGBRegressor(**params_xgb)
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=False
    )
    oof_preds["xgb"][valid_idx] = model_xgb.predict(X_val)
    test_preds["xgb"] += model_xgb.predict(test_df[features]) / folds.n_splits

# ============================================================
# ğŸ§© Stacking Layer
# ============================================================
train_stack = pd.DataFrame({k: v for k, v in oof_preds.items()})
test_stack = pd.DataFrame({k: v for k, v in test_preds.items()})

def build_interaction_features(df):
    out = df.copy()
    for c1, c2 in itertools.combinations(df.columns, 2):
        out[f"{c1}_x_{c2}"] = df[c1] * df[c2]
    out["stack_mean"] = df.mean(axis=1)
    out["stack_std"] = df.std(axis=1)
    return out

train_ext = build_interaction_features(train_stack)
test_ext = build_interaction_features(test_stack)

print(f"\nğŸ§± Base models: {train_stack.shape[1]}")
print(f"ğŸ§® Extended meta features: {train_ext.shape[1]}")

# ============================================================
# ğŸ”¹ RidgeCV Meta Model
# ============================================================
ridge_alphas = [1e-3, 1e-2, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0]
meta_model = RidgeCV(alphas=ridge_alphas, scoring="neg_root_mean_squared_error", cv=5)
meta_model.fit(train_ext, train_df[TARGET])

oof_final = meta_model.predict(train_ext)
pred_final = meta_model.predict(test_ext)

rmse_score = mean_squared_error(train_df[TARGET], oof_final, squared=False)
print("\n=================================")
print(f"ğŸ”¸ Final Stacking RMSE: {rmse_score:.5f}")
print(f"ğŸ”¸ Best Alpha: {meta_model.alpha_}")
print("=================================")

# ============================================================
# ğŸ“¤ Create Submission
# ============================================================
submission = pd.DataFrame({
    "id": test_df["id"],
    "accident_risk": pred_final
})
submission.to_csv("submission.csv", index=False)
print("\nâœ… submission.csv successfully saved!")



submission.head(10)

