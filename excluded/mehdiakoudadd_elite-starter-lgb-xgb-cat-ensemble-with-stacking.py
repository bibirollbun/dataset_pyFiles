# ===== Runtime config (you can tweak) =====
SEED = 42
N_SPLITS = 10              # K-folds for OOF
N_BAGS = 3                 # seed-bagging for stability
USE_DURATION = 'auto'      # 'auto' | 'yes' | 'no'
POWER_AVG = 1.25           # >1 prefers confident preds
RANK_BLEND = True          # rank-average inside ensemble

import os, random, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
set_seed(SEED)
print('Config =>', dict(SEED=SEED, N_SPLITS=N_SPLITS, N_BAGS=N_BAGS, USE_DURATION=USE_DURATION, POWER_AVG=POWER_AVG, RANK_BLEND=RANK_BLEND))


# ===== Data loading with Kaggle + local fallbacks =====
def detect_paths():
    kaggle_path = '/kaggle/input/playground-series-s5e8'
    if os.path.exists(kaggle_path):
        return kaggle_path
    # Local/dev fallbacks (handles user uploads like train-3.csv)
    return '/kaggle/working' if os.path.exists('/kaggle/working') else '/mnt/data'

INPUT_DIR = detect_paths()

def find_file(candidates):
    for c in candidates:
        p = os.path.join(INPUT_DIR, c)
        if os.path.exists(p):
            return p
    # Last chance: search directory
    for f in os.listdir(INPUT_DIR):
        if f.lower().startswith(('train','test','sample')) and f.lower().endswith('.csv'):
            return os.path.join(INPUT_DIR, f)
    return None

TRAIN_F = find_file(['train.csv', 'Train.csv', 'TRAIN.csv', 'train-3.csv'])
TEST_F  = find_file(['test.csv', 'Test.csv', 'TEST.csv'])
SAMPLE_SUB_F = find_file(['sample_submission.csv', 'sample-submission.csv', 'Sample_submission.csv'])

assert TRAIN_F and os.path.exists(TRAIN_F), f"train file not found in {INPUT_DIR}"
assert TEST_F  and os.path.exists(TEST_F),  f"test file not found in {INPUT_DIR}"
print('Using files:')
print('  train =>', TRAIN_F)
print('  test  =>', TEST_F)
print('  sample=>', SAMPLE_SUB_F)

train = pd.read_csv(TRAIN_F)
test  = pd.read_csv(TEST_F)
TARGET = 'y'
ID_COL = 'id'
assert TARGET in train.columns, f"Target '{TARGET}' not in train columns: {train.columns.tolist()}"
assert ID_COL in test.columns,   f"ID column '{ID_COL}' not in test columns: {test.columns.tolist()}"
print('Shapes:', train.shape, test.shape)
train.head(3)


# ===== Feature typing & quick info =====
def infer_feature_types(df, target=TARGET, id_col=ID_COL):
    feats = [c for c in df.columns if c not in [target, id_col]]
    num_cols, cat_cols = [], []
    for c in feats:
        if pd.api.types.is_numeric_dtype(df[c]): num_cols.append(c)
        else: cat_cols.append(c)
    return num_cols, cat_cols

num_cols, cat_cols = infer_feature_types(train)
if 'duration' in num_cols:
    if USE_DURATION == 'no':
        num_cols.remove('duration')
        print('[info] dropped duration (forced no)')
    elif USE_DURATION == 'auto':
        print('[info] duration present â€” consider USE_DURATION="no" if it leaks')
    else:
        print('[info] duration kept (forced yes)')
FEATS = num_cols + cat_cols
print(f"num={len(num_cols)} cat={len(cat_cols)} total={len(FEATS)}")


# ===== Adversarial validation (train vs test shift) =====
def adversarial_validation(train_df, test_df, feats, seed=SEED):
    try:
        from lightgbm import LGBMClassifier, log_evaluation
    except Exception:
        print('[warn] LightGBM unavailable â€” skipping AV')
        return None
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import OrdinalEncoder
    import numpy as np
    X_tr = train_df[feats].copy()
    X_te = test_df[feats].copy()
    y_tr = np.zeros(len(X_tr), dtype=int)
    y_te = np.ones(len(X_te), dtype=int)
    X = pd.concat([X_tr, X_te], axis=0).reset_index(drop=True)
    y = np.concatenate([y_tr, y_te])
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    if cat_cols:
        X[cat_cols] = enc.fit_transform(X[cat_cols])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    for tr_idx, va_idx in skf.split(X, y):
        m = LGBMClassifier(n_estimators=800, num_leaves=64, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
        m.fit(X.iloc[tr_idx], y[tr_idx],
              eval_set=[(X.iloc[va_idx], y[va_idx])],
              eval_metric='auc',
              callbacks=[log_evaluation(0)])  # silence logs
        oof[va_idx] = m.predict_proba(X.iloc[va_idx])[:,1]
    auc = roc_auc_score(y, oof)
    print(f"[AV] ROC-AUC(train vs test) = {auc:.4f} (0.5 ~ similar; higher => shift)")
    return auc

_ = adversarial_validation(train[FEATS], test[FEATS], FEATS, seed=SEED)


# ===== Feature engineering =====
def add_features(df):
    out = df.copy()
    # Standardized numeric features
    for c in num_cols:
        s = out[c].std()
        if pd.notnull(s) and s > 0:
            out[f'{c}_z'] = (out[c] - out[c].mean()) / s
    # Age bins (if exists)
    if 'age' in out.columns:
        out['age_bin'] = pd.cut(out['age'], bins=[-1,25,35,45,55,65,1e9], labels=False)
    # Cast typical bank string features to category
    for col in ['month','day_of_week','contact','poutcome','job','marital','education','default','housing','loan']:
        if col in out.columns and not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype('category')
    # Count 'unknown' tokens per row
    obj_like = [c for c in out.columns if out[c].dtype.name in ('category','object')]
    if obj_like:
        out['unknown_count'] = (out[obj_like] == 'unknown').sum(axis=1)
    # Safe interactions
    if {'campaign','pdays','previous'}.issubset(out.columns):
        out['cnt_intensity'] = out['campaign'].fillna(0) + out['previous'].fillna(0)
        out['pdays_is_999']  = (out['pdays'] == 999).astype(int)
    return out

train_fe = add_features(train)
test_fe  = add_features(test)

def recompute_types(df):
    feats = [c for c in df.columns if c not in [TARGET, ID_COL]]
    num, cat = [], []
    for c in feats:
        (num if pd.api.types.is_numeric_dtype(df[c]) else cat).append(c)
    return num, cat, feats

num_cols2, cat_cols2, FEATS2 = recompute_types(train_fe)
print('Post-FE:', len(num_cols2), 'numeric,', len(cat_cols2), 'categorical, total', len(FEATS2))


# ===== OOF Target Encoding with smoothing =====
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
import numpy as np

def oof_target_encode(train_df, test_df, cat_columns, target=TARGET, n_splits=N_SPLITS, seed=SEED, smoothing=20.0):
    if not cat_columns:
        return train_df.copy(), test_df.copy(), []
    global_mean = train_df[target].mean()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr = train_df.copy(); te = test_df.copy(); new_cols = []
    for col in cat_columns:
        new = f'te_{col}'; new_cols.append(new); tr[new] = np.nan
        for tr_idx, va_idx in skf.split(train_df, train_df[target]):
            fold_tr = train_df.iloc[tr_idx]
            stats = fold_tr.groupby(col)[target].agg(['mean','count']).rename(columns={'mean':'m','count':'n'})
            stats['te'] = (stats['n']*stats['m'] + smoothing*global_mean) / (stats['n'] + smoothing)
            tr.loc[va_idx, new] = train_df.iloc[va_idx][col].map(stats['te']).fillna(global_mean).values
        stats_full = train_df.groupby(col)[target].agg(['mean','count']).rename(columns={'mean':'m','count':'n'})
        stats_full['te'] = (stats_full['n']*stats_full['m'] + smoothing*global_mean) / (stats_full['n'] + smoothing)
        te[new] = test_df[col].map(stats_full['te']).fillna(global_mean).values
    return tr, te, new_cols

# Ordinal encode cats for model compatibility
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_enc = train_fe.copy(); test_enc = test_fe.copy()
if cat_cols2:
    train_enc[cat_cols2] = enc.fit_transform(train_enc[cat_cols2])
    test_enc[cat_cols2]  = enc.transform(test_enc[cat_cols2])

train_te, test_te, te_cols = oof_target_encode(train_enc, test_enc, cat_columns=cat_cols2, target=TARGET)
BASE_FEATS = [c for c in FEATS2 if c != TARGET]
ALL_FEATS  = list(dict.fromkeys(BASE_FEATS + te_cols))
print('ALL_FEATS =', len(ALL_FEATS))


# ===== Models: LGBM, XGB, Cat (LightGBM fixed with callbacks) =====
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

def fit_predict_lgb(train_df, test_df, feats, seed=SEED, n_splits=N_SPLITS):
    from lightgbm import LGBMClassifier, log_evaluation, early_stopping
    oof = np.zeros(len(train_df)); preds = np.zeros(len(test_df))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df[feats], train_df[TARGET])):
        X_tr, X_va = train_df.iloc[tr_idx][feats], train_df.iloc[va_idx][feats]
        y_tr, y_va = train_df.iloc[tr_idx][TARGET], train_df.iloc[va_idx][TARGET]
        model = LGBMClassifier(n_estimators=3000, learning_rate=0.02, num_leaves=64,
                               subsample=0.8, colsample_bytree=0.7, min_child_samples=50,
                               reg_lambda=2.0, random_state=seed+fold, n_jobs=-1)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  eval_metric='auc',
                  callbacks=[early_stopping(200), log_evaluation(0)])
        oof[va_idx] = model.predict_proba(X_va)[:,1]
        preds += model.predict_proba(test_df[feats])[:,1] / n_splits
    auc = roc_auc_score(train_df[TARGET], oof)
    print(f'[LGB] OOF AUC={auc:.6f}')
    return oof, preds

def fit_predict_xgb(train_df, test_df, feats, seed=SEED, n_splits=N_SPLITS):
    from xgboost import XGBClassifier
    oof = np.zeros(len(train_df)); preds = np.zeros(len(test_df))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df[feats], train_df[TARGET])):
        X_tr, X_va = train_df.iloc[tr_idx][feats], train_df.iloc[va_idx][feats]
        y_tr, y_va = train_df.iloc[tr_idx][TARGET], train_df.iloc[va_idx][TARGET]
        model = XGBClassifier(n_estimators=2500, max_depth=7, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
                              min_child_weight=30, tree_method='hist', random_state=seed+fold, n_jobs=-1)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc', verbose=False)
        oof[va_idx] = model.predict_proba(X_va)[:,1]
        preds += model.predict_proba(test_df[feats])[:,1] / n_splits
    auc = roc_auc_score(train_df[TARGET], oof)
    print(f'[XGB] OOF AUC={auc:.6f}')
    return oof, preds

def fit_predict_cat(train_df, test_df, feats, seed=SEED, n_splits=N_SPLITS):
    from catboost import CatBoostClassifier, Pool
    oof = np.zeros(len(train_df)); preds = np.zeros(len(test_df))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df[feats], train_df[TARGET])):
        X_tr, X_va = train_df.iloc[tr_idx][feats], train_df.iloc[va_idx][feats]
        y_tr, y_va = train_df.iloc[tr_idx][TARGET], train_df.iloc[va_idx][TARGET]
        train_pool = Pool(X_tr, y_tr); valid_pool = Pool(X_va, y_va)
        model = CatBoostClassifier(iterations=3000, depth=8, learning_rate=0.03,
                                   l2_leaf_reg=6.0, random_seed=seed+fold,
                                   loss_function='Logloss', eval_metric='AUC', verbose=False)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=False)
        oof[va_idx] = model.predict_proba(X_va)[:,1]
        preds += model.predict_proba(test_df[feats])[:,1] / n_splits
    auc = roc_auc_score(train_df[TARGET], oof)
    print(f'[CAT] OOF AUC={auc:.6f}')
    return oof, preds


# ===== Blending & stacking =====
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

def power_mean(p, pow_k=POWER_AVG, axis=0):
    p = np.clip(np.asarray(p), 1e-9, 1-1e-9)
    if pow_k == 1.0: return np.mean(p, axis=axis)
    return (np.mean(np.power(p, pow_k), axis=axis)) ** (1.0 / pow_k)

def rank_average(preds_matrix, axis=0):
    ranks = []
    for p in preds_matrix:
        r = pd.Series(p).rank(method='average') / len(p)
        ranks.append(r.values)
    return np.mean(np.vstack(ranks), axis=axis)

def blend_predictions(preds_list, use_rank=RANK_BLEND, pow_k=POWER_AVG):
    return rank_average(preds_list, axis=0) if use_rank else power_mean(preds_list, pow_k=pow_k, axis=0)

def stack_oof(oof_dict, y_true):
    X_stack = pd.DataFrame(oof_dict)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_stack, y_true)
    print('[STACK] Coefs:', dict(zip(X_stack.columns, lr.coef_.ravel())))
    return lr


# ===== Train with seed-bagging and ensemble =====
all_oof = []; all_test = []; labels = []
for b in range(N_BAGS):
    bag_seed = SEED + 1000*b
    set_seed(bag_seed)
    print(f"\n=== Bag {b+1}/{N_BAGS} | seed={bag_seed} ===")
    oof_lgb, pb_lgb = fit_predict_lgb(train_te, test_te, ALL_FEATS, seed=bag_seed, n_splits=N_SPLITS)
    oof_xgb, pb_xgb = fit_predict_xgb(train_te, test_te, ALL_FEATS, seed=bag_seed, n_splits=N_SPLITS)
    oof_cat, pb_cat = fit_predict_cat(train_te, test_te, ALL_FEATS, seed=bag_seed, n_splits=N_SPLITS)
    all_oof.extend([oof_lgb, oof_xgb, oof_cat])
    all_test.extend([pb_lgb, pb_xgb, pb_cat])
    labels.extend([f'lgb_b{b}', f'xgb_b{b}', f'cat_b{b}'])

oof_blend = blend_predictions(all_oof, use_rank=RANK_BLEND, pow_k=POWER_AVG)
auc_blend = roc_auc_score(train_te[TARGET], oof_blend)
print(f"\n[BLEND] OOF AUC={auc_blend:.6f} | models={labels} | rank_blend={RANK_BLEND} | power={POWER_AVG}")

oof_dict = {lab: all_oof[i] for i, lab in enumerate(labels)}
stacker = stack_oof(oof_dict, train_te[TARGET].values)
X_stack_test = pd.DataFrame({lab: p for lab, p in zip(labels, all_test)})
stack_preds = stacker.predict_proba(X_stack_test)[:,1]

final_preds = blend_predictions([rank_average(all_test), stack_preds], use_rank=True, pow_k=POWER_AVG)
print('Done ensembling.')


# ===== Export submission =====
sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: final_preds})
sub = sub.sort_values(ID_COL).reset_index(drop=True)
sub.to_csv('submission.csv', index=False)
sub.head()

