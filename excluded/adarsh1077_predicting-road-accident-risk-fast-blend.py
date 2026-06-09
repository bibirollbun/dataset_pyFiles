
# 1) Installs & imports
!pip -q install -U lightgbm catboost
import os, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
try:
    from lightgbm import early_stopping, log_evaluation
    LGB_HAS_CB = True
except Exception:
    LGB_HAS_CB = False
from catboost import CatBoostRegressor, Pool
np.set_printoptions(suppress=True); pd.set_option("display.max_columns", 120)
SEED = 42
print("Ready. SEED=", SEED)




DATA_DIR = ""  # set if needed
def find_file(name):
    if DATA_DIR and (Path(DATA_DIR)/name).exists(): return str(Path(DATA_DIR)/name)
    if Path(name).exists(): return name
    root = Path("/kaggle/input")
    if root.exists():
        for d in root.iterdir():
            p = d / name
            if p.exists(): return str(p)
    raise FileNotFoundError(name)
train = pd.read_csv(find_file("train.csv"))
test  = pd.read_csv(find_file("test.csv"))
print("train:", train.shape, "test:", test.shape)




IDCOL, TARGET = "id", "accident_risk"
CAT_COLS = ['holiday','lighting','public_road','road_signs_present','road_type','school_season','time_of_day','weather']
NUM_COLS = ['curvature','num_lanes','num_reported_accidents','speed_limit']

def clean_cats(df, cols):
    g = df.copy()
    for c in cols:
        if pd.api.types.is_bool_dtype(g[c]): g[c] = g[c].astype(bool)
        else: g[c] = g[c].astype(str).str.strip().str.lower()
    return g
def add_num_eng(df):
    g = df[NUM_COLS].copy()
    lanes = g['num_lanes'].replace(0,1)
    g['curvature_x_speed']  = g['curvature'] * g['speed_limit']
    g['accidents_per_lane'] = g['num_reported_accidents'] / lanes
    g['speed_per_lane']     = g['speed_limit'] / lanes
    g['curvature_sq']       = g['curvature'] ** 2
    g['log1p_accidents']    = np.log1p(g['num_reported_accidents'])
    return g.astype(np.float32)

train_c = clean_cats(train, CAT_COLS)
test_c  = clean_cats(test,  CAT_COLS)
tr_cat = pd.get_dummies(train_c[CAT_COLS], columns=CAT_COLS, dtype=np.uint8)
te_cat = pd.get_dummies(test_c[CAT_COLS],  columns=CAT_COLS, dtype=np.uint8)
tr_cat, te_cat = tr_cat.align(te_cat, join="outer", axis=1, fill_value=0)
tr_num = add_num_eng(train_c); te_num = add_num_eng(test_c)
X_train_lgb = pd.concat([tr_num, tr_cat], axis=1).astype(np.float32)
X_test_lgb  = pd.concat([te_num, te_cat], axis=1).astype(np.float32)
y = train[TARGET].astype(float)
print("LGB:", X_train_lgb.shape, X_test_lgb.shape)




PAIR_COLS = [('lighting','weather')]
def make_pair_feats(df, pairs):
    g = df.copy()
    for a,b in pairs: g[f'{a}__{b}'] = g[a].astype(str)+'|'+g[b].astype(str)
    return g
train_xc = make_pair_feats(train_c[CAT_COLS], PAIR_COLS)
test_xc  = make_pair_feats(test_c[CAT_COLS],  PAIR_COLS)
pair_names = [f'{a}__{b}' for a,b in PAIR_COLS]
CB_train = pd.concat([train_xc[CAT_COLS + pair_names].reset_index(drop=True),
                      tr_num.reset_index(drop=True)], axis=1)
CB_test  = pd.concat([test_xc[CAT_COLS + pair_names].reset_index(drop=True),
                      te_num.reset_index(drop=True)], axis=1)
cat_idx = [CB_train.columns.get_loc(c) for c in (CAT_COLS + pair_names)]
print("CB:", CB_train.shape, CB_test.shape, "#cat:", len(cat_idx))




def rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
def make_strat_bins(y, n_bins=10):
    y = pd.Series(y).astype(float)
    return pd.qcut(y.rank(method="first"), q=min(n_bins, y.nunique()), labels=False, duplicates="drop")
y_bins = make_strat_bins(y, 10)
FOLDS = 10
print("bins:", int(pd.Series(y_bins).nunique()), "folds:", FOLDS)




params_lgb = dict(objective="regression", metric="rmse",
                  n_estimators=4000, learning_rate=0.055,
                  num_leaves=64, min_data_in_leaf=100,
                  feature_fraction=0.80, bagging_fraction=0.70, bagging_freq=1,
                  max_bin=63, bin_construct_sample_cnt=200000,
                  first_metric_only=True, random_state=SEED, n_jobs=-1, verbosity=-1)
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
oof_lgb = np.zeros(len(X_train_lgb), dtype=np.float32)
test_lgb = np.zeros(len(X_test_lgb), dtype=np.float32)
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_lgb, y_bins), 1):
    X_tr, X_va = X_train_lgb.iloc[tr_idx], X_train_lgb.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    model = LGBMRegressor(**params_lgb)
    if LGB_HAS_CB:
        cbs = [early_stopping(stopping_rounds=100)]
        try: cbs.append(log_evaluation(period=50))
        except: pass
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=cbs)
    else:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    bi = getattr(model, "best_iteration_", None)
    pred_va = model.predict(X_va, num_iteration=bi) if bi else model.predict(X_va)
    oof_lgb[va_idx] = pred_va
    test_lgb += (model.predict(X_test_lgb, num_iteration=bi) if bi else model.predict(X_test_lgb)) / FOLDS
    print(f"LGB Fold {fold}: RMSE {rmse(y_va, pred_va):.6f}")
print("OOF LGB:", rmse(y, oof_lgb))




params_cb = dict(loss_function="RMSE", eval_metric="RMSE",
                 depth=7, learning_rate=0.08, l2_leaf_reg=3.0,
                 iterations=10000, random_seed=SEED,
                 task_type="GPU" if os.path.exists("/dev/nvidia0") else "CPU", devices="0",
                 rsm=0.85, border_count=64, od_type="Iter", od_wait=100, verbose=False)
oof_cb = np.zeros(len(CB_train), dtype=np.float32)
test_cb = np.zeros(len(CB_test), dtype=np.float32)
for fold, (tr_idx, va_idx) in enumerate(skf.split(CB_train, y_bins), 1):
    tr_pool = Pool(CB_train.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_idx)
    va_pool = Pool(CB_train.iloc[va_idx], y.iloc[va_idx], cat_features=cat_idx)
    cb = CatBoostRegressor(**params_cb)
    cb.fit(tr_pool, eval_set=va_pool, use_best_model=True, verbose=False)
    val_pred = cb.predict(va_pool)
    oof_cb[va_idx] = val_pred
    test_cb += cb.predict(Pool(CB_test, cat_features=cat_idx)) / FOLDS
    print(f"CB Fold {fold}: RMSE {rmse(y.iloc[va_idx], val_pred):.6f}")
print("OOF CB:", rmse(y, oof_cb))




weights = np.linspace(0,1,11)
best_w, best_rm = 0.5, 1e9
for w in weights:
    r = rmse(y, w*oof_lgb + (1-w)*oof_cb)
    if r < best_rm: best_rm, best_w = r, w
print(f"Best w(LGB)={best_w:.2f} | OOF RMSE={best_rm:.6f}")
pred = np.clip(best_w*test_lgb + (1-best_w)*test_cb, 0, 1)
sub = pd.DataFrame({'id': test[IDCOL].values, 'accident_risk': pred.astype(float)})
sub.to_csv('submission_blend_lgbm_cb_final.csv', index=False)
print('Saved: submission_blend_lgbm_cb.csv | rows:', len(sub))


