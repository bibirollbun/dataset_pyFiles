import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import os
import cudf
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


print('Train Shape: ', train.shape)
train.head(3)


print('Test Shape: ', test.shape)
test.head(3)


TARGET = "accident_risk"
print("Train/Test shapes:", train.shape, test.shape)

train_no_id = train.drop(columns=["id", TARGET], errors="ignore")
categorical_cols = train_no_id.select_dtypes(exclude=[np.number]).columns.tolist()
numeric_cols_raw = train_no_id.select_dtypes(include=[np.number]).columns.tolist()

known_bool_candidates = ["public_road", "road_signs_present", "school_season"]
inferred_bool = []
for col in train.columns:
    if str(train[col].dtype) == "bool":
        inferred_bool.append(col)
    else:
        vals = pd.Series(train[col]).dropna().unique()
        if len(vals) > 0 and set(pd.unique(vals)).issubset({0, 1, True, False}):
            inferred_bool.append(col)
BOOL_COLS = sorted(set(inferred_bool + [c for c in known_bool_candidates if c in train.columns]))

train_encoded = train.copy()
test_encoded  = test.copy()


label_maps = {}  
for col in categorical_cols:
    le = LabelEncoder()
    tr_vals = train_encoded[col].astype(str)
    le.fit(tr_vals)
    classes = list(le.classes_)
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    unk_idx = len(classes)  

    train_encoded[col] = tr_vals.map(mapping).fillna(unk_idx).astype(int)
    test_encoded[col]  = test_encoded[col].astype(str).map(mapping).fillna(unk_idx).astype(int)

    label_maps[col] = {"mapping": mapping, "unknown_index": unk_idx}

orig = train[categorical_cols + [TARGET]].copy()
global_mean = orig[TARGET].mean()

TE = []
for c in categorical_cols:
    means = orig.groupby(c)[TARGET].mean()
    n = f"TE_{c}"
    print(f"{n}, ", end="")
    train_encoded[n] = train[c].map(means).fillna(global_mean)
    test_encoded[n]  = test[c].map(means).fillna(global_mean)
    TE.append(n)
print("\nTarget encoding features created.")


def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = a / b
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

CATEGORICAL_COLS = [c for c in categorical_cols if c in train.columns]

def add_base_fe(df: pd.DataFrame, fit_stats=None):
    out = df.copy()
    stats = {} if fit_stats is None else dict(fit_stats)

    for col in BOOL_COLS:
        if col in out.columns:
            out[col] = out[col].astype(int)

    out["curvature_speed_interaction"]     = out["curvature"] * out["speed_limit"]
    out["accidents_speed_interaction"]     = out["num_reported_accidents"] * out["speed_limit"]
    out["accidents_curvature_interaction"] = out["num_reported_accidents"] * out["curvature"]

    out["speed_per_lane"]      = safe_div(out["speed_limit"], out["num_lanes"])
    out["curvature_per_lane"]  = safe_div(out["curvature"], out["num_lanes"])
    out["accidents_per_lane"]  = safe_div(out["num_reported_accidents"], out["num_lanes"])
    out["accidents_per_speed"] = safe_div(out["num_reported_accidents"], out["speed_limit"])
    out["curvature_per_speed"] = safe_div(out["curvature"], out["speed_limit"])

    out["curvature_squared"] = out["curvature"] ** 2
    out["curvature_cubed"]   = out["curvature"] ** 3
    out["speed_squared"]     = out["speed_limit"] ** 2

    out["inv_speed"]       = 1.0 / np.where(out["speed_limit"] == 0, 1, out["speed_limit"])
    out["log1p_accidents"] = np.log1p(out["num_reported_accidents"])

    if "high_curve_thr" not in stats:
        stats["high_curve_thr"] = out["curvature"].median()
    out["is_high_speed"]     = (out["speed_limit"] >= 60).astype(int)
    out["is_high_curvature"] = (out["curvature"] > stats["high_curve_thr"]).astype(int)
    out["poor_visibility"]   = (out["lighting"] != "daylight").astype(int)
    out["bad_weather"]       = (out["weather"]  != "clear").astype(int)
    out["school_time_window"]= ((out["time_of_day"].isin(["morning","afternoon"])) & (out["school_season"]==1)).astype(int)

    out["risky_conditions_count"] = out[["is_high_speed","poor_visibility","bad_weather","school_time_window"]].sum(axis=1)

    tod_map = {"morning": 0, "afternoon": 1, "evening": 2}
    out["time_of_day_idx"] = out["time_of_day"].map(tod_map).fillna(-1).astype(int)
    angle = 2 * np.pi * out["time_of_day_idx"].clip(lower=0) / 3.0
    out["tod_sin"] = np.sin(angle)
    out["tod_cos"] = np.cos(angle)

    for col in CATEGORICAL_COLS:
        key = f"{col}_freq_map"
        if fit_stats is None:
            stats[key] = out[col].value_counts()
        out[f"{col}_freq"] = out[col].map(stats[key]).fillna(0).astype(int)

    agg_num_cols = ["speed_limit", "curvature", "num_lanes", "num_reported_accidents"]
    aggs = ["mean","std","min","max"]
    for gcol in CATEGORICAL_COLS:
        k = f"{gcol}_grp"
        if fit_stats is None:
            grp = out.groupby(gcol, dropna=False)[agg_num_cols].agg(aggs)
            grp.columns = [f"{gcol}_{num}_{stat}" for num, stat in grp.columns]
            grp = grp.reset_index()
            stats[k] = grp
        out = out.merge(stats[k], how="left", on=gcol)

    def make_edges(arr, q=8):
        qs = np.linspace(0, 1, q+1)
        e = np.quantile(arr, qs)
        e = np.unique(e)
        e[0]  -= 1e-6
        e[-1] += 1e-6
        return e

    for base_col, prefix in [("speed_limit","speed"), ("curvature","curvature")]:
        key = f"{prefix}_edges"
        if fit_stats is None:
            stats[key] = make_edges(out[base_col].values, q=8)
        out[f"{prefix}_bin"] = pd.cut(out[base_col], bins=stats[key], labels=False, include_lowest=True)
        ohe = pd.get_dummies(out[f"{prefix}_bin"], prefix=f"{prefix}_bin")
        out = pd.concat([out, ohe], axis=1)
        out.drop(columns=[f"{prefix}_bin"], inplace=True)

    ratio_cols = ["speed_per_lane","curvature_per_lane","accidents_per_lane","accidents_per_speed","curvature_per_speed"]
    for col in ratio_cols:
        lo_key, hi_key = f"{col}_lo", f"{col}_hi"
        if fit_stats is None:
            stats[lo_key] = float(np.percentile(out[col].values, 0.1))
            stats[hi_key] = float(np.percentile(out[col].values, 99.9))
        out[col] = out[col].clip(stats[lo_key], stats[hi_key])

    return out, stats

train_fe, fe_stats = add_base_fe(train, fit_stats=None)
test_fe, _ = add_base_fe(test, fit_stats=fe_stats)

new_cols = [c for c in train_fe.columns if c not in train.columns]
train_encoded = train_encoded.join(train_fe[new_cols])
test_encoded  = test_encoded.join(test_fe[new_cols])
print(f"add_base_fe: added {len(new_cols)} columns.")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['curv_speed'] = df['curvature'] * df['speed_limit']
    df['lanes_speed_ratio'] = df['num_lanes'] / (df['speed_limit'] + 1)
    df['accidents_speed'] = df['num_reported_accidents'] * df['speed_limit']
    df['accidents_curv'] = df['num_reported_accidents'] * df['curvature']
    df['curvature_sq'] = df['curvature'] ** 2
    df['curvature_cube'] = df['curvature'] ** 3
    df['risk_intensity'] = (df['curvature'] * df['speed_limit']) / 50
    df['lane_capacity_risk'] = (5 - df['num_lanes']) * df['speed_limit']
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    df['accident_density'] = df['num_reported_accidents'] / (df['num_lanes'] * df['speed_limit'] + 1)
    df['speed_squared'] = df['speed_limit'] ** 2
    df['high_risk'] = ((df['curvature'] > 0.7) & (df['speed_limit'] >= 60)).astype(int)
    df['extreme_curve'] = (df['curvature'] > 0.8).astype(int)
    df['speed_bin'] = pd.cut(df['speed_limit'], bins=[0, 35, 50, 65, 100], labels=False, right=True)
    df['curve_bin'] = pd.cut(df['curvature'], bins=[0, 0.3, 0.6, 0.8, 1.0], labels=False, right=True)
    df['speed_bin'] = df['speed_bin'].fillna(-1).astype(int)
    df['curve_bin'] = df['curve_bin'].fillna(-1).astype(int)
    return df


train_eng2 = engineer_features(train)
test_eng2  = engineer_features(test)
engineer_added_all = list(set(train_eng2.columns) - set(train.columns))
engineer_new_cols = [c for c in engineer_added_all if c not in train_encoded.columns]
train_encoded = train_encoded.join(train_eng2[engineer_new_cols])
test_encoded  = test_encoded.join(test_eng2[engineer_new_cols])
print(f"engineer_features: added {len(engineer_new_cols)} columns.")

feature_cols = (
    train_encoded
    .drop(columns=["id", TARGET], errors="ignore")
    .select_dtypes(include=[np.number])
    .columns.tolist()
)
common_cols = [c for c in feature_cols if c in test_encoded.columns]

X_full = train_encoded[common_cols].values
y_full = train_encoded[TARGET].values
X_test_full = test_encoded[common_cols].values

print(f"Total numeric features used: {len(common_cols)}")


print('Train shape: ',train_encoded.shape)


print('Test shape: ',test_encoded.shape)


train_encoded.head(3)


TARGET = "accident_risk"
if 'selected_features' in globals() and len(selected_features) > 0:
    feat_cols = selected_features
else:
    feat_cols = (
        train_encoded
        .drop(columns=["id", TARGET], errors="ignore")
        .select_dtypes(include=[np.number])
        .columns.tolist()
    )

X = train_encoded[feat_cols].values
y = train_encoded[TARGET].values
X_test = test_encoded[feat_cols].values

N_SPLITS = 5
SEED = 42
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


xgb_params = dict(
    n_estimators=6000,
    learning_rate=0.03,
    max_depth=7,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=2.0,
    gamma=0.0,
    max_bin=512,
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="gpu_hist",
)

oof_xgb = np.zeros(len(y))
xgb_rmses, xgb_best = [], []
for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False, early_stopping_rounds=400)
    pred = m.predict(X_va, iteration_range=(0, m.best_iteration + 1))
    oof_xgb[va_idx] = pred
    rmse = mean_squared_error(y_va, pred, squared=False)
    xgb_rmses.append(rmse)
    xgb_best.append(m.best_iteration + 1)
    print(f"[XGB][CV] Fold {fold}: RMSE={rmse:.6f} | best_iter={m.best_iteration + 1}")

xgb_mean_rmse = float(np.mean(xgb_rmses))
xgb_std_rmse  = float(np.std(xgb_rmses))
xgb_mean_best = int(np.round(np.mean(xgb_best)))
print(f"\n[XGB] CV RMSE: {xgb_mean_rmse:.6f} ± {xgb_std_rmse:.6f}")
print(f"[XGB] Mean best iterations: {xgb_mean_best}")


cat_params = dict(
    loss_function="RMSE",
    eval_metric="RMSE",
    task_type="GPU",
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3.0,
    subsample=0.85,              
    bootstrap_type="Bernoulli",
    early_stopping_rounds=400,
    iterations=12000,            
    random_seed=SEED,
    verbose=False
)

oof_cat = np.zeros(len(y))
cat_rmses, cat_best = [], []
for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    train_pool = Pool(X_tr, y_tr)
    valid_pool = Pool(X_va, y_va)

    cm = CatBoostRegressor(**cat_params)
    cm.fit(train_pool, eval_set=valid_pool, verbose=False)
    pred = cm.predict(valid_pool)
    oof_cat[va_idx] = pred

    rmse = mean_squared_error(y_va, pred, squared=False)
    best_iter = cm.get_best_iteration() if cm.get_best_iteration() is not None else cm.tree_count_
    cat_rmses.append(rmse)
    cat_best.append(best_iter)
    print(f"[CAT][CV] Fold {fold}: RMSE={rmse:.6f} | best_iter={best_iter}")

cat_mean_rmse = float(np.mean(cat_rmses))
cat_std_rmse  = float(np.std(cat_rmses))
cat_mean_best = int(np.round(np.mean(cat_best)))
print(f"\n[CAT] CV RMSE: {cat_mean_rmse:.6f} ± {cat_std_rmse:.6f}")
print(f"[CAT] Mean best iterations: {cat_mean_best}")


lgb_params = dict(
    n_estimators=12000,
    learning_rate=0.03,
    num_leaves=255,
    max_depth=-1,             
    min_child_samples=20,
    subsample=0.85,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=2.0,
    objective="regression",
    device_type="gpu",        
    
)

oof_lgb = np.zeros(len(y))
lgb_rmses, lgb_best = [], []
for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    lm = lgb.LGBMRegressor(**lgb_params)
    lm.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=400, verbose=False)]
    )
    best_iter = lm.best_iteration_
    pred = lm.predict(X_va, num_iteration=best_iter)
    oof_lgb[va_idx] = pred

    rmse = mean_squared_error(y_va, pred, squared=False)
    lgb_rmses.append(rmse)
    lgb_best.append(best_iter if best_iter is not None else lgb_params["n_estimators"])
    print(f"[LGB][CV] Fold {fold}: RMSE={rmse:.6f} | best_iter={best_iter}")

lgb_mean_rmse = float(np.mean(lgb_rmses))
lgb_std_rmse  = float(np.std(lgb_rmses))
lgb_mean_best = int(np.round(np.mean(lgb_best)))
print(f"\n[LGB] CV RMSE: {lgb_mean_rmse:.6f} ± {lgb_std_rmse:.6f}")
print(f"[LGB] Mean best iterations: {lgb_mean_best}")

alphas = np.linspace(0.0, 1.0, 21)  # 0..1 step 0.05
best = (None, None, 9e9)            # (axgb, acat, rmse)
for axgb in alphas:
    for acat in alphas:
        a_lgb = 1.0 - axgb - acat
        if a_lgb < 0 or a_lgb > 1:  # keep simplex
            continue
        blend = axgb * oof_xgb + acat * oof_cat + a_lgb * oof_lgb
        rmse = mean_squared_error(y, blend, squared=False)
        if rmse < best[2]:
            best = (axgb, acat, rmse)
best_axgb, best_acat, best_rmse = best
best_algb = 1.0 - best_axgb - best_acat
print(f"\n[BLEND3] Best weights -> XGB: {best_axgb:.2f}, CAT: {best_acat:.2f}, LGB: {best_algb:.2f}")
print(f"[BLEND3] OOF RMSE = {best_rmse:.6f}")


xgb_seeds = [42, 2024, 777]
cat_seeds = [42, 2024, 777]
lgb_seeds = [42, 2024, 777]


xgb_test_preds = []
for s in xgb_seeds:
    p = xgb_params.copy()
    p["random_state"] = s
    p["n_estimators"] = max(xgb_mean_best, 200)
    xm = xgb.XGBRegressor(**p)
    xm.fit(X, y, verbose=False)
    xgb_test_preds.append(xm.predict(X_test))
xgb_test_pred = np.mean(np.column_stack(xgb_test_preds), axis=1)


cat_test_preds = []
for s in cat_seeds:
    p = cat_params.copy()
    p["random_seed"] = s
    p["iterations"] = max(cat_mean_best, 400)
    cm = CatBoostRegressor(**p)
    cm.fit(Pool(X, y), verbose=False)
    cat_test_preds.append(cm.predict(Pool(X_test)))
cat_test_pred = np.mean(np.column_stack(cat_test_preds), axis=1)


lgb_test_preds = []
for s in lgb_seeds:
    p = lgb_params.copy()
    p["random_state"] = s
    p["n_estimators"] = max(lgb_mean_best, 200)
    lm = lgb.LGBMRegressor(**p)
    lm.fit(
        X, y,
        eval_set=[(X, y)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=1, verbose=False)] 
    )
    lgb_test_preds.append(lm.predict(X_test, num_iteration=lm.best_iteration_))
lgb_test_pred = np.mean(np.column_stack(lgb_test_preds), axis=1)

final_pred = best_axgb * xgb_test_pred + best_acat * cat_test_pred + best_algb * lgb_test_pred


sub = sample_sub.copy()
target_cols = [c for c in sub.columns if c != "id"]
sub_col = target_cols[0] if target_cols else TARGET
sub_xgb = sub.copy(); sub_xgb[sub_col] = xgb_test_pred; sub_xgb.to_csv("submission_xgb.csv", index=False)
sub_cat = sub.copy(); sub_cat[sub_col] = cat_test_pred; sub_cat.to_csv("submission_cat.csv", index=False)
sub_lgb = sub.copy(); sub_lgb[sub_col] = lgb_test_pred; sub_lgb.to_csv("submission_lgb.csv", index=False)


sub_blend3 = sub.copy(); sub_blend3[sub_col] = final_pred.astype(float)
sub_blend3.to_csv("submission_blend3.csv", index=False)
sub_blend3.to_csv("submission.csv", index=False)
print(sub_blend3.head())




