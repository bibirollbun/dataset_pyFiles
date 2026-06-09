import os, gc, time, sys, math, json, warnings, pathlib, textwrap, random
warnings.filterwarnings("ignore")

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from catboost import CatBoostRegressor, Pool
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, PolynomialFeatures

VERSION = "04"               
CUTOFF_HOURS = 11.5          # stop training after this many hours (saves progress)
N_SPLITS = 5
SEED = 42

# "lgbm", "xgb", "cat", "hgb", "rf", "et", "gbr", "ridge", "lasso", "enet"
MODELS_TO_RUN = ["lgbm", "cat", "hgb"]

GPU_MODEL_SET  = ["lgbm", "xgb", "cat", "hgb"]       # may leverage GPU
CPU_MODEL_SET  = ["rf", "et", "gbr", "ridge", "lasso", "enet"]

# Preprocessing grid (edit this list to try other combos)
# encoder_type: "native_cats" (for LGB/CAT), "onehot", "ordinal", "target_mean", "freq"
# num_transform: list among ["none","log1p_acc","poly","bin_curvature","standardize","interact_cs"]
# rare_thresh: merge categories < frac into "RARE" (for non-native encoders only)
PREPROCESS_TO_RUN = [
    {"key":"native_base",  "encoder_type":"native_cats", "num_transform":["none"],                      "rare_thresh":None},
    {"key":"oh_base",      "encoder_type":"onehot",      "num_transform":["none"],                      "rare_thresh":None},
    {"key":"oh_rare",      "encoder_type":"onehot",      "num_transform":["none"],                      "rare_thresh":0.005},
    {"key":"oh_poly",      "encoder_type":"onehot",      "num_transform":["poly","interact_cs"],        "rare_thresh":None},
    {"key":"oh_bin",       "encoder_type":"onehot",      "num_transform":["bin_curvature","interact_cs"],"rare_thresh":None},
    {"key":"ord_std",      "encoder_type":"ordinal",     "num_transform":["standardize"],               "rare_thresh":0.005},
    {"key":"tgtm_mean",    "encoder_type":"target_mean", "num_transform":["interact_cs"],               "rare_thresh":0.005},
    {"key":"freq_poly",    "encoder_type":"freq",        "num_transform":["poly","log1p_acc"],          "rare_thresh":0.005},
]


# Paths
DATA_DIR = "/kaggle/input/playground-series-s5e10"
OUT_SUB_DIR = "submissions"
OUT_OOF_DIR = "oof"
OUT_RES_DIR = "results"
os.makedirs(OUT_SUB_DIR, exist_ok=True)
os.makedirs(OUT_OOF_DIR, exist_ok=True)
os.makedirs(OUT_RES_DIR, exist_ok=True)

# Global timer
_GLOBAL_START = time.time()
_CUTOFF_SECS = CUTOFF_HOURS * 3600.0
def time_left_ok(): return (time.time() - _GLOBAL_START) < _CUTOFF_SECS
def now_min(): return round((time.time() - _GLOBAL_START)/60.0, 2)

print(f"[INFO] VERSION={VERSION} | CUT-OFF={CUTOFF_HOURS}h | N_SPLITS={N_SPLITS} | SEED={SEED}")
print(f"[INFO] Models this run: {MODELS_TO_RUN}")
print(f"[INFO] Preprocess combos: {[p['key'] for p in PREPROCESS_TO_RUN]}")


train_path = os.path.join(DATA_DIR, "train.csv")
test_path  = os.path.join(DATA_DIR, "test.csv")
sub_path   = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(train_path), train_path
assert os.path.exists(test_path), test_path

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)
sample_sub = pd.read_csv(sub_path) if os.path.exists(sub_path) else pd.DataFrame({"id": test["id"], "accident_risk": 0.0})

print(train.shape, test.shape, sample_sub.shape)
display(train.head(3))
display(test.head(3))


TARGET = "accident_risk"
ID_COL = "id"

# Column groups
all_cols = [c for c in train.columns if c != TARGET]
cat_cols = [c for c in all_cols if train[c].dtype == "object"]
bool_cols = [c for c in all_cols if train[c].dtype == bool]
num_cols  = [c for c in all_cols if c not in cat_cols + bool_cols + [ID_COL]]

# treat bool like category where encoding is categorical; also keep numeric {0,1} for tree libs
cat_cols_all = cat_cols + bool_cols
num_cols_all = [c for c in num_cols if c != ID_COL]

features = [c for c in train.columns if c not in [TARGET]]
print("Categorical-like:", cat_cols_all)
print("Numeric-like:", num_cols_all)

y = train[TARGET].values
test_ids = test[ID_COL].values

# CV splitter (bin target for balance)
y_bins = pd.qcut(y, q=20, duplicates="drop").astype(str)
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
print("[INFO] KFold ready.")


def rmse(a, b): 
    return mean_squared_error(a, b, squared=False)

def evaluate_metrics(y_true, y_pred):
    return {"rmse": rmse(y_true, y_pred), "mae": mean_absolute_error(y_true, y_pred), "r2": r2_score(y_true, y_pred)}

def save_oof_and_submission(model_key, oof_pred, test_pred):
    oof_df = pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "oof_pred": oof_pred})
    oof_path = os.path.join(OUT_OOF_DIR, f"oof_{model_key}_v{VERSION}.csv")
    oof_df.to_csv(oof_path, index=False)
    sub_df = sample_sub.copy()
    sub_df[TARGET] = np.clip(test_pred, 0.0, 1.0)
    sub_path = os.path.join(OUT_SUB_DIR, f"submission_{model_key}_v{VERSION}.csv")
    sub_df.to_csv(sub_path, index=False)
    return oof_path, sub_path

def merge_rare_levels(sr: pd.Series, frac=0.005):
    if frac is None: return sr
    vc = sr.value_counts(normalize=True)
    rare = vc[vc < frac].index
    return sr.mask(sr.isin(rare), "RARE")

def kfold_target_mean_encoding(train_df, test_df, col, target, n_splits=5, seed=42):
    # Out-of-fold mean encoding for train; global mean for test
    global_mean = train_df[target].mean()
    oof = pd.Series(np.nan, index=train_df.index)
    for tr_idx, va_idx in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(train_df):
        means = train_df.iloc[tr_idx].groupby(col)[target].mean()
        oof.iloc[va_idx] = train_df.iloc[va_idx][col].map(means).fillna(global_mean)
    test_enc = test_df[col].map(train_df.groupby(col)[target].mean()).fillna(global_mean)
    return oof.astype(float), test_enc.astype(float)

def frequency_encoding(train_df, test_df, col):
    freq = train_df[col].value_counts()
    return train_df[col].map(freq).astype(float), test_df[col].map(freq).fillna(0).astype(float)


def build_model(name, use_gpu=True, lgb_features=None):
    name = name.lower()
    rng = SEED

    if name == "lgbm":
        params = dict(
            n_estimators=5000,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=128,
            min_data_in_leaf=64,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=0.5,
            objective="rmse",
            random_state=rng
        )
        if use_gpu:
            params.update(device_type="gpu")
        else:
            params.update(device_type="cpu")

        # monotone constraints for curvature & speed_limit if present in lgb_features
        mono = []
        if lgb_features is not None:
            for f in lgb_features:
                if f == "curvature":
                    mono.append(1)
                elif f == "speed_limit":
                    mono.append(1)
                else:
                    mono.append(0)
        # Note: sklearn wrapper accepts monotone_constraints; native API will ignore it unless we pass it there explicitly.
        params.update(monotone_constraints=mono if mono else None)
        model = lgb.LGBMRegressor(**params)

    elif name == "xgb":
        params = dict(
            n_estimators=5000,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=0.7,
            min_child_weight=10,
            random_state=rng,
            tree_method="hist",
        )
        if use_gpu:
            params.update(tree_method="gpu_hist", predictor="gpu_predictor")
        model = xgb.XGBRegressor(**params)

    elif name == "cat":
        params = dict(
            iterations=5000,
            depth=8,
            learning_rate=0.03,
            loss_function="RMSE",
            random_state=rng,
            l2_leaf_reg=6.0,
            bootstrap_type="Bernoulli",  # supports subsample
            subsample=0.8,
            verbose=False
        )
        if use_gpu:
            params.update(task_type="GPU")  # do NOT set rsm on GPU (unsupported for RMSE)
        else:
            params.update(rsm=0.8)
        model = CatBoostRegressor(**params)

    elif name == "hgb":
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=None,
            max_iter=2000,
            max_bins=255,
            min_samples_leaf=64,
            l2_regularization=0.1,
            random_state=rng
        )

    elif name == "rf":
        model = RandomForestRegressor(
            n_estimators=1000,
            max_depth=None,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=rng
        )

    elif name == "et":
        model = ExtraTreesRegressor(
            n_estimators=1000,
            max_depth=None,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=rng
        )

    elif name == "gbr":
        model = GradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            n_estimators=3000,
            max_depth=4,
            subsample=0.8,
            random_state=rng
        )

    elif name == "ridge":
        model = Pipeline(steps=[("est", Ridge(alpha=2.0, random_state=rng))])

    elif name == "lasso":
        model = Pipeline(steps=[("est", Lasso(alpha=0.001, random_state=rng, max_iter=10000))])

    elif name == "enet":
        model = Pipeline(steps=[("est", ElasticNet(alpha=0.001, l1_ratio=0.2, random_state=rng, max_iter=10000))])

    else:
        raise ValueError(f"Unknown model name: {name}")

    return model


def build_views_for_spec(spec):
    key = spec["key"]
    enc = spec["encoder_type"]
    num_tx = spec["num_transform"] if isinstance(spec["num_transform"], list) else [spec["num_transform"]]
    rare_thr = spec.get("rare_thresh", None)

    tr = train.copy()
    te = test.copy()

    # ----- Rare level merge (for non-native encoders) -----
    cats_for_encode = cat_cols_all.copy()
    if enc != "native_cats" and rare_thr is not None:
        for c in cats_for_encode:
            tr[c] = merge_rare_levels(tr[c], rare_thr)
            te[c] = merge_rare_levels(te[c], rare_thr)

    # ----- Numeric transforms / interactions -----
    # 1) log1p on num_reported_accidents (skew reduction)
    if "log1p_acc" in num_tx and "num_reported_accidents" in num_cols_all:
        tr["num_reported_accidents_log1p"] = np.log1p(tr["num_reported_accidents"])
        te["num_reported_accidents_log1p"] = np.log1p(te["num_reported_accidents"])

    # 2) curvature × speed_limit interaction
    if "interact_cs" in num_tx and {"curvature","speed_limit"}.issubset(tr.columns):
        tr["curv_x_speed"] = tr["curvature"] * tr["speed_limit"]
        te["curv_x_speed"] = te["curvature"] * te["speed_limit"]

    # 3) polynomial features for numerics (degree=2, no bias)
    poly_added = []
    if "poly" in num_tx:
        num_for_poly = [c for c in tr.columns if c in (num_cols_all + ["num_reported_accidents_log1p","curv_x_speed"]) and c != ID_COL]
        if len(num_for_poly) > 0:
            pf = PolynomialFeatures(degree=2, include_bias=False)
            tr_poly = pf.fit_transform(tr[num_for_poly])
            te_poly = pf.transform(te[num_for_poly])
            poly_names = pf.get_feature_names_out(num_for_poly)
            # avoid duplicating original columns
            for i, n in enumerate(poly_names):
                if n not in tr.columns:
                    tr[f"poly__{n}"] = tr_poly[:, i]
                    te[f"poly__{n}"] = te_poly[:, i]
                    poly_added.append(f"poly__{n}")

    # 4) bin curvature into deciles (as category)
    if "bin_curvature" in num_tx and "curvature" in tr.columns:
        tr["curvature_bin"] = pd.qcut(tr["curvature"], q=10, duplicates="drop").astype(str)
        te["curvature_bin"] = pd.qcut(te["curvature"], q=10, duplicates="drop").astype(str)
        cats_for_encode = cats_for_encode + ["curvature_bin"]

    # 5) standardize numerics (for linear models usually); done inside encoder branches to keep pipeline simple

    # ---------- Build three "views" ----------
    # View for native categorical libs
    train_lgb = tr.copy()
    test_lgb  = te.copy()
    for c in cat_cols_all + (["curvature_bin"] if "curvature_bin" in tr.columns else []):
        if c in train_lgb.columns:
            train_lgb[c] = train_lgb[c].astype("category")
            test_lgb[c]  = test_lgb[c].astype("category")
    lgb_features = [c for c in train_lgb.columns if c not in [TARGET]]

    # CatBoost view uses original strings (or merged) and numeric
    features_all = [c for c in tr.columns if c != TARGET]
    cat_idx_for_catboost = [i for i, c in enumerate(features_all) if c in cats_for_encode]

    # Encoded Matrix views for tree/linear models
    # A) onehot
    if enc == "onehot":
        onehot = ColumnTransformer(
            transformers=[("oh", OneHotEncoder(sparse=False, handle_unknown="ignore"), cats_for_encode)],
            remainder="passthrough"
        )
        X_oh = onehot.fit_transform(tr[cats_for_encode + [c for c in tr.columns if c in num_cols_all or c.startswith(("num_reported_accidents_log1p","poly__","curv_x_speed"))]])
        X_test_oh = onehot.transform(te[cats_for_encode + [c for c in te.columns if c in num_cols_all or c.startswith(("num_reported_accidents_log1p","poly__","curv_x_speed"))]])
        oh_feature_names = list(onehot.get_feature_names_out())

        # optional standardize numerics after onehot if requested
        if "standardize" in num_tx:
            # Identify numeric columns in the transformed matrix (the 'remainder=passthrough' part)
            # Easiest: standardize original numeric columns before one-hot to keep alignment simple
            pass  # already not necessary for tree models; for linear, we use a separate encoder below if needed

        X_enc = X_oh; X_test_enc = X_test_oh; enc_names = oh_feature_names

    # B) ordinal
    elif enc == "ordinal":
        ord_cols = cats_for_encode
        ord_enc = ColumnTransformer(
            transformers=[("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ord_cols)],
            remainder="passthrough"
        )
        X_ord = ord_enc.fit_transform(tr[ord_cols + [c for c in tr.columns if c in num_cols_all or c.startswith(("num_reported_accidents_log1p","poly__","curv_x_speed"))]])
        X_test_ord = ord_enc.transform(te[ord_cols + [c for c in te.columns if c in num_cols_all or c.startswith(("num_reported_accidents_log1p","poly__","curv_x_speed"))]])
        X_enc = X_ord; X_test_enc = X_test_ord; enc_names = list(ord_enc.get_feature_names_out())

        if "standardize" in num_tx:
            # Standardize the numeric slice (rough but effective): z-score the whole matrix
            scaler = StandardScaler(with_mean=False)  # keep sparse-safety; matrix is dense but this is fine
            X_enc = scaler.fit_transform(X_enc)
            X_test_enc = scaler.transform(X_test_enc)

    # C) target mean encoding
    elif enc == "target_mean":
        tr_tm = pd.DataFrame(index=tr.index)
        te_tm = pd.DataFrame(index=te.index)
        for c in cats_for_encode:
            tr_tm[c], te_tm[c] = kfold_target_mean_encoding(tr, te, c, TARGET, n_splits=N_SPLITS, seed=SEED)
        # append numerics
        use_nums = [c for c in tr.columns if c in num_cols_all or c.startswith(("num_reported_accidents_log1p","poly__","curv_x_speed"))]
        for c in use_nums:
            tr_tm[c] = tr[c].values
            te_tm[c] = te[c].values
        X_enc = tr_tm.values.astype(float)
        X_test_enc = te_tm.values.astype(float)
        enc_names = list(tr_tm.columns)

    # D) frequency encoding
    elif enc == "freq":
        tr_fe = pd.DataFrame(index=tr.index)
        te_fe = pd.DataFrame(index=te.index)
        for c in cats_for_encode:
            tr_fe[c], te_fe[c] = frequency_encoding(tr, te, c)
        use_nums = [c for c in tr.columns if c in num_cols_all or c.startswith(("num_reported_accidents_log1p","poly__","curv_x_speed"))]
        for c in use_nums:
            tr_fe[c] = tr[c].values
            te_fe[c] = te[c].values
        X_enc = tr_fe.values.astype(float)
        X_test_enc = te_fe.values.astype(float)
        enc_names = list(tr_fe.columns)

    # E) native_cats (no encoded matrix — will fallback to onehot for models that require matrices)
    elif enc == "native_cats":
        X_enc = None; X_test_enc = None; enc_names = None

    else:
        raise ValueError(f"Unknown encoder_type: {enc}")

    return {
        "spec_key": key,
        "encoder_type": enc,
        "X_enc": X_enc,
        "X_test_enc": X_test_enc,
        "enc_names": enc_names,
        "train_lgb": train_lgb,
        "test_lgb": test_lgb,
        "lgb_features": [c for c in lgb_features if c != TARGET],
        "features_all": features_all,
        "cat_idx_for_catboost": cat_idx_for_catboost
    }


results_rows = []

for prep in PREPROCESS_TO_RUN:
    if not time_left_ok():
        print(f"[STOP] Time limit reached at {now_min()} min BEFORE preprocessing '{prep['key']}'.")
        break

    print(f"\n================ Preprocessing: {prep['key']} (enc={prep['encoder_type']}, num={prep['num_transform']}) ================")
    views = build_views_for_spec(prep)

    for mdl_name in MODELS_TO_RUN:
        if not time_left_ok():
            print(f"[STOP] Time limit reached at {now_min()} min. Saving progress and exiting model loop.")
            break

        use_gpu = mdl_name in GPU_MODEL_SET
        model_key_base = f"{mdl_name}{'_gpu' if use_gpu else '_cpu'}"
        model_key = f"{model_key_base}__{prep['key']}"

        print(f"\n[MODEL] {model_key} | time={now_min()} min")

        # Build model (pass lgb_features for monotone map if LGBM)
        try:
            model = build_model(mdl_name, use_gpu=use_gpu, lgb_features=views.get("lgb_features", None))
        except Exception as e:
            print(f"[SKIP] Could not build {model_key}: {e}")
            continue

        oof_pred = np.zeros(len(train), dtype=float)
        test_pred_folds = np.zeros((len(test), N_SPLITS), dtype=float)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(train[ID_COL], y_bins), 1):
            if not time_left_ok():
                print(f"[STOP] Time limit reached mid-model at {now_min()} min. Saving partial results for {model_key}.")
                break

            y_tr, y_va = y[tr_idx], y[va_idx]
            enc_type = views["encoder_type"]

            if mdl_name == "lgbm":
                if enc_type == "native_cats":
                    dtr = lgb.Dataset(
                        views["train_lgb"].iloc[tr_idx][views["lgb_features"]],
                        label=y_tr,
                        categorical_feature=[c for c in cat_cols_all if c in views["lgb_features"]],
                        free_raw_data=False
                    )
                    dva = lgb.Dataset(
                        views["train_lgb"].iloc[va_idx][views["lgb_features"]],
                        label=y_va,
                        categorical_feature=[c for c in cat_cols_all if c in views["lgb_features"]],
                        free_raw_data=False
                    )
                    params = model.get_params()
                    # FIX (native API): drop params unknown to lgb.train()
                    drop_keys = {"monotone_constraints", "importance_type"}
                    fit_params = {k: v for k, v in params.items() if k not in drop_keys}

                    bst = lgb.train(
                        fit_params,
                        dtr,
                        valid_sets=[dtr, dva],
                        valid_names=["train","valid"],
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=200, verbose=False),
                            lgb.log_evaluation(period=200)
                        ]
                    )
                    oof_pred[va_idx] = bst.predict(
                        views["train_lgb"].iloc[va_idx][views["lgb_features"]],
                        num_iteration=bst.best_iteration
                    )
                    test_pred_folds[:, fold-1] = bst.predict(
                        views["test_lgb"][views["lgb_features"]],
                        num_iteration=bst.best_iteration
                    )
                    del dtr, dva, bst

                else:
                    # sklearn LGBM on encoded matrix
                    X_enc = views["X_enc"]; X_test_enc = views["X_test_enc"]
                    est = build_model("lgbm", use_gpu=use_gpu, lgb_features=None)
                    # FIX (sklearn API): remove 'verbose' kw; use callbacks instead
                    est.fit(
                        X_enc[tr_idx], y_tr,
                        eval_set=[(X_enc[va_idx], y_va)],
                        eval_metric="rmse",
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=200, verbose=False),
                            lgb.log_evaluation(period=200)
                        ]
                    )
                    oof_pred[va_idx] = est.predict(X_enc[va_idx])
                    test_pred_folds[:, fold-1] = est.predict(X_test_enc)
                    del est

            elif mdl_name == "cat":
                if enc_type == "native_cats":
                    train_pool = Pool(
                        views["train_lgb"].iloc[tr_idx][views["lgb_features"]], y_tr,
                        cat_features=[c for c in cat_cols_all if c in views["lgb_features"]]
                    )
                    valid_pool = Pool(
                        views["train_lgb"].iloc[va_idx][views["lgb_features"]], y_va,
                        cat_features=[c for c in cat_cols_all if c in views["lgb_features"]]
                    )
                    params = model.get_params()
                    cat = CatBoostRegressor(**params)
                    cat.fit(train_pool, eval_set=valid_pool, verbose=False, use_best_model=True, early_stopping_rounds=200)
                    oof_pred[va_idx] = cat.predict(valid_pool)
                    test_pred_folds[:, fold-1] = cat.predict(
                        Pool(views["test_lgb"][views["lgb_features"]],
                             cat_features=[c for c in cat_cols_all if c in views["lgb_features"]])
                    )
                    del train_pool, valid_pool, cat
                else:
                    X_enc = views["X_enc"]; X_test_enc = views["X_test_enc"]
                    params = model.get_params()
                    cat = CatBoostRegressor(**params)
                    cat.fit(X_enc[tr_idx], y_tr, eval_set=(X_enc[va_idx], y_va),
                            verbose=False, use_best_model=True, early_stopping_rounds=200)
                    oof_pred[va_idx] = cat.predict(X_enc[va_idx])
                    test_pred_folds[:, fold-1] = cat.predict(X_test_enc)
                    del cat

            elif mdl_name == "xgb":
                X_enc = views["X_enc"]; X_test_enc = views["X_test_enc"]
                if enc_type == "native_cats":
                    oh_cols = cat_cols_all
                    onehot = ColumnTransformer(
                        transformers=[("oh", OneHotEncoder(sparse=False, handle_unknown="ignore"), oh_cols)],
                        remainder="passthrough"
                    )
                    X_all = onehot.fit_transform(train[oh_cols + num_cols_all])
                    X_te_all = onehot.transform(test[oh_cols + num_cols_all])
                    X_enc = X_all; X_test_enc = X_te_all

                dtr = xgb.DMatrix(X_enc[tr_idx], label=y_tr)
                dva = xgb.DMatrix(X_enc[va_idx], label=y_va)
                dte = xgb.DMatrix(X_test_enc)
                params = model.get_params()
                n_estimators = params.pop("n_estimators")
                xg = xgb.train(
                    params,
                    dtr,
                    num_boost_round=n_estimators,
                    evals=[(dtr,"train"), (dva,"valid")],
                    verbose_eval=False,
                    early_stopping_rounds=200
                )
                oof_pred[va_idx] = xg.predict(dva, iteration_range=(0, xg.best_iteration+1))
                test_pred_folds[:, fold-1] = xg.predict(dte, iteration_range=(0, xg.best_iteration+1))
                del dtr, dva, dte, xg

            else:
                # sklearn models need matrices
                X_enc = views["X_enc"]; X_test_enc = views["X_test_enc"]
                if X_enc is None or X_test_enc is None:
                    oh = ColumnTransformer(
                        transformers=[("oh", OneHotEncoder(sparse=False, handle_unknown="ignore"), cat_cols_all)],
                        remainder="passthrough"
                    )
                    X_all = oh.fit_transform(train[cat_cols_all + num_cols_all])
                    X_te_all = oh.transform(test[cat_cols_all + num_cols_all])
                    X_enc = X_all; X_test_enc = X_te_all

                est = build_model(mdl_name, use_gpu=False)
                est.fit(X_enc[tr_idx], y_tr)
                oof_pred[va_idx] = est.predict(X_enc[va_idx])
                test_pred_folds[:, fold-1] = est.predict(X_test_enc)
                del est

            gc.collect()

        # aggregate & save
        test_pred = np.nanmean(test_pred_folds, axis=1)
        oof_metrics = evaluate_metrics(y, np.nan_to_num(oof_pred, nan=np.nanmean(oof_pred)))
        oof_path, sub_path = save_oof_and_submission(model_key, oof_pred, test_pred)

        print(f"[DONE] {model_key} | RMSE={oof_metrics['rmse']:.6f} | MAE={oof_metrics['mae']:.6f} | R2={oof_metrics['r2']:.4f}")
        print(f"       OOF: {oof_path} | SUB: {sub_path}")

        row = dict(model=model_key, prep=prep["key"], folds=N_SPLITS, **oof_metrics, time_min=now_min())
        results_rows.append(row)

        if not time_left_ok():
            print(f"[STOP] Time limit reached after completing {model_key}.")
            break

# Save combined leaderboard
results_df = pd.DataFrame(results_rows).sort_values(["rmse","model"])
res_csv_path = os.path.join(OUT_RES_DIR, f"results_01_v{VERSION}.csv")
results_df.to_csv(res_csv_path, index=False)
print(f"\n[RESULTS] Saved leaderboard -> {res_csv_path}")
display(results_df.head(20))


if len(results_rows) == 0:
    print("[WARN] No models were trained. Nothing to plot.")
else:
    top = results_df.nsmallest(10, "rmse")
    plt.figure(figsize=(9,6))
    labels = [f"{m} ({p})" for m,p in zip(top["model"], top["prep"])]
    plt.barh(range(len(top)), top["rmse"].values)
    plt.yticks(range(len(top)), labels, fontsize=9)
    plt.gca().invert_yaxis()
    plt.title(f"Top-10 by RMSE (v{VERSION})")
    plt.xlabel("RMSE")
    plt.tight_layout()
    fig_path = os.path.join(OUT_RES_DIR, f"results_01_v{VERSION}.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved -> {fig_path}")




