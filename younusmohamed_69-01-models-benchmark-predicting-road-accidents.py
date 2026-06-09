import os, gc, time, sys, math, json, warnings, pathlib, textwrap, random
warnings.filterwarnings("ignore")

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from catboost import CatBoostRegressor, Pool
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

VERSION = "06"               
CUTOFF_HOURS = 11.5           # <- stop training after this many hours (saving progress)
N_SPLITS = 5                 # <- KFold splits
SEED = 42

# "lgbm", "xgb", "cat", "hgb", "rf", "et", "gbr", "ridge", "lasso", "enet"
MODELS_TO_RUN = ["xgb"] 

GPU_MODEL_SET  = ["lgbm", "xgb", "cat", "hgb"]       # models that can leverage GPU (where available)
CPU_MODEL_SET  = ["rf", "et", "gbr", "ridge", "lasso", "enet"]  # pure CPU or minimal GPU benefit

# Track total wall-clock for cutoff
_GLOBAL_START = time.time()
_CUTOFF_SECS = CUTOFF_HOURS * 3600.0

def time_left_ok():
    return (time.time() - _GLOBAL_START) < _CUTOFF_SECS

def now_min():
    return round((time.time() - _GLOBAL_START)/60.0, 2)

print(f"[INFO] VERSION={VERSION} | CUT-OFF={CUTOFF_HOURS}h | N_SPLITS={N_SPLITS} | SEED={SEED}")
print(f"[INFO] Models this run: {MODELS_TO_RUN}")


# Paths
DATA_DIR = "/kaggle/input/playground-series-s5e10"
OUT_SUB_DIR = "submissions"
OUT_OOF_DIR = "oof"
OUT_RES_DIR = "results"
os.makedirs(OUT_SUB_DIR, exist_ok=True)
os.makedirs(OUT_OOF_DIR, exist_ok=True)
os.makedirs(OUT_RES_DIR, exist_ok=True)

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

# Treat bool as categorical for one-hot and as numeric {0,1} for tree libs that accept it
cat_cols_all = cat_cols + bool_cols
num_cols_all = [c for c in num_cols if c != ID_COL]

features = [c for c in train.columns if c not in [TARGET]]
print("Categorical-like:", cat_cols_all)
print("Numeric-like:", num_cols_all)

# View A (for XGB/RF/ET/HGB/Linear/GBR): One-Hot with handle_unknown='ignore'
onehot = ColumnTransformer(
    transformers=[
        ("oh", OneHotEncoder(sparse=False, handle_unknown="ignore"), cat_cols_all)
    ],
    remainder="passthrough"
)

X_oh = onehot.fit_transform(train[cat_cols_all + num_cols_all])
X_test_oh = onehot.transform(test[cat_cols_all + num_cols_all])
oh_feature_names = list(onehot.get_feature_names_out())

# View B (for LGBM with categorical dtype)
train_lgb = train.copy()
test_lgb  = test.copy()
for c in cat_cols_all:
    train_lgb[c] = train_lgb[c].astype("category")
    test_lgb[c]  = test_lgb[c].astype("category")
lgb_features = [c for c in features if c != ID_COL]

# View C (for CatBoost: keep strings and pass cat feature indices)
cat_idx_for_catboost = [i for i, c in enumerate(features) if c in cat_cols_all]

y = train[TARGET].values
test_ids = test[ID_COL].values

print(f"[OH] X={X_oh.shape}, X_test={X_test_oh.shape}")
print(f"[LGB] features={len(lgb_features)}, cats={len(cat_cols_all)}")
print(f"[CAT] cat_idx={cat_idx_for_catboost[:10]}{'...' if len(cat_idx_for_catboost)>10 else ''}")


def rmse(a, b): 
    return mean_squared_error(a, b, squared=False)

def evaluate_metrics(y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2":  r2_score(y_true, y_pred)
    }

def save_oof_and_submission(model_key, oof_pred, test_pred):
    # OOF
    oof_df = pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "oof_pred": oof_pred})
    oof_path = os.path.join(OUT_OOF_DIR, f"oof_{model_key}_v{VERSION}.csv")
    oof_df.to_csv(oof_path, index=False)
    # SUB
    sub_df = sample_sub.copy()
    sub_df[TARGET] = np.clip(test_pred, 0.0, 1.0)
    sub_path = os.path.join(OUT_SUB_DIR, f"submission_{model_key}_v{VERSION}.csv")
    sub_df.to_csv(sub_path, index=False)
    return oof_path, sub_path

# KFold splitter (bin the target for better balance)
y_bins = pd.qcut(y, q=20, duplicates="drop").astype(str)
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
print("[INFO] KFold ready.")

# Leaderboard collector
LEADERBOARD = []   # list of dicts per model


def build_model(name, use_gpu=True):
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
        # GPU hint
        if use_gpu:
            params.update(dict(device_type="gpu"))
        else:
            params.update(dict(device_type="cpu"))

        # EDA-informed monotonicity (if features present)
        # curvature ↑ ⇒ risk ↑, speed_limit ↑ ⇒ risk ↑
        # LightGBM expects an array aligned with lgb_features
        mono = []
        for f in lgb_features:
            if f == "curvature":
                mono.append(1)
            elif f == "speed_limit":
                mono.append(1)
            else:
                mono.append(0)
        params.update(monotone_constraints=mono)
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
            bootstrap_type="Bernoulli",
            subsample=0.8,
            # rsm=0.8,
            verbose=False
        )
        if use_gpu:
            params.update(task_type="GPU")
        model = CatBoostRegressor(**params)

    elif name == "hgb":
        # HistGradientBoosting (sklearn) (CPU; can be fast)
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
        model = Pipeline(steps=[
            ("est", Ridge(alpha=2.0, random_state=rng))
        ])

    elif name == "lasso":
        model = Pipeline(steps=[
            ("est", Lasso(alpha=0.001, random_state=rng, max_iter=10000))
        ])

    elif name == "enet":
        model = Pipeline(steps=[
            ("est", ElasticNet(alpha=0.001, l1_ratio=0.2, random_state=rng, max_iter=10000))
        ])
    else:
        raise ValueError(f"Unknown model name: {name}")

    return model


results_rows = []

for mdl_name in MODELS_TO_RUN:
    if not time_left_ok():
        print(f"[STOP] Time limit reached at {now_min()} min. Saving progress and exiting loop.")
        break

    use_gpu = mdl_name in GPU_MODEL_SET  # heuristic
    model_key = f"{mdl_name}{'_gpu' if use_gpu else '_cpu'}"

    print(f"\n[MODEL] {model_key} | time={now_min()} min")
    try:
        model = build_model(mdl_name, use_gpu=use_gpu)
    except Exception as e:
        print(f"[SKIP] Could not build {model_key}: {e}")
        continue

    oof_pred = np.zeros(len(train), dtype=float)
    test_pred_folds = np.zeros((len(test), N_SPLITS), dtype=float)

    # Fold training
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train[ID_COL], y_bins), 1):
        if not time_left_ok():
            print(f"[STOP] Time limit reached mid-model at {now_min()} min. Saving partial results.")
            break

        X_tr_oh, X_va_oh = X_oh[tr_idx], X_oh[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Branch by model type for data interface
        if mdl_name == "lgbm":
            dtr = lgb.Dataset(train_lgb.iloc[tr_idx][lgb_features], label=y_tr, categorical_feature=cat_cols_all, free_raw_data=False)
            dva = lgb.Dataset(train_lgb.iloc[va_idx][lgb_features], label=y_va, categorical_feature=cat_cols_all, free_raw_data=False)
            params = model.get_params()
            # Extract raw params for native API (faster with early stopping)
            fit_params = {k: params[k] for k in params if k not in ["monotone_constraints"]}
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
            oof_pred[va_idx] = bst.predict(train_lgb.iloc[va_idx][lgb_features], num_iteration=bst.best_iteration)
            test_pred_folds[:, fold-1] = bst.predict(test_lgb[lgb_features], num_iteration=bst.best_iteration)
            del dtr, dva, bst

        elif mdl_name == "cat":
            train_pool = Pool(train[features].iloc[tr_idx], y_tr, cat_features=cat_idx_for_catboost)
            valid_pool = Pool(train[features].iloc[va_idx], y_va, cat_features=cat_idx_for_catboost)
            params = model.get_params()
            cat = CatBoostRegressor(**params)
            cat.fit(train_pool, eval_set=valid_pool, verbose=False, use_best_model=True, early_stopping_rounds=200)
            oof_pred[va_idx] = cat.predict(valid_pool)
            test_pred_folds[:, fold-1] = cat.predict(Pool(test[features], cat_features=cat_idx_for_catboost))
            del train_pool, valid_pool, cat

        elif mdl_name == "xgb":
            # Use DMatrix for speed
            dtr = xgb.DMatrix(X_oh[tr_idx], label=y_tr)
            dva = xgb.DMatrix(X_oh[va_idx], label=y_va)
            dte = xgb.DMatrix(X_test_oh)
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
            # Sklearn-style fit on one-hot
            est = build_model(mdl_name, use_gpu=False)  # sklearn CPUs
            est.fit(X_tr_oh, y_tr)
            oof_pred[va_idx] = est.predict(X_va_oh)
            test_pred_folds[:, fold-1] = est.predict(X_test_oh)
            del est

        gc.collect()

    # If stopped mid-model, still aggregate whatever we have
    test_pred = np.nanmean(test_pred_folds, axis=1)
    oof_metrics = evaluate_metrics(y, np.nan_to_num(oof_pred, nan=np.nanmean(oof_pred)))
    oof_path, sub_path = save_oof_and_submission(model_key, oof_pred, test_pred)

    print(f"[DONE] {model_key} | RMSE={oof_metrics['rmse']:.6f} | MAE={oof_metrics['mae']:.6f} | R2={oof_metrics['r2']:.4f}")
    print(f"       OOF: {oof_path} | SUB: {sub_path}")

    row = dict(model=model_key, folds=N_SPLITS, **oof_metrics, time_min=now_min())
    results_rows.append(row)

    # Stop if over time (after saving)
    if not time_left_ok():
        print(f"[STOP] Time limit reached after completing {model_key}.")
        break

# Save raw leaderboard so far
results_df = pd.DataFrame(results_rows).sort_values("rmse")
res_csv_path = os.path.join(OUT_RES_DIR, f"results_01_v{VERSION}.csv")
results_df.to_csv(res_csv_path, index=False)
print(f"\n[RESULTS] Saved partial/full leaderboard -> {res_csv_path}")
display(results_df.head(20))


if len(results_rows) == 0:
    print("[WARN] No models were trained. Nothing to plot.")
else:
    top = results_df.nsmallest(10, "rmse")
    plt.figure(figsize=(8,5))
    plt.barh(range(len(top)), top["rmse"].values)
    plt.yticks(range(len(top)), top["model"].values)
    plt.gca().invert_yaxis()
    plt.title(f"Top-10 models by RMSE (v{VERSION})")
    plt.xlabel("RMSE")
    plt.tight_layout()
    fig_path = os.path.join(OUT_RES_DIR, f"results_01_v{VERSION}.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved -> {fig_path}")

