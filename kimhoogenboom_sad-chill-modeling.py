import os, math, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, RidgeCV


HAVE_XGB = True
HAVE_LGBM = True
try:
    from xgboost import XGBRegressor
except Exception:
    HAVE_XGB = False
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except Exception:
    HAVE_LGBM = False


HAVE_JOBLIB = True
try:
    import joblib
except Exception:
    HAVE_JOBLIB = False


SEED = 26
DATA_PATH = "/kaggle/input/playground-series-s5e9/"
CLIP_MIN, CLIP_MAX = 40.0, 220.0


# Speed-friendly defaults for Kaggle
FAST_MODE = False     # <- set False for max accuracy (slower)
SAMPLE_HEAVY = 500_000  # max rows for heavy steps (Optuna/PI/etc.)
SAMPLE_SHAP  = 120_000  # for SHAP training
SAMPLE_LEARN = 160_000  # for learning curve training
VAL_SHAP     = 5_000    # SHAP validation sample


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def stratify_bins(y, n_bins=20):
    y = pd.Series(y).astype(float)
    bins = pd.qcut(y, q=n_bins, duplicates="drop")
    return bins.cat.codes


def maybe_sample(X, y=None, n_max=160_000, seed=SEED):
    if not FAST_MODE:
        return (X, y) if y is not None else X
    if y is None:
        if X.shape[0] <= n_max: return X
        idx = np.random.RandomState(seed).choice(X.shape[0], size=n_max, replace=False)
        return X[idx]
    else:
        if X.shape[0] <= n_max: return X, y
        idx = np.random.RandomState(seed).choice(X.shape[0], size=n_max, replace=False)
        return X[idx], y[idx]


train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
submission = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))

target_col = "BeatsPerMinute"
id_col = "id" if "id" in train.columns else None
feature_cols = [c for c in train.columns if c not in [target_col, id_col]]

print(f"Train: {train.shape}, Test: {test.shape}")
print("Features:", feature_cols)
print("Submission columns:", submission.columns.tolist())


# Cast to float32 for speed/memory on CPU
X_full = train[feature_cols].astype(np.float32).values
y_full = train[target_col].astype(np.float32).values
X_test = test[feature_cols].astype(np.float32).values


# Stratified split by BPM quantiles to preserve distribution
y_bins = stratify_bins(y_full, n_bins=20)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_full, y_full, test_size=0.2, random_state=SEED, stratify=y_bins
)
print("Split:", X_tr.shape, X_va.shape)


y_bins = stratify_bins(y_full, n_bins=20)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_full, y_full, test_size=0.2, random_state=SEED, stratify=y_bins
)
print("Split:", X_tr.shape, X_va.shape)


def build_base_models(random_state=SEED):
    models = []

    # Linear baseline
    ridge = ("ridge", Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=random_state))
    ]))

    # RandomForest (trimmed in FAST_MODE)
    rf_params = dict(
        n_estimators=200 if FAST_MODE else 500,
        max_depth=16 if FAST_MODE else None,
        max_features="sqrt",
        min_samples_leaf=4 if FAST_MODE else 1,
        bootstrap=True,
        n_jobs=-1,
        random_state=random_state
    )
    rf = ("rf", RandomForestRegressor(**rf_params))

    # XGBoost
    if HAVE_XGB:
        xgb = ("xgb", XGBRegressor(
            n_estimators=400 if FAST_MODE else 700,
            learning_rate=0.08 if FAST_MODE else 0.05,
            max_depth=8,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.0, reg_lambda=1.0,
            tree_method="hist",
            n_jobs=-1, random_state=random_state
        ))
        models.append(xgb)

    # LightGBM
    if HAVE_LGBM:
        lgb = ("lgb", LGBMRegressor(
            n_estimators=500 if FAST_MODE else 900,
            learning_rate=0.07 if FAST_MODE else 0.05,
            max_depth=-1, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.0, reg_lambda=1.0,
            random_state=random_state, n_jobs=-1
        ))
        models.append(lgb)

    # In super-fast mode you can skip RF in stacking by removing it here
    models.append(rf)
    models.append(ridge)
    return models


def build_stack(models, random_state=SEED):
    meta = Ridge(alpha=1.0, random_state=random_state)
    return StackingRegressor(
        estimators=models,
        final_estimator=meta,
        passthrough=False,
        n_jobs=-1
    )


def evaluate_model(name, model, X_train, X_valid, y_train, y_valid):
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    score = rmse(y_valid, pred)
    print(f"[{name}] RMSE: {score:.5f}")
    return score, pred


def try_load_external_models():
    models = {}
    base = "/kaggle/input/saved-model"
    if not (HAVE_JOBLIB and os.path.isdir(base)):
        return models
    for key, fname in [("rf_ext","RandomForest.pkl"), ("xgb_ext","XGBoost.pkl"), ("lgb_ext","LightGBM.pkl")]:
        p = os.path.join(base, fname)
        if os.path.exists(p):
            try:
                models[key] = joblib.load(p)
                print(f"Loaded external model: {key} from {p}")
            except Exception as e:
                print(f"Could not load {p}: {e}")
    return models


def make_tempo_bands(y, bins=(80,110)):
    y = np.asarray(y)
    labels = np.zeros_like(y, dtype=int)     # low
    labels[(y >= bins[0]) & (y < bins[1])] = 1  # mid
    labels[y >= bins[1]] = 2                 # high
    return labels


class MixtureOfExpertsRegressor:
    def __init__(self, gating=None, experts=None, bands=(80,110)):
        self.bands = bands
        self.gating = gating
        self.experts = experts or {}
        self.classes_ = [0,1,2]

    def fit(self, X, y):
        y_bands = make_tempo_bands(y, self.bands)
        if self.gating is None:
            if HAVE_LGBM:
                self.gating = LGBMClassifier(n_estimators=300, learning_rate=0.07,
                                             random_state=SEED, n_jobs=-1)
            else:
                self.gating = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
        self.gating.fit(X, y_bands)

        self.experts = {}
        for band in self.classes_:
            idx = np.where(y_bands == band)[0]
            Xb, yb = X[idx], y[idx]
            if HAVE_LGBM:
                expert = LGBMRegressor(n_estimators=400 if FAST_MODE else 700,
                                       learning_rate=0.06 if FAST_MODE else 0.05,
                                       num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                                       random_state=SEED, n_jobs=-1)
            elif HAVE_XGB:
                expert = XGBRegressor(n_estimators=400 if FAST_MODE else 700,
                                      learning_rate=0.08 if FAST_MODE else 0.06,
                                      max_depth=8, subsample=0.8, colsample_bytree=0.8,
                                      n_jobs=-1, random_state=SEED, tree_method="hist")
            else:
                expert = RandomForestRegressor(n_estimators=200 if FAST_MODE else 400,
                                               max_depth=16 if FAST_MODE else None,
                                               max_features="sqrt", min_samples_leaf=4 if FAST_MODE else 1,
                                               random_state=SEED, n_jobs=-1)
            expert.fit(Xb, yb)
            self.experts[band] = expert
        return self

    def predict(self, X):
        if hasattr(self.gating, "predict_proba"):
            probs = self.gating.predict_proba(X)
        else:
            hard = self.gating.predict(X)
            probs = np.zeros((X.shape[0], len(self.classes_)))
            probs[np.arange(X.shape[0]), hard] = 1.0
        preds = np.zeros((X.shape[0], len(self.classes_)))
        for i, c in enumerate(self.classes_):
            preds[:, i] = self.experts[c].predict(X)
        return (probs * preds).sum(axis=1)


# Ridge baseline
ridge_baseline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=SEED))])
rmse_ridge, pred_ridge_va = evaluate_model("Ridge Baseline", ridge_baseline, X_tr, X_va, y_tr, y_va)

# LightGBM with early stopping
if HAVE_LGBM:
    lgb = LGBMRegressor(
        n_estimators=2000,               # large cap + ES
        learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1
    )
    lgb.fit(X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="l2",
            callbacks=[LGBMRegressor.early_stopping(stopping_rounds=100, verbose=False)] if hasattr(LGBMRegressor, 'early_stopping') else None)
    pred_lgb_va = lgb.predict(X_va, num_iteration=getattr(lgb, "best_iteration_", None))
    rmse_lgb = rmse(y_va, pred_lgb_va)
    print(f"[LightGBM+ES] RMSE: {rmse_lgb:.5f} (best_iter={getattr(lgb,'best_iteration_', None)})")
else:
    rmse_lgb, pred_lgb_va = None, None
    print("LightGBM not available.")

# XGBoost with early stopping
if HAVE_XGB:
    xgb = XGBRegressor(
        n_estimators=4000,               # large cap + ES
        learning_rate=0.05,
        max_depth=8, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=1.0, tree_method="hist",
        n_jobs=-1, random_state=SEED
    )
    xgb.fit(X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            verbose=False,
            early_stopping_rounds=100)
    pred_xgb_va = xgb.predict(X_va, iteration_range=(0, xgb.best_iteration+1))
    rmse_xgb = rmse(y_va, pred_xgb_va)
    print(f"[XGBoost+ES] RMSE: {rmse_xgb:.5f} (best_iter={xgb.best_iteration})")
else:
    rmse_xgb, pred_xgb_va = None, None
    print("XGBoost not available.")

# RandomForest (trimmed for speed in FAST_MODE)
rf = RandomForestRegressor(
    n_estimators=200 if FAST_MODE else 500,
    max_depth=16 if FAST_MODE else None,
    max_features="sqrt",
    min_samples_leaf=4 if FAST_MODE else 1,
    bootstrap=True, n_jobs=-1, random_state=SEED, verbose=1 if FAST_MODE else 0
)
rmse_rf, pred_rf_va = evaluate_model("RandomForest", rf, X_tr, X_va, y_tr, y_va)


# Optionally skip RF in stack to save time:
base_models = build_base_models(random_state=SEED)
#base_models = [(n,e) for (n,e) in base_models if n != "rf"]  # <- uncomment to drop RF from stack
stack = build_stack(base_models, random_state=SEED)
rmse_stack, pred_stack_va = evaluate_model("Stacked Ensemble", stack, X_tr, X_va, y_tr, y_va)


moe = MixtureOfExpertsRegressor(bands=(80,110))
moe.fit(X_tr, y_tr)
pred_moe_va = moe.predict(X_va)
rmse_moe = rmse(y_va, pred_moe_va)
print(f"[Mixture-of-Experts] RMSE: {rmse_moe:.5f}")


blend_cols = []
for p in [pred_lgb_va, pred_xgb_va, pred_rf_va, pred_ridge_va, pred_stack_va, pred_moe_va]:
    if p is not None:
        blend_cols.append(p)
blend_va = np.mean(np.vstack(blend_cols), axis=0)
rmse_blend = rmse(y_va, blend_va)
print(f"[Simple Blend Avg] RMSE: {rmse_blend:.5f}")


external = try_load_external_models()
rmse_ext_blend = None
if external:
    ext_cols = []
    if "rf_ext" in external:  ext_cols.append(external["rf_ext"].predict(X_va))
    if "xgb_ext" in external: ext_cols.append(external["xgb_ext"].predict(X_va))
    if "lgb_ext" in external: ext_cols.append(external["lgb_ext"].predict(X_va))
    if len(ext_cols) > 0:
        X_blend_va = np.column_stack(ext_cols)
        ridger = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0]))])
        ridger.fit(X_blend_va, y_va)
        p_ext = ridger.predict(X_blend_va)
        rmse_ext_blend = rmse(y_va, p_ext)
        print(f"[External Ridge Blend] RMSE: {rmse_ext_blend:.5f}")


cands = {
    "ridge": rmse_ridge,
    "lgb": rmse_lgb if rmse_lgb is not None else 1e9,
    "xgb": rmse_xgb if rmse_xgb is not None else 1e9,
    "rf": rmse_rf,
    "stack": rmse_stack,
    "moe": rmse_moe,
    "blend": rmse_blend,
}
if rmse_ext_blend is not None:
    cands["ext_blend"] = rmse_ext_blend

best_name = min(cands, key=cands.get)
print("\n=== RMSE Summary ===")
for k, v in cands.items():
    print(f"{k:>10}: {v:.5f}")
print(f"\nğŸ�† Best model for submission: {best_name}")


try:
    import optuna
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "optuna"])
    import optuna


import time
import optuna
from optuna.pruners import MedianPruner

# Sample for tuning to keep speed
X_tune, y_tune = maybe_sample(X_tr, y_tr, n_max=SAMPLE_HEAVY)

def objective_lgb(trial):
    if not HAVE_LGBM:
        return 1e9
    params = dict(
        n_estimators=4000,                 # cap + ES
        learning_rate=trial.suggest_float("lr", 0.02, 0.12),
        num_leaves=trial.suggest_int("num_leaves", 31, 127),
        max_depth=trial.suggest_int("max_depth", -1, 12),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 200),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
        random_state=SEED, n_jobs=-1
    )
    model = LGBMRegressor(**params)
    model.fit(X_tune, y_tune,
              eval_set=[(X_va, y_va)],
              eval_metric="l2",
              callbacks=[LGBMRegressor.early_stopping(stopping_rounds=100, verbose=False)] if hasattr(LGBMRegressor,'early_stopping') else None)
    pred = model.predict(X_va, num_iteration=getattr(model, "best_iteration_", None))
    return rmse(y_va, pred)

def objective_xgb(trial):
    if not HAVE_XGB:
        return 1e9
    params = dict(
        n_estimators=6000,                  # cap + ES
        learning_rate=trial.suggest_float("eta", 0.02, 0.12),
        max_depth=trial.suggest_int("max_depth", 4, 12),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 10.0),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("alpha", 0.0, 1.0),
        reg_lambda=trial.suggest_float("lambda", 0.0, 5.0),
        tree_method="hist", n_jobs=-1, random_state=SEED
    )
    model = XGBRegressor(**params)
    model.fit(X_tune, y_tune,
              eval_set=[(X_va, y_va)],
              eval_metric="rmse",
              early_stopping_rounds=100)
    pred = model.predict(X_va, iteration_range=(0, model.best_iteration+1))
    return rmse(y_va, pred)

N_TRIALS = 12 if FAST_MODE else 30   # keep small for Kaggle
TIMEOUT  = 900                       # 15 minutes safety

studies = {}
for name, obj in [("lgb", objective_lgb), ("xgb", objective_xgb)]:
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(obj, n_trials=N_TRIALS, timeout=TIMEOUT, show_progress_bar=False)
    print(f"\nOptuna {name} best value: {study.best_value:.5f}")
    print(f"Optuna {name} best params: {study.best_params}")
    studies[name] = study

# Build tuned models (if available)
lgb_tuned, xgb_tuned = None, None
if HAVE_LGBM and "lgb" in studies:
    bp = studies["lgb"].best_params
    lgb_tuned = LGBMRegressor(
        n_estimators=4000, random_state=SEED, n_jobs=-1,
        learning_rate=bp["lr"],
        num_leaves=bp["num_leaves"],
        max_depth=bp["max_depth"],
        subsample=bp["subsample"],
        colsample_bytree=bp["colsample_bytree"],
        min_child_samples=bp["min_child_samples"],
        reg_alpha=bp["reg_alpha"],
        reg_lambda=bp["reg_lambda"],
    )
    lgb_tuned.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)], eval_metric="l2",
                  callbacks=[LGBMRegressor.early_stopping(stopping_rounds=100, verbose=False)] if hasattr(LGBMRegressor,'early_stopping') else None)
    pred_lgb_tuned = lgb_tuned.predict(X_va, num_iteration=getattr(lgb_tuned, "best_iteration_", None))
    rmse_lgb_tuned = rmse(y_va, pred_lgb_tuned)
    print(f"[LightGBM Tuned] RMSE: {rmse_lgb_tuned:.5f}")

if HAVE_XGB and "xgb" in studies:
    bp = studies["xgb"].best_params
    xgb_tuned = XGBRegressor(
        n_estimators=6000, random_state=SEED, n_jobs=-1, tree_method="hist",
        learning_rate=bp["eta"],
        max_depth=bp["max_depth"],
        min_child_weight=bp["min_child_weight"],
        gamma=bp["gamma"],
        subsample=bp["subsample"],
        colsample_bytree=bp["colsample_bytree"],
        reg_alpha=bp["alpha"], reg_lambda=bp["lambda"],
    )
    xgb_tuned.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  eval_metric="rmse",
                  early_stopping_rounds=100)
    pred_xgb_tuned = xgb_tuned.predict(X_va, iteration_range=(0, xgb_tuned.best_iteration+1))
    rmse_xgb_tuned = rmse(y_va, pred_xgb_tuned)
    print(f"[XGBoost Tuned] RMSE: {rmse_xgb_tuned:.5f}")


# If tuned models are better, let them compete
if HAVE_LGBM and 'lgb_tuned' in locals():
    cands["lgb_tuned"] = rmse_lgb_tuned
if HAVE_XGB and 'xgb_tuned' in locals():
    cands["xgb_tuned"] = rmse_xgb_tuned

best_name = min(cands, key=cands.get)
print("\n=== UPDATED RMSE Summary (with Optuna) ===")
for k, v in cands.items():
    print(f"{k:>12}: {v:.5f}")
print(f"\nğŸ�† Updated best model for submission: {best_name}")


# Build final model by name (tuned boosters if they won)
if best_name == "ridge":
    final_model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=SEED))])
    final_model.fit(X_full, y_full)
    test_pred = final_model.predict(X_test)

elif best_name in ["lgb","lgb_tuned"]:
    if best_name == "lgb_tuned":
        bp = studies["lgb"].best_params
        final_model = LGBMRegressor(
            n_estimators=4000, random_state=SEED, n_jobs=-1,
            learning_rate=bp["lr"], num_leaves=bp["num_leaves"],
            max_depth=bp["max_depth"], subsample=bp["subsample"],
            colsample_bytree=bp["colsample_bytree"],
            min_child_samples=bp["min_child_samples"],
            reg_alpha=bp["reg_alpha"], reg_lambda=bp["reg_lambda"]
        )
    else:
        final_model = LGBMRegressor(
            n_estimators=900, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1
        )
    final_model.fit(X_full, y_full,
                    eval_set=[(X_va, y_va)],
                    eval_metric="l2",
                    verbose=False,
                    callbacks=[LGBMRegressor.early_stopping(stopping_rounds=100, verbose=False)] if hasattr(LGBMRegressor,'early_stopping') else None)
    test_pred = final_model.predict(X_test, num_iteration=getattr(final_model,"best_iteration_",None))

elif best_name in ["xgb","xgb_tuned"]:
    if best_name == "xgb_tuned":
        bp = studies["xgb"].best_params
        final_model = XGBRegressor(
            n_estimators=6000, learning_rate=bp["eta"], max_depth=bp["max_depth"],
            min_child_weight=bp["min_child_weight"], gamma=bp["gamma"],
            subsample=bp["subsample"], colsample_bytree=bp["colsample_bytree"],
            reg_alpha=bp["alpha"], reg_lambda=bp["lambda"],
            tree_method="hist", n_jobs=-1, random_state=SEED
        )
    else:
        final_model = XGBRegressor(
            n_estimators=700, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.0, reg_lambda=1.0,
            tree_method="hist", n_jobs=-1, random_state=SEED
        )
    final_model.fit(X_full, y_full,
                    eval_set=[(X_va, y_va)],
                    eval_metric="rmse",
                    verbose=False,
                    early_stopping_rounds=100)
    test_pred = final_model.predict(X_test, iteration_range=(0, final_model.best_iteration+1))

elif best_name == "rf":
    final_model = RandomForestRegressor(
        n_estimators=200 if FAST_MODE else 500,
        max_depth=16 if FAST_MODE else None,
        max_features="sqrt",
        min_samples_leaf=4 if FAST_MODE else 1,
        bootstrap=True, n_jobs=-1, random_state=SEED
    )
    final_model.fit(X_full, y_full)
    test_pred = final_model.predict(X_test)

elif best_name == "stack":
    base_models = build_base_models(random_state=SEED)
    # base_models = [(n,e) for (n,e) in base_models if n != "rf"]   # optional skip RF
    final_model = build_stack(base_models, random_state=SEED)
    final_model.fit(X_full, y_full)
    test_pred = final_model.predict(X_test)

elif best_name == "moe":
    final_model = MixtureOfExpertsRegressor(bands=(80,110))
    final_model.fit(X_full, y_full)
    test_pred = final_model.predict(X_test)

elif best_name == "blend":
    preds_test = []
    if HAVE_LGBM:
        lgb_final = LGBMRegressor(
            n_estimators=900 if not FAST_MODE else 600, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1
        )
        lgb_final.fit(X_full, y_full, eval_set=[(X_va, y_va)], eval_metric="l2",
                      verbose=False,
                      callbacks=[LGBMRegressor.early_stopping(stopping_rounds=100, verbose=False)] if hasattr(LGBMRegressor,'early_stopping') else None)
        preds_test.append(lgb_final.predict(X_test, num_iteration=getattr(lgb_final,"best_iteration_",None)))
    if HAVE_XGB:
        xgb_final = XGBRegressor(
            n_estimators=1200 if not FAST_MODE else 800, learning_rate=0.05,
            max_depth=8, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.0, reg_lambda=1.0, tree_method="hist",
            n_jobs=-1, random_state=SEED
        )
        xgb_final.fit(X_full, y_full, eval_set=[(X_va, y_va)],
                      eval_metric="rmse", verbose=False, early_stopping_rounds=100)
        preds_test.append(xgb_final.predict(X_test, iteration_range=(0, xgb_final.best_iteration+1)))
    rf_final = RandomForestRegressor(
        n_estimators=200 if FAST_MODE else 500,
        max_depth=16 if FAST_MODE else None,
        max_features="sqrt",
        min_samples_leaf=4 if FAST_MODE else 1,
        bootstrap=True, n_jobs=-1, random_state=SEED
    )
    rf_final.fit(X_full, y_full); preds_test.append(rf_final.predict(X_test))
    ridge_final = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=SEED))])
    ridge_final.fit(X_full, y_full); preds_test.append(ridge_final.predict(X_test))
    test_pred = np.mean(np.vstack(preds_test), axis=0)

elif best_name == "ext_blend":
    external = try_load_external_models()
    ext_cols_test, ext_cols_tr = [], []
    if "rf_ext" in external:
        ext_cols_test.append(external["rf_ext"].predict(X_test))
        ext_cols_tr.append(external["rf_ext"].predict(X_full))
    if "xgb_ext" in external:
        ext_cols_test.append(external["xgb_ext"].predict(X_test))
        ext_cols_tr.append(external["xgb_ext"].predict(X_full))
    if "lgb_ext" in external:
        ext_cols_test.append(external["lgb_ext"].predict(X_test))
        ext_cols_tr.append(external["lgb_ext"].predict(X_full))
    X_ext_test = np.column_stack(ext_cols_test)
    X_ext_tr = np.column_stack(ext_cols_tr)
    ridger = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0]))])
    ridger.fit(X_ext_tr, y_full)
    test_pred = ridger.predict(X_ext_test)

else:
    raise ValueError("Unknown winner.")


# Sanity checks & save
assert len(test_pred) == len(submission), "Prediction length mismatch with submission template."
out = submission.copy()
out["BeatsPerMinute"] = np.clip(test_pred, CLIP_MIN, CLIP_MAX).astype(float)
out.to_csv("submission.csv", index=False)

print("âœ… submission.csv written")
print("Prediction range:", float(out["BeatsPerMinute"].min()), "to", float(out["BeatsPerMinute"].max()))
print("Submission shape:", out.shape)
print("Columns:", out.columns.tolist())
submission




