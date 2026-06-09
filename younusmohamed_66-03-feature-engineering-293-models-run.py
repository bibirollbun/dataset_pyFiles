import os, gc, json, time, warnings
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

import optuna
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, RobustScaler, QuantileTransformer,
    PowerTransformer, FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# GPU-enabled libs
import lightgbm as lgb           # will use device_type='gpu'
import xgboost as xgb            # will use device='cuda'
from catboost import CatBoostClassifier  # will use task_type='GPU'

# Silence everything noisy
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.environ["CATBOOST_VERBOSE_LOGGING"] = "False"
os.environ["XGBOOST_ENABLE_STREAM_LOGGING"] = "0"

# LightGBM extra quiet
os.environ["LIGHTGBM_VERBOSE"] = "-1"

# ---- Run meta ----
CODE       = 3          
VERSION    = 20    
SEED       = 42
N_SPLITS   = 5
N_TRIALS   = 12         # hyperparam "sets" per (model, pipeline, stage)
MAX_HOURS  = 10       # wall-time budget
SAVE_TOP_K = None       # None or "ALL" -> save all sets; or an int (keep best K)

# Pick models & preprocessors & stages (we start at Stage 1 by request)
MODELS_THIS_VERSION   = ["xgb"]# ["lgbm", "xgb", "cat", "rf", "gbc", "logreg", "mlp"]
PREPROCS_THIS_VERSION = ["ohe_yeo", "ohe_yeo_eda", "ohe_std", "ohe_qt", "ohe_qt_eda", "ohe_robust"]
STAGES                = [2, 3]  # [1,2,3]

# Paths
DATA_DIR   = Path("/kaggle/input/playground-series-s5e8/")
OUT_DIR    = Path("model_outputs")
VER_DIR    = OUT_DIR / f"v{VERSION}"
OUT_DIR.mkdir(exist_ok=True)
VER_DIR.mkdir(exist_ok=True)
RESULTS_CSV = OUT_DIR / f"results_v{VERSION}.csv"  # flat table requested
HP_DIR      = VER_DIR / "params"
HP_DIR.mkdir(exist_ok=True)

# Timers
T0 = time.time()
def elapsed() -> timedelta:
    return timedelta(seconds=round(time.time() - T0))
def time_left_ok() -> bool:
    return (time.time() - T0) <= MAX_HOURS * 3600

print(f"Notebook start: {elapsed()}")
print(f"Saving to: {VER_DIR}")


train  = pd.read_csv(DATA_DIR / "train.csv")
test   = pd.read_csv(DATA_DIR / "test.csv")
TARGET = "y"

num_cols = train.select_dtypes(include="number").columns.drop(["id", TARGET]).tolist()
cat_cols = train.select_dtypes(include="object").columns.tolist()

X = train.drop(columns=[TARGET])
y = train[TARGET].astype(int).values

print(f"Rows: {len(train):,}  | Test: {len(test):,}")
print(f"Numerics: {len(num_cols)} | Categoricals: {len(cat_cols)}")


def safe_nan_guard(X):
    # ensure finite matrix after transforms
    return np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)

class AddEdaNumericFlags(BaseEstimator, TransformerMixin):
    """
    STAGE 1 — 'necessary & recommended' flags from your EDA:
      - pdays_is_neg1: never contacted before
      - recent_contact: pdays clipped at 0 is < 5 (very recent)
      - long_call: duration above ~Q3 (361 s)
      - any_prev: previous > 0
      - high_balance: balance above ~Q3 (1390)
    """
    def __init__(self, q3_duration=361, q3_balance=1390):
        self.q3_duration = q3_duration
        self.q3_balance  = q3_balance

    def fit(self, X, y=None): return self

    def transform(self, X):
        X = X.copy()
        cols = set(X.columns)
        if "pdays" in cols:
            X["pdays_is_neg1"]  = (X["pdays"] == -1).astype(int)
            X["recent_contact"] = (np.clip(X["pdays"], 0, None) < 5).astype(int)
        if "duration" in cols:
            X["long_call"] = (X["duration"] > self.q3_duration).astype(int)
        if "previous" in cols:
            X["any_prev"] = (X["previous"] > 0).astype(int)
        if "balance" in cols:
            X["high_balance"] = (X["balance"] > self.q3_balance).astype(int)
        return X

class AddFE50(BaseEstimator, TransformerMixin):
    """
    STAGE 2 — +50~70 meaningful features:
      - log1p/sqrt/square for heavy-tailed numerics
      - safe ratios & products among key numerics
      - 5-quantile bins for coarse groupings
      - pdays positive split and interactions with campaign/duration
    """
    def fit(self, X, y=None): return self

    def _log1p_cols(self, df, cols):
        for c in cols:
            df[f"log1p_{c}"] = np.log1p(np.where(c == "pdays", np.maximum(df[c], 0), df[c]))
        return df

    def _sqrt_cols(self, df, cols):
        for c in cols:
            df[f"sqrt_{c}"] = np.sqrt(np.clip(df[c], a_min=0, a_max=None))
        return df

    def _sq_cols(self, df, cols):
        for c in cols:
            df[f"sq_{c}"] = df[c] ** 2
        return df

    def _ratios_products_bins(self, df, cols):
        C = set(cols)
        def get(name): return name in C
        eps = 1e-6
        def rdiv(a, b):
            out = a / (b + eps)
            return np.clip(out, -1e9, 1e9)  # keep finite & reasonable

        # pairs using strongest numerics from EDA
        if get("balance") and get("duration"):
            df["balance_per_duration"] = rdiv(df["balance"], df["duration"] + 1)
            df["prod_balance_duration"] = df["balance"] * df["duration"]

        if get("duration") and get("campaign"):
            df["duration_per_campaign"] = rdiv(df["duration"], df["campaign"] + 1)
            df["prod_duration_campaign"] = df["duration"] * df["campaign"]

        if get("previous") and get("campaign"):
            df["previous_per_campaign"] = rdiv(df["previous"], df["campaign"] + 1)
            df["prod_previous_campaign"] = df["previous"] * df["campaign"]

        if get("balance") and get("previous"):
            df["balance_per_previous"] = rdiv(df["balance"], df["previous"] + 1)
            df["prod_balance_previous"] = df["balance"] * df["previous"]

        if get("pdays"):
            pdays_pos = np.clip(df["pdays"], 0, None)
            df["pdays_pos"] = pdays_pos
            if get("duration"):
                df["duration_per_pdays_pos"] = rdiv(df["duration"], pdays_pos + 1)
            if get("campaign"):
                df["pdays_pos_per_campaign"] = rdiv(pdays_pos, df["campaign"] + 1)

        # coarse quantile bins
        for c in ["age", "balance", "duration", "campaign", "pdays", "previous"]:
            if c in C:
                try:
                    df[f"{c}_qbin5"] = pd.qcut(df[c], q=5, labels=False, duplicates="drop").astype(float)
                except Exception:
                    df[f"{c}_qbin5"] = -1.0
        return df

    def transform(self, X):
        df = X.copy()
        base_cols = X.columns.tolist()
        df = self._log1p_cols(df, base_cols)
        df = self._sqrt_cols(df, base_cols)
        df = self._sq_cols(df, base_cols)
        df = self._ratios_products_bins(df, base_cols)
        return df

class AddFE100(BaseEstimator, TransformerMixin):
    """
    STAGE 3 — +50 more to reach ~100+:
      - selective pairwise products & diffs among base numerics
      - cubic terms for heavy-tailed offenders
      - simple row-wise aggregates across current numeric block
    """
    def fit(self, X, y=None): return self

    def transform(self, X):
        df = X.copy()
        # choose base numerics (ignore already-derived log/sqrt/sq/bin/etc.)
        base = [c for c in df.columns
                if df[c].dtype != object
                and not (c.startswith(("log1p_","sqrt_","sq_","prod_","diff_")) or
                         c.endswith("_qbin5") or
                         c.endswith(("_per_campaign","_per_previous","_per_duration")) or
                         c in ("pdays_pos","num_sum","num_mean"))]

        # pairwise limited interactions
        for i in range(len(base)):
            for j in range(i+1, len(base)):
                a, b = base[i], base[j]
                df[f"prod_{a}_{b}"] = df[a] * df[b]
                df[f"diff_{a}_{b}"] = df[a] - df[b]

        # cubic of key numerics
        for c in ["duration", "balance", "campaign", "previous"]:
            if c in df.columns:
                df[f"cube_{c}"] = df[c] ** 3

        # global aggregates
        numeric_now = df.select_dtypes(include=[np.number]).columns
        df["num_sum"]  = df[numeric_now].sum(axis=1)
        df["num_mean"] = df[numeric_now].mean(axis=1)
        return df


def num_block_base(tag: str):
    if tag in ("ohe_std",):
        scaler = ("std", StandardScaler())
    elif tag in ("ohe_qt", "ohe_qt_eda"):
        scaler = ("qt", QuantileTransformer(output_distribution="normal", n_quantiles=1000, subsample=10_000))
    elif tag in ("ohe_yeo", "ohe_yeo_eda"):
        scaler = ("yeo", PowerTransformer(method="yeo-johnson", standardize=True))
    elif tag in ("ohe_robust",):
        scaler = ("rob", RobustScaler())
    else:
        scaler = ("std", StandardScaler())

    return scaler

def make_numeric_pipeline(prep_tag: str, stage: int):
    steps = []
    # Add EDA flags:
    wants_eda = ("eda" in prep_tag) or (stage >= 1)
    if wants_eda:
        steps.append(("eda", AddEdaNumericFlags()))

    # Stage additions
    if stage >= 2:
        steps.append(("fe50", AddFE50()))
    if stage >= 3:
        steps.append(("fe100", AddFE100()))

    # Sanitize BEFORE imputer to kill ±inf/NaN from FE
    steps.append(("guard_pre", FunctionTransformer(safe_nan_guard)))

    # Impute -> drop zero-variance -> THEN scale (YJ/Qt/Std/Robust)
    steps.append(("imp", SimpleImputer(strategy="median")))
    steps.append(("vt", VarianceThreshold(1e-12)))           # <-- moved up
    steps.append(num_block_base(prep_tag))                    # <-- scaler after VT
    steps.append(("guard", FunctionTransformer(safe_nan_guard)))
    return Pipeline(steps)

def make_preprocessor(prep_tag: str, stage: int, *, for_cat: bool = False) -> ColumnTransformer:
    """ColumnTransformer for non-Cat models; if for_cat=True, pass categoricals through."""
    num_pipe = make_numeric_pipeline(prep_tag, stage)
    if for_cat:
        # CatBoost will handle categoricals natively; keep them raw (passthrough)
        return ColumnTransformer(
            transformers=[("num", num_pipe, num_cols),
                          ("cat", "passthrough", cat_cols)],
            remainder="drop", sparse_threshold=0.0
        )
    else:
        # One-hot for other models
        ohe_basic = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        ohe_heavy = OneHotEncoder(sparse_output=False, handle_unknown="ignore", min_frequency=0.01)
        cat_enc   = ohe_heavy if ("robust" in prep_tag or "qt" in prep_tag) else ohe_basic
        return ColumnTransformer(
            transformers=[("num", num_pipe, num_cols),
                          ("cat", cat_enc, cat_cols)],
            remainder="drop", sparse_threshold=0.0
        )

# ---------- Feature preview helpers ----------
# def _derive_numeric_names(num_pipe, X_num: pd.DataFrame) -> list[str]:
#     """Try to keep names from EDA/FE steps and apply VT mask."""
#     num_pipe = clone(num_pipe)
#     names = list(X_num.columns)

#     # EDA/FE steps output DataFrames with names
#     pre_vt_steps = []
#     for k, step in num_pipe.named_steps.items():
#         if k in ("vt", "guard"):  # apply separately
#             continue
#         pre_vt_steps.append((k, step))
#     pre_vt = Pipeline(pre_vt_steps)

#     X_pre = pre_vt.fit_transform(X_num.head(5000))
#     # Try to keep names if transformer returns DataFrame
#     if hasattr(X_pre, "columns"):
#         names = list(X_pre.columns)
#         Z = X_pre.to_numpy()
#     else:
#         # fallback: placeholder names
#         Z = X_pre
#         names = [f"num_{i}" for i in range(Z.shape[1])]

#     if "vt" in num_pipe.named_steps:
#         vt = num_pipe.named_steps["vt"]
#         vt.fit(Z)
#         mask = vt.get_support()
#         names = [n for n, keep in zip(names, mask) if keep]
#     return names

# def get_feature_names_for_preproc(preproc: ColumnTransformer, X_df: pd.DataFrame) -> list[str]:
#     """Return combined feature names ['num:...', 'cat:...']."""
#     # numeric names
#     num_spec = None
#     for name, tr, cols in preproc.transformers:
#         if name == "num":
#             num_spec = tr
#             break
#     assert num_spec is not None
#     num_names = _derive_numeric_names(num_spec, X_df[num_cols])

#     # cat names
#     cat_names = []
#     for name, tr, cols in preproc.transformers:
#         if name != "cat": 
#             continue
#         if tr == "passthrough":
#             cat_names = [f"{c}" for c in cols]
#         else:
#             enc = clone(tr)
#             enc.fit(X_df[cat_cols].head(10000))
#             cat_names = enc.get_feature_names_out(cat_cols).tolist()

#     return [f"num:{n}" for n in num_names] + [f"cat:{n}" for n in cat_names]

# def preview_stage_features(stage_label: str, preproc_tag: str, preproc: ColumnTransformer, X_df: pd.DataFrame):
#     feats = get_feature_names_for_preproc(preproc, X_df)
#     print(f"\n[Stage {stage_label}] {preproc_tag}: using {len(feats)} features")
#     print(feats)


# === PATCH: safer feature-name preview (no imputer/scaler run) ===

def _fg_only_pipeline(num_pipe: Pipeline) -> Pipeline:
    """Extract only feature-generating steps (EDA/FE50/FE100) to preserve names."""
    keep = []
    for k, step in num_pipe.named_steps.items():
        if isinstance(step, (AddEdaNumericFlags, AddFE50, AddFE100)):
            keep.append((k, step))
    return Pipeline(keep)

def _derive_numeric_names(num_pipe: Pipeline, X_num: pd.DataFrame) -> list[str]:
    num_pipe = clone(num_pipe)

    # Run only feature-generation steps
    fg = _fg_only_pipeline(num_pipe)
    sample = X_num.iloc[: max(1, min(5000, len(X_num)))]
    if len(fg.steps) == 0:
        df_pre = sample.copy()
    else:
        df_pre = fg.fit_transform(sample)

    if not hasattr(df_pre, "columns"):
        df_pre = pd.DataFrame(df_pre, columns=list(X_num.columns))

    names = list(df_pre.columns)

    Z = np.nan_to_num(df_pre.values, nan=0.0, posinf=1e9, neginf=-1e9)

    # Only apply VT if we actually have rows
    if Z.shape[0] > 0:
        vt = VarianceThreshold(1e-12)
        vt.fit(Z)
        mask = vt.get_support()
        names = [n for n, keep in zip(names, mask) if keep]

    return names

def get_feature_names_for_preproc(preproc: ColumnTransformer, X_df: pd.DataFrame) -> list[str]:
    """Return combined feature names ['num:...', 'cat:...'] using the safe preview extractor above."""
    # numeric names
    num_spec = None
    for name, tr, cols in preproc.transformers:
        if name == "num":
            num_spec = tr
            break
    assert num_spec is not None
    num_names = _derive_numeric_names(num_spec, X_df[num_cols])

    # categorical names
    cat_names = []
    for name, tr, cols in preproc.transformers:
        if name != "cat":
            continue
        if tr == "passthrough":
            cat_names = [f"{c}" for c in cols]
        else:
            enc = clone(tr)
            enc.fit(X_df[cat_cols].head(10000))
            cat_names = enc.get_feature_names_out(cat_cols).tolist()

    return [f"num:{n}" for n in num_names] + [f"cat:{n}" for n in cat_names]

def preview_stage_features(stage_label: str, preproc_tag: str, preproc: ColumnTransformer, X_df: pd.DataFrame):
    feats = get_feature_names_for_preproc(preproc, X_df)
    print(f"\n[Stage {stage_label}] {preproc_tag}: using {len(feats)} features")
    print(feats)


def make_estimator(mdl_name: str, seed: int, params: dict):
    if mdl_name == "logreg":
        from sklearn.linear_model import LogisticRegression
        est = LogisticRegression(
            solver="saga", penalty="l2", class_weight="balanced",
            max_iter=4000, tol=1e-3, random_state=seed
        )
        est.set_params(**params)
        return est

    if mdl_name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        est = RandomForestClassifier(
            n_estimators=400, n_jobs=-1,
            class_weight="balanced_subsample", random_state=seed
        )
        est.set_params(**params)
        return est

    if mdl_name == "gbc":
        from sklearn.ensemble import GradientBoostingClassifier
        est = GradientBoostingClassifier(random_state=seed)
        est.set_params(**params)
        return est

    if mdl_name == "mlp":
        from sklearn.neural_network import MLPClassifier
        est = MLPClassifier(max_iter=200, random_state=seed)
        est.set_params(**params)
        return est

    if mdl_name == "lgbm":
        est = lgb.LGBMClassifier(
            objective="binary", is_unbalance=True, random_state=seed,
            n_estimators=4000, device_type="gpu", verbosity=-1
        )
        est.set_params(**params)
        return est

    if mdl_name == "xgb":
        est = xgb.XGBClassifier(
            device="cuda", tree_method="hist", n_estimators=4000,
            eval_metric="auc", random_state=seed, n_jobs=-1, verbosity=0
        )
        est.set_params(**params)
        return est

    if mdl_name == "cat":
        est = CatBoostClassifier(
            task_type="GPU", loss_function="Logloss", eval_metric="AUC",
            random_seed=seed, od_type="Iter", od_wait=200,
            verbose=False, allow_writing_files=False,
            auto_class_weights="Balanced"
        )
        # You can tweak GPU memory use here if needed:
        # est.set_params(gpu_ram_part=0.15)
        est.set_params(**params)
        return est

    raise ValueError(mdl_name)

def suggest_params(trial, mdl_name: str):
    if mdl_name == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}

    if mdl_name == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 6, 24),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }

    if mdl_name == "gbc":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 400, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 200),
        }

    if mdl_name == "mlp":
        h1 = trial.suggest_int("h1", 64, 512, step=64)
        h2 = trial.suggest_int("h2", 32, 256, step=32)
        return {
            "hidden_layer_sizes": (h1, h2),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "solver": trial.suggest_categorical("solver", ["adam", "sgd"]),
        }

    if mdl_name == "lgbm":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", -1, 24),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 5.0),
            "max_bin": trial.suggest_int("max_bin", 64, 255),
        }

    if mdl_name == "xgb":
        return {
            "eta": trial.suggest_float("eta", 0.005, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 100.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            "max_bin": trial.suggest_int("max_bin", 128, 512),
        }

    if mdl_name == "cat":
        return {
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
            "border_count": trial.suggest_int("border_count", 64, 255),
        }

    raise ValueError(mdl_name)


def save_oof_and_sub(version, run_key, set_id, oof, test_pred, train_ids, test_ids):
    set_tag = f"set{set_id:02d}"
    oof_path = VER_DIR / f"oof_v{version}_{run_key}_{set_tag}.csv"
    sub_path = VER_DIR / f"sub_v{version}_{run_key}_{set_tag}.csv"
    pd.DataFrame({"id": train_ids, "oof": oof}).to_csv(oof_path, index=False)
    pd.DataFrame({"id": test_ids,  "y":  test_pred}).to_csv(sub_path, index=False)
    return oof_path, sub_path

def append_results_row(model_name, pipeline, auc_mean, minutes, params_set, params_dict):
    row = {
        "code": CODE,
        "model": f"{model_name} ({params_set})",  # e.g. "lgbm (set 01)"
        "pipeline": pipeline,
        "auc_mean": float(auc_mean),
        "run_min": float(minutes),
        "version": VERSION,
        "public score": "",        # left blank
        "params_set": params_set,  # extra explicit column
    }
    df = pd.DataFrame([row])
    if RESULTS_CSV.exists():
        all_prev = pd.read_csv(RESULTS_CSV)
        all_prev = pd.concat([all_prev, df], ignore_index=True)
    else:
        all_prev = df
    all_prev.to_csv(RESULTS_CSV, index=False)

    # also save the params to a JSON file
    with open(HP_DIR / f"params_{model_name}_{pipeline}_{params_set}.json", "w") as f:
        json.dump(params_dict, f, indent=2)

    print(f"   saved {model_name}_{pipeline} — {params_set}: CV-AUC={auc_mean:.6f}")


def run_one_set(mdl_name: str, preproc: ColumnTransformer, set_id: int, params: dict):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=float)
    test_pred = np.zeros(len(test), dtype=float)

    for f, (tr, val) in enumerate(skf.split(X, y), start=1):
        Xtr, Xval = X.iloc[tr], X.iloc[val]
        ytr, yval = y[tr], y[val]

        est = make_estimator(mdl_name, SEED, params)

        if mdl_name == "cat":
            # Build a CatBoost-friendly table: apply numeric stage transforms via preproc.num,
            # but pass categoricals raw (preproc has cat passthrough).
            preproc_fit = clone(preproc).fit(Xtr, ytr)
            Xtr_tf = preproc_fit.transform(Xtr)
            Xval_tf = preproc_fit.transform(Xval)
            Xte_tf  = preproc_fit.transform(test)

            # Build column names list for cat features indices
            # numeric names
            num_names = _derive_numeric_names(preproc.transformers[0][1], Xtr[num_cols])
            # categorical names (passthrough):
            cat_names = cat_cols[:]
            feat_names = num_names + cat_names

            # CatBoost can take Pools with feature names & categorical feature indices
            from catboost import Pool
            cat_idx = list(range(len(num_names), len(feat_names)))

            tr_pool  = Pool(Xtr_tf, label=ytr, cat_features=cat_idx, feature_names=feat_names)
            val_pool = Pool(Xval_tf, label=yval, cat_features=cat_idx, feature_names=feat_names)
            te_pool  = Pool(Xte_tf,               cat_features=cat_idx, feature_names=feat_names)

            est.fit(tr_pool, eval_set=val_pool, use_best_model=True)
            oof[val]   = est.predict_proba(val_pool)[:, 1]
            test_pred += est.predict_proba(te_pool)[:, 1] / N_SPLITS

        elif mdl_name in {"lgbm", "xgb"}:
            # For boosting libs with OHE preprocessors
            preproc_fit = clone(preproc).fit(Xtr, ytr)
            Xtr_p, Xval_p, Xte_p = preproc_fit.transform(Xtr), preproc_fit.transform(Xval), preproc_fit.transform(test)

            if mdl_name == "lgbm":
                est.fit(Xtr_p, ytr,
                        eval_set=[(Xval_p, yval)],
                        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)])
            else:
                est.fit(Xtr_p, ytr,
                        eval_set=[(Xval_p, yval)],
                        verbose=False, early_stopping_rounds=200)

            oof[val] = est.predict_proba(Xval_p)[:, 1]
            test_pred += est.predict_proba(Xte_p)[:, 1] / N_SPLITS

        else:
            # sklearn API models through a Pipeline
            pipe = Pipeline([("prep", preproc), ("est", est)])
            pipe.fit(Xtr, ytr)
            oof[val] = pipe.predict_proba(Xval)[:, 1]
            test_pred += pipe.predict_proba(test)[:, 1] / N_SPLITS

    cv_auc = roc_auc_score(y, oof)
    return cv_auc, oof, test_pred


print("Models this version:", MODELS_THIS_VERSION)
print("Preprocess this version:", PREPROCS_THIS_VERSION)
print("Stages:", STAGES)
print(f"Tuning trials per (model,prep,stage): {N_TRIALS}; Save top-k:", "ALL" if SAVE_TOP_K in (None, "ALL") else SAVE_TOP_K)

stop_all = False

for stage in STAGES:
    if stop_all or not time_left_ok():
        print(f"\n⏱️ Time budget reached at {elapsed()}. Stopping.")
        break

    # Preview feature names for each preprocessor once per stage
    for prep_tag in PREPROCS_THIS_VERSION:
        # preview for standard (non-cat) version
        preproc_preview = make_preprocessor(prep_tag, stage, for_cat=False)
        preview_stage_features(stage_label=f"s{stage}", preproc_tag=prep_tag,
                               preproc=preproc_preview, X_df=X)

    # Now train per model + preprocessor
    for mdl_name in MODELS_THIS_VERSION:
        for prep_tag in PREPROCS_THIS_VERSION:
            if stop_all or not time_left_ok():
                print(f"\n⏱️ Time budget reached at {elapsed()}. Stopping.")
                stop_all = True
                break

            # For CatBoost: use passthrough cats; for others: OHE cats
            preproc = make_preprocessor(prep_tag, stage, for_cat=(mdl_name == "cat"))
            run_key = f"{mdl_name}_{prep_tag}_s{stage}"
            print(f"\n==> Searching: {run_key} — {elapsed()}")

            # Optuna tuning
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))

            def objective(trial):
                if not time_left_ok():
                    raise optuna.exceptions.OptunaError("Time budget exceeded")
                params = suggest_params(trial, mdl_name)
                cv_auc, _, _ = run_one_set(mdl_name, preproc, set_id=trial.number+1, params=params)
                return cv_auc

            try:
                study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
            except optuna.exceptions.OptunaError:
                print("   (budget hit mid-search)")

            # sort trials by score
            trials_sorted = sorted(
                [t for t in study.trials if t.value is not None],
                key=lambda t: t.value, reverse=True
            )

            to_save = trials_sorted if SAVE_TOP_K in (None, "ALL") else trials_sorted[:int(SAVE_TOP_K)]

            # re-train each chosen params set, save OOF/sub & results
            for tr in to_save:
                if not time_left_ok():
                    print(f"⏱️ Time budget reached at {elapsed()} while saving. Stopping.")
                    stop_all = True
                    break

                set_id    = tr.number + 1
                params    = tr.params.copy()
                params_set = f"set {set_id:02d}"

                start = time.time()
                cv_auc, oof, test_pred = run_one_set(mdl_name, preproc, set_id=set_id, params=params)
                minutes = (time.time() - start) / 60.0

                # save OOF & submission
                oof_path, sub_path = save_oof_and_sub(
                    version=VERSION, run_key=run_key, set_id=set_id,
                    oof=oof, test_pred=test_pred,
                    train_ids=train.id.values, test_ids=test.id.values
                )

                # append results row (flat CSV) and params JSON
                append_results_row(
                    model_name=mdl_name, pipeline=prep_tag,
                    auc_mean=cv_auc, minutes=minutes,
                    params_set=params_set, params_dict=params
                )

            if stop_all:
                break
        if stop_all:
            break

print(f"\nDone at {elapsed()}. Results → {RESULTS_CSV}")


RESULTS_CSV




