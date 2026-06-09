import lightgbm as lgb, xgboost as xgb
import numpy as np, pandas as pd, gc, json, os, time, warnings

from catboost import CatBoostClassifier
from datetime import timedelta
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, PowerTransformer, OneHotEncoder,
    QuantileTransformer, RobustScaler, StandardScaler
)

warnings.filterwarnings("ignore")

SEED      = 42
N_SPLITS  = 5
MAX_HOURS = 11.7
T0        = time.time()

# output version folder
VERSION = "15_paramaware_sets"

# ---- knobs we can edit ----
RUN_MODELS = {"logreg","rf"} # {"logreg", "rf", "lgbm", "xgb", "cat", "gbc", "mlp"} 
RUN_PREPS  = {"ohe_qt", "ohe_yeo", "ohe_qt_eda", "ohe_yeo_eda", "ohe_robust", "ohe_std"}  # choose preprocessors
N_TUNE_ITERS = 12     # number of hyperparameter sets (RandomizedSearchCV trials)
SAVE_TOP_K_SETS = None  # None = save ALL tried sets; or put an int (e.g., 5) to limit saved sets per model+prep
# ---------------------------

def elapsed(): 
    return timedelta(seconds=round(time.time()-T0))

print("Notebook start:", elapsed())


DATA_DIR  = Path("/kaggle/input/playground-series-s5e8/")
OUT_DIR   = Path("model_outputs"); OUT_DIR.mkdir(exist_ok=True)

VER_DIR = OUT_DIR / f"v{VERSION}"; VER_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_VER_PATH    = VER_DIR / "results.csv"
RESULTS_MASTER_PATH = OUT_DIR / "results_master.csv"   # session master only (no reading old external files)

train  = pd.read_csv(DATA_DIR / "train.csv")
test   = pd.read_csv(DATA_DIR / "test.csv")
TARGET = "y"

num_cols = train.select_dtypes("number").columns.drop(["id", TARGET]).tolist()
cat_cols = train.select_dtypes("object").columns.tolist()

X = train.drop(columns=[TARGET])
y = train[TARGET].values

print(f"Numerics: {len(num_cols)} | Categoricals: {len(cat_cols)}")


def safe_nan_guard(X):
    return np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)

class AddEdaNumericFlags(BaseEstimator, TransformerMixin):
    def __init__(self, q3_duration=361, q3_balance=1390):
        self.q3_duration = q3_duration
        self.q3_balance  = q3_balance
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        pdays    = X["pdays"].values
        duration = X["duration"].values
        previous = X["previous"].values
        balance  = X["balance"].values
        X["pdays_is_neg1"]  = (pdays == -1).astype(int)
        X["long_call"]      = (duration > self.q3_duration).astype(int)
        X["any_prev"]       = (previous > 0).astype(int)
        X["recent_contact"] = (np.clip(pdays, 0, None) < 5).astype(int)
        X["high_balance"]   = (balance > self.q3_balance).astype(int)
        return X

ohe_basic = OneHotEncoder(handle_unknown="ignore", min_frequency=20,   sparse=False)
ohe_heavy = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=2000, sparse=False)

num_std = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  StandardScaler()),
    ("guard", FunctionTransformer(safe_nan_guard)),
])

num_qt = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("qt",  QuantileTransformer(output_distribution="normal", n_quantiles=1000, subsample=10_000)),
    ("guard", FunctionTransformer(safe_nan_guard)),
])

num_yeo = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("yeo", PowerTransformer(method="yeo-johnson", standardize=True)),
    ("guard", FunctionTransformer(safe_nan_guard)),
])

num_robust = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("rb",  RobustScaler()),
    ("vt",  VarianceThreshold(1e-12)),
    ("guard", FunctionTransformer(safe_nan_guard)),
])

num_qt_eda = Pipeline([
    ("eda", AddEdaNumericFlags()),
    ("imp", SimpleImputer(strategy="median")),
    ("qt",  QuantileTransformer(output_distribution="normal", n_quantiles=1000, subsample=10_000)),
    ("vt",  VarianceThreshold(1e-12)),
    ("guard", FunctionTransformer(safe_nan_guard)),
])

num_yeo_eda = Pipeline([
    ("eda", AddEdaNumericFlags()),
    ("imp", SimpleImputer(strategy="median")),
    ("yeo", PowerTransformer(method="yeo-johnson", standardize=True)),
    ("vt",  VarianceThreshold(1e-12)),
    ("guard", FunctionTransformer(safe_nan_guard)),
])

PREP_DICT = {
    "ohe_std": ColumnTransformer(
        [("num", num_std, num_cols), ("cat", ohe_basic, cat_cols)],
        remainder="drop", sparse_threshold=0.0
    ),
    "ohe_qt": ColumnTransformer(
        [("num", num_qt, num_cols), ("cat", ohe_heavy, cat_cols)],
        remainder="drop", sparse_threshold=0.0
    ),
    "ohe_yeo": ColumnTransformer(
        [("num", num_yeo, num_cols), ("cat", ohe_basic, cat_cols)],
        remainder="drop", sparse_threshold=0.0
    ),
    "ohe_qt_eda": ColumnTransformer(
        [("num", num_qt_eda, num_cols), ("cat", ohe_heavy, cat_cols)],
        remainder="drop", sparse_threshold=0.0
    ),
    "ohe_yeo_eda": ColumnTransformer(
        [("num", num_yeo_eda, num_cols), ("cat", ohe_basic, cat_cols)],
        remainder="drop", sparse_threshold=0.0
    ),
    "ohe_robust": ColumnTransformer(
        [("num", num_robust, num_cols), ("cat", ohe_heavy, cat_cols)],
        remainder="drop", sparse_threshold=0.0
    ),
}

print("Preprocessing variants:", list(PREP_DICT.keys()))


BASE_MODELS = {
    "logreg": (
        LogisticRegression(solver="saga", penalty="l2", class_weight="balanced",
                           max_iter=4000, tol=1e-3, random_state=SEED),
        {"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}
    ),
    "rf": (
        RandomForestClassifier(n_estimators=400, n_jobs=-1,
                               class_weight="balanced_subsample", random_state=SEED),
        {"max_depth": [None, 8, 12, 16, 20, 24],
         "min_samples_split": [2, 5, 10, 20],
         "min_samples_leaf": [1, 2, 4, 6],
         "max_features": [0.4, 0.6, 0.8, 1.0],
         "bootstrap": [True, False]}
    ),
    "lgbm": (
        lgb.LGBMClassifier(device_type="gpu", objective="binary",
                           random_state=SEED, is_unbalance=True, n_estimators=2000),
        {"learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
         "num_leaves": [31, 63, 95, 127, 191],
         "max_depth": [-1, 10, 12, 16, 20],
         "min_child_samples": [10, 20, 40, 60, 80],
         "subsample": [0.7, 0.85, 1.0],
         "colsample_bytree": [0.6, 0.8, 1.0],
         "reg_lambda": [0.0, 0.5, 1.0, 5.0, 10.0],
         "reg_alpha": [0.0, 0.5, 1.0, 5.0]}
    ),
    "xgb": (
        xgb.XGBClassifier(device="cuda", tree_method="hist", random_state=SEED,
                          eval_metric="auc", n_estimators=2000),
        {"eta": [0.02, 0.03, 0.05, 0.08, 0.1],
         "max_depth": [4, 5, 6, 7, 8],
         "min_child_weight": [1, 2, 3, 5, 8],
         "subsample": [0.7, 0.85, 1.0],
         "colsample_bytree": [0.6, 0.8, 1.0],
         "lambda": [0.5, 1.0, 5.0, 10.0],
         "alpha": [0.0, 0.5, 1.0, 5.0]}
    ),
    "cat": (
        CatBoostClassifier(
            task_type="GPU",
            random_seed=SEED,
            od_type="Iter",
            od_wait=200,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            allow_writing_files=False,
            auto_class_weights="Balanced",
            gpu_ram_part=0.12   # keep conservative to prevent OOM on CV folds
        ),
        {"depth": [6, 7, 8, 9, 10],
         "learning_rate": [0.03, 0.05, 0.07, 0.1],
         "l2_leaf_reg": [3, 5, 7, 10],
         "bagging_temperature": [0.0, 0.5, 1.0, 2.0],
         "border_count": [64, 128, 255]}
    ),
    "gbc": (
        GradientBoostingClassifier(random_state=SEED),
        {"learning_rate": [0.03, 0.05, 0.08, 0.1],
         "n_estimators": [300, 600, 900, 1200],
         "max_depth": [2, 3, 4, 5],
         "subsample": [0.7, 0.85, 1.0],
         "min_samples_leaf": [1, 2, 4]}
    ),
    "mlp": (
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=80, random_state=SEED),
        {"alpha": [1e-4, 3e-4, 1e-3, 3e-3],
         "learning_rate_init": [5e-4, 1e-3, 3e-3, 1e-2],
         "batch_size": [128, 256, 512]}
    ),
}

def resolved_preps():
    return [p for p in RUN_PREPS if p in PREP_DICT]


def save_results_table(rows):
    df = pd.DataFrame(rows)
    df["version"] = VERSION
    df.to_csv(RESULTS_VER_PATH, index=False)

    if RESULTS_MASTER_PATH.exists():
        old = pd.read_csv(RESULTS_MASTER_PATH)
        dfm = pd.concat([old, df], ignore_index=True)
    else:
        dfm = df.copy()
    dfm.to_csv(RESULTS_MASTER_PATH, index=False)


def eval_one_set(mdl_name, prep_tag, params, set_rank_idx, *, base_estimator, preproc, out_dir):
    """
    Returns (cv_auc, runtime_min, oof_path, sub_path).
    Saves OOF and submission for this (model, prep, set).
    set_rank_idx is 0-based; we format to 'set 01', 'set 02', ...
    """
    set_label = f"set {set_rank_idx+1:02d}"
    run_key   = f"{mdl_name}_{prep_tag}__{set_label.replace(' ', '_')}"

    start = time.time()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=float)
    test_preds = np.zeros(len(test), dtype=float)

    # prepare estimator with given params
    if mdl_name == "cat":
        est = base_estimator.__class__(**base_estimator.get_params())
        est.set_params(**params)
        for f, (tr, val) in enumerate(skf.split(X, y), 1):
            est.fit(X.iloc[tr], y[tr], cat_features=cat_cols)
            oof[val]    = est.predict_proba(X.iloc[val])[:, 1]
            test_preds += est.predict_proba(test)[:, 1] / N_SPLITS
    else:
        est = base_estimator.__class__(**base_estimator.get_params())
        est.set_params(**params)
        pipe = Pipeline([("prep", preproc), ("est", est)])
        for f, (tr, val) in enumerate(skf.split(X, y), 1):
            pipe.fit(X.iloc[tr], y[tr])
            oof[val]    = pipe.predict_proba(X.iloc[val])[:, 1]
            test_preds += pipe.predict_proba(test)[:, 1] / N_SPLITS

    cv_auc  = roc_auc_score(y, oof)
    runtime = (time.time()-start)/60

    # save artifacts
    oof_path = out_dir / f"oof_v{VERSION}_{run_key}.csv"
    sub_path = out_dir / f"sub_v{VERSION}_{run_key}.csv"
    pd.DataFrame({"id": train.id, "oof": oof}).to_csv(oof_path, index=False)
    pd.DataFrame({"id": test.id,  "y":   test_preds}).to_csv(sub_path, index=False)

    return cv_auc, runtime, str(oof_path), str(sub_path), set_label


results = []

models_to_run = [m for m in RUN_MODELS if m in BASE_MODELS]
preps_to_run  = resolved_preps()

print("Models this version:", models_to_run)
print("Preprocess this version:", preps_to_run)
print(f"Tuning trials per (model,prep): {N_TUNE_ITERS}; Save top-k: {SAVE_TOP_K_SETS or 'ALL'}")

for mdl_name in models_to_run:
    base_est, grid = BASE_MODELS[mdl_name]

    for prep_tag in preps_to_run:
        if (time.time()-T0) > MAX_HOURS*3600:
            print(f"\n !! Reached {MAX_HOURS} h – breaking loop.")
            break

        print(f"\n==> Searching: {mdl_name}_{prep_tag} — {elapsed()}")

        # Build the search object
        if mdl_name == "cat":
            search = RandomizedSearchCV(
                estimator=base_est,
                param_distributions=grid,
                n_iter=N_TUNE_ITERS,
                cv=3, scoring="roc_auc",
                random_state=SEED, n_jobs=-1, verbose=0
            )
            search.fit(X, y, cat_features=cat_cols)
        else:
            pipe = Pipeline([("prep", PREP_DICT[prep_tag]), ("est", base_est)])
            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions={f"est__{k}": v for k, v in grid.items()},
                n_iter=N_TUNE_ITERS,
                cv=3, scoring="roc_auc",
                random_state=SEED, n_jobs=-1, verbose=0
            )
            search.fit(X, y)

        # Rank all tried sets by mean CV score (best first)
        mean_scores = search.cv_results_["mean_test_score"]
        params_list = search.cv_results_["params"]
        order = np.argsort(-mean_scores)  # descending
        if SAVE_TOP_K_SETS is not None:
            order = order[:SAVE_TOP_K_SETS]

        # Evaluate each set with full 5-fold CV and persist artifacts
        for rank_idx, idx in enumerate(order):
            raw_params = params_list[idx]

            # strip "est__" for pipeline models
            if mdl_name == "cat":
                set_params = raw_params.copy()
            else:
                set_params = {k.replace("est__", "", 1): v for k, v in raw_params.items()}

            try:
                cv_auc, runtime, oof_path, sub_path, set_label = eval_one_set(
                    mdl_name, prep_tag, set_params, rank_idx,
                    base_estimator=base_est, preproc=PREP_DICT[prep_tag], out_dir=VER_DIR
                )

                results.append({
                    "model": f"{mdl_name} ({set_label})",
                    "pipeline": prep_tag,
                    "auc_mean": cv_auc,
                    "run_min": runtime,
                    "error": None,
                    "best_params": json.dumps(set_params),
                    "version": VERSION,
                    "oof_path": oof_path,
                    "sub_path": sub_path,
                })
                print(f"   saved {mdl_name}_{prep_tag} — {set_label}: CV-AUC={cv_auc:.6f}")

            except Exception as e:
                results.append({
                    "model": f"{mdl_name} (set {rank_idx+1:02d})",
                    "pipeline": prep_tag,
                    "auc_mean": np.nan,
                    "run_min": None,
                    "error": str(e),
                    "best_params": json.dumps(set_params),
                    "version": VERSION,
                    "oof_path": None,
                    "sub_path": None,
                })
                print(f"   FAILED set {rank_idx+1:02d} for {mdl_name}_{prep_tag}: {e}")

            save_results_table(results)
            gc.collect()


res_df = pd.DataFrame(results).sort_values("auc_mean", ascending=False)
print("\nBenchmark summary (this version)")
display(res_df)

pd.options.display.float_format = "{:.6f}".format
print(f"\nTotal elapsed: {elapsed()}")
print(f"Artifacts in: {VER_DIR}")




