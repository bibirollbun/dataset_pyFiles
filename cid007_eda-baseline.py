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


# ---------- Theme & Imports ----------
import warnings, os, random, json, math, time, gc, sys
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd

# Visuals
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import gridplot

# ML / FE
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFECV
from sklearn.linear_model import RidgeCV, ElasticNet
from sklearn.feature_selection import mutual_info_regression
from sklearn.utils import check_random_state
from scipy.stats import randint, uniform, loguniform
from itertools import combinations

# Encoders & models
try:
    import category_encoders as ce
except Exception:
    !pip -q install category-encoders
    import category_encoders as ce

try:
    import lightgbm as lgb
except Exception:
    !pip -q install lightgbm
    import lightgbm as lgb

try:
    import xgboost as xgb
except Exception:
    !pip -q install xgboost
    import xgboost as xgb

try:
    from catboost import CatBoostRegressor
except Exception:
    !pip -q install catboost
    from catboost import CatBoostRegressor

# Optional SHAP
try:
    import shap
    shap_available = True
except Exception:
    shap_available = False

from IPython.display import HTML, display

output_notebook()

SEED = 42
rng = check_random_state(SEED)
np.random.seed(SEED); random.seed(SEED)

THEME_CSS = """
<style>
:root{--bg:#0b1020;--card:#131a2a;--ink:#e6edf3;--muted:#94a3b8;--accent:#8b5cf6;--accent2:#22d3ee;}
html,body{background:var(--bg);color:var(--ink);}
.k-title{font-size:2.0rem;font-weight:800;letter-spacing:.3px;margin:12px 0 6px;}
.k-sub{font-size:1.0rem;color:var(--muted);margin-bottom:12px}
.card{background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.02));
border:1px solid rgba(255,255,255,.08);padding:14px 16px;border-radius:16px;box-shadow:0 6px 20px rgba(0,0,0,.25);margin:8px 0}
.k-mono{font-family:ui-monospace,Menlo,Consolas,monospace;}
.k-pill{border:1px dashed rgba(255,255,255,.18);border-radius:999px;padding:4px 10px;display:inline-block;color:var(--muted)}
</style>
"""
display(HTML(THEME_CSS))
display(HTML("<div class='card'><div class='k-title'>Kaggle S5E9 â€” Predict BPM</div><div class='k-sub'>Pretty EDA (Seaborn/Plotly/Bokeh) + DOEâ€‘style FE + Nested CV + Stacking</div></div>"))



INPUT_DIR = "/kaggle/input/playground-series-s5e9"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH  = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_PATH), "Add the competition dataset as an input to this notebook."

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB)

TARGET = "BeatsPerMinute"
ID_COL = "id"

display(HTML(f"<div class='card k-mono'>Loaded âœ”ï¸� â€” train: {train.shape}, "
             f"test: {test.shape}, sample_sub: {sample_sub.shape}</div>"))
display(HTML(f"<div class='card k-mono'>Target column: {TARGET} | ID column: {ID_COL}</div>"))


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif df[col].dtype == "float64":
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df

train = reduce_memory(train); test = reduce_memory(test)

feature_cols = [c for c in train.columns if c != TARGET]
cat_cols = [c for c in feature_cols if train[c].dtype == "object" or str(train[c].dtype).startswith("category")]
num_cols = [c for c in feature_cols if c not in cat_cols]

info_html = f"""
<div class="card k-mono">
Rows: {len(train):,} â€” Features: {len(feature_cols)} â€” Num: {len(num_cols)} â€” Cat: {len(cat_cols)}
</div>
"""
display(HTML(info_html))



fig = px.histogram(train, x=TARGET, nbins=60, marginal='box', title=f"Target Distribution â€” {TARGET}")
fig.update_layout(template='plotly_dark', height=420)
fig.show()




missing = train.isna().mean().sort_values(ascending=False)
miss_df = missing.reset_index()
miss_df.columns = ["feature","missing_rate"]
miss_df = miss_df[miss_df["missing_rate"]>0]
if len(miss_df):
    fig = px.bar(miss_df, x="feature", y="missing_rate", title="Missing Value Rate (Train)")
    fig.update_layout(template="plotly_dark", xaxis={"tickangle":45})
    fig.show()
else:
    display(HTML("<span class='k-pill'>No missing values detected in train</span>"))




samp = train.sample(min(4000, len(train)), random_state=SEED)
if num_cols:
    for chunk_start in range(0, len(num_cols), 6):
        cols_chunk = num_cols[chunk_start:chunk_start+6]
        melt_df = samp[cols_chunk + [TARGET]].melt(id_vars=TARGET, var_name="feature", value_name="value")
        fig = px.violin(melt_df, x="feature", y="value", color=None, box=True, points="outliers")
        fig.update_layout(template="plotly_dark", height=420, title=f"Violin Distributions (chunk {chunk_start//6+1})")
        fig.show()



if len(num_cols) >= 2:
    corr = train[num_cols + [TARGET]].corr(numeric_only=True)
    fig = px.imshow(corr, aspect="auto", color_continuous_scale="Viridis", title="Correlation Heatmap")
    fig.update_layout(template="plotly_dark", height=max(420, 16*len(corr)))
    fig.show()



if len(num_cols) >= 2:
    mi_sample = train.sample(min(20000, len(train)), random_state=SEED)
    mi_scores = mutual_info_regression(mi_sample[num_cols], mi_sample[TARGET], random_state=SEED)
    mi_df = pd.DataFrame({"feature": num_cols, "mi": mi_scores}).sort_values("mi", ascending=False)
    top_mi = mi_df.head(min(6, len(mi_df)))
    plots = []
    for f in top_mi["feature"]:
        sub = mi_sample[[f, TARGET]].dropna()
        cds = ColumnDataSource(sub)
        p = figure(width=340, height=280, background_fill_color="#0b1020")
        p.add_tools(HoverTool(tooltips=[    ("feat", f),     (TARGET, f"@{TARGET}{{0.00}}"),     (f, f"@{f}{{0.00}}")]))
        p.circle(x=f, y=TARGET, source=cds, size=4, alpha=0.5)
        p.title.text = f"{f} vs {TARGET}"
        plots.append(p)
    grid = gridplot(plots, ncols=3)
    show(grid)



FAST_MODE = True
MAX_BASE_TRANS_PER_FEATURE = 3

X = train[feature_cols].copy()
X_test = test[feature_cols].copy()

# Frequency encoding (non-leaky, fit on train only)
for c in cat_cols:
    freqs = X[c].value_counts(dropna=False)
    X[f"{c}__freq"] = X[c].map(freqs)
    X_test[f"{c}__freq"] = X_test[c].map(freqs).fillna(0)

def add_transforms(Xdf, Xtestdf, y, num_cols):
    X_new = Xdf.copy(); Xtest_new = Xtestdf.copy(); trans_added = []
    if not num_cols: return X_new, Xtest_new, trans_added
    base_mi = mutual_info_regression(Xdf[num_cols].fillna(Xdf[num_cols].median()), y, random_state=SEED)
    mi_base = pd.Series(base_mi, index=num_cols)

    def try_and_keep(name, col_values, test_values):
        nonlocal X_new, Xtest_new, trans_added
        Xcand = Xdf[num_cols].copy(); Xcand[name] = col_values
        mis = mutual_info_regression(Xcand.fillna(Xcand.median()), y, random_state=SEED)
        gain = mis[-1] - mi_base.mean()
        if gain > 0:
            X_new[name] = col_values; Xtest_new[name] = test_values; trans_added.append(name)

    for c in num_cols:
        col = Xdf[c]
        candidates = {
            f"{c}__sqrt": np.sqrt(np.clip(col, 0, None)),
            f"{c}__square": np.square(col),
        }
        if (col > 0).any():
            candidates[f"{c}__log1p"] = np.log1p(np.clip(col, a_min=0, a_max=None))
        kept = 0
        for name, vals in candidates.items():
            if kept >= MAX_BASE_TRANS_PER_FEATURE: break
            if name.endswith("sqrt"):
                test_vals = np.sqrt(np.clip(Xtestdf[c], 0, None))
            elif name.endswith("square"):
                test_vals = np.square(Xtestdf[c])
            else:
                test_vals = np.log1p(np.clip(Xtestdf[c], a_min=0, a_max=None))
            try_and_keep(name, vals, np.nan_to_num(test_vals)); kept += 1
    return X_new, Xtest_new, trans_added

X_base, X_test_base, base_added = add_transforms(X, X_test, train[TARGET].values, num_cols)
display(HTML(f"<div class='card k-mono'>Base transforms added: {len(base_added)}</div>"))



TOP_K_FOR_INTERACTIONS = 8
MAX_INTERACTIONS = 12 if FAST_MODE else 60
INTERACTION_GROUP_SIZE = 12

mi_scores = mutual_info_regression(train[num_cols].fillna(train[num_cols].median()), train[TARGET], random_state=SEED) if num_cols else []
mi_df = pd.DataFrame({"feature": num_cols, "mi": mi_scores}).sort_values("mi", ascending=False)
mi_top = mi_df.head(min(TOP_K_FOR_INTERACTIONS, len(mi_df)))["feature"].tolist()

interaction_defs = []
for a, b in combinations(mi_top, 2):
    interaction_defs.append((f"{a}__x__{b}", "prod", a, b))
    interaction_defs.append((f"{a}__div__{b}", "ratio", a, b))
    interaction_defs.append((f"{a}__minus__{b}", "diff", a, b))
interaction_defs = interaction_defs[:MAX_INTERACTIONS]

def quick_score(Xtrain, ytrain, Xval, yval):
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.07, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=SEED)
    model.fit(Xtrain, ytrain); pred = model.predict(Xval)
    return mean_squared_error(yval, pred, squared=False)

kept_interactions = []
Xtmp_train, Xtmp_val, ytmp_train, ytmp_val = train_test_split(X_base, train[TARGET], test_size=0.25, random_state=SEED)

for i in range(0, len(interaction_defs), INTERACTION_GROUP_SIZE):
    grp = interaction_defs[i:i+INTERACTION_GROUP_SIZE]
    Xadd_tr = Xtmp_train.copy(); Xadd_vl = Xtmp_val.copy()
    made_names = []
    for name, kind, a, b in grp:
        if kind == "prod":
            Xadd_tr[name] = Xtmp_train[a] * Xtmp_train[b]; Xadd_vl[name] = Xtmp_val[a] * Xtmp_val[b]
        elif kind == "ratio":
            Xadd_tr[name] = Xtmp_train[a] / (Xtmp_train[b].replace(0, np.nan))
            Xadd_vl[name] = Xtmp_val[a] / (Xtmp_val[b].replace(0, np.nan))
        else:
            Xadd_tr[name] = Xtmp_train[a] - Xtmp_train[b]; Xadd_vl[name] = Xtmp_val[a] - Xtmp_val[b]
        made_names.append(name)
    base_rmse = quick_score(Xtmp_train, ytmp_train, Xtmp_val, ytmp_val)
    new_rmse  = quick_score(Xadd_tr.fillna(0), ytmp_train, Xadd_vl.fillna(0), ytmp_val)
    if new_rmse + 1e-6 < base_rmse:
        kept_interactions.extend(made_names)

X_fe = X_base.copy(); X_test_fe = X_test_base.copy()
for name in kept_interactions:
    _, kind, a, b = [t for t in interaction_defs if t[0] == name][0]
    if kind == "prod":
        X_fe[name] = train[a] * train[b]; X_test_fe[name] = test[a] * test[b]
    elif kind == "ratio":
        X_fe[name] = train[a] / (train[b].replace(0, np.nan)); X_test_fe[name] = test[a] / (test[b].replace(0, np.nan))
    else:
        X_fe[name] = train[a] - train[b]; X_test_fe[name] = test[a] - test[b]

display(HTML(f"<div class='card k-mono'>Interactions kept: {len(kept_interactions)}</div>"))



RUN_RFECV = False  
selected_features = X_fe.columns.tolist()

if RUN_RFECV:
    sub_idx = np.random.RandomState(SEED).choice(len(X_fe), size=min(20000, len(X_fe)), replace=False)
    est = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31, random_state=SEED)
    rfecv = RFECV(estimator=est, step=0.2, min_features_to_select=25, scoring="neg_root_mean_squared_error", cv=3, n_jobs=-1)
    rfecv.fit(X_fe.iloc[sub_idx].fillna(0), train[TARGET].iloc[sub_idx])
    mask = rfecv.support_; selected_features = X_fe.columns[mask].tolist()
    X_fe = X_fe[selected_features]; X_test_fe = X_test_fe[selected_features]
    display(HTML(f"<div class='card k-mono'>RFECV selected features: {len(selected_features)}</div>"))
else:
    display(HTML("<div class='card'>Skipping RFECV to save time.</div>"))



num_cols_final = [c for c in X_fe.columns if c not in cat_cols]

IMPUTE_WITH = "iterative"  # or "knn"
if IMPUTE_WITH == "knn":
    num_imputer = KNNImputer(n_neighbors=5, weights="distance")
else:
    num_imputer = IterativeImputer(random_state=SEED, estimator=None, max_iter=10, sample_posterior=False)

num_pipeline = Pipeline([("imputer", num_imputer),
                         ("quant", QuantileTransformer(output_distribution="normal", random_state=SEED))])

def fit_target_encoder(X_tr, y_tr, X_va, cols):
    enc = ce.TargetEncoder(cols=cols, smoothing=0.2, min_samples_leaf=10,
                           handle_unknown="value", handle_missing="value")
    enc.fit(X_tr[cols], y_tr)
    return enc.transform(X_tr[cols]).add_prefix("te__"), enc.transform(X_va[cols]).add_prefix("te__"), enc



MODEL_SPECS = {
     "lgb": {
        "model": lgb.LGBMRegressor(
            random_state=SEED,
            n_estimators=1200,
            n_jobs=-1,
            force_col_wise=True,   # <- skip the auto-test message
            verbose=-1             # <- silence training logs
        ),
        "param_dist": {
            "num_leaves": randint(16, 96),
            "learning_rate": loguniform(1e-3, 0.2),
            "min_child_samples": randint(5, 40),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "max_depth": randint(3, 12),
            # Optional stabilizers:
            "min_split_gain": uniform(0.0, 0.2),   # discourage tiny/pointless splits
            "reg_lambda": loguniform(1e-3, 30.0),  # L2
            "reg_alpha": loguniform(1e-4, 10.0),   # L1
            "max_bin": randint(128, 255)          # fewer, denser bins
        },
        "n_iter": 20
    },
    "xgb": {
        "model": xgb.XGBRegressor(random_state=SEED, n_estimators=1500, tree_method="hist", booster="gbtree", n_jobs=-1, reg_alpha=0.0),
        "param_dist": {
            "max_depth": randint(3, 12),
            "learning_rate": loguniform(1e-3, 0.2),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "min_child_weight": randint(1, 20),
            "gamma": uniform(0, 3),
        },
        "n_iter": 20
    },
    "cat": {
        "model": CatBoostRegressor(random_state=SEED, depth=6, learning_rate=0.07, iterations=1200, loss_function="RMSE", verbose=False),
        "param_dist": {
            "depth": randint(4, 10),
            "learning_rate": loguniform(1e-3, 0.2),
            "l2_leaf_reg": loguniform(1e-2, 10),
            "bagging_temperature": uniform(0, 2),
        },
        "n_iter": 15
    },
    "en": {
        "model": ElasticNet(max_iter=2000, random_state=SEED),
        "param_dist": {
            "alpha": loguniform(1e-4, 10),
            "l1_ratio": uniform(0, 1),
        },
        "n_iter": 25
    }
}
SCORING = "neg_root_mean_squared_error"


'''
from sklearn.model_selection import KFold
N_SPLITS_OUTER = 5
N_SPLITS_INNER = 3

kf_outer = KFold(n_splits=N_SPLITS_OUTER, shuffle=True, random_state=SEED)

oof_preds = {k: np.zeros(len(train)) for k in MODEL_SPECS}
fold_scores = {k: [] for k in MODEL_SPECS}
models_per_fold = {k: [] for k in MODEL_SPECS}

X_mat = X_fe.copy(); X_test_mat = X_test_fe.copy()

for fold, (tr_idx, va_idx) in enumerate(kf_outer.split(X_mat, train[TARGET])):
    display(HTML(f"<div class='card'><b>Outer Fold {fold+1}/{N_SPLITS_OUTER}</b></div>"))
    Xtr, Xva = X_mat.iloc[tr_idx].copy(), X_mat.iloc[va_idx].copy()
    ytr, yva = train[TARGET].iloc[tr_idx].values, train[TARGET].iloc[va_idx].values

    # Target encoding per fold (leakage-safe)
    if cat_cols:
        tr_te, va_te, te = fit_target_encoder(Xtr, ytr, Xva, cat_cols)
        Xtr = pd.concat([Xtr.drop(columns=cat_cols, errors="ignore"), tr_te], axis=1)
        Xva = pd.concat([Xva.drop(columns=cat_cols, errors="ignore"), va_te], axis=1)

    # Numeric pipeline
    Xtr_num = pd.DataFrame(num_pipeline.fit_transform(Xtr), index=Xtr.index)
    Xva_num = pd.DataFrame(num_pipeline.transform(Xva), index=Xva.index)

    # Non-numeric remainder (e.g., encoded)
    non_num_cols_fold = [c for c in Xtr.columns if c not in num_cols_final]
    Xtr_other = Xtr[non_num_cols_fold].reset_index(drop=True)
    Xva_other = Xva[non_num_cols_fold].reset_index(drop=True)

    Xtr_final = pd.concat([Xtr_num.reset_index(drop=True), Xtr_other], axis=1)
    Xva_final = pd.concat([Xva_num.reset_index(drop=True), Xva_other], axis=1)

    for key, spec in MODEL_SPECS.items():
        rs = RandomizedSearchCV(
            spec["model"], param_distributions=spec["param_dist"], n_iter=spec["n_iter"],
            scoring=SCORING, cv=N_SPLITS_INNER, random_state=SEED, n_jobs=-1, verbose=0
        )
        rs.fit(Xtr_final, ytr)
        best_model = rs.best_estimator_
        models_per_fold[key].append(best_model)

        best_model.fit(Xtr_final, ytr)
        val_pred = best_model.predict(Xva_final)
        oof_preds[key][va_idx] = val_pred
        rmse = mean_squared_error(yva, val_pred, squared=False)
        fold_scores[key].append(rmse)
        display(HTML(f"<div class='card k-mono'>Model <b>{key}</b> RMSE: {rmse:.6f}</div>"))
'''


#Use only one model instead of four models, to make it simple and fast

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer


IMPUTE_WITH = "median" 
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("quant", QuantileTransformer(output_distribution="normal", random_state=SEED, n_quantiles=200)),  # fewer quantiles
])

# Convert FE matrices to float32 to keep trees lean
X_mat = X_fe.copy().astype(np.float32)
X_test_mat = X_test_fe.copy().astype(np.float32)

N_SPLITS = 3  # 3 folds is enough to validate quickly, change later once get proper cv
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_pred = np.zeros(len(train), dtype=np.float32)
test_preds = []

lgb_params = dict(
    n_estimators=5000,            # large cap + early stopping
    learning_rate=0.05,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    force_col_wise=True,
    verbose=-1,
)

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_mat, train[TARGET])):
    display(HTML(f"<div class='card'><b>FAST MODE â€” Fold {fold+1}/{N_SPLITS}</b></div>"))

    Xtr, Xva = X_mat.iloc[tr_idx].copy(), X_mat.iloc[va_idx].copy()
    ytr, yva = train[TARGET].iloc[tr_idx].values, train[TARGET].iloc[va_idx].values

    # Target encoding (leakage-safe)
    if cat_cols:
        tr_te, va_te, te = fit_target_encoder(Xtr, ytr, Xva, cat_cols)
        Xtr = pd.concat([Xtr.drop(columns=cat_cols, errors="ignore"), tr_te], axis=1)
        Xva = pd.concat([Xva.drop(columns=cat_cols, errors="ignore"), va_te], axis=1)

    # Numeric pipeline
    Xtr_num = pd.DataFrame(num_pipeline.fit_transform(Xtr), index=Xtr.index)
    Xva_num = pd.DataFrame(num_pipeline.transform(Xva), index=Xva.index)

    # Remainder (encoded cat)
    non_num_cols_fold = [c for c in Xtr.columns if c not in num_cols_final]
    Xtr_other = Xtr[non_num_cols_fold].reset_index(drop=True)
    Xva_other = Xva[non_num_cols_fold].reset_index(drop=True)

    Xtr_final = pd.concat([Xtr_num.reset_index(drop=True), Xtr_other], axis=1).astype(np.float32)
    Xva_final = pd.concat([Xva_num.reset_index(drop=True), Xva_other], axis=1).astype(np.float32)

    # Train LightGBM with early stopping
    lgbm = lgb.LGBMRegressor(**lgb_params)
    lgbm.fit(
        Xtr_final, ytr,
        eval_set=[(Xva_final, yva)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(period=0)]
    )

    # OOF + fold score
    val_pred = lgbm.predict(Xva_final, num_iteration=lgbm.best_iteration_)
    oof_pred[va_idx] = val_pred
    rmse = mean_squared_error(yva, val_pred, squared=False)
    display(HTML(f"<div class='card k-mono'>Fold RMSE: <b>{rmse:.5f}</b> | Best iters: {lgbm.best_iteration_}</div>"))

    # Prep test features with the same fold encoder/pipeline
    if cat_cols:
        te_full = te  # reuse fold encoder for test transform to avoid leakage
        X_test_fold = X_test_mat.drop(columns=cat_cols, errors="ignore")
        X_test_fold = pd.concat([X_test_fold, te_full.transform(X_test_mat[cat_cols]).add_prefix("te__")], axis=1)
    else:
        X_test_fold = X_test_mat.copy()

    X_test_fold_num  = pd.DataFrame(num_pipeline.transform(X_test_fold))
    X_test_fold_other = X_test_fold[[c for c in X_test_fold.columns if c not in num_cols_final]].reset_index(drop=True)
    X_test_final = pd.concat([X_test_fold_num.reset_index(drop=True), X_test_fold_other], axis=1).astype(np.float32)

    test_pred = lgbm.predict(X_test_final, num_iteration=lgbm.best_iteration_)
    test_preds.append(test_pred.astype(np.float32))

# Overall OOF
oof_rmse = mean_squared_error(train[TARGET], oof_pred, squared=False)
display(HTML(f"<div class='card'><b>FAST MODE â€” OOF RMSE:</b> {oof_rmse:.6f}</div>"))

# Test prediction (mean of folds)
test_pred_mean = np.mean(test_preds, axis=0)

# Submission
sub = sample_sub.copy()
if "id" in sub.columns:
    sub["id"] = test[ID_COL]
pred_col = [c for c in sub.columns if c != "id"]
pred_name = pred_col[0] if len(pred_col)==1 else (TARGET if TARGET in sub.columns else sub.columns[-1])
sub[pred_name] = test_pred_mean
sub_path = "/kaggle/working/submission.csv"
sub.to_csv(sub_path, index=False)
display(HTML(f"<div class='card k-mono'>âœ… FAST submission saved â†’ {sub_path}</div>"))



oof_df = pd.DataFrame({k: v for k, v in oof_preds.items()})
meta = RidgeCV(alphas=np.logspace(-4, 2, 20), cv=5)
meta.fit(oof_df, train[TARGET])
stack_oof = meta.predict(oof_df)
stack_rmse = mean_squared_error(train[TARGET], stack_oof, squared=False)
display(HTML(f"<div class='card'><b>Stacking Ridge OOF RMSE:</b> {stack_rmse:.6f}</div>"))



# Refit TE + numeric pipeline on full data
if cat_cols:
    enc_full = ce.TargetEncoder(cols=cat_cols, smoothing=0.2, min_samples_leaf=10, handle_unknown="value", handle_missing="value")
    enc_full.fit(X_mat[cat_cols], train[TARGET])
    X_full = pd.concat([X_mat.drop(columns=cat_cols, errors="ignore"), enc_full.transform(X_mat[cat_cols]).add_prefix("te__")], axis=1)
    X_test_full = pd.concat([X_test_mat.drop(columns=cat_cols, errors="ignore"), enc_full.transform(X_test_mat[cat_cols]).add_prefix("te__")], axis=1)
else:
    X_full = X_mat.copy(); X_test_full = X_test_mat.copy()

X_full_num  = pd.DataFrame(num_pipeline.fit_transform(X_full))
X_test_num  = pd.DataFrame(num_pipeline.transform(X_test_full))
X_full_other = X_full[[c for c in X_full.columns if c not in num_cols_final]].reset_index(drop=True)
X_test_other = X_test_full[[c for c in X_test_full.columns if c not in num_cols_final]].reset_index(drop=True)
X_full_final = pd.concat([X_full_num.reset_index(drop=True), X_full_other], axis=1)
X_test_final = pd.concat([X_test_num.reset_index(drop=True), X_test_other], axis=1)

# Refit each model on full data, using the last fold's tuned estimator as proxy
final_preds = {}; weights = {}
for key in MODEL_SPECS.keys():
    mean_rmse = float(np.mean(fold_scores[key]))
    display(HTML(f"<div class='card k-mono'>Refitting <b>{key}</b> â€” mean outer RMSE â‰ˆ {mean_rmse:.6f}</div>"))
    final_model = models_per_fold[key][-1]
    final_model.fit(X_full_final, train[TARGET].values)
    final_preds[key] = final_model.predict(X_test_final)
    weights[key] = 1.0 / (mean_rmse + 1e-9)

# Blend + Stack
blend = np.zeros(len(test)); w_sum = sum(weights.values())
for key in MODEL_SPECS.keys():
    blend += (weights[key]/w_sum) * final_preds[key]
stack_test = meta.predict(pd.DataFrame({k: final_preds[k] for k in MODEL_SPECS.keys()}))
final = 0.5 * blend + 0.5 * stack_test

sub = sample_sub.copy()
if "id" in sub.columns:
    sub["id"] = test[ID_COL]

pred_col = [c for c in sub.columns if c != "id"]
pred_name = pred_col[0] if len(pred_col)==1 else (TARGET if TARGET in sub.columns else sub.columns[-1])
sub[pred_name] = final
sub_path = "/kaggle/working/submission.csv"
sub.to_csv(sub_path, index=False)
display(HTML(f"<div class='card k-mono'>âœ… Submission saved â†’ {sub_path}</div>"))



try:
    ref_model = models_per_fold["lgb"][-1]
    importances = getattr(ref_model, "feature_importances_", None)
    if importances is not None:
        imp_df = pd.DataFrame({"feature": X_full_final.columns, "importance": importances}).sort_values("importance", ascending=False).head(30)
        fig = px.bar(imp_df, x="importance", y="feature", orientation="h", title="Top 30 Feature Importances (LGBM)")
        fig.update_layout(template="plotly_dark", height=600); fig.show()
except Exception:
    pass

score_rows = [{"model": k, "outer_rmse_mean": float(np.mean(v)), "outer_rmse_std": float(np.std(v))} for k, v in fold_scores.items()]
score_df = pd.DataFrame(score_rows).sort_values("outer_rmse_mean")
fig = px.bar(score_df, x="model", y="outer_rmse_mean", error_y="outer_rmse_std", title="Outerâ€‘CV RMSE by Model (lower is better)")
fig.update_layout(template="plotly_dark", height=420); fig.show()


