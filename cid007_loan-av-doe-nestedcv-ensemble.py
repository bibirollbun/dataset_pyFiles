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


print("â–¶ Setup starting...")

import os, sys, warnings, math, gc, json, time, itertools, textwrap, typing as T
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Polars for fast EDA where helpful
try:
    import polars as pl
    POLARS_OK = True
except Exception as e:
    POLARS_OK = False
    print("Polars not available, proceeding with pandas:", e)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Plotly/Bokeh
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = 'iframe'
    PLOTLY_OK = True
except Exception as e:
    PLOTLY_OK = False
    print("Plotly not available:", e)

try:
    from bokeh.io import output_notebook, show
    from bokeh.plotting import figure
    from bokeh.models import HoverTool, ColumnDataSource, FactorRange
    output_notebook()
    BOKEH_OK = True
except Exception as e:
    BOKEH_OK = False
    print("Bokeh not available:", e)

# Modeling stack
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin, clone

# Tree libraries
try:
    import lightgbm as lgb
    LGB_OK = True
except Exception as e:
    LGB_OK = False
    print("LightGBM not available:", e)

try:
    import xgboost as xgb
    XGB_OK = True
except Exception as e:
    XGB_OK = False
    print("XGBoost not available:", e)

try:
    from catboost import CatBoostClassifier, Pool
    CAT_OK = True
except Exception as e:
    CAT_OK = False
    print("CatBoost not available:", e)

# Encoders
try:
    import category_encoders as ce
    CE_OK = True
except Exception as e:
    CE_OK = False
    print("category_encoders not available; will fallback to pandas/get_dummies:", e)

# Optuna (optional)
try:
    import optuna
    OPTUNA_OK = True
except Exception as e:
    OPTUNA_OK = False
    print("Optuna not available, inner HPO will use RandomizedSearchCV:", e)

# W&B (optional)
try:
    import wandb
    WANDB_OK = True
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    #wandb.login(key=os.environ["wandb_key"])

except Exception as e:
    WANDB_OK = False
    print("Weights & Biases not available:", e)

from kaggle_secrets import UserSecretsClient
# Access secrets stored in Kaggle (Add-ons > Secrets)
user_secrets = UserSecretsClient()

# Try retrieving W&B key from either name (depending on which one you actually used)
wandb_api_key = None
try:
    wandb_api_key = user_secrets.get_secret("wandb_api_key")
except:
    try:
        wandb_api_key = user_secrets.get_secret("wandb_key")
    except:
        print("âš ï¸� Neither 'wandb_api_key' nor 'wandb_key' was found in Kaggle secrets.")

# Log into Weights & Biases if key was found
if wandb_api_key:
    wandb.login(key=wandb_api_key)
else:
    print("â�Œ W&B API key missing. Please check Kaggle Add-ons > Secrets.")

SEED = 42
np.random.seed(SEED)

print("âœ“ Setup complete.")
print("Read/Interpretation: If you saw 'not available' messages above, the notebook will still run with fallbacks.")

from IPython.display import HTML, display

output_notebook()

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



from pathlib import Path

COMP_PATH = Path('/kaggle/input/playground-series-s5e11')
ORIG_PATH = Path('/kaggle/input/loan-prediction-dataset-2025')  # nabiha zahid dataset

train = pd.read_csv(COMP_PATH/'train.csv')
test  = pd.read_csv(COMP_PATH/'test.csv')
sub   = pd.read_csv(COMP_PATH/'sample_submission.csv')

print("Competition shapes:", train.shape, test.shape, sub.shape)
print("Train columns:", list(train.columns)[:10], "...")
print(train.head(2))

# Original dataset (schema may differ; we'll standardize later)
orig_candidates = []
for f in ORIG_PATH.glob('*.csv'):
    try:
        df = pd.read_csv(f)
        orig_candidates.append((f.name, df.shape, list(df.columns)[:10]))
    except Exception as e:
        orig_candidates.append((f.name, "read_error", str(e)))

print("Original dataset files & peeks:", orig_candidates)
print("Read/Interpretation: If original schema differs, we will harmonize minimal common features for adversarial validation.")



def section(title):
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")
section("Helpers ready")
print("Read/Interpretation: We'll call section() before key blocks.")



section("Structure & Types")

def basic_overview(df, name="dataframe"):
    print(f"â–¶ {name}: shape={df.shape}")
    print(df.dtypes.value_counts())
    miss_rate = df.isna().mean().sort_values(ascending=False).head(20)
    print("\nTop missing columns:\n", miss_rate)
    display(df.describe(include='all').T.head(20))

basic_overview(train, "train")
basic_overview(test,  "test")

# Target distribution (assume column name 'Class' or 'loan_default' or check last col from sample_submission)
TARGET = sub.columns[-1]
ID_COL  = [c for c in train.columns if c.lower() in ("id","loan_id","row_id","customer_id","record_id")]
ID_COL  = ID_COL[0] if len(ID_COL)>0 else "id"
if ID_COL not in train.columns:
    # try default
    ID_COL = train.columns[0]

print("Heuristic ID column:", ID_COL)
print("Target:", TARGET)

import matplotlib.ticker as mtick
fig, ax = plt.subplots(figsize=(5,3))
train[TARGET].value_counts(normalize=True).sort_index().plot(kind='bar', ax=ax)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title("Target distribution")
ax.set_xlabel("Class")
ax.set_ylabel("Share")
plt.show()

print("Read/Interpretation: Check class balance; heavy skew suggests stratified CV and calibrated thresholds.")



section("Numeric vs Categorical detection")

# Heuristic split
num_cols = train.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = [c for c in train.columns if c not in num_cols + [TARGET]]
num_cols = [c for c in num_cols if c not in [TARGET]]

print("Numerical:", len(num_cols))
print("Categorical:", len(cat_cols))

# Seaborn pairplot sample (downsample for speed)
sample_idx = np.random.choice(len(train), size=min(1500, len(train)), replace=False)
sns.pairplot(
    train.loc[sample_idx, [*num_cols[:4], TARGET]],
    hue=TARGET, corner=True, diag_kind="kde"
)
plt.suptitle("Pairwise (subset) â€” quick separability check", y=1.02)
plt.show()

print("Read/Interpretation: Look for separability cues and nonlinearity; informs model choice (trees vs linear).")



section("Plotly & Bokeh univariate views")

if PLOTLY_OK:
    # Numeric histograms
    for col in num_cols[:8]:
        fig = px.histogram(train, x=col, color=TARGET, barmode="overlay", nbins=50, marginal="box", opacity=0.6)
        fig.update_layout(title=f"Distribution: {col} by {TARGET}")
        fig.show()

if BOKEH_OK and len(num_cols) >= 1:
    col = num_cols[0]
    df = train[[col, TARGET]].copy()
    cds = ColumnDataSource(df)
    p = figure(title=f"Bokeh: {col} vs {TARGET}", width=700, height=350, tools="pan,wheel_zoom,box_zoom,reset,hover,save")
    p.scatter(x=col, y=TARGET, source=cds, size=4, alpha=0.4)
    hover = p.select_one(HoverTool)
    hover.tooltips = [(col, f"@{col}"), (TARGET, f"@{TARGET}")]
    show(p)

print("Read/Interpretation: Heavy overlap suggests nonlinear models or interaction terms. Long tails hint at log/sqrt transforms.")



section("Missingness heatmap + leak sniff")

# Missingness heatmap
plt.figure(figsize=(10,4))
sns.heatmap(train[num_cols].isna(), cbar=False)
plt.title("Missingness pattern (numerics)")
plt.show()

# Simple leak sniff: any column perfectly predicts target?
from sklearn.tree import DecisionTreeClassifier
leaks = []
for col in num_cols + cat_cols:
    if col == TARGET: continue
    try:
        tmp = train[[col, TARGET]].dropna()
        if tmp[col].nunique() < 2: 
            continue
        acc = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(tmp[[col]], tmp[TARGET]).score(tmp[[col]], tmp[TARGET])
        if acc >= 0.995:
            leaks.append((col, acc, tmp.shape[0]))
    except Exception:
        pass

print("Potential leaks:", leaks if leaks else "None detected at max_depth=1")
print("Read/Interpretation: If any 'leak' column exists, consider removing or capping it.")



section("Train vs Test drift (KS and PSI)")

def population_stability_index(expected, actual, buckets=10):
    q = np.linspace(0, 1, buckets+1)
    try:
        cuts = np.unique(np.quantile(expected.dropna(), q))
    except Exception:
        return np.nan
    e_counts = pd.cut(expected, cuts, include_lowest=True).value_counts(normalize=True, sort=False)
    a_counts = pd.cut(actual,   cuts, include_lowest=True).value_counts(normalize=True, sort=False)
    psi = np.sum((a_counts - e_counts) * np.log((a_counts + 1e-8) / (e_counts + 1e-8)))
    return psi

drift_tbl = []
from scipy.stats import ks_2samp
for col in num_cols[:]:
    ks = ks_2samp(train[col].dropna(), test[col].dropna()).statistic
    psi = population_stability_index(train[col], test[col])
    drift_tbl.append((col, ks, psi))

drift_df = pd.DataFrame(drift_tbl, columns=["feature","KS","PSI"]).sort_values(["PSI","KS"], ascending=False)
display(drift_df.head(20))

if PLOTLY_OK:
    fig = px.scatter(drift_df, x="KS", y="PSI", text="feature", title="Train vs Test Drift â€” KS vs PSI")
    fig.update_traces(textposition="top center")
    fig.show()

print("Read/Interpretation: PSI>0.2 often suggests distribution shift. Drifted features are candidates for robust encodings or regularization.")



section("Adversarial Validation")

# Load original dataset 
orig_path = Path('/kaggle/input/loan-prediction-dataset-2025')
orig_files = sorted(orig_path.glob("*.csv"), key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
orig = None
for f in orig_files:
    try:
        tmp = pd.read_csv(f)
        if tmp.shape[0] > 100 and tmp.shape[1] > 5:
            orig = tmp.copy()
            print("Using original file:", f.name, "shape:", orig.shape)
            break
    except Exception as e:
        print("Skip", f.name, ":", e)

if orig is None:
    print("Could not load original dataset; skipping AV.")
else:
    # Standardize column names (lowercase + underscores)
    train_anon = train.drop(columns=[TARGET], errors="ignore").copy()
    train_anon.columns = [c.lower().strip() for c in train_anon.columns]
    orig_std = orig.copy()
    orig_std.columns = [c.lower().strip() for c in orig_std.columns]

    # Keep numeric overlap
    common = list(set(train_anon.columns).intersection(set(orig_std.columns)))
    # drop obvious IDs
    common = [c for c in common if c not in ("id","loan_id","row_id","customer_id","record_id")]
    # keep only numeric-like for quick AV
    common = [c for c in common if (pd.api.types.is_numeric_dtype(train_anon[c]) if c in train_anon.columns else False)
                                and (pd.api.types.is_numeric_dtype(orig_std[c]) if c in orig_std.columns else False)]

    X_tr = train_anon[common].copy()
    X_or = orig_std[common].copy()
    X_tr["__src__"] = 0
    X_or["__src__"] = 1
    av = pd.concat([X_tr, X_or], axis=0, ignore_index=True)
    av = av.dropna(axis=1, how="all")
    y_av = av["__src__"].values
    X_av = av.drop(columns=["__src__"])

    print("Common numeric features for AV:", len(X_av.columns))

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    if LGB_OK and X_av.shape[1] > 0:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = np.zeros(len(av))
        for tr_idx, va_idx in skf.split(X_av, y_av):
            lgbm = lgb.LGBMClassifier(
                n_estimators=1000, learning_rate=0.05, max_depth=-1,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=SEED
            )
            lgbm.fit(X_av.iloc[tr_idx], y_av[tr_idx],
                     eval_set=[(X_av.iloc[va_idx], y_av[va_idx])],
                     eval_metric="auc", callbacks=[lgb.early_stopping(50)])
            oof[va_idx] = lgbm.predict_proba(X_av.iloc[va_idx])[:,1]
        auc = roc_auc_score(y_av, oof)
        print(f"Adversarial AUC = {auc:.4f} (0.5=no shift, 1.0=separable)")
    else:
        print("LightGBM unavailable or no AV features; skipping AV AUC.")

print("Read/Interpretation: If AUC is high (>0.8), distributions differ. Consider robust encoders, remove drift-heavy features, or use CV split strategies aligned with shift.")



section("Feature Engineering")

SAFE_EPS = 1e-6

def make_base_transforms(df: pd.DataFrame, cols: T.List[str]) -> pd.DataFrame:
    out = {}
    for c in cols:
        x = df[c].astype(float)
        out[f"{c}__log"]   = np.log1p(np.clip(x, a_min=0, a_max=None))
        out[f"{c}__sqrt"]  = np.sqrt(np.clip(x, a_min=0, a_max=None))
        out[f"{c}__sq"]    = x**2
    return pd.DataFrame(out, index=df.index)

def make_pairwise(df: pd.DataFrame, cols: T.List[str], limit: int = 80) -> pd.DataFrame:
    out = {}
    pairs = list(itertools.combinations(cols, 2))[:limit]
    for a,b in pairs:
        xa, xb = df[a].astype(float), df[b].astype(float)
        out[f"{a}__plus__{b}"]  = xa + xb
        out[f"{a}__minus__{b}"] = xa - xb
        out[f"{a}__mul__{b}"]   = xa * xb
        out[f"{a}__div__{b}"]   = xa / (xb.abs() + SAFE_EPS)
    return pd.DataFrame(out, index=df.index)

def frequency_encode(train_series, test_series):
    freq = train_series.value_counts(dropna=False) / len(train_series)
    return train_series.map(freq), test_series.map(freq)

def target_encode(train_ser, y, test_ser, min_samples_leaf=20, smoothing=10):
    # Simple additive smoothing target encoding
    prior = y.mean()
    stats = train_ser.to_frame("cat").groupby("cat")[y.name].agg(["count","mean"])
    smoothing_val = 1 / (1 + np.exp(-(stats["count"] - min_samples_leaf) / smoothing))
    te = prior * (1 - smoothing_val) + stats["mean"] * smoothing_val
    return train_ser.map(te), test_ser.map(te)

print("Feature helpers ready.")

print("Read/Interpretation: We will create blocks incrementally and only keep helpful ones (by MI / model FI).")



section("Imputation + Drift check")

work = train.copy()
test_work = test.copy()

num_miss = [c for c in num_cols if work[c].isna().any()]
print(f"Columns with missing (numeric): {len(num_miss)}")

orig_corr = {}
for c in num_miss:
    orig_corr[c] = work[c].corr(work[c].dropna(), method="spearman")

# Pipeline: KNN then Iterative (on numerics only)
imputer_knn = KNNImputer(n_neighbors=5, weights="distance")
imputer_iter = IterativeImputer(random_state=SEED, max_iter=10, sample_posterior=False)

train_num = work[num_cols].copy()
test_num  = test_work[num_cols].copy()

train_num_knn = pd.DataFrame(imputer_knn.fit_transform(train_num), columns=num_cols, index=train.index)
test_num_knn  = pd.DataFrame(imputer_knn.transform(test_num),     columns=num_cols, index=test.index)

train_num_imp = pd.DataFrame(imputer_iter.fit_transform(train_num_knn), columns=num_cols, index=train.index)
test_num_imp  = pd.DataFrame(imputer_iter.transform(test_num_knn),     columns=num_cols, index=test.index)

# Put back
for c in num_cols:
    work[c] = train_num_imp[c]
    test_work[c] = test_num_imp[c]

# Correlation sanity (Spearman between original and imputed on overlap index)
corr_report = []
for c in num_miss:
    mask = train[c].notna()
    if mask.sum() > 5:
        corr_val = pd.Series(train.loc[mask, c]).corr(pd.Series(work.loc[mask, c]), method="spearman")
        corr_report.append((c, corr_val))
corr_df = pd.DataFrame(corr_report, columns=["feature","spearman_orig_vs_imputed"]).sort_values("spearman_orig_vs_imputed")
display(corr_df.head(15))

print("Read/Interpretation: Very low correlations suggest imputation altered rank structure; consider simpler methods for such features.")



section("Categorical encodings (fixed)")

X = work.drop(columns=[TARGET]).copy()
y = work[TARGET].astype(int).reset_index(drop=True)

# Detect categoricals
cat_cols = [c for c in X.columns if X[c].dtype == "object" or X[c].dtype.name == "category"]
print("Categoricals found:", len(cat_cols), "->", cat_cols[:15], "..." if len(cat_cols)>15 else "")

X = X.reset_index(drop=True)
test_work = test_work.reset_index(drop=True)

def frequency_encode(train_series: pd.Series, test_series: pd.Series):
    s_tr = train_series.fillna("__NA__")
    s_te = test_series.fillna("__NA__")
    freq = s_tr.value_counts(dropna=False) / len(s_tr)
    return s_tr.map(freq).astype(float), s_te.map(freq).fillna(0.0).astype(float)

def target_encode(train_ser: pd.Series,
                  y_ser: pd.Series,
                  test_ser: pd.Series,
                  min_samples_leaf: int = 20,
                  smoothing: float = 10.0):
    # Align and build a 2-col frame explicitly
    df = pd.DataFrame({
        "cat": train_ser.fillna("__NA__").values,
        "target": y_ser.values
    })
    stats = df.groupby("cat")["target"].agg(["count", "mean"])
    prior = df["target"].mean()
    # Smoothing towards prior
    smooth = 1.0 / (1.0 + np.exp(-(stats["count"] - min_samples_leaf) / smoothing))
    te_map = prior * (1.0 - smooth) + stats["mean"] * smooth
    # Map both train (OOF done outside) and test;
    tr_map = train_ser.fillna("__NA__").map(te_map).fillna(prior).astype(float)
    te_mapped = test_ser.fillna("__NA__").map(te_map).fillna(prior).astype(float)
    return tr_map, te_mapped, prior

X_fe = X.copy()
test_fe = test_work.copy()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for c in cat_cols:
    # Frequency
    tr_fe, te_fe = frequency_encode(X[c], test_fe[c])
    X_fe[f"{c}__freq"] = tr_fe
    test_fe[f"{c}__freq"] = te_fe

    # Target encoding with OOF to avoid leakage
    oof_te = np.zeros(len(X_fe), dtype=float)
    for tr_idx, va_idx in skf.split(X_fe, y):
        tr_ser = X.iloc[tr_idx][c]
        va_ser = X.iloc[va_idx][c]
        tr_y   = y.iloc[tr_idx]
        tr_map, va_map, _ = target_encode(tr_ser, tr_y, va_ser)
        oof_te[va_idx] = va_map.values
    # Fit on full train to map test
    tr_full, te_full, prior = target_encode(X[c], y, test_fe[c])
    X_fe[f"{c}__te"] = oof_te
    test_fe[f"{c}__te"] = te_full.values

# Drop raw categoricals (we keep encodings)
X_fe = X_fe.drop(columns=cat_cols)
test_fe = test_fe.drop(columns=cat_cols)

print("Encoded shape:", X_fe.shape, test_fe.shape)
print("Read/Interpretation: Added *_freq and *_te per categorical; unseen categories in test fall back to the train prior.")



section("Transforms & Interactions with scoring")

base_block = make_base_transforms(X_fe, [c for c in X_fe.columns if c != ID_COL])
pair_block = make_pairwise(X_fe, [c for c in X_fe.columns if c != ID_COL], limit=80)

# Score via MI
def mi_rank(Xblock, y, topk=150):
    Xb = Xblock.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mi = mutual_info_classif(Xb, y, random_state=SEED, discrete_features=False)
    df = pd.DataFrame({"feat": Xb.columns, "mi": mi}).sort_values("mi", ascending=False).head(topk)
    keep_cols = df["feat"].tolist()
    return Xb[keep_cols], keep_cols, df

Xb_base, base_keep, base_df = mi_rank(base_block, y, topk=100)
Xb_pair, pair_keep, pair_df = mi_rank(pair_block, y, topk=120)

display(base_df.head(10))
display(pair_df.head(10))

# Build final design matrix
X_final = pd.concat([X_fe, Xb_base, Xb_pair], axis=1)
test_final = pd.concat([test_fe, 
                        base_block[base_keep].reindex(test_fe.index).fillna(0.0),
                        pair_block[pair_keep].reindex(test_fe.index).fillna(0.0)], axis=1)

print("Final shapes:", X_final.shape, test_final.shape)

print("Read/Interpretation: We retained topâ€‘MI transform/interaction features to limit overfitting and runtime.")



section("Nested CV + HPO")

FEATURES = [c for c in X_final.columns if c not in (ID_COL,)]
X_mat = X_final[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
T_mat = test_final[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).values

outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def run_lgb_train(X_tr, y_tr, X_va, y_va):
    if not LGB_OK:
        return None
    params = dict(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=SEED
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(200)])
    return model

def run_xgb_train(X_tr, y_tr, X_va, y_va):
    if not XGB_OK:
        return None
    model = xgb.XGBClassifier(
        n_estimators=3000, learning_rate=0.03,
        max_depth=6, subsample=0.85, colsample_bytree=0.85,
        eval_metric="auc", tree_method="hist", random_state=SEED
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model

def run_cat_train(X_tr, y_tr, X_va, y_va):
    if not CAT_OK:
        return None
    model = CatBoostClassifier(
        iterations=3000, learning_rate=0.03, depth=6,
        loss_function="Logloss", eval_metric="AUC",
        random_seed=SEED, verbose=False
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    return model

oof = np.zeros(len(X_mat))
preds = []
fold_scores = []

if WANDB_OK:
    wandb.init(project="kaggle-s5e11-loan", config={"seed": SEED, "n_splits": outer.n_splits})

for fold, (tr_idx, va_idx) in enumerate(outer.split(X_mat, y)):
    Xtr, Xva = X_mat[tr_idx], X_mat[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    # Quick model selection: train 2-3 models, pick best by AUC on this fold
    candidates = []
    for trainer in (run_lgb_train, run_xgb_train, run_cat_train):
        m = trainer(Xtr, ytr, Xva, yva)
        if m is not None:
            proba = m.predict_proba(Xva)[:,1]
            auc = roc_auc_score(yva, proba)
            candidates.append((auc, m, proba))
    candidates.sort(key=lambda x: -x[0])
    best_auc, best_model, best_proba = candidates[0]
    oof[va_idx] = best_proba
    fold_scores.append(best_auc)

    # Save test predictions of each candidate for ensembling
    fold_pred = {}
    for name, m in [("lgb", candidates[0][1])] + ([("xgb", candidates[1][1])] if len(candidates)>1 else []) + ([("cat", candidates[2][1])] if len(candidates)>2 else []):
        if m is not None:
            fold_pred[name] = m.predict_proba(T_mat)[:,1]
    preds.append(fold_pred)

    if WANDB_OK:
        wandb.log({f"fold_{fold}_auc": best_auc, "fold": fold})

cv_auc = roc_auc_score(y, oof)
print(f"OOF AUC: {cv_auc:.5f}")
if WANDB_OK:
    wandb.log({"oof_auc": cv_auc})
    wandb.finish()

print("Read/Interpretation: OOF AUC reflects generalization; keep an eye on fold variance printed above if any.")



section("OOF Confusion Matrix + Misclassified Table (refined)")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 0/1 predictions from OOF probabilities
thr = 0.5  # tweak later if you calibrate thresholds
y_true = y
y_prob = oof
y_pred = (y_prob >= thr).astype(int)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0,1])
fig, ax = plt.subplots(figsize=(4,4))
ConfusionMatrixDisplay(cm, display_labels=[0,1]).plot(cmap="Blues", values_format="d", ax=ax)
ax.set_title(f"Confusion Matrix (OOF, thr={thr:.2f})")
plt.show()

# Misclassified index sets
mis_idx = np.where(y_true != y_pred)[0]
fp_idx  = np.where((y_true == 0) & (y_pred == 1))[0]
fn_idx  = np.where((y_true == 1) & (y_pred == 0))[0]

print(f"Misclassified: {len(mis_idx)} | FP: {len(fp_idx)} | FN: {len(fn_idx)} | Total: {len(y_true)}")

# Build a dataframe of misclassified rows with predictions/probs
mis_df = X_final.iloc[mis_idx].copy()
mis_df.insert(0, "true_label", y_true[mis_idx])
mis_df.insert(1, "pred_label", y_pred[mis_idx])
mis_df.insert(2, "pred_prob",  y_prob[mis_idx])

# Save full table to CSV for deeper inspection
mis_csv_path = "misclassified_oof_rows.csv"
mis_df.to_csv(mis_csv_path, index=False)
print(f"Saved full misclassified table -> {mis_csv_path}")

# Show a concise preview (full table also printed scrollably)
display(mis_df.head(15))
from IPython.display import display_html
display_html(mis_df.to_html(max_rows=100))  # scrollable



# - Determine important features
# - For top-K features, plot class-wise KDE and overlay misclassified samples
#
# Read/Interpretation
# - Red markers (y = -0.02) show where fails happen relative to class density.

section("Top-Feature Diagnostics with Misclassification Markers")

# Try model-derived importances if any candidate model 'best_model' exists
# (If not, fallback to a correlation proxy)
try:
    # If you kept a 'best_model' from the last outer fold; otherwise this will except
    importances = pd.Series(best_model.feature_importances_, index=FEATURES)
except Exception:
    # Correlation proxy (fast, safe)
    corr = np.corrcoef(X_mat.T, y_true)[-1, :-1] if X_mat.shape[1] == len(FEATURES) else np.corrcoef(X_final[FEATURES].values.T, y_true)[-1, :-1]
    importances = pd.Series(np.abs(corr), index=FEATURES)

top_feats = importances.sort_values(ascending=False).head(6).index.tolist()
print("Top important features:", top_feats)

for feat in top_feats:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.kdeplot(x=X_final[feat], hue=pd.Series(y_true, name="True Label"),
                fill=True, common_norm=False, alpha=0.35, ax=ax)
    # overlay misclassified markers slightly below x-axis
    ax.scatter(X_final.loc[mis_idx, feat],
               np.full(len(mis_idx), -0.02),
               s=12, alpha=0.7, color="red", label="Misclassified")
    ax.set_ylim(bottom=-0.05)
    ax.set_title(f"{feat} â€” class density + misclassification markers")
    ax.legend()
    plt.show()



# - Seaborn pairplot on a subset with misclassified overlay as hue
# - Keep it light with sampling to avoid huge render times
# Read/Interpretation
# - Red cluster islands = failure pockets; overlapping = ambiguous regions.

section("Pairplot with Misclassified Overlay")

pp_feats = top_feats[:4]  # pick 4 for readability
plot_df = X_final[pp_feats].copy()
plot_df["is_mis"] = 0
plot_df.loc[mis_idx, "is_mis"] = 1
plot_df["is_mis"] = plot_df["is_mis"].map({0:"Correct", 1:"Misclassified"})

# sample for speed
N = min(2000, len(plot_df))
samp_idx = np.random.choice(len(plot_df), N, replace=False)
sns.pairplot(plot_df.iloc[samp_idx], hue="is_mis", corner=True, diag_kind="kde",
             plot_kws=dict(alpha=0.6, s=20))
plt.suptitle("Pairplotâ€”Misclassified vs Correct (sampled)", y=1.02)
plt.show()



# - Train a SHAP-friendly model on full data quickly
# - Explain a sample of misclassified rows
#
# Read/Interpretation
# - Which features *push* probability the wrong way? Look at top drivers.

section("SHAP Analysis Focused on Misclassified Samples")

# Select a model type for SHAP
model_for_shap = None
model_name = None

# Simple train/valid split for early_stopping if trees
from sklearn.model_selection import train_test_split
Xtr, Xva, ytr, yva = train_test_split(X_final[FEATURES].values, y_true, test_size=0.15, random_state=SEED, stratify=y_true)

if LGB_OK:
    model_for_shap = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03,
                                        num_leaves=63, subsample=0.85, colsample_bytree=0.85,
                                        random_state=SEED)
    model_for_shap.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
                       callbacks=[lgb.early_stopping(200)])
    model_name = "lightgbm"
elif XGB_OK:
    model_for_shap = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.03,
                                       max_depth=6, subsample=0.85, colsample_bytree=0.85,
                                       eval_metric="auc", tree_method="hist", random_state=SEED)
    model_for_shap.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    model_name = "xgboost"
elif CAT_OK:
    model_for_shap = CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=6,
                                        loss_function="Logloss", eval_metric="AUC",
                                        random_seed=SEED, verbose=False)
    model_for_shap.fit(Xtr, ytr, eval_set=(Xva, yva))
    model_name = "catboost"
else:
    # Fallback: LogisticRegression (no early stopping)
    model_for_shap = LogisticRegression(max_iter=2000)
    model_for_shap.fit(X_final[FEATURES].values, y_true)
    model_name = "logreg"

print("SHAP model:", model_name)

# SHAP explainer
import shap
shap_samples = min(200, len(mis_idx))  # keep it light
mis_sample_idx = np.random.choice(mis_idx, shap_samples, replace=False)

X_mis = X_final[FEATURES].iloc[mis_sample_idx]
X_bg  = X_final[FEATURES].iloc[np.random.choice(len(X_final), min(500, len(X_final)), replace=False)]

if model_name in ["lightgbm", "xgboost", "catboost"]:
    explainer = shap.TreeExplainer(model_for_shap)
    shap_values = explainer.shap_values(X_mis)
    # Handle binary case shape differences across libs
    if isinstance(shap_values, list):  # XGB/Cat may return [neg_class, pos_class]
        # assume positive class at index 1
        shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_vals = shap_values
else:
    # KernelExplainer fallback (slower) â€” use small background
    explainer = shap.KernelExplainer(model_for_shap.predict_proba, X_bg, link="logit")
    shap_vals = explainer.shap_values(X_mis, nsamples=200)[1]  # positive class

# SHAP summary plot (feature impact on misclassified)
plt.figure(figsize=(7,5))
shap.summary_plot(shap_vals, X_mis, plot_type="bar", show=False, max_display=20)
plt.title("SHAP (bar) â€” drivers on misclassified")
plt.show()

plt.figure(figsize=(7,5))
shap.summary_plot(shap_vals, X_mis, show=False, max_display=20)
plt.title("SHAP (beeswarm) â€” misclassified samples")
plt.show()

print("Read/Interpretation:")
print("  â€¢ High positive SHAP values push probability towards 1 (loan paid back).")
print("  â€¢ If important features push incorrectly for misclassified cases,")
print("    it hints at feature bias/transform issues or label/shift noise.")



# - Project misclassified samples to 2D (PCA) and cluster with DBSCAN
# - Visualize clusters (matplotlib + optional Plotly)
#
# Read/Interpretation
# - Non-noise clusters (labels >=0) indicate repeatable failure modes to investigate.

section("Failure Clusters â€” DBSCAN on PCA(2) of Misclassified")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Use top features for clustering (keeps signal dense)
cluster_feats = top_feats[:8]
X_mis_sub = X_final[cluster_feats].iloc[mis_idx].copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)

scaler = StandardScaler()
Xm = scaler.fit_transform(X_mis_sub.values)

pca = PCA(n_components=2, random_state=SEED)
Z = pca.fit_transform(Xm)

db = DBSCAN(eps=0.6, min_samples=8)  # tweakable
labels = db.fit_predict(Z)

clu_df = pd.DataFrame({
    "pc1": Z[:,0],
    "pc2": Z[:,1],
    "cluster": labels,
    "true_label": y_true[mis_idx],
    "pred_prob":  y_prob[mis_idx]
}, index=mis_idx)

# Plot clusters
fig, ax = plt.subplots(figsize=(6,5))
palette = { -1:"lightgray" }  # noise
unique = sorted(np.unique(labels))
for lab in unique:
    sel = (labels == lab)
    ax.scatter(Z[sel,0], Z[sel,1], s=18, alpha=0.8, label=f"cluster {lab}")
ax.set_title("DBSCAN clusters on misclassified (PCA 2D)")
ax.legend(bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.show()

# Show a few rows from each non-noise cluster (for pattern discovery)
for lab in [l for l in unique if l != -1][:3]:
    print(f"\n--- Cluster {lab} (sample) ---")
    sample_ids = clu_df.index[clu_df["cluster"] == lab][:10]
    display(X_final.loc[sample_ids, cluster_feats].assign(true=y_true[sample_ids], pred=y_pred[sample_ids], p=y_prob[sample_ids]))



section("Ensembling")

# Collect OOF columns per model if we kept them
# For simplicity, we used 'best model per fold', so for stacking we'll rebuild CV with fixed models if present.

models_available = set()
for fold_pred in preds:
    models_available.update(list(fold_pred.keys()))
models_available = sorted(list(models_available))
print("Models available:", models_available)

# Build test prediction matrix averaged over folds, per model
test_pred_by_model = {m: np.zeros(T_mat.shape[0]) for m in models_available}
for fold_pred in preds:
    for m in models_available:
        if m in fold_pred:
            test_pred_by_model[m] += fold_pred[m] / len(preds)

# Weighted average: tune weights to maximize OOF AUC (proxy)
# We'll create synthetic OOF per model by refitting on full data quickly (fallback: use best model OOF only)
oof_by_model = {}
for m in models_available:
    # crude approach: assume relative ranking of folds' best matches m; fallback to oof if not present
    # Here we simply fallback to global oof since we did "best per fold"
    oof_by_model[m] = oof  # placeholder

# grid search simple weights
best_auc, best_w = -1, None
grid = np.linspace(0.0, 1.0, 11)
for w1 in grid:
    for w2 in grid:
        for w3 in grid:
            w = np.array([w1, w2, w3][:len(models_available)])
            if w.sum() == 0: 
                continue
            w = w / w.sum()
            oof_blend = np.zeros_like(oof, dtype=float)
            # with placeholder OOF, this collapses to identical oof; still useful scaffold
            # in a more advanced refit, each model would have its own OOF
            # Here we just evaluate weight simplex shape.
            oof_blend = oof  # placeholder
            auc = roc_auc_score(y, oof_blend)
            if auc > best_auc:
                best_auc, best_w = auc, w

print("Best blend weight (placeholder search):", dict(zip(models_available, best_w)))
final_pred = np.zeros(T_mat.shape[0])
for i, m in enumerate(models_available):
    final_pred += best_w[i] * test_pred_by_model[m]

print("Read/Interpretation: Stacking could further help, but weighted blend is simple and robust.")




section("Submission")

submission = sub.copy()
submission[submission.columns[-1]] = final_pred.clip(0,1)

out_path = Path("./submission.csv")
submission.to_csv(out_path, index=False)
print("Saved:", out_path.resolve())

print("Read/Interpretation: File written. Submit and compare LB with CV; if mismatch is large, revisit AV drift and encodings.")


