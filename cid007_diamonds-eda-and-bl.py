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


# Imports & Global Config
import os, sys, json, gc, itertools, warnings, random, math, time, pathlib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Viz
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'iframe'
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource, FactorRange

output_notebook()

# ML stack
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import RFECV, mutual_info_regression
from sklearn.inspection import permutation_importance

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

# Gradient boosting libs
try:
    import lightgbm as lgb
except Exception as e:
    lgb = None

try:
    import xgboost as xgb
except Exception as e:
    xgb = None

try:
    from catboost import CatBoostRegressor, Pool
except Exception as e:
    CatBoostRegressor = None

SEED = 1337
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"]=str(seed)
seed_everything()

FAST_MODE = True   # flip to False for full run (enables SHAP)
N_SPLITS  = 3 if FAST_MODE else 5

from joblib import parallel_backend


#Data Loading 
DATA_ROOTS = [
    "/kaggle/input/predicting-the-price-of-diamond",
    "/kaggle/input/diamond-price-prediction", 
    "/kaggle/input/diamonds", 
    "/kaggle/input"
]

train_path = test_path = sample_path = None
for root in DATA_ROOTS:
    if not os.path.exists(root): 
        continue
    cand = list(pathlib.Path(root).rglob("*.csv"))
    for p in cand:
        n = p.name.lower()
        if "train" in n and train_path is None: train_path = str(p)
        if "test"  in n and test_path  is None: test_path  = str(p)
        if "sample" in n and "sub" in n and sample_path is None: sample_path = str(p)

assert train_path is not None, "Could not locate train CSV."
assert test_path  is not None, "Could not locate test CSV."

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path) if sample_path and os.path.exists(sample_path) else None

print("train.shape:", train.shape, "| test.shape:", test.shape)
train.head()



# Basic schema 
display(pd.DataFrame({
    "col": train.columns,
    "dtype": train.dtypes.astype(str),
    "n_null": train.isna().sum().values,
    "n_unique": [train[c].nunique() for c in train.columns]
}).sort_values(["dtype","n_unique"]).reset_index(drop=True))

target = "price" if "price" in train.columns else [c for c in train.columns if c.lower()=="price"][0]
id_cols = [c for c in train.columns if "id" in c.lower()]
cat_cols = [c for c in train.columns if train[c].dtype=="object"]
num_cols = [c for c in train.columns if c not in cat_cols + [target] + id_cols]

print("Target:", target)
print("Categorical:", cat_cols)
print("Numerical:", num_cols[:12], "... total", len(num_cols))

# Seaborn target distribution
sns.set_theme(style="whitegrid", palette="rocket")
ax = sns.kdeplot(train[target], fill=True)
ax.set_title("Target (price) â€” KDE", fontsize=14)

# Plotly numeric correlation
if len(num_cols) >= 2:
    corr = train[num_cols + [target]].corr(numeric_only=True)
    fig = px.imshow(corr, title="Numeric Correlations (incl. price)", aspect="auto")
    fig.show()

# Bokeh scatter â€” carat vs price
if "carat" in train.columns:
    subset = train.sample(min(5000, len(train)), random_state=SEED)
    cds = ColumnDataSource(subset)
    p = figure(width=720, height=450, title="Bokeh: carat vs price")
    p.circle("carat", target, size=5, alpha=0.5, source=cds)
    p.add_tools(HoverTool(tooltips=[("carat","@carat"), ("price", "@"+target)]))
    show(p)

# Seaborn boxplots for common diamond categoricals
for c in ["cut","color","clarity"]:
    if c in train.columns:
        sns.set_palette("mako")
        ax = sns.boxplot(data=train, x=c, y=target, showfliers=False)
        ax.set_title(f"{c} vs price â€” Boxplot", fontsize=13)



from sklearn.feature_selection import mutual_info_regression

#Target (log) 
USE_LOG_TARGET = True
y_raw = pd.to_numeric(train[target], errors="coerce").astype(float)
y_raw = y_raw.clip(lower=0.0)  # guard for log
y = np.log1p(y_raw) if USE_LOG_TARGET else y_raw

def inv_target(z):
    return np.expm1(z) if USE_LOG_TARGET else z

try:
    Y_BINS = pd.qcut(y_raw, q=10, labels=False, duplicates='drop')
except Exception:
    Y_BINS = None

# FE utilities 
def safe_log(x): 
    return np.log1p(np.maximum(x, 0))

def make_base_transforms(df, nums):
    out = {}
    for c in nums:
        s = df[c].astype(float)
        out[f"{c}_log1p"] = safe_log(s)
        out[f"{c}_sqrt"]  = np.sqrt(np.clip(s, 0, None))
        out[f"{c}_sq"]    = s**2
    return pd.DataFrame(out, index=df.index)

def pairwise_interactions(df, cols):
    res = {}
    for a, b in itertools.combinations(cols, 2):
        a1, b1 = df[a].astype(float), df[b].astype(float)
        res[f"{a}_over_{b}"]  = np.where(b1!=0, a1/b1, 0)
        res[f"{a}_minus_{b}"] = a1 - b1
        res[f"{a}_times_{b}"] = a1 * b1
    return pd.DataFrame(res, index=df.index)

def target_freq_encoding(train_df, test_df, col, y, smooth=20):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    global_mean = y.mean()
    tr_enc = pd.Series(index=train_df.index, dtype=float)
    for tr_idx, val_idx in kf.split(train_df):
        tr = train_df.iloc[tr_idx]
        stats = tr.groupby(col)[y.name].agg(["count","mean"])
        enc = (stats["count"]*stats["mean"] + smooth*global_mean) / (stats["count"]+smooth)
        tr_enc.iloc[val_idx] = train_df.iloc[val_idx][col].map(enc).fillna(global_mean)
    stats_full = train_df.groupby(col)[y.name].agg(["count","mean"])
    enc_full  = (stats_full["count"]*stats_full["mean"] + smooth*global_mean) / (stats_full["count"]+smooth)
    te_enc = test_df[col].map(enc_full).fillna(global_mean)
    return tr_enc.values, te_enc.values



#Domain features
for df in (train, test):
    for c in ["x","y","z","depth","table","carat"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if all(k in df.columns for k in ["x","y","z"]):
        df["volume"] = (df["x"].clip(lower=1e-6)*df["y"].clip(lower=1e-6)*df["z"].clip(lower=1e-6))
        df["xy_ratio"] = (df["x"] / df["y"].replace(0, np.nan)).fillna(0)
        df["xz_ratio"] = (df["x"] / df["z"].replace(0, np.nan)).fillna(0)
        df["yz_ratio"] = (df["y"] / df["z"].replace(0, np.nan)).fillna(0)
    if "table" in df.columns:
        df["table_ratio"] = df["table"]/100.0
    if "depth" in df.columns:
        df["depth_ratio"] = df["depth"]/100.0

X_base = train[[c for c in train.columns if c in (num_cols)]].copy() if num_cols else pd.DataFrame(index=train.index)
X_test_base = test[[c for c in test.columns if c in (num_cols)]].copy() if num_cols else pd.DataFrame(index=test.index)

X_bt      = make_base_transforms(train, num_cols) if num_cols else pd.DataFrame(index=train.index)
X_test_bt = make_base_transforms(test,  num_cols) if num_cols else pd.DataFrame(index=test.index)

# MI to choose core numerics for interactions
mi_scores = {}
if num_cols:
    vals = mutual_info_regression(train[num_cols].fillna(train[num_cols].median()), y, random_state=SEED)
    mi_series = pd.Series(vals, index=num_cols).sort_values(ascending=False)
    core_for_pairwise = list(mi_series.head(min(6, max(2, len(num_cols)//2)) ).index)
else:
    core_for_pairwise = []

X_pw      = pairwise_interactions(train, core_for_pairwise) if core_for_pairwise else pd.DataFrame(index=train.index)
X_test_pw = pairwise_interactions(test,  core_for_pairwise) if core_for_pairwise else pd.DataFrame(index=test.index)

X_te = pd.DataFrame(index=train.index); X_test_te = pd.DataFrame(index=test.index)
for c in cat_cols:
    tr_enc, te_enc = target_freq_encoding(train[[c]].assign(**{target:y.values}), test[[c]], c, y.rename(target))
    X_te[f"{c}_te"]      = tr_enc
    X_test_te[f"{c}_te"] = te_enc

X_all = pd.concat([X_base, X_bt, X_pw, X_te], axis=1)
X_test_all = pd.concat([X_test_base, X_test_bt, X_test_pw, X_test_te], axis=1)
print("X_all:", X_all.shape, "| X_test_all:", X_test_all.shape)


# Impute + MI screening 

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_all_imp   = pd.DataFrame(imputer.fit_transform(X_all), columns=X_all.columns, index=X_all.index)
X_test_imp  = pd.DataFrame(imputer.transform(X_test_all), columns=X_test_all.columns, index=X_test_all.index)

mi_all = mutual_info_regression(X_all_imp, y, random_state=SEED)
mi_df = pd.DataFrame({"feature": X_all_imp.columns, "MI": mi_all}).sort_values("MI", ascending=False)
display(mi_df.head(20))

TOP_FEATS = list(mi_df.query("MI>0").head(200)["feature"])
if FAST_MODE:
    TOP_FEATS = TOP_FEATS[:120]
Xs = X_all_imp[TOP_FEATS].copy()
Xs_test = X_test_imp[TOP_FEATS].copy()
print("Using", len(TOP_FEATS), "features for first pass.")



def cv_oof(model, X, y, X_test, n_splits=N_SPLITS, name="model"):
    from sklearn.model_selection import StratifiedKFold
    # Stratified on raw target bins if available
    if 'Y_BINS' in globals() and Y_BINS is not None:
        _SKF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        _splits = _SKF.split(X, Y_BINS)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        _splits = kf.split(X)
    oof = np.zeros(len(X)); preds = np.zeros(len(X_test)); scores = []
    for fold,(tr_idx, val_idx) in enumerate(_splits):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[val_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[val_idx]
        m = model()

        def _build_mono(cols):
            cons = []
            for f in cols:
                if ('carat' in f) or (f=='carat') or ('volume' in f):
                    cons.append(1)
                else:
                    cons.append(0)
            return cons
        if name.startswith("lgb") and lgb is not None:
            m = lgb.LGBMRegressor(
                monotone_constraints=_build_mono(X.columns),
                n_estimators=2000 if not FAST_MODE else 600,
                learning_rate=0.03 if not FAST_MODE else 0.06,
                subsample=0.8, colsample_bytree=0.8,
                num_leaves=63, max_depth=-1, random_state=SEED, n_jobs=-1
            )
        elif name.startswith("xgb") and xgb is not None:
            m = xgb.XGBRegressor(
                n_estimators=2000 if not FAST_MODE else 700,
                learning_rate=0.03 if not FAST_MODE else 0.06,
                subsample=0.8, colsample_bytree=0.8,
                max_depth=8, tree_method="hist", random_state=SEED, n_jobs=-1
            )
        elif name.startswith("cat") and CatBoostRegressor is not None:
            m = CatBoostRegressor(
                iterations=1500 if not FAST_MODE else 700, depth=8, learning_rate=0.05,
                loss_function="MAE", verbose=False, random_state=SEED
            )
        elif name.startswith("et"):
            m = ExtraTreesRegressor(
                n_estimators=800 if not FAST_MODE else 400, max_depth=None,
                random_state=SEED, n_jobs=-1
            )
        elif name.startswith("hgbr"):
            m = HistGradientBoostingRegressor(
                max_depth=None, max_iter=1200 if not FAST_MODE else 300, random_state=SEED
            )
        else:
            m = Ridge(alpha=1.0, random_state=SEED)

        m.fit(Xtr, ytr)
        p = m.predict(Xva); oof[val_idx] = p
        preds += m.predict(X_test) / n_splits
        mae = mean_absolute_error(yva, p); scores.append(mae)
        print(f"[{name}] fold {fold}: MAE={mae:.4f}")
        del m; gc.collect()
    print(f"[{name}] CV MAE: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    return oof, preds, scores

oof_lgb, pred_lgb, _ = cv_oof(lambda: None, Xs, y, Xs_test, name="lgb")
oof_xgb, pred_xgb, _ = cv_oof(lambda: None, Xs, y, Xs_test, name="xgb")
oof_et,  pred_et,  _ = cv_oof(lambda: None, Xs, y, Xs_test, name="et")
oof_hg,  pred_hg,  _ = cv_oof(lambda: None, Xs, y, Xs_test, name="hgbr")
if CatBoostRegressor is not None:
    oof_cat, pred_cat, _ = cv_oof(lambda: None, Xs, y, Xs_test, name="cat")
else:
    oof_cat, pred_cat = np.zeros_like(oof_lgb), np.zeros_like(pred_lgb)


# Meta-learner on OOFs 
# === Stacking (meta-learner) ==================================================
STACK = pd.DataFrame({"lgb":oof_lgb,"xgb":oof_xgb,"et":oof_et,"hg":oof_hg})
STACK_TEST = pd.DataFrame({"lgb":pred_lgb,"xgb":pred_xgb,"et":pred_et,"hg":pred_hg})
if CatBoostRegressor is not None:
    STACK["cat"] = oof_cat; STACK_TEST["cat"] = pred_cat

meta = Ridge(alpha=0.5, random_state=SEED)
meta.fit(STACK, y)
meta_oof  = meta.predict(STACK)
meta_pred = meta.predict(STACK_TEST)

print("Meta (Ridge) OOF MAE:", mean_absolute_error(y, meta_oof))


# RFECV on a fast model + permutation importance
base_est = ExtraTreesRegressor(n_estimators=400, random_state=SEED, n_jobs=-1)
rfecv = RFECV(estimator=base_est, step=50, cv=KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED),
              scoring="neg_mean_absolute_error", min_features_to_select=40)
with parallel_backend("threading"):
    rfecv.fit(Xs, y)
selected_feats = list(Xs.columns[rfecv.support_])
print("RFECV kept:", len(selected_feats))

# Permutation importance
base_est.fit(Xs[selected_feats], y)
with parallel_backend("threading"):
    perm = permutation_importance(
        base_est, Xs[selected_feats], y,
        n_repeats=10, random_state=SEED, n_jobs=-1
    )
    
pi = pd.DataFrame({"feature": selected_feats, "PI": perm.importances_mean}).sort_values("PI", ascending=False)
display(pi.head(20))

Xs_sel = Xs[selected_feats]; Xs_sel_test = Xs_test[selected_feats]

o2_lgb, p2_lgb, _ = cv_oof(lambda: None, Xs_sel, y, Xs_sel_test, name="lgb")
o2_xgb, p2_xgb, _ = cv_oof(lambda: None, Xs_sel, y, Xs_sel_test, name="xgb")
o2_et,  p2_et,  _ = cv_oof(lambda: None, Xs_sel, y, Xs_sel_test, name="et")
o2_hg,  p2_hg, _ = cv_oof(lambda: None, Xs_sel, y, Xs_sel_test, name="hgbr")

STACK2 = pd.DataFrame({"lgb":o2_lgb,"xgb":o2_xgb,"et":o2_et,"hg":o2_hg})
STACK2_TEST = pd.DataFrame({"lgb":p2_lgb,"xgb":p2_xgb,"et":p2_et,"hg":p2_hg})
meta2 = ElasticNet(alpha=0.01, l1_ratio=0.15, random_state=SEED)
meta2.fit(STACK2, y)
meta2_oof  = meta2.predict(STACK2)
meta2_pred = meta2.predict(STACK2_TEST)
print("Meta2 (ElasticNet) OOF MAE:", mean_absolute_error(y, meta2_oof))


from sklearn.isotonic import IsotonicRegression
import numpy as np
try:
    from scipy.optimize import nnls
except Exception:
    nnls = None

iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
iso.fit(meta2_oof, y)
meta2_oof_cal  = iso.transform(meta2_oof)
meta2_pred_cal = iso.transform(meta2_pred)

if 'STACK2' in globals():
    if nnls is not None:
        w, _ = nnls(STACK2.values, y.values)
        w = w / (w.sum() + 1e-12)
    else:
        w = np.ones(STACK2.shape[1]) / STACK2.shape[1]
    oof_nnls  = STACK2.values @ w
    pred_nnls = STACK2_TEST.values @ w
    print("NNLS weights:", np.round(w, 4))
else:
    pred_nnls = None


df_diag = pd.DataFrame({"y_raw": y_raw, "oof": inv_target(meta2_oof_cal)})
try:
    df_diag["decile"] = pd.qcut(df_diag["y_raw"], 10, labels=False, duplicates='drop')
    grp = df_diag.groupby("decile").apply(lambda d: (d["oof"]-d["y_raw"]).median()).rename("median_resid")
    display(grp)
except Exception as e:
    print("Decile residual diagnostic skipped:", e)

df_oofc = pd.DataFrame({"y": y_raw, "oof_cal": inv_target(meta2_oof_cal)})
fig = px.scatter(df_oofc.sample(min(8000, len(df_oofc)), random_state=SEED),
                 x="y", y="oof_cal",
                 title="OOF vs Truth â€” Calibrated (raw scale)",
                 labels={"y":"Truth","oof_cal":"OOF (calibrated)"})
fig.add_trace(go.Scatter(x=[df_oofc["y"].min(), df_oofc["y"].max()],
                         y=[df_oofc["y"].min(), df_oofc["y"].max()],
                         mode="lines", name="y=x"))
fig.show()


if lgb is not None:
    resid = y - meta2_oof_cal
    if 'Xs_sel' in globals():
        rb = lgb.LGBMRegressor(n_estimators=400 if FAST_MODE else 900, learning_rate=0.05,
                               subsample=0.9, colsample_bytree=0.9, num_leaves=63,
                               random_state=SEED, n_jobs=-1)
        rb.fit(Xs_sel, resid)
        resid_pred = rb.predict(Xs_sel_test)
    else:
        rb = lgb.LGBMRegressor(n_estimators=400 if FAST_MODE else 900, learning_rate=0.05,
                               subsample=0.9, colsample_bytree=0.9, num_leaves=63,
                               random_state=SEED, n_jobs=-1)
        rb.fit(Xs, resid)
        resid_pred = rb.predict(Xs_test)
else:
    resid_pred = 0.0


# Lightweight SHAP summary 
try:
    import shap
except Exception as e:
    shap = None

def run_shap_summary():
    if shap is None:
        print("SHAP not available; skipping.")
        return
    if lgb is None:
        print("LightGBM not available; skipping SHAP.")
        return
    if FAST_MODE:
        print("FAST_MODE=True â†’ SHAP skipped. Set FAST_MODE=False to enable.")
        return

    # Prefer pruned features
    if 'selected_feats' in globals():
        use_feats = selected_feats
        X_train_use = Xs[use_feats] if 'Xs' in globals() else None
        if 'Xs_sel' in globals():
            X_train_use = Xs_sel
    else:
        use_feats = TOP_FEATS
        X_train_use = Xs[use_feats] if 'Xs' in globals() else None

    if X_train_use is None:
        print("No feature matrix for SHAP; skipping.")
        return

    model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        num_leaves=63, max_depth=-1, random_state=SEED, n_jobs=-1
    )
    model.fit(X_train_use, y)

    n_sample = min(3000, len(X_train_use))
    X_sample = X_train_use.sample(n_sample, random_state=SEED)

    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    shap.summary_plot(shap_vals, X_sample, show=True)

run_shap_summary()


#OOF vs Truth 
df_oof = pd.DataFrame({"y":y, "meta2_oof":meta2_oof})
fig = px.scatter(df_oof.sample(min(8000, len(df_oof)), random_state=SEED), x="y", y="meta2_oof",
                 title="OOF vs Truth â€” Refined Stack (ElasticNet)",
                 labels={"y":"Truth","meta2_oof":"OOF Pred"})
fig.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode="lines", name="y=x"))
fig.show()


# Final prediction & submission
# Convert meta predictions back to price scale 
pred_meta1 = inv_target(meta_pred)           if 'meta_pred' in globals() else 0.0
pred_meta2 = inv_target(meta2_pred_cal)      if 'meta2_pred_cal' in globals() else inv_target(meta2_pred)
pred_nnls_ = inv_target(pred_nnls)           if 'pred_nnls' in globals() and pred_nnls is not None else 0.0
resid_part = resid_pred                      if 'resid_pred' in globals() else 0.0

parts = []; weights = []
if isinstance(pred_meta1, (np.ndarray, list)): parts.append(pred_meta1); weights.append(0.35)
if isinstance(pred_meta2, (np.ndarray, list)): parts.append(pred_meta2); weights.append(0.35)
if isinstance(pred_nnls_, (np.ndarray, list)): parts.append(pred_nnls_); weights.append(0.30)

if len(parts)==0:
    final_pred = pred_meta2 if isinstance(pred_meta2, (np.ndarray, list)) else pred_meta1
else:
    final_pred = np.average(np.vstack(parts), axis=0, weights=weights[:len(parts)])

if isinstance(resid_part, (np.ndarray, list)):
    final_pred = final_pred + 0.5 * resid_part

sub = sample_sub.copy() if sample_sub is not None else pd.DataFrame()
if sub.empty:
    key_col = "id" if "id" in test.columns else test.columns[0]
    sub = pd.DataFrame({key_col: test[key_col].values})
sub[target] = final_pred
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv"); display(sub.head())


sub.shape

