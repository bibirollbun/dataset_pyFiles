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


import warnings, os, random
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
# Visuals
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'iframe'
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
output_notebook()

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression

import lightgbm as lgb
import xgboost as xgb
import category_encoders as ce

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

from IPython.display import HTML, display

SEED = 42
np.random.seed(SEED); random.seed(SEED)

THEME = """
<style>
:root{--bg:#0b1020;--ink:#e6edf3;--muted:#94a3b8;}
html,body{background:var(--bg);color:var(--ink);}
.card{background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.02));border:1px solid rgba(255,255,255,.08);
padding:12px 14px;border-radius:12px;box-shadow:0 6px 20px rgba(0,0,0,.25);margin:8px 0}
.k-mono{font-family:ui-monospace,Menlo,Consolas,monospace;}
</style>
"""
display(HTML(THEME))



INPUT_DIR = "/kaggle/input/playground-series-s5e9"
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

# Explicit target/id 
assert "BeatsPerMinute" in train.columns, f"Target 'BeatsPerMinute' not found. Columns: {list(train.columns)}"
TARGET = "BeatsPerMinute"
ID_COL = "id" if "id" in train.columns else train.columns[0]

display(HTML(f"<div class='card k-mono'>Loaded â€” train: {train.shape}, test: {test.shape}, sample_sub: {sample_sub.shape}</div>"))



def reduce_memory(df):
    for c in df.columns:
        if df[c].dtype == "int64": df[c] = pd.to_numeric(df[c], downcast="integer")
        elif df[c].dtype == "float64": df[c] = pd.to_numeric(df[c], downcast="float")
    return df

train = reduce_memory(train); test = reduce_memory(test)
feature_cols = [c for c in train.columns if c != TARGET]
cat_cols = [c for c in feature_cols if train[c].dtype == "object" or str(train[c].dtype).startswith("category")]
num_cols = [c for c in feature_cols if c not in cat_cols]
display(HTML(f"<div class='card k-mono'>Num: {len(num_cols)} | Cat: {len(cat_cols)}</div>"))



fig = px.histogram(train, x=TARGET, nbins=60, title=f"Target â€” {TARGET}")
fig.update_layout(template='plotly_dark', height=360); fig.show()

miss = train.isna().mean().sort_values(ascending=False)
dfm = miss[miss>0].reset_index(); dfm.columns = ["feature","missing_rate"]
if len(dfm):
    fig = px.bar(dfm, x="feature", y="missing_rate", title="Missing Rate (train)")
    fig.update_layout(template='plotly_dark', xaxis={'tickangle':45}); fig.show()



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



from itertools import combinations

X = train[feature_cols].copy(); X_test = test[feature_cols].copy()

# Frequency encoding for categoricals
for c in cat_cols:
    freq = X[c].value_counts(dropna=False)
    X[f"{c}__freq"] = X[c].map(freq)
    X_test[f"{c}__freq"] = X_test[c].map(freq).fillna(0)

# Base transforms 
def add_transforms_fast(Xdf, Xtestdf, y, num_cols):
    Xn, Xtn = Xdf.copy(), Xtestdf.copy()
    if not num_cols: return Xn, Xtn
    base_mi_mean = mutual_info_regression(Xdf[num_cols].fillna(Xdf[num_cols].median()), y, random_state=SEED).mean()
    for c in num_cols:
        col = Xdf[c]
        cands = {
            f"{c}__sqrt": np.sqrt(np.clip(col, 0, None)),
            f"{c}__log1p": np.log1p(np.clip(col, a_min=0, a_max=None))
        }
        kept = False
        for name, vals in cands.items():
            tmp = Xdf[num_cols].copy(); tmp[name] = vals
            mi_mean = mutual_info_regression(tmp.fillna(tmp.median()), y, random_state=SEED).mean()
            if mi_mean > base_mi_mean and not kept:
                Xn[name] = vals
                Xtn[name] = np.nan_to_num(np.log1p(np.clip(Xtestdf[c], a_min=0, a_max=None)) if "log1p" in name else np.sqrt(np.clip(Xtestdf[c], 0, None)))
                kept = True
    return Xn, Xtn

X_base, X_test_base = add_transforms_fast(X, X_test, train[TARGET].values, num_cols)

# Interaction set from top 6 MI features
mi = mutual_info_regression(train[num_cols].fillna(train[num_cols].median()), train[TARGET], random_state=SEED) if num_cols else []
mi_top = pd.Series(mi, index=num_cols).sort_values(ascending=False).head(min(6, len(num_cols))).index.tolist()

keep_inter = []
for (a,b) in list(combinations(mi_top, 2))[:6]:
    name = f"{a}__x__{b}"
    X_base[name] = train[a]*train[b]
    X_test_base[name] = test[a]*test[b]
    keep_inter.append(name)

X_fe = X_base.astype(np.float32); X_test_fe = X_test_base.astype(np.float32)
display(HTML(f"<div class='card k-mono'>Minimal FE â€” added base transforms + {len(keep_inter)} interactions</div>"))



num_cols_final = [c for c in X_fe.columns if c not in cat_cols]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("quant", QuantileTransformer(output_distribution="normal", random_state=SEED, n_quantiles=300))
])

def fit_target_encoder(X_tr, y_tr, X_va, cols):
    enc = ce.TargetEncoder(cols=cols, smoothing=0.2, min_samples_leaf=10, handle_unknown="value", handle_missing="value")
    enc.fit(X_tr[cols], y_tr)
    return enc.transform(X_tr[cols]).add_prefix("te__"), enc.transform(X_va[cols]).add_prefix("te__"), enc



N_SPLITS = 3
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_pred = np.zeros(len(train), dtype=np.float32)
test_preds = []


lgb_params = dict(
    n_estimators=5000,
    learning_rate=0.001,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    force_col_wise=True,
    verbose=-1,
)
'''
lgb_params = dict(
    n_estimators=12000, 
    learning_rate=0.02, 
    num_leaves=96,
    max_depth=-1,
    subsample=0.75, 
    colsample_bytree=0.75, 
    bagging_freq=1,
    min_child_samples=40, 
    min_split_gain=0.05,
    reg_alpha=0.3, 
    reg_lambda=3.0, 
    max_bin=255,
    extra_trees=True, 
    random_state=SEED, 
    n_jobs=-1, 
    force_col_wise=True, 
    verbose=-1
    )

lgb_params = dict(
        n_estimators=12000, learning_rate=0.02, num_leaves=64, max_depth=-1,
        subsample=0.8, colsample_bytree=0.8, feature_fraction_bynode=0.8,
        bagging_freq=1, min_child_samples=30, min_split_gain=0.0,
        reg_alpha=0.0, reg_lambda=2.0, max_bin=255,
        extra_trees=True, random_state=SEED, n_jobs=-1, force_col_wise=True, verbose=-1
    )'''
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_fe, train[TARGET])):
    display(HTML(f"<div class='card'><b>FAST â€” Fold {fold+1}/{N_SPLITS}</b></div>"))
    Xtr, Xva = X_fe.iloc[tr_idx].copy(), X_fe.iloc[va_idx].copy()
    ytr, yva = train[TARGET].iloc[tr_idx].values, train[TARGET].iloc[va_idx].values

    # Target encoding per fold
    if cat_cols:
        tr_te, va_te, te = fit_target_encoder(Xtr, ytr, Xva, cat_cols)
        Xtr = pd.concat([Xtr.drop(columns=cat_cols, errors="ignore"), tr_te], axis=1)
        Xva = pd.concat([Xva.drop(columns=cat_cols, errors="ignore"), va_te], axis=1)

    # Numeric pipeline
    Xtr_num = pd.DataFrame(num_pipeline.fit_transform(Xtr), index=Xtr.index)
    Xva_num = pd.DataFrame(num_pipeline.transform(Xva), index=Xva.index)

    non_num_cols_fold = [c for c in Xtr.columns if c not in num_cols_final]
    Xtr_other = Xtr[non_num_cols_fold].reset_index(drop=True)
    Xva_other = Xva[non_num_cols_fold].reset_index(drop=True)

    Xtr_final = pd.concat([Xtr_num.reset_index(drop=True), Xtr_other], axis=1).astype(np.float32)
    Xva_final = pd.concat([Xva_num.reset_index(drop=True), Xva_other], axis=1).astype(np.float32)

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        Xtr_final, ytr,
        eval_set=[(Xva_final, yva)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(period=0)]
    )

    val_pred = model.predict(Xva_final, num_iteration=model.best_iteration_)
    oof_pred[va_idx] = val_pred
    rmse = mean_squared_error(yva, val_pred, squared=False)
    display(HTML(f"<div class='card k-mono'>Fold RMSE: {rmse:.5f} | Best iters: {model.best_iteration_}</div>"))

    # Build test features under same encoder/pipeline
    if cat_cols:
        X_test_fold = X_test_fe.drop(columns=cat_cols, errors="ignore")
        X_test_fold = pd.concat([X_test_fold, te.transform(X_test_fe[cat_cols]).add_prefix('te__')], axis=1)
    else:
        X_test_fold = X_test_fe.copy()

    X_test_num = pd.DataFrame(num_pipeline.transform(X_test_fold))
    X_test_other = X_test_fold[[c for c in X_test_fold.columns if c not in num_cols_final]].reset_index(drop=True)
    X_test_final = pd.concat([X_test_num.reset_index(drop=True), X_test_other], axis=1).astype(np.float32)

    test_preds.append(model.predict(X_test_final, num_iteration=model.best_iteration_).astype(np.float32))

oof_rmse = mean_squared_error(train[TARGET], oof_pred, squared=False)
display(HTML(f"<div class='card'><b>FAST â€” OOF RMSE:</b> {oof_rmse:.6f}</div>"))




N_SPLITS_XGB = N_SPLITS
kf_xgb = KFold(n_splits=N_SPLITS_XGB, shuffle=True, random_state=SEED)

oof_xgb6 = np.zeros(len(train), dtype=np.float32)
oof_xgb8 = np.zeros(len(train), dtype=np.float32)
test_preds_xgb6, test_preds_xgb8 = [], []

xgb6_params = dict(tree_method="hist", n_estimators=15000, learning_rate=0.03,
                   max_depth=6, min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
                   reg_lambda=2.0, random_state=SEED, n_jobs=-1)

xgb8_params = dict(tree_method="hist", n_estimators=15000, learning_rate=0.03,
                   max_depth=8, min_child_weight=4, subsample=0.7, colsample_bytree=0.9,
                   reg_lambda=2.0, gamma=0.1, random_state=SEED, n_jobs=-1)

for fold, (tr_idx, va_idx) in enumerate(kf_xgb.split(X_fe, train[TARGET])):
    Xtr, Xva = X_fe.iloc[tr_idx].copy(), X_fe.iloc[va_idx].copy()
    ytr, yva = train[TARGET].iloc[tr_idx].values, train[TARGET].iloc[va_idx].values

    if cat_cols:
        tr_te, va_te, te_x = fit_target_encoder(Xtr, ytr, Xva, cat_cols)
        Xtr = pd.concat([Xtr.drop(columns=cat_cols, errors="ignore"), tr_te], axis=1)
        Xva = pd.concat([Xva.drop(columns=cat_cols, errors="ignore"), va_te], axis=1)

    Xtr_num = pd.DataFrame(num_pipeline.fit_transform(Xtr), index=Xtr.index)
    Xva_num = pd.DataFrame(num_pipeline.transform(Xva), index=Xva.index)
    non_num_cols_fold = [c for c in Xtr.columns if c not in num_cols_final]
    Xtr_other = Xtr[non_num_cols_fold].reset_index(drop=True)
    Xva_other = Xva[non_num_cols_fold].reset_index(drop=True)

    Xtr_final = pd.concat([Xtr_num.reset_index(drop=True), Xtr_other], axis=1).astype(np.float32)
    Xva_final = pd.concat([Xva_num.reset_index(drop=True), Xva_other], axis=1).astype(np.float32)

    m6 = xgb.XGBRegressor(**xgb6_params)
    m6.fit(Xtr_final, ytr, eval_set=[(Xva_final, yva)], eval_metric="rmse",
           verbose=False, early_stopping_rounds=300)
    pred6 = m6.predict(Xva_final)
    oof_xgb6[va_idx] = pred6

    m8 = xgb.XGBRegressor(**xgb8_params)
    m8.fit(Xtr_final, ytr, eval_set=[(Xva_final, yva)], eval_metric="rmse",
           verbose=False, early_stopping_rounds=300)
    pred8 = m8.predict(Xva_final)
    oof_xgb8[va_idx] = pred8

    if cat_cols:
        X_test_fold = X_test_fe.drop(columns=cat_cols, errors="ignore")
        X_test_fold = pd.concat([X_test_fold, te_x.transform(X_test_fe[cat_cols]).add_prefix('te__')], axis=1)
    else:
        X_test_fold = X_test_fe.copy()

    X_test_num = pd.DataFrame(num_pipeline.transform(X_test_fold))
    X_test_other = X_test_fold[[c for c in X_test_fold.columns if c not in num_cols_final]].reset_index(drop=True)
    X_test_final = pd.concat([X_test_num.reset_index(drop=True), X_test_other], axis=1).astype(np.float32)

    test_preds_xgb6.append(m6.predict(X_test_final).astype(np.float32))
    test_preds_xgb8.append(m8.predict(X_test_final).astype(np.float32))

pred_test_xgb6 = np.mean(test_preds_xgb6, axis=0)
pred_test_xgb8 = np.mean(test_preds_xgb8, axis=0)

oof_rmse_xgb6 = mean_squared_error(train[TARGET], oof_xgb6, squared=False)
oof_rmse_xgb8 = mean_squared_error(train[TARGET], oof_xgb8, squared=False)
display(HTML(f"<div class='card'>XGB OOF RMSE â€” xgb6: {oof_rmse_xgb6:.6f} | xgb8: {oof_rmse_xgb8:.6f}</div>"))


from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

pred_test_lgb = np.mean(test_preds, axis=0)

oof_stack_df = pd.DataFrame({
    "lgb":  oof_pred,
    "xgb6": oof_xgb6,
    "xgb8": oof_xgb8,
})
meta = RidgeCV(alphas=np.logspace(-4, 2, 25), cv=5)
meta.fit(oof_stack_df, train[TARGET])

stack_oof_fullfit = meta.predict(oof_stack_df)  
stack_rmse_fullfit = mean_squared_error(train[TARGET], stack_oof_fullfit, squared=False)

test_stack_df = pd.DataFrame({
    "lgb":  pred_test_lgb,
    "xgb6": pred_test_xgb6,
    "xgb8": pred_test_xgb8,
})
FINAL_TEST_PRED = meta.predict(test_stack_df)


kf_stack = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
stack_oof_cv = np.zeros(len(train), dtype=np.float32)
fold_rmse = { "lgb": [], "xgb6": [], "xgb8": [], "stack": [] }
fold_labels = []

for fold_i, (tr_idx, va_idx) in enumerate(kf_stack.split(oof_stack_df, train[TARGET])):
    meta_cv = RidgeCV(alphas=np.logspace(-4, 2, 25), cv=5)
    meta_cv.fit(oof_stack_df.iloc[tr_idx], train[TARGET].iloc[tr_idx])
    stack_oof_cv[va_idx] = meta_cv.predict(oof_stack_df.iloc[va_idx])

    # per-fold RMSEs
    yv = train[TARGET].iloc[va_idx].values
    fold_rmse["lgb"].append(mean_squared_error(yv, oof_pred[va_idx],  squared=False))
    fold_rmse["xgb6"].append(mean_squared_error(yv, oof_xgb6[va_idx], squared=False))
    fold_rmse["xgb8"].append(mean_squared_error(yv, oof_xgb8[va_idx], squared=False))
    fold_rmse["stack"].append(mean_squared_error(yv, stack_oof_cv[va_idx], squared=False))
    fold_labels.append(f"Fold {fold_i+1}")

stack_rmse_cv = mean_squared_error(train[TARGET], stack_oof_cv, squared=False)

display(HTML(
    f"<div class='card k-mono'><b>Stack OOF RMSE (CV):</b> {stack_rmse_cv:.6f} "
    f"| <span style='color:#94a3b8'>[Full-fit (optimistic): {stack_rmse_fullfit:.6f}]</span> "
    f"| Weights: {dict(zip(oof_stack_df.columns, meta.coef_))}</div>"
))

# --------- Visualizations ---------

# Per-fold RMSE bars
rmse_long = (pd.DataFrame({
    "fold": fold_labels,
    "lgb":   fold_rmse["lgb"],
    "xgb6":  fold_rmse["xgb6"],
    "xgb8":  fold_rmse["xgb8"],
    "stack": fold_rmse["stack"],
})
.melt(id_vars="fold", var_name="model", value_name="rmse"))

fig = px.bar(
    rmse_long, x="fold", y="rmse", color="model", barmode="group",
    title=f"Per-fold RMSE â€” LGB / XGB6 / XGB8 / STACK  (OOF CV stack RMSE: {stack_rmse_cv:.5f})"
)
fig.update_layout(template="plotly_dark", height=420)
fig.show()

# Global RMSE table bar
global_rmse = pd.DataFrame({
    "model": ["lgb", "xgb6", "xgb8", "stack_cv"],
    "rmse": [
        mean_squared_error(train[TARGET], oof_pred,  squared=False),
        mean_squared_error(train[TARGET], oof_xgb6, squared=False),
        mean_squared_error(train[TARGET], oof_xgb8, squared=False),
        stack_rmse_cv
    ]
})
fig = px.bar(global_rmse, x="model", y="rmse", title="Global OOF RMSE by model")
fig.update_layout(template="plotly_dark", height=300)
fig.show()

# Stacker weights
wdf = pd.DataFrame({"model": oof_stack_df.columns, "weight": meta.coef_})
fig = px.bar(wdf, x="model", y="weight", title="Blend Weights (RidgeCV full-fit)")
fig.update_layout(template="plotly_dark", height=300)
fig.show()

# OOF vs True scatter (sampled) for each model + ensemble
sample_idx = np.random.RandomState(SEED).choice(len(train), size=min(50000, len(train)), replace=False)
oof_compare = pd.DataFrame({
    "y": train[TARGET].values[sample_idx],
    "lgb":   oof_pred[sample_idx],
    "xgb6":  oof_xgb6[sample_idx],
    "xgb8":  oof_xgb8[sample_idx],
    "stack": stack_oof_cv[sample_idx],
})
for col in ["lgb","xgb6","xgb8","stack"]:
    fig = px.scatter(oof_compare, x="y", y=col, opacity=0.4, title=f"OOF vs Truth â€” {col.upper()}")
    fig.update_traces(marker=dict(size=4))
    ymin, ymax = oof_compare["y"].min(), oof_compare["y"].max()
    fig.add_trace(go.Scatter(x=[ymin, ymax], y=[ymin, ymax], mode="lines", name="y = x"))
    fig.update_layout(template="plotly_dark", height=320)
    fig.show()

# Residual histograms (faceted)
res_df = oof_compare.copy()
for col in ["lgb","xgb6","xgb8","stack"]:
    res_df[f"res_{col}"] = res_df[col] - res_df["y"]
res_long = res_df.melt(id_vars=["y"], value_vars=[f"res_{c}" for c in ["lgb","xgb6","xgb8","stack"]],
                       var_name="res_model", value_name="residual")
res_long["model"] = res_long["res_model"].str.replace("res_","", regex=False).str.upper()

fig = px.histogram(res_long, x="residual", facet_col="model", facet_col_wrap=2,
                   nbins=80, title="Residuals (pred âˆ’ true) â€” OOF (sampled)")
fig.update_layout(template="plotly_dark", height=520)
fig.show()

# Correlation heatmap of base OOF predictions (diversity â†’ better stacking)
corr = oof_stack_df.corr()
fig = px.imshow(corr, text_auto=".2f", title="Correlation of base model OOF predictions")
fig.update_layout(template="plotly_dark", height=360)
fig.show()



# Rebuild the same CV splits to compute per-fold RMSE from OOF predictions
kf_check = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_rmse = []
fold_ids = []
for fold_i, (_, va_idx) in enumerate(kf_check.split(X_fe, train[TARGET])):
    y_true_f = train[TARGET].iloc[va_idx].values
    y_pred_f = oof_pred[va_idx]
    fold_rmse.append(mean_squared_error(y_true_f, y_pred_f, squared=False))
    fold_ids.append(f"Fold {fold_i+1}")

#1 Bar chart of per-fold RMSE
rmse_df = pd.DataFrame({"fold": fold_ids, "rmse": fold_rmse})
fig = px.bar(rmse_df, x="fold", y="rmse", title=f"Per-fold RMSE (OOF) â€” OOF RMSE: {oof_rmse:.5f}")
fig.update_layout(template="plotly_dark", height=360)
fig.show()

#2 OOF vs. True scatter with y=x reference
oof_df = pd.DataFrame({"y_true": train[TARGET].values, "y_pred": oof_pred})
fig = px.scatter(oof_df.sample(min(40000, len(oof_df)), random_state=SEED),
                 x="y_true", y="y_pred", opacity=0.4,
                 title="OOF predictions vs. ground truth")
fig.update_traces(marker=dict(size=4))
fig.add_trace(go.Scatter(x=[oof_df.y_true.min(), oof_df.y_true.max()],
                         y=[oof_df.y_true.min(), oof_df.y_true.max()],
                         mode="lines", name="y = x"))
fig.update_layout(template="plotly_dark", height=380)
fig.show()

#3 histogram
resid = oof_df["y_pred"] - oof_df["y_true"]
fig = px.histogram(resid, nbins=80, title="OOF residuals (pred âˆ’ true)")
fig.update_layout(template="plotly_dark", height=320)
fig.show()



sub_blend = sample_sub.copy()
if "id" in sub_blend.columns:
    sub_blend["id"] = test[ID_COL]
pred_col = [c for c in sub_blend.columns if c != "id"]
pred_name = pred_col[0] if len(pred_col)==1 else (TARGET if TARGET in sub_blend.columns else sub_blend.columns[-1])

sub_blend[pred_name] = FINAL_TEST_PRED
sub_blend.to_csv("/kaggle/working/submission.csv", index=False)
sub_blend.to_csv("/kaggle/working/submission_blend.csv", index=False)
display(HTML("<div class='card k-mono'>âœ… Saved blended submission â†’ /kaggle/working/submission.csv</div>"))

# optional: also save LGB-only
sub_lgb = sample_sub.copy()
if "id" in sub_lgb.columns:
    sub_lgb["id"] = test[ID_COL]
sub_lgb[pred_name] = pred_test_lgb
sub_lgb.to_csv("/kaggle/working/submission_lgb.csv", index=False)

