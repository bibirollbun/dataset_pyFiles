import itertools, json, math, os, sys, warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize


print("Reading train / test")
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
y_true = train["Calories"].values
ids    = test["id"].values

ver_meta = {
    "v1": {"oof"     : "/kaggle/input/calorie-prediction-oof-predictions/oof_predictions_pg_s5e5v1.csv",
           "sub_dir" : "/kaggle/input/calorie-prediction-oof-predictions",
           "sub_fmt" : "sub_{model}_v1.csv"},
    "v2": {"oof"     : "/kaggle/input/calorie-prediction-oof-predictions/oof_predictions_pg_s5e5v2.csv",
           "sub_dir" : "/kaggle/input/calorie-prediction-oof-predictions",
           "sub_fmt" : "sub_{model}_v2.csv"},
    "v3": {"oof"     : "/kaggle/input/calorie-prediction-oof-predictions/oof_predictions_pg_s5e5v3.csv",
           "sub_dir" : "/kaggle/input/calorie-prediction-oof-predictions",
           "sub_fmt" : "sub_{model}_v3.csv"},
    #  v4 files are just sub_<Model>.csv (no “_v4” suffix)
    "v4": {"oof"     : "/kaggle/input/57-02-feature-engineering/oof_predictions_pg_s5e5.csv",
           "sub_dir" : "/kaggle/input/57-02-feature-engineering/subs",
           "sub_fmt" : "sub_{model}.csv"},
}

def load_version(tag):
    meta   = ver_meta[tag]
    oof_df = pd.read_csv(meta["oof"]).set_index("id")
    model_cols = [c for c in oof_df.columns if c != "Calories"]

    pred_test = {}
    for m in model_cols:
        # build expected filename and read it
        sub_fp = Path(meta["sub_dir"]) / meta["sub_fmt"].format(model=m)
        pred_test[m] = pd.read_csv(sub_fp).set_index("id")["Calories"]
    test_df = pd.DataFrame(pred_test)

    print(f" • {tag}: {len(oof_df):,} rows | {len(model_cols)} models")
    return oof_df[model_cols], test_df[model_cols]

versions = {tag: load_version(tag) for tag in ver_meta}


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, 0, None)))


cv_score = {
    # ── Version-1 (“no extra features”) ──
    "Ridge"      : 0.5648,
    "Lasso"      : 0.3475,
    "ElasticNet" : 0.4903,
    "HistGBR"    : 0.0635,
    "XGBoost"    : 0.060810,
    "LightGBM"   : 0.061795,
    "CatBoost"   : 0.0619,

    # ── Version-2 (“+4 features”) ──
    "Ridge_v2"      : 0.5301,
    "Lasso_v2"      : 0.3603,
    "ElasticNet_v2" : 0.4397,
    "HistGBR_v2"    : 0.0636,
    "XGBoost_v2"    : 0.0608,
    "LightGBM_v2"   : 0.0620,
    "CatBoost_v2"   : 0.0618,

    # ── Version-3 (“+20 features”) ──
    "Ridge_v3"      : 0.1639,
    "Lasso_v3"      : 0.1520,
    "ElasticNet_v3" : 0.1632,
    "HistGBR_v3"    : 0.0626,
    "XGBoost_v3"    : 0.0606,
    "LightGBM_v3"   : 0.0614,
    "CatBoost_v3"   : 0.0628,

    # ── Version-4 (“+140 features”; only 3 models) ──
    "XGBoost_v4"  : 0.0609,
    "LightGBM_v4" : 0.0615,
    "CatBoost_v4" : 0.0626,
}

lb_scores = {
    # V1
    "Ridge"      : 0.57370,
    "Lasso"      : 0.34456,
    "ElasticNet" : 0.48845,
    "HistGBR"    : 0.06264,
    "XGBoost"    : 0.05938,
    "LightGBM"   : 0.05976,
    "CatBoost"   : 0.06014,

    # V2  (only those you have – others omitted → equal weight)
    "HistGBR_v2"  : 0.06131,
    "XGBoost_v2"  : 0.05887,
    "LightGBM_v2" : 0.06018,
    "CatBoost_v2" : 0.05999,

    # V3
    "Ridge_v3"      : 0.16165,
    "Lasso_v3"      : 0.14929,
    "ElasticNet_v3" : 0.16090,
    "HistGBR_v3"    : 0.06193,
    "XGBoost_v3"    : 0.05881,
    "LightGBM_v3"   : 0.05968,
    "CatBoost_v3"   : 0.06080,

    # V4
    "XGBoost_v4"  : 0.05918,
    "LightGBM_v4" : 0.05958,
    "CatBoost_v4" : 0.06043,
}


all_cols   = []          # [(col_name, tag)]
all_scores = {}          # {col_name: cv_score}

for tag, (oof_df, _) in versions.items():
    for col in oof_df.columns:
        key = col if tag == "v1" else f"{col}_{tag}"
        all_cols.append((col, tag))
        all_scores[(col, tag)] = cv_score.get(key, np.inf)

# rank globally by CV score (ascending = better)
sorted_cols = sorted(all_cols, key=lambda x: all_scores[x])

top3 = sorted_cols[:3]
top5 = sorted_cols[:5]

print("Global top-3 columns :", top3)
print("Global top-5 columns :", top5)

# helper – given a version, keep only cols that are in the wanted list
def filter_version(tag, wanted):
    oof_df, test_df = versions[tag]
    keep = [c for c in oof_df.columns if (c, tag) in wanted]
    return oof_df[keep], test_df[keep]


combos = []

for subset_name, wanted in [("top3", top3), ("top5", top5)]:
    # build one big oof / test matrix that contains ONLY the wanted columns
    oof_parts, test_parts = [], []
    for tag in versions:
        oof_f, test_f = filter_version(tag, wanted)
        oof_parts.append(oof_f); test_parts.append(test_f)

    oof_all  = pd.concat(oof_parts,  axis=1)
    test_all = pd.concat(test_parts, axis=1)

    tag = f"ALL_{subset_name}"           # single “combo” name
    combos.append((tag, oof_all, test_all))

print(f"Total combos: {len(combos)}  ({[c[0] for c in combos]})")


def inv_lb_weights(cols):
    w = np.ones(len(cols))
    for i, c in enumerate(cols):
        base = c.split("_")[0]
        # search every version’s LB score; fall back to 1
        for key in lb_scores:
            if key.startswith(base):
                w[i] = 1.0 / lb_scores[key]; break
    return w / w.sum()

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
def ridge_blend(Xoof, Xtst, y):
    sc = StandardScaler()
    Xs, Xt = sc.fit_transform(Xoof), sc.transform(Xtst)
    mdl = Ridge(alpha=1.0, random_state=42).fit(Xs, y)
    return mdl.predict(Xs), mdl.predict(Xt)

def hill_climb(Xoof, y, iters=2000):
    best, best_w = 1e9, None
    X = Xoof.values
    for _ in range(iters):
        w = np.random.dirichlet(np.ones(X.shape[1]))
        s = rmsle(y, X @ w)
        if s < best: best, best_w = s, w
    return best_w

from scipy.optimize import minimize
def opt_slsqp(Xoof, y):
    d = Xoof.shape[1]
    f = lambda w: rmsle(y, Xoof.values @ w)
    con = {'type': 'eq', 'fun': lambda w: 1 - w.sum()}
    res = minimize(f, np.full(d, 1/d), bounds=[(0,1)]*d, constraints=con)
    return res.x


results, preds = {}, {}

for tag, oof_df, test_df in combos:
    print(f"\n▶ {tag}: {oof_df.shape[1]} columns")

    cols   = oof_df.columns
    Xoof   = oof_df.values
    Xtst   = test_df.values

    # mean
    po, pt = Xoof.mean(1), Xtst.mean(1)
    results[(tag, "mean")]  = rmsle(y_true, po)
    preds[(tag, "mean")]    = pt

    # weighted mean (inverse LB)
    w      = inv_lb_weights(cols)
    po, pt = (Xoof * w).sum(1), (Xtst * w).sum(1)
    results[(tag, "wmean")] = rmsle(y_true, po)
    preds[(tag, "wmean")]   = pt

    # rank average
    po = oof_df.rank(axis=1).mean(1)
    pt = test_df.rank(axis=1).mean(1)
    scale_o, scale_t = oof_df.max(1).mean(), test_df.max(1).mean()
    results[(tag, "rank")]  = rmsle(y_true, po / po.max() * scale_o)
    preds[(tag, "rank")]    = pt / pt.max() * scale_t

    # ridge
    po, pt = ridge_blend(oof_df, test_df, y_true)
    results[(tag, "ridge")] = rmsle(y_true, po)
    preds[(tag, "ridge")]   = pt

    # hill-climb
    w      = hill_climb(oof_df, y_true)
    results[(tag, "hill")]  = rmsle(y_true, Xoof @ w)
    preds[(tag, "hill")]    = Xtst @ w

    # SLSQP optimal
    w      = opt_slsqp(oof_df, y_true)
    results[(tag, "opt")]   = rmsle(y_true, Xoof @ w)
    preds[(tag, "opt")]     = Xtst @ w


res_df = (pd.Series(results, name="RMSLE")
            .sort_values()
            .reset_index()
            .rename(columns={"level_0": "Combo", "level_1": "Method"}))

plt.figure(figsize=(10, 0.5 * len(res_df) + 2))
plt.barh(res_df.index, res_df["RMSLE"], color="skyblue")
plt.yticks(res_df.index,
           [f"{c} | {m}" for c, m in res_df[["Combo", "Method"]].values])
plt.gca().invert_yaxis()
plt.title("OOF RMSLE – blends on global top-3 / top-5 models")
plt.xlabel("RMSLE")
for i, v in enumerate(res_df["RMSLE"]):
    plt.text(v + 0.00005, i, f"{v:.5f}", va="center")
plt.tight_layout()
plt.show()


out_dir = Path("submissions"); out_dir.mkdir(exist_ok=True)
for (combo, method), vec in preds.items():
    fn = out_dir / f"sub_{combo}_{method}.csv"
    pd.DataFrame({"id": ids, "Calories": vec}).to_csv(fn, index=False)
    print(f"Saved {fn.name:40s}  –  {len(vec):,} rows")




