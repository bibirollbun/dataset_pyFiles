# === BOOT / LOAD ===
import os, numpy as np, pandas as pd
IS_COMMIT = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Batch"  # True when Save&RunAll

DATA = "/kaggle/input/playground-series-s5e10"
SEED = 42
RISK_Q = 0.90
BENEFIT_TP, COST_FP = 5.0, 1.0

train = pd.read_csv(f"{DATA}/train.csv")
test  = pd.read_csv(f"{DATA}/test.csv")
sub   = pd.read_csv(f"{DATA}/sample_submission.csv")

def to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if not np.issubdtype(df[c].dtype, np.number):
            df[c] = df[c].astype(str)
    return df
print("Loaded:", train.shape, test.shape)



# === POLICY GRID (uses `train` loaded in Cell0) ===
df = train.copy()
y_high = df["accident_risk"] >= df["accident_risk"].quantile(RISK_Q)
base_p = y_high.mean()

gate_light   = (df["lighting"] != "daylight")
gate_weather = df["weather"].isin(["fog","rainy"])
gate = (gate_light | gate_weather)  # both_or

speed_grid = sorted(df["speed_limit"].unique().tolist())
curv_q_grid = [round(q,2) for q in np.arange(0.50, 0.96, 0.05)]

rows = []
for sl in speed_grid:
    for cq in curv_q_grid:
        ct  = df["curvature"].quantile(cq)
        pol = (df["speed_limit"] >= sl) & (df["curvature"] >= ct) & gate
        TP  = int((y_high & pol).sum()); FP = int((~y_high & pol).sum())
        cov = pol.mean()
        prec = TP / max(TP+FP, 1); rec = TP / y_high.sum()
        lift = prec / base_p if base_p>0 else np.nan
        net  = BENEFIT_TP*TP - COST_FP*FP
        rows.append((sl, cq, ct, cov, prec, rec, lift, net, TP, FP))

res = pd.DataFrame(rows, columns=[
    "speed_thr","curv_q","curv_thr","coverage","precision","recall","lift","net_value","TP","FP"
]).sort_values(["net_value","precision","lift"], ascending=[False,False,False])

res.head(10)
best = res.iloc[0]
print(f"[BEST] speed â‰¥ {best.speed_thr}, curvature â‰¥ {best.curv_thr:.2f} (q={best.curv_q})",
      f"â†’ cov {best.coverage:.2%}, prec {best.precision:.2%}, rec {best.recall:.2%},",
      f"lift {best.lift:.2f}, net {best.net_value:.0f}, TP {int(best.TP)}, FP {int(best.FP)}")



import numpy as np, pandas as pd

# -------- ì„¤ì •(ì›�í•˜ë©´ ë°”ê¿”) ----------
BENEFIT_TP = 5.0      # ì �ì¤‘ 1ê±´ ì�´ì�µ
COST_FP    = 1.0      # í—›ê°œì�… 1ê±´ ë¹„ìš©
RISK_Q     = 0.90     # 'ê³ ìœ„í—˜' ì •ì�˜ = ìƒ�ìœ„ 10%
GATE_MODE  = "both_or"  # ["none","light","weather","both_or","both_and"]
# -------------------------------------

df = train.copy()
y_high = df["accident_risk"] >= df["accident_risk"].quantile(RISK_Q)
base_p = y_high.mean()

# ê²Œì�´íŠ¸ ì •ì�˜
gate_all     = pd.Series(True, index=df.index)
gate_light   = (df["lighting"] != "daylight")
gate_weather = df["weather"].isin(["fog","rainy"])
gates = {
    "none": gate_all,
    "light": gate_light,
    "weather": gate_weather,
    "both_or": (gate_light | gate_weather),
    "both_and": (gate_light & gate_weather),
}
gate = gates[GATE_MODE]

# íƒ�ìƒ‰ ê²©ì��
speed_grid = sorted(df["speed_limit"].unique().tolist())
curv_q_grid = [round(q,2) for q in np.arange(0.50, 0.96, 0.05)]

rows = []
for sl in speed_grid:
    for cq in curv_q_grid:
        ct  = df["curvature"].quantile(cq)
        pol = (df["speed_limit"] >= sl) & (df["curvature"] >= ct) & gate
        TP  = int((y_high & pol).sum())
        FP  = int((~y_high & pol).sum())
        cov = pol.mean()
        prec = TP / max(TP+FP, 1)
        rec  = TP / y_high.sum()
        lift = (prec / base_p) if base_p > 0 else np.nan
        net  = BENEFIT_TP * TP - COST_FP * FP
        rows.append((sl, cq, ct, cov, prec, rec, lift, net, TP, FP))

res = pd.DataFrame(rows, columns=[
    "speed_thr","curv_q","curv_thr","coverage","precision","recall","lift","net_value","TP","FP"
])

# â�œ ë„¤ ìš°ì„ ìˆœìœ„ ì •ë ¬: net_value â†’ precision â†’ lift (ëª¨ë‘� ë‚´ë¦¼ì°¨ìˆœ)
res_best = res.sort_values(["net_value","precision","lift"], ascending=[False, False, False]).head(10)
display(res_best)

best = res_best.iloc[0]
print(
    f"[ì¶”ì²œ ì •ì±… @ {GATE_MODE}] speed â‰¥ {best.speed_thr}, curvature â‰¥ {best.curv_thr:.2f} (q={best.curv_q}) "
    f"â†’ coverage {best.coverage:.2%}, precision {best.precision:.2%}, "
    f"recall {best.recall:.2%}, lift {best.lift:.2f}, net {best.net_value:.0f}, "
    f"TP {int(best.TP)}, FP {int(best.FP)}"
)



# === Reproducibility & Config ===
SEED = 42
RISK_Q = 0.90                    # 'ê³ ìœ„í—˜' ì •ì�˜(ìƒ�ìœ„ 10%)
BENEFIT_TP, COST_FP = 5.0, 1.0   # ìˆœì�´ì�µ ê°€ì •
GATE_MODE = "both_or"            # ["none","light","weather","both_or","both_and"]
MODEL = "CatBoost(depth=8, lr=0.07, iters=1200)"
print(dict(SEED=SEED, RISK_Q=RISK_Q, BENEFIT_TP=BENEFIT_TP, COST_FP=COST_FP, GATE_MODE=GATE_MODE, MODEL=MODEL))



import pandas as pd, numpy as np
from pathlib import Path

DATA = Path("/kaggle/input/playground-series-s5e10")
train = pd.read_csv(DATA/"train.csv")
test  = pd.read_csv(DATA/"test.csv")
sub   = pd.read_csv(DATA/"sample_submission.csv")

target = "accident_risk"
train.head(3), train.shape, test.shape



desc = train.describe(include="all").T
nunique = train.nunique().sort_values(ascending=False).to_frame("nunique")
target_desc = train[target].describe()
train[target].hist(bins=40)
target_desc, nunique.head(10)



import pandas as pd, matplotlib.pyplot as plt, seaborn as sns

def target_heatmap(df, row, col, y='accident_risk'):
    pt = df.pivot_table(values=y, index=row, columns=col, aggfunc='mean')
    display(pt)  # ìˆ˜ì¹˜í‘œ ë¨¼ì €
    plt.figure(figsize=(6,4))
    sns.heatmap(pt, annot=True, fmt=".3f")
    plt.title(f"{y} mean by {row} Ã— {col}")
    plt.show()

# ë²”ì£¼Ã—ë²”ì£¼ (ì˜ˆ: ë‚ ì”¨Ã—ì¡°ëª…)
target_heatmap(train, 'weather', 'lighting')

# ìˆ˜ì¹˜Ã—ë²”ì£¼ (ì˜ˆ: ì œí•œì†�ë�„Ã—ì‹œê°„ëŒ€) â†’ ìˆ˜ì¹˜ëŠ” êµ¬ê°„ìœ¼ë¡œ ë‚˜ëˆ ì„œ
train['speed_bin'] = pd.qcut(train['speed_limit'], q=6, duplicates='drop')
target_heatmap(train, 'speed_bin', 'time_of_day')



for w in train['weather'].unique():
    target_heatmap(train[train['weather']==w], 'time_of_day', 'lighting')



import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

y = train['accident_risk'].values
X = train.drop(columns=['accident_risk'])

# 1) Interval íƒ€ì�…ì�„ ë¬¸ì��ì—´ë¡œ ë°”ê¾¸ê³  ë²”ì£¼í˜•ìœ¼ë¡œ í�¸ì�…
for c in X.columns:
    if pd.api.types.is_interval_dtype(X[c]):
        X[c] = X[c].astype(str)

# 2) ë²”ì£¼í˜• ê°�ì§€: object + category ë‘˜ ë‹¤
cat = X.select_dtypes(include=['object','category']).columns.tolist()
num = X.select_dtypes(include=[np.number]).columns.tolist()

pre = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat),
    ('num', 'passthrough', num)
])

tree = DecisionTreeRegressor(max_depth=3, random_state=42)
pipe = Pipeline([('pre', pre), ('model', tree)]).fit(X, y)

print(export_text(pipe.named_steps['model'],
                  feature_names=list(pipe.named_steps['pre'].get_feature_names_out())[:80]))



import numpy as np, pandas as pd
from pandas.api.types import (
    is_numeric_dtype, is_categorical_dtype, is_object_dtype,
    is_interval_dtype, is_bool_dtype, is_datetime64_any_dtype
)

target = "accident_risk"
y = train[target].values
X = train.drop(columns=[target]).copy()
X_test = test.copy()

# (ê¶Œì�¥) ì�˜ë¯¸ ì—†ëŠ” ì‹�ë³„ì�� ì œê±°
for c in ["id"]:
    if c in X.columns: 
        X.drop(columns=[c], inplace=True)
        if c in X_test.columns: X_test.drop(columns=[c], inplace=True)

def coerce_non_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        # 1) êµ¬ê°„í˜•(Interval) â†’ ë¬¸ì��ì—´
        if is_interval_dtype(df[c]):
            df[c] = df[c].astype(str)
        # 2) ì¹´í…Œê³ ë¦¬(Interval ì¹´í…Œê³ ë¦¬ í�¬í•¨) â†’ ë¬¸ì��ì—´
        elif is_categorical_dtype(df[c]):
            df[c] = df[c].astype(str)
        # 3) ì˜¤ë¸Œì �íŠ¸ â†’ ë¬¸ì��ì—´(ê·¸ëŒ€ë¡œ)
        elif is_object_dtype(df[c]):
            df[c] = df[c].astype(str)
        # 4) ë¶ˆë¦¬ì–¸ â†’ 0/1 (ë˜�ëŠ” ë¬¸ì��ì—´ë¡œ ë°”ê¿”ë�„ ë�¨)
        elif is_bool_dtype(df[c]):
            df[c] = df[c].astype(int)
        # 5) ë‚ ì§œí˜•(ì�ˆë‹¤ë©´) â†’ ë¬¸ì��ì—´(ê°„ë‹¨í•˜ê²Œ)  â€» í•„ìš” ì‹œ ì—°/ì›”/ìš”ì�¼ë¡œ íŒŒìƒ� ê¶Œì�¥
        elif is_datetime64_any_dtype(df[c]):
            df[c] = df[c].astype(str)
        # ë‚˜ë¨¸ì§€ëŠ” ìˆ«ì�� ê·¸ëŒ€ë¡œ ì‚¬ìš©
    return df

X = coerce_non_numeric(X)
X_test = coerce_non_numeric(X_test)

# CatBoostì—� ë„˜ê¸¸ ë²”ì£¼í˜• ì�¸ë�±ìŠ¤(ë¬¸ì��ì—´ ì»¬ëŸ¼)
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
cat_idx  = [X.columns.get_loc(c) for c in cat_cols]

X.dtypes.head(), len(cat_cols)



from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

params = dict(loss_function="RMSE", depth=8, iterations=1200,
              learning_rate=0.07, random_seed=42, verbose=False)

m = CatBoostRegressor(**params)
m.fit(Pool(X, y, cat_features=cat_idx))
print("Fitted:", m.is_fitted())

# (ì„ íƒ�) OOF í™•ì�¸í•˜ê³  ì‹¶ìœ¼ë©´ 5-foldë¡œ:
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# oof = np.zeros(len(X))
# for tr, va in kf.split(X, y):
#     mm = CatBoostRegressor(**params)
#     mm.fit(Pool(X.iloc[tr], y[tr], cat_features=cat_idx),
#            eval_set=Pool(X.iloc[va], y[va], cat_features=cat_idx),
#            early_stopping_rounds=200, verbose=False)
#     oof[va] = mm.predict(Pool(X.iloc[va], cat_features=cat_idx))
# print("OOF RMSE:", mean_squared_error(y, oof, squared=False))



import shap
explainer = shap.TreeExplainer(m)
sv = explainer.shap_values(Pool(X, cat_features=cat_idx))

# ì „ ë³€ìˆ˜ ìš”ì•½(ì¤‘ìš”ë�„ + ë°©í–¥ì„±)
shap.summary_plot(sv, X)

# ìƒ�í˜¸ì�‘ìš© ì˜ˆì‹œ: ì†�ë�„ Ã— ì¡°ëª…
shap.dependence_plot('speed_limit', sv, X, interaction_index='lighting')



import numpy as np, pandas as pd

df = train.copy()
y = df["accident_risk"].values

# ---[ì •ì±…/í�‰ê°€ ê°€ì •ê°’]---------------------------------------
SL_THR   = 60        # ì†�ë�„ ì�„ê³„ê°’ (ì˜ˆ: 60mph ì�´ìƒ�)
CURV_Q   = 0.80      # ê³¡ë¥  ìƒ�ìœ„ ëª‡ ë¶„ìœ„ ì�´ìƒ�ì�„ 'ë†’ë‹¤'ë¡œ ë³¼ì§€ (ì˜ˆ: ìƒ�ìœ„ 20%)
RISK_Q   = 0.90      # 'ê³ ìœ„í—˜' ë ˆì�´ë¸” = íƒ€ê¹ƒ ìƒ�ìœ„ 10% (ë¶„ë¥˜ìš©)
BENEFIT_TP = 5.0     # TPR 1ê±´(ê°œì�…ì�´ í•„ìš”í•œ ê³³ì�„ ì �ì¤‘)ë‹¹ ì�´ì�µ
COST_FP    = 1.0     # FPR 1ê±´(ë¶ˆí•„ìš” ê°œì�…)ë‹¹ ë¹„ìš©
# ------------------------------------------------------------

curv_thr = df["curvature"].quantile(CURV_Q)
policy = (df["speed_limit"] >= SL_THR) & (df["curvature"] >= curv_thr)

# 'ê³ ìœ„í—˜' ì •ì�˜(ë ˆì�´ë¸”) : ìƒ�ìœ„ RISK_Q ë¶„ìœ„
y_high = df["accident_risk"] >= df["accident_risk"].quantile(RISK_Q)

# ì§€í‘œê³„ì‚°
coverage   = policy.mean()                                     # ê°œì�… ë¹„ìœ¨(ì»¤ë²„ë¦¬ì§€)
baseline_p = y_high.mean()                                     # ì „ì²´ ê³ ìœ„í—˜ ë¹„ìœ¨
precision  = (y_high & policy).sum() / max(policy.sum(), 1)    # ì •ì±… ì �ìš© êµ¬ê°„ ë‚´ ê³ ìœ„í—˜ ë¹„ìœ¨
recall     = (y_high & policy).sum() / y_high.sum()            # ê³ ìœ„í—˜ ì¤‘ ì–¼ë§ˆë‚˜ ì�¡ì•˜ë‚˜
lift       = precision / baseline_p if baseline_p>0 else np.nan
net_value  = BENEFIT_TP * (y_high & policy).sum() - COST_FP * (policy & ~y_high).sum()

summary = pd.DataFrame({
    "speed_thr":[SL_THR], "curvature_quantile":[CURV_Q], "curv_thr":[curv_thr],
    "coverage":[coverage], "baseline_high_rate":[baseline_p],
    "precision":[precision], "recall":[recall], "lift":[lift], "net_value":[net_value]
})
summary



sl_grid   = [50,55,60,65,70]
cq_grid   = [0.6,0.7,0.8,0.9]
rows = []
q_risk = df["accident_risk"].quantile(RISK_Q)
y_high = df["accident_risk"] >= q_risk
base_p = y_high.mean()

for sl in sl_grid:
    for cq in cq_grid:
        ct = df["curvature"].quantile(cq)
        pol = (df["speed_limit"]>=sl) & (df["curvature"]>=ct)
        if pol.sum()==0: 
            rows.append((sl,cq,ct,0,base_p,0,0,0,0)); continue
        prec = (y_high & pol).sum()/pol.sum()
        rec  = (y_high & pol).sum()/y_high.sum()
        cov  = pol.mean()
        lift = prec/base_p if base_p>0 else np.nan
        net  = BENEFIT_TP*(y_high & pol).sum() - COST_FP*(pol & ~y_high).sum()
        rows.append((sl,cq,ct,cov,base_p,prec,rec,lift,net))

res = pd.DataFrame(rows, columns=["speed_thr","curv_q","curv_thr","coverage","baseline_high_rate","precision","recall","lift","net_value"])
res.sort_values(["net_value","lift","precision"], ascending=False).head(10)



BENEFIT_TP = 5.0   # ì �ì¤‘ 1ê±´ ì�´ì�µ
COST_FP    = 1.0   # í—›ê°œì�… 1ê±´ ë¹„ìš©
RISK_Q     = 0.90  # 'ê³ ìœ„í—˜' ë�¼ë²¨: ìƒ�ìœ„ 10%

import numpy as np, pandas as pd
df = train.copy()
y_high = df["accident_risk"] >= df["accident_risk"].quantile(RISK_Q)
base_p = y_high.mean()

def score_policy(speed_thr, curv_q):
    ct  = df["curvature"].quantile(curv_q)
    pol = (df["speed_limit"]>=speed_thr) & (df["curvature"]>=ct)
    TP  = int((y_high & pol).sum())
    FP  = int((~y_high & pol).sum())
    prec = TP / max(TP+FP, 1)
    rec  = TP / y_high.sum()
    cov  = pol.mean()
    lift = prec / base_p
    net  = BENEFIT_TP*TP - COST_FP*FP
    return speed_thr, curv_q, ct, cov, prec, rec, lift, net, TP, FP

speed_grid = [50,55,60,65,70]
curv_grid  = [round(q,2) for q in np.arange(0.55, 0.96, 0.05)]
rows = [score_policy(s,q) for s in speed_grid for q in curv_grid]

res = pd.DataFrame(rows, columns=[
    "speed_thr","curv_q","curv_thr","coverage","precision","recall","lift","net_value","TP","FP"
])

# ğŸ”� ìš°ì„ ìˆœìœ„ëŒ€ë¡œ ì •ë ¬: net_value â†’ precision â†’ lift
res_best = res.sort_values(["net_value","precision","lift"],
                           ascending=[False, False, False]).head(10)
display(res_best)

best = res_best.iloc[0]
print(f"ì¶”ì²œ ì •ì±… â�œ speed â‰¥ {best.speed_thr}, curvature â‰¥ {best.curv_thr:.2f} (q={best.curv_q}) "
      f"â†’ coverage {best.coverage:.2%}, precision {best.precision:.2%}, "
      f"recall {best.recall:.2%}, lift {best.lift:.2f}, net {best.net_value:.0f}, "
      f"TP {int(best.TP)}, FP {int(best.FP)}")



sub = train[train['curvature'] >= train['curvature'].quantile(0.55)]
sub['speed_limit'].value_counts()



def score_with_gate(speed_thr, curv_q, gate):
    df = train.copy()
    y_high = df['accident_risk'] >= df['accident_risk'].quantile(0.90)
    base_p = y_high.mean()
    ct  = df['curvature'].quantile(curv_q)
    pol = (df['speed_limit']>=speed_thr) & (df['curvature']>=ct) & gate
    TP  = int((y_high & pol).sum()); FP = int((~y_high & pol).sum())
    prec = TP / max(TP+FP,1); rec = TP / y_high.sum(); cov = pol.mean()
    lift = prec / base_p; net = 5*TP - 1*FP
    return prec, rec, cov, lift, net, TP, FP

gate_light = (train['lighting']!='daylight')
score_with_gate(50, 0.55, gate_light)



gate_weather = train['weather'].isin(['fog','rainy'])
gate_both    = gate_light | gate_weather   # OR ê²Œì�´íŠ¸ (ê²½ê³ /ê°�ì†� ì •ì±…ì—� ì �í•©)

for name, g in {'light':gate_light,'weather':gate_weather,'both':gate_both}.items():
    p,r,c,l,n,TP,FP = score_with_gate(50, 0.55, g)
    print(name, 'â†’',
          f'precision {p:.3f}, recall {r:.3f}, coverage {c:.3f}, lift {l:.2f}, net {n:.0f}, TP {TP}, FP {FP}')



gate_and = (train['lighting']!='daylight') & train['weather'].isin(['fog','rainy'])
score_with_gate(50, 0.55, gate_and)  # precision, recall, coverage, lift, net, TP, FP



# === BULLETPROOF SUBMISSION ===
import numpy as np, pandas as pd, os
from catboost import CatBoostRegressor, Pool

DATA = "/kaggle/input/playground-series-s5e10"
train = pd.read_csv(f"{DATA}/train.csv")
test  = pd.read_csv(f"{DATA}/test.csv")
sub   = pd.read_csv(f"{DATA}/sample_submission.csv")

y = train["accident_risk"].values
X = train.drop(columns=["accident_risk"]).copy()
X_test = test.copy()

# 1) train/test ê³µí†µ ì¹¼ëŸ¼ë§Œ ì‚¬ìš© (íŒŒìƒ�ì¹¼ëŸ¼ ìœ ì�… ì°¨ë‹¨)
common_cols = sorted(set(X.columns) & set(X_test.columns))
X = X[common_cols].copy()
X_test = X_test[common_cols].copy()

# 2) 'ìˆ«ì��ê°€ ì•„ë‹Œ' ëª¨ë“  ì¹¼ëŸ¼ì�„ ë¬¸ì��ì—´(=ë²”ì£¼)ë¡œ í†µì�¼  â†’ Interval/Categorical/Bool/Datetime ëª¨ë‘� ì»¤ë²„
num_cols = X.select_dtypes(include=[np.number]).columns
non_num_cols = [c for c in X.columns if c not in num_cols]
if non_num_cols:
    X[non_num_cols] = X[non_num_cols].astype(str)
    X_test[non_num_cols] = X_test[non_num_cols].astype(str)

cat_idx = [X.columns.get_loc(c) for c in non_num_cols]

# 3) ê°€ë²¼ìš´ ëª¨ë�¸(ì»¤ë°‹ ì•ˆì •ì„±)
m = CatBoostRegressor(loss_function="RMSE", depth=8, iterations=800,
                      learning_rate=0.07, random_seed=42, verbose=False)
m.fit(Pool(X, y, cat_features=cat_idx))
pred = m.predict(Pool(X_test, cat_features=cat_idx)).clip(0,1)

# 4) ì €ì�¥ + ì¡´ì�¬ í™•ì�¸
out = "/kaggle/working/submission.csv"
sub[sub.columns[1]] = pred
sub.to_csv(out, index=False)
print("Saved:", out, "| cols:", len(common_cols), "non_numeric:", len(non_num_cols))
print("Working files:", os.listdir("/kaggle/working"))


