# ================== China Real Estate Demand — CPU Baseline (quiet, tuned v3.1) ==================
# - Seasonal baseline (lag12), expanding means (sector & sector×calendar-month)
# - Per-sector linear detrend residual features (incl. lag1/lag12)
# - Purged rolling-origin CV; class-weighted zero classifier; Huber regressor on log1p
# - Tiny grid over (zero_threshold, ML_weight, soft_floor); seasonal-informed floor
# - **Fix**: deduplicate feature names to avoid LightGBM "appears more than one time" error
# - Quiet logs; preserves exact test order -> submission.csv

import os, re, gc, time, numpy as np, pandas as pd
try:
    import psutil
    _PROC = psutil.Process(os.getpid())
    def _mem_mb(): return _PROC.memory_info().rss / (1024**2)
except Exception:
    def _mem_mb(): return 0.0

import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor

# ---------- tiny logger ----------
_T0 = time.time()
def log(msg):
    print(f"[{time.time()-_T0:7.1f}s | {int(_mem_mb()):6d} MB] {msg}", flush=True)

# ---------- detect comp dir ----------
DATA_DIR = "/kaggle/input"
cand = [d for d in os.listdir(DATA_DIR) if any(k in d.lower() for k in ("real","estate","demand","china"))]
COMP_DIR = os.path.join(DATA_DIR, cand[0]) if cand else DATA_DIR
log(f"COMP_DIR = {COMP_DIR}")

# ---------- helpers ----------
def parse_month(s):
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s_norm = re.sub(r"^(\d{4})[-_/\.]([A-Za-z]{3,})$", r"\1 \2", s)  # "2019-Apr" -> "2019 Apr"
    dt = pd.to_datetime(s_norm, format="mixed", errors="coerce")
    if pd.isna(dt):
        dt = pd.to_datetime(s, format="mixed", errors="coerce")
    return dt

def sector_id(s):
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else -1

def add_time_features(df):
    df["year"] = df["month_dt"].dt.year
    df["month_num"] = df["month_dt"].dt.month
    df["month_sin"] = np.sin(2*np.pi*df["month_num"]/12.0)
    df["month_cos"] = np.cos(2*np.pi*df["month_num"]/12.0)
    df["t_idx"] = df.groupby("sector_id").cumcount().astype(float)
    return df

def make_lags_concat(df, group_cols, value_cols, lags=(1,2,3,6,12), rolls=(3,6,12)):
    gb = df.groupby(group_cols, sort=False)
    newcols = {}
    for col in value_cols:
        if col not in df.columns: 
            continue
        for L in lags:
            newcols[f"{col}_lag{L}"] = gb[col].shift(L)
        s = gb[col].shift(1)  # shift(1) prevents leakage
        for W in rolls:
            newcols[f"{col}_rollmean{W}"] = s.rolling(W).mean()
            newcols[f"{col}_rollsum{W}"]  = s.rolling(W).sum()
            newcols[f"{col}_rollmed{W}"]  = s.rolling(W).median()
    return pd.concat([df, pd.DataFrame(newcols, index=df.index)], axis=1)

def add_expanding_means(df, tgt):
    df["exp_mean_sector"] = (
        df.groupby("sector_id", sort=False)[tgt]
          .apply(lambda s: s.shift(1).expanding(min_periods=6).mean())
          .values
    )
    df["exp_mean_seasonal"] = (
        df.groupby(["sector_id","month_num"], sort=False)[tgt]
          .apply(lambda s: s.shift(1).expanding(min_periods=2).mean())
          .values
    )
    return df

def add_sector_detrend_residual(df, tgt):
    resid = np.zeros(len(df), dtype=float)
    for sid, idx in df.groupby("sector_id").indices.items():
        s = df.loc[idx, [tgt, "t_idx", "_is_test"]].copy()
        tr = (s["_is_test"] == 0).values
        if tr.sum() >= 6 and s["t_idx"].nunique() > 1:
            X = np.vstack([np.ones(tr.sum()), s.loc[tr,"t_idx"].values]).T
            y = s.loc[tr, tgt].values
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_hat = beta[0] + beta[1]*s["t_idx"].values
            resid[idx] = s[tgt].values - y_hat
        else:
            resid[idx] = 0.0
    df["sector_resid"] = resid
    df["sector_resid_lag1"] = df.groupby("sector_id")["sector_resid"].shift(1)
    df["sector_resid_lag12"] = df.groupby("sector_id")["sector_resid"].shift(12)
    return df

def comp_metric(y_true, y_pred, return_parts=False):
    y_true = np.asarray(y_true, float)
    y_pred = np.clip(np.asarray(y_pred, float), 0, None)
    ape = np.zeros_like(y_true)
    mask0 = (y_true == 0)
    if np.any(~mask0):
        ape[~mask0] = np.abs(y_pred[~mask0] - y_true[~mask0]) / np.abs(y_true[~mask0])
    ape[mask0] = np.where(y_pred[mask0] == 0, 0.0, 10.0)
    fail_pct = (ape > 1.0).mean()
    if fail_pct > 0.30:
        return (0.0, fail_pct, np.nan, np.nan) if return_parts else 0.0
    good = (ape <= 1.0)
    if good.sum() == 0:
        return (0.0, fail_pct, np.nan, 0.0) if return_parts else 0.0
    mape_good = ape[good].mean()
    frac_good = good.mean()
    score = float(np.clip(1.0 - (mape_good / max(frac_good, 1e-9)), 0.0, 1.0))
    return (score, fail_pct, mape_good, frac_good) if return_parts else score

# ---------- robust loaders (singular/plural variants) ----------
def first_existing(rel_paths):
    for rp in rel_paths:
        p = os.path.join(COMP_DIR, "train", rp)
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

log("Loading CSVs...")
nh   = first_existing(["new_house_transactions.csv"])  # required
assert nh is not None, "missing train/new_house_transactions.csv"

nhn  = first_existing(["new_house_transactions_nearby_sectors.csv",
                       "new_house_transactions_nearby_sector.csv"])
poh  = first_existing(["pre_owned_house_transactions.csv"])
pohn = first_existing(["pre_owned_house_transactions_nearby_sectors.csv",
                       "pre_owned_house_transactions_nearby_sector.csv"])
land  = first_existing(["land_transactions.csv"])
landn = first_existing(["land_transactions_nearby_sectors.csv",
                        "land_transactions_nearby_sector.csv"])
poi   = first_existing(["sector_POI.csv"])
cidx  = first_existing(["city_indexes.csv"])
csi   = first_existing(["city_search_index.csv"])
test  = pd.read_csv(os.path.join(COMP_DIR, "test.csv"))
log("Loaded.")

def fix_keys(df):
    if df is None: return None
    if "month" in df.columns:
        df["month"] = df["month"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    if "sector" in df.columns:
        df["sector"] = df["sector"].astype(str).str.strip()
    return df

for t in (nh, nhn, poh, pohn, land, landn, poi, cidx, csi):
    fix_keys(t)

# Parse test id -> month, sector
def split_id(id_str):
    s = str(id_str)
    mo = re.match(r"(\d{4}\s*[A-Za-z]{3,}|[\d]{4}[-_/\.][A-Za-z]{3,})_sector\s+(\d+)", s)
    if mo:
        month_part = mo.group(1).replace("-", " ").replace("_", " ").replace(".", " ")
        return month_part.strip(), f"sector {int(mo.group(2))}"
    parts = s.split("_sector")
    sec = re.search(r"(\d+)", parts[1]).group(1) if len(parts[1:]) and re.search(r"\d+", parts[1]) else "-1"
    return parts[0], f"sector {sec}"

test["month"], test["sector"] = zip(*test["id"].map(split_id))
test["month"]  = test["month"].astype(str).str.strip()
test["sector"] = test["sector"].astype(str).str.strip()

# target alias (singular -> plural)
if "amount_new_house_transaction" in nh.columns and "amount_new_house_transactions" not in nh.columns:
    nh = nh.rename(columns={"amount_new_house_transaction":"amount_new_house_transactions"})
if nhn is not None:
    nhn = nhn.rename(columns={c: c.replace("amount_new_house_transaction","amount_new_house_transactions")
                              for c in nhn.columns})

# ---------- build full (month x sector) grid ----------
months = pd.Index(sorted(
    set(nh["month"]) | set(test["month"]) |
    (set(nhn["month"])  if nhn  is not None else set()) |
    (set(poh["month"])  if poh  is not None else set()) |
    (set(pohn["month"]) if pohn is not None else set()) |
    (set(land["month"]) if land is not None else set()) |
    (set(landn["month"])if landn is not None else set())
))
sectors = pd.Index(sorted(
    set(nh["sector"]) | set(test["sector"]) |
    (set(nhn["sector"])  if nhn  is not None else set()) |
    (set(poh["sector"])  if poh  is not None else set()) |
    (set(pohn["sector"]) if pohn is not None else set()) |
    (set(land["sector"]) if land is not None else set()) |
    (set(landn["sector"])if landn is not None else set()) |
    (set(poi["sector"])  if poi  is not None else set())
))
grid = pd.MultiIndex.from_product([months, sectors], names=["month","sector"]).to_frame(index=False)
log(f"Grid shape: {grid.shape}")

# ---------- merge ----------
log("Merging all tables...")
df = grid.merge(nh, on=["month","sector"], how="left")
if nhn  is not None: df = df.merge(nhn,  on=["month","sector"], how="left", suffixes=("","_nhnear"))
if poh  is not None: df = df.merge(poh,  on=["month","sector"], how="left", suffixes=("","_poh"))
if pohn is not None: df = df.merge(pohn, on=["month","sector"], how="left", suffixes=("","_pohn"))
if land is not None: df = df.merge(land, on=["month","sector"], how="left", suffixes=("","_land"))
if landn is not None: df = df.merge(landn,on=["month","sector"], how="left", suffixes=("","_landn"))
if poi  is not None: df = df.merge(poi,  on=["sector"],       how="left")
log(f"After merge: {df.shape}")

# parse time + sector id; sort for lags
log("Parsing months & sorting...")
df["month_dt"] = df["month"].map(parse_month)
if df["month_dt"].isna().any():
    mfix = pd.to_datetime(df.loc[df["month_dt"].isna(), "month"].str.replace("-", " "),
                          format="mixed", errors="coerce")
    df.loc[df["month_dt"].isna(), "month_dt"] = mfix
assert df["month_dt"].notna().all(), "Unparseable month values exist."
df["sector_id"] = df["sector"].map(sector_id)
df = df.sort_values(["sector_id","month_dt"]).reset_index(drop=True)

# city_search_index
if csi is not None and {"month","search_volume"}.issubset(csi.columns):
    log("Aggregating city_search_index...")
    csi_agg = csi.groupby("month", as_index=False).agg(total_search_volume=("search_volume","sum"))
    if "source" in csi.columns:
        src_piv = (csi.pivot_table(index="month", columns="source", values="search_volume", aggfunc="sum")
                   .add_prefix("search_src_").reset_index())
        csi_agg = csi_agg.merge(src_piv, on="month", how="left")
    df = df.merge(csi_agg, on="month", how="left")

# city_indexes yearly join
if cidx is not None:
    log("Joining city_indexes by year...")
    if "city_indicator_data_year" in cidx.columns:
        cidx = cidx.rename(columns={"city_indicator_data_year":"year"})
    keep = ["year"] + [c for c in cidx.columns if c != "year" and pd.api.types.is_numeric_dtype(cidx[c])]
    cidx = cidx[keep].copy()
    df["year"] = df["month_dt"].dt.year
    df = df.merge(cidx, on="year", how="left")

# fills
log("Filling NaNs...")
price_cols = [c for c in df.columns if "price_" in c]
for c in df.columns:
    if c in ("month","sector") or ("price_" in c): 
        continue
    if re.search(r"(^num_|^area_|^amount_|^construction_area$|^planned_building_area$|^transaction_amount$|_nearby_sectors$|_nearby_sector$)", c):
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(0.0)
if price_cols:
    df[price_cols] = df.groupby("sector_id")[price_cols].ffill().bfill()
    df[price_cols] = df[price_cols].fillna(0.0)

target_col = "amount_new_house_transactions"
df[target_col] = df[target_col].fillna(0.0)

# mark test rows
df = df.merge(test[["month","sector"]].assign(_is_test=1), on=["month","sector"], how="left")
df["_is_test"] = df["_is_test"].fillna(0).astype(int)
is_test = df["_is_test"].astype(bool).values

# ---------- features ----------
log("Adding time features & building lags...")
df = add_time_features(df)

value_for_lags = [
    target_col,
    "num_new_house_transactions","area_new_house_transactions","price_new_house_transactions",
    "amount_new_house_transactions_nearby_sectors","area_new_house_transactions_nearby_sectors",
    "num_new_house_transactions_nearby_sectors","price_new_house_transactions_nearby_sectors",
    "amount_pre_owned_house_transactions","area_pre_owned_house_transactions","num_pre_owned_house_transactions",
    "price_pre_owned_house_transactions",
    "amount_pre_owned_house_transactions_nearby_sectors","area_pre_owned_house_transactions_nearby_sectors",
    "num_pre_owned_house_transactions_nearby_sectors","price_pre_owned_house_transactions_nearby_sectors",
    "transaction_amount","planned_building_area","construction_area","num_land_transactions",
    "transaction_amount_nearby_sectors","planned_building_area_nearby_sectors",
    "construction_area_nearby_sectors","num_land_transactions_nearby_sectors",
    "total_search_volume"
] + [c for c in df.columns if c.startswith("search_src_")]
value_for_lags = [c for c in value_for_lags if c in df.columns]

df = make_lags_concat(df, ["sector_id"], value_for_lags, lags=(1,2,3,6,12), rolls=(3,6,12))
log(f"Lagged df shape: {df.shape}")

# naive baselines
naive1  = df.get(f"{target_col}_lag1",  pd.Series(0.0, index=df.index))
naive12 = df.get(f"{target_col}_lag12", pd.Series(np.nan, index=df.index))
df["naive_pred"]    = naive1.fillna(naive12).fillna(0.0)
df["seasonal_pred"] = naive12.fillna(df["naive_pred"])

# expanding means & sector detrend residuals
df = add_expanding_means(df, target_col)
df = add_sector_detrend_residual(df, target_col)

# matrices
feat_cols = (
    ["sector_id","year","month_num","month_sin","month_cos","t_idx",
     "naive_pred","seasonal_pred","exp_mean_sector","exp_mean_seasonal",
     "sector_resid","sector_resid_lag1","sector_resid_lag12"] +
    [c for c in df.columns if re.search(r"(lag|rollmean|rollsum|rollmed)", c)]
)
# --------- DEDUP to avoid LightGBM duplicate feature-name error ---------
feat_cols = list(dict.fromkeys(feat_cols))  # preserves order, drops dups

X_all = df[feat_cols].copy()
y_all = df[target_col].values
X_train = X_all[~is_test].reset_index(drop=True)
y_train = y_all[~is_test]
X_test  = X_all[is_test].reset_index(drop=True)
log(f"Train X: {X_train.shape}, Test X: {X_test.shape}, y: {y_train.shape}")

# ---------- rolling-origin folds with 1-month purge ----------
train_months = np.sort(df.loc[~is_test, "month_dt"].dropna().unique())
n_folds = min(5, max(3, len(train_months)//6))
val_ix  = np.linspace(int(len(train_months)*0.6), len(train_months)-1, n_folds, dtype=int)

folds = []
for cp in val_ix:
    val_m   = pd.Timestamp(train_months[cp])
    purge_m = val_m - pd.offsets.MonthBegin(1)
    tr_mask = (df.loc[~is_test, "month_dt"] < purge_m)
    va_mask = (df.loc[~is_test, "month_dt"] == val_m)
    tr_idx  = np.where(tr_mask.values)[0]
    va_idx  = np.where(va_mask.values)[0]
    if len(tr_idx) and len(va_idx):
        folds.append((tr_idx, va_idx))
log(f"Using {len(folds)} folds (purged). Valid months: {[str(pd.Timestamp(train_months[i]).date()) for i in val_ix]}")

# ---------- two-stage models (quiet) ----------
y_zero = (y_train == 0).astype(int)
pos_w  = (len(y_zero) - y_zero.sum()) / max(y_zero.sum(), 1)

cls_params = dict(
    objective="binary", learning_rate=0.05, num_leaves=64,
    min_child_samples=96, subsample=0.9, subsample_freq=1,
    colsample_bytree=0.9, reg_lambda=2.0,
    n_estimators=12000, n_jobs=-1, random_state=42,
    verbosity=-1, force_col_wise=True, is_unbalance=False, scale_pos_weight=pos_w
)

reg_params = dict(
    objective="huber", alpha=0.9,
    learning_rate=0.05, num_leaves=96,
    min_child_samples=96, subsample=0.9, subsample_freq=1,
    colsample_bytree=0.9, reg_lambda=2.0,
    n_estimators=12000, n_jobs=-1, random_state=42,
    verbosity=-1, force_col_wise=True
)

thr_grid   = np.round(np.linspace(0.05, 0.55, 11), 3)
w_grid     = np.round(np.linspace(0.40, 0.90, 11), 2)
floor_grid = [0.05, 0.10, 0.15]

def cv_score_for(thr_zero, w_ml, floor_ratio):
    oof = np.zeros_like(y_train, float)
    for (tr, va) in folds:
        X_tr, X_va = X_train.iloc[tr], X_train.iloc[va]
        y_tr, y_va = y_train[tr], y_train[va]
        z_tr, z_va = y_zero[tr], y_zero[va]

        clf = LGBMClassifier(**cls_params)
        clf.fit(X_tr, z_tr, eval_set=[(X_va, z_va)], eval_metric="auc",
                callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)])
        pz = clf.predict_proba(X_va)[:,1]

        reg = LGBMRegressor(**reg_params)
        pos = (y_tr > 0)
        if pos.sum():
            reg.fit(X_tr.iloc[pos], np.log1p(y_tr[pos]),
                    eval_set=[(X_va, np.log1p(np.maximum(y_va,0)))], eval_metric="l2",
                    callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)])
            pred_ml = np.expm1(reg.predict(X_va))
        else:
            pred_ml = np.zeros(len(X_va))

        pred_nv   = X_va["naive_pred"].values
        pred_seas = X_va["seasonal_pred"].values
        base      = 0.5*pred_nv + 0.5*pred_seas
        pred      = w_ml*pred_ml + (1.0 - w_ml)*base

        mask_zero = (pz >= thr_zero)
        floor     = floor_ratio * np.maximum(pred_seas, pred_nv)
        pred[mask_zero] = np.maximum(pred[mask_zero], floor[mask_zero])

        oof[va] = np.clip(pred, 0, None)

    score, fail_pct, mape_good, frac_good = comp_metric(y_train, oof, return_parts=True)
    zero_rate = (oof == 0).mean()
    log(f"CV thr={thr_zero:.2f} w={w_ml:.2f} floor={floor_ratio:.2f} -> "
        f"score={score:.6f} | fail>{1.0:.0f}: {fail_pct*100:.1f}% | good={frac_good*100:.1f}% | zeros={zero_rate*100:.1f}%")
    return score

log("Starting grid search (thr_zero, w_ml, floor_ratio)...")
best = (-1, None, None, None)
for thr in thr_grid:
    for w in w_grid:
        for fl in floor_grid:
            sc = cv_score_for(thr, w, fl)
            if sc > best[0]:
                best = (sc, thr, w, fl)
best_score, best_thr, best_w, best_floor = best
log(f"Best CV = {best_score:.6f} @ thr={best_thr:.2f}, w={best_w:.2f}, floor={best_floor:.2f}")

# ---------- train full & predict ----------
log("Training full classifier...")
clf_full = LGBMClassifier(**cls_params)
ref_idx = folds[-1][1] if len(folds) else np.arange(len(X_train))[:max(500, len(X_train)//10)]
clf_full.fit(
    X_train, y_zero,
    eval_set=[(X_train.iloc[ref_idx], y_zero[ref_idx])],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)]
)
pz_test = clf_full.predict_proba(X_test)[:,1]

log("Training full regressor (pos-only, Huber)...")
reg_full = LGBMRegressor(**reg_params)
pos_full = (y_train > 0)
if pos_full.sum() > 0:
    reg_full.fit(
        X_train.iloc[pos_full], np.log1p(y_train[pos_full]),
        eval_set=[(X_train.iloc[ref_idx], np.log1p(np.maximum(y_train[ref_idx],0)))],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)]
    )
    pred_test_ml = np.expm1(reg_full.predict(X_test))
else:
    pred_test_ml = np.zeros(len(X_test), dtype=float)

pred_nv   = X_test["naive_pred"].values
pred_seas = X_test["seasonal_pred"].values
base = 0.5*pred_nv + 0.5*pred_seas
pred_test = best_w*pred_test_ml + (1.0 - best_w)*base

mask_zero = (pz_test >= best_thr)
floor = best_floor * np.maximum(pred_seas, pred_nv)
pred_test[mask_zero] = np.maximum(pred_test[mask_zero], floor[mask_zero])

pred_test = np.clip(pred_test, 0, None)
log(f"Test zeros after soft-zeroing: {(pred_test==0).mean()*100:.1f}%")
log("Inference done.")

# ---------- save in EXACT test order ----------
log("Writing submission...")
test_view = df.loc[is_test, ["month","sector"]].reset_index(drop=True)
pred_df = pd.DataFrame({"month": test_view["month"], "sector": test_view["sector"], "pred": pred_test})
sub = test[["id","month","sector"]].merge(pred_df, on=["month","sector"], how="left")
assert sub["pred"].isna().sum() == 0, "Some test ids did not find a prediction!"
sub = sub[["id"]].assign(new_house_transaction_amount=sub["pred"].clip(lower=0))
sub.to_csv("submission.csv", index=False)
log("Saved -> submission.csv")

# sanity checks & quick distribution
test_csv = pd.read_csv(os.path.join(COMP_DIR, "test.csv"))
sub_csv = pd.read_csv("submission.csv")
assert list(sub_csv.columns) == ["id","new_house_transaction_amount"]
assert len(sub_csv) == len(test_csv)
assert sub_csv["id"].equals(test_csv["id"])
vals = sub_csv["new_house_transaction_amount"].astype(float).values
log(f"Submission stats | zeros={np.mean(vals==0)*100:.1f}% | min={vals.min():.2f} | p50={np.median(vals):.2f} | p90={np.quantile(vals,0.90):.2f} | max={vals.max():.2f}")

del clf_full, reg_full
gc.collect()
log("Done.")





