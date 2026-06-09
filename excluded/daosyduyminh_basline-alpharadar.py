# =========================================================
# Alpha Radar: Solana Sprint — CatBoost Baseline (FIXED) + EDA + Submission
# =========================================================
import os, gc, random
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED); np.random.seed(SEED)

COMP_DIR = Path("/kaggle/input/alpha-radar-solana-sprint")
FULL_DIR = Path("/kaggle/input/pumpfun-30s-september-2025")
WORK_DIR = Path("/kaggle/working"); WORK_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TOKENS_GDRIVE_ID = "1EsqpZXPBU-6m0djDmccCrtUX07jV2fHA"
TARGET_TOKENS_CSV = WORK_DIR / "target_tokens.csv"

CHUNKSIZE = 250_000

# ---- deps
try:
    from catboost import CatBoostClassifier, Pool
except:
    !pip -q install --no-input catboost
    from catboost import CatBoostClassifier, Pool

try:
    import gdown; HAVE_GDOWN = True
except:
    try:
        !pip -q install --no-input gdown
        import gdown; HAVE_GDOWN = True
    except:
        HAVE_GDOWN = False

print(f"CatBoost OK | gdown: {HAVE_GDOWN}")

# -----------------------
# 1) Column setup
# -----------------------
KEY_COL  = "mint_token_id"
TIME_COL = "timestamp"

NUMERIC_CANDIDATES = [
    "index","token_quantity","creator_fee","creator_fee_pump","market_cap_usd",
    "token_delta","sol_delta","buy_count","sell_count","total_count",
    "token_volume","sol_volume","liquidity_ratio","virtual_sol_reserves","virtual_token_reserves",
    "consumed_gas","fee","relative_strength_index","bollinger_relative_position","volume_oscillator",
    "rate_of_change","money_flow_index","total_holders","current_holders","top10_percent_total",
    "creator_balance","creator_sold","holder_ratio","buy_sell_ratio",
]
STRING_DROP = ["holder","creator","trade_mode"]  # baseline bỏ cột quá nhiều giá trị khác nhau

def ensure_numeric(df, cols):
    use = [c for c in cols if c in df.columns]
    for c in use:
        if not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return use

# -----------------------
# 2) Eval reader
# -----------------------
def read_eval_chunks(eval_dir: Path):
    paths = sorted(eval_dir.glob("evaluation_set_30s_chunk_*.csv"))
    assert len(paths) == 5, f"Expected 5 eval chunks, got {len(paths)}"
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)

# -----------------------
# 3) FULL dataset files
# -----------------------
def list_full_paths(full_dir: Path):
    paths = sorted(full_dir.glob("september_2025_first30s_chunk_*.csv"))
    assert len(paths) > 0, "No full dataset CSVs found"
    return paths

# -----------------------
# 4) Chunk aggregator (NO MERGE CHAINS)
# -----------------------
def aggregate_chunk(chunk: pd.DataFrame):
    """Return a per-token aggregate for one chunk:
       - numeric *_sum, *_max, *_min
       - row_count
       - t_min, t_max (if timestamp present)
    """
    # drop heavy string cols
    drop_cols = [c for c in STRING_DROP if c in chunk.columns]
    if drop_cols:
        chunk = chunk.drop(columns=drop_cols)

    # numeric block
    num_cols = ensure_numeric(chunk, NUMERIC_CANDIDATES)
    chunk[num_cols] = chunk[num_cols].fillna(0)

    num_agg = chunk[[KEY_COL] + num_cols].groupby(KEY_COL).agg(["sum","max","min"])
    num_agg.columns = [f"{c}__{stat}" for c, stat in num_agg.columns.to_flat_index()]
    num_agg = num_agg.reset_index()

    # row_count
    cnt = chunk.groupby(KEY_COL, as_index=False).size().rename(columns={"size":"row_count"})

    # time span (optional)
    if TIME_COL in chunk.columns:
        # parse robustly; warnings ok
        ts = pd.to_datetime(chunk[TIME_COL], utc=True, errors="coerce")
        tdf = pd.DataFrame({KEY_COL: chunk[KEY_COL].values, TIME_COL: ts})
        tspan = tdf.groupby(KEY_COL)[TIME_COL].agg(["min","max"]).reset_index()
        tspan = tspan.rename(columns={"min":"t_min","max":"t_max"})
        # merge inside this chunk
        out = num_agg.merge(cnt, on=KEY_COL, how="left").merge(tspan, on=KEY_COL, how="left")
    else:
        out = num_agg.merge(cnt, on=KEY_COL, how="left")
        out["t_min"] = pd.NaT
        out["t_max"] = pd.NaT

    return out

def combine_aggregates(df_list):
    """Concat all per-chunk aggregates, then final groupby to combine:
       - *_sum → sum
       - *_max → max
       - *_min → min
       - row_count → sum
       - t_min → min, t_max → max
    """
    all_df = pd.concat(df_list, ignore_index=True)
    # collect columns by suffix
    sum_cols = [c for c in all_df.columns if c.endswith("__sum")]
    max_cols = [c for c in all_df.columns if c.endswith("__max")]
    min_cols = [c for c in all_df.columns if c.endswith("__min")]
    keep = [KEY_COL, "row_count", "t_min", "t_max"] + sum_cols + max_cols + min_cols
    keep = [c for c in keep if c in all_df.columns]
    all_df = all_df[keep]

    agg_dict = {c: "sum" for c in sum_cols}
    agg_dict.update({c: "max" for c in max_cols})
    agg_dict.update({c: "min" for c in min_cols})
    if "row_count" in all_df.columns: agg_dict["row_count"] = "sum"
    if "t_min"     in all_df.columns: agg_dict["t_min"]   = "min"
    if "t_max"     in all_df.columns: agg_dict["t_max"]   = "max"

    out = all_df.groupby(KEY_COL, as_index=False).agg(agg_dict)
    # lifespan
    if "t_min" in out.columns and "t_max" in out.columns:
        out["lifespan_seconds"] = (out["t_max"] - out["t_min"]).dt.total_seconds()
        out["lifespan_seconds"] = out["lifespan_seconds"].fillna(0)
    else:
        out["lifespan_seconds"] = 0.0
    return out.fillna(0)

def build_token_features_from_files(paths, verbose_every=3):
    agg_list = []
    for i, p in enumerate(paths, 1):
        print(f"[Build-feats] {p.name} ({i}/{len(paths)})")
        for chunk in pd.read_csv(p, chunksize=CHUNKSIZE):
            agg = aggregate_chunk(chunk)
            agg_list.append(agg)
            del chunk, agg
            gc.collect()
        if i % verbose_every == 0:
            print(f"  → partial groups stored: {len(agg_list)}")
    feats = combine_aggregates(agg_list)
    return feats

# -----------------------
# 5) Load eval + build features
# -----------------------
print("=== Load EVAL ===")
eval_df = read_eval_chunks(COMP_DIR)
eval_ids = eval_df[KEY_COL].drop_duplicates().reset_index(drop=True)
print("EVAL rows:", len(eval_df), "| unique tokens:", len(eval_ids))
assert len(eval_ids) == 64208, f"Expected 64208 eval tokens, got {len(eval_ids)}"
del eval_df; gc.collect()

print("=== Build FULL features (train) ===")
full_paths = list_full_paths(FULL_DIR)
train_feats = build_token_features_from_files(full_paths, verbose_every=2)
print("Train feats:", train_feats.shape)

print("=== Build EVAL features (test) ===")
eval_paths = sorted(COMP_DIR.glob("evaluation_set_30s_chunk_*.csv"))
eval_feats = build_token_features_from_files(eval_paths, verbose_every=3)
eval_feats = eval_feats[eval_feats[KEY_COL].isin(set(eval_ids))]
print("Eval feats:", eval_feats.shape)

# -----------------------
# 6) Targets
# -----------------------
def download_target_tokens_csv(dst_path: Path):
    if dst_path.exists(): return True
    if not HAVE_GDOWN:
        print("gdown not available — upload target_tokens.csv to /kaggle/working/")
        return False
    url = f"https://drive.google.com/uc?id={TARGET_TOKENS_GDRIVE_ID}"
    gdown.download(url, str(dst_path), quiet=False)
    return dst_path.exists() and dst_path.stat().st_size > 0

ok = download_target_tokens_csv(TARGET_TOKENS_CSV)
if not ok:
    raise SystemExit("Missing target_tokens.csv")

def load_target_list(path: Path):
    df = pd.read_csv(path)
    # đoán tên cột token
    cands = [c for c in df.columns if "mint" in c.lower() or "token" in c.lower()]
    assert cands, "Cannot find token id column in target csv"
    col = cands[0]
    t = df[[col]].dropna().drop_duplicates()
    t = t.rename(columns={col: KEY_COL})
    t[KEY_COL] = t[KEY_COL].astype(str)
    print("Targets:", len(t))
    return t

target_list = load_target_list(TARGET_TOKENS_CSV)

train_df = train_feats.merge(target_list.assign(is_target=1), on=KEY_COL, how="left")
train_df["is_target"] = train_df["is_target"].fillna(0).astype(int)

# (optional) tạo vài mean đơn giản từ sum/row_count
def add_simple_means(df):
    if "row_count" not in df.columns or df["row_count"].eq(0).all():
        return df
    for base in NUMERIC_CANDIDATES:
        s = f"{base}__sum"
        if s in df.columns:
            df[f"{base}__mean"] = df[s] / df["row_count"].clip(lower=1)
    return df

train_df = add_simple_means(train_df)
eval_feats = add_simple_means(eval_feats)

# Align feature columns
feature_cols = [c for c in train_df.columns if c not in [KEY_COL, "is_target", "t_min", "t_max"]]
# ensure numeric
for df_ in (train_df, eval_feats):
    for c in feature_cols:
        if c in df_.columns and (df_[c].dtype == object):
            df_[c] = pd.to_numeric(df_[c], errors="coerce").fillna(0)

# intersect features
feature_cols = [c for c in feature_cols if c in eval_feats.columns]

X = train_df[feature_cols].fillna(0)
y = train_df["is_target"].astype(int)
X_test = eval_feats[feature_cols].fillna(0)
test_ids = eval_feats[KEY_COL].values

print("Train shape:", X.shape, "| positives:", int(y.sum()))
print("Test  shape:", X_test.shape)

# -----------------------
# 7) EDA nhanh
# -----------------------
pos = int(y.sum()); n = len(y)
print(f"Class balance: pos={pos} ({pos/n:.2%})  neg={n-pos} ({1-pos/n:.2%})")
print("Top-10 var features:\n", X.var().sort_values(ascending=False).head(10))

# -----------------------
# 8) Train CatBoost
# -----------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

train_pool = Pool(X_tr, y_tr)
valid_pool = Pool(X_va, y_va)

params = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=SEED,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    iterations=2000,
    subsample=0.8,
    rsm=0.8,
    od_type="Iter",
    od_wait=100,
    verbose=200,
)
model = CatBoostClassifier(**params)
model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

va_pred = model.predict_proba(X_va)[:,1]
print(f"Valid AUC={roc_auc_score(y_va, va_pred):.4f} | AP={average_precision_score(y_va, va_pred):.4f}")

# chọn threshold baseline
THRESH = 0.5

# -----------------------
# 9) Inference + submission
# -----------------------
test_pred = model.predict_proba(X_test)[:,1]
sub_df = pd.DataFrame({KEY_COL: test_ids, "is_target": (test_pred >= THRESH).astype(int)})
assert len(sub_df) == 64208, f"Submission must have 64208 rows, got {len(sub_df)}"
sub_path = WORK_DIR / "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Saved submission:", sub_path, sub_df.shape)
display(sub_df.head())

# (optional) prediction details for deliverables
pred_detail = pd.DataFrame({
    KEY_COL: test_ids,
    "pred_score": test_pred,
    "threshold": THRESH,
    "is_target_pred": (test_pred >= THRESH).astype(int),
})
pred_detail.to_csv(WORK_DIR / "pred_detail_eval.csv", index=False)
print("Saved:", WORK_DIR / "pred_detail_eval.csv")


