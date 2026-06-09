# =========================================================
# Alpha Radar: CatBoost Imbalance Toolkit (GPU-safe params, auto CPU fallback)
#   - GPU fallback: tự chuyển CPU nếu CUDA lỗi/không có driver
#   - GPU-safe: KHÔNG dùng rsm trên GPU; CPU có thể dùng rsm
#   - Bernoulli bootstrap để dùng subsample
#   - Fix fragmented DataFrame khi tạo log1p_*
#   - Ép float32 để giảm RAM
# =========================================================

import os, gc, random, re, sys, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ---- Config ----
USE_GPU_WISH = True        # mong muốn dùng GPU; script sẽ tự fallback nếu không khả dụng
CHUNKSIZE = 250_000

COMP_DIR  = Path("/kaggle/input/alpha-radar-solana-sprint")
FULL_DIR  = Path("/kaggle/input/pumpfun-30s-september-2025")
WORK_DIR  = Path("/kaggle/working"); WORK_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TOKENS_GDRIVE_ID = "1EsqpZXPBU-6m0djDmccCrtUX07jV2fHA"
TARGET_TOKENS_CSV = WORK_DIR / "target_tokens.csv"

# ---- Deps ----
try:
    from catboost import CatBoostClassifier, Pool
    try:
        from catboost.utils import get_gpu_device_count
    except Exception:
        get_gpu_device_count = None
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])
    from catboost import CatBoostClassifier, Pool
    try:
        from catboost.utils import get_gpu_device_count
    except Exception:
        get_gpu_device_count = None

try:
    import gdown; HAVE_GDOWN = True
except Exception:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown; HAVE_GDOWN = True
    except Exception:
        HAVE_GDOWN = False

print(f"CatBoost OK | gdown: {HAVE_GDOWN}")

# =========================
# GPU detect & fallback
# =========================
def detect_gpu_for_catboost() -> bool:
    """
    Trả về True nếu có GPU usable cho CatBoost.
    Không đảm bảo 100%, nhưng đủ an toàn để thử.
    """
    # 1) Nếu Kaggle bật GPU thì thường có thiết bị /dev/nvidia0
    has_dev = any(Path(f"/dev/nvidia{i}").exists() for i in range(4))
    # 2) CatBoost đếm GPU
    cnt = 0
    if get_gpu_device_count is not None:
        try:
            cnt = int(get_gpu_device_count())
        except Exception:
            cnt = 0
    # 3) Env có thể chặn GPU
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    blocked = (cuda_visible.strip() == "-1")
    return (has_dev or cnt > 0) and not blocked

# Cờ USE_GPU thực tế (mong muốn + khả dụng)
USE_GPU = bool(USE_GPU_WISH and detect_gpu_for_catboost())
print("Will try GPU:", USE_GPU)

# =========================
# Columns & helpers
# =========================
KEY_COL  = "mint_token_id"
TIME_COL = "timestamp"

BASE_NUMS = [
    "index","token_quantity","creator_fee","creator_fee_pump","market_cap_usd",
    "token_delta","sol_delta","buy_count","sell_count","total_count",
    "token_volume","sol_volume","liquidity_ratio","virtual_sol_reserves","virtual_token_reserves",
    "consumed_gas","fee","relative_strength_index","bollinger_relative_position","volume_oscillator",
    "rate_of_change","money_flow_index","total_holders","current_holders","top10_percent_total",
    "creator_balance","creator_sold","holder_ratio","buy_sell_ratio",
]
ID_TX_COLS = ["holder","creator","trade_mode"]

WIN_LIST = [5, 10, 20, 30]
WIN_SUM_VARS = [
    "token_volume","sol_volume","buy_count","sell_count","total_count",
    "token_quantity","sol_delta","market_cap_usd"
]

MMSS_RE = re.compile(r"^\d{1,2}:\d{2}(\.\d+)?$")

def fast_parse_seconds(x: str):
    if isinstance(x, str):
        m = MMSS_RE.match(x)
        if m:
            mm, ss = x.split(":")
            try:
                return 60*int(mm) + float(ss)
            except:
                return np.nan
    return np.nan

def ensure_numeric(df, cols):
    use = [c for c in cols if c in df.columns]
    for c in use:
        if not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return use

# =========================
# Loaders
# =========================
def read_eval_chunks(eval_dir: Path):
    paths = sorted(eval_dir.glob("evaluation_set_30s_chunk_*.csv"))
    assert len(paths) == 5, f"Expected 5 eval chunks, got {len(paths)}"
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)

def list_full_paths(full_dir: Path):
    paths = sorted(full_dir.glob("september_2025_first30s_chunk_*.csv"))
    assert len(paths) > 0, "No full dataset CSVs found"
    return paths

# =========================
# Chunk aggregation + early behavior feats
# =========================
def aggregate_chunk(chunk: pd.DataFrame):
    has_holder  = "holder"   in chunk.columns
    has_creator = "creator"  in chunk.columns
    has_tm      = "trade_mode" in chunk.columns

    if TIME_COL in chunk.columns:
        t_abs = chunk[TIME_COL].map(fast_parse_seconds)
    else:
        t_abs = pd.Series(np.nan, index=chunk.index)
    if t_abs.isna().all():
        chunk["_row_order"] = chunk.groupby(KEY_COL).cumcount().astype(float)
        t_abs = chunk["_row_order"].values.astype(float)
    chunk["_t_abs"] = t_abs

    num_cols = ensure_numeric(chunk, BASE_NUMS)
    chunk[num_cols] = chunk[num_cols].fillna(0)

    tstats = chunk.groupby(KEY_COL)["_t_abs"].agg(["min","max","count"]).reset_index()\
                  .rename(columns={"min":"t_min_s","max":"t_max_s","count":"row_count"})
    chunk = chunk.merge(tstats[[KEY_COL,"t_min_s"]], on=KEY_COL, how="left")
    chunk["_t_rel"] = chunk["_t_abs"] - chunk["t_min_s"]

    win_aggs = []
    for W in WIN_LIST:
        mask = chunk["_t_rel"] <= W
        if not mask.any(): continue

        cols_sum = [v for v in WIN_SUM_VARS if v in chunk.columns]
        sub_num = chunk.loc[mask, [KEY_COL] + cols_sum]
        g = sub_num.groupby(KEY_COL).sum()
        g.columns = [f"{c}_w{W}_sum" for c in g.columns]
        g = g.reset_index()

        cnt = chunk.loc[mask, [KEY_COL]].groupby(KEY_COL).size().rename(f"row_count_w{W}").reset_index()
        g = g.merge(cnt, on=KEY_COL, how="left")

        if has_holder:
            hu = chunk.loc[mask, [KEY_COL, "holder"]].dropna()
            if not hu.empty:
                hu = hu.groupby(KEY_COL)["holder"].nunique().rename(f"unique_holders_w{W}").reset_index()
                g = g.merge(hu, on=KEY_COL, how="left")

        if has_creator:
            ca = chunk.loc[mask, [KEY_COL, "creator"]].dropna()
            if not ca.empty:
                ca["creator_seen"] = 1
                ca = ca.groupby(KEY_COL)["creator_seen"].max().rename(f"creator_seen_w{W}").reset_index()
                g = g.merge(ca, on=KEY_COL, how="left")

        if has_creator and has_tm:
            sub_ct = chunk.loc[mask, [KEY_COL, "creator", "trade_mode"]].dropna()
            if not sub_ct.empty:
                sub_ct["is_creator_trade"] = (sub_ct["creator"].astype(str).str.len()>0).astype(int)
                cre_trade = sub_ct.groupby(KEY_COL)["is_creator_trade"].max().rename(f"creator_traded_w{W}").reset_index()
                g = g.merge(cre_trade, on=KEY_COL, how="left")

                sub_ct["is_creator_sell"] = (sub_ct["trade_mode"].astype(str).str.lower()=="sell").astype(int)
                cre_sell = sub_ct.groupby(KEY_COL)["is_creator_sell"].max().rename(f"creator_sold_flag_w{W}").reset_index()
                g = g.merge(cre_sell, on=KEY_COL, how="left")

        win_aggs.append(g)

    num_agg = chunk[[KEY_COL] + num_cols].groupby(KEY_COL).agg(["sum","max","min"])
    num_agg.columns = [f"{c}__{stat}" for c, stat in num_agg.columns.to_flat_index()]
    num_agg = num_agg.reset_index()

    out = num_agg.merge(tstats, on=KEY_COL, how="left")
    for g in win_aggs:
        out = out.merge(g, on=KEY_COL, how="left")
    return out.fillna(0)

def combine_aggregates(df_list):
    all_df = pd.concat(df_list, ignore_index=True)

    sum_cols = [c for c in all_df.columns if c.endswith("__sum") or re.search(r"_w\d+_sum$", c)]
    max_cols = [c for c in all_df.columns if c.endswith("__max")]
    min_cols = [c for c in all_df.columns if c.endswith("__min")]
    cnt_cols = ["row_count"] + [c for c in all_df.columns if c.startswith("row_count_w")]
    extra_cols = [c for c in all_df.columns if c.startswith("unique_holders_w") or
                  c.startswith("creator_seen_w") or c.startswith("creator_traded_w") or
                  c.startswith("creator_sold_flag_w")]

    keep = [KEY_COL,"t_min_s","t_max_s"] + sum_cols + max_cols + min_cols + cnt_cols + extra_cols
    keep = [c for c in keep if c in all_df.columns]
    all_df = all_df[keep]

    agg = {c:"sum" for c in sum_cols + cnt_cols}
    agg.update({c:"max" for c in max_cols})
    agg.update({c:"min" for c in min_cols})
    for c in extra_cols: agg[c] = "max"
    agg["t_min_s"] = "min"; agg["t_max_s"] = "max"

    out = all_df.groupby(KEY_COL, as_index=False).agg(agg)
    out["lifespan_seconds"] = (out["t_max_s"] - out["t_min_s"]).clip(lower=0)
    return out.fillna(0)

def build_token_features_from_files(paths, verbose_every=3):
    agg_list = []
    for i, p in enumerate(paths, 1):
        print(f"[Build-feats] {p.name} ({i}/{len(paths)})")
        for chunk in pd.read_csv(p, chunksize=CHUNKSIZE):
            agg = aggregate_chunk(chunk)
            agg_list.append(agg); del chunk, agg
            gc.collect()
        if i % verbose_every == 0:
            print(f"  → partial groups stored: {len(agg_list)}")
    feats = combine_aggregates(agg_list)
    return feats

# =========================
# Load EVAL + FULL & feats
# =========================
print("=== Load EVAL ===")
eval_df = read_eval_chunks(COMP_DIR)
eval_ids = eval_df[KEY_COL].drop_duplicates().reset_index(drop=True)
print("EVAL rows:", len(eval_df), "| unique tokens:", len(eval_ids))
assert len(eval_ids) == 64208
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

# =========================
# Targets
# =========================
def download_target_tokens_csv(dst_path: Path):
    if dst_path.exists(): return True
    if not HAVE_GDOWN:
        print("gdown not available — upload target_tokens.csv to /kaggle/working/")
        return False
    url = f"https://drive.google.com/uc?id={TARGET_TOKENS_GDRIVE_ID}"
    gdown.download(url, str(dst_path), quiet=False)
    return dst_path.exists() and dst_path.stat().st_size > 0

ok = download_target_tokens_csv(TARGET_TOKENS_CSV)
if not ok: raise SystemExit("Missing target_tokens.csv")

def load_target_list(path: Path):
    df = pd.read_csv(path)
    cands = [c for c in df.columns if "mint" in c.lower() or "token" in c.lower()]
    assert cands, "Cannot find token id column in target csv"
    col = cands[0]
    t = df[[col]].dropna().drop_duplicates().rename(columns={col: KEY_COL})
    t[KEY_COL] = t[KEY_COL].astype(str)
    print("Targets:", len(t))
    return t

target_list = load_target_list(TARGET_TOKENS_CSV)
train_df = train_feats.merge(target_list.assign(is_target=1), on=KEY_COL, how="left")
train_df["is_target"] = train_df["is_target"].fillna(0).astype(int)

# =========================
# Post-agg features (đã sửa fragmented warning)
# =========================
def add_post_agg_features(df: pd.DataFrame) -> pd.DataFrame:
    if "row_count" not in df.columns:
        df["row_count"] = 1
    rc = df["row_count"].clip(lower=1)

    for base in ["token_volume","sol_volume","token_quantity","sol_delta","market_cap_usd","buy_count","sell_count","total_count"]:
        csum = f"{base}__sum"
        if csum in df.columns:
            df[f"{base}__per_tx"] = df[csum] / rc

    if "buy_count__sum" in df.columns and "total_count__sum" in df.columns:
        df["buy_ratio"]  = df["buy_count__sum"]  / df["total_count__sum"].clip(lower=1)
    if "sell_count__sum" in df.columns and "total_count__sum" in df.columns:
        df["sell_ratio"] = df["sell_count__sum"] / df["total_count__sum"].clip(lower=1)

    if "lifespan_seconds" in df.columns:
        ls = df["lifespan_seconds"].replace(0, np.nan)
        for base in ["token_volume","sol_volume","total_count","buy_count","sell_count"]:
            csum = f"{base}__sum"
            if csum in df.columns:
                df[f"{base}__per_sec"] = df[csum] / ls
        df.fillna(0, inplace=True)

    for W in WIN_LIST:
        tc = f"total_count_w{W}_sum"
        bc = f"buy_count_w{W}_sum"
        sc = f"sell_count_w{W}_sum"
        sv = f"sol_volume_w{W}_sum"
        tv = f"token_volume_w{W}_sum"
        rcw= f"row_count_w{W}"

        if tc in df.columns:
            df[f"buy_ratio_w{W}"]   = df.get(bc, 0) / df[tc].clip(lower=1)
            df[f"sell_ratio_w{W}"]  = df.get(sc, 0) / df[tc].clip(lower=1)
            df[f"per_tx_solv_w{W}"] = df.get(sv, 0) / df[tc].clip(lower=1)
            df[f"per_tx_tokv_w{W}"] = df.get(tv, 0) / df[tc].clip(lower=1)

        if rcw in df.columns:
            df[f"avg_tokq_w{W}"] = df.get(f"token_quantity_w{W}_sum", 0) / df[rcw].clip(lower=1)

        df[f"sol_per_sec_w{W}"]  = df.get(sv, 0) / max(1, W)
        df[f"tokv_per_sec_w{W}"] = df.get(tv, 0) / max(1, W)

    for a,b in [(10,5),(20,10),(30,20)]:
        feats = {
            "buy_ratio":     (f"buy_ratio_w{a}", f"buy_ratio_w{b}"),
            "sell_ratio":    (f"sell_ratio_w{a}", f"sell_ratio_w{b}"),
            "per_tx_solv":   (f"per_tx_solv_w{a}", f"per_tx_solv_w{b}"),
            "per_tx_tokv":   (f"per_tx_tokv_w{a}", f"per_tx_tokv_w{b}"),
            "sol_per_sec":   (f"sol_per_sec_w{a}", f"sol_per_sec_w{b}"),
            "tokv_per_sec":  (f"tokv_per_sec_w{a}", f"tokv_per_sec_w{b}"),
        }
        for name, (cA, cB) in feats.items():
            if cA in df.columns and cB in df.columns:
                df[f"{name}_delta_{a}-{b}"] = df[cA] - df[cB]

    weights = {5:1.0, 10:0.6, 20:0.35, 30:0.2}
    for base in ["total_count","buy_count","sell_count","sol_volume","token_volume"]:
        acc = 0.0; den = 0.0
        for W, w in weights.items():
            csum = f"{base}_w{W}_sum"
            if csum in df.columns:
                acc += w * df[csum]
                den += w * max(1, W)
        df[f"ewma_{base}_per_sec"] = acc / max(1e-9, den)

    # --- NEW: tạo log1p_* bằng concat một lần để tránh fragmented ---
    log_src_cols = [c for c in df.columns if c.endswith("__sum") or re.search(r"_w\d+_sum$", c) or c.endswith("__max")]
    if log_src_cols:
        tmp = np.log1p(df[log_src_cols].astype(float).clip(lower=0))
        tmp.columns = [f"log1p_{c}" for c in log_src_cols]
        df = pd.concat([df, tmp], axis=1)

    return df

train_df = add_post_agg_features(train_df)
eval_feats = add_post_agg_features(eval_feats)

# Align features
drop_non_features = [KEY_COL, "is_target", "t_min_s", "t_max_s"]
feature_cols = [c for c in train_df.columns if c not in drop_non_features]
for df_ in (train_df, eval_feats):
    for c in feature_cols:
        if c in df_.columns and df_[c].dtype == object:
            df_[c] = pd.to_numeric(df_[c], errors="coerce").fillna(0)

feature_cols = [c for c in feature_cols if c in eval_feats.columns]
X_full = train_df[feature_cols].fillna(0).astype(np.float32)
y_full = train_df["is_target"].astype(int)
X_test = eval_feats[feature_cols].fillna(0).astype(np.float32)
test_ids = eval_feats[KEY_COL].astype(str).values

print("Raw Train:", X_full.shape, "| Pos:", int(y_full.sum()), "Neg:", int((1-y_full).sum()))
print("Test     :", X_test.shape)

# =========================================================
# Imbalance toolkit
# =========================================================
def difficulty_score(df: pd.DataFrame) -> np.ndarray:
    comps = []
    for c in ["total_count__sum","buy_count__sum","sell_count__sum",
              "sol_volume__sum","token_volume__sum",
              "sol_per_sec_w10","tokv_per_sec_w10",
              "buy_ratio_w10","per_tx_solv_w10","per_tx_tokv_w10",
              "ewma_total_count_per_sec","ewma_sol_volume_per_sec","ewma_token_volume_per_sec"]:
        if c in df.columns:
            comps.append(df[c].values.astype(float))
    if not comps:
        return np.zeros(len(df), dtype=float)
    S = np.vstack(comps)
    ranks = np.apply_along_axis(lambda v: (pd.Series(v).rank().values-1)/max(1, len(v)-1), 1, S)
    return ranks.mean(axis=0)

def build_balanced_training(X: pd.DataFrame, y: pd.Series,
                            pos_aug_factor: int = 2,
                            neg_pos_ratio: float = 6.0,
                            hard_frac: float = 0.6,
                            jitter_std: float = 0.005,
                            random_state: int = 42):
    rs = np.random.RandomState(random_state)

    pos_idx = np.where(y.values==1)[0]
    neg_idx = np.where(y.values==0)[0]

    n_pos = len(pos_idx)
    n_neg_target = int(np.clip(neg_pos_ratio * n_pos, n_pos, len(neg_idx)))

    dscore = difficulty_score(X.iloc[neg_idx])
    order = np.argsort(-dscore)
    n_hard = int(hard_frac * n_neg_target)
    hard_sel = neg_idx[order[:n_hard]]

    rem = max(0, n_neg_target - n_hard)
    rand_pool = np.setdiff1d(neg_idx, hard_sel, assume_unique=False)
    if rem > 0 and len(rand_pool) > 0:
        rand_sel = rs.choice(rand_pool, size=min(rem, len(rand_pool)), replace=False)
        neg_sel = np.concatenate([hard_sel, rand_sel])
    else:
        neg_sel = hard_sel

    X_pos = X.iloc[pos_idx].copy()
    pos_blocks = [X_pos]
    for _ in range(max(0, pos_aug_factor-1)):
        X_j = X_pos.copy()
        num_cols = X_j.columns
        noise = rs.normal(0, jitter_std, size=X_j[num_cols].shape).astype(np.float32)
        mask = np.array([c.startswith("log1p_") or ("_per_" in c) or ("ratio" in c) for c in num_cols])
        noise[:, ~mask] = 0.0
        X_j[num_cols] = X_j[num_cols].values + noise
        pos_blocks.append(X_j)

    X_pos_aug = pd.concat(pos_blocks, axis=0, ignore_index=True)
    y_pos_aug = pd.Series(1, index=range(len(X_pos_aug)))

    X_neg = X.iloc[neg_sel].copy()
    y_neg = pd.Series(0, index=range(len(X_neg)))

    X_bal = pd.concat([X_pos_aug, X_neg], axis=0, ignore_index=True)
    y_bal = pd.concat([y_pos_aug, y_neg], axis=0, ignore_index=True)

    w = np.ones(len(X_bal), dtype=np.float32)
    pos_end = len(X_pos_aug)
    neg_is_hard = np.isin(neg_sel, hard_sel)
    w[:pos_end] = 1.5
    w[pos_end:][neg_is_hard] = 1.2
    w[pos_end:][~neg_is_hard] = 1.0

    perm = rs.permutation(len(X_bal))
    X_bal = X_bal.iloc[perm].reset_index(drop=True)
    y_bal = y_bal.iloc[perm].reset_index(drop=True)
    w     = w[perm]

    return X_bal.astype(np.float32), y_bal, w

X, y, sw = build_balanced_training(
    X_full, y_full,
    pos_aug_factor=2,
    neg_pos_ratio=6.0,
    hard_frac=0.6,
    jitter_std=0.005,
    random_state=SEED
)
print("Balanced Train:", X.shape, "| Pos%:", f"{100*y.mean():.2f}%", "| sw.mean=", f"{sw.mean():.3f}")

# ============ Winsorize + Rank-normalize ============
def winsorize99(x):
    lo, hi = np.nanpercentile(x, 0.1), np.nanpercentile(x, 99.9)
    return np.clip(x, lo, hi)

cap_cols = [c for c in X.columns if c.endswith("__sum") or re.search(r"_w\d+_sum$", c)]
for c in cap_cols[:150]:
    X[c] = winsorize99(X[c].values)
    X_test[c] = np.clip(X_test[c].values, X[c].min(), X[c].max())

def rank01(s):
    return (s.rank(method="average") - 1) / max(1, len(s)-1)

rank_cols = [c for c in X.columns if c.startswith("log1p_") or c in cap_cols]
for c in rank_cols[:180]:
    X[c] = rank01(X[c])
    X_test[c] = rank01(X_test[c])

print("Train(after transform):", X.shape, "Pos:", int(y.sum()))
print("Test :", X_test.shape)

# =========================
# CatBoost params (GPU-safe) + auto fallback CPU
# =========================
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve
from catboost import CatBoostError

pos_cnt = int(y.sum()); neg_cnt = len(y) - pos_cnt
w_pos_cls = float(np.sqrt(neg_cnt / max(1, pos_cnt)))
class_weights = [1.0, w_pos_cls]
print("class_weights:", class_weights)

def build_cat_params(use_gpu=True):
    params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        learning_rate=0.035,
        depth=10,
        l2_leaf_reg=6.0,
        iterations=8000,
        subsample=0.85,                 # Bernoulli supports subsample
        bootstrap_type="Bernoulli",
        class_weights=class_weights,
        od_type="Iter",
        od_wait=300,
        random_strength=0.8,
        leaf_estimation_iterations=8,
        verbose=200,
    )
    if use_gpu:
        params.update(dict(task_type="GPU", devices="0"))
        # IMPORTANT: rsm is NOT supported on GPU (except pairwise) -> do NOT set it
    else:
        params["rsm"] = 0.7
        params["thread_count"] = -1
    return params

params_base = build_cat_params(USE_GPU)
params_cv   = params_base.copy(); params_cv["verbose"] = 0

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(X), dtype=float)
test_preds = np.zeros(len(X_test), dtype=float)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    print(f"[CV] Fold {fold} | try GPU={params_cv.get('task_type','CPU')=='GPU'}")
    X_tr, y_tr, sw_tr = X.iloc[tr_idx], y.iloc[tr_idx], sw[tr_idx]
    X_va, y_va        = X.iloc[va_idx], y.iloc[va_idx]

    tried_gpu = (params_cv.get("task_type", "CPU") == "GPU")
    m = CatBoostClassifier(**params_cv)
    try:
        m.fit(Pool(X_tr, y_tr, weight=sw_tr), eval_set=Pool(X_va, y_va), use_best_model=True, verbose=False)
    except CatBoostError as e:
        msg = str(e)
        if tried_gpu and ("CUDA" in msg or "GPU" in msg or "cuda" in msg or "driver" in msg):
            print("  -> CUDA error detected; falling back to CPU for this fold.")
            params_cpu = build_cat_params(use_gpu=False)
            params_cpu["verbose"] = 0
            m = CatBoostClassifier(**params_cpu)
            m.fit(Pool(X_tr, y_tr, weight=sw_tr), eval_set=Pool(X_va, y_va), use_best_model=True, verbose=False)
        else:
            raise

    oof[va_idx] = m.predict_proba(X_va)[:,1]
    test_preds += m.predict_proba(X_test)[:,1] / skf.n_splits
    del m; gc.collect()

def best_thresh_iou_with_recall(y_true, scores, min_recall=0.75):
    prec, rec, thr = precision_recall_curve(y_true, scores)
    thr = np.r_[thr, [1.0]]
    P = float(y_true.sum())
    best_iou, best_t = -1.0, 0.5
    for p, r, t in zip(prec, rec, thr):
        if r < min_recall or p <= 0: 
            continue
        TP = r * P
        PosPred = TP / p
        FP = max(0.0, PosPred - TP)
        FN = max(0.0, P - TP)
        denom = TP + FP + FN
        if denom <= 0:
            continue
        iou = TP / denom
        if iou > best_iou:
            best_iou, best_t = iou, float(t)
    return best_t, best_iou

T_oof, best_iou = best_thresh_iou_with_recall(y.values, oof, min_recall=0.75)
print(f"[OOF] Best IoU={best_iou:.4f} at T={T_oof:.6f}")

# =========================
# Anti all-zero (quantile fuse + topK fuse)
# =========================
base_rate = float(y_full.mean())
target_rate = float(np.clip(base_rate*1.6, 0.015, 0.25))
T_quant = float(np.quantile(test_preds, 1.0 - target_rate))
T = min(T_oof, T_quant)

pred_bin = (test_preds >= T).astype(int)
pos_rate = pred_bin.mean()

if pos_rate < 0.015:
    for q in [0.985, 0.97, 0.95]:
        T_try = float(np.quantile(test_preds, q))
        pred_try = (test_preds >= T_try).astype(int)
        if pred_try.mean() >= 0.015:
            T = T_try
            pred_bin = pred_try
            pos_rate = pred_bin.mean()
            break

if pred_bin.sum() == 0:
    K = max(int(round(base_rate * len(test_preds))), 100)
    idx_topk = np.argsort(test_preds)[-K:]
    pred_bin = np.zeros_like(test_preds, dtype=int)
    pred_bin[idx_topk] = 1
    T = float(test_preds[idx_topk].min())
    pos_rate = pred_bin.mean()

print(f"[TEST Threshold] T_oof={T_oof:.6f} | T_quant={T_quant:.6f} | T_final={T:.6f} | pos_rate={pos_rate:.2%}")

# =========================
# Submission & artifacts
# =========================
sub_df = pd.DataFrame({KEY_COL: test_ids, "is_target": pred_bin})
assert len(sub_df) == 64208, f"Submission must have 64208 rows, got {len(sub_df)}"
sub_path = WORK_DIR / "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Saved submission:", sub_path, sub_df.shape)
display(sub_df.head())

pd.DataFrame({
    KEY_COL: test_ids,
    "pred_score": test_preds,
    "threshold": T,
    "is_target_pred": pred_bin,
}).to_csv(WORK_DIR/"pred_detail_eval.csv", index=False)

# Feature importance (top 40) — train full balanced for importances
# Dùng cùng chiến lược fallback như trên, nhưng huấn luyện 1 model duy nhất.
params_imp = build_cat_params(USE_GPU)
try:
    imp_model = CatBoostClassifier(**{**params_imp, "verbose": 0})
    imp_model.fit(Pool(X, y, weight=sw), verbose=False)
except CatBoostError as e:
    if "CUDA" in str(e) or "GPU" in str(e) or "driver" in str(e):
        print("  -> CUDA error on importance model; switching to CPU.")
        params_imp = build_cat_params(False)
        imp_model = CatBoostClassifier(**{**params_imp, "verbose": 0})
        imp_model.fit(Pool(X, y, weight=sw), verbose=False)
    else:
        raise

imp = pd.Series(imp_model.get_feature_importance(Pool(X, y, weight=sw)), index=X.columns)\
        .sort_values(ascending=False)
imp.head(40).to_csv(WORK_DIR/"feature_importance_top40.csv", index=False)
print("Saved pred_detail_eval.csv & feature_importance_top40.csv")

