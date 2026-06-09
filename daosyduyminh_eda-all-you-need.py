# =========================================================
# Alpha Radar — SOL Sprint EDA + Feature Engineering
# (SOTA-flavored, Kaggle-friendly, hardened & verbose logs)
# =========================================================
# What you get:
#  • Robust EDA (schema, missingness, heavy tails, ECDFs)
#  • Optional label join (auto gdown) → PR/AUPRC emphasis
#  • Univariate targets (AUPRC & Spearman)
#  • Correlation (Spearman on log1p)
#  • Time patterns (arrival curve & window deltas)
#  • Leakage scan heuristics (early vs late signals)
#  • Token-level Feature Engineering (30s):
#      - sums/max/min, per-tx, per-sec, ratios, deltas, log1p
#  • Artifacts: CSVs + PNGs + Markdown report for sharing
# =========================================================

import os, gc, re, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score

# ------------------ Config ------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
warnings.filterwarnings("ignore")

WORK_DIR = Path("/kaggle/working")
SAVE_DIR = WORK_DIR / "eda_artifacts"; SAVE_DIR.mkdir(parents=True, exist_ok=True)

COMP_DIR  = Path("/kaggle/input/alpha-radar-solana-sprint")
FULL_DIR  = Path("/kaggle/input/pumpfun-30s-september-2025")

KEY_COL   = "mint_token_id"
TIME_COL  = "timestamp"

CHUNKSIZE  = 250_000
MAX_ROWS_QC_SAMPLE = 1_000_000   # cap for heavy plots

# Labels (optional auto-download like your baseline)
TARGET_TOKENS_GDRIVE_ID = "1EsqpZXPBU-6m0djDmccCrtUX07jV2fHA"
TARGET_TOKENS_CSV = WORK_DIR / "target_tokens.csv"

# Plot defaults
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = True

# Base numeric fields (from baseline)
BASE_NUMS = [
    "index","token_quantity","creator_fee","creator_fee_pump","market_cap_usd",
    "token_delta","sol_delta","buy_count","sell_count","total_count",
    "token_volume","sol_volume","liquidity_ratio","virtual_sol_reserves","virtual_token_reserves",
    "consumed_gas","fee","relative_strength_index","bollinger_relative_position","volume_oscillator",
    "rate_of_change","money_flow_index","total_holders","current_holders","top10_percent_total",
    "creator_balance","creator_sold","holder_ratio","buy_sell_ratio",
]
STRING_DROP = ["holder","creator","trade_mode"]

# Windows used across the project
WIN_LIST = [5, 10, 20, 30]
WIN_SUM_VARS = [
    "token_volume","sol_volume","buy_count","sell_count","total_count",
    "token_quantity","sol_delta","market_cap_usd"
]

# ------------------ Helpers ------------------
MMSS_RE = re.compile(r"^\d{1,2}:\d{2}(\.\d+)?$")

def fast_parse_seconds(x: str):
    if isinstance(x, str):
        m = MMSS_RE.match(x)
        if m:
            mm, ss = x.split(":")
            try:
                return 60*int(mm) + float(ss)
            except Exception:
                return np.nan
    return np.nan

def ensure_numeric(df, cols):
    use = [c for c in cols if c in df.columns]
    for c in use:
        if not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return use

def log1p_clip(s):
    return np.log1p(np.asarray(s, dtype="float64").clip(min=0))

def ecdf(y):
    y = np.asarray(y); y = y[~np.isnan(y)]
    if len(y) == 0: return np.array([]), np.array([])
    x = np.sort(y); n = len(x); p = np.arange(1, n + 1) / n
    return x, p

def bootstrap_ci_mean(x, n_boot=300, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    means = []
    for _ in range(n_boot):
        s = rng.choice(x, size=len(x), replace=True)
        means.append(np.mean(s))
    lo, hi = np.quantile(means, [alpha/2, 1-alpha/2])
    return (np.mean(x), lo, hi)

def maybe_download_targets(dst_path: Path, file_id: str) -> bool:
    if dst_path.exists(): return True
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        print("[Labels] Downloading target_tokens.csv via gdown ...")
        gdown.download(url, str(dst_path), quiet=False)
        ok = dst_path.exists() and dst_path.stat().st_size > 0
        print("[Labels] Download OK" if ok else "[Labels] Download failed")
        return ok
    except Exception as e:
        print(f"[Labels] gdown not available / failed: {e}")
        return False

def list_eval_paths(eval_dir: Path):
    paths = sorted(eval_dir.glob("evaluation_set_30s_chunk_*.csv"))
    assert len(paths) == 5, f"Expected 5 eval chunks, got {len(paths)}"
    return paths

def read_eval_all(eval_dir: Path):
    dfs = [pd.read_csv(p) for p in list_eval_paths(eval_dir)]
    return pd.concat(dfs, ignore_index=True)

# ------------------ EDA Pipeline ------------------
print(">>> [1] SCHEMA & QUICK COUNTS")
eval_df = read_eval_all(COMP_DIR)
eval_df = eval_df.drop(columns=[c for c in STRING_DROP if c in eval_df.columns], errors="ignore")

print(f"[Info] Eval shape: {eval_df.shape}")
print(f"[Info] Eval unique tokens: {eval_df[KEY_COL].nunique()}")

# Sample for plots
eval_sample = eval_df.sample(MAX_ROWS_QC_SAMPLE, random_state=SEED).reset_index(drop=True) \
              if len(eval_df) > MAX_ROWS_QC_SAMPLE else eval_df.copy()

# Timestamp seconds for time plots
eval_sample["_t_seconds"] = eval_sample[TIME_COL].map(fast_parse_seconds) if TIME_COL in eval_sample.columns else np.nan

# Schema overview
schema = pd.DataFrame({
    "column": eval_sample.columns,
    "dtype": [str(eval_sample[c].dtype) for c in eval_sample.columns],
    "n_missing": [int(eval_sample[c].isna().sum()) for c in eval_sample.columns],
    "missing_rate": [float(eval_sample[c].isna().mean()) for c in eval_sample.columns],
    "n_unique": [int(eval_sample[c].nunique(dropna=True)) for c in eval_sample.columns],
}).sort_values(["dtype","column"])
schema.to_csv(SAVE_DIR / "schema_overview.csv", index=False)
print("[Save] schema_overview.csv")

# ------------------ Labels (optional) ------------------
print("\n>>> [2] TARGET MERGE (optional)")
labels_ok = False
if not TARGET_TOKENS_CSV.exists():
    labels_ok = maybe_download_targets(TARGET_TOKENS_CSV, TARGET_TOKENS_GDRIVE_ID)
else:
    labels_ok = True

if labels_ok:
    try:
        tgt = pd.read_csv(TARGET_TOKENS_CSV)
        tgt_cols = [c for c in tgt.columns if "mint" in c.lower() or "token" in c.lower()]
        assert len(tgt_cols) > 0, "target_tokens.csv: token id column not found"
        tgt = tgt[[tgt_cols[0]]].rename(columns={tgt_cols[0]: KEY_COL}).dropna()
        tgt[KEY_COL] = tgt[KEY_COL].astype(str)
        eval_sample[KEY_COL] = eval_sample[KEY_COL].astype(str)
        eval_sample = eval_sample.merge(tgt.assign(is_target=1), on=KEY_COL, how="left")
        eval_sample["is_target"] = eval_sample["is_target"].fillna(0).astype(int)
        print(f"[Labels] Positives (rows): {int(eval_sample['is_target'].sum())}")
    except Exception as e:
        print(f"[Labels] Failed to load/merge targets: {e}")
        eval_sample["is_target"] = 0
        labels_ok = False
else:
    eval_sample["is_target"] = 0

if not labels_ok:
    print("[Labels] Unavailable → running in UNLABELED mode (label-based blocks skipped).")

# ------------------ Class imbalance ------------------
print("\n>>> [3] CLASS IMBALANCE")
if labels_ok:
    pos_rows = int(eval_sample["is_target"].sum())
    row_prev = float(eval_sample["is_target"].mean())
    tok_prev = (eval_sample[[KEY_COL, "is_target"]].groupby(KEY_COL)["is_target"].max()).mean()
    print(f"[Class] Row positives: {pos_rows} ({row_prev:.3%}) | Token prevalence: {tok_prev:.3%}")
    with open(SAVE_DIR / "class_imbalance.txt","w") as f:
        f.write(f"RowPos={pos_rows} RowPrev={row_prev:.6f} TokenPrev={tok_prev:.6f}\n")
    print("[Save] class_imbalance.txt")
else:
    print("[Class] Skipped (no labels).")

# ------------------ Heavy tails ------------------
print("\n>>> [4] HEAVY-TAIL DIAGNOSTICS (log1p + ECDFs)")
NUM_CAND = ensure_numeric(eval_sample, [c for c in BASE_NUMS if c in eval_sample.columns])

def plot_hist_log1p(col):
    x = eval_sample[col].astype(float).values
    x = np.nan_to_num(x, nan=0.0)
    xl = log1p_clip(x)
    m, lo, hi = bootstrap_ci_mean(x)
    ml, lol, hil = bootstrap_ci_mean(xl)
    # Hist log1p
    fig, ax = plt.subplots()
    ax.hist(xl, bins=80, alpha=0.85)
    ax.set_title(f"{col} — log1p\nmean={m:.3g} [{lo:.3g},{hi:.3g}] | log1p mean={ml:.3g} [{lol:.3g},{hil:.3g}]")
    ax.set_xlabel("log1p(value)")
    fig.tight_layout(); fig.savefig(SAVE_DIR / f"hist_log1p__{col}.png"); plt.close(fig)
    # ECDF
    xs, ps = ecdf(xl)
    fig, ax = plt.subplots()
    if len(xs): ax.plot(xs, ps, lw=2)
    ax.set_title(f"ECDF(log1p({col}))"); ax.set_xlabel("log1p(value)"); ax.set_ylabel("F(x)")
    fig.tight_layout(); fig.savefig(SAVE_DIR / f"ecdf_log1p__{col}.png"); plt.close(fig)

for col in NUM_CAND[:24]:
    plot_hist_log1p(col)
print(f"[Save] {min(24,len(NUM_CAND))}x (hist+ecdf) under {SAVE_DIR}")

# ------------------ Univariate vs target ------------------
print("\n>>> [5] UNIVARIATE SCREENING (AUPRC & Spearman)")
uni_df = pd.DataFrame(columns=["feature","AUPRC","Spearman"])
if labels_ok and eval_sample["is_target"].sum() > 0:
    rows = []
    y = eval_sample["is_target"].values.astype(int)
    for col in NUM_CAND:
        x = eval_sample[col].astype("float64").values
        xl = log1p_clip(x)
        try:
            ap = max(average_precision_score(y, xl), average_precision_score(y, -xl))
        except Exception:
            ap = np.nan
        try:
            sr = pd.Series(xl).rank(pct=True).corr(pd.Series(y), method="spearman")
        except Exception:
            sr = np.nan
        rows.append({"feature": col, "AUPRC": ap, "Spearman": sr})
    uni_df = pd.DataFrame(rows)
    if not uni_df.empty: uni_df = uni_df.sort_values("AUPRC", ascending=False)
    uni_df.to_csv(SAVE_DIR / "univariate_rank_aucpr.csv", index=False)
    print("[Save] univariate_rank_aucpr.csv")
    print(uni_df.head(10))
else:
    print("[Uni] Skipped (no labels).")

# ------------------ Correlation (Spearman, log1p) ------------------
print("\n>>> [6] CORRELATION (Spearman on log1p)")
FEAT_FOR_CORR = [c for c in NUM_CAND if eval_sample[c].notna().mean() > 0.99][:40]
if FEAT_FOR_CORR:
    Xcorr = pd.DataFrame({c: log1p_clip(eval_sample[c].values) for c in FEAT_FOR_CORR})
    corr = Xcorr.corr(method="spearman")
    corr.to_csv(SAVE_DIR / "spearman_corr_log1p_top40.csv")
    print("[Save] spearman_corr_log1p_top40.csv")
else:
    print("[Corr] No dense numeric features available.")

# ------------------ Time patterns ------------------
print("\n>>> [7] TIME-BASED PATTERNS (arrival curve + window deltas)")
if "_t_seconds" in eval_sample.columns and eval_sample["_t_seconds"].notna().any():
    buckets = (eval_sample["_t_seconds"].round().clip(lower=0, upper=30)).astype("Int64")
    ts_curve = buckets.value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.plot(ts_curve.index.astype(float), ts_curve.values, lw=2)
    ax.set_title("Arrival curve (rows per second)"); ax.set_xlabel("second"); ax.set_ylabel("rows")
    fig.tight_layout(); fig.savefig(SAVE_DIR / "arrival_curve_rows_per_second.png"); plt.close(fig)
    print("[Save] arrival_curve_rows_per_second.png")

def plot_window_ratio(a, b, name):
    if a not in eval_sample.columns or b not in eval_sample.columns: return
    da = log1p_clip(eval_sample[a].values) - log1p_clip(eval_sample[b].values)
    fig, ax = plt.subplots()
    ax.hist(da, bins=80, alpha=0.85)
    ax.set_title(f"{name}: log1p({a}) - log1p({b})")
    fig.tight_layout(); fig.savefig(SAVE_DIR / f"win_delta__{name}.png"); plt.close(fig)

for v in ["token_volume","sol_volume","total_count","buy_count","sell_count"]:
    for (A,B) in [(10,5),(20,10),(30,20)]:
        plot_window_ratio(f"{v}_w{A}_sum", f"{v}_w{B}_sum", f"{v}_{A}-{B}")
print("[Save] window delta plots (where columns exist)")

# ------------------ Token-level quick summary ------------------
print("\n>>> [8] TOKEN-LEVEL 30s SUMMARY (sanity)")
cols_small = [KEY_COL,"token_volume","sol_volume","buy_count","sell_count","total_count"]
cols_small = [c for c in cols_small if c in eval_sample.columns]
tmp = eval_sample[cols_small].copy()
if len(tmp) > MAX_ROWS_QC_SAMPLE:
    tmp = tmp.sample(MAX_ROWS_QC_SAMPLE, random_state=SEED)
g = tmp.groupby(KEY_COL).agg(["sum","max","mean"])
g.columns = ["__".join(c) for c in g.columns.to_flat_index()]
g = g.reset_index()
g.to_csv(SAVE_DIR / "token_level_summary_small.csv", index=False)
print("[Save] token_level_summary_small.csv")
if labels_ok:
    tok_target = eval_sample.groupby(KEY_COL)["is_target"].max().rename("y").reset_index()
    g2 = g.merge(tok_target, on=KEY_COL, how="left")
    cand = [c for c in g2.columns if c.startswith("token_volume__sum")]
    if cand:
        c0 = cand[0]
        x1 = log1p_clip(g2.loc[g2["y"]==1, c0])
        x0 = log1p_clip(g2.loc[g2["y"]==0, c0])
        fig, ax = plt.subplots()
        ax.hist(x1, bins=80, alpha=0.6, label="target=1")
        ax.hist(x0, bins=80, alpha=0.6, label="target=0")
        ax.legend(); ax.set_title(f"Token-level {c0} (log1p) by class")
        fig.tight_layout(); fig.savefig(SAVE_DIR / f"token_level_{c0}_by_class.png"); plt.close(fig)
        print(f"[Save] token_level_{c0}_by_class.png")

# ------------------ Leak-scan heuristics ------------------
print("\n>>> [9] LEAKAGE-SCAN (early vs late)")
def collect_aucpr(cols):
    out = []
    if not labels_ok or eval_sample["is_target"].sum() == 0:
        return pd.DataFrame(columns=["feature","AUPRC"])
    y = eval_sample["is_target"].values.astype(int)
    for c in cols:
        if c not in eval_sample.columns: continue
        xl = log1p_clip(eval_sample[c].values)
        try:
            ap = max(average_precision_score(y, xl), average_precision_score(y, -xl))
        except Exception:
            ap = np.nan
        out.append({"feature": c, "AUPRC": ap})
    df = pd.DataFrame(out)
    if not df.empty: df = df.sort_values("AUPRC", ascending=False)
    return df

early_cols = [f"{v}_w5_sum" for v in WIN_SUM_VARS if f"{v}_w5_sum" in eval_sample.columns]
late_cols  = []
for v in ["token_volume","sol_volume","total_count","buy_count","sell_count"]:
    a, b = f"{v}_w30_sum", f"{v}_w20_sum"
    if a in eval_sample.columns and b in eval_sample.columns:
        eval_sample[f"{v}_w30_w20_delta"] = log1p_clip(eval_sample[a]) - log1p_clip(eval_sample[b])
        late_cols.append(f"{v}_w30_w20_delta")

early_rank = collect_aucpr(early_cols)
late_rank  = collect_aucpr(late_cols)

if not early_rank.empty:
    early_rank.to_csv(SAVE_DIR / "leakscan_early_aucpr.csv", index=False)
    print("[Save] leakscan_early_aucpr.csv")
else:
    print("[Leak] Early-window skipped/empty.")
if not late_rank.empty:
    late_rank.to_csv(SAVE_DIR / "leakscan_late_aucpr.csv", index=False)
    print("[Save] leakscan_late_aucpr.csv")
else:
    print("[Leak] Late-delta skipped/empty.")

# =========================================================
# FEATURE ENGINEERING (token-level 30s) — reusable with models
# =========================================================
print("\n>>> [10] FEATURE ENGINEERING (token-level 30s)")

def aggregate_chunk(chunk: pd.DataFrame):
    drop_cols = [c for c in STRING_DROP if c in chunk.columns]
    if drop_cols: chunk = chunk.drop(columns=drop_cols)

    # time to seconds
    t_abs = chunk[TIME_COL].map(fast_parse_seconds) if TIME_COL in chunk.columns else pd.Series(np.nan, index=chunk.index)
    if t_abs.isna().all():
        chunk["_row_order"] = chunk.groupby(KEY_COL).cumcount().astype(float)
        t_abs = chunk["_row_order"].values.astype(float)
    chunk["_t_abs"] = t_abs

    # numeric cleanup
    num_cols = ensure_numeric(chunk, BASE_NUMS)
    chunk[num_cols] = chunk[num_cols].fillna(0)

    # time stats
    tstats = chunk.groupby(KEY_COL)["_t_abs"].agg(["min","max","count"]).reset_index()
    tstats = tstats.rename(columns={"min":"t_min_s","max":"t_max_s","count":"row_count"})
    chunk = chunk.merge(tstats[[KEY_COL,"t_min_s"]], on=KEY_COL, how="left")
    chunk["_t_rel"] = chunk["_t_abs"] - chunk["t_min_s"]

    # window sums
    win_aggs = []
    for W in WIN_LIST:
        mask = chunk["_t_rel"] <= W
        if not mask.any(): continue
        sub = chunk.loc[mask, [KEY_COL] + [v for v in WIN_SUM_VARS if v in chunk.columns]].copy()
        g = sub.groupby(KEY_COL).sum()
        g.columns = [f"{c}_w{W}_sum" for c in g.columns]
        g = g.reset_index()
        cnt = chunk.loc[mask, [KEY_COL]].groupby(KEY_COL).size().rename(f"row_count_w{W}").reset_index()
        g = g.merge(cnt, on=KEY_COL, how="left")
        win_aggs.append(g)

    # 30s sums/max/min
    num_agg = chunk[[KEY_COL] + num_cols].groupby(KEY_COL).agg(["sum","max","min"])
    num_agg.columns = [f"{c}__{stat}" for c, stat in num_agg.columns.to_flat_index()]
    num_agg = num_agg.reset_index()

    out = num_agg.merge(tstats, on=KEY_COL, how="left")
    for g in win_aggs: out = out.merge(g, on=KEY_COL, how="left")
    return out.fillna(0)

def combine_aggregates(df_list):
    all_df = pd.concat(df_list, ignore_index=True)
    sum_cols = [c for c in all_df.columns if c.endswith("__sum") or re.search(r"_w\d+_sum$", c)]
    max_cols = [c for c in all_df.columns if c.endswith("__max")]
    min_cols = [c for c in all_df.columns if c.endswith("__min")]
    cnt_cols = ["row_count"] + [c for c in all_df.columns if c.startswith("row_count_w")]
    keep = [KEY_COL,"t_min_s","t_max_s"] + sum_cols + max_cols + min_cols + cnt_cols
    keep = [c for c in keep if c in all_df.columns]
    all_df = all_df[keep]
    agg = {c:"sum" for c in sum_cols + cnt_cols}
    agg.update({c:"max" for c in max_cols}); agg.update({c:"min" for c in min_cols})
    agg["t_min_s"] = "min"; agg["t_max_s"] = "max"
    out = all_df.groupby(KEY_COL, as_index=False).agg(agg)
    out["lifespan_seconds"] = (out["t_max_s"] - out["t_min_s"]).clip(lower=0)
    return out.fillna(0)

def build_token_features_from_files(paths, verbose_every=3):
    agg_list = []
    for i, p in enumerate(paths, 1):
        print(f"[FE] Reading {p.name} ({i}/{len(paths)})")
        for chunk in pd.read_csv(p, chunksize=CHUNKSIZE):
            agg = aggregate_chunk(chunk)
            agg_list.append(agg); del chunk, agg
            gc.collect()
        if i % verbose_every == 0:
            print(f"   → partial grouped chunks: {len(agg_list)}")
    feats = combine_aggregates(agg_list)
    return feats

# Build FE for FULL (train-ish) & EVAL (test-ish)
full_paths = sorted(FULL_DIR.glob("september_2025_first30s_chunk_*.csv"))
assert len(full_paths) > 0, "No FULL CSVs found"
eval_paths = list_eval_paths(COMP_DIR)

train_feats = build_token_features_from_files(full_paths, verbose_every=2)
eval_feats  = build_token_features_from_files(eval_paths,  verbose_every=2)

# Keep only eval tokens in eval_feats (safety)
eval_ids = pd.read_csv(eval_paths[0]).pipe(lambda df: df[[KEY_COL]].drop_duplicates())
for p in eval_paths[1:]:
    t = pd.read_csv(p)[[KEY_COL]].drop_duplicates()
    eval_ids = pd.concat([eval_ids, t], axis=0, ignore_index=True).drop_duplicates()
eval_feats = eval_feats[eval_feats[KEY_COL].isin(set(eval_ids[KEY_COL]))].reset_index(drop=True)

print(f"[FE] train_feats: {train_feats.shape} | eval_feats: {eval_feats.shape}")

# Post-agg features
def add_post_agg_features(df):
    if "row_count" not in df.columns: df["row_count"] = 1
    rc = df["row_count"].clip(lower=1)

    for base in ["token_volume","sol_volume","token_quantity","sol_delta","market_cap_usd","buy_count","sell_count","total_count"]:
        csum = f"{base}__sum"
        if csum in df.columns: df[f"{base}__per_tx"] = df[csum] / rc

    if "buy_count__sum" in df.columns and "total_count__sum" in df.columns:
        df["buy_ratio"]  = df["buy_count__sum"]  / df["total_count__sum"].clip(lower=1)
    if "sell_count__sum" in df.columns and "total_count__sum" in df.columns:
        df["sell_ratio"] = df["sell_count__sum"] / df["total_count__sum"].clip(lower=1)

    if "lifespan_seconds" in df.columns:
        ls = df["lifespan_seconds"].replace(0, np.nan)
        for base in ["token_volume","sol_volume","total_count","buy_count","sell_count"]:
            csum = f"{base}__sum"
            if csum in df.columns: df[f"{base}__per_sec"] = df[csum] / ls
        df.fillna(0, inplace=True)

    for W in WIN_LIST:
        tc = f"total_count_w{W}_sum"; bc = f"buy_count_w{W}_sum"; sc = f"sell_count_w{W}_sum"
        sv = f"sol_volume_w{W}_sum";  tv = f"token_volume_w{W}_sum"; rcw= f"row_count_w{W}"
        if tc in df.columns:
            df[f"buy_ratio_w{W}"]   = df.get(bc, 0) / df[tc].clip(lower=1)
            df[f"sell_ratio_w{W}"]  = df.get(sc, 0) / df[tc].clip(lower=1)
            df[f"per_tx_solv_w{W}"] = df.get(sv, 0) / df[tc].clip(lower=1)
            df[f"per_tx_tokv_w{W}"] = df.get(tv, 0) / df[tc].clip(lower=1)
        if rcw in df.columns:
            df[f"avg_tokq_w{W}"] = df.get(f"token_quantity_w{W}_sum", 0) / df[rcw].clip(lower=1)
        # per-sec proxy inside window
        df[f"sol_per_sec_w{W}"]  = df.get(sv, 0) / max(1, W)
        df[f"tokv_per_sec_w{W}"] = df.get(tv, 0) / max(1, W)

    for a,b in [(10,5),(20,10),(30,20)]:
        for name, (cA, cB) in {
            "buy_ratio":    (f"buy_ratio_w{a}", f"buy_ratio_w{b}"),
            "sell_ratio":   (f"sell_ratio_w{a}", f"sell_ratio_w{b}"),
            "per_tx_solv":  (f"per_tx_solv_w{a}", f"per_tx_solv_w{b}"),
            "per_tx_tokv":  (f"per_tx_tokv_w{a}", f"per_tx_tokv_w{b}"),
            "sol_per_sec":  (f"sol_per_sec_w{a}", f"sol_per_sec_w{b}"),
            "tokv_per_sec": (f"tokv_per_sec_w{a}", f"tokv_per_sec_w{b}"),
        }.items():
            if cA in df.columns and cB in df.columns:
                df[f"{name}_delta_{a}-{b}"] = df[cA] - df[cB]

    for c in list(df.columns):
        if c.endswith("__sum") or re.search(r"_w\d+_sum$", c) or c.endswith("__max"):
            df[f"log1p_{c}"] = np.log1p(df[c].astype(float).clip(lower=0))
    return df

train_feats = add_post_agg_features(train_feats)
eval_feats  = add_post_agg_features(eval_feats)

# Save FE (CSV + info)
train_feats.to_csv(SAVE_DIR / "train_token_features.csv", index=False)
eval_feats.to_csv(SAVE_DIR / "eval_token_features.csv",  index=False)
print("[Save] train_token_features.csv & eval_token_features.csv")

# Preview top-importance proxy = univariate AUPRC on train (if labels available)
fe_uni = pd.DataFrame()
if labels_ok:
    # build token-level label frame from EVAL labels only if intersecting with FE (optional small insight)
    # Note: your official labels map is for train universe; we demonstrate on available eval merge cautiously.
    lab = pd.read_csv(TARGET_TOKENS_CSV)
    lid = [c for c in lab.columns if "mint" in c.lower() or "token" in c.lower()][0]
    lab = lab[[lid]].rename(columns={lid: KEY_COL})
    lab[KEY_COL] = lab[KEY_COL].astype(str)
    # For train_feats, mark positives if they appear in target list (proxy prevalence)
    tdf = train_feats.copy()
    tdf[KEY_COL] = tdf[KEY_COL].astype(str)
    tdf = tdf.merge(lab.assign(y=1), on=KEY_COL, how="left")
    tdf["y"] = tdf["y"].fillna(0).astype(int)
    y = tdf["y"].values
    if y.sum() > 0:
        cand_cols = [c for c in tdf.columns if c not in [KEY_COL,"y","t_min_s","t_max_s"]]
        rows = []
        for col in cand_cols[:200]:  # cap for speed
            x = tdf[col].astype("float64").values
            xl = log1p_clip(x)
            try:
                ap = max(average_precision_score(y, xl), average_precision_score(y, -xl))
            except Exception:
                ap = np.nan
            rows.append({"feature": col, "AUPRC": ap})
        fe_uni = pd.DataFrame(rows).sort_values("AUPRC", ascending=False)
        fe_uni.to_csv(SAVE_DIR / "train_fe_univariate_aucpr_top.csv", index=False)
        print("[Save] train_fe_univariate_aucpr_top.csv")
    else:
        print("[FE-Uni] Labels merged but no positives visible on train_feats key space.")
else:
    print("[FE-Uni] Skipped (no labels).")

# =========================================================
# REPORT (Markdown) — so people can skim & upvote :)
# =========================================================
print("\n>>> [11] GENERATE MARKDOWN REPORT")
md = []
md.append("# Alpha Radar — SOL Sprint: EDA + Feature Engineering\n")
md.append("**SOTA-flavored diagnostics** with heavy-tail awareness, PR emphasis under imbalance, and leakage-scan heuristics. This notebook exports all plots/CSVs under `eda_artifacts/` for quick review.\n")
md.append("## Dataset Overview\n")
md.append(f"- Eval rows (sampled or full): **{len(eval_sample):,}**")
md.append(f"- Unique tokens (sampled): **{eval_sample[KEY_COL].nunique():,}**")
md.append(f"- Labels available: **{labels_ok}**")
if labels_ok:
    md.append(f"- Row-level prevalence (sampled): **{eval_sample['is_target'].mean():.4%}**")
md.append("\n## Saved Artifacts\n")
md.append("- `schema_overview.csv` — dtypes, missingness, cardinality")
md.append("- Heavy-tail plots: `hist_log1p__*.png`, `ecdf_log1p__*.png`")
if labels_ok:
    md.append("- `univariate_rank_aucpr.csv` — AUPRC & Spearman per feature (rows)")
md.append("- `spearman_corr_log1p_top40.csv` — correlation matrix (dense features)")
md.append("- `arrival_curve_rows_per_second.png` — micro-burst timing")
md.append("- `win_delta__*.png` — window delta histograms")
md.append("- `token_level_summary_small.csv` — per-token 30s summary")
if labels_ok:
    md.append("- `leakscan_early_aucpr.csv`, `leakscan_late_aucpr.csv` — leakage heuristics")
md.append("- **Feature Engineering outputs**: `train_token_features.csv`, `eval_token_features.csv`")
if not fe_uni.empty:
    md.append("- `train_fe_univariate_aucpr_top.csv` — FE univariate signals (proxy)")
md.append("\n## Method Notes\n")
md.append("- Use **AUPRC** over ROC for severe imbalance; report **ECDF** to avoid histogram binning artifacts.")
md.append("- Robust transforms (`log1p`) mitigate heavy tails; rank-based **Spearman** for non-Gaussian.")
md.append("- **Leakage scan** contrasts early-window signals with late-only deltas.")
md.append("- FE includes per-tx/per-sec/ratios/deltas + `log1p_*` of heavy sums.\n")
with open(SAVE_DIR / "README_report.md", "w") as f:
    f.write("\n".join(md))
print("[Save] README_report.md")

print("\nALL DONE. Browse artifacts in:", SAVE_DIR)


