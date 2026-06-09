# === Refactor Cell A: Config · Imports · Seeds · Timer · (optional) DuckDB PRAGMAs ===

# ---- Imports (single source of truth) ----
import os, math, time, json, gc
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling / metrics
from sklearn.metrics import average_precision_score
from sklearn.isotonic import IsotonicRegression

# LightGBM
from lightgbm import LGBMClassifier


import duckdb
con = duckdb.connect()

# ---- Global config ----
SEED = 1337
np.random.seed(SEED)

# Runtime knobs
SMALL_RUN = False           # fast-iteration mode
MODEL_THREADS = os.cpu_count() if os.cpu_count() else 2

# EDA & slicing
SLICE_MODE = "by_days"      # or "full"
EDA_START_DAY = 0
EDA_DAYS = 60               # default to 60-day window for better rolling coverage
FAST_PLOT_N = 200_000       # cap plots for speed

# Ops capacity (for P/R@k)
OPS = {"review_capacity_rate": 0.05}  # 5% default; tune to your ops queue

# Pandas options
pd.options.mode.copy_on_write = True
pd.options.display.max_columns = 200
pd.options.display.width = 180

# Plot style
sns.set_theme(context="notebook", style="whitegrid")

# ---- Tiny timer context ----
class Timer:
    def __init__(self, label): self.label = label
    def __enter__(self):
        self.t0 = time.time(); print(f"[⏱️] {self.label} ...", end=" ")
        return self
    def __exit__(self, *exc):
        dt = time.time() - self.t0
        print(f"done in {dt:.2f}s")

def set_seed(seed: int = SEED):
    np.random.seed(seed)
    # LightGBM and sklearn rely on passed random_state; set per estimator.

# ---- (Optional) DuckDB PRAGMAs if a connection is already available ----
if duckdb is not None and "con" in globals():
    con.execute(f"PRAGMA threads={MODEL_THREADS};")
    # Adjust memory limit to Kaggle constraints if desired
    con.execute("PRAGMA memory_limit='3GB';")
    con.execute("PRAGMA temp_directory='/kaggle/temp';")


#cell1

# Metrics
def aucpr(y_true, y_score) -> float:
    return float(average_precision_score(y_true, y_score))

def precision_recall_at_k(y_true, y_score, k_rate: float):
    n = len(y_score)
    k = max(1, int(math.floor(k_rate * n)))
    order = np.argsort(-y_score)
    topk = order[:k]
    tp = int(np.asarray(y_true)[topk].sum())
    prec = tp / k
    rec = tp / max(1, int(np.asarray(y_true).sum()))
    return prec, rec

# forward-chaining CV with purge
def build_time_folds(ts_days: np.ndarray, n_folds: int = 5, purge_days: int = 1):
    """
    Returns a list of dicts: {"train_idx":..., "valid_idx":...}
    Splits by unique day quantiles; training always precedes validation; purge gap enforced.
    """
    days = np.sort(np.unique(ts_days))
    folds = []
    # Cut points using quantiles
    cuts = np.quantile(days, np.linspace(0, 1, n_folds + 1)[1:-1], interpolation="nearest")
    cuts = np.unique(cuts)  # guard against duplicates
    prev_cut = days.min()
    for i, cp in enumerate(cuts):
        train_mask = ts_days < (cp - purge_days)
        valid_mask = (ts_days >= cp) & (ts_days < (cuts[i+1] if i+1 < len(cuts) else days.max()+1))
        tr_idx = np.flatnonzero(train_mask)
        va_idx = np.flatnonzero(valid_mask)
        if tr_idx.size and va_idx.size:
            folds.append({"train_idx": tr_idx, "valid_idx": va_idx})
    while len(folds) < n_folds and len(folds) > 0:
        folds.append(folds[-1])
    return folds

# ----- Monotone constraint vector -----
def build_monotone(features, monotone_on=("TransactionAmt",)):
    on = set(monotone_on)
    return [1 if f in on else 0 for f in features]

# Normalization
def ensure_norm_id_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Adds device_brand and email_tld if missing; returns a new frame with the columns ensured."""
    df = df_raw.copy()
    if "device_brand" not in df.columns:
        df["device_brand"] = (
            df["DeviceInfo"].astype(str).str.lower()
              .str.split("/", n=1, expand=False).str[0]
              .str.split(" ",  n=1, expand=False).str[0]
        )
    if "email_tld" not in df.columns:
        df["email_tld"] = (
            df["P_emaildomain"].astype(str).str.lower()
              .str.split(".", expand=False).str[-1]
        )
    return df

# Encoders
def fit_mappings(train_df: pd.DataFrame, cat: str, y: str, m: float = 50.0):
    """Return Series: category -> WOE (smoothed against global)."""
    g = train_df[y].mean()
    grp = train_df.groupby(cat, observed=False, dropna=False)[y].agg(["count","sum"]).rename(columns={"sum":"pos"})
    rate = (grp["pos"] + m * g) / (grp["count"] + m)
    odds_c = rate / (1.0 - rate + 1e-12)
    odds_g = g / (1.0 - g + 1e-12)
    woe = np.log(odds_c / (odds_g + 1e-12))
    woe.index = woe.index.astype(object)
    return woe

def fit_hier_maps(train_df: pd.DataFrame, child: str, parent: str, y: str, m: float = 50.0):
    """
    Returns two Series indexed by child:
      - child_woe: smoothed child WOE
      - backoff_woe: if count==1 → 50/50 blend with parent rate; else child smoothed rate
    """
    g = train_df[y].mean()
    par = train_df.groupby(parent, observed=False, dropna=False)[y].agg(count="size", pos="sum")
    par_rate = (par["pos"] + m * g) / (par["count"] + m)
    ch = (train_df.groupby([child,parent], observed=False, dropna=False)[y]
          .agg(count="size", pos="sum")).reset_index()
    ch["rate_child"] = (ch["pos"] + m * g) / (ch["count"] + m)
    ch = ch.merge(par_rate.rename("rate_parent").to_frame().reset_index(), on=parent, how="left")

    cnt = ch["count"].to_numpy()
    rc  = ch["rate_child"].to_numpy()
    rp  = ch["rate_parent"].to_numpy()
    r_back = rc.copy()
    mask_1 = cnt == 1
    r_back[mask_1] = 0.5*rc[mask_1] + 0.5*rp[mask_1]

    odds = lambda r: r / (1.0 - r + 1e-12)
    w_child = np.log(odds(rc)    / (g / (1.0 - g + 1e-12) + 1e-12))
    w_back  = np.log(odds(r_back)/ (g / (1.0 - g + 1e-12) + 1e-12))

    child_key = ch[child].astype(object)
    return pd.Series(w_child, index=child_key), pd.Series(w_back, index=child_key)

def apply_encoders_for_fold(df_raw: pd.DataFrame, tr_idx, va_idx, label_col: str = "isFraud",
                            clip: float = 3.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit maps on TRAIN rows only; transform TRAIN and VALID; return enc_tr, enc_va (float32).
    Emits columns:
      device_brand_woe, email_tld_woe,
      device_backoff_woe, email_backoff_woe,
      DeviceInfo_woe, P_emaildomain_woe
    """
    df = ensure_norm_id_columns(df_raw)
    tr = df.loc[tr_idx, ["device_brand","email_tld","DeviceInfo","P_emaildomain", label_col]].copy()
    va = df.loc[va_idx, ["device_brand","email_tld","DeviceInfo","P_emaildomain", label_col]].copy()

    # normalized
    w_brand = fit_mappings(tr.rename(columns={"device_brand":"cat"}), "cat", label_col)
    w_tld   = fit_mappings(tr.rename(columns={"email_tld":"cat"}),   "cat", label_col)
    # hierarchical
    _, w_back_dev = fit_hier_maps(tr.rename(columns={"DeviceInfo":"child","device_brand":"parent"}), "child", "parent", label_col)
    _, w_back_eml = fit_hier_maps(tr.rename(columns={"P_emaildomain":"child","email_tld":"parent"}),  "child", "parent", label_col)
    # raw field WOEs
    w_devinfo = fit_mappings(tr.rename(columns={"DeviceInfo":"cat"}),    "cat", label_col)
    w_email   = fit_mappings(tr.rename(columns={"P_emaildomain":"cat"}), "cat", label_col)

    def m(series, mp):
        return series.map(mp).fillna(0.0).astype("float32").clip(-clip, clip)

    enc_tr = pd.DataFrame(index=tr.index)
    enc_va = pd.DataFrame(index=va.index)
    enc_tr["device_brand_woe"]   = m(tr["device_brand"],  w_brand)
    enc_va["device_brand_woe"]   = m(va["device_brand"],  w_brand)
    enc_tr["email_tld_woe"]      = m(tr["email_tld"],     w_tld)
    enc_va["email_tld_woe"]      = m(va["email_tld"],     w_tld)
    enc_tr["device_backoff_woe"] = m(tr["DeviceInfo"],    w_back_dev)
    enc_va["device_backoff_woe"] = m(va["DeviceInfo"],    w_back_dev)
    enc_tr["email_backoff_woe"]  = m(tr["P_emaildomain"], w_back_eml)
    enc_va["email_backoff_woe"]  = m(va["P_emaildomain"], w_back_eml)
    enc_tr["DeviceInfo_woe"]     = m(tr["DeviceInfo"],    w_devinfo)
    enc_va["DeviceInfo_woe"]     = m(va["DeviceInfo"],    w_devinfo)

    return enc_tr.astype("float32"), enc_va.astype("float32")


# === Cell 3: Load → Join → Cache → Apply small-slice → Tiny EDA (DuckDB + pandas) ===
# Purpose:
# - Read CSVs once, do the big LEFT JOIN + sort in DuckDB
# - Add a coarse day clock (ts_day), cache as TEMP tables for reuse
# - Apply SMALL_RUN slicing (by contiguous days OR by random fraction) for quick iteration
# - Return only small summaries to pandas

# 1) Cache raw CSVs as TEMP TABLES (read once; reuse everywhere)
con.sql(f"""
CREATE OR REPLACE TEMP TABLE train_transaction_tbl AS
SELECT * FROM read_csv_auto('{DATA_DIR}/train_transaction.csv', AUTO_DETECT=TRUE);
""")
con.sql(f"""
CREATE OR REPLACE TEMP TABLE train_identity_tbl AS
SELECT * FROM read_csv_auto('{DATA_DIR}/train_identity.csv', AUTO_DETECT=TRUE);
""")

# 2) Enriched+sorted join cached as TEMP TABLE
con.sql("""
CREATE OR REPLACE TEMP TABLE train_enriched_tbl AS
SELECT
  tt.*,
  ti.* EXCLUDE (TransactionID),  -- avoid duplicate key
  CAST(FLOOR(tt.TransactionDT / (24*60*60)) AS BIGINT) AS ts_day
FROM train_transaction_tbl tt
LEFT JOIN train_identity_tbl ti USING (TransactionID)
ORDER BY tt.TransactionDT;
""")

# 3) Apply small-slice controls from Cell 1
if SMALL_RUN and SLICE_MODE == "by_days":
    where_clause = sql_time_slice_clause("ts_day")
    con.sql(f"""
    CREATE OR REPLACE TEMP TABLE train_slice_tbl AS
    SELECT * FROM train_enriched_tbl
    WHERE {where_clause};
    """)
elif SMALL_RUN and SLICE_MODE == "by_fraction":
    # Use DuckDB's Bernoulli sampler for a fast random subset (repeatable via SEED)
    frac_pct = max(0.0001, min(1.0, EDA_FRACTION)) * 100.0
    con.sql(f"""
    CREATE OR REPLACE TEMP TABLE train_slice_tbl AS
    SELECT * FROM train_enriched_tbl
    USING SAMPLE BERNOULLI ({frac_pct}) REPEATABLE ({SEED});
    """)
else:
    con.sql("CREATE OR REPLACE TEMP TABLE train_slice_tbl AS SELECT * FROM train_enriched_tbl;")

# 4) Global info on the (possibly sliced) working set
info = con.sql("""
SELECT 
  COUNT(*)::BIGINT AS n_rows,
  SUM(CASE WHEN isFraud=1 THEN 1 ELSE 0 END)::BIGINT AS n_fraud,
  AVG(CAST(isFraud AS DOUBLE)) AS fraud_rate,
  MIN(CAST(TransactionDT AS BIGINT)) AS tdt_min,
  MAX(CAST(TransactionDT AS BIGINT)) AS tdt_max,
  COUNT(DISTINCT ts_day)::BIGINT AS days_span
FROM train_slice_tbl;
""").to_df().iloc[0]

print(f"[INFO] working set shape: {info.n_rows:,} rows | fraud_rate={info.fraud_rate:.4f}")
print(f"[INFO] TransactionDT range: {int(info.tdt_min)} → {int(info.tdt_max)} (≈ {int(info.days_span)} days)")
print(f"[INFO] source tables cached: train_transaction_tbl, train_identity_tbl, train_enriched_tbl, train_slice_tbl")

# 5) Inspect a few anchor columns (present in this fork)
requested_anchors = [
    "TransactionAmt", "card1", "card2", "card3", "card4",
    "addr1", "addr2", "dist1", "dist2",
    "P_emaildomain", "R_emaildomain",
    "DeviceInfo", "DeviceType"
]
cols_df = con.sql("PRAGMA table_info('train_slice_tbl');").to_df()
present_cols = set(cols_df["name"].tolist())
anchor_cols = [c for c in requested_anchors if c in present_cols]

# Null rates & unique counts via SQL (bring back tiny frames)
null_rows, unique_rows = [], []
for c in anchor_cols:
    nr = con.sql(f"""
        SELECT SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS null_rate
        FROM train_slice_tbl;
    """).to_df()["null_rate"][0]
    uq = con.sql(f"SELECT COUNT(DISTINCT {c})::BIGINT AS n_unique FROM train_slice_tbl;").to_df()["n_unique"][0]
    null_rows.append((c, nr)); unique_rows.append((c, uq))

null_rates = (pd.DataFrame(null_rows, columns=["column","null_rate"])
              .sort_values("null_rate", ascending=False))
unique_counts = (pd.DataFrame(unique_rows, columns=["column","n_unique"])
                 .sort_values("n_unique", ascending=False))

print("\n[INFO] anchor column null rates (top 8):")
display(null_rates.head(8).style.format({"null_rate": "{:.2%}"}))

print("\n[INFO] anchor column unique counts (top 8):")
display(unique_counts.head(8))

# 6) Per-day volume & fraud rate (first 10 days of the working set)
daily = con.sql("""
SELECT 
  ts_day,
  COUNT(*)::BIGINT AS n,
  AVG(CAST(isFraud AS DOUBLE)) AS fraud_rate,
  median(TransactionAmt) AS amt_median
FROM train_slice_tbl
GROUP BY 1
ORDER BY 1
LIMIT 10;
""").to_df()

print("\n[INFO] daily volume & fraud rate (first 10 days in working set):")
display(daily.style.format({"fraud_rate": "{:.2%}", "amt_median": "{:.2f}"}))



# === Cell 4 (patched): Targeted EDA with seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# 1) Pull minimal columns once from DuckDB
min_cols = [
    "TransactionDT", "ts_day", "isFraud", "TransactionAmt",
    "DeviceInfo", "P_emaildomain", "R_emaildomain", "card1"
]
cols_df = con.sql("PRAGMA table_info('train_slice_tbl');").to_df()
present_cols = set(cols_df["name"].tolist())
use_cols = [c for c in min_cols if c in present_cols]

df = con.sql(f"SELECT {', '.join(use_cols)} FROM train_slice_tbl").to_df()
df = df.sort_values("TransactionDT").reset_index(drop=True)
df["isFraud"] = df["isFraud"].astype(np.uint8)
if "TransactionAmt" in df:
    df["amt_log1p"] = np.log1p(df["TransactionAmt"].clip(lower=0))

# Helper to plot safely with CoW on elsewhere
def plot_safe(func, *args, **kwargs):
    with pd.option_context("mode.copy_on_write", False):
        return func(*args, **kwargs)

# 2) Amount distribution vs label (log-scale)
if "amt_log1p" in df:
    df_plot = df.loc[:, ["amt_log1p", "isFraud"]].dropna().copy()
    if len(df_plot) > 200_000:
        df_plot = df_plot.sample(200_000, random_state=SEED).copy()
    def _plot_amount_hist():
        ax = sns.histplot(
            data=df_plot, x="amt_log1p", hue="isFraud",
            stat="density", common_norm=False, bins=60,
            element="step", fill=False
        )
        ax.set_title("Log(1+Amount) Distribution by Label")
        ax.set_xlabel("log1p(TransactionAmt)"); ax.set_ylabel("Density")
        plt.show()
    plot_safe(_plot_amount_hist)

# 3) Fraud-rate vs amount deciles
if "TransactionAmt" in df:
    amt = df["TransactionAmt"].astype(float)
    ranks = amt.rank(method="first")
    deciles = pd.qcut(ranks, 10, labels=False, duplicates="drop")
    tmp = pd.DataFrame({"decile": deciles, "isFraud": df["isFraud"]}).dropna().copy()
    def _plot_deciles():
        ax = sns.lineplot(data=tmp.groupby("decile", as_index=False)["isFraud"].mean(),
                          x="decile", y="isFraud", marker="o")
        ax.set_title("Fraud Rate by Amount Decile")
        ax.set_xlabel("Amount Decile (0=low … 9=high)"); ax.set_ylabel("Fraud Rate")
        plt.show()
    plot_safe(_plot_deciles)

# 4) Daily fraud-rate stability
if "ts_day" in df:
    daily = df.groupby("ts_day", as_index=False).agg(
        n=("isFraud","size"),
        fraud_rate=("isFraud","mean"),
        amt_med=("TransactionAmt","median") if "TransactionAmt" in df else ("isFraud","mean")
    ).copy()
    def _plot_daily():
        ax = sns.lineplot(data=daily, x="ts_day", y="fraud_rate", marker="o")
        ax.set_title("Daily Fraud Rate (working slice)")
        ax.set_xlabel("ts_day"); ax.set_ylabel("Fraud Rate")
        plt.show()
    plot_safe(_plot_daily)

# 5) New vs returning device
if "DeviceInfo" in df and "ts_day" in df:
    d_first = (df[["DeviceInfo","ts_day"]].dropna()
               .groupby("DeviceInfo", as_index=False)["ts_day"].min()
               .rename(columns={"ts_day":"first_day"}))
    tmp2 = df.merge(d_first, on="DeviceInfo", how="left")
    tmp2["is_new_device"] = (tmp2["ts_day"] == tmp2["first_day"]).astype(np.uint8)
    new_ret = tmp2.groupby("is_new_device", as_index=False)["isFraud"].mean().copy()
    new_ret["group"] = np.where(new_ret["is_new_device"] == 1, "NEW device", "RETURNING device")
    def _plot_newret():
        ax = sns.barplot(data=new_ret, x="group", y="isFraud")
        ax.set_title("Fraud Rate: New vs Returning Device")
        ax.set_xlabel(""); ax.set_ylabel("Fraud Rate")
        plt.show()
    plot_safe(_plot_newret)

# 6) Top email domains
if "P_emaildomain" in df:
    topk = df["P_emaildomain"].value_counts(dropna=True).head(10).index.tolist()
    em = df[df["P_emaildomain"].isin(topk)].copy()
    em_join = (em.groupby("P_emaildomain", as_index=False)["isFraud"].mean()
                 .merge(em["P_emaildomain"].value_counts()
                        .rename_axis("P_emaildomain").reset_index(name="n"),
                        on="P_emaildomain", how="left")
                 .sort_values("n", ascending=False))
    def _plot_email():
        ax = sns.barplot(data=em_join, x="P_emaildomain", y="isFraud")
        ax.set_title("Fraud Rate by Top P_emaildomain (top 10)")
        ax.set_xlabel("P_emaildomain"); ax.set_ylabel("Fraud Rate")
        plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        plt.show()
    plot_safe(_plot_email)

print("EDA complete (plots rendered with CoW-safe copies).")



df


# === Cell 5 (clean): Forward-chaining CV + leak-safe behavioral/velocity features in pandas ===
# Assumptions (IEEE-CIS standard columns present):
#   TransactionID, TransactionDT, ts_day, isFraud, TransactionAmt, card1, card2, DeviceInfo, P_emaildomain


# 1) Pull the working slice to pandas (only the columns we need)
df = con.sql("""
    SELECT 
        TransactionID, TransactionDT, ts_day, isFraud,
        TransactionAmt, card1, card2, DeviceInfo, P_emaildomain
    FROM train_slice_tbl
""").to_df()

# 2) Order chronologically and set a datetime-like clock for rolling windows
df = df.sort_values("TransactionDT").reset_index(drop=True)
df["isFraud"] = df["isFraud"].astype(np.uint8)
df["event_time"] = pd.to_datetime(df["TransactionDT"], unit="s", origin="unix")

# 3) Primary entity anchor (fixed for simplicity)
ENTITY = "card1"
g = df.groupby(ENTITY, dropna=False, sort=False)

# 4) As-of features (strictly past-looking)
# (a) first occurrence flag per entity
first_time = g["event_time"].transform("min")
df["is_new_entity"] = (df["event_time"] == first_time).astype(np.uint8)

# (b) time since previous txn by same entity (seconds); huge value for first txn
prev_time = g["event_time"].shift(1)
df["time_since_prev_sec"] = (df["event_time"] - prev_time).dt.total_seconds().fillna(1e12).astype("float64")

# (c) rolling counts in past 1D and 7D (exclude current by subtracting 1)
# --- PATCH for Cell 5: use DataFrame-groupby rolling with `on="event_time"` ---

# (c) Rolling counts in past 1D and 7D (exclude current by subtracting 1)
base_col = "TransactionID"

cnt_1d = (
    g.rolling("1D", on="event_time")[base_col]
     .count()
     .reset_index(level=0, drop=True)
     .astype("float64")
     - 1.0
)
cnt_7d = (
    g.rolling("7D", on="event_time")[base_col]
     .count()
     .reset_index(level=0, drop=True)
     .astype("float64")
     - 1.0
)
df["cnt_1d"] = np.clip(cnt_1d.to_numpy(), 0.0, None)
df["cnt_7d"] = np.clip(cnt_7d.to_numpy(), 0.0, None)

# (d) Rolling amount aggregates: past-1D sum (exclude current) and 7D median (includes current; acceptable)
amt_sum_1d = (
    g.rolling("1D", on="event_time")["TransactionAmt"]
     .sum()
     .reset_index(level=0, drop=True)
     .astype("float64")
     - df["TransactionAmt"].astype("float64").to_numpy()
)
amt_med_7d = (
    g.rolling("7D", on="event_time")["TransactionAmt"]
     .median()
     .reset_index(level=0, drop=True)
     .astype("float64")
)

df["amt_sum_1d"] = np.nan_to_num(amt_sum_1d, nan=0.0, posinf=0.0, neginf=0.0)
df["amt_median_7d"] = np.nan_to_num(amt_med_7d, nan=0.0, posinf=0.0, neginf=0.0)

# 5) Minimal feature matrix (ready for baselines)
X_cols = [
    "TransactionAmt",
    "is_new_entity", "time_since_prev_sec",
    "cnt_1d", "cnt_7d",
    "amt_sum_1d", "amt_median_7d"
]
label_col = "isFraud"
df_feat = df[["TransactionID", "ts_day", label_col] + X_cols].copy()

# 6) Forward-chaining CV (2 folds) with a 1-day purge gap between train and valid
unique_days = np.sort(df_feat["ts_day"].unique())
n_days = len(unique_days)
def _qday(q): 
    return unique_days[int(np.floor(q * (n_days - 1)))]

cut1 = _qday(0.50)
cut2 = _qday(0.75)
PURGE_DAYS = 1

folds = []
# Fold 1: train < cut1 - purge ; valid ∈ [cut1, cut2)
train_idx_1 = np.flatnonzero(df_feat["ts_day"].values < (cut1 - PURGE_DAYS))
valid_idx_1 = np.flatnonzero((df_feat["ts_day"].values >= cut1) & (df_feat["ts_day"].values < cut2))
folds.append({"train_idx": train_idx_1, "valid_idx": valid_idx_1})
# Fold 2: train < cut2 - purge ; valid ∈ [cut2, max]
train_idx_2 = np.flatnonzero(df_feat["ts_day"].values < (cut2 - PURGE_DAYS))
valid_idx_2 = np.flatnonzero(df_feat["ts_day"].values >= cut2)
folds.append({"train_idx": train_idx_2, "valid_idx": valid_idx_2})

print(f"[CV] ENTITY='{ENTITY}' | folds={len(folds)}")
for i, f in enumerate(folds, 1):
    print(f"  Fold {i}: train_n={len(f['train_idx']):,} | valid_n={len(f['valid_idx']):,}")
print(f"[FE] Feature matrix: {df_feat.shape[0]:,} rows × {len(X_cols)} features")



# === Cell 6A: Baselines on the sliced set — Rules Score + Logistic Regression ===
# Objective:
# - Establish quick baselines that mirror real first passes:
#   1) A hand-crafted "rules score" from classic risk signals
#   2) A simple logistic regression on our engineered features
# - Evaluate on each forward-chaining fold using AUCPR (primary) and precision/recall@k (ops)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (7, 4)

# Helpers from Cell 2 are available: aucpr, precision_recall_at_k, threshold_for_topk
K = OPS["review_capacity_rate"]

# --- 1) Define a hand-crafted rules score (train-fit scaling, then apply to valid to avoid leakage) ---
def compute_rules_raw(mat: pd.DataFrame) -> np.ndarray:
    """
    Heuristic risk score from a few features (no training):
      - amount ↑ (log1p)
      - new entity ↑
      - short time_since_prev ↑  (use 1 / (1 + log1p(gap)))
      - recent counts ↑
      - recent amount sums/medians ↑
    Returns an unscaled raw score (will be min-max scaled on the TRAIN fold only).
    """
    amt = np.log1p(mat["TransactionAmt"].astype("float64"))
    is_new = mat["is_new_entity"].astype("float64")
    gap = 1.0 / (1.0 + np.log1p(mat["time_since_prev_sec"].astype("float64")))
    c1 = np.log1p(mat["cnt_1d"].astype("float64"))
    c7 = np.log1p(mat["cnt_7d"].astype("float64"))
    s1 = np.log1p(np.clip(mat["amt_sum_1d"].astype("float64"), 0, None))
    m7 = np.log1p(np.clip(mat["amt_median_7d"].astype("float64"), 0, None))
    # weights are heuristic; they just capture relative importance
    raw = 0.6*amt + 0.9*is_new + 0.8*gap + 0.6*c1 + 0.3*c7 + 0.4*s1 + 0.2*m7
    return raw

def minmax_scale_train_apply(raw_train: np.ndarray, raw_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.min(raw_train), np.max(raw_train)
    rng = max(hi - lo, 1e-9)
    return (raw_train - lo) / rng, (raw_valid - lo) / rng

# --- 2) Logistic regression baseline (with standardization and class weighting) ---
def fit_logit_predict(train_X: pd.DataFrame, train_y: np.ndarray, valid_X: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_X.values)
    Xva = scaler.transform(valid_X.values)
    lr = LogisticRegression(
        penalty="l2",
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        n_jobs=None  # LogisticRegression ignores n_jobs for lbfgs; fine for our size
    )
    lr.fit(Xtr, train_y)
    return lr.predict_proba(Xva)[:, 1]

# --- 3) Loop over folds and evaluate ---
rows = []
for i, f in enumerate(folds, 1):
    tr_idx = f["train_idx"]; va_idx = f["valid_idx"]

    y_tr = df_feat.loc[tr_idx, label_col].to_numpy().astype(np.uint8)
    y_va = df_feat.loc[va_idx, label_col].to_numpy().astype(np.uint8)

    # --- Rules score ---
    raw_tr = compute_rules_raw(df_feat.loc[tr_idx, X_cols])
    raw_va = compute_rules_raw(df_feat.loc[va_idx, X_cols])
    s_tr, s_va = minmax_scale_train_apply(raw_tr, raw_va)

    rules_aupr = aucpr(y_va, s_va)
    p_at_k, r_at_k = precision_recall_at_k(y_va, s_va, K)

    rows.append({
        "fold": i, "model": "rules",
        "aupr": rules_aupr, "prec@k": p_at_k, "recall@k": r_at_k
    })

    # --- Logistic regression ---
    yhat_lr = fit_logit_predict(df_feat.loc[tr_idx, X_cols], y_tr,
                                df_feat.loc[va_idx, X_cols])
    lr_aupr = aucpr(y_va, yhat_lr)
    p_at_k, r_at_k = precision_recall_at_k(y_va, yhat_lr, K)

    rows.append({
        "fold": i, "model": "logistic",
        "aupr": lr_aupr, "prec@k": p_at_k, "recall@k": r_at_k
    })

res = pd.DataFrame(rows)
summary = (res.groupby("model", as_index=False)
             .agg(aupr_mean=("aupr","mean"),
                  aupr_std=("aupr","std"),
                  prec_k_mean=("prec@k","mean"),
                  recall_k_mean=("recall@k","mean")))

print("[BASELINES] AUCPR and P/R@k per fold:")
display(res.pivot(index="fold", columns="model", values="aupr").style.format("{:.4f}"))
print("\n[SUMMARY] Mean ± std across folds (k = {:.0%}):".format(K))
display(summary.style.format({
    "aupr_mean":"{:.4f}", "aupr_std":"{:.4f}",
    "prec_k_mean":"{:.4f}", "recall_k_mean":"{:.4f}"
}))

# Quick plot of AUCPR by model and fold
fig, ax = plt.subplots()
sns.barplot(data=res, x="fold", y="aupr", hue="model", ax=ax)
ax.set_title("Baseline AUCPR by Fold (sliced set)")
ax.set_xlabel("Fold"); ax.set_ylabel("AUCPR")
plt.tight_layout(); plt.show()


# === Cell 6B: LightGBM first pass (AUCPR-focused) on forward-chaining folds ===
# Objective:
# - Train LGBMClassifier per fold with AUCPR focus and mild regularization for imbalance
# - Evaluate AUCPR + precision/recall@k on each valid fold
# - Keep the best fold model for later explainability

from lightgbm import LGBMClassifier
import lightgbm as lgb

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (7, 4)

# Inputs prepared earlier:
# - df_feat: feature matrix with ['TransactionID','ts_day', label_col] + X_cols
# - X_cols: list of feature column names
# - folds: list of {'train_idx', 'valid_idx'}
# - OPS['review_capacity_rate'] -> K
# - COST if/when we do profit later
label_col = "isFraud"
K = OPS["review_capacity_rate"]

# LightGBM configuration (balanced, mild regularization, AUCPR-centric)
lgbm = LGBMClassifier(
    objective="binary",
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=200,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    class_weight="balanced",
    n_jobs=MODEL_THREADS,
    random_state=SEED
)

rows = []
models = []
best_model = None
best_fold = None
best_aupr = -np.inf

for i, f in enumerate(folds, 1):
    tr_idx, va_idx = f["train_idx"], f["valid_idx"]

    Xtr = df_feat.loc[tr_idx, X_cols].astype("float32")
    ytr = df_feat.loc[tr_idx, label_col].to_numpy(dtype=np.uint8)
    Xva = df_feat.loc[va_idx, X_cols].astype("float32")
    yva = df_feat.loc[va_idx, label_col].to_numpy(dtype=np.uint8)

    # Fit with early stopping on AUCPR
    mdl = lgbm.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

    # Predict on validation using best iteration
    yhat_va = mdl.predict_proba(Xva, raw_score=False)[:, 1]

    fold_aupr = aucpr(yva, yhat_va)
    p_at_k, r_at_k = precision_recall_at_k(yva, yhat_va, K)

    rows.append({
        "fold": i,
        "model": "lgbm",
        "aupr": fold_aupr,
        "prec@k": p_at_k,
        "recall@k": r_at_k,
        "best_iter": getattr(mdl, "best_iteration_", None)
    })
    models.append(mdl)

    if fold_aupr > best_aupr:
        best_aupr = fold_aupr
        best_model = mdl
        best_fold = i

res_lgbm = pd.DataFrame(rows)
summary = pd.DataFrame({
    "aupr_mean":   [res_lgbm["aupr"].mean()],
    "aupr_std":    [res_lgbm["aupr"].std()],
    "prec_k_mean": [res_lgbm["prec@k"].mean()],
    "recall_k_mean":[res_lgbm["recall@k"].mean()],
})

print("\n[SUMMARY] Mean ± std across folds (k = {:.0%}):".format(K))
display(summary.style.format({
    "aupr_mean":"{:.4f}", "aupr_std":"{:.4f}",
    "prec_k_mean":"{:.4f}", "recall_k_mean":"{:.4f}"
}))

print("[LGBM] AUCPR and P/R@k per fold:")
display(res_lgbm[["fold","aupr","prec@k","recall@k","best_iter"]].style.format({
    "aupr":"{:.4f}", "prec@k":"{:.4f}", "recall@k":"{:.4f}"
}))

# Quick plot of AUCPR by fold
fig, ax = plt.subplots()
sns.barplot(data=res_lgbm, x="fold", y="aupr", color=None, hue=None, ax=ax)
ax.set_title("LightGBM AUCPR by Fold (sliced set)")
ax.set_xlabel("Fold"); ax.set_ylabel("AUCPR")
plt.tight_layout(); plt.show()

print(f"[KEEP] Stored best LightGBM from fold {best_fold} (AUCPR={best_aupr:.4f}) as `best_model`.")



# === Cell 7: Calibration (Isotonic), reliability curves, and operating thresholds ===
# Objective:
# - Per fold: fit isotonic on TRAIN scores, apply to VALID scores
# - Plot reliability curve on VALID (pre vs post calibration)
# - Report AUCPR pre/post calibration
# - Compute capacity threshold (top-k) and profit-max threshold

from sklearn.isotonic import IsotonicRegression

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (6.5, 5)

label_col = "isFraud"
K = OPS["review_capacity_rate"]

def reliability_table(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    """Bin scores into equal-width bins on [0,1]; return bin-wise mean pred and empirical rate."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_score, bins, right=True)
    # clamp to [1, n_bins]
    idx = np.clip(idx, 1, n_bins)
    dfb = pd.DataFrame({"bin": idx, "y": y_true, "p": y_score})
    out = dfb.groupby("bin", as_index=False).agg(
        mean_pred=("p","mean"),
        frac_pos=("y","mean"),
        count=("y","size")
    )
    out["bin_left"] = bins[out["bin"] - 1]
    out["bin_right"] = bins[out["bin"]]
    return out

cal_rows = []
cal_models = []  # store per-fold isotonic models
reliability_plots = []  # optional hook if you want to save/inspect later

for i, f in enumerate(folds, 1):
    tr_idx, va_idx = f["train_idx"], f["valid_idx"]

    Xtr = df_feat.loc[tr_idx, X_cols].astype("float32").values
    ytr = df_feat.loc[tr_idx, label_col].to_numpy(dtype=np.uint8)
    Xva = df_feat.loc[va_idx, X_cols].astype("float32").values
    yva = df_feat.loc[va_idx, label_col].to_numpy(dtype=np.uint8)

    mdl = models[i-1]  # LightGBM from Cell 6B for this fold

    # Get raw (uncalibrated) scores
    s_tr_raw = mdl.predict_proba(Xtr)[:, 1]
    s_va_raw = mdl.predict_proba(Xva)[:, 1]

    # Fit isotonic on TRAIN (scores vs labels), apply to VALID
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(s_tr_raw, ytr)
    s_va_cal = iso.transform(s_va_raw)
    cal_models.append(iso)

    # AUCPR pre/post
    aupr_raw = aucpr(yva, s_va_raw)
    aupr_cal = aucpr(yva, s_va_cal)

    # Reliability tables (for plotting)
    tab_raw = reliability_table(yva, s_va_raw, n_bins=15)
    tab_cal = reliability_table(yva, s_va_cal, n_bins=15)

    # Capacity threshold (top-k on CALIBRATED scores)
    thr_cap = threshold_for_topk(s_va_cal, K)
    p_cap, r_cap = precision_recall_at_k(yva, s_va_cal, K)

    # Profit-max threshold (on CALIBRATED scores)
    # Use a grid of score quantiles for speed/robustness
    grid = np.quantile(s_va_cal, np.linspace(0.0, 1.0, 201))
    thr_profit, profit = best_threshold_by_profit(yva, s_va_cal, COST, grid=grid)

    cal_rows.append({
        "fold": i,
        "aupr_raw": aupr_raw,
        "aupr_cal": aupr_cal,
        "thr_capacity": float(thr_cap),
        "prec@k_cal": p_cap,
        "recall@k_cal": r_cap,
        "thr_profit": float(thr_profit),
        "profit": float(profit),
    })

    # --- Reliability plot (pre vs post) for this fold ---
    fig, ax = plt.subplots()
    ax.plot([0,1],[0,1], linestyle="--", linewidth=1, label="perfect")
    sns.lineplot(data=tab_raw, x="mean_pred", y="frac_pos", marker="o", label="raw", ax=ax)
    sns.lineplot(data=tab_cal, x="mean_pred", y="frac_pos", marker="o", label="calibrated", ax=ax)
    ax.set_title(f"Reliability (Fold {i})")
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Empirical fraud rate (bin)")
    ax.legend()
    plt.tight_layout()
    plt.show()

cal_df = pd.DataFrame(cal_rows)
print("[CALIBRATION] Per-fold metrics and thresholds (k = {:.0%}):".format(K))
display(cal_df.style.format({
    "aupr_raw":"{:.4f}", "aupr_cal":"{:.4f}",
    "thr_capacity":"{:.4f}", "prec@k_cal":"{:.4f}", "recall@k_cal":"{:.4f}",
    "thr_profit":"{:.4f}", "profit":"{:.2f}"
}))

# Simple summary deltas
delta = (cal_df["aupr_cal"] - cal_df["aupr_raw"]).mean()
print(f"[CALIBRATION] Mean AUCPR lift from isotonic: {delta:+.4f} (positive is good).")



# === Cell 8: Time-safe categorical rarity + WOE encodings → re-train LGBM, measure lift ===
# Objective:
# - For each fold, fit encoders on TRAIN ONLY for:
#     * Rarity: log1p(count)
#     * Smoothed target rate → WOE: log( odds_cat / odds_global )
# - Apply to VALID with train-fitted mappings (unseen -> neutral defaults)
# - Append features and re-run the same LGBM loop; compare to prior res_lgbm

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

label_col = "isFraud"
K = OPS["review_capacity_rate"]

cat_cols = ["P_emaildomain", "DeviceInfo"]

# Initialize container columns (one set per categorical)
for c in cat_cols:
    df_feat[f"{c}_rarity"] = np.nan
    df_feat[f"{c}_woe"] = np.nan

def fit_mappings(train_df: pd.DataFrame, cat: str, y: str) -> tuple[pd.Series, pd.Series, float]:
    """
    Fit train-only mappings:
      - rarity_map: category -> log1p(count)
      - woe_map:    category -> WOE with m-estimate smoothing
    Returns (rarity_map, woe_map, global_rate)
    """
    g = train_df[y].mean()
    grp = train_df.groupby(cat, observed=False, dropna=False)[y].agg(["count", "sum"]).rename(columns={"sum":"pos"})
    m = 50.0  # smoothing strength (pseudo-counts)
    rate = (grp["pos"] + m * g) / (grp["count"] + m)
    odds_cat = rate / (1.0 - rate + 1e-12)
    odds_glob = g / (1.0 - g + 1e-12)
    woe = np.log(odds_cat / (odds_glob + 1e-12))
    rarity_map = np.log1p(grp["count"])
    woe_map = woe
    rarity_map.name = f"{cat}_rarity_map"
    woe_map.name = f"{cat}_woe_map"
    return rarity_map, woe_map, float(g)

def apply_mappings(df_sub: pd.DataFrame, cat: str, rarity_map: pd.Series, woe_map: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    rarity_vals = df_sub[cat].map(rarity_map)
    woe_vals = df_sub[cat].map(woe_map)
    # Neutral defaults for unseen categories
    rarity_vals = rarity_vals.fillna(0.0).to_numpy(dtype=np.float32)
    woe_vals = woe_vals.fillna(0.0).to_numpy(dtype=np.float32)
    return rarity_vals, woe_vals

# Per-fold encode train & valid, filling df_feat columns
for i, f in enumerate(folds, 1):
    tr_idx, va_idx = f["train_idx"], f["valid_idx"]
    tr = df.loc[tr_idx, cat_cols + [label_col]].copy()
    va = df.loc[va_idx, cat_cols + [label_col]].copy()

    for cat in cat_cols:
        rarity_map, woe_map, g = fit_mappings(tr, cat, label_col)
        r_tr, w_tr = apply_mappings(tr, cat, rarity_map, woe_map)
        r_va, w_va = apply_mappings(va, cat, rarity_map, woe_map)

        df_feat.loc[tr_idx, f"{cat}_rarity"] = r_tr
        df_feat.loc[tr_idx, f"{cat}_woe"] = w_tr
        df_feat.loc[va_idx, f"{cat}_rarity"] = r_va
        df_feat.loc[va_idx, f"{cat}_woe"] = w_va

# Extended feature set
X_cols_ext = X_cols + [f"{c}_rarity" for c in cat_cols] + [f"{c}_woe" for c in cat_cols]

# --- Re-train LightGBM with the extended features (same config as Cell 6B) ---
lgbm_ext = LGBMClassifier(
    objective="binary",
    n_estimators=700,          # a touch higher to let it use the added signal
    learning_rate=0.05,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=200,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    class_weight="balanced",
    n_jobs=MODEL_THREADS,
    random_state=SEED
)

rows2 = []
models_ext = []

for i, f in enumerate(folds, 1):
    tr_idx, va_idx = f["train_idx"], f["valid_idx"]

    Xtr = df_feat.loc[tr_idx, X_cols_ext].astype("float32")
    ytr = df_feat.loc[tr_idx, label_col].to_numpy(dtype=np.uint8)
    Xva = df_feat.loc[va_idx, X_cols_ext].astype("float32")
    yva = df_feat.loc[va_idx, label_col].to_numpy(dtype=np.uint8)

    mdl = lgbm_ext.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

    yhat = mdl.predict_proba(Xva)[:, 1]
    aupr = aucpr(yva, yhat)
    p_at_k, r_at_k = precision_recall_at_k(yva, yhat, K)

    rows2.append({
        "fold": i, "model": "lgbm_ext",
        "aupr": aupr, "prec@k": p_at_k, "recall@k": r_at_k,
        "best_iter": getattr(mdl, "best_iteration_", None)
    })
    models_ext.append(mdl)

res_lgbm_ext = pd.DataFrame(rows2)

# --- Compare against prior LightGBM run (res_lgbm) ---
cmp = res_lgbm[["fold","aupr","recall@k"]].merge(
    res_lgbm_ext[["fold","aupr","recall@k"]],
    on="fold", suffixes=("_base","_ext")
)
cmp["Δ_aupr"] = cmp["aupr_ext"] - cmp["aupr_base"]
cmp["Δ_recall@k"] = cmp["recall@k_ext"] - cmp["recall@k_base"]

print("[ABLATION] Per-fold comparison — Base LGBM vs + (rarity + WOE):")
display(cmp.style.format({"aupr_base":"{:.4f}","aupr_ext":"{:.4f}",
                          "recall@k_base":"{:.4f}","recall@k_ext":"{:.4f}",
                          "Δ_aupr":"{:+.4f}","Δ_recall@k":"{:+.4f}"}))

delta_aupr = cmp["Δ_aupr"].mean()
delta_rec = cmp["Δ_recall@k"].mean()
print(f"[ABLATION] Mean lift: ΔAUCPR={delta_aupr:+.4f} | ΔRecall@k={delta_rec:+.4f} (k={K:.0%})")

# Update our current best extended model handle for later analysis if it beats prior best
best_ext_idx = int(res_lgbm_ext["aupr"].argmax())
best_model_ext = models_ext[best_ext_idx]
print(f"[KEEP] Stored best extended LGBM (fold {res_lgbm_ext.iloc[best_ext_idx]['fold']}) as `best_model_ext`.")



# === Cell 9: Wider slice + LightGBM DART (regularized) → compare metrics ===
# Objective:
# - Expand SMALL_RUN day window to ~60 days (without changing earlier cells manually)
# - Rebuild features (velocity + rarity/WOE) on the widened slice, leak-safe per fold
# - Train LGBM (DART) with lower LR, more trees, larger num_leaves, higher max_bin, stronger regularization
# - Evaluate AUCPR and recall@k; compare to prior extended LGBM (Cell 8) if available

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

label_col = "isFraud"
K = OPS["review_capacity_rate"]

# -----------------------
# 1) Rebuild the slice to ~60 days
# -----------------------
EDA_DAYS_LOCAL = 60  # widen the window
if SMALL_RUN and SLICE_MODE == "by_days":
    con.sql(f"""
    CREATE OR REPLACE TEMP TABLE train_slice_tbl AS
    SELECT *
    FROM train_enriched_tbl
    WHERE ts_day >= {EDA_START_DAY} AND ts_day < {EDA_START_DAY + EDA_DAYS_LOCAL};
    """)
else:
    # If you're using fraction mode or FULL run, leave slice as-is
    pass

# -----------------------
# 2) Pull needed columns and rebuild velocity/behavior features (as in Cell 5)
# -----------------------
df = con.sql("""
    SELECT 
        TransactionID, TransactionDT, ts_day, isFraud,
        TransactionAmt, card1, card2, DeviceInfo, P_emaildomain
    FROM train_slice_tbl
""").to_df()

df = df.sort_values("TransactionDT").reset_index(drop=True)
df["isFraud"] = df["isFraud"].astype(np.uint8)
df["event_time"] = pd.to_datetime(df["TransactionDT"], unit="s", origin="unix")

# Primary entity
ENTITY = "card1"
g = df.groupby(ENTITY, dropna=False, sort=False)

# As-of features
first_time = g["event_time"].transform("min")
df["is_new_entity"] = (df["event_time"] == first_time).astype(np.uint8)

prev_time = g["event_time"].shift(1)
df["time_since_prev_sec"] = (df["event_time"] - prev_time).dt.total_seconds().fillna(1e12).astype("float64")

base_col = "TransactionID"
cnt_1d = (
    g.rolling("1D", on="event_time")[base_col]
     .count().reset_index(level=0, drop=True).astype("float64") - 1.0
)
cnt_7d = (
    g.rolling("7D", on="event_time")[base_col]
     .count().reset_index(level=0, drop=True).astype("float64") - 1.0
)
df["cnt_1d"] = np.clip(cnt_1d.to_numpy(), 0.0, None)
df["cnt_7d"] = np.clip(cnt_7d.to_numpy(), 0.0, None)

amt_sum_1d = (
    g.rolling("1D", on="event_time")["TransactionAmt"]
     .sum().reset_index(level=0, drop=True).astype("float64")
     - df["TransactionAmt"].astype("float64").to_numpy()
)
amt_med_7d = (
    g.rolling("7D", on="event_time")["TransactionAmt"]
     .median().reset_index(level=0, drop=True).astype("float64")
)
df["amt_sum_1d"] = np.nan_to_num(amt_sum_1d, nan=0.0, posinf=0.0, neginf=0.0)
df["amt_median_7d"] = np.nan_to_num(amt_med_7d, nan=0.0, posinf=0.0, neginf=0.0)

feature_cols = [
    "TransactionAmt",
    "is_new_entity", "time_since_prev_sec",
    "cnt_1d", "cnt_7d",
    "amt_sum_1d", "amt_median_7d"
]
df_feat = df[["TransactionID", "ts_day", label_col] + feature_cols].copy()
X_cols = feature_cols  # rebuild handle

# -----------------------
# 3) Forward-chaining folds (2 folds, purge = 1 day)
# -----------------------
unique_days = np.sort(df_feat["ts_day"].unique())
n_days = len(unique_days)
def _qday(q): 
    return unique_days[int(np.floor(q * (n_days - 1)))]
cut1 = _qday(0.50)
cut2 = _qday(0.75)
PURGE_DAYS = 1
folds_wide = []
train_idx_1 = np.flatnonzero(df_feat["ts_day"].values < (cut1 - PURGE_DAYS))
valid_idx_1 = np.flatnonzero((df_feat["ts_day"].values >= cut1) & (df_feat["ts_day"].values < cut2))
folds_wide.append({"train_idx": train_idx_1, "valid_idx": valid_idx_1})
train_idx_2 = np.flatnonzero(df_feat["ts_day"].values < (cut2 - PURGE_DAYS))
valid_idx_2 = np.flatnonzero(df_feat["ts_day"].values >= cut2)
folds_wide.append({"train_idx": train_idx_2, "valid_idx": valid_idx_2})

print(f"[CV] Wider window folds built: {len(folds_wide)} | days={int(unique_days.min())}..{int(unique_days.max())}")

# -----------------------
# 4) Time-safe rarity + WOE encodings per fold (P_emaildomain, DeviceInfo)
# -----------------------
cat_cols = ["P_emaildomain", "DeviceInfo"]
for c in cat_cols:
    # ensure columns exist
    if c not in df.columns:
        df[c] = np.nan

# init encoded columns
for c in cat_cols:
    df_feat[f"{c}_rarity"] = np.nan
    df_feat[f"{c}_woe"] = np.nan

def fit_mappings(train_df: pd.DataFrame, cat: str, y: str) -> tuple[pd.Series, pd.Series, float]:
    g_rate = train_df[y].mean()
    grp = train_df.groupby(cat, observed=False, dropna=False)[y].agg(["count", "sum"]).rename(columns={"sum":"pos"})
    m = 50.0  # smoothing strength
    rate = (grp["pos"] + m * g_rate) / (grp["count"] + m)
    odds_cat = rate / (1.0 - rate + 1e-12)
    odds_glob = g_rate / (1.0 - g_rate + 1e-12)
    woe = np.log(odds_cat / (odds_glob + 1e-12))
    rarity_map = np.log1p(grp["count"]); rarity_map.name = f"{cat}_rarity_map"
    woe_map = woe; woe_map.name = f"{cat}_woe_map"
    return rarity_map, woe_map, float(g_rate)

def apply_mappings(df_sub: pd.DataFrame, cat: str, rarity_map: pd.Series, woe_map: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    r = df_sub[cat].map(rarity_map).fillna(0.0).to_numpy(dtype=np.float32)
    w = df_sub[cat].map(woe_map).fillna(0.0).to_numpy(dtype=np.float32)
    return r, w

for f in folds_wide:
    tr_idx, va_idx = f["train_idx"], f["valid_idx"]
    tr = df.loc[tr_idx, cat_cols + [label_col]].copy()
    va = df.loc[va_idx, cat_cols + [label_col]].copy()
    for cat in cat_cols:
        rarity_map, woe_map, g_rate = fit_mappings(tr, cat, label_col)
        r_tr, w_tr = apply_mappings(tr, cat, rarity_map, woe_map)
        r_va, w_va = apply_mappings(va, cat, rarity_map, woe_map)
        df_feat.loc[tr_idx, f"{cat}_rarity"] = r_tr
        df_feat.loc[tr_idx, f"{cat}_woe"] = w_tr
        df_feat.loc[va_idx, f"{cat}_rarity"] = r_va
        df_feat.loc[va_idx, f"{cat}_woe"] = w_va

X_cols_ext = X_cols + [f"{c}_rarity" for c in cat_cols] + [f"{c}_woe" for c in cat_cols]

# -----------------------
# 5) Train LGBM DART on widened slice folds
# -----------------------
lgbm_dart = LGBMClassifier(
    objective="binary",
    boosting_type="dart",
    n_estimators=600,          # more trees
    learning_rate=0.025,        # lower LR
    num_leaves=64,              # slightly larger tree
    max_bin=511,                # finer bins for heavy-tailed features
    min_child_samples=200,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,              # stronger L1
    reg_lambda=2.0,             # stronger L2
    class_weight="balanced",
    n_jobs=MODEL_THREADS,
    random_state=SEED
)

rows_wide = []
models_wide = []

for i, f in enumerate(folds_wide, 1):
    tr_idx, va_idx = f["train_idx"], f["valid_idx"]
    Xtr = df_feat.loc[tr_idx, X_cols_ext].astype("float32")
    ytr = df_feat.loc[tr_idx, label_col].to_numpy(dtype=np.uint8)
    Xva = df_feat.loc[va_idx, X_cols_ext].astype("float32")
    yva = df_feat.loc[va_idx, label_col].to_numpy(dtype=np.uint8)

    mdl = lgbm_dart.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="average_precision",
    )
    yhat = mdl.predict_proba(Xva)[:, 1]
    aupr = aucpr(yva, yhat)
    p_at_k, r_at_k = precision_recall_at_k(yva, yhat, K)

    rows_wide.append({
        "fold": i, "model": "lgbm_dart_wide",
        "aupr": aupr, "prec@k": p_at_k, "recall@k": r_at_k,
        "best_iter": getattr(mdl, "best_iteration_", None)
    })
    models_wide.append(mdl)

res_lgbm_dart_wide = pd.DataFrame(rows_wide)

print("[DART WIDE] AUCPR & recall@k:")
display(res_lgbm_dart_wide[["fold","aupr","prec@k","recall@k","best_iter"]].style.format({
    "aupr":"{:.4f}", "prec@k":"{:.4f}", "recall@k":"{:.4f}"
}))

# -----------------------
# 6) Compare to prior extended run (Cell 8) if available
# -----------------------
if "res_lgbm_ext" in globals():
    cmp = res_lgbm_ext[["fold","aupr","recall@k"]].merge(
        res_lgbm_dart_wide[["fold","aupr","recall@k"]],
        on="fold", suffixes=("_ext30d","_dart60d")
    )
    cmp["Δ_aupr"] = cmp["aupr_dart60d"] - cmp["aupr_ext30d"]
    cmp["Δ_recall@k"] = cmp["recall@k_dart60d"] - cmp["recall@k_ext30d"]
    print("[COMPARE] Prior 30d LGBM_ext vs 60d LGBM_DART:")
    display(cmp.style.format({
        "aupr_ext30d":"{:.4f}","aupr_dart60d":"{:.4f}",
        "recall@k_ext30d":"{:.4f}","recall@k_dart60d":"{:.4f}",
        "Δ_aupr":"{:+.4f}","Δ_recall@k":"{:+.4f}"
    }))
else:
    print("[NOTE] No prior res_lgbm_ext found; using current results as new baseline.")


# === Cell 10A: Personalized amount deviation (per card1) — 7d & 30d windows ===
# Objective:
# - Add standard and robust deviation features per entity (card1) using time-safe rolling windows
#   * amt_z7, amt_z30       : (amt - mean) / std
#   * amt_rz7, amt_rz30     : (amt - median) / (IQR/1.349)
#   * amt_pct_med7          : amt / (median_7d + eps)
#   * amt_above_med7        : max(0, amt - median_7d)

import numpy as np
import pandas as pd

ENTITY = "card1"
amt = df["TransactionAmt"].astype("float64")

gdf = df.groupby(ENTITY, dropna=False, sort=False)

# --- 7-day window stats ---
roll7_mean = (
    gdf.rolling("7D", on="event_time")["TransactionAmt"]
       .mean().reset_index(level=0, drop=True).astype("float64")
)
roll7_std = (
    gdf.rolling("7D", on="event_time")["TransactionAmt"]
       .std(ddof=0).reset_index(level=0, drop=True).astype("float64")
)
roll7_med = (
    gdf.rolling("7D", on="event_time")["TransactionAmt"]
       .median().reset_index(level=0, drop=True).astype("float64")
)
# IQR as 75th - 25th
roll7_q75 = (
    gdf.rolling("7D", on="event_time")["TransactionAmt"]
       .quantile(0.75).reset_index(level=0, drop=True).astype("float64")
)
roll7_q25 = (
    gdf.rolling("7D", on="event_time")["TransactionAmt"]
       .quantile(0.25).reset_index(level=0, drop=True).astype("float64")
)
roll7_iqr = (roll7_q75 - roll7_q25).to_numpy()
eps = 1e-9

df["amt_z7"] = (amt.to_numpy() - roll7_mean.to_numpy()) / (np.sqrt(roll7_std.to_numpy()**2) + eps)
df["amt_rz7"] = (amt.to_numpy() - roll7_med.to_numpy()) / (roll7_iqr / 1.349 + eps)
df["amt_pct_med7"] = amt.to_numpy() / (roll7_med.to_numpy() + eps)
df["amt_above_med7"] = np.clip(amt.to_numpy() - roll7_med.to_numpy(), 0.0, None)

# --- 30-day window stats ---
roll30_mean = (
    gdf.rolling("30D", on="event_time")["TransactionAmt"]
       .mean().reset_index(level=0, drop=True).astype("float64")
)
roll30_std = (
    gdf.rolling("30D", on="event_time")["TransactionAmt"]
       .std(ddof=0).reset_index(level=0, drop=True).astype("float64")
)
roll30_med = (
    gdf.rolling("30D", on="event_time")["TransactionAmt"]
       .median().reset_index(level=0, drop=True).astype("float64")
)
roll30_q75 = (
    gdf.rolling("30D", on="event_time")["TransactionAmt"]
       .quantile(0.75).reset_index(level=0, drop=True).astype("float64")
)
roll30_q25 = (
    gdf.rolling("30D", on="event_time")["TransactionAmt"]
       .quantile(0.25).reset_index(level=0, drop=True).astype("float64")
)
roll30_iqr = (roll30_q75 - roll30_q25).to_numpy()

df["amt_z30"] = (amt.to_numpy() - roll30_mean.to_numpy()) / (np.sqrt(roll30_std.to_numpy()**2) + eps)
df["amt_rz30"] = (amt.to_numpy() - roll30_med.to_numpy()) / (roll30_iqr / 1.349 + eps)

# Clean infinities/nans for modeling
for c in ["amt_z7","amt_z30","amt_rz7","amt_rz30","amt_pct_med7","amt_above_med7"]:
    df[c] = np.nan_to_num(df[c].astype("float32"), nan=0.0, posinf=0.0, neginf=0.0)

# Attach to feature frame (keeps ts_day/label alongside)
new_feats = ["amt_z7","amt_z30","amt_rz7","amt_rz30","amt_pct_med7","amt_above_med7"]
df_feat = df_feat.merge(df[["TransactionID"] + new_feats], on="TransactionID", how="left")
print(f"[10A] Added personalized amount features: {new_feats}")



# === Cell 10B: Short-horizon burstiness (per card1) — 5/15/60min counts + inter-arrival CV ===
# Objective:
# - cnt_5m, cnt_15m, cnt_60m : rolling counts in past minutes (exclude current)
# - gap_mean5, gap_std5, gap_cv5 : stats of last-5 inter-arrival gaps (seconds)

import numpy as np
import pandas as pd

ENTITY = "card1"
base_col = "TransactionID"
gdf = df.groupby(ENTITY, dropna=False, sort=False)

# --- Rolling counts in trailing 5/15/60 minutes (exclude current by -1) ---
cnt_5m = (
    gdf.rolling("5min", on="event_time")[base_col]
       .count().reset_index(level=0, drop=True).astype("float64") - 1.0
)
cnt_15m = (
    gdf.rolling("15min", on="event_time")[base_col]
       .count().reset_index(level=0, drop=True).astype("float64") - 1.0
)
cnt_60m = (
    gdf.rolling("60min", on="event_time")[base_col]
       .count().reset_index(level=0, drop=True).astype("float64") - 1.0
)

df["cnt_5m"] = np.clip(cnt_5m.to_numpy(), 0.0, None)
df["cnt_15m"] = np.clip(cnt_15m.to_numpy(), 0.0, None)
df["cnt_60m"] = np.clip(cnt_60m.to_numpy(), 0.0, None)

# --- Inter-arrival gaps: last-5 events per card1 ---
df["gap_sec"] = (df["event_time"] - gdf["event_time"].shift(1)).dt.total_seconds()

roll_gap_mean5 = (
    df.groupby(ENTITY, dropna=False, sort=False)["gap_sec"]
      .rolling(5).mean().reset_index(level=0, drop=True).astype("float64")
)
roll_gap_std5 = (
    df.groupby(ENTITY, dropna=False, sort=False)["gap_sec"]
      .rolling(5).std(ddof=0).reset_index(level=0, drop=True).astype("float64")
)

eps = 1e-6
df["gap_mean5"] = np.nan_to_num(roll_gap_mean5.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
df["gap_std5"]  = np.nan_to_num(roll_gap_std5.to_numpy(),  nan=0.0, posinf=0.0, neginf=0.0)
df["gap_cv5"]   = np.nan_to_num(df["gap_std5"] / (df["gap_mean5"] + eps), nan=0.0, posinf=0.0, neginf=0.0)

# Attach to feature frame
new_feats = ["cnt_5m","cnt_15m","cnt_60m","gap_mean5","gap_std5","gap_cv5"]
df_feat = df_feat.merge(df[["TransactionID"] + new_feats], on="TransactionID", how="left")
print(f"[10B] Added burstiness features: {new_feats}")



# === Cell 10C: Normalized device/email → rarity + WOE (train-only per fold) ===
# Objective:
# - Parse DeviceInfo → device_brand; P_emaildomain → email_tld
# - Fit rarity + WOE maps on TRAIN ONLY for each fold; apply to VALID
# - Append four features to df_feat

import numpy as np
import pandas as pd

# 1) Normalization (simple, deterministic parsing)
df["device_brand"] = (
    df["DeviceInfo"].astype(str).str.lower()
      .str.split("/", n=1, expand=False).str[0]
      .str.split(" ", n=1, expand=False).str[0]
)

df["email_tld"] = (
    df["P_emaildomain"].astype(str).str.lower()
      .str.split(".", expand=False).str[-1]
)

# 2) Prepare columns on df_feat
for c in ["device_brand_rarity","device_brand_woe","email_tld_rarity","email_tld_woe"]:
    df_feat[c] = np.nan

# Reuse the leakage-safe helpers from Cell 8:
#   fit_mappings(train_df, cat, label_col) -> (rarity_map, woe_map, g_rate)
#   apply_mappings(df_sub, cat, rarity_map, woe_map) -> (r_vals, w_vals)

# 3) Per-fold train-only fitting and valid application
for f in folds:
    tr_idx, va_idx = f["train_idx"], f["valid_idx"]

    # device_brand
    rarity_map, woe_map, _ = fit_mappings(df.loc[tr_idx, ["device_brand", label_col]], "device_brand", label_col)
    r_tr, w_tr = apply_mappings(df.loc[tr_idx, ["device_brand"]], "device_brand", rarity_map, woe_map)
    r_va, w_va = apply_mappings(df.loc[va_idx, ["device_brand"]], "device_brand", rarity_map, woe_map)
    df_feat.loc[tr_idx, "device_brand_rarity"] = r_tr
    df_feat.loc[tr_idx, "device_brand_woe"]    = w_tr
    df_feat.loc[va_idx, "device_brand_rarity"] = r_va
    df_feat.loc[va_idx, "device_brand_woe"]    = w_va

    # email_tld
    rarity_map, woe_map, _ = fit_mappings(df.loc[tr_idx, ["email_tld", label_col]], "email_tld", label_col)
    r_tr, w_tr = apply_mappings(df.loc[tr_idx, ["email_tld"]], "email_tld", rarity_map, woe_map)
    r_va, w_va = apply_mappings(df.loc[va_idx, ["email_tld"]], "email_tld", rarity_map, woe_map)
    df_feat.loc[tr_idx, "email_tld_rarity"] = r_tr
    df_feat.loc[tr_idx, "email_tld_woe"]    = w_tr
    df_feat.loc[va_idx, "email_tld_rarity"] = r_va
    df_feat.loc[va_idx, "email_tld_woe"]    = w_va

# 4) Clean and confirm
new_feats = ["device_brand_rarity","device_brand_woe","email_tld_rarity","email_tld_woe"]
df_feat[new_feats] = df_feat[new_feats].astype("float32").fillna(0.0)

print(f"[10C] Added normalized device/email encodings: {new_feats}")



df_feat


# === Cell 10D: Hierarchical risk with smoothing (child → parent backoff) ===
# Objective:
# - For each fold, fit smoothed child risk and parent risk on TRAIN ONLY
# - Back off to parent when child is sparse (count < 2) per your rule
# - Encode as WOE vs global: child_woe, child_backoff_woe for email and device
#
# Assumes 10C created: df['device_brand'], df['email_tld']
# Uses label_col = "isFraud", folds from earlier

import numpy as np
import pandas as pd

label_col = "isFraud"
m = 50.0  # m-estimate smoothing strength

pairs = [
    ("P_emaildomain", "email_tld", "email"),
    ("DeviceInfo",    "device_brand", "device"),
]

# Prepare output columns in df_feat
out_cols = []
for _, _, tag in pairs:
    for suffix in ["child_woe", "backoff_woe"]:
        col = f"{tag}_{suffix}"
        df_feat[col] = np.nan
        out_cols.append(col)

def _fit_hier_maps(train_df: pd.DataFrame, child: str, parent: str, y: str):
    """Return child map with smoothed child rate, count, parent key & parent smoothed rate; plus global rate."""
    g = train_df[y].mean()
    # Parent stats
    par = (train_df.groupby(parent, observed=False, dropna=False)[y]
           .agg(count="size", pos="sum"))
    par_rate = (par["pos"] + m * g) / (par["count"] + m)
    # Child stats (with parent key)
    ch = (train_df.groupby([child, parent], observed=False, dropna=False)[y]
          .agg(count="size", pos="sum")).reset_index()
    ch["rate_child"] = (ch["pos"] + m * g) / (ch["count"] + m)
    # attach parent rate
    ch = ch.merge(par_rate.rename("rate_parent").to_frame().reset_index(), on=parent, how="left")
    return ch, float(g)

def _make_woe(rate: np.ndarray, g: float) -> np.ndarray:
    odds_c = rate / (1.0 - rate + 1e-12)
    odds_g = g / (1.0 - g + 1e-12)
    return np.log(odds_c / (odds_g + 1e-12))

for f in folds:
    tr_idx, va_idx = f["train_idx"], f["valid_idx"]
    dtr = df.loc[tr_idx, :].copy()
    dva = df.loc[va_idx, :].copy()

    for child, parent, tag in pairs:
        # Train-only maps
        ch_map, g = _fit_hier_maps(dtr[[child, parent, label_col]], child, parent, label_col)

        # Backoff rule:
        # - If child count >= 2: use child smoothed rate
        # - Else: blend child & parent (simple average when count==1; parent-only when count==0)
        cnt = ch_map["count"].to_numpy()
        r_child = ch_map["rate_child"].to_numpy()
        r_parent = ch_map["rate_parent"].to_numpy()

        r_backoff = r_child.copy()
        mask_1 = cnt == 1
        r_backoff[mask_1] = 0.5 * r_child[mask_1] + 0.5 * r_parent[mask_1]
        mask_0 = cnt == 0  # (not present; defensive if ever empty)
        if mask_0.any():
            r_backoff[mask_0] = r_parent[mask_0]

        # Build mapping Series keyed by child value
        # (child may map to multiple parents in theory; training data determines mapping used at inference)
        child_key = ch_map[child].astype(object)
        child_woe_map = pd.Series(_make_woe(r_child, g), index=child_key, name=f"{tag}_child_woe_map")
        backoff_woe_map = pd.Series(_make_woe(r_backoff, g), index=child_key, name=f"{tag}_backoff_woe_map")

        # Apply to TRAIN and VALID partitions (unseen -> 0.0 neutral)
        df_feat.loc[tr_idx, f"{tag}_child_woe"]   = dtr[child].map(child_woe_map).fillna(0.0).to_numpy(dtype=np.float32)
        df_feat.loc[tr_idx, f"{tag}_backoff_woe"] = dtr[child].map(backoff_woe_map).fillna(0.0).to_numpy(dtype=np.float32)
        df_feat.loc[va_idx, f"{tag}_child_woe"]   = dva[child].map(child_woe_map).fillna(0.0).to_numpy(dtype=np.float32)
        df_feat.loc[va_idx, f"{tag}_backoff_woe"] = dva[child].map(backoff_woe_map).fillna(0.0).to_numpy(dtype=np.float32)

# Finalize types
df_feat[out_cols] = df_feat[out_cols].astype("float32").fillna(0.0)

print(f"[10D] Added hierarchical risk features: {out_cols}")



# === Cell 10E: Re-train LGBM on 30d ablation with all new features; compare metrics ===
# Objective:
# - Use folds from the ablation setup (forward-chaining, purge) — i.e., `folds`
# - Train LightGBM (same config as Cell 8) on features: base + 10A + 10B + 10C + 10D
# - Report AUCPR and recall@k per fold; compare to res_lgbm_ext (Cell 8)
# - Keep best fold model as `best_model_ablate`
label_col = "isFraud"
K = OPS["review_capacity_rate"]

# --- 1) Assemble the final feature list used in this ablation run ---
fe_10a = ["amt_z7","amt_z30","amt_rz7","amt_rz30","amt_pct_med7","amt_above_med7"]
fe_10b = ["cnt_5m","cnt_15m","cnt_60m","gap_mean5","gap_std5","gap_cv5"]
fe_10c = ["device_brand_rarity","device_brand_woe","email_tld_rarity","email_tld_woe"]
fe_10d = ["email_child_woe","email_backoff_woe","device_child_woe","device_backoff_woe"]

X_cols_ablate = X_cols + [f for f in fe_10a + fe_10b + fe_10c + fe_10d]

# Safety: ensure numeric dtype and no NaNs/Infs
df_feat[X_cols_ablate] = df_feat[X_cols_ablate].astype("float32")
df_feat[X_cols_ablate] = df_feat[X_cols_ablate].replace([np.inf, -np.inf], 0.0).fillna(0.0)

# --- 2) LightGBM config (same spirit as Cell 8 for fair comparison) ---
lgbm_ablate = LGBMClassifier(
    objective="binary",
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=200,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    class_weight="balanced",
    n_jobs=MODEL_THREADS,
    random_state=SEED
)

rows_ablate, models_ablate = [], []
best_model_ablate, best_fold, best_aupr = None, None, -np.inf

for i, f in enumerate(folds, 1):
    tr_idx, va_idx = f["train_idx"], f["valid_idx"]

    Xtr = df_feat.loc[tr_idx, X_cols_ablate].values
    ytr = df_feat.loc[tr_idx, label_col].to_numpy(dtype=np.uint8)
    Xva = df_feat.loc[va_idx, X_cols_ablate].values
    yva = df_feat.loc[va_idx, label_col].to_numpy(dtype=np.uint8)

    mdl = lgbm_ablate.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    yhat = mdl.predict_proba(Xva)[:, 1]
    aupr = aucpr(yva, yhat)
    p_at_k, r_at_k = precision_recall_at_k(yva, yhat, K)

    rows_ablate.append({
        "fold": i, "model": "lgbm_ablate_30d",
        "aupr": aupr, "prec@k": p_at_k, "recall@k": r_at_k,
        "best_iter": getattr(mdl, "best_iteration_", None)
    })
    models_ablate.append(mdl)

    if aupr > best_aupr:
        best_aupr, best_model_ablate, best_fold = aupr, mdl, i

res_lgbm_ablate = pd.DataFrame(rows_ablate)

print("[ABLATION-ALL] AUCPR & recall@k per fold (30d):")
display(res_lgbm_ablate[["fold","aupr","prec@k","recall@k","best_iter"]].style.format({
    "aupr":"{:.4f}", "prec@k":"{:.4f}", "recall@k":"{:.4f}"
}))

# --- 3) Compare to prior extended (Cell 8) if available ---
if "res_lgbm_ext" in globals():
    cmp_all = res_lgbm_ext[["fold","aupr","recall@k"]].merge(
        res_lgbm_ablate[["fold","aupr","recall@k"]],
        on="fold", suffixes=("_ext","_ablate")
    )
    cmp_all["Δ_aupr"] = cmp_all["aupr_ablate"] - cmp_all["aupr_ext"]
    cmp_all["Δ_recall@k"] = cmp_all["recall@k_ablate"] - cmp_all["recall@k_ext"]
    print("[COMPARE] Cell 8 (rarity+WOE) vs 10E (all new features) — 30d slice:")
    display(cmp_all.style.format({
        "aupr_ext":"{:.4f}","aupr_ablate":"{:.4f}",
        "recall@k_ext":"{:.4f}","recall@k_ablate":"{:.4f}",
        "Δ_aupr":"{:+.4f}","Δ_recall@k":"{:+.4f}"
    }))
else:
    print("[NOTE] No prior res_lgbm_ext found; treat current as baseline for this slice.")

print(f"[KEEP] Best ablation model: fold {best_fold}, AUCPR={best_aupr:.4f} → saved as `best_model_ablate`.")



# === Cell 11A: Diagnostics — permutation importance, mean |SHAP|, rank correlation, FP/FN case studies ===
# Purpose:
# - Compare best_model_ext (Cell 8) vs best_model_ablate (Cell 10E) on the same 30-day CV context
# - Permutation importance (AUCPR) on each model's best-fold VALID set
# - Mean |SHAP| on each model's best-fold VALID set (sample up to 5k rows)
# - Spearman rank correlation between importance vectors
# - FP/FN case studies (top-20 each) with top-5 per-row SHAP contributors


from sklearn.metrics import make_scorer, average_precision_score
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

import shap
shap.utils._legacy = True  # keep TreeExplainer happy on some versions

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (7, 4)

# --- Helper: get best fold meta (indices, features, etc.) for a given results table and model handle ---
def get_best_fold(res_df: pd.DataFrame, models_list: list, X_cols_list: list[str] | None = None):
    best_idx = int(res_df["aupr"].argmax())
    best_fold = int(res_df.iloc[best_idx]["fold"])
    model = models_list[best_idx]
    # use global folds (ablation used `folds`)
    f = folds[best_fold - 1]
    va_idx = f["valid_idx"]
    tr_idx = f["train_idx"]
    # decide feature set: for ext run, use X_cols_ext; for ablate, use X_cols_ablate if present, else fall back
    if X_cols_list is None:
        Xc = X_cols  # fallback
    else:
        Xc = X_cols_list
    return best_fold, model, tr_idx, va_idx, Xc

# --- Identify best folds & assemble data for both models ---
assert "best_model_ext" in globals(), "best_model_ext not found (from Cell 8)."
assert "best_model_ablate" in globals(), "best_model_ablate not found (from Cell 10E)."

# For the extended model from Cell 8
best_fold_ext, mdl_ext, tr_idx_ext, va_idx_ext, Xc_ext = get_best_fold(res_lgbm_ext, models_ext, X_cols_ext)
Xva_ext = df_feat.loc[va_idx_ext, Xc_ext].astype("float32").values
yva_ext = df_feat.loc[va_idx_ext, "isFraud"].to_numpy(dtype=np.uint8)

# For the ablation-all model from Cell 10E
best_fold_abl, mdl_abl, tr_idx_abl, va_idx_abl, Xc_abl = get_best_fold(res_lgbm_ablate, models_ablate, X_cols_ablate)
Xva_abl = df_feat.loc[va_idx_abl, Xc_abl].astype("float32").values
yva_abl = df_feat.loc[va_idx_abl, "isFraud"].to_numpy(dtype=np.uint8)

print(f"[DIAG] best_fold_ext={best_fold_ext} | best_fold_abl={best_fold_abl}")
print(f"[DIAG] ext valid n={len(yva_ext):,} | ablate valid n={len(yva_abl):,}")

# --- 1) Permutation importance (AUCPR scorer) on VALID sets ---
ap_scorer = make_scorer(average_precision_score, needs_threshold=True)

pi_ext = permutation_importance(
    estimator=mdl_ext, X=Xva_ext, y=yva_ext, scoring=ap_scorer,
    n_repeats=5, random_state=SEED, n_jobs=MODEL_THREADS
)
pi_abl = permutation_importance(
    estimator=mdl_abl, X=Xva_abl, y=yva_abl, scoring=ap_scorer,
    n_repeats=5, random_state=SEED, n_jobs=MODEL_THREADS
)

imp_ext = (pd.DataFrame({
    "feature": Xc_ext,
    "perm_imp_mean": pi_ext.importances_mean,
    "perm_imp_std": pi_ext.importances_std
}).sort_values("perm_imp_mean", ascending=False).reset_index(drop=True))

imp_abl = (pd.DataFrame({
    "feature": Xc_abl,
    "perm_imp_mean": pi_abl.importances_mean,
    "perm_imp_std": pi_abl.importances_std
}).sort_values("perm_imp_mean", ascending=False).reset_index(drop=True))

print("[PERM] Top 15 features — Extended model (Cell 8):")
display(imp_ext.head(15).style.format({"perm_imp_mean":"{:.5f}","perm_imp_std":"{:.5f}"}))

print("[PERM] Top 15 features — Ablation model (Cell 10E):")
display(imp_abl.head(15).style.format({"perm_imp_mean":"{:.5f}","perm_imp_std":"{:.5f}"}))

# --- 2) Mean |SHAP| on best validation folds (sample up to 5k rows for speed) ---
SAMPLE_N = min(5000, len(yva_ext), len(yva_abl))

# Extended model SHAP on its valid split
idx_sample_ext = np.random.default_rng(SEED).choice(len(yva_ext), size=SAMPLE_N, replace=False)
expl_ext = shap.TreeExplainer(mdl_ext)
shap_ext = expl_ext.shap_values(Xva_ext[idx_sample_ext])
# shap_values can be [array] or list per class; handle binary class list
if isinstance(shap_ext, list):
    shap_vals_ext = shap_ext[1]
else:
    shap_vals_ext = shap_ext
mean_abs_shap_ext = np.abs(shap_vals_ext).mean(axis=0)
shap_imp_ext = pd.DataFrame({"feature": Xc_ext, "mean_|shap|": mean_abs_shap_ext}).sort_values("mean_|shap|", ascending=False)

# Ablation model SHAP on its valid split
idx_sample_abl = np.random.default_rng(SEED+1).choice(len(yva_abl), size=SAMPLE_N, replace=False)
expl_abl = shap.TreeExplainer(mdl_abl)
shap_abl = expl_abl.shap_values(Xva_abl[idx_sample_abl])
if isinstance(shap_abl, list):
    shap_vals_abl = shap_abl[1]
else:
    shap_vals_abl = shap_abl
mean_abs_shap_abl = np.abs(shap_vals_abl).mean(axis=0)
shap_imp_abl = pd.DataFrame({"feature": Xc_abl, "mean_|shap|": mean_abs_shap_abl}).sort_values("mean_|shap|", ascending=False)

print("[SHAP] Top 15 features — Extended model (Cell 8):")
display(shap_imp_ext.head(15).style.format({"mean_|shap|":"{:.6f}"}))
print("[SHAP] Top 15 features — Ablation model (Cell 10E):")
display(shap_imp_abl.head(15).style.format({"mean_|shap|":"{:.6f}"}))

# --- 3) Rank correlation between importance vectors (Spearman on aligned feature sets) ---
# Align by shared features (different runs may have different columns)
shared_ext = set(imp_ext["feature"])
shared_abl = set(imp_abl["feature"])
shared = sorted(shared_ext & shared_abl)

rank_ext = imp_ext.set_index("feature").loc[shared, "perm_imp_mean"].rank(ascending=False)
rank_abl = imp_abl.set_index("feature").loc[shared, "perm_imp_mean"].rank(ascending=False)
rho, pval = spearmanr(rank_ext.values, rank_abl.values)

print(f"[RANK] Spearman ρ between permutation importances (shared features, n={len(shared)}): ρ={rho:.3f}, p={pval:.2g}")

# --- 4) FP/FN case studies with top-5 SHAP contributors (best fold of each model) ---
def fp_fn_tables(model, Xva, yva, X_cols, sample_seed=SEED, top_k=20):
    # scores
    s = model.predict_proba(Xva)[:, 1]
    # Compute SHAP for all or a sample to get per-row attributions
    n = len(yva)
    take = min(5000, n)
    idx_samp = np.random.default_rng(sample_seed).choice(n, size=take, replace=False)
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(Xva[idx_samp])
    if isinstance(sv, list):
        sv = sv[1]
    # Build a quick index mapping for the sampled rows
    samp_mask = np.zeros(n, dtype=bool)
    samp_mask[idx_samp] = True

    # FPs: y=0 with highest scores
    fp_idx = np.where((yva == 0))[0]
    fp_sorted = fp_idx[np.argsort(-s[fp_idx])][:top_k]

    # FNs: y=1 with lowest scores
    fn_idx = np.where((yva == 1))[0]
    fn_sorted = fn_idx[np.argsort(s[fn_idx])][:top_k]

    def top5_shap_for_row(row_i):
        # if row not in sample, return placeholder (to keep quick)
        if not samp_mask[row_i]:
            return "n/a (not in SHAP sample)"
        local = sv[np.where(idx_samp == row_i)[0][0]]
        top5 = np.argsort(-np.abs(local))[:5]
        pairs = [f"{X_cols[j]}:{local[j]:+.3f}" for j in top5]
        return "; ".join(pairs)

    fp_rows = []
    for i in fp_sorted:
        fp_rows.append({
            "row_idx": int(i),
            "score": float(s[i]),
            "top5_shap": top5_shap_for_row(i)
        })
    fn_rows = []
    for i in fn_sorted:
        fn_rows.append({
            "row_idx": int(i),
            "score": float(s[i]),
            "top5_shap": top5_shap_for_row(i)
        })
    return pd.DataFrame(fp_rows), pd.DataFrame(fn_rows)

fp_ext, fn_ext = fp_fn_tables(mdl_ext, Xva_ext, yva_ext, Xc_ext, sample_seed=SEED, top_k=20)
fp_abl, fn_abl = fp_fn_tables(mdl_abl, Xva_abl, yva_abl, Xc_abl, sample_seed=SEED+1, top_k=20)

print("[CASE] Top-20 False Positives — Extended model (valid fold):")
display(fp_ext.head(20).style.format({"score":"{:.4f}"}))
print("[CASE] Top-20 False Negatives — Extended model (valid fold):")
display(fn_ext.head(20).style.format({"score":"{:.4f}"}))

print("[CASE] Top-20 False Positives — Ablation model (valid fold):")
display(fp_abl.head(20).style.format({"score":"{:.4f}"}))
print("[CASE] Top-20 False Negatives — Ablation model (valid fold):")
display(fn_abl.head(20).style.format({"score":"{:.4f}"}))

print("Diagnostics complete: permutation importances, mean|SHAP|, rank correlation, FP/FN case studies.")



# === Cell 11B: Sanity checks — coverage, window effectiveness, WOE tails, redundancy ===
# Purpose:
# - For the 30-day ablation context (folds), quantify:
#   (1) Coverage: % non-null, % non-zero per feature on TRAIN vs VALID
#   (2) Window effectiveness: % rows where short-horizon / deviation features actually "fire"
#   (3) Encoding smoothness: WOE tails (p01/p99, %(|woe|>3))
#   (4) Redundancy: |Spearman| ≥ 0.90 among new features (10A–10D)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

label_col = "isFraud"
EPS = 1e-6

# --- Feature groups from earlier cells ---
fe_10a = ["amt_z7","amt_z30","amt_rz7","amt_rz30","amt_pct_med7","amt_above_med7"]
fe_10b = ["cnt_5m","cnt_15m","cnt_60m","gap_mean5","gap_std5","gap_cv5"]
fe_10c = ["device_brand_rarity","device_brand_woe","email_tld_rarity","email_tld_woe"]
fe_10d = ["email_child_woe","email_backoff_woe","device_child_woe","device_backoff_woe"]

# Include original encoders from Cell 8 if present
enc_base = [c for c in [ "P_emaildomain_rarity","P_emaildomain_woe",
                         "DeviceInfo_rarity","DeviceInfo_woe"] if c in df_feat.columns]

new_all = [c for c in (fe_10a + fe_10b + fe_10c + fe_10d + enc_base) if c in df_feat.columns]
woe_cols = [c for c in new_all if "woe" in c]

def coverage_table(X: pd.DataFrame) -> pd.DataFrame:
    n = len(X)
    non_null = X.notna().sum() / max(n,1)
    non_zero = (X.fillna(0) != 0).sum() / max(n,1)
    return pd.DataFrame({"non_null": non_null, "non_zero": non_zero})

def window_effectiveness_table(X: pd.DataFrame) -> pd.DataFrame:
    # Define "fires" per feature family
    fires = {}
    for c in X.columns:
        if c.startswith("cnt_"):               # burstiness counts
            fires[c] = (X[c] > 0)
        elif c in ["amt_above_med7"]:         # positive deviation
            fires[c] = (X[c] > EPS)
        elif c.startswith("amt_"):            # other amount deviations (z / pct / robust z)
            fires[c] = (X[c].abs() > EPS)
        else:
            # for encoders/rarity, treat any non-zero as "fires"
            fires[c] = (X[c].fillna(0) != 0)
    eff = {c: fires[c].mean() for c in X.columns}
    return pd.DataFrame.from_dict(eff, orient="index", columns=["fires_rate"])

# --- 1) Coverage & window effectiveness per fold and split ---
rows_cov = []
rows_eff = []

for i, f in enumerate(folds, 1):
    tr_idx, va_idx = f["train_idx"], f["valid_idx"]
    Xtr = df_feat.loc[tr_idx, new_all].copy()
    Xva = df_feat.loc[va_idx, new_all].copy()

    cov_tr = coverage_table(Xtr)
    cov_tr["fold"] = i; cov_tr["split"] = "train"
    cov_va = coverage_table(Xva)
    cov_va["fold"] = i; cov_va["split"] = "valid"
    rows_cov.append(cov_tr.reset_index(names="feature"))
    rows_cov.append(cov_va.reset_index(names="feature"))

    eff_tr = window_effectiveness_table(Xtr)
    eff_tr["fold"] = i; eff_tr["split"] = "train"
    eff_va = window_effectiveness_table(Xva)
    eff_va["fold"] = i; eff_va["split"] = "valid"
    rows_eff.append(eff_tr.reset_index(names="feature"))
    rows_eff.append(eff_va.reset_index(names="feature"))

cov_df = pd.concat(rows_cov, ignore_index=True)
eff_df = pd.concat(rows_eff, ignore_index=True)

print("[COVERAGE] % non-null and % non-zero per feature (mean over folds):")
cov_summary = (cov_df.groupby(["feature","split"], as_index=False)[["non_null","non_zero"]]
               .mean().sort_values(["split","non_zero"], ascending=[True, False]))
display(cov_summary.style.format({"non_null":"{:.1%}","non_zero":"{:.1%}"}))

print("[WINDOW EFFECTIVENESS] % rows where feature 'fires' (mean over folds):")
eff_summary = (eff_df.groupby(["feature","split"], as_index=False)["fires_rate"]
               .mean().sort_values(["split","fires_rate"], ascending=[True, False]))
display(eff_summary.style.format({"fires_rate":"{:.1%}"}))

# --- 2) Encoding smoothness (WOE tails) on VALID rows across folds ---
if woe_cols:
    va_idx_all = np.concatenate([f["valid_idx"] for f in folds])
    woe_frame = df_feat.loc[va_idx_all, woe_cols].copy()

    pct01 = woe_frame.quantile(0.01, numeric_only=True)
    pct99 = woe_frame.quantile(0.99, numeric_only=True)
    heavy_tail = (woe_frame.abs() > 3).sum() / max(len(woe_frame),1)

    woe_smooth = (pd.DataFrame({
        "feature": woe_cols,
        "p01": [pct01.get(c, np.nan) for c in woe_cols],
        "p99": [pct99.get(c, np.nan) for c in woe_cols],
        "%|woe|>3": [heavy_tail.get(c, np.nan) for c in woe_cols],
    })
    .sort_values("%|woe|>3", ascending=False))
    print("[WOE SMOOTHNESS] tails on VALID (aggregated across folds):")
    display(woe_smooth.style.format({"p01":"{:.2f}","p99":"{:.2f}","%|woe|>3":"{:.1%}"}))
else:
    print("[WOE SMOOTHNESS] No WOE columns found in df_feat.")

# --- 3) Redundancy: Spearman correlation among new features (VALID rows across folds) ---
if new_all:
    va_idx_all = np.concatenate([f["valid_idx"] for f in folds])
    Xcorr = df_feat.loc[va_idx_all, new_all].astype("float32").copy()
    # Drop all-constant columns to avoid NaNs in correlation
    nunique = Xcorr.nunique(dropna=False)
    keep = nunique[nunique > 1].index.tolist()
    Xcorr = Xcorr[keep]

    # Compute Spearman pairwise
    corr_mat = Xcorr.corr(method="spearman").abs()
    # List highly correlated pairs (upper triangle, exclude diag)
    hi_pairs = []
    cols = corr_mat.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            rho = corr_mat.iat[i, j]
            if np.isfinite(rho) and rho >= 0.90:
                hi_pairs.append((cols[i], cols[j], float(rho)))
    red_df = pd.DataFrame(hi_pairs, columns=["feature_a","feature_b","|spearman|"])
    print("[REDUNDANCY] Highly correlated feature pairs (|ρ| ≥ 0.90) on VALID:")
    if len(red_df):
        display(red_df.sort_values("|spearman|", ascending=False))
    else:
        print("  None ≥ 0.90")
else:
    print("[REDUNDANCY] No features to analyze.")

# --- 4) Compact findings to guide pruning/tuning ---
LOW_COV_THRESH = 0.10   # <10% non-zero on VALID
LOW_EFF_THRESH = 0.10   # <10% fires on VALID
HEAVY_WOE_THRESH = 0.05 # >5% of rows with |woe|>3 on VALID

low_cov = cov_summary[cov_summary["split"]=="valid"].query("non_zero < @LOW_COV_THRESH")[["feature","non_zero"]]
low_eff = eff_summary[eff_summary["split"]=="valid"].query("fires_rate < @LOW_EFF_THRESH")[["feature","fires_rate"]]
heavy_woe = woe_smooth[woe_smooth["%|woe|>3"] > HEAVY_WOE_THRESH] if woe_cols else pd.DataFrame(columns=["feature"])

print("\n=== Suggested actions ===")
if not low_cov.empty:
    print("- Prune or postpone (low coverage on VALID):")
    display(low_cov.sort_values("non_zero").style.format({"non_zero":"{:.1%}"}))
else:
    print("- No low-coverage features (<10% non-zero) on VALID.")

if not low_eff.empty:
    print("- Window ineffectiveness (rarely fires on VALID):")
    display(low_eff.sort_values("fires_rate").style.format({"fires_rate":"{:.1%}"}))
else:
    print("- All windowed/deviation features fire on ≥10% of VALID rows.")

if woe_cols:
    if not heavy_woe.empty:
        print("- Consider clipping WOE to ±3 (features with heavy tails >5% of VALID):")
        display(heavy_woe[["feature","%|woe|>3"]].style.format({"%|woe|>3":"{:.1%}"}))
    else:
        print("- WOE tails look reasonable (≤5% beyond |3|).")

if 'red_df' in locals() and len(red_df):
    print("- Redundancy detected (|Spearman| ≥ 0.90): prefer keeping only one from each pair.")
else:
    print("- No high redundancy ≥ 0.90 detected.")



# === Cell 12: Restructure & prune features + stronger regularization + monotone constraints (30d ablation) ===
# Purpose:
# - Keep short-term / baseline features appropriate for a 30d slice
# - Clip WOE tails, add mild extra regularization, enforce sensible monotonicity
# - Re-evaluate vs Cell 10E (ablation-all) to confirm reduced variance / better AUCPR or Recall@k


label_col = "isFraud"
K = OPS["review_capacity_rate"]

# 1) Select features to KEEP (short-term/baseline only for 30d)
fe_keep = [
    # personalized amount deviation (7d, short/baseline)
    "amt_z7", "amt_rz7", "amt_pct_med7",
    # burstiness (pick the most generally-informative singletons)
    "cnt_15m", "gap_cv5",
    # normalized encoders
    "device_brand_woe", "email_tld_woe",
    # hierarchical backoffs
    "device_backoff_woe", "email_backoff_woe",
    # baseline encoders from Cell 8 (keep continuity)
    "P_emaildomain_woe", "DeviceInfo_woe",
]

# Compose final feature list = base velocity block + pruned extensions
X_cols_restruct = X_cols + fe_keep

# 2) Clip WOE tails to ±3 (stabilize contributions on small slice)
woe_cols_all = [c for c in X_cols_restruct if "woe" in c]
df_feat[woe_cols_all] = df_feat[woe_cols_all].clip(lower=-3.0, upper=+3.0)

# 3) Ensure numeric & clean (no NaNs/±inf)
df_feat[X_cols_restruct] = df_feat[X_cols_restruct].astype("float32").replace([np.inf, -np.inf], 0.0).fillna(0.0)

# 4) Monotone constraint vector (aligned to X_cols_restruct)
#    +1 for features we know should increase risk when larger
#    Here: TransactionAmt, amt_pct_med7. Others unconstrained.
mono = []
for feat in X_cols_restruct:
    if feat in ["TransactionAmt", "amt_pct_med7"]:
        mono.append(+1)
    else:
        mono.append(0)

# 5) LightGBM config: slightly stronger regularization than 10E
lgbm_pruned = LGBMClassifier(
    objective="binary",
    n_estimators=900,
    learning_rate=0.045,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=250,   # ↑ for extra regularization
    subsample=0.75,          # ↓ a bit to reduce variance
    colsample_bytree=0.75,   # ↓ a bit to reduce variance
    max_bin=255,
    reg_alpha=0.0,
    reg_lambda=1.5,          # ↑ L2 a touch
    class_weight="balanced",
    monotone_constraints=mono,
    n_jobs=MODEL_THREADS,
    random_state=SEED
)

rows_restruct, models_restruct = [], []
best_model_restruct, best_fold_restruct, best_aupr_restruct = None, None, -np.inf

for i, f in enumerate(folds, 1):
    tr_idx, va_idx = f["train_idx"], f["valid_idx"]

    Xtr = df_feat.loc[tr_idx, X_cols_restruct].values
    ytr = df_feat.loc[tr_idx, label_col].to_numpy(dtype=np.uint8)
    Xva = df_feat.loc[va_idx, X_cols_restruct].values
    yva = df_feat.loc[va_idx, label_col].to_numpy(dtype=np.uint8)

    mdl = lgbm_pruned.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="average_precision"
    )
    yhat = mdl.predict_proba(Xva)[:, 1]
    aupr = aucpr(yva, yhat)
    p_at_k, r_at_k = precision_recall_at_k(yva, yhat, K)

    rows_restruct.append({
        "fold": i, "model": "lgbm_pruned_30d",
        "aupr": aupr, "prec@k": p_at_k, "recall@k": r_at_k,
        "best_iter": getattr(mdl, "best_iteration_", None)
    })
    models_restruct.append(mdl)

    if aupr > best_aupr_restruct:
        best_aupr_restruct, best_model_restruct, best_fold_restruct = aupr, mdl, i

res_lgbm_restruct = pd.DataFrame(rows_restruct)

print("[RESTRUCT] AUCPR & recall@k per fold (30d):")
display(res_lgbm_restruct[["fold","aupr","prec@k","recall@k","best_iter"]].style.format({
    "aupr":"{:.4f}", "prec@k":"{:.4f}", "recall@k":"{:.4f}"
}))

# 6) Compare against ablation-all (Cell 10E) to validate pruning/constraints/regularization
if "res_lgbm_ablate" in globals():
    cmp_r = res_lgbm_ablate[["fold","aupr","recall@k"]].merge(
        res_lgbm_restruct[["fold","aupr","recall@k"]],
        on="fold", suffixes=("_ablate_all","_pruned")
    )
    cmp_r["Δ_aupr"] = cmp_r["aupr_pruned"] - cmp_r["aupr_ablate_all"]
    cmp_r["Δ_recall@k"] = cmp_r["recall@k_pruned"] - cmp_r["recall@k_ablate_all"]
    print("[COMPARE] 10E (all new features) vs 12 (pruned/monotone/regularized) — 30d slice:")
    display(cmp_r.style.format({
        "aupr_ablate_all":"{:.4f}","aupr_pruned":"{:.4f}",
        "recall@k_ablate_all":"{:.4f}","recall@k_pruned":"{:.4f}",
        "Δ_aupr":"{:+.4f}","Δ_recall@k":"{:+.4f}"
    }))

print(f"[KEEP] Best pruned model: fold {best_fold_restruct}, AUCPR={best_aupr_restruct:.4f} → saved as `best_model_restruct`.")



# === Cell 13: Micro-sweep diagnostics on 30d — clipping / constraints / regularization / burstiness ===
# Purpose:
# - Run 4 small variants vs the Cell 12 pruned baseline to see which guardrail affected ranking most.
# - Report AUCPR, precision/recall@k per fold and Δ vs baseline.

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

label_col = "isFraud"
K = OPS["review_capacity_rate"]

# ----- Baseline feature set from Cell 12 -----
fe_keep = [
    "amt_z7", "amt_rz7", "amt_pct_med7",
    "cnt_15m", "gap_cv5",
    "device_brand_woe", "email_tld_woe",
    "device_backoff_woe", "email_backoff_woe",
    "P_emaildomain_woe", "DeviceInfo_woe",
]
X_cols_base = X_cols + fe_keep  # base velocity block + curated features (Cell 12)

# Utility: build monotone vector aligned to given feature list
def build_monotone(features, amt_pct_monotone=True):
    mono = []
    for f in features:
        if f == "TransactionAmt":
            mono.append(+1)
        elif f == "amt_pct_med7" and amt_pct_monotone:
            mono.append(+1)
        else:
            mono.append(0)
    return mono

# Helper: train/eval one configuration across folds
def run_config(name, X_cols_cfg, params, mono_vec):
    rows, models = [], []
    for i, f in enumerate(folds, 1):
        tr_idx, va_idx = f["train_idx"], f["valid_idx"]

        # Local matrices (never mutate df_feat in place in this sweep)
        Xtr = df_feat.loc[tr_idx, X_cols_cfg].astype("float32").copy()
        Xva = df_feat.loc[va_idx, X_cols_cfg].astype("float32").copy()
        ytr = df_feat.loc[tr_idx, label_col].to_numpy(dtype=np.uint8)
        yva = df_feat.loc[va_idx, label_col].to_numpy(dtype=np.uint8)

        # Optional: apply a local "soft clip" to WOE columns if requested in params
        clip_bounds = params.pop("_woe_soft_clip", None)
        if clip_bounds is not None:
            lo, hi = clip_bounds
            woe_cols_local = [c for c in X_cols_cfg if "woe" in c]
            if len(woe_cols_local):
                # NOTE: since Cell 12 clipped to ±3 in-place, values are already <= ±3; 
                # this soft-clip to ±4.5 won't expand magnitudes. This test is here for completeness.
                Xtr[woe_cols_local] = Xtr[woe_cols_local].clip(lo, hi)
                Xva[woe_cols_local] = Xva[woe_cols_local].clip(lo, hi)

        # Clean numerics
        Xtr = Xtr.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        Xva = Xva.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        clf = LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            n_jobs=MODEL_THREADS,
            random_state=SEED,
            monotone_constraints=mono_vec,
            **params
        )
        clf.fit(
            Xtr.values, ytr,
            eval_set=[(Xva.values, yva)],
            eval_metric="average_precision",
        )
        yhat = clf.predict_proba(Xva.values)[:, 1]
        aupr = aucpr(yva, yhat)
        p_at_k, r_at_k = precision_recall_at_k(yva, yhat, K)

        rows.append({
            "config": name, "fold": i,
            "aupr": aupr, "prec@k": p_at_k, "recall@k": r_at_k,
            "best_iter": getattr(clf, "best_iteration_", None)
        })
        models.append(clf)
    return pd.DataFrame(rows), models

# ----- Define configs -----
# Base (Cell 12) for reference
params_base = dict(
    n_estimators=900, learning_rate=0.045,
    num_leaves=64, max_depth=-1,
    min_child_samples=250,
    subsample=0.75, colsample_bytree=0.75,
    max_bin=255,
    reg_alpha=0.0, reg_lambda=1.5,
)
mono_base = build_monotone(X_cols_base, amt_pct_monotone=True)

# 1) WOE soft-clip (requested ±4.5; no effect since prior step clipped in-place to ±3)
params_clip = params_base.copy()
params_clip["_woe_soft_clip"] = (-4.5, 4.5)
mono_clip = mono_base  # same constraints

# 2) Monotone relax (only TransactionAmt)
params_mono_relax = params_base.copy()
mono_relax = build_monotone(X_cols_base, amt_pct_monotone=False)

# 3) Looser regularization
params_loose = dict(
    n_estimators=900, learning_rate=0.045,
    num_leaves=64, max_depth=-1,
    min_child_samples=200,
    subsample=0.80, colsample_bytree=0.80,
    max_bin=255,
    reg_alpha=0.0, reg_lambda=1.0,
)

# 4) Burstiness horizon tweak (swap cnt_15m → add cnt_5m & cnt_60m)
fe_burst_alt = [c for c in fe_keep if c != "cnt_15m"] + ["cnt_5m", "cnt_60m"]
X_cols_burst_alt = X_cols + fe_burst_alt
params_burst = params_base.copy()
mono_burst = build_monotone(X_cols_burst_alt, amt_pct_monotone=True)

# ----- Run configs -----
res_clip, _    = run_config("woe_soft_clip_±4.5", X_cols_base, params_clip.copy(), mono_clip)
res_mono, _    = run_config("monotone_amt_only", X_cols_base, params_mono_relax.copy(), mono_relax)
res_loose, _   = run_config("looser_reg", X_cols_base, params_loose.copy(), mono_base)
res_burst, _   = run_config("burst_5m+60m", X_cols_burst_alt, params_burst.copy(), mono_burst)

# ----- Compare to Cell 12 baseline -----
def summarize_vs_baseline(res_cfg: pd.DataFrame, res_base: pd.DataFrame, label: str):
    m = res_base[["fold","aupr","recall@k"]].merge(
        res_cfg[["fold","aupr","recall@k"]], on="fold", suffixes=("_base","_cfg")
    )
    m["Δ_aupr"] = m["aupr_cfg"] - m["aupr_base"]
    m["Δ_recall@k"] = m["recall@k_cfg"] - m["recall@k_base"]
    mean_row = pd.DataFrame([{
        "config": label,
        "aupr_base_mean": res_base["aupr"].mean(),
        "aupr_cfg_mean":  res_cfg["aupr"].mean(),
        "Δ_aupr_mean":    m["Δ_aupr"].mean(),
        "rec_base_mean":  res_base["recall@k"].mean(),
        "rec_cfg_mean":   res_cfg["recall@k"].mean(),
        "Δ_rec_mean":     m["Δ_recall@k"].mean()
    }])
    return m, mean_row

# Baseline results from Cell 12:
res_base = res_lgbm_restruct.copy()
tables = []
means = []

for label, res_cfg in [
    ("woe_soft_clip_±4.5", res_clip),
    ("monotone_amt_only",  res_mono),
    ("looser_reg",         res_loose),
    ("burst_5m+60m",       res_burst),
]:
    comp, mean_row = summarize_vs_baseline(res_cfg, res_base, label)
    print(f"[COMPARE vs Cell12] {label}:")
    display(comp.style.format({
        "aupr_base":"{:.4f}","aupr_cfg":"{:.4f}",
        "recall@k_base":"{:.4f}","recall@k_cfg":"{:.4f}",
        "Δ_aupr":"{:+.4f}","Δ_recall@k":"{:+.4f}"
    }))
    means.append(mean_row)

summary = pd.concat(means, ignore_index=True)
print("[SUMMARY] Mean deltas vs Cell 12 pruned baseline:")
display(summary.style.format({
    "aupr_base_mean":"{:.4f}","aupr_cfg_mean":"{:.4f}","Δ_aupr_mean":"{:+.4f}",
    "rec_base_mean":"{:.4f}","rec_cfg_mean":"{:.4f}","Δ_rec_mean":"{:+.4f}"
}))


# === Cell 14: 60-day slice + DART LightGBM (3 forward-chaining folds) with validated feature choices ===
# Objective:
# - Rebuild a 60-day slice and features
# - Train LightGBM DART with monotone constraint only on TransactionAmt
# - 3 forward-chaining folds (purge=1 day); report AUCPR & P/R@k

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

label_col = "isFraud"
K = OPS["review_capacity_rate"]

# -----------------------
# 1) Rebuild the 60-day slice
# -----------------------
EDA_DAYS_60 = 60
if SMALL_RUN and SLICE_MODE == "by_days":
    con.sql(f"""
    CREATE OR REPLACE TEMP TABLE train_slice_tbl AS
    SELECT *
    FROM train_enriched_tbl
    WHERE ts_day >= {EDA_START_DAY} AND ts_day < {EDA_START_DAY + EDA_DAYS_60};
    """)
# else: leave FULL or fraction mode as-is

# -----------------------
# 2) Pull and rebuild features on 60-day
# -----------------------
df = con.sql("""
    SELECT 
        TransactionID, TransactionDT, ts_day, isFraud,
        TransactionAmt, card1, card2, DeviceInfo, P_emaildomain
    FROM train_slice_tbl
""").to_df()

# Chrono ordering & event time
df = df.sort_values("TransactionDT").reset_index(drop=True)
df["isFraud"] = df["isFraud"].astype(np.uint8)
df["event_time"] = pd.to_datetime(df["TransactionDT"], unit="s", origin="unix")

# Entity and groupby
ENTITY = "card1"
g = df.groupby(ENTITY, dropna=False, sort=False)
base_col = "TransactionID"
amt = df["TransactionAmt"].astype("float64")
eps = 1e-9

# --- Base velocity/behavior (as-of) ---
first_time = g["event_time"].transform("min")
df["is_new_entity"] = (df["event_time"] == first_time).astype(np.uint8)

prev_time = g["event_time"].shift(1)
df["time_since_prev_sec"] = (df["event_time"] - prev_time).dt.total_seconds().fillna(1e12).astype("float64")

cnt_1d = (g.rolling("1D", on="event_time")[base_col].count().reset_index(level=0, drop=True).astype("float64") - 1.0)
cnt_7d = (g.rolling("7D", on="event_time")[base_col].count().reset_index(level=0, drop=True).astype("float64") - 1.0)
df["cnt_1d"] = np.clip(cnt_1d.to_numpy(), 0.0, None)
df["cnt_7d"] = np.clip(cnt_7d.to_numpy(), 0.0, None)

amt_sum_1d = (
    g.rolling("1D", on="event_time")["TransactionAmt"].sum()
     .reset_index(level=0, drop=True).astype("float64")
     - amt.to_numpy()
)
amt_med_7d = (
    g.rolling("7D", on="event_time")["TransactionAmt"].median()
     .reset_index(level=0, drop=True).astype("float64")
)
df["amt_sum_1d"] = np.nan_to_num(amt_sum_1d, nan=0.0, posinf=0.0, neginf=0.0)
df["amt_median_7d"] = np.nan_to_num(amt_med_7d, nan=0.0, posinf=0.0, neginf=0.0)

# --- Personalized amount (7d) ---
roll7_mean = g.rolling("7D", on="event_time")["TransactionAmt"].mean().reset_index(level=0, drop=True).astype("float64")
roll7_std  = g.rolling("7D", on="event_time")["TransactionAmt"].std(ddof=0).reset_index(level=0, drop=True).astype("float64")
roll7_med  = g.rolling("7D", on="event_time")["TransactionAmt"].median().reset_index(level=0, drop=True).astype("float64")
roll7_q75  = g.rolling("7D", on="event_time")["TransactionAmt"].quantile(0.75).reset_index(level=0, drop=True).astype("float64")
roll7_q25  = g.rolling("7D", on="event_time")["TransactionAmt"].quantile(0.25).reset_index(level=0, drop=True).astype("float64")
roll7_iqr  = (roll7_q75 - roll7_q25).to_numpy()

df["amt_z7"]        = (amt.to_numpy() - roll7_mean.to_numpy()) / (np.sqrt(roll7_std.to_numpy()**2) + eps)
df["amt_rz7"]       = (amt.to_numpy() - roll7_med.to_numpy())  / (roll7_iqr / 1.349 + eps)
df["amt_pct_med7"]  = amt.to_numpy() / (roll7_med.to_numpy() + eps)
df["amt_above_med7"]= np.clip(amt.to_numpy() - roll7_med.to_numpy(), 0.0, None)

# --- Burstiness (5m, 60m) + gap CV on last 5 events ---
cnt_5m  = (g.rolling("5min",  on="event_time")[base_col].count().reset_index(level=0, drop=True).astype("float64") - 1.0)
cnt_60m = (g.rolling("60min", on="event_time")[base_col].count().reset_index(level=0, drop=True).astype("float64") - 1.0)
df["cnt_5m"]  = np.clip(cnt_5m.to_numpy(), 0.0, None)
df["cnt_60m"] = np.clip(cnt_60m.to_numpy(), 0.0, None)

df["gap_sec"] = (df["event_time"] - g["event_time"].shift(1)).dt.total_seconds()
roll_gap_mean5 = df.groupby(ENTITY, dropna=False, sort=False)["gap_sec"].rolling(5).mean().reset_index(level=0, drop=True).astype("float64")
roll_gap_std5  = df.groupby(ENTITY, dropna=False, sort=False)["gap_sec"].rolling(5).std(ddof=0).reset_index(level=0, drop=True).astype("float64")
df["gap_mean5"] = np.nan_to_num(roll_gap_mean5.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
df["gap_std5"]  = np.nan_to_num(roll_gap_std5.to_numpy(),  nan=0.0, posinf=0.0, neginf=0.0)
df["gap_cv5"]   = np.nan_to_num(df["gap_std5"] / (df["gap_mean5"] + 1e-6), nan=0.0, posinf=0.0, neginf=0.0)

# Base feature cols (velocity + amounts)
base_feats = [
    "TransactionAmt",
    "is_new_entity","time_since_prev_sec",
    "cnt_1d","cnt_7d","amt_sum_1d","amt_median_7d",
    "amt_z7","amt_rz7","amt_pct_med7","amt_above_med7",
    "cnt_5m","cnt_60m","gap_cv5"
]

# Start df_feat frame
df_feat = df[["TransactionID","ts_day",label_col] + base_feats].copy()

# -----------------------
# 3) Train-only encoders per fold (normalized + hierarchical + baseline raw)
# -----------------------
# Normalization
df["device_brand"] = (
    df["DeviceInfo"].astype(str).str.lower()
      .str.split("/", n=1, expand=False).str[0]
      .str.split(" ",  n=1, expand=False).str[0]
)
df["email_tld"] = (
    df["P_emaildomain"].astype(str).str.lower()
      .str.split(".", expand=False).str[-1]
)

# Utility mappers (reuse earlier logic)
def fit_mappings(train_df: pd.DataFrame, cat: str, y: str):
    g_rate = train_df[y].mean()
    grp = train_df.groupby(cat, observed=False, dropna=False)[y].agg(["count","sum"]).rename(columns={"sum":"pos"})
    m = 50.0
    rate = (grp["pos"] + m * g_rate) / (grp["count"] + m)
    odds_c = rate / (1.0 - rate + 1e-12)
    odds_g = g_rate / (1.0 - g_rate + 1e-12)
    woe = np.log(odds_c / (odds_g + 1e-12))
    rarity = np.log1p(grp["count"])
    return rarity, woe, float(g_rate)

def apply_map(df_sub: pd.DataFrame, key: str, series_map: pd.Series):
    return df_sub[key].map(series_map).fillna(0.0).to_numpy(dtype=np.float32)

def fit_hier_maps(train_df: pd.DataFrame, child: str, parent: str, y: str, m: float = 50.0):
    g = train_df[y].mean()
    par = (train_df.groupby(parent, observed=False, dropna=False)[y].agg(count="size", pos="sum"))
    par_rate = (par["pos"] + m * g) / (par["count"] + m)
    ch = (train_df.groupby([child,parent], observed=False, dropna=False)[y].agg(count="size", pos="sum")).reset_index()
    ch["rate_child"]  = (ch["pos"] + m * g) / (ch["count"] + m)
    ch = ch.merge(par_rate.rename("rate_parent").to_frame().reset_index(), on=parent, how="left")
    cnt = ch["count"].to_numpy()
    rc  = ch["rate_child"].to_numpy()
    rp  = ch["rate_parent"].to_numpy()
    r_back = rc.copy()
    mask_1 = cnt == 1
    r_back[mask_1] = 0.5*rc[mask_1] + 0.5*rp[mask_1]
    odds = lambda r: r / (1.0 - r + 1e-12)
    w_child  = np.log(odds(rc)    / (g / (1.0 - g + 1e-12) + 1e-12))
    w_back   = np.log(odds(r_back)/ (g / (1.0 - g + 1e-12) + 1e-12))
    child_key = ch[child].astype(object)
    return (pd.Series(w_child, index=child_key),
            pd.Series(w_back,  index=child_key))

# Prepare columns to receive encodings
enc_cols = [
    "device_brand_woe","email_tld_woe",
    "device_backoff_woe","email_backoff_woe",
    "P_emaildomain_woe","DeviceInfo_woe"
]
for c in enc_cols:
    df_feat[c] = np.nan

# -----------------------
# 4) Build 3 forward-chaining folds (purge=1 day)
# -----------------------
unique_days = np.sort(df_feat["ts_day"].unique())
n_days = len(unique_days)
def _qday(q): 
    return unique_days[int(np.floor(q * (n_days - 1)))]
# thirds
c1 = _qday(1/3)
c2 = _qday(2/3)
PURGE_DAYS = 1
folds_60 = []
# Fold1: valid [c1, c2)
train_idx_1 = np.flatnonzero(df_feat["ts_day"].values < (c1 - PURGE_DAYS))
valid_idx_1 = np.flatnonzero((df_feat["ts_day"].values >= c1) & (df_feat["ts_day"].values < c2))
folds_60.append({"train_idx": train_idx_1, "valid_idx": valid_idx_1})
# Fold2: valid [c2, max)
train_idx_2 = np.flatnonzero(df_feat["ts_day"].values < (c2 - PURGE_DAYS))
valid_idx_2 = np.flatnonzero(df_feat["ts_day"].values >= c2)
folds_60.append({"train_idx": train_idx_2, "valid_idx": valid_idx_2})
# Fold3: expand/train to c2, validate on the last ~1/3 again with a different split (sliding backstop)
# (To get 3 folds without shrinking valid too much, create an early split as well)
c0 = _qday(0.0)  # start
mid = _qday(0.5)
train_idx_0 = np.flatnonzero(df_feat["ts_day"].values < (mid - PURGE_DAYS))
valid_idx_0 = np.flatnonzero((df_feat["ts_day"].values >= mid) & (df_feat["ts_day"].values < c2))
folds_60.insert(0, {"train_idx": train_idx_0, "valid_idx": valid_idx_0})  # make it Fold1

print(f"[CV-60d] Built {len(folds_60)} folds over days {int(unique_days.min())}..{int(unique_days.max())}")

# -----------------------
# 5) Per-fold: fit encoders (train-only), apply to train/valid
# -----------------------
for f in folds_60:
    tr_idx, va_idx = f["train_idx"], f["valid_idx"]
    tr = df.loc[tr_idx, ["device_brand","email_tld","DeviceInfo","P_emaildomain", label_col]].copy()
    va = df.loc[va_idx, ["device_brand","email_tld","DeviceInfo","P_emaildomain", label_col]].copy()

    # normalized encoders
    _, w_brand, _ = fit_mappings(tr.rename(columns={"device_brand":"cat"}), "cat", label_col)
    _, w_tld,   _ = fit_mappings(tr.rename(columns={"email_tld":"cat"}),   "cat", label_col)
    df_feat.loc[tr_idx, "device_brand_woe"] = apply_map(tr, "device_brand", w_brand)
    df_feat.loc[va_idx, "device_brand_woe"] = apply_map(va, "device_brand", w_brand)
    df_feat.loc[tr_idx, "email_tld_woe"]    = apply_map(tr, "email_tld",   w_tld)
    df_feat.loc[va_idx, "email_tld_woe"]    = apply_map(va, "email_tld",   w_tld)

    # hierarchical backoffs
    w_child_dev, w_back_dev = fit_hier_maps(tr.rename(columns={"DeviceInfo":"child","device_brand":"parent"}), "child", "parent", label_col)
    w_child_eml, w_back_eml = fit_hier_maps(tr.rename(columns={"P_emaildomain":"child","email_tld":"parent"}), "child", "parent", label_col)
    df_feat.loc[tr_idx, "device_backoff_woe"] = apply_map(tr.rename(columns={"DeviceInfo":"child"}), "child", w_back_dev)
    df_feat.loc[va_idx, "device_backoff_woe"] = apply_map(va.rename(columns={"DeviceInfo":"child"}), "child", w_back_dev)
    df_feat.loc[tr_idx, "email_backoff_woe"]  = apply_map(tr.rename(columns={"P_emaildomain":"child"}), "child", w_back_eml)
    df_feat.loc[va_idx, "email_backoff_woe"]  = apply_map(va.rename(columns={"P_emaildomain":"child"}), "child", w_back_eml)

    # baseline WOEs on raw fields
    _, w_devinfo, _ = fit_mappings(tr.rename(columns={"DeviceInfo":"cat"}), "cat", label_col)
    _, w_email,   _ = fit_mappings(tr.rename(columns={"P_emaildomain":"cat"}), "cat", label_col)
    df_feat.loc[tr_idx, "DeviceInfo_woe"]    = apply_map(tr, "DeviceInfo",    w_devinfo)
    df_feat.loc[va_idx, "DeviceInfo_woe"]    = apply_map(va, "DeviceInfo",    w_devinfo)
    df_feat.loc[tr_idx, "P_emaildomain_woe"] = apply_map(tr, "P_emaildomain", w_email)
    df_feat.loc[va_idx, "P_emaildomain_woe"] = apply_map(va, "P_emaildomain", w_email)

# Fill NaNs & cast
for c in enc_cols:
    df_feat[c] = df_feat[c].astype("float32").fillna(0.0)

# -----------------------
# 6) Train DART LGBM across folds (with soft WOE clipping on the fly)
# -----------------------
X_cols_60 = base_feats + enc_cols

def build_monotone(features):
    mono = []
    for f in features:
        mono.append(+1 if f == "TransactionAmt" else 0)
    return mono

lgbm_dart = LGBMClassifier(
    objective="binary",
    boosting_type="dart",
    n_estimators=100,
    learning_rate=0.01,
    num_leaves=96,
    max_bin=1023,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.85,
    reg_alpha=1.0,
    reg_lambda=1.5,
    class_weight="balanced",
    monotone_constraints=build_monotone(X_cols_60),
    n_jobs=MODEL_THREADS,
    random_state=SEED
)

rows_60, models_60 = [], []
for i, f in enumerate(folds_60, 1):
    tr_idx, va_idx = f["train_idx"], f["valid_idx"]
    Xtr = df_feat.loc[tr_idx, X_cols_60].astype("float32").copy()
    Xva = df_feat.loc[va_idx, X_cols_60].astype("float32").copy()
    ytr = df_feat.loc[tr_idx, label_col].to_numpy(dtype=np.uint8)
    yva = df_feat.loc[va_idx, label_col].to_numpy(dtype=np.uint8)

    # Soft clip WOEs at ±3.5 (do not mutate df_feat)
    woe_cols_local = [c for c in X_cols_60 if "woe" in c]
    if woe_cols_local:
        Xtr[woe_cols_local] = Xtr[woe_cols_local].clip(-3.5, 3.5)
        Xva[woe_cols_local] = Xva[woe_cols_local].clip(-3.5, 3.5)

    # Clean numerics
    Xtr = Xtr.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    Xva = Xva.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    mdl = lgbm_dart.fit(
        Xtr.values, ytr,
        eval_set=[(Xva.values, yva)],
        eval_metric="average_precision",
    )
    yhat = mdl.predict_proba(Xva.values)[:, 1]
    aupr = aucpr(yva, yhat)
    p_at_k, r_at_k = precision_recall_at_k(yva, yhat, K)

    rows_60.append({
        "fold": i, "model": "lgbm_dart_60d",
        "aupr": aupr, "prec@k": p_at_k, "recall@k": r_at_k,
        "best_iter": getattr(mdl, "best_iteration_", None)
    })
    models_60.append(mdl)

res_lgbm_dart_60 = pd.DataFrame(rows_60)
summary_60 = pd.DataFrame({
    "aupr_mean":   [res_lgbm_dart_60["aupr"].mean()],
    "aupr_std":    [res_lgbm_dart_60["aupr"].std()],
    "prec_k_mean": [res_lgbm_dart_60["prec@k"].mean()],
    "recall_k_mean":[res_lgbm_dart_60["recall@k"].mean()],
})

print("[60D DART] AUCPR & P/R@k per fold:")
display(res_lgbm_dart_60[["fold","aupr","prec@k","recall@k","best_iter"]].style.format({
    "aupr":"{:.4f}", "prec@k":"{:.4f}", "recall@k":"{:.4f}"
}))
print("\n[SUMMARY] 60-day DART (3 folds):")
display(summary_60.style.format({
    "aupr_mean":"{:.4f}", "aupr_std":"{:.4f}",
    "prec_k_mean":"{:.4f}", "recall_k_mean":"{:.4f}"
}))
print("[KEEP] Stored fold-wise models in `models_60` and results in `res_lgbm_dart_60`.")



# === Cell 16: Finalize 60-day DART model (5-fold CV, no HPO) ===
# Purpose:
# - Train LightGBM DART with a sensible, split-friendly config (max_bin=1023, min_child_samples=150)
# - Per-fold train-only encoders; soft-clip WOE at input; monotone only on TransactionAmt
# - Report AUCPR & P/R@k; save best model + encoder maps for write-up/export

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

label_col = "isFraud"
K = OPS["review_capacity_rate"]

# --- Feature lists (from Cell 14) ---
base_feats = [
    "TransactionAmt",
    "is_new_entity","time_since_prev_sec",
    "cnt_1d","cnt_7d","amt_sum_1d","amt_median_7d",
    "amt_z7","amt_rz7","amt_pct_med7","amt_above_med7",
    "cnt_5m","cnt_60m","gap_cv5"
]
enc_cols = [
    "device_brand_woe","email_tld_woe",
    "device_backoff_woe","email_backoff_woe",
    "P_emaildomain_woe","DeviceInfo_woe"
]
X_cols_60 = base_feats + enc_cols

# --- CV folds (prefer 5-folds; else fallback to 3) ---
cv_splits = folds_5 if "folds_5" in globals() and len(folds_5) >= 2 else folds_60

# --- Encoder helpers (from Cell 14); returns enc_tr, enc_va and also the maps for export ---
def fit_mappings(train_df: pd.DataFrame, cat: str, y: str):
    g_rate = train_df[y].mean()
    grp = train_df.groupby(cat, observed=False, dropna=False)[y].agg(["count","sum"]).rename(columns={"sum":"pos"})
    m = 50.0
    rate = (grp["pos"] + m * g_rate) / (grp["count"] + m)
    odds_c = rate / (1.0 - rate + 1e-12)
    odds_g = g_rate / (1.0 - g_rate + 1e-12)
    woe = np.log(odds_c / (odds_g + 1e-12))
    return woe, float(g_rate), m

def fit_hier_maps(train_df: pd.DataFrame, child: str, parent: str, y: str, m: float = 50.0):
    g = train_df[y].mean()
    par = train_df.groupby(parent, observed=False, dropna=False)[y].agg(count="size", pos="sum")
    par_rate = (par["pos"] + m * g) / (par["count"] + m)
    ch = (train_df.groupby([child,parent], observed=False, dropna=False)[y]
          .agg(count="size", pos="sum")).reset_index()
    ch["rate_child"] = (ch["pos"] + m * g) / (ch["count"] + m)
    ch = ch.merge(par_rate.rename("rate_parent").to_frame().reset_index(), on=parent, how="left")
    cnt = ch["count"].to_numpy()
    rc  = ch["rate_child"].to_numpy()
    rp  = ch["rate_parent"].to_numpy()
    # backoff: if count==1, 50/50 blend; else child
    r_back = rc.copy()
    mask_1 = cnt == 1
    r_back[mask_1] = 0.5*rc[mask_1] + 0.5*rp[mask_1]
    odds = lambda r: r / (1.0 - r + 1e-12)
    w_child = np.log(odds(rc)    / (g / (1.0 - g + 1e-12) + 1e-12))
    w_back  = np.log(odds(r_back)/ (g / (1.0 - g + 1e-12) + 1e-12))
    key = ch[child].astype(object)
    return pd.Series(w_child, index=key), pd.Series(w_back, index=key), float(g), m

def apply_encoders_for_fold(tr_idx, va_idx, clip=3.5):
    tr = df.loc[tr_idx, ["device_brand","email_tld","DeviceInfo","P_emaildomain", label_col]].copy()
    va = df.loc[va_idx, ["device_brand","email_tld","DeviceInfo","P_emaildomain", label_col]].copy()

    # normalized cats
    w_brand, g_brand, m_brand = fit_mappings(tr.rename(columns={"device_brand":"cat"}), "cat", label_col)
    w_tld,   g_tld,   m_tld   = fit_mappings(tr.rename(columns={"email_tld":"cat"}),   "cat", label_col)
    # hierarchical backoffs
    w_child_dev, w_back_dev, g_dev, m_dev = fit_hier_maps(tr.rename(columns={"DeviceInfo":"child","device_brand":"parent"}), "child", "parent", label_col)
    w_child_eml, w_back_eml, g_eml, m_eml = fit_hier_maps(tr.rename(columns={"P_emaildomain":"child","email_tld":"parent"}),  "child", "parent", label_col)
    # baseline raw WOEs
    w_devinfo, g_dv, m_dv = fit_mappings(tr.rename(columns={"DeviceInfo":"cat"}),    "cat", label_col)
    w_email,   g_em, m_em = fit_mappings(tr.rename(columns={"P_emaildomain":"cat"}), "cat", label_col)

    def m(series, mp):
        return series.map(mp).fillna(0.0).astype("float32").clip(-clip, clip)

    enc_tr = pd.DataFrame(index=tr.index)
    enc_va = pd.DataFrame(index=va.index)
    enc_tr["device_brand_woe"] = m(tr["device_brand"], w_brand)
    enc_va["device_brand_woe"] = m(va["device_brand"], w_brand)
    enc_tr["email_tld_woe"]    = m(tr["email_tld"],   w_tld)
    enc_va["email_tld_woe"]    = m(va["email_tld"],   w_tld)
    enc_tr["device_backoff_woe"] = m(tr["DeviceInfo"],   w_back_dev)
    enc_va["device_backoff_woe"] = m(va["DeviceInfo"],   w_back_dev)
    enc_tr["email_backoff_woe"]  = m(tr["P_emaildomain"],w_back_eml)
    enc_va["email_backoff_woe"]  = m(va["P_emaildomain"],w_back_eml)
    enc_tr["DeviceInfo_woe"]     = m(tr["DeviceInfo"],   w_devinfo)
    enc_va["DeviceInfo_woe"]     = m(va["DeviceInfo"],   w_devinfo)
    enc_tr["P_emaildomain_woe"]  = m(tr["P_emaildomain"],w_email)
    enc_va["P_emaildomain_woe"]  = m(va["P_emaildomain"],w_email)

    maps = {
        "w_brand": w_brand, "w_tld": w_tld,
        "w_child_dev": w_child_dev, "w_back_dev": w_back_dev,
        "w_child_eml": w_child_eml, "w_back_eml": w_back_eml,
        "w_devinfo": w_devinfo, "w_email": w_email,
        "globals": {
            "g_brand": g_brand, "g_tld": g_tld, "g_dev": g_dev, "g_eml": g_eml,
            "g_devinfo": g_dv, "g_email": g_em,
            "m": 50.0, "clip": clip
        }
    }
    return enc_tr.astype("float32"), enc_va.astype("float32"), maps

def build_monotone(features):
    return [1 if f == "TransactionAmt" else 0 for f in features]

# --- Final DART config (split-friendly & stable) ---
params_final = dict(
    objective="binary",
    boosting_type="dart",
    xgboost_dart_mode=True,
    learning_rate=0.025,
    n_estimators=100,
    num_leaves=31,
    max_bin=1023,           # encourage splits; reduces -inf best gain
    min_child_samples=75,  # allow more splits per node than 200
    subsample=0.9,
    colsample_bytree=0.85,
    reg_alpha=1.0,
    reg_lambda=1.5,
    class_weight="balanced",
    n_jobs=MODEL_THREADS,
    random_state=SEED
)

rows_final, models_final, maps_per_fold = [], [], []
for i, f in enumerate(cv_splits, 1):
    tr_idx, va_idx = f["train_idx"], f["valid_idx"]
    # Base numerics
    Xtr_base = df_feat.loc[tr_idx, base_feats].astype("float32").copy()
    Xva_base = df_feat.loc[va_idx, base_feats].astype("float32").copy()
    ytr = df_feat.loc[tr_idx, label_col].to_numpy(dtype=np.uint8)
    yva = df_feat.loc[va_idx, label_col].to_numpy(dtype=np.uint8)

    # Train-only encoders for this fold
    enc_tr, enc_va, maps = apply_encoders_for_fold(tr_idx, va_idx, clip=3.5)
    maps_per_fold.append(maps)

    # Assemble matrices
    Xtr = pd.concat([Xtr_base.reset_index(drop=True), enc_tr.reset_index(drop=True)], axis=1)[X_cols_60]
    Xva = pd.concat([Xva_base.reset_index(drop=True), enc_va.reset_index(drop=True)], axis=1)[X_cols_60]

    # Model with monotone only on amount
    clf = LGBMClassifier(**params_final, monotone_constraints=build_monotone(X_cols_60))
    clf.fit(
        Xtr.values, ytr,
        eval_set=[(Xva.values, yva)],
        eval_metric="average_precision"
    )
    yhat = clf.predict_proba(Xva.values)[:, 1]
    aupr = aucpr(yva, yhat)
    p_at_k, r_at_k = precision_recall_at_k(yva, yhat, OPS["review_capacity_rate"])

    rows_final.append({
        "fold": i, "aupr": aupr, "prec@k": p_at_k, "recall@k": r_at_k,
        "best_iter": getattr(clf, "best_iteration_", None) or params_final["n_estimators"]
    })
    models_final.append(clf)

res_lgbm_final = pd.DataFrame(rows_final)
summary_final = pd.DataFrame({
    "aupr_mean":   [res_lgbm_final["aupr"].mean()],
    "aupr_std":    [res_lgbm_final["aupr"].std()],
    "prec_k_mean": [res_lgbm_final["prec@k"].mean()],
    "recall_k_mean":[res_lgbm_final["recall@k"].mean()],
})

print("[FINAL 60D DART] AUCPR & P/R@k per fold (5-fold CV if available):")
display(res_lgbm_final[["fold","aupr","prec@k","recall@k","best_iter"]].style.format({
    "aupr":"{:.4f}", "prec@k":"{:.4f}", "recall@k":"{:.4f}"
}))
print("\n[SUMMARY] Final 60-day DART (no HPO):")
display(summary_final.style.format({
    "aupr_mean":"{:.4f}", "aupr_std":"{:.4f}",
    "prec_k_mean":"{:.4f}", "recall_k_mean":"{:.4f}"
}))
print("[KEEP] Saved per-fold models in `models_final`, results in `res_lgbm_final`, and encoder `maps_per_fold` for the write-up.")


# === Cell 17 (fixed): Calibration (Isotonic) + Thresholds on 60-day 5-fold OOF ===
# - Fully self-contained: includes encoder helpers used in Cell 16
# - Builds OOF raw & calibrated scores (time-true), then prints AUCPR + capacity thresholds

from sklearn.isotonic import IsotonicRegression

label_col = "isFraud"
capacity_grid = [0.01, 0.03, 0.05, 0.10, OPS["review_capacity_rate"]]

# Feature lists (same as Cell 16)
base_feats = [
    "TransactionAmt",
    "is_new_entity","time_since_prev_sec",
    "cnt_1d","cnt_7d","amt_sum_1d","amt_median_7d",
    "amt_z7","amt_rz7","amt_pct_med7","amt_above_med7",
    "cnt_5m","cnt_60m","gap_cv5"
]
enc_cols = [
    "device_brand_woe","email_tld_woe",
    "device_backoff_woe","email_backoff_woe",
    "P_emaildomain_woe","DeviceInfo_woe"
]
X_cols_60 = base_feats + enc_cols

# CV splits & trained models from Cell 16
cv_splits = folds_5 if "folds_5" in globals() and len(folds_5) >= 2 else folds_60
assert len(models_final) >= len(cv_splits), "models_final does not match the number of CV splits."

# ---------- helpers (same logic as Cell 16) ----------
def fit_mappings(train_df: pd.DataFrame, cat: str, y: str):
    g_rate = train_df[y].mean()
    grp = train_df.groupby(cat, observed=False, dropna=False)[y].agg(["count","sum"]).rename(columns={"sum":"pos"})
    m = 50.0
    rate = (grp["pos"] + m * g_rate) / (grp["count"] + m)
    odds_c = rate / (1.0 - rate + 1e-12)
    odds_g = g_rate / (1.0 - g_rate + 1e-12)
    woe = np.log(odds_c / (odds_g + 1e-12))
    return woe, float(g_rate), m

def fit_hier_maps(train_df: pd.DataFrame, child: str, parent: str, y: str, m: float = 50.0):
    g = train_df[y].mean()
    par = train_df.groupby(parent, observed=False, dropna=False)[y].agg(count="size", pos="sum")
    par_rate = (par["pos"] + m * g) / (par["count"] + m)
    ch = (train_df.groupby([child,parent], observed=False, dropna=False)[y]
          .agg(count="size", pos="sum")).reset_index()
    ch["rate_child"] = (ch["pos"] + m * g) / (ch["count"] + m)
    ch = ch.merge(par_rate.rename("rate_parent").to_frame().reset_index(), on=parent, how="left")
    cnt = ch["count"].to_numpy()
    rc  = ch["rate_child"].to_numpy()
    rp  = ch["rate_parent"].to_numpy()
    r_back = rc.copy()
    mask_1 = cnt == 1
    r_back[mask_1] = 0.5*rc[mask_1] + 0.5*rp[mask_1]
    odds = lambda r: r / (1.0 - r + 1e-12)
    w_child = np.log(odds(rc)    / (g / (1.0 - g + 1e-12) + 1e-12))
    w_back  = np.log(odds(r_back)/ (g / (1.0 - g + 1e-12) + 1e-12))
    key = ch[child].astype(object)
    return pd.Series(w_child, index=key), pd.Series(w_back, index=key), float(g), m

def apply_encoders_for_fold(tr_idx, va_idx, clip=3.5):
    tr = df.loc[tr_idx, ["device_brand","email_tld","DeviceInfo","P_emaildomain", label_col]].copy()
    va = df.loc[va_idx, ["device_brand","email_tld","DeviceInfo","P_emaildomain", label_col]].copy()

    w_brand, _, _ = fit_mappings(tr.rename(columns={"device_brand":"cat"}), "cat", label_col)
    w_tld,   _, _ = fit_mappings(tr.rename(columns={"email_tld":"cat"}),   "cat", label_col)
    w_child_dev, w_back_dev, _, _ = fit_hier_maps(tr.rename(columns={"DeviceInfo":"child","device_brand":"parent"}), "child", "parent", label_col)
    w_child_eml, w_back_eml, _, _ = fit_hier_maps(tr.rename(columns={"P_emaildomain":"child","email_tld":"parent"}),  "child", "parent", label_col)
    w_devinfo, _, _ = fit_mappings(tr.rename(columns={"DeviceInfo":"cat"}),    "cat", label_col)
    w_email,   _, _ = fit_mappings(tr.rename(columns={"P_emaildomain":"cat"}), "cat", label_col)

    def m(series, mp):
        return series.map(mp).fillna(0.0).astype("float32").clip(-clip, clip)

    enc_tr = pd.DataFrame(index=tr.index)
    enc_va = pd.DataFrame(index=va.index)
    enc_tr["device_brand_woe"] = m(tr["device_brand"],  w_brand)
    enc_va["device_brand_woe"] = m(va["device_brand"],  w_brand)
    enc_tr["email_tld_woe"]    = m(tr["email_tld"],     w_tld)
    enc_va["email_tld_woe"]    = m(va["email_tld"],     w_tld)
    enc_tr["device_backoff_woe"] = m(tr["DeviceInfo"],   w_back_dev)
    enc_va["device_backoff_woe"] = m(va["DeviceInfo"],   w_back_dev)
    enc_tr["email_backoff_woe"]  = m(tr["P_emaildomain"],w_back_eml)
    enc_va["email_backoff_woe"]  = m(va["P_emaildomain"],w_back_eml)
    enc_tr["DeviceInfo_woe"]     = m(tr["DeviceInfo"],   w_devinfo)
    enc_va["DeviceInfo_woe"]     = m(va["DeviceInfo"],   w_devinfo)
    enc_tr["P_emaildomain_woe"]  = m(tr["P_emaildomain"],w_email)
    enc_va["P_emaildomain_woe"]  = m(va["P_emaildomain"],w_email)
    return enc_tr.astype("float32"), enc_va.astype("float32")

def build_matrices_for_fold(tr_idx, va_idx, clip=3.5):
    Xtr_base = df_feat.loc[tr_idx, base_feats].astype("float32").copy()
    Xva_base = df_feat.loc[va_idx, base_feats].astype("float32").copy()
    enc_tr, enc_va = apply_encoders_for_fold(tr_idx, va_idx, clip=clip)
    Xtr = pd.concat([Xtr_base.reset_index(drop=True), enc_tr.reset_index(drop=True)], axis=1)[X_cols_60]
    Xva = pd.concat([Xva_base.reset_index(drop=True), enc_va.reset_index(drop=True)], axis=1)[X_cols_60]
    ytr = df_feat.loc[tr_idx, label_col].to_numpy(dtype=np.uint8)
    yva = df_feat.loc[va_idx, label_col].to_numpy(dtype=np.uint8)
    return Xtr.values, ytr, Xva.values, yva

def pr_at_capacity(y_true, scores, cap_rate):
    n = len(scores)
    k = max(1, int(np.floor(cap_rate * n)))
    order = np.argsort(-scores)
    topk = order[:k]
    thr = float(scores[order[k-1]]) if k <= n else float(scores[order[-1]])
    tp = int(y_true[topk].sum())
    prec = tp / k
    rec = tp / max(1, int(y_true.sum()))
    return prec, rec, thr, k

# ---------- build OOF raw & calibrated ----------
n_total = len(df_feat)
raw_oof = np.full(n_total, np.nan, dtype=np.float32)
cal_oof = np.full(n_total, np.nan, dtype=np.float32)
y_oof   = df_feat[label_col].to_numpy(dtype=np.uint8)

for i, f in enumerate(cv_splits, 1):
    tr_idx, va_idx = f["train_idx"], f["valid_idx"]
    Xtr, ytr, Xva, yva = build_matrices_for_fold(tr_idx, va_idx, clip=3.5)
    mdl = models_final[i-1]

    raw_tr = mdl.predict_proba(Xtr)[:, 1]
    raw_va = mdl.predict_proba(Xva)[:, 1]

    # time-true calibration slice inside TRAIN: last 20% of days
    tr_days = df_feat.iloc[tr_idx]["ts_day"].to_numpy()
    cutoff = np.quantile(tr_days, 0.80)
    calib_mask = tr_days >= cutoff
    if calib_mask.sum() < 100:
        cutoff = np.quantile(tr_days, 0.90)
        calib_mask = tr_days >= cutoff

    if 50 <= calib_mask.sum() < len(tr_idx):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_tr[calib_mask], ytr[calib_mask])
        cal_va = iso.transform(raw_va)
    else:
        cal_va = raw_va.copy()

    raw_oof[va_idx] = raw_va.astype(np.float32)
    cal_oof[va_idx] = cal_va.astype(np.float32)

filled = np.isfinite(raw_oof).sum()
print(f"[CAL] Filled OOF predictions for {filled:,} rows out of {n_total:,}.")

# ---------- metrics & thresholds ----------
oo_raw = aucpr(y_oof[np.isfinite(raw_oof)], raw_oof[np.isfinite(raw_oof)])
oo_cal = aucpr(y_oof[np.isfinite(cal_oof)], cal_oof[np.isfinite(cal_oof)])
print(f"[OOF] AUCPR raw={oo_raw:.4f} | calibrated={oo_cal:.4f}")

rows = []
for cap in capacity_grid:
    p_raw, r_raw, thr_raw, k_raw = pr_at_capacity(y_oof, raw_oof, cap)
    p_cal, r_cal, thr_cal, k_cal = pr_at_capacity(y_oof, cal_oof, cap)
    rows.append({
        "capacity": cap,
        "alerts_raw": k_raw, "thr_raw": thr_raw, "prec_raw": p_raw, "recall_raw": r_raw,
        "alerts_cal": k_cal, "thr_cal": thr_cal, "prec_cal": p_cal, "recall_cal": p_cal and r_cal  # keep both
    })
thr_table = pd.DataFrame(rows)
thr_table["recall_cal"] = [pr_at_capacity(y_oof, cal_oof, cap)[1] for cap in capacity_grid]  # fix typo above

print("[THRESHOLDS] Precision/Recall at capacity levels (OOF):")
display(thr_table.style.format({
    "capacity":"{:.0%}",
    "prec_raw":"{:.4f}","recall_raw":"{:.4f}","thr_raw":"{:.5f}",
    "prec_cal":"{:.4f}","recall_cal":"{:.4f}","thr_cal":"{:.5f}"
}))

# ---------- save artifacts ----------
oof_df = pd.DataFrame({
    "TransactionID": df_feat["TransactionID"].values,
    "y": y_oof.astype(np.uint8),
    "score_raw": raw_oof.astype(np.float32),
    "score_cal": cal_oof.astype(np.float32),
})
oof_path = "oof_60d_scores.csv"
thr_path = "thresholds_60d_oof.csv"
oof_df.to_csv(oof_path, index=False)
thr_table.to_csv(thr_path, index=False)
print(f"[SAVE] OOF scores → {oof_path}")
print(f"[SAVE] Threshold table → {thr_path}")



# === Cell 18: Model card + artifacts snapshot ===
# Purpose:
# - Emit a concise, portable model card (JSON) and persist small artifacts for the portfolio write-up.

import json
import time

# Metrics summary from Cell 16
summary = {
    "aupr_mean": float(res_lgbm_final["aupr"].mean()),
    "aupr_std": float(res_lgbm_final["aupr"].std()),
    "prec_k_mean": float(res_lgbm_final["prec@k"].mean()),
    "recall_k_mean": float(res_lgbm_final["recall@k"].mean()),
    "folds": int(len(res_lgbm_final)),
}

# Thresholds summary from Cell 17
thr_preview = pd.read_csv("thresholds_60d_oof.csv").to_dict(orient="records")

# Params used (from Cell 16)
params_used = dict(**params_final)
params_used["monotone_constraints"] = "+1 on TransactionAmt only"
params_used["woe_clip"] = "±3.5 at input time"
params_used["encoder_m"] = 50.0

# Feature list
features_used = {
    "base_feats": base_feats,
    "enc_feats": enc_cols
}

# CV description
cv_desc = {
    "window_days": 60,
    "cv": "forward-chaining",
    "folds": int(len(cv_splits)),
    "purge_days": 1,
    "label": label_col,
    "entity": "card1",
}

# Risks & mitigations
risks = [
    "Data drift in categorical distributions (email/domain/device).",
    "Cold-start entities (new cards/devices) with little history.",
    "Threshold mismatch when fraud base rate shifts.",
]
mitigations = [
    "Monitor PSI on inputs monthly; retrain maps weekly on rolling window.",
    "Back-off encoders to parent (brand/TLD); default WOE=0.0 for unseen.",
    "Capacity-linked thresholds; recalibrate (isotonic) periodically.",
]

# Future work
future = [
    "Compact HPO over (learning_rate × n_estimators), num_leaves, min_child_samples (50–200), max_bin (511–1023), subsample/colsample (0.8–1.0), DART drop_rate/skip_drop.",
    "Add EWMA features (6h/24h) and merchant/network graph signals.",
    "Profit curve with ops cost/benefit to set operating threshold directly.",
]

model_card = {
    "meta": {
        "created_utc": int(time.time()),
        "project": "IEEE-CIS Fraud Detection (portfolio)",
        "version": "v1.0-60d-dart-nohpo",
    },
    "framing": {
        "objective": "Rank fraudulent transactions for review.",
        "primary_metric": "AUCPR (average precision)",
        "ops_metrics": "Precision/Recall@k at capacity",
    },
    "data_cv": cv_desc,
    "features": features_used,
    "encoders": {
        "type": "WOE + hierarchical back-off (train-only per fold)",
        "clip": "±3.5",
        "m_smoothing": 50.0,
        "unseen_handling": "WOE=0.0 (neutral)",
    },
    "model": {
        "algo": "LightGBM (DART)",
        "params": params_used,
    },
    "results": {
        "cv_summary": summary,
        "thresholds_preview": thr_preview,
        "oof_files": {
            "scores": "oof_60d_scores.csv",
            "thresholds": "thresholds_60d_oof.csv"
        }
    },
    "risks": risks,
    "mitigations": mitigations,
    "future_work": future
}

# Save artifacts
with open("model_card_60d.json", "w") as f:
    json.dump(model_card, f, indent=2)

res_lgbm_final.to_csv("cv_results_60d.csv", index=False)

print("[SAVE] Model card → model_card_60d.json")
print("[SAVE] CV results → cv_results_60d.csv")
print("[SAVE] OOF scores → oof_60d_scores.csv (from Cell 17)")
print("[SAVE] Thresholds → thresholds_60d_oof.csv (from Cell 17)")

# Quick peek
print("\nModel card preview:")
print(json.dumps({k: model_card[k] for k in ("meta","framing","results")}, indent=2))


