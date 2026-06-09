# cell 1 - init

import os, json, math, warnings, random, time
import numpy as np, pandas as pd

# logging for run-time
def log(msg, level="INFO"):
    print(f"[{level} {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

class Timer:
    def __init__(self, label): self.label=label
    def __enter__(self):
        self._t0=time.time(); log(f"{self.label} — started"); return self
    def __exit__(self, *exc):
        log(f"{self.label} — finished in {int((time.time()-self._t0)*1000)} ms")

def set_global_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception: pass

# config
class CFG:
    # project & paths
    PROJECT     = "kaggle-lmsys-pairwise"
    DATASET_DIR = "/kaggle/input/lmsys-chatbot-arena"     # fixed Kaggle input
    WORK_DIR    = "/kaggle/working"                       # ephemeral outputs
    # artifacts (kept stable across cells)
    BASELINE_ART_DIR = os.path.join(WORK_DIR, "artifacts_baseline_v2")
    ALIGN_DIR        = os.path.join(WORK_DIR, "artifacts_align_v2")

    # runtime
    SEED       = 42
    N_THREADS  = max(1, (os.cpu_count() or 4) - 1)

    # data handling
    TIE_POLICY = "drop"     # we'll drop ties (A/B labels cleanly defined)
    # fast-dev switch for quick notebook runs
    DEV_MODE       = True   # flip to False for full data
    DEV_MAX_TRAIN  = 5_000  # cap train rows
    DEV_MAX_VALID  = 1_000  # cap valid rows

    # CV choices
    CV_FOLDS       = 5
    CV_REFIT_TEXT  = True   # strict leakage-free CV by default

# train.csv schema
SCHEMA = {
    "prompt":          "prompt",
    "response_a":      "response_a",
    "response_b":      "response_b",
    "model_a":         "model_a",
    "model_b":         "model_b",
    "winner_model_a":  "winner_model_a",
    "winner_model_b":  "winner_model_b",
    "winner_tie":      "winner_tie",
}

def maybe_downsample(df, kind: str, seed: int = CFG.SEED):
    """
    If CFG.DEV_MODE is True, cap dataframe size for speed.
    kind ∈ {'train','valid','generic'} controls cap size.
    """
    if not CFG.DEV_MODE: return df
    n = len(df)
    cap = CFG.DEV_MAX_TRAIN if kind == "train" else CFG.DEV_MAX_VALID if kind == "valid" else min(CFG.DEV_MAX_TRAIN, 6000)
    if n > cap:
        return df.sample(n=cap, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)

def clean_text_series(s):
    return s.astype("string").fillna("").str.replace(r"\s+", " ", regex=True).str.strip()

# artifacts
os.makedirs(CFG.BASELINE_ART_DIR, exist_ok=True)
os.makedirs(CFG.ALIGN_DIR, exist_ok=True)

# seed
warnings.filterwarnings("ignore")
set_global_seed(CFG.SEED)

print("=== CONFIG ===")
print(json.dumps({
    "dataset_dir": CFG.DATASET_DIR,
    "work_dir": CFG.WORK_DIR,
    "seed": CFG.SEED,
    "threads": CFG.N_THREADS,
    "dev_mode": CFG.DEV_MODE,
    "dev_caps": {"train": CFG.DEV_MAX_TRAIN, "valid": CFG.DEV_MAX_VALID},
    "tie_policy": CFG.TIE_POLICY,
    "cv_folds": CFG.CV_FOLDS,
    "cv_refit_text": CFG.CV_REFIT_TEXT,
}, indent=2))


TEST_CSV = os.path.join(CFG.DATASET_DIR, "test.csv")
print(pd.read_csv(TEST_CSV, nrows=0).columns.tolist())


# cell 2 - datapaths & EDA

import os
import numpy as np
import pandas as pd

with Timer("load-train/test"):
    TRAIN_CSV = os.path.join(CFG.DATASET_DIR, "train.csv")
    TEST_CSV  = os.path.join(CFG.DATASET_DIR, "test.csv")
    assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
    assert os.path.exists(TEST_CSV),  f"Missing: {TEST_CSV}"

    tr_cols = [
        SCHEMA["prompt"], SCHEMA["response_a"], SCHEMA["response_b"],
        SCHEMA["model_a"], SCHEMA["model_b"],
        SCHEMA["winner_model_a"], SCHEMA["winner_model_b"], SCHEMA["winner_tie"],
    ]
    te_cols = [
        SCHEMA["prompt"], SCHEMA["response_a"], SCHEMA["response_b"], 
    ]

    train_df = pd.read_csv(TRAIN_CSV, usecols=tr_cols, dtype="string", keep_default_na=True)
    test_df  = pd.read_csv(TEST_CSV,  usecols=te_cols, dtype="string", keep_default_na=True)

    for c in [SCHEMA["prompt"], SCHEMA["response_a"], SCHEMA["response_b"]]:
        train_df[c] = clean_text_series(train_df[c])
        test_df[c]  = clean_text_series(test_df[c])

    log(f"train shape: {train_df.shape} | test shape: {test_df.shape}")
    display(train_df.head(3))
    display(test_df.head(3))

# null counts
with Timer("null-counts"):
    key_cols_train = tr_cols
    key_cols_test  = te_cols

    def null_table(df, cols):
        n = len(df)
        s = df[cols].isna().sum()
        return pd.DataFrame({"nulls": s, "pct": (s / max(n,1) * 100).round(2)}).sort_values("nulls", ascending=False)

    log("Nulls — train (key columns)")
    display(null_table(train_df, key_cols_train))
    log("Nulls — test (key columns)")
    display(null_table(test_df, key_cols_test))

# label distribution
with Timer("label-eda"):
    a_win = train_df[SCHEMA["winner_model_a"]].astype("Int64")
    b_win = train_df[SCHEMA["winner_model_b"]].astype("Int64")
    ties  = train_df[SCHEMA["winner_tie"]].astype("Int64")

    bad_tie = (ties == 1) & ((a_win.fillna(0) != 0) | (b_win.fillna(0) != 0))
    bad_notie = (ties != 1) & ~(((a_win.fillna(0) == 1) & (b_win.fillna(0) == 0)) | ((a_win.fillna(0) == 0) & (b_win.fillna(0) == 1)))
    invalid_n = int((bad_tie | bad_notie).sum())

    n_total = len(train_df)
    n_tie   = int((ties == 1).sum())
    mask_nt = (ties != 1)
    n_nt    = int(mask_nt.sum())
    n_a     = int(((a_win == 1) & mask_nt).sum())
    n_b     = int(((b_win == 1) & mask_nt).sum())

    tie_rate = n_tie / max(n_total,1)
    a_rate_nt = n_a / max(n_nt,1)
    b_rate_nt = n_b / max(n_nt,1)
    a_bias = a_rate_nt - 0.5  # positive => A favored

    print(f"Rows total: {n_total:,}")
    print(f"Ties: {n_tie:,} ({tie_rate:.2%}) | Non-ties: {n_nt:,}")
    print(f"Among non-ties -> A wins: {n_a:,} ({a_rate_nt:.2%}) | B wins: {n_b:,} ({b_rate_nt:.2%}) | A-bias: {a_bias:+.4f}")
    print(f"Inconsistent label rows (to drop later): {invalid_n:,}")

# text-length stats
with Timer("text-length-stats"):
    def len_stats(df, col):
        s = df[col].astype("string")
        Lc = s.str.len().astype("float32")
        Lw = (s.str.count(" ") + 1).astype("float32")
        q = [0.05, 0.5, 0.95]
        return pd.DataFrame({
            "mean_chars": [float(Lc.mean())],
            "p05_chars":  [float(Lc.quantile(q[0]))],
            "p50_chars":  [float(Lc.quantile(q[1]))],
            "p95_chars":  [float(Lc.quantile(q[2]))],
            "mean_words": [float(Lw.mean())],
            "p50_words":  [float(Lw.quantile(0.5))],
            "p95_words":  [float(Lw.quantile(0.95))],
        })

    cols_txt = [SCHEMA["prompt"], SCHEMA["response_a"], SCHEMA["response_b"]]
    out = []
    for c in cols_txt:
        t = len_stats(train_df, c); t.index = [c]; out.append(t)
    stats_tbl = pd.concat(out, axis=0)
    display(stats_tbl)

# tiny peek
if CFG.DEV_MODE:
    with Timer("dev-peek"):
        tr_dev = maybe_downsample(train_df, kind="train")
        te_dev = maybe_downsample(test_df, kind="generic")
        print(f"[DEV MODE] preview sizes -> train: {len(tr_dev):,} | test: {len(te_dev):,}")



# cell 3 - labels & ties

with Timer("load-raw"):
    tr_cols = [
        SCHEMA["prompt"], SCHEMA["response_a"], SCHEMA["response_b"],
        SCHEMA["model_a"], SCHEMA["model_b"],
        SCHEMA["winner_model_a"], SCHEMA["winner_model_b"], SCHEMA["winner_tie"],
    ]
    test_cols = ["id", SCHEMA["prompt"], SCHEMA["response_a"], SCHEMA["response_b"]]

    train_df = pd.read_csv(TRAIN_CSV, usecols=tr_cols, dtype="string", keep_default_na=True)
    test_df  = pd.read_csv(TEST_CSV,  usecols=test_cols, dtype="string", keep_default_na=True)
    log(f"loaded: train={train_df.shape} | test={test_df.shape}")

# labels & tie-policy
with Timer("labels-clean"):
    a_win = train_df[SCHEMA["winner_model_a"]].astype("Int64")
    b_win = train_df[SCHEMA["winner_model_b"]].astype("Int64")
    tie   = train_df[SCHEMA["winner_tie"]].astype("Int64")

    # inconsistent rows: tie==1 but any winner==1, or tie!=1 but not exactly one winner==1
    bad_tie   = (tie == 1) & ((a_win.fillna(0) != 0) | (b_win.fillna(0) != 0))
    exactly_one_winner = ((a_win.fillna(0) == 1) & (b_win.fillna(0) == 0)) | ((a_win.fillna(0) == 0) & (b_win.fillna(0) == 1))
    bad_notie = (tie != 1) & (~exactly_one_winner)
    invalid_mask = (bad_tie | bad_notie)

    n_total = len(train_df)
    n_invalid = int(invalid_mask.sum())
    n_ties = int((tie == 1).sum())

    # apply tie policy (drop ties)
    keep_mask = (~invalid_mask) & (tie != 1)
    clean = train_df.loc[keep_mask].copy().reset_index(drop=True)

    # binary label: y=1 if A wins else 0
    clean["y"] = (clean[SCHEMA["winner_model_a"]].astype("Int64") == 1).astype("int8")

    log(f"rows total={n_total:,} | drop ties={n_ties:,} | drop invalid={n_invalid:,} | kept (non-ties, valid)={len(clean):,}")

with Timer("sanitize-text"):
    for c in [SCHEMA["prompt"], SCHEMA["response_a"], SCHEMA["response_b"], SCHEMA["model_a"], SCHEMA["model_b"]]:
        clean[c] = clean[c].astype("string").fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    for c in [SCHEMA["prompt"], SCHEMA["response_a"], SCHEMA["response_b"]]:
        test_df[c] = test_df[c].astype("string").fillna("").str.replace(r"\s+", " ", regex=True).str.strip()

with Timer("label-distribution"):
    y = clean["y"].values
    a_rate = float(y.mean())
    b_rate = 1.0 - a_rate
    a_bias = a_rate - 0.5
    print(f"A wins: {a_rate:.2%} | B wins: {b_rate:.2%} | A-bias: {a_bias:+.4f}")

with Timer("maybe-downsample-train"):
    clean_dev = maybe_downsample(clean, kind="train")
    log(f"train rows after DEV cap (if any): {len(clean_dev):,}")

# persist
with Timer("persist-parquets"):
    pairs_train = clean_dev[[SCHEMA["prompt"], SCHEMA["response_a"], SCHEMA["response_b"], SCHEMA["model_a"], SCHEMA["model_b"], "y"]].copy()
    pairs_train_full = clean[[SCHEMA["prompt"], SCHEMA["response_a"], SCHEMA["response_b"], SCHEMA["model_a"], SCHEMA["model_b"], "y"]].copy()
    keep_test = [SCHEMA["prompt"], SCHEMA["response_a"], SCHEMA["response_b"]]
    pairs_test  = test_df[keep_test].copy()

    pt_path = os.path.join(CFG.WORK_DIR, "pairs_train.parquet")
    ps_path = os.path.join(CFG.WORK_DIR, "pairs_test.parquet")
    pt_full_path = os.path.join(CFG.WORK_DIR, "pairs_train_full.parquet")
    pairs_train.to_parquet(pt_path, index=False)
    pairs_test.to_parquet(ps_path, index=False)
    pairs_train_full.to_parquet(pt_full_path, index=False)
    log(f"saved -> {pt_path} ({pairs_train.shape})")
    log(f"saved -> {ps_path} ({pairs_test.shape})")
    log(f"saved -> {pt_full_path} ({pairs_train_full.shape})")

display(pairs_train.head(3))
display(pairs_test.head(3))


#cell4 EDA
import re

PT_PATH = os.path.join(CFG.WORK_DIR, "pairs_train.parquet")
assert os.path.exists(PT_PATH), "Run Cell 3 first to create pairs_train.parquet"
df = pd.read_parquet(PT_PATH)

P, A, B = SCHEMA["prompt"], SCHEMA["response_a"], SCHEMA["response_b"]
MA, MB  = SCHEMA["model_a"], SCHEMA["model_b"]

INCLUDE_MODEL_EDA = False 

with Timer("basic-stats"):
    n = len(df); n_prompts = df[P].nunique()
    y = df["y"].astype(int).values
    a_rate = float(y.mean()); b_rate = 1.0 - a_rate; a_bias = a_rate - 0.5
    print(f"rows={n:,} | unique prompts={n_prompts:,}")
    print(f"y: A wins={a_rate:.2%} | B wins={b_rate:.2%} | A-bias={a_bias:+.4f}")

# numeric deltas
with Timer("compute-del-numeric"):
    def _num_feats(s: pd.Series):
        s = s.astype("string")
        n_char = s.str.len()
        n_space = s.str.count(" ")
        n_word = (n_space + 1).clip(lower=1)
        return pd.DataFrame({
            "n_char": n_char.astype("float32"),
            "n_word": n_word.astype("float32"),
            "avg_wlen": (n_char / n_word).clip(upper=30).astype("float32"),
            "n_q": s.str.count(r"\?").astype("float32"),
            "n_e": s.str.count(r"!").astype("float32"),
            "n_nl": s.str.count(r"\n").astype("float32"),
            "has_codefence": s.str.contains(r"```").astype("int8"),
        })

    NA = _num_feats(df[A]); NB = _num_feats(df[B])
    d_num = (NA - NB).astype("float32")
    d_num.columns = [f"del_{c}" for c in d_num.columns]

# jaccard delta
with Timer("compute-del-jaccard"):
    tok = re.compile(r"\w+")
    def _tokset(x: str): return set(tok.findall((x or "").lower()))
    def _jacc(p, r):
        Pset, Rset = _tokset(p), _tokset(r)
        if not Pset and not Rset: return 1.0
        u = len(Pset | Rset)
        return (len(Pset & Rset) / u) if u else 0.0

    jac_a = np.array([_jacc(p, r) for p, r in zip(df[P].values, df[A].values)], dtype="float32")
    jac_b = np.array([_jacc(p, r) for p, r in zip(df[P].values, df[B].values)], dtype="float32")
    d_align = pd.DataFrame({"del_jaccard": jac_a - jac_b})

eda = pd.concat([d_num, d_align, df[[MA, MB, "y"]]], axis=1)
del_cols = [c for c in eda.columns if c.startswith("del_")]
print("delta columns:", del_cols)

# means by class
with Timer("delta-summaries"):
    def _summ_with_class(col: str) -> pd.Series:
        v = pd.to_numeric(eda[col], errors="coerce").values
        yv = eda["y"].astype(int).values

        v1 = v[yv == 1]  # A wins
        v0 = v[yv == 0]  # B wins

        # means
        m1 = float(np.nanmean(v1)) if v1.size else np.nan
        m0 = float(np.nanmean(v0)) if v0.size else np.nan

        # effect size (Cohen's d, pooled std)
        s1 = np.nanstd(v1, ddof=1) if v1.size > 1 else 0.0
        s0 = np.nanstd(v0, ddof=1) if v0.size > 1 else 0.0
        n1 = max(len(v1), 1); n0 = max(len(v0), 1)
        sp = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / max(n1 + n0 - 2, 1))
        d  = (m1 - m0) / (sp if sp > 0 else 1.0)

        # share of positives (A>B)
        pct_pos1 = float(np.nanmean(v1 > 0)) if v1.size else np.nan
        pct_pos0 = float(np.nanmean(v0 > 0)) if v0.size else np.nan

        diff = m1 - m0
        favors = "A (y=1)" if diff > 0 else "B (y=0)" if diff < 0 else "tie"

        return pd.Series({
            "mean_y1_Awin": m1,
            "mean_y0_Bwin": m0,
            "diff_m1_m0": diff,
            "cohen_d": d,
            "pct_pos_y1": pct_pos1,
            "pct_pos_y0": pct_pos0,
            "favors_when_higher": favors
        })

    key_cols = ["del_n_char","del_n_word","del_avg_wlen","del_n_q","del_n_e","del_n_nl",
                "del_has_codefence","del_jaccard"]
    key_cols = [c for c in key_cols if c in eda.columns]

    summ = pd.concat({c: _summ_with_class(c) for c in key_cols}, axis=1).T
    summ.index.name = "feature"
    summ["abs_cohen_d"] = summ["cohen_d"].abs()
    summ = summ.sort_values("abs_cohen_d", ascending=False)

    display(summ.round(4))

# dupes
with Timer("prompt-reuse-dups"):
    per_prompt = df[P].value_counts()
    print("pairs per prompt (p50/p90/p99/max):",
          np.percentile(per_prompt.values, [50,90,99]).round(1).tolist(),
          int(per_prompt.max()))
    dup_a = int(df[A].duplicated().sum()); dup_b = int(df[B].duplicated().sum())
    print(f"duplicate responses → A:{dup_a:,} | B:{dup_b:,}")

if INCLUDE_MODEL_EDA:
    with Timer("model-identity"):
        total = df[MA].value_counts().add(df[MB].value_counts(), fill_value=0).astype(int)
        win_a = df.loc[df["y"]==1, MA].value_counts()
        win_b = df.loc[df["y"]==0, MB].value_counts()
        wins = win_a.add(win_b, fill_value=0).astype(int)
        win_rate = (wins / total.replace(0, np.nan)).rename("win_rate")
        id_tbl = (pd.concat([total.rename("appearances"), wins.rename("wins"), win_rate], axis=1)
                    .sort_values("appearances", ascending=False))
        display(id_tbl.head(20))

# persist
with Timer("persist-eda-summary"):
    eda_summary = {
        "rows": int(n),
        "unique_prompts": int(n_prompts),
        "y_rate_A": a_rate,
        "A_bias": a_bias,
        "prompt_pairs_p50_p90_p99_max": [float(np.percentile(per_prompt.values, p)) for p in [50,90,99]] + [int(per_prompt.max())],
        "delta_signals": summ.round(4).to_dict(orient="index"),
        "include_model_eda": bool(INCLUDE_MODEL_EDA),
    }
    if INCLUDE_MODEL_EDA:
        eda_summary["top_models"] = id_tbl.head(10)[["appearances","wins","win_rate"]].round(4).to_dict(orient="index")
    with open(os.path.join(CFG.WORK_DIR, "eda_summary.json"), "w") as f:
        json.dump(eda_summary, f, indent=2)
    log("Saved eda_summary.json")


# cell 5 - folds
from sklearn.model_selection import StratifiedGroupKFold

PT_PATH = os.path.join(CFG.WORK_DIR, "pairs_train.parquet")
assert os.path.exists(PT_PATH), "Run Cell 3 first."
df = pd.read_parquet(PT_PATH)

P, A, B = SCHEMA["prompt"], SCHEMA["response_a"], SCHEMA["response_b"]
MA, MB  = SCHEMA["model_a"], SCHEMA["model_b"]

# reused feats
tok = re.compile(r"\w+")
def _tokset(x: str): return set(tok.findall((x or "").lower()))

def _num_feats(s: pd.Series) -> pd.DataFrame:
    s = s.astype("string")
    n_char = s.str.len()
    n_space = s.str.count(" ")
    n_word = (n_space + 1).clip(lower=1)
    return pd.DataFrame({
        "n_char": n_char.astype("float32"),
        "n_word": n_word.astype("float32"),
        "avg_wlen": (n_char / n_word).clip(upper=30).astype("float32"),
        "n_q": s.str.count(r"\?").astype("float32"),
        "n_e": s.str.count(r"!").astype("float32"),
        "n_nl": s.str.count(r"\n").astype("float32"),
        "has_codefence": s.str.contains(r"```").astype("int8"),
    })

def _jacc_series(prompt_s: pd.Series, resp_s: pd.Series) -> np.ndarray:
    out = np.empty(len(prompt_s), dtype="float32")
    for i, (p, r) in enumerate(zip(prompt_s.astype("string").values, resp_s.astype("string").values)):
        Pset, Rset = _tokset(p), _tokset(r)
        if not Pset and not Rset: out[i] = 1.0
        else:
            u = len(Pset | Rset); out[i] = (len(Pset & Rset) / u) if u else 0.0
    return out

with Timer("compute fold diagnostics features"):
    # prompt/basic lengths
    pr_len_w = (df[P].astype("string").str.count(" ") + 1).astype("float32")
    pr_len_c = df[P].astype("string").str.len().astype("float32")
    # response stats
    NA = _num_feats(df[A]); NB = _num_feats(df[B])
    # deltas 
    d_num = (NA - NB).astype("float32")
    d_num.columns = [f"del_{c}" for c in d_num.columns]
    # alignment delta (Jaccard)
    del_jaccard = _jacc_series(df[P], df[A]) - _jacc_series(df[P], df[B])
    d_align = pd.DataFrame({"del_jaccard": del_jaccard})
    eda = pd.concat([
        pd.DataFrame({"prompt_words": pr_len_w, "prompt_chars": pr_len_c}).astype("float32"),
        NA.add_prefix("A_").astype("float32"),
        NB.add_prefix("B_").astype("float32"),
        d_num, d_align,
        df[["y", P]]
    ], axis=1)

# fold
n_folds = CFG.CV_FOLDS
splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=CFG.SEED) \
.split(X=np.zeros(len(df)), y=df["y"].values, groups=df[P].values)
groups = df[P].astype("string").values
idx = np.arange(len(df))
fold_assign = -np.ones(len(df), dtype="int8")

fold_rows = []
diag = []

with Timer(f"build {n_folds} folds"):
    for fold, (tr_idx, va_idx) in enumerate(splitter, 1):
        fold_assign[va_idx] = fold

        # leakage check
        prompts_tr = set(df.iloc[tr_idx][P].values.tolist())
        prompts_va = set(df.iloc[va_idx][P].values.tolist())
        inter = prompts_tr & prompts_va
        assert len(inter) == 0, f"Prompt leakage detected in fold {fold}!"

        va = eda.iloc[va_idx].copy()
        # per-fold validation diagnostics
        n_va = len(va)
        uniq_prompts = int(va[P].nunique())
        y_mean = float(va["y"].astype(int).mean())  # A win %
        # prompt lengths
        pw_mean = float(va["prompt_words"].mean()); pc_mean = float(va["prompt_chars"].mean())
        # response means
        a_w_mean = float(va["A_n_word"].mean()); b_w_mean = float(va["B_n_word"].mean())
        a_c_mean = float(va["A_n_char"].mean()); b_c_mean = float(va["B_n_char"].mean())

        # selected deltas: coverage & structure
        sel = ["del_n_char","del_n_word","del_avg_wlen","del_n_q","del_n_e","del_n_nl","del_has_codefence","del_jaccard"]
        sel = [c for c in sel if c in va.columns]
        del_means = {c: float(np.nanmean(va[c].values)) for c in sel}

        diag.append({
            "fold": fold,
            "valid_rows": n_va,
            "valid_unique_prompts": uniq_prompts,
            "valid_y_mean_Awin": y_mean,
            "prompt_mean_words": pw_mean,
            "prompt_mean_chars": pc_mean,
            "A_mean_words": a_w_mean,
            "A_mean_chars": a_c_mean,
            "B_mean_words": b_w_mean,
            "B_mean_chars": b_c_mean,
            **{f"mean_{k}": v for k, v in del_means.items()},
        })

        fold_rows.append(pd.DataFrame({"row_idx": va_idx, "fold": fold, "group_prompt": df.iloc[va_idx][P].values}))

# persist
cv_map = pd.concat(fold_rows, axis=0).sort_values(["fold","row_idx"]).reset_index(drop=True)
cv_path = os.path.join(CFG.WORK_DIR, "cv_folds_gkf.parquet")
cv_map.to_parquet(cv_path, index=False)

diag_path = os.path.join(CFG.WORK_DIR, "cv_fold_diag.json")
with open(diag_path, "w") as f:
    json.dump(diag, f, indent=2)

log(f"Saved fold assignment → {cv_path}")
log(f"Saved per-fold diagnostics → {diag_path}")

# summary
diag_df = pd.DataFrame(diag)
display(diag_df.round(4))
print("\n[Fold variability]")
for col in ["valid_y_mean_Awin","prompt_mean_words","A_mean_words","B_mean_words","mean_del_jaccard"]:
    if col in diag_df.columns:
        mu, sd = diag_df[col].mean(), diag_df[col].std(ddof=1)
        print(f"{col}: {mu:.4f} ± {sd:.4f}")


# cell 6 - vectorize & baseline
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import gc

warnings.filterwarnings("ignore")

PT_PATH  = os.path.join(CFG.WORK_DIR, "pairs_train.parquet")
CV_PATH  = os.path.join(CFG.WORK_DIR, "cv_folds_gkf.parquet")
assert os.path.exists(PT_PATH) and os.path.exists(CV_PATH), "Run Cells 3 and 5 first."

df = pd.read_parquet(PT_PATH).reset_index(drop=True)
cv_map = pd.read_parquet(CV_PATH)  # columns: row_idx, fold, group_prompt

P, A, B = SCHEMA["prompt"], SCHEMA["response_a"], SCHEMA["response_b"]

# recompute
def numeric_feats(s: pd.Series) -> pd.DataFrame:
    s = s.astype("string")
    n_char = s.str.len().astype("float32")
    n_space = s.str.count(" ").astype("float32")
    n_word = (n_space + 1.0).clip(lower=1.0)
    return pd.DataFrame({
        "n_char": n_char,
        "n_word": n_word,
        "avg_wlen": (n_char / n_word).clip(upper=30.0).astype("float32"),
        "n_q": s.str.count(r"\?").astype("float32"),
        "n_e": s.str.count(r"!").astype("float32"),
        "n_nl": s.str.count(r"\n").astype("float32"),
        "has_codefence": s.str.contains(r"```").astype("int8"),
    })

ART_DIR = os.path.join(CFG.WORK_DIR, "artifacts_sgd_v1")
os.makedirs(ART_DIR, exist_ok=True)

# use row_idx for stability
truth_tbl = pd.DataFrame({
    "row_idx": np.arange(len(df), dtype=int),
    "y": df["y"].astype(int).to_numpy()
})

# fold loop
n_folds = int(getattr(CFG, "CV_FOLDS", 5))
oof = np.full(len(df), np.nan, dtype=np.float32) 
fold_metrics = []

for fold in range(1, n_folds+1):
    va_idx = cv_map.loc[cv_map["fold"] == fold, "row_idx"].to_numpy()
    tr_idx = cv_map.loc[cv_map["fold"] != fold, "row_idx"].to_numpy()

    tr = df.iloc[tr_idx].reset_index(drop=True)
    va = df.iloc[va_idx].reset_index(drop=True)

    # char tf-idf delta per-fold
    vec_char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(4, 5), min_df=5, lowercase=True,
        max_features=200_000, sublinear_tf=True, dtype=np.float32
    )
    vec_char.fit(pd.concat([tr[A], tr[B]], axis=0).astype("string").tolist())

    XA_tr = vec_char.transform(tr[A]); XB_tr = vec_char.transform(tr[B])
    XA_va = vec_char.transform(va[A]); XB_va = vec_char.transform(va[B])
    X_char_tr = XA_tr - XB_tr
    X_char_va = XA_va - XB_va

    # scaled numeric delta per-fold
    numA_tr = numeric_feats(tr[A]); numB_tr = numeric_feats(tr[B])
    numA_va = numeric_feats(va[A]); numB_va = numeric_feats(va[B])
    del_num_tr = (numA_tr - numB_tr).astype("float32")
    del_num_va = (numA_va - numB_va).astype("float32")

    scaler_num = MaxAbsScaler().fit(del_num_tr.values)
    X_num_tr = scaler_num.transform(del_num_tr.values)
    X_num_va = scaler_num.transform(del_num_va.values)

    # assemble vec_char & numeric
    X_tr = hstack([X_char_tr, csr_matrix(X_num_tr)], format="csr")
    X_va = hstack([X_char_va, csr_matrix(X_num_va)], format="csr")

    clf = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=1e-4,
        max_iter=20, tol=1e-3, n_iter_no_change=3, random_state=CFG.SEED
    )
    clf.fit(X_tr, tr["y"].astype(int).values)

    p_va = clf.predict_proba(X_va)[:, 1].astype(np.float32)

    pred_tbl = pd.DataFrame({
        "row_idx": va_idx,   
        "p": p_va
    })
    eval_tbl = pred_tbl.merge(truth_tbl, on="row_idx", how="inner")

    p_eval = eval_tbl["p"].to_numpy(dtype=np.float32)
    y_eval = eval_tbl["y"].to_numpy(dtype=int)
    assert len(p_eval) == len(y_eval) > 0, f"Fold {fold}: empty or mismatched after alignment."

    yhat = (p_eval >= 0.5).astype("int8")
    auc_val = np.nan
    if (y_eval.min() != y_eval.max()):
        try:
            auc_val = roc_auc_score(y_eval, p_eval)
        except Exception:
            auc_val = np.nan

    fold_metrics.append({
        "fold": fold,
        "acc": accuracy_score(y_eval, yhat),
        "auc": auc_val,
        "brier": brier_score_loss(y_eval, p_eval),
        "logloss": log_loss(y_eval, p_eval, labels=[0,1]),
        "n_valid": int(len(y_eval))
    })

    # oof per row_idx
    oof[eval_tbl["row_idx"].to_numpy()] = p_eval

    # cleanup big mats
    del X_char_tr, X_char_va, XA_tr, XB_tr, XA_va, XB_va, X_tr, X_va
    gc.collect()

# aggregate oof metrics
m_df = pd.DataFrame(fold_metrics)
acc_mu, acc_sd = float(np.nanmean(m_df["acc"])), float(np.nanstd(m_df["acc"], ddof=1))
auc_mu, auc_sd = float(np.nanmean(m_df["auc"])), float(np.nanstd(m_df["auc"], ddof=1))
bri_mu, bri_sd = float(np.nanmean(m_df["brier"])), float(np.nanstd(m_df["brier"], ddof=1))
ll_mu,  ll_sd  = float(np.nanmean(m_df["logloss"])), float(np.nanstd(m_df["logloss"], ddof=1))

print(f"[CV MEAN]  Acc={acc_mu:.4f}±{acc_sd:.4f} | AUC={auc_mu:.4f}±{auc_sd:.4f} | "
      f"Brier={bri_mu:.4f}±{bri_sd:.4f} | LogLoss={ll_mu:.4f}±{ll_sd:.4f}")
display(m_df.round(4))

# persist oof & metrics
fold_by_row = cv_map.set_index("row_idx").reindex(range(len(df)))["fold"].to_numpy()

oof_tbl = pd.DataFrame({
    "row_idx": np.arange(len(df), dtype=int),
    "fold": fold_by_row,
    "y": truth_tbl["y"].values,
    "oof_pred_sgd": oof,  # may contain NaN for rows never evaluated
})
oof_path = os.path.join(ART_DIR, "oof_sgd.parquet")
oof_tbl.to_parquet(oof_path, index=False)

summary = {
    "model": "SGDClassifier(log_loss,l2,alpha=1e-4,max_iter=20,early_stopping)",
    "features": {
        "del_char_tfidf": {"analyzer": "char_wb", "ngram_range": [4,5], "min_df": 5, "max_features": 200_000, "sublinear_tf": True},
        "del_num": ["n_char","n_word","avg_wlen","n_q","n_e","n_nl","has_codefence"]
    },
    "cv": {"folds": int(n_folds), "group": "prompt"},
    "metrics_by_fold": m_df.to_dict(orient="records"),
    "metrics_mean": {"acc": acc_mu, "auc": auc_mu, "brier": bri_mu, "logloss": ll_mu}
}
sum_path = os.path.join(ART_DIR, "cv_sgd_summary.json")
with open(sum_path, "w") as f:
    json.dump(summary, f, indent=2)

log(f"Saved OOF → {oof_path}")
log(f"Saved CV summary → {sum_path}")

# oof sanity
with Timer("oof-quicklook"):
    y = oof_tbl["y"].values
    p = oof_tbl["oof_pred_sgd"].values
    mask = np.isfinite(p)

    plt.figure(figsize=(5.5,4.2))
    plt.hist(p[np.logical_and(mask, y==1)], bins=30, alpha=0.6, label="y=1 (A wins)")
    plt.hist(p[np.logical_and(mask, y==0)], bins=30, alpha=0.6, label="y=0 (B wins)")
    plt.xlabel("OOF predicted p(A wins)")
    plt.ylabel("Count")
    plt.title("OOF probability distribution by class (SGD baseline)")
    plt.legend()
    plt.show()

    idx_finite = np.where(mask)[0]
    if len(idx_finite) > 0:
        sample_idx = np.random.RandomState(CFG.SEED).choice(idx_finite, size=min(100, len(idx_finite)), replace=False)
        sample = oof_tbl.iloc[sample_idx].copy()
        sample["pred_class"] = (sample["oof_pred_sgd"] >= 0.5).astype(int)
        sample["correct"] = (sample["pred_class"] == sample["y"]).astype(int)
        display(sample.sort_values("oof_pred_sgd").reset_index(drop=True).head(10))
        display(sample.sort_values("oof_pred_sgd", ascending=False).reset_index(drop=True).head(10))
        display(sample.sample(min(10, len(sample)), random_state=CFG.SEED+1).reset_index(drop=True))
    else:
        print("[WARN] No finite OOF predictions to sample.")


# cell 7 - baseline introspection
warnings.filterwarnings("ignore")

PT_PATH  = os.path.join(CFG.WORK_DIR, "pairs_train.parquet")
assert os.path.exists(PT_PATH), "Run earlier cells to create pairs_train.parquet."

df = pd.read_parquet(PT_PATH).reset_index(drop=True)
P, A, B = SCHEMA["prompt"], SCHEMA["response_a"], SCHEMA["response_b"]
y = df["y"].astype(int).to_numpy()

INT_DIR = os.path.join(CFG.WORK_DIR, "artifacts_sgd_introspect_v1")
os.makedirs(INT_DIR, exist_ok=True)

# global pipeline for introspection
vec_char = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(4, 5), min_df=5, lowercase=True,
    max_features=200_000, sublinear_tf=True, dtype=np.float32
)

vec_char.fit(pd.concat([df[A], df[B]], axis=0).astype("string").tolist())

XA = vec_char.transform(df[A].astype("string"))
XB = vec_char.transform(df[B].astype("string"))
X_char = XA - XB
del XA, XB; gc.collect()

def numeric_feats(s: pd.Series) -> pd.DataFrame:
    s = s.astype("string")
    n_char = s.str.len().astype("float32")
    n_space = s.str.count(" ").astype("float32")
    n_word = (n_space + 1.0).clip(lower=1.0)
    return pd.DataFrame({
        "n_char": n_char,
        "n_word": n_word,
        "avg_wlen": (n_char / n_word).clip(upper=30.0).astype("float32"),
        "n_q": s.str.count(r"\?").astype("float32"),
        "n_e": s.str.count(r"!").astype("float32"),
        "n_nl": s.str.count(r"\n").astype("float32"),
        "has_codefence": s.str.contains(r"```").astype("int8"),
    })

numA = numeric_feats(df[A]); numB = numeric_feats(df[B])
X_num_df = (numA - numB).astype("float32")
num_cols = X_num_df.columns.tolist()

scaler_num = MaxAbsScaler().fit(X_num_df.values)
X_num = scaler_num.transform(X_num_df.values)

X = hstack([X_char, csr_matrix(X_num)], format="csr")

clf = SGDClassifier(
    loss="log_loss", penalty="l2", alpha=1e-4,
    max_iter=20, tol=1e-3, n_iter_no_change=3, random_state=CFG.SEED
)
clf.fit(X, y)

probs = clf.predict_proba(X)[:, 1].astype(np.float32)
preds = (probs >= 0.5).astype(int)

glob_metrics = {
    "acc": float(accuracy_score(y, preds)),
    "auc": float(roc_auc_score(y, probs)) if (y.min()!=y.max()) else float("nan"),
    "brier": float(brier_score_loss(y, probs)),
    "logloss": float(log_loss(y, probs, labels=[0,1]))
}

char_vocab = vec_char.vocabulary_              # token -> col_id
char_rev = np.empty(len(char_vocab), dtype=object)
for tok, cid in char_vocab.items():
    char_rev[cid] = tok

feat_names = list(char_rev) + num_cols  # order must match X columns

# coefficient ranking
coef = clf.coef_.ravel()
assert len(coef) == X.shape[1] == len(feat_names)

k_top = int(getattr(CFG, "TOP_FEATURES", 50))
order_pos = np.argsort(-coef)[:k_top]               
order_neg = np.argsort(coef)[:k_top]                
order_abs = np.argsort(-np.abs(coef))[:k_top]       

top_feats = {
    "top_positive": [{"feature": feat_names[i], "weight": float(coef[i])} for i in order_pos],
    "top_negative": [{"feature": feat_names[i], "weight": float(coef[i])} for i in order_neg],
    "top_magnitude": [{"feature": feat_names[i], "weight": float(coef[i])} for i in order_abs],
}

# persist
with open(os.path.join(INT_DIR, "top_features_sgd.json"), "w") as f:
    json.dump({"metrics_full_fit": glob_metrics, **top_feats}, f, indent=2)

# per-example contributions
bias = float(clf.intercept_.ravel()[0])

def explain_example(row_idx: int, n_show: int = 20):
    xr = X.getrow(row_idx)
    idx = xr.indices
    vals = xr.data
    contrib = vals * coef[idx]
    order = np.argsort(contrib)
    worst = [{"feature": feat_names[i], "contribution": float(contrib[j])}
             for j, i in zip(order[:n_show], idx[order[:n_show]])]
    best  = [{"feature": feat_names[i], "contribution": float(contrib[j])}
             for j, i in zip(order[::-1][:n_show], idx[order[::-1][:n_show]])]
    logit = bias + float(contrib.sum())
    prob  = float(1.0 / (1.0 + np.exp(-logit)))
    return {"row_idx": int(row_idx), "bias": bias, "logit": logit, "prob": prob,
            "top_positive_contrib": best, "top_negative_contrib": worst}

sample_size = min(25, len(df))
rng = np.random.RandomState(getattr(CFG, "SEED", 1337))
sample_rows = rng.choice(np.arange(len(df)), size=sample_size, replace=False)
explanations = [explain_example(int(r), n_show=15) for r in sample_rows]
with open(os.path.join(INT_DIR, "example_explanations.json"), "w") as f:
    json.dump({"examples": explanations}, f, indent=2)

# fast ablation-based importance
K_ABLATE = int(getattr(CFG, "ABLATE_K", 200))
DEV_N    = int(getattr(CFG, "ABLATE_DEV_N", 3000))

cand_idx = np.argsort(-np.abs(coef))[:K_ABLATE]
dev_idx  = rng.choice(np.arange(len(df)), size=min(DEV_N, len(df)), replace=False)
X_dev    = X[dev_idx].tocsr(copy=True)
y_dev    = y[dev_idx]
p_dev    = clf.predict_proba(X_dev)[:, 1].astype(np.float32)
base_auc = roc_auc_score(y_dev, p_dev) if (y_dev.min()!=y_dev.max()) else np.nan
base_ll  = log_loss(y_dev, p_dev, labels=[0,1])
base_bri = brier_score_loss(y_dev, p_dev)

abl_records = []
X_dev_csc = X_dev.tocsc(copy=True)

for j in cand_idx:
    col_start = X_dev_csc.indptr[j]
    col_end   = X_dev_csc.indptr[j+1]
    orig_data = X_dev_csc.data[col_start:col_end].copy()

    X_dev_csc.data[col_start:col_end] = 0.0

    p_drop = clf.predict_proba(X_dev_csc.tocsr())[:, 1].astype(np.float32)

    X_dev_csc.data[col_start:col_end] = orig_data

    auc_drop = roc_auc_score(y_dev, p_drop) if (y_dev.min()!=y_dev.max()) else np.nan
    ll_drop  = log_loss(y_dev, p_drop, labels=[0,1])
    bri_drop = brier_score_loss(y_dev, p_drop)

    abl_records.append({
        "feature": feat_names[j],
        "j": int(j),
        "coef": float(coef[j]),
        "delta_auc": float((auc_drop - base_auc) if (not np.isnan(base_auc) and not np.isnan(auc_drop)) else np.nan),
        "delta_logloss": float(ll_drop - base_ll),
        "delta_brier": float(bri_drop - base_bri),
    })

abl_df = pd.DataFrame(abl_records).sort_values(["delta_logloss"], ascending=False).reset_index(drop=True)
abl_path_parquet = os.path.join(INT_DIR, "ablation_importance.parquet")
abl_path_csv     = os.path.join(INT_DIR, "ablation_importance.csv")
abl_df.to_parquet(abl_path_parquet, index=False)
abl_df.to_csv(abl_path_csv, index=False)

# persist 
manifest = {
    "notes": "Global interpretation-only linear model; not used for OOF. Same featurization as baseline.",
    "vectorizer": {
        "analyzer": "char_wb", "ngram_range": [4,5], "min_df": 5,
        "max_features": 200_000, "sublinear_tf": True
    },
    "numeric_block": num_cols,
    "model": "SGDClassifier(log_loss,l2,alpha=1e-4,max_iter=20)",
    "global_fit_metrics": glob_metrics,
    "artifacts": {
        "top_features_json": os.path.basename(os.path.join(INT_DIR, "top_features_sgd.json")),
        "example_explanations_json": os.path.basename(os.path.join(INT_DIR, "example_explanations.json")),
        "ablation_importance_parquet": os.path.basename(abl_path_parquet),
        "ablation_importance_csv": os.path.basename(abl_path_csv),
    }
}
with open(os.path.join(INT_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print("[INTROSPECT] Wrote artifacts:")
for k, v in manifest["artifacts"].items():
    print(f"  - {k}: {v}")

gc.collect()



# cell 8 - sigmoid calibration (Platt)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

OOF_DIR = os.path.join(CFG.WORK_DIR, "artifacts_sgd_v1")
OOF_PATH = os.path.join(OOF_DIR, "oof_sgd.parquet")
assert os.path.exists(OOF_PATH), "Run Cell 6 first to produce OOF."

oof_tbl = pd.read_parquet(OOF_PATH)
assert {"row_idx","fold","y","oof_pred_sgd"}.issubset(oof_tbl.columns)

oof = oof_tbl["oof_pred_sgd"].to_numpy()
y   = oof_tbl["y"].astype(int).to_numpy()
mask = np.isfinite(oof)
oof = oof[mask]
y   = y[mask]

def _safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log(1.0 - p)

def expected_calibration_error(y_true, p_pred, n_bins=15):
    """ECE with equal-width bins in [0,1]."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p_pred, bins) - 1
    ece = 0.0
    per_bin = []
    n = len(p_pred)
    for b in range(n_bins):
        m = (idx == b)
        if not np.any(m):
            per_bin.append({"bin": b, "count": 0, "conf": None, "acc": None, "gap": None})
            continue
        conf = float(np.mean(p_pred[m]))
        acc  = float(np.mean(y_true[m]))
        gap  = abs(conf - acc)
        w    = np.mean(m)
        ece += w * gap
        per_bin.append({"bin": b, "count": int(m.sum()), "conf": conf, "acc": acc, "gap": gap})
    return float(ece), per_bin

def reliability_table(y_true, p_pred, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p_pred, bins) - 1
    rows = []
    for b in range(n_bins):
        m = (idx == b)
        lo, hi = bins[b], bins[b+1]
        if np.any(m):
            rows.append({
                "bin_lo": float(lo), "bin_hi": float(hi),
                "n": int(m.sum()),
                "mean_pred": float(np.mean(p_pred[m])),
                "frac_pos":  float(np.mean(y_true[m]))
            })
        else:
            rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": 0,
                         "mean_pred": None, "frac_pos": None})
    return pd.DataFrame(rows)

pre = {}
pre["brier"] = float(brier_score_loss(y, oof))
pre["logloss"] = float(log_loss(y, oof, labels=[0,1]))
pre["ece"], pre_bins = expected_calibration_error(y, oof, n_bins=15)

K = int(getattr(CFG, "CALIB_FOLDS", 5))
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=CFG.SEED)

logits = _safe_logit(oof)
calibrated = np.empty_like(oof, dtype=np.float32)

fold_params = []
for k, (tr, va) in enumerate(skf.split(logits, y), start=1):
    lr = LogisticRegression(
        penalty="none", solver="lbfgs", max_iter=1000, random_state=CFG.SEED
    )
    lr.fit(logits[tr].reshape(-1,1), y[tr])
    p_va = lr.predict_proba(logits[va].reshape(-1,1))[:,1].astype(np.float32)
    calibrated[va] = p_va
    fold_params.append({
        "fold": k,
        "coef_A": float(lr.coef_.ravel()[0]),
        "bias_B": float(lr.intercept_.ravel()[0])
    })

post = {}
post["brier"] = float(brier_score_loss(y, calibrated))
post["logloss"] = float(log_loss(y, calibrated, labels=[0,1]))
post["ece"], post_bins = expected_calibration_error(y, calibrated, n_bins=15)

CAL_DIR = os.path.join(CFG.WORK_DIR, "artifacts_calibration_v1")
os.makedirs(CAL_DIR, exist_ok=True)

cal_tbl = pd.DataFrame({
    "row_idx": oof_tbl.loc[mask, "row_idx"].to_numpy(),
    "fold":    oof_tbl.loc[mask, "fold"].to_numpy(),
    "y":       y,
    "p_raw":   oof.astype(np.float32),
    "p_cal":   calibrated.astype(np.float32)
})
cal_path = os.path.join(CAL_DIR, "oof_calibrated.parquet")
cal_tbl.to_parquet(cal_path, index=False)

rel_pre  = reliability_table(y, oof, n_bins=15)
rel_post = reliability_table(y, calibrated, n_bins=15)

plt.figure(figsize=(6.0,5.0))
# perfect calibration line
plt.plot([0,1], [0,1])
# pre
mp = rel_pre["mean_pred"].to_numpy()
fp = rel_pre["frac_pos"].to_numpy()
mask_pre = np.isfinite(mp) & np.isfinite(fp)
plt.plot(mp[mask_pre], fp[mask_pre], marker="o", label=f"Pre (ECE={pre['ece']:.3f})")
# post
mp2 = rel_post["mean_pred"].to_numpy()
fp2 = rel_post["frac_pos"].to_numpy()
mask_post = np.isfinite(mp2) & np.isfinite(fp2)
plt.plot(mp2[mask_post], fp2[mask_post], marker="o", label=f"Post (ECE={post['ece']:.3f})")
plt.xlabel("Mean predicted probability")
plt.ylabel("Empirical fraction positive")
plt.title("Reliability curve — sigmoid calibration")
plt.legend()
plt.tight_layout()
plot_path = os.path.join(CAL_DIR, "reliability_curve.png")
plt.savefig(plot_path, dpi=150)
plt.show()

# histogram
plt.figure(figsize=(6.0,4.5))
plt.hist(oof[y==1], bins=30, alpha=0.6, label="y=1 (pre)")
plt.hist(oof[y==0], bins=30, alpha=0.6, label="y=0 (pre)")
plt.xlabel("Raw OOF predicted p(A wins)")
plt.ylabel("Count")
plt.title("OOF probability distribution by class — pre calibration")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6.0,4.5))
plt.hist(calibrated[y==1], bins=30, alpha=0.6, label="y=1 (post)")
plt.hist(calibrated[y==0], bins=30, alpha=0.6, label="y=0 (post)")
plt.xlabel("Calibrated OOF predicted p(A wins)")
plt.ylabel("Count")
plt.title("OOF probability distribution by class — post calibration")
plt.legend()
plt.tight_layout()
plt.show()

# persist
summary = {
    "method": "Sigmoid (Platt) with K-fold on OOF logits",
    "k_folds": K,
    "pre": pre,
    "post": post,
    "fold_params": fold_params,
    "artifacts": {
        "oof_calibrated_parquet": os.path.basename(cal_path),
        "reliability_curve_png": os.path.basename(plot_path)
    }
}
with open(os.path.join(CAL_DIR, "calibration_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("[CALIB] Pre  — Brier={:.4f} | LogLoss={:.4f} | ECE={:.4f}".format(pre["brier"], pre["logloss"], pre["ece"]))
print("[CALIB] Post — Brier={:.4f} | LogLoss={:.4f} | ECE={:.4f}".format(post["brier"], post["logloss"], post["ece"]))
print(f"[CALIB] Saved calibrated OOF → {cal_path}")
print(f"[CALIB] Saved summary → {os.path.join(CAL_DIR,'calibration_summary.json')}")
gc.collect()


# cell 9 - baseline tweaks
from sklearn.feature_extraction.text import HashingVectorizer


warnings.filterwarnings("ignore")

PT_PATH  = os.path.join(CFG.WORK_DIR, "pairs_train.parquet")
CV_PATH  = os.path.join(CFG.WORK_DIR, "cv_folds_gkf.parquet")
assert os.path.exists(PT_PATH) and os.path.exists(CV_PATH), "Run Cells 3,5 first."

df = pd.read_parquet(PT_PATH).reset_index(drop=True)
cv_map = pd.read_parquet(CV_PATH)
P, A, B = SCHEMA["prompt"], SCHEMA["response_a"], SCHEMA["response_b"]

truth_tbl = pd.DataFrame({
    "row_idx": np.arange(len(df), dtype=int),
    "y": df["y"].astype(int).to_numpy()
})

# copy-paste
def numeric_feats(s: pd.Series) -> pd.DataFrame:
    s = s.astype("string")
    n_char = s.str.len().astype("float32")
    n_space = s.str.count(" ").astype("float32")
    n_word = (n_space + 1.0).clip(lower=1.0)
    return pd.DataFrame({
        "n_char": n_char,
        "n_word": n_word,
        "avg_wlen": (n_char / n_word).clip(upper=30.0).astype("float32"),
        "n_q": s.str.count(r"\?").astype("float32"),
        "n_e": s.str.count(r"!").astype("float32"),
        "n_nl": s.str.count(r"\n").astype("float32"),
        "has_codefence": s.str.contains(r"```").astype("int8"),
    })

def run_elasticnet_oof(vec_factory, art_dir, label_key):
    os.makedirs(art_dir, exist_ok=True)

    n_folds = int(getattr(CFG, "CV_FOLDS", 5))
    oof = np.full(len(df), np.nan, dtype=np.float32)
    fold_metrics = []

    for fold in range(1, n_folds+1):
        va_idx = cv_map.loc[cv_map["fold"] == fold, "row_idx"].to_numpy()
        tr_idx = cv_map.loc[cv_map["fold"] != fold, "row_idx"].to_numpy()

        tr = df.iloc[tr_idx].reset_index(drop=True)
        va = df.iloc[va_idx].reset_index(drop=True)

        vec_char = vec_factory()
        if hasattr(vec_char, "fit"):
            vec_char.fit(pd.concat([tr[A], tr[B]], axis=0).astype("string").tolist())

        XA_tr = vec_char.transform(tr[A].astype("string"))
        XB_tr = vec_char.transform(tr[B].astype("string"))
        XA_va = vec_char.transform(va[A].astype("string"))
        XB_va = vec_char.transform(va[B].astype("string"))
        X_char_tr = XA_tr - XB_tr
        X_char_va = XA_va - XB_va

        numA_tr = numeric_feats(tr[A]); numB_tr = numeric_feats(tr[B])
        numA_va = numeric_feats(va[A]); numB_va = numeric_feats(va[B])
        del_num_tr = (numA_tr - numB_tr).astype("float32").values
        del_num_va = (numA_va - numB_va).astype("float32").values

        scaler_num = MaxAbsScaler().fit(del_num_tr)
        X_num_tr = scaler_num.transform(del_num_tr)
        X_num_va = scaler_num.transform(del_num_va)

        X_tr = hstack([X_char_tr, csr_matrix(X_num_tr)], format="csr")
        X_va = hstack([X_char_va, csr_matrix(X_num_va)], format="csr")

        clf = SGDClassifier(
            loss="log_loss", penalty="elasticnet",
            l1_ratio=float(getattr(CFG, "EN_L1_RATIO", 0.2)),
            alpha=float(getattr(CFG, "EN_ALPHA", 1e-4)),
            max_iter=int(getattr(CFG, "EN_MAX_ITER", 30)),
            tol=1e-3, n_iter_no_change=3, random_state=CFG.SEED
        )
        clf.fit(X_tr, tr["y"].astype(int).values)
        p_va = clf.predict_proba(X_va)[:, 1].astype(np.float32)

        pred_tbl = pd.DataFrame({"row_idx": va_idx, "p": p_va})
        eval_tbl = pred_tbl.merge(truth_tbl, on="row_idx", how="inner")
        p_eval = eval_tbl["p"].to_numpy(dtype=np.float32)
        y_eval = eval_tbl["y"].to_numpy(dtype=int)
        assert len(p_eval) == len(y_eval) > 0, f"Fold {fold}: empty/mismatch."

        yhat = (p_eval >= 0.5).astype("int8")
        auc_val = np.nan
        if (y_eval.min() != y_eval.max()):
            try:
                auc_val = roc_auc_score(y_eval, p_eval)
            except Exception:
                auc_val = np.nan

        fold_metrics.append({
            "fold": fold,
            "acc": accuracy_score(y_eval, yhat),
            "auc": auc_val,
            "brier": brier_score_loss(y_eval, p_eval),
            "logloss": log_loss(y_eval, p_eval, labels=[0,1]),
            "n_valid": int(len(y_eval))
        })

        oof[eval_tbl["row_idx"].to_numpy()] = p_eval

        # cleanup!
        del XA_tr, XB_tr, XA_va, XB_va, X_char_tr, X_char_va, X_tr, X_va
        gc.collect()

    m_df = pd.DataFrame(fold_metrics)
    acc_mu, acc_sd = float(np.nanmean(m_df["acc"])), float(np.nanstd(m_df["acc"], ddof=1))
    auc_mu, auc_sd = float(np.nanmean(m_df["auc"])), float(np.nanstd(m_df["auc"], ddof=1))
    bri_mu, bri_sd = float(np.nanmean(m_df["brier"])), float(np.nanstd(m_df["brier"], ddof=1))
    ll_mu,  ll_sd  = float(np.nanmean(m_df["logloss"])), float(np.nanstd(m_df["logloss"], ddof=1))

    # persist
    fold_by_row = cv_map.set_index("row_idx").reindex(range(len(df)))["fold"].to_numpy()
    oof_tbl = pd.DataFrame({
        "row_idx": np.arange(len(df), dtype=int),
        "fold": fold_by_row,
        "y": truth_tbl["y"].values,
        label_key: oof.astype(np.float32)
    })
    oof_path = os.path.join(art_dir, f"{label_key}.parquet")
    oof_tbl.to_parquet(oof_path, index=False)

    # Persist summary
    summary = {
        "model": "SGDClassifier(log_loss, elasticnet)",
        "params": {
            "alpha": float(getattr(CFG, "EN_ALPHA", 1e-4)),
            "l1_ratio": float(getattr(CFG, "EN_L1_RATIO", 0.2)),
            "max_iter": int(getattr(CFG, "EN_MAX_ITER", 30))
        },
        "metrics_by_fold": m_df.to_dict(orient="records"),
        "metrics_mean": {"acc": acc_mu, "auc": auc_mu, "brier": bri_mu, "logloss": ll_mu}
    }
    sum_path = os.path.join(art_dir, f"{label_key}_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{label_key}] Acc={acc_mu:.4f}±{acc_sd:.4f} | AUC={auc_mu:.4f}±{auc_sd:.4f} | "
          f"Brier={bri_mu:.4f}±{bri_sd:.4f} | LogLoss={ll_mu:.4f}±{ll_sd:.4f}")
    print(f"[{label_key}] Saved OOF → {oof_path}")
    print(f"[{label_key}] Saved summary → {sum_path}")
    return m_df, oof_tbl

# A: tf-idf + elastic-net
def tfidf_factory():
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(4,5), min_df=5, lowercase=True,
        max_features=int(getattr(CFG, "EN_TFIDF_MAX_FEAT", 300_000)),
        sublinear_tf=True, dtype=np.float32
    )

ART_EN_TFIDF = os.path.join(CFG.WORK_DIR, "artifacts_en_tfidf_v1")
m_tfidf, oof_tfidf = run_elasticnet_oof(tfidf_factory, ART_EN_TFIDF, label_key="oof_pred_en_tfidf")

# B: hasingvectorizer + elastic-net
def hashing_factory():
    return HashingVectorizer(
        analyzer="char", ngram_range=(4,5),
        n_features=int(getattr(CFG, "EN_HASH_NFEATURES", 2**20)),
        alternate_sign=False, norm="l2", lowercase=True
    )

ART_EN_HASH = os.path.join(CFG.WORK_DIR, "artifacts_en_hash_v1")
m_hash, oof_hash = run_elasticnet_oof(hashing_factory, ART_EN_HASH, label_key="oof_pred_en_hash")

# summary
def summarize(m_df, name):
    return pd.DataFrame([{
        "model": name,
        "Acc_mean": np.nanmean(m_df["acc"]),
        "AUC_mean": np.nanmean(m_df["auc"]),
        "Brier_mean": np.nanmean(m_df["brier"]),
        "LogLoss_mean": np.nanmean(m_df["logloss"])
    }])

comp = pd.concat([summarize(m_tfidf, "EN + TFIDF(4-5)"),
                  summarize(m_hash,  "EN + Hash(4-5, 2^20)")], ignore_index=True)
display(comp.round(4))
gc.collect()


# cell 10 - hybrid lite
ART_SGD   = os.path.join(CFG.WORK_DIR, "artifacts_sgd_v1",       "oof_sgd.parquet")
ART_EN_TF = os.path.join(CFG.WORK_DIR, "artifacts_en_tfidf_v1",  "oof_pred_en_tfidf.parquet")
ART_EN_HS = os.path.join(CFG.WORK_DIR, "artifacts_en_hash_v1",   "oof_pred_en_hash.parquet")

assert os.path.exists(ART_SGD),   "Run Cell 6 first."
assert os.path.exists(ART_EN_TF), "Run Cell 9 (TF-IDF+EN) first."
assert os.path.exists(ART_EN_HS), "Run Cell 9 (Hash+EN) first."

sgd = pd.read_parquet(ART_SGD)[["row_idx","fold","y","oof_pred_sgd"]]
en_tf = pd.read_parquet(ART_EN_TF)[["row_idx","oof_pred_en_tfidf"]]
en_hs = pd.read_parquet(ART_EN_HS)[["row_idx","oof_pred_en_hash"]]

tbl = sgd.merge(en_tf, on="row_idx", how="left").merge(en_hs, on="row_idx", how="left")
tbl = tbl.sort_values("row_idx").reset_index(drop=True)

# copy-paste
def safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)

base_cols = ["oof_pred_sgd", "oof_pred_en_tfidf", "oof_pred_en_hash"]
X_all = np.column_stack([safe_logit(tbl[c].to_numpy()) for c in base_cols])
y_all = tbl["y"].astype(int).to_numpy()
fold_all = tbl["fold"].astype(int).to_numpy()

# no NaN allowed
mask = np.isfinite(X_all).all(axis=1)
X_all, y_all, fold_all, tbl = X_all[mask], y_all[mask], fold_all[mask], tbl.loc[mask].reset_index(drop=True)

# meta-logisitic on same oof
n_folds = int(getattr(CFG, "CV_FOLDS", 5))
skf = StratifiedKFold(n_splits=n_folds, shuffle=False) 
fold_ids = sorted(tbl["fold"].unique())
assert len(fold_ids) == n_folds, "Fold count mismatch vs CFG.CV_FOLDS."

meta_oof = np.full(len(tbl), np.nan, dtype=np.float32)
coef_list, intercept_list = [], []

for k, f in enumerate(fold_ids, start=1):
    va_mask = (tbl["fold"].to_numpy() == f)
    tr_mask = ~va_mask

    X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
    X_va, y_va = X_all[va_mask], y_all[va_mask]

    meta = LogisticRegression(
        penalty="l2", C=float(getattr(CFG, "META_C", 1.0)),
        solver="lbfgs", max_iter=1000, random_state=CFG.SEED
    )
    meta.fit(X_tr, y_tr)
    p_va = meta.predict_proba(X_va)[:,1].astype(np.float32)
    meta_oof[va_mask] = p_va

    coef_list.append(meta.coef_.ravel().tolist())
    intercept_list.append(float(meta.intercept_.ravel()[0]))

fold_metrics = []
for f in fold_ids:
    m = (tbl["fold"].to_numpy() == f)
    y_f = y_all[m]; p_f = meta_oof[m]
    yhat = (p_f >= 0.5).astype("int8")
    auc_val = np.nan
    if y_f.min() != y_f.max():
        try: auc_val = roc_auc_score(y_f, p_f)
        except Exception: auc_val = np.nan
    fold_metrics.append({
        "fold": int(f),
        "acc": accuracy_score(y_f, yhat),
        "auc": auc_val,
        "brier": brier_score_loss(y_f, p_f),
        "logloss": log_loss(y_f, p_f, labels=[0,1]),
        "n_valid": int(len(y_f))
    })

m_df = pd.DataFrame(fold_metrics).sort_values("fold")
acc_mu, acc_sd = float(np.nanmean(m_df["acc"])), float(np.nanstd(m_df["acc"], ddof=1))
auc_mu, auc_sd = float(np.nanmean(m_df["auc"])), float(np.nanstd(m_df["auc"], ddof=1))
bri_mu, bri_sd = float(np.nanmean(m_df["brier"])), float(np.nanstd(m_df["brier"], ddof=1))
ll_mu,  ll_sd  = float(np.nanmean(m_df["logloss"])), float(np.nanstd(m_df["logloss"], ddof=1))

# persist
HYB_DIR = os.path.join(CFG.WORK_DIR, "artifacts_hybrid_lite_v1")
os.makedirs(HYB_DIR, exist_ok=True)

full_pred = np.full(tbl["row_idx"].max()+1, np.nan, dtype=np.float32)
full_pred[tbl["row_idx"].to_numpy()] = meta_oof

oof_hyb = pd.DataFrame({
    "row_idx": tbl["row_idx"].to_numpy(),
    "fold":    tbl["fold"].to_numpy(),
    "y":       y_all,
    "oof_pred_hybrid_lite": meta_oof
}).sort_values("row_idx")
oof_path = os.path.join(HYB_DIR, "oof_hybrid_lite.parquet")
oof_hyb.to_parquet(oof_path, index=False)

# summary
summary = {
    "meta_model": "LogisticRegression on logits of base models",
    "base_features": base_cols,
    "coef_by_fold": coef_list,
    "intercept_by_fold": intercept_list,
    "metrics_by_fold": m_df.to_dict(orient="records"),
    "metrics_mean": {"acc": acc_mu, "auc": auc_mu, "brier": bri_mu, "logloss": ll_mu}
}
with open(os.path.join(HYB_DIR, "hybrid_lite_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"[HYB-LITE] Acc={acc_mu:.4f}±{acc_sd:.4f} | AUC={auc_mu:.4f}±{auc_sd:.4f} | "
      f"Brier={bri_mu:.4f}±{bri_sd:.4f} | LogLoss={ll_mu:.4f}±{ll_sd:.4f}")
print(f"[HYB-LITE] Saved OOF → {oof_path}")
print(f"[HYB-LITE] Saved summary → {os.path.join(HYB_DIR, 'hybrid_lite_summary.json')}")
gc.collect()


# cell 11 - calibrate hybrid-lite

HYB_DIR = os.path.join(CFG.WORK_DIR, "artifacts_hybrid_lite_v1")
OOF_PATH = os.path.join(HYB_DIR, "oof_hybrid_lite.parquet")
assert os.path.exists(OOF_PATH), "Run Cell 10 (hybrid-lite) first."

oof_tbl = pd.read_parquet(OOF_PATH)
assert {"row_idx","fold","y","oof_pred_hybrid_lite"}.issubset(oof_tbl.columns)

p_raw = oof_tbl["oof_pred_hybrid_lite"].to_numpy()
y     = oof_tbl["y"].astype(int).to_numpy()
mask  = np.isfinite(p_raw)
p_raw = p_raw[mask]; y = y[mask]

#copy-paste
def safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps)
    return np.log(p) - np.log(1-p)

def expected_calibration_error(y_true, p_pred, n_bins=15):
    bins = np.linspace(0,1,n_bins+1); idx = np.digitize(p_pred, bins)-1
    ece=0.0; per=[]
    for b in range(n_bins):
        m = (idx==b)
        if not np.any(m):
            per.append({"bin":b,"count":0,"conf":None,"acc":None,"gap":None}); continue
        conf = float(np.mean(p_pred[m])); acc = float(np.mean(y_true[m]))
        gap  = abs(conf-acc); w = np.mean(m)
        ece += w*gap
        per.append({"bin":b,"count":int(m.sum()),"conf":conf,"acc":acc,"gap":gap})
    return float(ece), per

def reliability_table(y_true, p_pred, n_bins=15):
    bins = np.linspace(0,1,n_bins+1); idx = np.digitize(p_pred, bins)-1
    rows=[]
    for b in range(n_bins):
        m=(idx==b); lo,hi=bins[b],bins[b+1]
        rows.append({
            "bin_lo":float(lo),"bin_hi":float(hi),
            "n":int(m.sum()),
            "mean_pred": float(np.mean(p_pred[m])) if np.any(m) else None,
            "frac_pos":  float(np.mean(y_true[m])) if np.any(m) else None
        })
    return pd.DataFrame(rows)

pre_brier = float(brier_score_loss(y, p_raw))
pre_ll    = float(log_loss(y, p_raw, labels=[0,1]))
pre_ece, _ = expected_calibration_error(y, p_raw, n_bins=15)

K = int(getattr(CFG, "CALIB_FOLDS", 5))
skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=CFG.SEED)
z = safe_logit(p_raw)
p_cal = np.empty_like(p_raw, dtype=np.float32)
fold_params = []

for k, (tr, va) in enumerate(skf.split(z, y), start=1):
    lr = LogisticRegression(penalty="none", solver="lbfgs", max_iter=1000, random_state=CFG.SEED)
    lr.fit(z[tr].reshape(-1,1), y[tr])
    p_cal[va] = lr.predict_proba(z[va].reshape(-1,1))[:,1].astype(np.float32)
    fold_params.append({"fold":k, "A": float(lr.coef_.ravel()[0]), "B": float(lr.intercept_.ravel()[0])})

post_brier = float(brier_score_loss(y, p_cal))
post_ll    = float(log_loss(y, p_cal, labels=[0,1]))
post_ece, _ = expected_calibration_error(y, p_cal, n_bins=15)

cal_tbl = oof_tbl.loc[mask, ["row_idx","fold","y"]].copy()
cal_tbl["p_raw"] = p_raw.astype(np.float32)
cal_tbl["p_cal"] = p_cal.astype(np.float32)

CAL_DIR = os.path.join(CFG.WORK_DIR, "artifacts_calibration_hybrid_v1")
os.makedirs(CAL_DIR, exist_ok=True)
cal_path = os.path.join(CAL_DIR, "oof_hybrid_lite_calibrated.parquet")
cal_tbl.to_parquet(cal_path, index=False)

rel_pre  = reliability_table(y, p_raw, n_bins=15)
rel_post = reliability_table(y, p_cal, n_bins=15)

plt.figure(figsize=(6,5))
plt.plot([0,1],[0,1])
mp, fp = rel_pre["mean_pred"].to_numpy(), rel_pre["frac_pos"].to_numpy()
m1 = np.isfinite(mp) & np.isfinite(fp)
plt.plot(mp[m1], fp[m1], marker="o", label=f"Pre (ECE={pre_ece:.3f})")
mp2, fp2 = rel_post["mean_pred"].to_numpy(), rel_post["frac_pos"].to_numpy()
m2 = np.isfinite(mp2) & np.isfinite(fp2)
plt.plot(mp2[m2], fp2[m2], marker="o", label=f"Post (ECE]={post_ece:.3f})")
plt.xlabel("Mean predicted probability"); plt.ylabel("Empirical fraction positive")
plt.title("Hybrid-Lite Reliability (Platt)"); plt.legend(); plt.tight_layout()
plot_path = os.path.join(CAL_DIR, "reliability_hybrid_lite.png")
plt.savefig(plot_path, dpi=150); plt.show()

# summary
summary = {
    "model": "Hybrid-Lite (meta-logistic on base logits)",
    "method": "Platt (sigmoid) with Stratified K-fold on OOF logits",
    "k_folds": K,
    "pre": {"brier": pre_brier, "logloss": pre_ll, "ece": pre_ece},
    "post": {"brier": post_brier, "logloss": post_ll, "ece": post_ece},
    "fold_params": fold_params,
    "artifacts": {"calibrated_oof_parquet": os.path.basename(cal_path),
                  "reliability_png": os.path.basename(plot_path)}
}
with open(os.path.join(CAL_DIR, "calibration_hybrid_lite_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("[HYB-LITE CALIB] Pre  — Brier={:.4f} LogLoss={:.4f} ECE={:.4f}".format(pre_brier, pre_ll, pre_ece))
print("[HYB-LITE CALIB] Post — Brier={:.4f} LogLoss={:.4f} ECE={:.4f}".format(post_brier, post_ll, post_ece))
print(f"[HYB-LITE CALIB] Saved calibrated OOF → {cal_path}")
print(f"[HYB-LITE CALIB] Saved summary → {os.path.join(CAL_DIR, 'calibration_hybrid_lite_summary.json')}")



# cell 12 - final run



WORK = CFG.WORK_DIR
PT_TRAIN = os.path.join(WORK, "pairs_train.parquet")
PT_TEST  = os.path.join(WORK, "pairs_test.parquet") 
assert os.path.exists(PT_TRAIN) and os.path.exists(PT_TEST), "Missing pairs_train.parquet or pairs_test.parquet."

OOF_SGD   = os.path.join(WORK, "artifacts_sgd_v1",       "oof_sgd.parquet")
OOF_EN_TF = os.path.join(WORK, "artifacts_en_tfidf_v1",  "oof_pred_en_tfidf.parquet")
OOF_EN_HS = os.path.join(WORK, "artifacts_en_hash_v1",   "oof_pred_en_hash.parquet")
OOF_HYB   = os.path.join(WORK, "artifacts_hybrid_lite_v1", "oof_hybrid_lite.parquet")
assert os.path.exists(OOF_SGD) and os.path.exists(OOF_EN_TF) and os.path.exists(OOF_EN_HS) and os.path.exists(OOF_HYB), \
    "Run Cells 6, 9, and 10 first to generate OOF artifacts."

df_tr = pd.read_parquet(PT_TRAIN).reset_index(drop=True)
df_te = pd.read_parquet(PT_TEST).reset_index(drop=True)
P, A, B = SCHEMA["prompt"], SCHEMA["response_a"], SCHEMA["response_b"]
y_tr = df_tr["y"].astype(int).to_numpy()

# copy-paste
def numeric_feats(s: pd.Series) -> pd.DataFrame:
    s = s.astype("string")
    n_char = s.str.len().astype("float32")
    n_space = s.str.count(" ").astype("float32")
    n_word = (n_space + 1.0).clip(lower=1.0)
    return pd.DataFrame({
        "n_char": n_char,
        "n_word": n_word,
        "avg_wlen": (n_char / n_word).clip(upper=30.0).astype("float32"),
        "n_q": s.str.count(r"\?").astype("float32"),
        "n_e": s.str.count(r"!").astype("float32"),
        "n_nl": s.str.count(r"\n").astype("float32"),
        "has_codefence": s.str.contains(r"```").astype("int8"),
    })

def delta_numeric(train_df: pd.DataFrame, test_df: pd.DataFrame):
    trA, trB = numeric_feats(train_df[A]), numeric_feats(train_df[B])
    teA, teB = numeric_feats(test_df[A]),  numeric_feats(test_df[B])
    d_tr = (trA - trB).astype("float32").values
    d_te = (teA - teB).astype("float32").values
    scaler = MaxAbsScaler().fit(d_tr)
    return scaler.transform(d_tr), scaler.transform(d_te)

def safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps)
    return np.log(p) - np.log(1-p)

def run_sgd_l2_full(train_df, test_df):
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(4,5), min_df=5, lowercase=True,
        max_features=200_000, sublinear_tf=True, dtype=np.float32
    )
    vec.fit(pd.concat([train_df[A], train_df[B]], axis=0).astype("string").tolist())
    XA_tr, XB_tr = vec.transform(train_df[A].astype("string")), vec.transform(train_df[B].astype("string"))
    XA_te, XB_te = vec.transform(test_df[A].astype("string")),  vec.transform(test_df[B].astype("string"))
    X_char_tr, X_char_te = XA_tr - XB_tr, XA_te - XB_te

    X_num_tr, X_num_te = delta_numeric(train_df, test_df)
    X_tr = hstack([X_char_tr, csr_matrix(X_num_tr)], format="csr")
    X_te = hstack([X_char_te, csr_matrix(X_num_te)], format="csr")

    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                        max_iter=30, tol=1e-3, n_iter_no_change=3, random_state=CFG.SEED)
    clf.fit(X_tr, train_df["y"].astype(int).values)
    p_te = clf.predict_proba(X_te)[:,1].astype(np.float32)
    return p_te

def run_en_tfidf_full(train_df, test_df):
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(4,5), min_df=5, lowercase=True,
        max_features=int(getattr(CFG, "EN_TFIDF_MAX_FEAT", 300_000)),
        sublinear_tf=True, dtype=np.float32
    )
    vec.fit(pd.concat([train_df[A], train_df[B]], axis=0).astype("string").tolist())
    XA_tr, XB_tr = vec.transform(train_df[A].astype("string")), vec.transform(train_df[B].astype("string"))
    XA_te, XB_te = vec.transform(test_df[A].astype("string")),  vec.transform(test_df[B].astype("string"))
    X_char_tr, X_char_te = XA_tr - XB_tr, XA_te - XB_te

    X_num_tr, X_num_te = delta_numeric(train_df, test_df)
    X_tr = hstack([X_char_tr, csr_matrix(X_num_tr)], format="csr")
    X_te = hstack([X_char_te, csr_matrix(X_num_te)], format="csr")

    clf = SGDClassifier(loss="log_loss", penalty="elasticnet",
                        l1_ratio=float(getattr(CFG, "EN_L1_RATIO", 0.2)),
                        alpha=float(getattr(CFG, "EN_ALPHA", 1e-4)),
                        max_iter=int(getattr(CFG, "EN_MAX_ITER", 30)),
                        tol=1e-3, n_iter_no_change=3, random_state=CFG.SEED)
    clf.fit(X_tr, train_df["y"].astype(int).values)
    p_te = clf.predict_proba(X_te)[:,1].astype(np.float32)
    return p_te

def run_en_hash_full(train_df, test_df):
    vec = HashingVectorizer(
        analyzer="char", ngram_range=(4,5),
        n_features=int(getattr(CFG, "EN_HASH_NFEATURES", 2**20)),
        alternate_sign=False, norm="l2", lowercase=True
    )
    # Hashing has no fit
    XA_tr, XB_tr = vec.transform(train_df[A].astype("string")), vec.transform(train_df[B].astype("string"))
    XA_te, XB_te = vec.transform(test_df[A].astype("string")),  vec.transform(test_df[B].astype("string"))
    X_char_tr, X_char_te = XA_tr - XB_tr, XA_te - XB_te

    X_num_tr, X_num_te = delta_numeric(train_df, test_df)
    X_tr = hstack([X_char_tr, csr_matrix(X_num_tr)], format="csr")
    X_te = hstack([X_char_te, csr_matrix(X_num_te)], format="csr")

    clf = SGDClassifier(loss="log_loss", penalty="elasticnet",
                        l1_ratio=float(getattr(CFG, "EN_L1_RATIO", 0.2)),
                        alpha=float(getattr(CFG, "EN_ALPHA", 1e-4)),
                        max_iter=int(getattr(CFG, "EN_MAX_ITER", 30)),
                        tol=1e-3, n_iter_no_change=3, random_state=CFG.SEED)
    clf.fit(X_tr, train_df["y"].astype(int).values)
    p_te = clf.predict_proba(X_te)[:,1].astype(np.float32)
    return p_te

# on full data
print("[FINAL] Fitting full baselines and predicting test...")
p_te_sgd   = run_sgd_l2_full(df_tr, df_te)
p_te_en_tf = run_en_tfidf_full(df_tr, df_te)
p_te_en_hs = run_en_hash_full(df_tr, df_te)

# train meta-learner on oof
def load_oof(path, col):
    t = pd.read_parquet(path)
    assert col in t.columns
    return t.sort_values("row_idx").reset_index(drop=True)[col].to_numpy()

p_oof_sgd   = load_oof(OOF_SGD,   "oof_pred_sgd")
p_oof_en_tf = load_oof(OOF_EN_TF, "oof_pred_en_tfidf")
p_oof_en_hs = load_oof(OOF_EN_HS, "oof_pred_en_hash")
y_oof       = pd.read_parquet(OOF_SGD).sort_values("row_idx")["y"].astype(int).to_numpy()

# keep only on all valid
mask = np.isfinite(p_oof_sgd) & np.isfinite(p_oof_en_tf) & np.isfinite(p_oof_en_hs)
Z_train = np.column_stack([safe_logit(p_oof_sgd[mask]),
                           safe_logit(p_oof_en_tf[mask]),
                           safe_logit(p_oof_en_hs[mask])])
y_train = y_oof[mask]

meta = LogisticRegression(penalty="l2", C=float(getattr(CFG, "META_C", 1.0)),
                          solver="lbfgs", max_iter=1000, random_state=CFG.SEED)
meta.fit(Z_train, y_train)

Z_test = np.column_stack([safe_logit(p_te_sgd),
                          safe_logit(p_te_en_tf),
                          safe_logit(p_te_en_hs)])
p_te_hybrid = meta.predict_proba(Z_test)[:,1].astype(np.float32)

# platt on full
oof_hyb = pd.read_parquet(OOF_HYB).sort_values("row_idx").reset_index(drop=True)
p_oof_hybrid = oof_hyb["oof_pred_hybrid_lite"].to_numpy()
mask_h = np.isfinite(p_oof_hybrid)
Z_h = safe_logit(p_oof_hybrid[mask_h]).reshape(-1,1)
y_h = oof_hyb.loc[mask_h, "y"].astype(int).to_numpy()

platt = LogisticRegression(penalty="none", solver="lbfgs", max_iter=1000, random_state=CFG.SEED)
platt.fit(Z_h, y_h)

# Calibrate test
z_te = safe_logit(p_te_hybrid).reshape(-1,1)
p_te_hybrid_cal = platt.predict_proba(z_te)[:,1].astype(np.float32)

# ---- 4) Persist predictions (both raw and calibrated) ----
OUT_DIR = os.path.join(WORK, "artifacts_final_v1")
os.makedirs(OUT_DIR, exist_ok=True)

pred_tbl = pd.DataFrame({
    "row_idx": np.arange(len(df_te), dtype=int),
    "p_sgd":   p_te_sgd,
    "p_en_tfidf": p_te_en_tf,
    "p_en_hash":  p_te_en_hs,
    "p_hybrid_raw": p_te_hybrid,
    "p_hybrid_cal": p_te_hybrid_cal
})
pred_path = os.path.join(OUT_DIR, "test_predictions.parquet")
pred_tbl.to_parquet(pred_path, index=False)

# kaggle submission csv
csv_path = os.path.join(OUT_DIR, "submission.csv")
pred_tbl[["row_idx","p_hybrid_cal"]].rename(columns={"p_hybrid_cal": "prediction"}).to_csv(csv_path, index=False)

# meta data
summary = {
    "meta": {
        "coef": meta.coef_.ravel().tolist(),
        "intercept": float(meta.intercept_.ravel()[0]),
        "features": ["logit_sgd", "logit_en_tfidf", "logit_en_hash"]
    },
    "platt": {
        "A": float(platt.coef_.ravel()[0]),
        "B": float(platt.intercept_.ravel()[0])
    },
    "artifacts": {
        "test_predictions_parquet": os.path.basename(pred_path),
        "submission_csv": os.path.basename(csv_path)
    }
}
with open(os.path.join(OUT_DIR, "final_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"[FINAL] Wrote test predictions → {pred_path}")
print(f"[FINAL] Wrote submission CSV → {csv_path}")
print(f"[FINAL] Meta weights: {summary['meta']}, Platt params: {summary['platt']}")
gc.collect()



# cell 13 - model card
import os, json, textwrap, pandas as pd, numpy as np

WORK = CFG.WORK_DIR

# Try to pull key numbers from earlier summaries (best-effort)
def try_load(path, default=None):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

sgd_sum   = try_load(os.path.join(WORK, "artifacts_sgd_v1", "cv_sgd_summary.json"))
en_tf_sum = try_load(os.path.join(WORK, "artifacts_en_tfidf_v1", "oof_pred_en_tfidf_summary.json"))  # our earlier cell saved *_summary.json; adapt if needed
en_hash_sum = try_load(os.path.join(WORK, "artifacts_en_hash_v1", "oof_pred_en_hash_summary.json"))
hyb_sum   = try_load(os.path.join(WORK, "artifacts_hybrid_lite_v1", "hybrid_lite_summary.json"))
cal_hyb   = try_load(os.path.join(WORK, "artifacts_calibration_hybrid_v1", "calibration_hybrid_lite_summary.json"))
final_sum = try_load(os.path.join(WORK, "artifacts_final_v1", "final_summary.json"))

def fmt_metrics(d):
    if not d: return "—"
    m = d.get("metrics_mean", d.get("post", {}))
    if not m: return "—"
    # accept either {acc, auc, brier, logloss} or {brier, logloss, ece}
    keys = [k for k in ["acc","auc","brier","logloss","ece"] if k in m]
    return ", ".join([f"{k.upper()}={m[k]:.4f}" for k in keys])

card = f"""
# Model Card — Pairwise Response Preference (Fast, Interpretable, Calibrated)

## Overview
This notebook trains **fast linear baselines** on Δ(char-TFIDF) + tiny numeric deltas, and blends them with a **meta-logistic** (“hybrid-lite”). We then apply **Platt (sigmoid) calibration** for reliable probabilities. The entire pipeline runs in ~5 minutes on Kaggle.

## Data
- Train: `pairs_train.parquet` (columns: `{P}`, `{A}`, `{B}`, `y`)
- Test:  `pairs_test.parquet`  (columns: `{P}`, `{A}`, `{B}`)

## Features
- **Δ char TF-IDF**: `analyzer="char_wb"`, `ngram_range=(4,5)`, sublinear TF; optional 300k max features.
- **Tiny numeric deltas**: `n_char, n_word, avg_wlen, n_q, n_e, n_nl, has_codefence` between A and B.

## Models
- **SGD(LogLoss, L2)** baseline.
- **Elastic-Net + TF-IDF** (L1 ratio default 0.2).
- **Elastic-Net + Hashing** (2^20 features).
- **Hybrid-Lite**: meta **logistic regression on base logits** (no leakage; trained on OOF).
- **Calibration**: Platt on **OOF hybrid logits** (then applied to test).

## Evaluation (OOF)
- SGD: {fmt_metrics(sgd_sum)}
- EN + TF-IDF: {fmt_metrics(en_tf_sum)}
- EN + Hash: {fmt_metrics(en_hash_sum)}
- Hybrid-Lite (pre-cal): {fmt_metrics(hyb_sum)}
- Hybrid-Lite (post-cal): {fmt_metrics(cal_hyb)}

## Final Inference
- Train each baseline on **full train**.
- Predict test with each baseline → meta-logistic (weights: {final_sum.get('meta', {}).get('coef', []) if final_sum else '[]'}) → **Hybrid raw**.
- Apply Platt (A={final_sum.get('platt', {}).get('A', '—')}, B={final_sum.get('platt', {}).get('B', '—')}) → **Hybrid calibrated**.
- Artifacts:
  - `artifacts_final_v1/test_predictions.parquet`
  - `artifacts_final_v1/submission.csv` (uses calibrated hybrid)

## Why this design
- **Speed & Simplicity**: pure linear models over n-grams are **fast** and surprisingly strong.
- **Interpretability**: we provide **feature attributions**, **n-gram rankings**, and **per-example contributions** (Cell 7).
- **Probability Quality**: **calibration** improves Brier/LogLoss and supports downstream decision thresholds.
- **Reproducibility**: no internet model pulls; fully deterministic config via `CFG`.

## Limitations
- Char n-grams can miss long-range semantics; unusual formats may require extra features.
- If distribution shifts (domain/topic), **recalibrate** and/or update n-gram ranges/min_df.

## Ethical & Safety
- Predictions reflect training data preferences; avoid use in sensitive settings without human review and fairness checks.

## Runtime Notes
- Typical run-time: ~10 minutes on Kaggle P100/T4.
- Memory: dominated by TF-IDF matrices (set `max_features` lower if constrained).

"""

OUT = os.path.join(WORK, "artifacts_final_v1")
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "model_card.md"), "w") as f:
    f.write(textwrap.dedent(card).strip() + "\n")

print(f"[CARD] Wrote model card → {os.path.join(OUT, 'model_card.md')}")



pd.read_csv("/kaggle/working/artifacts_final_v1/submission.csv")


sub = pd.read_csv("/kaggle/working/artifacts_final_v1/submission.csv")
sub["pred_label"] = (sub["prediction"] >= 0.5).astype(int)  # 1 = A wins, 0 = B wins
print(sub.head())



te = pd.read_parquet(f"{CFG.WORK_DIR}/pairs_test.parquet")
view = sub.merge(te.reset_index().rename(columns={"index":"row_idx"}), on="row_idx", how="left")
view[["row_idx","prediction", SCHEMA["prompt"], SCHEMA["response_a"], SCHEMA["response_b"]]].head()


