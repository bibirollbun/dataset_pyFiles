# === 1. Imports ===
import pandas as pd
import numpy as np
import os
import glob
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import jaccard_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
print("Libraries imported.")


# === 2. Load Data & Create Labels ===

# --- 1. Define File Paths ---
DATA_PATH = '/kaggle/input/alpha-radar-solana-sprint'
TARGET_TOKEN_PATH = '/kaggle/input/alpha-rader-target-tokens/Alpha Radar Target Tokens.csv'

# --- 2. Load Target Tokens ---
print(f"Attempting to load target tokens from: {TARGET_TOKEN_PATH}")
target_tokens_df = pd.read_csv(TARGET_TOKEN_PATH)
target_token_set = set(target_tokens_df['Target Token Addresses'])
print(f"Loaded {len(target_token_set)} target (positive) tokens successfully.")

# --- 3. Load Training Transactions ---
print("Loading training transaction data (Sample_Dataset.csv)...")
train_tx_df = pd.read_csv(os.path.join(DATA_PATH, 'Sample_Dataset.csv'))
print(f"Loaded {len(train_tx_df)} training transactions.")

# --- 4. Load & Concatenate Evaluation Transactions ---
print("Loading evaluation transaction data (chunks)...")
eval_files = sorted(glob.glob(os.path.join(DATA_PATH, 'evaluation_set_30s_chunk_*.csv')))
eval_dfs = []
for f in tqdm(eval_files, desc="Loading eval chunks"):
    eval_dfs.append(pd.read_csv(f))
eval_tx_df = pd.concat(eval_dfs, ignore_index=True)
print(f"Loaded {len(eval_tx_df)} evaluation transactions from {len(eval_files)} files.")





FULL_DIR ="/kaggle/input/pumpfun-30s-september-2025"

train_files = sorted(glob.glob(os.path.join(FULL_DIR, 'september_2025_first30s_chunk_*.csv')))
train_dfs = []
for f in tqdm(train_files, desc="Loading eval chunks"):
    train_dfs.append(pd.read_csv(f))
train_tx_df = pd.concat(train_dfs, ignore_index=True)
print(f"Loaded {len(train_tx_df)} evaluation transactions from {len(train_files)} files.")




# --- Normalize "mm:ss(.fff)" → "hh:mm:ss(.fff)" safely ---
s = train_tx_df['timestamp'].astype(str).str.strip()

# unify decimal separator and drop stray trailing dots (e.g., "12:03.")
s = s.str.replace(',', '.', regex=False).str.replace(r'\.$', '', regex=True)

# prepend "00:" when there is only one colon → "mm:ss" becomes "00:mm:ss"
one_colon = s.str.count(':') == 1
s = np.where(one_colon, '00:' + s, s)

# parse to Timedelta
ts = pd.to_timedelta(s)

# --- Compute per-token gaps and keep rows with gap ≤ 30s (preserve original order) ---
train_tx_df = train_tx_df.assign(_ts=ts).sort_values(['mint_token_id', '_ts'])
gap = train_tx_df.groupby('mint_token_id')['_ts'].diff()

keep_sorted = gap.isna() | (gap <= pd.Timedelta(seconds=30))
keep_idx = train_tx_df.index[keep_sorted]

# back to original order
train_tx_df = train_tx_df.loc[keep_idx].sort_index(kind='mergesort').reset_index(drop=True)


# --- 5. Get all unique tokens from the training transaction data ---
unique_train_tokens = train_tx_df['mint_token_id'].unique()
print(f"Found {len(unique_train_tokens)} unique tokens in the training transaction set.")




# --- 6. Create the y_train (ground truth) DataFrame ---
y_train_list = [1 if token in target_token_set else 0 for token in unique_train_tokens]
y_train_df = pd.DataFrame({ # Save as DataFrame for merging
    'mint_token_id': unique_train_tokens,
    'is_target': y_train_list
})
y_train = y_train_df.set_index('mint_token_id')['is_target'] # For model training
print(f"Created y_train with {len(y_train)} entries.")

# --- 7. Check Class Imbalance & Get Ratio ---
print("\n--- Training Set Class Imbalance ---")
print(y_train.value_counts(normalize=True))
imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nCalculated imbalance ratio (scale_pos_weight): {imbalance_ratio:.2f}")


import polars as pl

KEY = "mint_token_id"
EPS = 1e-9


def engineer_features_single_pl(df: pl.DataFrame) -> pl.DataFrame:
    out = df.with_columns(
        pl.col(["mint_token_id", "creator", "holder"]).cast(
            pl.Categorical
        )
    )

    out = out.with_columns(
        (
            pl.col("virtual_sol_reserves") / (pl.col("virtual_token_reserves") + EPS)
        ).alias("price_spt")
    )


    out = out.with_columns(
        (pl.col("token_delta") / (pl.col("virtual_token_reserves") + EPS)).alias(
            "impact_token"
        )
    )
    out = out.with_columns(
        (pl.col("sol_delta") / (pl.col("virtual_sol_reserves") + EPS)).alias(
            "impact_sol"
        )
    )

    out = out.with_columns(
        (
            (pl.col("buy_count") - pl.col("sell_count")) / (pl.col("total_count") + EPS)
        ).alias("count_imbalance"),
        (pl.col("buy_count") / (pl.col("total_count") + EPS)).alias("buy_share"),
        ((pl.col("buy_count") + 1) / (pl.col("total_count") + 2)).alias(
            "buy_share_smoothed"
        ),
    )

    out = out.with_columns(
        (pl.col("token_volume") / (pl.col("current_holders") + EPS)).alias(
            "turnover_per_holder"
        )
    )

    out = out.with_columns(
        (pl.col("consumed_gas") / (pl.col("total_count") + EPS)).alias("gas_per_trade")
    )
    out = out.with_columns(
        (pl.col("fee") / (pl.col("total_count") + EPS)).alias("fee_per_trade")
    )
    out = out.with_columns(
        (pl.col("fee") / (pl.col("sol_volume") + EPS)).alias("fee_per_sol_volume")
    )

    # ---------------- Streaks & time-since events ----------------
    # 1) Identify buy-dominance (True where buy_count > sell_count)
    # 1) flag dominance
    out = out.with_columns(
        ((pl.col("buy_count") - pl.col("sell_count")) > 0).alias("_is_buy_dom")
    )

    # EWMs via map_groups (tiny spans; robust for short histories)    # ---------------- Holder dynamics ----------------

    out = out.with_columns(
        (pl.col("total_holders") - pl.col("current_holders")).alias("holder_churn"),
        (pl.col("current_holders") / (pl.col("total_holders") + EPS)).alias(
            "holder_utilization"
        ),
    )
    out = out.with_columns(
        (pl.col("top10_percent_total") / (pl.col("current_holders") + EPS)).alias(
            "top10_share_of_holders"
        )
    )

    out = out.with_columns(
        (pl.col("bollinger_relative_position") > 1.0)
        .cast(pl.Int8)
        .alias("bb_break_high"),
        (pl.col("bollinger_relative_position") < 0.0)
        .cast(pl.Int8)
        .alias("bb_break_low"),
    )

    return out


train_tx_df["same_holder_creator"] = (
    train_tx_df["holder"] == train_tx_df["creator"]
).astype(int)
eval_tx_df["same_holder_creator"] = (
    eval_tx_df["holder"] == eval_tx_df["creator"]
).astype(int)

m = {"buy": 0, "sell": 1}
train_tx_df["trade_mode"] = train_tx_df["trade_mode"].replace(m)
eval_tx_df["trade_mode"]   = eval_tx_df["trade_mode"].replace(m)


# ✅ Let pandas parse automatically
# Parse timestamps safely (invalid ones become NaT)
eval_tx_df["timestamp"] = pd.to_datetime(
    eval_tx_df["timestamp"], utc=True, errors="coerce"
)

# Convert to seconds since midnight (NaT → NaN)
eval_tx_df["timestamp"] = (
    +eval_tx_df["timestamp"].dt.minute * 60
    + eval_tx_df["timestamp"].dt.second
    + eval_tx_df["timestamp"].dt.microsecond / 1e6
)

# Safely format only valid rows
eval_tx_df["timestamp"] = eval_tx_df["timestamp"].apply(
    lambda x: f"{int(x//60):02d}:{x%60:04.1f}" if pd.notna(x) else None
)


# df is your original DataFrame
train_tx_df = engineer_features_single_pl(pl.from_pandas(train_tx_df)).to_pandas()
eval_tx_df = engineer_features_single_pl(pl.from_pandas(eval_tx_df)).to_pandas()


cols = train_tx_df.select_dtypes(include='bool').columns
train_tx_df[cols] = train_tx_df[cols].astype('int8')      # compact ints
eval_tx_df[cols] = eval_tx_df[cols].astype('int8')      # compact ints



def create_features(df, numeric_cols=None, categorical_cols=None):
    """
    V1: Simple, robust baseline feature set.
    """
    import numpy as np
    import pandas as pd

    print("Starting V1 feature aggregation...")

    numeric_cols = set(numeric_cols or [])
    categorical_cols = set(categorical_cols or [])

    # --- Lists of desired metrics ---
    # last_state_cols = [
    #     'market_cap_usd','buy_count','sell_count','total_count','total_holders',
    #     'current_holders','top10_percent_total','creator_balance','creator_sold',
    #     'buy_sell_ratio','relative_strength_index','bollinger_relative_position',
    #     'money_flow_index'
    # ]
    # sum_cols = [
    #     'sol_volume','token_volume','consumed_gas','fee','sol_delta',
    #     'top10_share_of_holders','bb_break_low','bb_break_high','_is_buy_dom',
    #     'buy_share_smoothed', "trade_mode"
    # ]
    # mean_cols = [
    #     'volume_oscillator','rate_of_change','liquidity_ratio',
    #     'fee_per_trade','fee_per_sol_volume','impact_token','impact_sol',
    #     'gas_per_trade','turnover_per_holder','price_spt','holder_utilization',
    #     'holder_churn'
    # ]

    last_state_cols = ['market_cap_usd', 'buy_count', 'sell_count', 'total_count', 'total_holders', 'current_holders', 'top10_percent_total', 'creator_balance', 'creator_sold', 'buy_sell_ratio', 
                       'relative_strength_index', 'bollinger_relative_position', 'money_flow_index', "gas_per_trade", "fee_per_trade", "turnover_per_holder"]
    first_state_cols = ['market_cap_usd', 'buy_count', 'sell_count', 'total_count', 'total_holders', 'current_holders', 'top10_percent_total', 'creator_balance', 'creator_sold', 'buy_sell_ratio', 
                       'relative_strength_index', 'bollinger_relative_position', 'money_flow_index']
    sum_cols = []
    mean_cols = ['volume_oscillator', 'rate_of_change', 'liquidity_ratio', "same_holder_creator", "trade_mode",'sol_volume', 'token_volume', 'consumed_gas', 'fee', 'sol_delta', 'current_holders', 'creator_balance',
                'buy_count', 'sell_count', 'total_count', "gas_per_trade", "fee_per_trade", "turnover_per_holder"]
    std_cols = ['volume_oscillator', 'rate_of_change', 'liquidity_ratio','sol_volume', 'token_volume', 'consumed_gas', 'fee', 'sol_delta']

    
    # --- Build the aggregation spec based on columns that ACTUALLY exist ---
    aggregations = {}

    def add_agg(col, agg_func):
        if col not in aggregations:
            aggregations[col] = []
        if agg_func not in aggregations[col]:
            aggregations[col].append(agg_func)

    # Coerce any column we plan to treat as numeric (if present) to numeric
    numeric_targets = set(last_state_cols + sum_cols + mean_cols + ['creator_fee', 'market_cap_usd'])
    for col in (numeric_targets & set(df.columns)):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Add aggregations for columns present in df (not in the provided lists)
    for col in last_state_cols:
        if col in df.columns: add_agg(col, 'last')
            
    # for col in first_state_cols:
    #     if col in df.columns: add_agg(col, 'first')

    
    for col in sum_cols:
        if col in df.columns: add_agg(col, 'sum')

    for col in mean_cols:
        if col in df.columns: add_agg(col, 'mean')

    # for col in std_cols:
    #     if col in df.columns: add_agg(col, 'std')

    if 'holder' in df.columns: add_agg('holder', 'nunique')
    if 'creator' in df.columns: add_agg('creator', 'first')
    if 'creator_fee' in df.columns: add_agg('creator_fee', 'first')
    if 'creator' in df.columns: add_agg('creator', 'last')
    if 'creator_fee' in df.columns: add_agg('creator_fee', 'last')

    
    if 'market_cap_usd' in df.columns:
        add_agg('market_cap_usd', 'std')
        # change over window
        add_agg('market_cap_usd', lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 0 else 0)

    if not aggregations:
        raise ValueError("No matching columns found in df for the requested aggregations.")

    # --- Sort so 'last' truly means last by timestamp within each token ---
    try:
        df_sorted = df.sort_values(by=['mint_token_id', 'timestamp'])
    except KeyError:
        print("Warning: 'timestamp' column not found for sorting. Proceeding without sorting.")
        df_sorted = df

    feature_df = df_sorted.groupby('mint_token_id', sort=False).agg(aggregations)

    # Flatten MultiIndex columns
    new_cols = []
    for col in feature_df.columns:
        if isinstance(col, tuple):
            base, aggname = col
            new_cols.append(f"{base}_change" if '<lambda>' in str(aggname) else f"{base}_{aggname}")
        else:
            new_cols.append(col)
    feature_df.columns = new_cols

    # Renames
    if 'holder_nunique' in feature_df.columns:
        feature_df = feature_df.rename(columns={'holder_nunique': 'unique_holders'})
    # if 'creator_first' in feature_df.columns:
    #     feature_df = feature_df.rename(columns={'creator_first': 'creator'})

    # feature_df = feature_df.rename(columns={'creator_first': 'creator'})

    
    
    # --- Fill NaNs ---
    if 'market_cap_usd_std' in feature_df.columns:
        feature_df['market_cap_usd_std'] = feature_df['market_cap_usd_std'].fillna(0)
    if 'buy_sell_ratio_last' in feature_df.columns:
        feature_df['buy_sell_ratio_last'] = feature_df['buy_sell_ratio_last'].fillna(1000).replace(np.inf, 1000)

    # Numeric fills
    num_cols_out = feature_df.select_dtypes(include=['number']).columns
    feature_df[num_cols_out] = feature_df[num_cols_out].fillna(0)

    # Categorical fills (keep dtype if any survived)
    for c in feature_df.select_dtypes(include=['category']).columns:
        feature_df[c] = feature_df[c].cat.add_categories(['missing']).fillna('missing')

    # Optional: ensure expected columns exist even if missing in the source df
    # (comment out if you don't want placeholder zeros)
    expected_suffixes = (
        [f"{c}_last" for c in last_state_cols if c in df.columns] +
        [f"{c}_sum" for c in sum_cols if c in df.columns] +
        [f"{c}_mean" for c in mean_cols if c in df.columns] +
        (['market_cap_usd_std'] if 'market_cap_usd' in df.columns else []) +
        (['market_cap_usd_change'] if 'market_cap_usd' in df.columns else []) +
        (['unique_holders'] if 'holder' in df.columns else []) +
        (['creator'] if 'creator' in df.columns else []) +
        (['creator_fee_first'] if 'creator_fee' in df.columns else [])
    )
    for col in expected_suffixes:
        if col not in feature_df.columns:
            feature_df[col] = 0

    print("V1 Feature aggregation complete.")
    return feature_df



# === 4. Run Feature Engineering (V1) ===

# --- 1. Identify Shared Columns ---
train_cols = set(train_tx_df.columns)
eval_cols = set(eval_tx_df.columns)
shared_cols = list(train_cols.intersection(eval_tx_df.columns))
print(f"Column(s) missing from evaluation set: {train_cols - eval_cols}")

all_numeric_cols = ['token_quantity', 'creator_fee', 'creator_fee_pump', 'market_cap_usd', 'token_delta', 'sol_delta', 'buy_count', 'sell_count', 'total_count', 'token_volume', 'sol_volume', 'liquidity_ratio', 'virtual_sol_reserves', 'virtual_token_reserves', 'consumed_gas', 'fee', 'relative_strength_index', 'bollinger_relative_position', 'volume_oscillator', 'rate_of_change', 'money_flow_index', 'total_holders', 'current_holders', 'top10_percent_total', 'creator_balance', 'creator_sold', 'holder_ratio', 'buy_sell_ratio']
all_categorical_cols = ['holder', 'trade_mode', 'creator']
numeric_cols = [col for col in all_numeric_cols if col in shared_cols]
categorical_cols = [col for col in all_categorical_cols if col in shared_cols]

# --- 2. Create Training Features (X_train) ---
print("Processing Training Data (V1)...")
X_train = create_features(train_tx_df,
                          numeric_cols=numeric_cols,
                          categorical_cols=categorical_cols)

# --- 3. Create Test Features (X_test) ---
print("\nProcessing Evaluation Data (V1)...")
X_test = create_features(eval_tx_df,
                         numeric_cols=numeric_cols,
                         categorical_cols=categorical_cols)

# X_train = X_train.drop(columns=X_train.select_dtypes(include=['object', 'category', 'string']).columns)

# --- 4. Align Training Labels (y_train) ---
print("\nAligning data...") 
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
y_train = y_train.loc[X_train.index] # Get the aligned series

# --- 5. Define Categorical Features for CatBoost ---
cat_features = ["creator_first", "creator_last"]
# --- 6. Final Verification ---
print("\n--- Final Shapes ---")
print(f"X_train shape:      {X_train.shape}")
print(f"y_train_aligned shape: {y_train.shape}")
print(f"X_test shape:       {X_test.shape}")
print(f"Categorical features: {cat_features}")

required_rows = 64208
if X_test.shape[0] == required_rows:
    print(f"\nSUCCESS: X_test has exactly {required_rows} rows.")
else:
    print(f"\nWARNING: X_test has {X_test.shape[0]} rows.")


del train_tx_df 
import gc
gc.collect()


# === CatBoost CV + Multiple Ensemble Options (GPU) ===
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import jaccard_score
import numpy as np

class JaccardMetric:
    def is_max_optimal(self):
        return True  # higher Jaccard is better

    def evaluate(self, approxes, targets, weights):
        # approxes = raw prediction margins; convert to probabilities
        prob = 1 / (1 + np.exp(-np.array(approxes[0])))
        y_true = np.array(targets)

        # Jaccard needs binary predictions, so choose threshold (typical 0.5)
        y_pred = (prob >= 0.5).astype(int)

        score = jaccard_score(y_true, y_pred)

        # CatBoost expects: (metric_value, weight_sum)
        # weight_sum is usually sum(weights) or number of samples
        weight_sum = sum(weights) if weights is not None else len(y_true)
        return score, weight_sum

    def get_final_error(self, error, weight):
        # final metric value
        return error

jaccard_metric = JaccardMetric()


# -------------------- Config --------------------
N_SPLITS = 5
SEEDS = [42, 1337, 2025, 777, 31415, 423411, 13, 2025]  # 3–5 seeds is usually enough

# Choose one: 'prob_mean', 'logit_mean', 'weighted_prob', 'weighted_logit', 'stacking_logreg'
ENSEMBLE_MODE = "prob_mean" # or weighted_prob?

# For weighting modes
WEIGHT_METRIC = "oof_ap"  # 'oof_ap' (PR-AUC) or 'oof_auc'
GAMMA = 1.25  # >1 emphasizes better seeds, 1.0=plain normalize

# For stacking mode
STACK_FEATURE_SPACE = "logit"  # 'logit' or 'prob'
STACK_LOGREG_KW = dict(
    solver="lbfgs",
    max_iter=1000,
    n_jobs=None,
    class_weight=None,  # consider 'balanced' if classes are very imbalanced
)

SCALE_POS_WEIGHT = (
    imbalance_ratio  # From earlier cell, optional if you want to re-enable
)


# -------------------- Utilities --------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def f1_opt_threshold(y_true: np.ndarray, proba: np.ndarray):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1_vals = (2 * prec[:-1] * rec[:-1]) / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    best_idx = np.nanargmax(f1_vals)
    return float(thr[best_idx]), float(f1_vals[best_idx])


def metrics_block(y_true: np.ndarray, proba: np.ndarray, prefix=""):
    thr, f1_at_thr = f1_opt_threshold(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    ap = average_precision_score(y_true, proba)
    print(
        f"{prefix}F1={f1_at_thr:.4f} (thr={thr:.3f}) | AUC={auc:.4f} | PR-AUC={ap:.4f}"
    )
    return dict(best_thr=thr, f1=f1_at_thr, auc=auc, ap=ap)


# -------------------- CatBoost params --------------------
base_params = {
    "iterations": 3000,
    "learning_rate": 0.01,
    "depth": 5,
    "l2_leaf_reg": 10.0,
    "random_strength": 1.5,
    "min_data_in_leaf": 20,
    "subsample": 0.9,
    "bootstrap_type": "Bernoulli",
    "one_hot_max_size": 8,
    "loss_function": "Logloss",
    "eval_metric": "F1",
    "custom_metric": ["F1", "AUC"],
    # "scale_pos_weight": SCALE_POS_WEIGHT,  # optional to re-enable
    "early_stopping_rounds": 250,
    "use_best_model": True,
    "verbose": 250,
    "task_type": "GPU",
    "devices": "0:1",
    "auto_class_weights": "SqrtBalanced",
}

# -------------------- Training (per-seed CV) --------------------
print(f"--- CatBoost CV • {len(SEEDS)} seed(s) • Ensemble='{ENSEMBLE_MODE}' ---")

all_oof_margins = []  # shape -> [n_train, n_seeds]
all_test_margins = []  # shape -> [n_test,  n_seeds]
seed_summaries = []  # per-seed oof metrics (for weighting)

for s_i, seed in enumerate(SEEDS, 1):
    print(f"\n===== Seed {s_i}/{len(SEEDS)} — value={seed} =====")
    params = dict(base_params, random_seed=seed)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    oof_margin = np.zeros(len(X_train), dtype=float)
    test_margin = np.zeros(len(X_test), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\n--- Seed {seed} • Fold {fold}/{N_SPLITS} ---")
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        test_pool = Pool(X_test, cat_features=cat_features)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool)

        # Raw margins (log-odds)
        val_m = model.predict(val_pool, prediction_type="RawFormulaVal")
        test_m = model.predict(test_pool, prediction_type="RawFormulaVal")

        # store
        oof_margin[va_idx] = val_m
        test_margin += test_m / N_SPLITS

        # fold metrics
        val_p = sigmoid(val_m)
        _ = metrics_block(y_val, val_p, prefix=f"Seed {seed} • Fold {fold}: ")
        gc.collect()

    # seed-level OOF metrics
    oof_p = sigmoid(oof_margin)
    m = metrics_block(y_train, oof_p, prefix=f"Seed {seed} OOF: ")

    all_oof_margins.append(oof_margin)
    all_test_margins.append(test_margin)
    seed_summaries.append(
        {
            "seed": seed,
            "oof_f1": float(m["f1"]),
            "oof_auc": float(m["auc"]),
            "oof_ap": float(m["ap"]),
            "best_oof_thr": float(m["best_thr"]),
        }
    )

# -------------------- Stack / Blend --------------------
print("\n--- Aggregating across seeds ---")
all_oof_margins = np.stack(all_oof_margins, axis=1)  # [n_train, n_seeds]
all_test_margins = np.stack(all_test_margins, axis=1)  # [n_test,  n_seeds]

# convenience conversions
oof_probs_by_seed = sigmoid(all_oof_margins)  # [n_train, n_seeds]
test_probs_by_seed = sigmoid(all_test_margins)  # [n_test,  n_seeds]


def compute_weights(metric_name=WEIGHT_METRIC, gamma=GAMMA):
    w_raw = np.array([s[metric_name] for s in seed_summaries], dtype=float)
    finite = np.isfinite(w_raw)
    if not finite.all():
        safe_min = np.min(w_raw[finite])
        w_raw = np.where(finite, w_raw, safe_min)
    w_raw = np.clip(w_raw, 1e-12, None)
    w = w_raw**gamma
    w = w / w.sum()
    print(f"Seed weights ({metric_name}, gamma={gamma}):", np.round(w, 6))
    return w


if ENSEMBLE_MODE == "prob_mean":
    oof_blend_p = oof_probs_by_seed.mean(axis=1)
    test_blend_p = test_probs_by_seed.mean(axis=1)

elif ENSEMBLE_MODE == "logit_mean":
    oof_margin_blend = all_oof_margins.mean(axis=1)
    test_margin_blend = all_test_margins.mean(axis=1)
    oof_blend_p = sigmoid(oof_margin_blend)
    test_blend_p = sigmoid(test_margin_blend)

elif ENSEMBLE_MODE == "weighted_prob":
    w = compute_weights()
    oof_blend_p = (oof_probs_by_seed * w).sum(axis=1)
    test_blend_p = (test_probs_by_seed * w).sum(axis=1)

elif ENSEMBLE_MODE == "weighted_logit":
    w = compute_weights()
    oof_margin_blend = (all_oof_margins * w).sum(axis=1)
    test_margin_blend = (all_test_margins * w).sum(axis=1)
    oof_blend_p = sigmoid(oof_margin_blend)
    test_blend_p = sigmoid(test_margin_blend)

elif ENSEMBLE_MODE == "stacking_logreg":
    # features = per-seed outputs (OOF & TEST), in chosen space
    if STACK_FEATURE_SPACE == "logit":
        X_meta = all_oof_margins
        X_testm = all_test_margins
    elif STACK_FEATURE_SPACE == "prob":
        X_meta = oof_probs_by_seed
        X_testm = test_probs_by_seed
    else:
        raise ValueError("STACK_FEATURE_SPACE must be 'logit' or 'prob'")

    print(
        f"Training meta-learner (LogisticRegression) on {STACK_FEATURE_SPACE} features..."
    )
    meta = LogisticRegression(**STACK_LOGREG_KW)
    meta.fit(X_meta, y_train)

    oof_blend_p = meta.predict_proba(X_meta)[:, 1]
    test_blend_p = meta.predict_proba(X_testm)[:, 1]
else:
    raise ValueError("ENSEMBLE_MODE not recognized.")

# -------------------- Global threshold & final metrics --------------------
print("\n--- Blended OOF metrics ---")
stats = metrics_block(y_train, oof_blend_p, prefix="BLENDED OOF: ")
best_thr_blend = stats["best_thr"]

test_pred_proba = test_blend_p.copy()
test_pred_label = (test_pred_proba >= best_thr_blend).astype(int)

print("\nGenerated:")
print(" • Arrays: oof_blend_p, test_pred_proba, test_pred_label")
print(" • Dicts: seed_summaries, stats")
print(f" • Ensemble: {ENSEMBLE_MODE} (weights used if applicable).")


# === 6. Find Optimal Threshold (Exact PR-curve, Jaccard w/ Recall Floor) — ENSEMBLE-AWARE ===
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    recall_score,
    jaccard_score,
    f1_score,
    precision_score,
)

RECALL_FLOOR = 0.75  # competition requirement
SAFETY_MARGIN = 0.00  # e.g., 0.02 if you want recall ≥ 0.77
FLOOR = RECALL_FLOOR + SAFETY_MARGIN

# >>> CHANGE HERE: use the ensemble OOF probabilities <<<
# If you're not ensembling, set SCORE_OOF = oof_preds
SCORE_OOF = oof_blend_p  # from the ensemble block
# Optional sanity check
assert SCORE_OOF.shape[0] == len(y_train), "OOF length mismatch."

# 1) PR curve breakpoints from OOF predictions
prec, rec, thr = precision_recall_curve(y_train, SCORE_OOF)
# thresholds align with prec[:-1], rec[:-1]
prec_, rec_, thr_ = prec[:-1], rec[:-1], thr

# 2) Compute F1 and Jaccard at each breakpoint
eps = 1e-12
f1_vals = (2 * prec_ * rec_) / np.clip(prec_ + rec_, eps, None)
jacc_vals = f1_vals / np.clip(2 - f1_vals, eps, None)  # J = F1 / (2 - F1)

# 3) Feasible indices: meet the recall floor
feasible = rec_ >= FLOOR

best_threshold = None
best_jaccard = None
best_recall = None

if np.any(feasible):
    # Maximize Jaccard among feasible points
    i_rel = np.nanargmax(jacc_vals[feasible])
    i = np.where(feasible)[0][i_rel]
    best_threshold = float(thr_[i])
    best_jaccard = float(jacc_vals[i])
    best_recall = float(rec_[i])
else:
    # No breakpoint meets the floor
    if rec[0] >= FLOOR:
        # threshold just below the smallest score -> predict "all positives"
        best_threshold = np.nextafter(SCORE_OOF.min(), -np.inf)
    else:
        # fallback to highest-recall breakpoint
        idx_max_rec = int(np.argmax(rec_))
        best_threshold = float(thr_[idx_max_rec])
        print(
            f"WARNING: No threshold meets recall ≥ {FLOOR:.2f} on OOF; "
            f"falling back to highest-recall breakpoint (rec={rec_[idx_max_rec]:.4f})."
        )

# 4) Final OOF metrics & confusion matrix at the chosen threshold
final_oof_preds = (SCORE_OOF >= best_threshold).astype(int)
oof_recall = recall_score(y_train, final_oof_preds, zero_division=0)
oof_precision = precision_score(y_train, final_oof_preds, zero_division=0)
oof_f1 = f1_score(y_train, final_oof_preds, zero_division=0)
oof_jaccard = jaccard_score(y_train, final_oof_preds, zero_division=0)

print("\n--- Optimal Threshold Search (Exact PR-curve) ---")
print(f"Recall floor required: {RECALL_FLOOR:.2f} | safety margin: {SAFETY_MARGIN:.2f}")
print(f"Chosen threshold: {best_threshold:.6f}")
print(
    f"OOF metrics @ thr: Jaccard={oof_jaccard:.4f} | Recall={oof_recall:.4f} | "
    f"Precision={oof_precision:.4f} | F1={oof_f1:.4f}"
)

# 5) Confusion Matrix (report Jaccard alongside)
cm = confusion_matrix(y_train, final_oof_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted 0", "Predicted 1"],
    yticklabels=["Actual 0", "Actual 1"],
)
plt.title(
    f"OOF Confusion Matrix (thr={best_threshold:.4f})\n"
    f"Jaccard={oof_jaccard:.4f} | Recall={oof_recall:.4f} | Precision={oof_precision:.4f} | F1={oof_f1:.4f}"
)
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.show()

# Keep this handy for inference on test:
optimal_threshold = best_threshold

# >>> APPLY TO TEST <<<
# Use the ensemble test probabilities you produced earlier
# (from the ensemble block: test_pred_proba)
test_pred_label = (test_pred_proba >= optimal_threshold).astype(int)


# === 7. Create Submission Files ===

# --- 1. Apply the optimal threshold ---
final_binary_preds = (test_pred_proba > (best_threshold)).astype(int)

# --- 2. Create the main submission.csv file (with correct order) ---
correct_token_order = pd.Series(eval_tx_df['mint_token_id']).unique()
submission_df_ordered = pd.DataFrame({'mint_token_id': correct_token_order})

predictions_map = pd.Series(test_pred_proba, index=X_test.index)
binary_predictions_map = pd.Series(final_binary_preds, index=X_test.index)

submission_df_ordered['is_target'] = submission_df_ordered['mint_token_id'].map(binary_predictions_map)
submission_df_ordered['prediction_value'] = submission_df_ordered['mint_token_id'].map(predictions_map)

# --- 3. Create the deliverable_details.csv file ---
deliverable_df = pd.DataFrame({
    'token': submission_df_ordered['mint_token_id'],
    'threshold': best_threshold,
    'prediction_value': submission_df_ordered['prediction_value'],
    'isTargetToken': submission_df_ordered['is_target']
})

# --- 4. Save the files ---
final_submission_csv = submission_df_ordered[['mint_token_id', 'is_target']]
final_submission_csv.to_csv('submission.csv', index=False)
deliverable_df.to_csv('deliverable_details.csv', index=False)

print("\n--- Submission Files Created ---")
print(f"submission.csv shape: {final_submission_csv.shape}")
print(f"deliverable_details.csv shape: {deliverable_df.shape}")
print("\nAll done!")

