import os, gc, random, re, sys, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import glob
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.linear_model import LinearRegression # <-- No longer needed
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import jaccard_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
print("Libraries imported.")

# --- Setup Paths and IDs ---
SEED = 42
random.seed(SEED); np.random.seed(SEED)
USE_GPU_WISH = True
CHUNKSIZE = 250_000

DATA_PATH = '/kaggle/input/alpha-radar-solana-sprint/'
WORK_DIR  = Path("/kaggle/working"); WORK_DIR.mkdir(parents=True, exist_ok=True)
KEY_COL  = "mint_token_id"
TIME_COL = "timestamp"

TARGET_TOKENS_GDRIVE_ID = "1EsqpZXPBU-6m0djDmccCrtUX07jV2fHA"
TARGET_TOKEN_PATH = WORK_DIR / "target_tokens.csv" # <-- Use our GDrive path

# --- CatBoost/gdown Installation ---
try:
    from catboost import CatBoostClassifier, Pool
    try: from catboost.utils import get_gpu_device_count
    except Exception: get_gpu_device_count = None
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])
    from catboost import CatBoostClassifier, Pool
    try: from catboost.utils import get_gpu_device_count
    except Exception: get_gpu_device_count = None

try:
    import gdown; HAVE_GDOWN = True
except Exception:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown; HAVE_GDOWN = True
    except Exception:
        HAVE_GDOWN = False
print(f"CatBoost OK | gdown: {HAVE_GDOWN}")

# --- GPU Detection ---
def detect_gpu_for_catboost() -> bool:
    has_dev = any(Path(f"/dev/nvidia{i}").exists() for i in range(4))
    cnt = 0
    if get_gpu_device_count is not None:
        try: cnt = int(get_gpu_device_count())
        except Exception: cnt = 0
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    blocked = (cuda_visible.strip() == "-1")
    return (has_dev or cnt > 0) and not blocked
USE_GPU = bool(USE_GPU_WISH and detect_gpu_for_catboost())
print("Will try GPU:", USE_GPU)

# === 2. Load Data & Create Labels ===

# --- Helper Functions for Data Loading ---
def download_target_tokens_csv(dst_path: Path):
    if dst_path.exists(): return True
    if not HAVE_GDOWN:
        print("gdown not available — upload target_tokens.csv to /kaggle/working/")
        return False
    url = f"https://drive.google.com/uc?id={TARGET_TOKENS_GDRIVE_ID}"
    gdown.download(url, str(dst_path), quiet=False)
    return dst_path.exists() and dst_path.stat().st_size > 0

def load_target_list(path: Path):
    df = pd.read_csv(path)
    # --- MODIFIED: Use the correct column name from baseline ---
    cands = [c for c in df.columns if "Target Token Addresses" in c] 
    if not cands: # Fallback to our logic
         cands = [c for c in df.columns if "mint" in c.lower() or "token" in c.lower()]
    # --- End Modification ---
    assert cands, "Cannot find token id column in target csv"
    col = cands[0]
    t = df[[col]].dropna().drop_duplicates().rename(columns={col: KEY_COL})
    t[KEY_COL] = t[KEY_COL].astype(str)
    print(f"Loaded {len(t)} target (positive) tokens successfully.")
    return t

# --- 2. Load Target Tokens (Using GDrive) ---
print(f"Attempting to download target tokens to: {TARGET_TOKEN_PATH}")
ok = download_target_tokens_csv(TARGET_TOKEN_PATH)
if not ok: raise SystemExit("Missing target_tokens.csv")

target_list_df = load_target_list(TARGET_TOKEN_PATH)
target_token_set = set(target_list_df['mint_token_id'])
# --- End ---

# --- 3. Load Training Transactions ---
print("Loading training transaction data (Sample_Dataset.csv)...")
try:
    train_tx_df = pd.read_csv(os.path.join(DATA_PATH, 'Sample_Dataset.csv'))
    print(f"Loaded {len(train_tx_df)} training transactions.")
except FileNotFoundError:
    print("FATAL: Sample_Dataset.csv not found. This strategy relies on it.")
    # Add fallback just in case
    FULL_DIR  = Path("/kaggle/input/pumpfun-30s-september-2025")
    full_paths = sorted(FULL_DIR.glob("september_2025_first30s_chunk_*.csv"))
    train_dfs = [pd.read_csv(p) for p in full_paths]
    train_tx_df = pd.concat(train_dfs, ignore_index=True)
    print(f"Loaded {len(train_tx_df)} training transactions from full dataset (FALLBACK).")


# --- 4. Load & Concatenate Evaluation Transactions ---
print("Loading evaluation transaction data (chunks)...")
eval_files = sorted(glob.glob(os.path.join(DATA_PATH, 'evaluation_set_30s_chunk_*.csv')))
eval_dfs = []
for f in tqdm(eval_files, desc="Loading eval chunks"):
    eval_dfs.append(pd.read_csv(f))
eval_tx_df = pd.concat(eval_dfs, ignore_index=True)
print(f"Loaded {len(eval_tx_df)} evaluation transactions from {len(eval_files)} files.")

# --- 5. Get all unique tokens from the training transaction data ---
unique_train_tokens = train_tx_df['mint_token_id'].unique()
print(f"Found {len(unique_train_tokens)} unique tokens in the training transaction set.")

# --- 6. Create the y_train (ground truth) DataFrame ---
y_train_list = [1 if token in target_token_set else 0 for token in unique_train_tokens]
y_train_df = pd.DataFrame({
    'mint_token_id': unique_train_tokens,
    'is_target': y_train_list
})
# --- MODIFIED: Create the Series y_train as in the baseline ---
y_train = y_train_df.set_index('mint_token_id')['is_target'] # For model training
print(f"Created y_train with {len(y_train)} entries.")
# --- End Modification ---

# --- 7. Check Class Imbalance & Get Ratio ---
print("\n--- Training Set Class Imbalance ---")
print(y_train.value_counts(normalize=True))
imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nCalculated imbalance ratio (scale_pos_weight): {imbalance_ratio:.2f}")


# === 3. Feature Engineering (Baseline V1) ===

# --- Define the Feature Engineering Function (V1 from baseline) ---
def create_features(df, numeric_cols, categorical_cols):
    """
    V1: Our simple, robust baseline feature set.
    """
    
    print("Starting V1 feature aggregation...")
    
    # --- Force all numeric columns to be numeric *before* aggregation ---
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    aggregations = {}
    def add_agg(col, agg_func):
        if col not in aggregations: aggregations[col] = []
        if agg_func not in aggregations[col]: aggregations[col].append(agg_func)

    last_state_cols = ['market_cap_usd', 'buy_count', 'sell_count', 'total_count', 'total_holders', 'current_holders', 'top10_percent_total', 'creator_balance', 'creator_sold', 'buy_sell_ratio', 'relative_strength_index', 'bollinger_relative_position', 'money_flow_index']
    for col in last_state_cols:
        if col in numeric_cols: add_agg(col, 'last')
    sum_cols = ['sol_volume', 'token_volume', 'consumed_gas', 'fee', 'sol_delta']
    for col in sum_cols:
         if col in numeric_cols: add_agg(col, 'sum')
    mean_cols = ['volume_oscillator', 'rate_of_change', 'liquidity_ratio']
    for col in mean_cols:
         if col in numeric_cols: add_agg(col, 'mean')
    if 'holder' in categorical_cols: add_agg('holder', 'nunique')
    if 'creator' in categorical_cols: add_agg('creator', 'first')
    if 'creator_fee' in numeric_cols: add_agg('creator_fee', 'first')
    if 'market_cap_usd' in numeric_cols:
        add_agg('market_cap_usd', 'std')
        add_agg('market_cap_usd', lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 0 else 0)

    final_aggregations = {col: funcs for col, funcs in aggregations.items()}
            
    try:
        df_sorted = df.sort_values(by=['mint_token_id', 'timestamp'])
    except KeyError:
        print("Warning: 'timestamp' column not found for sorting. Proceeding without sorting.")
        # --- ADDED: Fallback sort by row order ---
        df['_row_order'] = df.groupby('mint_token_id').cumcount()
        df_sorted = df.sort_values(by=['mint_token_id', '_row_order'])
        # --- End Add ---
        
    feature_df = df_sorted.groupby('mint_token_id').agg(final_aggregations)
    
    # Flatten MultiIndex Columns
    new_cols = []
    for col in feature_df.columns:
        if isinstance(col, tuple):
            if '<lambda>' in str(col[1]): new_cols.append(f"{col[0]}_change")
            else: new_cols.append(f"{col[0]}_{col[1]}")
        else: new_cols.append(col)
    feature_df.columns = new_cols
    
    if 'holder_nunique' in feature_df.columns: feature_df = feature_df.rename(columns={'holder_nunique': 'unique_holders'})
    if 'creator_first' in feature_df.columns: feature_df = feature_df.rename(columns={'creator_first': 'creator'})

    # --- V1 Filling NaNs ---
    if 'market_cap_usd_std' in feature_df.columns: feature_df['market_cap_usd_std'] = feature_df['market_cap_usd_std'].fillna(0)
    if 'buy_sell_ratio_last' in feature_df.columns: 
        feature_df['buy_sell_ratio_last'] = feature_df['buy_sell_ratio_last'].fillna(1000).replace(np.inf, 1000)

    feature_df = feature_df.fillna(0)
    
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

# --- 4. NO Creator DNA in this version ---

# --- 5. Align Training Labels (y_train) ---
print("\nAligning data...")
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
X_train = X_train.reindex(columns=X_test.columns, fill_value=0) # Align both ways

y_train_aligned = y_train.reindex(X_train.index).fillna(0).astype(int) # Get the aligned series

# --- 6. Define Categorical Features for CatBoost ---
cat_features = [col for col in ['creator'] if col in X_train.columns]

# --- 7. Final Verification ---
print("\n--- Final Shapes ---")
print(f"X_train shape:      {X_train.shape}")
print(f"y_train_aligned shape: {y_train_aligned.shape}")
print(f"X_test shape:       {X_test.shape}")
print(f"Categorical features: {cat_features}")

required_rows = 64208
if X_test.shape[0] == required_rows:
    print(f"\nSUCCESS: X_test has exactly {required_rows} rows.")
else:
    print(f"\nWARNING: X_test has {X_test.shape[0]} rows.")
    # Add reindexing logic from baseline
    correct_token_order = pd.Series(eval_tx_df['mint_token_id'].unique())
    X_test = X_test.reindex(correct_token_order, fill_value=0)
    print(f"Reindexed X_test shape: {X_test.shape}")
    if X_test.shape[0] == required_rows:
         print("SUCCESS after reindexing.")
    else:
         print("FATAL: Reindexing failed to fix row count.")

# Clean up
del train_tx_df, eval_dfs, target_list_df; gc.collect()


# === 5. Model Training (V1 - Tuned) ===
# --- 1. Define Model Parameters ---
N_SPLITS = 5
RANDOM_STATE = 42
SCALE_POS_WEIGHT = imbalance_ratio # From Cell 2

# --- MODIFIED: Tuned Hyperparameters ---
cb_params = {
    'iterations': 4000,             # More iterations
    'learning_rate': 0.02,          # Slower learning rate
    'depth': 7,                     # Deeper trees
    'l2_leaf_reg': 10.0,            # Added regularization
    'loss_function': 'Logloss',
    'eval_metric': 'F1',
    'scale_pos_weight': SCALE_POS_WEIGHT,
    'random_seed': RANDOM_STATE,
    'early_stopping_rounds': 200,     # More patience
    'verbose': 250,
    'cat_features': cat_features
}
# --- End Modification ---

if USE_GPU:
    cb_params['task_type'] = 'GPU'
    cb_params['devices'] = '0'
else:
    cb_params.pop('task_type', None)
    cb_params.pop('devices', None)

# --- 2. Set up Cross-Validation ---
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# --- 3. Prepare Arrays to Store Predictions ---
oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))
models = []

print("--- Starting CatBoost Cross-Validation (V15 - Tuned Baseline) ---")
for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train_aligned)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train_aligned.iloc[train_index], y_train_aligned.iloc[val_index]

    train_pool = Pool(X_train_fold, y_train_fold, cat_features=cat_features)
    val_pool = Pool(X_val_fold, y_val_fold, cat_features=cat_features)

    model = CatBoostClassifier(**cb_params)
    
    try:
        model.fit(train_pool, eval_set=val_pool)
    except Exception as e:
        print(f"Error during training (fold {fold+1}): {e}")
        if "GPU" in str(e) or "Cuda" in str(e) or "driver" in str(e):
            print("GPU error, falling back to CPU...")
            cb_params_cpu = cb_params.copy()
            cb_params_cpu.pop('task_type', None)
            cb_params_cpu.pop('devices', None)
            model = CatBoostClassifier(**cb_params_cpu)
            model.fit(train_pool, eval_set=val_pool)

    oof_preds[val_index] = model.predict_proba(val_pool)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

    models.append(model)
    print(f"Fold {fold+1} finished.")
    del X_train_fold, X_val_fold, y_train_fold, y_val_fold, train_pool, val_pool; gc.collect()

print("\n--- Cross-Validation Complete ---")
print("OOF predictions generated.")
print("Test predictions generated.")

# === 6. Find Optimal Threshold (Fine Search) ===
# --- 1. Find Best Threshold (Fine Search) ---
thresholds = np.linspace(0.01, 0.99, 500)
best_jaccard = -1
best_recall = -1
best_threshold = 0
RECALL_TARGET = 0.75 # <-- Using the 0.163 baseline's 75% target

print("Starting fine threshold search (500 steps)...")
for threshold in thresholds:
    oof_binary_preds = (oof_preds > threshold).astype(int)
    recall = recall_score(y_train_aligned, oof_binary_preds, zero_division=0)

    if recall >= RECALL_TARGET: # Must have at least 75% Recall
        jaccard = jaccard_score(y_train_aligned, oof_binary_preds, zero_division=0)
        if jaccard > best_jaccard: # If it does, find the best Jaccard
            best_jaccard = jaccard
            best_recall = recall
            best_threshold = threshold

print("--- Optimal Threshold Search (V15) ---")
if best_threshold > 0:
    print(f"Optimal Threshold: {best_threshold:.4f}")
    print(f"   Best Jaccard: {best_jaccard:.4f}")
    print(f"       OOF Recall: {best_recall:.4f}")
else:
    print("\nWARNING: Could not find a threshold that achieves 75% recall.")
    print("Fallback: Finding best Jaccard score regardless of recall...")
    for threshold in thresholds:
        oof_binary_preds = (oof_preds > threshold).astype(int)
        jaccard = jaccard_score(y_train_aligned, oof_binary_preds, zero_division=0)
        if jaccard > best_jaccard:
            best_jaccard = jaccard
            best_recall = recall_score(y_train_aligned, oof_binary_preds, zero_division=0)
            best_threshold = threshold
    print(f"Fallback Optimal Threshold: {best_threshold:.4f}")
    print(f"   Best Jaccard (fallback): {best_jaccard:.4f}")
    print(f"       OOF Recall (fallback): {best_recall:.4f}")


# --- 2. Show Confusion Matrix ---
print("\n--- OOF Confusion Matrix (V15) ---")
final_oof_preds = (oof_preds > best_threshold).astype(int)
cm = confusion_matrix(y_train_aligned, final_oof_preds)

try:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title(f"OOF Confusion Matrix (Threshold = {best_threshold:.4f})")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig("oof_confusion_matrix.png") # Save the plot
    print("Saved OOF confusion matrix plot to oof_confusion_matrix.png")
except Exception as e:
    print(f"Could not generate plot: {e}")


# === 7. Create Submission Files ===
# --- 1. Apply the optimal threshold ---
final_binary_preds = (test_preds > best_threshold).astype(int)

# --- 2. Create the main submission.csv file (with correct order) ---
# --- MODIFIED: Use X_test.index for correct order ---
submission_df_ordered = pd.DataFrame({'mint_token_id': X_test.index})

predictions_map = pd.Series(test_preds, index=X_test.index)
binary_predictions_map = pd.Series(final_binary_preds, index=X_test.index)

submission_df_ordered['is_target'] = submission_df_ordered['mint_token_id'].map(binary_predictions_map)
submission_df_ordered['prediction_value'] = submission_df_ordered['mint_token_id'].map(predictions_map)
# --- End Modification ---

submission_df_ordered['is_target'] = submission_df_ordered['is_target'].fillna(0).astype(int)
submission_df_ordered['prediction_value'] = submission_df_ordered['prediction_value'].fillna(0)


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
print(f"deliverable_df.csv shape: {deliverable_df.shape}")
print(f"Final positive prediction rate: {final_submission_csv['is_target'].mean():.2%}")

print("\nAll done!")

