!pip install lifelines


!pip install py7zr


!pip install pycox torchtuples


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

# Any results you write to the current directory are saved as output.
import os
print(os.listdir("../input"))


!ls -d /kaggle/input/kkbox-churn-prediction-challenge/*


import os
import py7zr
import pandas as pd
import json

INPUT_PATH = "/kaggle/input/kkbox-churn-prediction-challenge"
OUTPUT_PATH = "/kaggle/working/extracted"
os.makedirs(OUTPUT_PATH, exist_ok=True)


def safe_preview_csv(path, n=10):
    try:
        df = pd.read_csv(path, nrows=n)
        display(df)
    except Exception as e:
        print(f"Failed to read CSV: {e}")


def preview_file(path, preview_rows=10):
    fname = os.path.basename(path)

    # CSV
    if fname.endswith(".csv"):
        print(f"\n--- {fname} (CSV) ---")
        safe_preview_csv(path, preview_rows)
        return

    # TSV
    if fname.endswith(".tsv"):
        print(f"\n--- {fname} (TSV) ---")
        try:
            df = pd.read_csv(path, sep="\t", nrows=preview_rows)
            display(df)
        except Exception as e:
            print(f"Failed to read TSV: {e}")
        return

    # JSON
    if fname.endswith(".json"):
        print(f"\n--- {fname} (JSON) ---")
        try:
            with open(path) as j:
                data = json.load(j)
            print(json.dumps(data, indent=2)[:2000])
        except Exception as e:
            print(f"Failed to read JSON: {e}")
        return

    # Plain text / code formats
    if fname.endswith((".txt", ".md", ".py", ".log", ".scala")):
        print(f"\n--- {fname} (text) ---")
        try:
            with open(path, "r", errors="ignore") as t:
                print("".join(t.readlines()[:preview_rows]))
        except Exception as e:
            print(f"Failed to read text: {e}")
        return

    # FALLBACK FOR BINARY / UNKNOWN
    print(f"\n--- {fname} (unknown/binary) ---")
    try:
        with open(path, "rb") as b:
            print(b.read(200))
    except Exception as e:
        print(f"(unreadable) {e}")


def inspect_folder(folder, preview_rows=10):
    for root, dirs, files in os.walk(folder):
        for f in files:
            preview_file(os.path.join(root, f), preview_rows)


def inspect_7z(filename, preview_rows=10):
    src = f"{INPUT_PATH}/{filename}"
    dst = f"{OUTPUT_PATH}/{filename.replace('.7z', '')}"

    os.makedirs(dst, exist_ok=True)

    print(f"\n=== Extracting {filename} ===")

    with py7zr.SevenZipFile(src, "r") as archive:
        archive.extractall(path=dst)

    print("\nContents:")
    for item in os.listdir(dst):
        print(" -", item)

    print("\n=== PREVIEW ===")
    inspect_folder(dst, preview_rows)


files = [
    "train_v2.csv.7z",
    "members_v3.csv.7z",
    "transactions_v2.csv.7z",
    "user_logs_v2.csv.7z",
    "sample_submission_v2.csv.7z"    
]

for f in files:
    inspect_7z(f)


!ls -d /kaggle/working/extracted/*


mem_df = pd.read_csv('/kaggle/working/extracted/members_v3.csv/members_v3.csv')


mem_df[mem_df['msno']=="moRTKhKIDvb+C8ZHOgmaF4dXMLk0jOn65d7a8tQ2Eds="]


mem_df['registration_init_time_dt']=pd.to_datetime(mem_df['registration_init_time'], format='%Y%m%d')


mem_df = mem_df.drop(columns=['registration_init_time'])
mem_df.head()


import sys, subprocess
try:
    import pycox, torch  # quick check
except Exception:
    print("Installing pycox + torchtuples + light torch (CPU)...")
    !pip install --no-cache-dir pycox torchtuples
    # install a CPU-only torch if available (pytorch cpu wheels tend to be auto-selected)
    !pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu


import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import gc  # For garbage collection to free memory


# --- Paths ---
EXTRACTED = "/kaggle/working/extracted"
train_v2_path = os.path.join(EXTRACTED, "train_v2.csv", "data", "churn_comp_refresh", "train_v2.csv")
transactions_v2_path = os.path.join(EXTRACTED, "transactions_v2.csv", "data", "churn_comp_refresh", "transactions_v2.csv")
members_path = os.path.join(EXTRACTED, "members_v3.csv", "members_v3.csv")

# --- Load datasets ---
print("Loading train...")
train = pd.read_csv(train_v2_path, usecols=["msno","is_churn"])
print(f"Train shape: {train.shape}")

print("Loading members...")
members = pd.read_csv(members_path)
print(f"Members shape: {members.shape}")

# --- Load transactions (only necessary columns) ---
print("Loading transactions (this may take a while)...")
transactions = pd.read_csv(
    transactions_v2_path, 
    usecols=['msno', 'transaction_date', 'membership_expire_date']
)
print(f"Transactions shape: {transactions.shape}")
print(transactions.head(10))


# --- Preprocess transactions ---
print("Processing transactions...")
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'], format='%Y%m%d', errors='coerce')
transactions['membership_expire_date'] = pd.to_datetime(transactions['membership_expire_date'], format='%Y%m%d', errors='coerce')

# Remove invalid dates
transactions = transactions.dropna(subset=['transaction_date', 'membership_expire_date'])

# Aggregate per user
print("Aggregating transactions per user...")
trans_agg = transactions.groupby('msno', as_index=False).agg(
    first_trans=('transaction_date', 'min'),
    last_expire=('membership_expire_date', 'max'),
    num_transactions=('transaction_date', 'count')  # Transaction count as feature
)

# Free up memory
del transactions
gc.collect()
print(f"Aggregated transactions shape: {trans_agg.shape}")
print(trans_agg.head(10))


# --- Merge with train ---
print("Merging datasets...")
df = train.merge(trans_agg, on='msno', how='left')
df = df.merge(mem_df, on='msno', how='left')
print(f"Merged shape: {df.shape}")

# df table
print(df.head(10))

# Free memory
del train, trans_agg, members
gc.collect()


# --- Calculate duration ---
print("Calculating duration...")
df['duration'] = (df['last_expire'] - df['first_trans']).dt.days

# Handle invalid durations (negative or NaN)
# For users with no transaction history, use a default of 30 days
df['duration'] = df['duration'].apply(lambda x: max(1, x) if pd.notna(x) else 30)

# Event observed
df['event_observed'] = df['is_churn'].astype(int)

print(f"Duration range: [{df['duration'].min()}, {df['duration'].max()}]")
print(f"Event rate: {df['event_observed'].mean():.2%}")
print(f"Rows with NaN first_trans: {df['first_trans'].isna().sum()}")
print(f"Rows with NaN last_expire: {df['last_expire'].isna().sum()}")


# --- Clean member features ---
print("Cleaning features...")

# 1. Age (bd) - filter outliers and fill missing
df['bd'] = df['bd'].apply(lambda x: x if (x >= 0 and x <= 90) else np.nan)

# 2. Gender - encode and handle missing
df['gender'] = df['gender'].map({'male': 1, 'female': 0})
df.fillna({'gender': -1}, inplace=True)
# df['gender'].fillna(-1, inplace=True)  # -1 for unknown

# 4. Registration via - categorical
df['registered_via'].fillna(-1, inplace=True)

print("Gender distribution:")
print(df['gender'].value_counts())
print(df.head(10))


# --- Remove rows with invalid duration or missing key features ---
print("\n=== Final Data Validation ===")
initial_rows = len(df)

# Remove rows with invalid duration
df = df[df['duration'] > 0].copy()
print(f"Removed {initial_rows - len(df)} rows with invalid duration")

# Remove rows with missing event
df = df.dropna(subset=['event_observed'])
print(f"Final dataset: {len(df)} rows")

# --- Define feature columns ---
feature_cols = [
    'bd', 
    'gender', 
    'city', 
    'registered_via',
    'num_transactions'
]

# Add registration features if they exist
if 'registration_year' in df.columns:
    feature_cols.extend(['registration_year', 'registration_month', 'account_age_days', 'registration_missing'])
elif 'has_registration_data' in df.columns:
    feature_cols.append('has_registration_data')

print(f"\nFeatures to use ({len(feature_cols)}): {feature_cols}")

# Create final dataframe with only needed columns
df_final = df[feature_cols + ['duration', 'event_observed']].copy()

# Final check for any remaining NaNs
print("\n=== Missing Values Check ===")
missing_summary = df_final.isnull().sum()
if missing_summary.sum() > 0:
    print("⚠️ WARNING: Some columns still have missing values:")
    print(missing_summary[missing_summary > 0])
    print("\nFilling remaining NaNs with -1...")
    df_final = df_final.fillna(-1)
else:
    print("✓ No missing values!")

print("\n=== Final Dataset Summary ===")
print(f"Shape: {df_final.shape}")
print(f"\nDuration stats:")
print(df_final['duration'].describe())
print(f"\nEvent distribution:")
print(df_final['event_observed'].value_counts())

# Free memory
del df
gc.collect()

print("\n✓ Data cleaning complete!")


# --- Sample for faster training (OPTIONAL) ---
# Comment this out to use full dataset once you verify it works
SAMPLE_SIZE = 50000  # Adjust based on your memory

if len(df_final) > SAMPLE_SIZE:
    df_sample = df_final.sample(SAMPLE_SIZE, random_state=42)
    print(f"Sampled {SAMPLE_SIZE} rows from {len(df_final)} total rows")
else:
    df_sample = df_final.copy()
    print(f"Using all {len(df_final)} rows")

print(f"\nSample event rate: {df_sample['event_observed'].mean():.2%}")


# --- Prepare features and target ---
print("=== Preparing Data for Cox Model ===")

# Get feature columns (everything except duration and event_observed)
feature_cols = [col for col in df_sample.columns if col not in ['duration', 'event_observed']]

print(f"Using {len(feature_cols)} features: {feature_cols}")

X = df_sample[feature_cols].values.astype('float32')
durations = df_sample['duration'].values.astype('float32')
events = df_sample['event_observed'].values.astype('float32')

print(f"\nData shapes:")
print(f"  X: {X.shape}")
print(f"  Durations: {durations.shape}")
print(f"  Events: {events.shape}")

print(f"\nData ranges:")
print(f"  Duration: [{durations.min():.1f}, {durations.max():.1f}] days")
print(f"  Event rate: {events.mean():.2%}")

# Check for any remaining NaNs or infs
if np.isnan(X).any():
    print(f"⚠️ WARNING: X contains {np.isnan(X).sum()} NaN values")
if np.isinf(X).any():
    print(f"⚠️ WARNING: X contains {np.isinf(X).sum()} Inf values")


# --- Standardize features ---
print("\n=== Standardizing Features ===")

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

print(f"Standardized X shape: {X_std.shape}")
print(f"Mean of first feature: {X_std[:, 0].mean():.6f} (should be ~0)")
print(f"Std of first feature: {X_std[:, 0].std():.6f} (should be ~1)")

# Verify no NaNs after scaling
if np.isnan(X_std).any():
    print(f"⚠️ ERROR: Standardization created {np.isnan(X_std).sum()} NaN values!")
    print("This usually means a feature had zero variance (all same value)")
else:
    print("✓ No NaN values after standardization")


# --- Train/test split with stratification ---
print("\n=== Splitting Train/Test ===")

X_train, X_test, durations_train, durations_test, events_train, events_test = train_test_split(
    X_std, durations, events, 
    test_size=0.2, 
    random_state=42, 
    stratify=events
)

print(f"Train size: {X_train.shape[0]:,} samples")
print(f"Test size: {X_test.shape[0]:,} samples")
print(f"Train event rate: {events_train.mean():.2%}")
print(f"Test event rate: {events_test.mean():.2%}")

# Check duration distributions
print(f"\nTrain duration: [{durations_train.min():.1f}, {durations_train.max():.1f}]")
print(f"Test duration: [{durations_test.min():.1f}, {durations_test.max():.1f}]")


print("\n=== Converting to PyTorch Tensors ===")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
durations_train_tensor = torch.tensor(durations_train, dtype=torch.float32)
events_train_tensor = torch.tensor(events_train, dtype=torch.float32)
durations_test_tensor = torch.tensor(durations_test, dtype=torch.float32)
events_test_tensor = torch.tensor(events_test, dtype=torch.float32)

print("✓ Tensors created successfully!")
print(f"  X_train: {X_train_tensor.shape}")
print(f"  X_test: {X_test_tensor.shape}")


# --- Build and train Cox model ---
print("\n=== Building and Training Cox Model ===")

in_features = X_train_tensor.shape[1]

# Create neural network for Cox model
net = tt.practical.MLPVanilla(
    in_features, 
    [32, 32],  # Two hidden layers with 32 nodes each
    out_features=1, 
    batch_norm=True, 
    dropout=0.1
)

# Create Cox model
model = CoxPH(net, tt.optim.Adam(lr=0.01))

print(f"Model architecture: {in_features} -> 32 -> 32 -> 1")

# Train the model
print("\nTraining...")
log = model.fit(
    X_train_tensor,
    (durations_train_tensor, events_train_tensor),
    batch_size=256,
    epochs=100,
    verbose=True,
    val_data=(X_test_tensor, (durations_test_tensor, events_test_tensor)),
    val_batch_size=256
)

print("\n✓ Training complete!")


from sklearn.metrics import brier_score_loss

def compute_brier_score_at_time(surv_df, durations, events, time_point):

    # Get survival probability at time_point
    if time_point in surv_df.index:
        surv_prob = surv_df.loc[time_point].values
    else:
        # Find closest time point
        idx = (surv_df.index - time_point).abs().argmin()
        surv_prob = surv_df.iloc[idx].values
    
    # Binary outcome: did event occur before time_point?
    # For Brier score: 1 if event occurred before time_point, 0 otherwise
    y_true = np.zeros(len(durations))
    
    for i in range(len(durations)):
        if events[i] == 1 and durations[i] <= time_point:
            # Event occurred before time_point
            y_true[i] = 1
        elif events[i] == 0 and durations[i] <= time_point:
            # Censored before time_point - exclude from calculation
            y_true[i] = np.nan
        else:
            # Event occurred after time_point or not yet observed
            y_true[i] = 0
    
    # Remove censored observations
    mask = ~np.isnan(y_true)
    y_true_clean = y_true[mask]
    surv_prob_clean = surv_prob[mask]
    
    if len(y_true_clean) == 0:
        return np.nan
    
    # Brier score uses (1 - survival_prob) as predicted event probability
    y_pred = 1 - surv_prob_clean
    
    # Calculate Brier score (MSE between predicted and actual)
    brier = np.mean((y_pred - y_true_clean) ** 2)
    
    return brier

# Calculate Brier scores at multiple time points
time_points = [30, 60, 90, 120, 180, 365]


# --- Evaluate model on TRAINING data ---
print("\n" + "="*60)
print("EVALUATING ON TRAINING DATA (to check for overfitting)")
print("="*60)

# Compute baseline hazards
_ = model.compute_baseline_hazards()

# Predict survival curves on TRAINING data
surv_train = model.predict_surv_df(X_train_tensor)

# Create EvalSurv object for training data
ev_train = EvalSurv(surv_train, durations_train, events_train, censor_surv='km')

# 1. C-INDEX on training data
c_index_train = ev_train.concordance_td()
print(f"\nC-INDEX (Train): {c_index_train:.4f}")

# --- Same for TRAINING data ---
print("\n=== BRIER SCORES (TRAINING DATA) ===")
brier_scores_train = []
valid_times_train = []

for t in time_points:
    if t <= durations_train.max() and t >= durations_train.min():
        try:
            bs = compute_brier_score_at_time(surv_train, durations_train, events_train, t)
            if not np.isnan(bs):
                brier_scores_train.append(bs)
                valid_times_train.append(t)
                print(f"Day {t:3d}: {bs:.4f}")
        except Exception as e:
            print(f"Day {t:3d}: Could not compute ({e})")

if len(brier_scores_train) > 0:
    ibs_train = np.mean(brier_scores_train)
    print(f"\nIntegrated Brier Score (Train): {ibs_train:.4f}")
else:
    print("\nCould not compute Integrated Brier Score")
    
# 3. Time-dependent AUC at specific time points on training data
print("\n=== Time-Dependent AUC at Specific Time Points (Train) ===")
try:
    from sklearn.metrics import roc_auc_score
    
    time_points = [30, 60, 90]
    
    for t in time_points:
        if t <= durations_train.max() and t >= durations_train.min():
            # Get survival probability at time t
            surv_prob_at_t = surv_train.loc[t] if t in surv_train.index else surv_train.iloc[(surv_train.index - t).abs().argmin()]
            
            # Create binary outcome
            y_true_at_t = ((durations_train <= t) & (events_train == 1)).astype(int)
            
            if len(np.unique(y_true_at_t)) > 1:
                risk_scores = 1 - surv_prob_at_t.values
                auc_at_t = roc_auc_score(y_true_at_t, risk_scores)
                print(f"AUC at day {t}: {auc_at_t:.4f}")
            else:
                print(f"AUC at day {t}: Cannot compute (only one class present)")
                
except Exception as e:
    print(f"Could not compute time-dependent AUC: {e}")


# --- Evaluate model ---
print("\n=== Evaluating Model ===")

# Compute baseline hazards
_ = model.compute_baseline_hazards()

# Predict survival curves
surv = model.predict_surv_df(X_test_tensor)

# Create EvalSurv object
ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')

# 1. C-INDEX (Concordance Index)
c_index = ev.concordance_td()
print(f"\n{'='*50}")
print(f"C-INDEX: {c_index:.4f}")
print(f"{'='*50}")

# 2. INTEGRATED BRIER SCORE
print("\n=== BRIER SCORES (TEST DATA) ===")
brier_scores_test = []
valid_times_test = []

for t in time_points:
    if t <= durations_test.max() and t >= durations_test.min():
        try:
            bs = compute_brier_score_at_time(surv, durations_test, events_test, t)
            if not np.isnan(bs):
                brier_scores_test.append(bs)
                valid_times_test.append(t)
                print(f"Day {t:3d}: {bs:.4f}")
        except Exception as e:
            print(f"Day {t:3d}: Could not compute ({e})")

# Calculate Integrated Brier Score (average across time points)
if len(brier_scores_test) > 0:
    ibs_test = np.mean(brier_scores_test)
    print(f"\nIntegrated Brier Score (Test): {ibs_test:.4f}")
else:
    print("\nCould not compute Integrated Brier Score")

# Alternative: Time-dependent AUC at specific time points
print("\n=== Time-Dependent AUC at Specific Time Points ===")
try:
    from sklearn.metrics import roc_auc_score
    
    # Evaluate AUC at 30, 60, and 90 days
    time_points = [30, 60, 90]
    
    for t in time_points:
        if t <= durations_test.max() and t >= durations_test.min():
            # Get survival probability at time t
            surv_prob_at_t = surv.loc[t] if t in surv.index else surv.iloc[(surv.index - t).abs().argmin()]
            
            # Create binary outcome: did event occur before time t?
            y_true_at_t = ((durations_test <= t) & (events_test == 1)).astype(int)
            
            # Only compute AUC if we have both classes
            if len(np.unique(y_true_at_t)) > 1:
                # Use 1 - survival probability as risk score
                risk_scores = 1 - surv_prob_at_t.values
                auc_at_t = roc_auc_score(y_true_at_t, risk_scores)
                print(f"AUC at day {t}: {auc_at_t:.4f}")
            else:
                print(f"AUC at day {t}: Cannot compute (only one class present)")
                
except Exception as e:
    print(f"Could not compute time-dependent AUC: {e}")

print("\n" + "="*50)
print("EVALUATION COMPLETE")
print("="*50)


print("\n" + "="*60)
print("COMPARISON: TRAINING vs TEST PERFORMANCE")
print("="*60)

# Compare C-index
print(f"\n=== C-INDEX ===")
print(f"  Train: {c_index_train:.4f}")
print(f"  Test:  {c_index:.4f}")
print(f"  Difference: {c_index_train - c_index:.4f}")

if c_index_train - c_index > 0.05:
    print("  ⚠️  WARNING: Significant overfitting detected!")
    print("     (Train C-index is >0.05 higher than test)")
elif c_index_train - c_index > 0.02:
    print("  ⚠️  Mild overfitting detected")
    print("     (Train C-index is 0.02-0.05 higher than test)")
else:
    print("  ✅ Good generalization! Model is not overfitting significantly.")

print(f"\n=== Brier Score ===")
print(f"  Train: {ibs_train:.4f}")
print(f"  Test:  {ibs_test:.4f}")
print(f"  Difference: {ibs_train - ibs_test:.4f}")

# Compare AUC at day 30 if available
print(f"\n=== Time-Dependent AUC Comparison ===")
try:
    for t in [30, 60, 90]:
        if t <= durations_test.max() and t <= durations_train.max():
            # Test AUC
            surv_prob_test = surv.loc[t] if t in surv.index else surv.iloc[(surv.index - t).abs().argmin()]
            y_true_test = ((durations_test <= t) & (events_test == 1)).astype(int)
            
            # Train AUC
            surv_prob_train = surv_train.loc[t] if t in surv_train.index else surv_train.iloc[(surv_train.index - t).abs().argmin()]
            y_true_train = ((durations_train <= t) & (events_train == 1)).astype(int)
            
            if len(np.unique(y_true_test)) > 1 and len(np.unique(y_true_train)) > 1:
                auc_test = roc_auc_score(y_true_test, 1 - surv_prob_test.values)
                auc_train = roc_auc_score(y_true_train, 1 - surv_prob_train.values)
                
                print(f"\nDay {t}:")
                print(f"  Train AUC: {auc_train:.4f}")
                print(f"  Test AUC:  {auc_test:.4f}")
                print(f"  Difference: {auc_train - auc_test:.4f}")
except Exception as e:
    print(f"Could not compare AUCs: {e}")

print("\n" + "="*60)




