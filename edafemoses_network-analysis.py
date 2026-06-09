# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

print("Starting Phase 1 & 2: Data Ingestion, Preprocessing, and Feature Engineering (Vectorized Labeling - Fixed Labels)...\n")

# --- 1. Dataset Consolidation ---
# Define the list of dataset filenames with their full Kaggle paths
dataset_filenames = [
    "/kaggle/input/network-malware-detection-connection-analysis/CTU-IoT-Malware-Capture-35-1conn.log.labeled.csv",
    "/kaggle/input/network-malware-detection-connection-analysis/CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv",
    "/kaggle/input/network-malware-detection-connection-analysis/CTU-IoT-Malware-Capture-3-1conn.log.labeled.csv",
    "/kaggle/input/network-malware-detection-connection-analysis/CTU-IoT-Malware-Capture-8-1conn.log.labeled.csv",
    "/kaggle/input/network-malware-detection-connection-analysis/CTU-IoT-Malware-Capture-9-1conn.log.labeled.csv",
    "/kaggle/input/network-malware-detection-connection-analysis/CTU-IoT-Malware-Capture-20-1conn.log.labeled.csv",
    "/kaggle/input/network-malware-detection-connection-analysis/CTU-IoT-Malware-Capture-21-1conn.log.labeled.csv",
    "/kaggle/input/network-malware-detection-connection-analysis/CTU-IoT-Malware-Capture-34-1conn.log.labeled.csv",
    "/kaggle/input/network-malware-detection-connection-analysis/CTU-IoT-Malware-Capture-42-1conn.log.labeled.csv",
    "/kaggle/input/network-malware-detection-connection-analysis/CTU-IoT-Malware-Capture-44-1conn.log.labeled.csv",
    "/kaggle/input/network-malware-detection-connection-analysis/CTU-IoT-Malware-Capture-48-1conn.log.labeled.csv",
    "/kaggle/input/network-malware-detection-connection-analysis/CTU-IoT-Malware-Capture-60-1conn.log.labeled.csv"
]

# Define a common header string for all files based on your provided data
common_header = "ts|uid|id.orig_h|id.orig_p|id.resp_h|id.resp_p|proto|service|duration|orig_bytes|resp_bytes|conn_state|local_orig|local_resp|missed_bytes|history|orig_pkts|orig_ip_bytes|resp_pkts|resp_ip_bytes|tunnel_parents|label|detailed-label"
common_columns = common_header.split('|')

# Define dtypes for memory efficiency during initial load
dtype_mapping = {
    'ts': np.float32,
    'uid': 'object',
    'id.orig_h': 'object',
    'id.orig_p': np.int32,
    'id.resp_h': 'object',
    'id.resp_p': np.int32,
    'proto': 'category',
    'service': 'category',
    'duration': np.float32,
    'orig_bytes': np.float32,
    'resp_bytes': np.float32,
    'conn_state': 'category',
    'local_orig': 'category',
    'local_resp': 'category',
    'missed_bytes': np.float32,
    'history': 'object', # Keep as object for now to extract flags, then drop
    'orig_pkts': np.int32,
    'orig_ip_bytes': np.float32,
    'resp_pkts': np.int32,
    'resp_ip_bytes': np.float32,
    'tunnel_parents': 'category',
    'label': 'category',
    'detailed-label': 'category'
}

all_dataframes = []
for filename in dataset_filenames:
    filepath = filename
    try:
        df_temp = pd.read_csv(
            filepath,
            sep='|',
            names=common_columns,
            skiprows=1,
            skipinitialspace=True,
            dtype={k: v for k, v in dtype_mapping.items() if k not in ['duration', 'orig_bytes', 'resp_bytes']},
            low_memory=False
        )
        all_dataframes.append(df_temp)
        print(f"Successfully loaded {filename} with {len(df_temp)} rows.")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}. Please ensure it exists at the specified Kaggle path. Skipping.")
    except Exception as e:
        print(f"Error loading {filename}: {e}. Skipping.")

if all_dataframes:
    df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\nAll datasets consolidated. Total rows: {len(df)}")
else:
    print("\nNo dataframes were loaded. Please check your file paths and ensure files exist.")
    exit()


# --- Initial Data Information (Before Cleaning) ---
print("\n--- Initial Data Information (Before Cleaning) ---")
df.info(memory_usage='deep')

print("\n--- Descriptive Statistics for Numerical Columns (Before Cleaning) ---")
print(df.describe())

print("\n--- Missing Values Count (Before Cleaning) ---")
print(df.isnull().sum())

print("\n--- Unique Values for Key Categorical Columns (Before Cleaning) ---")
for col in ['proto', 'service', 'conn_state', 'label', 'detailed-label', 'history']: # Include history for initial check
    if col in df.columns:
        print(f"\nUnique values for '{col}':")
        print(df[col].value_counts())
    else:
        print(f"Column '{col}' not found.")

# --- Handling Missing Values and Type Conversion (Revised Phase 1) ---
numerical_cols_to_impute_zero = [
    'duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
    'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes'
]
for col in numerical_cols_to_impute_zero:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype_mapping[col])
        df[col] = df[col].fillna(0)
        print(f"Filled NaN in '{col}' with 0 and converted to numeric ({dtype_mapping[col]}).")

categorical_cols_to_process = [
    'service', 'local_orig', 'local_resp', 'tunnel_parents',
    'proto', 'conn_state', 'label', 'detailed-label'
]
for col in categorical_cols_to_process:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace('', np.nan)
        df[col] = df[col].fillna('unknown')
        df[col] = df[col].astype('category')
        print(f"Processed '{col}' (stripped, filled unknown, converted to category).")

if 'ts' in df.columns:
    df['ts'] = pd.to_numeric(df['ts'], errors='coerce').astype(dtype_mapping['ts'])
    df['ts'] = df['ts'].fillna(df['ts'].median())
    print("Converted 'ts' to numeric and handled any non-numeric values.")

columns_to_drop_early = ['uid', 'local_orig', 'local_resp', 'tunnel_parents']
existing_columns_to_drop = [col for col in columns_to_drop_early if col in df.columns]
if existing_columns_to_drop:
    df = df.drop(existing_columns_to_drop, axis=1)
    print(f"Dropped columns: {existing_columns_to_drop}.")

print("\n--- Phase 1: Data Ingestion and Initial Preprocessing Completed. ---\n")

# --- Phase 2: Feature Engineering and Selection ---
print("Starting Phase 2: Feature Engineering and Selection...\n")

# 1. Feature Extraction from IP Addresses (Vectorized for efficiency)
internal_ip_patterns = [
    r'^10\.',
    r'^172\.(1[6-9]|2[0-9]|3[0-1])\.',
    r'^192\.168\.'
]
combined_internal_ip_pattern = '|'.join(internal_ip_patterns)

def check_internal_ips_vectorized(ip_series):
    ip_series_str = ip_series.astype(str)
    return ip_series_str.str.contains(combined_internal_ip_pattern, regex=True, na=False).astype(np.int8)

if 'id.orig_h' in df.columns:
    df['is_orig_internal'] = check_internal_ips_vectorized(df['id.orig_h'])
    print("Created 'is_orig_internal' feature (vectorized).")
if 'id.resp_h' in df.columns:
    df['is_resp_internal'] = check_internal_ips_vectorized(df['id.resp_h'])
    print("Created 'is_resp_internal' feature (vectorized).")

df = df.drop(columns=['id.orig_h', 'id.resp_h'], errors='ignore')
print("Dropped original IP address columns (id.orig_h, id.resp_h).")


# 2. Temporal Features from `ts`
if 'ts' in df.columns: # This check is redundant if 'ts' is dropped earlier, but harmless.
    df['timestamp'] = pd.to_datetime(df['ts'], unit='s', errors='coerce')
    df['hour_of_day'] = df['timestamp'].dt.hour.astype(np.int8)
    df['day_of_week'] = df['timestamp'].dt.dayofweek.astype(np.int8)
    print("Extracted 'hour_of_day' and 'day_of_week' features.")
    df = df.drop(columns=['ts', 'timestamp'], errors='ignore')
    print("Dropped original 'ts' and 'timestamp' columns.")


# 3. Feature Transformation for Numerical Features (Log Transformation)
numerical_features_for_log_transform = [
    'duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
    'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes'
]

for col in numerical_features_for_log_transform:
    if col in df.columns:
        df[f'log_{col}'] = np.log1p(df[col]).astype(np.float32)
        df = df.drop(columns=[col])
        print(f"Applied log1p transformation to '{col}'.")


# 4. One-Hot Encoding for Categorical Features (excluding history)
categorical_features_to_ohe = [
    'proto', 'service', 'conn_state'
]

existing_categorical_features_to_ohe = [col for col in categorical_features_to_ohe if col in df.columns]

if existing_categorical_features_to_ohe:
    df = pd.get_dummies(df, columns=existing_categorical_features_to_ohe, prefix=existing_categorical_features_to_ohe, dummy_na=False)
    print(f"Applied One-Hot Encoding to: {existing_categorical_features_to_ohe}")
else:
    print("No categorical features found for One-Hot Encoding (or already processed).")

# --- New: Feature Engineering for 'history' column (individual flags) ---
if 'history' in df.columns:
    # Common Zeek/Bro connection history flags
    history_flags = ['S', 'h', 'A', 'D', 'f', 'R', 'c', 'w', 'i', 'q', 't', 'g']
    print(f"Extracting individual flags from 'history' column: {history_flags}")

    # Ensure history column is string type for .str.contains
    df['history'] = df['history'].astype(str).fillna('') # Fill NaN with empty string for safety

    for flag in history_flags:
        # Create a new binary column for each flag
        df[f'history_has_{flag}'] = df['history'].str.contains(flag, regex=False, na=False).astype(np.int8)
        print(f"  - Created 'history_has_{flag}' feature.")

    # Drop the original history column after extracting flags
    df = df.drop(columns=['history'], errors='ignore')
    print("Dropped original 'history' column.")
else:
    print("History column not found for flag extraction.")


# --- New: Consolidated Label Handling (Vectorized using np.select) ---
# Create a binary 'is_malicious' label
df['is_malicious'] = df['label'].apply(lambda x: 1 if 'Malicious' in str(x) else 0).astype(np.int8)
print("Created 'is_malicious' binary label.")

# Clean label columns for consistent string comparison
df['detailed_label_clean'] = df['detailed-label'].astype(str).str.strip()
df['label_clean'] = df['label'].astype(str).str.strip()

# Define conditions and corresponding choices for 'attack_type' using np.select
conditions = [
    # 1. Specific detailed-labels (highest priority)
    (df['detailed_label_clean'] == 'PartOfAHorizontalPortScan'),
    (df['detailed_label_clean'] == 'C&C'),
    (df['detailed_label_clean'] == 'Attack'),
    (df['detailed_label_clean'] == 'HeartBeat'),
    (df['detailed_label_clean'] == 'Torii'),
    (df['detailed_label_clean'] == 'FileDownload'),

    # 2. Specific malicious types from 'label' (if not already caught by detailed-label)
    # These conditions are now independent of detailed_label_clean being generic,
    # relying on np.select's order to prioritize detailed_label_clean first.
    (df['label_clean'].str.contains('DDoS', na=False)),
    (df['label_clean'].str.contains('PartOfAHorizontalPortScan', na=False)), # Redundant if caught by detailed-label, but harmless due to order
    (df['label_clean'].str.contains('C&C', na=False)), # Redundant if caught by detailed-label, but harmless due to order
    (df['label_clean'].str.contains('Attack', na=False)), # Redundant if caught by detailed-label, but harmless due to order
    (df['label_clean'].str.contains('FileDownload', na=False)), # Redundant if caught by detailed-label, but harmless due to order

    # 3. Generic 'Malicious' label (if not caught by any specific malicious type)
    (df['label_clean'].str.contains('Malicious', na=False))
]

choices = [
    'PartOfAHorizontalPortScan',
    'C&C',
    'Attack',
    'HeartBeat',
    'Torii',
    'FileDownload',
    'DDoS',
    'PartOfAHorizontalPortScan',
    'C&C',
    'Attack',
    'FileDownload',
    'General_Malware'
]

# Use np.select to apply these conditions. Default will be 'Benign_Traffic'
# for anything not caught by the malicious conditions.
df['attack_type'] = np.select(conditions, choices, default='Benign_Traffic')

# Drop the temporary clean columns
df = df.drop(columns=['detailed_label_clean', 'label_clean'], errors='ignore')

# Convert to category at the very end for memory efficiency
df['attack_type'] = df['attack_type'].astype('category')
print("Created 'attack_type' consolidated multi-class label (vectorized).")

# Drop original 'label' and 'detailed-label' columns as new consolidated ones are created
df = df.drop(columns=['label', 'detailed-label'], errors='ignore')
print("Dropped original 'label' and 'detailed-label' columns.")


# 5. Feature Scaling (MinMaxScaler)
features_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()
features_to_scale = [col for col in features_to_scale if col not in ['is_malicious']]

if features_to_scale:
    scaler = MinMaxScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale]).astype(np.float32)
    print(f"Applied MinMaxScaler to {len(features_to_scale)} numerical features.")
else:
    print("No numerical features found for scaling.")

# Final check of the DataFrame after Phase 2
print("\n--- Data Information After Phase 2 (Feature Engineering & Label Consolidation) ---")
df.info(memory_usage='deep')
print("\n--- Sample of Data After Phase 2 ---")
print(df.head())
print("\n--- Value Counts for New Labels ---")
print("is_malicious:\n", df['is_malicious'].value_counts())
print("\nattack_type:\n", df['attack_type'].value_counts())


# --- Output the processed DataFrame to a CSV file ---
output_filepath = "/kaggle/working/processed_network_traffic.csv"
df.to_csv(output_filepath, index=False)
print(f"\nProcessed data saved to: {output_filepath}")

print("\nPhase 2: Feature Engineering and Selection Completed.")



!rm -rf /kaggle/working/*


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

print("Starting Phase 3: Dataset Splitting and Preparation for Model Training...\n")

# --- Load the processed data from Phase 2 ---
input_filepath = "/kaggle/working/processed_network_traffic.csv"
try:
    # Read with appropriate dtypes to save memory, especially for boolean and category columns
    # We need to infer dtypes for the OHE columns, so let's load and then downcast/assign categories
    df = pd.read_csv(input_filepath, low_memory=False)
    print(f"Successfully loaded processed data from {input_filepath} with {len(df)} rows.")

    # Re-apply category and int8 dtypes where appropriate after loading from CSV
    # CSVs don't preserve category dtypes directly, so re-convert
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to category if cardinality is low
            if df[col].nunique() / len(df) < 0.5: # Heuristic for low cardinality
                df[col] = df[col].astype('category')
        elif df[col].dtype == 'bool':
            # Bool columns from get_dummies might be loaded as bool, convert to int8 for consistency
            df[col] = df[col].astype(np.int8)
        elif df[col].dtype == np.float64:
            # Downcast float64 to float32 if possible
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == np.int64:
            # Downcast int64 to int32 if possible
            df[col] = df[col].astype(np.int32)
    print("Re-assigned optimized dtypes after loading CSV.")

except FileNotFoundError:
    print(f"Error: Processed data file not found at {input_filepath}. Please ensure Phase 1 & 2 ran successfully.")
    exit()
except Exception as e:
    print(f"Error loading processed data: {e}")
    exit()

print("\n--- Data Information After Loading for Phase 3 ---")
df.info(memory_usage='deep')
print("\n--- Value Counts for Labels ---")
print("is_malicious:\n", df['is_malicious'].value_counts())
print("\nattack_type:\n", df['attack_type'].value_counts())

# Define target columns
IS_MALICIOUS_COL = 'is_malicious'
ATTACK_TYPE_COL = 'attack_type'

# Separate features (X) from targets (y)
# Exclude both target columns from features for all splits
feature_columns = [col for col in df.columns if col not in [IS_MALICIOUS_COL, ATTACK_TYPE_COL]]
X = df[feature_columns]
y_is_malicious = df[IS_MALICIOUS_COL]
y_attack_type = df[ATTACK_TYPE_COL]

# Set a random state for reproducibility
RANDOM_STATE = 42
OUTPUT_DIR = "/kaggle/working/"
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

# --- 1. Dataset Splitting for Deep Learning Anomaly Detector ---
print("\n--- Splitting data for Anomaly Detector (AD) ---")
# The AD is trained only on Benign traffic.
# Its test set will contain both Benign and Malicious traffic for evaluation.

# Separate Benign and Malicious traffic
benign_df = df[df[IS_MALICIOUS_COL] == 0].copy()
malicious_df = df[df[IS_MALICIOUS_COL] == 1].copy()

print(f"Benign samples: {len(benign_df)}")
print(f"Malicious samples: {len(malicious_df)}")

# Split Benign data for AD training and a portion for AD testing
# AD_TRAIN_SIZE: Portion of benign data used to train the AD (e.g., 80%)
# AD_TEST_BENIGN_SIZE: Portion of benign data used in the AD test set (e.g., 20%)
# Note: The AD test set will also include ALL malicious data.
AD_TRAIN_RATIO = 0.8
AD_TEST_BENIGN_RATIO = 0.2 # This is the ratio of the original benign_df

# Split benign data into training for AD and a portion for the AD test set
X_ad_train_benign, X_ad_test_benign, _, _ = train_test_split(
    benign_df[feature_columns], benign_df[IS_MALICIOUS_COL],
    test_size=AD_TEST_BENIGN_RATIO, random_state=RANDOM_STATE, stratify=benign_df[IS_MALICIOUS_COL] # Stratify on is_malicious (all 0s, so it's consistent)
)
# The anomaly detector's training data consists only of benign traffic features
ad_train_data = X_ad_train_benign
print(f"Anomaly Detector Training Data (Benign only): {len(ad_train_data)} samples.")

# The full test set for the anomaly detector combines the held-out benign data and all malicious data
ad_test_data = pd.concat([X_ad_test_benign, malicious_df[feature_columns]], ignore_index=True)
ad_test_labels = pd.concat([pd.Series(0, index=X_ad_test_benign.index), malicious_df[IS_MALICIOUS_COL]], ignore_index=True)
print(f"Anomaly Detector Test Data (Benign + Malicious): {len(ad_test_data)} samples.")
print(f"  - Benign in AD Test: {len(X_ad_test_benign)}")
print(f"  - Malicious in AD Test: {len(malicious_df)}")


# Save AD datasets
ad_train_data.to_csv(os.path.join(OUTPUT_DIR, 'ad_train_benign.csv'), index=False)
ad_test_data.to_csv(os.path.join(OUTPUT_DIR, 'ad_test_features.csv'), index=False)
ad_test_labels.to_csv(os.path.join(OUTPUT_DIR, 'ad_test_labels.csv'), index=False)
print(f"Saved AD training and test data to {OUTPUT_DIR}")


# --- 2. Dataset Splitting for Supervised Classifiers (KNN/Random Forest) ---
print("\n--- Splitting data for Supervised Classifiers (CLF) ---")
# This split uses the full dataset with the 'attack_type' as target.
# It's a standard train-validation-test split.

# First, split into training + validation set and a final test set
# Stratify by 'attack_type' to maintain class distribution, especially for minority classes
X_train_val, X_test_clf, y_train_val_clf, y_test_clf = train_test_split(
    X, y_attack_type, test_size=0.2, random_state=RANDOM_STATE, stratify=y_attack_type
)
print(f"Classifier Train+Val samples: {len(X_train_val)}")
print(f"Classifier Test samples: {len(X_test_clf)}")

# Then, split the training + validation set into distinct training and validation sets
# Stratify again by 'attack_type'
X_train_clf, X_val_clf, y_train_clf, y_val_clf = train_test_split(
    X_train_val, y_train_val_clf, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val_clf
) # 0.25 of 0.8 is 0.2, so 60% train, 20% val, 20% test

print(f"Classifier Training samples: {len(X_train_clf)}")
print(f"Classifier Validation samples: {len(X_val_clf)}")

# Save CLF datasets
X_train_clf.to_csv(os.path.join(OUTPUT_DIR, 'clf_train_features.csv'), index=False)
y_train_clf.to_csv(os.path.join(OUTPUT_DIR, 'clf_train_labels.csv'), index=False)

X_val_clf.to_csv(os.path.join(OUTPUT_DIR, 'clf_val_features.csv'), index=False)
y_val_clf.to_csv(os.path.join(OUTPUT_DIR, 'clf_val_labels.csv'), index=False)

X_test_clf.to_csv(os.path.join(OUTPUT_DIR, 'clf_test_features.csv'), index=False)
y_test_clf.to_csv(os.path.join(OUTPUT_DIR, 'clf_test_labels.csv'), index=False)
print(f"Saved CLF training, validation, and test data to {OUTPUT_DIR}")

print("\nPhase 3: Dataset Splitting and Preparation for Model Training Completed.")



import os

# Specify the name of the CSV file you want to delete
file_to_delete = "knn_model.joblib"  # Replace with the actual name of your CSV file

# Construct the full path to the file
file_path = f"/kaggle/working/{file_to_delete}"

# Check if the file exists before attempting to delete it
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"File '{file_to_delete}' deleted successfully from /kaggle/working.")
else:
    print(f"File '{file_to_delete}' not found in /kaggle/working.")

# You can optionally list the files in the directory to verify the deletion
print("\nFiles remaining in /kaggle/working:")
print(os.listdir("/kaggle/working"))


import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler # Needed if we save/load scaler separately
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib # For saving scikit-learn models and scalers
import matplotlib.pyplot as plt # For plotting ROC/PR curves (optional, but good for visualization)
import seaborn as sns # For confusion matrix visualization (optional)

print("Starting Phase 4: Model Implementation and Training...\n")

# Define output directory
OUTPUT_DIR = "/kaggle/working/"
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

# --- Helper function to re-apply dtypes after loading CSVs ---
def reapply_dtypes(df_loaded):
    """Re-applies optimized dtypes to a DataFrame loaded from CSV."""
    for col in df_loaded.columns:
        if df_loaded[col].dtype == 'object':
            if df_loaded[col].nunique() / len(df_loaded) < 0.5:
                df_loaded[col] = df_loaded[col].astype('category')
        elif df_loaded[col].dtype == 'bool':
            df_loaded[col] = df_loaded[col].astype(np.int8)
        elif df_loaded[col].dtype == np.float64:
            df_loaded[col] = df_loaded[col].astype(np.float32)
        elif df_loaded[col].dtype == np.int64:
            df_loaded[col] = df_loaded[col].astype(np.int32)
    return df_loaded

# --- Load Data Splits from Phase 3 ---
print("Loading data splits from Phase 3...")

# Anomaly Detector (AD) Data
try:
    ad_train_benign = pd.read_csv(os.path.join(OUTPUT_DIR, 'ad_train_benign.csv'), low_memory=False)
    ad_train_benign = reapply_dtypes(ad_train_benign)
    print(f"Loaded AD training (benign): {len(ad_train_benign)} samples.")

    ad_test_features = pd.read_csv(os.path.join(OUTPUT_DIR, 'ad_test_features.csv'), low_memory=False)
    ad_test_features = reapply_dtypes(ad_test_features)
    ad_test_labels = pd.read_csv(os.path.join(OUTPUT_DIR, 'ad_test_labels.csv'), low_memory=False).squeeze() # Squeeze to Series
    ad_test_labels = reapply_dtypes(pd.DataFrame(ad_test_labels)).squeeze() # Reapply dtypes to Series
    print(f"Loaded AD test features: {len(ad_test_features)} samples.")
    print(f"Loaded AD test labels: {len(ad_test_labels)} samples.")

except FileNotFoundError as e:
    print(f"Error loading AD data: {e}. Ensure Phase 3 ran correctly.")
    exit()

# Classifier (CLF) Data
try:
    clf_train_features = pd.read_csv(os.path.join(OUTPUT_DIR, 'clf_train_features.csv'), low_memory=False)
    clf_train_features = reapply_dtypes(clf_train_features)
    clf_train_labels = pd.read_csv(os.path.join(OUTPUT_DIR, 'clf_train_labels.csv'), low_memory=False).squeeze()
    clf_train_labels = reapply_dtypes(pd.DataFrame(clf_train_labels)).squeeze()
    print(f"Loaded CLF training features: {len(clf_train_features)} samples.")
    print(f"Loaded CLF training labels: {len(clf_train_labels)} samples.")

    clf_val_features = pd.read_csv(os.path.join(OUTPUT_DIR, 'clf_val_features.csv'), low_memory=False)
    clf_val_features = reapply_dtypes(clf_val_features)
    clf_val_labels = pd.read_csv(os.path.join(OUTPUT_DIR, 'clf_val_labels.csv'), low_memory=False).squeeze()
    clf_val_labels = reapply_dtypes(pd.DataFrame(clf_val_labels)).squeeze()
    print(f"Loaded CLF validation features: {len(clf_val_features)} samples.")
    print(f"Loaded CLF validation labels: {len(clf_val_labels)} samples.")

    clf_test_features = pd.read_csv(os.path.join(OUTPUT_DIR, 'clf_test_features.csv'), low_memory=False)
    clf_test_features = reapply_dtypes(clf_test_features)
    clf_test_labels = pd.read_csv(os.path.join(OUTPUT_DIR, 'clf_test_labels.csv'), low_memory=False).squeeze()
    clf_test_labels = reapply_dtypes(pd.DataFrame(clf_test_labels)).squeeze()
    print(f"Loaded CLF test features: {len(clf_test_features)} samples.")
    print(f"Loaded CLF test labels: {len(clf_test_labels)} samples.")

except FileNotFoundError as e:
    print(f"Error loading CLF data: {e}. Ensure Phase 3 ran correctly.")
    exit()


# --- Configure TensorFlow for GPU/CPU (to avoid TPU issues) ---
# Check for GPU availability and set up strategy
strategy = tf.distribute.get_strategy() # Default strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # Detect TPU
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU.")
except ValueError:
    # No TPU found, check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            strategy = tf.distribute.MirroredStrategy() # For multi-GPU, or single GPU
            print(f"Running on GPU(s): {len(gpus)}")
        except RuntimeError as e:
            print(f"Error setting up GPU: {e}")
            print("Falling back to CPU.")
            strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
    else:
        print("No GPU or TPU found. Running on CPU.")
        strategy = tf.distribute.OneDeviceStrategy("/cpu:0")


# --- 1. Anomaly Detection Model (Deep Learning - Autoencoder) ---
print("\n--- Training Anomaly Detector (Autoencoder) ---")

input_dim = ad_train_benign.shape[1]
encoding_dim = int(input_dim / 2) # Example: half the input dimension
hidden_dim = int(encoding_dim / 2) # Further compression

# Build and compile the Autoencoder within the strategy scope
with strategy.scope():
    # Input Layer
    input_layer = Input(shape=(input_dim,))

    # Encoder
    encoder = Dense(encoding_dim, activation="relu")(input_layer)
    encoder = Dense(hidden_dim, activation="relu")(encoder) # Bottleneck layer

    # Decoder
    decoder = Dense(encoding_dim, activation="relu")(encoder)
    decoder = Dense(input_dim, activation="sigmoid")(decoder) # Sigmoid for scaled [0,1] data

    # Autoencoder Model
    autoencoder = Model(inputs=input_layer, outputs=decoder)

    # Compile the Autoencoder
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

print("Autoencoder model summary:")
autoencoder.summary()

# Convert Pandas DataFrames to NumPy arrays for Keras fit, especially with strategies
ad_train_benign_np = ad_train_benign.values.astype(np.float32)
ad_test_features_np = ad_test_features.values.astype(np.float32)

# Train the Autoencoder on benign data
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = autoencoder.fit(
    ad_train_benign_np, ad_train_benign_np, # Input and target are the same for autoencoder
    epochs=50,
    batch_size=256, # Adjust batch size based on GPU/CPU memory
    shuffle=True,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

print("\nAutoencoder training complete.")

# Save the trained Autoencoder model
autoencoder.save(os.path.join(OUTPUT_DIR, 'autoencoder_model.h5'))
print(f"Saved Autoencoder model to {os.path.join(OUTPUT_DIR, 'autoencoder_model.h5')}")

# --- Evaluate Autoencoder and Determine Anomaly Threshold ---
print("\n--- Evaluating Autoencoder and determining anomaly threshold ---")

# Get reconstruction errors on the full AD test set
ad_test_predictions = autoencoder.predict(ad_test_features_np)
mse = np.mean(np.power(ad_test_features_np - ad_test_predictions, 2), axis=1)

# Separate errors for benign and malicious samples in the test set
benign_errors = mse[ad_test_labels == 0]
malicious_errors = mse[ad_test_labels == 1]

# Determine a threshold for anomaly detection
threshold = np.percentile(benign_errors, 95)
print(f"Calculated anomaly threshold (95th percentile of benign errors): {threshold}")

# Classify test samples as anomalous (1) or normal (0) based on the threshold
ad_test_predictions_binary = (mse > threshold).astype(int)

# Evaluate the anomaly detector's performance
ad_report = classification_report(ad_test_labels, ad_test_predictions_binary, target_names=['Benign', 'Malicious'])
print("\nAnomaly Detector Classification Report (Benign vs. Malicious):")
print(ad_report)

ad_conf_matrix = confusion_matrix(ad_test_labels, ad_test_predictions_binary)
print("\nAnomaly Detector Confusion Matrix:")
print(ad_conf_matrix)

# Calculate AUC-ROC score
roc_auc = roc_auc_score(ad_test_labels, mse)
print(f"\nAnomaly Detector AUC-ROC Score: {roc_auc:.4f}")

# Save anomaly detector evaluation report
with open(os.path.join(OUTPUT_DIR, 'autoencoder_ad_report.txt'), 'w') as f:
    f.write("Anomaly Detector Classification Report (Benign vs. Malicious):\n")
    f.write(ad_report)
    f.write("\nAnomaly Detector Confusion Matrix:\n")
    f.write(str(ad_conf_matrix))
    f.write(f"\nAnomaly Detector AUC-ROC Score: {roc_auc:.4f}\n")
    f.write(f"Calculated Anomaly Threshold: {threshold}\n")
print(f"Saved Autoencoder AD report to {os.path.join(OUTPUT_DIR, 'autoencoder_ad_report.txt')}")


# --- 2. Supervised Classifiers (KNN and Random Forest) ---
print("\n--- Training Supervised Classifiers ---")

# Convert Pandas DataFrames to NumPy arrays for scikit-learn models if needed,
# though scikit-learn generally handles DataFrames well.
# We'll stick to DataFrames for scikit-learn as they are usually fine.
X_train_clf_data = clf_train_features
y_train_clf_data = clf_train_labels

X_val_clf_data = clf_val_features
y_val_clf_data = clf_val_labels

X_test_clf_data = clf_test_features
y_test_clf_data = clf_test_labels


# --- K-Nearest Neighbors (KNN) Classifier ---
print("\n--- Training K-Nearest Neighbors (KNN) Classifier ---")
knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

# Train KNN
knn_model.fit(X_train_clf_data, y_train_clf_data)
print("KNN training complete.")

# Save the trained KNN model
joblib.dump(knn_model, os.path.join(OUTPUT_DIR, 'knn_model.joblib'))
print(f"Saved KNN model to {os.path.join(OUTPUT_DIR, 'knn_model.joblib')}")

# Evaluate KNN on Validation Set
print("\nEvaluating KNN on Validation Set:")
knn_val_predictions = knn_model.predict(X_val_clf_data)
knn_val_report = classification_report(y_val_clf_data, knn_val_predictions)
print(knn_val_report)

# Evaluate KNN on Test Set
print("\nEvaluating KNN on Test Set:")
knn_test_predictions = knn_model.predict(X_test_clf_data)
knn_test_report = classification_report(y_test_clf_data, knn_test_predictions)
print(knn_test_report)

# Save KNN evaluation report
with open(os.path.join(OUTPUT_DIR, 'knn_clf_report.txt'), 'w') as f:
    f.write("KNN Classification Report (Validation Set):\n")
    f.write(knn_val_report)
    f.write("\nKNN Classification Report (Test Set):\n")
    f.write(knn_test_report)
print(f"Saved KNN report to {os.path.join(OUTPUT_DIR, 'knn_clf_report.txt')}")


# --- Random Forest Classifier ---
print("\n--- Training Random Forest Classifier ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)

# Train Random Forest
rf_model.fit(X_train_clf_data, y_train_clf_data)
print("Random Forest training complete.")

# Save the trained Random Forest model
joblib.dump(rf_model, os.path.join(OUTPUT_DIR, 'random_forest_model.joblib'))
print(f"Saved Random Forest model to {os.path.join(OUTPUT_DIR, 'random_forest_model.joblib')}")

# Evaluate Random Forest on Validation Set
print("\nEvaluating Random Forest on Validation Set:")
rf_val_predictions = rf_model.predict(X_val_clf_data)
rf_val_report = classification_report(y_val_clf_data, rf_val_predictions)
print(rf_val_report)

# Evaluate Random Forest on Test Set
print("\nEvaluating Random Forest on Test Set:")
rf_test_predictions = rf_model.predict(X_test_clf_data)
rf_test_report = classification_report(y_test_clf_data, rf_test_predictions)
print(rf_test_report)

# Save Random Forest evaluation report
with open(os.path.join(OUTPUT_DIR, 'random_forest_clf_report.txt'), 'w') as f:
    f.write("Random Forest Classification Report (Validation Set):\n")
    f.write(rf_val_report)
    f.write("\nRandom Forest Classification Report (Test Set):\n")
    f.write(rf_test_report)
print(f"Saved Random Forest report to {os.path.join(OUTPUT_DIR, 'random_forest_clf_report.txt')}")

print("\nPhase 4: Model Implementation and Training Completed.")



import os
import json
import subprocess
import shutil # Added for file operations

# --- Configuration ---
# IMPORTANT: Replace 'your-kaggle-username' with your actual Kaggle username.
# You can find your username on your Kaggle profile page.
KAGGLE_USERNAME = "edafemoses" # Updated username

# IMPORTANT: Choose a unique and descriptive ID for your dataset.
# This will be part of the dataset URL (e.g., kaggle.com/your-username/my-new-dataset-id)
# Use lowercase letters, numbers, and hyphens only.
DATASET_ID = "processed-network-traffic-data"

# Title that will appear on Kaggle for your dataset
DATASET_TITLE = "Processed Network Traffic Data"

# Description for your dataset
DATASET_DESCRIPTION = "This dataset contains processed network traffic data generated from a Kaggle notebook."

# Path to your CSV file in the current notebook's working directory
CSV_FILE_NAME = "processed_network_traffic.csv"
SOURCE_FILE_PATH = f"/kaggle/working/{CSV_FILE_NAME}"

# --- Kaggle API Key Setup ---
# This section is added to handle the kaggle.json file from your uploaded dataset.

KAGGLE_API_KEY_DATASET_PATH = "/kaggle/input/kaggle-api-key/"
KAGGLE_JSON_FILE_NAME = "kaggle (1).json" # As per your description

# Define the target directory for kaggle.json (standard Kaggle API location)
KAGGLE_CONFIG_DIR = os.path.expanduser("~/.kaggle") # Resolves to /root/.kaggle/ in Kaggle env

print("--- Setting up Kaggle API Key ---")
# Create the .kaggle directory if it doesn't exist
os.makedirs(KAGGLE_CONFIG_DIR, exist_ok=True)
print(f"Ensured Kaggle config directory exists: {KAGGLE_CONFIG_DIR}")

# Source path of the kaggle.json file within the uploaded dataset
source_kaggle_json_path = os.path.join(KAGGLE_API_KEY_DATASET_PATH, KAGGLE_JSON_FILE_NAME)
# Destination path for the kaggle.json file
destination_kaggle_json_path = os.path.join(KAGGLE_CONFIG_DIR, "kaggle.json") # Rename to kaggle.json

if os.path.exists(source_kaggle_json_path):
    # Copy the kaggle.json file
    shutil.copy(source_kaggle_json_path, destination_kaggle_json_path)
    print(f"Copied '{source_kaggle_json_path}' to '{destination_kaggle_json_path}'")

    # Set appropriate permissions (read/write for owner only)
    os.chmod(destination_kaggle_json_path, 0o600)
    print(f"Set permissions for '{destination_kaggle_json_path}' to 600 (owner read/write).")
else:
    print(f"Error: '{KAGGLE_JSON_FILE_NAME}' not found in '{KAGGLE_API_KEY_DATASET_PATH}'.")
    print("Please ensure your 'kaggle-api-key' dataset is added to the notebook and contains the file.")
    # Exit or raise an error if the key isn't found, as subsequent steps will fail
    exit("Kaggle API key not found. Aborting dataset creation.")

print("Kaggle API Key setup complete.\n")

# --- Prepare Dataset Directory and Metadata ---

# Create a temporary directory where the dataset files will be staged
# This directory will contain the CSV and the metadata file.
DATASET_STAGING_DIR = f"/kaggle/working/{DATASET_ID}"
os.makedirs(DATASET_STAGING_DIR, exist_ok=True)
print(f"Created staging directory: {DATASET_STAGING_DIR}")

# Define the metadata for your new Kaggle dataset
# This JSON structure is required by the Kaggle API
dataset_metadata = {
    "title": DATASET_TITLE,
    "id": f"{KAGGLE_USERNAME}/{DATASET_ID}",
    "licenses": [{"name": "CC0-1.0"}], # Common license for public domain data
    "resources": [
        {
            "path": CSV_FILE_NAME,
            "description": DATASET_DESCRIPTION
        }
    ]
}

# Write the metadata to a JSON file inside the staging directory
METADATA_FILE_PATH = os.path.join(DATASET_STAGING_DIR, "dataset-metadata.json")
with open(METADATA_FILE_PATH, "w") as f:
    json.dump(dataset_metadata, f, indent=4)
print(f"Generated dataset metadata file: {METADATA_FILE_PATH}")

# Move the processed CSV file into the staging directory
# This is crucial so that the Kaggle API picks it up
DESTINATION_FILE_PATH = os.path.join(DATASET_STAGING_DIR, CSV_FILE_NAME)
if os.path.exists(SOURCE_FILE_PATH):
    os.rename(SOURCE_FILE_PATH, DESTINATION_FILE_PATH)
    print(f"Moved '{SOURCE_FILE_PATH}' to '{DESTINATION_FILE_PATH}'")
else:
    print(f"Error: Source file '{SOURCE_FILE_PATH}' not found. Please ensure it exists.")
    exit("Processed CSV file not found. Aborting dataset creation.") # Exit if CSV is missing

# --- Use Kaggle API to Create/Update the Dataset ---

print("\nAttempting to create/update Kaggle dataset...")
print(f"Dataset will be located at: https://www.kaggle.com/{KAGGLE_USERNAME}/{DATASET_ID}")

# Command to create a new dataset
create_command = f"kaggle datasets create -p {DATASET_STAGING_DIR} -r zip --dir-mode skip"

# Command to update an existing dataset (create a new version)
update_command = f"kaggle datasets version -p {DATASET_STAGING_DIR} -m 'Update with latest processed data' --dir-mode skip"

try:
    # Try to create the dataset first
    print(f"Executing command: {create_command}")
    result = subprocess.run(create_command, shell=True, capture_output=True, text=True, check=True)
    print("Kaggle API Output (Create):")
    print(result.stdout)
    if result.stderr:
        print("Kaggle API Error (Create):")
        print(result.stderr)
    print("\nDataset creation initiated. Check your Kaggle profile for status.")

except subprocess.CalledProcessError as e:
    # If creation fails, it might be because the dataset already exists.
    # In that case, we should use the 'version' command.
    print(f"\nDataset creation failed (Error: {e.returncode}). This might mean the dataset already exists.")
    print("Trying to update (create a new version) instead...")
    print(f"Executing command: {update_command}")
    try:
        result = subprocess.run(update_command, shell=True, capture_output=True, text=True, check=True)
        print("Kaggle API Output (Update):")
        print(result.stdout)
        if result.stderr:
            print("Kaggle API Error (Update):")
            print(result.stderr)
        print("\nDataset update initiated. Check your Kaggle profile for status.")
    except subprocess.CalledProcessError as e_update:
        print(f"\nFailed to update dataset as well (Error: {e_update.returncode}).")
        print("Please check the error message above and ensure:")
        print("1. Your Kaggle API token is correctly configured (which this script attempts to do).")
        print("2. The DATASET_ID is unique if creating, or matches an existing dataset you own if updating.")
        print("3. The CSV file exists at the specified source path.")
        print("Kaggle API Error (Update):")
        print(e_update.stderr)
    except Exception as e_gen:
        print(f"An unexpected error occurred during dataset update: {e_gen}")
except Exception as e:
    print(f"An unexpected error occurred during dataset creation: {e}")

print("\n--- Script Finished ---")
print("You can verify the dataset on Kaggle at:")
print(f"https://www.kaggle.com/{KAGGLE_USERNAME}/{DATASET_ID}")



import json
import numpy as np
import hdbscan
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from stable_baselines3 import PPO
from stable_baselines3.common.envs import SimpleMultiObsEnv
from joblib import Parallel, delayed
import gym
from gym import spaces
from collections import deque, Counter
import itertools
import copy

# --- Global Debug Flags ---
DEBUG_MODE = True
DEBUG_INFERENCE = True
DEBUG_GRID_PRINT = True
DEBUG_OBJECT_DETECTION = False

# --- Data Loading and Utility Functions ---
def load_json_file(filepath):
    """Loads a JSON file from the given filepath."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        if DEBUG_MODE:
            print(f"Successfully loaded: {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None

def print_grid(grid, title="Grid"):
    """Prints a grid in a readable format, handling large grids."""
    if not DEBUG_GRID_PRINT:
        return
    print(f"\n--- {title} ---")
    if not grid or not grid[0]:
        print("[Empty or Malformed Grid]")
        return
    max_rows, max_cols = 10, 10
    h, w = len(grid), len(grid[0])
    if h > max_rows or w > max_cols:
        print(f"Grid (dims: {h}x{w}), showing top-left {min(h, max_rows)}x{min(w, max_cols)}:")
        for r in range(min(h, max_rows)):
            print(' '.join(map(str, grid[r][:min(w, max_cols)])))
    else:
        for row in grid:
            print(' '.join(map(str, row)))
    print("-" * (len(title) + 8))

def get_grid_dimensions(grid):
    """Returns (height, width) of a grid."""
    if not grid or not isinstance(grid, list) or not grid[0] or not isinstance(grid[0], list):
        return (0, 0)
    return (len(grid), len(grid[0]))

def get_unique_colors(grid):
    """Returns a sorted list of unique colors (integers) present in a grid."""
    if not grid or not grid[0]:
        return []
    return sorted(list(set(itertools.chain.from_iterable(grid))))

def grids_equal(grid1, grid2):
    """Checks if two grids are identical."""
    return np.array_equal(np.array(grid1, dtype=int), np.array(grid2, dtype=int))

def pad_grid(grid, max_size=30, pad_value=0):
    """Pads a grid to max_size x max_size with pad_value."""
    h, w = get_grid_dimensions(grid)
    if h == 0 or w == 0:
        return [[pad_value] * max_size for _ in range(max_size)]
    padded = np.full((max_size, max_size), pad_value, dtype=int)
    padded[:min(h, max_size), :min(w, max_size)] = np.array(grid)[:min(h, max_size), :min(w, max_size)]
    return padded.tolist()

# --- Transformer Model for Pattern Recognition ---
class GridTransformer(nn.Module):
    def __init__(self, grid_size=30, d_visual_size=30, d_model=128, nhead=8, num_layers=4):
        super(GridTransformer, self).__init__()
        self.grid_size = grid_size
        self.d_model = d_model
        self.embedding = nn.Embedding(10, d_model)  # Assuming values 0-9
        self.pos_encoding = self.generate_pos_encoding(grid_size, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, 256)

    def generate_pos_encoding(self, grid_size, d_model):
        position = torch.arange(grid_size * grid_size).reshape(grid_size, grid_size)
        pos_encoding = torch.zeros(grid_size, grid_size, d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pos_encoding[:, :, 0::2] = torch.sin(position.unsqueeze(-1) * div_term)
        pos_encoding[:, :, 1::2] = torch.cos(position.unsqueeze(-1) * div_term)
        return pos_encoding

    def forward(self, x):
        batch, h, w = x.shape
        # Pad or truncate to grid_size
        if h != self.grid_size or w != self.grid_size:
            x_np = x.cpu().numpy()
            x_padded = np.array([pad_grid(x_np[i], self.grid_size) for i in range(batch)])
            x = torch.tensor(x_padded, device=x.device, dtype=x.dtype)
            h, w = self.grid_size, self.grid_size
        x = self.embedding(x.long()).reshape(batch, h * w, self.d_model)
        x += self.pos_encoding[:h, :w, :].reshape(h * w, self.d_model).unsqueeze(0)
        x = self.transformer_encoder(x)
        x = self.fc(x.mean(dim=1))
        return x

# --- RL Environment for Transformation Rule Selection ---
class ARCPatternEnv(SimpleMultiObsEnv):
    def __init__(self, train_data):
        super().__init__()
        self.train_data = train_data
        self.action_space = spaces.Discrete(10)  # 10 transformation rules
        self.observation_space = spaces.Box(low=0, high=255, shape=(256,), dtype=np.float32)
        self.transformations = [
            self.apply_expansion_00576224,
            self.apply_tiling_007bbfb7,
            self.apply_value_replace_009d5c81,
            self.apply_neighbor_rule_00d62c1b,
            self.apply_block_rule_42918530,
            lambda x: x,  # Identity
            lambda x: rotate_grid(x, 1),
            lambda x: rotate_grid(x, 3),
            lambda x: flip_grid(x, 1),
            lambda x: apply_gravity(x, 'down')
        ]
        self.current_challenge = None
        self.current_input = None

    def reset(self):
        challenge_id = np.random.choice(list(self.train_data.keys()))
        self.current_challenge = challenge_id
        train_pairs = self.train_data[challenge_id]['train']
        pair = train_pairs[np.random.randint(len(train_pairs))]
        self.current_input = np.array(pair['input'])
        state = self.get_state(self.current_input)
        return state

    def get_state(self, grid):
        grid_padded = np.array(pad_grid(grid, max_size=30))
        grid_tensor = torch.tensor(grid_padded, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            state = transformer_model(grid_tensor).numpy().flatten()
        return state

    def step(self, action):
        transformed = self.transformations[action](self.current_input)
        reward = self.compute_reward(transformed)
        done = True
        info = {}
        state = self.get_state(self.current_input)
        return state, reward, done, info

    def compute_reward(self, transformed):
        for pair in self.train_data[self.current_challenge]['train']:
            if np.array_equal(transformed, np.array(pair['output'])):
                return 1.0
        return -0.1

    def apply_expansion_00576224(self, grid):
        try:
            a, b = grid[0]
            c, d = grid[1]
            output = np.zeros((6, 6), dtype=int)
            output[0] = [a, b, a, b, a, b]
            output[1] = [c, d, c, d, c, d]
            output[2] = [b, a, b, a, b, a]
            output[3] = [d, c, d, c, d, c]
            output[4] = [a, b, a, b, a, b]
            output[5] = [c, d, c, d, c, d]
            return output
        except:
            return grid  # Fallback to identity if shape is incompatible

    def apply_tiling_007bbfb7(self, grid):
        try:
            output = np.zeros((9, 9), dtype=int)
            output[:3, :3] = grid[:3, :3]
            output[6:9, 6:9] = grid[:3, :3]
            output[3:6, 3:6] = grid[:3, :3]
            output[:3, 3:6] = np.rot90(grid[:3, :3], 2)
            output[6:9, 3:6] = np.rot90(grid[:3, :3], 2)
            output[3:6, :3] = np.rot90(grid[:3, :3], 1)
            output[3:6, 6:9] = np.rot90(grid[:3, :3], -1)
            return output
        except:
            return grid  # Fallback to identity if shape is incompatible

    def apply_value_replace_009d5c81(self, grid):
        try:
            ones_count = np.sum(grid == 1)
            replace_value = {3: 7, 4: 3, 5: 2}.get(ones_count, 0)
            output = np.zeros_like(grid)
            output[grid == 8] = replace_value
            return output
        except:
            return grid  # Fallback to identity if shape is incompatible

    def apply_neighbor_rule_00d62c1b(self, grid):
        try:
            output = grid.copy()
            h, w = grid.shape
            for i in range(h):
                for j in range(w):
                    if grid[i, j] == 3:
                        neighbors = []
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                neighbors.append(grid[ni, nj])
                        if neighbors.count(3) == 2:
                            output[i, j] = 4
            return output
        except:
            return grid  # Fallback to identity if shape is incompatible

    def apply_block_rule_42918530(self, grid):
        try:
            output = grid.copy()
            h, w = grid.shape
            for i in range(0, h, 6):
                for j in range(0, w, 6):
                    block = grid[i:i+5, j:j+5]
                    if block.size == 25:
                        primary_value = np.max(block)
                        if primary_value == 0:
                            continue
                        for bi in range(5):
                            for bj in range(5):
                                if block[bi, bj] != 0:
                                    neighbors = []
                                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                        ni, nj = bi + di, bj + dj
                                        if 0 <= ni < 5 and 0 <= nj < 5:
                                            neighbors.append(block[ni, nj])
                                    if neighbors.count(0) > 0:
                                        output[i + bi, j + bj] = primary_value
            return output
        except:
            return grid  # Fallback to identity if shape is incompatible

# --- Object Detection and Feature Engineering ---
def extract_object_features(grid, coords):
    """Extracts features for a connected component."""
    if not coords:
        return None
    min_r = min(c[0] for c in coords)
    max_r = max(c[0] for c in coords)
    min_c = min(c[1] for c in coords)
    max_c = max(c[1] for c in coords)
    bbox = (min_r, min_c, max_r, max_c)
    obj_grid = [[grid[r][c] if (r, c) in coords else -1 for c in range(min_c, max_c + 1)] for r in range(min_r, max_r + 1)]
    colors_in_obj = [grid[r][c] for r, c in coords]
    color_counts = Counter(colors_in_obj)
    return {
        'coords': sorted(coords),
        'size': len(coords),
        'colors': color_counts,
        'primary_color': color_counts.most_common(1)[0][0],
        'bbox': bbox,
        'height': max_r - min_r + 1,
        'width': max_c - min_c + 1,
        'grid': obj_grid
    }

def get_objects(grid, ignore_colors=[0], diagonal_connectivity=False):
    """Identifies connected components using BFS."""
    h, w = get_grid_dimensions(grid)
    if h == 0 or w == 0:
        return []
    visited = [[False for _ in range(w)] for _ in range(h)]
    objects = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)] if not diagonal_connectivity else [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for r in range(h):
        for c in range(w):
            if not visited[r][c] and grid[r][c] not in ignore_colors:
                coords = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    curr_r, curr_c = q.popleft()
                    coords.append((curr_r, curr_c))
                    for dr, dc in directions:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] not in ignore_colors:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                if coords:
                    obj = extract_object_features(grid, coords)
                    objects.append(obj)
                    if DEBUG_OBJECT_DETECTION:
                        print(f"Detected object of size {obj['size']} with colors {obj['colors']}")
    return objects

# --- Transformation Primitives ---
def recolor_grid(grid, color_map):
    h, w = get_grid_dimensions(grid)
    new_grid = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if new_grid[r][c] in color_map:
                new_grid[r][c] = color_map[new_grid[r][c]]
    return new_grid

def rotate_grid(grid, k):
    return np.rot90(np.array(grid, dtype=int), k=-k).tolist()

def flip_grid(grid, axis):
    return np.flip(np.array(grid, dtype=int), axis=axis).tolist()

def crop_to_content(grid, background_color=0):
    coords = [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] != background_color]
    if not coords:
        return []
    min_r, min_c, max_r, max_c = extract_object_features(grid, coords)['bbox']
    return [row[min_c:max_c+1] for row in grid[min_r:max_r+1]]

def apply_gravity(grid, direction='down', background_color=0):
    h, w = get_grid_dimensions(grid)
    new_grid = [[background_color for _ in range(w)] for _ in range(h)]
    if direction == 'down':
        for c in range(w):
            column_cells = [grid[r][c] for r in range(h) if grid[r][c] != background_color]
            for i, cell_color in enumerate(column_cells):
                new_grid[h - len(column_cells) + i][c] = cell_color
    return new_grid

def apply_transformation(grid, rule):
    current_grid = copy.deepcopy(grid)
    if rule['type'] == 'composite':
        for step in rule['sequence']:
            current_grid = apply_transformation(current_grid, step)
            if current_grid is None:
                return None
        return current_grid
    try:
        rule_type = rule['type']
        params = rule.get('params', {})
        if rule_type == 'identity':
            return current_grid
        elif rule_type == 'recolor':
            return recolor_grid(current_grid, params['color_map'])
        elif rule_type == 'rotate':
            return rotate_grid(current_grid, params['k'])
        elif rule_type == 'flip':
            return flip_grid(current_grid, params['axis'])
        elif rule_type == 'crop_to_content':
            return crop_to_content(current_grid, params.get('background_color', 0))
        elif rule_type == 'apply_gravity':
            return apply_gravity(current_grid, params.get('direction', 'down'), params.get('background_color', 0))
        elif rule_type == 'custom':
            return params['function'](current_grid)
        else:
            return None
    except (KeyError, TypeError) as e:
        if DEBUG_INFERENCE:
            print(f"Error applying transformation {rule}: {e}")
        return None

# --- Rule Inference with Clustering ---
def extract_features(challenge_id, data, transformer_model):
    features = []
    train_pairs = data.get(challenge_id, {}).get('train', [])
    for pair in train_pairs:
        input_grid = pair.get('input', [])
        output_grid = pair.get('output', [])
        if not input_grid or not output_grid:
            continue
        try:
            input_grid_padded = np.array(pad_grid(input_grid, max_size=30))
            output_grid_padded = np.array(pad_grid(output_grid, max_size=30))
            input_tensor = torch.tensor(input_grid_padded, dtype=torch.float32).unsqueeze(0)
            output_tensor = torch.tensor(output_grid_padded, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                input_feat = transformer_model(input_tensor).numpy().flatten()
                output_feat = transformer_model(output_tensor).numpy().flatten()
            features.append(np.concatenate([input_feat, output_feat]))
        except Exception as e:
            if DEBUG_MODE:
                print(f"Error processing grid for challenge {challenge_id}: {e}")
            continue
    return np.array(features) if features else np.array([])

# --- Main Solver Logic ---
def solve_task(task, env, model):
    train_pairs = task.get('train', [])
    test_inputs = task.get('test', [])
    predictions = []
    for test_pair in test_inputs:
        test_input_grid = np.array(test_pair.get('input', []))
        if test_input_grid.size == 0:
            predictions.append([test_input_grid.tolist(), test_input_grid.tolist()])
            continue
        state = env.get_state(test_input_grid)
        action, _ = model.predict(state)
        try:
            predicted_grid = env.transformations[action](test_input_grid)
            attempt1 = predicted_grid.tolist()
        except Exception as e:
            if DEBUG_INFERENCE:
                print(f"Error applying transformation: {e}")
            attempt1 = test_input_grid.tolist()  # Fallback to input
        attempt2 = copy.deepcopy(test_input_grid).tolist()  # Fallback: copy input
        predictions.append([attempt1, attempt2])
    return predictions

# --- Submission and Evaluation ---
def generate_submission(challenges_data, env, model, output_path='/kaggle/working/submission.json'):
    submission = {}
    print("\n--- Generating Submission ---")
    task_ids = list(challenges_data.keys())
    for i, task_id in enumerate(task_ids):
        print(f"Processing task {i+1}/{len(task_ids)}: {task_id}")
        task_data = challenges_data[task_id]
        task_predictions = solve_task(task_data, env, model)
        submission[task_id] = [{'attempt_1': pred_pair[0], 'attempt_2': pred_pair[1]} for pred_pair in task_predictions]
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)
    print(f"\nSubmission fileGenerated and saved to {output_path}")
    return submission

def evaluate_predictions(challenges_data, solutions_data, env, model):
    print("\n--- Running Evaluation ---")
    total_tasks = 0
    correct_predictions = 0
    task_ids = list(challenges_data.keys())
    for i, task_id in enumerate(task_ids):
        print(f"Evaluating Task {i+1}/{len(task_ids)}: {task_id}", end="")
        total_tasks += 1
        task_data = challenges_data[task_id]
        predictions = solve_task(task_data, env, model)
        ground_truth_outputs = solutions_data.get(task_id, [])
        is_task_fully_correct = True
        if len(predictions) != len(ground_truth_outputs):
            is_task_fully_correct = False
        else:
            for i, pred_pair in enumerate(predictions):
                gt_output = ground_truth_outputs[i]
                if not (grids_equal(pred_pair[0], gt_output) or grids_equal(pred_pair[1], gt_output)):
                    is_task_fully_correct = False
                    break
        if is_task_fully_correct:
            correct_predictions += 1
            print(" -> Correct")
        else:
            print(" -> Incorrect")
    accuracy = (correct_predictions / total_tasks) * 100 if total_tasks > 0 else 0
    print(f"\n--- Evaluation Results ---")
    print(f"Total Evaluation Tasks: {total_tasks}")
    print(f"Correctly Solved Tasks: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

# --- Main Execution Flow ---
if __name__ == "__main__":
    DATA_PATH = '/kaggle/input/arc-prize-2025/'
    training_challenges = load_json_file(DATA_PATH + 'arc-agi_training_challenges.json')
    training_solutions = load_json_file(DATA_PATH + 'arc-agi_training_solutions.json')
    evaluation_challenges = load_json_file(DATA_PATH + 'arc-agi_evaluation_challenges.json')
    evaluation_solutions = load_json_file(DATA_PATH + 'arc-agi_evaluation_solutions.json')
    test_challenges = load_json_file(DATA_PATH + 'arc-agi_test_challenges.json')

    # Combine training and evaluation data for learning
    combined_data = {}
    if training_challenges and training_solutions:
        for cid in training_challenges:
            if cid in training_solutions:
                combined_data[cid] = training_challenges[cid]
                combined_data[cid]['train'] = [
                    {'input': pair['input'], 'output': training_solutions[cid][i]}
                    for i, pair in enumerate(training_challenges[cid]['train'])
                    if i < len(training_solutions[cid])
                ]
    if evaluation_challenges and evaluation_solutions:
        for cid in evaluation_challenges:
            if cid in evaluation_solutions:
                combined_data[cid] = evaluation_challenges[cid]
                combined_data[cid]['train'] = [
                    {'input': pair['input'], 'output': evaluation_solutions[cid][i]}
                    for i, pair in enumerate(evaluation_challenges[cid]['train'])
                    if i < len(evaluation_solutions[cid])
                ]

    # Initialize Transformer Model
    transformer_model = GridTransformer(grid_size=30, d_model=128, nhead=8, num_layers=4)

    # Cluster Training Examples
    def compute_features(cid):
        return extract_features(cid, combined_data, transformer_model)

    feature_dict = dict(zip(combined_data.keys(), Parallel(n_jobs=-1)(delayed(compute_features)(cid) for cid in combined_data.keys())))
    all_features = np.vstack([f for f in feature_dict.values() if f.size > 0])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, cluster_selection_epsilon=0.5)
    clusters = clusterer.fit_predict(all_features)

    # Initialize RL Environment and Agent
    env = ARCPatternEnv(combined_data)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10000)

    # Evaluate on Evaluation Set
    if evaluation_challenges and evaluation_solutions:
        evaluate_predictions(evaluation_challenges, evaluation_solutions, env, model)

    # Generate Submission for Test Set
    if test_challenges:
        generate_submission(test_challenges, env, model)
    else:
        print("\nTest challenges data not found. Cannot generate submission file.")


!pip install hdbscan stable_baselines3


import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib # For loading scikit-learn models

print("Starting Phase 5: Hybrid System Integration and Final Evaluation...\n")

# Define output directory
OUTPUT_DIR = "/kaggle/working/"
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

# --- Helper function to re-apply dtypes after loading CSVs ---
def reapply_dtypes(df_loaded):
    """Re-applies optimized dtypes to a DataFrame loaded from CSV."""
    for col in df_loaded.columns:
        if df_loaded[col].dtype == 'object':
            if df_loaded[col].nunique() / len(df_loaded) < 0.5:
                df_loaded[col] = df_loaded[col].astype('category')
        elif df_loaded[col].dtype == 'bool':
            df_loaded[col] = df_loaded[col].astype(np.int8)
        elif df_loaded[col].dtype == np.float64:
            df_loaded[col] = df_loaded[col].astype(np.float32)
        elif df_loaded[col].dtype == np.int64:
            df_loaded[col] = df_loaded[col].astype(np.int32)
    return df_loaded

# --- Load Trained Models from Phase 4 ---
print("Loading trained models from Phase 4...")
autoencoder = None # Initialize autoencoder to None
rf_model = None    # Initialize rf_model to None
try:
    # Added compile=False to load_model to bypass potential issues with custom objects/metrics
    autoencoder = load_model(os.path.join(OUTPUT_DIR, 'autoencoder_model.h5'), compile=False)
    print("Loaded Autoencoder model.")
    rf_model = joblib.load(os.path.join(OUTPUT_DIR, 'random_forest_model.joblib'))
    print("Loaded Random Forest model.")
except FileNotFoundError as e:
    print(f"Error loading models: {e}. Ensure Phase 4 ran correctly and models were saved.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading models: {e}")
    exit()

# Add explicit checks to ensure models were loaded
if autoencoder is None:
    print("Error: Autoencoder model was not loaded correctly. Exiting.")
    exit()
if rf_model is None:
    print("Error: Random Forest model was not loaded correctly. Exiting.")
    exit()


# --- Load the full processed data to re-create the AD test set with multi-class labels ---
# This is crucial for evaluating the hybrid system against the true multi-class attack_type.
# UPDATED PATH: Now loading from /kaggle/input/processed-network-traffic-data/
input_filepath = "/kaggle/input/processed-network-traffic-data/processed_network_traffic.csv"
df_full_processed = None # Initialize to None
try:
    df_full_processed = pd.read_csv(input_filepath, low_memory=False)
    df_full_processed = reapply_dtypes(df_full_processed)
    print(f"Successfully loaded full processed data from {input_filepath} with {len(df_full_processed)} rows.")
except FileNotFoundError:
    print(f"Error: Processed data file not found at {input_filepath}. Please ensure Phase 1 & 2 ran successfully and the file is in this input path.")
    exit() # Exit immediately if file not found
except Exception as e:
    print(f"Error loading full processed data: {e}")
    exit() # Exit immediately on other loading errors

# Add an explicit check to ensure df_full_processed was loaded
if df_full_processed is None or df_full_processed.empty:
    print("Error: df_full_processed is empty or was not loaded correctly. Exiting.")
    exit()

# Define target columns
IS_MALICIOUS_COL = 'is_malicious'
ATTACK_TYPE_COL = 'attack_type'

# Separate features (X) from targets (y)
feature_columns = [col for col in df_full_processed.columns if col not in [IS_MALICIOUS_COL, ATTACK_TYPE_COL]]
X_full = df_full_processed[feature_columns]
y_is_malicious_full = df_full_processed[IS_MALICIOUS_COL]
y_attack_type_full = df_full_processed[ATTACK_TYPE_COL]

# --- Re-create the AD Test Set with Multi-class Labels for Hybrid Evaluation ---
print("\nRe-creating AD test set with multi-class labels for hybrid system evaluation...")
# This logic mirrors the split from Phase 3 to ensure consistency
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
AD_TEST_BENIGN_RATIO = 0.2

benign_df_full = df_full_processed[df_full_processed[IS_MALICIOUS_COL] == 0].copy()
malicious_df_full = df_full_processed[df_full_processed[IS_MALICIOUS_COL] == 1].copy()

# Split benign data into training for AD (not used here, but for consistency) and a portion for the AD test set
X_ad_train_benign_temp, X_ad_test_benign_hybrid, y_ad_train_benign_temp, y_ad_test_benign_hybrid = train_test_split(
    benign_df_full[feature_columns], benign_df_full[ATTACK_TYPE_COL], # Use ATTACK_TYPE_COL here
    test_size=AD_TEST_BENIGN_RATIO, random_state=RANDOM_STATE, stratify=benign_df_full[IS_MALICIOUS_COL] # Stratify on is_malicious for benign split
)

# Combine held-out benign features/labels with all malicious features/labels
hybrid_test_features = pd.concat([X_ad_test_benign_hybrid, malicious_df_full[feature_columns]], ignore_index=True)
hybrid_test_true_labels = pd.concat([y_ad_test_benign_hybrid, malicious_df_full[ATTACK_TYPE_COL]], ignore_index=True)

print(f"Hybrid System Test Features: {len(hybrid_test_features)} samples.")
print(f"Hybrid System True Labels: {len(hybrid_test_true_labels)} samples.")
print(f"  - Benign samples in hybrid test: {len(X_ad_test_benign_hybrid)}")
print(f"  - Malicious samples in hybrid test: {len(malicious_df_full)}")


# --- Determine Anomaly Threshold (from Phase 4 report) ---
# Load the threshold saved in the autoencoder_ad_report.txt
try:
    with open(os.path.join(OUTPUT_DIR, 'autoencoder_ad_report.txt'), 'r') as f:
        for line in f:
            if "Calculated Anomaly Threshold:" in line:
                anomaly_threshold = float(line.split(":")[1].strip())
                break
    print(f"\nLoaded Anomaly Threshold from report: {anomaly_threshold}")
except FileNotFoundError:
    print("Error: autoencoder_ad_report.txt not found. Cannot load anomaly threshold.")
    print("Please manually set a threshold or re-run Phase 4.")
    # Fallback to a default if file not found, or exit
    anomaly_threshold = 1.0e-5 # A reasonable default if not found
except Exception as e:
    print(f"Error parsing anomaly threshold from report: {e}. Using default.")
    anomaly_threshold = 1.0e-5 # Fallback


# --- Implement Hybrid Detection System ---
print("\nImplementing Hybrid Detection System...")

# Convert features to numpy array for Autoencoder prediction
hybrid_test_features_np = hybrid_test_features.values.astype(np.float32)

# 1. Anomaly Detection Step (Autoencoder)
print("  - Running Autoencoder for anomaly detection...")

# Define a tf.function for prediction to explicitly control device placement
@tf.function
def predict_on_cpu(model, data):
    with tf.device('/CPU:0'):
        return model(data) # Use model(data) for direct call within tf.function

try:
    reconstructions = predict_on_cpu(autoencoder, hybrid_test_features_np)
    mse = np.mean(np.power(hybrid_test_features_np - reconstructions.numpy(), 2), axis=1) # .numpy() to convert EagerTensor
except Exception as e:
    print(f"Error during Autoencoder prediction: {e}")
    print("Attempting prediction without explicit tf.function/tf.device for debugging.")
    # Fallback if the tf.function approach also fails
    try:
        reconstructions = autoencoder.predict(hybrid_test_features_np)
        mse = np.mean(np.power(hybrid_test_features_np - reconstructions, 2), axis=1)
    except Exception as e_fallback:
        print(f"Fallback Autoencoder prediction also failed: {e_fallback}")
        print("Cannot proceed with hybrid system evaluation without Autoencoder predictions. Exiting.")
        exit()


# Classify as anomalous (1) or normal (0) based on the threshold
is_anomalous_prediction = (mse > anomaly_threshold).astype(int)
print(f"  - Classified {np.sum(is_anomalous_prediction)} samples as anomalous.")
print(f"  - Classified {len(is_anomalous_prediction) - np.sum(is_anomalous_prediction)} samples as normal.")


# 2. Supervised Classification Step (Random Forest for anomalous traffic)
print("  - Running Random Forest for multi-class classification on anomalous traffic...")

# Initialize hybrid_system_predictions with the correct categories from hybrid_test_true_labels
# This ensures that all possible attack types are recognized when assigning predictions.
hybrid_system_predictions = pd.Series('Benign_Traffic', index=hybrid_test_features.index,
                                      dtype=pd.CategoricalDtype(categories=hybrid_test_true_labels.cat.categories))


# Identify samples predicted as anomalous by the Autoencoder
anomalous_indices = hybrid_test_features[is_anomalous_prediction == 1].index

if not anomalous_indices.empty:
    X_anomalous = hybrid_test_features.loc[anomalous_indices]
    # Predict attack type for anomalous samples using Random Forest
    rf_attack_predictions = rf_model.predict(X_anomalous)
    hybrid_system_predictions.loc[anomalous_indices] = rf_attack_predictions
    print(f"  - Random Forest classified {len(anomalous_indices)} anomalous samples.")
else:
    print("  - No anomalous samples detected by Autoencoder. Random Forest not applied.")

# The dtype conversion below is now redundant because it's set during initialization
# hybrid_system_predictions = hybrid_system_predictions.astype(hybrid_test_true_labels.dtype)


# --- Final Evaluation of the Hybrid System ---
print("\n--- Evaluating Hybrid System Performance ---")
hybrid_report = classification_report(hybrid_test_true_labels, hybrid_system_predictions)
print("\nHybrid System Classification Report:")
print(hybrid_report)

# Fix: The confusion_matrix function was incorrectly passed 'hybrid_conf_matrix' as the second argument.
# It should be the predicted labels.
hybrid_conf_matrix = confusion_matrix(hybrid_test_true_labels, hybrid_system_predictions)
print("\nHybrid System Confusion Matrix:")
print(hybrid_conf_matrix)

# Save hybrid system evaluation report
with open(os.path.join(OUTPUT_DIR, 'hybrid_system_report.txt'), 'w') as f:
    f.write("Hybrid System Classification Report:\n")
    f.write(hybrid_report)
    f.write("\nHybrid System Confusion Matrix:\n")
    f.write(str(hybrid_conf_matrix))
print(f"Saved Hybrid System report to {os.path.join(OUTPUT_DIR, 'hybrid_system_report.txt')}")

print("\nPhase 5: Hybrid System Integration and Final Evaluation Completed.")


