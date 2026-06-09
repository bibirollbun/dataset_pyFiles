import os
import numpy as np
import pandas as pd
import warnings
import pickle
import gc

from scipy.stats import entropy
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# ------------------- CONFIGURATION -------------------
TEST_PATH            = "/kaggle/working/"
BASE_PATH            = "/kaggle/input/hms-harmful-brain-activity-classification"
PREPROCESSED_PATH    = "/kaggle/input/preprocessing/preprocessed/eeg"
TRAIN_LABELS_PATH    = os.path.join(BASE_PATH, "train.csv")
SUCCESS_FILE_PATH    = os.path.join(os.path.dirname(PREPROCESSED_PATH), "success.csv")

CLASSES    = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
N_CLASSES  = len(CLASSES)
BATCH_SIZE = 500

print("Starting SVM-based EEG classification (handling NaNs via median imputation)\n")

# ------------------- KL DIVERGENCE FUNCTION -------------------
def kl_divergence_numpy(y_true_onehot, y_pred_proba, epsilon=1e-7):
    """
    Compute mean KL divergence between true one-hot and predicted probability vectors.
    """
    y_pred = np.clip(y_pred_proba, epsilon, 1.0 - epsilon)
    y_true = np.clip(y_true_onehot,   epsilon, 1.0       )
    kl = np.sum(y_true * np.log(y_true / y_pred), axis=1)
    return np.mean(kl)


# ------------------- FEATURE EXTRACTION -------------------
def extract_features(eeg_data: np.ndarray) -> np.ndarray:
    """
    Given a 2D array (channels Ã— timepoints), extract per-channel
    statistical features + cross-channel statistics.
    Returns a 1D array of length 251.
    """
    features = []

    # 1) Per-channel (19 channels):
    for ch in range(eeg_data.shape[0]):
        channel_data = eeg_data[ch, :]

        # Basic stats
        features.extend([
            np.mean(channel_data),
            np.std(channel_data),
            np.var(channel_data),
            np.median(channel_data),
            np.min(channel_data),
            np.max(channel_data),
            np.percentile(channel_data, 25),
            np.percentile(channel_data, 75),
            np.sum(np.abs(channel_data)),
            np.sum(channel_data ** 2),
        ])

        # Additional measures
        features.extend([
            np.abs(np.mean(channel_data)),
            np.sqrt(np.mean(channel_data ** 2)),
            entropy(np.abs(channel_data) + 1e-10),
        ])

    # 2) Cross-channel features (global)
    features.extend([
        np.mean(eeg_data),
        np.std(eeg_data),
        np.var(eeg_data),
        np.corrcoef(eeg_data).mean(),
    ])

    return np.array(features, dtype=np.float32)


def process_eeg_file(eeg_id: str) -> np.ndarray:
    """
    Load the preprocessed .npy file, handle NaNs/Infs, zâ€�normalize each channel,
    then return the 251â€�dim feature vector. On error, returns None.
    """
    eeg_path = os.path.join(PREPROCESSED_PATH, f"{eeg_id}.npy")
    try:
        eeg_data = np.load(eeg_path).astype(np.float32)

        # Replace NaN/Inf per channel with that channel's mean
        if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
            for ch in range(eeg_data.shape[0]):
                channel = eeg_data[ch, :]
                mask = np.isnan(channel) | np.isinf(channel)
                if np.any(mask):
                    channel[mask] = np.nanmean(channel)

        # Zâ€�normalize each channel independently
        eeg_data = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / (
            np.std(eeg_data, axis=1, keepdims=True) + 1e-7
        )

        # Extract features
        feats = extract_features(eeg_data)
        return feats

    except Exception as e:
        print(f"Error loading/processing EEG {eeg_id}: {e}")
        return None


class BatchDataGenerator:
    """
    Yields (X_batch, y_batch) in chunks of batch_size. Uses process_eeg_file().
    """
    def __init__(self, eeg_ids: list, labels: np.ndarray, batch_size: int = BATCH_SIZE):
        self.eeg_ids    = eeg_ids
        self.labels     = labels
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, len(self.eeg_ids), self.batch_size):
            batch_ids    = self.eeg_ids[i : i + self.batch_size]
            batch_labels = self.labels[i : i + self.batch_size]

            batch_features = []
            batch_y        = []

            for eeg_id, lbl in zip(batch_ids, batch_labels):
                feats = process_eeg_file(eeg_id)
                if feats is not None:
                    batch_features.append(feats)
                    batch_y.append(lbl)

            if batch_features:
                yield np.vstack(batch_features), np.array(batch_y, dtype=np.int32)


# ------------------- LOAD & PREPROCESS LABELS -------------------
try:
    train_df = pd.read_csv(TRAIN_LABELS_PATH)
    print(f"Loaded {len(train_df)} annotations, {train_df['eeg_id'].nunique()} unique EEG IDs")
except Exception as e:
    print(f"Error loading train.csv: {e}")
    raise SystemExit

# Encode expert_consensus â†’ integer labels [0..5]
label_encoder = LabelEncoder()
label_encoder.fit(CLASSES)
train_df['label'] = label_encoder.transform(train_df['expert_consensus'])

# Determine which EEG IDs successfully preprocessed
if os.path.exists(SUCCESS_FILE_PATH):
    try:
        success_df  = pd.read_csv(SUCCESS_FILE_PATH)
        success_ids = set(success_df['eeg_id'].astype(str).tolist())
        print(f"Found success.csv with {len(success_ids)} successful IDs")
    except Exception as e:
        print(f"Error loading success.csv: {e}")
        success_ids = set(train_df['eeg_id'].astype(str).tolist())
else:
    print("No success.csv found â†’ using all EEG IDs from train.csv")
    success_ids = set(train_df['eeg_id'].astype(str).tolist())

# Filter to only those that were successfully preprocessed
valid_df = train_df[train_df['eeg_id'].astype(str).isin(success_ids)].copy()
eeg_ids = valid_df['eeg_id'].astype(str).tolist()
labels  = valid_df['label'].values
print(f"Processing {len(eeg_ids)} valid training samples\n")

# ------------------- TRAIN/VALIDATION SPLIT -------------------
train_ids, val_ids, train_labels, val_labels = train_test_split(
    eeg_ids,
    labels,
    test_size=0.20,
    stratify=labels,
    random_state=42
)
print(f"-> Train IDs: {len(train_ids)},  Val IDs: {len(val_ids)}\n")

# ------------------- EXTRACT FEATURES FOR TRAIN & VAL -------------------
print("Extracting features for TRAIN set in batches...")
all_train_feats = []
all_train_lbls  = []

for Xb, yb in BatchDataGenerator(train_ids, train_labels, batch_size=BATCH_SIZE):
    all_train_feats.append(Xb)
    all_train_lbls.append(yb)
    gc.collect()

X_train = np.vstack(all_train_feats)
y_train = np.hstack(all_train_lbls)
print(f"  Completed: X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")

del all_train_feats, all_train_lbls
gc.collect()

print("\nExtracting features for VALID set in batches...")
all_val_feats = []
all_val_lbls  = []

for Xb, yb in BatchDataGenerator(val_ids, val_labels, batch_size=BATCH_SIZE):
    all_val_feats.append(Xb)
    all_val_lbls.append(yb)
    gc.collect()

X_val = np.vstack(all_val_feats)
y_val = np.hstack(all_val_lbls)
print(f"  Completed: X_val.shape = {X_val.shape}, y_val.shape = {y_val.shape}\n")

del all_val_feats, all_val_lbls
gc.collect()

# ------------------- IMPUTE MISSING VALUES -------------------
print("Imputing NaNs (if any) with median of each feature...")
imputer = SimpleImputer(strategy='median')

# Fit on X_train and transform both train & val
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed   = imputer.transform(X_val)

# (Optional) Save the imputer for later inference
with open(os.path.join(TEST_PATH, "svm_imputer.pkl"), "wb") as f:
    pickle.dump(imputer, f)

# ------------------- STANDARD SCALING -------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled   = scaler.transform(X_val_imputed)

# Save the scaler for later inference
with open(os.path.join(TEST_PATH, "svm_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# ------------------- TRAIN SVM WITH PROBABILITIES -------------------
print("Training SVM (RBF kernel, probability=True, class_weight='balanced')...\n")
svm_clf = SVC(
    kernel="rbf",
    C=1.0,
    probability=True,
    class_weight="balanced",
    random_state=42
)

svm_clf.fit(X_train_scaled, y_train)

# ------------------- PREDICTIONS & METRICS -------------------
train_proba = svm_clf.predict_proba(X_train_scaled)  # shape = (n_train, 6)
val_proba   = svm_clf.predict_proba(X_val_scaled)    # shape = (n_val, 6)

train_pred = np.argmax(train_proba, axis=1)
val_pred   = np.argmax(val_proba,   axis=1)

train_acc = accuracy_score(y_train, train_pred)
val_acc   = accuracy_score(y_val,   val_pred)

train_true_onehot = np.eye(N_CLASSES)[y_train]
val_true_onehot   = np.eye(N_CLASSES)[y_val]

train_kl = kl_divergence_numpy(train_true_onehot, train_proba)
val_kl   = kl_divergence_numpy(val_true_onehot,   val_proba)

print("=== FINAL RESULTS ===")
print(f"Train Accuracy     â†’ {train_acc * 100:.2f}%")
print(f"Validation Accuracyâ†’ {val_acc   * 100:.2f}%\n")

print(f"Train KL Divergenceâ†’ {train_kl:.6f}")
print(f"Val   KL Divergenceâ†’ {val_kl:.6f}")
print("=====================\n")

if train_acc >= 0.75 and val_acc >= 0.75:
    print("âœ” Both TRAIN and VALIDATION accuracy â‰¥ 75%")
else:
    print("âš  Did not reach â‰¥ 75% on train and/or validation. Consider hyperparameter tuning.")

# ------------------- SAVE SVM MODEL & ARTIFACTS -------------------
os.makedirs(os.path.join(TEST_PATH, "models_svm"), exist_ok=True)
with open(os.path.join(TEST_PATH, "models_svm", "svm_model.pkl"), "wb") as f:
    pickle.dump(svm_clf, f)

with open(os.path.join(TEST_PATH, "models_svm", "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

print("\nSaved:")
print(f"  â€¢ SVM model   â†’ {os.path.join(TEST_PATH, 'models_svm', 'svm_model.pkl')}")
print(f"  â€¢ LabelEncoderâ†’ {os.path.join(TEST_PATH, 'models_svm', 'label_encoder.pkl')}")
print(f"  â€¢ Imputer     â†’ {os.path.join(TEST_PATH, 'svm_imputer.pkl')}")
print(f"  â€¢ Scaler      â†’ {os.path.join(TEST_PATH, 'svm_scaler.pkl')}\n")

print("ğŸŸ¢ SVM training + evaluation complete!")


