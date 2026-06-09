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


import os
import math
import numpy as np
import polars as pl
import tensorflow as tf
import albumentations as A
import cv2
import torch
import torchaudio
from scipy.signal import butter, filtfilt



BASE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
TRAIN_EEG_PATH = os.path.join(BASE_PATH, "train_eegs")
TEST_EEG_PATH = os.path.join(BASE_PATH, "test_eegs")

n_fft = 800
win_length = 256
hop_length = 44

spec_transform = torchaudio.transforms.Spectrogram(
    n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=None
)

spec_transforms = A.Compose([
    A.Resize(height=96, width=224, interpolation=cv2.INTER_CUBIC, always_apply=True)
])



def MAD(signal, axis=-1):
    median = np.median(signal, axis=axis, keepdims=True)
    abs_dev = np.abs(signal - median)
    mad = np.median(abs_dev, axis=axis, keepdims=True)
    return mad * 1.4826

def butter_filter(data, fs=200, cutoff_freq=[0.25, 50], order=5, btype="bandpass"):
    nyq = 0.5 * fs
    low = cutoff_freq[0] / nyq
    high = cutoff_freq[1] / nyq
    b, a = butter(order, [low, high], btype=btype)
    return filtfilt(b, a, data)

import math
import numpy as np

def bin_array(array, bin_size=4, axis=-1):
    #print(f"Original shape: {array.shape}")
    
    length = array.shape[axis]
    
    # Calculate padding length, ensure it's only added when necessary
    pad_len = (math.ceil(length / bin_size) * bin_size) - length
    #print(f"Padding length: {pad_len}")
    
    # Pad the array to align it with bin_size
    if pad_len > 0:
        array = np.pad(array, [(0, 0)] * axis + [(0, pad_len)] + [(0, 0)] * (array.ndim - axis - 1), mode="reflect")
    #print(f"Padded shape: {array.shape}")
    
    # Calculate the new shape after binning
    num_bins = array.shape[axis] // bin_size  # Number of bins
    new_shape = list(array.shape)
    new_shape[axis] = num_bins  # The first dimension becomes the number of bins
    new_shape.insert(axis + 1, bin_size)  # The second dimension corresponds to bin_size
    #print(f"New shape after insertion: {new_shape}")
    
    # Reshape and compute the mean across the bins
    reshaped_array = array.reshape(new_shape).mean(axis=axis + 1)  # Take mean across the bin_size dimension
    #print(f"Reshaped array shape: {reshaped_array.shape}")
    
    return reshaped_array





def safe_reshape_eeg(eeg, target_shape=(19, 2500)):
    """Safely reshape EEG array to (19, 2500) by trimming or padding if needed."""
    total_channels = target_shape[0]
    target_length = target_shape[1]
    expected_size = total_channels * target_length

    current_size = eeg.shape[0] * eeg.shape[1]
    if current_size < expected_size:
        # Pad at the end with reflection
        pad_size = expected_size - current_size
        eeg = np.pad(eeg, [(0, 0), (0, pad_size)], mode='reflect')
    elif current_size > expected_size:
        # Trim at the end
        eeg = eeg[:, :expected_size // eeg.shape[0]]
    return eeg.reshape(target_shape)

def compute_eeg_chain(df):
    cols = ["Fp1","Fp2","Fz","Cz","Pz","F3","F4","F7","F8","C3","C4","P3","P4","T3","T4","T5","T6","O1","O2"]
    eeg = [df[col].to_numpy() for col in cols]
    
    ekg = butter_filter(df["EKG"].to_numpy(), cutoff_freq=[0.5, 20.0])
    ekg = bin_array(ekg).reshape(1, -1)

    def pair(a, b): return bin_array(butter_filter(a - b))
    
    ll = [pair(eeg[0], eeg[7]), pair(eeg[7], eeg[13]), pair(eeg[13], eeg[15]), pair(eeg[15], eeg[17])]
    lp = [pair(eeg[0], eeg[5]), pair(eeg[5], eeg[9]), pair(eeg[9], eeg[11]), pair(eeg[11], eeg[17])]
    rp = [pair(eeg[1], eeg[6]), pair(eeg[6], eeg[10]), pair(eeg[10], eeg[12]), pair(eeg[12], eeg[18])]
    rl = [pair(eeg[1], eeg[8]), pair(eeg[8], eeg[14]), pair(eeg[14], eeg[16]), pair(eeg[16], eeg[18])]
    mid = [pair(eeg[2], eeg[3]), pair(eeg[3], eeg[4])]

    chains = np.stack([ll, lp, rp, rl])
    mid = np.stack(mid)
    
    return chains, mid, ekg

def proc_eeg(eeg, mid, ekg):
    eeg[np.isnan(eeg) | np.isinf(eeg)] = 0
    mid[np.isnan(mid) | np.isinf(mid)] = 0
    ekg[np.isnan(ekg) | np.isinf(ekg)] = 0

    eeg -= eeg.mean(axis=-1, keepdims=True)
    mid -= mid.mean(axis=-1, keepdims=True)

    std = np.median(MAD(eeg, axis=-1)) + 1e-5
    eeg = np.clip(eeg / std, -10, 10)
    mid = np.clip(mid / std, -10, 10)
    ekg = ekg / (MAD(ekg, axis=-1).mean() + 1e-5)

    eeg = eeg.reshape(16, -1)
    eeg = np.concatenate([eeg, mid, ekg], axis=0)

    eeg = safe_reshape_eeg(eeg, target_shape=(19, 2500))  # ğŸ‘ˆ Replace old reshape line

    return eeg




@torch.no_grad()
def compute_spec(signal):
    signal = torch.tensor(signal, dtype=torch.float32)
    spec = spec_transform(signal)
    spec = spec[:, :, 2:98]
    spec = torch.abs(spec) / 15
    spec = torch.log(spec.clip(math.exp(-4), math.exp(7)))
    spec = spec.mean(dim=1)
    return spec.numpy()

def compute_spec_eeg(a, b):
    return butter_filter(a - b, cutoff_freq=[0.25, 40], order=5)

def compute_spec_chain(df):
    eeg = [df[col].to_numpy() for col in ["Fp1","Fp2","Fz","Cz","Pz","F3","F4","F7","F8","C3","C4","P3","P4","T3","T4","T5","T6","O1","O2"]]
    
    def pair(a, b): return compute_spec_eeg(a, b)
    ll = [pair(eeg[0], eeg[7]), pair(eeg[7], eeg[13]), pair(eeg[13], eeg[15]), pair(eeg[15], eeg[17])]
    lp = [pair(eeg[0], eeg[5]), pair(eeg[5], eeg[9]), pair(eeg[9], eeg[11]), pair(eeg[11], eeg[17])]
    rp = [pair(eeg[1], eeg[6]), pair(eeg[6], eeg[10]), pair(eeg[10], eeg[12]), pair(eeg[12], eeg[18])]
    rl = [pair(eeg[1], eeg[8]), pair(eeg[8], eeg[14]), pair(eeg[14], eeg[16]), pair(eeg[16], eeg[18])]
    
    chain = np.stack([ll, lp, rp, rl])
    chain = chain / (MAD(chain, axis=-1).mean() + 1e-5)
    return compute_spec(chain[:, 0])



def resolve_path(eeg_id, mode="train"):
    subdir = TRAIN_EEG_PATH if mode == "train" else TEST_EEG_PATH
    return os.path.join(subdir, f"{eeg_id}.parquet")

def compute_spec_from_file(path):
    try:
        df = pl.read_parquet(path).fill_null(0)
        return compute_spec_chain(df)
    except Exception as e:
        print(f"Failed to read or process spec from {path}: {e}")
        return None

def compute_eeg_from_file(path):
    try:
        df = pl.read_parquet(path).fill_null(0)
        return compute_eeg_chain(df)
    except Exception as e:
        print(f"Failed to read or process EEG from {path}: {e}")
        return None



def proc_kspec(x):
    # Slice the data (based on your specific needs)
    x = x[:, 2:98]
    
    # Handle NaN and infinite values
    x[np.isnan(x) | np.isinf(x)] = 0
    
    # Apply log transform (clip to avoid extremely small or large values)
    x = np.log(np.clip(x, np.exp(-4), np.exp(7)))
    
    # Normalize along the correct axis (axis=1 for channels)
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-5)
    
    # Apply any additional spec transformations
    x = spec_transforms(image=x)["image"]
    
    # Print shape to debug
    #print(x.shape)
    
    # Adjust reshape based on the actual number of elements
    return x.reshape(4, 48, 112)  # Adjust based on the data size




def proc_eeg_spec(x):
    x = x[:, :, 2:-2]
    x[np.isnan(x) | np.isinf(x)] = 0
    x += 1
    return x.reshape(4, 96, 224)



test_file = "1000913311.parquet"
path = os.path.join(BASE_PATH, "train_eegs", test_file)

import polars as pl
df = pl.read_parquet(path).fill_null(0)

print("Shape:", df.shape)
print("Columns:", df.columns)



eeg, mid, ekg = compute_eeg_chain(df)
processed_eeg = proc_eeg(eeg, mid, ekg)
print("EEG shape:", processed_eeg.shape)

spec = compute_spec_chain(df)
processed_spec = proc_kspec(spec)
print("Spec shape:", processed_spec.shape)



import os
import numpy as np
from tqdm import tqdm
import polars as pl

# Config
SAVE_DIR = "/kaggle/working/preprocessed"
BATCH_SIZE = 100  # Change this to control batch size

# Create output directories
os.makedirs(os.path.join(SAVE_DIR, "eeg"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "spec"), exist_ok=True)

# Input EEG directory
eeg_dir = os.path.join(BASE_PATH, "train_eegs")
eeg_files = sorted(f for f in os.listdir(eeg_dir) if f.endswith(".parquet"))

# Skip already processed files (resumable batches)
already_processed = set(f.replace(".npy", "") for f in os.listdir(os.path.join(SAVE_DIR, "eeg")))
eeg_files = [f for f in eeg_files if f.replace(".parquet", "") not in already_processed]

# Tracking
failed_files = []
success_files = []
partial_success_files = []

# Process in batches
for i in range(0, len(eeg_files), BATCH_SIZE):
    batch = eeg_files[i:i + BATCH_SIZE]
    print(f"\nğŸ”„ Processing batch {i // BATCH_SIZE + 1} / {(len(eeg_files) - 1) // BATCH_SIZE + 1}")

    for fname in tqdm(batch, desc="Processing EEGs"):
        eeg_id = fname.replace(".parquet", "")
        path = os.path.join(eeg_dir, fname)

        try:
            df = pl.read_parquet(path).fill_null(0)
            eeg_success, spec_success = False, False

            # EEG Processing
            try:
                eeg, mid, ekg = compute_eeg_chain(df)
                eeg_processed = proc_eeg(eeg, mid, ekg)
                np.save(os.path.join(SAVE_DIR, "eeg", f"{eeg_id}.npy"), eeg_processed)
                eeg_success = True
            except Exception as e:
                print(f"[EEG FAIL] {eeg_id}: {e}")
                failed_files.append((eeg_id, "eeg"))

            # Spectrogram Processing
            try:
                spec = compute_spec_chain(df)
                spec_processed = proc_kspec(spec)
                np.save(os.path.join(SAVE_DIR, "spec", f"{eeg_id}.npy"), spec_processed)
                spec_success = True
            except Exception as e:
                print(f"[SPEC FAIL] {eeg_id}: {e}")
                failed_files.append((eeg_id, "spec"))

            # Logging result
            if eeg_success and spec_success:
                success_files.append(eeg_id)
            elif eeg_success or spec_success:
                partial_success_files.append(eeg_id)

        except Exception as e:
            print(f"[FILE FAIL] {eeg_id}: {e}")
            failed_files.append((eeg_id, "file"))

# Summary
print(f"\nâœ… Fully preprocessed files: {len(success_files)}")
print(f"âš ï¸�  Partially preprocessed files: {len(partial_success_files)}")
print(f"â�Œ Total failed files: {len(failed_files)}")

# Optional: Save summaries
import pandas as pd

pd.DataFrame(success_files, columns=["eeg_id"]).to_csv(os.path.join(SAVE_DIR, "success.csv"), index=False)
pd.DataFrame(partial_success_files, columns=["eeg_id"]).to_csv(os.path.join(SAVE_DIR, "partial_success.csv"), index=False)
pd.DataFrame(failed_files, columns=["eeg_id", "fail_type"]).to_csv(os.path.join(SAVE_DIR, "failures.csv"), index=False)


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import os

# Load metadata
train_df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')

# Directory where you saved preprocessed data
PREPROCESSED_DIR = "/kaggle/working/preprocessed"

# Function to load numpy arrays
def load_preprocessed_data(eeg_id):
    eeg_path = os.path.join(PREPROCESSED_DIR, "eeg", f"{eeg_id}.npy")
    spec_path = os.path.join(PREPROCESSED_DIR, "spec", f"{eeg_id}.npy")
    
    eeg = np.load(eeg_path) if os.path.exists(eeg_path) else None
    spec = np.load(spec_path) if os.path.exists(spec_path) else None
    
    return eeg, spec

# Get unique EEG IDs from metadata
eeg_ids = train_df['eeg_id'].unique()


# Initialize lists to store features and labels
X = []
y = []

# Define the target columns (6 seizure types)
target_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

for eeg_id in eeg_ids[:2000]:  # Using first 2000 samples for demo (adjust as needed)
    eeg, spec = load_preprocessed_data(eeg_id)
    
    if eeg is not None and spec is not None:
        # Flatten and concatenate features
        eeg_flat = eeg.flatten()
        spec_flat = spec.flatten()
        features = np.concatenate([eeg_flat, spec_flat])
        
        # Get corresponding labels from metadata
        labels = train_df[train_df['eeg_id'] == eeg_id][target_cols].values.mean(axis=0)
        pred_class = np.argmax(labels)  # Convert to single class label
        
        X.append(features)
        y.append(pred_class)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)


# Standardize features first
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce dimensions
pca = PCA(n_components=100)  # Adjust based on your needs
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.2f}")


X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)


# Initialize SVM - using linear kernel for efficiency
svm = SVC(kernel='linear', C=1.0, random_state=42, class_weight='balanced')

# Train the model
svm.fit(X_train, y_train)

# Evaluate
train_score = svm.score(X_train, y_train)
test_score = svm.score(X_test, y_test)

print(f"Training Accuracy: {train_score:.2f}")
print(f"Test Accuracy: {test_score:.2f}")

# Detailed classification report
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_cols))


import joblib

# Save the trained model
joblib.dump(svm, '/kaggle/working/eeg_svm_model.pkl')

# To load later:
# svm = joblib.load('/kaggle/working/eeg_svm_model.pkl')


import os

model_path = '/kaggle/working/eeg_svm_model.pkl'
print(f"File exists: {os.path.exists(model_path)}")

# List all files in /kaggle/working
print("Files in /kaggle/working:")
print(os.listdir('/kaggle/working'))


import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import os
from scipy.stats import entropy  # For KL-Divergence

# Load metadata
train_df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')

# Directory where you saved preprocessed data
PREPROCESSED_DIR = "/kaggle/working/preprocessed"

# Function to load numpy arrays
def load_preprocessed_data(eeg_id):
    eeg_path = os.path.join(PREPROCESSED_DIR, "eeg", f"{eeg_id}.npy")
    spec_path = os.path.join(PREPROCESSED_DIR, "spec", f"{eeg_id}.npy")
    
    eeg = np.load(eeg_path) if os.path.exists(eeg_path) else None
    spec = np.load(spec_path) if os.path.exists(spec_path) else None
    
    return eeg, spec

# Function to calculate KL-Divergence
def calculate_kl_divergence(y_true, y_pred, num_classes=6):
    """Calculate KL divergence between true and predicted distributions"""
    # Create probability distributions
    true_dist = np.bincount(y_true, minlength=num_classes) / len(y_true)
    pred_dist = np.bincount(y_pred, minlength=num_classes) / len(y_pred)
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    true_dist = true_dist + epsilon
    pred_dist = pred_dist + epsilon
    
    # Normalize
    true_dist = true_dist / np.sum(true_dist)
    pred_dist = pred_dist / np.sum(pred_dist)
    
    return entropy(true_dist, pred_dist)

# Get unique EEG IDs from metadata
eeg_ids = train_df['eeg_id'].unique()

# Initialize lists to store features and labels
X = []
y = []

# Define the target columns (6 seizure types)
target_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

for eeg_id in eeg_ids[:2000]:  # Using first 2000 samples for demo (adjust as needed)
    eeg, spec = load_preprocessed_data(eeg_id)
    
    if eeg is not None and spec is not None:
        # Flatten and concatenate features
        eeg_flat = eeg.flatten()
        spec_flat = spec.flatten()
        features = np.concatenate([eeg_flat, spec_flat])
        
        # Get corresponding labels from metadata
        labels = train_df[train_df['eeg_id'] == eeg_id][target_cols].values.mean(axis=0)
        pred_class = np.argmax(labels)  # Convert to single class label
        
        X.append(features)
        y.append(pred_class)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Standardize features first
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce dimensions
pca = PCA(n_components=100)  # Adjust based on your needs
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.2f}")

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize SVM - using linear kernel for efficiency
svm = SVC(kernel='linear', C=1.0, random_state=42, class_weight='balanced')

# Train the model
svm.fit(X_train, y_train)

# Evaluate
train_score = svm.score(X_train, y_train)
test_score = svm.score(X_test, y_test)

print(f"Training Accuracy: {train_score:.2f}")
print(f"Test Accuracy: {test_score:.2f}")

# Detailed classification report
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_cols))

# Calculate KL-Divergence
kl_div = calculate_kl_divergence(y_test, y_pred)
print(f"\nKL Divergence between true and predicted distributions: {kl_div:.4f}")

# Save the trained model
import joblib
joblib.dump(svm, '/kaggle/working/eeg_svm_model.pkl')


#new code with apparently higher accuracy


import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from scipy.stats import entropy
import joblib

# Load metadata
train_df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
PREPROCESSED_DIR = "/kaggle/working/preprocessed"

# Columns with seizure types
target_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

# Load preprocessed data
def load_preprocessed_data(eeg_id):
    eeg_path = os.path.join(PREPROCESSED_DIR, "eeg", f"{eeg_id}.npy")
    spec_path = os.path.join(PREPROCESSED_DIR, "spec", f"{eeg_id}.npy")
    eeg = np.load(eeg_path) if os.path.exists(eeg_path) else None
    spec = np.load(spec_path) if os.path.exists(spec_path) else None
    return eeg, spec

# Statistical feature extractor
def extract_stat_features(data):
    stats = []
    for channel in data:  # Assumes shape (channels, time) or (freq, time)
        stats.extend([
            np.mean(channel), np.std(channel), np.max(channel),
            np.min(channel), np.median(channel),
            np.percentile(channel, 25), np.percentile(channel, 75)
        ])
    return stats

# KL divergence calculator
def calculate_kl_divergence(y_true, y_pred, num_classes=6):
    true_dist = np.bincount(y_true, minlength=num_classes) / len(y_true)
    pred_dist = np.bincount(y_pred, minlength=num_classes) / len(y_pred)
    epsilon = 1e-10
    true_dist += epsilon
    pred_dist += epsilon
    true_dist /= true_dist.sum()
    pred_dist /= pred_dist.sum()
    return entropy(true_dist, pred_dist)

# Prepare data
X, y = [], []
eeg_ids = train_df['eeg_id'].unique()

for eeg_id in eeg_ids[:2000]:  # Limit to 2000 samples
    eeg, spec = load_preprocessed_data(eeg_id)
    if eeg is None or spec is None:
        continue

    row = train_df[train_df['eeg_id'] == eeg_id][target_cols].mean()
    if row.max() < 0.5:
        continue  # Skip unclear labels

    label = np.argmax(row)
    eeg_features = extract_stat_features(eeg)
    spec_features = extract_stat_features(spec)
    combined_features = eeg_features + spec_features
    X.append(combined_features)
    y.append(label)

# Convert to arrays
X = np.array(X)
y = np.array(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X_scaled)
print(f"PCA - Components selected: {X_pca.shape[1]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
clf = RandomForestClassifier(
    n_estimators=200, max_depth=20, random_state=42, class_weight='balanced'
)
clf.fit(X_train, y_train)

# Evaluate
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
print(f"Train Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")

# Classification report
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_cols))

# KL Divergence
kl_div = calculate_kl_divergence(y_test, y_pred)
print(f"KL Divergence: {kl_div:.4f}")

# Save model
joblib.dump(clf, '/kaggle/working/eeg_rf_model.pkl')
print("Model saved as 'eeg_rf_model.pkl'")



#new code for svm above is random forest


import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from scipy.stats import entropy
import joblib

# Load metadata
train_df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
PREPROCESSED_DIR = "/kaggle/working/preprocessed"
target_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

def load_preprocessed_data(eeg_id):
    eeg_path = os.path.join(PREPROCESSED_DIR, "eeg", f"{eeg_id}.npy")
    spec_path = os.path.join(PREPROCESSED_DIR, "spec", f"{eeg_id}.npy")
    eeg = np.load(eeg_path) if os.path.exists(eeg_path) else None
    spec = np.load(spec_path) if os.path.exists(spec_path) else None
    return eeg, spec

def extract_stat_features(data):
    stats = []
    for channel in data:
        stats.extend([
            np.mean(channel), np.std(channel), np.max(channel),
            np.min(channel), np.median(channel),
            np.percentile(channel, 25), np.percentile(channel, 75)
        ])
    return stats

def calculate_kl_divergence(y_true, y_pred, num_classes=6):
    true_dist = np.bincount(y_true, minlength=num_classes) / len(y_true)
    pred_dist = np.bincount(y_pred, minlength=num_classes) / len(y_pred)
    epsilon = 1e-10
    true_dist += epsilon
    pred_dist += epsilon
    true_dist /= true_dist.sum()
    pred_dist /= pred_dist.sum()
    return entropy(true_dist, pred_dist)

# Prepare data
X, y = [], []
eeg_ids = train_df['eeg_id'].unique()

for eeg_id in eeg_ids[:2000]:  # Adjust range for full training
    eeg, spec = load_preprocessed_data(eeg_id)
    if eeg is None or spec is None:
        continue

    row = train_df[train_df['eeg_id'] == eeg_id][target_cols].mean()
    if row.max() < 0.5:
        continue  # Skip low-confidence labels

    label = np.argmax(row)
    eeg_features = extract_stat_features(eeg)
    spec_features = extract_stat_features(spec)
    combined_features = eeg_features + spec_features
    X.append(combined_features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA (retain 95% variance)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, stratify=y, random_state=42
)

# SVM with RBF kernel (higher accuracy)
svm = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42)
svm.fit(X_train, y_train)

# Evaluate
train_acc = svm.score(X_train, y_train)
test_acc = svm.score(X_test, y_test)
print(f"Train Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")

# Report
y_pred = svm.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_cols))

# KL Divergence
kl_div = calculate_kl_divergence(y_test, y_pred)
print(f"KL Divergence: {kl_div:.4f}")

# Save model
joblib.dump(svm, '/kaggle/working/eeg_svm_rbf_model.pkl')
print("Model saved as 'eeg_svm_rbf_model.pkl'")



#new latest


import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from scipy.stats import entropy
import joblib

# Load metadata
train_df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
PREPROCESSED_DIR = "/kaggle/working/preprocessed"
target_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

def load_preprocessed_data(eeg_id):
    eeg_path = os.path.join(PREPROCESSED_DIR, "eeg", f"{eeg_id}.npy")
    spec_path = os.path.join(PREPROCESSED_DIR, "spec", f"{eeg_id}.npy")
    eeg = np.load(eeg_path) if os.path.exists(eeg_path) else None
    spec = np.load(spec_path) if os.path.exists(spec_path) else None
    return eeg, spec

def extract_stat_features(data):
    stats = []
    for channel in data:
        stats.extend([
            np.mean(channel), np.std(channel), np.max(channel),
            np.min(channel), np.median(channel),
            np.percentile(channel, 25), np.percentile(channel, 75)
        ])
    return stats

def calculate_kl_divergence(y_true, y_pred, num_classes=6):
    true_dist = np.bincount(y_true, minlength=num_classes) / len(y_true)
    pred_dist = np.bincount(y_pred, minlength=num_classes) / len(y_pred)
    epsilon = 1e-10
    true_dist += epsilon
    pred_dist += epsilon
    true_dist /= true_dist.sum()
    pred_dist /= pred_dist.sum()
    return entropy(true_dist, pred_dist)

# Prepare data
X, y = [], []
eeg_ids = train_df['eeg_id'].unique()

for eeg_id in eeg_ids:  # âœ… Use all available EEG IDs
    eeg, spec = load_preprocessed_data(eeg_id)
    if eeg is None or spec is None:
        continue

    row = train_df[train_df['eeg_id'] == eeg_id][target_cols].mean()
    if row.max() < 0.5:
        continue  # Skip low-confidence labels

    label = np.argmax(row)
    eeg_features = extract_stat_features(eeg)
    spec_features = extract_stat_features(spec)
    combined_features = eeg_features + spec_features
    X.append(combined_features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA (retain 95% variance)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, stratify=y, random_state=42
)

# SVM with RBF kernel (high accuracy)
svm = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42)
svm.fit(X_train, y_train)

# Evaluate
train_acc = svm.score(X_train, y_train)
test_acc = svm.score(X_test, y_test)
print(f"Train Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")

# Report
y_pred = svm.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_cols))

# KL Divergence
kl_div = calculate_kl_divergence(y_test, y_pred)
print(f"KL Divergence: {kl_div:.4f}")

# Save model
joblib.dump(svm, '/kaggle/working/eeg_svm_rbf_model.pkl')
print("Model saved as 'eeg_svm_rbf_model.pkl'")



#the above code uses all data


#the below code is random forest which uses all data


import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from scipy.stats import entropy
import joblib

# Load metadata
train_df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
PREPROCESSED_DIR = "/kaggle/working/preprocessed"
target_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

def load_preprocessed_data(eeg_id):
    eeg_path = os.path.join(PREPROCESSED_DIR, "eeg", f"{eeg_id}.npy")
    spec_path = os.path.join(PREPROCESSED_DIR, "spec", f"{eeg_id}.npy")
    eeg = np.load(eeg_path) if os.path.exists(eeg_path) else None
    spec = np.load(spec_path) if os.path.exists(spec_path) else None
    return eeg, spec

def extract_stat_features(data):
    stats = []
    for channel in data:
        stats.extend([
            np.mean(channel), np.std(channel), np.max(channel),
            np.min(channel), np.median(channel),
            np.percentile(channel, 25), np.percentile(channel, 75)
        ])
    return stats

def calculate_kl_divergence(y_true, y_pred, num_classes=6):
    true_dist = np.bincount(y_true, minlength=num_classes) / len(y_true)
    pred_dist = np.bincount(y_pred, minlength=num_classes) / len(y_pred)
    epsilon = 1e-10
    true_dist += epsilon
    pred_dist += epsilon
    true_dist /= true_dist.sum()
    pred_dist /= pred_dist.sum()
    return entropy(true_dist, pred_dist)

# Prepare dataset
X, y = [], []
eeg_ids = train_df['eeg_id'].unique()

for eeg_id in eeg_ids:  # âœ… Use all available EEG IDs
    eeg, spec = load_preprocessed_data(eeg_id)
    if eeg is None or spec is None:
        continue

    row = train_df[train_df['eeg_id'] == eeg_id][target_cols].mean()
    if row.max() < 0.5:
        continue  # Skip low-confidence labels

    label = np.argmax(row)
    eeg_features = extract_stat_features(eeg)
    spec_features = extract_stat_features(spec)
    combined_features = eeg_features + spec_features
    X.append(combined_features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA (retain 95% variance)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, stratify=y, random_state=42
)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Evaluation
train_acc = rf.score(X_train, y_train)
test_acc = rf.score(X_test, y_test)
print(f"Train Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")

# Classification Report
y_pred = rf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_cols))

# KL Divergence
kl_div = calculate_kl_divergence(y_test, y_pred)
print(f"KL Divergence: {kl_div:.4f}")

# Save model
joblib.dump(rf, '/kaggle/working/eeg_rf_model.pkl')
print("Model saved as 'eeg_rf_model.pkl'")



#sankalp code svm


import numpy as np
import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from scipy.stats import entropy
import joblib

# Load metadata
train_df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
PREPROCESSED_DIR = "/kaggle/working/preprocessed"
target_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

def load_preprocessed_data(eeg_id):
    eeg_path = os.path.join(PREPROCESSED_DIR, "eeg", f"{eeg_id}.npy")
    spec_path = os.path.join(PREPROCESSED_DIR, "spec", f"{eeg_id}.npy")
    eeg = np.load(eeg_path) if os.path.exists(eeg_path) else None
    spec = np.load(spec_path) if os.path.exists(spec_path) else None
    return eeg, spec

def extract_stat_features(data):
    stats = []
    for channel in data:
        stats.extend([
            np.mean(channel), np.std(channel), np.max(channel),
            np.min(channel), np.median(channel),
            np.percentile(channel, 25), np.percentile(channel, 75)
        ])
    return stats

def calculate_kl_divergence(y_true, y_pred, num_classes=6):
    true_dist = np.bincount(y_true, minlength=num_classes) / len(y_true)
    pred_dist = np.bincount(y_pred, minlength=num_classes) / len(y_pred)
    epsilon = 1e-10
    true_dist += epsilon
    pred_dist += epsilon
    true_dist /= true_dist.sum()
    pred_dist /= pred_dist.sum()
    return entropy(true_dist, pred_dist)

# Prepare dataset
X, y = [], []
eeg_ids = train_df['eeg_id'].unique()

for eeg_id in eeg_ids:  # âœ… Use all available EEG IDs
    eeg, spec = load_preprocessed_data(eeg_id)
    if eeg is None or spec is None:
        continue

    row = train_df[train_df['eeg_id'] == eeg_id][target_cols].mean()
    if row.max() < 0.5:
        continue  # Skip low-confidence labels

    label = np.argmax(row)
    eeg_features = extract_stat_features(eeg)
    spec_features = extract_stat_features(spec)
    combined_features = eeg_features + spec_features
    X.append(combined_features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA (retain 95% variance)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, stratify=y, random_state=42
)

# XGBoost Classifier
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
xgb.fit(X_train, y_train)

# Evaluation
train_acc = xgb.score(X_train, y_train)
test_acc = xgb.score(X_test, y_test)
print(f"Train Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")

# Report
y_pred = xgb.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_cols))

# KL Divergence
kl_div = calculate_kl_divergence(y_test, y_pred)
print(f"KL Divergence: {kl_div:.4f}")

# Save model
joblib.dump(xgb, '/kaggle/working/eeg_xgboost_model.pkl')
print("Model saved as 'eeg_xgboost_model.pkl'")



#xg boost


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import log_loss, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from xgboost import XGBClassifier

# Configuration
TEST_PATH = "/kaggle/working/"
BASE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
PREPROCESSED_PATH = "/kaggle/input/preprocessing/preprocessed"
TRAIN_LABELS_PATH = os.path.join(BASE_PATH, "train.csv")
MODEL_OUTPUT_PATH = os.path.join(TEST_PATH, "models")

# Define our classes
CLASSES = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']

# Metrics functions
def kl_divergence(y_true, y_pred):
    """
    Calculate KL divergence between true and predicted probabilities
    
    Args:
        y_true: One-hot encoded ground truth (N, C)
        y_pred: Predicted probabilities (N, C)
        
    Returns:
        Mean KL divergence
    """
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    y_true = np.clip(y_true, epsilon, 1.0)
    
    kl_div = np.sum(y_true * np.log(y_true / y_pred), axis=1)
    return np.mean(kl_div)

def kl_divergence_scorer(estimator, X, y):
    """
    Scorer function for GridSearchCV that calculates -KL divergence
    (negative because GridSearchCV maximizes score)
    """
    y_pred = estimator.predict_proba(X)
    return -kl_divergence(y, y_pred)  # Use soft labels directly

# Load the training labels
train_df = pd.read_csv(TRAIN_LABELS_PATH)
print(f"Loaded {len(train_df)} training samples")

# Aggregate annotations by eeg_id
print("Aggregating annotations by eeg_id...")
vote_columns = [f"{cls.lower()}_vote" for cls in CLASSES]
aggregated_labels = []

grouped = train_df.groupby('eeg_id')
eeg_ids = []
y_soft = []

for eeg_id, group in tqdm(grouped, total=len(grouped)):
    votes = group[vote_columns].sum()
    total_votes = votes.sum()
    if total_votes == 0:
        print(f"Warning: No votes for eeg_id {eeg_id}, skipping...")
        continue
    probs = votes / total_votes
    eeg_ids.append(eeg_id)
    y_soft.append(probs.values)

y_soft = np.array(y_soft)
print(f"Aggregated to {len(eeg_ids)} unique EEG IDs")

# Load features
print("Loading features...")
X = []

success_df = pd.read_csv(os.path.join(PREPROCESSED_PATH, "success.csv"))
success_ids = set(success_df['eeg_id'].tolist())

def load_features(eeg_id):
    try:
        eeg_path = os.path.join(PREPROCESSED_PATH, "eeg", f"{eeg_id}.npy")
        eeg_data = np.load(eeg_path).astype(np.float32)
        eeg_features = eeg_data.flatten()
        
        spec_path = os.path.join(PREPROCESSED_PATH, "spec", f"{eeg_id}.npy")
        spec_data = np.load(spec_path).astype(np.float32)
        spec_features = spec_data.flatten()
        
        features = np.concatenate([eeg_features, spec_features])
        return features
    except Exception as e:
        print(f"Error loading features for {eeg_id}: {e}")
        return None

N = 17300
indices_to_remove = []

for idx, eeg_id in tqdm(enumerate(eeg_ids), total=min(len(eeg_ids), N)):
    if idx >= N:
        break
    if eeg_id not in success_ids:
        indices_to_remove.append(idx)
        continue
    features = load_features(eeg_id)
    if features is not None:
        X.append(features)
    else:
        indices_to_remove.append(idx)

# Remove entries with missing features
for idx in sorted(indices_to_remove, reverse=True):
    y_soft = np.delete(y_soft, idx, axis=0)
    eeg_ids.pop(idx)

X = np.array(X)
print(f"Loaded features for {len(X)} samples")
print(f"Feature vector shape: {X.shape}")
print(f"Label shape: {y_soft.shape}")

max_cores = min(4, multiprocessing.cpu_count() - 1)
print(f"Using {max_cores} cores for parallel processing")

# Hard labels for stratification
y_hard = np.argmax(y_soft, axis=1)

n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

reduced_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 8],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

base_xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=len(CLASSES),
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_jobs=1,
    verbosity=1,
    random_state=42
)

print("Training with cross-validation...")

cv_scores = []
cv_models = []
cv_predictions = []
cv_feature_importances = []
all_fold_predictions = {}

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_hard)):
    print(f"\nFold {fold+1}/{n_splits}")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_soft[train_idx], y_soft[val_idx]
    eeg_ids_val = [eeg_ids[i] for i in val_idx]

    grid_search = GridSearchCV(
        estimator=base_xgb,
        param_grid=reduced_param_grid,
        scoring=kl_divergence_scorer,
        cv=2,
        n_jobs=1,
        verbose=1
    )
    grid_search.fit(X_train, y_train.argmax(axis=1))

    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    y_val_pred_proba = best_model.predict_proba(X_val)
    kl_score = kl_divergence(y_val, y_val_pred_proba)
    print(f"Fold {fold+1} KL Divergence: {kl_score:.6f}")
    ll_score = log_loss(y_val.argmax(axis=1), y_val_pred_proba)
    print(f"Fold {fold+1} Log Loss: {ll_score:.6f}")

    cv_scores.append(kl_score)
    cv_models.append(best_model)
    cv_predictions.append((y_val, y_val_pred_proba))
    cv_feature_importances.append(best_model.feature_importances_)

    for i, eeg_id in enumerate(eeg_ids_val):
        all_fold_predictions[eeg_id] = y_val_pred_proba[i]

os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
for fold, model in enumerate(cv_models):
    with open(os.path.join(MODEL_OUTPUT_PATH, f"xgb_model_fold_{fold}.pkl"), "wb") as f:
        pickle.dump(model, f)

mean_importance = np.mean(cv_feature_importances, axis=0)
feature_importance_df = pd.DataFrame({
    'Feature': [f"Feature_{i}" for i in range(len(mean_importance))],
    'Importance': mean_importance
})
top_features = feature_importance_df.sort_values('Importance', ascending=False).head(30)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=top_features)
plt.title('Top 30 Feature Importances')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_OUTPUT_PATH, 'feature_importance.png'))
plt.close()

mean_kl = np.mean(cv_scores)
std_kl = np.std(cv_scores)
print(f"\nCross-validation results:")
print(f"Mean KL Divergence: {mean_kl:.6f} Â± {std_kl:.6f}")

print("\nTraining final model on all data with epoch-like output...")
final_model = XGBClassifier(
    **grid_search.best_params_,
    objective='multi:softprob',
    num_class=len(CLASSES),
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_jobs=max_cores,
    verbosity=1,
    random_state=42
)
final_model.fit(X, y_soft.argmax(axis=1), eval_set=[(X, y_soft.argmax(axis=1))], verbose=True)

with open(os.path.join(MODEL_OUTPUT_PATH, "xgb_model_final.pkl"), "wb") as f:
    pickle.dump(final_model, f)

print("Final model saved to:", os.path.join(MODEL_OUTPUT_PATH, "xgb_model_final.pkl"))

best_fold = np.argmin(cv_scores)
y_val, y_val_pred_proba = cv_predictions[best_fold]
y_val_pred = np.argmax(y_val_pred_proba, axis=1)
y_val_hard = np.argmax(y_val, axis=1)

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_val_hard, y_val_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(cmap=plt.cm.Blues)
plt.title(f'Normalized Confusion Matrix - Best Fold (KL: {cv_scores[best_fold]:.6f})')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_OUTPUT_PATH, 'confusion_matrix.png'))
plt.close()

def create_prediction_file():
    test_eeg_path = os.path.join(BASE_PATH, "test_eegs")
    test_files = [f.replace(".parquet", "") for f in os.listdir(test_eeg_path) if f.endswith(".parquet")]

    submission_df = pd.DataFrame({'eeg_id': test_files})
    for cls in CLASSES:
        submission_df[cls] = 0.0

    print("Generating predictions for test data...")

    for eeg_id in tqdm(test_files):
        eeg_feature_path = os.path.join("/kaggle/working/preprocessed/eeg", f"{eeg_id}.npy")
        spec_feature_path = os.path.join("/kaggle/working/preprocessed/spec", f"{eeg_id}.npy")

        try:
            eeg_features = np.load(eeg_feature_path)
            spec_features = np.load(spec_feature_path)
            features = np.concatenate([eeg_features.flatten(), spec_features.flatten()]).reshape(1, -1)
        except Exception as e:
            print(f"Warning: Missing or error loading features for {eeg_id}. Error: {e}")
            features = None

        if features is not None:
            probs = final_model.predict_proba(features)[0]
            for i, cls in enumerate(CLASSES):
                submission_df.loc[submission_df['eeg_id'] == eeg_id, cls] = probs[i]
        else:
            uniform_prob = 1.0 / len(CLASSES)
            for cls in CLASSES:
                submission_df.loc[submission_df['eeg_id'] == eeg_id, cls] = uniform_prob

    submission_path = "/kaggle/working/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print("Submission file saved to:", submission_path)

    return submission_df

submission_df = create_prediction_file


