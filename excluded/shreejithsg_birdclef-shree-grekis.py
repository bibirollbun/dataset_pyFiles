import pandas as pd

df = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")
freq_table = df["primary_label"].value_counts()
print(freq_table.head(20)) 



top10 = ['grekis', 'compau', 'trokin', 'roahaw', 'banana', 'whtdov', 'socfly1', 'yeofly1', 'bobfly1', 'wbwwre1']

# Positive = grekis
pos_df = df[df['primary_label'] == 'grekis'].copy()
pos_df['label'] = 1

# Negative = rest of top 10 (excluding grekis)
neg_df = df[df['primary_label'].isin(top10) & (df['primary_label'] != 'grekis')].copy()
neg_df['label'] = 0

from sklearn.utils import resample

# Downsample negatives to match positives
neg_df_balanced = resample(neg_df, 
                           replace=False, 
                           n_samples=len(pos_df), 
                           random_state=42)

combined_df = pd.concat([pos_df, neg_df_balanced]).sample(frac=1, random_state=42)

# Combine
combined_df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1, random_state=42)

# Preview
combined_df[['primary_label', 'filename', 'label']].head()


neg_df = df[df['primary_label'].isin(top10) & (df['primary_label'] != 'grekis')].copy()
neg_df['label'] = 0

from sklearn.utils import resample

neg_df_balanced = resample(neg_df, 
                           replace=False, 
                           n_samples=len(pos_df), 
                           random_state=42)

combined_df = pd.concat([pos_df, neg_df_balanced]).sample(frac=1, random_state=42)
combined_df


import numpy as np
import librosa

def extract_binary_features(filepath, sr=32000, n_fft=1024, hop_length=512, percentile=85):
    """
    Converts an audio file into a 3200-dim binary frequency-bin vector using adaptive thresholding.
    
    Args:
        filepath (str): Full path to the audio file
        sr (int): Sampling rate
        n_fft (int): FFT window size
        hop_length (int): Number of samples between frames
        percentile (float): Percentile threshold for activity in a frequency bin (0â€“100)

    Returns:
        np.ndarray: Binary vector of shape (3200,)
    """
    # Load audio
    y, sr_actual = librosa.load(filepath, sr=sr)
    
    # Compute STFT (complex)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    
    # Power and convert to decibels
    S_power = np.abs(S) ** 2
    S_db = librosa.power_to_db(S_power, ref=np.max)

    # Frequency values
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Frequency binning: 0 to 16000 Hz, in 5 Hz intervals â†’ 3200 bins
    bin_edges = np.linspace(0, 16000, 3201)
    binary_vector = np.zeros(3200, dtype=int)

    # Compute threshold based on global energy distribution
    energy_per_freq = S_db.max(axis=1)  # Max across time for each frequency
    adaptive_threshold = np.percentile(energy_per_freq, percentile)

    # Assign 1s where max energy in bin > threshold
    for i in range(3200):
        f_start = bin_edges[i]
        f_end = bin_edges[i+1]
        bin_mask = (freqs >= f_start) & (freqs < f_end)
        
        if np.any(bin_mask):
            max_energy = np.max(energy_per_freq[bin_mask])
            if max_energy > adaptive_threshold:
                binary_vector[i] = 1

    return binary_vector


path = "/kaggle/input/birdclef-2025/train_audio/" + combined_df.iloc[0]["filename"]
vec = extract_binary_features(path)
print(vec.shape) 
print(vec.sum()) 


import os
import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from tqdm import tqdm

# Settings
thresholds_to_try = np.linspace(70, 90, 10)  # 10 thresholds
n_splits = 5
AUDIO_BASE = "/kaggle/input/birdclef-2025/train_audio/"
N_FILES = 100  
results = {}

def extract_binary_features(path, percentile=85):
    y, sr = librosa.load(path, sr=32000)
    S = librosa.stft(y, n_fft=1024, hop_length=512)
    S_power = np.abs(S) ** 2
    S_db = librosa.power_to_db(S_power, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    bin_edges = np.linspace(0, 16000, 3201)
    binary_vector = np.zeros(3200, dtype=int)
    energy_per_freq = S_db.max(axis=1)
    adaptive_threshold = np.percentile(energy_per_freq, percentile)

    for j in range(3200):
        f_start = bin_edges[j]
        f_end = bin_edges[j+1]
        bin_mask = (freqs >= f_start) & (freqs < f_end)
        if np.any(bin_mask):
            max_energy = np.max(energy_per_freq[bin_mask])
            if max_energy > adaptive_threshold:
                binary_vector[j] = 1

    return binary_vector



# Main Loop
for thresh in thresholds_to_try:
    print(f"\nğŸ”µ Threshold = {thresh:.1f}%")

    X = []
    y = []

    for i in tqdm(range(N_FILES), desc=f"Extracting with {thresh:.1f}"):
        path = AUDIO_BASE + combined_df.iloc[i]["filename"]

        try:
            vec = extract_binary_features(path, percentile=thresh)
            X.append(vec)
            y.append(combined_df.iloc[i]['label'])
        except Exception as e:
            print(f"â�Œ Failed for file {i}: {e}")

    X = np.stack(X)
    y = np.array(y)

    # 5-Fold Cross-Validation
    f1_scores = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = XGBClassifier(
            n_estimators=500,
            scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        f1_scores.append(f1)

    avg_f1 = np.mean(f1_scores)
    print(f"âœ… Average F1 @ {thresh:.1f}% = {avg_f1:.4f}")

    results[thresh] = avg_f1

# Summary
print("\nğŸ“Š Final Results:")
for k, v in results.items():
    print(f"Threshold {k:.1f}% -> Avg F1: {v:.4f}")

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(list(results.keys()), list(results.values()), marker='o')
plt.xlabel('Percentile Threshold')
plt.ylabel('Average F1-Score (5-Fold CV)')
plt.title('Threshold vs F1-Score')
plt.grid(True)
plt.show()


path = "/kaggle/input/birdclef-2025/train_audio/" + combined_df.iloc[0]["filename"]
vec = extract_binary_features(path, percentile=76.7)
print(vec.shape) 
print(vec.sum()) 


import seaborn as sns

plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title('Positive vs Negative Samples (After Processing)')
plt.xlabel('Class (0=Negative, 1=Positive)')
plt.ylabel('Count')
plt.show()


import os
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

# Storage
X = []
y = []

# Full base path to audio files
AUDIO_BASE = "/kaggle/input/birdclef-2025/train_audio/"

for _, row in tqdm(combined_df.iterrows(), total=len(combined_df)):
    full_path = os.path.join(AUDIO_BASE, row['filename'])

    try:
        # Load audio
        y_raw, sr = sf.read(full_path)
        samples_per_chunk = sr * 5  # 5 seconds
        num_chunks = len(y_raw) // samples_per_chunk

        if num_chunks == 0:
            continue  # skip if too short

        # Extract binary vectors per chunk
        chunk_features = []

        for i in range(num_chunks):
            start = i * samples_per_chunk
            end = start + samples_per_chunk
            chunk = y_raw[start:end]

            # Extract features for chunk
            S = librosa.stft(chunk, n_fft=1024, hop_length=512)
            S_power = np.abs(S) ** 2
            S_db = librosa.power_to_db(S_power, ref=np.max)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
            bin_edges = np.linspace(0, 16000, 3201)
            binary_vector = np.zeros(3200, dtype=int)
            energy_per_freq = S_db.max(axis=1)
            adaptive_threshold = np.percentile(energy_per_freq, 85)

            for j in range(3200):
                f_start = bin_edges[j]
                f_end = bin_edges[j+1]
                bin_mask = (freqs >= f_start) & (freqs < f_end)
                if np.any(bin_mask):
                    max_energy = np.max(energy_per_freq[bin_mask])
                    if max_energy > adaptive_threshold:
                        binary_vector[j] = 1

            chunk_features.append(binary_vector)

        # Now aggregate across all chunks
        chunk_features = np.array(chunk_features)

        # median aggregation across chunks
        final_vector = np.median(chunk_features, axis=0)

        # binarize again (keep 0 or 1 only)
        final_vector = (final_vector > 0.5).astype(int)

        # Save final vector
        X.append(final_vector)
        y.append(row['label'])

    except Exception as e:
        print(f"â�Œ Failed for {row['filename']}: {e}")


X = np.stack(X)
y = np.array(y)

print("âœ… Final Feature Matrix shape:", X.shape)
print("âœ… Final Labels shape:", y.shape)


from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

f1_scores_rf = []
auc_scores_rf = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nğŸ”µ Fold {fold+1}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    clf = XGBClassifier(
        n_estimators=500,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    f1_scores_rf.append(f1)
    auc_scores_rf.append(auc)

    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC:  {auc:.4f}")

print(f"\nğŸ“Š Average F1 Score: {np.mean(f1_scores_rf):.4f}")
print(f"ğŸ“ˆ Average AUC-ROC:  {np.mean(auc_scores_rf):.4f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.bar(range(1, len(auc_scores_rf)+1), auc_scores_rf, color='skyblue')
plt.ylim(0.75,0.85)
plt.xlabel('Fold')
plt.ylabel('AUC Score')
plt.title('Fold-wise AUC Scores (XGB)')
plt.grid(axis='y')
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_val, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix on Validation Fold")
plt.show()


from sklearn.utils import resample
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Target bird: compau
bird = 'compau'

# Positive and negative split
pos_df = df[df['primary_label'] == bird].copy()
pos_df['label'] = 1

neg_df = df[df['primary_label'].isin(top10) & (df['primary_label'] != bird)].copy()
neg_df['label'] = 0

# Downsample negative class to match
neg_df_balanced = resample(neg_df, replace=False, n_samples=len(pos_df), random_state=42)

combined_df = pd.concat([pos_df, neg_df_balanced]).sample(frac=1, random_state=42)

# --- Feature extraction ---
X = []
y = []

for _, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc=f"Processing {bird}"):
    full_path = os.path.join(AUDIO_BASE, row['filename'])
    try:
        y_raw, sr = sf.read(full_path)
        samples_per_chunk = sr * 5
        num_chunks = len(y_raw) // samples_per_chunk
        if num_chunks == 0: continue

        chunk_features = []
        for i in range(num_chunks):
            chunk = y_raw[i*samples_per_chunk : (i+1)*samples_per_chunk]
            S = librosa.stft(chunk, n_fft=1024, hop_length=512)
            S_db = librosa.power_to_db(np.abs(S)**2, ref=np.max)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
            bin_edges = np.linspace(0, 16000, 3201)
            binary_vector = np.zeros(3200, dtype=int)
            energy_per_freq = S_db.max(axis=1)
            adaptive_threshold = np.percentile(energy_per_freq, 76.7)

            for j in range(3200):
                bin_mask = (freqs >= bin_edges[j]) & (freqs < bin_edges[j+1])
                if np.any(bin_mask) and np.max(energy_per_freq[bin_mask]) > adaptive_threshold:
                    binary_vector[j] = 1
            chunk_features.append(binary_vector)

        # Aggregate
        chunk_features = np.array(chunk_features)
        median_vector = np.median(chunk_features, axis=0)
        final_vector = (median_vector > 0.5).astype(int)

        X.append(final_vector)
        y.append(row['label'])

    except Exception as e:
        print(f"â�Œ {row['filename']} failed: {e}")

X = np.stack(X)
y = np.array(y)

# --- Train + CV ---
f1s, aucs = [], []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    clf = XGBClassifier(
        n_estimators=500,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_val)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)

    f1s.append(f1)
    aucs.append(auc)

print(f"\nğŸ“Š compau | F1 Score: {np.mean(f1s):.4f} | AUC-ROC: {np.mean(aucs):.4f}")


from sklearn.utils import resample
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Target bird: trokin
bird = 'trokin'

# Positive and negative split
pos_df = df[df['primary_label'] == bird].copy()
pos_df['label'] = 1

neg_df = df[df['primary_label'].isin(top10) & (df['primary_label'] != bird)].copy()
neg_df['label'] = 0

# Downsample negative class to match
neg_df_balanced = resample(neg_df, replace=False, n_samples=len(pos_df), random_state=42)

combined_df = pd.concat([pos_df, neg_df_balanced]).sample(frac=1, random_state=42)

# --- Feature extraction ---
X = []
y = []

for _, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc=f"Processing {bird}"):
    full_path = os.path.join(AUDIO_BASE, row['filename'])
    try:
        y_raw, sr = sf.read(full_path)
        samples_per_chunk = sr * 5
        num_chunks = len(y_raw) // samples_per_chunk
        if num_chunks == 0: continue

        chunk_features = []
        for i in range(num_chunks):
            chunk = y_raw[i*samples_per_chunk : (i+1)*samples_per_chunk]
            S = librosa.stft(chunk, n_fft=1024, hop_length=512)
            S_db = librosa.power_to_db(np.abs(S)**2, ref=np.max)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
            bin_edges = np.linspace(0, 16000, 3201)
            binary_vector = np.zeros(3200, dtype=int)
            energy_per_freq = S_db.max(axis=1)
            adaptive_threshold = np.percentile(energy_per_freq, 76.7)

            for j in range(3200):
                bin_mask = (freqs >= bin_edges[j]) & (freqs < bin_edges[j+1])
                if np.any(bin_mask) and np.max(energy_per_freq[bin_mask]) > adaptive_threshold:
                    binary_vector[j] = 1
            chunk_features.append(binary_vector)

        # Aggregate
        chunk_features = np.array(chunk_features)
        median_vector = np.median(chunk_features, axis=0)
        final_vector = (median_vector > 0.5).astype(int)

        X.append(final_vector)
        y.append(row['label'])

    except Exception as e:
        print(f"â�Œ {row['filename']} failed: {e}")

X = np.stack(X)
y = np.array(y)

# --- Train + CV ---
f1s, aucs = [], []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    clf = XGBClassifier(
        n_estimators=500,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_val)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)

    f1s.append(f1)
    aucs.append(auc)

print(f"\nğŸ“Š trokin | F1 Score: {np.mean(f1s):.4f} | AUC-ROC: {np.mean(aucs):.4f}")


import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Settings
AUDIO_BASE = "/kaggle/input/birdclef-2025/train_audio/"
df = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")
top10 = ['grekis', 'compau', 'trokin', 'roahaw', 'banana', 'whtdov', 'socfly1', 'yeofly1', 'bobfly1', 'wbwwre1']
target_birds = ['grekis', 'compau', 'trokin']
models = {}

# Main loop
for bird in target_birds:
    print(f"\n==================== ğŸ�¦ Training for '{bird}' ====================")

    # Create labels
    pos_df = df[df['primary_label'] == bird].copy()
    pos_df['label'] = 1
    neg_df = df[df['primary_label'].isin(top10) & (df['primary_label'] != bird)].copy()
    neg_df['label'] = 0

    # Balance
    neg_df_balanced = resample(neg_df, replace=False, n_samples=min(len(pos_df), len(neg_df)), random_state=42)
    combined_df = pd.concat([pos_df, neg_df_balanced]).sample(frac=1, random_state=42)

    X, y = [], []

    for _, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc=f"Extracting {bird}"):
        full_path = os.path.join(AUDIO_BASE, row['filename'])
        try:
            y_raw, sr = sf.read(full_path)
            samples_per_chunk = sr * 5
            num_chunks = len(y_raw) // samples_per_chunk
            if num_chunks == 0:
                continue

            chunk_features = []
            for i in range(num_chunks):
                chunk = y_raw[i*samples_per_chunk : (i+1)*samples_per_chunk]
                S = librosa.stft(chunk, n_fft=1024, hop_length=512)
                S_db = librosa.power_to_db(np.abs(S)**2, ref=np.max)
                freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
                bin_edges = np.linspace(0, 16000, 3201)
                binary_vector = np.zeros(3200, dtype=int)
                energy_per_freq = S_db.max(axis=1)
                adaptive_threshold = np.percentile(energy_per_freq, 76.7)

                for j in range(3200):
                    bin_mask = (freqs >= bin_edges[j]) & (freqs < bin_edges[j+1])
                    if np.any(bin_mask) and np.max(energy_per_freq[bin_mask]) > adaptive_threshold:
                        binary_vector[j] = 1

                chunk_features.append(binary_vector)

            # Aggregate chunks
            median_vector = np.median(np.array(chunk_features), axis=0)
            final_vector = (median_vector > 0.5).astype(int)

            X.append(final_vector)
            y.append(row['label'])

        except Exception as e:
            print(f"â�Œ {row['filename']} failed: {e}")

    # Final dataset
    X = np.stack(X)
    y = np.array(y)

    # Train with Stratified 5-Fold CV
    f1s, aucs = [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = XGBClassifier(
            n_estimators=500,
            scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )

        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_val)[:, 1]
        y_pred = (y_proba > 0.5).astype(int)

        f1s.append(f1_score(y_val, y_pred))
        aucs.append(roc_auc_score(y_val, y_proba))

    print(f"ğŸ“Š {bird} | Avg F1 Score: {np.mean(f1s):.4f} | Avg AUC-ROC: {np.mean(aucs):.4f}")
    models[bird] = clf


results = {
    'grekis': {'f1': [0.7592], 'auc': [0.8308]},
    'compau': {'f1': [0.8360], 'auc': [0.9178]},
    'trokin': {'f1': [0.7841], 'auc': [0.8589]},
}


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# F1 + AUC per fold line plots
plt.figure(figsize=(12, 5))
for idx, metric in enumerate(['f1', 'auc']):
    plt.subplot(1, 2, idx+1)
    for bird in results:
        plt.plot(results[bird][metric], marker='o', label=bird)
    plt.title(f"{metric.upper()} Score")
    plt.xlabel("Fold" if len(results[bird][metric]) > 1 else "Single Fold")
    plt.ylabel(f"{metric.upper()}")
    plt.ylim(0.5, 1.0)
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.show()


# Barplot of average scores
bird_names = list(results.keys())
avg_f1 = [np.mean(results[bird]['f1']) for bird in bird_names]
avg_auc = [np.mean(results[bird]['auc']) for bird in bird_names]

df_plot = pd.DataFrame({
    'Bird': bird_names,
    'Avg F1': avg_f1,
    'Avg AUC-ROC': avg_auc
})

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.barplot(x='Bird', y='Avg F1', data=df_plot)
plt.ylim(0.5, 1.0)
plt.title('Average F1 Score')

plt.subplot(1, 2, 2)
sns.barplot(x='Bird', y='Avg AUC-ROC', data=df_plot)
plt.ylim(0.5, 1.0)
plt.title('Average AUC-ROC')
plt.tight_layout()
plt.show()



import os
import librosa
import numpy as np
import pandas as pd

# Paths
AUDIO_BASE_TEST = '/kaggle/input/birdclef-2025/test_soundscapes/'
class_labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))  # All 206 birds

# Submission storage
submission_rows = []

# Loop over test soundscapes
test_files = sorted([f for f in os.listdir(AUDIO_BASE_TEST) if f.endswith('.ogg')])
for file in test_files:
    full_path = os.path.join(AUDIO_BASE_TEST, file)
    signal, sr = librosa.load(full_path, sr=32000)
    samples_per_chunk = sr * 5

    for i in range(0, len(signal), samples_per_chunk):
        chunk = signal[i:i+samples_per_chunk]
        if len(chunk) < samples_per_chunk:
            continue  # skip short ones

        try:
            # Binary feature extraction
            S = librosa.stft(chunk, n_fft=1024, hop_length=512)
            S_power = np.abs(S) ** 2
            S_db = librosa.power_to_db(S_power, ref=np.max)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
            bin_edges = np.linspace(0, 16000, 3201)
            binary_vector = np.zeros(3200, dtype=int)
            energy_per_freq = S_db.max(axis=1)
            adaptive_threshold = np.percentile(energy_per_freq, 76.7)

            for j in range(3200):
                bin_mask = (freqs >= bin_edges[j]) & (freqs < bin_edges[j+1])
                if np.any(bin_mask) and np.max(energy_per_freq[bin_mask]) > adaptive_threshold:
                    binary_vector[j] = 1

            # Predict with models
            row_id = file.replace(".ogg", "") + f"_{(i//samples_per_chunk+1)*5}"
            row = [row_id]

            for bird in class_labels:
                if bird in models:
                    prob = models[bird].predict_proba(binary_vector.reshape(1, -1))[:, 1][0]
                    row.append(prob)
                else:
                    row.append(0.001)  # default prob for birds you didnâ€™t train

            submission_rows.append(row)

        except Exception as e:
            print(f"â�Œ Error processing chunk from {file}: {e}")

# Save submission
submission_df = pd.DataFrame(submission_rows, columns=['row_id'] + class_labels)
submission_df.to_csv("submission.csv", index=False)


import os
import librosa
import numpy as np
import pandas as pd

AUDIO_BASE_TEST = '/kaggle/input/birdclef-2025/test_soundscapes/'

# Class labels
class_labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))

# Your known target bird
target_bird = 'grekis'  # model trained for grekis detection

# Submission dataframe
submission_rows = []

# For each soundscape
test_soundscapes = [os.path.join(AUDIO_BASE_TEST, f) for f in sorted(os.listdir(AUDIO_BASE_TEST)) if f.endswith('.ogg')]

for soundscape_path in test_soundscapes:
    sig, sr = librosa.load(soundscape_path, sr=32000)

    samples_per_chunk = sr * 5

    for i in range(0, len(sig), samples_per_chunk):
        chunk = sig[i:i+samples_per_chunk]
        if len(chunk) < samples_per_chunk:
            continue  # skip short last chunk

        # extract binary features
        try:
            S = librosa.stft(chunk, n_fft=1024, hop_length=512)
            S_power = np.abs(S)**2
            S_db = librosa.power_to_db(S_power, ref=np.max)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
            bin_edges = np.linspace(0, 16000, 3201)
            binary_vector = np.zeros(3200, dtype=int)
            energy_per_freq = S_db.max(axis=1)
            adaptive_threshold = np.percentile(energy_per_freq, 76.7)

            for j in range(3200):
                f_start = bin_edges[j]
                f_end = bin_edges[j+1]
                bin_mask = (freqs >= f_start) & (freqs < f_end)
                if np.any(bin_mask):
                    max_energy = np.max(energy_per_freq[bin_mask])
                    if max_energy > adaptive_threshold:
                        binary_vector[j] = 1

            # Predict
            pred_prob = clf_rf.predict_proba(binary_vector.reshape(1, -1))[:,1][0]

            # Build prediction row
            row_id = os.path.basename(soundscape_path).split('.')[0] + f'_{(i//samples_per_chunk+1)*5}'
            row = [row_id]

            for bird in class_labels:
                if bird == target_bird:
                    row.append(pred_prob)
                else:
                    row.append(0.001)  # small probability for all other birds

            submission_rows.append(row)

        except Exception as e:
            print(f"â�Œ Error in {soundscape_path}: {e}")

# Create dataframe
submission_df = pd.DataFrame(submission_rows, columns=['row_id'] + class_labels)

# Save
submission_df.to_csv('submission.csv', index=False)

