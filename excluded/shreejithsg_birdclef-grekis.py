# ===============================
# ğŸ“¦ Imports
# ===============================

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt


# ===============================
# ğŸ“š Settings
# ===============================

AUDIO_BASE_TRAIN = '/kaggle/input/birdclef-2025/train_audio/'
AUDIO_BASE_TEST = '/kaggle/input/birdclef-2025/test_soundscapes/'

# Top 10 birds used
top10 = ['grekis', 'compau', 'trokin', 'roahaw', 'banana', 'whtdov', 'socfly1', 'yeofly1', 'bobfly1', 'wbwwre1']

# Full list of class labels (all 206 birds for submission)
class_labels = sorted(os.listdir(AUDIO_BASE_TRAIN))


# ===============================
# ğŸ”¥ Feature Extraction Function
# ===============================

def extract_binary_features_from_chunk(chunk, percentile=76.7):
    sr = 32000
    S = librosa.stft(chunk, n_fft=1024, hop_length=512)
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


# ===============================
# ğŸ�—ï¸� Build Training Dataset
# ===============================

print("ğŸ”µ Building training dataset...")

train_meta = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')
# DO NOT TOUCH train_meta['filename']


# Create Positive and Negative Samples
pos_df = train_meta[train_meta['primary_label'] == 'grekis'].copy()
pos_df['label'] = 1

neg_df = train_meta[(train_meta['primary_label'].isin(top10)) & (train_meta['primary_label'] != 'grekis')].copy()
neg_df['label'] = 0

# Balance the classes
from sklearn.utils import resample
neg_df_balanced = resample(neg_df, replace=False, n_samples=len(pos_df), random_state=42)

combined_df = pd.concat([pos_df, neg_df_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)


# ===============================
# ğŸ”¥ Extract Features for Training
# ===============================

X = []
y = []

print("ğŸ”µ Extracting training features...")

for _, row in tqdm(combined_df.iterrows(), total=len(combined_df)):
    path = os.path.join(AUDIO_BASE_TRAIN, row['filename'])

    try:
        y_raw, sr = sf.read(path)
        samples_per_chunk = sr * 5
        num_chunks = len(y_raw) // samples_per_chunk

        if num_chunks == 0:
            continue

        chunk_features = []
        for i in range(num_chunks):
            start = i * samples_per_chunk
            end = start + samples_per_chunk
            chunk = y_raw[start:end]

            vec = extract_binary_features_from_chunk(chunk, percentile=76.7)
            chunk_features.append(vec)

        chunk_features = np.array(chunk_features)
        final_vector = np.median(chunk_features, axis=0)
        final_vector = (final_vector > 0.5).astype(int)

        X.append(final_vector)
        y.append(row['label'])

    except Exception as e:
        print(f"â�Œ Failed for {row['filename']}: {e}")

X = np.stack(X)
y = np.array(y)

print(f"âœ… Final Training Feature Shape: {X.shape}")
print(f"âœ… Final Training Labels Shape: {y.shape}")


# ===============================
# ğŸ“ˆ Train Random Forest with Cross Validation
# ===============================

print("\nğŸ”µ Training Random Forest...")

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

f1_scores_rf = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nğŸ”µ Fold {fold+1}/{n_splits}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    clf_rf = RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_val)

    f1 = f1_score(y_val, y_pred_rf)
    print(f"âœ… Fold {fold+1} F1-Score: {f1:.4f}")

    f1_scores_rf.append(f1)

avg_f1_rf = np.mean(f1_scores_rf)
print(f"\nğŸ“Š Average F1-Score across {n_splits} folds: {avg_f1_rf:.4f}")


# ===============================
# ğŸ§ª Predict on Test Soundscapes
# ===============================

print("\nğŸ”µ Predicting on test soundscapes...")

submission_rows = []

test_soundscapes = [os.path.join(AUDIO_BASE_TEST, f) for f in sorted(os.listdir(AUDIO_BASE_TEST)) if f.endswith('.ogg')]

for soundscape_path in tqdm(test_soundscapes, desc="Processing Test Soundscapes"):
    try:
        sig, sr = librosa.load(soundscape_path, sr=32000)
    except Exception as e:
        print(f"â�Œ Error reading {soundscape_path}: {e}")
        continue

    samples_per_chunk = sr * 5

    for i in range(0, len(sig), samples_per_chunk):
        chunk = sig[i:i+samples_per_chunk]
        if len(chunk) < samples_per_chunk:
            continue

        try:
            vec = extract_binary_features_from_chunk(chunk, percentile=76.7)
            vec = vec.reshape(1, -1)

            prob = clf_rf.predict_proba(vec)[0][1]  # Probability for grekis

            row_id = os.path.basename(soundscape_path).split('.')[0] + f'_{(i//samples_per_chunk+1)*5}'
            row = [row_id]

            for bird in class_labels:
                if bird == 'grekis':
                    row.append(prob)
                else:
                    row.append(0.001)  # Small probability for other birds

            submission_rows.append(row)

        except Exception as e:
            print(f"â�Œ Error processing chunk {i}: {e}")


# ===============================
# ğŸ’¾ Create Submission File
# ===============================

submission_df = pd.DataFrame(submission_rows, columns=['row_id'] + class_labels)
submission_df.to_csv('submission.csv', index=False)

print("âœ… Final submission.csv created!")

submission_df.head()

