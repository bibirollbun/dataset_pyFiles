import os
import warnings
import numpy as np
import pandas as pd
import torchaudio
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


RND = 42
OUTPUT_DIR = '/kaggle/input/is-that-audio-aicc-round-1-2'
TARGET_SR = 16000
N_MFCC = 40
MAX_EXAMPLES = None

# Paths
train_csv = os.path.join(OUTPUT_DIR, 'train.csv')
test_csv  = os.path.join(OUTPUT_DIR, 'test.csv')
submission_out = "/kaggle/working/submission.csv"

# Load metadata
train_df = pd.read_csv(train_csv)
test_df  = pd.read_csv(test_csv)

if MAX_EXAMPLES is not None:
    train_df = train_df.iloc[:MAX_EXAMPLES].reset_index(drop=True)
    test_df  = test_df.iloc[:MAX_EXAMPLES].reset_index(drop=True)

# Prepare transforms
mfcc_transform = torchaudio.transforms.MFCC(sample_rate=TARGET_SR, n_mfcc=N_MFCC, melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": 64})

def load_audio(path, target_sr=TARGET_SR):
    """Load audio, convert to mono, and resample to target_sr if necessary.
    Returns a float32 torch.Tensor of shape (1, samples) and the sample rate.
    """
    waveform, sr = torchaudio.load(path)  # waveform: (channels, samples)

    waveform = waveform.to(torch.float32)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    return waveform, sr

def extract_mfcc_vector(path):
    """Return a numpy vector of length N_MFCC by computing MFCC and mean-pooling over time.
    If anything fails, return a zero vector and log a warning."""
    try:
        path = os.path.join(OUTPUT_DIR, path.lstrip("./"))
        
        if not os.path.isfile(path):
            warnings.warn(f'Audio file not found: {path}')
            return np.zeros(N_MFCC, dtype=np.float32)
        waveform, sr = load_audio(path, TARGET_SR)

        with torch.no_grad():
            mfcc = mfcc_transform(waveform)  # (n_mfcc, time)
        mfcc_np = mfcc.squeeze().cpu().numpy()

        if mfcc_np.ndim == 1:
            vec = np.pad(mfcc_np, (0, max(0, N_MFCC - mfcc_np.shape[0])), 'constant')[:N_MFCC]
            return vec.astype(np.float32)
        vec = mfcc_np.mean(axis=-1)
        if vec.shape[0] != N_MFCC:
            vec = np.pad(vec, (0, max(0, N_MFCC - vec.shape[0])), 'constant')[:N_MFCC]

        return vec.astype(np.float32)
    except Exception as e:
        warnings.warn(f'Failed to extract MFCC for {path}: {e}')
        return np.zeros(N_MFCC, dtype=np.float32)

# --- Extract features for train ---
print("Extracting MFCCs for train set...")
train_paths = train_df['path'].tolist()
X_train = np.stack([extract_mfcc_vector(p) for p in train_paths])
y_train = train_df['label'].values
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# --- Extract features for test ---
print("Extracting MFCCs for test set...")
test_paths = test_df['path'].tolist()
X_test = np.stack([extract_mfcc_vector(p) for p in test_paths])
print(f"X_test shape: {X_test.shape}")

# --- Label encode and train SVC ---
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(class_weight='balanced', random_state=RND))
])

print("Training SVC baseline...")
model.fit(X_train, y_train_enc)

# --- Predict and write submission ---
y_test_pred_enc = model.predict(X_test)
y_test_pred = le.inverse_transform(y_test_pred_enc)

submission = pd.DataFrame({'id': test_df['id'].values, 'label': y_test_pred})
submission.to_csv(submission_out, index=False)
print("✅ submission.csv created at:", submission_out)
print(submission.head())


