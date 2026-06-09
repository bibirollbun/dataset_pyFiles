import numpy as np
import pandas as pd
import torchaudio
import os
from tqdm.auto import tqdm

K = 16000
IN_DIR = "/kaggle/input/audio-demixing-aicc-round-2/"

def load_signal(path, K):
    wav, _ = torchaudio.load(path)
    arr = wav[0].numpy()
    if len(arr) >= K:
        return arr[:K].astype(np.float32)
    else:
        return np.pad(arr, (0, K - len(arr)), mode='constant').astype(np.float32)

# Load CSVs
train = pd.read_csv("/kaggle/input/audio-demixing-aicc-round-2/train.csv")
test = pd.read_csv("/kaggle/input/audio-demixing-aicc-round-2/test.csv")

# Load raw training signals
print("Loading training signals...")
X_train = np.vstack([load_signal(os.path.join(IN_DIR, p), K) for p in tqdm(train["file"], desc="X_train")])
Y1_train = np.vstack([load_signal(os.path.join(IN_DIR, p), K) for p in tqdm(train["sig_1"], desc="Y1_train")])
Y2_train = np.vstack([load_signal(os.path.join(IN_DIR, p), K) for p in tqdm(train["sig_2"], desc="Y2_train")])

# Load raw test signals
print("Loading test signals...")
if "sig_1" in test.columns:
    X_test = np.vstack([load_signal(os.path.join(IN_DIR, p), K) for p in tqdm(test["sig_1"], desc="X_test")])
else:
    X_test = np.tile(X_train[0], (len(test), 1))

test_IDs = test["ID"].astype(str).values


from sklearn.linear_model import Ridge

# Flatten for single-step regression
X_flat = X_train.reshape(-1, 1)
Y1_flat = Y1_train.reshape(-1)
Y2_flat = Y2_train.reshape(-1)

# Train Ridge regressors
print("Training Ridge models...")
m1 = Ridge(alpha=1.0)
m2 = Ridge(alpha=1.0)
m1.fit(X_flat, Y1_flat)
m2.fit(X_flat, Y2_flat)

# Predict on raw test signals
X_test_flat = X_test.reshape(-1, 1)
pred1_flat = m1.predict(X_test_flat)
pred2_flat = m2.predict(X_test_flat)

pred1 = pred1_flat.reshape(len(X_test), K).astype(np.float32)
pred2 = pred2_flat.reshape(len(X_test), K).astype(np.float32)


import base64
import pandas as pd
from tqdm.auto import tqdm
import csv

def encode_array_base85(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr.astype(np.float32).ravel())
    return base64.b85encode(arr.tobytes()).decode("ascii")

# Encode predictions
rows = []
for i in tqdm(range(len(test_IDs)), desc="Encoding rows"):
    b1 = encode_array_base85(pred1[i])
    b2 = encode_array_base85(pred2[i])
    rows.append((str(test_IDs[i]), b1, b2))

# Build submission DataFrame
submission = pd.DataFrame(rows, columns=["ID", "sig_1", "sig_2"])
submission.to_csv("/kaggle/working/submission.csv", index=False, quoting=csv.QUOTE_ALL)
print("✓ submission.csv written")

