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


import matplotlib.pyplot as plt

# Load labels
df = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")

# Display basic info
print(df.shape)
print(df.head())


# Plot class distribution
labels = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
df[labels].mean().plot(kind='bar', title='Average Class Distribution')
plt.ylabel('Average Vote Probability')
plt.show()


# Load EEG file again
import pyarrow.parquet as pq

eeg_path = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/1000913311.parquet"
eeg_df = pq.read_table(eeg_path).to_pandas()

# See all available columns
print(eeg_df.columns.tolist())


import matplotlib.pyplot as plt

# Select a few EEG channels to visualize
channels = eeg_df.columns[:4]  # Take first 4 channels (e.g., 'Fp1', 'Fp2', 'F3', 'F4')

# Plot the EEG signals
eeg_df[channels].plot(figsize=(15, 5), title="EEG Channel Signals (Sample ID: 1000913311)")
plt.xlabel("Time step (5 ms each @200Hz)")
plt.ylabel("Signal amplitude (Î¼V)")
plt.grid(True)
plt.show()


import pandas as pd

# Load train.csv
df = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")

# Pick a sample eeg_id
eeg_id = 1000913311

# Get corresponding spectrogram_id(s)
spec_id = df[df["eeg_id"] == eeg_id]["spectrogram_id"].iloc[0]
print(f"SPECTROGRAM ID: {spec_id}")



import pyarrow.parquet as pq

# Construct the correct file path using spectrogram_id
spec_path = f"/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/{spec_id}.parquet"

# Read the file
spectrogram = pq.read_table(spec_path).to_pandas()

# Inspect shape
print(spectrogram.shape)
spectrogram.head()


import matplotlib.pyplot as plt
import numpy as np

spec_data = spectrogram.drop(columns=["time"]).values  # shape: (300, 400)

img = spec_data.reshape(300, 20, 20)  # shape: (time, freq, channels)
img = img.transpose(1, 0, 2)  # now shape = (freq, time, channels)

import matplotlib.pyplot as plt

channel_idx = 10
plt.figure(figsize=(10, 6))
plt.imshow(img[:, :, channel_idx], cmap="magma", aspect="auto")
plt.title(f"Spectrogram of Channel {channel_idx}")
plt.xlabel("Time Steps")
plt.ylabel("Frequency Bins")
plt.colorbar(label="Power")
plt.show()


eeg_id = 1000913311
label_row = df[df["eeg_id"] == eeg_id]
print(label_row[labels])


import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Load EEG file
eeg_path = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/1000913311.parquet"
eeg_df = pq.read_table(eeg_path).to_pandas()

# Basic stats for each EEG channel
features = {}
for col in eeg_df.columns:
    signal = eeg_df[col].values
    features[f"{col}_mean"] = np.mean(signal)
    features[f"{col}_std"] = np.std(signal)
    features[f"{col}_skew"] = skew(signal)
    features[f"{col}_kurt"] = kurtosis(signal)


from scipy.signal import welch

def bandpower(signal, sf, band, window_sec=2):
    band = np.asarray(band)
    freqs, psd = welch(signal, sf, nperseg=window_sec * sf)
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapz(psd[idx_band], freqs[idx_band])

sf = 200  # Sampling frequency (Hz)
bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}

for col in eeg_df.columns:
    for band_name, band_range in bands.items():
        bp = bandpower(eeg_df[col].values, sf, band_range)
        features[f"{col}_{band_name}_power"] = bp


features_df = pd.DataFrame([features])
features_df.head()


from scipy.stats import skew, kurtosis
from scipy.signal import welch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def extract_features_from_eeg(file_path, sf=200):
    df = pq.read_table(file_path).to_pandas()
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
    
    def bandpower(signal, sf, band):
        freqs, psd = welch(signal, sf, nperseg=sf*2)
        idx = (freqs >= band[0]) & (freqs <= band[1])
        return np.trapz(psd[idx], freqs[idx])

    features = {}
    for col in df.columns:
        sig = df[col].values
        features[f"{col}_mean"] = np.mean(sig)
        features[f"{col}_std"] = np.std(sig)
        features[f"{col}_skew"] = skew(sig)
        features[f"{col}_kurt"] = kurtosis(sig)
        for name, rng in bands.items():
            features[f"{col}_{name}_power"] = bandpower(sig, sf, rng)
    
    return pd.Series(features)


import lightgbm as lgb

def build_train_lgbm(X, y):
    models = {}
    for i, col in enumerate(y.columns):
        m = lgb.LGBMRegressor(n_estimators=200)
        m.fit(X, y[col])
        models[col] = m
    return models

def predict_with_lgbm(models, X):
    preds = np.stack([models[col].predict(X) for col in models], axis=1)
    return preds / preds.sum(axis=1, keepdims=True)  # Normalize


import torch

def prepare_spectrogram_tensor(spectrogram_df):
    spec = spectrogram_df.drop(columns=["time"]).values
    spec = spec.reshape(300, 20, 20).transpose(2, 0, 1)  # (channels=20, time=300, freq=20)
    return torch.tensor(spec).float().unsqueeze(0)  # shape: [1, 20, 300, 20]


import torch.nn as nn

class SpectrogramCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 75 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def blend_predictions(cnn_preds, lgbm_preds, w1=0.6, w2=0.4):
    blend = w1 * cnn_preds + w2 * lgbm_preds
    return blend / blend.sum(axis=1, keepdims=True)


def main():
    # Load sample EEG & Spectrogram file paths
    eeg_path = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/1000913311.parquet"
    spec_path = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/1000913311.parquet"

    # === EEG Tabular Features ===
    X = pd.DataFrame([extract_features_from_eeg(eeg_path)])
    y = pd.DataFrame([[1, 0, 0, 0, 0, 0]], columns=["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"])  # dummy target
    lgbm_models = build_train_lgbm(X, y)
    lgbm_preds = predict_with_lgbm(lgbm_models, X)

    # === CNN Prediction ===
    spec_df = pq.read_table(spec_path).to_pandas()
    cnn_input = prepare_spectrogram_tensor(spec_df)
    cnn_model = SpectrogramCNN()
    with torch.no_grad():
        cnn_preds = cnn_model(cnn_input).numpy()

    # === Blend ===
    ensemble = blend_predictions(cnn_preds, lgbm_preds)
    print("Blended Prediction:", ensemble)


import os
from tqdm import tqdm

def batch_extract_eeg_features(eeg_dir, limit=None):
    feature_list = []
    ids = []
    
    files = os.listdir(eeg_dir)
    if limit: files = files[:limit]
    
    for f in tqdm(files):
        eeg_id = int(f.replace(".parquet", ""))
        try:
            path = os.path.join(eeg_dir, f)
            features = extract_features_from_eeg(path)
            feature_list.append(features)
            ids.append(eeg_id)
        except Exception as e:
            print(f"Failed on {f}: {e}")
    
    X = pd.DataFrame(feature_list)
    X["eeg_id"] = ids
    return X.set_index("eeg_id")


train_df = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")

# Votes as labels
label_cols = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
labels = train_df.groupby("eeg_id")[label_cols].mean()


features_df = batch_extract_eeg_features("/kaggle/input/hms-harmful-brain-activity-classification/train_eegs", limit=5000)

# Join features with labels
data = features_df.join(labels, how="inner")

# Split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(data.drop(columns=label_cols), data[label_cols], test_size=0.2)

# Train LGBM
lgbm_models = build_train_lgbm(X_train, y_train)
lgbm_preds_val = predict_with_lgbm(lgbm_models, X_val)


def batch_predict_cnn(cnn_model, spectro_dir, ids):
    cnn_model.eval()
    preds = []
    with torch.no_grad():
        for eid in tqdm(ids):
            try:
                df = pq.read_table(f"{spectro_dir}/{eid}.parquet").to_pandas()
                tensor = prepare_spectrogram_tensor(df)
                pred = cnn_model(tensor).numpy()[0]
                preds.append(pred)
            except:
                preds.append([1/6] * 6)  # fallback: uniform
    return np.array(preds)


import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrogramCNN(nn.Module):
    def __init__(self):
        super(SpectrogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 6)  # 6 classes for 6 EEG pattern votes

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))   # [B, 32, 64, 64]
        x = self.pool2(F.relu(self.conv2(x)))   # [B, 64, 32, 32]
        x = x.view(x.size(0), -1)               # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


# Instantiate model
cnn_model = SpectrogramCNN()

# Load trained weights (optional)
# cnn_model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))

cnn_model.eval()  # Important for inference


# This assumes cnn_model is defined and in eval mode
def batch_predict_cnn(model, spectrogram_dir, eeg_ids):
    preds = []
    model.eval()

    for eeg_id in eeg_ids:
        try:
            path = f"{spectrogram_dir}/{eeg_id}.parquet"
            df = pd.read_parquet(path)

            # Prepare tensor
            img = df.drop(columns="time").values.reshape(3, 128, 128)
            tensor = torch.tensor(img).float().unsqueeze(0)  # (1, 3, 128, 128)

            with torch.no_grad():
                output = model(tensor)
                preds.append(output.cpu().numpy()[0])

        except Exception as e:
            print(f"Missing or broken file: {eeg_id}, using uniform prediction")
            preds.append([1/6]*6)  # fallback if spectrogram missing

    return np.array(preds)



cnn_preds_val = batch_predict_cnn(
    cnn_model, 
    "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms", 
    list(X_val.index)
)


ensemble_val = blend_predictions(cnn_preds_val, lgbm_preds_val)


from scipy.special import rel_entr

def kl_divergence(y_true, y_pred):
    return np.mean(np.sum(rel_entr(y_true, y_pred), axis=1))

kl = kl_divergence(y_val.values, ensemble_val)
print(f"Validation KL Divergence: {kl:.5f}")



# Load test EEGs
test_df = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
test_ids = test_df["eeg_id"].unique()

# Extract features
X_test = batch_extract_eeg_features("/kaggle/input/hms-harmful-brain-activity-classification/test_eegs", limit=None)
X_test = X_test.loc[X_test.index.isin(test_ids)]

# Predict
lgbm_test_preds = predict_with_lgbm(lgbm_models, X_test)
cnn_test_preds = batch_predict_cnn(cnn_model, "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms", list(X_test.index))

# Blend
test_preds = blend_predictions(cnn_test_preds, lgbm_test_preds)


submission = pd.DataFrame(test_preds, columns=label_cols)
submission.insert(0, "eeg_id", X_test.index)
submission.to_csv("submission.csv", index=False)




