#!pip install torch
#!pip install torchaudio
#!pip install scikit-learn


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import librosa
import numpy as np
import pandas as pd
import gc
import dataclasses
import torchaudio
import traceback
from pathlib import Path
from typing import Optional, Callable, Tuple, List
from torchaudio.transforms import Resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader


test_directory = "/kaggle/input/birdclef-2025/test_soundscapes"
submission = "/kaggle/input/birdclef-2025/sample_submission.csv"
train_file = "/kaggle/input/birdclef-2025/train.csv"
taxonomy_file = "/kaggle/input/birdclef-2025/taxonomy.csv"

@dataclasses.dataclass
class AudioParameters:
    sample_rate: int = 32000
    max_freq: int = 16000
    min_freq: int = 20

params = AudioParameters()

submission_df = pd.read_csv(submission)
index_to_class = submission_df.columns.drop("row_id").tolist()
class_to_index = {label: idx for idx, label in enumerate(index_to_class)}
available_files = set(os.listdir(test_directory))
submission_basenames = set(x.split("_")[0] for x in submission_df["row_id"])
file_paths = [
    os.path.join(test_directory, fname)
    for fname in os.listdir(test_directory)
    if Path(fname).stem in submission_basenames and fname.endswith(".ogg")
]


class CNNmodel(nn.Module):
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        temp = torch.zeros(1, 1, 128, 313)
        with torch.no_grad():
            x = self._forward_features(temp)
        self.fc1 = nn.Linear(x.shape[1], num_classes)

    def _forward_features(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = self.fc1(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNmodel(num_classes=len(index_to_class)).to(device)


mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=params.sample_rate,
    n_fft=1024,
    hop_length=512,
    n_mels=128
).to(device)

@torch.no_grad()
def predict(model, file_paths, device, chunk_size=5.0, sample_rate=32000):
    model.eval()
    predictions = []
    row_ids = []

    for file_path in file_paths:
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            print(f"Could not load {file_path}: {e}")
            continue

        if sr != sample_rate:
            waveform = Resample(sr, sample_rate)(waveform)

        total_samples = waveform.shape[1]
        step = int(chunk_size * sample_rate)

        for start in range(0, total_samples, step):
            end = start + step
            if end > total_samples:
                break

            chunk = waveform[:, start:end].to(device)
            spectrogram = mel_transform(chunk)
            spectrogram = spectrogram.log2().clamp(min=-10)
            spectrogram = spectrogram.unsqueeze(0)

            output = model(spectrogram)
            prob = torch.sigmoid(output).cpu().numpy()

            seconds = int(start / sample_rate)
            row_id = f"{Path(file_path).stem}_{seconds}"
            row_ids.append(row_id)
            predictions.append(prob.squeeze())

    return np.array(predictions), row_ids


submission_ids = []
matrix = []

if not file_paths:
    print("No test files found. Returning original sample submission.")
else:
    with ThreadPoolExecutor(max_workers=4) as executor:
        for audio_file in file_paths:
            preds, ids = predict(model, [audio_file], device)
            if ids:
                submission_ids.extend(ids)
                matrix.extend(preds)
            gc.collect()

if matrix:
    pred_df = pd.DataFrame(
        np.hstack([np.array(submission_ids).reshape(-1, 1), np.array(matrix).reshape(-1, len(index_to_class))]),
        columns=["row_id"] + index_to_class
    )
    pred_df[index_to_class] = pred_df[index_to_class].astype(float).round(6)

    for i, row in pred_df.iterrows():
        if row["row_id"] in submission_df["row_id"].values:
            submission_df.loc[submission_df["row_id"] == row["row_id"], index_to_class] = row[index_to_class]
else:
    print("No predictions generated. Filling with zeros.")
    submission_df[index_to_class] = 0.0

assert submission_df.shape == pd.read_csv(submission).shape, "Submission shape mismatch"
submission_df.to_csv("submission.csv", index=False)
print("Final submission shape:", submission_df.shape)
print(submission_df.head())

