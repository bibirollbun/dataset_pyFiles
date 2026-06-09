import os
import numpy as np
import pandas as pd
import librosa
import torch
import dataclasses
from typing import Optional, Callable, Tuple, List
# Import data from files
train_csv = '/kaggle/input/birdclef-2025/train.csv'
test_data = '/kaggle/input/birdclef-2025/test_soundscapes'
submission = '/kaggle/input/birdclef-2025/sample_submission.csv'
taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'

# Define audio parameter class
@dataclasses.dataclass
class AudioParam:
    SR: int = 32_000  # Sample rate
    NFFT: int = 2048  # Number of FFT points
    NMEL: int = 128   # Number of Mel bands
    FMAX: int = 16_000 # Maximum frequency
    FMIN: int = 20   # Minimum frequency
    HOP_LENGTH: int = NFFT // 4  # Hop length

audio_param = AudioParam()

# Load sample_submission CSV to get bird class names
submission_csv = pd.read_csv(submission)
bird_classes = submission_csv.columns.drop('row_id').tolist()  # List of bird classes

file_names = [os.path.join(test_data, filepath) for filepath in os.listdir(test_data) if filepath.endswith(".ogg")]

# Use a single file for debugging.  This makes the matrix dimension calculations easier.
if len(file_names) == 0:
    file_names = [
        '/kaggle/input/birdclef-2025/train_soundscapes/H03_20230505_190000.ogg',
    ]


"""
define an audio processing pipeline to prepare the input data for the CNN model
"""

def pipeline(x: np.ndarray) -> np.ndarray:
    """
    Converts audio data to a mel spectrogram and then to a dB scale.
    """
    mels = librosa.feature.melspectrogram(
        y=x,
        sr=audio_param.SR,
        n_fft=audio_param.NFFT,
        n_mels=audio_param.NMEL,
        fmax=audio_param.FMAX,
        fmin=audio_param.FMIN,
        hop_length=audio_param.HOP_LENGTH,
    )
    db_map = librosa.power_to_db(mels, ref=np.max)
    db_map = (db_map + 80) / (80 + 1e-6)  # Normalize to [0, 1] - Added small constant
    if np.isnan(db_map).any():
        print('Warning: NaN values detected in db_map!')
        db_map = np.nan_to_num(db_map) #Replace with 0

    return db_map[None, :, :]  # Add a channel dimension (1, height, width)


def process_audio_segments(filepath: str) -> Tuple[List[torch.Tensor], List[str]]:
    """
    Load audio, split into 5-second segments, and apply preprocessing.

    Returns:
        List of tensors ready for model input and corresponding row_ids.
    """

    x, _ = librosa.load(filepath, sr=audio_param.SR)
    if x.size == 0:
        print(f"Warning: Audio file {filepath} is empty!")
        return [], []

    num_segments = int(np.floor(len(x) / audio_param.SR / 5))
    processed_segments = []
    row_ids = []

    for i in range(num_segments):
        start = i * audio_param.SR * 5
        end = (i + 1) * audio_param.SR * 5
        segment = x[start:end]

        # Convert to mel spectrogram
        segment = pipeline(segment)

        # Convert to tensor
        segment_tensor = torch.from_numpy(segment).float().unsqueeze(0)
        processed_segments.append(segment_tensor)

        filepath_name = os.path.basename(filepath).split(".")[0]
        row_id = f"{filepath_name}_{(i + 1) * 5}"
        row_ids.append(row_id)

    return processed_segments, row_ids



import torch.nn as nn


#  a simpler, randomly initialized CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self._to_linear = None
        self.fc1 = nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        
        if self._to_linear is None:
            self._to_linear = x.shape[1]
            if self._to_linear == 0:
               return torch.zeros((1, len(bird_classes)))
            self.fc1 = nn.Linear(self._to_linear, len(bird_classes))
        x = self.fc1(x)
        return x


model = SimpleCNN(num_classes=len(bird_classes)) # initialize the model
model.eval() # Set the model to evaluation mode.

# prediction fuction
@torch.no_grad()
def predict(filepath: str) -> Tuple[np.ndarray, List[str]]:

    segments, row_ids = process_audio_segments(filepath)
    if not segments:
        return np.array([]), []

    outputs = []
    
    for seg in segments:
        out = model(seg).sigmoid().cpu().numpy()
        outputs.append(out[0])

    return np.array(outputs), row_ids


import gc
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor


row_id = []
matrix = []

#Using a ThreadPoolExecutor to parallelize the predictions
with ThreadPoolExecutor(max_workers=6) as executor:
    for filepath_idx, (filepath) in enumerate(file_names):
        out, rid = predict(filepath)
        if len(rid) > 0:
            row_id.extend(rid)
            matrix.extend(out)
        else:
            print(f"Warning: No predictions generated for file: {filepath}")
        gc.collect()

    matrix = np.array(matrix).reshape(-1, len(bird_classes))
    row_id = np.array(row_id).reshape(-1, 1)
    matrix = np.hstack([row_id, matrix])

    # Create a Pandas DataFrame from the results.
    sub = pd.DataFrame(matrix, columns=["row_id", *bird_classes])
    sub.to_csv('submission.csv', index=False)
gc.collect()

