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
import torch
import torch.nn as nn
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm.notebook import tqdm
import timm

# Config
class CFG:
    sample_rate = 32000
    duration = 5  # seconds
    n_mels = 128
    n_fft = 2048
    hop_length = 512
    num_classes = 206
    model_name = 'efficientnet_b0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_dir = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'


class EfficientNetFrozen(nn.Module):
    def __init__(self, model_name='efficientnet_b0', n_classes=206):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, in_chans=1, num_classes=0)
        self.classifier = nn.Linear(self.backbone.num_features, n_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


def audio_to_logmel(y, cfg):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels
    )
    logmel = librosa.power_to_db(mel)
    # Normalize per frequency bin
    logmel = (logmel - logmel.mean(axis=1, keepdims=True)) / (logmel.std(axis=1, keepdims=True) + 1e-6)
    return logmel


import numpy as np
import torch

def predict_on_soundscape_with_smoothing(
    file_path,
    model,
    cfg,
    smoothing_window: int = 5
):
    """
    Predict on one soundscape, then apply a moving‐average filter along the time axis
    to smooth per-class probabilities across adjacent chunks.

    Args:
      file_path: Path to the .ogg soundscape.
      model: Your trained torch.nn.Module.
      cfg: CFG object (contains sample_rate, duration, etc.).
      smoothing_window: Number of chunks over which to average. Must be an odd integer.
                        (e.g. 3 => average over [i-1, i, i+1].)
    Returns:
      row_ids: List of strings (e.g. "soundscape_5", "soundscape_10", …)
      smoothed_preds: List of 1D NumPy arrays (length=num_classes), one per chunk,
                      after smoothing.
    """
    # Load entire audio (mono)
    y, _ = librosa.load(file_path, sr=cfg.sample_rate)
    chunk_size = cfg.duration * cfg.sample_rate
    num_chunks = len(y) // chunk_size

    raw_preds = []   # will store one (num_classes,) array per chunk
    stem = Path(file_path).stem

    # 1) Collect raw predictions for each chunk
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = y[start:end]

        # If last chunk is shorter (shouldn’t happen if len(y)//chunk_size is exact),
        # pad with zeros
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        # Convert to log‐mel, then tensor → model → sigmoid
        logmel = audio_to_logmel(chunk, cfg)
        tensor = (
            torch.tensor(logmel)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(cfg.device)
        )

        with torch.no_grad():
            output = model(tensor)          # shape: (1, num_classes)
            prob   = torch.sigmoid(output)  # shape: (1, num_classes)
            prob   = prob.cpu().numpy().squeeze()  # → (num_classes,)
        raw_preds.append(prob)

    # If there are no chunks (very short file), return empty
    if len(raw_preds) == 0:
        return [], []

    raw_preds = np.stack(raw_preds, axis=0)  
    # raw_preds shape: (num_chunks, num_classes)

    # 2) Build a uniform (moving‐average) kernel of length smoothing_window
    if smoothing_window < 1 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be a positive odd integer")
    kernel = np.ones(smoothing_window, dtype=np.float32) / smoothing_window

    # 3) Apply 1D convolution (per class) along the “time” axis.
    #    We pad at both ends with 'edge' (i.e. repeat the first/last) so that the
    #    smoothed array has the same length = num_chunks.
    pad_len = smoothing_window // 2
    smoothed_preds = np.empty_like(raw_preds)

    # For each class idx, convolve its 1D time series with the uniform kernel.
    for cls in range(raw_preds.shape[1]):
        series = raw_preds[:, cls]  # shape: (num_chunks,)
        # pad with edge values so that index 0 uses series[0], index -1 uses series[-1], etc.
        padded = np.pad(series, (pad_len, pad_len), mode="edge")
        conved = np.convolve(padded, kernel, mode="valid")  
        # After 'valid', conved.shape = (num_chunks,)
        smoothed_preds[:, cls] = conved

    # 4) Build row_ids and turn the smoothed_preds back into a list of arrays
    row_ids = []
    smoothed_list = []
    for i in range(num_chunks):
        timestamp = (i + 1) * cfg.duration
        row_id = f"{stem}_{timestamp}"
        row_ids.append(row_id)
        smoothed_list.append(smoothed_preds[i])

    return row_ids, smoothed_list



def generate_submission_with_smoothing(model, cfg, smoothing_window=5):
    model.eval()
    test_files = list(Path(cfg.test_dir).glob("*.ogg"))

    all_row_ids = []
    all_preds   = []

    for file_path in tqdm(test_files):
        # Use the "with_smoothing" version here:
        row_ids, preds = predict_on_soundscape_with_smoothing(
            file_path, model, cfg, smoothing_window=smoothing_window
        )
        all_row_ids.extend(row_ids)
        all_preds.extend(preds)

    # Build DataFrame just like before, but using smoothed preds
    pred_df = pd.DataFrame(
        all_preds, 
        columns=[f"class_{i}" for i in range(cfg.num_classes)]
    )
    pred_df.insert(0, "row_id", all_row_ids)

    sample = pd.read_csv(cfg.submission_csv)
    pred_df = pred_df.set_index("row_id")
    sample  = sample.set_index("row_id")
    final   = sample.copy()
    final.loc[pred_df.index] = pred_df
    final   = final.reset_index()
    final.to_csv("submission.csv", index=False)
    print("✅ submission_smoothed.csv created.")



# Load model
model = EfficientNetFrozen(model_name=CFG.model_name, n_classes=CFG.num_classes)
model.load_state_dict(torch.load("/kaggle/input/efficientnet-v4/efficientnet_b0_frozen_overall_best.pth", map_location=CFG.device))
model.to(CFG.device)

# Generate predictions
# Generate submission
generate_submission_with_smoothing(
    model=model,
    cfg=CFG,
    smoothing_window=3
)

