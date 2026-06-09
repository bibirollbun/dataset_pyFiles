import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import math

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from transformers import AutoModelForImageClassification
import torchaudio
import librosa

from tqdm import tqdm


paths = [
    "/kaggle/input/0602-models-v5/0602models/efficientnet-b2_fold1/checkpoint-14000",
    "/kaggle/input/0602-models-v5/0602models/efficientnet-b2_fold2/checkpoint-14000",
    "/kaggle/input/0602-models-v5/0602models/efficientnet-b2_fold3/checkpoint-14000",
    "/kaggle/input/0602-models-v5/0602models/regnet-y-008_fold1/checkpoint-14000",
    "/kaggle/input/0602-models-v5/0602models/regnet-y-008_fold2/checkpoint-14000",
    "/kaggle/input/0602-models-v5/0602models/regnet-y-008_fold3/checkpoint-14000"
]
models = []
for path in paths:
    model = AutoModelForImageClassification.from_pretrained(path)
    model.eval()
    models.append(model)                                             


# Configuration for mel spectrogram
mel_spec_params = {
    "sample_rate": 32000,
    "n_mels": 128,
    "f_min": 20,
    "f_max": 16000,
    "n_fft": 1024,
    "hop_length": 500,
    "normalized": True,
    "center": True,
    "pad_mode": "constant",
    "norm": "slaney",
    "mel_scale": "slaney"
}
top_db = 80
segment_length = 5 * mel_spec_params["sample_rate"]  # 5 seconds

mel_transform = torchaudio.transforms.MelSpectrogram(**mel_spec_params)
db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=top_db)

def normalize_melspec(X, eps=1e-6):
    """Normalize mel spectrogram"""
    mean = X.mean((1, 2), keepdim=True)
    std = X.std((1, 2), keepdim=True)
    Xstd = (X - mean) / (std + eps)

    norm_min, norm_max = (
        Xstd.min(-1)[0].min(-1)[0],
        Xstd.max(-1)[0].max(-1)[0],
    )
    fix_ind = (norm_max - norm_min) > eps * torch.ones_like(
        (norm_max - norm_min)
    )
    V = torch.zeros_like(Xstd)
    if fix_ind.sum():
        V_fix = Xstd[fix_ind]
        norm_max_fix = norm_max[fix_ind, None, None]
        norm_min_fix = norm_min[fix_ind, None, None]
        V_fix = torch.max(
            torch.min(V_fix, norm_max_fix),
            norm_min_fix,
        )
        V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
        V[fix_ind] = V_fix
    return V

def prepare_spec(wav):
    """Create mel spectrogram from audio file"""
    
    # Create mel spectrogram
    mel_spectrogram = mel_transform(wav)
    mel_spectrogram = db_transform(mel_spectrogram)
    mel_spectrogram = normalize_melspec(mel_spectrogram)
    
    # Scale to 0-255 range for image-like processing
    mel_spectrogram = mel_spectrogram * 255
    
    # Convert to 3-channel image format (RGB)
    mel_spectrogram = mel_spectrogram.expand(3, -1, -1).permute(1, 2, 0).numpy()

    # Final formatting: Convert to [C, H, W] format
    spec = mel_spectrogram.transpose(2, 0, 1)
    
    return torch.from_numpy(spec)


# List of test soundscapes (only visible during submission)
test_soundscape_path = '/kaggle/input/birdclef-2025/test_soundscapes'
test_soundscapes = [os.path.join(test_soundscape_path, afile) for afile in sorted(os.listdir(test_soundscape_path)) if afile.endswith('.ogg')]

#test_soundscape_path = '/kaggle/input/birdclef-2025/train_soundscapes'
#test_soundscapes = [os.path.join(test_soundscape_path, afile) for afile in sorted(os.listdir(test_soundscape_path)) if afile.endswith('.ogg')]
#test_soundscapes = test_soundscapes[:5]

preds = [np.empty(shape=(0, 206), dtype='float32') for _ in range(len(models))]
ids = []

for soundscape in tqdm(test_soundscapes):
    # Load audio
    sig, rate = librosa.load(path=soundscape, sr=None)

    # Split into 5s chunks
    chunks = []
    for i in range(0, len(sig), rate*5):
        chunk = sig[max(0, int(i-rate*2.5)):int(i+rate*7.5)]
        chunk = torch.from_numpy(chunk).unsqueeze(0)
        chunks.append(chunk)

    # create IDs for each chunk
    filename = os.path.basename(soundscape).split('.')[0]
    rec_ids = [f'{filename}_{(frame_id+1)*5}' for frame_id in range(len(chunks))]
    ids += rec_ids
    
    for m_idx, model in enumerate(models):
        rec_preds = np.empty(shape=(0, 206), dtype='float32')  # predictions for recording
        for i, chunk in enumerate(chunks):
            spec = prepare_spec(chunk)
            logits = model(spec.unsqueeze(0)).logits
            chunk_preds = nn.functional.softmax(logits, dim=-1)  # predictions for chunk
            chunk_preds = chunk_preds.detach().numpy()
            chunk_preds = chunk_preds[:, :206]  # drop last column (no call probabilities)
            rec_preds = np.concatenate([rec_preds, chunk_preds], axis=0)
        # average predictions for each chunk with neighboring chunks (window width 5)
        smooth_preds = rec_preds.copy()
        for i in range (len(chunks)):
            smooth_preds[i, :] = rec_preds[max(0, i-2):i+3].mean(axis = 0)            
        preds[m_idx] = np.concatenate([preds[m_idx], smooth_preds], axis=0)

# ensembling
preds = np.array(preds)
#preds = preds.mean(axis=0, keepdims=True)
#preds = preds.min(axis=0, keepdims=True)
preds = preds.max(axis=0, keepdims=True)
preds = preds.squeeze()

# Class labels from train audio
labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))
# Save prediction as csv
pred_df = pd.DataFrame(ids, columns=['row_id'])
pred_df.loc[:, labels] = preds
pred_df.to_csv('submission.csv', index=False)
pred_df

