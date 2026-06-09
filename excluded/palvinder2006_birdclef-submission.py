import os
import glob
import json
import random
import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed

import timm
from tqdm import tqdm


class Config:
    train_dir = "/kaggle/input/birdclef-2025/train_audio"
    train_csv = "/kaggle/input/birdclef-2025/train.csv"
    train_soundscape = "/kaggle/input/birdclef-2025/train_soundscapes"
    test_soundscape = "/kaggle/input/birdclef-2025/test_soundscapes"
    sample_submission_csv = "/kaggle/input/birdclef-2025/sample_submission.csv"
    
    sr = 32000
    n_fft = 1024
    hop_length = 500
    n_mels = 128
    fmin = 40
    fmax = 15000
    power = 2
    chunk_seconds = 5
    image_shape = (128, 640, 1)  # height, width, channels
    num_classes = 206

    submission_mode = len(glob.glob(test_soundscape + "/*.ogg")) > 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed set to: {seed}")

set_seed(42)


if Config.submission_mode:
    sound_dir = sorted(glob.glob(f"{Config.test_soundscape}/*.ogg"))
else:
    sound_dir = sorted(glob.glob(f"{Config.train_soundscape}/*.ogg"))[:5]

print(f"[Info] Total .ogg files found: {len(sound_dir)}")


def process(audio_path):
    filename = audio_path.split("/")[-1].split(".")[0]
    data, _ = librosa.load(audio_path, sr=Config.sr)
    
    # Scale up the data
    data = data * 1024

    # Divide into 5-second chunks
    chunk_duration = 5
    min_len = chunk_duration * Config.sr
    
    local_mapper = {}

    for i in range(0, len(data), min_len):
        t = i // Config.sr
        row_id = f"{filename}_{t + chunk_duration}"
        chunk_5s = data[i: i + min_len]
        
        if len(chunk_5s) < min_len:
            continue  # Skip incomplete chunks

        chunk_10s = np.tile(chunk_5s, 2)
        chunk_10s = chunk_10s.reshape(-1, len(chunk_10s))

        # Mel spectrogram
        mel_sp = librosa.feature.melspectrogram(
            y=chunk_10s[0],
            sr=Config.sr,
            fmin=Config.fmin,
            fmax=Config.fmax,
            power=Config.power,
            n_mels=Config.n_mels,
            n_fft=Config.n_fft,
            hop_length=Config.hop_length
        )
        mel_sp = librosa.power_to_db(mel_sp, ref=1.0)

        # Normalize
        eps = 1e-12
        mel_sp = (mel_sp - mel_sp.min()) / (mel_sp.max() - mel_sp.min() + eps)
        mel_sp = mel_sp[:, :Config.image_shape[1]]  # crop or pad if needed
        mel_sp = np.expand_dims(mel_sp, axis=0)     # add channel dimension

        local_mapper[row_id] = mel_sp

    return local_mapper

# Load audio files in parallel
all_mappers = Parallel(n_jobs=-1, backend='loky')(
    delayed(process)(path) for path in sound_dir
)

# Merge all into a single dictionary
global_mapper = {}
for mapper in all_mappers:
    global_mapper.update(mapper)

print(f"[Info] Processed and extracted features for {len(global_mapper)} chunks")


# Print number of entries
print(f"Total processed segments: {len(global_mapper)}")

# Preview a single entry
for key, value in global_mapper.items():
    print(f"Row ID: {key}")
    print(f"Mel shape: {value.shape}")
    break  # Print only the first entry


global_mapper.keys()


class Model(nn.Module):
    def __init__(self, model_name: str = "tf_efficientnet_b3"):
        super().__init__()
        self.base_model = timm.create_model(
            model_name=model_name,
            pretrained=False,
            in_chans=1,
            num_classes=Config.num_classes
        )

    def forward(self, x):
        return self.base_model(x)

model_paths = [
    "/kaggle/input/birdclef-dataset-v3/checkpoint_epoch_4.pth",
    "/kaggle/input/birdclef-dataset-v3/checkpoint_epoch_5.pth",
    
]

device = Config.device
model_pool = []
for path in model_paths:
    model = Model(model_name="tf_efficientnet_b3").to(device)
    state_dict = torch.load(path, map_location=device)
    state_dict = {k.replace("backbone.", "base_model."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model_pool.append(model)

print(f"[INFO] Loaded {len(model_pool)} models.")


class TestDataset(Dataset):
    def __init__(self, mapper):
        self.mapper = mapper
        self.ids = list(mapper.keys())

    def __len__(self):
        return len(self.mapper)

    def __getitem__(self, idx):
        row_id = self.ids[idx]
        x = self.mapper[row_id]
        if x.shape == (1, 128, 640):
            x = x.squeeze(0)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        return row_id, x


test_loader = DataLoader(TestDataset(global_mapper), batch_size=16, shuffle=False)

pred_mapper = {}

for row_ids, mels in test_loader:
    mels = mels.to(device)
    batch_preds = []

    with torch.no_grad():
        for model in model_pool:
            outputs = model(mels)
            probs = torch.sigmoid(outputs).cpu().numpy()
            batch_preds.append(probs)

    avg_preds = np.mean(batch_preds, axis=0)

    for i, row_id in enumerate(row_ids):
        pred_mapper[row_id] = avg_preds[i]

print(f"[INFO] Predictions stored for {len(pred_mapper)} row_ids.")


pred_mapper


sample_df = pd.read_csv(Config.sample_submission_csv)
submission_columns = sample_df.columns.tolist()


# Convert pred_mapper to DataFrame
submission_df = pd.DataFrame.from_dict(pred_mapper, orient="index")
submission_df.columns = submission_columns[1:]  # exclude "row_id"
submission_df["row_id"] = submission_df.index
submission_df = submission_df[submission_columns]  # reorder columns


# Save submission
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

# Display top few rows
print("Submission file saved at:", submission_path)
submission_df.head()




