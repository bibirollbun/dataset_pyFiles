import pandas as pd
import numpy as np
import librosa
import glob

import torch
import torch.nn as nn
import albumentations
import os
import random
from joblib import Parallel, delayed
import json
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
import timm
from warnings import filterwarnings
import pandas.api.types

import sklearn.metrics
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
import gc

filterwarnings("ignore")



class Config:
    train_dir = "/kaggle/input/birdclef-2025/train_audio"
    seed = 42
    train_csv = "/kaggle/input/birdclef-2025/train.csv"
    train_soundscapes = "/kaggle/input/birdclef-2025/train_soundscapes"
    test_soundscapes = "/kaggle/input/birdclef-2025/test_soundscapes"
    sample_submission_csv = "/kaggle/input/birdclef-2025/sample_submission.csv"
    sr = int(32e3)
    n_fft = 1024
    hop_length = 500
    n_mels = 128
    fmin = 40
    fmax = 15000
    power = 2
    num_classes = 206
    image_shape = (128, 640, 1)
    submission_mode = len(glob.glob("/kaggle/input/birdclef-2025/test_soundscapes/*.ogg")) > 0



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Set seed: {seed}")

set_seed()



test_files = glob.glob(Config.test_soundscapes + "/*.ogg")
# sound_dir = test_files
if len(test_files) > 0:
    sound_dir = test_files
    print("[INFO] Submission mode: using test_soundscapes.")
else:
    sound_dir = glob.glob(Config.train_soundscapes + "/*.ogg")[:5]
    print("[INFO] Debug mode: using train_soundscapes.")
sound_dir


%%time
def process(audio_path):
    filename = audio_path.split("/")[-1].split(".")[0]
    data, _ = librosa.load(audio_path, sr=Config.sr)

    data = data * 1024

    # Dividing the data into 5s chunks
    chunk_duration = 5
    min_len = chunk_duration * Config.sr

    local_mapper = {}

    for i in range(0, len(data), min_len):
        # Making row ids
        t = i // Config.sr
        row_id = f"{filename}_{t + chunk_duration}"

        chunk_5s = data[i : i + min_len]
        chunk_10s = np.tile(chunk_5s, 2)
        chunk_10s = chunk_10s.reshape(-1, len(chunk_10s))

        # Converting to mel spectrogram
        mel_sp = librosa.feature.melspectrogram(
            y=chunk_10s,
            sr=Config.sr,
            fmin=Config.fmin,
            fmax=Config.fmax,
            power=Config.power,
            n_mels=Config.n_mels,
            n_fft=Config.n_fft,
            hop_length=Config.hop_length
        )
        mel_sp = librosa.power_to_db(mel_sp, ref=1)

        # Normalizing the features
        eps = 1e-12
        mel_sp = (mel_sp - mel_sp.min()) / ((mel_sp.max() - mel_sp.min()) + eps)
        mel_sp = mel_sp[:, :, :640]
        local_mapper[row_id] = mel_sp

    return local_mapper
# Loading audio files
all_mappers = Parallel(
    n_jobs=-1,
    backend="loky"
)(delayed(process)(filepath) for filepath in sound_dir)

# Creating complete mapping
global_mapper = {}
for mapper in all_mappers: 
    global_mapper.update(mapper)

print(f"[INFO] Loaded all audio files, total_items: {len(global_mapper)}")



global_mapper.keys()


%%time

models = [
    "/kaggle/input/lastoflast/fold_0_tf_efficientnet_b0_epoch_6_val_auc_0.9553_val_loss_41.2952 (1).pth",
    "/kaggle/input/lastoflast/fold_2_regnety_008_epoch_9_val_auc_0.9446_val_loss_48.3645.pth"
]

device = "cuda" if torch.cuda.is_available() else "cpu"

class Model(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        self.base_model = timm.create_model(
            model_name=model_name,
            num_classes=Config.num_classes,
            pretrained=False,
            in_chans=1,
        )

    def forward(self, x):
        return self.base_model(x)

# Sửa tại đây
model_infos = [
    ("tf_efficientnet_b0", models[0]),
    ("regnety_008",         models[1])
]

models_pool = []
for model_name, model_path in model_infos:
    model = Model(model_name=model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    models_pool.append(model)

print("[INFO] Loaded all models")



class TestDataset(torch.utils.data.Dataset):
    def __init__(self, mapper):
        self.mapper = mapper
        self.ids = list(self.mapper.keys())

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx): return self.ids[idx], self.mapper[self.ids[idx]]

test_loader = torch.utils.data.DataLoader(
    dataset = TestDataset(global_mapper),
    batch_size = 16,
    num_workers = 2,
    shuffle = False,
    drop_last = False
)

# To capture the model prediction per row id
pred_mapper = {}

for (row_ids, mels) in test_loader:
    mels_t = torch.tensor(mels).to(device)

    model_preds = []

    with torch.no_grad():
        for model in models_pool:
            outputs = model(mels_t)
            probs = torch.sigmoid(outputs).detach().cpu().numpy().squeeze()
            model_preds.append(probs)  # Prediction of every model on current batch

    # Averaging model predictions
    mel_preds = np.mean(model_preds, axis=0)

    for idx, row_id in enumerate(row_ids):
        pred_mapper[row_id] = mel_preds[idx]

    del mels_t

print(len(global_mapper), len(pred_mapper.keys()))



# Creating submission df
sample_df = pd.read_csv(Config.sample_submission_csv)
submission_df = pd.DataFrame(data=list(pred_mapper.values()), columns=sample_df.columns[1:])
submission_df.insert(0, 'row_id', list(pred_mapper.keys()))
submission_df = submission_df[sample_df.columns]
submission_df.to_csv("/kaggle/working/submission.csv", index=False)
submission_df

