import pandas as pd
import numpy as np
import librosa
import glob
import torch
import torch.nn as nn
import os
import random
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import seaborn as sns
from ast import literal_eval
import timm
import pandas.api.types

import sklearn.metrics
from tqdm import tqdm
import gc
from warnings import filterwarnings
filterwarnings("ignore")


class Config:
    train_dir = "/kaggle/input/birdclef-2025/train_audio"
    seed = 42
    train_csv = "/kaggle/input/birdclef-2025/train.csv"
    sample_submission_csv = "/kaggle/input/birdclef-2025/sample_submission.csv"
    train_soundscapes = "/kaggle/input/birdclef-2025/train_soundscapes"
    test_soundscapes = "/kaggle/input/birdclef-2025/test_soundscapes"
    # test_soundscapes = "/kaggle/input/birdclef-2025/test_audio"
    sr = int(32e3)

    num_classes = 206
    n_fft = 2048
    hop_length = 500

    n_mels = 256
    fmin = 50
    fmax = 16000
    power = 2
    image_shape = (128, 640, 1)
    submission_mode = len(glob.glob("/kaggle/input/birdclef-2025/test_soundscapes/*.ogg")) > 0 


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    # reproducible weight initialization
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    print(f"[info] set seed: {seed}")

set_seed()


if Config.submission_mode:
    sound_dir = glob.glob(Config.test_soundscapes + "/*.ogg")
else:
    sound_dir = glob.glob(Config.train_soundscapes + "/*.ogg")[:5]

sound_dir


%%time
def process(audio_path):
    filename = audio_path.split("/")[-1].split(".")[0]
    data, _ = librosa.load(audio_path, sr = Config.sr)

    data = data * 1024

    # Dividing the data into 5s chunks
    chunk_duration = 5
    min_len = chunk_duration * Config.sr

    local_mapper = {}
    for i in range(0, len(data), min_len):
        # Making row ids
        t = i // Config.sr
        row_id = f"{filename}_{t + chunk_duration}"

        chunk_5s = data[i: i + min_len]
        chunk_10s = np.tile(chunk_5s, 2)

        chunk_10s = chunk_10s.reshape(-1, len(chunk_10s))

        # Converting to mel spectrogram
        mel_sp = librosa.feature.melspectrogram(
            y = chunk_10s,
            sr = Config.sr,
            fmin = Config.fmin,
            fmax = Config.fmax,
            power = Config.power,
            n_mels = Config.n_mels,
            n_fft = Config.n_fft,
            hop_length = Config.hop_length
        )

        mel_sp = librosa.power_to_db(mel_sp, ref = 1)

        # Normalizing the features
        eps = 1e-12
        mel_sp = (mel_sp - mel_sp.min())/(mel_sp.max() - mel_sp.min() + eps)

        mel_sp = mel_sp[:, :, :640]
        local_mapper[row_id] = mel_sp
    return local_mapper    

# Loading audio files
all_mappers = Parallel(
    n_jobs = -1,
    backend = "loky"
)(delayed(process)(filepath) for filepath in sound_dir)

# Creating complete mapping
global_mapper = {}
for mapper in all_mappers: global_mapper.update(mapper)

print(f"[INFO] loaded all audio files, total_items: {len(global_mapper)}")


for key, val in global_mapper.items():
    print(key)
    print(val)
    break


%%time

model_paths = [
    '/kaggle/input/effnetb0-mixup-epoch20-fold-3/fold_1_epoch_6_best_effnetB0_val_auc_0.9798_val_loss_0.011425114528982332.pth',
    '/kaggle/input/effnetb0-mixup-epoch20-fold-3/fold_2_epoch_1_best_effnetB0_val_auc_0.9930_val_loss_0.006461535613466642.pth',
    #'/kaggle/input/effnetb0-mixup-epoch20-fold-3/fold_0_epoch_19_best_effnetB0_val_auc_0.9527_val_loss_0.015609626224437954.pth'
]

device = torch.device("cpu")  # Explicit for Kaggle

class Model(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.base_model = timm.create_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
            in_chans=1
        )

    def forward(self, x):
        return self.base_model(x)

def load_models(model_paths, model_name, num_classes):
    models = []
    for path in model_paths:
        model = Model(model_name, num_classes)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        models.append(model)
    return models

models_pool = load_models(
    model_paths,
    model_name="tf_efficientnet_b0",
    num_classes=Config.num_classes
)

print(f"[INFO] Loaded {len(models_pool)} models.")



def mini_tta_batch(mels_t):
    """Return original and time-shifted versions"""
    shifted = torch.roll(mels_t, shifts=random.randint(-20, 20), dims=2)
    return [mels_t, shifted]


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, mapper):
        self.mapper = mapper
        self.ids = list(self.mapper.keys())

    def __len__(self): return len(self.mapper)

    # def __getitem__(self, idx): return self.ids[idx], self.mapper[self.ids[idx]]

    def __getitem__(self, idx):
        row_id = self.ids[idx]
        mel_np = self.mapper[row_id]           # NumPy array
        mel_tensor = torch.from_numpy(mel_np)  # Convert once, fast
        return row_id, mel_tensor

test_loader = torch.utils.data.DataLoader(
    test_ds := TestDataset(global_mapper),
    batch_size=16,
    num_workers=0,      # Safer for CPU-only
    shuffle=False,
    drop_last=False
)

pred_mapper = {}
best_thresholds = np.load("/kaggle/input/effnetb0-birdclef2025-fold2-epoch1-best-threshold/best_thresholds.npy")

# Use only 1 model to avoid timeout
model = models_pool[0]
model.eval()

# Set all models to eval mode just to be safe
for model in models_pool:
    model.eval()
# Use torch.no_grad() globally
with torch.no_grad():
    for row_ids, mels_t in test_loader:
        mels_t = mels_t.to(device).float()  # Ensure float32 type

        # Mini TTA
        versions = mini_tta_batch(mels_t)
    
        # Collect predictions
        batch_preds = []
        for v in versions:
            outputs = model(v)
            probs = torch.sigmoid(outputs).cpu().numpy()
            batch_preds.append(probs)

        # Average and apply threshold
        avg_preds = np.mean(batch_preds, axis=0)
        for i in range(len(row_ids)):
            pred = (avg_preds[i] >= best_thresholds).astype(float)
            pred_mapper[row_ids[i]] = pred
        
# Sanity check
print(f"[INFO] Total samples: {len(global_mapper)} — Predictions generated: {len(pred_mapper)}")

# Create submission file
sample_sub = pd.read_csv(Config.sample_submission_csv)
columns = sample_sub.columns

pred_values = list(pred_mapper.values())
row_ids = list(pred_mapper.keys())

sub_df = pd.DataFrame(data=pred_values, columns=columns[1:])
sub_df.insert(0, 'row_id', row_ids)

sub_df.to_csv('submission.csv', index=False)
print(f"[INFO] Submission with TTA saved with shape: {sub_df.shape}")

