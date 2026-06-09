import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import librosa
import matplotlib.pyplot as plt
import cv2

import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



class CFG:
    
    seed = 42
    debug = True  
    apex = False
    print_freq = 100
    num_workers = 2
    
    OUTPUT_DIR = '/kaggle/working/'

    train_csv = '/kaggle/input/birdclef-2025/train.csv'
    train_datadir = '/kaggle/input/birdclef-2025/train_audio'
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'

    model = models.efficientnet_b0()

    class CustomResNet(nn.Module):
        def __init__(self, model):
            super(CustomResNet, self).__init__()
            self.model = model
            self.part = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3
            )
        def forward(self, x):
            x = self.part(x)
            return x
    
    pretrained = True
    in_channels = 1

    LOAD_DATA = True  
    sample_rate = 32000
    target_duration = 5.0
    target_shape = (256, 256)
    
    n_fft = 1024
    hop_length = 512
    n_mels = 128
    f_min = 20
    f_max = 16000
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10  
    batch_size = 32  
    criterion = 'BCEWithLogitsLoss'

    n_fold = 5
    selected_folds = [0, 1, 2, 3, 4]   

    optimizer = 'AdamW'
    lr = 5e-4 
    weight_decay = 1e-5
  
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = epochs

    aug_prob = 0.5  
    mixup_alpha = 0.5  
    
    def update_debug_settings(self):
        if self.debug:
            self.epochs = 2
            self.selected_folds = [0]

cfg = CFG()


tax_df   = pd.read_csv("/kaggle/input/birdclef-2025/taxonomy.csv")   # has columns ["common_name","class_name",…]
train_df = pd.read_csv('/kaggle/input/birdclef-2025/train.csv')

# 2) identify all common_names that belong to class Aves
aves_names = tax_df.loc[
    tax_df["class_name"] == "Aves", 
    "common_name"
].unique()

# 3) filter out any rows in train_df whose common_name is in that list
filtered = train_df[train_df["common_name"].isin(aves_names)]

# 4) save result
filtered.to_csv("train_only_aves.csv", index=False)


df = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")
len(df)


train_only_aves.


file_path = "/kaggle/input/train-audio-human-timestamps/train_audio_speech_timestamps.pkl"

try:
    with open(file_path, 'rb') as file:
        human_segments = pickle.load(file)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except EOFError:
    print("Error: Incomplete data in pickle file")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


train_dataset = pd.read_csv(cfg.train_csv)
audio, _ = librosa.load(cfg.train_datadir + "/" + train_dataset["filename"][0])

print(len(audio))
mel_spec = audio2melspec(cfg, audio)

# Plot
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spec, sr=cfg.sample_rate, x_axis='time', y_axis='mel', fmax= cfg.f_max, cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()


def audio2melspec(cfg, audio_data):
    """Convert audio data to mel spectrogram"""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y= audio_data,
        sr= cfg.sample_rate,
        n_fft= cfg.n_fft,
        hop_length= cfg.hop_length,
        n_mels= cfg.n_mels,
        fmin= cfg.f_min,
        fmax= cfg.f_max,
        power=2.0,
        pad_mode="reflect",
        norm='slaney',
        htk=True,
        center=True,
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

    return mel_spec_norm

def process_audio_segment(audio_data):
    """Process audio segment to get mel spectrogram"""
    if len(audio_data) < n_length:
        audio_data = np.pad(audio_data,
                          (0, n_length - len(audio_data)),
                          mode='constant')

    mel_spec = audio2melspec(audio_data)

    if mel_spec.shape != img_size:
        mel_spec = cv2.resize(mel_spec, img_size, interpolation=cv2.INTER_LINEAR)

    return mel_spec.astype(np.float32)

def remove_human_voice(audio, sr, segments, mode='excise'):
    """
    Remove or zero‐out human‐voice intervals from a 1D audio array.
    """
    if not segments:
        return audio

    if mode == 'excise':
        mask = np.ones(len(audio), dtype=bool)
        for seg in segments:
            i0 = int(seg['start'] * sr)
            i1 = int(seg['end']   * sr)
            mask[i0:i1] = False
        return audio[mask]

    elif mode == 'zero':
        out = audio.copy()
        for seg in segments:
            i0 = int(seg['start'] * sr)
            i1 = int(seg['end']   * sr)
            out[i0:i1] = 0.0
        return out

    else:
        raise ValueError(f"Unknown mode {mode!r}")
def preprocess_audio_to_dict(
    cfg,
    human_segments: dict,
    mode: str = 'excise',
    n_length: int = 16000,
    img_size: tuple = (256, 256),
    path_col: str = 'filename'
) -> dict:
    """
    For each entry in cfg.train_csv[path_col]:
      1. load at cfg.sample_rate
      2. remove human‐voice segments if present
      3. pad/truncate to n_length
      4. compute & normalize mel‐spectrogram
      5. resize to img_size

    Returns:
      spect_dict: { filename (str) → mel‐spec np.ndarray of shape img_size }
    """
    spect_dict = {}
    df = pd.read_csv(cfg.train_csv)

    for fname in df[path_col]:
        fp = cfg.train_datadir + "/" +fname

        # 1) load
        audio, sr = librosa.load(fp, sr=cfg.sample_rate, mono=True)

        # 2) remove voice
        segments = human_segments.get(fp, [])
        audio = remove_human_voice(audio, sr, segments, mode=mode)

        # 3) pad / truncate
        if len(audio) < n_length:
            audio = np.pad(audio, (0, n_length - len(audio)), mode='constant')
        else:
            audio = audio[:n_length]

        # 4) mel‐spec
        mel = audio2melspec(cfg, audio)

        # 5) resize if needed
        if mel.shape != img_size:
            mel = cv2.resize(mel, img_size, interpolation=cv2.INTER_LINEAR)

        spect_dict[fname] = mel.astype(np.float32)

    return spect_dict



train_df


train_df = pd.read_csv(cfg.train_csv)
mel_specs = preprocess_audio_to_dict(
    cfg,
    human_segments,
    mode='excise',      # fully cut out speech segments
    n_length=5*32000,   # e.g. 5-second clips at 32 kHz
    img_size=(256,256),
    path_col='filename'
)



# 2. save to .npy
np.save('mel_specs.npy', mel_specs)


df_prepped.to_csv("mel_specs.csv")


df_prepped2.to_pickle('mel_specs.pkl')      
# reading
with open('/kaggle/working/mel_specs.pkl', 'rb') as f:
    df_restored = pickle.load(f)


with open('/kaggle/working/mel_specs.pkl', 'rb') as f:
    df_restored = pickle.load(f)


from IPython.display import FileLink

# display a link
display(FileLink('mel_specs.npy'))


# Plot
plt.figure(figsize=(10, 4))
librosa.display.specshow(df_prepped["mel_spec"].iloc[3], sr=cfg.sample_rate, x_axis='time', y_axis='mel', fmax= cfg.f_max, cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()


df_prepped2 = df_prepped[['primary_label', 'filename', 'mel_spec']]




