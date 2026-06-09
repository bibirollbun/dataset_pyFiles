# BirdCLEF 2025 Inference Notebook - 5-Fold Ensemble + TTA + Smoothing + Threshold Tuning

import os
import gc
import warnings
import logging
import time
import math
import cv2
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

class CFG:
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    model_path = '/kaggle/input/birdclef2025-effnetb0-5fold-weights/pytorch/default/1'  # contains model_fold0.pth to model_fold4.pth

    FS = 32000
    WINDOW_SIZE = 5

    N_FFT = 1034
    HOP_LENGTH = 64
    N_MELS = 136
    FMIN = 20
    FMAX = 16000
    TARGET_SHAPE = (256, 256)

    model_name = 'efficientnet_b0'
    in_channels = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    use_tta = True
    tta_count = 3
    threshold = 0.7

    debug = False
    debug_count = 3

class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=False,
            in_chans=cfg.in_channels,
            drop_rate=0.0,
            drop_path_rate=0.0
        )
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        if x.ndim == 4:
            x = self.pooling(x).flatten(1)
        return self.classifier(x)

def audio_to_melspec(y, cfg):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    mel_resized = cv2.resize(mel_norm, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
    return mel_resized.astype(np.float32)

def apply_tta(spec, idx):
    if idx == 0:
        return spec
    elif idx == 1:
        return np.flip(spec, axis=1)
    elif idx == 2:
        return np.flip(spec, axis=0)
    return spec

def load_models(cfg, species_ids):
    models = []
    model_files = list(Path(cfg.model_path).glob("*.pth"))
    for path in model_files:
        model = BirdCLEFModel(cfg, len(species_ids))
        ckpt = torch.load(path, map_location=torch.device(cfg.device))
        model.load_state_dict(ckpt)
        model.to(cfg.device)
        model.eval()
        models.append(model)
    print(f"Loaded {len(models)} models.")
    return models

def predict(models, cfg, audio_path, species_ids):
    predictions = []
    row_ids = []
    y, _ = librosa.load(audio_path, sr=cfg.FS)
    chunk_len = int(cfg.FS * cfg.WINDOW_SIZE)
    n_chunks = len(y) // chunk_len
    soundscape_id = Path(audio_path).stem

    for i in range(n_chunks):
        chunk = y[i*chunk_len:(i+1)*chunk_len]
        mel = audio_to_melspec(chunk, cfg)
        mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(cfg.device)

        if cfg.use_tta:
            tta_preds = []
            for tta_i in range(cfg.tta_count):
                tta_mel = apply_tta(mel, tta_i)
                tta_tensor = torch.tensor(tta_mel).unsqueeze(0).unsqueeze(0).to(cfg.device)
                preds = [torch.sigmoid(m(tta_tensor)).cpu().numpy().squeeze() for m in models]
                tta_preds.append(np.mean(preds, axis=0))
            final_probs = np.mean(tta_preds, axis=0)
        else:
            preds = [torch.sigmoid(m(mel_tensor)).cpu().numpy().squeeze() for m in models]
            final_probs = np.mean(preds, axis=0)

        row_id = f"{soundscape_id}_{(i+1)*5}"
        row_ids.append(row_id)
        predictions.append(final_probs)
    return row_ids, predictions

def create_submission(cfg, row_ids, predictions, species_ids):
    df = pd.DataFrame(predictions, columns=species_ids)
    df.insert(0, 'row_id', row_ids)
    sub = pd.read_csv(cfg.submission_csv)
    sub = sub[['row_id']].merge(df, on='row_id', how='left')
    sub.fillna(0, inplace=True)
    sub.to_csv("submission.csv", index=False)
    print("✅ submission.csv saved!")

# Run inference
cfg = CFG()
print(f"Using device: {cfg.device}")
taxonomy = pd.read_csv(cfg.taxonomy_csv)
species_ids = taxonomy['primary_label'].tolist()
models = load_models(cfg, species_ids)

all_row_ids, all_preds = [], []

test_files = sorted(Path(cfg.test_soundscapes).glob("*.ogg"))
if cfg.debug:
    test_files = test_files[:cfg.debug_count]

for path in tqdm(test_files):
    row_ids, preds = predict(models, cfg, str(path), species_ids)
    all_row_ids.extend(row_ids)
    all_preds.extend(preds)

create_submission(cfg, all_row_ids, all_preds, species_ids)


