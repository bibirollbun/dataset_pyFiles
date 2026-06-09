import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import librosa

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
from torchaudio.transforms import TimeMasking, FrequencyMasking

import torchvision
from torchvision import models
from torchvision.transforms import functional as F_t

import timm

from sklearn.metrics import roc_auc_score


class CFG:
    SUBMISSION = True
    num_workers = 4

    OUTPUT_DIR = '/kaggle/working/'

    train_datadir = Path('/kaggle/input/birdclef-2025/train_audio')
    train_csv = '/kaggle/input/birdclef-2025/train.csv'
    test_soundscapes = Path('/kaggle/input/birdclef-2025/test_soundscapes')
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    # model_path = '/kaggle/input/bird_v1/pytorch/crossentropy_loss/1/model_crossentropy_loss.pt'
    model_path = '/kaggle/input/bird_v1/pytorch/bceloss/3/model_bce_loss(2).pt'

    model_name = 'efficientnet_b1'

    SR = 32000
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (256, 256)
    
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 10
    batch_size = 64

cfg = CFG()


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg
        
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=not cfg.SUBMISSION,
            in_chans=1,
            drop_rate=0.2,    
            drop_path_rate=0.2
        )
        
        backbone_out = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(backbone_out, num_classes)
        
    def forward(self, x):
        logits = self.backbone(x)
        return logits


taxonomies = pd.read_csv(cfg.taxonomy_csv)


tax2id = {t["primary_label"]: i for i, t in taxonomies.iterrows()}
id2tax = {i: t for t, i in tax2id.items()}


def audio2melspec(audio_data, cfg):
    # Convert raw audio to Log Mel Spectogram and normalize
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.SR,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm


def process_audio_segment(audio_data, cfg):
    # add padding if needed, if audio is less than 5s
    if len(audio_data) < cfg.SR * cfg.TARGET_DURATION:
        audio_data = np.pad(
            audio_data, 
            (0, int(cfg.SR * cfg.TARGET_DURATION - len(audio_data))), 
            mode='constant'
        )
    
    mel_spec = audio2melspec(audio_data, cfg)

    # most models were trained on 224x224, so resize spectogram to this size
    if mel_spec.shape != cfg.TARGET_SHAPE:
        mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

    return mel_spec.astype(np.float32)


df = pd.read_csv(cfg.train_csv)
df.head()


df = df[['primary_label', 'filename', 'secondary_labels']]
df.head()


melspecs = {}


def process_item(item):
    # load audio and take middle 5s (I hope middle 5s is representive)
    audio_path = cfg.train_datadir / item["filename"]
    waveform, sample_rate = librosa.load(audio_path, sr=cfg.SR)
    
    length = int(cfg.TARGET_DURATION * cfg.SR)

    step = length // 2

    start = max(0, waveform.shape[0] // 2 - length)
    end = min(waveform.shape[0], waveform.shape[0] // 2 + 1)

    mel_specs = []
    for i in range(start, end, step):
        waveform = waveform[i : i + length]
        mel_spec = process_audio_segment(waveform, cfg)
        mel_specs.append( mel_spec )
    
    secondary_labels = eval(item["secondary_labels"])
    label = [item["primary_label"]] + secondary_labels

    labels = [tax2id[i] for i in label if i]
    
    return item["filename"], mel_specs, labels


if not cfg.SUBMISSION:
    ## use 4 threads to convert audio 2 log mel spectogram
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
        futures = [executor.submit(process_item, item) for _, item in df.iterrows()]

        for future in tqdm(as_completed(futures), total=len(futures)):
            filename, mel_specs, labels = future.result()

            for i, m in enumerate(mel_specs):
                melspecs[f"{filename}_{i}"] = {
                    "mel_spec": m,
                    "labels": labels
                }


class BirdDataset(Dataset):
    def __init__(self, melspecs):
        self.melspecs = list(melspecs.values())

        self.time_mask = TimeMasking(time_mask_param=50)
        self.freq_mask = FrequencyMasking(freq_mask_param=30)

    def __len__(self):
        return len(self.melspecs)

    def random_brightness_contrast(self, mel_spec):
        # Apply random brightness and contrast
        mel_spec = F_t.adjust_brightness(mel_spec, random.uniform(0.7, 1.3))
        mel_spec = F_t.adjust_contrast(mel_spec, random.uniform(0.7, 1.3))

        return mel_spec

    def __getitem__(self, idx):
        item = self.melspecs[idx]

        # mel spectogram
        mel_spec = torch.Tensor( item["mel_spec"] ).unsqueeze(0)

        # Augmentation: Time masking
        if random.random() < 0.5:
            mel_spec = self.time_mask(mel_spec)
        
        # Augmentation: Frequency masking
        if random.random() < 0.5:
            mel_spec = self.freq_mask(mel_spec)

        # Augmentation: Random brightness/contrast
        if random.random() < 0.5:
            mel_spec = self.random_brightness_contrast(mel_spec)

        labels = item["labels"]
        
        targets = np.zeros((len(tax2id), ))
        targets[labels] = 1.0
    
        return mel_spec, targets


if not cfg.SUBMISSION:
    # create datasets/dataloaders
    from sklearn.model_selection import train_test_split
    
    # train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # train_df = train_df.reset_index(drop=True)
    # val_df = val_df.reset_index(drop=True)

    filenames = list(melspecs.keys())
    train_files, val_files = train_test_split(filenames, test_size=0.2, random_state=42)
    
    # Reconstruct train and val dicts
    train_data = {fname: melspecs[fname] for fname in train_files}
    val_data = {fname: melspecs[fname] for fname in val_files}
    
    train_ds = BirdDataset(train_data)
    val_ds = BirdDataset(val_data,)
    
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=4, pin_memory=True)

    print(len(train_ds), len(train_dl))


model = BirdCLEFModel(cfg, len(id2tax))

if cfg.SUBMISSION:
    model.load_state_dict(torch.load(cfg.model_path, weights_only=True))
    model.eval()


sum(p.numel() for p in model.parameters()) / 1_000_000


if not cfg.SUBMISSION:
    backbone_params = [p for n, p in model.backbone.named_parameters() if "classifier" not in n]

    optimizer = torch.optim.Adam([
        {
            'params': backbone_params,
            'lr': 1e-4
        },
        {
            'params': model.backbone.classifier.parameters(),
            'lr': 1e-3
        },
    ])


sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000


def macro_roc_auc_with_positive_labels(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positive_cols = y_true.sum(axis=0) > 0

    return roc_auc_score(y_true[:, positive_cols],
                         y_score[:, positive_cols],
                         average='macro')


if not cfg.SUBMISSION:
    model.to(cfg.device)

    # BCELossWithLogits showed better results (expected)
    # since we need to predict 1 class, but many. And we need to maximize their probs
    loss_fn = nn.BCEWithLogitsLoss()
    
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        
        train_loss = 0
        all_train_true = []
        all_train_scores = []
    
        for x, y in tqdm(train_dl):
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
    
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item() * x.size(0)
    
            y_true_batch = y
            probs = torch.sigmoid(logits)

            all_train_true.append(y_true_batch.cpu().detach().numpy())
            all_train_scores.append(probs.cpu().detach().numpy())
    
        train_true = np.vstack(all_train_true)
        train_scores = np.vstack(all_train_scores)
        train_auc = macro_roc_auc_with_positive_labels(train_true, train_scores)
        avg_train_loss = train_loss / len(train_dl.dataset)

        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} — Train AUC: {train_auc:.4f}")

        ### Validation
        val_loss = 0
        all_val_true = []
        all_val_scores = []
        model.eval()
        
        with torch.no_grad():
            for x, y in tqdm(val_dl):
                x, y = x.to(cfg.device), y.to(cfg.device)
        
                logits = model(x)
                loss = loss_fn(logits, y)
                val_loss += loss.item() * x.size(0)
                
                y_true_batch = y
                probs = torch.sigmoid(logits)

                all_val_true.append(y_true_batch.cpu().detach().numpy())
                all_val_scores.append(probs.cpu().detach().numpy())

            val_true = np.vstack(all_val_true)
            val_scores = np.vstack(all_val_scores)
            val_auc = macro_roc_auc_with_positive_labels(val_true, val_scores)
            avg_val_loss = val_loss / len(val_dl.dataset)
    
            print(f"[Epoch {epoch}] Val Loss:   {avg_val_loss:.4f} — Val AUC:   {val_auc:.4f}")


# model.to("cpu")
# torch.save(model.state_dict(), "./model_bce_loss.pt")


def process_item(audio_path):
    # now get EACH 5 seconds from sound
    waveform, sample_rate = librosa.load(audio_path, sr=cfg.SR)
    
    length = int(cfg.TARGET_DURATION * cfg.SR)

    mel_specs = []
    for i in range(0, len(waveform), sample_rate * 5):
        start = i
        end = i + sample_rate * 5
        w = waveform[start : end]

        mel_spec = process_audio_segment(w, cfg)
        mel_specs.append( mel_spec )
        
    return mel_specs


# make submission
if cfg.SUBMISSION:
    class_labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))
    
    test_soundscape_path = '/kaggle/input/birdclef-2025/test_soundscapes/'
    test_soundscapes = [os.path.join(test_soundscape_path, afile) for afile in sorted(os.listdir(test_soundscape_path)) if afile.endswith('.ogg')]

    if not test_soundscapes:
        print("Test dir is empty. Using first 10 samples from train dir")
        test_soundscapes = list(cfg.train_datadir.glob("*/*.ogg"))[:10]
    
    predictions = pd.DataFrame(columns=['row_id'] + class_labels)
    for audio_path in test_soundscapes:
        mel_specs = process_item(audio_path)

        for k, mel_spec in enumerate(mel_specs):
            row_id = os.path.basename(audio_path).split('.')[0] + f'_{k * 5 + 5}'

            mel_spec = torch.Tensor( mel_spec ).unsqueeze(0)
            logits = model( mel_spec.unsqueeze(0) )[0]
            scores = F.softmax(logits, dim=0).tolist()
    
            new_row = pd.DataFrame(
                [[row_id] + list(scores)],
                columns=['row_id'] + class_labels
            )

            predictions = pd.concat([predictions, new_row], axis=0, ignore_index=True)

    
    sample_submission = pd.read_csv(cfg.submission_csv)

    assert set(sample_submission.columns) == set(predictions.columns)
    predictions.to_csv("submission.csv", index=False, float_format='%.16f')
    print("CSV saved")




