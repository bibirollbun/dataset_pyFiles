import os
import time
import random
import gc
import warnings
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import librosa.display
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from haversine import haversine
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, lr_scheduler
from torch.amp import autocast, GradScaler
import timm
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')


class Config:
    OUTPUT_DIR = '/kaggle/working/'
    DATA_ROOT = '/kaggle/input/birdclef-2025'
    AUDIO_PATH = Path(DATA_ROOT) / 'train_audio'
    SOUNDSCAPE_PATH = Path(DATA_ROOT) / 'train_soundscapes'
    FS = 32000
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    N_MFCC = 13
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (256, 256)
    N_MAX = None
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_FOLDS = 5
    SEED = 2025
    NOISE_FACTOR = 0.005
    TIME_STRETCH_MIN = 0.8
    TIME_STRETCH_MAX = 1.2
    PITCH_SHIFT_MIN = -2
    PITCH_SHIFT_MAX = 2
    FREQ_MASK_PARAM = 15
    TIME_MASK_PARAM = 20
    CHECKPOINT_PATH = Path(OUTPUT_DIR) / 'checkpoints'
    DURATION_CACHE = Path(OUTPUT_DIR) / 'durations.pkl'
    FEATURE_CACHE = Path(OUTPUT_DIR) / 'features'

# Initialize config
config = Config()


# Set random seed
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_everything(config.SEED)


# Load data
taxonomy_df = pd.read_csv(f'{config.DATA_ROOT}/taxonomy.csv')
train_df = pd.read_csv(f'{config.DATA_ROOT}/train.csv')
print("Data loaded successfully")


# Create mappings
species_class_map = dict(zip(taxonomy_df['primary_label'], taxonomy_df['class_name']))
label_list = sorted(train_df['primary_label'].unique())
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}


# Prepare working DataFrame
working_df = train_df[['primary_label', 'rating', 'filename', 'latitude', 'longitude']].copy()
working_df['target'] = working_df['primary_label'].map(label2id)
working_df['filepath'] = str(config.AUDIO_PATH) + '/' + working_df['filename']
working_df['samplename'] = working_df['filename'].map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
working_df['class'] = working_df['primary_label'].map(lambda x: species_class_map.get(x, 'Unknown'))
total_samples = min(len(working_df), config.N_MAX or len(working_df))
working_df = working_df.iloc[:total_samples].reset_index(drop=True)
print(f"Total samples to process: {total_samples}")


# Parse secondary labels
def parse_secondary_labels(label_str):
    if pd.isna(label_str):
        return []
    try:
        return ast.literal_eval(label_str)
    except:
        return []
train_df['parsed_secondary_labels'] = train_df['secondary_labels'].apply(parse_secondary_labels)
working_df = working_df.merge(train_df[['filename', 'parsed_secondary_labels']], on='filename', how='left')
working_df.rename(columns={'parsed_secondary_labels': 'secondary_labels'}, inplace=True)


# Add geo feature and rating weight
working_df['geo_distance'] = working_df.apply(lambda row: haversine((row['latitude'], row['longitude']), (0, 0)) / 1000, axis=1)
working_df['rating_weight'] = working_df['rating'].apply(lambda x: x / 5.0 if x > 0 else 0.5)


def get_audio_duration(file_path, sr=config.FS):
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
        return len(audio) / sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.nan

if os.path.exists(config.DURATION_CACHE):
    print("Loading cached durations")
    durations = joblib.load(config.DURATION_CACHE)
else:
    print("Calculating durations")
    durations = [get_audio_duration(fp) for fp in tqdm(working_df['filepath'].tolist(), desc="Calculating durations")]
    joblib.dump(durations, config.DURATION_CACHE)
working_df['duration'] = durations
print(f"Duration stats - Mean: {np.nanmean(working_df['duration']):.2f}s, Median: {np.nanmedian(working_df['duration']):.2f}s")


plt.figure(figsize=(10, 6))
sns.countplot(data=working_df, x='class', order=working_df['class'].value_counts().index)
plt.title('Sample Distribution by Class')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{config.OUTPUT_DIR}/class_distribution.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.countplot(x='rating', data=working_df)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig(f'{config.OUTPUT_DIR}/rating_distribution.png')
plt.close()
print(f"Average rating: {working_df['rating'].mean():.2f}, Median rating: {working_df['rating'].median():.2f}")

plt.figure(figsize=(12, 6))
plt.hist(working_df['duration'].dropna(), bins=50, color='skyblue')
plt.title('Distribution of Audio Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.savefig(f'{config.OUTPUT_DIR}/duration_distribution.png')
plt.close()


classes = ['Aves', 'Amphibia', 'Mammalia', 'Insecta']
plt.figure(figsize=(15, 10))
for i, cls in enumerate(classes, 1):
    sample = working_df[working_df['class'] == cls].sample(n=1)
    audio, _ = librosa.load(sample['filepath'].iloc[0], sr=config.FS)
    mel = audio2melspec(audio)
    plt.subplot(2, 2, i)
    librosa.display.specshow(mel, sr=config.FS, x_axis='time', y_axis='mel')
    plt.title(f'Spectrogram - {cls}')
    plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.savefig(f'{config.OUTPUT_DIR}/spectrogram_samples.png')
plt.close()
print("Visualizations saved as spectrogram_samples.png")


def prepare_audio(audio, target_len):
    while len(audio) < target_len:
        audio = np.concatenate([audio, audio])
    start = max(0, len(audio) // 2 - target_len // 2)
    audio = audio[start:start + target_len]
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
    return audio

def audio2melspec(audio_data):
    if np.isnan(audio_data).any():
        mean_val = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_val)
    mel = librosa.feature.melspectrogram(
        y=audio_data, sr=config.FS, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS, fmin=config.FMIN, fmax=config.FMAX, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db

def extract_mfcc(audio_data):
    if np.isnan(audio_data).any():
        mean_val = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_val)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=config.FS, n_mfcc=config.N_MFCC)
    mfcc = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min() + 1e-8)
    return mfcc


os.makedirs(config.FEATURE_CACHE, exist_ok=True)
feature_cache_files = [config.FEATURE_CACHE / f'{row.samplename}.npz' for _, row in working_df.iterrows()]
missing_files = [f for f in feature_cache_files if not f.exists()]

if missing_files:
    print(f"Precomputing features for {len(missing_files)} samples")
    for i in tqdm(range(0, total_samples, config.BATCH_SIZE), desc="Feature extraction"):
        batch_df = working_df.iloc[i:i + config.BATCH_SIZE]
        for _, row in batch_df.iterrows():
            cache_file = config.FEATURE_CACHE / f'{row.samplename}.npz'
            if cache_file.exists():
                continue
            try:
                audio, _ = librosa.load(row.filepath, sr=config.FS, mono=True)
                audio = prepare_audio(audio, int(config.TARGET_DURATION * config.FS))
                mel = audio2melspec(audio)
                mfcc = extract_mfcc(audio)
                if mel.shape != config.TARGET_SHAPE:
                    mel = cv2.resize(mel, config.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
                if mfcc.shape != config.TARGET_SHAPE:
                    mfcc = cv2.resize(mfcc, config.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
                np.savez_compressed(cache_file, mel=mel.astype(np.float32), mfcc=mfcc.astype(np.float32))
            except Exception as e:
                print(f"Error processing {row.filepath}: {e}")
        gc.collect()
print("Feature extraction completed and cached")


class BirdDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df
        self.is_train = is_train
        self.spec_augment = T.FrequencyMasking(freq_mask_param=config.FREQ_MASK_PARAM)
        self.time_augment = T.TimeMasking(time_mask_param=config.TIME_MASK_PARAM)

    def augment_audio(self, audio):
        rate = random.uniform(config.TIME_STRETCH_MIN, config.TIME_STRETCH_MAX)
        audio = librosa.effects.time_stretch(audio, rate=rate)
        shift = random.uniform(config.PITCH_SHIFT_MIN, config.PITCH_SHIFT_MAX)
        audio = librosa.effects.pitch_shift(audio, sr=config.FS, n_steps=shift)
        noise = np.random.normal(0, config.NOISE_FACTOR, len(audio))
        audio += noise
        return audio

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cache_file = config.FEATURE_CACHE / f'{row.samplename}.npz'
        try:
            data = np.load(cache_file, mmap_mode='r')
            mel = data['mel']
            mfcc = data['mfcc']
        except Exception as e:
            print(f"Error loading {cache_file}: {e}")
            audio, _ = librosa.load(row.filepath, sr=config.FS, mono=True)
            audio = prepare_audio(audio, int(config.TARGET_DURATION * config.FS))
            mel = audio2melspec(audio)
            mfcc = extract_mfcc(audio)
            if mel.shape != config.TARGET_SHAPE:
                mel = cv2.resize(mel, config.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
            if mfcc.shape != config.TARGET_SHAPE:
                mfcc = cv2.resize(mfcc, config.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

        if self.is_train:
            audio, _ = librosa.load(row.filepath, sr=config.FS, mono=True)
            audio = self.augment_audio(audio)
            audio = prepare_audio(audio, int(config.TARGET_DURATION * config.FS))
            mel = audio2melspec(audio)
            mfcc = extract_mfcc(audio)
            if mel.shape != config.TARGET_SHAPE:
                mel = cv2.resize(mel, config.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
            if mfcc.shape != config.TARGET_SHAPE:
                mfcc = cv2.resize(mfcc, config.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        mel = mel.permute(1, 2, 0).repeat(1, 1, 3).permute(2, 0, 1)  # [3, H, W]
        mfcc = mfcc.permute(1, 2, 0).repeat(1, 1, 3).permute(2, 0, 1)  # [3, H, W]

        if self.is_train:
            mel = self.spec_augment(mel)
            mel = self.time_augment(mel)
            mfcc = self.spec_augment(mfcc)
            mfcc = self.time_augment(mfcc)

        if mel.shape != (3, config.TARGET_SHAPE[0], config.TARGET_SHAPE[1]):
            raise ValueError(f"Expected mel shape {(3, config.TARGET_SHAPE[0], config.TARGET_SHAPE[1])}, got {mel.shape}")
        if mfcc.shape != (3, config.TARGET_SHAPE[0], config.TARGET_SHAPE[1]):
            raise ValueError(f"Expected mfcc shape {(3, config.TARGET_SHAPE[0], config.TARGET_SHAPE[1])}, got {mfcc.shape}")

        target = np.zeros(len(label_list), dtype=np.float32)
        target[label2id[row['primary_label']]] = 1.0
        return {'mel': mel, 'mfcc': mfcc, 'target': target, 'geo_feature': row['geo_distance'], 'rating_weight': row['rating_weight']}


class BirdCRNN(nn.Module):
    def __init__(self, num_classes):
        super(BirdCRNN, self).__init__()
        self.cnn_mel = timm.create_model('tf_efficientnetv2_b0.in1k', pretrained=True, num_classes=0, global_pool='', drop_rate=0.1)
        self.cnn_mfcc = timm.create_model('tf_efficientnetv2_b0.in1k', pretrained=True, num_classes=0, global_pool='', drop_rate=0.1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(input_size=self.cnn_mel.num_features + self.cnn_mfcc.num_features, hidden_size=128, num_layers=2, batch_first=True)
        self.geo_fc = nn.Linear(1, 64)
        self.rating_fc = nn.Linear(1, 64)
        self.fc = nn.Linear(128 + 64 + 64, num_classes)

    def forward(self, mel, mfcc, geo_feature, rating_weight):
        mel = self.cnn_mel(mel)
        if mel.dim() != 4:
            raise ValueError(f"Expected 4D tensor for mel, got {mel.dim()}D tensor with shape {mel.shape}")
        mel = self.pool(mel).view(mel.size(0), -1)

        mfcc = self.cnn_mfcc(mfcc)
        if mfcc.dim() != 4:
            raise ValueError(f"Expected 4D tensor for mfcc, got {mfcc.dim()}D tensor with shape {mfcc.shape}")
        mfcc = self.pool(mfcc).view(mfcc.size(0), -1)

        x = torch.cat([mel, mfcc], dim=1).unsqueeze(1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        geo_emb = F.relu(self.geo_fc(geo_feature.unsqueeze(-1)))
        rating_emb = F.relu(self.rating_fc(rating_weight.unsqueeze(-1)))
        x = torch.cat([x, geo_emb, rating_emb], dim=1)
        return self.fc(x)


def get_dataloader(df, is_train=True):
    dataset = BirdDataset(df, is_train)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=is_train, num_workers=4, pin_memory=True)

def train_epoch(model, dataloader, optimizer, scheduler, scaler, device):
    model.train()
    loss_l = []
    for batch in tqdm(dataloader, desc="Training"):
        mel = batch['mel'].to(device, non_blocking=True)
        mfcc = batch['mfcc'].to(device, non_blocking=True)
        target = batch['target'].to(device, non_blocking=True)
        geo_feature = torch.tensor(batch['geo_feature'], dtype=torch.float32).to(device, non_blocking=True)
        rating_weight = torch.tensor(batch['rating_weight'], dtype=torch.float32).to(device, non_blocking=True)
        with autocast(device_type=device.type):
            pred = model(mel, mfcc, geo_feature, rating_weight)
            class_counts = working_df['primary_label'].value_counts().sort_index()
            num_samples = len(working_df)
            class_weights = (1.0 / class_counts) / (1.0 / num_samples) * (num_samples / len(class_counts))
            weights = torch.tensor([class_weights[label2id[id2label[i]]] for i in range(len(label_list))], dtype=torch.float32).to(device)
            loss = nn.BCEWithLogitsLoss(weight=weights)(pred, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        loss_l.append(loss.item())
        torch.cuda.empty_cache()
        gc.collect()
    return np.mean(loss_l)

def valid_epoch(model, dataloader, device):
    model.eval()
    preds_l, targets_l = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            mel = batch['mel'].to(device, non_blocking=True)
            mfcc = batch['mfcc'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            geo_feature = torch.tensor(batch['geo_feature'], dtype=torch.float32).to(device, non_blocking=True)
            rating_weight = torch.tensor(batch['rating_weight'], dtype=torch.float32).to(device, non_blocking=True)
            with autocast(device_type=device.type):
                pred = model(mel, mfcc, geo_feature, rating_weight)
            preds_l.append(torch.sigmoid(pred).cpu().numpy())
            targets_l.append(target.cpu().numpy())
    preds = np.concatenate(preds_l)
    targets = np.concatenate(targets_l)
    auc = roc_auc_score(targets, preds, average='macro')
    preds_binary = (preds > 0.5).astype(int)
    precision = precision_score(targets, preds_binary, average='macro', zero_division=0)
    recall = recall_score(targets, preds_binary, average='macro', zero_division=0)
    f1 = f1_score(targets, preds_binary, average='macro', zero_division=0)
    return auc, precision, recall, f1


if __name__ == "__main__":
    skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED)
    for fold, (train_idx, val_idx) in enumerate(skf.split(working_df, working_df['primary_label'])):
        print(f"\nStarting fold {fold}")
        train_df_fold = working_df.iloc[train_idx].reset_index(drop=True)
        val_df_fold = working_df.iloc[val_idx].reset_index(drop=True)

        train_loader = get_dataloader(train_df_fold, is_train=True)
        val_loader = get_dataloader(val_df_fold, is_train=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BirdCRNN(num_classes=len(label_list)).to(device)
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
        scaler = GradScaler()

        best_auc = 0
        for epoch in range(config.EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device)
            val_auc, val_precision, val_recall, val_f1 = valid_epoch(model, val_loader, device)
            print(f"Fold {fold}, Epoch {epoch}, Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
            scheduler.step(val_auc)
            if val_auc > best_auc:
                best_auc = val_auc
                os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)
                torch.save(model.state_dict(), f"{config.CHECKPOINT_PATH}/model_fold{fold}.pth")
                print(f"Saved best model for fold {fold} with AUC {best_auc:.4f}")
        gc.collect()
        torch.cuda.empty_cache()

