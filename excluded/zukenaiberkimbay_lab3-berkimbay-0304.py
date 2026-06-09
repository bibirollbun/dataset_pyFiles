import os
import random
import copy
import csv

import numpy as np
import pandas as pd
import librosa

from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.models import resnet50

from skimage.transform import resize
from skimage import exposure, util

from sklearn.model_selection import KFold

device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 24
SR = 48000
CLIP_SECONDS = 10
CLIP_SAMPLES = CLIP_SECONDS * SR

LEARNING_RATE = 1e-4
EPOCHS = 20
N_FOLDS = 5

train_csv_path = "/kaggle/input/rfcx-species-audio-detection/train_tp.csv"
train_audio_dir = "/kaggle/input/rfcx-species-audio-detection/train"
test_audio_dir = "/kaggle/input/rfcx-species-audio-detection/test"
sample_sub_path = "/kaggle/input/rfcx-species-audio-detection/sample_submission.csv"


train_df = pd.read_csv(train_csv_path)

f_min = train_df["f_min"].min()
f_max = train_df["f_max"].max()

f_min = int(f_min * 0.9)
f_max = int(f_max * 1.1)

def spec_to_image(spec):
    spec_resized = resize(spec, (224, 400))
    eps = 1e-6
    mean = spec_resized.mean()
    std = spec_resized.std()
    spec_norm = (spec_resized - mean) / (std + eps)
    spec_min = spec_norm.min()
    spec_max = spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min + eps)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled


class SimpleAugment:
    def __init__(self):
        self.transforms = [self.add_noise, self.change_contrast]

    def add_noise(self, image):
        noisy = util.random_noise(image)
        return np.stack([noisy] * 3)

    def change_contrast(self, image):
        img = exposure.rescale_intensity(image)
        return np.stack([img] * 3)

    def apply(self, image):
        func = random.choice(self.transforms)
        return func(image)

recording_ids = train_df["recording_id"].tolist()
labels = train_df["species_id"].tolist()

train_spects = {}

def process_one(index):
    rec_id = recording_ids[index]
    wav_path = os.path.join(train_audio_dir, rec_id + ".flac")
    wav, sr = librosa.load(wav_path, sr=None)

    t_min = int(train_df.at[index, "t_min"] * sr)
    t_max = int(train_df.at[index, "t_max"] * sr)

    center = int((t_min + t_max) / 2)
    start = max(center - CLIP_SAMPLES // 2, 0)
    end = min(start + CLIP_SAMPLES, len(wav))
    start = end - CLIP_SAMPLES if end - start < CLIP_SAMPLES else start

    segment = wav[start:end]
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=sr,
        fmin=f_min,
        fmax=f_max
    )
    mel_db = librosa.power_to_db(mel, top_db=80)
    img = spec_to_image(mel_db)
    return rec_id, img

with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_one, range(len(train_df))))

for rec_id, img in results:
    train_spects[rec_id] = img

augmenter = SimpleAugment()


class SpectrogramDataset(Dataset):
    def __init__(self, rec_ids, targets, mode):
        self.rec_ids = rec_ids
        self.targets = targets
        self.mode = mode
        self.storage = train_spects

    def __len__(self):
        return len(self.rec_ids)

    def __getitem__(self, idx):
        rec_id = self.rec_ids[idx]
        label = self.targets[idx]
        img = self.storage[rec_id]

        if self.mode == "train":
            img3 = augmenter.apply(img)
        else:
            img3 = np.stack([img] * 3)

        return img3, label

def build_model():
    model = resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, in_features // 2),
        nn.ReLU(inplace=True),
        nn.Linear(in_features // 2, NUM_CLASSES)
    )
    return model.to(device)

criterion = nn.CrossEntropyLoss()



def train_one_fold(model, train_loader, valid_loader, optimizer, scheduler):
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for batch_imgs, batch_labels in train_loader:
            optimizer.zero_grad()
            x = batch_imgs.to(device, dtype=torch.float32)
            y = batch_labels.to(device, dtype=torch.long)
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        valid_losses = []
        all_true = []
        all_pred = []

        with torch.no_grad():
            for batch_imgs, batch_labels in valid_loader:
                x = batch_imgs.to(device, dtype=torch.float32)
                y = batch_labels.to(device, dtype=torch.long)
                outputs = model(x)
                loss = criterion(outputs, y)
                valid_losses.append(loss.item())
                all_true.append(y.cpu().numpy())
                all_pred.append(outputs.cpu().numpy())

        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
        acc = np.mean(all_pred.argmax(axis=1) == all_true)

        train_loss = float(np.mean(train_losses))
        valid_loss = float(np.mean(valid_losses))

        print("epoch:", epoch, "train_loss:", round(train_loss, 4),
              "val_loss:", round(valid_loss, 4),
              "val_acc:", round(acc, 4))

        scheduler.step(valid_loss)
        if acc > best_acc:
            best_acc = acc
            best_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_wts)
    return model

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=563)

all_rec_ids = np.array(recording_ids)
all_labels = np.array(labels)

for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(all_rec_ids, all_labels)):
    print("Fold", fold_idx)
    x_train = all_rec_ids[train_idx]
    y_train = all_labels[train_idx]
    x_valid = all_rec_ids[valid_idx]
    y_valid = all_labels[valid_idx]

    train_dataset = SpectrogramDataset(x_train, y_train, mode="train")
    valid_dataset = SpectrogramDataset(x_valid, y_valid, mode="valid")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, drop_last=False)

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)

    model = train_one_fold(model, train_loader, valid_loader, optimizer, scheduler)
    torch.save(model.state_dict(), f"model_fold_{fold_idx}.pth")

    del train_dataset, valid_dataset, train_loader, valid_loader, model
    torch.cuda.empty_cache()



def load_test_file(file_name):
    path = os.path.join(test_audio_dir, file_name)
    wav, sr = librosa.load(path, sr=None)
    total_len = len(wav)
    segments = int(np.ceil(total_len / CLIP_SAMPLES))

    clips = []
    for i in range(segments):
        start = i * CLIP_SAMPLES
        end = start + CLIP_SAMPLES
        if end > total_len:
            start = max(total_len - CLIP_SAMPLES, 0)
            end = total_len
        segment = wav[start:end]
        if len(segment) < CLIP_SAMPLES:
            pad = np.zeros(CLIP_SAMPLES - len(segment))
            segment = np.concatenate([segment, pad])
        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=sr,
            fmin=f_min,
            fmax=f_max
        )
        mel_db = librosa.power_to_db(mel, top_db=80)
        img = spec_to_image(mel_db)
        img3 = np.stack([img] * 3)
        clips.append(img3)
    return np.stack(clips)

sample_sub = pd.read_csv(sample_sub_path)
test_files = sample_sub["recording_id"].tolist()

fold_models = []
for fold_idx in range(N_FOLDS):
    m = build_model()
    m.load_state_dict(torch.load(f"model_fold_{fold_idx}.pth", map_location=device))
    m.eval()
    fold_models.append(m)

pred_rows = []

for rec_id in test_files:
    file_name = rec_id + ".flac"
    batch = load_test_file(file_name)
    batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)

    with torch.no_grad():
        fold_probs = []
        for m in fold_models:
            outputs = m(batch_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            fold_probs.append(probs)
        fold_probs = np.mean(fold_probs, axis=0)

    mean_probs = fold_probs.mean(axis=0)

    row = [rec_id] + [float(mean_probs[c]) for c in range(NUM_CLASSES)]
    pred_rows.append(row)

cols = ["recording_id"] + [f"s{i}" for i in range(NUM_CLASSES)]
pred_df = pd.DataFrame(pred_rows, columns=cols)
pred_df.to_csv("submission.csv", index=False)


import pandas as pd
sub = pd.read_csv("submission.csv")
sub.head()

