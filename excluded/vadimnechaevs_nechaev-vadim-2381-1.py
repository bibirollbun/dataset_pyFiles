import kagglehub

audio_dataset_path = kagglehub.competition_download("freesound-audio-tagging")
efficientnet_weights_path = kagglehub.model_download(
    "tensorflow/efficientnet/TensorFlow2/b0-classification/1"
)

print("Источники успешно загружены.")




import os
import cv2
import numpy as np
import pandas as pd

import librosa
import librosa.display

from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models




device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device}")




meta = pd.read_csv("../input/freesound-audio-tagging/train.csv")

# Формируем словарь классов
unique_tags = sorted(meta["label"].unique())
tag_to_id = {tag: idx for idx, tag in enumerate(unique_tags)}
print(f"Количество классов: {len(unique_tags)}")

# Пути к данным
TRAIN_AUDIO = "../input/freesound-audio-tagging/audio_train/"
TEST_AUDIO = "../input/freesound-audio-tagging/audio_test/"




class FSDataset(Dataset):
    def __init__(self, df, is_test=False):
        self.df = df
        self.test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["fname"]
        label = row["label"]

        filepath = (TEST_AUDIO if self.test else TRAIN_AUDIO) + filename

        try:
            wave, _ = librosa.load(filepath)
            mel = librosa.feature.melspectrogram(y=wave)
            mel = librosa.power_to_db(mel, ref=np.max)

            img = cv2.resize(mel, (128, 128))
        except Exception:
            img = np.zeros((128, 128))

        tensor = torch.tensor(np.repeat(img[None], 3, axis=0), dtype=torch.float32)

        if self.test:
            return tensor

        return tensor, tag_to_id[label]




BATCH = 64
EPOCHS = 10

train_df, val_df = train_test_split(
    meta, test_size=0.2, shuffle=True, random_state=5
)

train_data = FSDataset(train_df)
val_data = FSDataset(val_df)

train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=BATCH, shuffle=True)

print(f"Тренировочных примеров: {len(train_df)}")
print(f"Валидационных примеров: {len(val_df)}")




model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(1280, len(unique_tags))
model = model.to(device)




criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)




for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    total_train_correct = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        total_train_correct += (logits.argmax(1) == batch_y).sum().item()

    # --- Оценка ---
    model.eval()
    total_val_loss = 0
    total_val_correct = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_val_loss += loss.item()
            total_val_correct += (logits.argmax(1) == batch_y).sum().item()

    print(
        f"Epoch {epoch}: "
        f"train_loss={total_train_loss/len(train_loader):.4f}, "
        f"val_loss={total_val_loss/len(val_loader):.4f}, "
        f"train_acc={total_train_correct/len(train_df):.4f}, "
        f"val_acc={total_val_correct/len(val_df):.4f}"
    )




test_csv = pd.read_csv("../input/freesound-audio-tagging/sample_submission.csv")
test_dataset = FSDataset(test_csv, is_test=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)

model.eval()
raw_preds = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        raw_preds.append(out.cpu())

raw_preds = torch.cat(raw_preds)
probs = torch.softmax(raw_preds, dim=1).numpy()




submission = test_csv.copy()

for i in range(len(submission)):
    pred_class = unique_tags[np.argmax(probs[i])]
    submission.loc[i, "label"] = pred_class

submission.to_csv("submission_final.csv", index=False)

submission.head()


