import os
import numpy as np
import pandas as pd
import cv2
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import models
from IPython.display import Audio, display


# Функция оценки MAP@3
def compute_map_at_3(true_labels, pred_probs):
    top3 = np.argsort(pred_probs, axis=1)[:, ::-1][:, :3]
    scores = []
    for i, true in enumerate(true_labels):
        if true in top3[i]:
            rank = np.where(top3[i] == true)[0][0] + 1
            scores.append(1.0 / rank)
        else:
            scores.append(0.0)
    return np.mean(scores)


# Пути к данным
TRAIN_AUDIO_DIR = "../input/freesound-audio-tagging/audio_train/"
TEST_AUDIO_DIR = "../input/freesound-audio-tagging/audio_test/"
TRAIN_CSV_PATH = "../input/freesound-audio-tagging/train.csv"
SAMPLE_SUB_PATH = "../input/freesound-audio-tagging/sample_submission.csv"

# Загрузка меток
labels_df = pd.read_csv(TRAIN_CSV_PATH)
print("Пример обучающих данных:")
print(labels_df.head())


SR = 22050          # Частота дискретизации
DURATION = 5.0      # Длительность в секундах
N_MELS = 128        # Количество мел-фильтров
FMAX = 11025        # Максимальная частота
IMG_H, IMG_W = 128, 128  # Размер изображения спектрограммы


class FreesoundDataset(Dataset):
    def __init__(self, metadata, audio_dir, label_map=None, is_test=False):
        self.meta = metadata
        self.audio_dir = audio_dir
        self.label_map = label_map
        self.is_test = is_test

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        fname = self.meta.iloc[idx]["fname"]
        full_path = os.path.join(self.audio_dir, fname)

        waveform, _ = librosa.load(full_path, sr=SR, duration=DURATION)
        if len(waveform) < SR * DURATION:
            waveform = np.pad(waveform, (0, int(SR * DURATION - len(waveform))), mode="constant")

        # Преобразование в мел-спектрограмму
        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=SR, n_mels=N_MELS, fmax=FMAX)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Приведение к фиксированному размеру
        try:
            resized = cv2.resize(mel_db, (IMG_W, IMG_H))
        except:
            resized = np.zeros((IMG_H, IMG_W))

        # Трёхканальное изображение для EfficientNet
        image = np.stack([resized] * 3, axis=0).astype(np.float32)

        if self.is_test:
            return torch.tensor(image)
        else:
            label_str = self.meta.iloc[idx]["label"]
            label_id = self.label_map[label_str]
            return torch.tensor(image), label_id


# Определяем cpu или gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {DEVICE}")


#Пример звука из датасета
sample_file = labels_df.iloc[0]["fname"]
sample_label = labels_df.iloc[0]["label"]
print(f"\nМетка примера: {sample_label}")
display(Audio(filename=os.path.join(TRAIN_AUDIO_DIR, sample_file)))


verified_df = labels_df[labels_df["manually_verified"] == 1].reset_index(drop=True)
print(f"Количество записей с проверенными метками: {len(verified_df)}")


# Метки
class_names = sorted(verified_df["label"].unique())
num_classes = len(class_names)
print(f"Количество уникальных меток: {num_classes}")


label2id = {}
id2label = {}
for i, cls in enumerate(class_names):
    label2id[cls] = i
    id2label[i] = cls


# Разделение на обучающую и валидационную выборки
train_split, val_split = train_test_split(
    verified_df,
    test_size=0.2,
    stratify=verified_df["label"],
    random_state=42
)
print(f"Обучающих примеров: {len(train_split)}")
print(f"Валидационных примеров: {len(val_split)}")


# Создание загрузчиков
BATCH_SIZE = 32
train_dataset = FreesoundDataset(train_split, TRAIN_AUDIO_DIR, label2id)
val_dataset = FreesoundDataset(val_split, TRAIN_AUDIO_DIR, label2id)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# Инициализация модели
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(DEVICE)

# Оптимизатор и функция потерь
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)



# Обучение
best_score = 0.0
EPOCHS = 15

for epoch in range(EPOCHS):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Валидация
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.concatenate(all_targets)
    val_metric = compute_map_at_3(all_targets, all_preds)
    print(f"Эпоха {epoch + 1}/{EPOCHS} — MAP@3 на валидации: {val_metric:.4f}")

    if val_metric > best_score:
        best_score = val_metric
        torch.save(model.state_dict(), "model.pth")

    scheduler.step(val_metric)


# Подготовка тестовых данных
test_meta = pd.read_csv(SAMPLE_SUB_PATH)
test_dataset = FreesoundDataset(test_meta, TEST_AUDIO_DIR, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model.load_state_dict(torch.load("model.pth"))
model.eval()

test_probs = []
with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(DEVICE)
        logits = model(inputs)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        test_probs.append(probs)

test_probs = np.vstack(test_probs)



# Формирование решения в формате csv
predictions = []
for i, row in test_meta.iterrows():
    ids = np.argsort(test_probs[i])[::-1][:3]
    labels = [id2label[idx] for idx in ids]
    predictions.append({"fname": row["fname"], "label": " ".join(labels)})

submission_df = pd.DataFrame(predictions)
submission_df.to_csv("submission.csv", index=False)


print("\nПримеры меток на тестовом наборе:")
examples = submission_df.iloc[20:26]
for _, ex in examples.iterrows():
    print(f"\nФайл: {ex['fname']}")
    print(f"Предсказанные метки: {ex['label']}")
    display(Audio(filename=os.path.join(TEST_AUDIO_DIR, ex["fname"])))

