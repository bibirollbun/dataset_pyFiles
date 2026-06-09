import os
import csv
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.model_selection import KFold
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
from skimage import exposure, util
from skimage.transform import resize
from skimage.filters import gaussian
import librosa

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {DEVICE}")

# Константы
NUM_CLASSES = 24
SAMPLE_RATE = 48000
CLIP_DURATION_SEC = 10
CLIP_LENGTH_SAMPLES = CLIP_DURATION_SEC * SAMPLE_RATE
INITIAL_FMIN = float("inf")
INITIAL_FMAX = float("-inf")
LEARNING_RATE = 2e-4
EPOCHS = 20
NUM_FOLDS = 5


# Функции обработки изображений

class AudioAugmentationPipeline:
    def __init__(self):
        self.transforms = [
            self._horizontal_flip,
            self._vertical_flip,
            self._add_gaussian_noise,
            self._rescale_contrast
        ]

    def _horizontal_flip(self, img):
        return np.stack([img[:, ::-1]] * 3, axis=0)

    def _vertical_flip(self, img):
        return np.stack([img[::-1, :]] * 3, axis=0)

    def _add_gaussian_noise(self, img):
        noisy = util.random_noise(img)
        return np.stack([noisy] * 3, axis=0)

    def _rescale_contrast(self, img):
        enhanced = exposure.rescale_intensity(img)
        return np.stack([enhanced] * 3, axis=0)

    def apply_random_transform(self, img):
        transform = random.choice(self.transforms)
        return transform(img)


# Преобразование спектрограммы в нормализованное изображение

def spectrogram_to_image(spec: np.ndarray) -> np.ndarray:
    resized = resize(spec, (224, 400), anti_aliasing=True)
    eps = 1e-6

    normalized = (resized - resized.mean()) / (resized.std() + eps)
    min_val, max_val = normalized.min(), normalized.max()
    scaled = 255 * (normalized - min_val) / (max_val - min_val + eps)
    return scaled.astype(np.uint8)


# Создание модели
def build_model(num_classes=24):
    model = resnet50(weights=None)
    weights_path = "/kaggle/input/resnet50-weights-offline/resnet50-weights.pth"
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)
    
# Загрузка и анализ метаданных

train_metadata = pd.read_csv("/kaggle/input/rfcx-species-audio-detection/train_tp.csv")

f_min = min(train_metadata["f_min"]) * 0.9
f_max = max(train_metadata["f_max"]) * 1.1
F_MIN = int(f_min)
F_MAX = int(f_max)


# Обработка аудиофайлов

recording_ids = train_metadata["recording_id"].tolist()
species_labels = train_metadata["species_id"].tolist()
precomputed_spectrograms = {}

def extract_spectrogram(index: int):
    rec_id = recording_ids[index]
    label = species_labels[index]

    filepath = f"/kaggle/input/rfcx-species-audio-detection/train/{rec_id}.flac"
    audio, sr = librosa.load(filepath, sr=None)

    t_min_sec = train_metadata.at[index, "t_min"]
    t_max_sec = train_metadata.at[index, "t_max"]
    t_min = int(t_min_sec * sr)
    t_max = int(t_max_sec * sr)

    center = (t_min + t_max) // 2
    start = max(center - CLIP_LENGTH_SAMPLES // 2, 0)
    end = min(start + CLIP_LENGTH_SAMPLES, len(audio))
    if end - start < CLIP_LENGTH_SAMPLES:
        start = end - CLIP_LENGTH_SAMPLES

    segment = audio[int(start):int(end)]
    mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, fmin=F_MIN, fmax=F_MAX)
    db_spec = librosa.power_to_db(mel_spec, top_db=80)

    image = spectrogram_to_image(db_spec)
    return rec_id, image

# Параллельная предварительная обработка
with ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(extract_spectrogram, range(len(recording_ids))), 
                        total=len(recording_ids), desc="Предварительная обработка"))
    precomputed_spectrograms.update(results)


# Датасет для обучения

class SpectrogramDataset(Dataset):
    def __init__(self, recording_ids, labels, split_type, augmentation_pipeline=None):
        self.recording_ids = recording_ids
        self.labels = labels
        self.split_type = split_type
        self.augmentation = augmentation_pipeline
        self.cache = precomputed_spectrograms

    def __len__(self):
        return len(self.recording_ids)

    def __getitem__(self, idx):
        rec_id = self.recording_ids[idx]
        label = self.labels[idx]
        spec_img = self.cache[rec_id]

        if self.split_type == "train" and self.augmentation:
            image = self.augmentation.apply_random_transform(spec_img)
        else:
            image = np.stack([spec_img] * 3, axis=0)

        return torch.from_numpy(image).float(), torch.tensor(label, dtype=torch.long)


# Обучение модели с валидацией

def train_model(model, criterion, train_loader, val_loader, optimizer, scheduler):
    best_acc = 0.0
    best_weights = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = []

        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        model.eval()
        val_loss = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss.append(loss.item())
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        accuracy = np.mean(np.argmax(all_preds, axis=1) == all_labels)

        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)
        print(f"Эпоха {epoch:02d} | Train loss: {avg_train_loss:.5f} | "
              f"Val loss: {avg_val_loss:.5f} | Val acc: {accuracy:.5f}")

        scheduler.step(avg_val_loss)
        if accuracy > best_acc:
            best_acc = accuracy
            best_weights = model.state_dict()

    model.load_state_dict(best_weights)
    return model


# Кросс-валидация и обучение

kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=563)
augmenter = AudioAugmentationPipeline()

for fold, (train_idx, val_idx) in enumerate(kfold.split(recording_ids)):
    print(f"\nОбучение на фолде {fold}")

    X_train = [recording_ids[i] for i in train_idx]
    y_train = [species_labels[i] for i in train_idx]
    X_val = [recording_ids[i] for i in val_idx]
    y_val = [species_labels[i] for i in val_idx]

    train_dataset = SpectrogramDataset(X_train, y_train, "train", augmentation_pipeline=augmenter)
    val_dataset = SpectrogramDataset(X_val, y_val, "valid")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, drop_last=False)

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)

    trained_model = train_model(model, nn.CrossEntropyLoss(), train_loader, val_loader, optimizer, scheduler)
    torch.save(trained_model.state_dict(), f"./model{fold}.pt")

    del train_dataset, val_dataset, train_loader, val_loader, trained_model


# Загрузка ансамбля моделей

ensemble_models = []
for i in range(NUM_FOLDS):
    model = build_model()
    model.load_state_dict(torch.load(f"./model{i}.pt", map_location=DEVICE))
    model.eval()
    ensemble_models.append(model)
    os.remove(f"./model{i}.pt")

if torch.cuda.is_available():
    ensemble_models = [m.cuda() for m in ensemble_models]


# Обработка тестовых данных

def process_test_file(filename: str):
    path = f"/kaggle/input/rfcx-species-audio-detection/test/{filename}"
    audio, sr = librosa.load(path, sr=None)

    num_segments = int(np.ceil(len(audio) / CLIP_LENGTH_SAMPLES))
    segments = []

    for i in range(num_segments):
        start = i * CLIP_LENGTH_SAMPLES
        end = start + CLIP_LENGTH_SAMPLES
        if end > len(audio):
            segment = audio[-CLIP_LENGTH_SAMPLES:]
        else:
            segment = audio[start:end]

        mel = librosa.feature.melspectrogram(y=segment, sr=sr, fmin=F_MIN, fmax=F_MAX)
        db_mel = librosa.power_to_db(mel, top_db=80)
        img = spectrogram_to_image(db_mel)
        rgb_img = np.stack([img] * 3, axis=0)
        segments.append(rgb_img)

    return torch.tensor(np.stack(segments, axis=0), dtype=torch.float32)

def predict_on_file(filename: str, models):
    file_id = filename.split(".")[0]
    segments = process_test_file(filename)

    if torch.cuda.is_available():
        segments = segments.cuda()

    predictions = []
    for model in models:
        with torch.no_grad():
            outputs = model(segments)
            max_per_class = torch.max(outputs, dim=0).values
            predictions.append(max_per_class.cpu())

    averaged = torch.mean(torch.stack(predictions), dim=0)
    return [file_id] + [val.item() for val in averaged]

def create_submission(test_files, models, output_path="submission.csv"):
    header = ["recording_id"] + [f"s{i}" for i in range(NUM_CLASSES)]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(predict_on_file, fn, models) for fn in test_files]
            for future in tqdm(futures, desc="Создание предсказаний"):
                writer.writerow(future.result())


# Запуск на тестовых данных

test_files = os.listdir("/kaggle/input/rfcx-species-audio-detection/test/")
print(f"Найдено тестовых файлов: {len(test_files)}")

create_submission(test_files, ensemble_models)

