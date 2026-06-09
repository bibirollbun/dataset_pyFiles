import os
import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from sklearn.model_selection import KFold
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import librosa
from skimage import exposure, util
from skimage.transform import resize

warnings.filterwarnings("ignore")

# Устройство
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {DEVICE}")

NUM_CLASSES = 24
SAMPLE_RATE = 48_000
SEGMENT_LENGTH_SAMPLES = 10 * SAMPLE_RATE  # 10 секунд
LEARNING_RATE = 2e-4
NUM_EPOCHS = 20
NUM_FOLDS = 5


def compute_frequency_bounds(metadata_path: str) -> Tuple[int, int]:
    """Определяет глобальные границы частот с запасом"""
    df = pd.read_csv(metadata_path)
    f_min = int(df["f_min"].min() * 0.9)
    f_max = int(df["f_max"].max() * 1.1)
    return f_min, f_max


F_MIN, F_MAX = compute_frequency_bounds("/kaggle/input/rfcx-species-audio-detection/train_tp.csv")


def spec_to_image(spec: np.ndarray, target_shape: Tuple[int, int] = (224, 400)) -> np.ndarray:
    """Нормализует и масштабирует спектрограмму в uint8 изображение"""
    spec_resized = resize(spec, target_shape, anti_aliasing=True)
    eps = 1e-6
    spec_norm = (spec_resized - spec_resized.mean()) / (spec_resized.std() + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min + eps)
    return spec_scaled.astype(np.uint8)


def get_pretrained_resnet(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Загружает ResNet50 и заменяет последний слой под нужное число классов"""
    model = resnet50(weights=None)
    weights_path = "/kaggle/input/resnet50-weights/resnet50-weights.pth"
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)



class AudioImageAugmenter:
    """Применяет случайные аугментации к одноканальному изображению спектрограммы"""

    def __init__(self):
        self.transforms = [
            self._horizontal_flip,
            self._vertical_flip,
            self._add_noise,
            self._contrast_stretch,
        ]

    @staticmethod
    def _horizontal_flip(img: np.ndarray) -> np.ndarray:
        return np.stack([img[:, ::-1]] * 3, axis=0)

    @staticmethod
    def _vertical_flip(img: np.ndarray) -> np.ndarray:
        return np.stack([img[::-1, :]] * 3, axis=0)

    @staticmethod
    def _add_noise(img: np.ndarray) -> np.ndarray:
        noisy = util.random_noise(img, mode="gaussian")
        return np.stack([noisy] * 3, axis=0)

    @staticmethod
    def _contrast_stretch(img: np.ndarray) -> np.ndarray:
        stretched = exposure.rescale_intensity(img)
        return np.stack([stretched] * 3, axis=0)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Применяет случайную аугментацию"""
        transform = random.choice(self.transforms)
        return transform(img)


def process_single_audio(
    recording_id: str,
    species_id: int,
    t_min: float,
    t_max: float,
    audio_dir: str = "/kaggle/input/rfcx-species-audio-detection/train",
    sr_target: int = SAMPLE_RATE,
) -> Tuple[str, np.ndarray]:
    """Загружает аудиофайл, вырезает сегмент, строит мел-спектрограмму и конвертирует в изображение"""
    audio_path = os.path.join(audio_dir, f"{recording_id}.flac")
    wav, sr = librosa.load(audio_path, sr=None)

    t_min_samp = int(t_min * sr)
    t_max_samp = int(t_max * sr)
    center = int(round((t_min_samp + t_max_samp) / 2))
    start = max(center - SEGMENT_LENGTH_SAMPLES // 2, 0)
    end = min(start + SEGMENT_LENGTH_SAMPLES, len(wav))
    if end - start < SEGMENT_LENGTH_SAMPLES:
        start = max(0, end - SEGMENT_LENGTH_SAMPLES)

    segment = wav[start:end]
    mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, fmin=F_MIN, fmax=F_MAX)
    mel_db = librosa.power_to_db(mel_spec, top_db=80)
    image = spec_to_image(mel_db)
    return recording_id, image


def load_training_data(metadata_path: str) -> dict:
    """Параллельно обрабатывает все обучающие аудиофайлы и возвращает словарь {recording_id: image}"""
    df = pd.read_csv(metadata_path)
    audio_data = {}

    def _process_row(idx: int):
        row = df.iloc[idx]
        return process_single_audio(
            recording_id=row["recording_id"],
            species_id=row["species_id"],
            t_min=row["t_min"],
            t_max=row["t_max"],
        )

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(_process_row, range(len(df))), total=len(df)))

    for rec_id, img in results:
        audio_data[rec_id] = img

    return audio_data


AUDIO_CACHE = load_training_data("/kaggle/input/rfcx-species-audio-detection/train_tp.csv")


class AudioImageDataset(Dataset):
    def __init__(
        self,
        recording_ids: List[str],
        labels: List[int],
        data_split: str,  # "train" или "valid"
        audio_cache: dict,
        augmenter: AudioImageAugmenter = None,
    ):
        self.recording_ids = recording_ids
        self.labels = labels
        self.split = data_split
        self.cache = audio_cache
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.recording_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec_id = self.recording_ids[idx]
        label = self.labels[idx]
        img = self.cache[rec_id]

        if self.split == "train" and self.augmenter is not None:
            img_rgb = self.augmenter(img)
        else:
            img_rgb = np.stack([img] * 3, axis=0)

        return torch.from_numpy(img_rgb).float(), label


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int = NUM_EPOCHS,
) -> nn.Module:
    best_model_state = None
    best_acc = 0.0

    for epoch in tqdm(range(1, num_epochs + 1), desc="Эпохи"):
        # Обучение
        model.train()
        train_losses = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Валидация
        model.eval()
        val_losses = []
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_losses.append(loss.item())
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(targets.cpu().numpy())

        # Оценка
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds).argmax(axis=1)
        val_acc = np.mean(y_pred == y_true)
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        print(
            f"Эпоха {epoch:02d} | Train Loss: {avg_train_loss:.5f} | "
            f"Val Loss: {avg_val_loss:.5f} | Val Acc: {val_acc:.5f}"
        )

        scheduler.step(avg_val_loss)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()

    model.load_state_dict(best_model_state)
    return model


def run_kfold_training():
    df = pd.read_csv("/kaggle/input/rfcx-species-audio-detection/train_tp.csv")
    recording_ids = df["recording_id"].tolist()
    labels = df["species_id"].tolist()

    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=563)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(recording_ids)):
        print(f"\n=== Обучение на фолде {fold} ===")

        X_train = [recording_ids[i] for i in train_idx]
        y_train = [labels[i] for i in train_idx]
        X_val = [recording_ids[i] for i in val_idx]
        y_val = [labels[i] for i in val_idx]

        train_dataset = AudioImageDataset(
            recording_ids=X_train,
            labels=y_train,
            data_split="train",
            audio_cache=AUDIO_CACHE,
            augmenter=AudioImageAugmenter(),
        )
        val_dataset = AudioImageDataset(
            recording_ids=X_val,
            labels=y_val,
            data_split="valid",
            audio_cache=AUDIO_CACHE,
        )

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, drop_last=False)

        model = get_pretrained_resnet()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)
        criterion = nn.CrossEntropyLoss()

        model = train_one_fold(model, train_loader, val_loader, criterion, optimizer, scheduler)
        torch.save(model.state_dict(), f"./model{fold}.pt")

        # Очистка памяти
        del model, train_dataset, val_dataset, train_loader, val_loader


run_kfold_training()


def load_test_segments(test_file: str, test_dir: str = "/kaggle/input/rfcx-species-audio-detection/test") -> List[np.ndarray]:
    """Разбивает тестовый аудиофайл на 10-секундные сегменты и конвертирует в RGB-изображения"""
    path = os.path.join(test_dir, test_file)
    wav, sr = librosa.load(path, sr=None)
    num_segments = int(np.ceil(len(wav) / SEGMENT_LENGTH_SAMPLES))
    segments = []

    for i in range(num_segments):
        start = i * SEGMENT_LENGTH_SAMPLES
        end = start + SEGMENT_LENGTH_SAMPLES
        if end > len(wav):
            segment = wav[-SEGMENT_LENGTH_SAMPLES:]
        else:
            segment = wav[start:end]

        mel = librosa.feature.melspectrogram(y=segment, sr=sr, fmin=F_MIN, fmax=F_MAX)
        mel_db = librosa.power_to_db(mel, top_db=80)
        img = spec_to_image(mel_db)
        img_rgb = np.stack([img] * 3, axis=0)
        segments.append(img_rgb)

    return segments


def ensemble_predict(
    test_file: str,
    models: List[nn.Module],
) -> List[Any]:
    segments = load_test_segments(test_file)
    batch = torch.stack([torch.from_numpy(s).float() for s in segments]).to(DEVICE)

    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model(batch)
            max_output, _ = torch.max(outputs, dim=0)  # максимум по сегментам
            predictions.append(max_output.cpu())

    mean_prediction = torch.mean(torch.stack(predictions), dim=0).tolist()
    recording_id = Path(test_file).stem
    return [recording_id] + mean_prediction


def save_submission_file(rows: List[List], filename: str = "submission.csv"):
    header = ["recording_id"] + [f"s{i}" for i in range(NUM_CLASSES)]
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def generate_and_save_submission(test_dir: str = "/kaggle/input/rfcx-species-audio-detection/test"):
    test_files = [f for f in os.listdir(test_dir) if f.endswith(".flac")]
    print(f"Найдено тестовых файлов: {len(test_files)}")

    # Загрузка ансамбля
    models = []
    for fold in range(NUM_FOLDS):
        model = get_pretrained_resnet()
        model.load_state_dict(torch.load(f"./model{fold}.pt", map_location=DEVICE))
        models.append(model)
        os.remove(f"./model{fold}.pt")  # очистка

    if torch.cuda.is_available():
        models = [m.cuda() for m in models]

    # Предсказание
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(ensemble_predict, f, models) for f in test_files]
        for future in tqdm(futures, desc="Инференс"):
            results.append(future.result())

    save_submission_file(results)


generate_and_save_submission()

