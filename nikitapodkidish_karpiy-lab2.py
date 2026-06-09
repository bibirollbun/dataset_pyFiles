import sys
import os
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


package_path = '../input/resnest50-fast-package/resnest-0.0.6b20200701/resnest'
if Path(package_path).exists():
    shutil.copytree(package_path, 'resnet', dirs_exist_ok=True)
    sys.path.append('./resnet')
    from resnest.torch import resnest50
else:
    print("Пакет resnest не найден")


# --- Конфигурация ---
@dataclass
class Config:
    num_classes: int = 397
    sr: int = 32_000
    duration: int = 5
    threshold: float = 0.21
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_audio_dir: Path = Path("../input/birdclef-2021/test_soundscapes")
    train_metadata: Path = Path("../input/birdclef-2021/train_metadata.csv")
    checkpoint_path: Path = Path("../input/kkiller-birdclef-models-public/birdclef_resnest50_fold0_epoch_10_f1_val_06471_20210417161101.pth")

cfg = Config()
print(f"Запуск на устройстве: {cfg.device}")


# --- Утилиты для обработки аудио ---
class AudioTransform:
    """Класс для превращения аудиосигнала в Mel-спектрограмму (картинку)"""
    def __init__(self, sr=32000, n_mels=128, fmin=0, fmax=None):
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sr // 2
        self.n_fft = sr // 10
        self.hop_length = sr // 40  # (10 * 4)

    def audio_to_melspec(self, audio: np.array) -> np.array:
        """Генерирует мел-спектрограмму из сырого аудио"""
        melspec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sr, 
            n_mels=self.n_mels, 
            fmin=self.fmin, 
            fmax=self.fmax, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        melspec = librosa.power_to_db(melspec).astype(np.float32)
        return melspec

    def to_image(self, melspec: np.array) -> np.array:
        """Переводит спектрограмму в 3-канальное изображение (RGB)"""
        mean, std = melspec.mean(), melspec.std()
        image = (melspec - mean) / (std + 1e-6)
        
        _min, _max = image.min(), image.max()
        if (_max - _min) > 1e-6:
            image = np.clip(image, _min, _max)
            image = 255 * (image - _min) / (_max - _min)
        else:
            image = np.zeros_like(image)
            
        image = image.astype(np.uint8)
        image = np.stack([image, image, image]) 
        
        return image.astype("float32") / 255.0


# --- Dataset ---
class InferenceDataset:
    def __init__(self, file_paths, cfg: Config):
        self.file_paths = file_paths
        self.cfg = cfg
        self.transformer = AudioTransform(sr=cfg.sr)
        self.chunk_len = cfg.duration * cfg.sr

    def __len__(self):
        return len(self.file_paths)

    def read_audio(self, filepath):
        """Читает аудио и ресемплирует если нужно"""
        audio, orig_sr = sf.read(filepath, dtype="float32")
        if orig_sr != self.cfg.sr:
            audio = librosa.resample(audio, orig_sr, self.cfg.sr, res_type="kaiser_fast")
        return audio

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        audio = self.read_audio(path)
        
        # нарезка аудио на куски по 5 секунд
        chunks = []
        for i in range(0, len(audio), self.chunk_len):
            segment = audio[i : i + self.chunk_len]
            # отбрасываем, если кусок короче 5 секунд (конец файла)
            if len(segment) < self.chunk_len:
                continue
            
            # звук в картинку
            melspec = self.transformer.audio_to_melspec(segment)
            image = self.transformer.to_image(melspec)
            chunks.append(image)
            
        return np.stack(chunks) if chunks else np.array([])


# --- Модель ---
def load_model(path: Path, device: torch.device):
    """Загружает архитектуру ResNeSt50 и веса"""
    model = resnest50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
    
    state_dict = torch.load(path, map_location='cpu')
    clean_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    model.to(device)
    model.eval()
    return model


# --- Подготовка данных ---
test_files = list(cfg.test_audio_dir.glob("*.ogg"))
if not test_files:
    print("Тестовые файлы не найдены, используем примеры из train...")
    cfg.test_audio_dir = Path("../input/birdclef-2021/train_soundscapes")
    test_files = list(cfg.test_audio_dir.glob("*.ogg"))[:5]

print(f"Найдено файлов для обработки: {len(test_files)}")

df_meta = pd.read_csv(cfg.train_metadata)
LABELS = sorted(df_meta["primary_label"].unique())
ID_TO_BIRD = {i: label for i, label in enumerate(LABELS)}


# --- Инференс (Предсказание) ---
model = load_model(cfg.checkpoint_path, cfg.device)
dataset = InferenceDataset(test_files, cfg)

predictions = []
row_ids = []

print("Начинаем классификацию...")
with torch.no_grad():
    for i in tqdm(range(len(dataset))):
        batch_images = dataset[i] # Получение 5-сек отрезков из одного файла
        if len(batch_images) == 0:
            continue
            
        file_name = test_files[i].stem
        file_parts = file_name.split("_")
        site = file_parts[1]
        
        # Перевод в тензор и отправка на GPU
        inputs = torch.from_numpy(batch_images).to(cfg.device)
        
        # Прогон через модель
        logits = model(inputs)
        probs = torch.sigmoid(logits).cpu().numpy() # Вероятности (0-1)
        
        # Ответ для каждого отрезка
        for chunk_idx, prob_vec in enumerate(probs):
            seconds = (chunk_idx + 1) * 5
            row_id = f"{file_parts[0]}_{site}_{seconds}"
            
            # Фильтр по порогу
            detected_indices = np.where(prob_vec > cfg.threshold)[0]
            
            if len(detected_indices) > 0:
                birds = " ".join([ID_TO_BIRD[idx] for idx in detected_indices])
            else:
                birds = "nocall"
                
            row_ids.append(row_id)
            predictions.append(birds)


# --- Сохранение ---
submission = pd.DataFrame({
    "row_id": row_ids,
    "birds": predictions
})

sample_sub_path = Path("../input/birdclef-2021/sample_submission.csv")
if sample_sub_path.exists():
    sample_sub = pd.read_csv(sample_sub_path)
    submission = sample_sub.drop("birds", axis=1).merge(submission, on="row_id", how="left")
    submission["birds"] = submission["birds"].fillna("nocall")

submission.to_csv("submission.csv", index=False)
print("Файл submission.csv сохранен.")
print(submission.head())

