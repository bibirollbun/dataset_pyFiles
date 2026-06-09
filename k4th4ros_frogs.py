# Импорты
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import librosa
import os
import csv
import warnings
import gc
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from skimage.transform import resize
from skimage import exposure
import copy

warnings.filterwarnings('ignore')
print('Библиотеки загружены')


# Конфигурация
class Config:
    NUM_CLASSES = 24
    SAMPLE_RATE = 48000
    SEGMENT_LENGTH = 10
    SEGMENT_SAMPLES = SEGMENT_LENGTH * SAMPLE_RATE
    
    IMG_HEIGHT = 224
    IMG_WIDTH = 384
    TOP_DB = 80
    
    NUM_FOLDS = 5
    EPOCHS = 18
    BATCH_SIZE = 6
    LEARNING_RATE = 1.5e-4
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_SEED = 2024

cfg = Config()
print(f'Устройство: {cfg.DEVICE}')
print(f'Сегменты: {cfg.SEGMENT_LENGTH} сек')
print(f'Изображения: {cfg.IMG_HEIGHT}×{cfg.IMG_WIDTH}')
print(f'Batch size: {cfg.BATCH_SIZE} (уменьшен для памяти)')
print(f'Фолдов: {cfg.NUM_FOLDS}, Эпох: {cfg.EPOCHS}')


def audio_to_spectr(audio_segment, sr, f_min, f_max):
    mel_spec = librosa.feature.melspectrogram(
        y=audio_segment, 
        sr=sr, 
        fmin=f_min, 
        fmax=f_max
    )
    mel_db = librosa.power_to_db(mel_spec, top_db=cfg.TOP_DB)
    return mel_db


def normalize_spectr(spec):
    spec_resized = resize(spec, (cfg.IMG_HEIGHT, cfg.IMG_WIDTH))
    
    # Z-score нормализация
    eps = 1e-6
    mean = spec_resized.mean()
    std = spec_resized.std()
    spec_norm = (spec_resized - mean) / (std + eps)
    
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min + eps)
    spec_scaled = spec_scaled.astype(np.uint8)
    
    return spec_scaled


def apply_contrast(img, вероятность=0.5):
    if np.random.rand() < вероятность:
        return exposure.rescale_intensity(img)
    return img


print('Функции обработки готовы')


# Загрузка и предобработка данных

print('Загрузка метаданных...')
train_tp = pd.read_csv('/kaggle/input/rfcx-species-audio-detection/train_tp.csv')

f_min_global = int(train_tp['f_min'].min() * 0.9)
f_max_global = int(train_tp['f_max'].max() * 1.1)

print(f'Частотный диапазон: {f_min_global}-{f_max_global} Hz')
print(f'Записей: {len(train_tp)}, Видов: {cfg.NUM_CLASSES}')

# Предобработка всех аудио
print('\nПредобработка аудио...')
аудио_кэш = {}

for idx, row in tqdm(train_tp.iterrows(), total=len(train_tp)):
    recording_id = row['recording_id']
    
    # Загрузка аудио
    audio_path = f'/kaggle/input/rfcx-species-audio-detection/train/{recording_id}.flac'
    wav, sr = librosa.load(audio_path, sr=None)
    
    # Извлечение сегмента с центром на активности
    t_min = int(row['t_min'] * sr)
    t_max = int(row['t_max'] * sr)
    центр = int((t_min + t_max) / 2)
    
    начало = max(0, центр - cfg.SEGMENT_SAMPLES // 2)
    конец = min(len(wav), начало + cfg.SEGMENT_SAMPLES)
    
    if конец - начало < cfg.SEGMENT_SAMPLES:
        начало = max(0, конец - cfg.SEGMENT_SAMPLES)
    
    segment = wav[начало:конец]
    
    # Преобразование в спектрограмму
    mel_spec = audio_to_spectr(segment, sr, f_min_global, f_max_global)
    img = normalize_spectr(mel_spec)
    
    аудио_кэш[recording_id] = img

print(f'✓ Предобработано {len(аудио_кэш)} записей')


# Dataset с Mixup аугментацией

class RFCXDataset(Dataset):
    def __init__(self, recording_ids, labels, cache, is_train=True, use_mixup=True):
        self.recording_ids = recording_ids
        self.labels = labels
        self.cache = cache
        self.is_train = is_train
        self.use_mixup = use_mixup and is_train
    
    def __len__(self):
        return len(self.recording_ids)
    
    def __getitem__(self, idx):
        rec_id = self.recording_ids[idx]
        label = self.labels[idx]
        
        # Получаем изображение из кэша
        img = self.cache[rec_id].copy()
        
        # Аугментации для тренировки
        if self.is_train:
            # Контраст
            img = apply_contrast(img, вероятность=0.5)
            
            # Horizontal flip
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
            
            # Vertical flip
            if np.random.rand() < 0.3:
                img = img[::-1, :]
        
        # Преобразование в 3 канала (RGB для ImageNet)
        img = np.stack([img, img, img], axis=0)  # (3, H, W)
        
        return torch.FloatTensor(img), label


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


print('Dataset с Mixup готов')


# Модель: EfficientNet-B0 вместо ResNet50

def создать_модель():
    # Используем pretrained EfficientNet-B0
    model = models.efficientnet_b0(pretrained=True)
    
    # Заменяем классификатор
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_features, cfg.NUM_CLASSES)
    )
    
    return model.to(cfg.DEVICE)


# Тестируем
test_model = создать_модель()
total_params = sum(p.numel() for p in test_model.parameters())
print(f'EfficientNet-B0: {total_params:,} параметров')
del test_model


# Функция обучения с Mixup и Label Smoothing

def обучить_модель(model, train_loader, val_loader, fold_num):
    # Loss с label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    
    # Оптимизатор
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )
    
    # Cosine Annealing scheduler (вместо ReduceLROnPlateau)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=cfg.EPOCHS,
        eta_min=1e-6
    )
    
    best_acc = 0.0
    best_weights = None
    
    print(f'\n=== Fold {fold_num} ===')
    
    for epoch in range(1, cfg.EPOCHS + 1):
        # TRAIN
        model.train()
        train_losses = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)
            
            # Mixup аугментация
            if np.random.rand() < 0.5:  # 50% вероятность
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels, alpha=cfg.MIXUP_ALPHA
                )
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + \
                       (1 - lam) * criterion(outputs, labels_b)
            else:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
            
            # Очистка памяти каждые 10 батчей
            if batch_idx % 10 == 0 and cfg.DEVICE == 'cuda':
                torch.cuda.empty_cache()
        
        # VALIDATION
        model.eval()
        val_losses = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(cfg.DEVICE)
                labels = labels.to(cfg.DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_losses.append(loss.item())
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = correct / total
        
        print(f'Epoch {epoch:2d}: '
              f'train_loss={np.mean(train_losses):.4f}, '
              f'val_loss={np.mean(val_losses):.4f}, '
              f'val_acc={val_acc:.4f}')
        
        # Сохраняем лучшую модель
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
        
        scheduler.step()
        
        # Очистка памяти после эпохи
        if cfg.DEVICE == 'cuda':
            torch.cuda.empty_cache()
    
    # Загружаем лучшие веса
    model.load_state_dict(best_weights)
    print(f'Лучшая точность: {best_acc:.4f}')
    
    return model


print('Функция обучения готова')


# K-Fold обучение
import gc

recording_ids = train_tp['recording_id'].values
labels = train_tp['species_id'].values

kfold = KFold(
    n_splits=cfg.NUM_FOLDS, 
    shuffle=True, 
    random_state=cfg.RANDOM_SEED
)

trained_models = []

for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(recording_ids)):
    print(f'\n{"="*60}')
    print(f'Fold {fold_idx}/{cfg.NUM_FOLDS}')
    print(f'{"="*60}')
    
    # Очистка памяти перед каждым фолдом
    if cfg.DEVICE == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    # Разделение данных
    train_ids = recording_ids[train_indices]
    train_labels = labels[train_indices]
    val_ids = recording_ids[val_indices]
    val_labels = labels[val_indices]
    
    # Datasets
    train_dataset = RFCXDataset(
        train_ids, train_labels, аудио_кэш, 
        is_train=True, use_mixup=True
    )
    val_dataset = RFCXDataset(
        val_ids, val_labels, аудио_кэш, 
        is_train=False, use_mixup=False
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        drop_last=True,
        num_workers=2,  # Параллельная загрузка
        pin_memory=True if cfg.DEVICE == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False, 
        drop_last=False,
        num_workers=2,
        pin_memory=True if cfg.DEVICE == 'cuda' else False
    )
    
    # Создание и обучение модели
    model = создать_модель()
    model = обучить_модель(model, train_loader, val_loader, fold_idx)
    
    # Сохранение
    torch.save(model.state_dict(), f'model_fold_{fold_idx}.pt')
    trained_models.append(model)
    
    # Очистка памяти
    del train_dataset, val_dataset, train_loader, val_loader
    if cfg.DEVICE == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

print('\n' + '='*60)
print('Обучение всех фолдов завершено')
print('='*60)


# Функции для инференса

def load_test_file(file_path):
    wav, sr = librosa.load(file_path, sr=None)
    
    # Количество сегментов
    num_segments = int(np.ceil(len(wav) / cfg.SEGMENT_SAMPLES))
    
    segments = []
    for i in range(num_segments):
        start = i * cfg.SEGMENT_SAMPLES
        end = start + cfg.SEGMENT_SAMPLES
        
        if end > len(wav):
            # Последний сегмент - берем с конца
            segment = wav[-cfg.SEGMENT_SAMPLES:]
        else:
            segment = wav[start:end]
        
        # Преобразование в спектрограмму
        mel_spec = audio_to_spectr(
            segment, sr, f_min_global, f_max_global
        )
        img = normalize_spectr(mel_spec)
        
        # 3 канала
        img_rgb = np.stack([img, img, img], axis=0)
        segments.append(img_rgb)
    
    return np.array(segments)


def predict_file(file_name, models):
    file_path = f'/kaggle/input/rfcx-species-audio-detection/test/{file_name}'
    
    # Загрузка сегментов
    segments = load_test_file(file_path)
    segments_tensor = torch.FloatTensor(segments).to(cfg.DEVICE)
    
    # Предсказания от каждой модели
    ensemble_preds = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model(segments_tensor)
            # Max агрегация по сегментам (как в оригинале)
            max_pred = outputs.max(dim=0)[0]
            ensemble_preds.append(max_pred.cpu())
    
    # Усреднение по моделям ансамбля
    final_pred = torch.stack(ensemble_preds).mean(dim=0)
    
    # Формирование результата
    file_id = file_name.split('.')[0]
    result = [file_id] + final_pred.numpy().tolist()
    
    return result


print('Функции инференса готовы')


# Генерация submission

print('Генерация предсказаний...\n')

test_files = os.listdir('/kaggle/input/rfcx-species-audio-detection/test/')
print(f'Тестовых файлов: {len(test_files)}')

if cfg.DEVICE == 'cuda':
    trained_models = [m.cuda() for m in trained_models]

# Предсказания
predictions = []
for test_file in tqdm(test_files):
    pred = predict_file(test_file, trained_models)
    predictions.append(pred)

# Сохранение submission
with open('submission.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    
    # Header
    header = ['recording_id'] + [f's{i}' for i in range(cfg.NUM_CLASSES)]
    writer.writerow(header)
    
    # Predictions
    for pred in predictions:
        writer.writerow(pred)

print('\n' + '='*60)
print('Все процессы завершены')
print('='*60)
print('\nSubmission сохранен: submission.csv')

