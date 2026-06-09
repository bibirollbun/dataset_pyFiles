# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.filterwarnings('ignore')


import os
from pathlib import Path
import random
import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm 
import librosa
import soundfile as sf


# Конфиг с оптимизацией
DATA_ROOT = Path("/kaggle/input/birdclef-2021")
TRAIN_AUDIO_DIR = DATA_ROOT / "train_short_audio"            
TRAIN_META = DATA_ROOT / "train_metadata.csv"


SR = 32000              # целевая дискретизация
DURATION = 5            # длительность для большего контекста
AUDIO_LEN = SR * DURATION
N_MELS = 224            
FMIN = 20
FMAX = 16000            
BATCH = 128 if torch.cuda.is_available() else 64  # батч
NUM_WORKERS = 2  # воркеры
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


# Воспроизводимость
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything()
print("Device:", DEVICE)


# Упрощенные утилиты для аудио
def load_audio(path, sr=SR, duration=DURATION):
    try:        
        audio, orig_sr = librosa.load(path, sr=sr, duration=duration, mono=True)
        return audio.astype(np.float32)
    except Exception as e:
        print(f"Error loading {path}: {e}")        
        return np.zeros(AUDIO_LEN, dtype=np.float32)

def random_crop_or_pad(x, length=AUDIO_LEN):    
    if len(x) >= length:
        start = np.random.randint(0, len(x) - length + 1)
        return x[start:start+length]
    else:
        # Простой паддинг в конец
        return np.pad(x, (0, length - len(x)), mode='constant')

def compute_mel_spec(y, sr=32000, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    try:
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=n_mels,
            n_fft=2048,
            hop_length=512,
            fmin=fmin,
            fmax=fmax,
            power=2.0
        )
        
        mel_db = librosa.power_to_db(mel, ref=1.0)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        return mel_db.astype(np.float32)
    except Exception as e:        
        return np.zeros((n_mels, 313), dtype=np.float32)  


# Аугментации
def audio_augmentations(audio, apply_prob=0.5):
    augmented = audio.copy()
    
    # Добавление шума
    if np.random.rand() < apply_prob:
        noise = np.random.normal(0, 0.005, audio.shape).astype(np.float32)
        augmented += noise
    
    if np.random.rand() < apply_prob:
        shift = int(len(audio) * 0.2 * np.random.uniform(-1, 1))
        if abs(shift) > 0:
            augmented = np.roll(augmented, shift)
    
    return augmented


def spec_augment_simple(mel):
    mel = mel.copy()
    T, F = mel.shape[1], mel.shape[0]
    
    # Маскировка времени
    if T > 10:
        t_mask_width = min(20, T // 4)
        t0 = np.random.randint(0, T - t_mask_width)
        mel[:, t0:t0+t_mask_width] = mel.min()
    
    # Частотная маскировка
    if F > 5:
        f_mask_width = min(5, F // 8)
        f0 = np.random.randint(0, F - f_mask_width)
        mel[f0:f0+f_mask_width, :] = mel.min()
    
    return mel


# Подготовка данных
meta = pd.read_csv(TRAIN_META)
labels = sorted(meta['primary_label'].unique().tolist())
LABEL2ID = {l:i for i,l in enumerate(labels)}
ID2LABEL = {i:l for l,i in LABEL2ID.items()}
NUM_CLASSES = len(labels)
print("Found classes:", NUM_CLASSES)


# Формирование DataFrame
if TRAIN_AUDIO_DIR.exists():
    audio_files = []
    for bird_dir in TRAIN_AUDIO_DIR.iterdir():
        if bird_dir.is_dir():
            for audio_file in bird_dir.glob("*.ogg"):
                audio_files.append({
                    "filename": audio_file.name,
                    "primary_label": bird_dir.name,
                    "filepath": str(audio_file)
                })
    df_short = pd.DataFrame(audio_files)
else:
    raise FileNotFoundError(f"Папка {TRAIN_AUDIO_DIR} не найдена!")

print("Всего файлов:", len(df_short))


# # Фильтрация файлов
# def filter_valid_files_simple(df, max_files_per_class=500):
#     valid_files = []
#     class_counts = {}
    
#     for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering files"):
#         label = row["primary_label"]
                
#         if label not in class_counts:
#             class_counts[label] = 0
        
#         if class_counts[label] < max_files_per_class:
#             try:
#                 # Быстрая проверка файла
#                 audio = load_audio_fast(row["filepath"], duration=1.0)
#                 if len(audio) > 1000:  
#                     valid_files.append(row)
#                     class_counts[label] += 1
#             except:
#                 continue
    
#     filtered_df = pd.DataFrame(valid_files)
#     print(f"Filtered to {len(filtered_df)} files across {len(class_counts)} classes")
#     return filtered_df


class SimpleBirdDataset(Dataset):
    def __init__(self, df, label2id, train=True):
        self.df = df.reset_index(drop=True).copy()
        self.label2id = label2id
        self.train = train
        # если в df есть метка primary_label
        self.labels = [self.label2id[row] if row in self.label2id else 0
                       for row in self.df['primary_label'].values]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        label_idx = self.labels[idx]

        # Загружаем аудио обычной функцией
        audio = load_audio(filepath, sr=SR, duration=DURATION)  # гарантированно определена

        # Crop / pad
        audio = random_crop_or_pad(audio, length=AUDIO_LEN)

        # Аугментации аудио (временные/шум)
        if self.train and np.random.rand() < 0.7:
            audio = audio_augmentations(audio, apply_prob=0.5)

        # Мел-спектр
        mel = compute_mel_spec(audio, sr=SR, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)

        # Спец-аугментация на спектрограмме
        if self.train and np.random.rand() < 0.5:
            mel = spec_augment_simple(mel)

        # Нормализация
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # Преобразуем в тензор (1, n_mels, T)
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(label_idx, dtype=torch.long)
        return mel_tensor, label


import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import librosa
from sklearn.model_selection import train_test_split

def create_bird_dataframe_fast(audio_dir, max_files_per_class=150):
    audio_dir = Path(audio_dir)
    records = []
    bird_dirs = sorted([d for d in audio_dir.iterdir() if d.is_dir()])

    for bird_dir in tqdm(bird_dirs, desc="Scanning bird dirs"):
        files = list(bird_dir.glob("*.ogg"))
        if not files:
            continue
        random.shuffle(files)
        files = files[:max_files_per_class]
        for f in files:
            records.append({
                "filename": f.name,
                "primary_label": bird_dir.name,
                "filepath": str(f)
            })
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} files from {df['primary_label'].nunique()} classes")
    return df


TRAIN_AUDIO_DIR = Path("/kaggle/input/birdclef-2021/train_short_audio")
df_short_clean = create_bird_dataframe_fast(TRAIN_AUDIO_DIR, max_files_per_class=150)


# Проверяем результат
if not df_short_clean.empty:
    print("DataFrame created successfully!")
    print(f"Shape: {df_short_clean.shape}")
    print(f"Columns: {df_short_clean.columns.tolist()}")
    print("\nFirst 3 rows:")
    print(df_short_clean.head(3))
    
    # Разделение данных - ВАЖНО: правильные отступы!
    df_train, df_val = train_test_split(
        df_short_clean, 
        test_size=0.15, 
        stratify=df_short_clean['primary_label'], 
        random_state=SEED
    )
    print(f"Train size: {len(df_train)}, Val size: {len(df_val)}")
    
    # Проверяем распределение
    print(f"Train classes: {df_train['primary_label'].nunique()}")
    print(f"Val classes: {df_val['primary_label'].nunique()}")
    
else:
    print("Failed to create DataFrame!")


# маленькая проверка
df_test = df_short_clean.sample(n=min(10, len(df_short_clean)), random_state=SEED)
ds_test = SimpleBirdDataset(df_test, LABEL2ID, train=False)

# Попытка получить пару (mel, label)
x, y = ds_test[0]
print("x.shape:", x.shape, "y:", y)


# Создаем датасеты и даталоадеры
print("Creating datasets and dataloaders...")
train_ds = SimpleBirdDataset(df_train, LABEL2ID, train=True)
val_ds = SimpleBirdDataset(df_val, LABEL2ID, train=False)

train_dl = DataLoader(
    train_ds, 
    batch_size=BATCH, 
    shuffle=True, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_dl = DataLoader(
    val_ds, 
    batch_size=BATCH, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}")


# Создаем модель
class BirdClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            in_chans=1,
            num_classes=num_classes,
            drop_rate=0.2
        )
    
    def forward(self, x):
        return self.backbone(x)

model = BirdClassifier(NUM_CLASSES)
model.to(DEVICE)


# Оптимизатор и функция потерь
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


# Функции обучения
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        correct += (preds.argmax(1) == y).sum().item()
        total += x.size(0)
    
    return total_loss / total, correct / total

def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            loss = criterion(preds, y)
            
            total_loss += loss.item() * x.size(0)
            correct += (preds.argmax(1) == y).sum().item()
            total += x.size(0)
    
    return total_loss / total, correct / total


from torch.cuda.amp import autocast, GradScaler
import time

EPOCHS = 8
scaler = GradScaler()
best_acc = 0.0
best_epoch = 0

print("\n Начало обучения...\n")
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    
    # === Обучение ===
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS} [train]", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        
        with autocast():  # ускорение и экономия памяти
            preds = model(x)
            loss = criterion(preds, y)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        correct += (preds.argmax(1) == y).sum().item()
        total += x.size(0)

    train_loss = total_loss / total
    train_acc = correct / total

    # === Валидация ===
    model.eval()
    val_loss, val_acc = 0, 0
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in tqdm(val_dl, desc=f"Epoch {epoch}/{EPOCHS} [val]", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            loss = criterion(preds, y)
            val_loss += loss.item() * x.size(0)
            correct += (preds.argmax(1) == y).sum().item()
            total += x.size(0)
    val_loss /= total
    val_acc = correct / total

    # === Сохранение лучшей модели ===
    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), "best_bird_model.pth")

    t1 = time.time()
    print(f"[{epoch:02d}/{EPOCHS}] "
          f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
          f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f} | "
          f"time={t1 - t0:.1f}s")

print(f"\n Обучение завершено! Лучшая эпоха: {best_epoch}, точность = {best_acc:.3f}")

