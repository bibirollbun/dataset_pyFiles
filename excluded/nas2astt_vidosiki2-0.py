import os
import sys
import re
import gc
import platform
import random
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import einops

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm
import glob
import cv2

from numpy import array
from numpy import argmax

from tqdm import tqdm
from tqdm.contrib import tzip

import wandb

import warnings

warnings.simplefilter('ignore')


train_csv = pd.read_csv('/kaggle/input/what-on-the-video/train.csv')
print(train_csv.shape)
train_csv.head()


miss_csv = pd.read_csv('/kaggle/input/missing-videos/missing_videos.csv', sep=';')
miss_csv.columns = ['path', 'labels']
print(miss_csv.shape)
miss_csv.head()


import os
import pandas as pd
from collections import defaultdict

# Путь к корневой папке с видео
root_dir = "/kaggle/input/vidosiki0/videos"

# Словарь для хранения информации: {название_видео: [классы]}
video_classes = defaultdict(list)

# Проходим по всем подпапкам
for class_name in os.listdir(root_dir):
    class_dir = os.path.join(root_dir, class_name)
    
    # Проверяем, что это директория
    if os.path.isdir(class_dir):
        # Проходим по всем файлам в папке класса
        for video_file in os.listdir(class_dir):
            # Добавляем класс для этого видео
            video_classes[video_file].append(class_name)

# Создаем датафрейм
df = pd.DataFrame({
    'path': list(video_classes.keys()),
    'labels': [', '.join(classes) for classes in video_classes.values()]
})

print(df.shape)
df.head(10)



df_train = pd.concat([train_csv, miss_csv, df], axis=0, ignore_index=True)


df_train.shape


df_train['labels'].value_counts()


df_train['labels'] = df_train['labels'].str.replace(' ', '')   
df_train['labels'] = df_train['labels'].str.replace('.', ',')
 
df_train['labels'].value_counts()


df_train['labels'] = df_train['labels'].str.replace('animals', 'animal')   
df_train['labels'] = df_train['labels'].str.replace('flowers', 'flower')  
df_train['labels'] = df_train['labels'].str.replace('clouds', 'cloud')  
df_train['labels'].value_counts()


from sklearn.preprocessing import MultiLabelBinarizer

# Разбиваем строки с классами на списки
df_train['label_split'] = df_train['labels'].str.split(',')

# Создаем кодировщик
mlb = MultiLabelBinarizer()

# Применяем one-hot encoding
one_hot_encoded = mlb.fit_transform(df_train['label_split'])

# Создаем DataFrame с one-hot кодированием
one_hot_df = pd.DataFrame(one_hot_encoded, columns=mlb.classes_)

# Объединяем с исходными данными
result_df = pd.concat([df_train, one_hot_df], axis=1)

result_df.head(30)


result_df.drop('labels', axis=1, inplace=True)
result_df.drop('label_split', axis=1, inplace=True)
result_df.head(20)


result_df.shape


result_df.to_csv('/kaggle/working/video_labels.csv', index=False)


import cv2
import numpy as np
from PIL import Image
def read_video(path, img_size, transform=None, frames_num=11):

    frames = []
    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return frames
    
    # Вычисляем шаг между кадрами (не менее 1)
    step = max(1, total_frames // frames_num)
    
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if ret and len(frames) < frames_num:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, img_size)
            
            if transform:
                frame = Image.fromarray(frame)
                frame = transform(frame)
                
            frames.append(frame)
    
    cap.release()
    return frames[:frames_num]


# Извлечь 11 кадров размером 224x224
frames = read_video("/kaggle/input/what-on-the-video/train/132080839-paris-tree-lined-avenue_preview.mp4", img_size=(224, 224))

# # С дополнительными преобразованиями (например, albumentations)
# import albumentations as A
# transform = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.2),
# ])

# frames = read_video("/kaggle/input/what-on-the-video/train/160929_120_London_TowerBridge4_1080p_preview.mp4", img_size=(224, 224), transform=transform)


import matplotlib.pyplot as plt

def show_frames_grid(frames, cols=4, figsize=(15, 10)):
    """
    Показывает кадры в виде сетки
    
    Параметры:
        frames: список кадров
        cols: количество столбцов в сетке
        figsize: размер фигуры
    """
    rows = (len(frames) + cols - 1) // cols
    plt.figure(figsize=figsize)
    
    for i, frame in enumerate(frames):
        plt.subplot(rows, cols, i+1)
        plt.imshow(frame)
        plt.axis('off')
        plt.title(f'Кадр {i+1}')
    
    plt.tight_layout()
    plt.show()

show_frames_grid(frames)





import os
import shutil

# Путь к исходным данным и списку категорий
source_base = "/kaggle/input/vidosiki0/videos"
categories = ["animals", "car", "cloud", "dance", "fire", "flowers", "food", "sunset", "water"]

# Путь к целевой папке
target_dir = "/kaggle/working/videos"
os.makedirs(target_dir, exist_ok=True)

# Множество для отслеживания уже скопированных файлов
copied_files = set()

for category in categories:
    category_path = os.path.join(source_base, category)
    for filename in os.listdir(category_path):
        if filename in copied_files:
            continue  # Пропустить дубликаты

        src_path = os.path.join(category_path, filename)
        dst_path = os.path.join(target_dir, filename)

        # Убедиться, что это файл
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            copied_files.add(filename)

print(f"Скопировано {len(copied_files)} уникальных видео в {target_dir}")



import os
import shutil

source_dir = "/kaggle/input/what-on-the-video/train"
target_dir = "/kaggle/working/videos"

os.makedirs(target_dir, exist_ok=True)

# Список уже существующих файлов в целевой папке
existing_files = set(os.listdir(target_dir))

# Копируем файлы, если их ещё нет
for filename in os.listdir(source_dir):
    if filename not in existing_files:
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(target_dir, filename)

        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)

print(f"Видео из {source_dir} скопированы в {target_dir}")



path_to_directory = '/kaggle/working/videos'
video_name = '_import_62ca6256159466.11484720_preview.mp4'
path_to_video = os.path.join(path_to_directory, video_name)
print(path_to_video)


frames = read_video(path_to_video, img_size=(224, 224))
show_frames_grid(frames)





pip install av


import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.io as io

# === Настройки ===
VIDEO_DIR = "/kaggle/working/videos"
CSV_PATH = "/kaggle/working/video_labels.csv"  # путь к CSV с названиями и метками
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 9  # число категорий

# === 1. DataFrame с метками ===
df = pd.read_csv(CSV_PATH)
df['video_path'] = df['path'].apply(lambda x: os.path.join(VIDEO_DIR, x))

# === 2. Предобработка видео ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # уменьшаем размер кадров
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# === 3. Кастомный Dataset ===
from torch.utils.data import Dataset
import torch
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, dataframe, img_size=(128, 128), transform=None, frames_num=11):
        self.df = dataframe
        self.img_size = img_size
        self.transform = transform
        self.frames_num = frames_num

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['video_path']
    
        frames = read_video(path, img_size=self.img_size, transform=self.transform, frames_num=self.frames_num)
    
        if len(frames) < self.frames_num:
            # Если кадров мало, докинь нулевые тензоры
            n_missing = self.frames_num - len(frames)
            zero_frame = torch.zeros((3, self.img_size[1], self.img_size[0]), dtype=torch.float32)
            frames += [zero_frame] * n_missing
    
        frames = torch.stack(frames)  # [T, C, H, W]
    
        labels = torch.tensor(row.iloc[1:10].values.astype('float32'))
    
        return frames, labels


# === 4. Модель ===
class VideoClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):  # [B, T, C, H, W] -> [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv3d(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))  # сигмоида для multilabel

# === 5. Тренировка ===
def train_model(model, dataloader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for videos, labels in dataloader:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            outputs = model(videos)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# === Запуск ===
dataset = VideoDataset(df, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
videos, labels = videos.to(DEVICE), labels.to(DEVICE)

model = VideoClassifier(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

train_model(model, dataloader, optimizer, criterion, epochs=5)


import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random

# === Настройки ===
VIDEO_DIR = "/kaggle/working/videos"
CSV_PATH = "/kaggle/working/video_labels.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 9
SEED = 42

# === Фиксация seed для воспроизводимости ===
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# === DataFrame ===
df = pd.read_csv(CSV_PATH)
df['video_path'] = df['path'].apply(lambda x: os.path.join(VIDEO_DIR, x))

# === Трансформации ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# === Dataset ===
class VideoDataset(Dataset):
    def __init__(self, dataframe, img_size=(128, 128), transform=None, frames_num=11):
        self.df = dataframe
        self.img_size = img_size
        self.transform = transform
        self.frames_num = frames_num

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['video_path']
    
        frames = read_video(path, img_size=self.img_size, transform=self.transform, frames_num=self.frames_num)
    
        if len(frames) < self.frames_num:
            n_missing = self.frames_num - len(frames)
            zero_frame = torch.zeros((3, self.img_size[1], self.img_size[0]), dtype=torch.float32)
            frames += [zero_frame] * n_missing
    
        frames = torch.stack(frames)  # [T, C, H, W]
    
        labels = torch.tensor(row.iloc[1:10].values.astype('float32'))
    
        return frames, labels

# === Модель ===
class VideoClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):  # [B, T, C, H, W] -> [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv3d(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

# === Тренировка ===
def train_model(model, dataloader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for videos, labels in dataloader:
            videos, labels = videos.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            outputs = model(videos)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# === Запуск ===
dataset = VideoDataset(df, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

model = VideoClassifier(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

train_model(model, dataloader, optimizer, criterion, epochs=5)

