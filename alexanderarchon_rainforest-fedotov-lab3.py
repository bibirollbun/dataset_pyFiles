!pip install timm


import os
import pandas as pd
import librosa
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from skimage.transform import resize
from PIL import Image

import torch
import torch.utils.data as torchdata
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import timm
import warnings
warnings.filterwarnings('ignore')


# Константы
TRAIN_TP = '/kaggle/input/rfcx-species-audio-detection/train_tp.csv'
AUDIO_DATA = '/kaggle/input/rfcx-species-audio-detection/train/'
WORKING_DIR = '/kaggle/working/'
fft = 2048
hop = 512
sr = 48000
length = 10 * sr
save_to_disk = False
num_birds = 24
batch_size = 16


# Установка seed для воспроизводимости
rng_seed = 1234
random.seed(rng_seed)
np.random.seed(rng_seed)
os.environ['PYTHONHASHSEED'] = str(rng_seed)
torch.manual_seed(rng_seed)
torch.cuda.manual_seed(rng_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Загрузка данных
df = pd.read_csv(TRAIN_TP)
fmin = int(df['f_min'].min() * 0.9)
fmax = int(df['f_max'].max() * 1.1)


# Функция для визуализации спектрограмм
def visualize_spectrograms(df, num_samples=2):
    """Визуализация случайных спектрограмм"""
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i in range(num_samples):
        # Выбираем случайную запись
        idx = np.random.randint(len(df))
        row = df.iloc[idx]
        
        # Загружаем аудио
        wav, sr = librosa.load(f"{AUDIO_DATA}{row['recording_id']}.flac", sr=None)
        
        # Вырезаем сегмент
        t_min = float(row['t_min']) * sr
        t_max = float(row['t_max']) * sr
        center = np.round((t_min + t_max) / 2)
        beginning = center - length / 2
        if beginning < 0:
            beginning = 0
        
        ending = beginning + length
        if ending > len(wav):
            ending = len(wav)
            beginning = ending - length
            
        slice = wav[int(beginning):int(ending)]
        
        # Создаем спектрограмму
        mel_spec = librosa.feature.melspectrogram(
            y=slice, n_fft=fft, hop_length=hop, 
            sr=sr, fmin=fmin, fmax=fmax, power=1.5
        )
        
        # Нормализуем
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Визуализация
        plt.subplot(num_samples, 2, i*2 + 1)
        librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop, 
                                fmin=fmin, fmax=fmax, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram {i+1}: Species {row["species_id"]}')
        
        plt.subplot(num_samples, 2, i*2 + 2)
        plt.imshow(mel_spec_db, aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(f'Raw Mel Spectrogram {i+1}')
        plt.xlabel('Time frames')
        plt.ylabel('Mel bins')
    
    plt.tight_layout()
    plt.savefig(f'{WORKING_DIR}/sample_spectrograms.png', dpi=100)
    plt.show()

# Выводим примеры спектрограмм
print("Визуализация примеров спектрограмм...")
visualize_spectrograms(df, num_samples=2)


# Создание спектрограмм для обучения
print("\nСоздание спектрограмм для обучения...")
for idx, row in tqdm(df.iterrows(), total=min(100, len(df)), desc='Создание спектрограмм'):
    wav, sr = librosa.load(f"{AUDIO_DATA}{row['recording_id']}.flac", sr=None)
    
    t_min = float(row['t_min']) * sr
    t_max = float(row['t_max']) * sr
    
    center = np.round((t_min + t_max) / 2)
    beginning = center - length / 2
    if beginning < 0:
        beginning = 0
    
    ending = beginning + length
    if ending > len(wav):
        ending = len(wav)
        beginning = ending - length
        
    slice = wav[int(beginning):int(ending)]
    
    mel_spec = librosa.feature.melspectrogram(
        y=slice, n_fft=fft, hop_length=hop, 
        sr=sr, fmin=fmin, fmax=fmax, power=1.5
    )
    mel_spec = resize(mel_spec, (224, 400))
    
    # Нормализация
    mel_spec = mel_spec - np.min(mel_spec)
    mel_spec = mel_spec / (np.max(mel_spec) + 1e-8)
    mel_spec = mel_spec * 255
    mel_spec = np.round(mel_spec)    
    mel_spec = mel_spec.astype('uint8')
    
    # Сохранение
    bmp = Image.fromarray(mel_spec, 'L')
    bmp.save(f"{WORKING_DIR}{row['recording_id']}_{row['species_id']}_{int(center)}.bmp")


# Dataset
class RainforestDataset(torchdata.Dataset):
    def __init__(self, filelist, augment=False):
        self.specs = []
        self.labels = []
        self.augment = augment
        
        for f in tqdm(filelist, desc='Загрузка данных'):
            label = int(str.split(f, '_')[1])
            label_array = np.zeros(num_birds, dtype=np.float32)
            label_array[label] = 1.0
            self.labels.append(label_array)

            img = Image.open(WORKING_DIR + f)
            mel_spec = np.array(img, dtype=np.float32)
            img.close()

            # Нормализация
            mel_spec = mel_spec / 255.0
            
            # Аугментация
            if self.augment:
                mel_spec = self._augment_spectrogram(mel_spec)
            
            # Конвертация в 3 канала
            mel_spec = np.stack((mel_spec, mel_spec, mel_spec))
            
            self.specs.append(mel_spec)
    
    def _augment_spectrogram(self, spec):
        """Простая аугментация спектрограммы"""
        if random.random() > 0.5:
            # Случайное отражение по времени
            spec = np.fliplr(spec)
        
        if random.random() > 0.5:
            # Добавление шума
            noise = np.random.normal(0, 0.02, spec.shape)
            spec = spec + noise
            spec = np.clip(spec, 0, 1)
        
        return spec
    
    def __len__(self):
        return len(self.specs)
    
    def __getitem__(self, item):
        spec = torch.tensor(self.specs[item], dtype=torch.float32)
        label = torch.tensor(self.labels[item], dtype=torch.float32)
        return spec, label


# Подготовка данных
print("\nПодготовка данных...")
file_list = []
label_list = []

for f in os.listdir(WORKING_DIR):
    if '.bmp' in f:
        file_list.append(f)
        label = str.split(f, '_')[1]
        label_list.append(label)


# Стратифицированное разделение
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng_seed)
train_files = []
val_files = []

for fold_id, (train_index, val_index) in enumerate(skf.split(file_list, label_list)):
    if fold_id == 0:
        train_files = np.take(file_list, train_index)
        val_files = np.take(file_list, val_index)

print(f"Train samples: {len(train_files)}, Val samples: {len(val_files)}")



# Создание датасетов и загрузчиков
train_dataset = RainforestDataset(train_files, augment=True)
val_dataset = RainforestDataset(val_files, augment=False)

train_loader = torchdata.DataLoader(
    train_dataset, batch_size=batch_size, 
    sampler=torchdata.RandomSampler(train_dataset),
    num_workers=2
)
val_loader = torchdata.DataLoader(
    val_dataset, batch_size=batch_size, 
    sampler=torchdata.SequentialSampler(val_dataset),
    num_workers=2
)


# Модель
class EnhancedBirdClassifier(nn.Module):
    def __init__(self, base_model_name='tf_efficientnet_b4_ns', num_classes=24):
        super(EnhancedBirdClassifier, self).__init__()
        
        # Базовый претренированный энкодер
        self.base_model = timm.create_model(
            base_model_name, 
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        
        # Размер фичей
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 400)
            features = self.base_model(dummy)
            feature_size = features.shape[1]
            spatial_size = features.shape[2] * features.shape[3]
        
        # Attention механизм
        self.attention = nn.Sequential(
            nn.Conv2d(feature_size, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Классификатор с BatchNorm и Dropout
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            
            nn.Linear(512, num_classes)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        # Extract features
        features = self.base_model(x)
        
        # Attention weights
        attention_weights = self.attention(features)
        
        # Apply attention
        attended_features = features * attention_weights
        
        # Global pooling
        pooled = self.global_pool(attended_features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output, attention_weights

# Инициализация модели
print("\nИнициализация модели...")
model = EnhancedBirdClassifier(base_model_name='tf_efficientnet_b4_ns', num_classes=num_birds)


# Оптимизатор и планировщик
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=0.001, 
    weight_decay=0.01
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10, 
    T_mult=1, 
    eta_min=1e-6
)


# Функция потерь с учетом дисбаланса классов
class_counts = df['species_id'].value_counts().sort_index().values
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * num_birds
pos_weights = torch.tensor(class_weights, dtype=torch.float32)

loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weights)


# Перемещение на GPU если доступно
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = model.to(device)
loss_function = loss_function.to(device)


# Функция для расчета точности
def calculate_accuracy(outputs, targets):
    """Вычисляет точность предсказаний"""
    with torch.no_grad():
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()
        correct = (preds == targets).float().sum()
        accuracy = correct / targets.numel()
    return accuracy.item()


# Обучение модели
print("\nНачало обучения...")
best_val_accuracy = 0
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in tqdm(range(30), desc='Эпохи'):
    # Обучение
    model.train()
    epoch_train_loss = 0
    epoch_train_acc = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        output, _ = model(data)
        loss = loss_function(output, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_train_loss += loss.item()
        epoch_train_acc += calculate_accuracy(output, target)
    
    train_losses.append(epoch_train_loss / len(train_loader))
    train_accuracies.append(epoch_train_acc / len(train_loader))
    
    # Валидация
    model.eval()
    epoch_val_loss = 0
    epoch_val_acc = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            output, _ = model(data)
            loss = loss_function(output, target)
            
            epoch_val_loss += loss.item()
            epoch_val_acc += calculate_accuracy(output, target)
    
    val_losses.append(epoch_val_loss / len(val_loader))
    val_accuracies.append(epoch_val_acc / len(val_loader))
    
    # Обновление планировщика
    scheduler.step()
    
    # Сохранение лучшей модели
    current_val_acc = val_accuracies[-1]
    if current_val_acc > best_val_accuracy:
        best_val_accuracy = current_val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': best_val_accuracy,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, f'{WORKING_DIR}/best_model.pth')
    
    # Вывод статистики
    if (epoch + 1) % 5 == 0:
        print(f'\nEpoch {epoch + 1}:')
        print(f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}')
        print(f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')


# Загрузка лучшей модели для тестирования
print("\nЗагрузка лучшей модели для тестирования...")
checkpoint = torch.load(f'{WORKING_DIR}/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("\nПеревод модели в режим оценки...")


# Функция для обработки тестовых файлов
def load_test_file(f):
    wav, sr = librosa.load('/kaggle/input/rfcx-species-audio-detection/test/' + f, sr=None)

    segments = len(wav) / length
    segments = int(np.ceil(segments))
    
    mel_array = []
    
    for i in range(segments):
        if (i + 1) * length > len(wav):
            slice = wav[len(wav) - length:len(wav)]
        else:
            slice = wav[i * length:(i + 1) * length]

        mel_spec = librosa.feature.melspectrogram(
            y=slice, n_fft=fft, hop_length=hop, 
            sr=sr, fmin=fmin, fmax=fmax, power=1.5
        )
        mel_spec = resize(mel_spec, (224, 400))
    
        mel_spec = mel_spec - np.min(mel_spec)
        mel_spec = mel_spec / (np.max(mel_spec) + 1e-8)
        mel_spec = np.stack((mel_spec, mel_spec, mel_spec))

        mel_array.append(mel_spec)
    
    return mel_array


# Обработка тестовых данных
print("\nОбработка тестовых данных...")
results = []
test_files = os.listdir('/kaggle/input/rfcx-species-audio-detection/test/')

for file_name in tqdm(test_files, desc='Processing test files'):
    data = load_test_file(file_name)
    data = torch.tensor(data).float().to(device)
    
    with torch.no_grad():
        outputs = []
        for segment in data:
            segment = segment.unsqueeze(0)
            output, _ = model(segment)
            outputs.append(torch.sigmoid(output))
        
        aggregated = torch.stack(outputs).max(dim=0)[0]
        
    file_id = file_name.split('.')[0]
    row = [file_id] + aggregated.cpu().numpy().flatten().tolist()
    results.append(row)



# Сохранение результатов
columns = ['recording_id'] + [f's{i}' for i in range(num_birds)]
submission_df = pd.DataFrame(results, columns=columns)
submission_df.to_csv(f'{WORKING_DIR}/submission.csv', index=False)

print(f"\nСоздан файл submission.csv с {len(submission_df)} записями")
print("Готово!")

