import pandas as pd
import numpy as np
import os
import cv2
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import models
from IPython.display import Audio, display


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


#Загрузка датасета
train_path = '../input/freesound-audio-tagging/audio_train/'
train_df = pd.read_csv("../input/freesound-audio-tagging/train.csv")
train_df.head()


# Выводим пример звукового файла
fname = train_df.iloc[25]['fname']
label = train_df.iloc[25]['label']
print(f"Метка: {label}")

audio_path = os.path.join(train_path, fname)
Audio(filename=audio_path)


class AudioDataset(Dataset):
    def __init__(self, df, test=False):
        self.df = df
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = self.df.iloc[idx]['fname']
        path = (test_path if self.test else train_path) + fname
        
        signal, _ = librosa.load(path, sr=22050, duration=4.0)
        if len(signal) < 22050 * 4:
            signal = np.pad(signal, (0, 22050 * 4 - len(signal)), mode='constant')
        
        mel = librosa.feature.melspectrogram(y=signal, sr=22050, n_mels=128, fmax=11025)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        try:
            mel_resized = cv2.resize(mel_db, IMG_SIZE[::-1])
        except:
            mel_resized = np.zeros(IMG_SIZE)
        
        X = np.stack([mel_resized] * 3, axis=0)

        if self.test:
            return torch.tensor(X, dtype=torch.float32)
        else:
            label = self.df.iloc[idx]['label']
            y = label_to_idx[label]
            return torch.tensor(X, dtype=torch.float32), y


print(f"Всего обучающих примеров: {len(train_df)}")


# Оставим только вручную проверенные (более надёжные)
clean_df = train_df[train_df['manually_verified'] == 1].reset_index(drop=True)
print(f"Проверенные примеры: {len(clean_df)}")


# Разделение
train_data, val_data = train_test_split(
    clean_df, test_size=0.2, stratify=clean_df['label'], random_state=42
)


print(f"Размер тренировочного датасета: {len(train_data)}")
print(f"Размер валидационного датасета: {len(val_data)}")


# Кодирование меток
unique_labels = sorted(clean_df['label'].unique())
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
num_classes = len(unique_labels)
print(f"Число классов: {num_classes}")
idx_to_label


IMG_SIZE = (128, 128)
batch_size = 32


train_loader = DataLoader(AudioDataset(train_data), batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(AudioDataset(val_data), batch_size=batch_size, shuffle=False, num_workers=0)


# Модель
from torchvision.models import EfficientNet_B0_Weights
weights = EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)


# Обучение
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)



# МЕТРИКА MAP@3
def map_at_3(y_true, y_pred_probs):
    """
    y_true: массив истинных меток (целые числа)
    y_pred_probs: матрица вероятностей (N x num_classes)
    Возвращает MAP@3
    """
    top3_preds = np.argsort(y_pred_probs, axis=1)[:, ::-1][:, :3]  # (N, 3)
    scores = []
    for i, true_label in enumerate(y_true):
        if true_label in top3_preds[i]:
            rank = np.where(top3_preds[i] == true_label)[0][0] + 1  # 1-based
            scores.append(1.0 / rank)
        else:
            scores.append(0.0)
    return np.mean(scores)


best_map3 = 0.0
for epoch in range(15):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    
    # Валидация по MAP@3
    model.eval()
    all_probs = []
    all_true = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_true.append(y.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_true = np.concatenate(all_true)
    val_map3 = map_at_3(all_true, all_probs)
    print(f"Epoch {epoch+1}: Val MAP@3 = {val_map3:.4f}")
    
    if val_map3 > best_map3:
        best_map3 = val_map3
        torch.save(model.state_dict(), "/kaggle/working/best_model.pth")
    scheduler.step(val_map3)

print(f"Лучший валидационный MAP@3: {best_map3:.4f}")


test_path = '../input/freesound-audio-tagging/audio_test/'
test_df = pd.read_csv('../input/freesound-audio-tagging/sample_submission.csv')
test_loader = DataLoader(AudioDataset(test_df, test=True), batch_size=batch_size, shuffle=False, num_workers=0)


print(f"Размер  датасета: {len(test_df)}")


# Предсказание на тесте (топ 3 метки)
model.load_state_dict(torch.load("/kaggle/working/best_model.pth"))
model.eval()

all_probs = []
with torch.no_grad():
    for x in test_loader:
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)

all_probs = np.vstack(all_probs)

# Формируем решение
submission = []
for i, fname in enumerate(test_df['fname']):
    top3_idx = np.argsort(all_probs[i])[::-1][:3]
    top3_labels = [idx_to_label[idx] for idx in top3_idx]
    submission.append({'fname': fname, 'label': ' '.join(top3_labels)})

# Преобразуем в DataFrame
submission_df = pd.DataFrame(submission)


# Пример 5 предсказаний
print("Примеры предсказаний на тестовых данных:\n")

num_examples = 5
for i in range(num_examples):
    fname = submission_df.iloc[i]['fname']
    pred_labels = submission_df.iloc[i]['label']
    
    print(f"Файл: {fname}")
    print(f"Предсказанные метки (top-3): {pred_labels}")
    
    # Путь к аудиофайлу
    audio_path = os.path.join(test_path, fname)
    
    # Отображаем аудиоплеер
    display(Audio(filename=audio_path))
    print("-" * 60)


# Сохраняем решение
pd.DataFrame(submission).to_csv("submission.csv", index=False)


# Проверка формата
check = pd.read_csv("submission.csv")
print(check.head(10))
print("\nРазмер:", check.shape)
print("Колонки:", check.columns.tolist())

