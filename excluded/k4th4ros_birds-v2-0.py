import shutil
import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

print("Установка ResNeSt...")
try:
    shutil.copytree('../input/resnest50-fast-package/resnest-0.0.6b20200701/resnest', 
                    'resnet', dirs_exist_ok=True)
    os.system('pip install "./resnet" --no-deps')
    print("ResNeSt установлен")
except FileNotFoundError:
    print("Путь к resnest не найден, пробуем альтернативный...")
    os.system('pip install ../input/resnest50-fast-package/resnest-0.0.6b20200701 --no-deps')

from resnest.torch import resnest50

SR = 32000
DURATION = 5
THRESHOLD = 0.25
BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "../input/birdclef-2021"
TEST_AUDIO = os.path.join(DATA_ROOT, "test_soundscapes")
WEIGHTS = "../input/kkiller-birdclef-models-public/birdclef_resnest50_fold0_epoch_10_f1_val_06471_20210417161101.pth"

print(f"\nИнициализация завершена")
print(f"Device: {DEVICE}")
print(f"Threshold: {THRESHOLD}")


train_meta = pd.read_csv(os.path.join(DATA_ROOT, "train_metadata.csv"))
species = sorted(train_meta["primary_label"].unique())
encoder = LabelEncoder().fit(species)
NUM_CLASSES = len(species)

print(f"Видов: {NUM_CLASSES}")


def audio_to_melspec(audio, sr=SR):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=128, fmin=0, fmax=sr//2,
        n_fft=sr//10, hop_length=sr//40
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    mean, std = log_mel.mean(), log_mel.std()
    normalized = (log_mel - mean) / (std + 1e-8)
    
    vmin, vmax = normalized.min(), normalized.max()
    if vmax - vmin > 1e-6:
        scaled = 255 * (normalized - vmin) / (vmax - vmin)
    else:
        scaled = np.zeros_like(normalized)
    
    return scaled.astype(np.uint8)

def to_rgb(spec):
    return np.stack([spec] * 3, axis=0).astype(np.float32) / 255.0

print("Функции готовы")


class SoundscapeDataset(Dataset):
    def __init__(self, df, audio_dir, sr=SR, duration=DURATION):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.sr = sr
        self.duration = duration
        self.cache = {}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        row_id = row["row_id"]
        
        file_id = "_".join(row_id.split("_")[:2])
        end_time = int(row_id.split("_")[-1])
        
        try:
            if file_id not in self.cache:
                filename = next(f for f in os.listdir(self.audio_dir) if f.startswith(file_id))
                audio, orig_sr = librosa.load(
                    os.path.join(self.audio_dir, filename), 
                    sr=None, res_type='kaiser_fast'
                )
                if orig_sr != self.sr:
                    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sr)
                self.cache[file_id] = audio
            
            audio = self.cache[file_id]
            
            start = max(0, (end_time - self.duration) * self.sr)
            end = min(len(audio), end_time * self.sr)
            segment = audio[start:end]
            
            if len(segment) < self.duration * self.sr:
                segment = np.pad(segment, (0, self.duration * self.sr - len(segment)))
            
            spec = audio_to_melspec(segment, self.sr)
            return to_rgb(spec)
        
        except:
            return np.zeros((3, 128, 313), dtype=np.float32)

print("Dataset готов")


def load_model(weights_path, num_classes):
    model = resnest50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    state = torch.load(weights_path, map_location="cpu")
    state = {k.replace("model.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    
    model.to(DEVICE)
    model.eval()
    return model

print("Загрузка модели...")
model = load_model(WEIGHTS, NUM_CLASSES)
print("Модель загружена")


@torch.no_grad()
def predict_batch(batch, model, threshold=THRESHOLD):
    inputs = torch.from_numpy(batch).to(DEVICE)
    logits = model(inputs)
    probs = torch.sigmoid(logits).cpu().numpy()
    
    predictions = []
    for prob in probs:
        indices = np.where(prob > threshold)[0]
        if len(indices) == 0:
            predictions.append("nocall")
        else:
            labels = encoder.inverse_transform(indices)
            predictions.append(" ".join(sorted(labels)))
    
    return predictions

print("Inference функция готова")


test_df = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))

if len(test_df) < 10:
    print("Используем train_soundscapes для теста")
    test_df = pd.read_csv(os.path.join(DATA_ROOT, "train_soundscape_labels.csv"))
    audio_dir = os.path.join(DATA_ROOT, "train_soundscapes")
else:
    audio_dir = TEST_AUDIO

print(f"Сегментов: {len(test_df)}")


dataset = SoundscapeDataset(test_df, audio_dir)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

all_predictions = []

print("Запуск инференса...")
for batch in tqdm(loader, desc="Processing"):
    batch_array = np.stack([b.numpy() for b in batch])
    preds = predict_batch(batch_array, model, THRESHOLD)
    all_predictions.extend(preds)

print(f"\nОбработано {len(all_predictions)} сегментов")


submission = pd.DataFrame({
    "row_id": test_df["row_id"],
    "birds": all_predictions
})

submission.to_csv("submission.csv", index=False)

nocall_pct = (submission["birds"] == "nocall").sum() / len(submission) * 100

print(f"\n{'='*60}")
print("СТАТИСТИКА")
print(f"{'='*60}")
print(f"Всего сегментов: {len(submission)}")
print(f"nocall: {nocall_pct:.1f}%")
print(f"с птицами: {100-nocall_pct:.1f}%")
print(f"\nПервые 10 строк:")
print(submission.head(10))
print(f"\nСохранено в submission.csv")


import shutil

if os.path.exists('resnet'):
    shutil.rmtree('resnet')
    print("Временная папка resnet удалена")

