import shutil
shutil.copytree('../input/resnest50-fast-package/resnest-0.0.6b20200701/resnest', 'resnet', dirs_exist_ok=True) 
!pip install "./resnet" --no-deps


import os
import json
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from resnest.torch import resnest50
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt


#1. Определение путей и устройств
DATA_DIR = "../input/birdclef-2021"
TEST_AUDIO_DIR = os.path.join(DATA_DIR, "test_soundscapes")
SAMPLE_RATE = 32000
SEGMENT_SEC = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#2. Подготовка меток
metadata = pd.read_csv(os.path.join(DATA_DIR, "train_metadata.csv"))
all_birds = sorted(set(metadata["primary_label"]))

label_encoder = LabelEncoder().fit(all_birds)
NUM_CLASSES = len(all_birds)
print(f"Using {NUM_CLASSES} scored bird species.")


#Пример названий птиц
all_birds[:11]


#3. Аудио в спектрограмму
def audio_to_melspec(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=128, fmin=0, fmax=sr//2,
        n_fft=sr//10, hop_length=sr//40
    )
    db = librosa.power_to_db(mel, ref=np.max)
    
    # Z-нормализация
    db = (db - db.mean()) / (db.std() + 1e-8)
    
    db_min, db_max = db.min(), db.max()
    if db_max - db_min > 1e-6:
        db = 255 * (db - db_min) / (db_max - db_min)
    else:
        db = np.zeros_like(db)
    return db.astype(np.uint8)

def spec_to_tensor(spec: np.ndarray) -> np.ndarray:
    # Convert 2D spectrogram to 3-channel float tensor [C, H, W]."""
    rgb = np.stack([spec] * 3, axis=0).astype(np.float32) / 255.0
    return rgb


class BirdSoundDataset(Dataset):
    def __init__(self, df, audio_dir, sr=32000, segment_sec=5):
        self.df = df
        self.audio_dir = audio_dir
        self.sr = sr
        self.segment_sec = segment_sec
        self.cache = {}  # кеширование

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row_id = self.df.iloc[idx]["row_id"]
        parts = row_id.split("_")
        file_id, site, end_sec = parts[0], parts[1], int(parts[2])
        prefix = f"{file_id}_{site}"

        try:
            if prefix not in self.cache:
                audio_file = next(f for f in os.listdir(self.audio_dir) if f.startswith(prefix))
                audio_path = os.path.join(self.audio_dir, audio_file)
                audio, sr = librosa.load(audio_path, sr=None, res_type='kaiser_fast')
                if sr != self.sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
                self.cache[prefix] = audio
            else:
                audio = self.cache[prefix]

            start = max(0, (end_sec - self.segment_sec) * self.sr)
            end = min(len(audio), end_sec * self.sr)
            segment = audio[start:end]
            if len(segment) < self.segment_sec * self.sr:
                segment = np.pad(segment, (0, self.segment_sec * self.sr - len(segment)))

            mel = audio_to_melspec(segment, self.sr)
            tensor = spec_to_tensor(mel)
            return tensor

        except Exception:
            return np.zeros((3, 128, 313), dtype=np.float32)


#4. Загрузка модели
def load_resnest_model(weights_path: str, num_classes: int):
    model = resnest50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    # Загрузка весов
    state_dict = torch.load(weights_path, map_location="cpu")
    new_state = {}
    for k, v in state_dict.items():
        new_key = k.replace("model.", "") if k.startswith("model.") else k
        new_state[new_key] = v
    model.load_state_dict(new_state)
    
    model.to(device)
    model.eval()
    return model

# Загружаем предобученную модель
MODEL_PATH = "../input/kkiller-birdclef-models-public/birdclef_resnest50_fold0_epoch_10_f1_val_06471_20210417161101.pth"
model = load_resnest_model(MODEL_PATH, NUM_CLASSES)



#5. Inference с порогом
def predict_batch(batch: np.ndarray, model, threshold: float = 0.1):
    with torch.no_grad():
        inputs = torch.from_numpy(batch).to(device)
        logits = model(inputs)
        probs = torch.sigmoid(logits).cpu().numpy()
    
    predictions = []
    for p in probs:
        active = np.where(p > threshold)[0]
        if len(active) == 0:
            predictions.append("nocall")
        else:
            bird_names = label_encoder.inverse_transform(active)
            predictions.append(" ".join(sorted(bird_names)))
    return predictions


#6. Работа с данными 
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
if len(test_df) < 5:  
    test_df = pd.read_csv(os.path.join(DATA_DIR, "train_soundscape_labels.csv"))
    audio_dir = os.path.join(DATA_DIR, "train_soundscapes")
else:
    audio_dir = TEST_AUDIO_DIR

print(f"Processing {len(test_df)} segments...")


# Создаём датасет и DataLoader
dataset = BirdSoundDataset(test_df, audio_dir)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)


all_preds = []
for batch in tqdm(dataloader, desc="Inference"):
    batch_np = np.stack([b.numpy() for b in batch])
    preds = predict_batch(batch_np, model, threshold=0.1)
    all_preds.extend(preds)


from IPython.display import Audio, display

#Найдём индексы, где предсказание — не "nocall"
non_nocall_indices = [i for i, pred in enumerate(all_preds) if pred != "nocall"]
indices_to_show = non_nocall_indices[:6]
num_viz = len(indices_to_show)

if num_viz > 0:
    for idx, i in enumerate(indices_to_show):
        #Получаем аудио
        row_id = test_df.iloc[i]["row_id"]
        parts = row_id.split("_")
        file_id, site, end_sec = parts[0], parts[1], int(parts[2])
        prefix = f"{file_id}_{site}"

        audio_file = next(f for f in os.listdir(audio_dir) if f.startswith(prefix))
        audio_path = os.path.join(audio_dir, audio_file)
        audio_full, sr_orig = librosa.load(audio_path, sr=None, res_type='kaiser_fast')
        if sr_orig != SAMPLE_RATE:
            audio_full = librosa.resample(audio_full, orig_sr=sr_orig, target_sr=SAMPLE_RATE)
        
        start = max(0, (end_sec - SEGMENT_SEC) * SAMPLE_RATE)
        end = min(len(audio_full), end_sec * SAMPLE_RATE)
        segment_audio = audio_full[start:end]
        if len(segment_audio) < SEGMENT_SEC * SAMPLE_RATE:
            segment_audio = np.pad(segment_audio, (0, SEGMENT_SEC * SAMPLE_RATE - len(segment_audio)))

        #Получаем спектрограмму
        spec_tensor = dataset[i] 
        pred_birds = all_preds[i]
        spec_img = spec_tensor[0]

        #Выводим аудио
        print(f"\nСегмент {i+1} | row_id: {row_id} | Prediction: '{pred_birds}'")
        display(Audio(segment_audio, rate=SAMPLE_RATE))

        #Выводим спектрограмму
        plt.figure(figsize=(14, 3))
        im = plt.imshow(spec_img, aspect='auto', origin='lower', cmap='magma')
        plt.title(f"Mel-spectrogram | {row_id}", fontsize=12)
        plt.xlabel("Time frames")
        plt.ylabel("Mel bins")
        plt.colorbar(im, shrink=0.6)
        plt.tight_layout()
        plt.show()

else:
    print("Нет сегментов с предсказанием, отличным от 'nocall'.")


#7. Сохранение submission
submission = pd.DataFrame({
    "row_id": test_df["row_id"],
    "birds": all_preds
})
submission.to_csv("submission.csv", index=False)


print(submission)

