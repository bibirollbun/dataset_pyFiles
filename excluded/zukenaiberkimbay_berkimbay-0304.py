import os
import math
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

# Конфигурация
N_CLASSES = 397
SAMPLE_RATE = 32000
CLIP_LEN = 5
CONF_THRESH = 0.2  # Порог уверенности

# Определение устройства
ACCELERATOR = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {ACCELERATOR}")

# Пути к данным
BASE_DIR = Path("../input/birdclef-2021")
TEST_DIR = BASE_DIR / "test_soundscapes"
SUB_FILE = BASE_DIR / "sample_submission.csv"
LABELS_FILE = None

# Проверка режима (Submit vs Offline Debug)
if len(list(TEST_DIR.glob("*.ogg"))) == 0:
    print("Debug mode: using train_soundscapes")
    TEST_DIR = BASE_DIR / "train_soundscapes"
    SUB_FILE = None
    LABELS_FILE = BASE_DIR / "train_soundscape_labels.csv"

print(f"Reading audio from: {TEST_DIR}")



!ls /kaggle/input/resnest50-fast-package/resnest-0.0.6b20200701/resnest


import sys
# Добавляем пути к локальному пакету resnest
package_path = "/kaggle/input/resnest50-fast-package/resnest-0.0.6b20200701"
sys.path.append(package_path)
sys.path.append(os.path.join(package_path, "resnest"))

from resnest.torch import resnest50
print("ResNeSt library loaded.")


LOCATION_BIRDS = {}

if LABELS_FILE and LABELS_FILE.exists():
    df_labels = pd.read_csv(LABELS_FILE)
    for row in df_labels.itertuples(index=False):
        if isinstance(row.birds, str) and row.birds != "nocall":
            bird_list = row.birds.split()
            if row.site not in LOCATION_BIRDS:
                LOCATION_BIRDS[row.site] = set()
            LOCATION_BIRDS[row.site].update(bird_list)

    print("Unique birds per location:")
    for loc, birds in LOCATION_BIRDS.items():
        print(f"Location {loc}: {len(birds)}")
else:
    print("Labels file not found, skipping stats.")



file_list = []
for p in TEST_DIR.glob("*.ogg"):
    parts = p.stem.split("_")
    file_list.append((p.stem, parts[0], parts[1], parts[2], p))

meta_df = pd.DataFrame(file_list, columns=["filename", "id", "site", "date", "filepath"])
print(f"Total soundscapes: {len(meta_df)}")
meta_df.head()


class AudioToSpec:
    def __init__(self, rate, mels, f_min, f_max, **config):
        self.rate = rate
        self.mels = mels
        self.f_min = f_min
        self.f_max = f_max
        config["n_fft"] = config.get("n_fft", rate // 10)
        config["hop_length"] = config.get("hop_length", rate // 40)
        self.config = config

    def __call__(self, signal):
        spec = librosa.feature.melspectrogram(
            y=signal, sr=self.rate, n_mels=self.mels, 
            fmin=self.f_min, fmax=self.f_max, **self.config
        )
        return librosa.power_to_db(spec).astype(np.float32)

def norm_image(img, eps=1e-6):
    mean_val = img.mean()
    std_val = img.std()
    img = (img - mean_val) / (std_val + eps)
    
    min_v, max_v = img.min(), img.max()
    if (max_v - min_v) > eps:
        img = np.clip(img, min_v, max_v)
        img = 255 * (img - min_v) / (max_v - min_v)
        return img.astype(np.uint8)
    return np.zeros_like(img, dtype=np.uint8)


class InferenceDataset(Dataset):
    def __init__(self, df, sr=SAMPLE_RATE, duration=CLIP_LEN):
        self.df = df
        self.sr = sr
        self.duration = duration
        self.chunk_len = int(duration * sr)
        self.converter = AudioToSpec(sr=sr, mels=128, f_min=0, f_max=sr//2)

    def __len__(self):
        return len(self.df)

    def _process_chunk(self, chunk):
        spec = self.converter(chunk)
        img = norm_image(spec)
        img = img.astype("float32", copy=False) / 255.0
        return np.stack([img, img, img])

    def __getitem__(self, idx):
        path = self.df.loc[idx, "filepath"]
        raw_audio, _ = sf.read(path, dtype="float32")
        
        # Если частота не совпадает, ресемплим (но обычно в датасете 32k)
        if len(raw_audio) > 0:
            # Просто заглушка, librosa.resample тяжелый, надеемся на совпадение SR
            pass 

        # Нарезка
        chunks = []
        for i in range(0, len(raw_audio), self.chunk_len):
            if i + self.chunk_len <= len(raw_audio):
                chunks.append(raw_audio[i : i + self.chunk_len])
        
        # Если файл пустой или странный
        if not chunks:
            chunks = [np.zeros(self.chunk_len, dtype=np.float32)]

        images = [self._process_chunk(c) for c in chunks]
        return np.stack(images)


class InferenceDataset(Dataset):
    def __init__(self, df, sr=SAMPLE_RATE, duration=CLIP_LEN):
        self.df = df
        self.sr = sr
        self.duration = duration
        self.chunk_len = int(duration * sr)
        # ИСПРАВЛЕНИЕ: передаем rate=sr (было sr=sr)
        self.converter = AudioToSpec(rate=sr, mels=128, f_min=0, f_max=sr//2)

    def __len__(self):
        return len(self.df)

    def _process_chunk(self, chunk):
        spec = self.converter(chunk)
        img = norm_image(spec)
        img = img.astype("float32", copy=False) / 255.0
        return np.stack([img, img, img])

    def __getitem__(self, idx):
        path = self.df.loc[idx, "filepath"]
        # Читаем аудио. Если файл битый, вернем заглушку
        try:
            raw_audio, _ = sf.read(path, dtype="float32")
        except:
            raw_audio = np.array([])
        
        chunks = []
        for i in range(0, len(raw_audio), self.chunk_len):
            # Берем куски только полной длины (или паддим, если нужно, но здесь просто режем)
            # В оригинале мы паддили короткие, но для простоты берем полные 5 сек
            # Но чтобы не терять хвосты, лучше дополнить нулями, если кусок < 5 сек
            chunk = raw_audio[i : i + self.chunk_len]
            if len(chunk) < self.chunk_len:
                padding = np.zeros(self.chunk_len - len(chunk), dtype="float32")
                chunk = np.concatenate([chunk, padding])
            chunks.append(chunk)
            
        if not chunks:
            chunks = [np.zeros(self.chunk_len, dtype=np.float32)]

        images = [self._process_chunk(c) for c in chunks]
        return np.stack(images)



ds_test = InferenceDataset(df=meta_df)
print(f"Dataset size: {len(ds_test)}")

first_item = ds_test[0]
print(f"Batch shape: {first_item.shape}")

plt.figure(figsize=(8, 3))
plt.imshow(first_item[0].transpose(1, 2, 0))
plt.title("Sample Spectrogram")
plt.axis("off")
plt.show()



import gc
gc.collect()
torch.cuda.empty_cache()


meta_train = pd.read_csv(BASE_DIR / "train_metadata.csv")
unique_labels = sorted(meta_train["primary_label"].unique())
CLASS_TO_ID = {lbl: i for i, lbl in enumerate(unique_labels)}
ID_TO_CLASS = {i: lbl for lbl, i in CLASS_TO_ID.items()}
print(f"Classes count: {len(CLASS_TO_ID)}")

def get_model(ckpt_path):
    model = resnest50(pretrained=False)
    # Меняем последний слой
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_TO_ID))
    
    weights = torch.load(ckpt_path, map_location="cpu")
    # Убираем префикс "model." если есть
    clean_weights = {k.replace("model.", ""): v for k, v in weights.items()}
    
    model.load_state_dict(clean_weights, strict=True)
    model.to(ACCELERATOR)
    model.eval()
    return model

model_path = Path("/kaggle/input/kkiller-birdclef-models-public/birdclef_resnest50_fold0_epoch_10_f1_val_06471_20210417161101.pth")
models_list = [get_model(model_path)]
print("Model loaded successfully.")


@torch.no_grad()
def decode_output(probs, threshold=CONF_THRESH):
    # probs: tensor [batch, n_classes]
    # Возвращаем имена птиц
    results = []
    for row in probs:
        indices = torch.where(row > threshold)[0]
        if len(indices) == 0:
            results.append("nocall")
        else:
            names = [ID_TO_CLASS[x.item()] for x in indices]
            # Сортируем по вероятности (опционально), здесь просто список
            results.append(" ".join(names))
    return results

@torch.no_grad()
def inference_loop(models, dataset, batch_size=32):
    final_preds = []
    
    for i in tqdm(range(len(dataset)), desc="Processing files"):
        file_imgs = dataset[i] # [num_windows, 3, H, W]
        n_windows = file_imgs.shape[0]
        file_predictions = []

        for start in range(0, n_windows, batch_size):
            end = min(start + batch_size, n_windows)
            input_tensor = torch.tensor(file_imgs[start:end]).to(ACCELERATOR)
            
            # Ансамбль (здесь одна модель, но код готов к нескольким)
            avg_preds = torch.zeros((end-start, len(CLASS_TO_ID)), device=ACCELERATOR)
            for m in models:
                logits = m(input_tensor)
                avg_preds += torch.sigmoid(logits)
            
            avg_preds /= len(models)
            file_predictions.extend(decode_output(avg_preds))
        
        final_preds.append(file_predictions)
        
    return final_preds


all_predictions = inference_loop(models_list, ds_test)
print(f"Processed {len(all_predictions)} files.")
print(f"First 5 windows of file 1: {all_predictions[0][:5]}")


def make_submission(meta, preds, step=CLIP_LEN):
    res = []
    for row, p_list in zip(meta.itertuples(), preds):
        base_id = row.id
        site = row.site
        for i, label in enumerate(p_list):
            seconds = (i + 1) * step
            row_id = f"{base_id}_{site}_{seconds}"
            res.append({"row_id": row_id, "birds": label})
    
    df_res = pd.DataFrame(res)
    
    # Мердж с сэмплом если он есть
    if SUB_FILE:
        sample = pd.read_csv(SUB_FILE, usecols=["row_id"])
        df_res = sample.merge(df_res, on="row_id", how="left").fillna("nocall")
        
    return df_res

submission_df = make_submission(meta_df, all_predictions)
submission_df.head()


submission_df.to_csv("submission.csv", index=False)
print("File saved: submission.csv")


def calc_f1_metric(true_list, pred_list):
    # Метрика F1 Micro row-wise
    tp_tot, fp_tot, fn_tot = 0, 0, 0
    
    for t, p in zip(true_list, pred_list):
        t = "nocall" if pd.isna(t) else t
        p = "nocall" if pd.isna(p) else p
        
        set_t = set(t.split()) if t != "nocall" else set()
        set_p = set(p.split()) if p != "nocall" else set()
        
        tp = len(set_t & set_p)
        fp = len(set_p - set_t)
        fn = len(set_t - set_p)
        
        tp_tot += tp
        fp_tot += fp
        fn_tot += fn

    if (tp_tot + fp_tot + fn_tot) == 0: return 0.0
    
    prec = tp_tot / (tp_tot + fp_tot + 1e-9)
    rec = tp_tot / (tp_tot + fn_tot + 1e-9)
    
    if (prec + rec) == 0: return 0.0
    return 2 * prec * rec / (prec + rec)

if LABELS_FILE:
    print("Calculating offline score...")
    true_df = pd.read_csv(LABELS_FILE)
    combined = true_df.merge(submission_df, on="row_id", suffixes=("_real", "_pred"))
    
    metric = calc_f1_metric(combined["birds_real"], combined["birds_pred"])
    print(f"Validation F1 Score: {metric:.4f}")
else:
    print("Submission mode. Score available on Leaderboard.")

