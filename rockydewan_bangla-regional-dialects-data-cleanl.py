import os, random, numpy as np, librosa, soundfile as sf, matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from IPython.display import Audio as ipdAudio


BASE_DIR = "/kaggle/input/ben10/ben10"
TRAIN_DIR = os.path.join(BASE_DIR, "16_kHz_train_audio")
VALID_DIR = os.path.join(BASE_DIR, "16_kHz_valid_audio")

RAW_AUDIO_DIR = BASE_DIR  
CLEAN_AUDIO_DIR = "./clean_audio"  
os.makedirs(CLEAN_AUDIO_DIR, exist_ok=True)

SAMPLE_RATE = 16000
ENERGY_THRESHOLD = 0.001  

print("Base folder:", BASE_DIR)
print("Train folder:", TRAIN_DIR)
print("Valid folder:", VALID_DIR)
print("Output folder:", CLEAN_AUDIO_DIR)



files = []
for d in [TRAIN_DIR, VALID_DIR]:
    for dp, _, fs in os.walk(d):
        for f in fs:
            if f.lower().endswith(".wav"):
                files.append(os.path.join(dp, f))

print(f"Total audio files found (train+valid): {len(files)}")


import librosa.display

def inspect_random_audio():
    f = random.choice(files)
    y, sr = librosa.load(f, sr=None, mono=False)
    print(f"Example file: {f}\nOriginal sr={sr}, shape={y.shape}")
    plt.figure(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Raw waveform example")
    plt.show()
    return ipdAudio(y, rate=sr)

inspect_random_audio()


def preprocess_audio(src_path, dst_path, target_sr=SAMPLE_RATE):
    try:
        y, sr = librosa.load(src_path, sr=None, mono=True)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        y = librosa.util.normalize(y)
        y = y - np.mean(y)
        sf.write(dst_path, y, target_sr)
        return True
    except Exception as e:
        print(f"Failed: {src_path} -> {e}")
        return False



for f in tqdm(files, desc="Preprocessing train+valid"):
    dst = os.path.join(CLEAN_AUDIO_DIR, Path(f).name)
    preprocess_audio(f, dst)


def is_silent(path):
    try:
        y, sr = librosa.load(path, sr=None)
        return np.mean(y**2) < ENERGY_THRESHOLD
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return True  # if failed, treat as silent

kept = []
removed = 0
for f in tqdm(os.listdir(CLEAN_AUDIO_DIR), desc="Filtering silent files"):
    full_path = os.path.join(CLEAN_AUDIO_DIR, f)
    if is_silent(full_path):
        os.remove(full_path)
        removed += 1
    else:
        kept.append(full_path)

print(f"âœ… Kept {len(kept)} non-silent clips, removed {removed} silent ones.")
print(f"ðŸŽ§ Cleaned dataset saved in: {CLEAN_AUDIO_DIR}")



import random
def preview_cleaned():
    f = random.choice(os.listdir(CLEAN_AUDIO_DIR))
    path = os.path.join(CLEAN_AUDIO_DIR, f)
    y, sr = librosa.load(path, sr=None)
    print("Preview:", path)
    plt.figure(figsize=(12,3))
    librosa.display.waveshow(y, sr=sr)
    plt.show()
    return ipdAudio(y, rate=sr)

preview_cleaned()


