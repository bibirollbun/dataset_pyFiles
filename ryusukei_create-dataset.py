import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa
import cv2
import pickle
#pytorch関連
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import torchvision.transforms as transforms


class CFG:
    # 音声データの変換関係のパラメータ
    FS = 32000
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    TARGET_DURATION = 5
    TARGET_SHAPE = (256, 256)

    extract_human_voice = False
    debug = False


cfg = CFG()


def exclude_human_voice(audio_data: np.array, voice_segments: list[dict], cfg: CFG) -> np.ndarray:
    """train_audioに含まれる人の声を削除する."""
    # 除外区間の前後をつなげていく
    include_segments = []
    prev_end_sample = 0
    
    for seg in voice_segments:
        start_sample = int(seg['start'] * cfg.FS)
        end_sample = int(seg['end'] * cfg.FS)
    
        if start_sample > prev_end_sample:
            include_segments.append(audio_data[prev_end_sample:start_sample])
    
        prev_end_sample = end_sample
    
    # 最後に残りがあれば追加
    if prev_end_sample < len(audio_data):
        include_segments.append(audio_data[prev_end_sample:])
    
    # すべて連結
    cleaned_audio = np.concatenate(include_segments)
    
    return cleaned_audio

def audio2melspec(audio_data: np.ndarray, cfg: CFG) -> np.ndarray:
    """音声データをデシベル単位のメルスペクトログラムに変換し、正規化(Min-Max正規化)する"""

    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        hop_length=cfg.HOP_LENGTH
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

    return mel_spec_norm

def preprocess_audio_file(audio_path: str, voice_segments: dict, cfg: CFG) -> np.ndarray:
    """音声データを読み込み、真ん中の5秒間を正規化したメルスペクトログラムに変換する(db単位)"""
    
    try:
        # 読み込み
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
        if cfg.extract_human_voice:
            # 人間の声を除く
            audio_data = exclude_human_voice(audio_data, voice_segments, cfg)
        
        target_samples = int(cfg.TARGET_DURATION * cfg.FS)
        start_idx = max(0, int(len(audio_data) / 2 - target_samples / 2))
        end_idx = min(len(audio_data), start_idx + target_samples)
        center_audio = audio_data[start_idx: end_idx]

        if len(center_audio) < target_samples:
            center_audio = np.pad(
                center_audio,
                pad_width = (0, target_samples - len(center_audio)),
                mode = 'constant'
            )

        mel_spec = audio2melspec(center_audio, cfg)

        if mel_spec.shape != cfg.TARGET_SHAPE:
            mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

        return mel_spec.astype(np.float32)

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


# 実際にメルスペクトログラム変換していく
train_csv = pd.read_csv("/kaggle/input/birdclef-2025/train.csv")
with open("/kaggle/input/bc25-separation-voice-from-data/train_voice_data.pkl", "rb") as f:
    train_voice_data = pickle.load(f)

images = []
labels = []

cnt_for_debug = 0

for _, row in tqdm(train_csv.iterrows(), total=28564):
    audio_path = os.path.join("/kaggle/input/birdclef-2025/train_audio", row["filename"])
    voice_segments = train_voice_data.get(audio_path, [])
    
    image = preprocess_audio_file(
        audio_path=audio_path,
        voice_segments=voice_segments,
        cfg=cfg
    )
    images.append(image)
    labels.append(row['primary_label'])

    if cfg.debug:
        cnt_for_debug += 1
        if cnt_for_debug >= 10:
            break


print(np.array(images).shape)
print(np.array(labels).shape)


np.savez("/kaggle/working/my_dataset_exclude_human_voice.npz", images=np.array(images), labels=np.array(labels))

