import os
import cv2
import math
import time
import librosa
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm


import torch
import warnings
warnings.filterwarnings("ignore")


class Config:
 
    DEBUG_MODE = False
    
    OUTPUT_DIR = '/kaggle/working/'
    DATA_ROOT = '/kaggle/input/birdclef-2025'
    FS = 32000
    
    # Mel spectrogram parameters
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 256
    FMIN = 50
    FMAX = 14000
    
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (256, 256)  
    
    N_MAX = 50 if DEBUG_MODE else None  

config = Config()


import os
import numpy as np
import librosa
import random

# ─── 1. 简单的纯 Python 随机增强函数 ───────────────────────────────────────────
def augment_audio_librosa(audio: np.ndarray, sr: int, target_len: int) -> np.ndarray:
    # 1) 添加高斯噪声
    if random.random() < 0.5:
        noise_amp = np.random.uniform(0.001, 0.015)
        audio = audio + noise_amp * np.random.randn(len(audio))
    # 2) 随机时域拉伸（并补偿长度）
    if random.random() < 0.3:
        rate = np.random.uniform(0.8, 1.25)
        audio = librosa.effects.time_stretch(audio, rate)
        # 裁剪／填零回到 target_len
        if len(audio) < target_len:
            pad = target_len - len(audio)
            audio = np.pad(audio, (pad//2, pad-pad//2), mode="constant")
        else:
            start = (len(audio) - target_len)//2
            audio = audio[start:start+target_len]
    # 3) 随机音高偏移
    if random.random() < 0.4:
        n_steps = np.random.uniform(-4, 4)
        audio = librosa.effects.pitch_shift(audio, sr, n_steps)
    # 4) 随机时间平移（roll）
    if random.random() < 0.4:
        shift_amt = int(np.random.uniform(-0.5, 0.5) * len(audio))
        audio = np.roll(audio, shift_amt)
    return audio


print(f"Debug mode: {'ON' if config.DEBUG_MODE else 'OFF'}")
print(f"Max samples to process: {config.N_MAX if config.N_MAX is not None else 'ALL'}")

print("Loading taxonomy data...")
taxonomy_df = pd.read_csv(f'{config.DATA_ROOT}/taxonomy.csv')
species_class_map = dict(zip(taxonomy_df['primary_label'], taxonomy_df['class_name']))

print("Loading training metadata...")
train_df = pd.read_csv(f'{config.DATA_ROOT}/train.csv')


label_list = sorted(train_df['primary_label'].unique())
label_id_list = list(range(len(label_list)))
label2id = dict(zip(label_list, label_id_list))
id2label = dict(zip(label_id_list, label_list))

print(f'Found {len(label_list)} unique species')
working_df = train_df[['primary_label', 'rating', 'filename']].copy()
working_df['target'] = working_df.primary_label.map(label2id)
working_df['filepath'] = config.DATA_ROOT + '/train_audio/' + working_df.filename
working_df['samplename'] = working_df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
working_df['class'] = working_df.primary_label.map(lambda x: species_class_map.get(x, 'Unknown'))
total_samples = min(len(working_df), config.N_MAX or len(working_df))
print(f'Total samples to process: {total_samples} out of {len(working_df)} available')
print(f'Samples by class:')
print(working_df['class'].value_counts())


def audio2melspec(audio_data):
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=config.FS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        fmin=config.FMIN,
        fmax=config.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm


from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def process_row(row_dict):
    """
    1) 读取音频并截取中心固定长度
    2) 随机增强（纯 Python）
    3) 生成 Mel 频谱
    4) 返回 name, mel_spec
    """
    try:
        # 动态导入保持一致
        import numpy as np, librosa, os
        from types import SimpleNamespace

        row = SimpleNamespace(**row_dict)
        audio, _ = librosa.load(row.filepath, sr=config.FS)

        # 中心截取或填零
        tgt_len = int(config.TARGET_DURATION * config.FS)
        if len(audio) < tgt_len:
            pad = tgt_len - len(audio)
            audio = np.pad(audio, (pad//2, pad-pad//2), mode='constant')
        else:
            start = (len(audio) - tgt_len)//2
            audio = audio[start:start+tgt_len]

        # 随机增强
        if config.AUGMENT:
            audio = augment_audio_librosa(audio, sr=config.FS, target_len=tgt_len)

        # Mel 频谱
        mel_spec = audio2melspec(
            audio,
            sr=config.FS,
            n_mels=config.N_MELS,
            hop_length=config.HOP_LENGTH,
            win_length=config.WIN_LENGTH
        )

        name = getattr(row, 'id', None) or os.path.basename(row.filepath)
        return name, mel_spec

    except Exception as e:
        name = row_dict.get('id', row_dict.get('filepath', 'unknown'))
        return name, f"ERROR {e}"
        
# 启动处理
print("Starting parallel audio processing...")
start_time = time.time()

from types import SimpleNamespace
all_bird_data = {}
errors = []

# 转为 dict 传递到多进程（DataFrame 行不能直接传）
rows = working_df.head(config.N_MAX or len(working_df)).to_dict(orient='records')

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_row, row) for row in rows]
    for future in tqdm(as_completed(futures), total=len(rows)):
        name, result = future.result()
        if isinstance(result, str) and result.startswith("ERROR"):
            errors.append((name, result))
        else:
            all_bird_data[name] = result

end_time = time.time()
print(f"✅ Done! Time: {end_time - start_time:.2f}s")
print(f"✅ Success: {len(all_bird_data)}")
print(f"❌ Errors: {len(errors)}")



import matplotlib.pyplot as plt

samples = []
displayed_classes = set()

max_samples = min(4, len(all_bird_data))

for i, row in working_df.iterrows():
    if i >= (config.N_MAX or len(working_df)):
        break
        
    if row['samplename'] in all_bird_data:
        if config.DEBUG_MODE:
            if row['class'] not in displayed_classes:
                samples.append((row['samplename'], row['class'], row['primary_label']))
                displayed_classes.add(row['class'])
        else:
            if row['class'] not in displayed_classes:
                samples.append((row['samplename'], row['class'], row['primary_label']))
                displayed_classes.add(row['class'])
        
        if len(samples) >= max_samples:  
            break

if samples:
    plt.figure(figsize=(16, 12))
    
    for i, (samplename, class_name, species) in enumerate(samples):
        plt.subplot(2, 2, i+1)
        plt.imshow(all_bird_data[samplename], aspect='auto', origin='lower', cmap='viridis')
        plt.title(f"{class_name}: {species}")
        plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    debug_note = "debug_" if config.DEBUG_MODE else ""
    plt.savefig(f'{debug_note}melspec_examples.png')
    plt.show()


import numpy as np

np.save('/kaggle/working/all_bird_data.npy', all_bird_data)





