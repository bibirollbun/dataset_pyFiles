import os
import gc
import warnings
import logging
import time
import math
import cv2
import random 
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


# --- Configuration ---
class Config:
    DEBUG_MODE = False
    N_MAX_DEBUG = 100 

    # --- OUTPUT ---
    OUTPUT_DIR = '/kaggle/working/'
    OUTPUT_NPY_FILE = os.path.join(OUTPUT_DIR, f'birdclef25_melspec_5s_randcrop_32k_2048fft_512hop_128mel_rs256.npy')

    # --- INPUT DATA ---
    DATA_ROOT = '/kaggle/input/birdclef-2025'
    AUDIO_SUBDIR = 'train_audio' 

    # --- AUDIO PARAMETERS ---
    FS = 32000
    TARGET_DURATION = 5.0
    USE_RANDOM_CROP = True 

    # --- MEL SPECTROGRAM PARAMETERS ---
    N_FFT = 2048
    HOP_LENGTH = 512
    WIN_LENGTH = 2048 
    N_MELS = 128  
    FMIN = 20
    FMAX = 16000

    # --- TARGET SHAPE & RESIZE ---
    DO_RESIZE = True #
    TARGET_SHAPE = (256, 256)

config = Config()


# --- Setup ---
print(f"--- Configuration ---")
print(f"DEBUG_MODE: {'ON' if config.DEBUG_MODE else 'OFF'} ({config.N_MAX_DEBUG if config.DEBUG_MODE else 'ALL'} samples)")
print(f"Target Duration: {config.TARGET_DURATION}s")
print(f"Use Random Crop: {config.USE_RANDOM_CROP}")
print(f"Sample Rate: {config.FS}")
print(f"Mel Params: N_FFT={config.N_FFT}, HOP={config.HOP_LENGTH}, WIN={config.WIN_LENGTH}, N_MELS={config.N_MELS}, FMIN={config.FMIN}, FMAX={config.FMAX}")
print(f"Resize Spectrogram: {config.DO_RESIZE}")
if config.DO_RESIZE:
    print(f"Target Shape (after resize): {config.TARGET_SHAPE}")
else:
    calculated_shape = (config.N_MELS, int(np.floor((config.TARGET_DURATION * config.FS) / config.HOP_LENGTH) + 1))
    print(f"Target Shape (original): {calculated_shape}")
    config.TARGET_SHAPE = calculated_shape 
print(f"Output NPY file: {config.OUTPUT_NPY_FILE}")
print(f"--------------------")

os.makedirs(config.OUTPUT_DIR, exist_ok=True)

print("Loading training metadata...")
train_df = pd.read_csv(f'{config.DATA_ROOT}/train.csv')
working_df = train_df[['filename']].copy() 
working_df['filepath'] = config.DATA_ROOT + '/' + config.AUDIO_SUBDIR + '/' + working_df.filename
# *** Tạo samplename khớp với training: bỏ '.ogg' ***
working_df['samplename'] = working_df.filename.map(lambda x: x.replace('.ogg',''))

total_samples_available = len(working_df)
samples_to_process = min(total_samples_available, config.N_MAX_DEBUG if config.DEBUG_MODE else total_samples_available)

print(f'Total samples to process: {samples_to_process} out of {total_samples_available} available')


# --- Hàm xử lý Audio ---
def audio_to_melspectrogram(audio_data, config):
    """Converts audio data to a Mel spectrogram according to Config."""
    if np.isnan(audio_data).any():
        # print(f"Warning: NaN found in audio data, replacing with mean.")
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal if not np.isnan(mean_signal) else 0.0)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=config.FS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
        n_mels=config.N_MELS,
        fmin=config.FMIN,
        fmax=config.FMAX,
        power=2.0,
        center=True,
        pad_mode="reflect"
    )

    # Convert to decibels and normalize to [0, 1]
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    if mel_spec_db.max() == mel_spec_db.min(): # Handle silent clips
        return np.zeros((config.N_MELS, int(np.floor((config.TARGET_DURATION * config.FS) / config.HOP_LENGTH) + 1)), dtype=np.float32)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

    return mel_spec_norm

def process_single_file(filepath, samplename, config):
    """Loads audio, extracts/pads 5s, creates Mel spectrogram, and resizes if needed."""
    try:
        audio_data, _ = librosa.load(filepath, sr=config.FS)

        target_samples = int(config.TARGET_DURATION * config.FS)
        current_samples = len(audio_data)
        processed_audio = np.zeros(target_samples, dtype=np.float32) # Khởi tạo mảng 0

        if current_samples == 0:
            # print(f"Warning: Empty audio file {filepath}. Returning zero spec.")
            # Tạo spec zero với shape gốc trước resize
            orig_shape = (config.N_MELS, int(np.floor(target_samples / config.HOP_LENGTH) + 1))
            mel_spec_norm = np.zeros(orig_shape, dtype=np.float32)

        elif current_samples < target_samples:
            # Pad audio ngắn
            start = random.randint(0, target_samples - current_samples) # Pad ngẫu nhiên 2 đầu
            processed_audio[start : start + current_samples] = audio_data
            mel_spec_norm = audio_to_melspectrogram(processed_audio, config)

        elif current_samples == target_samples:
            processed_audio = audio_data
            mel_spec_norm = audio_to_melspectrogram(processed_audio, config)

        else: 
            if config.USE_RANDOM_CROP:
                # *** Lấy 5 giây ngẫu nhiên ***
                max_start_idx = current_samples - target_samples
                start_idx = random.randint(0, max_start_idx)
                processed_audio = audio_data[start_idx : start_idx + target_samples]
            else:
                processed_audio = audio_data[:target_samples]
            mel_spec_norm = audio_to_melspectrogram(processed_audio, config)

        # --- Resize ---
        final_spec = mel_spec_norm
        if config.DO_RESIZE:
            if final_spec.shape[1] == 0:
                 print(f"Warning: Spectrogram for {samplename} has zero width before resize. Creating zero array.")
                 final_spec = np.zeros(config.TARGET_SHAPE, dtype=np.float32)
            elif final_spec.shape != config.TARGET_SHAPE:
                 # cv2.resize cần (width, height)
                 final_spec = cv2.resize(final_spec, (config.TARGET_SHAPE[1], config.TARGET_SHAPE[0]), interpolation=cv2.INTER_LINEAR)


        return samplename, final_spec.astype(np.float32)

    except Exception as e:
        # print(f"Error processing {filepath}: {e}")
        return samplename, None


# --- Vòng lặp xử lý chính ---
print("\nProcessing audio files...")
start_time = time.time()

all_spectrograms = {}
error_count = 0

process_df = working_df.head(samples_to_process)

for _, row in tqdm(process_df.iterrows(), total=samples_to_process):
    samplename, spec = process_single_file(row['filepath'], row['samplename'], config)
    if spec is not None:
        all_spectrograms[samplename] = spec
    else:
        error_count += 1

# --- Kết thúc và Lưu ---
end_time = time.time()
print(f"\nProcessing finished in {(end_time - start_time):.2f} seconds.")
print(f"Successfully generated {len(all_spectrograms)} spectrograms.")
if error_count > 0:
    print(f"Encountered errors in {error_count} files.")

# --- Lưu file NPY ---
if all_spectrograms:
    print(f"\nSaving spectrogram data to {config.OUTPUT_NPY_FILE}...")
    np.save(config.OUTPUT_NPY_FILE, all_spectrograms, allow_pickle=True)
    print(f"Data saved successfully. Shape of first spec: {next(iter(all_spectrograms.values())).shape}")
else:
    print("\nNo spectrogram data generated to save.")


# --- Visualize ---
if config.DEBUG_MODE and all_spectrograms: 
    print("\nVisualizing examples...")
    visualize_keys = random.sample(list(all_spectrograms.keys()), min(4, len(all_spectrograms)))
    plt.figure(figsize=(12, 5 * (len(visualize_keys)//2 + len(visualize_keys)%2))) 
    for i, key in enumerate(visualize_keys):
        plt.subplot( (len(visualize_keys)+1)//2 , 2, i + 1)
        spec_to_show = all_spectrograms[key]
        if config.DO_RESIZE:
             plt.imshow(spec_to_show, aspect='auto', origin='lower', cmap='viridis')
             plt.title(f"{key}\nShape: {spec_to_show.shape} (Resized)")
             plt.colorbar()
        else:
             librosa.display.specshow(spec_to_show, sr=config.FS, hop_length=config.HOP_LENGTH,
                                     x_axis='time', y_axis='mel', fmin=config.FMIN, fmax=config.FMAX, cmap='viridis')
             plt.title(f"{key}\nShape: {spec_to_show.shape}")
             plt.colorbar(format='%+2.0f dB') 

    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'debug_melspec_examples.png'))
    print(f"Visualization saved to {os.path.join(config.OUTPUT_DIR, 'debug_melspec_examples.png')}")
    plt.show() 


import os
dataset_name = "melspec-train-audio-update"

os.makedirs(dataset_name, exist_ok=True)
!cp birdclef25_melspec_5s_randcrop_32k_2048fft_512hop_128mel_rs256.npy {dataset_name}/


import os

os.environ['KAGGLE_USERNAME'] = ''
os.environ['KAGGLE_KEY'] = ''


metadata = {
    "title": "MelSpec Train Audio",
    "id": f"{os.environ['KAGGLE_USERNAME']}/{dataset_name}",
    "licenses": [{"name": "CC0-1.0"}]
}

import json
with open(f"{dataset_name}/dataset-metadata.json", "w") as f:
    json.dump(metadata, f)


!kaggle datasets create -p {dataset_name}

