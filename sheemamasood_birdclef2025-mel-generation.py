# ğŸ“¦ Basic Utilities
import os
import math
import time
import random
import logging
import warnings
from pathlib import Path

# ğŸ“Š Data Handling & Evaluation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
import pickle

# ğŸ�§ Audio Processing
import librosa
import librosa.display
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F

# ğŸ”¥ PyTorch and Model Utilities
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from dataclasses import dataclass
from typing import List

# transformers
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForSequenceClassification
from torch.optim import AdamW

# ğŸ–¼ï¸� Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# ğŸ”� Progress Tracking
from tqdm.notebook import tqdm  # for notebooks
from tqdm import tqdm  # for scripts
from tqdm.auto import tqdm

# ğŸ§  Pretrained Models
import timm

# âœ… Confirm librosa
print(f"librosa version : {librosa.__version__}")
print(f"librosa files   : {librosa.__file__}")

print("âœ… All libraries successfully imported!")



import torch

# Check for CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("âœ… CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("â�Œ CUDA not available. Using CPU.")



class Config:
    # Audio settings
    FS = 32000  # Sampling rate (audio)

    # Mel spectrogram parameters (for converting audio to image)
    N_FFT = 1024       # FFT window size
    HOP_LENGTH = 512   # Step size for each frame
    FS = 32000
    FMIN = 50          # Minimum Mel frequency
    FMAX = 14000       # Maximum Mel frequency
    
    # RGB image shape (C, H, W)
    TARGET_DURATION = 10.0
    N_MELS = 128
    MEL_SHAPE = (256, 256)      
    TARGET_SHAPE = (3, 256, 256)  


    # No limit on the number of samples during training (full dataset)
    N_MAX = None  

    # flag for training mode
    TRAINING_MODE = True  
    
    # Additional training-specific configurations
    EPOCHS = 10  
    BATCH_SIZE = 32  
    LEARNING_RATE = 0.001  

# Create the config object
config = Config()


# Root path where all files and folders are stored
DATA_ROOT = '/kaggle/input/birdclef-2025'

# Load CSVs
train_df = pd.read_csv(os.path.join(DATA_ROOT, 'train.csv'))
taxonomy_df = pd.read_csv(os.path.join(DATA_ROOT, 'taxonomy.csv'))
location_df = pd.read_csv(os.path.join(DATA_ROOT, 'recording_location.txt'), delimiter='\t')
sample_submission = pd.read_csv(os.path.join(DATA_ROOT, 'sample_submission.csv'))



print(f"âœ… Loaded train_df: {train_df.shape}")
print(f"âœ… Loaded taxonomy_df: {taxonomy_df.shape}")
print(f"âœ… Loaded location_df: {location_df.shape}")
print(f"âœ… Loaded sample_submission: {sample_submission.shape}")



# Load your VAD-cleaned CSV file
clean_train_df = pd.read_csv('/kaggle/input/birdcleft-clean-and-vad-filtered-data/train_audio_10sec_chunks_VAD_filtered.csv')
chunked_train_df = pd.read_csv("/kaggle/input/birdcleft-clean-and-vad-filtered-data/train_audio_10sec_chunks.csv")
print(f"âœ… Loaded chunked_train_df: {chunked_train_df.shape}")
print(f"âœ… Loaded clean_train_df: {clean_train_df.shape}")


working_df = pd.read_csv("/kaggle/input/melspectrogramofbirdclef-2025/working_df.csv")
print(f"âœ… Loaded working_df: {working_df.shape}")

soundscape_chunked_df = pd.read_csv("/kaggle/input/birdcleft-clean-and-vad-filtered-data/soundscape_10sec_chunks.csv")
clean_soundscape_df = pd.read_csv("/kaggle/input/birdcleft-clean-and-vad-filtered-data/clean_soundscapes_chunks_10sec_vad_filtered.csv")

print(f"âœ… Loaded soundscape_chunked_df: {soundscape_chunked_df.shape}")
print(f"âœ… Loaded clean_soundscape_df: {clean_soundscape_df.shape}")


# Load the master label list and set NUM_CLASSES
with open("/kaggle/input/birdcleft-clean-and-vad-filtered-data/master_label_list.pkl", "rb") as f:
    master_labels = pickle.load(f)
NUM_CLASSES = len(master_labels)  # should be 206


print(f"total number of labels in full data : {len(master_labels)}")  # Should print 206



def validate_df(df, name):
    print(f"\nğŸ”� Validating: {name}")
    print("-" * 50)
    
    # Check for NaNs
    nan_summary = df.isna().sum()
    if nan_summary.sum() == 0:
        print("âœ… No NaNs found.")
    else:
        print("âš ï¸� NaNs found:")
        display(nan_summary[nan_summary > 0])
    
    # Show data types
    print("\nğŸ“Š Data Info:")
    display(df.info())
    
    # Sample rows
    print("\nğŸ§¾ Sample Rows:")
    display(df.sample(3, random_state=42))
    
    # Check essential columns (just an example set â€” adjust as needed)
    expected_cols = ['chunk_id', 'filepath', 'start_sample', 'end_sample']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"â�Œ Missing essential columns: {missing_cols}")
    else:
        print("âœ… All essential columns are present.")
        
    #unique filename
    print(f"Unique audio files in df: {df['filename'].nunique()}")
    print(f"Average duration in df: {df['duration'].mean()}")
    
    # Sanity check: duration and sample range
    if 'duration' in df.columns and 'start_sample' in df.columns and 'end_sample' in df.columns:
        duration_errors = df[df['end_sample'] <= df['start_sample']]
        if len(duration_errors) > 0:
            print(f"â�Œ {len(duration_errors)} rows have invalid sample ranges!")
        else:
            print("âœ… Sample ranges are valid.")

# Run validation
validate_df(working_df, "working_df (raw/rating filtered)")
validate_df(clean_train_df, "clean_df (VAD filtered)")
validate_df(clean_soundscape_df , "clean_soundscape_df(VAD FILTERED)")


print("Total rows:", len(clean_train_df))
print("Unique samplenames:", clean_train_df['samplename'].nunique())
print(clean_train_df['samplename'].value_counts().head(10))



print("Total rows:", len(clean_soundscape_df))
print("Unique samplenames:", clean_soundscape_df['samplename'].nunique())
print(clean_soundscape_df['samplename'].value_counts().head(10))



# Function jo audio ko mel spectrogram me convert karta hai
# def audio2melspec(audio_data, config):
#     # Agar NaN ho to usko remove karte hain
#     if np.isnan(audio_data).any():
#         mean_val = np.nanmean(audio_data)
#         audio_data = np.nan_to_num(audio_data, nan=mean_val)

#     # Mel spectrogram
#     mel = librosa.feature.melspectrogram(
#         y=audio_data,
#         sr=config.FS,
#         n_fft=config.N_FFT,
#         hop_length=config.HOP_LENGTH,
#         n_mels=config.N_MELS,
#         fmin=config.FMIN,
#         fmax=config.FMAX,
#         power=2.0
#     )

#     # Usko decibels me convert karna
#     mel_db = librosa.power_to_db(mel, ref=np.max)

#     # Normalize karna
#     mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

#     return mel_db


# # Function to process audio dataset
# def process_audio(df, label, config):
#     print(f"Processing {label} audio data...")
#     start_time = time.time()

#     bird_data = {}
#     errors = []
#     target_len = int(config.TARGET_DURATION * config.FS)

#     for k, row in tqdm(df.iterrows(), total=len(df)):
#         try:
#             audio, _ = librosa.load(row.filepath, sr=config.FS)
#             audio = prepare_audio(audio, target_len)
#             mel = audio2melspec(audio, config)

#             if mel.shape != config.TARGET_SHAPE:
#                 mel = cv2.resize(mel, config.TARGET_SHAPE)

#             bird_data[row.samplename] = mel.astype(np.float32)

#         except Exception as e:
#             errors.append((row.filepath, str(e)))

#     end_time = time.time()

#     print(f"\nFinished processing '{label}' in {end_time - start_time:.1f} seconds")
#     print(f" Successfully processed: {len(bird_data)} files of {label}")
#     print(f"Failed: {len(errors)} files")

#     np.savez_compressed(f'{label}.npz', **bird_data)
#     print(f"Saved data as '{label}.npz'\n")

#     return bird_data, errors



# ========== MEL FUNCTION ===================
# config object has these:
mel_transform = T.MelSpectrogram(
    sample_rate=config.FS,
    n_fft=config.N_FFT,
    hop_length=config.HOP_LENGTH,
    n_mels=config.N_MELS,
    f_min=config.FMIN,
    f_max=config.FMAX,
    power=2.0,
).to(device)  # ğŸ‘ˆ GPU-par move

# Your new GPU-compatible function
def audio2melspec_gpu(audio_data):
    # Convert numpy to torch tensor and add batch/channel dims
    if np.isnan(audio_data).any():
        mean_val = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_val)

    waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).to(device)  # shape: (1, samples)

    # Apply mel spectrogram
    mel = mel_transform(waveform)

    # Convert to decibel scale
    mel_db = F.amplitude_to_DB(mel, multiplier=10.0, amin=1e-10, db_multiplier=0.0)

    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    return mel_db.squeeze(0).cpu().numpy()  # shape: (n_mels, time)

def is_valid_5sec_audio(audio, sample_rate=32000, duration_sec=5):
    expected_len = sample_rate * duration_sec
    return len(audio) == expected_len

    
# ====== PROCESS FUNCTION ===================


def prepare_audio(audio, target_len):
    """
    Ensure audio is exactly `target_len` samples.
    - If too short: zero-pad
    - If too long: center-trim
    """
    current_len = len(audio)

    if current_len < target_len:
        # Zero pad (centered)
        pad_left = (target_len - current_len) // 2
        pad_right = target_len - current_len - pad_left
        audio = np.pad(audio, (pad_left, pad_right), mode='constant')

    elif current_len > target_len:
        # Center trim
        start = (current_len - target_len) // 2
        audio = audio[start: start + target_len]

    return audio





print(config.FS, config.TARGET_DURATION, config.TARGET_SHAPE)



def process_audio_with_batches(df, label, config, batch_size=1000):
    print(f"ğŸ”„ Processing {label} audio data with batch size {batch_size}...\n")
    start_time = time.time()

    bird_data = {}
    errors = []
    target_len = int(config.TARGET_DURATION * config.FS)
    batch_num = 0
    success_count = 0

    for i, row in tqdm(df.iterrows(), total=len(df), leave=True, dynamic_ncols=True):
        try:
            # Load and resample if needed
            audio, sr = torchaudio.load(row.filepath)
            audio = audio.mean(dim=0).numpy()  # mono

            if sr != config.FS:
                audio = torchaudio.functional.resample(
                    torch.tensor(audio), orig_freq=sr, new_freq=config.FS
                ).numpy()

            # Preprocess and convert to mel
            audio = prepare_audio(audio, target_len)
            mel = audio2melspec_gpu(audio)
            mel = cv2.resize(mel, config.MEL_SHAPE[::-1])
            mel = cv2.cvtColor((mel * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            mel = mel.transpose(2, 0, 1)

            # Save with unique key
            bird_data[row.chunk_id] = mel.astype(np.float32)
            success_count += 1

            # Save batch
            if (i + 1) % batch_size == 0:
                batch_filename = f'{label}_batch_{batch_num}.npz'
                np.savez_compressed(batch_filename, **bird_data)
                #print(f"ğŸ’¾ Batch {batch_num} saved with {len(bird_data)} samples âœ…")
                tqdm.write(f"ğŸ’¾ Batch {batch_num} saved with {len(bird_data)} samples âœ…")
                bird_data.clear()
                batch_num += 1

        except Exception as e:
            #print(f"â�Œ Error on {row.chunk_id}: {e}")
            tqdm.write(f"â�Œ Error on {row.chunk_id}: {e}")
            errors.append((row.chunk_id, row.filepath, str(e)))

    # Save remaining samples
    if bird_data:
        batch_filename = f'{label}_batch_{batch_num}.npz'
        np.savez_compressed(batch_filename, **bird_data)
        print(f"ğŸ’¾ Final batch {batch_num} saved with {len(bird_data)} samples âœ…")

    end_time = time.time()
    print(f"\nâœ… Finished processing '{label}' in {end_time - start_time:.1f} seconds")
    print(f"ğŸŸ¢ Total successful: {success_count}")
    print(f"ğŸ”´ Total failed: {len(errors)}")

    return errors



if torch.cuda.is_available():
    torch.cuda.empty_cache()


errors = process_audio_with_batches(clean_soundscape_df, label='clean_soundscape_mel_specs', config=config, batch_size=1000)



if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ========== APPLY TO DFs ==========
#clean_mel_specs, errors = process_audio(df=clean_train_df, label="clean_train_audio_mel_specs", config=config)

errors = process_audio_with_batches(clean_train_df, label='clean_train_mel_specs', config=config, batch_size=1000)



def process_full_audio_with_batches(df, label, config, batch_size=1000):
    print(f"ğŸ”„ Processing {label} audio data with batch size {batch_size}...")
    start_time = time.time()

    bird_data = {}
    errors = []
    target_len = int(config.TARGET_DURATION * config.FS)
    batch_num = 0
    success_count = 0

    for i, row in tqdm(df.iterrows(), total=len(df), leave=True, dynamic_ncols=True):
        try:
            # Load audio using torchaudio (GPU-friendly)
            audio, sr = torchaudio.load(row.filepath)
            audio = audio.mean(dim=0).numpy()  # convert to mono

            # Resample if necessary
            if sr != config.FS:
                audio = torchaudio.functional.resample(
                    torch.tensor(audio), orig_freq=sr, new_freq=config.FS
                ).numpy()

            # Pad/trim to fixed length
            audio = prepare_audio(audio, target_len)

            # Convert to mel spectrogram (GPU)
            mel = audio2melspec_gpu(audio)

            # Resize and convert to RGB
            mel = cv2.resize(mel, config.MEL_SHAPE[::-1])
            mel = cv2.cvtColor((mel * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            mel = mel.transpose(2, 0, 1)  # (3, H, W)

            # Use filename as key
            bird_data[row.filename] = mel.astype(np.float32)
            success_count += 1

            # Save in batches
            if (i + 1) % batch_size == 0:
                batch_filename = f'{label}_batch_{batch_num}.npz'
                np.savez_compressed(batch_filename, **bird_data)
                tqdm.write(f"ğŸ’¾ Batch {batch_num} saved with {len(bird_data)} samples âœ…")
                bird_data.clear()
                batch_num += 1

        except Exception as e:
            tqdm.write(f"â�Œ Error on {row.filename}: {e}")
            errors.append((row.filename, row.filepath, str(e)))

    # Save any remaining data
    if bird_data:
        batch_filename = f'{label}_batch_{batch_num}.npz'
        np.savez_compressed(batch_filename, **bird_data)
        print(f"ğŸ’¾ Final batch {batch_num} saved with {len(bird_data)} samples âœ…")

    end_time = time.time()
    print(f"\nâœ… Finished processing '{label}' in {end_time - start_time:.1f} seconds")
    print(f"ğŸŸ¢ Total successful: {success_count}")
    print(f"ğŸ”´ Total failed: {len(errors)}")

    return errors



errors = process_full_audio_with_batches(working_df, label='working_data_mel_specs', config=config, batch_size=1000)




