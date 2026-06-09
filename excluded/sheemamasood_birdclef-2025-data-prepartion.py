# Basic utilities
import os
import math
import time
import random
import gc 
import logging
import warnings
from pathlib import Path
import librosa.display
import matplotlib.pyplot as plt
import torch
import torchaudio


# Data handling
import numpy as np
import pandas as pd

# Visualization
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm  

print("all libraries imported in the environment")


# Audio processing
import librosa
print(f"librosa version : {librosa.__version__}")
print(f"librosa files :{librosa.__file__}")

y = np.random.randn(32000)
D = librosa.stft(y, hop_length=512)
D_stretched = librosa.phase_vocoder(D, rate=1.1, hop_length=512)
print(D_stretched.shape)


class Config:
    # Audio settings
    FS = 32000  # Sampling rate (audio)

    # Mel spectrogram parameters (for converting audio to image)
    N_FFT = 1024       # FFT window size
    HOP_LENGTH = 512   # Step size for each frame
    N_MELS = 128       # Number of mel bands
    FMIN = 50          # Minimum Mel frequency
    FMAX = 14000       # Maximum Mel frequency

    # Parameters for audio duration and spectrogram size
    TARGET_DURATION = 10.0  # Length of each audio (in seconds)
    TARGET_SHAPE = (3, 256, 256)  # Size of the spectrogram image

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
working_df = pd.read_csv("/kaggle/input/melspectrogramofbirdclef-2025/working_df.csv")


print(f"âœ… Loaded train_df: {train_df.shape}")
print(f"âœ… Loaded taxonomy_df: {taxonomy_df.shape}")
print(f"âœ… Loaded location_df: {location_df.shape}")
print(f"âœ… Loaded sample_submission: {sample_submission.shape}")
print(f"âœ… Loaded working_df: {working_df.shape}")



print("="*50)
print(f"ğŸ—ºï¸�  Working_df Shape: {working_df.shape}")
print("="*50)

print("="*50)
print("\nğŸ“Š working Data Types:")
print(working_df.info())
print("="*50)


#  Missing Values Check
print("="*50)
print("\nâ�Œ Missing Values in Working Data:")
print(working_df.isnull().sum())
print("="*50)

print("="*50)
print("\nğŸ”¹ Sample Rows from working Data:")
display(working_df.sample(5))
print("="*50)



# Ek bar run karo:
import pickle
all_labels = sorted(working_df['primary_label'].unique())  # full_df = poora data
with open("master_label_list.pkl", "wb") as f:
    pickle.dump(all_labels, f)


def split_audio_into_10sec_chunks(df, sample_rate=config.FS, chunk_len_sec=10):
    chunk_records = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        y, sr = librosa.load(row['filepath'], sr=sample_rate)
        duration_sec = librosa.get_duration(y=y, sr=sr)

        samples_per_chunk = int(chunk_len_sec * sr)
        num_chunks = int(duration_sec // chunk_len_sec)

        for i in range(num_chunks):
            start_sample = i * samples_per_chunk
            end_sample = (i + 1) * samples_per_chunk

            chunk_records.append({
                'chunk_id': f"{row['samplename']}_chunk{i}",
                'primary_label': row['primary_label'],
                'rating': row['rating'],
                'filename': row['filename'],
                'target': row['target'],
                'filepath': row['filepath'],
                'samplename': row['samplename'],
                'class': row['class'],
                'secondary_labels': row['secondary_labels'],
                'secondary_target': row['secondary_target'],
                'duration': row['duration'],
                'start_sec': i * chunk_len_sec,
                'end_sec': (i + 1) * chunk_len_sec,
                'start_sample': start_sample,
                'end_sample': end_sample
            })

    chunked_df = pd.DataFrame(chunk_records)
    return chunked_df




# Split all dataset into 10 sec chunks
chunked_df = split_audio_into_10sec_chunks(working_df)

chunked_df.to_csv("10_sec_chunked_dataset.csv", index=False)
print("CSV saved successfully!")

#Total Chunks Created
print(f"Total 10-sec chunks: {len(chunked_df)}")

#Chunks per Class (Label)
print(chunked_df['primary_label'].value_counts())

#Average Chunks per File
avg_chunks = chunked_df.groupby("filename").size().mean()
print(f"Average chunks per file: {avg_chunks:.2f}")

#Histogram Plot

chunked_df['primary_label'].value_counts().plot(kind='bar', figsize=(12, 5))
plt.title("Number of 10-sec Chunks per Class")
plt.xlabel("Class")
plt.ylabel("Chunk Count")
plt.tight_layout()
plt.savefig("Number of 10-sec Chunks per Class.png")
plt.show()


import os
import librosa
import pandas as pd
from tqdm import tqdm

def split_soundscape_files_into_chunks(folder_path, sample_rate=config.FS, chunk_len_sec=10):
    soundscape_files = [f for f in os.listdir(folder_path) if f.endswith('.ogg')]
    chunk_records = []

    # tqdm for progress bar (same as your function)
    for file in tqdm(soundscape_files, desc="Processing soundscapes"):
        file_path = os.path.join(folder_path, file)
        try:
            y, sr = librosa.load(file_path, sr=sample_rate)
            duration_sec = librosa.get_duration(y=y, sr=sr)
            samples_per_chunk = int(chunk_len_sec * sr)
            num_chunks = int(duration_sec // chunk_len_sec)

            samplename = file.replace(".ogg", "")

            for i in range(num_chunks):
                start_sample = i * samples_per_chunk
                end_sample = (i + 1) * samples_per_chunk

                chunk_records.append({
                    'chunk_id': f"{samplename}_chunk{i}",
                    'filename': file,
                    'filepath': file_path,
                    'samplename': samplename,
                    'start_sec': i * chunk_len_sec,
                    'end_sec': (i + 1) * chunk_len_sec,
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'duration': duration_sec,
                })
        except Exception as e:
            print(f"â�Œ Failed to process {file}: {e}")

    chunked_df = pd.DataFrame(chunk_records)
    return chunked_df



folder_path = "/kaggle/input/birdclef-2025/train_soundscapes"

soundscape_chunked_df = split_soundscape_files_into_chunks(folder_path)
soundscape_chunked_df.to_csv("soundscape_10sec_chunks.csv", index=False)

print("âœ… Soundscape chunk CSV saved successfully!")



soundscape_chunked_df.head(2)


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

vad_model, (get_speech_timestamps, _, read_audio, _, _) = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    trust_repo=True
)
vad_model = vad_model.to(device)


vad_model.eval()

clean_chunks = []

# Define silence threshold (tune this if needed)
SILENCE_THRESHOLD = 0.01  # very low amplitude, tweak if needed

for idx, row in tqdm(chunked_df.iterrows(), total=len(chunked_df)):
    # Load full audio file
    waveform, sr = torchaudio.load(row['filepath'])
    
    # Extract chunk samples from waveform
    start_sample = row['start_sample']
    end_sample = row['end_sample']
    chunk_audio = waveform[0, start_sample:end_sample].to(device)
    
    # --- Silence check ---
    if chunk_audio.abs().mean() < SILENCE_THRESHOLD:
        continue  # skip silent chunk
    
    # --- VAD check (human voice) ---
    speech_timestamps = get_speech_timestamps(chunk_audio, vad_model, sampling_rate=sr)
    
    # If NO human speech detected and not silent, keep chunk
    if len(speech_timestamps) == 0:
        clean_chunks.append(row)

# Create new DataFrame with only clean & non-silent chunks
clean_train_chunked_df = pd.DataFrame(clean_chunks)

# Save to CSV
clean_train_chunked_df.to_csv('train_audio_10sec_chunks_VAD_filtered.csv', index=False)

print(f"Clean and non-silent chunks count: {len(clean_train_chunked_df)}")
print("Saved to clean_chunks_10sec_vad_filtered.csv")



vad_model.eval()

clean_chunks = []

# Define silence threshold (tune this if needed)
SILENCE_THRESHOLD = 0.01  # very low amplitude, tweak if needed

for idx, row in tqdm(soundscape_chunked_df.iterrows(), total=len(soundscape_chunked_df)):
    # Load full audio file
    waveform, sr = torchaudio.load(row['filepath'])

    # Extract chunk samples from waveform
    start_sample = row['start_sample']
    end_sample = row['end_sample']
    chunk_audio = waveform[0, start_sample:end_sample].to(device)

    # --- Silence check ---
    if chunk_audio.abs().mean() < SILENCE_THRESHOLD:
        continue  # skip silent chunk

    # --- VAD check (human voice) ---
    speech_timestamps = get_speech_timestamps(chunk_audio, vad_model, sampling_rate=sr)

    # If NO human speech detected and not silent, keep chunk
    if len(speech_timestamps) == 0:
        clean_chunks.append(row)

# Create new DataFrame with only clean & non-silent chunks
clean_soundscape_chunked_df = pd.DataFrame(clean_chunks)

# Save to CSV
clean_soundscape_chunked_df.to_csv('clean_soundscapes_chunks_10sec_vad_filtered.csv', index=False)

print(f"âœ… Clean and non-silent chunks count: {len(clean_soundscape_chunked_df)}")
print("ğŸ“� Saved to clean_chunks_10sec_vad_filtered_from_soundscapes.csv")



chunked_df.to_csv("train_audio_10sec_chunks.csv", index=False)
clean_train_chunked_df.to_csv("train_audio_10sec_chunks_VAD_filtered.csv", index=False)

soundscape_chunked_df.to_csv("soundscape_10sec_chunks.csv", index=False)
clean_soundscape_chunked_df.to_csv("clean_soundscapes_chunks_10sec_vad_filtered.csv", index=False)
print("All files are saved")




