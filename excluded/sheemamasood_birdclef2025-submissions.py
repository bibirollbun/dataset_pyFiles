# Standard Libraries
import os
import gc
import time
import math
import random
import warnings
import logging
from pathlib import Path
from glob import glob
from typing import Union
import copy

# Data Handling
import numpy as np
import pandas as pd
import joblib
import pickle
import collections

# Audio Processing
import librosa
import librosa.display
import torchaudio

# Machine Learning & PyTorch
import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm

# Visualization
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Progress Bars
from tqdm import tqdm
from tqdm.notebook import tqdm as notebook_tqdm

# Logging and Warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# Check versions
print(f"librosa version : {librosa.__version__}")
print(f"librosa files : {librosa.__file__}")
print("âœ… All libraries imported in the environment.")

import sys
sys.path.append('/kaggle/input/birdcleft-clean-and-vad-filtered-data')

from utils_vad import get_speech_timestamps


# ===================== CONFIG =====================
class Config:
    # General
    seed = 42
    print_freq = 100
    num_workers = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Mel spectrogram parameters (for converting audio to image)
    N_FFT = 1024       # FFT window size
    HOP_LENGTH = 512   # Step size for each frame
    FS = 32000
    FMIN = 50          # Minimum Mel frequency
    FMAX = 14000       # Maximum Mel frequency
    N_MELS = 128
     
    SR = 32000
    TARGET_DURATION= 5
    train_duration = 10
    # Mel/Image
    MEL_SHAPE = (3, 256, 256)
    TARGET_SHAPE = (3, 256, 256)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths (change these if needed)
    test_soundscapes = "/kaggle/input/birdclef-2025/test_soundscapes"
    submission_csv = "/kaggle/input/birdclef-2025/sample_submission.csv"
    model_path = "/kaggle/input/birdcleft-clean-and-vad-filtered-data/best_model_187.pth"
    backbone_weights = "/kaggle/input/birdcleft-clean-and-vad-filtered-data/seresnext_backbone_weights.pth"
    master_labels = "/kaggle/input/birdcleft-clean-and-vad-filtered-data/valid_labels.pkl"
    vad_model_path = "/kaggle/input/birdcleft-clean-and-vad-filtered-data/silero_vad.jit"
    
config = Config()

device = torch.device(config.device)

species_ids = pd.read_csv(config.submission_csv).columns[1:].tolist()

print("test_soundscapes path:", config.test_soundscapes)


# ===================== DATA UTILS =====================
def load_species_ids(submission_csv):
    """Load 206 submission class names from sample_submission.csv (skip 'row_id')."""
    return pd.read_csv(submission_csv).columns[1:].tolist()

def load_master_labels(master_labels):
    """Load 187 model class labels (output order) from a pickle file."""
    with open(master_labels, 'rb') as f:
        return pickle.load(f)

def load_vad_model(jit_path, device='cpu'):
    """
    Load Silero VAD JIT model from file.
    Args:
        jit_path (str): Path to the .jit model file.
        device (str or torch.device): 'cpu' or 'cuda'
    Returns:
        torch.jit.ScriptModule: Loaded model in eval mode.
    """
    model = torch.jit.load(jit_path, map_location=device)
    model.eval()
    return model

# ====== MEL SPECTROGRAM TRANSFORM ======
def get_mel_transform(cfg):
    return T.MelSpectrogram(
        sample_rate=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        f_min=cfg.FMIN,
        f_max=cfg.FMAX,
        power=2.0,
    ).to(cfg.DEVICE)

mel_transform = get_mel_transform(config)

def audio2melspec_gpu(audio_data, cfg, mel_transform):
    if np.isnan(audio_data).any():
        mean_val = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_val)
    waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).to(cfg.DEVICE)
    mel = mel_transform(waveform)
    #mel_db = F.amplitude_to_DB(mel, multiplier=10.0, amin=1e-10, db_multiplier=0.0)
    mel_db = F.amplitude_to_DB(mel, multiplier=10.0, amin=1e-10, db_multiplier=0.0)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.squeeze(0).cpu().numpy()

def prepare_audio(audio, target_len):
    current_len = len(audio)
    if current_len < target_len:
        pad_left = (target_len - current_len) // 2
        pad_right = target_len - current_len - pad_left
        audio = np.pad(audio, (pad_left, pad_right), mode='constant')
    elif current_len > target_len:
        start = (current_len - target_len) // 2
        audio = audio[start: start + target_len]
    return audio

# ======  GET TEST FILES ======
def get_test_files(test_dir, fallback_dir=None, fallback_n=10):
    test_files = list(Path(test_dir).glob('*.ogg'))
    if len(test_files) == 0 and fallback_dir:
        test_files = sorted(list(Path(fallback_dir).glob('*.ogg')))[:fallback_n]
    return test_files

# ======  CHUNKING ======
def get_chunks_for_files(soundscape_files, sample_rate=32000, chunk_len_sec=5):
    chunk_records = []
    for file_path in tqdm(soundscape_files, desc="Processing test soundscapes"):
        file = Path(file_path).name
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
                    'filepath': str(file_path),
                    'samplename': samplename,
                    'start_sec': i * chunk_len_sec,
                    'end_sec': (i + 1) * chunk_len_sec,
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'duration': duration_sec,
                    'row_id': f"{samplename}_{(i + 1) * chunk_len_sec}",  
                })
        except Exception as e:
            print(f"â�Œ Failed to process {file}: {e}")
    chunked_df = pd.DataFrame(chunk_records)
    return chunked_df





# ====== 3. DATASET CLASS ======
class BirdTestMelDataset(Dataset):
    def __init__(self, chunked_df, cfg, mel_transform):
        self.df = chunked_df.reset_index(drop=True)
        self.cfg = cfg
        self.mel_transform = mel_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio, sr = torchaudio.load(row.filepath)
        audio = audio.mean(dim=0).numpy()
        if sr != self.cfg.FS:
            audio = torchaudio.functional.resample(
                torch.tensor(audio), orig_freq=sr, new_freq=self.cfg.FS
            ).numpy()
        chunk_audio = audio[row.start_sample:row.end_sample]
        chunk_audio = prepare_audio(chunk_audio, int(self.cfg.TARGET_DURATION * self.cfg.FS))
        mel = audio2melspec_gpu(chunk_audio, self.cfg, self.mel_transform)
        mel = cv2.resize(mel, self.cfg.MEL_SHAPE[1:])
        mel = cv2.cvtColor((mel * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        mel = mel.transpose(2, 0, 1)
        mel = mel.astype(np.float32) 
        mel_tensor = torch.tensor(mel, dtype=torch.float32)

        return mel_tensor, row['row_id']

def vad_and_silence_filter(
    chunked_df,
    vad_model,
    get_speech_timestamps,
    device,
    silence_threshold=0.01,
    sampling_rate_col='sample_rate',
    save_csv_path=None,
    description="VAD and silence filter"
):
    """
    Remove silent and human-voice chunks from the DataFrame using VAD and amplitude threshold.

    Args:
        chunked_df (pd.DataFrame): DataFrame with columns ['filepath', 'start_sample', 'end_sample', ...].
        vad_model: Trained VAD PyTorch model.
        get_speech_timestamps: Silero function for speech detection.
        device: torch.device
        silence_threshold (float): Mean amplitude below which chunk is considered silent.
        sampling_rate_col (str): Column name for sample rate (if per-row), else use torchaudio.load default.
        save_csv_path (str, optional): If provided, saves cleaned DataFrame to this path.
        description (str): For logging.

    Returns:
        pd.DataFrame: Filtered DataFrame with only clean, non-silent chunks.
    """
    vad_model.eval()
    clean_chunks = []

    print(f"\n== Cleaning chunks: Removing silent & human-voice segments ({description}) ==")

    for idx, row in tqdm(chunked_df.iterrows(), total=len(chunked_df)):
        # Load audio
        waveform, sr = torchaudio.load(row['filepath'])
        # Use per-row sample_rate if available
        if sampling_rate_col in row:
            sr = int(row[sampling_rate_col])

        start_sample = int(row['start_sample'])
        end_sample = int(row['end_sample'])
        chunk_audio = waveform[0, start_sample:end_sample].to(device)

        # Silence check
        if chunk_audio.abs().mean().item() < silence_threshold:
            continue

        # VAD (human voice) check
        speech_timestamps = get_speech_timestamps(chunk_audio, vad_model, sampling_rate=sr)
        if len(speech_timestamps) == 0:
            clean_chunks.append(row)

    clean_df = pd.DataFrame(clean_chunks)

    if save_csv_path is not None:
        clean_df.to_csv(save_csv_path, index=False)
        print(f"ğŸ“� Saved to {save_csv_path}")

    print(f"âœ… Clean and non-silent chunks count: {len(clean_df)}")

    return clean_df



def run_test_pipeline(
    test_dir,
    fallback_dir=None,
    chunk_len_sec=5,
    batch_size=4,
    fallback_n=10,
    n_print=2,
    do_cleaning=True,
    vad_model=None,
    get_speech_timestamps=None,
    silence_threshold=0.01,
    cleaning_desc="VAD filtering"
):
    print("== Getting test files ==")
    test_files = get_test_files(test_dir, fallback_dir, fallback_n)
    print(f"Found {len(test_files)} test files.")
    if len(test_files) == 0:
        raise RuntimeError("No test files found.")

    print("\n== Chunking test files ==")
    chunked_df = get_chunks_for_files(test_files, sample_rate=config.FS, chunk_len_sec=chunk_len_sec)
    display(chunked_df.head())

    # === CLEANING STEP ===
    clean_df = chunked_df
    if do_cleaning:
        if vad_model is None or get_speech_timestamps is None:
            raise ValueError("vad_model and get_speech_timestamps must be provided for cleaning.")
        print("\n== Cleaning chunks: Removing silent & human-voice segments ==")
        clean_df = vad_and_silence_filter(
            chunked_df,
            vad_model,
            get_speech_timestamps,
            device=device,
            silence_threshold=silence_threshold,
            description=cleaning_desc
        )

    print("\n== Initializing Dataset & Loader ==")
    dataset = BirdTestMelDataset(clean_df, config, mel_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"\n== Dataset shape check ==")
    for i in range(min(n_print, len(dataset))):
        mel, row_id = dataset[i]
        print(f"RowID: {row_id}")
        print(f"  MEL shape: {mel.shape}")
        print(f"  MEL dtype: {mel.dtype}")
        print(f"  MEL min/max: {mel.min()} / {mel.max()}")
        print(f"  MEL first 5 vals: {mel.flatten()[:5]}")
        print("-" * 40)

    print("\n== Loader batch check ==")
    for batch in loader:
        mels, row_ids = batch
        print(f"Batch MELS shape: {mels.shape}")  # (B, 3, 256, 256)
        print(f"Batch MELS dtype: {mels.dtype}")
        print(f"Batch first row_id: {row_ids[0]}")
        print(f"Batch MELS min/max: {mels.min().item()} / {mels.max().item()}")
        break  # just first batch

    print("All checks done.")
    # RETURN BOTH dfs!
    return chunked_df, clean_df, dataset, loader

# 3. Model load karo (bahar, function ke andar nahi)
vad_model= load_vad_model('/kaggle/input/birdcleft-clean-and-vad-filtered-data/silero_vad.jit')  

##============TEST RUN=====================
# Make sure vad_model and get_speech_timestamps are defined, then pass to your pipeline:

# 4. Pipeline run karo
chunked_df ,clean_df, test_dataset, test_loader = run_test_pipeline(
    test_dir="/kaggle/input/birdclef-2025/test_soundscapes",
    fallback_dir="/kaggle/input/birdclef-2025/train_soundscapes",
    chunk_len_sec=5,
    batch_size=4,
    fallback_n=10,
    n_print=2,
    do_cleaning=True,
    vad_model=vad_model,  # Yahan pass karo
    get_speech_timestamps=get_speech_timestamps,
    silence_threshold=0.01,
    cleaning_desc="Filtering test soundscapes"
)


def plot_multiple_mels(dataset, indices, title_prefix='Sample Mel Spectrogram'):
    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))  # 1 row, n columns

    for i, idx in enumerate(indices):
        mel_tensor, _ = dataset[idx]
        mel = mel_tensor[0].numpy()

        ax = axes[i] if n > 1 else axes
        im = ax.imshow(mel, aspect='auto', origin='lower')
        ax.set_title(f"{title_prefix} #{idx}")
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Mel Bands")
        ax.label_outer()  # Only show outer labels for clean look

    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    plt.tight_layout()
    plt.show()

    # Example usage:
print("Test Samples:")
plot_multiple_mels(test_dataset, indices=list(range(5)), title_prefix='test Sample Mel')


#=========INFERENCE CODE=============
def infer_model(model, 
                dataloader, 
                device, 
                master_labels, 
                save_preds_csv=None,
                num_batches=None):
    """
    Run inference on test dataloader, return DataFrame with predictions.
    Assumes dataloader yields (mel_tensor, row_id).
    """
    model.eval()
    all_row_ids = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
            # Unpack batch: (inputs, row_ids)
            inputs, row_ids = batch
            inputs = inputs.to(device)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()  # shape [B, num_classes]

            all_row_ids.extend(list(row_ids))
            all_probs.append(probs)

            if num_batches and batch_idx + 1 >= num_batches:
                break

    all_probs = np.vstack(all_probs)  # shape: [num_samples, num_classes]
    # Make sure DataFrame columns use master_labels:
    df = pd.DataFrame(all_probs, columns=master_labels)
    df.insert(0, "row_id", all_row_ids)
    

    if save_preds_csv:
        df.to_csv(save_preds_csv, index=False)
        print(f"Saved predictions to: {save_preds_csv}")

    # Optional: Print top-5 for first few samples
    print("\nSample predictions:")
    for i in range(min(3, len(df))):
        top5_idx = df.iloc[i, 1:].values.argsort()[-5:][::-1]
        top5_probs = df.iloc[i, 1:].values[top5_idx]
        print(f"\nRow: {df.loc[i, 'row_id']}")
        for rank, (cls_idx, prob) in enumerate(zip(top5_idx, top5_probs), 1):
            print(f"  {rank}. {master_labels[cls_idx]} ({prob:.3f})")

    return df


# ===================== MODEL =====================
class ImprovedBirdCLEFModel(nn.Module):
    def __init__(self, num_classes=187, backbone_weights=None, device=device):
        super().__init__()
        self.backbone = timm.create_model(
            "seresnext26t_32x4d",
            pretrained=False,
            in_chans=3,
            num_classes=0
        )
        if backbone_weights:
            state_dict = torch.load(backbone_weights, map_location=device)
            self.backbone.load_state_dict(state_dict, strict=False)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

def load_model(model_path, device, num_classes, backbone_weights):
    model = ImprovedBirdCLEFModel(num_classes=num_classes, backbone_weights=backbone_weights, device=device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model



#=========================SUBMISSIONS CODE================
def create_submission(row_ids, predictions, master_labels, species_ids, submission_csv):
    """
    row_ids: list of row_id strings
    predictions: numpy array shape (N, 187) [model output]
    master_labels: list of 187 model class labels (your cleaned data)
    species_ids: list of 206 submission class labels (from sample_submission.csv)
    sample_sub_csv: path to sample_submission.csv
    """
    print("Creating submission dataframe...")

    # Mapping: for each submission class, find its index in model output, or -1 if missing
    model2sub_idx = [master_labels.index(lbl) if lbl in master_labels else -1 for lbl in species_ids]

    sub_dict = {'row_id': row_ids}
    for i, lbl in enumerate(species_ids):
        idx = model2sub_idx[i]
        if idx != -1:
            sub_dict[lbl] = predictions[:, idx]
        else:
            sub_dict[lbl] = np.zeros(len(row_ids), dtype=np.float32)

    sub_df = pd.DataFrame(sub_dict)

    # Ensure all columns (same order) as sample_submission
    sample_sub = pd.read_csv(config.submission_csv)
    for col in sample_sub.columns:
        if col not in sub_df.columns:
            sub_df[col] = 0.0
    sub_df = sub_df[sample_sub.columns]
    return sub_df

def smooth_submission(submission_path):
    """
    submission_path: path to CSV file (will overwrite with smoothed)
    """
    print("Smoothing submission predictions...")
    sub = pd.read_csv(submission_path)
    cols = sub.columns[1:]  # all label columns
    groups = sub['row_id'].str.rsplit('_', n=1).str[0]

    for group, idx in sub.groupby(groups).groups.items():
        arr = sub.loc[idx, cols].values
        arr_sm = arr.copy()
        if len(arr) > 1:
            arr_sm[0] = 0.8 * arr[0] + 0.2 * arr[1]
            arr_sm[-1] = 0.8 * arr[-1] + 0.2 * arr[-2]
            for i in range(1, len(arr) - 1):
                arr_sm[i] = 0.2 * arr[i-1] + 0.6 * arr[i] + 0.2 * arr[i+1]
        sub.loc[idx, cols] = arr_sm

    sub.to_csv(submission_path, index=False)
    print(f"Smoothed submission saved to {submission_path}")


device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    # 1. Load VAD model & utility
    vad_model = load_vad_model('/kaggle/input/birdcleft-clean-and-vad-filtered-data/silero_vad.jit')  
    print("VAD loaded")

    # 2. Prepare test chunks & loader (return BOTH dfs!)
    chunked_df, clean_df, test_dataset, test_loader = run_test_pipeline(
        test_dir="/kaggle/input/birdclef-2025/test_soundscapes",
        fallback_dir="/kaggle/input/birdclef-2025/train_soundscapes",
        chunk_len_sec=5,
        batch_size=32,
        fallback_n=10,
        n_print=2,
        do_cleaning=True,
        vad_model=vad_model,
        get_speech_timestamps=get_speech_timestamps,
        silence_threshold=0.01,
        cleaning_desc="Filtering test soundscapes"
    )

    # 3. Load model
    model = load_model(
        model_path=config.model_path,
        device=device,
        num_classes=187,
        backbone_weights=config.backbone_weights
    )
    print("âœ… Model loaded!")

    # 4. Load label lists
    master_labels = load_master_labels(config.master_labels)        # 187 model classes
    species_ids = load_species_ids(config.submission_csv)           # 206 submission columns

    # 5. Run inference on only clean chunks
    test_preds_df = infer_model(
        model=model,
        dataloader=test_loader,
        device=config.DEVICE,
        master_labels=master_labels,
        save_preds_csv="test_preds.csv",
        num_batches=None
    )
    # test_preds_df columns: ["row_id"] + master_labels (187)

    # 6. Zero-fill for all chunks (incl. filtered ones)
    all_row_ids = chunked_df["row_id"].tolist()  # <-- use original chunked_df for all 120 row_ids qk submission me subki prediction chhaye chahy zero hi ho
    full_preds_df = pd.DataFrame({'row_id': all_row_ids}).merge(
        test_preds_df, on='row_id', how='left'
    )
    full_preds_df = full_preds_df.fillna(0.0)

    # 7. Create & save submission
    predictions = full_preds_df[master_labels].values  # shape: (N, 187)
    row_ids = full_preds_df["row_id"].tolist()
    submission_df = create_submission(
        row_ids=row_ids,
        predictions=predictions,
        master_labels=master_labels,
        species_ids=species_ids,
        submission_csv="sample_submission.csv"
    )
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file saved as submission.csv")

    # 8. (Optional) BirdCLEF smoothing
    smooth_submission('submission.csv')
    print("Smoothed submission saved.")


pd.read_csv("submission.csv")










