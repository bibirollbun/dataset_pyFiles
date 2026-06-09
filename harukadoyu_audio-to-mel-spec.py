import os
import logging
import random
import gc
import glob
import time
import cv2
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import timm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


class CFG:
    
    seed = 42
    debug = False
    apex = False
    print_freq = 100
    num_workers = 2
    
    OUTPUT_DIR = '/kaggle/working/'

    train_datadir = '/kaggle/input/birdclef-2025/train_audio'
    train_csv = '/kaggle/input/birdclef-2025/train.csv'
    train_soundscapes = '/kaggle/input/birdclef-2025/train_soundscapes'
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'

    spectrogram_npy = '/kaggle/input/birdclef25-mel-spectrograms/birdclef2025_melspec_5sec_256_256.npy'


    human_voice_data = '/kaggle/input/bc25-separation-voice-from-data-by-silero-vad/train_voice_data.pkl'
    
    model_name = 'efficientnet_b0'  
    pretrained = True
    in_channels = 1

    FS = 32000
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (256, 256)

    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000  
    #N_FFT = 2048
    #HOP_LENGTH = 128
    #N_MELS = 512
    #FMIN = 50
    #FMAX = 16000
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10  
    batch_size = 64
    criterion = 'BCEWithLogitsLoss'

    n_fold = 5
    selected_folds = [0, 1, 2, 3, 4]   

    optimizer = 'AdamW'
    lr = 5e-4 
    weight_decay = 1e-5
  
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = epochs

    aug_prob = 0.5  
    mixup_alpha = 0.5  

    crop_mode = "random"
    human_threshold = 0.5
    
    def update_debug_settings(self):
        if self.debug:
            self.epochs = 2
            self.selected_folds = [0]

cfg = CFG()


def prepare_working_df(cfg):
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    species_class_map = dict(zip(taxonomy_df['primary_label'], taxonomy_df['class_name']))
    train_df = pd.read_csv(cfg.train_csv)
    
    label_list = sorted(train_df['primary_label'].unique())
    label_id_list = list(range(len(label_list)))
    label2id = dict(zip(label_list, label_id_list))
    id2label = dict(zip(label_id_list, label_list))

    working_df = train_df[['primary_label', 'rating', 'filename']].copy()
    working_df['target'] = working_df.primary_label.map(label2id)
    working_df['filepath'] = cfg.train_datadir + '/' + working_df.filename
    working_df['samplename'] = working_df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
    working_df['class'] = working_df.primary_label.map(lambda x: species_class_map.get(x, 'Unknown'))
    
    return working_df


def audio2melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram"""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm


def remove_human_voice(audio_data, voice_times, sr):
    """Removes human voice segments from the audio data safely."""
    keep_segments = []
    last_end = 0
    for segment in voice_times:
        start = round(segment['start'] * sr)
        end = round(segment['end'] * sr)
        # Keep segment before this human voice part
        keep_segments.append(audio_data[last_end:start])
        last_end = end
    # Add the last remaining part after final voice segment
    keep_segments.append(audio_data[last_end:])
    return np.concatenate(keep_segments)

def discard_audio_with_human(audio_path, human_voice_data, start_idx,
                             end_idx, sr, threshold=0.5):
    """
    Returns True if human voice portion in the given segment exceeds the threshold ratio.
    """
    if audio_path not in human_voice_data:
        return False

    voice_segments = human_voice_data[audio_path]
    voice_duration = 0.0
    segment_length = end_idx - start_idx

    for seg in voice_segments:
        voice_start = seg['start'] * sr
        voice_end = seg['end'] * sr
        # Check overlap with segment
        overlap_start = max(start_idx, voice_start)
        overlap_end = min(end_idx, voice_end)
        if overlap_end > overlap_start:
            voice_duration += (overlap_end - overlap_start)

    voice_ratio = voice_duration / segment_length
    return voice_ratio > threshold

def find_all_rms_segments(audio_data, target_samples, step_size):
    """
    Slide over audio and return list of (start_idx, rms) sorted by rms descending
    """
    rms_segments = []
    for start_idx in range(0, len(audio_data) - target_samples + 1, step_size):
        segment = audio_data[start_idx : start_idx + target_samples]
        rms = np.sqrt(np.mean(segment ** 2))
        rms_segments.append((start_idx, rms))
    # Sort by RMS descending
    rms_segments.sort(key=lambda x: x[1], reverse=True)
    return rms_segments

def process_audio_file(audio_path, human_voice_data, cfg):
    """Process a single audio file to get the mel spectrogram"""
    try:
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
        target_samples = int(cfg.TARGET_DURATION * cfg.FS)

        if len(audio_data) < target_samples:
            n_copy = math.ceil(target_samples / len(audio_data))
            if n_copy > 1:
                audio_data = np.concatenate([audio_data] * n_copy)

        # Select segment based on mode
        if cfg.crop_mode == "rms":
            step_size = int(cfg.FS)
            rms_segments = find_all_rms_segments(audio_data, target_samples, step_size)

            start_idx = None
            for candidate_start, _ in rms_segments:
                end_idx = candidate_start + target_samples
                if human_voice_data is not None:
                    if discard_audio_with_human(audio_path, human_voice_data, candidate_start, end_idx, cfg.FS, cfg.human_threshold):
                        continue  # discard, try next best segment
                start_idx = candidate_start
                break
            if start_idx is None:
                # No segment without too much human voice found
                return None
        elif cfg.crop_mode == "random":
            max_offset = len(audio_data) - target_samples
            start_idx = random.randint(0, max(0, max_offset))
            end_idx = start_idx + target_samples
            if human_voice_data is not None:
                if discard_audio_with_human(audio_path, human_voice_data, start_idx, end_idx, cfg.FS,
                                           cfg.human_threshold):
                    return None
        else:  # mode == "first"
            start_idx = 0
            end_idx = start_idx + target_samples
            if human_voice_data is not None:
                if discard_audio_with_human(audio_path, human_voice_data, start_idx, end_idx, cfg.FS,
                                           cfg.human_threshold):
                    return None

        cropped_audio = audio_data[start_idx:end_idx]

        if len(cropped_audio) < target_samples:
            cropped_audio = np.pad(cropped_audio, 
                                 (0, target_samples - len(cropped_audio)), 
                                 mode='constant')

        mel_spec = audio2melspec(cropped_audio, cfg)
        
        if mel_spec.shape != cfg.TARGET_SHAPE:
            mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

        return mel_spec.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def save_discarded_filenames(filepaths):
    with open("discarded_files.txt", "w") as f:
        for filepath in filepaths:
            f.write(filepath + "\n")
    print(f"Saved {len(filepaths)} discarded file paths to 'discarded_files.txt'")

def generate_spectrograms(cfg):
    """Generate spectrograms from audio files"""
    print("Generating mel spectrograms from audio files...")
    start_time = time.time()

    all_bird_data = {}
    discarded_files = []
    errors = []

    df = prepare_working_df(cfg)
    human_voice_data = pd.read_pickle(cfg.human_voice_data)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        if cfg.debug and i >= 100:
            break
        
        try:
            samplename = row['samplename']
            filepath = row['filepath']
            
            mel_spec = process_audio_file(filepath, human_voice_data, cfg)
            
            if mel_spec is not None:
                all_bird_data[samplename] = mel_spec
            else:
                discarded_files.append(filepath)

        except Exception as e:
            print(f"Error processing {row.filepath}: {e}")
            errors.append((row.filepath, str(e)))

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    print(f"Successfully processed {len(all_bird_data)} files out of {len(df)}")
    print(f"Failed to process {len(errors)} files")

    save_discarded_filenames(discarded_files)

    return all_bird_data


cfg = CFG()
cfg.crop_mode = "rms"
bird_data = generate_spectrograms(cfg)
np.save("birdclef2025_melspec_5sec_256_256_nohuman_rms.npy", bird_data)


def save_pseudo_filenames(filepaths):
    with open("pseudo_files.txt", "w") as f:
        for filepath in filepaths:
            f.write(filepath + "\n")
    print(f"Saved {len(filepaths)} file paths to 'pseudo_files.txt'")

def generate_spectrograms_pseudo(cfg):
    """Generate spectrograms from pseudo audio files"""
    print("Generating mel spectrograms from pseudo audio files...")
    start_time = time.time()

    all_bird_data = {}
    pseudo_files = []
    errors = []

    audio_files = glob.glob(os.path.join(cfg.train_soundscapes, '**', '*.ogg'), recursive=True)

    for i, filepath in tqdm(enumerate(audio_files), total=len(audio_files)):
        if cfg.debug and i >= 100:
            break
        
        try:
            mel_spec = process_audio_file(filepath, None, cfg)
            
            if mel_spec is not None:
                all_bird_data[filepath] = mel_spec
                pseudo_files.append(filepath)

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            errors.append((filepath, str(e)))

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    print(f"Successfully processed {len(all_bird_data)} files out of {len(audio_files)}")
    print(f"Failed to process {len(errors)} files")

    save_pseudo_filenames(pseudo_files)
    
    return all_bird_data

#cfg = CFG()
#cfg.crop_mode = "random"
#pseudo_data = generate_spectrograms_pseudo(cfg)
#np.save("birdclef2025_melspec_5sec_256_256_pseudo.npy", pseudo_data)

