import os
import cv2
import math
import time
import librosa
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


class Config:
 
    DEBUG_MODE = False
    
    OUTPUT_DIR = '/kaggle/working/'
    DATA_ROOT = '/kaggle/input/birdclef-2025'
    FS = 32000
    
    N_FFT = 2048 
    HOP_LENGTH = 512
    N_CHROMA = 12 
    
    FMIN = 50 
    BINS_PER_OCTAVE = 36 
    
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (256, 256)  
    
    N_MAX = 50 if DEBUG_MODE else None  


config = Config()
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


def audio2chromagram(audio_data):
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    # Generate constant-Q chromagram
    chromagram = librosa.feature.chroma_cqt(
        y=audio_data,
        sr=config.FS,
        hop_length=config.HOP_LENGTH,
        fmin=config.FMIN,
        n_chroma=config.N_CHROMA,
        bins_per_octave=config.BINS_PER_OCTAVE
    )
    
    eps = 1e-6
    chroma_log = np.log(chromagram + eps)
    chroma_norm = (chroma_log - chroma_log.min()) / (chroma_log.max() - chroma_log.min() + eps)
    
    return chroma_norm


print("Starting audio processing for chromagrams...")
print(f"{'DEBUG MODE - Processing only 50 samples' if config.DEBUG_MODE else 'FULL MODE - Processing all samples'}")
start_time = time.time()

all_bird_data = {}
errors = []

for i, row in tqdm(working_df.iterrows(), total=total_samples):
    if config.N_MAX is not None and i >= config.N_MAX:
        break
    
    try:
        audio_data, _ = librosa.load(row.filepath, sr=config.FS)

        target_samples = int(config.TARGET_DURATION * config.FS)

        if len(audio_data) < target_samples:
            n_copy = math.ceil(target_samples / len(audio_data))
            if n_copy > 1:
                audio_data = np.concatenate([audio_data] * n_copy)

        start_idx = max(0, int(len(audio_data) / 2 - target_samples / 2))
        end_idx = min(len(audio_data), start_idx + target_samples)
        center_audio = audio_data[start_idx:end_idx]

        if len(center_audio) < target_samples:
            center_audio = np.pad(center_audio, 
                                 (0, target_samples - len(center_audio)), 
                                 mode='constant')

        chromagram = audio2chromagram(center_audio)

        if chromagram.shape != config.TARGET_SHAPE:
            chromagram = cv2.resize(chromagram, config.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

        all_bird_data[row.samplename] = chromagram.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {row.filepath}: {e}")
        errors.append((row.filepath, str(e)))

end_time = time.time()
print(f"Processing completed in {end_time - start_time:.2f} seconds")
print(f"Successfully processed {len(all_bird_data)} files out of {total_samples} total")
print(f"Failed to process {len(errors)} files")


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
        plt.ylabel('Pitch Class')
        plt.xlabel('Time')
        plt.colorbar(label='Energy')
    
    plt.tight_layout()
    debug_note = "debug_" if config.DEBUG_MODE else ""
    plt.savefig(f'{debug_note}chromagram_examples.png')
    plt.show()



def save_chromagram_data(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    np_save_path = os.path.join(output_dir, 'chromagram_data.npz')
    print(f"Saving chromagram data to {np_save_path}")
    np.savez_compressed(np_save_path, **all_bird_data)
    
    data_list = []
    for samplename, _ in all_bird_data.items():
        row = working_df[working_df['samplename'] == samplename].iloc[0]
        data_list.append({
            'samplename': samplename,
            'primary_label': row['primary_label'],
            'class': row['class'],
            'target': row['target']
        })
    
    data_df = pd.DataFrame(data_list)
    
    csv_path = os.path.join(output_dir, 'chromagram_metadata.csv')
    data_df.to_csv(csv_path, index=False)
    print(f"Saved metadata to {csv_path}")
    
    return np_save_path, csv_path


def load_chromagram_data(np_path):
    data = np.load(np_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


save_path, meta_path = save_chromagram_data(config.OUTPUT_DIR)
print(f"Saved chromagram data to {save_path}")
print(f"Saved metadata to {meta_path}")


def prepare_data_for_modeling():
    print("Example: Preparing chromagram data for model training")
    
    X = np.array([all_bird_data[name] for name in all_bird_data.keys()])
    samplenames = list(all_bird_data.keys())
    labels = [working_df[working_df['samplename'] == name].iloc[0]['target'] for name in samplenames]
    y = np.array(labels)
    
    print(f"Input shape: {X.shape}, Target shape: {y.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    return X, y, samplenames


X, y, samplenames = prepare_data_for_modeling()

chroma_shapes = [all_bird_data[name].shape for name in all_bird_data.keys()]
unique_shapes = set(chroma_shapes)
print(f"Unique chromagram shapes: {unique_shapes}")

chroma_min = min([np.min(all_bird_data[name]) for name in all_bird_data.keys()])
chroma_max = max([np.max(all_bird_data[name]) for name in all_bird_data.keys()])
print(f"Value range: [{chroma_min:.4f}, {chroma_max:.4f}]")

print("Notebook completed!")

