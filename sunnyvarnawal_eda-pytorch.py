import re
import os
import gc
import timm
import time
import torch
import wandb
import folium
import librosa
import torchaudio
import librosa.display
import concurrent.futures

import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn as nn
import plotly.express as px
from torchvision import models
from IPython.display import Audio
from shapely.geometry import Point
import torchaudio.transforms as AT
from folium.plugins import FastMarkerCluster

import warnings
warnings.filterwarnings('ignore')


class CFG:
    def __init__(self):
        self.test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
        self.submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
        self.taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
        self.model_path = '/kaggle/input/birdclef25-effnetb0-starter-weight'  

        # Audio Parameters
        self.SR = 32000
        self.WINDOW_SIZE = 5
        self.N_FFT = 1000
        self.HOP_LENGTH = 512
        self.N_MELS = 144
        self.FMIN = 50
        self.FMAX = 14000
        self.TARGET_SHAPE = (256, 256)

        # Model Parameters
        self.model_name = 'efficientnet_b0'
        self.in_channels = 1
        self.device = 'cpu'

        # Inference parameters
        self.batch_size = 16
        self.use_tta = False
        self.tta_count = 3
        self.threshold = 0.5

        self.use_specific_folds = False
        self.folds = [0, 1]

        # Debug
        self.debug = False
        self.debug_count = 3

cfg = CFG()


cfg.model_path


print(f'Using device: {cfg.device}')
print(f'Loading taxonomy data...')
taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
species_ids = taxonomy_df['primary_label'].tolist()
num_classes = len(species_ids)
print(f'Number of classes: {num_classes}')


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg

        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=False, # No need for pretrained Weights
            in_chans=cfg.in_channels, 
            drop_rate = 0.0,
            drop_path_rate=0.0        
        )

        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        elif 'resnet' in cfg.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim= backbone_out
        self.classifier = nn.Linear(backbone_out, num_classes)

    def forward(self, x):
        features = self.backbone(x)

        if isinstance(features, dict):
            features = features['features']

        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)

        logits = self.classifier(features)
        return logits


def audio2melspec(audio_data, cfg):
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.SR,
        n_fft = cfg.N_FFT,
        hop_length = cfg.HOP_LENGTH,
        n_mels = cfg.N_MELS,
        fmin= cfg.FMIN,
        fmax= cfg.FMAX,
        power=2.0
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db-mel_spec_db.min())/(mel_spec_db.max()-mel_spec_db.min()+1e-8)
    return mel_spec_norm

def process_audio_segment(audio_data, cfg):
    '''Process audio segment to get mel spectrogram'''
    if len(audio_data) < cfg.SR * cfg.WINDOW_SIZE:
        # PADDING 
        audio_data = np.pad(audio_data, (0, cfg.SR*cfg.WINDOW_SIZE-len(audio_data)), mode='constant')
    mel_spec = audio2melspec(audio_data, cfg)

    # Resize if needed
    if mel_spec.shape != cfg.TARGET_SHAPE:
        mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
    return mel_spec.astype(np.float32)


def apply_tta(spec, tta_idx):
    '''Apply test-time augmentation'''
    if tta_idx == 0:
        return spec
    elif tta_idx == 1:
        return np.flip(spec, axis=1) # horizontal flip => time shift 
    elif tta_idx == 2:
        return np.flip(spec, axis=0)  # vertical flip => frequency shift
    else:
        return spec


def find_model_files(cfg):
    '''Find all .pth model files in the specified model directory'''
    model_files = []

    # convert the model directory to a path object
    model_dir = Path(cfg.model_path)

    # find all .pth files
    for path in model_dir.glob('**/*.pth'):
        model_files.append(str(path))

    return model_files

def load_models(cfg, num_classes):
    '''Load all found model files and prepare them for ensemble'''
    models = []

    # find model files
    model_files = find_model_files(cfg)

    if not model_files:
        print(f'Warning: No model files found under {cfg.model_path}!')
        return models

    print(f'Found a total of {len(model_files)} model files.')

    # Load each model files 
    for model_path in model_files:
        try:
            print(f'Loading model: {model_path}')
            checkpoint = torch.load(model_path, map_location=torch.device(cfg.device))
            model = BirdCLEFModel(cfg, num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(cfg.device)
            model.eval()
            models.append(model)
        except Exception as e:
            print(f'Error Loading model {model_path}: {e}')
    return models

def predict_on_spectrogram(audio_path, models, cfg, species_ids):
    '''Process a single audio file and predict species presence for each 5-second segment'''
    predictions = []
    row_ids = []
    soundscape_id = Path(audio_path).stem
    
    try:
        print(f'Processing {soundscape_id}')
        audio_data, _ = libros.load(audio_path, sr=cfg.SR)

        total_segments = len(audio_data)//(cfg.SR*cfg.WINDOW_SIZE)

        for segment_idx in range(total_segments):
            # Extract current 5-second segment
            st_sample = segment_idx * cfg.SR + cfg.WINDOW_SIZE
            end_sample= st_sample + cfg.SR + cfg.WINDOW_SIZE
            segment_audio = audio_data[st_sample:end_sample]

            # Calculate end time in seconds for row_id
            end_time_sec = (segment_idx+1)*cfg.WINDOW_SIZE

            row_id = f'{soundscape_id}_{end_time_sec}'
            row_ids.append(row_id)

            # Process the audio segment and get mel spectrogram
            if cfg.use_tta:
                # Use test-time augmentation
                all_preds = []
                for tta_idx in range(cfg.tta_count):
                    mel_spec = process_audio_segment(segment_audio, cfg)
                    mel_spec = apply_tta(mel_spec, tta_idx)

                    # convert to tensor and add batch and channel dimensions
                    mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    mel_spec = mel_spec.to(cfg.device)

                    # Handle single model case without ensemble 
                    if len(models)==1:
                        with torch.no_grad():
                            outputs = models[0](mel_spec)
                            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                            all_preds.append(probs)
                    else:
                        # Get prediction from each models for ensemble 
                        segment_preds = []
                        for model in models:
                            with torch.no_grad():
                                outputs = model(mel_spec)
                                probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                                segment_preds.append(probs)

                        # Average predictions from all models
                        avg_preds = np.mean(segment_preds, axis=0)
                        all_preds.append(avg_preds)

                    # Average TTA predictions
                    final_preds = np.mean(all_preds, axis=0)

            else:
                # No TTA -> just use original spectrogram
                mel_spec = process_audio_segment(segment_audio, cfg)
                mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                mel_spec = mel_spec.to(cfg.device)

                # Handle single model case without ensemble
                if len(models) == 1:
                    with torch.no_grad():
                        outputs = models[0](mel_spec)
                        final_preds = torch.sigmoid(outputs).cpu().numpy().squeeze()

                else:
                    # Get predictions from each model for ensemble
                    segment_preds = []
                    for model in models:
                        with torch.no_grad():
                            outputs = model(mel_spec)
                            prob = torch.sigmoid(outputs).cpu().numpy().squeeze()
                            segment_preds.append(probs)

                    final_preds = np.mean(segment_preds, axis=0)

            predictions.append(final_preds)

    except Exception as e:
        print(f'Error processing {audio_path}: {e}')

    return row_ids, predictions


def run_inference(cfg, models, species_ids):
    '''Run inference on all test soundscapes'''
    # Get list of test soundscapes
    test_files = list(Path(cfg.test_soundscapes).glob('*.ogg'))

    if cfg.debug:
        print(f'Debug mode enabled, using only {cfg.debug_count} files')
        test_files = test_files[:cfg.debug_count]

    print(f'Found {len(test_files)} test soundscapes')

    # Initialize lists for predictions
    all_row_ids = []
    all_predictions = []

    # Process each soundscape
    for audio_path in tqdm(test_files):
        row_ids, predictions = predict_on_spectrogram(str(audio_path), models, cfg, species_ids)
        all_row_ids.extend(row_ids)
        all_predictions.extend(predictions)

    return all_row_ids, all_predictions


def create_submission(row_ids, predictions, species_ids, cfg):
    '''Create submission dataframe'''
    print('Creating submission dataframe...')

    submission_dict = {'row_id': row_ids}

    # Add predicitions for each species
    for i, species in enumerate(species_ids):
        submission_dict[species] = [pred[i] for pred in predictions]

    # Create dataframe
    submission_df = pd.DataFrame(submission_dict)
    # Set row_id as index
    submission_df.set_index('row_id', inplace=True)
    # Verify the submission format against sample submission
    sample_sub = pd.read_csv(cfg.submission_csv, index_col='row_id')

    # Check if all species columns are present
    missing_cols = set(sample_sub.columns)-set(submission_df.columns)

    if missing_cols:
        print(f'Warning: Missing {len(missing_cols)} species columns in submission')

        # Add missing columns with zeros
        for col in missing_cols:
            submission_df[col]=0.0

     # Ensure columns are in the same order as sample submission
    submission_df = submission_df[sample_sub.columns]
    
    # Reset the index to include row_id as a column
    submission_df = submission_df.reset_index()
    
    return submission_df


def main():
    start_time = time.time()
    print(f'Starting BirdCLEF-2025 inference...')
    print(f'TTA Enabled: {cfg.use_tta} (variations: {cfg.tta_count if cfg.use_tta else 0})')

    # Load models 
    models = load_models(cfg, num_classes)

    if not models:
        print('No models found! Please check model paths')
        return

    print(f"Model usage: {'Single model' if len(models)==1 else f'Ensemble of {len(models)} models'}")

    # Run inference on test soundscapes
    row_ids, predictions = run_inference(cfg, models, species_ids)

     # Create submission dataframe
    submission_df = create_submission(row_ids, predictions, species_ids, cfg)
    
    # Save submission file
    submission_path = 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    
    end_time = time.time()
    print(f"Inference completed in {(end_time - start_time)/60:.2f} minutes")


if __name__ == "__main__":
    main()




