import os
import gc
import warnings
import logging
import time
import math
import cv2
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


def apply_power_to_low_ranked_cols(
    p: np.ndarray,
    top_k: int = 30,
    exponent: float = 2.0,
    inplace: bool = True
) -> np.ndarray:
    if not inplace:
        p = p.copy()
    tail_cols = np.argsort(-p.max(axis=0))[top_k:]
    p[:, tail_cols] = p[:, tail_cols] ** exponent
    return p


class CFG:
 
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    model_path = '/kaggle/input/efficientnetb0_birdclef_trained/pytorch/default/1'  
    
    FS = 32000  
    WINDOW_SIZE = 5  
    
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    TARGET_SHAPE = (256, 256)
    
    model_name = 'efficientnet_b0'
    in_channels = 1
    device = 'cpu'
    
    batch_size = 16
    threshold = 0.5
    
cfg = CFG()


print(f"Using device: {cfg.device}")
taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
species_ids = taxonomy_df['primary_label'].tolist()
num_classes = len(species_ids)
print(f"Number of classes: {num_classes}")


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg
        
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=False,  
            in_chans=cfg.in_channels,
            drop_rate=0.0,    
            drop_path_rate=0.0
        )
        
        backbone_out = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
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

def process_audio_segment(audio_data, cfg):
    if len(audio_data) < cfg.FS * cfg.WINDOW_SIZE:
        audio_data = np.pad(audio_data, 
                          (0, cfg.FS * cfg.WINDOW_SIZE - len(audio_data)), 
                          mode='constant')
    
    mel_spec = audio2melspec(audio_data, cfg)
    
    # Resize if needed
    if mel_spec.shape != cfg.TARGET_SHAPE:
        mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
        
    return mel_spec.astype(np.float32)


def find_model_files(cfg):

    model_files = []
    
    model_dir = Path(cfg.model_path)
    
    for path in model_dir.glob('**/*.pth'):
        model_files.append(str(path))
    
    return model_files


def load_models(cfg, num_classes):

    models = []
    
    model_files = find_model_files(cfg)
    
    print(f"Found a total of {len(model_files)} model files.")
    
    for model_path in model_files:
        try:
            print(f"Loading model: {model_path}")
            checkpoint = torch.load(model_path, map_location=torch.device(cfg.device))
            
            model = BirdCLEFModel(cfg, num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(cfg.device)
            model.eval()
            
            models.append(model)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
    
    return models


def predict_on_spectrogram(audio_path, models, cfg, species_ids):
    """Process a single audio file and predict species presence for each 5-second segment"""
    predictions = []
    row_ids = []
    soundscape_id = Path(audio_path).stem
    
    try:
        print(f"Processing {soundscape_id}")
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)
        
        total_segments = int(len(audio_data) / (cfg.FS * cfg.WINDOW_SIZE))
        
        for segment_idx in range(total_segments):
            start_sample = segment_idx * cfg.FS * cfg.WINDOW_SIZE
            end_sample = start_sample + cfg.FS * cfg.WINDOW_SIZE
            segment_audio = audio_data[start_sample:end_sample]
            
            end_time_sec = (segment_idx + 1) * cfg.WINDOW_SIZE
            row_id = f"{soundscape_id}_{end_time_sec}"
            row_ids.append(row_id)

            mel_spec = process_audio_segment(segment_audio, cfg)
            
            mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            mel_spec = mel_spec.to(cfg.device)
            with torch.no_grad():
                outputs = models[0](mel_spec)
                final_preds = torch.sigmoid(outputs).cpu().numpy().squeeze()
                    
            predictions.append(final_preds)
            
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
    
    return row_ids, predictions


# def run_inference(cfg, models, species_ids):
#     test_files = list(Path(cfg.test_soundscapes).glob('*.ogg'))
    
#     print(f"Found {len(test_files)} test soundscapes")

#     all_row_ids = []
#     all_predictions = []

#     for audio_path in tqdm(test_files):
#         row_ids, predictions = predict_on_spectrogram(str(audio_path), models, cfg, species_ids)
#         all_row_ids.extend(row_ids)
#         all_predictions.extend(predictions)
    
#     return all_row_ids, all_predictions

def run_inference(cfg, models, species_ids):
    test_files = list(Path(cfg.test_soundscapes).glob('*.ogg'))
    
    print(f"Found {len(test_files)} test soundscapes")

    all_row_ids = []
    all_predictions = []
    grouped_preds = {}

    for audio_path in tqdm(test_files):
        row_ids, predictions = predict_on_spectrogram(str(audio_path), models, cfg, species_ids)
        
        predictions = np.array(predictions)
        predictions = apply_power_to_low_ranked_cols(predictions, top_k=30, exponent=2.0)
        smoothed_preds = predictions.copy()
        for i in range(1, len(predictions) - 1):
            smoothed_preds[i] = (
                0.2 * predictions[i - 1] +
                0.6 * predictions[i] +
                0.2 * predictions[i + 1]
            )
        smoothed_preds[0] = 0.8 * predictions[0] + 0.2 * predictions[1]
        smoothed_preds[-1] = 0.8 * predictions[-1] + 0.2 * predictions[-2]

        all_row_ids.extend(row_ids)
        all_predictions.extend(smoothed_preds)
    
    return all_row_ids, all_predictions


def create_submission(row_ids, predictions, species_ids, cfg):
    print("Creating submission dataframe...")

    submission_dict = {'row_id': row_ids}
    
    for i, species in enumerate(species_ids):
        submission_dict[species] = [pred[i] for pred in predictions]

    submission_df = pd.DataFrame(submission_dict)

    submission_df.set_index('row_id', inplace=True)

    sample_sub = pd.read_csv(cfg.submission_csv, index_col='row_id')

    missing_cols = set(sample_sub.columns) - set(submission_df.columns)
    if missing_cols:
        print(f"Warning: Missing {len(missing_cols)} species columns in submission")
        for col in missing_cols:
            submission_df[col] = 0.0

    submission_df = submission_df[sample_sub.columns]

    submission_df = submission_df.reset_index()
    
    return submission_df


def main():
    start_time = time.time()
    print("Starting BirdCLEF-2025 inference...")

    models = load_models(cfg, num_classes)
    
    if not models:
        print("No models found! Please check model paths.")
        return
    
    print(f"Model usage: {'Single model' if len(models) == 1 else f'Ensemble of {len(models)} models'}")

    row_ids, predictions = run_inference(cfg, models, species_ids)

    submission_df = create_submission(row_ids, predictions, species_ids, cfg)

    submission_path = 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    
    end_time = time.time()
    print(f"Inference completed in {(end_time - start_time)/60:.2f} minutes")


if __name__ == "__main__":
    main()




