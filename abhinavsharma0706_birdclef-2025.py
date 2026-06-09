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

# Suppress warnings and limit logging output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

class CFG:
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    model_path = '/kaggle/input/birdclef-2025-efficientnet-b0'
    
    FS = 32000
    WINDOW_SIZE = 5
    N_FFT = 1034
    HOP_LENGTH = 64
    N_MELS = 136
    FMIN = 20
    FMAX = 16000
    TARGET_SHAPE = (256, 256)
    
    model_name = 'efficientnet_b0'
    in_channels = 1
    device = 'cpu'
    
    batch_size = 16
    use_tta = False
    tta_count = 3
    threshold = 0.7
    
    use_specific_folds = False
    folds = [0, 1]
    
    debug = False
    debug_count = 3



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



class BirdCLEF2025Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.taxonomy_df = None
        self.species_ids = []
        self.models = []
        self._load_taxonomy()

    def _load_taxonomy(self):
        print("Loading taxonomy data...")
        self.taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = self.taxonomy_df['primary_label'].tolist()
        print(f"Number of classes: {len(self.species_ids)}")

    def audio2melspec(self, audio_data):
        if np.isnan(audio_data).any():
            mean_signal = np.nanmean(audio_data)
            audio_data = np.nan_to_num(audio_data, nan=mean_signal)
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.cfg.FS,
            n_fft=self.cfg.N_FFT,
            hop_length=self.cfg.HOP_LENGTH,
            n_mels=self.cfg.N_MELS,
            fmin=self.cfg.FMIN,
            fmax=self.cfg.FMAX,
            power=2.0
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        return mel_spec_norm

    def process_audio_segment(self, audio_data):
        if len(audio_data) < self.cfg.FS * self.cfg.WINDOW_SIZE:
            audio_data = np.pad(
                audio_data,
                (0, self.cfg.FS * self.cfg.WINDOW_SIZE - len(audio_data)),
                mode='constant'
            )
        mel_spec = self.audio2melspec(audio_data)
        if mel_spec.shape != self.cfg.TARGET_SHAPE:
            mel_spec = cv2.resize(mel_spec, self.cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
        return mel_spec.astype(np.float32)



    def find_model_files(self):
        model_files = []
        model_dir = Path(self.cfg.model_path)
        for path in model_dir.glob('**/*.pth'):
            model_files.append(str(path))
        return model_files

    def load_models(self):
        self.models = []
        model_files = self.find_model_files()
        if not model_files:
            print(f"Warning: No model files found under {self.cfg.model_path}!")
            return self.models

        print(f"Found a total of {len(model_files)} model files.")
        
        if self.cfg.use_specific_folds:
            filtered_files = []
            for fold in self.cfg.folds:
                fold_files = [f for f in model_files if f"fold{fold}" in f]
                filtered_files.extend(fold_files)
            model_files = filtered_files
            print(f"Using {len(model_files)} model files for the specified folds ({self.cfg.folds}).")
        
        for model_path in model_files:
            try:
                print(f"Loading model: {model_path}")
                checkpoint = torch.load(model_path, map_location=torch.device(self.cfg.device))
                model = BirdCLEFModel(self.cfg, len(self.species_ids))
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(self.cfg.device)
                model.eval()
                self.models.append(model)
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
        
        return self.models



    def predict(self, input_tensor):
        preds = []
        input_tensor = input_tensor.to(self.cfg.device)
        for model in self.models:
            with torch.no_grad():
                output = model(input_tensor)
                output = torch.sigmoid(output)
                preds.append(output.cpu().numpy())
        preds = np.mean(preds, axis=0)
        return preds

    def segment_audio(self, audio_data):
        segment_length = self.cfg.FS * self.cfg.WINDOW_SIZE
        segments = []
        num_segments = math.ceil(len(audio_data) / segment_length)
        for i in range(num_segments):
            start = i * segment_length
            end = min((i + 1) * segment_length, len(audio_data))
            segment = audio_data[start:end]
            if len(segment) < segment_length:
                segment = np.pad(segment, (0, segment_length - len(segment)), mode='constant')
            segments.append(segment)
        return segments

    def predict_soundscape(self, file_path):
        y, sr = librosa.load(file_path, sr=self.cfg.FS, mono=True)
        segments = self.segment_audio(y)
        predictions = []

        for segment in segments:
            mel = self.process_audio_segment(segment)
            mel_tensor = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            pred = self.predict(mel_tensor)[0]
            predictions.append(pred)

        predictions = np.array(predictions)
        averaged_preds = predictions.mean(axis=0)
        return averaged_preds



def create_submission(predictor, test_files, output_csv='submission.csv'):
    id_list = []
    prediction_list = []

    for file_path in tqdm(test_files, desc="Predicting"):
        preds = predictor.predict_soundscape(file_path)
        file_id = os.path.splitext(os.path.basename(file_path))[0]
        id_list.append(file_id)
        prediction_list.append(preds)

    prediction_array = np.array(prediction_list)
    df = pd.DataFrame(prediction_array, columns=predictor.cfg.LABELS)
    df.insert(0, 'filename', id_list)
    df.to_csv(output_csv, index=False)
    print(f"Submission file saved to {output_csv}")
    return df



def run(self):
    print("[INFO] Starting BirdCLEF2025 pipeline...")

    # Step 1: Load test files
    test_files = glob.glob(os.path.join(self.cfg.test_dir, "*.ogg"))
    print(f"[INFO] Found {len(test_files)} test files.")

    # Step 2: Initialize predictor
    predictor = SoundscapePredictor(self.cfg)

    # Step 3: Create submission
    submission_df = create_submission(predictor, test_files)

    # Step 4: Save submission
    submission_df.to_csv("submission.csv", index=False)
    print("[INFO] Submission file saved as submission.csv.")


